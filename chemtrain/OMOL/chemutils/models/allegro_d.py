"""Custom implementation of Allegro-Jax.
 """
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, Any, Tuple, List, Optional
from collections import OrderedDict

import haiku as hk
import jax
from jax import random, lax, numpy as jnp, nn as jax_nn
from jax_md import space, partition, nn, util
import e3nn_jax as e3nn

import numpy as onp

from jax_md_mod.model import layers, sparse_graph
from jax_md import util as md_util

from chemutils.models.layers import AtomicEnergyLayer

import e3x
import e3nn_jax as e3nn
from jax.scipy.special import sph_harm

class AllegroCoefficientsLayer(hk.Module):
    def __init__(self,
                 epsilon: float,
                 irreps_out: e3nn.Irreps,
                 max_ell: int = 3,
                 mlp_n_hidden: int = 32,
                 mlp_n_layers: int = 3,
                 mlp_activation: Callable = jax.nn.silu,
                 num_species: int = 100,
                 p: int = 6,
                 name: str = "AllegroCoefficientsLayer"):
        super().__init__(name=name)
        self.epsilon = epsilon
        self.max_ell = max_ell
        self.irreps_out = irreps_out
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.mlp_activation = mlp_activation
        self.num_species = num_species
        self.p = p

        # Parameters for species embeddings
        self.embedding = hk.Embed(self.num_species, self.mlp_n_hidden)



    def __call__(self,
                 vectors: e3nn.IrrepsArray,  # [n_edges, 3]
                 x: jnp.ndarray,  # [n_edges, features]
                 V: e3nn.IrrepsArray,  # [n_edges, irreps]
                 senders: jnp.ndarray,  # [n_edges]
                 species: jnp.ndarray,  # [n_nodes]
                 is_training: bool = False) -> e3nn.IrrepsArray:
        """
        Predicts the coefficients c^{(I)}_{nℓm} for each atom I based on embeddings and irreps.

        Args:
            vectors: Relative displacement vectors between neighboring atoms.
            x: Feature vectors for each edge.
            V: Irreps for each edge.
            senders: Indices of central atoms.
            species: Atomic species for each atom.
            is_training: Whether the model is in training mode.

        Returns:
            Predicted coefficients in irreps.
        """
        # Obtain species embeddings for senders
        species_embeddings = self.embedding(species[senders])

        # Concatenate species embeddings with edge features
        edge_features = jnp.concatenate([x, species_embeddings], axis=-1)

        # Apply multi-layer perceptron to predict coefficients
        coefficients = e3nn.haiku.MultiLayerPerceptron(
            [self.mlp_n_hidden] * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False
        )(edge_features)

        # Apply smoothing envelope based on distance
        distances = e3nn.norm(vectors).array.squeeze(1)
        coefficients = layers.SmoothingEnvelope(self.p)(distances)[:, None] * coefficients

        # Project to desired irreps
        projected_coefficients = e3nn.haiku.Linear(self.irreps_out)(coefficients)

        return projected_coefficients

def radial_basis(r, sigma_n):
    return jnp.exp(-0.5 * (r / sigma_n) ** 2)


def angular_basis(theta, phi, l, m):
    return sph_harm(m, l, theta, phi).real

def compute_local_density(R_I, coefficients, radial_basis, angular_basis):
    """
    Compute the local electron density contribution for atom I.

    Args:
        R_I: Position of atom I (3D vector).
        coefficients: Predicted coefficients c_{n\ell m}^{(I)}.
        radial_basis: Radial basis functions.
        angular_basis: Angular (spherical harmonics) basis functions.

    Returns:
        Local electron density contribution (function of r).
    """
    def density_at_r(r, theta, phi):
        density = 0.0
        for n, ell, m in coefficients:
            radial_contribution = radial_basis(r - R_I, sigma_n=n)
            angular_contribution = angular_basis(theta, phi, ell, m)
            density += coefficients[(n, ell, m)] * radial_contribution * angular_contribution
        return density
        
    return density_at_r

def compute_total_density(r, atoms, coefficients, radial_basis, angular_basis):
    """
    Compute the total electron density at a point r by summing the local densities from all atoms.

    Args:
        r: Position in space where the total density is evaluated (3D vector).
        atoms: List of atom positions in the system.
        coefficients: Dictionary of predicted coefficients c_{nℓm}^{(I)} for each atom.
        radial_basis: Radial basis functions.
        angular_basis: Angular (spherical harmonics) basis functions.

    Returns:
        Total electron density at point r.
    """
    total_density = 0.0
    for R_I in atoms:
        # Compute the local density for atom I at position r
        local_density = compute_local_density(R_I, coefficients, radial_basis, angular_basis)
        total_density += local_density(r)
    return total_density
    
class Allegro(hk.Module):
    """Allegro for molecular property prediction.

    This model takes as input a sparse representation of a molecular graph
    - consisting of pairwise distances and angular triplets - and predicts
    pairwise properties. Global properties can be obtained by summing over
    pairwise predictions.

    This custom implementation follows the original Allegro-Jax
    (https://github.com/mariogeiger/allegro-jax).
    """
    def __init__(self,
                 avg_num_neighbors: float,
                 max_ell: int = 3,
                 irreps: e3nn.Irreps = 128 * e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e"),
                 mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
                 mlp_n_hidden: int = 1024,
                 mlp_n_layers: int = 3,
                 embed_dim: int = 32,
                 num_species: int = 100,
                 envelope_p: int = 6,
                 n_radial_basis: int = 8,
                 output_irreps: e3nn.Irreps = e3nn.Irreps("0e"),
                 num_layers: int = 1, #3,
                 name: str = 'Allegro',
                 *args, **kwargs):
        """Initializes the Allegro model

        Args:
            avg_num_neighbors: Average number of neighboring atoms.
            max_ell: Maximum rotation order (l_{max}) to which features in the
                     network are truncated.
            irreps: Irreducible representations to consider.
            mlp_activation: Activation function in MLPs.
            mlp_n_hidden: Number of nodes in hidden layers of two-body MLP and
                          latent MLP.
            mlp_n_layers: Number of hidden layers in MLPs.
            p: Polynomial order of polynomial envelope for weighting two-body
               features by atomic distance.
            n_radial_basis: Number of Bessel basis functions.
            radial_cutoff: Radial cut-off distance of edges.
            output_irreps: Irreducible representations in output layer.
            num_layers: Number of tensor product layers.
            name: Name of Allegro model.
        """
        super().__init__(name=name)

        self.max_ell = max_ell
        self.irreps = irreps
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.envelope_p = envelope_p
        self.output_irreps = output_irreps
        self.num_layers = num_layers

        epsilon = hk.get_parameter(
            "varepsilon", shape=(),
            init=hk.initializers.Constant(jnp.sqrt(avg_num_neighbors))
        )
        self.epsilon = 1 / jnp.sqrt(1 + epsilon ** 2)

        self.alpha = hk.get_parameter(
            "residual_alpha", shape=(), init=hk.initializers.Constant(1.0)
        )

        self.embedding = hk.Embed(num_species, embed_dim)
        self.radial_basis = layers.RadialBesselLayer(
            cutoff=1.0, num_radial=n_radial_basis, envelope_p=envelope_p
        )

        self.particle_energy = hk.Embed(num_species, 1)

        # self.dropout = e3nn.haiku.Dropout(0.5)


    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        species: jnp.ndarray,  # [n_nodes]
        is_training: bool = False,
    ) -> e3nn.IrrepsArray:
        """Predicts pairwise quantities for a given conformation.

        Args:
            node_attrs: Species information (jax.nn.one_hot(z, num_species)).
            vectors: Relative displacement vectors r_{ij} between neighboring
                     atoms from i to j.
            senders: Indices of central atoms i.
            receivers: Indices of neighboring atoms j.
            edge_feats:
            is_training: Set model into training mode (True: dropout is applied).

        Returns:
            An array of predicted pairwise quantities in irreducible representations.
        """
        num_edges = vectors.shape[0]
        assert vectors.shape == (num_edges, 3)
        assert senders.shape == (num_edges,)
        assert receivers.shape == (num_edges,)

        assert vectors.irreps in ["1o", "1e"]
        irreps = e3nn.Irreps(self.irreps)
        irreps_out = e3nn.Irreps(self.output_irreps)

        # list of irreps of layers in network
        irreps_layers = [irreps] * self.num_layers + [irreps_out]
        # filter irreps in order to match desired output irreps via tensor products
        irreps_layers = self.filter_layers(irreps_layers, self.max_ell)

        # calculate Euclidean norm of vectors, convert into array
        d = e3nn.norm(vectors).array.squeeze(1)
        x = jnp.concatenate(
            [
                self.radial_basis(d),
                self.embedding(species[senders]),
                self.embedding(species[receivers]),
            ],
            axis=1,
        )

        # Protection against exploding dummy edges:
        x = jnp.where(d[:, None] == 0.0, 0.0, x)

        # initialize mlp and evaluate at x
        x = e3nn.haiku.MultiLayerPerceptron(
            (
                self.mlp_n_hidden // 8,
                self.mlp_n_hidden // 4,
                self.mlp_n_hidden // 2,
                self.mlp_n_hidden,
            ),
            self.mlp_activation,
            output_activation=False,
        )(x)

        # weight mlp output according to value of polynomial envelope at distance d
        x = layers.SmoothingEnvelope(self.envelope_p)(d)[:, None] * x
        assert x.shape == (num_edges, self.mlp_n_hidden)

        # keep irreps from first layer that satisfy mirroring of input vectors
        irreps_Y = irreps_layers[0].filter(
            keep=lambda mir: vectors.irreps[0].ir.p ** mir.ir.l == mir.ir.p
        )
        # apply spherical harmonics according to irreps of first layer to vectors
        V = e3nn.spherical_harmonics(irreps_Y, vectors, True)
        V = e3nn.concatenate([V, self.embedding(species[senders]), self.embedding(species[receivers])])

        w = e3nn.haiku.MultiLayerPerceptron((V.irreps.num_irreps,),None)(x)
        V = w * V / V.irreps.num_irreps
        assert V.shape == (num_edges, V.irreps.dim)

        for irreps in irreps_layers[1:]:
            y, V = AllegroLayer(
                epsilon=self.epsilon,
                max_ell=self.max_ell,
                output_irreps=irreps,
                mlp_activation=self.mlp_activation,
                mlp_n_hidden=self.mlp_n_hidden,
                mlp_n_layers=self.mlp_n_layers,
                p=self.envelope_p,
            )(vectors, x, V, senders)

            # residual update as weighted sum
            # alpha = 0.5
            x = (x + self.alpha ** 2 * y) / (1 + self.alpha**2)

            # x = self.dropout(hk.next_rng_key(), x, is_training)
            # V = self.dropout(hk.next_rng_key(), V, is_training)

        x = e3nn.haiku.MultiLayerPerceptron((128,),None)(x)

        # x = self.dropout(hk.next_rng_key(), x, is_training)

        xV = e3nn.haiku.Linear(irreps_out)(e3nn.concatenate([x, V]))

        if xV.irreps != irreps_out:
            raise ValueError(f"output_irreps {irreps_out} is not reachable")

        xV = layers.SmoothingEnvelope(self.envelope_p)(d)[:, None] * xV

        return xV


    def filter_layers(self, layer_irreps: List[e3nn.Irreps], max_ell: int) -> List[e3nn.Irreps]:
        """Shape irreducible representations of tensor product layers in order
        to match desired output irreps via tensor products.

        Args:
            layer_irreps: Irreducible representations of tensor product layers.
            max_ell: Maximum rotation order (l_{max}) to which features in the
                     network are truncated.

        Returns:
            Updated list of irreducible representations of tensor product layers.
        """
        layer_irreps = list(layer_irreps)
        # initialize filtered as last layer from layer_irreps
        filtered = [e3nn.Irreps(layer_irreps[-1])]
        # propagate through network from output to input
        for irreps in reversed(layer_irreps[:-1]):
            irreps = e3nn.Irreps(irreps)
            # filter all irreps, remaining should satisfy tensor product with consecutive layer
            irreps = irreps.filter(
                # tensor product of consecutive layers in the nn structure
                keep=e3nn.tensor_product(
                    # irreps of subsequent layer in nn structure
                    filtered[0],
                    # only irreps considering spherical harmonics (eg. '1x0e+1x1o+1x2e+1x3o' for lmax=3)
                    e3nn.Irreps.spherical_harmonics(lmax=max_ell),
                ).regroup() # regroup the same irreps together
            )
            filtered.insert(0, irreps)
        return filtered


class AllegroLayer(hk.Module):
    """Tensor product layer of Allegro.

    This model updates invariant two-body features and equivariant latent
    features by applying tensor prodict operations.

    This custom implementation follows the original Allegro-Jax
    (https://github.com/mariogeiger/allegro-jax).
    """
    def __init__(self,
                 epsilon: float,
                 max_ell: int = 3,
                 output_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e"),
                 mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
                 mlp_n_hidden: int = 64,
                 mlp_n_layers: int = 3,
                 p: int = 6,
                 name: str = 'TensorProduct'):
        """Tensor product layer.

        Args:
            avg_num_neighbors: Average number of neighboring atoms.
            max_ell: Maximum rotation order (l_{max}) to which features in the
                     network are truncated.
            irreps: Irreducible representations to consider.
            mlp_activation: Activation function in MLPs.
            mlp_n_hidden: Number of nodes in hidden layers of two-body MLP and
                          latent MLP.
            mlp_n_layers: Number of hidden layers in MLPs.
            p: Polynomial order of polynomial envelope for weighting two-body
               features by atomic distance d.
            name: Name of Embedding block.
        """
        super().__init__(name=name)
        self.epsilon = epsilon
        self.max_ell = max_ell
        self.output_irreps = output_irreps
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.envelope_p = p


    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        x: jnp.ndarray,  # [n_edges, features]
        V: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        """Returns output of the Tensor product layer."""
        num_edges = vectors.shape[0]
        assert vectors.shape == (num_edges, 3)
        assert x.shape == (num_edges, x.shape[-1])
        assert V.shape == (num_edges, V.irreps.dim)
        assert senders.shape == (num_edges,)

        irreps_out = e3nn.Irreps(self.output_irreps)

        w = e3nn.haiku.MultiLayerPerceptron((V.irreps.mul_gcd,),None)(x)
        Y = e3nn.spherical_harmonics(range(self.max_ell + 1), vectors, True)
        wY = e3nn.scatter_sum(
            w[:, :, None] * Y[:, None, :], dst=senders, map_back=True
        ) * self.epsilon
        assert wY.shape == (num_edges, V.irreps.mul_gcd, wY.irreps.dim)

        V = e3nn.tensor_product(
            wY, V.mul_to_axis(), filter_ir_out="0e" + irreps_out
        ).axis_to_mul()

        if "0e" in V.irreps:
            x = jnp.concatenate([x, V.filter(keep="0e").array], axis=1)
            V = V.filter(drop="0e")

        x = e3nn.haiku.MultiLayerPerceptron(
            (self.mlp_n_hidden,) * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False,
        )(x)

        lengths = e3nn.norm(vectors).array
        x =  layers.SmoothingEnvelope(self.envelope_p)(lengths) * x
        assert x.shape == (num_edges, self.mlp_n_hidden)

        V = e3nn.haiku.Linear(irreps_out)(V)
        assert V.shape == (num_edges, V.irreps.dim)

        return (x, V)


allegro_default_kwargs = OrderedDict(
    embed_dim = 32,
    output_irreps = "1x0e", # Reduce the output trough two linear layers
    hidden_irreps =  128 * e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e"),
    mlp_n_hidden=64,
    mlp_n_layers=2,
    max_ell = 3,
    n_radial_basis = 8,
    envelope_p = 6,
    num_layers=1,
    num_species=100,
)


def allegro_neighborlist_pp(displacement: space.DisplacementFn,
                            r_cutoff: float,
                            n_species: int = 100,
                            positions_test: jnp.ndarray = None,
                            neighbor_test: partition.NeighborList = None,
                            max_edge_multiplier: float = 1.25,
                            max_edges=None,
                            avg_num_neighbors: float = None,
                            mode: str = "energy",
                            per_particle: bool = False,
                            positive_species: bool = False,
                            **nequip_kwargs
                            ) -> Tuple[nn.InitFn, Callable[[Any, md_util.Array],
                                                                 md_util.Array]]:
    """Allegro model for property prediction.

    Args:
        displacement: Jax_md displacement function
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        n_species: Number of different atom species the network is supposed
            to process.
        positions_test: Sample positions to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        max_edges: Expected maximum of valid edges.
        nequip_escn: Use NequIPESCN instead of NequIP (more computational
            efficient).
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "property_prediction" (default),
            returns the learned node features. If "energy_prediction", returns
            the total energy of the system.
        per_particle: Return per-particle energies instead of total energy.
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        nequip_kwargs: Kwargs to change the default structure of NequIP.
            For definition of the kwargs, see NequIP.


    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """
    kwargs = deepcopy(allegro_default_kwargs)
    kwargs.update(nequip_kwargs)

    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    _avg_num_neighbors = None
    if positions_test is not None and neighbor_test is not None:
        if neighbor_test.format == partition.Dense:
            print('Capping edges and triplets. Beware of overflow, which is'
                  ' currently not being detected.')

            testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
                displacement, positions_test, neighbor_test, r_cutoff)
            max_edges = jnp.int32(jnp.ceil(testgraph.n_edges * max_edge_multiplier))

            # cap maximum edges and angles to avoid overflow from multiplier
            n_particles, n_neighbors = neighbor_test.idx.shape
            max_edges = min(max_edges, n_particles * n_neighbors)

            print(f"Estimated max. {max_edges} edges.")

            _avg_num_neighbors = testgraph.n_edges / n_particles
        else:
            n_particles = neighbor_test.idx.shape[0]
            _avg_num_neighbors = onp.sum(neighbor_test.idx[0] < n_particles)
            _avg_num_neighbors /= n_particles

    if avg_num_neighbors is None:
        avg_num_neighbors = _avg_num_neighbors
    assert avg_num_neighbors is not None, (
        "Average number of neighbors not set and no test graph was provided."
    )

    @hk.without_apply_rng
    @hk.transform
    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            print(f"[Allegro] Use default species")
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        elif positive_species:
            species -= 1
        if mask is None:
            print(f"[Allegro] Use default mask")
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        # Compute the displacements for all edges
        dyn_displacement = partial(displacement, **dynamic_kwargs)

        if neighbor.format == partition.Dense:
            graph, _ = sparse_graph.sparse_graph_from_neighborlist(
                dyn_displacement, position, neighbor, r_cutoff,
                species, max_edges=max_edges, species_mask=mask
            )
            senders = graph.idx_i
            receivers = graph.idx_j
        else:
            assert neighbor.idx.shape == (2, neighbor.idx.shape[1]), "Neighbor list has wrong shape."
            senders, receivers = neighbor.idx

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        vectors = jnp.where(
            jnp.logical_and(
                senders < position.shape[0],
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff / jnp.sqrt(3))
        vectors /= r_cutoff
        vectors = e3nn.IrrepsArray(
            e3nn.Irreps("1o"), vectors
        )

        net = Allegro(
            avg_num_neighbors=avg_num_neighbors,
            max_ell=kwargs["max_ell"],
            irreps=kwargs["hidden_irreps"],
            mlp_n_hidden = kwargs["mlp_n_hidden"],
            mlp_n_layers = kwargs["mlp_n_layers"],
            envelope_p=kwargs["envelope_p"],
            n_radial_basis=kwargs["n_radial_basis"],
            output_irreps=kwargs["output_irreps"],
            num_layers=kwargs["num_layers"],
            embed_dim=kwargs["embed_dim"],
            num_species=n_species,
        )

        features = net(vectors, senders, receivers, species)

        if mode == "energy":
            per_edge_energies, = features.array.T

            per_node_energies = layers.high_precision_segment_sum(
                per_edge_energies, senders, num_segments=position.shape[0])
            per_atom_energies = AtomicEnergyLayer(n_species)(per_node_energies, species)
            per_atom_energies *= mask

            if per_particle:
                return per_atom_energies
            else:
                return md_util.high_precision_sum(per_atom_energies)

        elif mode == "energy_and_charge":
            raise NotImplementedError("Mode 'energy_and_charge' not implemented.")
        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)


