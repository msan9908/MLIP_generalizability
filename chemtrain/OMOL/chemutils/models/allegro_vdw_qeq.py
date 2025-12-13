"""Custom implementation of Allegro-Jax.
 """
import functools
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, Any, Tuple, List, Optional, Union, Iterable
from collections import OrderedDict

import haiku as hk
import jax
import jaxopt
from jax import random, lax, numpy as jnp, nn as jax_nn, tree_util
from jax_md import space, partition, nn, util, energy, smap
import e3nn_jax as e3nn

import numpy as onp
from jax import debug
from jax_md.energy import lennard_jones_neighbor_list
from jax_md_mod.model import layers, sparse_graph
from jax_md_mod import custom_electrostatics
from jax_md import util as md_util
from jax_md_mod.model.layers import SmoothingEnvelope

from jaxopt import ProjectedGradient

from chemutils.models.layers import AtomicEnergyLayer, ScaleShiftLayer
from sympy.integrals.intpoly import hyperplane_parameters

allegro_default_kwargs = OrderedDict(
    embed_dim = 32,
    output_irreps = "2x0e", # Reduce the output trough two linear layers
    hidden_irreps =  128 * e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e"),
    mlp_n_hidden=64,
    mlp_n_layers=2,
    max_ell = 3,
    n_radial_basis = 8,
    envelope_p = 6,
    num_layers=1,
    num_species=200,
    charge_embed_dim=16,
    max_radius=0.5,
    min_radius=0.01
)



class AllegroQeq(hk.Module):
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
                 num_layers: Union[int, Iterable[int]] = 1, #3,
                 charge_embed_n_hidden: int = 16,
                 charge_embed_n_layers: int = 1,
                 exclude_electrostatics: bool = False,
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
        self.charge_embed_n_hidden = charge_embed_n_hidden
        self.charge_embed_n_layers = charge_embed_n_layers
        self.num_species = num_species
        self.exclude_electrostatics = exclude_electrostatics

        epsilon = hk.get_parameter(
            "varepsilon", shape=(),
            init=hk.initializers.Constant(jnp.sqrt(avg_num_neighbors))
        )
        self.epsilon = 1 / jnp.sqrt(1 + jax_nn.softplus(epsilon))

        self.alpha = hk.get_parameter(
            "residual_alpha", shape=(), init=hk.initializers.Constant(0.0)
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
        # node_features: e3nn.IrrepsArray = None,
        charge_qeq_fn: Callable,
        vdw_fn: Callable,
        is_training: bool = False,
        exclude_correction: bool = False,
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
        try:
            pre_qeq, post_qeq = self.num_layers
        except TypeError:
            pre_qeq = self.num_layers
            post_qeq = self.num_layers

        irreps_layers = [irreps] * (pre_qeq + post_qeq + 1) + [irreps_out]
        # filter irreps in order to match desired output irreps via tensor products
        irreps_layers = self.filter_layers(irreps_layers, self.max_ell)

        # if node_features is not None:
        #     print(f"Called with extra node features")
        #     edge_feats = [node_features[senders], node_features[receivers]]
        # else:
        #     edge_feats = []

        # calculate Euclidean norm of vectors, convert into array

        # Get scalar features
        # scalar_edge_feats = [f.filter(keep="0e").array for f in edge_feats]

        d = e3nn.norm(vectors).array.squeeze(1)
        x = jnp.concatenate(
            [
                self.radial_basis(d),
                self.embedding(species[senders]),
                self.embedding(species[receivers]),
                # *scalar_edge_feats,
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
        V = e3nn.concatenate([
            V,
            self.embedding(species[senders]),
            self.embedding(species[receivers]),
            # *edge_feats
        ])

        w = e3nn.haiku.MultiLayerPerceptron((V.irreps.num_irreps,),None)(x)
        V = w * V / V.irreps.num_irreps
        assert V.shape == (num_edges, V.irreps.dim)

        charge_out = None
        for idx, irreps in enumerate(irreps_layers[1:]):
            if idx == pre_qeq and self.exclude_electrostatics == "skip":
                print(f"Skipping CELLI layer")
                continue

            if idx == pre_qeq and not self.exclude_electrostatics:
                (y, V), charge_out,vdw_pot = AllegroQeqLayer(
                    max_radius=1.0,
                    num_species=self.num_species,
                    mlp_n_hidden=self.mlp_n_hidden,
                    irreps_out=irreps,
                    epsilon=self.epsilon,
                    max_ell=self.max_ell,
                    mlp_activation=self.mlp_activation,
                    mlp_n_layers=self.mlp_n_layers,
                    p=self.envelope_p,
                    charge_embed_n_hidden=self.charge_embed_n_hidden,
                    charge_embed_n_layers=self.charge_embed_n_layers,
                )(
                    vectors,
                    x,
                    V,
                    senders,
                    species,
                    charge_qeq_fn,
                    vdw_fn,
                    exclude_correction=exclude_correction,
                )
            else:
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
            x = (x + jax_nn.softplus(self.alpha) * y) / (1 + jax_nn.softplus(self.alpha))

            # x = self.dropout(hk.next_rng_key(), x, is_training)
            # V = self.dropout(hk.next_rng_key(), V, is_training)

        # TODO: What dimensions are actually required?
        x = e3nn.haiku.MultiLayerPerceptron((self.mlp_n_hidden,),None)(x)

        # x = self.dropout(hk.next_rng_key(), x, is_training)

        xV = e3nn.haiku.Linear(irreps_out)(e3nn.concatenate([x, V]))

        if xV.irreps != irreps_out:
            raise ValueError(f"output_irreps {irreps_out} is not reachable")

        # Perform a nonlinear activation
        # TODO: Maybe change
        # num_vectors = irreps_out.filter(drop=["0e", "0o"]).num_irreps  # Multiplicity of (l > 0) irreps
        # xV = e3nn.haiku.Linear(
        #     (irreps_out + e3nn.Irreps(f"{num_vectors}x0e")).simplify()
        # )(xV)
        # xV = e3nn.gate(xV, even_act=jax_nn.swish, even_gate_act=jax_nn.sigmoid)
        # xV = e3nn.haiku.Linear(irreps_out)(xV)

        xV = layers.SmoothingEnvelope(self.envelope_p)(d)[:, None] * xV

        return xV, charge_out, vdw_pot


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

        print(f"Out irreps {V.irreps}")

        if "0e" in V.irreps:
            x = jnp.concatenate([x, V.filter(keep="0e").array], axis=1)
            V = V.filter(drop="0e")

        x = e3nn.haiku.MultiLayerPerceptron(
            (self.mlp_n_hidden,) * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False,
        )(x)

        lengths = e3nn.norm(vectors).array
        x = layers.SmoothingEnvelope(self.envelope_p)(lengths) * x
        assert x.shape == (num_edges, self.mlp_n_hidden)

        V = e3nn.haiku.Linear(irreps_out)(V)
        assert V.shape == (num_edges, V.irreps.dim)
        print(f"Irreps after layer are {V.irreps}")

        return (x, V)


class AllegroQeqLayer(hk.Module):

    def __init__(self,
                 epsilon: float,
                 irreps_out: e3nn.Irreps,
                 max_ell: int = 3,
                 charge_embed_n_hidden: int = 16,
                 charge_embed_n_layers: int = 1,
                 max_radius: float = 1.0,
                 num_species: int = 100,
                 mlp_n_hidden: int = 32,
                 mlp_n_layers: int = 3,
                 p: int = 6,
                 mlp_activation: Callable = None,
                 ):
        super().__init__()

        self.epsilon = epsilon
        self.envelope_p = p

        self.charge_embed_dim = charge_embed_n_hidden
        self.charge_embed_layers = charge_embed_n_layers
        self.max_radius = max_radius

        self.mlp_n_hidden = mlp_n_hidden

        self.mlp_activation = mlp_activation
        self.mlp_n_layers = mlp_n_layers

        self.radius = hk.get_parameter("radius", (num_species,), jnp.float32, init=hk.initializers.Constant(0.0))
        self.hardness = hk.get_parameter("hardness", (num_species,), jnp.float32, init=hk.initializers.Constant(10.0))

        self.chi_shift_scale = ScaleShiftLayer(1.0, False)
        self.gammas_shift_scale = ScaleShiftLayer(4.0, 0.5)



        self.charge_embed = hk.get_parameter(
            "charge_embed", (num_species, charge_embed_n_hidden),
            jnp.float32, init=hk.initializers.RandomNormal(1.0)
        )

        self.sigma_shift_scale = ScaleShiftLayer(0.001, 0.01)
        self.eps_shift_scale = ScaleShiftLayer(0.00001, 0.01)

        #self.sigma = hk.get_parameter("sigma", (num_species, charge_embed_n_hidden), jnp.float32, init=hk.initializers.Constant(0.3))
        #self.eps = hk.get_parameter("eps", (num_species, charge_embed_n_hidden), jnp.float32, init=hk.initializers.Constant(0.2))


    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        x: jnp.ndarray,  # [n_edges, features]
        V: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges]
        species: jnp.ndarray,
        charge_fn: Callable,
        vdw_fn: Callable,
        exclude_correction: bool = False,
    ) -> e3nn.IrrepsArray:
        lengths = jnp.sum(vectors.array ** 2, axis=-1) ** 0.5

        # Calculate charges via EEM method.
        chis = e3nn.haiku.MultiLayerPerceptron(
            [self.charge_embed_dim] * self.charge_embed_layers + [1,],
            self.mlp_activation,
            output_activation=False
        )(x)
        chis = jax.ops.segment_sum(chis, senders, species.size).squeeze()
        chis = self.chi_shift_scale(chis)

        # Like for 4GNN
        # gammas = jax_nn.softplus(self.radius[species]) / jnp.log(2.0)
        # gammas = 1 + jnp.log(jnp.sqrt(self.radius[species] ** 2 + 1) + self.radius[species]) / 10
        #gammas = (self.radius[species] ) / jnp.log(2.0)
        gammas = self.radius[species]
        #gammas = jax.nn.softplus(gammas)
        hardness = jax_nn.softplus(self.hardness[species])
        """gammas = e3nn.haiku.MultiLayerPerceptron(
            [16]  + [1,],
            self.mlp_activation,
            output_activation=False
        )(x)
        gammas = jax.ops.segment_sum(gammas, senders, species.size).squeeze()"""
        gammas = self.gammas_shift_scale(gammas)

        # Charge equilibration
        #debug.print("hardness: {}", hardness)
        #debug.print("gammas: {}", gammas)
        #debug.print("chis: {}", chis)
        charges, pot = charge_fn(gammas, chis, hardness)
        #debug.print("charges: {}", charges)


        sigma = e3nn.haiku.MultiLayerPerceptron(
            [32] * 1 + [1,],
            self.mlp_activation,
            output_activation=False
        )(x)


        eps = e3nn.haiku.MultiLayerPerceptron(
            [32] * 1 + [1,],
            self.mlp_activation,
            output_activation=False
        )(x)

        sigma = jax.ops.segment_sum(sigma, senders, species.size).squeeze()
        
        #sigma = self.sigma_shift_scale(sigma)
        #sigma = jax_nn.softplus(sigma)
        sigma = jax_nn.sigmoid(sigma)

        sigma = sigma*0.15 + 0.15
        eps = jax.ops.segment_sum(eps, senders, species.size).squeeze()
        
        #eps = self.eps_shift_scale(eps)
        #eps = jax_nn.softplus(eps)

        eps = jax_nn.sigmoid(eps)
        eps = eps * 1.7 + 0.3

        vdw = vdw_fn(sigma,eps)



        w = e3nn.haiku.MultiLayerPerceptron(
            [self.charge_embed_dim] * self.charge_embed_layers,
            self.mlp_activation,
            output_activation=False
        )(
            jnp.concatenate((
                charges[:, jnp.newaxis],
                self.charge_embed[species]), axis=-1
            )
        )

        x = e3nn.haiku.MultiLayerPerceptron(
            [self.mlp_n_hidden] * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False
        )(jnp.concatenate([x, w[senders]], axis=-1))
        x = layers.SmoothingEnvelope(self.envelope_p)(lengths)[:, None] * x

        return (x, V), (charges, pot), vdw


allegro_default_kwargs = OrderedDict(
    embed_dim = 32,
    output_irreps = "2x0e", # Reduce the output trough two linear layers
    hidden_irreps =  128 * e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e"),
    mlp_n_hidden=64,
    mlp_n_layers=2,
    max_ell = 3,
    n_radial_basis = 8,
    envelope_p = 6,
    num_layers=1,
    num_species=100,
    charge_embed_dim=16,
    max_radius=0.5,
    min_radius=0.01
)


def allegro_neighborlist_pp(displacement: space.DisplacementFn,
                            r_cutoff: float,
                            coulomb_cutoff: float = 1.0,
                            n_species: int = 100,
                            positions_test: jnp.ndarray = None,
                            neighbor_test: partition.NeighborList = None,
                            max_edge_multiplier: float = 1.25,
                            max_edges=None,
                            avg_num_neighbors: float = None,
                            mode: str = "energy",
                            per_particle: bool = False,
                            positive_species: bool = False,
                            exclude_correction: bool = False,
                            electrostatics: str = 'pme',
                            solver: str = 'direct',
                            exclude_electrostatics: bool = False,
                            grid=None,
                            alpha: float = 0.0,
                            max_local=None,
                            fractional_coordinates: bool = True,
                            coulomb_onset: float = None,
                            box=None,
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

    if coulomb_onset is None:
        coulomb_onset = 0.8 * coulomb_cutoff
    else:
        print(f"Set coulomb onset to {coulomb_onset}")

    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    assert not per_particle, "Per-particle energies not yet implemented."

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
    assert max_edges is not None, (
        "Requires maximum number of edges within NN cutoff."
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

        charge_eq_energy = custom_electrostatics.charge_eq_energy_neighborlist(
            dyn_displacement, r_onset=coulomb_onset, r_cutoff=coulomb_cutoff,
            electrostatics=electrostatics, solver=solver, grid=grid, alpha=alpha,
             fractional_coordinates=fractional_coordinates,
            box=box
        )

        
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

        # Remove all edges between replicated atoms
        if "reps" in dynamic_kwargs.keys():
            print(f"Remove replicated senders in neighbor list")
            invalid_idx = jnp.sum(mask) // dynamic_kwargs["reps"]
        else:
            invalid_idx = position.shape[0]

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        vectors = jnp.where(
            jnp.logical_and(
                senders < invalid_idx,
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff)
        vectors /= r_cutoff
        # Sort vectors by length and remove up to max_edges edges
        lengths = jnp.linalg.norm(vectors, axis=-1)
        sort_idx = jnp.argsort(lengths)
        vectors = vectors[sort_idx][:max_edges]
        senders = senders[sort_idx][:max_edges]
        receivers = receivers[sort_idx][:max_edges]

        vectors = e3nn.IrrepsArray(
            e3nn.Irreps("1o"), vectors
        )

        net = AllegroQeq(
            avg_num_neighbors=avg_num_neighbors,
            max_ell=kwargs["max_ell"],
            irreps=kwargs["hidden_irreps"],
            mlp_n_hidden = kwargs["mlp_n_hidden"],
            mlp_n_layers = kwargs["mlp_n_layers"],
            envelope_p=kwargs["envelope_p"],
            n_radial_basis=kwargs["n_radial_basis"],
            # output_irreps=kwargs["output_irreps"],
            output_irreps="1x0e",
            num_layers=kwargs["num_layers"],
            embed_dim=kwargs["embed_dim"],
            num_species=n_species,
            charge_embed_n_hidden=kwargs["charge_embed_dim"],
            charge_embed_n_layers=kwargs["charge_embed_layers"],
            mlp_activation=jax_nn.mish,
            exclude_electrostatics=exclude_electrostatics,
        )

        vdw_neighbor_fn, vdw_energy = lennard_jones_neighbor_list(
            dyn_displacement, box_size=box, r_onset=coulomb_onset, 
            r_cutoff=coulomb_cutoff,  
            fractional_coordinates=fractional_coordinates)
        debug.print("neighbor: {neighbor}", neighbor=neighbor)
        debug.print("position: {position}", position=position)

        def vdw_fn_old(sigma, eps):
            """Van der Waals energy function."""
            sigma2 = sigma *mask #* 0.1
            eps2 = eps *mask #* 0.1
            debug.print("sigma: {}", sigma2)
            debug.print("eps: {}", eps2)
            
            
            #vdw_neighbor = vdw_neighbor_fn.allocate(position)
            #vdw_neighbor_fn 
            #vdw_nbrs = vdw_neighbor_fn(position)

            vdw_pot = vdw_energy(position, neighbor,
            species=species,sigma=sigma2, 
            epsilon=eps2, mask=mask, **dynamic_kwargs)

            return vdw_pot

        def vdw_fn(eps_ij, sig_ij, vdw_onset=coulomb_onset, vdw_cutoff=coulomb_cutoff):
            R, nbrs = position, neighbor
            eps_ij = eps_ij*mask
            #sig_ij = sig_ij *mask


            def energy_i(i):
                Ri = R[i]
                nbr_ids = nbrs.idx[i]  # shape (n_max_neighbors,)

                Rj = R[nbr_ids]
                valid = mask[i]

                dR = jax.vmap(dyn_displacement, in_axes=(None, 0))(Ri, Rj)  # shape (n_max_neighbors, 3)
                debug.print("dR: {dR}", dR=dR)

                dR = jnp.where(
                    jnp.logical_and(
                        i < invalid_idx,
                        nbr_ids < R.shape[0]
                    )[:, jnp.newaxis], dR, vdw_cutoff)

                dist = jnp.linalg.norm(dR, axis=-1)

                debug.print("dist: {dist}", dist=dist)

                m = dist > 1e-7
                dist = jnp.where(m, dist, 1e-7)

                eps = eps_ij[i]
                sig = sig_ij[i]

                inv_r = sig / dist
                inv_r6 = inv_r ** 6
                inv_r12 = inv_r6 ** 2
                lj_pair = 4 * eps * (inv_r12 - inv_r6)

                # Smooth switching
                switch = jnp.where(dist < vdw_onset, 1.0,
                        jnp.where(dist > vdw_cutoff, 0.0,
                        (vdw_cutoff - dist) / (vdw_cutoff - vdw_onset)))

                #energy = switch * lj_pair
                return jnp.sum(switch * lj_pair * valid )

            total = 0.5 * jnp.sum(jax.vmap(energy_i)(jnp.arange(R.shape[0])))
            return total
            

        def charge_qeq_fn(gammas, chis, hardness):
            assert "radius" in dynamic_kwargs.keys(), "Radius not in dynamic_kwargs."
            #debug.print("gammas: {}", gammas)

            if kwargs.get("learn_radius", False):
                gammas *= dynamic_kwargs["radius"]
                print(f"Learn radius")
            else:
                print("Fix the radius")
                gammas = dynamic_kwargs["radius"]

            # Replicate the outputs
            if "reps" in dynamic_kwargs:
                print(f"Replicate outputs in charge_qeq_fn")

                n_local = jnp.sum(mask) // dynamic_kwargs["reps"]
                tile_idx = jnp.mod(jnp.arange(position.shape[0]), n_local)

                gammas = gammas[tile_idx]
                chis = chis[tile_idx]
                hardness = hardness[tile_idx]
            #hardness = jax.nn.softplus(hardness) 
            #chis = jax.nn.softplus(chis) 
            #debug.print("gammas: {}", gammas)
            gammas = jax.nn.softplus(gammas) 

            #debug.print("gammas: {}", gammas)

            _, charges = charge_eq_energy(
                position, neighbor, gammas, chi=chis, idmp=hardness, mask=mask, **dynamic_kwargs
            )
            #debug.print("charges: {}", charges)


            # Add to account for effect of charges on total potential
            dcharge = dynamic_kwargs.get("dcharge", jnp.zeros_like(charges))
            charges += dcharge
            # Do not optimize hardness and gammas on energy and force (only
            # indirectly through charges)
            pot = charge_eq_energy(
                position, neighbor, jax.lax.stop_gradient(gammas), mask=mask,
                charge=charges, **dynamic_kwargs
            )

            return charges, pot

        features, qeq_features, vdw_pot = net(
            vectors, senders, receivers, species, charge_qeq_fn, vdw_fn,
            exclude_correction=exclude_correction
        )
        debug.print("vdw_pot: {vdw_pot}", vdw_pot=vdw_pot)
        
        if mode in ["energy", "energy_and_charge"]:
            per_edge_energies, = features.array.T

            per_node_energies = layers.high_precision_segment_sum(
                per_edge_energies, senders, num_segments=position.shape[0])

            # Replicate the outputs
            if "reps" in dynamic_kwargs:
                print(f"Replicate potential outputs")

                n_local = jnp.sum(mask) // dynamic_kwargs["reps"]
                tile_idx = jnp.mod(jnp.arange(position.shape[0]), n_local)
                per_node_energies = per_node_energies[tile_idx]

            if exclude_correction:
                per_node_energies = jnp.zeros_like(per_node_energies)
            

            per_atom_energies = AtomicEnergyLayer(n_species)(per_node_energies, species)
            per_atom_energies *= mask

            if exclude_electrostatics:
                total_pot = md_util.high_precision_sum(per_atom_energies)
                charges = jnp.zeros_like(mask, dtype=jnp.float32)
            else:
                charges, elec_pot = qeq_features
                debug.print("elec_pot: {elec_pot}", elec_pot=elec_pot)
                total_pot = elec_pot + md_util.high_precision_sum(per_atom_energies) + vdw_pot

            if mode == "energy_and_charge":
                return total_pot, charges
            else:
                return total_pot

        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)


