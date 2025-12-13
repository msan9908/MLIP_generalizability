"""Custom implementation of Allegro-Jax.
"""
import functools
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, Any, Tuple, List, Optional, Union, Iterable
from collections import OrderedDict
from typing import Sequence

import haiku as hk
import jax
import jaxopt
from jax import random, lax, numpy as jnp, nn as jax_nn, tree_util
from jax_md import space, partition, nn, util, energy, smap
import e3nn_jax as e3nn
import flax.linen as flaxnn

import numpy as onp
from jax import debug
#from jax import nn
from jax_md_mod.model import layers, sparse_graph
from jax_md_mod import custom_electrostatics
from jax_md import util as md_util
from jax_md_mod.model.layers import SmoothingEnvelope
from jaxopt import ProjectedGradient
import flax.linen as flaxnn
from chemutils.models.layers import AtomicEnergyLayer, ScaleShiftLayer
from sympy.integrals.intpoly import hyperplane_parameters
from .fast_attention_hk import EuclideanFastAttention
import e3x

allegro_default_kwargs = OrderedDict(
    embed_dim = 32,
    output_irreps = "2x0e", # Reduce the output trough two linear layers
    hidden_irreps =  128 * e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e"),
    mlp_n_hidden=64,
    mlp_n_layers=2,
    max_ell = 2,
    n_radial_basis = 8,
    envelope_p = 6,
    num_layers=1,
    num_species=200,
    max_radius=0.5,
    min_radius=0.01,
    era_use_in_iterations=[0,1],  # Apply EFA at first and third layers
    cutoff=1,
    num_features=128,
    num_iterations=3,
    era_lebedev_num=50,
    era_include_pseudotensors=False,
    era_tensor_integration=False,
    self_interaction=False,
    era_max_degree=0,
    era_qk_num_features=16,
    era_v_num_features=32,
    era_max_frequency=jnp.pi,
    era_max_length=1.5,
    atomic_dipole_embedding=False,

)

class EFAIntegrationBlock(hk.Module):
    def __init__(
        self,
        num_features: int,
        era_use_in_iterations: Optional[Sequence[int]] = None,
        emulate_era_block: bool = False,
        era_lebedev_num: int = 86,
        era_qk_num_features: int = 64,
        era_v_num_features: int = 64,
        era_activation_fn: Callable = e3x.nn.silu,
        era_num_frequencies: int = 8,
        era_max_frequency: float = 2.0,
        era_max_length: float = 1.5,
        era_tensor_integration: bool = True,
        era_ti_max_degree_sph: int = 1,
        era_ti_max_degree: int = 1,
        era_max_degree: int = 0,
        era_ti_parametrize_coupling_paths: bool = False,
        era_ti_degree_scaling_constants: str = "2**-degree",
        last_layer_kernel_init: Optional[Callable] = jax.nn.initializers.zeros,
        name: str = "EFAIntegrationBlock"
    ):
        super().__init__(name=name)
        self.num_features = num_features
        self.era_use_in_iterations = era_use_in_iterations
        self.emulate_era_block = emulate_era_block
        self.era_lebedev_num = era_lebedev_num
        self.era_qk_num_features = era_qk_num_features
        self.era_v_num_features = era_v_num_features
        self.era_activation_fn = era_activation_fn
        self.era_num_frequencies = era_num_frequencies
        self.era_max_frequency = era_max_frequency
        self.era_max_length = era_max_length
        self.era_tensor_integration = era_tensor_integration
        self.era_ti_max_degree_sph = era_ti_max_degree_sph
        self.era_ti_max_degree = era_ti_max_degree
        self.era_max_degree = era_max_degree
        self.era_ti_parametrize_coupling_paths = era_ti_parametrize_coupling_paths
        self.era_ti_degree_scaling_constants = era_ti_degree_scaling_constants
        self.last_layer_kernel_init = last_layer_kernel_init

    
    def __call__(self, species_emb, positions, batch_segments, graph_mask, i: int):
        x = species_emb
        x = x[:, None, None, :]  # (N, 1, 1, num_features)

        if self.emulate_era_block:
            y_nl = e3x.nn.Dense(self.num_features)(x)
        else:
            efa_input = e3x.nn.change_max_degree_or_type(
                x, max_degree=self.era_max_degree, include_pseudotensors=False
            )

            efa_module = EuclideanFastAttention(
                lebedev_num=self.era_lebedev_num,
                num_features_qk=self.era_qk_num_features,
                num_features_v=self.era_v_num_features,
                activation_fn=self.era_activation_fn,
                epe_num_frequencies=self.era_num_frequencies,
                epe_max_frequency=self.era_max_frequency,
                epe_max_length=self.era_max_length,
                tensor_integration=self.era_tensor_integration,
                ti_max_degree_sph=self.era_ti_max_degree_sph,
                ti_max_degree=self.era_ti_max_degree,
                ti_parametrize_coupling_paths=self.era_ti_parametrize_coupling_paths,
                ti_degree_scaling_constants=self.era_ti_degree_scaling_constants,
                name=f'EuclideanRopeAttention_{i}'
            )

            y_nl = efa_module(efa_input, positions, batch_segments, graph_mask)

        return y_nl
    

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
                #============ EFA Integration Parameters ============
                era_use_in_iterations: list[int] | None = None,
                emulate_era_block: bool = False,
                era_lebedev_num: int = 86,
                era_qk_num_features: int = 64,
                era_v_num_features: int = 64,
                era_activation_fn: callable = e3x.nn.gelu,
                era_num_frequencies: int = 8,
                era_max_frequency: float = 2.0,
                num_features: int = 64,
                era_max_length: float = 5.0,
                era_tensor_integration: bool = True,
                era_ti_max_degree_sph: int = 1,
                era_ti_max_degree: int = 1,
                era_ti_parametrize_coupling_paths: bool = False,
                era_ti_degree_scaling_constants: Optional[Sequence[float]] = None,
                last_layer_kernel_init: callable = hk.initializers.Constant(0.0),
                position: jnp.ndarray = None,
                species: jnp.ndarray = None,

                # ===================================================
                *args, **kwargs
    ):
        super().__init__(name=name)
        self.position = position
        self.species = species
        self.max_ell = max_ell
        self.irreps = irreps
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.envelope_p = envelope_p
        self.output_irreps = output_irreps
        self.num_layers = num_layers
        self.era_use_in_iterations = era_use_in_iterations
        self.emulate_era_block = emulate_era_block
        self.era_lebedev_num = era_lebedev_num
        self.era_qk_num_features = era_qk_num_features
        self.era_v_num_features = era_v_num_features
        self.era_activation_fn = era_activation_fn
        self.era_num_frequencies = era_num_frequencies
        self.era_max_frequency = era_max_frequency
        self.era_max_length = era_max_length
        self.era_tensor_integration = era_tensor_integration        
        self.era_ti_parametrize_coupling_paths = era_ti_parametrize_coupling_paths
        self.era_ti_max_degree_sph = era_ti_max_degree_sph
        self.era_ti_max_degree = era_ti_max_degree
        self.era_ti_degree_scaling_constants = era_ti_degree_scaling_constants
        self.last_layer_kernel_init = last_layer_kernel_init
        self.num_features = num_features

        epsilon = hk.get_parameter(
            "varepsilon", shape=(),
            init=hk.initializers.Constant(jnp.sqrt(avg_num_neighbors))
        )
        self.epsilon = 1 / jnp.sqrt(1 + epsilon ** 2)

        self.alpha = hk.get_parameter(
            "residual_alpha", shape=(), init=hk.initializers.Constant(1.0)
        )

        self.embedding = hk.Embed(num_species, embed_dim)
        


        #self.efn_embed = e3x.nn.Embed(num_embeddings=num_species, features=embed_dim)(species)
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
        positions: jnp.ndarray,  # [n_nodes,3] ?
        mask: jnp.ndarray,  # [n_nodes]
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
        
        species_emb = self.embedding(species)
        num_graphs = 1
        batch_segments = jnp.repeat(jnp.arange(num_graphs), positions.shape[0])
        graph_mask = mask
        
        # 3. Create EFAIntegrationBlock
        efa_node_features = EFAIntegrationBlock(num_features=self.num_features,  # Match edge feature dimension
                era_use_in_iterations=self.era_use_in_iterations,
                emulate_era_block=self.emulate_era_block,
                era_lebedev_num=self.era_lebedev_num,
                era_qk_num_features=self.era_qk_num_features,
                era_v_num_features=self.era_v_num_features,
                era_activation_fn=self.era_activation_fn,
                era_num_frequencies=self.era_num_frequencies,
                era_max_frequency=self.era_max_frequency,
                era_max_length=self.era_max_length,
                era_tensor_integration=self.era_tensor_integration,
                era_ti_max_degree_sph=self.era_ti_max_degree_sph,
                era_ti_max_degree=self.era_ti_max_degree,
                era_ti_parametrize_coupling_paths=self.era_ti_parametrize_coupling_paths,
                era_ti_degree_scaling_constants=self.era_ti_degree_scaling_constants,
                last_layer_kernel_init=self.last_layer_kernel_init,
                name='EFAIntegrationBlock',)(species_emb, positions, batch_segments, graph_mask, 0)

            # Initialize
        key = jax.random.PRNGKey(0)
        #variables = efa_model.init(key, species_emb, positions, batch_segments, graph_mask, 0)

        # Apply
        
             
        #refined_node_features_flat = refined_node_features.reshape(refined_node_features.shape[0], -1)
        refined_node_features = species_emb[:, None, None, :] + hk.Linear(self.num_features)(efa_node_features)#e3x.nn.add(e3x.nn.Dense(self.num_features)(refined_node_features), species_emb)
        # Same for senders/receivers
        original_shape = refined_node_features.shape
        # Flatten all dims except the first (num_nodes)
        refined_node_features = jnp.reshape(refined_node_features, (original_shape[0], -1))  # shape: (N, D)

        refined_node_features = hk.Linear(self.num_features)(refined_node_features)               # (N, num_features)
        refined_node_features = jax.nn.silu(refined_node_features)
        refined_node_features = hk.Linear(self.num_features)(refined_node_features)  # Final (N, num_features)


        send_feats = refined_node_features[senders]
        recv_feats = refined_node_features[receivers]

        """

        x = jnp.concatenate(
        [
            self.radial_basis(d),
            send_feats,
            recv_feats,
        ],
            axis=1,
        )"""

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
        #V = e3nn.concatenate([V, self.embedding(species[senders]), self.embedding(species[receivers])])
        V = e3nn.concatenate([V, send_feats, recv_feats])

        w = e3nn.haiku.MultiLayerPerceptron((V.irreps.num_irreps,),None)(x)
        V = w * V / V.irreps.num_irreps
        assert V.shape == (num_edges, V.irreps.dim)
        

        for layer_idx, irreps in enumerate(irreps_layers[1:]):
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

        x = e3nn.haiku.MultiLayerPerceptron((self.mlp_n_hidden,),None)(x)
       

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




def efa_neighborlist_pp(displacement: space.DisplacementFn,
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
                            exclude_correction: bool = False,
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
        print("vectors /= r_cutoff Works")
        # Sort vectors by length and remove up to max_edges edges
        lengths = jnp.linalg.norm(vectors, axis=-1)
        sort_idx = jnp.argsort(lengths)
        vectors = vectors[sort_idx][:max_edges]
        senders = senders[sort_idx][:max_edges]
        receivers = receivers[sort_idx][:max_edges]
        print("receivers Works")

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
            # output_irreps=kwargs["output_irreps"],
            output_irreps="1x0e",
            num_layers=kwargs["num_layers"],
            embed_dim=kwargs["embed_dim"],
            num_species=n_species,
            mlp_activation=jax_nn.mish,
            position=jnp.array(position, jnp.float32),
            mask=jnp.array(mask, jnp.bool),
            species=jnp.array(species, jnp.int32),
        )
        print("net Works")

        

        features = net(
            vectors, senders, receivers, species,  positions=position,  mask=mask)

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

            total_pot = md_util.high_precision_sum(per_atom_energies)
            charges = jnp.zeros_like(mask, dtype=jnp.float32)
            print(f"charges, elec_pot   work {total_pot}")
            #debug.print("charges: {}", charges)
            #debug.print("total_pot: {}", total_pot)

            if mode == "energy_and_charge":
                return total_pot, charges
            else:
                return total_pot, charges

        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)


