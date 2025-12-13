"""Custom implementation of Allegro-Jax.
"""
import functools
from copy import deepcopy
from typing import Callable,  Any, Tuple,  Optional, Union
from collections import OrderedDict
from typing import Sequence
import math

import haiku as hk
import jax
from jax import ops
from jax import  numpy as jnp, nn as jax_nn, tree_util
from jax_md import space, partition, nn
import e3nn_jax as e3nn

import numpy as onp
from jax import debug
#from jax import nn
from jax_md_mod.model import  sparse_graph
from jax_md_mod import custom_electrostatics
from jax_md import util as md_util
from .utils import safe_norm
from .layers import (
    EquivariantProductBasisLayer,
    InteractionLayer,
    LinearNodeEmbeddingLayer,
    LinearReadoutLayer,
    NonLinearReadoutLayer
)
from jax_md_mod.model.layers import (
    RadialBesselLayer,
    SmoothingEnvelope
)
import flax.linen as flaxnn
from chemutils.models.layers import AtomicEnergyLayer
from .fast_attention import EuclideanFastAttention
import e3x

efa_default_kwargs = OrderedDict(

    era_use_in_iterations=[0,1],  # Apply EFA at given layers
    num_features_efa=128,
    era_lebedev_num=50,
    era_include_pseudotensors=False,
    era_tensor_integration=False,
    self_interaction=False,
    era_max_degree=0,
    era_qk_num_features=16,
    era_num_frequencies=8,
    era_v_num_features=32,
    era_max_frequency=jnp.pi,
    era_max_length=0.5,
    atomic_dipole_embedding=False,
    era_ti_parametrize_coupling_paths=False,
    era_ti_degree_scaling_constants=None,
    era_ti_max_degree_sph=None,
    era_ti_max_degree=None,
    last_layer_kernel_init=jax.nn.initializers.zeros

)

class EFAIntegrationBlock(flaxnn.Module):
    num_features: int
    era_use_in_iterations: list[int] | None = None
    emulate_era_block: bool = False
    era_lebedev_num: int = 86
    era_qk_num_features: int = 64
    era_v_num_features: int = 64
    era_activation_fn: callable = e3x.nn.silu
    era_num_frequencies: int = 8
    era_max_frequency: float = 2.0
    era_max_length: float = 0.5,
    era_tensor_integration: bool = True
    era_ti_max_degree_sph: int = 1
    era_ti_max_degree: int = 1
    era_max_degree: int = 0
    era_ti_parametrize_coupling_paths: bool = False
    era_ti_degree_scaling_constants: str = "2**-degree"
    last_layer_kernel_init: callable = jax.nn.initializers.zeros


    @flaxnn.compact
    def __call__(self, species_emb, positions, batch_segments, graph_mask, i):
        # Embedding species
          
        x = species_emb
        x = x[:, None, None, :]  # reshape as needed for EFA  (N, 1, 1, num_features)

        if self.emulate_era_block:
            y_nl = e3x.nn.Dense(self.num_features)(x)
        else:
            efa_input = e3x.nn.change_max_degree_or_type(
                x, max_degree=self.era_max_degree, include_pseudotensors=False
            )

            y_nl = EuclideanFastAttention(
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
            )(efa_input, positions, batch_segments, graph_mask)

        
        return y_nl




class MACE(hk.Module):
    def __init__(
        self,
        
        output_irreps: e3nn.Irreps,  # Irreps of the output, default 1x0e
        num_interactions: int,  # Number of interactions (layers), default 2
        hidden_irreps: e3nn.Irreps,  # 256x0e or 128x0e + 128x1o
        readout_mlp_irreps: e3nn.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        avg_num_neighbors: float,
        num_species: int,
        num_features: int = None,  # Number of features per node, default gcd of hidden_irreps multiplicities
        n_radial_basis: int = 8,  # Number of radial basis functions
        envelope_p: int = 6,  # Order of the envelope polynomial
        max_ell: int = 3,  # Max spherical harmonic degree, default 3
        epsilon: Optional[float] = None,
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        activation: Callable = jax.nn.silu,  # activation function
        gate: Callable = jax.nn.sigmoid,  # gate function
        soft_normalization: Optional[float] = None,
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        interaction_irreps: Union[str, e3nn.Irreps] = "o3_restricted",  # or o3_full
        skip_connection_first_layer: bool = False,
        
        

        #============ EFA Integration Parameters ============
                era_use_in_iterations: list[int] | None = None,
                emulate_era_block: bool = False,
                era_lebedev_num: int = 86,
                era_qk_num_features: int = 64,
                era_v_num_features: int = 64,
                era_activation_fn: callable = e3x.nn.gelu,
                era_num_frequencies: int = 8,
                era_max_frequency: float = 2.0,
                num_features_efa: int = 64,
                era_max_length: float = 0.5,
                era_tensor_integration: bool = True,
                era_ti_max_degree_sph: int = 1,
                era_ti_max_degree: int = 1,
                era_ti_parametrize_coupling_paths: bool = False,
                era_ti_degree_scaling_constants: Optional[Sequence[float]] = None,
                last_layer_kernel_init: callable = flaxnn.initializers.zeros,
                era_include_pseudotensors: bool = False,
                self_interaction: bool = False,
                atomic_dipole_embedding: bool = False,
                era_max_degree: int = 0,
                position: jnp.ndarray = None,
                species: jnp.ndarray = None,
                 *args, **kwargs,


    ):
        super().__init__()

        output_irreps = e3nn.Irreps(output_irreps)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)



        if num_features is None:
            self.num_features = functools.reduce(
                math.gcd, (mul for mul, _ in hidden_irreps)
            )
            self.hidden_irreps = e3nn.Irreps(
                [(mul // self.num_features, ir) for mul, ir in hidden_irreps]
            )
        else:
            self.num_features = num_features
            self.hidden_irreps = hidden_irreps

        if interaction_irreps == "o3_restricted":
            self.interaction_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        elif interaction_irreps == "o3_full":
            self.interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(max_ell))
        else:
            self.interaction_irreps = e3nn.Irreps(interaction_irreps)

        self.correlation = correlation
        self.epsilon = epsilon
        self.readout_mlp_irreps = readout_mlp_irreps
        self.activation = activation
        self.gate = gate
        self.num_interactions = num_interactions
        self.output_irreps = output_irreps
        self.num_species = num_species
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal
        self.max_ell = max_ell
        self.soft_normalization = soft_normalization
        self.skip_connection_first_layer = skip_connection_first_layer


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
        self.num_features_efa = num_features_efa
        self.era_include_pseudotensors = era_include_pseudotensors
        self.self_interaction = self_interaction
        self.atomic_dipole_embedding = atomic_dipole_embedding
        self.era_max_degree = era_max_degree

        epsilon = hk.get_parameter(
            "varepsilon", shape=(),
            init=hk.initializers.Constant(jnp.sqrt(avg_num_neighbors))
        )
        self.epsilon = 1 / jnp.sqrt(1 + epsilon ** 2)

        # Embeddings
        self.node_embedding = LinearNodeEmbeddingLayer(
            self.num_species, self.num_features * self.hidden_irreps
        )
        self.radial_embedding = RadialBesselLayer(
            cutoff=1.0, num_radial=n_radial_basis, envelope_p=envelope_p
        )
        self.envelope_p = envelope_p

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        positions: jnp.ndarray,  # [n_nodes,3] ?
        mask: jnp.ndarray,  # [n_nodes]
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
        is_training: bool = False,

    ) -> e3nn.IrrepsArray:
        assert vectors.ndim == 2 and vectors.shape[1] == 3
        assert node_species.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)

        # Embeddings
        node_feats = self.node_embedding(node_species).astype(vectors.dtype)  # [n_nodes, feature * irreps]
        rbf = self.radial_embedding(safe_norm(vectors.array, axis=-1))

        # Interactions
        outputs = []
        node_outputs = None
        
        for i in range(self.num_interactions):
            first = i == 0
            last = i == self.num_interactions - 1
            if i in self.era_use_in_iterations:
                use_efa = True
            else:
                use_efa = False



            hidden_irreps = (
                self.hidden_irreps
                if not last
                else self.hidden_irreps.filter(self.output_irreps)
            )

            node_outputs, node_feats = MACELayer(
                first=first,
                last=last,
                use_efa=use_efa,
                num_features=self.num_features,
                interaction_irreps=self.interaction_irreps,
                hidden_irreps=hidden_irreps,
                max_ell=self.max_ell,
                epsilon=self.epsilon,
                activation=self.activation,
                gate=self.gate,
                num_species=self.num_species,
                correlation=self.correlation,
                output_irreps=self.output_irreps,
                readout_mlp_irreps=self.readout_mlp_irreps,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                soft_normalization=self.soft_normalization,
                skip_connection_first_layer=self.skip_connection_first_layer,
                name=f"layer_{i}",
                era_use_in_iterations = self.era_use_in_iterations,
                emulate_era_block = self.emulate_era_block,
                era_lebedev_num = self.era_lebedev_num,
                era_qk_num_features = self.era_qk_num_features,
                era_v_num_features = self.era_v_num_features,
                era_activation_fn = self.era_activation_fn,
                era_num_frequencies = self.era_num_frequencies,
                era_max_frequency = self.era_max_frequency,
                era_max_length = self.era_max_length,
                era_tensor_integration = self.era_tensor_integration    ,    
                era_ti_parametrize_coupling_paths = self.era_ti_parametrize_coupling_paths,
                era_ti_max_degree_sph = self.era_ti_max_degree_sph,
                era_ti_max_degree = self.era_ti_max_degree,
                era_ti_degree_scaling_constants = self.era_ti_degree_scaling_constants,
                last_layer_kernel_init = self.last_layer_kernel_init,
                num_features_efa =self. num_features_efa,
                era_include_pseudotensors = self.era_include_pseudotensors,
                self_interaction = self.self_interaction,
                atomic_dipole_embedding = self.atomic_dipole_embedding,
                era_max_degree = self.era_max_degree
            )(
                vectors,
                node_feats,
                node_species,
                rbf,
                senders,
                receivers,
                node_species,
                positions,  
                mask,
                node_mask,
                is_training,
            )

            outputs += [node_outputs]  # list of [n_nodes, output_irreps]

        
        return node_outputs # e3nn.stack(outputs, axis=1)  # [n_nodes, num_interactions, output_irreps]


class MACELayer(hk.Module):
    def __init__(
        self,
        *,
        first: bool,
        last: bool,
        use_efa: bool,
        num_features: int,
        interaction_irreps: e3nn.Irreps,
        hidden_irreps: e3nn.Irreps,
        activation: Callable,
        gate: Callable,
        num_species: int,
        epsilon: Optional[float],
        name: Optional[str],
        # InteractionBlock:
        max_ell: int,
        # EquivariantProductBasisBlock:
        correlation: int,
        symmetric_tensor_product_basis: bool,
        off_diagonal: bool,
        soft_normalization: Optional[float],
        # ReadoutBlock:
        output_irreps: e3nn.Irreps,
        readout_mlp_irreps: e3nn.Irreps,
        skip_connection_first_layer: bool = False,
                #============ EFA Integration Parameters ============
                era_use_in_iterations: list[int] | None = None,
                emulate_era_block: bool = False,
                era_lebedev_num: int = 64,
                era_qk_num_features: int = 64,
                era_v_num_features: int = 64,
                era_activation_fn: callable = e3x.nn.gelu,
                era_num_frequencies: int = 8,
                era_max_frequency: float = 2.0,
                num_features_efa: int = 64,
                era_max_length: float = 0.5,
                era_tensor_integration: bool = True,
                era_ti_max_degree_sph: int = 1,
                era_ti_max_degree: int = 1,
                era_ti_parametrize_coupling_paths: bool = False,
                era_ti_degree_scaling_constants: Optional[Sequence[float]] = None,
                last_layer_kernel_init: callable = flaxnn.initializers.zeros,
                era_include_pseudotensors: bool = False,
                self_interaction: bool = False,
                atomic_dipole_embedding: bool = False,
                era_max_degree: int = 0,
                position: jnp.ndarray = None,
                species: jnp.ndarray = None,

        
        
    ) -> None:
        super().__init__(name=name)

        # self.dropout = e3nn.haiku.Dropout(p=0.5)

        self.first = first
        self.last = last
        self.use_efa = use_efa
        self.num_features = num_features
        self.interaction_irreps = interaction_irreps
        self.hidden_irreps = hidden_irreps
        self.max_ell = max_ell
        self.activation = activation
        self.gate = gate
        self.num_species = num_species
        self.epsilon = epsilon
        self.correlation = correlation
        self.output_irreps = output_irreps
        self.readout_mlp_irreps = readout_mlp_irreps
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal
        self.soft_normalization = soft_normalization
        self.skip_connection_first_layer = skip_connection_first_layer
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
        self.num_features_efa = num_features_efa
        

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        species: jnp.ndarray,  # [n_nodes]
        positions: jnp.ndarray,  # [n_nodes,3] ?
        mask: jnp.ndarray,  # [n_nodes]
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling    
        is_training: bool = False,
    
    ):
        if node_mask is None:
            node_mask = jnp.ones(node_specie.shape[0], dtype=jnp.bool_)

        sc = None
        if self.use_efa :
            # Use refined node features from EFA
            
            species_emb = node_feats.filter(keep="0e").array   # [n_nodes, irreps]
            #scalar_irreps = e3nn.Irreps("0e")  # scalar invariant output

            # Map whatever irreps you have to 0e (invariant)
            #species_emb = e3nn.haiku.Linear(self.num_features_efa)(species_emb)
            
            num_graphs = 1
            batch_segments = jnp.repeat(jnp.arange(num_graphs), positions.shape[0])
            graph_mask = mask
            
            # 3. Create EFAIntegrationBlock
            efa_model = EFAIntegrationBlock(num_features=self.num_features_efa,  # Match edge feature dimension
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
                    name='EFAIntegrationBlock',)

            # Initialize
            key = jax.random.PRNGKey(0)
            variables = efa_model.init(key, species_emb, positions, batch_segments, graph_mask, 0)
            # Apply
            efa_node_features = efa_model.apply(variables, species_emb, positions, batch_segments, graph_mask, 0)
                
            #refined_node_features_flat = refined_node_features.reshape(refined_node_features.shape[0], -1)
            #refined_node_features = species_emb + e3nn.haiku.Linear([self.num_features_efa])(efa_node_features)#e3x.nn.add(e3x.nn.Dense(self.num_features)(refined_node_features), species_emb)
            
            # Same for senders/receivers
            #original_shape = species_emb.shape
            # Flatten all dims except the first (num_nodes)
            #refined_node_features = jnp.reshape(efa_node_features, (original_shape[0], -1))  # shape: (N, D)
            efa_node_features = efa_node_features.reshape(efa_node_features.shape[0], -1)

            refined_node_features = hk.Linear(self.num_features_efa)(efa_node_features)               # (N, num_features)
            refined_node_features = jax.nn.silu(refined_node_features)
            refined_node_features = hk.Linear(self.num_features_efa)(refined_node_features)  # Final (N, num_features)


            #node_feats = e3nn.concatenate([node_feats, refined_node_features]).simplify() 

        
        if not self.first or self.skip_connection_first_layer:
            sc = e3nn.haiku.Linear(
                self.num_features * self.hidden_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp",
            )(
                node_specie, node_feats
            )  # [n_nodes, feature * hidden_irreps]

            # print(f"Use dropout for hidden irreps")
            # sc = self.dropout(hk.next_rng_key(), sc, is_training=is_training)

        node_feats = InteractionLayer(
            target_irreps=self.num_features * self.interaction_irreps,
            epsilon=self.epsilon,
            max_ell=self.max_ell,
            activation=self.activation,
            
        )(
            vectors=vectors,
            node_feats=node_feats,
            radial_embedding=radial_embedding,
            receivers=receivers,
            senders=senders,
            
        )
        

        node_feats *= self.epsilon

        if self.first:
            # Selector TensorProduct
            node_feats = e3nn.haiku.Linear(
                self.num_features * self.interaction_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp_first",
            )(node_specie, node_feats)

        




        node_feats = EquivariantProductBasisLayer(
            target_irreps=self.num_features * self.hidden_irreps,
            correlation=self.correlation,
            num_species=self.num_species,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )(node_feats=node_feats, node_specie=node_specie)

        
        # node_feats = self.dropout(hk.next_rng_key(), node_feats, is_training=is_training)

        if self.soft_normalization is not None:

            def phi(n):
                n = n / self.soft_normalization
                return 1.0 / (1.0 + n * e3nn.sus(n))

            node_feats = e3nn.norm_activation(
                node_feats, [phi] * len(node_feats.irreps)
            )
        
        if sc is not None:
            node_feats = node_feats + sc  # [n_nodes, feature * hidden_irreps]

        
        if not self.last:
            if self.use_efa:
                node_outputs = LinearReadoutLayer(self.output_irreps)(
                jnp.concatenate([node_feats.array, refined_node_features], axis=-1)
            )  # [n_nodes, output_irreps]

            else:
                node_outputs = LinearReadoutLayer(self.output_irreps)(
                node_feats
            )
            
        else:  # Non linear readout for last layer
            if self.use_efa:
                node_outputs = NonLinearReadoutLayer(
                self.readout_mlp_irreps,
                self.output_irreps,
                activation=self.activation,
                gate=self.gate
            )(
                jnp.concatenate([node_feats.array, refined_node_features], axis=-1)
            )  # [n_nodes, output_irreps]
            else:
                node_outputs = NonLinearReadoutLayer(
                    self.readout_mlp_irreps,
                    self.output_irreps,
                    activation=self.activation,
                    gate=self.gate
                )(
                    node_feats
                )  # [n_nodes, output_irreps]

        return node_outputs, node_feats


def efa_mace_neighborlist_pp(displacement: space.DisplacementFn,
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
                         box=None,
                         fractional_coordinates: bool = True,
                         **mace_kwargs
                         ) -> Tuple[nn.InitFn, Callable[[Any, md_util.Array],
                                                               md_util.Array]]:
    """MACE EFA model for property prediction.

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
    kwargs = deepcopy(efa_default_kwargs)
    kwargs.update(mace_kwargs)

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
            print(f"[MACE] Use default species")
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        elif positive_species:
            species -= 1
        if mask is None:
            print(f"[MACE] Use default mask")
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        box = dynamic_kwargs.get("box")
        if fractional_coordinates:
            if box is None:
                box = jnp.eye(3, dtype=position.dtype)
            real_position = jnp.einsum('ij,mj->mi', box, position)
        else:
            real_position = position

        # Compute the displacements for all edges
        dyn_displacement = functools.partial(displacement, **dynamic_kwargs)



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

        net = MACE(
            avg_num_neighbors=avg_num_neighbors,
            num_species=n_species,
            position=real_position,
            species=species,

            **kwargs,
        )


        
        features = net(
            vectors, senders, receivers, species,  positions=real_position,  mask=mask)

        if mode in ["energy", "energy_and_charge"]:
            _ = dynamic_kwargs.pop("reps", None)

            per_atom_energies, = features.array.T
            per_atom_energies = AtomicEnergyLayer(n_species)(per_atom_energies, species)
            per_atom_energies *= mask


            total_pot =  md_util.high_precision_sum(per_atom_energies)
            charges = jnp.zeros_like(mask, dtype=jnp.float32)

            if per_particle:
                return per_atom_energies
            else:
                return total_pot, charges

        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)
