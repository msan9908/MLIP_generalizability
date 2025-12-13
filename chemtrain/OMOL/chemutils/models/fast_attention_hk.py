import e3x
from e3x.nn.modules import initializers

import numpy as np

import haiku as hk
import e3nn_jax as e3nn


import jax
import jax.numpy as jnp
import jaxtyping

from typing import Any, Callable, Optional, Sequence, Union
from . import tensor_integration
from . import rope


InitializerFn = initializers.InitializerFn
Array = jaxtyping.Array
Bool = jaxtyping.Bool
Float = jaxtyping.Float
Integer = jaxtyping.Integer
UInt32 = jaxtyping.UInt32
Shape = Sequence[Union[int, Any]]
Dtype = Any  # This could be a real type if support for that is added.
PRNGKey = UInt32[Array, '2']
PrecisionLike = jax.lax.PrecisionLike


def frequency_init_fn(
        rng, num_frequencies, num_features, max_frequency, max_length, dtype
):
    """Init function for Euclidean Rope frequencies.

    Args:
    rng: jax.PRNGKey
    num_frequencies: Number of frequencies.
    num_features: Number of features.
    max_frequency: Maximal frequency.
    max_length: Maximal length.
    dtype:

    Returns:
    Vector of frequency values from `[0, ..., max_frequency/max_length]`.
    """
    if num_features // 2 > 1:
        return (
                jnp.linspace(0, max_frequency, int(num_features / 2), dtype=dtype)
                / max_length
        )
    else:
        return jnp.array([max_frequency], dtype=dtype) / max_length



class EuclideanFastAttention(hk.Module):
    def __init__(
        self,
        lebedev_num: int = 6,
        parametrized: bool = True,
        num_features_qk: Optional[int] = None,
        max_degree_qk: Optional[int] = None,
        include_pseudotensors_qk: Optional[bool] = None,
        num_features_v: Optional[int] = None,
        max_degree_v: Optional[int] = None,
        include_pseudotensors_v: Optional[bool] = None,
        activation_fn: Optional[Callable[..., Any]] = lambda u: u,
        tensor_integration: bool = False,
        ti_max_degree_sph: Optional[int] = None,
        ti_include_pseudotensors: Optional[bool] = None,
        ti_max_degree: Optional[int] = None,
        ti_parametrize_coupling_paths: bool = False,
        ti_degree_scaling_constants: Optional[Sequence[float]] = None,
        epe_frequencies_init_fn: Optional[Callable[..., Any]] = None,
        epe_num_frequencies: Optional[int] = None,
        epe_max_frequency: Optional[float] = None,
        epe_max_length: Optional[float] = None,
        epe_frequencies_trainable: bool = False,
        param_dtype: jnp.dtype = jnp.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.lebedev_num = lebedev_num
        self.parametrized = parametrized
        self.num_features_qk = num_features_qk
        self.max_degree_qk = max_degree_qk
        self.include_pseudotensors_qk = include_pseudotensors_qk
        self.num_features_v = num_features_v
        self.max_degree_v = max_degree_v
        self.include_pseudotensors_v = include_pseudotensors_v
        self.activation_fn = activation_fn
        self.tensor_integration = tensor_integration
        self.ti_max_degree_sph = ti_max_degree_sph
        self.ti_include_pseudotensors = ti_include_pseudotensors
        self.ti_max_degree = ti_max_degree
        self.ti_parametrize_coupling_paths = ti_parametrize_coupling_paths
        self.ti_degree_scaling_constants = ti_degree_scaling_constants
        self.epe_frequencies_init_fn = epe_frequencies_init_fn
        self.epe_num_frequencies = epe_num_frequencies
        self.epe_max_frequency = epe_max_frequency
        self.epe_max_length = epe_max_length
        self.epe_frequencies_trainable = epe_frequencies_trainable
        self.param_dtype = param_dtype

        # Modules defined here will be called in __call__
        if self.parametrized:
            self.q_proj = e3x.nn.Dense(self.num_features_qk, dtype=param_dtype)
            self.k_proj = e3x.nn.Dense(self.num_features_qk, dtype=param_dtype)
            self.v_proj = e3x.nn.Dense(self.num_features_v, dtype=param_dtype)

    def __call__(self, inputs, positions, batch_segments, graph_mask):
        """
        Given equivariant input features and node positions, calculate a Euclidean fast attention update.
        Args:
            inputs (): (num_nodes, 1 or 2, (max_degree + 1)**2, num_features) - Equivariant node features
                The convention for equivariant features follows https://e3x.readthedocs.io/stable/index.html so
                check it out for a detailed introduction.
            positions (): (num_nodes, 3) - Node positions.
            batch_segments (): (num_nodes) - The batch a node belongs to. For example assume a batch of two
                molecules / graphs, where the first has 3 atoms and the second has 2 atoms. The batch_segments
                would then be [0, 0, 0, 1, 1]. Since JAX requires static shapes for jit compilation,
                batch_segments are usually padded towards a fixed length, such that the remaining entries
                are filled with a padding index. For a fixed batch size of 7 nodes, this yields
                batch_segments [0, 0, 0, 1, 1, 2, 2]. The corresponding graph_mask is then
                [True, True, False]. For details about graph batching see also
                https://e3x.readthedocs.io/stable/examples/md17_ethanol.html and
                https://jraph.readthedocs.io/en/latest/api.html#batching-padding-utilities.
            graph_mask (): (num_graphs) - Labels which graphs are "true" graphs and which are padded.
                I.e. for the batch_segments example from above, it would be
                [True, True] and [True, True, False].

        Returns:
            Updated features. Output shape depends on the specific settings, but will be the same shape as
            `inputs` for default settings.


        """

        max_degree_inputs = int(np.rint(np.sqrt(inputs.shape[-2]) - 1).item())

        # if no tensor integration is performed, max_degree_sph can not be set.
        if not self.tensor_integration:

            if self.ti_max_degree_sph is not None:
                raise ValueError(
                    'ti_max_degree_sph can only be set if tensor_integration=True. '
                    f'received {self.tensor_integration=} and {self.ti_max_degree_sph=}'
                )

            if self.ti_max_degree is not None:
                raise ValueError(
                    'ti_max_degree can only be set if tensor_integration=True. '
                    f'received {self.tensor_integration=} and {self.ti_max_degree=}'
                )

            if self.ti_include_pseudotensors is not None:
                raise ValueError(
                    'ti_include_pseudotensors can only be set if tensor_integration=True. '
                    f'received {self.tensor_integration=} and {self.ti_include_pseudotensors=}'
                )

            if self.ti_parametrize_coupling_paths:
                raise ValueError(
                    f'one can only parametrize coupling paths if tensor_integration=True. '
                    f'received {self.tensor_integration=} and {self.ti_parametrize_coupling_paths=}'
                )

        num_features_qk = (
            inputs.shape[-1]
            if self.num_features_qk is None
            else self.num_features_qk
        )

        num_features_v = (
            inputs.shape[-1]
            if self.num_features_v is None
            else self.num_features_v
        )

        if self.parametrized:
            q = e3nn.haiku.Linear(num_features_qk)(inputs) # (N, 1 or 2, (max_degree_in + 1)**2, qk_num_features)
            k = e3nn.haiku.Linear(num_features_qk)(inputs)  # (N, 1 or 2, (max_degree_in + 1)**2, qk_num_features)
            v = e3nn.haiku.Linear(num_features_v)(inputs)# (N, 1 or 2, (max_degree_in + 1)**2, qk_num_features)
        else:
            if self.num_features_qk is not None:
                raise ValueError(
                    "Down projection of query and key to `qk_num_features ="
                    f" {self.num_features_qk} is only possible with `parametrized ="
                    " True`."
                )
            if self.num_features_v is not None:
                raise ValueError(
                    "Down projection of value to `v_num_features ="
                    f" {self.num_features_v}` is only possible with `parametrized ="
                    " True`."
                )
            q = inputs  # (N, 1 or 2, (max_degree_in + 1)**2, num_features)
            k = inputs  # (N, 1 or 2, (max_degree_in + 1)**2, num_features)
            v = inputs  # (N, 1 or 2, (max_degree_in + 1)**2, num_features)

        # Lebedev grid.
        with jax.ensure_compile_time_eval():
            grid_u, grid_w = e3x.so3.lebedev_quadrature(
                num=self.lebedev_num
            )

        # If frequencies are trainable, initialize them as params.
        if self.epe_frequencies_trainable:
            frequencies = self.param(
                "frequencies",
                self.epe_frequencies_init_fn,
                self.epe_num_frequencies,
                num_features_qk,
                self.epe_max_frequency,
                self.epe_max_length,
                self.param_dtype,
            )
        # Otherwise just call the init function for the frequencies.
        else:
            frequencies = self.epe_frequencies_init_fn(
                None,  # no RNG key needed.
                self.epe_num_frequencies,
                num_features_qk,
                self.epe_max_frequency,
                self.epe_max_length,
                self.param_dtype,
            )

        # Perform the linear scaling attention aggregation.
        beta = rope.apply(
            q=self.activation_fn(q),
            k=self.activation_fn(k),
            v=v,
            pos=positions,
            theta=frequencies,
            grid_u=grid_u,
            grid_w=grid_w,
            batch_segments=batch_segments,
            graph_mask=graph_mask,
            include_pseudotensors_qk=self.include_pseudotensors_qk,
            include_pseudotensors_v=self.include_pseudotensors_v,
            max_degree_qk=self.max_degree_qk,
            max_degree_v=self.max_degree_v,
            # Determines values at grid points are present or already summed over.
            do_integration=not self.tensor_integration,
        )  # (N, M, P, L, F) or (N, P, L, F)

        # If no tensor integration is required, return beta which is already numerically integrated.
        if not self.tensor_integration:
            return beta

        # Perform tensor product integration. It first builds tensor product between beta and
        # the spherical harmonics expansion for all grid points and then sums over the grid points.
        else:
            # if tensor integration is performed, max_degree_sph is either set explicitly or set
            # to maximal input degree.
            if self.ti_max_degree_sph is None:
                ti_max_degree_sph = max_degree_inputs
            else:
                ti_max_degree_sph = self.ti_max_degree_sph

            # output degree is either set explicitly or set to maximal input degree.
            if self.ti_max_degree is None:
                ti_max_degree = max_degree_inputs
            else:
                ti_max_degree = self.ti_max_degree

            # include_pseudotensors
            if self.ti_include_pseudotensors is None:
                ti_include_pseudotensors = inputs.shape[-3] == 2
            else:
                ti_include_pseudotensors = self.ti_include_pseudotensors

            # Expand grid points in spherical harmonics basis.
            grid_u_sph = jnp.expand_dims(
                e3x.so3.spherical_harmonics(
                    grid_u,
                    max_degree=ti_max_degree_sph
                ),
                axis=(-3, -1)
            )  # (M, 1, (L_Y+1)**2), 1)

            if self.ti_degree_scaling_constants is not None:
                if len(self.ti_degree_scaling_constants) != ti_max_degree_sph + 1:
                    raise ValueError(
                        f'the number of constants in `ti_degree_scaling_constants` must equal the number '
                        f'of degrees in the spherical harmonics vector. '
                        f'received {self.ti_degree_scaling_constants=} and {ti_max_degree_sph=}.'
                    )

                repeats = np.array([2*o+1 for o in range(ti_max_degree_sph + 1)])
                c = jnp.array(self.ti_degree_scaling_constants)  # (L_Y+1, )
                c = jnp.repeat(c, repeats=repeats, total_repeat_length=repeats.sum())  # ((L_Y+1)**2, )

                grid_u_sph = grid_u_sph * c[None, None, :, None]

            # Calculate the tensor product between beta and the spherical harmonics expansion on the grid
            # and then integrate.
            return tensor_integration.TensorIntegration(
                include_pseudotensors=ti_include_pseudotensors,
                max_degree=ti_max_degree,
                parametrize_coupling_paths=self.ti_parametrize_coupling_paths
            )(
                jnp.repeat(grid_u_sph, axis=-1, repeats=beta.shape[-1]),
                beta,
                grid_w
            )  # (N, M, P, (L_out+1)**2, F)
