# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exporting potential models to MLIR."""

import abc
import functools

import jax
from fontTools.misc.cython import returns
from jax import numpy as jnp, export, lax

from typing import Dict, NamedTuple, Any, List, Tuple, Callable

import jax_md_mod
from jax_md import util as md_util

from . import graphs, shape_util
from ._protobuf import model_pb2 as model_proto


class Exporter(metaclass=abc.ABCMeta):
    """Exports a potential model to an MLIR module.

    Usage:
        To export a potential model, subclass this class, select an appropriate
        graph type and define the energy function:

        ```python
            class LennardJonesExport(exporter.Exporter):

                graph_type = graphs.DeviceSparseNeighborList

                def energy_fn(self, pos, species, graph):

                neighbors = partition.NeighborList(
                    jnp.stack((graph.senders, graph.receivers)),
                    pos, None, None, graph.senders.size, partition.Sparse,
                    None, None, None
                )

                assert neighbors.idx.shape[0] == 2, "Wrong shape"

                apply_fn = custom_energy.customn_lennard_jones_neighbor_list(
                    lambda ra, rb, **kwargs: rb - ra, None, None,
                    sigma=3.165, epsilon=1.0, r_onset=4.0, r_cutoff=5.0,
                    initialize_neighbor_list=False
                )

                return apply_fn(pos, neighbors)

            mlir_module = LennardJonesExport().export()
        ```

    Attributes:
        graph_type: Specifies the required neighborhood representation and
            how to generate it from the input data.

    """

    # Use the default graph containing the full neighbor indices
    graph_type: graphs.NeighborList = graphs.SimpleSparseNeighborList

    num_mpl: int = 0

    mask: bool = False

    _symbols: List[str] = []
    _constraints: List[str] = []
    _init_fns: List[Callable] = []
    _proto: model_proto.Model = None

    @abc.abstractmethod
    def energy_fn(self, position, species, graph):
        """Computes the energy for positions and a graph representation.

        Args:
            position: (N, dim) Array of particle positions, including ghost
                atoms that are not within the local domain.
            species: (N) Array of atoms species.
            graph: Graph representation of the neighborhood around atoms.

        Returns:
            Must return an energy contribution associated to each particle.

        """
        pass

    @staticmethod
    @shape_util.define_symbols("n_atoms")
    def _define_position_shapes(n_atoms, **kwargs):
        shape_defs = (
            jax.ShapeDtypeStruct((n_atoms, 3), jnp.float32),
            jax.ShapeDtypeStruct((n_atoms,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        )

        return shape_defs

    def _add_shapes(self, init_fn):
        init_fn(self._symbols, self._constraints, self._init_fns)

    def _create_shapes(self):
        all_symbols = ",".join(self._symbols)
        symbols = {
            key: symb for key, symb in zip(
                self._symbols,
                export.symbolic_shape(all_symbols, constraints=self._constraints),
            )
        }
        shapes = []
        for init_fn in self._init_fns:
            shapes.extend(init_fn(**symbols))

        # Reset
        self._symbols, self._constraints, self._init_fns = [], [], []
        return shapes


    def _energy_fn(self, position, species, n_local, n_ghost, *graph_args):
        # Expects particles to be sorted by local, ghost, and padding atoms

        valid_mask = jnp.arange(position.shape[0]) < (n_local + n_ghost)
        ghost_mask = jnp.arange(position.shape[0]) < n_local

        graph, build_statistics = self.graph_type.create_from_args(
            position, species, ghost_mask, valid_mask, *graph_args)
        graph = lax.stop_gradient(graph)

        @functools.partial(jax.grad, has_aux=True)
        def force_and_aux(pos):
            if self.mask:
                per_atom_energies = self.energy_fn(pos, species, valid_mask, graph)
            else:
                per_atom_energies = self.energy_fn(pos, species, graph)

            assert per_atom_energies.shape == ghost_mask.shape, (
                f"Per particle energies have shape {per_atom_energies.shape}, "
                f"but should have shape {ghost_mask.shape}."
            )

            # Attention: Force is negative gradient of potential
            total_neg_energy = jnp.float32(-1.0) * md_util.high_precision_sum(
                per_atom_energies * valid_mask)
            local_energy = md_util.high_precision_sum(
                ghost_mask * per_atom_energies)

            # Differentiate w.r.t. the total potential in the box, but exclude
            # ghost atom contributions to the total potential
            aux = local_energy, *build_statistics
            return total_neg_energy, aux

        return force_and_aux(position)

    def export(self) -> None:
        """Exports the potential model to an MLIR module."""

        proto = model_proto.Model()

        # Hard-coded for now
        proto.neighbor_list.cutoff = 5.0
        proto.neighbor_list.num_mpl = self.num_mpl

        self.graph_type.set_properties(proto)

        # Using the ghost mask in the last layer we can compute correct forces
        # by accounting for their contribution to the gradient but
        # mask them out when we compute the total potential to not count
        # them double.
        self._add_shapes(self._define_position_shapes)
        self._add_shapes(self.graph_type.create_symbolic_input_format)

        shapes = self._create_shapes()

        exp: export.Exported = export.export(
            jax.jit(self._energy_fn), platforms=["cuda"])(*shapes)

        proto.mlir_module = exp.mlir_module()

        self._proto = proto

    def __str__(self):
        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )

        return str(self._proto)

    def save(self, file: str):
        """Saves the exported protobuffer to a file."""

        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )

        with open(file, "wb") as f:
            f.write(self._proto.SerializeToString())
