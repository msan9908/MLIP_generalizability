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

"""Utilities to create graphs for reference data."""

import functools

import jax
from jax import numpy as jnp, lax

from jax_md_mod import custom_partition
from jax_md import partition, space

from chemtrain import util

from typing import Tuple, Optional


def allocate_neighborlist(dataset,
                          displacement: space.DisplacementOrMetricFn,
                          box: space.Box,
                          r_cutoff: float,
                          capacity_multiplier: float = 1.0,
                          disable_cell_list: bool = True,
                          fractional_coordinates: bool = True,
                          format: partition.NeighborListFormat = partition.NeighborListFormat.Dense,
                          pairwise_distances: bool = True,
                          box_key: str = None,
                          mask_key: str = None,
                          reps_key: str = None,
                          batch_size: int = 1000,
                          init_kwargs: dict = None,
                          count_triplets: bool = False,
                          **static_kwargs) -> Tuple[partition.NeighborList,
                                                   Tuple[int, int, float, Optional[int]]]:
    """Allocates an optimally sized neighbor list.

    Args:
        dataset: A dictionary containing the dataset with key ``"R"`` for
            positions.
        displacement: A function `d(R_a, R_b)` that computes the displacement
            between pairs of points.
        box: Either a float specifying the size of the box, an array of
            shape `[spatial_dim]` specifying the box size for a cubic box in each
            spatial dimension, or a matrix of shape `[spatial_dim, spatial_dim]` that
            is _upper triangular_ and specifies the lattice vectors of the box.
        r_cutoff: A scalar specifying the neighborhood radius.
        capacity_multiplier: A floating point scalar specifying the fractional
            increase in maximum neighborhood occupancy we allocate compared with the
            maximum in the example positions.
        disable_cell_list: An optional boolean. If set to `True` then the neighbor
            list is constructed using only distances. This can be useful for
            debugging but should generally be left as `False`.
        fractional_coordinates: An optional boolean. Specifies whether positions
            will be supplied in fractional coordinates in the unit cube, :math:`[0, 1]^d`.
            If this is set to True then the `box_size` will be set to `1.0` and the
            cell size used in the cell list will be set to `cutoff / box_size`.
        format: The format of the neighbor list; see the :meth:`NeighborListFormat` enum
            for details about the different choices for formats. Defaults to `Dense`.
        pairwise_distances: Computes pairwise distances between every particles
            for every sample.
        box_key: The key in the dataset dictionary that contains the box. If
            not provided, uses the box argument.
        mask_key: The key in the dataset dictionary that contains the mask. If
            not provided, all particles are considered valid.
        batch_size: Evaluate multiple samples in parallel.
        init_kwargs: Keyword arguments passed to the neighbor list allocation,
            e.g., to specify a capacity multiplier.
        **static_kwargs: kwargs that get threaded through the calculation of
            example positions.

    Returns:
        Returns a neighbor list that fits the dataset.

    """

    # We use the masked neighbor list to avoid interference of masked particles
    # and required neighbor list capacity.
    neighbor_fn = custom_partition.masked_neighbor_list(
        displacement, box, r_cutoff, dr_threshold=0.0,
        capacity_multiplier=capacity_multiplier,
        disable_cell_list=disable_cell_list,
        fractional_coordinates=fractional_coordinates, format=format,
        **static_kwargs
    )

    assert pairwise_distances, (
        "Currently, this function only works when computing distances between "
        "all pairs of particles (``pairwise_distances=True``)."
    )

    @jax.jit
    def find_max_neighbors_and_edges(dataset):
        def number_of_neighbors(input):
            position, box, mask, reps = input

            if box is None:
                metric = space.canonicalize_displacement_or_metric(displacement)
            else:
                metric = space.canonicalize_displacement_or_metric(
                    functools.partial(displacement, box=box))

            pair_distances = space.map_product(metric)(position, position)

            # Find neighbors, discarding self-interactions and masked particles.
            is_neighbor = pair_distances <= r_cutoff
            is_neighbor = jnp.logical_and(
                is_neighbor, ~jnp.eye(is_neighbor.shape[0], dtype=jnp.bool_))

            # Invalid particles cannot receive or send edges.
            if mask is not None:
                is_neighbor = jnp.logical_and(is_neighbor, mask[jnp.newaxis, :])
                is_neighbor = jnp.logical_and(is_neighbor, mask[:, jnp.newaxis])

            # Remove all replicated receivers
            if reps is not None:
                print(f"Remove replicated senders")
                max_local = jnp.sum(mask) // reps
                include = max_local < jnp.arange(is_neighbor.shape[0])
                is_neighbor = jnp.where(include[:, jnp.newaxis], is_neighbor, False)

            # Sets the number of neighbors to 0 for masked particles
            neighbors = jnp.sum(is_neighbor, axis=1)
            if mask is not None:
                neighbors *= mask

            # Compute the number of triplets.
            # First, we evaluate whether the pair of nodes are connected by an
            # edge to the same node.
            ji, jk = jax.vmap(
                functools.partial(jnp.meshgrid, indexing="ij")
            )(is_neighbor, is_neighbor)

            extra_out = []
            if count_triplets:
                # We mask out pairs of identical edges.
                is_triplet = jnp.logical_and(ji, jk)
                is_triplet = jnp.logical_and(
                    is_triplet,
                    ~jnp.eye(is_triplet.shape[0], dtype=jnp.bool_)[jnp.newaxis, ...]
                )

                extra_out += [jnp.sum(is_triplet)]

            avg_neighbors = jnp.mean(neighbors)
            if mask is not None:
                avg_neighbors /= jnp.mean(mask)

            max_neighbors = jnp.max(neighbors)
            max_edges = jnp.sum(neighbors)

            return max_neighbors, max_edges, avg_neighbors, *extra_out

        # We find the sample with the maximum number of neighbors or edges
        return util.batch_map(
            number_of_neighbors,
            (
                dataset["R"],
                dataset.get(box_key),
                dataset.get(mask_key),
                dataset.get(reps_key)
            ),
            batch_size=batch_size
        )

    n_neighbors, n_edges, avg_neighbors, *extra = find_max_neighbors_and_edges(dataset)

    print(
        f"The dataset has max. {jnp.max(n_neighbors)} neighbors per particle "
        f"and max. {jnp.max(n_edges)} edges in total.")

    if format == partition.Dense:
        # The maximum neighbors per particle determine the capacity of the
        # neighbor list.
        sample_idx = jnp.argmax(n_neighbors)
    elif format == partition.Sparse:
        # The maximum number of edges determine the capacity of the neighbor list.
        sample_idx = jnp.argmax(n_edges)

    extra_out = []
    if count_triplets:
        n_triplets, = extra
        extra_out += [jnp.max(n_triplets)]

    if init_kwargs is None:
        init_kwargs = {}
    if box_key is not None:
        init_kwargs['box'] = jnp.asarray(dataset[box_key][sample_idx])
    if mask_key is not None:
        init_kwargs['mask'] = jnp.asarray(dataset[mask_key][sample_idx])

    nbrs_init = neighbor_fn.allocate(
        jnp.asarray(dataset["R"][sample_idx]), **init_kwargs)

    return nbrs_init, (n_neighbors.max(), n_edges.max(), avg_neighbors.mean(), *extra_out)
