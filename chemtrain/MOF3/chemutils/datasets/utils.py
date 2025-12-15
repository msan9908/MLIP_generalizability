import functools

import numpy as onp

import jax
from jax import numpy as jnp, tree_util

from jax_md_mod.model import sparse_graph
from jax_md import partition


def show_progress(block_num, block_size, total_size):
    print(f"Progess: {round(block_num * block_size / total_size *100,2)} %", end="\r")


def estimate_edge_and_triplet_count(dataset, displacement_fn, r_cutoff=0.5, capacity_multiplier=1.25):
    """Iterates through all data of all splits and determines the minimum dimensions of the graph.

    Args:
        dataset: Dictionary containing multiple splits of the full dataset.
        displacement_fn: Function to compute displacement between particles.
        r_cutoff: Cutoff radius within particles are considered neighbors
        capacity_multiplier: Factor to estimate max. number of neighbors based
            on the computes number of neighbors.

    Returns:
        Returns arrays containing the maximum number of neighbors, edges, and
        triplets of the graph and a sufficiently large neighbor list for the
        given dataset.

    """

    @jax.jit
    def compute_and_check_overflow(position, neighbor, box):
        neighbor = neighbor.update(position, box=box)
        dynamic_displacement = functools.partial(displacement_fn, box=box)

        # Compute a graph and return the estimated number of weights and triplets
        graph, _ = sparse_graph.sparse_graph_from_neighborlist(
            dynamic_displacement, position, neighbor, r_cutoff
        )

        max_triplets = jnp.int32(jnp.ceil(graph.n_triplets))
        max_edges = jnp.int32(jnp.ceil(graph.n_edges))

        return max_edges, max_triplets, neighbor.did_buffer_overflow

    n_samples = 0
    for split in dataset.keys():
        n_samples += dataset[split]['box'].shape[0]

    all_max_neighbors, all_max_edges, all_max_triplets = onp.zeros((3, n_samples), dtype=int)

    neighbor = None
    overflow = False

    neighbor_fn = partition.neighbor_list(
        displacement_fn, 1.0, r_cutoff, capacity_multiplier=capacity_multiplier,
        disable_cell_list=False, dr_threshold=0.0, fractional_coordinates=True
    )

    def compute_and_check(position, neighbor, box, overflow):
        if neighbor is None or overflow:
            print(f"Re-compute the neighborlist for samples {idx} in split {split}")
            neighbor = neighbor_fn.allocate(position, box=box)

            return compute_and_check(position, neighbor, box, False)

        return *compute_and_check_overflow(position, neighbor, box), neighbor

    n_iter = 0
    for split in dataset.keys():
        for idx in range(dataset[split]['box'].shape[0]):
            box, position = dataset[split]['box'], dataset[split]['R']

            # assert onp.all(position <= 1.0), f"Fractional coordinates are wrong."

            max_edges, max_triplets, overflow, neighbor = compute_and_check(
                jnp.asarray(position[idx]), neighbor, jnp.asarray(box[idx]), overflow
            )

            all_max_edges[n_iter] = int(max_edges)
            all_max_triplets[n_iter] = int(max_triplets)
            all_max_neighbors[n_iter] = int(neighbor.idx.shape[1])


    return all_max_neighbors, all_max_edges, all_max_triplets, neighbor


def make_supercell(dataset, a=1, b=1, c=1, fractional=True):
    """Transforms the boxes of the dataset into supercells.

    Args:
        dataset: Dictionary containing multiple splits of the full dataset.
        a, b, c: Repetitions of the cells in the directions of the base
            vectors.
        fractional: Whether positions are given in fractional coordinates.

    Returns:
        Returns the dataset tiled into supercells.

    """

    assert fractional, "Not implemented in real space."

    # Move all particles into the original boxes
    for split in dataset.keys():
        dataset[split]["R"] = onp.mod(dataset[split]["R"], 1.0)

    # For all samples and all particles, we add all combinations of the multiples
    # of a, b, and c.
    @functools.partial(jax.vmap, in_axes=(0, None, None, None)) # All samples
    @functools.partial(jax.vmap, in_axes=(0, None, None, None)) # All particles
    @functools.partial(jax.vmap, in_axes=(None, None, None, 0)) # All c repetitions
    @functools.partial(jax.vmap, in_axes=(None, None, 0, None)) # All b repetitions
    @functools.partial(jax.vmap, in_axes=(None, 0, None, None)) # All a repetitions
    def tile_positions_and_forces(subset, a_off, b_off, c_off):
        subset["R"] += jnp.asarray([a_off, b_off, c_off])
        # We do not need to changes forces, returned as is
        return subset

    for split in dataset.keys():
        a_off = jnp.arange(a)
        b_off = jnp.arange(b)
        c_off = jnp.arange(c)

        # We also add the forces to replicate them
        tiled_subset = tile_positions_and_forces(
            {"R": dataset[split]["R"], "F": dataset[split]["F"]},
            a_off, b_off, c_off
        )

        # We now combine the replicas of the box into a single large one.
        # Additionally, we have to scale back into fractional coordinates
        n_samples = tiled_subset["R"].shape[0]
        dataset[split]["R"] = tiled_subset["R"].reshape((n_samples, -1, 3))
        dataset[split]["R"] *= jnp.asarray([1/a, 1/b, 1/c])
        dataset[split]["F"] = tiled_subset["F"].reshape((n_samples, -1, 3))

        # Virial normalized by volume is intensive, internal energy and box
        # are extensive

        dataset[split]["U"] *= a * b * c
        dataset[split]["box"] *= onp.asarray([a, b, c])[:, None]

    # Ensure that dataset entries are still numpy arrays
    return tree_util.tree_map(onp.asarray, dataset)