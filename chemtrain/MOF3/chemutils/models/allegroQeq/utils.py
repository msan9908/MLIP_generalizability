from functools import partial

import haiku as hk
import numpy as onp
import jax
from jax import random, lax, numpy as jnp
from jax_md import space, util, partition
import e3nn_jax as e3nn

from jax_md_mod.training import data_utils
from chemutils.models.allegro.allegro import Allegro


def allegro_neighborlist(config, dataset, fractional=False):
    """Initialize custom implementation of Allegro-Jax model and build energy function template.

    This function provides an interface for the Allegro haiku model to be used
    as a jax_md energy_fn. Analogous to jax_md energy_fns, the initialized
    Allegro energy_fn requires particle positions and a dense neighbor list as
    input - plus other dynamic kwargs, if applicable.

    From particle positions and neighbor list, the potential energy is computed.
    Due to the constant shape requirement of jit of the neighborlist in jax_md,
    the neighbor list contains many masked edges, i.e. pairwise interactions
    that only "fill" the neighbor list, but are set to 0 during computation.
    This translates to masked edges and triplets in the sparse graph
    representation.

    Args:
        config: Jax_md displacement function.
        dataset: Training dataset. Used for initialization of neighbor list and model parameters.
        species: Array encoding species information.
        fractional: Bool specifying whether to use fractional coordinates.

    Returns:
        A tuple of energy function, initial model parameters and neighbor list.
    """
    # Estimate per-particle shift
    if "U" in dataset["training"].keys():
        config["model"]["energy_shift"] = onp.mean(
            dataset["training"]["U"] / dataset["training"]["R"].shape[1])
    else:
        config["model"]["energy_shift"] = 0

    displacement_fn, _ = space.periodic_general(1.0, fractional_coordinates=fractional)

    # We estimate the maximum number of edges and triplets and also initialize
    # a sufficiently big neighbor list.
    max_neighbor, max_edges, max_triplets, nbrs_init = data_utils.estimate_edge_and_triplet_count(
        reduce_dataset(dataset), displacement_fn, r_cutoff=config["model"]["r_cutoff"], capacity_multiplier=1.25, fractional_coordinates=fractional
    )

    print(f"Estimated: "
          f"\tMax. neighbors: {max_neighbor.max()},"
          f"\tMax. edges: {max_edges.max()},"
          f"\tMax. triplets: {max_triplets.max()}")

    max_edges = int(max_edges.max() * config["model"]["edge_multiplier"])
    max_triplets = int(max_triplets.max() * config["model"]["edge_multiplier"] ** 2)

    config["model"]["max_edges"] = max_edges
    config["model"]["max_triplets"] = max_triplets
    config["model"]["max_neighbor"] = nbrs_init.idx.shape[1]

    pot_shift = config["model"].get("energy_shift")

    # Set up NN model
    r_init = jnp.asarray(dataset['training']['R'][0])
    box_init = jnp.asarray(dataset['training']['box'][0])

    key = random.PRNGKey(21)

    displacement_fn, shift_fn = space.periodic_general(
        box_init, fractional_coordinates=fractional)

    model_kwargs = config["model"].get("model_kwargs", {})
    default_kwargs = dict(
        max_ell=model_kwargs.get("max_ell", 3),
        irreps=model_kwargs.get("n_irreps", 128) * e3nn.Irreps(
            model_kwargs.get("irreps", "0o + 1o + 1e + 2e + 2o + 3o + 3e")),
        mlp_n_hidden=model_kwargs.get("hidden_dim", 1024),
        mlp_n_layers=model_kwargs.get("n_layer", 3),
        n_radial_basis=model_kwargs.get("n_radial_basis", 8),
        output_irreps=e3nn.Irreps("0e"),
        num_layers=model_kwargs.get("num_layers", 1),  # 3,
        p=model_kwargs.get("p", 6),
    )
    species = dataset["training"]["species"][0]
    n_species = jnp.unique(species).shape[0]

    @hk.transform
    def haiku_model(pos, neighbor, species, box=None, per_particle=False, export=False, **dynamic_kwargs):
        """Evaluates the Allegro model and predicts the potential energy.

        Args:
            pos: Jax_md state-position. (N_particles x dim) array of
                 particle positions.
            box: Simulation box.
            species: (N_particles,) Array encoding atom types. If None, assumes
                     all particles to belong to the same species.
            **dynamic_kwargs: Dynamic kwargs.

        Returns:
            Potential energy value of state.
        """
        allegro_model = Allegro(
            avg_num_neighbors=max_edges / r_init.shape[0],
            radial_cutoff=config["model"]["r_cutoff"],  # In nm
            **default_kwargs)

        # Create a neighbor list with maximum capacity first
        if neighbor.format == partition.Dense:
            dense_idx = neighbor.idx
            senders = jnp.arange(dense_idx.shape[0]).repeat(dense_idx.shape[1])
            receivers = dense_idx.ravel()
        else:
            senders, receivers = neighbor.idx

        # number of neighbors
        #neighbor_mask = dense_idx < pos.shape[0]
        #n_neighbors = jnp.sqrt(jnp.sum(neighbor_mask*1.0, axis=1)) + 1 - neighbor_mask

        # Sort the indices of the receivers (invalids will be last)
        # and only keep a pre-defined amount
        if not export:
            _, sorted_idx = lax.top_k(-receivers, max_edges)
            senders = senders[sorted_idx]
            receivers = receivers[sorted_idx]

        # Mask out all invalid neighbors
        mask = receivers < pos.shape[0]

        # Assemble the Allegro Haiku Model
        node_attrs = jax.nn.one_hot(species, n_species)

        # Only apply PBC if box is provided
        if box is None:
            displacements = pos[senders, :] - pos[receivers, :]
        else:
            displacements = jax.vmap(
                partial(displacement_fn, box=box)
            )(pos[senders, :], pos[receivers, :])
            displacements = jnp.where(mask[:, None], displacements, config["model"]["r_cutoff"])

        vectors = e3nn.IrrepsArray("1o", displacements)

        maybe_energy = allegro_model(node_attrs, vectors, senders, receivers).array
        maybe_energy = (maybe_energy.T * mask).T

        if per_particle:
            return jax.ops.segment_sum(maybe_energy, senders, pos.shape[0])[:, 0]
        else:
            return util.high_precision_sum(maybe_energy) #/(n_neighbors**0.5)

    init_params = haiku_model.init(key, r_init, nbrs_init, box=box_init, species=species)
    init_params['learnable_shift'] = 0.0

    def energy_fn_template(energy_params):

        def energy_fn(pos, neighbor, rng=None, **dynamic_kwargs):
            if 'species' in dynamic_kwargs.keys():
                species = dynamic_kwargs.pop('species')
            else:
                print("Use default species")
                species = jnp.asarray(species)
            assert 'box' in dynamic_kwargs.keys(), 'box not in dynamic_kwargs'

            if pot_shift is not None:
                shift = pot_shift * pos.shape[0]
            else:
                shift = 0.0

            if rng is not None:
                print(f"Are we trainig?")
                is_training = True
            else:
                rng = random.PRNGKey(21)
                is_training = False

            # Remove this parameter again from the dictionary
            params = {key: value for key, value in energy_params.items()
                      if key != "learnable_shift"}
            shift += energy_params.get('learnable_shift', 0.0) * pos.shape[0]

            gnn_energy = haiku_model.apply(params, rng, pos, neighbor, species, is_training=is_training, **dynamic_kwargs)

            # Disable the learned energy shift
            if config["model"].get("no_shift", False):
                return gnn_energy

            return gnn_energy + shift

        return energy_fn

    return energy_fn_template, init_params, nbrs_init, config

def reduce_dataset(dataset:dict[str, dict], max_samples: int=10, **kwargs) -> dict[str, dict]:
    if max_samples is not None:
        train = {key: arr[:max_samples] for key, arr in dataset["training"].items()}
        val = {key: arr[:max_samples] for key, arr in dataset["validation"].items()}
        test = {key: arr[:max_samples] for key, arr in dataset["testing"].items()}

    return {"training": train, "validation": val, "testing": test}