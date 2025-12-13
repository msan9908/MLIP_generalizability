import os
import sys
import argparse

from pathlib import Path
import pickle

import eval_utils


if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
jax.config.update("jax_debug_nans", True)

import numpy as onp
from typing import Union, Tuple, Callable
from jax import tree_util 

import jax.numpy as jnp
import jax

import jax.numpy as jnp

from jax.sharding import PartitionSpec as P

from jax_md_mod import io, custom_quantity, custom_space, custom_energy, custom_partition
from jax_md import simulate, partition, space, util, energy, quantity as snapshot_quantity

from collections import OrderedDict

from chemtrain.data import graphs
from chemtrain import quantity, trainers, util as chem_util
from chemtrain.trainers import ForceMatching, extensions
from chemtrain.quantity import property_prediction
from collections import Counter
from chemutils.datasets import utils as data_utils
import train_utils 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tomli

import json
import gzip
import pandas as pd
from ase.io import read
from pymatgen.core.structure import Structure
import time


def load_saved_datasets(save_path):
    """ Load the saved datasets, removing 'qmof_ids' key before returning. """
    
    # Convert loaded data into dictionaries
    
    dataset_small_raw = onp.load(save_path + 'small_molecules.npz', allow_pickle=True)

    # Load large molecules dataset
    dataset_large = onp.load(save_path + 'large_molecules.npz', allow_pickle=True)

    # Extract keys and create dataset_splits in the correct nested dictionary format
    dataset_splits = {
        'training': dict(dataset_small_raw['training'].item()),
        'validation': dict(dataset_small_raw['validation'].item()),
        'testing': dict(dataset_small_raw['testing'].item()),
    }
    dataset_large = {key: dataset_large[key] for key in dataset_large.keys() }
    
    print(f"Datasets loaded from {save_path}")
    
    return dataset_splits, dataset_large


def filter_dataset_by_species(reference_dataset, target_dataset):
    # Extract valid species from the reference dataset
    valid_species = set(onp.unique(reference_dataset['species'][reference_dataset['mask']]))

    # Prepare list to store filtered structures
    keep_indices = []

    for i in range(target_dataset['species'].shape[0]):
        species = target_dataset['species'][i][target_dataset['mask'][i]]
        if all(sp in valid_species for sp in species):
            keep_indices.append(i)

    # Filter all arrays in the target dataset
    filtered_dataset = {
        key: value[keep_indices] for key, value in target_dataset.items()
    }

    print(f"Kept {len(keep_indices)} out of {target_dataset['species'].shape[0]} structures.")
    return filtered_dataset


def process_dataset(dataset, charge=None, max_U=onp.inf):
    """Removes NaNs from charge data and creates weights for charge and energy."""

    for split in dataset.keys():
        # Weight the potential by the number of particles. The per-particle
        # potential should have equal weight, so the error for larger systems
        # should contribute more. On the other hand, since we compute the
        # MSE for the forces, we have to correct for systems with masked
        # out particles.

        corr = 1.0
        if 'reps' in dataset[split].keys():
            corr = onp.mean(dataset[split]['reps']) / dataset[split]['reps']


        n_particles = onp.sum(dataset[split]['mask'], axis=1)

        # Sort out samples with too high potential energy
        

        max_particles = dataset[split]['mask'].shape[1]

        weights = n_particles / onp.mean(n_particles, keepdims=True)
        dataset[split]['U_weight'] = weights * corr

        # Remove NaNs from the samples

        dataset[split]['charge_weight'] = max_particles / n_particles

    return dataset

def evaluate_model(model_dir: str, save_path: str, dataset_path: str, use_coulomb_nbrs: bool):
    def load_saved_datasets(path):
        dataset_small_raw = onp.load(os.path.join(path, 'small_molecules.npz'), allow_pickle=True)
        dataset_large = onp.load(os.path.join(path, 'large_molecules.npz'), allow_pickle=True)

        dataset_splits = {
            'training': dict(dataset_small_raw['training'].item()),
            'validation': dict(dataset_small_raw['validation'].item()),
            'testing': dict(dataset_small_raw['testing'].item()),
        }
        dataset_large = {key: dataset_large[key] for key in dataset_large.keys()}
        return dataset_splits, dataset_large

    def process_dataset(dataset):
        for split in dataset.keys():
            corr = 1.0
            if 'reps' in dataset[split].keys():
                corr = onp.mean(dataset[split]['reps']) / dataset[split]['reps']

            n_particles = onp.sum(dataset[split]['mask'], axis=1)
            max_particles = dataset[split]['mask'].shape[1]

            weights = n_particles / onp.mean(n_particles, keepdims=True)
            dataset[split]['U_weight'] = weights * corr
            dataset[split]['charge_weight'] = max_particles / n_particles
        return dataset

    def setup_model(model_dir, dataset, displacement_fn,use_coulomb_nbrs=False):
        """Setup the model for evaluation."""
        model = Path(model_dir)
        with open(model / "config.toml", "rb") as f:
            config = tomli.load(f)

        if config["model"]["type"]== "DimeNetPP":
            nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, dataset['training']['box'][0],
            config["model"]["r_cutoff"], mask_key="mask", box_key="box",
            format=partition.Dense, fractional_coordinates=True
        )
            
        else:
            nbrs_init, (max_neighbors, max_edges, avg_num_neighbors, max_triplets) = graphs.allocate_neighborlist(
                dataset["training"], displacement_fn, dataset['training']['box'][0],
                config["model"]["r_cutoff"], mask_key="mask", box_key="box",
                format=partition.Sparse, fractional_coordinates=True, capacity_multiplier=1.1,  count_triplets=True
            )
        max_neighbors = int(max_neighbors * 1.1)
        max_edges = int(max_edges * 1.1)

        if use_coulomb_nbrs and config["model"]["type"] != "DimeNetPP":
            nbrs_init, _ = graphs.allocate_neighborlist(
                dataset["training"], displacement_fn, dataset['training']['box'][0],
                config["model"]["coulomb_cutoff"], mask_key="mask", reps_key="reps",
                format=partition.Sparse, box_key="box", fractional_coordinates=True,
                capacity_multiplier=1.1,
            )

        energy_fn_template, init_params = train_utils.define_model(
            config, dataset, nbrs_init, max_edges, per_particle=False,
            avg_num_neighbors=avg_num_neighbors, positive_species=True,
            displacement_fn=displacement_fn,
            exclude_correction=config["optimizer"].get("exclude_correction", False),
            electrostatics='pme'
        )

        optimizer = train_utils.init_optimizer(config, dataset)

        energy_params = tree_util.tree_map(jnp.asarray,
            onp.load(model / "best_params.pkl", allow_pickle=True))

        trainer = ForceMatching(
            energy_params, optimizer, energy_fn_template, nbrs_init,
            batch_per_device=config["optimizer"]["batch"] // len(jax.devices()),
            batch_cache=config["optimizer"]["cache"],
            gammas=config["gammas"],
            energy_fn_has_aux=True,
            additional_targets={"charge": custom_quantity.get_aux("charge")},
            weights_keys={"U": "U_weight", "charge": "charge_weight"},
        )
        return energy_params, trainer

    # === Load and preprocess datasets ===
    dataset_small, dataset_large = load_saved_datasets(dataset_path)

    filter_dataset_by_species(dataset_small["training"], dataset_small["testing"])
    filter_dataset_by_species(dataset_small["training"], dataset_large)

    q_ids_small = {
        "training": dataset_small['training'].pop('qmof_ids'),
        "validation": dataset_small['validation'].pop('qmof_ids'),
        "testing": dataset_small['testing'].pop('qmof_ids'),
    }
    NA_small = {
        "training": dataset_small['training'].pop('original_num_atoms'),
        "validation": dataset_small['validation'].pop('original_num_atoms'),
        "testing": dataset_small['testing'].pop('original_num_atoms'),
    }

    q_ids_large = dataset_large.pop('qmof_ids')
    NA_large = dataset_large.pop('original_num_atoms')

    dataset_small = process_dataset(dataset_small)
    displacement_fn, _ = space.periodic_general(box=dataset_small['training']['box'][0], fractional_coordinates=True)

    # === Evaluate on small molecules ===
    _, model = setup_model(model_dir, dataset_small, displacement_fn,use_coulomb_nbrs=use_coulomb_nbrs)
    start_time = time.time()

    predictions = {
        split: model.predict(dataset_small[split], batch_size=100 // len(jax.devices()))
        for split in ['training', 'validation', 'testing']
    }

    stop = time.time()
    elapsed = stop - start_time
    print(f"Inference time small: {elapsed:.2f} s")


    for split in ['training', 'validation', 'testing']:
        pred, ref = eval_utils.unpadd_predictions(predictions[split], dataset_small[split])
        charge_diff, energy_diff = eval_utils.compute_errors(split.capitalize(), pred, ref)

        onp.savez_compressed(os.path.join(save_path, f"{split}_errors.npz"),
            charge_diff=onp.array(charge_diff, dtype=object),
            energy_diff=energy_diff,
            q_ids=q_ids_small[split],
            NA=NA_small[split]
        )

    # === Evaluate on large molecules ===
    dataset_large_full = {"training": dataset_large}
    dataset_large = {"training": {k: v[:400] for k, v in dataset_large.items()}}  # for setup

    dataset_large_full = process_dataset(dataset_large_full)
    _, model = setup_model(model_dir, dataset_large, displacement_fn)

    start_time = time.time()
    large_predictions = model.predict(dataset_large_full['training'], batch_size=20)
    stop = time.time()
    elapsed = stop - start_time 
    print(f"Inference time large: {elapsed:.2f} s")
    large_pred, large_ref = eval_utils.unpadd_predictions(large_predictions, dataset_large_full['training'])
    charge_diff_large, energy_diff_large = eval_utils.compute_errors("Large Molecules", large_pred, large_ref)

    onp.savez_compressed(os.path.join(save_path, 'large_molecules_errors.npz'),
        charge_diff=onp.array(charge_diff_large, dtype=object),
        energy_diff=energy_diff_large,
        q_ids=q_ids_large,
        NA=NA_large
    )

    print(f"✓ Evaluation completed for {model_dir}")


def evaluate_cluster_or_maxsep_model(model_dir: str, save_path: str, dataset_path: str, use_coulomb_nbrs: bool):
    def load_dataset(path):
        raw = onp.load(path, allow_pickle=True)
        dataset = {key: dict(raw[key].item()) for key in raw.files}
        return dataset

    def process_dataset(dataset):
        for split in dataset.keys():
            n_particles = onp.sum(dataset[split]['mask'], axis=1)
            max_particles = dataset[split]['mask'].shape[1]

            weights = n_particles / onp.mean(n_particles, keepdims=True)
            dataset[split]['U_weight'] = weights
            dataset[split]['charge_weight'] = max_particles / n_particles
        return dataset

    def setup_model(model_dir, dataset, displacement_fn,use_coulomb_nbrs=False):
        """Setup the model for evaluation."""
        model = Path(model_dir)
        with open(model / "config.toml", "rb") as f:
            config = tomli.load(f)



        if config["model"]["type"]== "DimeNetPP":
            nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, dataset['training']['box'][0],
            config["model"]["r_cutoff"], mask_key="mask", box_key="box",
            format=partition.Dense, fractional_coordinates=True
        )
            
        else:
            nbrs_init, (max_neighbors, max_edges, avg_num_neighbors, max_triplets) = graphs.allocate_neighborlist(
                dataset["training"], displacement_fn, dataset['training']['box'][0],
                config["model"]["r_cutoff"], mask_key="mask", box_key="box",
                format=partition.Sparse, fractional_coordinates=True, capacity_multiplier=1.1,  count_triplets=True
            )
        max_neighbors = int(max_neighbors * 1.1)
        max_edges = int(max_edges * 1.1)



        if use_coulomb_nbrs and config["model"]["type"] != "DimeNetPP":
            nbrs_init, _ = graphs.allocate_neighborlist(
                dataset["training"], displacement_fn, dataset['training']['box'][0],
                config["model"]["coulomb_cutoff"], mask_key="mask", reps_key="reps",
                format=partition.Sparse, box_key="box", fractional_coordinates=True,
                capacity_multiplier=1.1,
            )

        energy_fn_template, _ = train_utils.define_model(
            config, dataset, nbrs_init, max_edges, per_particle=False,
            avg_num_neighbors=avg_num_neighbors, positive_species=True,
            displacement_fn=displacement_fn,
            exclude_correction=config["optimizer"].get("exclude_correction", False),
            electrostatics='pme'
        )

        optimizer = train_utils.init_optimizer(config, dataset)

        energy_params = tree_util.tree_map(jnp.asarray,
            onp.load(model / "best_params.pkl", allow_pickle=True))

        trainer = ForceMatching(
            energy_params, optimizer, energy_fn_template, nbrs_init,
            batch_per_device=config["optimizer"]["batch"] // len(jax.devices()),
            batch_cache=config["optimizer"]["cache"],
            gammas=config["gammas"],
            energy_fn_has_aux=True,
            additional_targets={"charge": custom_quantity.get_aux("charge")},
            weights_keys={"U": "U_weight", "charge": "charge_weight"},
        )
        return energy_params, trainer

    dataset = load_dataset(dataset_path)
    filter_dataset_by_species(dataset["training"], dataset["testing"])

    q_ids_small = {
        "training": dataset['training'].pop('names'),
        "validation": dataset['validation'].pop('names'),
        "testing": dataset['testing'].pop('names'),
    }

    dataset = process_dataset(dataset)

    displacement_fn, _ = space.periodic_general(box=dataset['training']['box'][0], fractional_coordinates=True)

    _, model = setup_model(model_dir, dataset, displacement_fn,use_coulomb_nbrs=use_coulomb_nbrs)

    for split in dataset:
        start_time = time.time()
        predictions = model.predict(dataset[split], batch_size=100 // len(jax.devices()))
        
        stop = time.time()
        elapsed = stop - start_time 
        print(f"Inference time {split}: {elapsed:.2f} s")
        pred, ref = eval_utils.unpadd_predictions(predictions, dataset[split])
        charge_diff, energy_diff = eval_utils.compute_errors(split.capitalize(), pred, ref)

        os.makedirs(save_path, exist_ok=True)
        onp.savez_compressed(os.path.join(save_path, f"{split}_errors.npz"),
            charge_diff=onp.array(charge_diff, dtype=object),
            energy_diff=energy_diff,
            q_ids=q_ids_small[split],
            #NA=dataset[split].pop('original_num_atoms')
        )

    print(f"✓ Evaluation completed for {model_dir}")