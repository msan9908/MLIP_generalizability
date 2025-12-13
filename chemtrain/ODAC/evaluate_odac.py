import os
import sys
import argparse
from sklearn import linear_model
from jax import tree_util 

from pathlib import Path
import pickle
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import jax
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax_md_mod import io, custom_quantity, custom_space, custom_energy, custom_partition
from jax_md import simulate, partition, space, util, energy, quantity as snapshot_quantity
from collections import OrderedDict
from chemtrain.data import preprocessing, graphs
from chemtrain import quantity, trainers, util as chem_util
from chemtrain.trainers import ForceMatching
from collections import Counter
from chemutils.datasets import utils as data_utils
import train_utils 
from ase.db import connect
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from ase.io import read
from pymatgen.core.structure import Structure
from typing import Dict, Any
import numpy as onp
from ase.data import atomic_numbers
from ase.units import Bohr
import tomli
import eval_utils
import time



atomic_number_to_radius= {
    # Atomic numbers 1-20
    1: 31,   2: 28,   3: 128,  4: 96,   5: 84,
    6: 76,   7: 71,   8: 66,   9: 57,   10: 58,
    11: 166, 12: 141, 13: 121, 14: 111, 15: 107,
    16: 105, 17: 102, 18: 106, 19: 203, 20: 176,

    # Transition metals (21-30)
    21: 170, 22: 160, 23: 153, 24: 139, 25: 139,
    26: 132, 27: 126, 28: 124, 29: 132, 30: 122,

    # Main group (31-38)
    31: 122, 32: 120, 33: 119, 34: 120, 35: 120,
    36: 116, 37: 220, 38: 195,

    # Transition metals (39-48)
    39: 190, 40: 175, 41: 164, 42: 154, 43: 147,
    44: 146, 45: 142, 46: 139, 47: 145, 48: 144,

    # Main group (49-54)
    49: 142, 50: 139, 51: 139, 52: 138, 53: 139,
    54: 140,

    # Heavy elements (55-86)
    55: 244, 56: 215, 57: 207, 58: 204, 59: 203,
    60: 201, 72: 175, 73: 170, 74: 162, 75: 151,
    76: 144, 77: 141, 78: 136, 79: 136, 80: 132,
    81: 145, 82: 146, 83: 148, 84: 140, 85: 150,
    86: 150
}
symbol_to_atomic_number = atomic_numbers

def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("--traj", type=str, required=True,
                        help="Path to lowest_energy_metal_organics.traj")
    parser.add_argument("--train_indices", type=str, default=None,
                        help="Path to train indices file (optional)")
    parser.add_argument("--test_indices", type=str, default=None,
                        help="Path to test indices file (optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for random split (default=42)")
    parser.add_argument("--sl", type=bool, default=False,
                        help="Small/large molecule split")
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        model=OrderedDict(
            r_cutoff=0.5,
            edge_multiplier=1.15,
              type="DimeNetPP",
            model_kwargs=OrderedDict(
               embed_size=128,
                n_interaction_blocks=2,
                 out_embed_size=128,
                 num_rbf=6,
                 num_sbf=7,
                 num_residual_before_skip=1,
                 num_residual_after_skip=2,
                 num_dense_out=3,
             ),

            
            exclude_electrostatics=False,
        ),
        optimizer=OrderedDict(
            init_lr=5e-2,
            lr_decay=1e-3,  #
            epochs=100,
            batch=12, 
            cache=25,
            weight_decay=1e-2,
            type="ADAM",
            power=2,
            exclude_correction=False,
            optimizer_kwargs=OrderedDict(
                b1=0.9,
                b2=0.99,
                eps=1e-8,
            )
        ),
        pre_optimizer=OrderedDict(
            init_lr=1e-2,
            lr_decay=1e-0,  
            epochs=0,
            batch=32,  
            cache=25,
            weight_decay=1e-5,
            exclude_correction=False,
            type="ADAM",
            optimizer_kwargs=OrderedDict(
                b1=0.9,
                b2=0.99,
                eps=1e-8,
            )
        ),
        dataset=OrderedDict(
            total_charge='total_charge', 
            spin='spin'
        ),
        gammas=OrderedDict(
            U=2e-3,   
            F=5e-1,
            charge=0, 
        ),
        pre_gammas=OrderedDict(
            U=0,
            F=0,
            charge= 1e1
        ),
    )


def remove_rare_atoms(splits: dict, threshold: int = 10, large_dataset: dict = None):
    """
    Remove rare atom species consistently across dataset splits and (optionally) a large dataset.

    Args:
        splits (dict): { 'training', 'validation', 'testing' } datasets (each a dict).
        threshold (int): Minimum count for atom species to be considered common.
        large_dataset (dict, optional): The large dataset (dict). If None, only splits are filtered.

    Returns:
        (filtered_splits, filtered_large):
            - filtered_splits: dict with same structure as splits
            - filtered_large: dict with same structure as large_dataset (or None if not provided)
    """
    # 1. Aggregate species across splits (+ large if provided)
    all_species = []
    for ds in splits.values():
        all_species.append(onp.concatenate(ds['species']))
    if large_dataset is not None and len(large_dataset['species']) > 0:
        all_species.append(onp.concatenate(large_dataset['species']))

    all_species = onp.concatenate(all_species)
    unique_atoms, counts = onp.unique(all_species, return_counts=True)

    # 2. Identify rare species
    rare_atoms = set(unique_atoms[counts < threshold])
    if not rare_atoms:
        print("No rare atom types found across datasets.")
        if large_dataset is not None:
            return splits, large_dataset
        else:
            return splits

    print(f"Removing all structures containing these rare atoms: {rare_atoms}\n")

    # 3. Apply consistent filtering to each split
    filtered_splits = {}
    for split_name, ds in splits.items():
        mask = onp.array([
            not any(atom in rare_atoms for atom in species)
            for species in ds['species']
        ])
        print(f"Split '{split_name}': kept {mask.sum()}/{len(mask)} samples.")
        filtered_splits[split_name] = {
            k: (onp.array(v)[mask] if isinstance(v, onp.ndarray) else [v[i] for i in range(len(v)) if mask[i]])
            for k, v in ds.items()
        }

    # 4. Apply same filtering to large dataset (if provided)
    filtered_large = None
    if large_dataset is not None:
        mask_large = onp.array([
            not any(atom in rare_atoms for atom in species)
            for species in large_dataset['species']
        ])
        print(f"Large dataset: kept {mask_large.sum()}/{len(mask_large)} samples.")
        filtered_large = {
            k: (onp.array(v)[mask_large] if isinstance(v, onp.ndarray) else [v[i] for i in range(len(v)) if mask_large[i]])
            for k, v in large_dataset.items()
        }

        return filtered_splits, filtered_large
    else:
        return filtered_splits


def load_and_transform_odac25(path, num_load=None):
    """Load 'metal_organics' structures from a given path and transform to dataset format."""

    structures = []

    all_atoms = read(path, index=":")
    if num_load is not None:
        all_atoms = all_atoms[:num_load]
    print(all_atoms[0])
    print(all_atoms[0].info)
    print(all_atoms[0].get_forces())
    #print(all_atoms[0].get_charges())
    for atoms in all_atoms:
        try:
            energy = atoms.get_total_energy()
            total_charge = atoms.info.get("charge", 0)
            total_spin = atoms.info.get("spin", 0)  # <-- Read total spin
            charges = atoms.info.get("charges", [0.0] * len(atoms))
            forces = atoms.get_forces()

            atoms_data = []
            for j, atom in enumerate(atoms):
                atoms_data.append({
                    "element": atom.symbol,
                    "position": atom.position.tolist(),
                    "force": forces[j].tolist(),
                    "charge": charges[j]
                })

            structure = {
                "atoms": atoms_data,
                "energy": energy,
                "total_charge": int(total_charge),
                "total_spin": total_spin  # <-- Store spin
            }

            if atoms.cell.rank == 3:
                structure["lattice"] = atoms.cell.tolist()

            structures.append(structure)

        except Exception as e:
            print(f"Skipped structure due to error: {e}")
            continue

    return transform_to_dataset_format(structures)


def transform_to_dataset_format(structures, num_train=None, with_info=False):
    """Transform parsed data into the required dataset format and scale."""

    num_structures = len(structures)
    max_atoms = max(len(mol["atoms"]) for mol in structures)

    F = onp.zeros((num_structures, max_atoms, 3))

    R = onp.zeros((num_structures, max_atoms, 3))
    U = onp.zeros((num_structures,))
    box = onp.zeros((num_structures, 3, 3))
    type_array = onp.zeros((num_structures,))
    mask = onp.zeros((num_structures, max_atoms), dtype=bool)
    species = onp.zeros((num_structures, max_atoms), dtype=int)
    charges = onp.zeros((num_structures, max_atoms))
    radius = onp.ones((num_structures, max_atoms))
    Total_charge = onp.zeros((num_structures,))
    Total_spin = onp.zeros((num_structures,))  # <-- Add spin array

    for i, structure in enumerate(structures):
        for j, atom in enumerate(structure['atoms']):
            if j < max_atoms:
                pos = atom['position']
                force = atom['force']
                elem = atom['element']

                R[i, j, :] = pos
                F[i, j, :] = force
                species[i, j] = symbol_to_atomic_number.get(elem, 0)
                mask[i, j] = True
                charges[i, j] = atom['charge']
                radius[i, j] = atomic_number_to_radius.get(elem, 1.0) / 100.0

        U[i] = structure['energy']
        box[i] = onp.array(structure.get('lattice')).swapaxes(0, 1)
        Total_charge[i] = structure['total_charge']
        Total_spin[i] = structure.get('total_spin', 0)  # <-- Assign spin
        type_array[i] = structure['total_charge']

    dataset = {
        'F': F,
        'R': R,
        'U': U,
        'box': box,
        'mask': mask,
        'species': species,
        'charge': charges,
        'total_charge': Total_charge,
        'radius': radius,
    }

    return dataset

def scale_dataset(dataset, scale_R, scale_U, scale_e, fractional=False):
    
    """Scale dataset positions and energies."""

    R_max = dataset["R"].max()
    R_min = dataset["R"].min()
    print(f"Original positions: min={R_min}, max={R_max}")

    if fractional:
        inv_box = onp.linalg.inv(dataset['box'])
        dataset['R'] = onp.einsum('nij,nmj->nmi', inv_box, dataset['R'],    optimize="optimal" )

    else:
        dataset['R'] = dataset['R'] * scale_R


    dataset['box'] *= scale_R #* onp.tile(box_size * onp.eye(3), (dataset['R'].shape[0], 1, 1))
    #dataset['box'] = scale_R * onp.tile(box * onp.eye(3), (dataset['R'].shape[0], 1, 1))
    spec = onp.arange(1, dataset['species'].max() + 1)
    
    # Get a matrix with number of unique species for each sample
    counts = onp.sum(spec[None, None, :] == dataset['species'][:, :, None], axis=1)

    #counts = counts[:, jonp.any(counts, axis=0)]
    # Solve for the mean potential contribution
    model = linear_model.Ridge(alpha=1e-6, fit_intercept=False, positive=True)
    model.fit(-counts, dataset['U'])

    #per_species_energy = onp.linalg.lstsq(counts, dataset['U'][:, None])[0]
    per_species_energy = -model.coef_

    
    print(f"Per particle energy: {per_species_energy}")
    dataset['U'] -= onp.dot(counts, per_species_energy).squeeze()

    dataset['U'] *= scale_U

    dataset['F'] *=  scale_U / scale_R

    dataset['radius'] *= scale_R
    dataset['charge'] *= scale_e
    dataset['total_charge'] *= scale_e

    return dataset

def filter_dataset(dataset, filter_options, fractional):
    if filter_options is None: return dataset

    if "r_cutoff" in filter_options:
        # Remove all samples which could violate the minimum image convention.
        # If the cutoff is smaller than half the shorted box vector,
        r_cutoff = filter_options["r_cutoff"]

        if filter_options.get('replicate', True):
            assert fractional, "Replication only works with fractional coordinates."

            dataset['reps'] = onp.ones(dataset['R'].shape[0], dtype=int)

            # Replicate the samples to ensure that the cutoff is not violated.
            for dim in range(3):
                # Compute the number of replications in that dimension
                box_vectors = onp.linalg.norm(dataset['box'], axis=2)
                print(f"Box vectors between {onp.min(box_vectors)} and {onp.max(box_vectors)}")

                n_rep = onp.int32(onp.ceil(2 * r_cutoff / box_vectors[:, dim]))
                print(f"Shape of reps is {dataset['reps'].shape} and {n_rep.shape}")

                # Ensure that the padding is large enough
                n_samples, max_particles = dataset['mask'].shape
                n_particles = onp.sum(dataset['mask'], axis=1)

                # Make padding larger
                padd_size = onp.max([onp.max(n_rep * n_particles) - max_particles, 0])

                if 'max_particles' in filter_options:
                    if (max_particles + padd_size) > filter_options['max_particles']:
                        print(f"Apply max particle constraint.")
                        padd_size = filter_options['max_particles'] - max_particles

                if padd_size > 0:
                    print(
                        f"Increasing size from {max_particles} to {padd_size + max_particles}.")

                    new_R = onp.zeros((n_samples, max_particles + padd_size, 3))
                    new_F = onp.zeros((n_samples, max_particles + padd_size, 3))
                    new_mask = onp.zeros((n_samples, max_particles + padd_size),
                                         dtype=bool)
                    new_species = onp.zeros(
                        (n_samples, max_particles + padd_size), dtype=int)
                    new_radius = onp.ones(
                        (n_samples, max_particles + padd_size))

                    new_R[:, :max_particles, :] = dataset['R']
                    new_F[:, :max_particles, :] = dataset['F']
                    new_mask[:, :max_particles] = dataset['mask']
                    new_species[:, :max_particles] = dataset['species']
                    new_radius[:, :max_particles] = dataset['radius']

                    dataset['R'] = new_R
                    dataset['F'] = new_F
                    dataset['mask'] = new_mask
                    dataset['species'] = new_species
                    dataset['radius'] = new_radius

                max_particles += padd_size

                
                sel = n_rep * n_particles <= max_particles
                print(f"Removed {onp.sum(~sel)} from {sel.size} samples due to capacity.")

                n_samples, max_particles = dataset['mask'].shape
                n_particles = n_particles[sel]
                n_rep = n_rep[sel]

                # Apply selection to all keys in the dataset
                dataset = {key: value[sel] for key, value in dataset.items()}

                @jax.jit
                def replicate_in_dim(sample, np, nr):

                    # Iterates over all particles periodically
                    tile_idx = jnp.mod(jnp.arange(max_particles), np)

                    shift = jnp.floor((jnp.arange(max_particles) + 0.5) / np)
                    shift = shift[:, jnp.newaxis] * (jnp.arange(3) == dim)[jnp.newaxis, :]

                    print(f"Shape of shift is {shift.shape}")

                    new_r = sample["R"][tile_idx, :] + shift
                    new_r *= jnp.ones(3).at[dim].set(1 / nr)[jnp.newaxis, :]

                    print(f"Shape of new_r is {new_r.shape}")

                    mask = jnp.arange(max_particles) < (nr * np)

                    print(f"Shape of mask is {mask.shape}")

                    sample["mask"] = mask
                    sample["R"] = new_r * mask[..., jnp.newaxis]
                    sample["species"] = sample["species"][tile_idx] * mask
                    sample["charge"] = sample["charge"][tile_idx] * mask
                    sample["radius"] = sample["radius"][tile_idx] * mask + (1 - mask)
                    sample["total_charge"] *= nr
                    sample["U"] *= nr
                    sample["F"] = sample["F"][tile_idx, :] * mask[..., jnp.newaxis]
                    sample["box"] *= jnp.ones(3).at[dim].set(nr)[jnp.newaxis, :]
                
                    sample["reps"] *= nr

                    return sample

                with jax.default_device(jax.devices("cpu")[0]):
                    dataset = chem_util.batch_map(
                        lambda s: replicate_in_dim(*s), (dataset, n_particles, n_rep),
                        batch_size=100
                    )
                    dataset = {
                        key: onp.array(val, copy=True)
                        for key, val in dataset.items()
                    }

                print(f"After operation shape of reps are {dataset['reps'].shape} and {n_rep.shape}")


            box_vectors = onp.linalg.norm(dataset['box'], axis=2)
            print(
                f"Box vectors between {onp.min(box_vectors)} and {onp.max(box_vectors)}")

        else:
            box_vectors = onp.linalg.norm(dataset['box'], axis=2)
            sel = box_vectors.min(axis=1) > 2 * r_cutoff
            print(f"Removed {onp.sum(~sel)} from {sel.size} samples due to cutoff.")

            for key in dataset.keys():
                dataset[key] = dataset[key][sel]


    return dataset

def split_dataset(dataset, train_size=0.9, test_size=0.3,
                  train_indices_file=None, test_indices_file=None, seed=42):
    """
    Split dataset into training, validation, and testing sets.

    If train/test index files are provided, those indices are used.
    Otherwise, a random split is performed.
    """
    num_samples = len(dataset['R'])

    if train_indices_file is not None and test_indices_file is not None:
        # --- Use precomputed split ---
        train_indices = onp.loadtxt(train_indices_file, dtype=int)
        test_indices = onp.loadtxt(test_indices_file, dtype=int)

        # Validation = 10% of training set
        n_val = max(1, int(0.1 * len(train_indices)))
        val_indices = train_indices[:n_val]
        train_indices = train_indices[n_val:]
        print(f"Using precomputed indices: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    else:
        # --- Use random split ---
        print("⚠️ No indices provided — falling back to random split.")
        indices = onp.arange(num_samples)
        trainval_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=seed
        )
        train_indices, val_indices = train_test_split(
            trainval_indices, train_size=train_size, random_state=seed
        )
        print(f"Random split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    return {
        'training': {key: dataset[key][train_indices] for key in dataset.keys()},
        'validation': {key: dataset[key][val_indices] for key in dataset.keys()},
        'testing': {key: dataset[key][test_indices] for key in dataset.keys()},
    }



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

        n_particles =onp.sum(dataset[split]['mask'], axis=1)


        max_particles = dataset[split]['mask'].shape[1]

        weights = n_particles /onp.mean(n_particles, keepdims=True)
        dataset[split]['U_weight'] = weights * corr
        dataset[split]['F_weight'] = max_particles / n_particles

        for charge_key in ['charge']:
            charge_data = dataset[split].pop(charge_key)

            # Add data if selected as target
            if charge is not None and charge == charge_key:
                dataset[split]['charge'] = charge_data

        if 'charge' not in dataset[split].keys(): continue

        # Remove NaNs from the samples
        is_nan =onp.isnan(dataset[split]['charge'])
        is_nan =onp.any(onp.isnan(dataset[split]['charge']), axis=-1)
        dataset[split]['charge'][is_nan, :] = 0.0
        dataset[split]['charge_weight'] = max_particles / n_particles
        dataset[split]['charge_weight'] *= ~is_nan /onp.mean(~is_nan,
                                                              keepdims=True)

    return dataset



def process_dataset_large(dataset, charge=None, max_U=onp.inf):
        """Removes NaNs from charge data and creates weights for charge and energy."""

    
        # Weight the potential by the number of particles. The per-particle
        # potential should have equal weight, so the error for larger systems
        # should contribute more. On the other hand, since we compute the
        # MSE for the forces, we have to correct for systems with masked
        # out particles.
        corr = 1.0
        if 'reps' in dataset.keys():
            corr = onp.mean(dataset['reps']) / dataset['reps']

        n_particles =onp.sum(dataset['mask'], axis=1)


        max_particles = dataset['mask'].shape[1]

        weights = n_particles /onp.mean(n_particles, keepdims=True)
        dataset['U_weight'] = weights * corr
        dataset['F_weight'] = max_particles / n_particles

        for charge_key in ['charge']:
            charge_data = dataset.pop(charge_key)

            # Add data if selected as target
            if charge is not None and charge == charge_key:
                dataset['charge'] = charge_data

        

        # Remove NaNs from the samples
        is_nan =onp.isnan(dataset['charge'])
        is_nan =onp.any(onp.isnan(dataset['charge']), axis=-1)
        dataset['charge'][is_nan, :] = 0.0
        dataset['charge_weight'] = max_particles / n_particles
        dataset['charge_weight'] *= ~is_nan /onp.mean(~is_nan,
                                                              keepdims=True)

        return dataset
def identify_large_small_indices(dataset, max_atoms: int):
    """
    Identify indices of small and large samples based on atom count
    (computed from non-zero mask entries).

    Args:
        dataset (dict): Dataset dictionary, must contain key 'mask'.
                        Shape: (n_samples, n_atoms_max)
        max_atoms (int): Threshold to categorize samples.

    Returns:
        (small_indices, large_indices): onp arrays of indices.
    """
    # Atom count = number of nonzero mask entries per sample
    num_atoms = onp.sum(dataset['mask'], axis=1)
    all_indices = onp.arange(len(num_atoms))

    small_mask = num_atoms < max_atoms
    large_mask = ~small_mask

    small_indices = all_indices[small_mask]
    large_indices = all_indices[large_mask]

    print(f"Identified {len(small_indices)} small and {len(large_indices)} large samples (threshold = {max_atoms}).")
    return small_indices, large_indices
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


def main():
    #config = get_default_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")

    args = parser.parse_args()



    models = {
    """# --- Cluster (42) ---
    "allegro_efa_odac10_cluster_42": {
        "path": "output/Qeq_odac_efa_allegro_EFA_2025_10_10_0efde4bb-28b4-46c7-bfa2-bbab342c4a5d",
        "type": "Cluster"
    },
    "allegro_c_odac10_cluster_42": {
        "path": "output/Qeq_odac_c_allegro_AllegroQeq_2025_10_11_7517260f-b7df-47cd-9cba-7e99f69c0f2a",
        "type": "Cluster"
    },
    "allegro_noc_odac10_cluster_42": {
        "path": "output/Qeq_odac_noc_allegro_AllegroQeq_2025_10_13_49f51082-63fc-4a29-9604-13604e97b9a8",
        "type": "Cluster"},
    "MACE_noc_odac10_cluster_42": {
        "path": "output/Qeq_odac_noc_MACE_MACE_2025_10_14_addcdd9f-321f-449e-a42c-7e37f8829e7b",
        "type": "Cluster"
    },
    "MACE_EFA_odac10_cluster_42": {
        "path": "output/Qeq_odac_efa_MACE_MACE_EFA_2025_10_28_79bd3785-7314-46bd-972a-c2641d05ffa5",
        "type": "Cluster"
    },
    "MACE_c_odac10_cluster_42": {
        "path": "output/Qeq_odac_c_MACE_MACE_2025_10_16_e81b8c3f-e35d-4067-a845-04e17ca20a71",
        "type": "Cluster"
    },
    "MACE_MP4_odac10_cluster_42": {
        "path": "output/Qeq_odac_MP4_MACE_MACE_2025_10_18_f6826388-ed94-4345-866a-8d0a22fbade9",
        "type": "Cluster"
    },
    "Dimnet_odac10_cluster_42": {
        "path": "output/Qeq_odac_efa_allegro_DimeNetPP_2025_10_20_1f7647d5-63d2-42f0-97a0-de2376d32649",
        "type": "Cluster"
    },

    # --- Maxsep (42) ---
    "allegro_efa_odac_maxsep_42": {
        "path": "output/Qeq_odac_efa_allegro_EFA_2025_10_9_8384984c-b6df-4420-90e2-3c86efb61b00",
        "type": "Maxsep"
    },
    "allegro_c_odac_maxsep_42": {
        "path": "output/Qeq_odac_c_allegro_AllegroQeq_2025_10_10_0834ab1f-6f6a-4ffc-8cfe-df9bc7db6e68",
        "type": "Maxsep"
    },
    "allegro_noc_odac_maxsep_42": {
        "path": "output/Qeq_odac_noc_allegro_AllegroQeq_2025_10_11_afc4690c-95b9-4976-b929-cf5e7f1d2e3a",
        "type": "Maxsep"
    },
    "MACE_noc_odac_maxsep_42": {
        "path": "output/Qeq_odac_noc_MACE_MACE_2025_10_13_1dd7aa7b-d1f9-4b76-a599-5ce1c725872c",
        "type": "Maxsep"
    },
    "MACE_EFA_odac_maxsep_42": {
        "path": "output/Qeq_odac_efa_MACE_MACE_EFA_2025_10_26_e70e06f3-1c4a-42a0-a7b5-ae18e4703501",
        "type": "Maxsep"
    },
    "MACE_c_odac_maxsep_42": {
        "path": "output/Qeq_odac_c_MACE_MACE_2025_10_14_794c97c3-ea73-4354-a1e0-c210f0642bfe",
        "type": "Maxsep"
    },
    "MACE_MP4_odac_maxsep_42": {
        "path": "output/Qeq_odac_MP4_MACE_MACE_2025_10_16_09cba5d2-5859-46f6-b2d6-77c7f3dfc5ff",
        "type": "Maxsep"
    },
    "Dimnet_odac_maxsep_42": {
        "path": "output/Qeq_odac_efa_allegro_DimeNetPP_2025_10_18_0b8deef0-22f2-4e97-a7d8-f60c8a32203b",
        "type": "Maxsep"
    },



    # --- SL (42) ---
    "allegro_efa_odac10_SL_42": {
        "path": "output/Qeq_odac_efa_allegro_EFA_2025_10_24_bc442338-a25a-4de0-aa31-d99c831e24d9",
        "type": "SL"
    },
    "allegro_c_odac10_SL_42": {
        "path": "output/Qeq_odac_c_allegro_AllegroQeq_2025_10_25_64c4e419-ea46-49cb-856f-422f298b5e4f",
        "type": "SL"
    },
    "allegro_noc_odac10_SL_42": {
        "path": "output/Qeq_odac_noc_allegro_AllegroQeq_2025_10_26_3e2384e7-85dc-4408-bc5d-abca5a159cbf",
        "type": "SL"
    },
    "MACE_noc_odac10_SL_42": {
        "path": "output/Qeq_odac_noc_MACE_MACE_2025_10_27_5bd85662-a132-42f2-80d4-518c6f9f6ad8",
        "type": "SL"
    },
    "MACE_EFA_odac10_SL_42": {
        "path": "output/Qeq_odac_efa_MACE_MACE_EFA_2025_10_28_3e4117fb-36cf-4029-b322-e2bc18c68af1",
        "type": "SL"
    },
    "MACE_c_odac10_SL_42": {
        "path": "output/Qeq_odac_c_MACE_MACE_2025_10_30_61a38460-b0fb-4293-9063-783e10857875",
        "type": "SL"
    },
    "MACE_MP4_odac10_SL_42": {
        "path": "output/Qeq_odac_MP4_MACE_MACE_2025_10_31_7d33006c-7a0c-42d2-8a85-f860b7943ade",
        "type": "SL"
    },
    "Dimnet_odac10_SL_42": {
        "path": "output/Qeq_odac_efa_allegro_DimeNetPP_2025_11_2_ec7eea4c-c3ba-422c-b97f-6417285c1f1a",
        "type": "SL"
    },

    # --- Rand (42) ---
    "allegro_efa_odac10_rand_42": {
        "path": "output/Qeq_odac_efa_allegro_EFA_2025_10_20_ec7b88e8-dff1-4fb8-9a0b-782684a5ed33",
        "type": "Rand"
    },
    "allegro_c_odac10_rand_42": {
        "path": "output/Qeq_odac_c_allegro_AllegroQeq_2025_10_8_f84b2899-1ba8-409f-aefd-4ec5c6b4d57a",
        "type": "Rand"
    },
    "allegro_noc_odac10_rand_42": {
        "path": "output/Qeq_odac_noc_allegro_AllegroQeq_2025_10_10_4ac07ef5-ea90-4007-8729-7eafd1b753e3",
        "type": "Rand"
    },
    "MACE_noc_odac10_rand_42": {
        "path": "output/Qeq_odac_noc_MACE_MACE_2025_10_11_5c62e427-00a5-41fb-aba6-d01e6108d43a",
        "type": "Rand"
    },
    """
    "MACE_EFA_odac10_rand_42": {
        "path": "output/Qeq_odac_efa_MACE_MACE_EFA_2025_10_23_5a340baa-7af0-42fc-9426-6f921ad22bb3",
        "type": "Rand"
    },

    }



    print(jax.devices())
    dataset = load_and_transform_odac25('lowest10_energy_mof.traj')
    #out_dir = train_utils.create_out_dir(config, tag="omol_dimnet")
    #print(f"Output directory: {out_dir}")
    dataset_scalled = scale_dataset(dataset, scale_R=0.1, scale_U=96.485, scale_e=11.7871, fractional=True)
    #dataset = filter_dataset(dataset, filter_options={"r_cutoff": 0.5, 'max_particles':160000}, fractional=True)
    print(dataset.keys())
    for model_name, info in models.items():

        model = Path(info['path'])
        model_type = info['type']
        with open(model / "config.toml", "rb") as f:
            config = tomli.load(f)
        if  model_type=="SL":
            small_idx, large_idx = identify_large_small_indices(dataset_scalled, max_atoms=300)
            dataset_scalled = filter_dataset(dataset_scalled, filter_options={"r_cutoff": 0.5, 'max_particles':1200}, fractional=True)
            
            large_dataset = {k: onp.array(v)[large_idx] for k, v in dataset_scalled.items()}
            dataset = {k: onp.array(v)[small_idx] for k, v in dataset_scalled.items()}
            train_indices = None
            test_indices = None
            dataset = split_dataset(
                dataset,      
                train_indices_file=train_indices,
                test_indices_file=test_indices,
                seed=42
            )
            
            
        
        elif model_type=='Maxsep':
            dataset = filter_dataset(dataset_scalled, filter_options={"r_cutoff": 0.5, 'max_particles':1200}, fractional=True)
            train_indices = 'train_indices_maxsep_odac10_2.txt'
            test_indices = 'test_indices_maxsep_odac10_2.txt'

        elif model_type=='Cluster':
            dataset = filter_dataset(dataset_scalled, filter_options={"r_cutoff": 0.5, 'max_particles':1200}, fractional=True)
            train_indices = 'train_indices_cluster_odac10_2.txt'
            test_indices = 'test_indices_cluster_odac10_2.txt'
        else:
            dataset = filter_dataset(dataset_scalled, filter_options={"r_cutoff": 0.5, 'max_particles':1200}, fractional=True)
            train_indices = None
            test_indices = None            



        if  model_type=="SL":
            dataset, large_dataset = remove_rare_atoms(dataset,  threshold=10, large_dataset=large_dataset)
            large_dataset = filter_dataset_by_species(dataset["training"], large_dataset)
            large_dataset = process_dataset_large(large_dataset, charge="charge")
        else:
            dataset_split = split_dataset(
                dataset,      
                train_indices_file=train_indices,
                test_indices_file=test_indices,
                seed=42
            )
            dataset = remove_rare_atoms(dataset_split,  threshold=10)
            dataset["testing"] = filter_dataset_by_species(dataset["training"], dataset["testing"])
        dataset = process_dataset(dataset, charge="charge")
        for key in dataset.keys():
            print(f"Split {key} has {len(dataset[key]['U'])} samples")
            dataset[key]["F_lr"] = dataset[key]["F"]
            dataset[key]["U_lr"] = dataset[key]["U"]
            #dataset[key].pop("box")

        print(f"Max num atoms: {len(dataset['training']['charge'][0])}")


        displacement_fn, _ = space.periodic_general(box=dataset['training']['box'][0], fractional_coordinates=True)#has to be TRUE

        mini_dataset = {'training':{}}
        for key in dataset['training'].keys():
            mini_dataset['training'][key] = dataset['training'][key][:200]
            
        if config["model"]["type"]== "DimeNetPP":
            nbrs_init, (max_neighbors, max_edges, avg_num_neighbors, max_triplets) = graphs.allocate_neighborlist(
                        mini_dataset["training"], displacement_fn, mini_dataset['training']['box'][0],
                        config["model"]["r_cutoff"], mask_key="mask", box_key="box",
                        format=partition.Dense, fractional_coordinates=True, capacity_multiplier=1.1,  count_triplets=True
                    )
            max_neighbors = int(max_neighbors * 1.3)
            max_edges = int(max_edges * 1.3)
            print(f"Neighbors: {nbrs_init}")
            print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")
            nbrs_init, _ = graphs.allocate_neighborlist(
                    mini_dataset["training"], displacement_fn, mini_dataset['training']['box'][0],
                    config["model"]["r_cutoff"], mask_key="mask",
                    format=partition.Dense, box_key="box", fractional_coordinates=True,
                    capacity_multiplier=1.15,
                )
            print(f"Neighbors: {nbrs_init}")
            energy_fn_template, init_params = train_utils.define_model(
                config, dataset, nbrs_init, max_edges, per_particle=False,
                avg_num_neighbors=avg_num_neighbors, positive_species=True,
                displacement_fn=displacement_fn, 
                exclude_correction=False,
                exclude_electrostatics=config["model"]["exclude_electrostatics"],
                fractional_coordinates=True,
                max_triplets=max_triplets

            )

        #elif "_c_" in model_name:
            
            

        else:
            config['model']['alpha'] = 4.5
            if "_c_" in model_name:
                config['model']['coulomb_cutoff'] = 0.5
            else:
                config['model']['coulomb_cutoff'] = config['model']['r_cutoff']
            nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
                mini_dataset["training"], displacement_fn, mini_dataset['training']['box'][0],
                config["model"]["r_cutoff"], mask_key="mask", box_key="box",
                format=partition.Sparse, fractional_coordinates=True
            )
            max_neighbors = int(max_neighbors * 1.1)
            max_edges = int(max_edges * 1.1)
            print(f"Neighbors: {nbrs_init}")
            print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")
            nbrs_init, _ = graphs.allocate_neighborlist(
                mini_dataset["training"], displacement_fn, mini_dataset['training']['box'][0],
                config["model"]["r_cutoff"], mask_key="mask",  reps_key="reps",
                format=partition.Sparse, box_key="box", fractional_coordinates=True,
                capacity_multiplier=1.1,
            )
            print(f"Neighbors: {nbrs_init}")
            energy_fn_template, init_params = train_utils.define_model(
                config, dataset, nbrs_init, max_edges, per_particle=False,
                avg_num_neighbors=avg_num_neighbors, positive_species=True,
                displacement_fn=displacement_fn, exclude_correction=config["optimizer"].get("exclude_correction", False),
                electrostatics='pme'
            )

        optimizer = train_utils.init_optimizer(config, dataset)

        energy_params = tree_util.tree_map(jnp.asarray,
            onp.load(model / "best_params.pkl", allow_pickle=True))

        model = ForceMatching(
            energy_params, optimizer, energy_fn_template, nbrs_init,
            batch_per_device=config["optimizer"]["batch"] // len(jax.devices()),
            batch_cache=config["optimizer"]["cache"],
            gammas=config["gammas"],
            energy_fn_has_aux=True,
            additional_targets={"charge": custom_quantity.get_aux("charge")},
            weights_keys={"U": "U_weight", "charge": "charge_weight"},
        )
        save_path = os.path.join('results', model_name)
        
        


        for split in dataset:
                start_time = time.time()
                predictions = model.predict(dataset[split], batch_size=100 // len(jax.devices()))
                
                stop = time.time()
                elapsed = stop - start_time 
                print(f"Inference time {split}: {elapsed:.2f} s")
                pred, ref = eval_utils.unpadd_predictions(predictions, dataset[split])
                charge_diff, energy_diff, force_diff = eval_utils.compute_errors(split.capitalize(), pred, ref)

                os.makedirs(save_path, exist_ok=True)
                onp.savez_compressed(os.path.join(save_path, f"{split}_errors.npz"),
                    charge_diff=onp.array(charge_diff, dtype=object),
                    energy_diff=energy_diff,
                    force_diff=force_diff,

                    #NA=dataset[split].pop('original_num_atoms')
                )

                print(f"✓ Evaluation completed for {model_name}")
        if model_type =="SL":
                start_time = time.time()
                predictions = model.predict(large_dataset, batch_size=100 // len(jax.devices()))
                
                stop = time.time()
                elapsed = stop - start_time 
                print(f"Inference time Large: {elapsed:.2f} s")
                pred, ref = eval_utils.unpadd_predictions(predictions, large_dataset)
                charge_diff, energy_diff, force_diff = eval_utils.compute_errors('Large', pred, ref)

                os.makedirs(save_path, exist_ok=True)
                onp.savez_compressed(os.path.join(save_path, "large_errors.npz"),
                    charge_diff=onp.array(charge_diff, dtype=object),
                    energy_diff=energy_diff,
                    force_diff=force_diff,

                    #NA=dataset[split].pop('original_num_atoms')
                )

                print(f"✓ Evaluation completed for {model_name}")

    


if __name__ == "__main__":
    main()