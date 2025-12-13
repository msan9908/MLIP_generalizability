import os
import sys
import argparse

from pathlib import Path
import pickle
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
jax.config.update("jax_debug_nans", True)

import numpy as onp
from typing import Union, Tuple, Callable

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

import json
import gzip
import pandas as pd
from ase.io import read
from pymatgen.core.structure import Structure



def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")

    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        model=OrderedDict(
            r_cutoff=0.45,
            edge_multiplier=1.15,
            coulomb_cutoff=0.5,
            type="MACE_EFA",
            model_kwargs=OrderedDict(
                hidden_irreps="64x0e + 128x1o + 64x1e",
                max_ell=3,
                num_interactions=2,
                readout_mlp_irreps = "16x0e",
                correlation=3,
                ),
 
        ),

        optimizer=OrderedDict(
            init_lr=3e-3,
            lr_decay=1e-2,  #
            epochs=80,
            batch=6, 
            cache=100,
            weight_decay=1e-3,
            exclude_correction=False,
            type="ADAM",
            power=2,
            optimizer_kwargs=OrderedDict(
                b1=0.9,
                b2=0.99,
                eps=1e-8,
            )
        ),
        pre_optimizer=OrderedDict(
            init_lr=1e-2,
            lr_decay=1e-2,  
            epochs=0,
            batch=6,  
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
        ),
        gammas=OrderedDict(
            U=1e-2,   
            charge=0 , 
        ),
        pre_gammas=OrderedDict(
            U=0,
            charge= 1e1
        ),
    )


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



def remove_rare_atoms(dataset_small,threshold=10):
    # Count occurrences of each atom type in the small dataset
    unique_atoms, counts = onp.unique(dataset_small['species'], return_counts=True)
    
    # Identify rare atom types (occurring less than `threshold` times)
    rare_atoms = set(unique_atoms[counts < threshold])
    
    if not rare_atoms:
        print("No rare atoms found. Returning original datasets.")
        return dataset_small # No filtering needed

    print(f"Removing MOFs containing rare atoms: {rare_atoms}")

    # Create masks to filter out MOFs containing rare atoms
    small_keep_mask = onp.array([
        not any(atom in rare_atoms for atom in species) 
        for species in dataset_small['species']
    ])



    # Apply masks to both datasets
    dataset_small_filtered = {key: value[small_keep_mask] for key, value in dataset_small.items()}

    return dataset_small_filtered


def scale_dataset(dataset, scale_R, scale_U, scale_e, fractional=True):
    
    """Scale dataset positions and energies."""


    if fractional:
       
        inv_box = onp.linalg.inv(dataset['box'])
        dataset['R'] = onp.einsum('nij,nmj->nmi', inv_box, dataset['R'],    optimize="optimal" )

    else:
        dataset['R'] = dataset['R'] * scale_R


    dataset['box'] *= scale_R #* onp.tile(box_size * onp.eye(3), (dataset['R'].shape[0], 1, 1))
    
    
    
    dataset['U'] *= scale_U
    dataset['radius'] *= scale_R

    dataset['charge'] *= scale_e
    dataset['total_charge'] *= scale_e

    return dataset


Box = Union[float, util.Array]


def _rectangular_boxtensor(box: Box) -> Box:
    """Transforms a 1-dimensional box to a 2D box tensor."""
    spatial_dim = box.shape[0]
    return jnp.eye(spatial_dim).at[jnp.diag_indices(spatial_dim)].set(box)


def init_fractional_coordinates(box: Box) -> Tuple[Box, Callable]:
    """Returns a 2D box tensor and a scale function that projects positions
    within a box in real space to the unit-hypercube as required by fractional
    coordinates.

    Args:
        box: A 1 or 2-dimensional box

    Returns:
        A tuple (box, scale_fn) of a 2D box tensor and a scale_fn that scales
        positions in real-space to the unit hypercube.
    """
    if box.ndim != 2:  # we need to transform to box tensor
        box_tensor = _rectangular_boxtensor(box)
    else:
        box_tensor = box
    inv_box_tensor = space.inverse(box_tensor)
    scale_fn = lambda positions: jnp.dot(positions, inv_box_tensor)
    return box_tensor, scale_fn


def load_qmof_data(data_path):
    # Load optimized geometries
    mofs = read(f'{data_path}/qmof-geometries.xyz', index=':')

    # Load reference codes
    refcodes = onp.genfromtxt(f'{data_path}/qmof-refcodes.csv', delimiter=',', dtype=str)

    # Load material properties
    with open(f'{data_path}/qmof.json') as f:
        qmof_data = json.load(f)

    qmof_df = pd.json_normalize(qmof_data).set_index('name')

    # Load structure data
    with gzip.open(f'{data_path}/qmof_structure_data.json.gz') as f:
        struc_data = json.load(f)
    qmof_strucs = {entry['name']: Structure.from_dict(entry['structure']) for entry in struc_data}
    #print(mofs[0], refcodes[0])
    print(qmof_df.columns)
    return mofs, refcodes, qmof_df, qmof_strucs

def construct_upper_triangular_box(lattice):
    """Convert a Pymatgen lattice to an upper-triangular triclinic box representation."""
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = onp.radians(lattice.alpha), onp.radians(lattice.beta), onp.radians(lattice.gamma)

    # Compute box matrix in upper-triangular form
    box = onp.array([
        [a, b * onp.cos(gamma), c * onp.cos(beta)],
        [0, b * onp.sin(gamma), c * (onp.cos(alpha) - onp.cos(beta) * onp.cos(gamma)) / onp.sin(gamma)],
        [0, 0, c * onp.sqrt(1 - (onp.cos(alpha) ** 2) - (onp.cos(beta) ** 2) - (onp.cos(gamma) ** 2) +
                            2 * onp.cos(alpha) * onp.cos(beta) * onp.cos(gamma)) / onp.sin(gamma)]
    ])



    return box





def preprocess_mof_data(data_path, split_method="cluster", val_ratio=0.1, save_path=None, seed=4):
    mofs, refcodes, qmof_df, qmof_strucs = load_qmof_data(data_path)

    # Process all MOFs into single dataset
    all_mofs, all_ids = [], []
    for i, mof in enumerate(mofs):
        all_mofs.append(mof)
        all_ids.append(refcodes[i])  # Using refcodes directly as IDs

    def transform_to_dataset(mof_list, names):
        num_structures = len(mof_list)
        max_atoms = max(len(mof) for mof in mof_list) if mof_list else 0

        R = onp.zeros((num_structures, max_atoms, 3), dtype=float)
        species = onp.zeros((num_structures, max_atoms), dtype=int)
        mask = onp.zeros((num_structures, max_atoms), dtype=bool)
        radius = onp.zeros((num_structures, max_atoms), dtype=float) 
        charges_array = onp.zeros((num_structures, max_atoms), dtype=float)
        total_charge = onp.zeros((num_structures,), dtype=float)
        U = onp.zeros((num_structures,), dtype=float)
        box = onp.zeros((num_structures, 3, 3), dtype=float)

        
        valid_names = []

        valid_indices = []

        for i, (mof, name) in enumerate(zip(mof_list, names)):
            if name not in qmof_strucs or name not in qmof_df.index:
                print(f"Warning: Missing data for {name}")
                continue

            structure = qmof_strucs[name]
            #structure2  = structure.copy()
            #structure2.to_primitive()
            #print(structure2.lattice)
            cell = mof.cell 
            
            box[i] = onp.stack(onp.array(cell).swapaxes(0, 1))

            tc = 0
            missing_charge = False
            
            for j, atom in enumerate(mof):
                if j < max_atoms:
                    
                    charge = structure[j].properties.get("pbe_ddec_charge",10)
                    
                    if charge >= 10:
                        print(f"Skipping {name}: Missing charge for atom {j}")
                        missing_charge = True
                        break  # Skip the entire MOF
                    
                    if charge > 6:
                        print(f"Warning: Unexpected charge value for {name} --- {structure[j]}")

                    #structure=structure.to_primitive()
                    R[i, j] =  structure[j].coords #
                    species[i, j] = atom.number
                    mask[i, j] = True
                    radius[i, j] = (1+atomic_number_to_radius.get(atom.number, 1.0)) / 100.0  # pm to A
                    charges_array[i, j] = charge
                    tc += charge
            
            if missing_charge:
                continue  # Skip MOF if any charge is missing

            total_charge[i] = onp.round(jnp.sum(charges_array[i]))
            energy = qmof_df.loc[name].get('outputs.pbe.energy_total', 9999999)

            if energy > 0:
                print(f"Skipping {name}: Invalid energy data")
                continue  # Skip MOF if energy is invalid

            U[i] = energy
            valid_indices.append(i)
            valid_names.append(name)

        # Trim arrays to only keep valid indices
        valid_count = len(valid_indices)
        print(valid_count, len(mof_list))
        R = R[:valid_count]
        box = box[:valid_count]
        mask = mask[:valid_count]
        radius = radius[:valid_count]
        species = species[:valid_count]
        charges_array = charges_array[:valid_count]
        total_charge = total_charge[:valid_count]
        U = U[:valid_count]
        print(mask[0])
        
        valid_names = onp.array(valid_names[:valid_count])
        return {
        'R': R,
        'box': box,
        'mask': mask,
        'radius': radius,
        'species': species,
        'charge': charges_array,
        'total_charge': total_charge,
        'U': U,
        'names': valid_names  # Save valid QMOF IDs
    }
    dataset = transform_to_dataset(all_mofs, all_ids)
    scaled_dataset = scale_dataset(dataset, scale_R=0.1, scale_U=96.48, 
                                   scale_e=11.87, fractional=True)
    scaled_dataset = remove_rare_atoms(scaled_dataset, threshold=10)

    spec = onp.arange(scaled_dataset['species'].min(), scaled_dataset['species'].max() + 1)

    counts = onp.sum(spec[None, None, :] == scaled_dataset['species'][:, :, None], axis=1)
    # # Solve for the mean potential contribution
    per_species_energy = onp.linalg.lstsq(counts, scaled_dataset['U'][:, None])[0]
    
    print(f"Per particle energy: {per_species_energy}")
    scaled_dataset['U'] -= onp.dot(counts, per_species_energy).squeeze()

    all_names_old = scaled_dataset['names'].tolist()

    # Remove names for filtering
    scaled_dataset.pop('names')

    # Store indices for mapping
    original_indices = list(range(len(all_names_old)))

    # Add indices to dataset temporarily
    scaled_dataset['__idx__'] = onp.array(original_indices)

    # Filter
    scaled_dataset = filter_dataset(scaled_dataset, filter_options={"r_cutoff": 0.5,'max_particles':800}, fractional=True)

    # Get filtered names using surviving indices
    filtered_indices = scaled_dataset['__idx__']
    all_names = [all_names_old[i] for i in filtered_indices]

    # Remove helper column
    scaled_dataset.pop('__idx__')

    def load_refcode_splits(split_method):
        train_file = f"train_refcodes_{split_method}.csv"
        test_file = f"test_refcodes_{split_method}.csv"
        if os.path.exists(train_file) and os.path.exists(test_file):
            train_refcodes = onp.loadtxt(train_file, dtype=str).tolist()
            test_refcodes = onp.loadtxt(test_file, dtype=str).tolist()
            return train_refcodes, test_refcodes
        else:
            return None, None

    train_refcodes, test_refcodes = load_refcode_splits(split_method)

    if train_refcodes is None or test_refcodes is None:
        # Fallback: random split
        print(f"[INFO] No refcode splits found for method '{split_method}', using random split")
        num_samples = len(all_names)
        indices = onp.arange(num_samples)
        onp.random.seed(4)
        onp.random.shuffle(indices)

        num_train = int(num_samples * (1 - val_ratio) * 0.5)  # 80% of non-val for training
        num_val = int(num_samples * val_ratio)
        num_test = num_samples - num_train - num_val

        train_indices = indices[:num_train].tolist()
        val_indices = indices[num_train:num_train + num_val].tolist()
        test_indices = indices[num_train + num_val:].tolist()

    else:
        # Map to processed dataset indices using names (refcodes)
        train_indices = [i for i, name in enumerate(all_names) if name in train_refcodes]
        test_indices = [i for i, name in enumerate(all_names) if name in test_refcodes]

        # Split train into train/validation
        num_val = int(len(train_indices) * val_ratio)
        onp.random.seed(seed)
        shuffled_train = onp.random.permutation(train_indices)
        val_indices = shuffled_train[:num_val].tolist()
        train_indices = shuffled_train[num_val:].tolist()

    # ==============================
    # Final dataset splits
    # ==============================
    dataset_splits = {
        'training': {key: scaled_dataset[key][train_indices] for key in scaled_dataset.keys()},
        'validation': {key: scaled_dataset[key][val_indices] for key in scaled_dataset.keys()},
        'testing': {key: scaled_dataset[key][test_indices] for key in scaled_dataset.keys()},
    }

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        onp.savez_compressed(os.path.join(save_path, 'mof_dataset.npz'), **dataset_splits)
        print(f"Dataset saved with {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test samples")

    return dataset_splits


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
                    new_mask = onp.zeros((n_samples, max_particles + padd_size),
                                         dtype=bool)
                    new_species = onp.zeros(
                        (n_samples, max_particles + padd_size), dtype=int)
                    new_radius = onp.ones(
                        (n_samples, max_particles + padd_size))

                    new_R[:, :max_particles, :] = dataset['R']
                    new_mask[:, :max_particles] = dataset['mask']
                    new_species[:, :max_particles] = dataset['species']
                    new_radius[:, :max_particles] = dataset['radius']

                    dataset['R'] = new_R
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




def main():

    config = get_default_config()
    print(jax.devices())
    data_path = '13147324/qmof_database'
    out_dir = train_utils.create_out_dir(config, tag="MACE_EFA_rand")
    dataset = preprocess_mof_data(data_path, split_method="None", val_ratio=0.1,
                                 save_path=None, seed=4)
    dataset = process_dataset(dataset, charge="charge")
    print("Box!!!!!",dataset['training']['box'][0])

    displacement_fn, _ = space.periodic_general(box=dataset['training']['box'][0], fractional_coordinates=True)#has to be TRUE
        
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, dataset['training']['box'][0],
            config["model"]["r_cutoff"], mask_key="mask", box_key="box",
            format=partition.Sparse, fractional_coordinates=True
        )
    
    max_neighbors = int(max_neighbors * 1.1)
    max_edges = int(max_edges * 1.1)
    print(f"Neighbors: {nbrs_init}")
    print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")
    nbrs_init, _ = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, dataset['training']['box'][0],
            config["model"]["r_cutoff"], mask_key="mask", 
            format=partition.Sparse, box_key="box", fractional_coordinates=True,
            capacity_multiplier=1.1,
        )
    print(f"Neighbors: {nbrs_init}")
    print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")
    energy_fn_template, init_params = train_utils.define_model(
            config, dataset, nbrs_init, max_edges, per_particle=False,
            avg_num_neighbors=avg_num_neighbors, positive_species=True,
            displacement_fn=displacement_fn, exclude_correction=config["optimizer"].get("exclude_correction", False),
            electrostatics='pme'
        )

    optimizer = train_utils.init_optimizer(config, dataset)

    pre_optimizer = train_utils.init_optimizer(config, dataset, key="pre_optimizer")

    pretrain_trainer_fm = trainers.ForceMatching(
        init_params, pre_optimizer, energy_fn_template, nbrs_init,
        batch_per_device=config["pre_optimizer"]["batch"] // len(jax.devices()),
        batch_cache=config["pre_optimizer"]["cache"],
        gammas=config["pre_gammas"],
        energy_fn_has_aux=True,
        additional_targets={
            "charge": custom_quantity.get_aux("charge"),
        },
        # error_fns={
        #     "charge": max_likelihood.mae_loss
        # },
        weights_keys={
            "U": "U_weight",
            "charge": "charge_weight"
        },
        log_file=out_dir / "pretraining.log",
        # penalty_fn=penalty_fn
    )

    # extensions.log_batch_progress(trainer_fm, frequency=100)

    pretrain_trainer_fm.set_dataset(
        dataset['training'], stage='training')
    pretrain_trainer_fm.set_dataset(
        dataset['validation'], stage='validation', include_all=True)
    # pretrain_trainer_fm.set_dataset(
    #     dataset['testing'], stage='testing', include_all=True)

    if config["pre_optimizer"]["epochs"] > 0:
        pretrain_trainer_fm.train(config["pre_optimizer"]["epochs"])

        # test_predictions = pretrain_trainer_fm.predict(dataset['testing'],
        #                                       batch_size=config["optimizer"][
        #                                                      "batch"] // len(
        #                                           jax.devices()))
    
    config['model']['alpha'] = 4.5
    energy_fn_template, init_params = train_utils.define_model(
        config, dataset, nbrs_init, max_edges, per_particle=False,
        avg_num_neighbors=avg_num_neighbors, positive_species=True,
        displacement_fn=displacement_fn, exclude_correction=config["optimizer"].get("exclude_correction", False),
        electrostatics='pme'
    )


    trainer_fm = trainers.ForceMatching(
        pretrain_trainer_fm.best_params, optimizer, energy_fn_template, nbrs_init,
        batch_per_device=config["optimizer"]["batch"] // len(jax.devices()),
        batch_cache=config["optimizer"]["cache"],
        gammas=config["gammas"],
        energy_fn_has_aux=True,
        additional_targets={
            "charge": custom_quantity.get_aux("charge"),
        },
        weights_keys={
            "U": "U_weight",
            "charge": "charge_weight",
        },

        log_file=out_dir / "training.log",
    )

    # extensions.log_batch_progress(trainer_fm, frequency=100)

    trainer_fm.set_dataset(
        dataset['training'], stage='training')
    trainer_fm.set_dataset(
        dataset['validation'], stage='validation', include_all=True)
    trainer_fm.set_dataset(
        dataset['testing'], stage='testing', include_all=True)

    # predictions = trainer_fm.predict(dataset['training'], batch_size=config["optimizer"]["batch"])
    
    assert not onp.any(onp.isnan(dataset['training']["R"])), "Predicted NaN energies"
    assert not onp.any(onp.isnan(dataset['training']["box"])), "Predicted NaN forces"
    assert not onp.any(onp.isnan(dataset['training']["charge"])), "Predicted NaN forces"

    # Train and save the results to a new folder
    trainer_fm.train(config["optimizer"]["epochs"])

    train_utils.save_training_results(config, out_dir, trainer_fm)

    # test_predictions = trainer_fm.predict(dataset['testing'], batch_size=config["optimizer"]["batch"] // len(jax.devices()))
    train_predictions = trainer_fm.predict(dataset['training'], batch_size=config["optimizer"]["batch"] // len(jax.devices()))

    test_predictions = trainer_fm.predict(dataset['testing'], batch_size=config["optimizer"]["batch"] // len(jax.devices()))

    mae_U_test = onp.sqrt(onp.mean((test_predictions['U'] / onp.sum(dataset['testing']['mask'], axis=1) / 96.49 - dataset['testing']['U'] / onp.sum(dataset['testing']['mask'], axis=1) / 96.49) ** 2 ))
    print(f"Energy Testing  RMSE: {mae_U_test*1000:.1f} (meV/atom)")
    mae_U_train = onp.sqrt(onp.mean((train_predictions['U']/ onp.sum(dataset['training']['mask'], axis=1) / 96.49 - dataset['training']['U']/ onp.sum(dataset['training']['mask'], axis=1) / 96.49) ** 2 ))
    print(f"Energy Training RMSE: {mae_U_train*1000:.1f} (meV/atom)")
    mae_C = onp.sqrt(onp.mean((train_predictions['charge'] / 11.7871 - dataset['training']['charge']/ 11.7871) ** 2))
    print(f"Charge Training RMSE: {mae_C*1000:.1f} (me)")
    mae_C = onp.sqrt(onp.mean((test_predictions['charge'] / 11.7871 - dataset['testing']['charge']/ 11.7871) ** 2))
    print(f"Charge Testing RMSE: {mae_C*1000:.1f} (me)")
    mae_C = onp.mean(onp.abs(test_predictions['charge'] / 11.7871 - dataset['testing']['charge']/ 11.7871) )
    print(f"Charge Testing MAE: {mae_C*1000:.1f} (me)")
    print(out_dir)


    # train_utils.plot_predictions(test_predictions, dataset["testing"], out_dir, f"preds_testing")
    train_utils.plot_predictions(train_predictions, dataset["training"], out_dir, f"preds_training")
    train_utils.plot_convergence(trainer_fm, out_dir)

if __name__ == "__main__":
    main()


