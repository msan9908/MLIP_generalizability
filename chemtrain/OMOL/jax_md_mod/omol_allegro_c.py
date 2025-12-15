import os
import sys
import argparse
from sklearn import linear_model

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

# Define symbol-to-radius mapping (expand as needed)

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
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch", type=int, default=100)
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        model=OrderedDict(
            r_cutoff=0.45,
            edge_multiplier=1.15,
            type="AllegroQeq",
            model_kwargs=OrderedDict(

                hidden_irreps=" 128x1e + 128x1o ",
                max_ell=2 ,
                radius_scale=1,
                embed_dim=64,
                #num_layers=2,
                num_layers = (2,1),
                charge_embed_dim=128,
                charge_embed_layers=2,
                grid=[15,15, 15],
                learn_radius=True,
                alpha=4.5,
                exclude_electrostatics=False,
            ),
            coulomb_onset=0.4,
            coulomb_cutoff=0.5,
        ),
        optimizer=OrderedDict(
            init_lr=1e-2,
            lr_decay=5e-3,  #
            epochs=170,
            batch=12, 
            cache=50,
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
            epochs=12,
            batch=12,  
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
            F=1e-1,
            charge=3e4, 
        ),
        pre_gammas=OrderedDict(
            U=0,
            F=0,
            charge= 1e1
        ),
    )

def load_and_transform_omol25(path, num_load=None):
    """Load 'metal_organics' structures from a given path and transform to dataset format."""

    structures = []

    all_atoms = read(path, index=":")
    if num_load is not None:
        all_atoms = all_atoms[:num_load]
    print(all_atoms[0])
    print(all_atoms[0].info)
    for atoms in all_atoms:
        
        try:
            energy =atoms.get_total_energy()
            total_charge = atoms.info.get("charge", 0)
            charges = atoms.info.get("nbo_charges", [0.0] * len(atoms))
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
                "total_charge": total_charge
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
        box[i] = onp.array(structure.get('lattice', onp.zeros((3, 3))))
        Total_charge[i] = structure['total_charge']
        type_array[i] = structure['total_charge']


    return {
        'F': F,
        'R': R,
        'U': U,
        'box': box,
        'type': type_array,
        'mask': mask,
        'species': species,
        'charge': charges,
        'total_charge': Total_charge,
        'radius': radius
    }

def scale_dataset(dataset, scale_R, scale_U, scale_e, fractional=False):
    
    """Scale dataset positions and energies."""

    R_max = dataset["R"].max()
    R_min = dataset["R"].min()
    box = 10 * (R_max - R_min)
    print(f"Original positions: min={R_min}, max={R_max}")

    if fractional:
        inv_box = onp.linalg.inv(dataset['orig_box'])
        dataset['R'] = onp.einsum('nij,nmj->nmi', inv_box, dataset['R'],    optimize="optimal" )

    else:
        dataset['R'] = dataset['R'] * scale_R


    #dataset['box'] *= scale_R #* onp.tile(box_size * onp.eye(3), (dataset['R'].shape[0], 1, 1))
    dataset['box'] = scale_R * onp.tile(box * onp.eye(3), (dataset['R'].shape[0], 1, 1))
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

    dataset['F'] *=  scale_U / 0.0529177

    dataset['radius'] *= scale_R
    dataset['charge'] *= scale_e
    dataset['total_charge'] *= scale_e

    return dataset

def split_dataset(dataset, size=0.8):
    num_train = len(dataset['R'])
    train_size = int(size * num_train)
    
    indices = onp.arange(num_train)
    train_indices, val_indices = train_test_split(indices, train_size=train_size, random_state=42)
    print(train_size)
    return {
        'training': {key: dataset[key][train_indices] for key in dataset.keys()},
        'validation': {key: dataset[key][val_indices] for key in dataset.keys()},
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


def main():
    config = get_default_config()
    print(jax.devices())
    dataset = load_and_transform_omol25("filtered_metal_complexes.traj")
    out_dir = train_utils.create_out_dir(config, tag="omol")
    dataset = scale_dataset(dataset, scale_R=0.1, scale_U=96.485, scale_e=11.7871, fractional=False)
    #dataset = filter_dataset(dataset, filter_options={"r_cutoff": 0.5, 'max_particles':160000}, fractional=True)
    print(dataset.keys())
    dataset = split_dataset(dataset, size=0.85)
    dataset = process_dataset(dataset, charge="charge")
    for key in dataset.keys():
        print(f"Split {key} has {len(dataset[key]['U'])} samples")
        dataset[key]["F_lr"] = dataset[key]["F"]
        dataset[key]["U_lr"] = dataset[key]["U"]

    displacement_fn, _ = space.periodic_general(box=dataset['training']['box'][0], fractional_coordinates=False)
        
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, dataset['training']['box'][0],
            config["model"]["r_cutoff"], mask_key="mask", box_key="box", 
            format=partition.Sparse, fractional_coordinates=False
        )
    max_neighbors = int(max_neighbors * 1.1)
    max_edges = int(max_edges * 1.1)
    print(f"Neighbors: {nbrs_init}")
    print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")
    nbrs_init, _ = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, dataset['training']['box'][0],
            config["model"]["coulomb_cutoff"], mask_key="mask",  box_key="box",
            format=partition.Sparse, fractional_coordinates=True,
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
        weights_keys={
            "U": "U_weight",
            "F": "F_weight",
            "charge": "charge_weight"
        },
        log_file=out_dir / "pretraining.log",
        # penalty_fn=penalty_fn
    )


    pretrain_trainer_fm.set_dataset(
        dataset['training'], stage='training')
    pretrain_trainer_fm.set_dataset(
        dataset['validation'], stage='validation', include_all=True)
    # pretrain_trainer_fm.set_dataset(

    if config["pre_optimizer"]["epochs"] > 0:
        pretrain_trainer_fm.train(config["pre_optimizer"]["epochs"])

    
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
            "F": "F_weight",
            "charge": "charge_weight",
        },

        log_file=out_dir / "training.log",
    )
    trainer_fm.set_dataset(
            dataset['training'], stage='training')
    trainer_fm.set_dataset(
            dataset['validation'], stage='validation', include_all=True)

        
    trainer_fm.train(config["optimizer"]["epochs"])
    train_utils.save_training_results(config, out_dir, trainer_fm)

    train_predictions = trainer_fm.predict(dataset['training'], batch_size=config["optimizer"]["batch"] // len(jax.devices()))
    val_predictions = trainer_fm.predict(dataset['validation'], batch_size=config["optimizer"]["batch"] // len(jax.devices()))
    

    mae_U_train = onp.sqrt(onp.mean((train_predictions['U'] / onp.sum(dataset['training']['mask'], axis=1) / 96.49 - dataset['training']['U']/ onp.sum(dataset['training']['mask'], axis=1) / 96.49) ** 2 )) 
    print(f"Energy Training RMSE: {mae_U_train*1000:.1f} (meV/atom)")
    mae_C = onp.sqrt(onp.mean((train_predictions['charge'] / 11.7871 - dataset['training']['charge']/ 11.7871) ** 2)) 
    print(f"Charge Training RMSE: {mae_C*1000:.1f} (me)")
    mae_F_test = onp.sqrt(onp.mean((train_predictions['F'] / 96.49 * 0.1 - dataset['training']['F'] / 96.49 * 0.1) ** 2))
    print(f"Force Training RMSE: {mae_F_test*1000:.2f} (meV/A)")

    # train_utils.plot_predictions(test_predictions, dataset["testing"], out_dir, f"preds_testing")
    train_utils.plot_predictions(train_predictions, dataset["training"], out_dir, f"preds_training")
    train_utils.plot_predictions(val_predictions, dataset["validation"], out_dir, f"preds_val")
    train_utils.plot_convergence(trainer_fm, out_dir)


if __name__ == "__main__":
    main()