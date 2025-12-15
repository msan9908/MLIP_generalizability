"""Downloads and prepares the ANI1-x dataset."""

import os

from urllib import request
from pathlib import Path
import zipfile
import lzma

import h5py
from sys import stdout

import jax
import numpy as onp
from scipy.special import jnp_zeros

from jax_md_mod import custom_space
from chemtrain.data import preprocessing

import jax.numpy as jnp

from . import utils

def download_ani1x(root="./_data", scale_R=0.1, scale_U=2625.5, scale_e=11.787, fractional=True, max_samples=None, **kwargs):
    with jax.default_device(jax.devices("cpu")[0]):
        data_dir = download_source(root=root)

        dataset = load_and_padd_samples(data_dir)
        dataset = scale_dataset(dataset, scale_R, scale_U, scale_e, fractional)

        train, val, test = preprocessing.train_val_test_split(dataset, **kwargs, shuffle=True, shuffle_seed=11)

        if max_samples is not None:
            train = {key: arr[:max_samples] for key, arr in train.items()}
            val = {key: arr[:max_samples] for key, arr in val.items()}
            test = {key: arr[:max_samples] for key, arr in test.items()}

        return {"training": train, "validation": val, "testing": test}


def download_source(root="./_data"):
    """Downloads and unpacks the QM7-X dataset."""
    url = "https://springernature.figshare.com/ndownloader/files/18112775"

    data_dir = Path(root)
    data_dir.mkdir(exist_ok=True, parents=True)

    if not (data_dir / "ANI_1x").exists():
        (data_dir / "ANI_1x").mkdir()
        print(f"Download ANI1_x dataset from {url}")
        request.urlretrieve(url, data_dir / "ANI_1x/ANI1_x.hdf5", utils.show_progress)

    return data_dir / "ANI_1x"


def load_and_padd_samples(data_dir):
    """Loads and padds the atom data."""

    # Do not process more than once
    if (data_dir / "ANI1_x.npz").exists():
        return dict(onp.load(data_dir / "ANI1_x.npz"))

    with h5py.File(data_dir / "ANI1_x.hdf5", "r") as file:
        max_atoms = max([file[mol]["atomic_numbers"].size for mol in file.keys()])
        n_samples = sum([file[mol]["coordinates"].shape[0] for mol in file.keys()])

        mols = list(file.keys())
        mols.sort()

        print(f"Found {len(file.keys())} molecules with a maximum of {max_atoms} atoms and {n_samples} samples.")

        # Reserve memory for the complete padded dataset
        dataset = {
            "id": onp.zeros((n_samples,), dtype=int),
            "R": onp.zeros((n_samples, max_atoms, 3)),
            "F": onp.zeros((n_samples, max_atoms, 3)),
            "U": onp.zeros((n_samples,)),
            "species": onp.zeros((n_samples, max_atoms), dtype=int),
            "mask": onp.zeros((n_samples, max_atoms), dtype=bool),
            "hirshfeld_charges": onp.zeros((n_samples, max_atoms)),
            "cm5_charges": onp.zeros((n_samples, max_atoms)),
        }

        idx = 0
        for id, mol in enumerate(mols):
            confs = file[mol]
            n_samples, n_atoms, _ = confs["coordinates"].shape

            dataset["id"][idx:idx + n_samples] = onp.broadcast_to(id, (n_samples,))
            dataset["mask"][idx:idx + n_samples] = onp.broadcast_to(onp.arange(max_atoms) < n_atoms, (n_samples, max_atoms))
            dataset["species"][idx:idx + n_samples, :n_atoms] = onp.broadcast_to(onp.asarray(confs["atomic_numbers"], dtype=int), (n_samples, n_atoms))

            dataset["R"][idx:idx + n_samples, :n_atoms, :] = onp.asarray(confs["coordinates"])
            dataset["F"][idx:idx + n_samples, :n_atoms, :] = onp.asarray(confs["wb97x_dz.forces"])
            dataset["U"][idx:idx + n_samples] = onp.asarray(confs["wb97x_dz.energy"])

            dataset["hirshfeld_charges"][idx:idx + n_samples, :n_atoms] = onp.asarray(confs["wb97x_dz.hirshfeld_charges"])
            dataset["cm5_charges"][idx:idx + n_samples, :n_atoms] = onp.asarray(confs["wb97x_dz.cm5_charges"])

            idx += n_samples

    onp.savez(data_dir / "ANI1_x.npz", **dataset)

    return dataset


def scale_dataset(dataset, scale_R, scale_U, scale_e, fractional=True):
    """Scales the dataset."""

    box = 10 * (dataset["R"].max() - dataset["R"].min())

    if fractional:
        dataset['R'] = dataset['R'] / box
    else:
        dataset['R'] = dataset['R'] * scale_R

    scale_F = scale_U / scale_R
    dataset['box'] = scale_R * onp.tile(box * onp.eye(3), (dataset['R'].shape[0], 1, 1))
    dataset['U'] *= scale_U
    dataset['F'] *= scale_F
    dataset['hirshfeld_charges'] *= scale_e
    dataset['cm5_charges'] *= scale_e

    return dataset


def process_dataset(dataset, charge=None):
    """Removes NaNs from charge data and creates weights for charge and energy."""

    for split in dataset.keys():
        # Weight the potential by the number of particles. The per-particle
        # potential should have equal weight, so the error for larger systems
        # should contribute more. On the other hand, since we compute the
        # MSE for the forces, we have to correct for systems with masked
        # out particles.

        n_particles = onp.sum(dataset[split]['mask'], axis=1)
        max_particles = dataset[split]['mask'].shape[1]

        weights = n_particles / onp.mean(n_particles, keepdims=True)
        dataset[split]['U_weight'] = weights
        dataset[split]['F_weight'] = max_particles / n_particles

        for charge_key in ['cm5_charges', 'hirshfeld_charges']:
            charge_data = dataset[split].pop(charge_key)

            # Add data if selected as target
            if charge is not None and charge == charge_key:
                dataset[split]['charge'] = charge_data

        if 'charge' not in dataset[split].keys(): continue

        # Remove NaNs from the samples
        is_nan = onp.isnan(dataset[split]['charge'])
        is_nan = onp.any(is_nan, axis=1)
        print(f"IsNAN: {is_nan}")
        dataset[split]['charge'][is_nan, :] = 0.0
        dataset[split]['charge_weight'] = onp.logical_not(is_nan) / onp.mean(onp.logical_not(is_nan))
        dataset[split]['charge_weight'] *= max_particles / n_particles

    return dataset


if __name__ == "__main__":
    print(download_ani1x(root="/home/paul/Datasets"))
