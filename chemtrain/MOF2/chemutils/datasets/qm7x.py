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


def download_qm7x(root="./", scale_R=0.1, scale_U=96.4853722, fractional=True, max_samples=None, **kwargs):
    with jax.default_device(jax.devices("cpu")[0]):
        data_dir = download_source(root=root)

        dataset = load_and_padd_samples(data_dir)
        dataset = scale_dataset(dataset, scale_R, scale_U, fractional)

        train, val, test = preprocessing.train_val_test_split(dataset, **kwargs, shuffle=True, shuffle_seed=11)

        if max_samples is not None:
            train = {key: arr[:max_samples] for key, arr in train.items()}
            val = {key: arr[:max_samples] for key, arr in val.items()}
            test = {key: arr[:max_samples] for key, arr in test.items()}

        return {"training": train, "validation": val, "testing": test}


def download_source(root="./"):
    """Downloads and unpacks the QM7-X dataset."""
    url = "https://zenodo.org/api/records/4288677/files-archive"

    data_dir = Path(root) / "_data"
    data_dir.mkdir(exist_ok=True, parents=True)

    if not (data_dir / "QM7x_DB").exists():
        print(f"Download QM7x dataset from {url}")
        request.urlretrieve(url, data_dir / "QM7x_DB.zip", utils.show_progress)

        with zipfile.ZipFile(data_dir / "QM7x_DB.zip") as zip_f:
            zip_f.extractall(data_dir / "QM7x_DB")

    for f in (data_dir / "QM7x_DB").iterdir():
        if f.suffix == ".xz" and not f.with_suffix(".hdf5").exists():
            print(f"Unpack {f}")
            with lzma.open(f) as f_src:
                with open(f.with_suffix(".hdf5"), "wb") as f_out:
                    f_out.write(f_src.read())

    return data_dir / "QM7x_DB"


def duplicates(data_dir):
    """Loads duplicate IDs."""
    DupMols = []
    for line in open(data_dir / 'DupMols.dat', 'r'):
        DupMols.append(line.rstrip('\n'))

    return DupMols


def load_and_padd_samples(data_dir):
    """Loads and padds the atom data."""

    # Do not process more than once
    if (data_dir / "QM7x.npz").exists():
        return dict(onp.load(data_dir / "QM7x.npz"))

    duplicate_list = duplicates(data_dir)

    ## for all sets of molecules (representing individual files):
    set_ids = ['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000']

    # First run through all files determines the maximum number of atoms
    n_atoms = []
    idx = 0
    for id in set_ids:
        fMOL = h5py.File(data_dir / f"{id}.hdf5", 'r')

        for mid, mol in fMOL.items():
            for cid, conf in mol.items():
                # Do not add duplicates to dataset
                if cid in duplicate_list: continue

                # Count the atomic numbers
                if idx % 10000 == 0:
                    print(f"[{idx}] Number of atoms for {mid}:{cid} is {len(conf['atNUM'])}")
                n_atoms.append(len(conf['atNUM']))
                idx += 1

        fMOL.close()

    n_samples = len(n_atoms)
    max_atoms = max(n_atoms)

    print(f"Processed dataset and discovered that the maximum number of atoms is {max_atoms}")

    # Reserve memory for the complete padded dataset
    dataset = {
        "R": onp.zeros((n_samples, max_atoms, 3)),
        "F": onp.zeros((n_samples, max_atoms, 3)),
        "U": onp.zeros((n_samples,)),
        "ID": onp.zeros((n_samples,), dtype=int),
        "species": onp.zeros((n_samples, max_atoms), dtype=int),
        "mask": onp.zeros((n_samples, max_atoms), dtype=bool),
    }

    idx = 0
    for id in set_ids:
        fMOL = h5py.File(data_dir / f"{id}.hdf5", 'r')

        for mid, mol in fMOL.items():
            for cid, conf in mol.items():

                # Do not add duplicates to dataset
                if cid in duplicate_list: continue

                if (idx * 1000) % n_samples == 0:
                    print(f"Processed {idx * 100 / n_samples : .1f} %: {mid}:{cid}")

                # Count the atomic numbers
                n_atoms = len(conf['atNUM'])

                dataset["R"][idx, :n_atoms] = onp.asarray(conf['atXYZ'])
                dataset["F"][idx, :n_atoms] = onp.asarray(conf['totFOR'])
                dataset["U"][idx] = float(conf['ePBE0+MBD'][0])
                dataset["mask"][idx] = onp.arange(max_atoms) < n_atoms
                dataset["ID"][idx] = int(mid)
                dataset["species"][idx, :n_atoms] = onp.asarray(conf['atNUM'])

                idx += 1

        fMOL.close()

    onp.savez(data_dir / "QM7x.npz", **dataset)

    return dataset


def scale_dataset(dataset, scale_R=0.1, scale_U=96.4853722, fractional=True):
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
    dataset['species'] = onp.asarray(dataset['species'], dtype=int)

    return dataset


# atom_positions, atom_numbers, atom_energies, atom_forces, atom_molecule = get_atoms_buffer_property_buffer()
#
# # Pad the data, encode data into numpy arrays
# padded_pos, padded_forces, padded_species = pad_forces_positions_species(species=atom_numbers,
#                                                                          position_data=atom_positions,
#                                                                          forces_data=atom_forces)
# atom_energies = onp.array(atom_energies)
# atom_molecule = onp.array(atom_molecule)
# # Save
# onp.save("QM7x_DB/atom_positions_QM7x.npy", padded_pos)
# onp.save("QM7x_DB/atom_numbers_QM7x.npy", padded_species)
# onp.save("QM7x_DB/atom_energies_QM7x.npy", atom_energies)
# onp.save("QM7x_DB/atom_forces_QM7x.npy", padded_forces)
# onp.save("QM7x_DB/atom_molecule_QM7x.npy", atom_molecule)
#
# # Shuffle the data and save a shuffled set
# indices = onp.arange(len(atom_energies))
# onp.random.shuffle(indices)
# print("First 20 shuffled indices: ", indices[:20])
#
# shuffled_padded_pos = padded_pos[indices]
# shuffled_padded_species = padded_species[indices]
# shuffled_atom_energies = atom_energies[indices]
# shuffled_padded_forces = padded_forces[indices]
# shuffled_atom_molecule = atom_molecule[indices]
#
# # Save the shuffled data
# onp.save("QM7x_DB/shuffled_atom_positions_QM7x.npy", shuffled_padded_pos)
# onp.save("QM7x_DB/shuffled_atom_numbers_QM7x.npy", shuffled_padded_species)
# onp.save("QM7x_DB/shuffled_atom_energies_QM7x.npy", shuffled_atom_energies)
# onp.save("QM7x_DB/shuffled_atom_forces_QM7x.npy", shuffled_padded_forces)
# onp.save("QM7x_DB/shuffled_atom_molecule_QM7x.npy", shuffled_atom_molecule)
#
#
# print("Done")

if __name__ == "__main__":
    print(download_qm7x())
