import zipfile
from urllib import request
from pathlib import Path

import hashlib

import numpy as onp

import jax

from jax_md_mod import custom_space

from chemtrain.data import preprocessing


from ase.io import read


def download_and_prepare_dataset(root="./", train_ratio=0.7, val_ratio=0.1, scale_U=1.0, scale_R=1.0):

    url = "https://archive.materialscloud.org/record/file?record_id=1387&filename=Ag_warm_nospin.xyz"
    md5 = "e62d46b12e4a300257feaef6e1a75149"

    data_dir = Path(root) / "_data"
    data_dir.mkdir(exist_ok=True, parents=True)

    if not (data_dir/"Ag_warm_nospin.xyz").exists():
        request.urlretrieve(url, data_dir/"Ag_warm_nospin.xyz")

    # Check downloaded file
    with open(data_dir/"Ag_warm_nospin.xyz", "rb") as f:
        hash = hashlib.file_digest(f, "md5")
        if hash.hexdigest() != md5:
            raise ValueError(f"Hash mismatch: {hash.hexdigest()} != {md5}")

    atoms = read(data_dir / 'Ag_warm_nospin.xyz', index=':', format='extxyz')
    n_atoms = 71

    dataset = {
        "R": onp.zeros((len(atoms), n_atoms, 3)),
        "F": onp.zeros((len(atoms), n_atoms, 3)),
        "U": onp.zeros((len(atoms),)),
        "species": onp.zeros((len(atoms), n_atoms), dtype=int),
        "box": onp.zeros((len(atoms), 3, 3)),
    }

    for idx, atom in enumerate(atoms):
        assert atom.get_atomic_numbers().shape[0] == n_atoms, f"Number of atoms is {atom.get_atomic_numbers().shape[0]}"

        dataset["R"][idx, :] = onp.asarray(atom.get_positions())
        dataset["F"][idx, :] = onp.asarray(atom.get_forces())
        dataset["U"][idx] = onp.asarray(atom.get_potential_energy())
        dataset["species"][idx, :] = onp.asarray(atom.get_atomic_numbers())
        dataset["box"][idx, :] = onp.asarray(atom.get_cell())

    splits = preprocessing.train_val_test_split(
        dataset, train_ratio=train_ratio, val_ratio=val_ratio, shuffle=False)

    dataset = {
        "training": splits[0],
        "validation": splits[1],
        "testing": splits[2],
    }

    dataset = scale_dataset(dataset, scale_U=scale_U, scale_R=scale_R, fractional=True)

    print(dataset)

    return dataset


def scale_dataset(dataset, scale_U=1.0, scale_R=1.0, fractional=True):
    """Scales a dataset of positions from real space to fractional coordinates.

    Args:
        dataset: Dictionary containing multiple splits of the full dataset.
            Each split is again a dictionary, containing the keys
            ``["box", "R", "U", "F", "virial", "type"]``.
        scale_U: Unit conversion factor for the energy
        scale_R: Unit conversion factor for lengths
        fractional: Whether to scale the dataset in fractional coordinates.

    Returns:
        Returns all splits of the data in the correct units.

    """

    scale_F = scale_U / scale_R
    for split in dataset.keys():

        _, scale_fn = custom_space.init_fractional_coordinates(
            dataset[split]['box'][0])
        vmap_scale_fn = jax.vmap(lambda R, box: scale_fn(R, box=box),
                                 in_axes=(0, 0))

        if fractional:
            dataset[split]['R'] = vmap_scale_fn(dataset[split]['R'],
                                                dataset[split]['box'])
        else:
            dataset[split]['R'] = dataset[split]['R'] * scale_R

        dataset[split]['box'] *= scale_R
        dataset[split]['U'] *= scale_U
        dataset[split]['F'] *= scale_F

    return dataset


if __name__ == "__main__":
    download_and_prepare_dataset()
