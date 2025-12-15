import os
from pathlib import Path
from urllib import request
import zipfile
import json
import subprocess

import mdtraj
import numpy as onp

from jax_md_mod import io

from chemtrain.data import preprocessing
from chemutils.datasets import utils


_urls = {
    "0.51": "https://escholarship.org/content/qt6sd403pz/supp/FreeSolv-0.51.zip"
}


def download_freesolv(root="./_data",
                      scale_R=1.0,
                      scale_U=4.184,
                      fractional=True,
                      version='0.51',
                      subsets=None, # TODO: Implement options to select subsets
                      **kwargs):

    dataset, info = download_and_process(_urls[version], version, root=root)
    dataset = scale_dataset(dataset, scale_R, scale_U, fractional)
    training, validation, testing = preprocessing.train_val_test_split(
        dataset, **kwargs, shuffle=True, shuffle_seed=11)


    dataset = {
        "training": training,
        "validation": validation,
        "testing": testing
    }

    return dataset, info


def download_and_process(url: str, version: str, root: str="./_data"):
    """Downloads and prepares the FreeSolv database for use with the ReSolv method.

    This function generates initial conformations for all SMILES strings using
    the command-line tool OpenBabel (must be installed).

    """
    data_dir = Path(root)
    data_dir.mkdir(exist_ok=True, parents=True)

    data_dir = data_dir / "FREE_SOLV"
    if (data_dir / "database.npz").exists():
        dataset = dict(onp.load(data_dir / "database.npz"))
        info = get_dataset_info(data_dir)
        return dataset, info

    if not (data_dir / "source.zip").exists():
        data_dir.mkdir()
        print(f"Download FREE_SOLV dataset from {url}")
        request.urlretrieve(url, data_dir / "source.zip", utils.show_progress)

        with zipfile.ZipFile(data_dir / "source.zip", 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    # Create initial conformations using openbabel
    with open(data_dir / f"FreeSolv-{version}/database.json", "r") as f:
        database = json.load(f)

    # Create initial conformations using openbabel
    conf_dir = data_dir / "confs"
    conf_dir.mkdir(exist_ok=True)

    raw_data = {
        "R": [],
        "dF": [],
        "species": [],
        "mass": [],
        "bonds": []
    }

    ids = []
    smiles = []

    for key, value in enumerate(database.items()):
        if not (conf_dir / f"{key}.pdb").exists():
            command = 'echo "{}" | obabel -ismi -O {}.pdb --gen3d slowest'.format(value["smiles"], conf_dir / key)
            print(f"Run command {command}")
            subprocess.run(command, check=True, shell=True)

        _, pos, mass, species = io.load_box(f"{conf_dir / key}.pdb")
        top = mdtraj.load_topology(f"{conf_dir / key}.pdb")

        # Create bondlist
        bondlist = onp.zeros((2, 2 * top.n_bonds), dtype=int)
        for idx, (b1, b2) in enumerate(top.bonds):
            bondlist[0, idx] = b1.index
            bondlist[1, idx] = b2.index

            # Make undirected
            bondlist[0, 2 * idx] = b2.index
            bondlist[1, 2 * idx] = b1.index


        # Skip conformations without positions
        if onp.all(pos == 0.0):
            print(f"Skip conformation {key} with all-zero positions.")
            continue
        else:
            # Only save molecules with successful generation
            ids.append(key)
            smiles.append(value["smiles"])

        raw_data["R"].append(pos)
        raw_data["species"].append(species)
        raw_data["mass"].append(mass)
        raw_data["dF"].append(float(value["expt"]))
        raw_data["bonds"].append(bondlist)

    with open(data_dir / "smiles.csv", "w") as f:
        for idx, (key, smile) in enumerate(zip(ids, smiles)):
            f.write(f"{idx};{key};{database[key]['smiles']}\n")

    # Process the raw data by padding all atoms
    max_atoms = max([len(species) for species in raw_data["species"]])
    max_bonds = max([bonds.shape[1] for bonds in raw_data["bonds"]])
    n_samples = len(raw_data["R"])

    data = {
        "id": onp.arange(n_samples, dtype=int),
        "R": onp.zeros((n_samples, max_atoms, 3), dtype=float),
        "dF": onp.zeros((n_samples,), dtype=float),
        "species": onp.zeros((n_samples, max_atoms), dtype=int),
        "mask": onp.zeros((n_samples, max_atoms), dtype=bool),
        "mass": onp.zeros((n_samples, max_atoms), dtype=float),
        "bonds": onp.full((n_samples, 2, max_bonds), max_atoms, dtype=int)
    }

    for idx, entry in enumerate(zip(*raw_data.values())):
        pos, free_eng, species, mass, bonds = entry

        data["R"][idx, :mass.size] = pos
        data["dF"][idx] = free_eng
        data["mask"][idx, :mass.size] = True
        data["species"][idx, :mass.size] = species
        data["mass"][idx, :mass.size] = mass
        data["bonds"][idx, :, :bonds.shape[1]] = bonds

    onp.savez(data_dir / "database.npz", **data)

    return data, get_dataset_info(data_dir)


def get_dataset_info(data_dir):
    ids, keys, smis = onp.loadtxt(data_dir / "smiles.csv", delimiter=";", dtype=str, unpack=True)
    info_dict = {
        int(id): {"key": key, "smile": smi}
        for id, key, smi in zip(ids, keys, smis)
    }
    return info_dict


def scale_dataset(dataset, scale_R, scale_U, fractional=True):
    """Scales the dataset."""

    # Make large enough to fit individual structures
    box = 10 * (dataset["R"].max(axis=(1, 2)) - dataset["R"].min(axis=(1, 2)))

    if fractional:
        dataset['R'] = dataset['R'] / box[:, onp.newaxis, onp.newaxis]
    else:
        dataset['R'] = dataset['R'] * scale_R

    dataset['dF'] = dataset['dF'] * scale_U
    dataset['box'] = scale_R * box[:, onp.newaxis, onp.newaxis] * onp.tile(onp.eye(3), (box.size, 1, 1))

    return dataset


if __name__ == "__main__":
    download_freesolv("/home/paul/Datasets")

