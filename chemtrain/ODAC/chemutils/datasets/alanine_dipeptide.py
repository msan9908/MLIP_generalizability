"""Downloads and prepares the ANI1-x dataset."""

import os
import importlib

from urllib import request
from pathlib import Path
import zipfile
import lzma

import h5py
from sys import stdout

import jax
import numpy as onp
from scipy.special import jnp_zeros

from jax_md_mod import custom_space, io, custom_quantity
from jax_md_mod.model import prior
from jax_md import space

from chemtrain.data import preprocessing

import jax.numpy as jnp

from . import utils

def get_dataset(root="./_data", scale_R=1.0, scale_U=1.0, fractional=True, **kwargs):
    """Returns the mapped CG dataset and the topology."""
    with jax.default_device(jax.devices("cpu")[0]):
        data_dir = download_source(root=root)

        dataset = load_dataset(data_dir)
        dataset = scale_dataset(dataset, scale_R, scale_U, fractional)

        train, val, test = preprocessing.train_val_test_split(dataset, **kwargs, shuffle=True, shuffle_seed=11)

        return {"training": train, "validation": val, "testing": test}, load_topology(data_dir)


def load_topology(data_dir):
    """Loads the topology of the system."""
    mapping = {
        "ACE": {
            "CH3": 0,
            "C": 1,
            "O": 2,
        },
        "ALA": {
            "N": 3,
            "CA": 4,
            "CB": 0,
            "C": 1,
            "O": 2,
        },
        "NME": {
            "C": 0, # Should be CH3, but mdtraj returns C
            "N": 3
        },
    }

    def mapping_fn(name="", residue="", **kwargs):
        return mapping[residue][name]

    mdtraj = importlib.import_module("mdtraj")
    top = mdtraj.load_topology(data_dir / "heavy_2_7nm.gro")
    topology = prior.Topology.from_mdtraj(top, mapping=mapping_fn)

    return topology


def load_dataset(data_dir):
    """Loads the data."""

    force_dataset = preprocessing.get_dataset(data_dir / "forces_heavy_100ns.npy")
    position_dataset = preprocessing.get_dataset(data_dir / "positions_heavy_100ns.npy")

    box, _, _, _ = io.load_box(data_dir / "heavy_2_7nm.gro")
    # Box size fixed to 2.7 nm
    box = onp.diag(box)
    box = onp.tile(box, (force_dataset.shape[0], 1, 1))

    dataset = {
        "R": position_dataset,
        "F": force_dataset,
        "box": box,
    }

    return dataset


def download_source(root="./_data"):
    """Downloads the forces and position data."""
    out_dir = Path(root) / "Alanine_Dipeptide"

    position_url = "https://drive.usercontent.google.com/download?id=1yKVHiI8y7ZNzyduh8bosR6YKScLyFezU&export=download&confirm=t&uuid=cff71a05-a45a-446e-bf2f-17f62e84263f"
    force_url = "https://drive.usercontent.google.com/download?id=1JhRQcZ3tE2w-mLqTGN0JHJQst5uZijJx&export=download&confirm=t&uuid=ad4f279f-18b7-4b65-a144-ab1115110549"
    confs_url = "https://raw.githubusercontent.com/tummfm/relative-entropy/refs/heads/main/examples/alanine_dipeptide/data/confs/heavy_2_7nm.gro"

    forces_path = out_dir / "forces_heavy_100ns.npy"
    positions_path = out_dir / "positions_heavy_100ns.npy"
    conf_path = out_dir / "heavy_2_7nm.gro"

    out_dir.mkdir(exist_ok=True, parents=True)
    if not Path(forces_path).exists():
        request.urlretrieve(force_url, forces_path, utils.show_progress)
    if not Path(positions_path).exists():
        request.urlretrieve(position_url, positions_path, utils.show_progress)
    if not Path(conf_path).exists():
        request.urlretrieve(confs_url, conf_path, utils.show_progress)

    return out_dir


def scale_dataset(dataset, scale_R, scale_U, fractional=True, center=True):
    """Scales the dataset."""

    if fractional:
        dataset['R'] = preprocessing.scale_dataset_fractional(
            dataset['R'], box=dataset['box'])
        dataset['R'] = onp.asarray(dataset['R'])
    else:
        dataset['R'] = dataset['R'] * scale_R

    if center:
        # Compute the displacement to the first particle and the center of mass
        displacement, _ = space.periodic_general(1.0, fractional_coordinates=fractional)
        align_fn = custom_quantity.init_rigid_body_alignment(
            displacement, dataset['R'][0], box=dataset['box'][0])

        dataset['R'] = onp.asarray(
            jax.vmap(lambda R, box: align_fn(R, box=box))(dataset['R'], dataset['box']))


    scale_F = scale_U / scale_R
    dataset['box'] *= scale_R
    dataset['F'] *= scale_F

    return dataset


if __name__ == "__main__":
    print(get_dataset(root="/home/paul/Datasets"))
