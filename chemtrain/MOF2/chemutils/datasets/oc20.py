"""Downloads and prepares the ANI1-x dataset."""

import os

from tempfile import NamedTemporaryFile

from urllib import request
from pathlib import Path
import tarfile
import lzma

import h5py
from sys import stdout

import jax
import mdtraj
import numpy as onp
import sklearn
from jax import random
from scipy.special import jnp_zeros

from jax_md_mod import custom_space
from chemtrain.data import preprocessing
from chemtrain import util as chem_util

from ase.io import read

import jax.numpy as jnp

from . import utils

_name = "OC20"


symbol_to_atomic_number = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27,
    "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
    "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47,
    "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67,
    "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77,
    "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87,
    "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97,
    "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106,
    "Bh": 107, "Hs": 108, "Mt": 109,
    "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115,
    "Lv": 116, "Ts": 117, "Og": 118
}


# Covalent single bond radi in pm from https://doi.org/10.1002/chem.200800987
symbol_to_radius = {
    "H": 32, "He": 46, "Li": 133, "Be": 102, "B": 85, "C": 75, "N": 71,
    "O": 63, "F": 64, "Ne": 67, "Na": 155, "Mg": 139, "Al": 126, "Si": 116,
    "P": 111, "S": 103, "Cl": 99, "Ar": 96, "K": 196, "Ca": 171, "Sc": 148,
    "Ti": 136, "V": 134, "Cr": 122, "Mn": 119, "Fe": 116, "Co": 111,
    "Ni": 110, "Cu": 112, "Zn": 118, "Ga": 124, "Ge": 121, "As": 121,
    "Se": 116, "Br": 114, "Kr": 117, "Rb": 210, "Sr": 185, "Y": 163,
    "Zr": 154, "Nb": 147, "Mo": 138, "Tc": 128, "Ru": 125, "Rh": 125,
    "Pd": 120, "Ag": 128, "Cd": 136, "In": 142, "Sn": 140, "Sb": 140,
    "Te": 136, "I": 133, "Xe": 131, "Cs": 232, "Ba": 196, "La": 180,
    "Ce": 163, "Pr": 176, "Nd": 174, "Pm": 173, "Sm": 172, "Eu": 168,
    "Gd": 169, "Tb": 168, "Dy": 167, "Ho": 166, "Er": 165, "Tm": 164,
    "Yb": 170, "Lu": 162, "Hf": 152, "Ta": 146, "W": 137, "Re": 131,
    "Os": 129, "Ir": 122, "Pt": 123, "Au": 124, "Hg": 133, "Tl": 144,
    "Pb": 144, "Bi": 151, "Po": 145, "At": 147, "Rn": 142, "Fr": 223,
    "Ra": 201, "Ac": 186, "Th": 175, "Pa": 169, "U": 170, "Np": 171,
    "Pu": 172, "Am": 166, "Cm": 166, "Bk": 168, "Cf": 168, "Es": 165,
    "Fm": 167, "Md": 173, "No": 176, "Lr": 161, "Rf": 157, "Db": 149,
    "Sg": 143, "Bh": 141, "Hs": 134, "Mt": 129, "Ds": 128, "Rg": 121,
    "Cn": 122, "Nh": 136, "Fl": 143, "Mc": 162, "Lv": 175, "Ts": 165,
    "Og": 157
}


def radius_lookup(dataset):
    species = onp.arange(1, dataset['species'].max() + 1)

    radii = [1.0] + [
        symbol_to_radius[mdtraj.element.Element.getByAtomicNumber(s).symbol]
        for s in species
    ]


    dataset["radius"] = onp.asarray(radii, dtype=float)[dataset['species']]
    dataset["radius"] /= 100.0 # Convert from pm to Angstrom

    return dataset


def download_oc20(root="./_data", scale_R=0.1, scale_U=96.485, seed=11, fractional=True, max_samples=None, normalize=True, filter_options=None, **kwargs):
    with jax.default_device(jax.devices("cpu")[0]):
        data_dir = download_source(root=root)

        split1, split2 = random.split(random.PRNGKey(seed=seed))

        # Training split
        dataset, derived = load_and_padd_samples(data_dir, split="training")
        train, derived = scale_dataset(dataset, derived, scale_R, scale_U, fractional, normalize)
        if max_samples is not None:
            train = {
                key: random.permutation(split1, arr, axis=0)[:max_samples]
                for key, arr in train.items()
            }

        train = filter_dataset(train, filter_options, fractional)

        dataset, _ = load_and_padd_samples(data_dir, split="validation")
        val, _ = scale_dataset(dataset, derived, scale_R, scale_U, fractional, normalize)
        if max_samples is not None:
            val = {
                key: random.permutation(split2, arr, axis=0)[:max_samples]
                for key, arr in val.items()
            }

        val = filter_dataset(val, filter_options, fractional)

        # Ingnore info for now

        # train, val, test = preprocessing.train_val_test_split(dataset, **kwargs, shuffle=True, shuffle_seed=11)

            # test = {key: arr[:max_samples] for key, arr in test.items()}

        return {"training": train, "validation": val} # "testing": test}


def download_source(root="./_data"):
    """Downloads and unpacks the QM7-X dataset."""
    url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar"

    data_dir = Path(root) / f"{_name}"
    data_dir.mkdir(exist_ok=True, parents=True)

    if not (data_dir / "source.tar").exists():
        print(f"Download OC20 dataset from {url}")
        request.urlretrieve(url, data_dir / "source.tar", utils.show_progress)

    with tarfile.open(data_dir / "source.tar", "r") as tar:
        tar.extractall(data_dir)

    url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar"
    if not (data_dir / "source_val_id.tar").exists():
        print(f"Download OC20 validation dataset from {url}")
        request.urlretrieve(url, data_dir / "source_val_id.tar", utils.show_progress)

    with tarfile.open(data_dir / "source_val_id.tar", "r") as tar:
        tar.extractall(data_dir)

    url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_ads.tar"
    if not (data_dir / "source_val_ood_ads.tar").exists():
        print(f"Download OC20 validation dataset from {url}")
        request.urlretrieve(url, data_dir / "source_val_ood_ads.tar", utils.show_progress)

    with tarfile.open(data_dir / "source_val_ood_ads.tar", "r") as tar:
        tar.extractall(data_dir)

    url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_cat.tar"
    if not (data_dir / "source_val_ood_cat.tar").exists():
        print(f"Download OC20 validation dataset from {url}")
        request.urlretrieve(url, data_dir / "source_val_ood_cat.tar", utils.show_progress)

    with tarfile.open(data_dir / "source_val_ood_cat.tar", "r") as tar:
        tar.extractall(data_dir)

    url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar"
    if not (data_dir / "source_val_ood_both.tar").exists():
        print(f"Download OC20 validation dataset from {url}")
        request.urlretrieve(url, data_dir / "source_val_ood_both.tar", utils.show_progress)

    with tarfile.open(data_dir / "source_val_ood_both.tar", "r") as tar:
        tar.extractall(data_dir)

    return data_dir


def load_and_padd_samples(data_dir, split="training"):
    """Loads and padds the atom data."""

    # Do not process more than once
    if (data_dir / f"{_name}_{split}.npz").exists():
        dataset = dict(onp.load(data_dir / f"{_name}_{split}.npz", mmap_mode="r"))

        if split == "training":
            derived = dict(onp.load(data_dir / f"{_name}_derived.npz"))
        else:
            derived = None

        return dataset, derived

    subsets = []

    extxyz_files = []
    txt_files = []
    setid = []

    if split == "training":
        source_dir = data_dir / "s2ef_train_2M/s2ef_train_2M"

        extxyz_files = sorted(
            [f.absolute() for f in source_dir.iterdir() if f.name.endswith(".extxyz.xz")])
        txt_files = sorted(
            [f.absolute() for f in source_dir.iterdir() if f.name.endswith(".txt.xz")])
        setid = [0] * len(txt_files)

    elif split == "validation":
        sets = [
            "s2ef_val_id", "s2ef_val_ood_ads", "s2ef_val_ood_both",
            "s2ef_val_ood_cat"
        ]

        for idx, s in enumerate(sets):
            source_dir = data_dir / s / s

            extxyz_files += sorted(
                [f.absolute() for f in source_dir.iterdir() if f.name.endswith(".extxyz.xz")])
            txt_files += sorted(
                [f.absolute() for f in source_dir.iterdir() if f.name.endswith(".txt.xz")])

            setid += [idx + 1] * len(txt_files)

    else:
        raise ValueError(f"Split {split} unknown.")

    file_pairs = [
        (extxyz_file, txt_file, id)
        for extxyz_file, txt_file, id in zip(extxyz_files, txt_files, setid)
    ]

    for idx, (extxyz_file, txt_file, id) in enumerate(file_pairs):
        print(f"[{idx + 1}/{len(file_pairs)}] Processing {extxyz_file}")

        # Decompress and temporarily save `.extxyz` file to work around ASE issues
        with lzma.open(extxyz_file, 'rb') as f:
            extxyz_content = f.read()

        with NamedTemporaryFile(delete=False, suffix=".extxyz") as temp_file:
            temp_file.write(extxyz_content)
            temp_filepath = temp_file.name
        try:
            # Read the structures using ASE
            structures = list(read(temp_filepath, index=':'))
        finally:
            # Clean up the temporary file
            os.remove(temp_filepath)

        # Decompress and read the `.txt` file
        with lzma.open(txt_file, 'rt') as f:
            txt_lines = f.readlines()

        # Ensure consistency between `.extxyz` and `.txt` files
        if len(structures) != len(txt_lines):
            raise ValueError(
                f"Mismatch in structures and metadata for {extxyz_file}")

        # Get the maximum number of atoms and number of samples in the subset

        max_atoms = max([len(structure.get_positions()) for structure in structures])
        n_samples = len(structures)

        subset = {
            "R": onp.zeros((n_samples, max_atoms, 3), dtype=float),
            "F": onp.zeros((n_samples, max_atoms, 3), dtype=float),
            "U": onp.zeros((n_samples,), dtype=float),
            "dU": onp.zeros((n_samples,), dtype=float),
            "box": onp.zeros((n_samples, 3, 3), dtype=float),
            "species": onp.zeros((n_samples, max_atoms), dtype=int),
            "mask": onp.zeros((n_samples, max_atoms), dtype=bool),
            "total_charge": onp.zeros((n_samples,), dtype=float),
            "subset": onp.zeros((n_samples,), dtype=int)
        }

        for i, structure in enumerate(structures):

            # Parse metadata from `.txt` file
            system_id, frame_number, reference_energy = txt_lines[
                i].strip().split(',')

            # Extract positions, forces, and species
            positions = structure.get_positions()
            forces = structure.get_forces(apply_constraint=False)
            atomic_numbers = structure.get_atomic_numbers()

            n_atoms = len(positions)

            box = onp.array(structure.cell).swapaxes(0, 1)

            subset["R"][i, :n_atoms, :] = positions
            subset["F"][i, :n_atoms, :] = forces
            subset["U"][i] = structure.get_potential_energy()
            subset["dU"][i] = structure.get_potential_energy() - float(reference_energy)
            subset["box"][i, ...] = box
            subset["species"][i, :n_atoms] = atomic_numbers
            subset["mask"][i, :n_atoms] = True
            subset["total_charge"][i] = 0.0
            subset["id"] = id

        subsets.append(subset)

    # Concatenate all subsets
    n_samples = sum([subset["R"].shape[0] for subset in subsets])
    max_atoms = max([subset["R"].shape[1] for subset in subsets])

    dataset = {
        "R": onp.zeros((n_samples, max_atoms, 3), dtype=float),
        "F": onp.zeros((n_samples, max_atoms, 3), dtype=float),
        "U": onp.zeros((n_samples,), dtype=float),
        "dU": onp.zeros((n_samples,), dtype=float),
        "box": onp.zeros((n_samples, 3, 3), dtype=float),
        "species": onp.zeros((n_samples, max_atoms), dtype=int),
        "mask": onp.zeros((n_samples, max_atoms), dtype=bool),
        "total_charge": onp.zeros((n_samples,), dtype=float),
        "subset": onp.zeros((n_samples,), dtype=int)
    }

    idx = 0
    for s in subsets:
        n_samples, n_atoms, _ = s["R"].shape

        dataset["R"][idx:idx + n_samples, :n_atoms, :] = s["R"]
        dataset["F"][idx:idx + n_samples, :n_atoms, :] = s["F"]
        dataset["U"][idx:idx + n_samples] = s["U"]
        dataset["dU"][idx:idx + n_samples] = s["dU"]
        dataset["box"][idx:idx + n_samples, ...] = s["box"]
        dataset["species"][idx:idx + n_samples, :n_atoms] = s["species"]
        dataset["mask"][idx:idx + n_samples, :n_atoms] = s["mask"]
        dataset["total_charge"][idx:idx + n_samples] = s["total_charge"]
        dataset["subset"][idx:idx + n_samples] = s["subset"]

        idx += n_samples

    # Compute a correction for the potential energy

    if split == "training":
        # Exclude masked
        spec = onp.arange(1, dataset['species'].max() + 1)

        # # Get a matrix with number of unique species for each sample
        counts = onp.sum(spec[None, None, :] == dataset['species'][:, :, None],
                         axis=1)
        mask = jnp.any(counts, axis=0)
        counts = counts[:, mask]
        # # Solve for the mean potential contribution
        model = sklearn.linear_model.Ridge(
            alpha=1e-6, fit_intercept=False, positive=True)
        model.fit(-counts, dataset['U'])

        # per_species_energy = onp.linalg.lstsq(counts, dataset['U'][:, None])[0]
        per_species_energy = onp.zeros_like(spec, dtype=float)
        per_species_energy[mask] = -model.coef_
        print(f"Learned species energy: {per_species_energy}")

        model = sklearn.linear_model.Ridge(
            alpha=1e-6, fit_intercept=False, positive=False
        )
        model.fit(-counts, dataset['dU'])

        ad_species_energy = onp.zeros_like(spec, dtype=float)
        ad_species_energy[mask] = -model.coef_
        print(f"Learned adsorbate species energy: {ad_species_energy}")

        # Save fitted energies to simplify processing
        derived = {
            "per_species_energy": per_species_energy,
            "ad_species_energy": ad_species_energy
        }

        onp.savez(data_dir / f"{_name}_derived.npz", **derived)
    else:
        derived = None

    onp.savez(data_dir / f"{_name}_{split}.npz", **dataset)

    return dataset, derived


def scale_dataset(dataset, derived, scale_R, scale_U, fractional, normalize):
    """Scale dataset positions and energies."""

    if fractional:
        inv_box = onp.linalg.inv(dataset['box'])
        dataset['R'] =  onp.einsum(
            'nij,nmj->nmi', inv_box, dataset['R'],
            optimize="optimal"
        )
    else:
        dataset['R'] *= scale_R

    scale_F = scale_U / scale_R
    dataset['box'] *= scale_R

    mask = dataset['species'] > 0
    if normalize:
        dataset['U'] -= onp.sum(
            mask * derived['per_species_energy'][dataset['species'] - 1], axis=1)
        # dataset['dU'] -= onp.sum(
        #     mask * dataset['ad_species_energy'][dataset['species'] - 1], axis=1)

    dataset['U'] *= scale_U
    dataset['dU'] *= scale_U

    derived['per_species_energy'] *= scale_U
    derived['ad_species_energy'] *= scale_U

    dataset['F'] *= scale_F

    dataset = radius_lookup(dataset)
    dataset['radius'] *= scale_R

    return dataset, derived


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

                # sel = n_rep * n_particles <= max_particles
                # print(f"Removed {onp.sum(~sel)} from {sel.size} samples due to capacity.")
                #
                # n_samples, max_particles = dataset['mask'].shape
                # n_particles = n_particles[sel]
                # n_rep = n_rep[sel]
                #
                # # Apply selection to all keys in the dataset
                # dataset = {key: value[sel] for key, value in dataset.items()}
                #
                # # Iterates over all particles periodically
                # tile_idx = onp.mod(
                #     onp.arange(max_particles)[onp.newaxis, :], n_particles[:, onp.newaxis]
                # )
                # sample_idx = onp.arange(dataset["U"].size)[:, onp.newaxis]
                #
                # print(f"Shape of tile_idx is {tile_idx.shape}")
                #
                # shift = onp.floor((onp.arange(max_particles)[jnp.newaxis, :] + 0.5) / n_particles[:, jnp.newaxis])
                # shift = shift[..., jnp.newaxis] * (onp.arange(3) == dim)[jnp.newaxis, jnp.newaxis, :]
                #
                # print(f"Shape of shift is {shift.shape}")
                #
                #
                # new_r = dataset["R"][sample_idx, tile_idx, :] + shift
                # new_r[..., dim] = new_r[..., dim] / n_rep[:, jnp.newaxis]
                #
                # print(f"Shape of new_r is {new_r.shape}")
                #
                # mask = onp.arange(max_particles)[jnp.newaxis, :] < (n_rep * n_particles)[:, jnp.newaxis]
                #
                # print(f"Shape of mask is {mask.shape}")
                #
                # dataset["mask"] = mask
                # dataset["R"] = new_r * mask[..., jnp.newaxis]
                # dataset["F"] = dataset["F"][sample_idx, tile_idx, :] * mask[..., jnp.newaxis]
                # dataset["species"] = dataset["species"][sample_idx, tile_idx] * mask
                # dataset["radius"] = dataset["radius"][sample_idx, tile_idx] * mask + (1 - mask)
                # dataset["total_charge"] *= n_rep
                # dataset["U"] *= n_rep
                # dataset["dU"] *= n_rep
                # dataset["box"][..., dim] *= n_rep[:, jnp.newaxis]
                # dataset["reps"] *= n_rep
                # print(f"After operation shape of reps are {dataset['reps'].shape} and {n_rep.shape}")

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
                    sample["F"] = sample["F"][tile_idx, :] * mask[..., jnp.newaxis]
                    sample["species"] = sample["species"][tile_idx] * mask
                    sample["radius"] = sample["radius"][tile_idx] * mask + (1 - mask)
                    sample["total_charge"] *= nr
                    sample["U"] *= nr
                    sample["dU"] *= nr
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

        energy_target = filter_options.get("energy_target", "adsorbation")
        if energy_target == "adsorbation":
            print(f"Train on adsorbation energy")
            dataset["U"] = dataset["dU"]

    return dataset


def process_dataset(dataset):
    """Removes NaNs from charge data and creates weights for charge and energy."""

    for split in dataset.keys():
        # Weight the potential by the number of particles. The per-particle
        # potential should have equal weight, so the error for larger systems
        # should contribute more. On the other hand, since we compute the
        # MSE for the forces, we have to correct for systems with masked
        # out particles.

        # Correct for larger contribution of potential
        corr = 1.0
        if 'reps' in dataset[split].keys():
            corr = onp.mean(dataset[split]['reps']) / dataset[split]['reps']

        n_particles = onp.sum(dataset[split]['mask'], axis=1)
        max_particles = dataset[split]['mask'].shape[1]

        weights = n_particles / onp.mean(n_particles, keepdims=True)
        dataset[split]['U_weight'] = weights * corr
        dataset[split]['F_weight'] = max_particles / n_particles

    return dataset


template = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
{}
ITEM: BOX BOUNDS pp pp pp
{}
ITEM: ATOMS id type x y z
{}
"""

def write_lammps_data(dataset, filename, idx=0):
    """Determining the bounding box"""

    sample = {key: dataset[key][idx] for key in dataset.keys()}
    n_atoms = sample['mask'].sum()

    sample["R"] = onp.dot(sample["R"], sample["box"].T) * 10
    lo, hi = onp.min(sample["R"], axis=0), onp.max(sample["R"], axis=0)
    box = "\n".join([f"{l:.3f} {h:.3f}" for l, h in zip(lo, hi)])

    coords = ""
    for idx, r, s in zip(range(n_atoms), sample['R'], sample['species']):
        coords += f"{idx + 1} {s} {r[0]:.3f} {r[1]:.3f} {r[2]:.3f}\n"

    with open(filename, "w") as file:
        file.write(template.format(n_atoms, box, coords))


if __name__ == "__main__":
    dataset = download_oc20(
        root="/home/paul/Datasets", filter_options={"r_cutoff": 0.5}
    )

    write_lammps_data(dataset['training'], "oc20_02.lammpstrj", 0)
    write_lammps_data(dataset['training'], "oc20_12.lammpstrj", 100)
    write_lammps_data(dataset['training'], "oc20_22.lammpstrj", 1000)
    write_lammps_data(dataset['training'], "oc20_32.lammpstrj", 4000)

