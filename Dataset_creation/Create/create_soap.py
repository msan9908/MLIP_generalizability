import os
import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from sparse import save_npz

# ---------------- Settings ----------------
basepath = os.getcwd()  # folder where SOAP matrices will be stored
traj_path = "lowest_energy_metal_organics.traj"  # our filtered dataset
soap_params = {
    'rcut': 4.0,
    'sigma': 0.1,
    'nmax': 9,
    'lmax': 9,
    'rbf': 'gto',
    'average': 'off',       # keep per-atom fingerprints
    'crossover': True       # for compression
}

# ---------------- Prepare folder ----------------
soap_dir = os.path.join(basepath, 'soap_matrices')
os.makedirs(soap_dir, exist_ok=True)

# ---------------- Read structures ----------------
structures = read(traj_path, index=':')  # list of ASE Atoms

# ---------------- Use sources as "refcodes" ----------------
refcodes = [s.info.get("source", f"struct_{i}") for i, s in enumerate(structures)]

# ---------------- Determine unique species ----------------
species = []
for structure in structures:
    syms = np.unique(structure.get_chemical_symbols())
    species.extend([sym for sym in syms if sym not in species])
species.sort()
print(f"Unique species: {species}")

# ---------------- Initialize SOAP ----------------
soap = SOAP(
    species=species,
    periodic=False,  # molecules are not periodic
    sigma=soap_params['sigma'],
    r_cut=soap_params['rcut'],
    n_max=soap_params['nmax'],
    l_max=soap_params['lmax'],
    rbf=soap_params['rbf'],
    average=soap_params['average'],
    compression={'mode': 'crossover'},
    sparse=True
)

# ---------------- Generate SOAP fingerprints ----------------
for i, structure in enumerate(structures):
    refcode = refcodes[i].replace("/", "_")  # replace slashes to avoid filesystem issues
    soap_filename = os.path.join(soap_dir, f"soap_{refcode}.npz")
    if os.path.exists(soap_filename):
        continue

    soap_matrix = soap.create(structure)
    save_npz(soap_filename, soap_matrix)
    print(f"Saved SOAP matrix: {soap_filename}")
