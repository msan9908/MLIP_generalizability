from collections import defaultdict
from fairchem.core.datasets import AseDBDataset
from ase.io import Trajectory

# Path to the train split LMDB
dataset_path = "/Users/michalsanocki/Downloads/train_4M"
save_path = 'lowest_energy_metal_organics.traj'
# Load dataset
dataset = AseDBDataset({"src": dataset_path})
print(f"Loaded dataset with {len(dataset)} structures\n")

# Dictionary: root -> (best_index, best_energy)
root_best = {}

for i in range(len(dataset)):
    atoms = dataset.get_atoms(i)
    source = atoms.info.get("source", "")

    if source.startswith("omol/metal_organics/outputs_low_spin_241118/"):
        parts = source.split("/")
        if len(parts) >= 4:
            root = "/".join(parts[:4]) + "/"  # e.g. omol/.../99624_1_-1_1/

            energy = atoms.get_total_energy()

            if root not in root_best or energy < root_best[root][1]:
                root_best[root] = (i, energy)

# Print summary
print(f"Found {len(root_best)} unique roots with lowest-energy structures:\n")



# Write the new dataset
traj = Trajectory(save_path, mode="w")
for root, (best_idx, best_energy) in root_best.items():
    atoms = dataset.get_atoms(best_idx)
    traj.write(atoms)

traj.close()
print(f"Saved all lowest-energy structures to {save_path}")
"""for root, (best_idx, best_energy) in root_best.items():
    atoms = dataset.get_atoms(best_idx)
    print(f"Root: {root}")
    print(f"  Lowest Energy: {best_energy:.6f} eV")
    print("  Composition:   ", atoms.info.get("composition"))
    print("  Num atoms:     ", atoms.info.get("num_atoms"))
    print("  Charge:        ", atoms.info.get("charge"))
    print("  Spin:          ", atoms.info.get("spin"))
    print()
"""