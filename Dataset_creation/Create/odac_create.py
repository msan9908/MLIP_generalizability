from collections import defaultdict
import heapq
from fairchem.core.datasets import AseDBDataset
from ase.io import Trajectory

# Path to the train split LMDB
dataset_path = "./train/mof"
save_path = "lowest10_energy_mof.traj"
dataset = AseDBDataset({"src": dataset_path})
print(f"Loaded dataset with {len(dataset)} structures")

# Store top 10 lowest energies per MOF using min-heaps
top10_per_mof = defaultdict(list)  # mof_name -> [(energy, idx), ...]

for idx in range(len(dataset)):
    atoms = dataset.get_atoms(idx)
    info = atoms.info

    mof_name = info.get("mof_name")
    energy = info.get("energy")

    # Skip if missing metadata or contains underscores
    if not mof_name or energy is None or "_" in mof_name:
        continue

    # Push new energy into heap
    heapq.heappush(top10_per_mof[mof_name], (-energy, idx))  # store negative energy for max-heap

    # If more than 10, remove the worst (highest energy)
    if len(top10_per_mof[mof_name]) > 10:
        heapq.heappop(top10_per_mof[mof_name])

print("Unique MOF names identified:")
for name in sorted(top10_per_mof.keys()):
    print(f"  • {name}")
print(f"\nFound {len(top10_per_mof)} unique MOFs with up to 10 lowest-energy configurations.\n")

# Write selected atoms to trajectory
traj = Trajectory(save_path, mode="w")

for mof_name, entries in top10_per_mof.items():
    # Convert back to (energy, idx), sorted by energy ascending
    lowest_entries = sorted([(-e, i) for e, i in entries], key=lambda x: x[0])
    for rank, (energy, idx) in enumerate(lowest_entries, 1):
        atoms = dataset.get_atoms(idx)
        atoms.info["selected_lowest_rank"] = rank
        atoms.info["selected_energy"] = energy
        traj.write(atoms)

traj.close()
print(f"Saved up to 10 lowest-energy configurations per MOF to '{save_path}'.")
