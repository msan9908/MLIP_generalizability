
import os
from copy import deepcopy

import numpy as onp

import matplotlib.pyplot as plt
from cycler import cycler



def unpadd_predictions(predictions, reference, info=None):
    predictions = deepcopy(predictions)
    reference = deepcopy(reference)

    mask = reference["mask"]

    for split in [predictions, reference]:
        if info is not None:
            spec = onp.arange(1, reference['species'].max() + 1)

            counts = onp.sum(
                mask[..., None] * (spec[None, None, :] == reference["species"][:, :, None]),
                axis=1)
            counts = counts[:, onp.any(counts, axis=0)]

            split["U"] += onp.dot(counts, info["per_species_energy"]).squeeze()

        # Filter charge and species using the mask
        split["charge"] = split["charge"].ravel()[mask.ravel()]
        
        split["species"] = reference["species"].ravel()[mask.ravel()]  # Filter species
        split["U"] /= onp.sum(mask, axis=1)

    return predictions, reference


def compute_errors(key, predictions, reference):
    """Compute the errors between predictions and reference data."""

    # Calculate charge and energy differences
    charge_diff = (predictions["charge"] - reference["charge"]) / 11.7871 * 1000
    energy_diff = (predictions["U"] - reference["U"]) / 96.485 * 1000

    # Group charge errors by molecule using mask
    mask = reference["mask"]
    species = reference["species"]  # Assuming species is provided in reference
    charge_diff_per_molecule = []
    species_per_molecule = []

    start = 0
    for i in range(mask.shape[0]):  # Loop over molecules
        num_atoms = onp.sum(mask[i])  # Count valid atoms in molecule
        charge_diff_per_molecule.append(charge_diff[start:start + num_atoms])
        species_per_molecule.append(species[start:start + num_atoms])
        start += num_atoms  # Move to next molecule

    # Flatten lists for per-species calculation
    charge_diff_flat = onp.concatenate(charge_diff_per_molecule)
    species_flat = onp.concatenate(species_per_molecule)


    # Calculate per-species MAE
    unique_species = onp.unique(species_flat)  # Get unique species
    unique_species = unique_species[unique_species != 0]  # Exclude 0 (masked atoms)

    species_mae = {}
    for s in unique_species:
        species_mask = (species_flat == s)  # Create a 1D boolean mask
        #print(f"species_mask shape for species {s}: {species_mask.shape}")
        species_charge_diff = charge_diff_flat[species_mask]  # Apply 1D mask
        species_mae[s] = onp.mean(onp.abs(species_charge_diff))  # MAE for species

    # Average MAE across all species
    avg_species_mae = onp.mean(list(species_mae.values()))

    # Print results
    print(template.format(
        key,
        onp.mean(energy_diff ** 2.0) ** 0.5, onp.mean(onp.abs(energy_diff)),
        onp.mean(charge_diff ** 2.0) ** 0.5, onp.mean(onp.abs(charge_diff)),
        avg_species_mae
    ))

    print(species_mae)
    return charge_diff_per_molecule, energy_diff


template = """

{} errors
==========

Energy: {:.4f} meV/atom (RMSE) and {:.4f} meV/atom (MAE)
Charge: {:.4f} [me] (RMSE) and {:.4f} [me] (MAE)
Charge: {:.4f} [me] (SMAE)
"""




def compute_errors_prev(key, predictions, reference):
    """Compute the errors between predictions and reference data."""

    charge_diff = (predictions["charge"] - reference["charge"]) / 11.7871 * 1000
    energy_diff = (predictions["U"] - reference["U"]) / 96.485 * 1000

     # Group charge errors by molecule using mask
    mask = reference["mask"]
    charge_diff_per_molecule = []
    
    start = 0
    for i in range(mask.shape[0]):  # Loop over molecules
        num_atoms = onp.sum(mask[i])  # Count valid atoms in molecule
        charge_diff_per_molecule.append(charge_diff[start:start + num_atoms])
        start += num_atoms  # Move to next molecule

    print(template.format(
        key,
        onp.mean(energy_diff ** 2.0) ** 0.5, onp.mean(onp.abs(energy_diff)),
        onp.mean(charge_diff ** 2.0) ** 0.5, onp.mean(onp.abs(charge_diff))
    ))
    return charge_diff_per_molecule,  energy_diff



from cycler import cycler

def plot_diagonal(ax, x, y):
    min_xy = onp.concatenate([x, y]).min()
    max_xy = onp.concatenate([x, y]).max()
    ax.plot([min_xy, max_xy], [min_xy, max_xy], ":", color="black")


def plot_predictions(*ax, predictions, reference_data, label="_", color="tab20:blue"):
    # Simplifies comparison to reported values
    scale_energy = 96.485  # [eV] -> [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]
    scale_charge = 11.7871

    cmap = plt.get_cmap('tab20')
    if len(ax) > 0:
        ax1, ax2, ax3 = ax
        fig = ax1.get_figure()
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

    # fig.suptitle("Predictions")

    ax1.set_title(f"Energy")
    ax1.plot(reference_data["U"] / scale_energy , predictions["U"] / scale_energy, "o", label="_", color=color, markersize=0.5)
    ax1.set_xlabel("Ref. U [eV/atom]")
    ax1.set_ylabel("Pred. U [eV/atom]")
    plot_diagonal(ax1, reference_data["U"] / scale_energy, predictions["U"] / scale_energy)


    ax3.set_title(f"Charge")
    ax3.plot(reference_data["charge"].ravel() / scale_charge , predictions["charge"].ravel() / scale_charge, "o", label=label, color=color, markersize=0.5)
    ax3.set_xlabel("Ref. Q [e]")
    ax3.set_ylabel("Pred. Q [e]")
    ax3.legend(loc="lower right", prop={'size': 5})
    plot_diagonal(ax3, reference_data["charge"] / scale_charge, predictions["charge"] / scale_charge)

    return fig, (ax1, ax2, ax3)
