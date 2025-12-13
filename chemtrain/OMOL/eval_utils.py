
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

        split["charge"] = split["charge"].ravel()[mask.ravel()]
        split["F"] = split["F"].reshape(-1, 3)[mask.ravel(), :]
        split["U"] /= onp.sum(mask, axis=1)

    return predictions, reference


template = """

{} errors
==========

Energy: {:.4f} meV/atom (RMSE) and {:.4f} meV/atom (MAE)
Force: {:.4f} meV/Å (RMSE) and {:.4f} meV/Å (MAE)
Charge: {:.4f} [me] (RMSE) and {:.4f} [me] (MAE)

"""

def compute_errors(key, predictions, reference):
    """Compute the errors between predictions and reference data."""

    charge_diff = (predictions["charge"] - reference["charge"]) / 11.7871 * 1000
    force_diff = (predictions["F"] - reference["F"]) / 96.485 / 10 * 1000
    energy_diff = (predictions["U"] - reference["U"]) / 96.485 * 1000

    print(template.format(
        key,
        onp.mean(energy_diff ** 2.0) ** 0.5, onp.mean(onp.abs(energy_diff)),
        onp.mean(force_diff ** 2.0) ** 0.5, onp.mean(onp.abs(force_diff)),
        onp.mean(charge_diff ** 2.0) ** 0.5, onp.mean(onp.abs(charge_diff))
    ))
    return charge_diff, energy_diff , force_diff


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
