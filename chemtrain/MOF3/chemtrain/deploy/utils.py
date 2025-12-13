
from jax import ops, numpy as jnp

from jax_md_mod.model import layers
from jax_md.util import high_precision_sum

def edge_to_atom_energies(n_atoms, per_edge_energies, senders):
    """Assigns energies of edges to per-atom energies.

    Args:
        n_atoms: Number of atoms
        per_edge_energies: Energies of each edge
        senders: Sender atoms of the edge

    Returns:
        Returns the energies per atoms. Edge energies are assigned to the
        senders of the edge.

    """
    per_atom_energies = layers.high_precision_segment_sum(
        per_edge_energies, senders, num_segments=n_atoms
    )
    return per_atom_energies
