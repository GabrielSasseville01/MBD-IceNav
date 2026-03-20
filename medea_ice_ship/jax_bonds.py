"""
JAX-compatible graph algorithms for the 2-D ice bond topology.

Provides a GPU-friendly replacement for the BFS-based
find_connected_components() in bonds.py.

Algorithm
---------
Iterative label propagation (Shiloach-Vishkin style):
  1. Initialize labels[i] = i  (every element is its own component)
  2. For each intact bond (i, j): scatter min(labels[i], labels[j]) back to
     both endpoints via .at[].min().
  3. Repeat n_iter times.

After n_iter >= graph_diameter iterations the labels converge to the minimum
element index in each connected component.  Elements with the same label
belong to the same component.

Unlike BFS:
  * Fixed output shape (n,) — no variable-length lists.
  * Fully vectorised — one jax.lax.scan over edges per iteration.
  * JIT-compilable and GPU-compatible.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Connected components via label propagation
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n", "n_iter"))
def find_connected_components_jax(
    n: int,
    bond_i: jnp.ndarray,       # (n_bonds,) int32  — first bond endpoint
    bond_j: jnp.ndarray,       # (n_bonds,) int32  — second bond endpoint
    bond_intact: jnp.ndarray,  # (n_bonds,) bool   — True = bond not broken
    n_iter: int = 64,
) -> jnp.ndarray:               # (n,) int32  — label[i] = min index in component
    """
    JAX-compatible connected components via iterative label propagation.

    Returns an integer label array where label[i] is the minimum element
    index in the connected component containing element i.  Elements with the
    same label value belong to the same component.

    Parameters
    ----------
    n : int  (static — controls jnp.arange and scan length)
        Number of ice elements.
    bond_i, bond_j : (n_bonds,) int32
        Bond endpoint indices.
    bond_intact : (n_bonds,) bool
        True for intact (non-broken) bonds.
    n_iter : int  (static)
        Number of propagation iterations.  Must be >= diameter of the bond
        graph for the result to be exact.  Default 64 is conservative for
        typical compact ice floes; increase for very elongated configurations.

    Returns
    -------
    labels : (n,) int32
        Component label per element (minimum element index in component).
    """
    labels = jnp.arange(n, dtype=jnp.int32)

    def prop_step(labels: jnp.ndarray, _) -> tuple:
        label_i = labels[bond_i]   # (n_bonds,) — current label of i-endpoint
        label_j = labels[bond_j]   # (n_bonds,) — current label of j-endpoint
        min_ij = jnp.minimum(label_i, label_j)  # (n_bonds,)

        # Only propagate along intact bonds; use n (out-of-range) as sentinel
        # so that broken bonds do not scatter any update.
        sentinel = jnp.full_like(min_ij, n)
        min_for_i = jnp.where(bond_intact, min_ij, sentinel)
        min_for_j = jnp.where(bond_intact, min_ij, sentinel)

        new_labels = labels
        new_labels = new_labels.at[bond_i].min(min_for_i)
        new_labels = new_labels.at[bond_j].min(min_for_j)
        return new_labels, None

    labels, _ = jax.lax.scan(prop_step, labels, None, length=n_iter)
    return labels


# ---------------------------------------------------------------------------
# Component centre-of-mass velocity  (rigid-body drift helper)
# ---------------------------------------------------------------------------

@jax.jit
def component_com_velocities_jax(
    labels: jnp.ndarray,      # (n,) int32  — component label per element
    masses: jnp.ndarray,      # (n,)        — element masses [kg]
    velocities: jnp.ndarray,  # (n, 2)      — element velocities [m/s]
) -> jnp.ndarray:              # (n, 2)      — COM velocity for each element's component
    """
    Compute the centre-of-mass velocity for each connected component, then
    broadcast back to a per-element array.

    Useful for rigid-body drift mode (ship_trajectory=None): instead of
    integrating each element independently, all elements in the same component
    move at the component COM velocity.

    Implementation uses the fact that labels ∈ [0, n) so we can scatter into
    n-sized accumulator arrays without knowing the number of components.

    Parameters
    ----------
    labels : (n,) int32
        Component labels from find_connected_components_jax.
    masses : (n,)
        Element masses.
    velocities : (n, 2)
        Element velocities.

    Returns
    -------
    v_com : (n, 2)
        COM velocity of element i's component, for each i.
    """
    n = masses.shape[0]

    # Accumulate mass-weighted momenta per component label
    momentum = masses[:, None] * velocities          # (n, 2)
    total_m  = jnp.zeros(n).at[labels].add(masses)   # (n,) — mass per label
    total_px = jnp.zeros(n).at[labels].add(momentum[:, 0])
    total_py = jnp.zeros(n).at[labels].add(momentum[:, 1])

    # COM velocity per label (safe divide)
    safe_m   = jnp.maximum(total_m, 1e-30)
    v_com_x  = total_px / safe_m   # (n,) indexed by label
    v_com_y  = total_py / safe_m   # (n,) indexed by label

    # Broadcast back to each element via its label
    return jnp.stack([v_com_x[labels], v_com_y[labels]], axis=1)
