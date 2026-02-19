"""
DPDEM-style bond fracture mechanics for the ship-ice simulation.

Implements the hybrid failure criterion, Mohr-Coulomb shear strength,
damage variable, and Benzeggagh-Kenane fracture energy (Eqs. 14–19)
using NumPy so the ship-ice simulation can use the same physics as
the 3D DPDEM library without a JAX dependency.

Reference: Wang et al. (2024), Liu and Ji (2021), Benzeggagh and Kenane (1996)
"""

import numpy as np


def macaulay_bracket(x: np.ndarray) -> np.ndarray:
    """⟨x⟩ = max(0, x). Only tensile (positive) contributes to failure."""
    return np.maximum(x, 0.0)


# Max stress magnitude to avoid overflow in squares and ratios (Pa)
_STRESS_CLIP = 1e12
# Min stress to allow damage progression (avoid D=1 from zero-load numerical noise)
_SIGMA_MIN_FOR_DAMAGE = 1e4
# Min equivalent deformation (m) to initiate damage (avoid failure from noise)
_DELTA_M_MIN_INITIATE = 1e-5
# Max damage increase per step so fracture is visible over several frames (not instant)
_DAMAGE_RATE_CAP = 0.15


def compute_strength_criterion(
    sigma: np.ndarray,
    tau: np.ndarray,
    sigma_bar: np.ndarray,
    tau_bar: np.ndarray,
) -> np.ndarray:
    """
    Mixed-mode strength criterion: (⟨σ⟩/σ̄)² + (τ/τ̄)².
    Failure when >= 1.
    """
    sigma_pos = macaulay_bracket(np.clip(sigma, -_STRESS_CLIP, _STRESS_CLIP))
    tau = np.clip(tau, -_STRESS_CLIP, _STRESS_CLIP)
    sigma_bar = np.maximum(np.clip(sigma_bar, 1e-10, _STRESS_CLIP), 1e-10)
    tau_bar = np.maximum(np.clip(tau_bar, 1e-10, _STRESS_CLIP), 1e-10)
    normal_term = (sigma_pos / sigma_bar) ** 2
    shear_term = (tau / tau_bar) ** 2
    return np.clip(normal_term + shear_term, 0.0, 1e10)


def compute_mohr_coulomb_strength(
    C: float,
    mu_b: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Shear strength τ̄ = C - μ_b * σ. Compression (σ < 0) increases τ̄."""
    return np.maximum(C - mu_b * sigma, 1e-6)


def compute_equivalent_deformation(delta_n: np.ndarray, delta_s: np.ndarray) -> np.ndarray:
    """δ_m = √(⟨δ_n⟩² + δ_s²)."""
    delta_n = np.clip(delta_n, -1e3, 1e3)
    delta_s = np.clip(np.abs(delta_s), 0.0, 1e3)
    delta_n_pos = macaulay_bracket(delta_n)
    return np.sqrt(np.clip(delta_n_pos ** 2 + delta_s ** 2 + 1e-20, 0.0, 1e12))


def compute_damage_variable(
    delta_m: np.ndarray,
    delta_m_0: np.ndarray,
    delta_m_f: np.ndarray,
    delta_m_max: np.ndarray,
) -> np.ndarray:
    """
    D = δ_m^f * (δ_m^max - δ_m^0) / (δ_m^max * (δ_m^f - δ_m^0)).
    D in [0, 1]; 1 = fully failed.
    """
    delta_m = np.clip(delta_m, 0.0, 1e6)
    delta_m_0 = np.clip(delta_m_0, 0.0, 1e6)
    delta_m_f = np.maximum(np.clip(delta_m_f, 1e-10, 1e6), 1e-10)
    delta_m_max = np.maximum(np.clip(delta_m_max, 0.0, 1e6), delta_m_0 + 1e-10)
    denom = delta_m_max * (delta_m_f - delta_m_0 + 1e-10)
    denom = np.maximum(denom, 1e-20)
    D = delta_m_f * (delta_m_max - delta_m_0) / denom
    return np.clip(D, 0.0, 1.0)


def compute_fracture_energy_ratio(delta_n: np.ndarray, delta_s: np.ndarray) -> np.ndarray:
    """G_II / G_I = δ_s² / ⟨δ_n⟩²."""
    delta_n_pos = macaulay_bracket(delta_n)
    return delta_s ** 2 / (delta_n_pos ** 2 + 1e-10)


def compute_benzeggagh_kenane_energy(
    G_I: np.ndarray,
    G_II: np.ndarray,
    G_c_I: float,
    G_c_II: float,
    eta: float = 1.75,
) -> np.ndarray:
    """G_c = G_c^I + (G_c^II - G_c^I) * (G_II / (G_I + G_II))^η."""
    G_total = G_I + G_II + 1e-10
    ratio = G_II / G_total
    return G_c_I + (G_c_II - G_c_I) * np.power(ratio, eta)


def compute_critical_deformation(G_c: np.ndarray, sigma_m_0: np.ndarray) -> np.ndarray:
    """δ_m^f = 2 * G_c / σ_m^0."""
    return 2.0 * G_c / (np.abs(sigma_m_0) + 1e-10)


def update_bond_damage_state(
    delta_n: np.ndarray,
    delta_s: np.ndarray,
    sigma_n: np.ndarray,
    sigma_t: np.ndarray,
    sigma_bar: np.ndarray,
    C: float,
    mu_b: float,
    G_c_I: float,
    G_c_II: float,
    eta_bk: float,
    current_damage: np.ndarray,
    current_delta_m_max: np.ndarray,
    current_delta_m_0: np.ndarray,
    intact: np.ndarray,
) -> tuple:
    """
    Update damage and intact state for all bonds (vectorized).

    Returns:
        new_damage: (n_bonds,)
        new_delta_m_max: (n_bonds,)
        new_delta_m_0: (n_bonds,)
        new_intact: (n_bonds,) bool
    """
    delta_n = np.clip(delta_n, -1e3, 1e3)
    delta_s = np.clip(delta_s, -1e3, 1e3)
    sigma_bar = np.clip(sigma_bar, 1e-10, _STRESS_CLIP)
    delta_m = compute_equivalent_deformation(delta_n, delta_s)
    new_delta_m_max = np.maximum(current_delta_m_max, delta_m)

    tau_bar = compute_mohr_coulomb_strength(C, mu_b, sigma_n)
    criterion = compute_strength_criterion(sigma_n, sigma_t, sigma_bar, tau_bar)
    # Only start damage when criterion is met AND deformation is above noise (avoid instant fracture at t=0)
    damage_initiating = (
        (criterion >= 1.0) & (current_delta_m_0 <= 0.0) & (delta_m > _DELTA_M_MIN_INITIATE)
    )
    new_delta_m_0 = np.where(damage_initiating, delta_m, current_delta_m_0)

    G_ratio = compute_fracture_energy_ratio(delta_n, delta_s)
    G_I = 1.0 / (1.0 + G_ratio + 1e-10)
    G_II = G_ratio / (1.0 + G_ratio + 1e-10)
    G_c = compute_benzeggagh_kenane_energy(G_I, G_II, G_c_I, G_c_II, eta_bk)

    sigma_n = np.clip(sigma_n, -_STRESS_CLIP, _STRESS_CLIP)
    sigma_t = np.clip(sigma_t, -_STRESS_CLIP, _STRESS_CLIP)
    sigma_m_0_raw = np.sqrt(np.clip(macaulay_bracket(sigma_n) ** 2 + sigma_t ** 2 + 1e-20, 0.0, 1e24))
    sigma_m_0 = np.maximum(sigma_m_0_raw, 1e-10)
    delta_m_f = compute_critical_deformation(G_c, sigma_m_0)

    # Only apply damage when stress is meaningful (otherwise zero-load gives D=1 from formula)
    meaningful_stress = sigma_m_0_raw >= _SIGMA_MIN_FOR_DAMAGE
    has_damage = (new_delta_m_0 > 0.0) & meaningful_stress
    computed_damage = np.where(
        has_damage,
        compute_damage_variable(delta_m, new_delta_m_0, delta_m_f, new_delta_m_max),
        current_damage,
    )
    new_damage = np.where(meaningful_stress, np.maximum(computed_damage, current_damage), current_damage)
    # Cap damage increase per step so fracture progresses over several frames (not instant)
    new_damage = np.minimum(new_damage, current_damage + _DAMAGE_RATE_CAP)
    new_intact = intact & (new_damage < 1.0)

    return new_damage, new_delta_m_max, new_delta_m_0, new_intact
