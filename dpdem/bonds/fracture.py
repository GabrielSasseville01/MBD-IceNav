"""
Fracture mechanics for bond failure in DPDEM.

Implements the mixed-mode fracture criterion and damage evolution
from Eqs. 14-19 in the paper.

Reference: Wang et al. (2024), Liu and Ji (2021), Benzeggagh and Kenane (1996)
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from ..config import ETA_BK


@partial(jax.jit, static_argnums=())
def macaulay_bracket(x: float) -> float:
    """
    Macaulay bracket: ⟨x⟩ = max(0, x)
    
    Returns x if x >= 0, else 0.
    
    Args:
        x: Input value
    
    Returns:
        max(0, x)
    """
    return jnp.maximum(x, 0.0)


@partial(jax.jit, static_argnums=())
def compute_strength_criterion(
    sigma: float,
    tau: float,
    sigma_bar: float,
    tau_bar: float,
) -> float:
    """
    Compute the mixed-mode strength criterion.
    
    Implements Eq. 14:
    (⟨σ⟩/σ̄)² + (τ/τ̄)² >= 1
    
    Returns the left-hand side value. Failure occurs when >= 1.
    
    Args:
        sigma: Normal stress [Pa] (positive = tension)
        tau: Shear stress magnitude [Pa]
        sigma_bar: Normal strength [Pa]
        tau_bar: Shear strength [Pa]
    
    Returns:
        Strength criterion value (>= 1 means failure initiation)
    """
    # Only tensile stress contributes (Macaulay bracket)
    sigma_pos = macaulay_bracket(sigma)
    
    # Compute criterion
    normal_term = (sigma_pos / sigma_bar) ** 2
    shear_term = (tau / tau_bar) ** 2
    
    return normal_term + shear_term


@partial(jax.jit, static_argnums=())
def compute_mohr_coulomb_strength(
    C: float,
    mu_b: float,
    sigma: float,
) -> float:
    """
    Compute Mohr-Coulomb shear strength.
    
    Implements Eq. 15:
    τ̄ = C - μ_b * σ
    
    Note: σ is positive in tension, so compression (negative σ)
    increases shear strength.
    
    Args:
        C: Cohesion strength [Pa]
        mu_b: Internal friction coefficient [-]
        sigma: Normal stress [Pa] (positive = tension)
    
    Returns:
        Shear strength τ̄ [Pa]
    """
    return C - mu_b * sigma


@partial(jax.jit, static_argnums=())
def compute_damage_variable(
    delta_m: float,
    delta_m_0: float,
    delta_m_f: float,
    delta_m_max: float,
) -> float:
    """
    Compute damage variable D.
    
    Implements Eq. 16:
    D = δ_m^f * (δ_m^max - δ_m^0) / (δ_m^max * (δ_m^f - δ_m^0))
    
    D ranges from 0 (undamaged) to 1 (fully failed).
    
    Args:
        delta_m: Current equivalent deformation [m]
        delta_m_0: Deformation at damage initiation [m]
        delta_m_f: Deformation at complete failure [m]
        delta_m_max: Maximum deformation in loading history [m]
    
    Returns:
        Damage variable D [0, 1]
    """
    # Ensure delta_m_max >= delta_m_0
    delta_m_max = jnp.maximum(delta_m_max, delta_m_0 + 1e-10)
    
    # Compute damage
    numerator = delta_m_f * (delta_m_max - delta_m_0)
    denominator = delta_m_max * (delta_m_f - delta_m_0)
    
    D = numerator / (denominator + 1e-10)
    
    # Clamp to [0, 1]
    return jnp.clip(D, 0.0, 1.0)


@partial(jax.jit, static_argnums=())
def compute_effective_stiffness(
    D: float,
    k: float,
) -> float:
    """
    Compute effective stiffness with damage.
    
    Implements Eq. 17:
    k' = (1 - D) * k
    
    Args:
        D: Damage variable [0, 1]
        k: Undamaged stiffness
    
    Returns:
        Effective stiffness
    """
    return (1.0 - D) * k


@partial(jax.jit, static_argnums=())
def compute_benzeggagh_kenane_energy(
    G_I: float,
    G_II: float,
    G_c_I: float,
    G_c_II: float,
    eta: float = ETA_BK,
) -> float:
    """
    Compute critical fracture energy using Benzeggagh-Kenane model.
    
    Implements Eq. 18:
    G_c = G_c^I + (G_c^II - G_c^I) * (G_II / (G_I + G_II))^η
    
    Args:
        G_I: Current Mode I (tensile) fracture energy [J/m²]
        G_II: Current Mode II (shear) fracture energy [J/m²]
        G_c_I: Critical Mode I fracture energy [J/m²]
        G_c_II: Critical Mode II fracture energy [J/m²]
        eta: B-K exponent (1.75 for brittle materials)
    
    Returns:
        Critical mixed-mode fracture energy G_c [J/m²]
    """
    G_total = G_I + G_II + 1e-10
    ratio = G_II / G_total
    
    G_c = G_c_I + (G_c_II - G_c_I) * jnp.power(ratio, eta)
    
    return G_c


@partial(jax.jit, static_argnums=())
def compute_fracture_energy_ratio(
    delta_n: float,
    delta_s: float,
) -> float:
    """
    Compute the ratio of shear to tensile fracture energies.
    
    Implements Eq. 19:
    G_II / G_I = δ_s² / ⟨δ_n⟩²
    
    Args:
        delta_n: Normal deformation [m] (positive = tension)
        delta_s: Shear deformation [m]
    
    Returns:
        Ratio G_II / G_I
    """
    delta_n_pos = macaulay_bracket(delta_n)
    
    # Avoid division by zero
    return delta_s ** 2 / (delta_n_pos ** 2 + 1e-10)


@partial(jax.jit, static_argnums=())
def compute_critical_deformation(
    G_c: float,
    sigma_m_0: float,
) -> float:
    """
    Compute critical deformation at bond failure.
    
    δ_m^f = 2 * G_c / σ_m^0
    
    where σ_m^0 is the stress at damage initiation.
    
    Args:
        G_c: Critical fracture energy [J/m²]
        sigma_m_0: Stress at damage initiation [Pa]
    
    Returns:
        Critical deformation δ_m^f [m]
    """
    return 2.0 * G_c / (jnp.abs(sigma_m_0) + 1e-10)


@partial(jax.jit, static_argnums=())
def compute_equivalent_deformation(
    delta_n: float,
    delta_s: float,
) -> float:
    """
    Compute equivalent deformation for mixed-mode loading.
    
    δ_m = √(⟨δ_n⟩² + δ_s²)
    
    Args:
        delta_n: Normal deformation [m]
        delta_s: Shear deformation magnitude [m]
    
    Returns:
        Equivalent deformation [m]
    """
    delta_n_pos = macaulay_bracket(delta_n)
    return jnp.sqrt(delta_n_pos ** 2 + delta_s ** 2)


@partial(jax.jit, static_argnums=())
def check_bond_failure(
    sigma_n: float,
    sigma_t: float,
    sigma_bar: float,
    C: float,
    mu_b: float,
    delta_m: float,
    delta_m_f: float,
    intact: bool,
) -> Tuple[bool, bool]:
    """
    Check if a bond has failed.
    
    A bond fails when:
    1. The strength criterion is exceeded (damage initiation), AND
    2. The equivalent deformation exceeds the critical value (complete failure)
    
    Args:
        sigma_n: Normal stress [Pa]
        sigma_t: Shear stress magnitude [Pa]
        sigma_bar: Normal strength [Pa]
        C: Cohesion [Pa]
        mu_b: Internal friction coefficient
        delta_m: Current equivalent deformation [m]
        delta_m_f: Critical deformation at failure [m]
        intact: Whether bond is currently intact
    
    Returns:
        damage_initiated: Whether damage has initiated (strength exceeded)
        failed: Whether bond has completely failed
    """
    # Compute shear strength (Mohr-Coulomb)
    tau_bar = compute_mohr_coulomb_strength(C, mu_b, sigma_n)
    tau_bar = jnp.maximum(tau_bar, 1e-6)  # Ensure positive
    
    # Check strength criterion
    criterion = compute_strength_criterion(sigma_n, sigma_t, sigma_bar, tau_bar)
    damage_initiated = criterion >= 1.0
    
    # Check for complete failure
    failed = delta_m >= delta_m_f
    
    # Only fail if was intact and now failed
    failed = intact & failed
    
    return damage_initiated, failed


@partial(jax.jit, static_argnums=())
def update_bond_state(
    sigma_n: float,
    sigma_t: float,
    delta_n: float,
    delta_s: float,
    sigma_bar: float,
    C: float,
    mu_b: float,
    G_c_I: float,
    G_c_II: float,
    current_damage: float,
    current_delta_m_max: float,
    current_delta_m_0: float,
    intact: bool,
) -> Tuple[float, float, float, bool]:
    """
    Update bond damage state based on current loading.
    
    Args:
        sigma_n: Current normal stress [Pa]
        sigma_t: Current shear stress magnitude [Pa]
        delta_n: Normal deformation [m]
        delta_s: Shear deformation [m]
        sigma_bar: Normal strength [Pa]
        C: Cohesion [Pa]
        mu_b: Internal friction coefficient
        G_c_I: Mode I critical fracture energy [J/m²]
        G_c_II: Mode II critical fracture energy [J/m²]
        current_damage: Current damage variable
        current_delta_m_max: Current maximum deformation
        current_delta_m_0: Current damage initiation deformation
        intact: Whether bond is intact
    
    Returns:
        new_damage: Updated damage variable
        new_delta_m_max: Updated maximum deformation
        new_delta_m_0: Updated damage initiation deformation
        new_intact: Updated intact status
    """
    # Compute equivalent deformation
    delta_m = compute_equivalent_deformation(delta_n, delta_s)
    
    # Update maximum deformation
    new_delta_m_max = jnp.maximum(current_delta_m_max, delta_m)
    
    # Compute shear strength
    tau_bar = compute_mohr_coulomb_strength(C, mu_b, sigma_n)
    tau_bar = jnp.maximum(tau_bar, 1e-6)
    
    # Check if damage initiates
    criterion = compute_strength_criterion(sigma_n, sigma_t, sigma_bar, tau_bar)
    damage_initiating = (criterion >= 1.0) & (current_delta_m_0 == 0.0)
    
    # Set damage initiation deformation
    new_delta_m_0 = jnp.where(damage_initiating, delta_m, current_delta_m_0)
    
    # Compute critical fracture energy (B-K model)
    G_ratio = compute_fracture_energy_ratio(delta_n, delta_s)
    G_I = 1.0 / (1.0 + G_ratio + 1e-10)  # Normalized
    G_II = G_ratio / (1.0 + G_ratio + 1e-10)
    G_c = compute_benzeggagh_kenane_energy(G_I, G_II, G_c_I, G_c_II)
    
    # Compute critical deformation
    sigma_m_0 = jnp.sqrt(macaulay_bracket(sigma_n) ** 2 + sigma_t ** 2 + 1e-10)
    delta_m_f = compute_critical_deformation(G_c, sigma_m_0)
    
    # Only compute damage if damage has initiated
    has_damage = new_delta_m_0 > 0.0
    
    new_damage = jnp.where(
        has_damage,
        compute_damage_variable(delta_m, new_delta_m_0, delta_m_f, new_delta_m_max),
        0.0
    )
    
    # Ensure damage only increases
    new_damage = jnp.maximum(new_damage, current_damage)
    
    # Check for complete failure
    new_intact = intact & (new_damage < 1.0)
    
    return new_damage, new_delta_m_max, new_delta_m_0, new_intact
