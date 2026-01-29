"""
Lindqvist formula for ice resistance validation.

The Lindqvist formula is a semi-empirical method for estimating
ice resistance on ships, used to validate DEM simulation results.

Reference: Liu et al. (2021), Fig. 11 in Wang et al. (2024)
"""

import jax.numpy as jnp
from typing import Tuple


def lindqvist_ice_resistance(
    L: float,
    B: float,
    T: float,
    phi: float,
    alpha: float,
    h_i: float,
    v: float,
    sigma_f: float = 0.6e6,
    rho_i: float = 920.0,
    rho_w: float = 1025.0,
    g: float = 9.81,
    mu: float = 0.1,
) -> Tuple[float, float, float, float]:
    """
    Calculate ice resistance using the Lindqvist formula.
    
    The Lindqvist formula estimates three components of ice resistance:
    - R_c: Crushing resistance
    - R_b: Bending resistance  
    - R_s: Submersion resistance
    
    Args:
        L: Ship length [m]
        B: Ship breadth [m]
        T: Ship draft [m]
        phi: Stem angle [rad]
        alpha: Waterline entrance angle [rad]
        h_i: Ice thickness [m]
        v: Ship velocity [m/s]
        sigma_f: Ice flexural strength [Pa]
        rho_i: Ice density [kg/m³]
        rho_w: Water density [kg/m³]
        g: Gravitational acceleration [m/s²]
        mu: Ice-hull friction coefficient [-]
    
    Returns:
        R_total: Total ice resistance [N]
        R_c: Crushing resistance [N]
        R_b: Bending resistance [N]
        R_s: Submersion resistance [N]
    """
    # Characteristic length
    l_c = jnp.sqrt(h_i / jnp.sqrt(rho_w * g / sigma_f))
    
    # Crushing component
    # R_c = 0.5 * sigma_f * h_i^2 * tan(phi) * (1 + mu*cot(phi)) / (1 - mu*tan(phi))
    tan_phi = jnp.tan(phi)
    cot_phi = 1.0 / tan_phi
    
    R_c = 0.5 * sigma_f * h_i**2 * tan_phi * (1 + mu * cot_phi) / (1 - mu * tan_phi + 1e-10)
    
    # Bending component
    # R_b = C_b * sigma_f * B * h_i^(3/2) / sqrt(E_i/rho_w*g)
    # Simplified form using characteristic length
    C_b = 0.15  # Empirical coefficient
    E_i = 2.4e9  # Ice elastic modulus [Pa]
    
    R_b = C_b * sigma_f * B * h_i**1.5 * jnp.sqrt(rho_w * g / E_i)
    
    # Submersion component
    # R_s = rho_w * g * h_i * B * T * (1 + 0.5 * phi) * C_s
    C_s = 0.85  # Submersion coefficient
    
    R_s = rho_w * g * h_i * B * T * (1 + 0.5 * phi) * C_s
    
    # Velocity-dependent correction
    # The above is for low speed; add velocity correction
    Fr = v / jnp.sqrt(g * l_c + 1e-10)  # Froude number based on characteristic length
    velocity_factor = 1 + 0.1 * Fr
    
    R_total = (R_c + R_b + R_s) * velocity_factor
    
    return R_total, R_c, R_b, R_s


def lindqvist_xuelong_validation(
    h_i: float,
    v: float = 5.0,
) -> float:
    """
    Calculate ice resistance for R/V Xuelong using Lindqvist formula.
    
    Uses parameters from Table 2 in the paper.
    
    Args:
        h_i: Ice thickness [m]
        v: Ship velocity [m/s]
    
    Returns:
        Ice resistance [N]
    """
    from ..config import XUELONG_PARAMS
    
    R_total, _, _, _ = lindqvist_ice_resistance(
        L=XUELONG_PARAMS.length,
        B=XUELONG_PARAMS.breadth,
        T=XUELONG_PARAMS.draft,
        phi=XUELONG_PARAMS.stem_angle,
        alpha=XUELONG_PARAMS.waterline_angle,
        h_i=h_i,
        v=v,
    )
    
    return R_total


def compare_with_dem_results(
    dem_force_mean: float,
    dem_force_peak: float,
    dem_force_peak_mean: float,
    h_i: float,
    v: float = 5.0,
) -> dict:
    """
    Compare DEM simulation results with Lindqvist formula.
    
    Following the methodology in Section 3.2 and Fig. 11.
    
    Args:
        dem_force_mean: Mean ice force from DEM [N]
        dem_force_peak: Peak ice force from DEM [N]
        dem_force_peak_mean: Mean of peak values from DEM [N]
        h_i: Ice thickness [m]
        v: Ship velocity [m/s]
    
    Returns:
        Dictionary with comparison metrics
    """
    R_lindqvist = lindqvist_xuelong_validation(h_i, v)
    
    return {
        'lindqvist': R_lindqvist,
        'dem_mean': dem_force_mean,
        'dem_peak': dem_force_peak,
        'dem_peak_mean': dem_force_peak_mean,
        'ratio_mean': dem_force_mean / R_lindqvist,
        'ratio_peak_mean': dem_force_peak_mean / R_lindqvist,
        'ratio_peak': dem_force_peak / R_lindqvist,
    }


def generate_lindqvist_curve(
    h_i_range: jnp.ndarray,
    v: float = 5.0,
) -> jnp.ndarray:
    """
    Generate Lindqvist resistance curve for a range of ice thicknesses.
    
    Args:
        h_i_range: Array of ice thicknesses [m]
        v: Ship velocity [m/s]
    
    Returns:
        Array of ice resistances [N]
    """
    resistances = []
    for h_i in h_i_range:
        R = lindqvist_xuelong_validation(float(h_i), v)
        resistances.append(R)
    
    return jnp.array(resistances)
