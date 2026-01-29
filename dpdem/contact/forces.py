"""
Hertz-Mindlin contact force model for DPDEM.

Implements the non-linear contact force model from Eqs. 9-10 in the paper.

Reference: Wang et al. (2024), Liu and Ji (2021, 2022)
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from ..config import (
    compute_effective_modulus,
    compute_effective_radius,
    compute_normal_stiffness,
    compute_max_tangential_overlap,
    compute_equivalent_mass,
)


@partial(jax.jit, static_argnums=())
def hertz_normal_force(
    delta_n: float,
    delta_n_dot: float,
    k_n: float,
    c_n: float,
) -> float:
    """
    Compute Hertzian normal contact force.
    
    Implements Eq. 9 from the paper:
    
    F_n = F_n^e + F_n^v = k_n * δ_n^(3/2) + c_n * √δ_n * δ̇_n
    
    Args:
        delta_n: Normal overlap [m] (positive when penetrating)
        delta_n_dot: Normal overlap rate [m/s] (positive when approaching)
        k_n: Normal stiffness [N/m^(3/2)]
        c_n: Normal damping coefficient [N·s/m^(3/2)]
    
    Returns:
        Normal force magnitude [N] (positive = repulsive)
    """
    # Only compute force if in contact
    delta_n_pos = jnp.maximum(delta_n, 0.0)
    
    # Elastic force (Hertzian)
    F_n_elastic = k_n * jnp.power(delta_n_pos, 1.5)
    
    # Viscous force
    F_n_viscous = c_n * jnp.sqrt(delta_n_pos) * delta_n_dot
    
    # Total normal force (ensure it doesn't become tensile)
    F_n_total = F_n_elastic + F_n_viscous
    
    return jnp.maximum(F_n_total, 0.0)


@partial(jax.jit, static_argnums=())
def mindlin_tangential_force(
    delta_t: jnp.ndarray,
    delta_t_dot: jnp.ndarray,
    F_n_elastic: float,
    mu: float,
    delta_t_max: float,
    c_t: float,
    m_AB: float,
) -> jnp.ndarray:
    """
    Compute Mindlin tangential contact force.
    
    Implements Eq. 10 from the paper:
    
    F_t = F_t^e + F_t^v
    
    F_t^e = μ|F_n^e| * [1 - (1 - δ_t/δ_t^max)^(3/2)] * (δ_t/|δ_t|)
    
    F_t^v = c_t * √(6μm_AB|F_n^e|√(1 - δ_t/δ_t^max) / δ_t^max) * δ̇_t
    
    Args:
        delta_t: (3,) tangential displacement [m]
        delta_t_dot: (3,) tangential velocity [m/s]
        F_n_elastic: Elastic normal force magnitude [N]
        mu: Friction coefficient [-]
        delta_t_max: Maximum tangential overlap [m]
        c_t: Tangential damping coefficient [-]
        m_AB: Equivalent mass [kg]
    
    Returns:
        (3,) tangential force vector [N]
    """
    delta_t_mag = jnp.linalg.norm(delta_t)
    
    # Avoid division by zero
    delta_t_mag_safe = jnp.maximum(delta_t_mag, 1e-10)
    delta_t_max_safe = jnp.maximum(delta_t_max, 1e-10)
    
    # Direction of tangential displacement
    delta_t_dir = delta_t / delta_t_mag_safe
    
    # Ratio of tangential displacement to maximum
    ratio = jnp.minimum(delta_t_mag / delta_t_max_safe, 1.0)
    
    # Elastic tangential force (Mindlin)
    # F_t^e = μ|F_n^e| * [1 - (1 - δ_t/δ_t^max)^(3/2)] * (δ_t/|δ_t|)
    F_t_elastic_mag = mu * jnp.abs(F_n_elastic) * (1.0 - jnp.power(1.0 - ratio, 1.5))
    F_t_elastic = F_t_elastic_mag * delta_t_dir
    
    # Viscous tangential force
    # c_t * √(6μm_AB|F_n^e|√(1 - δ_t/δ_t^max) / δ_t^max) * δ̇_t
    sqrt_term = jnp.sqrt(jnp.maximum(1.0 - ratio, 0.0))
    viscous_coef = c_t * jnp.sqrt(
        6.0 * mu * m_AB * jnp.abs(F_n_elastic) * sqrt_term / delta_t_max_safe
    )
    F_t_viscous = viscous_coef * delta_t_dot
    
    # Total tangential force
    F_t_total = F_t_elastic + F_t_viscous
    
    # Apply Coulomb limit: |F_t| <= μ|F_n|
    F_t_mag = jnp.linalg.norm(F_t_total)
    F_t_max = mu * jnp.abs(F_n_elastic)
    
    # Scale down if exceeding Coulomb limit
    scale = jnp.where(
        F_t_mag > F_t_max,
        F_t_max / (F_t_mag + 1e-10),
        1.0
    )
    
    return F_t_total * scale


@partial(jax.jit, static_argnums=())
def compute_contact_force(
    delta_n: float,
    delta_n_dot: float,
    delta_t: jnp.ndarray,
    delta_t_dot: jnp.ndarray,
    contact_normal: jnp.ndarray,
    E: float,
    nu: float,
    R_i: float,
    R_j: float,
    m_i: float,
    m_j: float,
    mu: float,
    c_n: float = 0.0,
    c_t: float = 0.0,
) -> Tuple[jnp.ndarray, float]:
    """
    Compute total contact force between two particles.
    
    Args:
        delta_n: Normal overlap [m]
        delta_n_dot: Normal overlap rate [m/s]
        delta_t: (3,) tangential displacement [m]
        delta_t_dot: (3,) tangential velocity [m/s]
        contact_normal: (3,) unit contact normal
        E: Elastic modulus [Pa]
        nu: Poisson's ratio [-]
        R_i: Equivalent radius of particle i [m]
        R_j: Equivalent radius of particle j [m]
        m_i: Mass of particle i [kg]
        m_j: Mass of particle j [kg]
        mu: Friction coefficient [-]
        c_n: Normal damping coefficient [-]
        c_t: Tangential damping coefficient [-]
    
    Returns:
        F_contact: (3,) total contact force on particle i [N]
        F_n_mag: Normal force magnitude [N]
    """
    # Compute derived parameters
    E_star = compute_effective_modulus(E, nu)
    R_star = compute_effective_radius(R_i, R_j)
    k_n = (4.0 / 3.0) * E_star * jnp.sqrt(R_star)
    m_AB = compute_equivalent_mass(m_i, m_j)
    delta_t_max = compute_max_tangential_overlap(mu, jnp.maximum(delta_n, 1e-10), nu)
    
    # Normal force
    F_n_elastic = k_n * jnp.power(jnp.maximum(delta_n, 0.0), 1.5)
    F_n_viscous = c_n * jnp.sqrt(jnp.maximum(delta_n, 0.0)) * delta_n_dot
    F_n_mag = jnp.maximum(F_n_elastic + F_n_viscous, 0.0)
    F_n = F_n_mag * contact_normal
    
    # Tangential force
    F_t = mindlin_tangential_force(
        delta_t, delta_t_dot, F_n_elastic, mu, delta_t_max, c_t, m_AB
    )
    
    # Total force on particle i (normal pushes apart, tangential opposes sliding)
    F_contact = F_n - F_t
    
    return F_contact, F_n_mag


@partial(jax.jit, static_argnums=())
def compute_contact_moment(
    F_contact: jnp.ndarray,
    contact_point: jnp.ndarray,
    position: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute moment from contact force about particle center.
    
    M = r × F
    
    where r = contact_point - position
    
    Args:
        F_contact: (3,) contact force [N]
        contact_point: (3,) contact point location [m]
        position: (3,) particle center [m]
    
    Returns:
        (3,) moment about particle center [N·m]
    """
    r = contact_point - position
    return jnp.cross(r, F_contact)


@partial(jax.jit, static_argnums=())
def update_tangential_displacement(
    delta_t_old: jnp.ndarray,
    v_t: jnp.ndarray,
    contact_normal: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Update tangential displacement for a contact.
    
    The tangential displacement is accumulated incrementally and
    must be corrected for rotation of the contact plane.
    
    Args:
        delta_t_old: (3,) previous tangential displacement
        v_t: (3,) tangential velocity at contact
        contact_normal: (3,) current contact normal
        dt: Time step [s]
    
    Returns:
        (3,) updated tangential displacement
    """
    # Project old displacement onto current tangent plane
    # This corrects for rotation of the contact
    delta_t_proj = delta_t_old - jnp.dot(delta_t_old, contact_normal) * contact_normal
    
    # Add new displacement
    delta_t_new = delta_t_proj + v_t * dt
    
    return delta_t_new


@partial(jax.jit, static_argnums=())
def compute_damping_coefficients(
    E: float,
    nu: float,
    R_i: float,
    R_j: float,
    m_i: float,
    m_j: float,
    restitution: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute damping coefficients from restitution coefficient.
    
    Uses the method from Tsuji et al. (1992).
    
    Args:
        E: Elastic modulus [Pa]
        nu: Poisson's ratio [-]
        R_i: Equivalent radius of particle i [m]
        R_j: Equivalent radius of particle j [m]
        m_i: Mass of particle i [kg]
        m_j: Mass of particle j [kg]
        restitution: Coefficient of restitution [-]
    
    Returns:
        c_n: Normal damping coefficient
        c_t: Tangential damping coefficient
    """
    # Effective parameters
    m_AB = compute_equivalent_mass(m_i, m_j)
    E_star = compute_effective_modulus(E, nu)
    R_star = compute_effective_radius(R_i, R_j)
    k_n = (4.0 / 3.0) * E_star * jnp.sqrt(R_star)
    
    # Damping from restitution
    # This is an approximation; exact relationship depends on velocity
    ln_e = jnp.log(jnp.maximum(restitution, 0.01))
    beta = -ln_e / jnp.sqrt(jnp.pi**2 + ln_e**2)
    
    c_n = 2.0 * beta * jnp.sqrt(k_n * m_AB)
    c_t = c_n  # Often taken equal for simplicity
    
    return c_n, c_t
