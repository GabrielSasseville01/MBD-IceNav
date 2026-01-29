"""
Rigid body dynamics for sea ice elements.

Implements Newton's second law for translational and rotational motion
of ice elements (Eqs. 2-3 in the paper).

Reference: Wang et al. (2024), Liu et al. (2022)
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from ..kinematics.quaternion import (
    quaternion_to_rotation_matrix,
    euler_equations_local,
    euler_equations_global,
    transform_to_local,
)


@partial(jax.jit, static_argnums=())
def compute_ice_element_acceleration(
    F_contact: jnp.ndarray,
    F_bond: jnp.ndarray,
    F_buoyancy: jnp.ndarray,
    F_drag: jnp.ndarray,
    mass: float,
    g: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute linear acceleration of an ice element.
    
    Implements Eq. 2:
    m_i * dv_i/dt = Σ(F_n,ij + F_t,ij) + m_i * g
    
    Extended to include buoyancy and drag forces.
    
    Args:
        F_contact: (3,) total contact force [N]
        F_bond: (3,) total bond force [N]
        F_buoyancy: (3,) buoyancy force [N]
        F_drag: (3,) drag force [N]
        mass: Element mass [kg]
        g: (3,) gravity vector [m/s²]
    
    Returns:
        (3,) linear acceleration [m/s²]
    """
    # Total force
    F_total = F_contact + F_bond + F_buoyancy + F_drag + mass * g
    
    # Acceleration (Newton's second law)
    acceleration = F_total / mass
    
    return acceleration


@partial(jax.jit, static_argnums=())
def compute_ice_element_angular_acceleration(
    M_contact: jnp.ndarray,
    M_bond: jnp.ndarray,
    quaternion: jnp.ndarray,
    omega: jnp.ndarray,
    inertia_local: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute angular acceleration of an ice element.
    
    Implements Eq. 3:
    I_i * dω_i/dt = Σ(M_n,ij + M_t,ij)
    
    Using Euler's equations in the local frame (Eqs. 6-8).
    
    Args:
        M_contact: (3,) total contact moment in global frame [N·m]
        M_bond: (3,) total bond moment in global frame [N·m]
        quaternion: (4,) element orientation
        omega: (3,) angular velocity in global frame [rad/s]
        inertia_local: (3,) diagonal inertia tensor [I_xx, I_yy, I_zz] [kg·m²]
    
    Returns:
        (3,) angular acceleration in global frame [rad/s²]
    """
    # Total moment
    M_total = M_contact + M_bond
    
    # Use Euler's equations in global frame (handles frame transformation internally)
    alpha = euler_equations_global(omega, M_total, quaternion, inertia_local)
    
    return alpha


@partial(jax.jit, static_argnums=())
def update_ice_element_state(
    position: jnp.ndarray,
    velocity: jnp.ndarray,
    quaternion: jnp.ndarray,
    omega: jnp.ndarray,
    acceleration: jnp.ndarray,
    alpha: jnp.ndarray,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Update ice element state using semi-implicit Euler integration.
    
    Args:
        position: (3,) current position [m]
        velocity: (3,) current velocity [m/s]
        quaternion: (4,) current orientation
        omega: (3,) current angular velocity [rad/s]
        acceleration: (3,) linear acceleration [m/s²]
        alpha: (3,) angular acceleration [rad/s²]
        dt: Time step [s]
    
    Returns:
        new_position: (3,) updated position
        new_velocity: (3,) updated velocity
        new_quaternion: (4,) updated orientation (normalized)
        new_omega: (3,) updated angular velocity
    """
    # Semi-implicit Euler: update velocity first, then position
    new_velocity = velocity + acceleration * dt
    new_position = position + new_velocity * dt
    
    new_omega = omega + alpha * dt
    
    # Update quaternion
    from ..kinematics.quaternion import quaternion_derivative, quaternion_normalize
    dq = quaternion_derivative(quaternion, new_omega)
    new_quaternion = quaternion + dq * dt
    new_quaternion = quaternion_normalize(new_quaternion)
    
    return new_position, new_velocity, new_quaternion, new_omega


def compute_all_ice_element_accelerations(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    quaternions: jnp.ndarray,
    omegas: jnp.ndarray,
    masses: jnp.ndarray,
    inertias_local: jnp.ndarray,
    contact_forces: jnp.ndarray,
    contact_moments: jnp.ndarray,
    bond_forces: jnp.ndarray,
    bond_moments: jnp.ndarray,
    volumes: jnp.ndarray,
    rho_ice: float,
    rho_water: float,
    C_d: float,
    g: jnp.ndarray,
    v_water: jnp.ndarray,
    waterline_z: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute accelerations for all ice elements.
    
    Args:
        positions: (N, 3) positions
        velocities: (N, 3) velocities
        quaternions: (N, 4) orientations
        omegas: (N, 3) angular velocities
        masses: (N,) masses
        inertias_local: (N, 3) diagonal inertias
        contact_forces: (N, 3) total contact forces per element
        contact_moments: (N, 3) total contact moments per element
        bond_forces: (N, 3) total bond forces per element
        bond_moments: (N, 3) total bond moments per element
        volumes: (N,) element volumes
        rho_ice: Ice density
        rho_water: Water density
        C_d: Drag coefficient
        g: (3,) gravity vector
        v_water: (3,) water current velocity
        waterline_z: Waterline z-coordinate
    
    Returns:
        accelerations: (N, 3) linear accelerations
        alphas: (N, 3) angular accelerations
    """
    from ..hydrodynamics.drag import compute_ice_element_buoyancy, compute_ice_element_drag
    
    n_particles = len(positions)
    accelerations = jnp.zeros((n_particles, 3))
    alphas = jnp.zeros((n_particles, 3))
    
    for i in range(n_particles):
        # Hydrodynamic forces
        F_buoy = compute_ice_element_buoyancy(
            positions[i], volumes[i], rho_ice, rho_water, g, waterline_z
        )
        F_drag = compute_ice_element_drag(
            positions[i], velocities[i], volumes[i], v_water, C_d, rho_water, waterline_z
        )
        
        # Linear acceleration
        a = compute_ice_element_acceleration(
            contact_forces[i], bond_forces[i], F_buoy, F_drag, masses[i], g
        )
        
        # Angular acceleration
        alpha = compute_ice_element_angular_acceleration(
            contact_moments[i], bond_moments[i],
            quaternions[i], omegas[i], inertias_local[i]
        )
        
        accelerations = accelerations.at[i].set(a)
        alphas = alphas.at[i].set(alpha)
    
    return accelerations, alphas
