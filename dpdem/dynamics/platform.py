"""
Rigid body dynamics for the floating platform.

Implements 6-DOF motion equations for the IRSPP platform (Eqs. 20-21).

Reference: Wang et al. (2024), Liu and Ji (2018, 2022)
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from ..kinematics.quaternion import (
    quaternion_to_rotation_matrix,
    euler_equations_global,
    quaternion_derivative,
    quaternion_normalize,
    transform_to_local,
)
from ..hydrodynamics.buoyancy import compute_total_buoyancy
from ..hydrodynamics.drag import compute_total_drag


@partial(jax.jit, static_argnums=())
def compute_platform_forces(
    F_ice: jnp.ndarray,
    F_buoyancy: jnp.ndarray,
    F_drag: jnp.ndarray,
    F_mooring: jnp.ndarray,
    mass: float,
    g: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute total force on the platform.
    
    Implements part of Eq. 20:
    F_i + F_b + F_d + F_p + F_g = m * du/dt
    
    Args:
        F_ice: (3,) ice contact force [N]
        F_buoyancy: (3,) buoyancy force [N]
        F_drag: (3,) drag force [N]
        F_mooring: (3,) mooring/propulsion force [N]
        mass: Platform mass [kg]
        g: (3,) gravity vector [m/s²]
    
    Returns:
        (3,) total force [N]
    """
    F_gravity = mass * g
    return F_ice + F_buoyancy + F_drag + F_mooring + F_gravity


@partial(jax.jit, static_argnums=())
def compute_platform_moments(
    M_ice: jnp.ndarray,
    M_buoyancy: jnp.ndarray,
    M_drag: jnp.ndarray,
    M_mooring: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute total moment on the platform.
    
    Implements part of Eq. 21:
    M_i + M_b + M_d + M_p = I * dω/dt
    
    Args:
        M_ice: (3,) moment from ice forces [N·m]
        M_buoyancy: (3,) buoyancy moment [N·m]
        M_drag: (3,) drag moment [N·m]
        M_mooring: (3,) mooring/propulsion moment [N·m]
    
    Returns:
        (3,) total moment [N·m]
    """
    return M_ice + M_buoyancy + M_drag + M_mooring


@partial(jax.jit, static_argnums=())
def compute_platform_accelerations(
    F_total: jnp.ndarray,
    M_total: jnp.ndarray,
    mass: float,
    inertia: jnp.ndarray,
    quaternion: jnp.ndarray,
    omega: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute linear and angular accelerations of the platform.
    
    Implements Eqs. 20-21.
    
    Args:
        F_total: (3,) total force [N]
        M_total: (3,) total moment [N·m]
        mass: Platform mass [kg]
        inertia: (3, 3) inertia tensor in body frame [kg·m²]
        quaternion: (4,) platform orientation
        omega: (3,) angular velocity in global frame [rad/s]
    
    Returns:
        acceleration: (3,) linear acceleration [m/s²]
        alpha: (3,) angular acceleration [rad/s²]
    """
    # Linear acceleration
    acceleration = F_total / mass
    
    # Angular acceleration using Euler's equations
    # Transform moment to body frame
    R = quaternion_to_rotation_matrix(quaternion)
    M_local = R.T @ M_total
    omega_local = R.T @ omega
    
    # Diagonal inertia approximation
    I_diag = jnp.diag(inertia)
    
    # Euler's equations in local frame
    alpha_local = euler_equations_local_full(omega_local, M_local, I_diag)
    
    # Transform back to global
    alpha = R @ alpha_local
    
    return acceleration, alpha


@partial(jax.jit, static_argnums=())
def euler_equations_local_full(
    omega_local: jnp.ndarray,
    M_local: jnp.ndarray,
    I_diag: jnp.ndarray,
) -> jnp.ndarray:
    """
    Euler's equations for a general (diagonal) inertia tensor.
    
    I_xx * dω_x/dt = M_x - (I_zz - I_yy) * ω_y * ω_z
    I_yy * dω_y/dt = M_y - (I_xx - I_zz) * ω_z * ω_x
    I_zz * dω_z/dt = M_z - (I_yy - I_xx) * ω_x * ω_y
    
    Args:
        omega_local: (3,) angular velocity in body frame
        M_local: (3,) moment in body frame
        I_diag: (3,) diagonal inertia [I_xx, I_yy, I_zz]
    
    Returns:
        (3,) angular acceleration in body frame
    """
    I_xx, I_yy, I_zz = I_diag
    omega_x, omega_y, omega_z = omega_local
    M_x, M_y, M_z = M_local
    
    alpha_x = (M_x - (I_zz - I_yy) * omega_y * omega_z) / I_xx
    alpha_y = (M_y - (I_xx - I_zz) * omega_z * omega_x) / I_yy
    alpha_z = (M_z - (I_yy - I_xx) * omega_x * omega_y) / I_zz
    
    return jnp.array([alpha_x, alpha_y, alpha_z])


@partial(jax.jit, static_argnums=())
def update_platform_state(
    position: jnp.ndarray,
    velocity: jnp.ndarray,
    quaternion: jnp.ndarray,
    omega: jnp.ndarray,
    acceleration: jnp.ndarray,
    alpha: jnp.ndarray,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Update platform state using semi-implicit Euler integration.
    
    Args:
        position: (3,) current position [m]
        velocity: (3,) current velocity [m/s]
        quaternion: (4,) current orientation
        omega: (3,) current angular velocity [rad/s]
        acceleration: (3,) linear acceleration [m/s²]
        alpha: (3,) angular acceleration [rad/s²]
        dt: Time step [s]
    
    Returns:
        new_position: Updated position
        new_velocity: Updated velocity
        new_quaternion: Updated orientation (normalized)
        new_omega: Updated angular velocity
    """
    # Semi-implicit Euler
    new_velocity = velocity + acceleration * dt
    new_position = position + new_velocity * dt
    
    new_omega = omega + alpha * dt
    
    # Quaternion integration
    dq = quaternion_derivative(quaternion, new_omega)
    new_quaternion = quaternion + dq * dt
    new_quaternion = quaternion_normalize(new_quaternion)
    
    return new_position, new_velocity, new_quaternion, new_omega


def compute_platform_dynamics(
    platform_position: jnp.ndarray,
    platform_velocity: jnp.ndarray,
    platform_quaternion: jnp.ndarray,
    platform_omega: jnp.ndarray,
    platform_mass: float,
    platform_inertia: jnp.ndarray,
    mesh_vertices: jnp.ndarray,
    mesh_faces: jnp.ndarray,
    F_ice: jnp.ndarray,
    M_ice: jnp.ndarray,
    F_mooring: jnp.ndarray,
    M_mooring: jnp.ndarray,
    rho_water: float,
    g: jnp.ndarray,
    v_water: jnp.ndarray,
    C_d_F: float,
    C_d_M: float,
    r_eff: float,
    waterline_z: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute platform dynamics including all forces.
    
    Args:
        platform_position: (3,) platform CoG position
        platform_velocity: (3,) platform velocity
        platform_quaternion: (4,) platform orientation
        platform_omega: (3,) platform angular velocity
        platform_mass: Platform mass [kg]
        platform_inertia: (3, 3) inertia tensor
        mesh_vertices: (n_verts, 3) hull mesh vertices (local frame)
        mesh_faces: (n_faces, 3) hull mesh faces
        F_ice: (3,) total ice contact force
        M_ice: (3,) total ice contact moment
        F_mooring: (3,) mooring/propulsion force
        M_mooring: (3,) mooring/propulsion moment
        rho_water: Water density
        g: (3,) gravity vector
        v_water: (3,) water current velocity
        C_d_F: Force drag coefficient
        C_d_M: Moment drag coefficient
        r_eff: Effective radius for drag
        waterline_z: Waterline z-coordinate
    
    Returns:
        acceleration: (3,) linear acceleration
        alpha: (3,) angular acceleration
    """
    # Buoyancy
    F_buoy, M_buoy = compute_total_buoyancy(
        mesh_vertices, mesh_faces,
        platform_position, platform_quaternion,
        rho_water, g, waterline_z
    )
    
    # Drag
    F_drag, M_drag = compute_total_drag(
        mesh_vertices, mesh_faces,
        platform_position, platform_quaternion,
        platform_velocity, platform_omega,
        v_water, C_d_F, C_d_M, rho_water, r_eff, waterline_z
    )
    
    # Total forces and moments
    F_total = compute_platform_forces(
        F_ice, F_buoy, F_drag, F_mooring, platform_mass, g
    )
    
    M_total = compute_platform_moments(
        M_ice, M_buoy, M_drag, M_mooring
    )
    
    # Compute accelerations
    acceleration, alpha = compute_platform_accelerations(
        F_total, M_total, platform_mass, platform_inertia,
        platform_quaternion, platform_omega
    )
    
    return acceleration, alpha
