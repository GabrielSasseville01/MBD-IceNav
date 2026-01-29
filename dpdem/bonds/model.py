"""
Bond force model for DPDEM.

Implements the inter-element bonding model from Eqs. 11-13 in the paper.
Bonds connect adjacent ice elements and transmit forces until fracture.

Reference: Wang et al. (2024), Liu and Ji (2021)
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List
import numpy as np

from ..kinematics.quaternion import (
    quaternion_to_rotation_matrix,
    transform_to_global,
    transform_to_local,
)


@partial(jax.jit, static_argnums=())
def compute_stiffness_matrix(E: float, nu: float) -> jnp.ndarray:
    """
    Compute the elastic stiffness matrix for bonds.
    
    From Eq. 11:
    K = E(1-ν)/((1+ν)(1-2ν)) * [[1, 0], [0, (1-2ν)/(2(1-ν))]]
    
    This gives k_n (normal stiffness) and k_s (shear stiffness).
    
    Args:
        E: Elastic modulus [Pa]
        nu: Poisson's ratio [-]
    
    Returns:
        (2, 2) stiffness matrix [k_n, k_s on diagonal]
    """
    prefactor = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    k_n = prefactor
    k_s = prefactor * (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))
    
    return jnp.array([
        [k_n, 0.0],
        [0.0, k_s]
    ])


@partial(jax.jit, static_argnums=())
def compute_bond_strain(
    position_i: jnp.ndarray,
    quaternion_i: jnp.ndarray,
    local_pos_i: jnp.ndarray,
    position_j: jnp.ndarray,
    quaternion_j: jnp.ndarray,
    local_pos_j: jnp.ndarray,
    initial_length: float,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute normal and tangential strain in a bond.
    
    Args:
        position_i: (3,) position of particle i
        quaternion_i: (4,) orientation of particle i
        local_pos_i: (3,) bond attachment point in particle i's local frame
        position_j: (3,) position of particle j
        quaternion_j: (4,) orientation of particle j
        local_pos_j: (3,) bond attachment point in particle j's local frame
        initial_length: Initial bond length [m]
    
    Returns:
        epsilon_n: Normal strain [-]
        epsilon_t: (3,) tangential strain direction (magnitude is strain)
    """
    # Transform attachment points to world frame
    R_i = quaternion_to_rotation_matrix(quaternion_i)
    R_j = quaternion_to_rotation_matrix(quaternion_j)
    
    attach_i_world = position_i + R_i @ local_pos_i
    attach_j_world = position_j + R_j @ local_pos_j
    
    # Current bond vector
    bond_vector = attach_j_world - attach_i_world
    current_length = jnp.linalg.norm(bond_vector)
    
    # Bond direction (from i to j)
    bond_direction = bond_vector / (current_length + 1e-10)
    
    # Normal strain (extension is positive)
    epsilon_n = (current_length - initial_length) / initial_length
    
    # Tangential strain from rotation
    # Use initial bond direction in each particle's frame
    initial_direction_i = local_pos_j - local_pos_i  # Approximate
    initial_direction_i = initial_direction_i / (jnp.linalg.norm(initial_direction_i) + 1e-10)
    
    # Current bond direction in particle i's frame
    current_direction_i = R_i.T @ bond_direction
    
    # Tangential displacement is the perpendicular component
    tangent = current_direction_i - jnp.dot(current_direction_i, initial_direction_i) * initial_direction_i
    epsilon_t = tangent  # This approximates tangential strain
    
    return epsilon_n, epsilon_t


@partial(jax.jit, static_argnums=())
def compute_bond_strain_rate(
    velocity_i: jnp.ndarray,
    angular_velocity_i: jnp.ndarray,
    position_i: jnp.ndarray,
    quaternion_i: jnp.ndarray,
    local_pos_i: jnp.ndarray,
    velocity_j: jnp.ndarray,
    angular_velocity_j: jnp.ndarray,
    position_j: jnp.ndarray,
    quaternion_j: jnp.ndarray,
    local_pos_j: jnp.ndarray,
    initial_length: float,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute strain rate in a bond.
    
    Args:
        velocity_i: (3,) velocity of particle i
        angular_velocity_i: (3,) angular velocity of particle i
        position_i: (3,) position of particle i
        quaternion_i: (4,) orientation of particle i
        local_pos_i: (3,) bond attachment point in particle i's frame
        velocity_j, angular_velocity_j, position_j, quaternion_j, local_pos_j: Same for j
        initial_length: Initial bond length
    
    Returns:
        epsilon_n_dot: Normal strain rate [1/s]
        epsilon_t_dot: (3,) tangential strain rate
    """
    # Transform attachment points
    R_i = quaternion_to_rotation_matrix(quaternion_i)
    R_j = quaternion_to_rotation_matrix(quaternion_j)
    
    attach_i_world = position_i + R_i @ local_pos_i
    attach_j_world = position_j + R_j @ local_pos_j
    
    # Bond direction
    bond_vector = attach_j_world - attach_i_world
    current_length = jnp.linalg.norm(bond_vector)
    bond_direction = bond_vector / (current_length + 1e-10)
    
    # Velocities at attachment points
    r_i = R_i @ local_pos_i
    r_j = R_j @ local_pos_j
    
    v_attach_i = velocity_i + jnp.cross(angular_velocity_i, r_i)
    v_attach_j = velocity_j + jnp.cross(angular_velocity_j, r_j)
    
    # Relative velocity
    v_rel = v_attach_j - v_attach_i
    
    # Normal strain rate
    v_n = jnp.dot(v_rel, bond_direction)
    epsilon_n_dot = v_n / initial_length
    
    # Tangential strain rate
    v_t = v_rel - v_n * bond_direction
    epsilon_t_dot = v_t / initial_length
    
    return epsilon_n_dot, epsilon_t_dot


@partial(jax.jit, static_argnums=())
def compute_elastic_stress(
    epsilon_n: float,
    epsilon_t: jnp.ndarray,
    E: float,
    nu: float,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute elastic stress in a bond.
    
    Implements Eq. 11:
    σ_e = K · ε
    
    Args:
        epsilon_n: Normal strain [-]
        epsilon_t: (3,) tangential strain
        E: Elastic modulus [Pa]
        nu: Poisson's ratio [-]
    
    Returns:
        sigma_n: Normal stress [Pa]
        sigma_t: (3,) tangential stress [Pa]
    """
    K = compute_stiffness_matrix(E, nu)
    k_n = K[0, 0]
    k_s = K[1, 1]
    
    sigma_n = k_n * epsilon_n
    sigma_t = k_s * epsilon_t
    
    return sigma_n, sigma_t


@partial(jax.jit, static_argnums=())
def compute_viscous_stress(
    epsilon_n_dot: float,
    epsilon_t_dot: jnp.ndarray,
    E: float,
    nu: float,
    beta: float,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute viscous stress in a bond.
    
    Implements Eq. 12:
    σ_v = β * K · ε̇
    
    Args:
        epsilon_n_dot: Normal strain rate [1/s]
        epsilon_t_dot: (3,) tangential strain rate
        E: Elastic modulus [Pa]
        nu: Poisson's ratio [-]
        beta: Damping coefficient [-]
    
    Returns:
        sigma_n_v: Normal viscous stress [Pa]
        sigma_t_v: (3,) tangential viscous stress [Pa]
    """
    K = compute_stiffness_matrix(E, nu)
    k_n = K[0, 0]
    k_s = K[1, 1]
    
    sigma_n_v = beta * k_n * epsilon_n_dot
    sigma_t_v = beta * k_s * epsilon_t_dot
    
    return sigma_n_v, sigma_t_v


@partial(jax.jit, static_argnums=())
def compute_bond_force(
    position_i: jnp.ndarray,
    quaternion_i: jnp.ndarray,
    velocity_i: jnp.ndarray,
    angular_velocity_i: jnp.ndarray,
    local_pos_i: jnp.ndarray,
    position_j: jnp.ndarray,
    quaternion_j: jnp.ndarray,
    velocity_j: jnp.ndarray,
    angular_velocity_j: jnp.ndarray,
    local_pos_j: jnp.ndarray,
    initial_length: float,
    bond_area: float,
    n_bond_points: int,
    E: float,
    nu: float,
    beta: float,
    damage: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, float]:
    """
    Compute bond force between two particles.
    
    Implements Eq. 13:
    F_b = (σ_e + σ_v) · A · n
    
    where A is bond area and n is number of bond points.
    
    Args:
        position_i, quaternion_i, velocity_i, angular_velocity_i: State of particle i
        local_pos_i: Bond attachment in particle i's frame
        position_j, quaternion_j, velocity_j, angular_velocity_j: State of particle j
        local_pos_j: Bond attachment in particle j's frame
        initial_length: Initial bond length [m]
        bond_area: Bond cross-sectional area [m²]
        n_bond_points: Number of bond points
        E: Elastic modulus [Pa]
        nu: Poisson's ratio [-]
        beta: Damping coefficient [-]
        damage: Damage variable D [0, 1]
    
    Returns:
        F_bond: (3,) force on particle i from bond [N]
        M_bond: (3,) moment on particle i from bond [N·m]
        sigma_n_total: Total normal stress [Pa]
        sigma_t_mag: Tangential stress magnitude [Pa]
    """
    # Compute strains
    epsilon_n, epsilon_t = compute_bond_strain(
        position_i, quaternion_i, local_pos_i,
        position_j, quaternion_j, local_pos_j,
        initial_length
    )
    
    epsilon_n_dot, epsilon_t_dot = compute_bond_strain_rate(
        velocity_i, angular_velocity_i, position_i, quaternion_i, local_pos_i,
        velocity_j, angular_velocity_j, position_j, quaternion_j, local_pos_j,
        initial_length
    )
    
    # Compute stresses
    sigma_n_e, sigma_t_e = compute_elastic_stress(epsilon_n, epsilon_t, E, nu)
    sigma_n_v, sigma_t_v = compute_viscous_stress(epsilon_n_dot, epsilon_t_dot, E, nu, beta)
    
    # Total stress
    sigma_n_total = sigma_n_e + sigma_n_v
    sigma_t_total = sigma_t_e + sigma_t_v
    sigma_t_mag = jnp.linalg.norm(sigma_t_total)
    
    # Apply damage (softening)
    effective_stiffness = 1.0 - damage
    sigma_n_total = sigma_n_total * effective_stiffness
    sigma_t_total = sigma_t_total * effective_stiffness
    
    # Bond direction
    R_i = quaternion_to_rotation_matrix(quaternion_i)
    R_j = quaternion_to_rotation_matrix(quaternion_j)
    
    attach_i_world = position_i + R_i @ local_pos_i
    attach_j_world = position_j + R_j @ local_pos_j
    
    bond_vector = attach_j_world - attach_i_world
    bond_direction = bond_vector / (jnp.linalg.norm(bond_vector) + 1e-10)
    
    # Force per bond point, then multiply by number of points and area
    # Force direction: normal along bond, tangential perpendicular
    F_n = sigma_n_total * bond_area / n_bond_points * bond_direction
    F_t = sigma_t_total * bond_area / n_bond_points  # Already in world frame direction
    
    # Total force on particle i (bond pulls if in tension)
    F_bond = (F_n + F_t) * n_bond_points
    
    # Moment about particle i center
    r_i = R_i @ local_pos_i
    M_bond = jnp.cross(r_i, F_bond)
    
    return F_bond, M_bond, sigma_n_total, sigma_t_mag


def initialize_bonds(
    positions: np.ndarray,
    quaternions: np.ndarray,
    polyhedra_vertices: List[np.ndarray],
    adjacency_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize bonds between adjacent ice elements.
    
    Bonds are created between particles whose surfaces are within
    the adjacency threshold.
    
    Args:
        positions: (N, 3) particle positions
        quaternions: (N, 4) particle orientations
        polyhedra_vertices: List of vertex arrays for each particle
        adjacency_threshold: Maximum distance for bond creation [m]
    
    Returns:
        particle_i: Indices of first particles
        particle_j: Indices of second particles
        areas: Bond areas
        n_bond_points: Number of bond points
        local_positions_i: Attachment points in particle i frames
        local_positions_j: Attachment points in particle j frames
        initial_lengths: Initial bond lengths
    """
    n_particles = len(positions)
    
    bonds_i = []
    bonds_j = []
    areas = []
    n_points = []
    local_pos_i = []
    local_pos_j = []
    init_lengths = []
    
    # Check all pairs (could be optimized with spatial hashing)
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # Distance between centers
            dist = np.linalg.norm(positions[i] - positions[j])
            
            # Quick rejection if too far apart
            if dist > adjacency_threshold * 3:
                continue
            
            # Check if surfaces are adjacent
            # Use centroids for simplicity
            direction = positions[j] - positions[i]
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Approximate contact check using equivalent spheres
            # In a full implementation, check actual surface proximity
            gap = dist - adjacency_threshold
            
            if gap < adjacency_threshold:
                # Create bond
                bonds_i.append(i)
                bonds_j.append(j)
                
                # Bond area (approximate from element size)
                area = adjacency_threshold ** 2 * 0.5  # Approximate
                areas.append(area)
                
                # Number of bond points
                n_points.append(4)  # Typical value
                
                # Attachment points (at surface along connecting direction)
                local_i = direction * adjacency_threshold * 0.5
                local_j = -direction * adjacency_threshold * 0.5
                local_pos_i.append(local_i)
                local_pos_j.append(local_j)
                
                # Initial length
                init_lengths.append(dist)
    
    if len(bonds_i) == 0:
        # Return empty arrays
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([]),
            np.array([], dtype=np.int32),
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            np.array([]),
        )
    
    return (
        np.array(bonds_i, dtype=np.int32),
        np.array(bonds_j, dtype=np.int32),
        np.array(areas),
        np.array(n_points, dtype=np.int32),
        np.array(local_pos_i),
        np.array(local_pos_j),
        np.array(init_lengths),
    )
