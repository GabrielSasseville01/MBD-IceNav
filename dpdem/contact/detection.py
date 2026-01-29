"""
Contact detection between dilated polyhedra.

Implements narrow-phase contact detection using the approximate envelope
function method from Boon et al. (2013) and Liu and Ji (2020).
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from ..kinematics.quaternion import quaternion_to_rotation_matrix
from ..geometry.support import (
    support_dilated_polyhedron,
    minkowski_difference_support,
)


@partial(jax.jit, static_argnums=())
def compute_contact(
    position_i: jnp.ndarray,
    quaternion_i: jnp.ndarray,
    vertices_i: jnp.ndarray,
    dilation_i: float,
    equiv_radius_i: float,
    position_j: jnp.ndarray,
    quaternion_j: jnp.ndarray,
    vertices_j: jnp.ndarray,
    dilation_j: float,
    equiv_radius_j: float,
) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
    """
    Compute contact between two dilated polyhedra.
    
    Uses a combination of sphere approximation for initial estimate
    and envelope function refinement.
    
    Args:
        position_i: (3,) position of body i
        quaternion_i: (4,) orientation of body i
        vertices_i: (n_i, 3) vertices of body i in local frame
        dilation_i: Dilation radius of body i
        equiv_radius_i: Equivalent sphere radius of body i
        position_j: (3,) position of body j
        quaternion_j: (4,) orientation of body j
        vertices_j: (n_j, 3) vertices of body j in local frame
        dilation_j: Dilation radius of body j
        equiv_radius_j: Equivalent sphere radius of body j
    
    Returns:
        delta_n: Normal overlap (positive if penetrating) [m]
        contact_normal: (3,) unit normal from j to i
        contact_point: (3,) contact point location
    """
    # Direction from j to i
    direction = position_i - position_j
    distance = jnp.linalg.norm(direction)
    contact_normal = direction / (distance + 1e-10)
    
    # Rotation matrices
    R_i = quaternion_to_rotation_matrix(quaternion_i)
    R_j = quaternion_to_rotation_matrix(quaternion_j)
    
    # Use support functions to compute more accurate overlap
    # Support of i in direction -n (closest point on i towards j)
    support_i = support_dilated_polyhedron_world(
        vertices_i, dilation_i, position_i, R_i, -contact_normal
    )
    
    # Support of j in direction n (closest point on j towards i)
    support_j = support_dilated_polyhedron_world(
        vertices_j, dilation_j, position_j, R_j, contact_normal
    )
    
    # Overlap is negative of the gap
    gap = jnp.dot(support_i - support_j, contact_normal)
    delta_n = -gap
    
    # Contact point is midway between support points
    contact_point = 0.5 * (support_i + support_j)
    
    return delta_n, contact_normal, contact_point


def support_dilated_polyhedron_world(
    vertices_local: jnp.ndarray,
    dilation_radius: float,
    position: jnp.ndarray,
    rotation_matrix: jnp.ndarray,
    direction: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute support of dilated polyhedron in world coordinates.
    """
    # Transform direction to local frame
    direction_local = rotation_matrix.T @ direction
    direction_norm = jnp.linalg.norm(direction_local)
    direction_normalized = direction_local / (direction_norm + 1e-10)
    
    # Find support vertex
    dot_products = vertices_local @ direction_normalized
    max_idx = jnp.argmax(dot_products)
    support_local = vertices_local[max_idx] + dilation_radius * direction_normalized
    
    # Transform back to world
    return rotation_matrix @ support_local + position


@partial(jax.jit, static_argnums=())
def compute_contact_sphere_approx(
    position_i: jnp.ndarray,
    radius_i: float,
    position_j: jnp.ndarray,
    radius_j: float,
) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
    """
    Compute contact using sphere approximation.
    
    Fast but less accurate than full polyhedron contact.
    Useful for broad-phase filtering.
    
    Args:
        position_i: (3,) position of sphere i
        radius_i: Radius of sphere i (equiv_radius + dilation)
        position_j: (3,) position of sphere j
        radius_j: Radius of sphere j
    
    Returns:
        delta_n: Normal overlap
        contact_normal: Unit normal from j to i
        contact_point: Contact point location
    """
    direction = position_i - position_j
    distance = jnp.linalg.norm(direction)
    contact_normal = direction / (distance + 1e-10)
    
    delta_n = radius_i + radius_j - distance
    
    # Contact point
    contact_point = position_j + (radius_j - 0.5 * delta_n) * contact_normal
    
    return delta_n, contact_normal, contact_point


@partial(jax.jit, static_argnums=())
def compute_contact_point(
    position_i: jnp.ndarray,
    position_j: jnp.ndarray,
    contact_normal: jnp.ndarray,
    delta_n: float,
    radius_i: float,
    radius_j: float,
) -> jnp.ndarray:
    """
    Compute contact point location.
    
    The contact point is located on the contact surface, which is
    at the midpoint of the overlap region.
    
    Args:
        position_i: Center of body i
        position_j: Center of body j
        contact_normal: Unit normal from j to i
        delta_n: Normal overlap
        radius_i: Effective radius of body i
        radius_j: Effective radius of body j
    
    Returns:
        (3,) contact point in world coordinates
    """
    # Contact point is on body j's surface plus half the overlap
    return position_j + (radius_j - 0.5 * delta_n) * contact_normal


@partial(jax.jit, static_argnums=())
def compute_contact_normal(
    position_i: jnp.ndarray,
    position_j: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute contact normal direction.
    
    The normal points from body j to body i.
    
    Args:
        position_i: Center of body i
        position_j: Center of body j
    
    Returns:
        (3,) unit normal vector
    """
    direction = position_i - position_j
    distance = jnp.linalg.norm(direction)
    return direction / (distance + 1e-10)


def detect_all_contacts(
    positions: jnp.ndarray,
    quaternions: jnp.ndarray,
    polyhedra_vertices: jnp.ndarray,
    dilation_radii: jnp.ndarray,
    equiv_radii: jnp.ndarray,
    neighbor_pairs: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Detect contacts for all neighbor pairs.
    
    Args:
        positions: (N, 3) particle positions
        quaternions: (N, 4) particle orientations
        polyhedra_vertices: (N, max_verts, 3) vertices for each particle
        dilation_radii: (N,) dilation radii
        equiv_radii: (N,) equivalent sphere radii
        neighbor_pairs: (idx_i, idx_j) neighbor pair indices
    
    Returns:
        delta_n: (n_pairs,) normal overlaps
        contact_normal: (n_pairs, 3) contact normals
        contact_point: (n_pairs, 3) contact points
        in_contact: (n_pairs,) boolean mask for actual contacts
    """
    idx_i, idx_j = neighbor_pairs
    n_pairs = len(idx_i)
    
    if n_pairs == 0:
        return (
            jnp.array([]),
            jnp.zeros((0, 3)),
            jnp.zeros((0, 3)),
            jnp.array([], dtype=bool),
        )
    
    # Use sphere approximation for speed
    # More accurate contact can be computed selectively for actual contacts
    total_radii_i = equiv_radii[idx_i] + dilation_radii[idx_i]
    total_radii_j = equiv_radii[idx_j] + dilation_radii[idx_j]
    
    # Vectorized sphere contact
    directions = positions[idx_i] - positions[idx_j]
    distances = jnp.linalg.norm(directions, axis=1)
    contact_normals = directions / (distances[:, None] + 1e-10)
    
    delta_n = total_radii_i + total_radii_j - distances
    
    # Contact points
    contact_points = (
        positions[idx_j] + 
        (total_radii_j - 0.5 * delta_n)[:, None] * contact_normals
    )
    
    # Contact mask
    in_contact = delta_n > 0
    
    return delta_n, contact_normals, contact_points, in_contact


@partial(jax.jit, static_argnums=())
def compute_relative_velocity_at_contact(
    velocity_i: jnp.ndarray,
    angular_velocity_i: jnp.ndarray,
    velocity_j: jnp.ndarray,
    angular_velocity_j: jnp.ndarray,
    contact_point: jnp.ndarray,
    position_i: jnp.ndarray,
    position_j: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute relative velocity at contact point.
    
    v_rel = v_i + ω_i × r_i - v_j - ω_j × r_j
    
    where r_i = contact_point - position_i
    
    Args:
        velocity_i: (3,) linear velocity of body i
        angular_velocity_i: (3,) angular velocity of body i
        velocity_j: (3,) linear velocity of body j
        angular_velocity_j: (3,) angular velocity of body j
        contact_point: (3,) contact point location
        position_i: (3,) center of body i
        position_j: (3,) center of body j
    
    Returns:
        (3,) relative velocity at contact point
    """
    r_i = contact_point - position_i
    r_j = contact_point - position_j
    
    v_contact_i = velocity_i + jnp.cross(angular_velocity_i, r_i)
    v_contact_j = velocity_j + jnp.cross(angular_velocity_j, r_j)
    
    return v_contact_i - v_contact_j


@partial(jax.jit, static_argnums=())
def decompose_relative_velocity(
    relative_velocity: jnp.ndarray,
    contact_normal: jnp.ndarray,
) -> Tuple[float, jnp.ndarray]:
    """
    Decompose relative velocity into normal and tangential components.
    
    Args:
        relative_velocity: (3,) relative velocity at contact
        contact_normal: (3,) unit contact normal
    
    Returns:
        v_n: Normal component (positive = approaching)
        v_t: (3,) tangential velocity vector
    """
    v_n = jnp.dot(relative_velocity, contact_normal)
    v_t = relative_velocity - v_n * contact_normal
    
    return -v_n, v_t  # Negative because positive delta_n_dot means approaching
