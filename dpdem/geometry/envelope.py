"""
Approximate envelope function for contact detection.

Implements the contact detection method based on approximate envelope
functions as described by Boon et al. (2013) and Liu and Ji (2020).

The envelope function provides a smooth potential field around a dilated
polyhedron, enabling efficient computation of contact overlap and normal.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple


@partial(jax.jit, static_argnums=())
def signed_distance_to_plane(
    point: jnp.ndarray,
    plane_point: jnp.ndarray,
    plane_normal: jnp.ndarray,
) -> float:
    """
    Compute signed distance from a point to a plane.
    
    Positive = point is on the side the normal points to
    Negative = point is on the opposite side
    
    Args:
        point: (3,) query point
        plane_point: (3,) point on the plane
        plane_normal: (3,) unit normal of the plane
    
    Returns:
        Signed distance to plane
    """
    return jnp.dot(point - plane_point, plane_normal)


@partial(jax.jit, static_argnums=())
def signed_distance_to_polyhedron(
    point: jnp.ndarray,
    vertices: jnp.ndarray,
    faces: jnp.ndarray,
    face_sizes: jnp.ndarray,
    face_normals: jnp.ndarray,
) -> float:
    """
    Compute signed distance from a point to a convex polyhedron.
    
    For a convex polyhedron, the signed distance is:
    - Negative if point is inside (distance to nearest face)
    - Positive if point is outside (distance to nearest face)
    
    Args:
        point: (3,) query point
        vertices: (n_verts, 3) polyhedron vertices
        faces: (n_faces, max_verts) face vertex indices
        face_sizes: (n_faces,) vertices per face
        face_normals: (n_faces, 3) outward face normals
    
    Returns:
        Signed distance (negative inside, positive outside)
    """
    n_faces = len(faces)
    
    # For each face, compute signed distance to the face plane
    max_signed_dist = -jnp.inf
    
    for i in range(n_faces):
        # Get a point on the face (first vertex)
        face_point = vertices[faces[i, 0]]
        face_normal = face_normals[i]
        
        # Signed distance to this face's plane
        dist = signed_distance_to_plane(point, face_point, face_normal)
        max_signed_dist = jnp.maximum(max_signed_dist, dist)
    
    return max_signed_dist


@partial(jax.jit, static_argnums=())
def contact_potential(
    point: jnp.ndarray,
    vertices: jnp.ndarray,
    faces: jnp.ndarray,
    face_sizes: jnp.ndarray,
    face_normals: jnp.ndarray,
    dilation_radius: float,
    alpha: float = 0.5,
) -> float:
    """
    Compute approximate envelope potential for contact detection.
    
    The potential is a weighted combination of the polyhedral potential
    and a spherical potential, providing a smooth field for contact.
    
    Reference: Boon et al. (2013), Liu and Ji (2020)
    
    Args:
        point: (3,) query point
        vertices: (n_verts, 3) polyhedron vertices
        faces: (n_faces, max_verts) face vertex indices
        face_sizes: (n_faces,) vertices per face
        face_normals: (n_faces, 3) outward face normals
        dilation_radius: Sphere dilation radius
        alpha: Weighting between polyhedral (0) and spherical (1) potential
    
    Returns:
        Potential value (negative inside dilated polyhedron, positive outside)
    """
    # Polyhedral component: signed distance minus dilation
    poly_dist = signed_distance_to_polyhedron(
        point, vertices, faces, face_sizes, face_normals
    )
    poly_potential = poly_dist - dilation_radius
    
    # Spherical component: distance to centroid minus effective radius
    centroid = jnp.mean(vertices, axis=0)
    
    # Effective radius for sphere approximation
    # Use average distance from centroid to vertices plus dilation
    vertex_distances = jnp.linalg.norm(vertices - centroid, axis=1)
    effective_radius = jnp.mean(vertex_distances) + dilation_radius
    
    dist_to_centroid = jnp.linalg.norm(point - centroid)
    sphere_potential = dist_to_centroid - effective_radius
    
    # Weighted combination
    return alpha * sphere_potential + (1.0 - alpha) * poly_potential


@partial(jax.jit, static_argnums=())
def compute_contact_overlap(
    position_i: jnp.ndarray,
    rotation_i: jnp.ndarray,
    vertices_i: jnp.ndarray,
    faces_i: jnp.ndarray,
    face_sizes_i: jnp.ndarray,
    face_normals_i: jnp.ndarray,
    dilation_i: float,
    position_j: jnp.ndarray,
    rotation_j: jnp.ndarray,
    vertices_j: jnp.ndarray,
    faces_j: jnp.ndarray,
    face_sizes_j: jnp.ndarray,
    face_normals_j: jnp.ndarray,
    dilation_j: float,
) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
    """
    Compute contact overlap between two dilated polyhedra.
    
    Uses a combination of centroid distance and envelope functions
    to estimate the penetration depth and contact normal.
    
    Args:
        position_i, rotation_i: Position and rotation of body i
        vertices_i, faces_i, face_sizes_i, face_normals_i, dilation_i: Geometry of body i
        position_j, rotation_j: Position and rotation of body j
        vertices_j, faces_j, face_sizes_j, face_normals_j, dilation_j: Geometry of body j
    
    Returns:
        delta_n: Normal overlap (positive if penetrating)
        contact_normal: Unit normal pointing from j to i
        contact_point: Contact point location
    """
    # Transform vertices to world frame
    vertices_i_world = (rotation_i @ vertices_i.T).T + position_i
    vertices_j_world = (rotation_j @ vertices_j.T).T + position_j
    
    # Transform face normals
    normals_i_world = (rotation_i @ face_normals_i.T).T
    normals_j_world = (rotation_j @ face_normals_j.T).T
    
    # Centroids in world frame
    centroid_i = jnp.mean(vertices_i_world, axis=0)
    centroid_j = jnp.mean(vertices_j_world, axis=0)
    
    # Direction from j to i
    direction = centroid_i - centroid_j
    distance = jnp.linalg.norm(direction)
    contact_normal = direction / (distance + 1e-10)
    
    # Compute potential of centroid_j in body i's field
    # First transform to body i's local frame
    centroid_j_local_i = rotation_i.T @ (centroid_j - position_i)
    potential_j_in_i = contact_potential(
        centroid_j_local_i,
        vertices_i, faces_i, face_sizes_i, face_normals_i,
        dilation_i, alpha=0.3
    )
    
    # Compute potential of centroid_i in body j's field
    centroid_i_local_j = rotation_j.T @ (centroid_i - position_j)
    potential_i_in_j = contact_potential(
        centroid_i_local_j,
        vertices_j, faces_j, face_sizes_j, face_normals_j,
        dilation_j, alpha=0.3
    )
    
    # Overlap is negative of the sum of potentials
    # (when both potentials are negative, the objects overlap)
    delta_n = -(potential_j_in_i + potential_i_in_j)
    
    # Contact point is midway between centroids, offset along normal
    contact_point = 0.5 * (centroid_i + centroid_j)
    
    return delta_n, contact_normal, contact_point


def compute_contact_overlap_vectorized(
    positions_i: jnp.ndarray,
    rotations_i: jnp.ndarray,
    polyhedra_i: 'Polyhedron',
    positions_j: jnp.ndarray,
    rotations_j: jnp.ndarray,
    polyhedra_j: 'Polyhedron',
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Vectorized contact overlap computation for multiple pairs.
    
    Note: This function assumes all polyhedra have the same shape structure.
    For heterogeneous polyhedra, use a loop over pairs.
    
    Args:
        positions_i: (N, 3) positions of first bodies
        rotations_i: (N, 3, 3) rotation matrices of first bodies
        polyhedra_i: Polyhedron data for first bodies
        positions_j: (N, 3) positions of second bodies
        rotations_j: (N, 3, 3) rotation matrices of second bodies
        polyhedra_j: Polyhedron data for second bodies
    
    Returns:
        delta_n: (N,) normal overlaps
        contact_normal: (N, 3) contact normals
        contact_point: (N, 3) contact points
    """
    # Simple sphere-based approximation for vectorized computation
    # More accurate narrow-phase can be done selectively
    
    direction = positions_i - positions_j
    distance = jnp.linalg.norm(direction, axis=1, keepdims=True)
    contact_normal = direction / (distance + 1e-10)
    
    # Use equivalent sphere radii for quick overlap estimate
    r_i = polyhedra_i.equivalent_radius + polyhedra_i.dilation_radius
    r_j = polyhedra_j.equivalent_radius + polyhedra_j.dilation_radius
    
    delta_n = r_i + r_j - distance.squeeze()
    
    contact_point = 0.5 * (positions_i + positions_j)
    
    return delta_n, contact_normal, contact_point
