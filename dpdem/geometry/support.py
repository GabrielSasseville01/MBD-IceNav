"""
Support function for dilated polyhedra.

The support function returns the point on a convex shape that is
furthest in a given direction. For dilated polyhedra (Minkowski sum
of polyhedron + sphere), the support is:

    support(A ⊕ B, d) = support(A, d) + support(B, d)

where support of a sphere is simply r * d/|d|.

Reference: Eq. 1 in Wang et al. (2024)
"""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=())
def support_polyhedron(
    vertices: jnp.ndarray,
    direction: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute support point of a polyhedron in a given direction.
    
    The support point is the vertex furthest in the given direction.
    
    Args:
        vertices: (n_vertices, 3) array of polyhedron vertices
        direction: (3,) direction vector (does not need to be normalized)
    
    Returns:
        (3,) support point on the polyhedron
    """
    # Compute dot products with direction
    dot_products = vertices @ direction
    
    # Find vertex with maximum dot product
    max_idx = jnp.argmax(dot_products)
    
    return vertices[max_idx]


@partial(jax.jit, static_argnums=())
def support_sphere(
    center: jnp.ndarray,
    radius: float,
    direction: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute support point of a sphere in a given direction.
    
    Args:
        center: (3,) sphere center
        radius: Sphere radius
        direction: (3,) direction vector
    
    Returns:
        (3,) support point on the sphere
    """
    direction_norm = jnp.linalg.norm(direction)
    direction_normalized = direction / (direction_norm + 1e-10)
    
    return center + radius * direction_normalized


@partial(jax.jit, static_argnums=())
def support_dilated_polyhedron(
    vertices: jnp.ndarray,
    dilation_radius: float,
    direction: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute support point of a dilated polyhedron in a given direction.
    
    For a dilated polyhedron (Minkowski sum A ⊕ B where A is a polyhedron
    and B is a sphere of radius r):
    
        support(A ⊕ B, d) = support(A, d) + r * d/|d|
    
    This is Eq. 1 from the paper:
        A ⊕ B = {x + y | x ∈ A, y ∈ B}
    
    Args:
        vertices: (n_vertices, 3) array of polyhedron vertices (in local frame)
        dilation_radius: Radius of dilation sphere [m]
        direction: (3,) direction vector
    
    Returns:
        (3,) support point on the dilated polyhedron
    """
    # Normalize direction
    direction_norm = jnp.linalg.norm(direction)
    direction_normalized = direction / (direction_norm + 1e-10)
    
    # Support of polyhedron
    poly_support = support_polyhedron(vertices, direction)
    
    # Support of sphere (centered at origin)
    sphere_support = dilation_radius * direction_normalized
    
    # Minkowski sum: add supports
    return poly_support + sphere_support


@partial(jax.jit, static_argnums=())
def support_dilated_polyhedron_world(
    vertices_local: jnp.ndarray,
    dilation_radius: float,
    position: jnp.ndarray,
    rotation_matrix: jnp.ndarray,
    direction: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute support of a dilated polyhedron in world coordinates.
    
    Args:
        vertices_local: (n_vertices, 3) vertices in local frame
        dilation_radius: Dilation sphere radius
        position: (3,) position of polyhedron center in world frame
        rotation_matrix: (3, 3) rotation from local to world frame
        direction: (3,) direction in world frame
    
    Returns:
        (3,) support point in world coordinates
    """
    # Transform direction to local frame
    direction_local = rotation_matrix.T @ direction
    
    # Compute support in local frame
    support_local = support_dilated_polyhedron(
        vertices_local, dilation_radius, direction_local
    )
    
    # Transform support back to world frame
    support_world = rotation_matrix @ support_local + position
    
    return support_world


def minkowski_difference_support(
    vertices_a: jnp.ndarray,
    dilation_radius_a: float,
    position_a: jnp.ndarray,
    rotation_a: jnp.ndarray,
    vertices_b: jnp.ndarray,
    dilation_radius_b: float,
    position_b: jnp.ndarray,
    rotation_b: jnp.ndarray,
    direction: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute support of the Minkowski difference A - B.
    
    Used in GJK algorithm for collision detection.
    
    support(A - B, d) = support(A, d) - support(B, -d)
    
    Args:
        vertices_a, dilation_radius_a, position_a, rotation_a: Shape A parameters
        vertices_b, dilation_radius_b, position_b, rotation_b: Shape B parameters
        direction: Search direction
    
    Returns:
        Support point on the Minkowski difference
    """
    # Support of A in direction d
    support_a = support_dilated_polyhedron_world(
        vertices_a, dilation_radius_a, position_a, rotation_a, direction
    )
    
    # Support of B in direction -d
    support_b = support_dilated_polyhedron_world(
        vertices_b, dilation_radius_b, position_b, rotation_b, -direction
    )
    
    # Minkowski difference
    return support_a - support_b
