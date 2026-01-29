"""
Triangular mesh utilities for DPDEM.

Handles mesh operations for hull surfaces and polyhedron triangulation.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List
import numpy as np


@partial(jax.jit, static_argnums=())
def compute_triangle_area(
    v0: jnp.ndarray,
    v1: jnp.ndarray,
    v2: jnp.ndarray,
) -> float:
    """
    Compute area of a triangle from its vertices.
    
    Area = 0.5 * ||(v1 - v0) × (v2 - v0)||
    
    Args:
        v0, v1, v2: (3,) vertex positions
    
    Returns:
        Triangle area
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = jnp.cross(edge1, edge2)
    return 0.5 * jnp.linalg.norm(cross)


@partial(jax.jit, static_argnums=())
def compute_triangle_normal(
    v0: jnp.ndarray,
    v1: jnp.ndarray,
    v2: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute unit normal of a triangle from its vertices.
    
    Normal direction follows right-hand rule: (v1-v0) × (v2-v0)
    
    Args:
        v0, v1, v2: (3,) vertex positions
    
    Returns:
        (3,) unit normal vector
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = jnp.cross(edge1, edge2)
    norm = jnp.linalg.norm(cross)
    return cross / (norm + 1e-10)


@partial(jax.jit, static_argnums=())
def compute_triangle_centroid(
    v0: jnp.ndarray,
    v1: jnp.ndarray,
    v2: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute centroid of a triangle.
    
    Args:
        v0, v1, v2: (3,) vertex positions
    
    Returns:
        (3,) centroid position
    """
    return (v0 + v1 + v2) / 3.0


def triangulate_polyhedron_surface(
    vertices: jnp.ndarray,
    faces: jnp.ndarray,
    face_sizes: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert a polyhedron with polygonal faces to a triangle mesh.
    
    Uses fan triangulation for each face.
    
    Args:
        vertices: (n_verts, 3) vertex positions
        faces: (n_faces, max_verts) face vertex indices
        face_sizes: (n_faces,) number of vertices per face
    
    Returns:
        vertices: (n_verts, 3) same vertices
        triangles: (n_triangles, 3) triangle vertex indices
    """
    triangles = []
    n_faces = len(faces)
    
    for i in range(n_faces):
        n_verts = int(face_sizes[i])
        face_verts = faces[i, :n_verts]
        
        # Fan triangulation from first vertex
        for j in range(1, n_verts - 1):
            triangles.append([
                int(face_verts[0]),
                int(face_verts[j]),
                int(face_verts[j + 1])
            ])
    
    return vertices, jnp.array(triangles)


def compute_mesh_surface_area(
    vertices: jnp.ndarray,
    triangles: jnp.ndarray,
) -> float:
    """
    Compute total surface area of a triangle mesh.
    
    Args:
        vertices: (n_verts, 3) vertex positions
        triangles: (n_triangles, 3) triangle vertex indices
    
    Returns:
        Total surface area
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    crosses = jnp.cross(edge1, edge2)
    areas = 0.5 * jnp.linalg.norm(crosses, axis=1)
    
    return jnp.sum(areas)


def compute_mesh_volume(
    vertices: jnp.ndarray,
    triangles: jnp.ndarray,
) -> float:
    """
    Compute volume enclosed by a closed triangle mesh.
    
    Uses the divergence theorem: V = (1/6) * Σ v0 · (v1 × v2)
    
    Args:
        vertices: (n_verts, 3) vertex positions
        triangles: (n_triangles, 3) triangle vertex indices
    
    Returns:
        Enclosed volume (assumes outward-facing normals)
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    
    # Signed volume of each tetrahedron with origin
    signed_volumes = jnp.einsum('ij,ij->i', v0, jnp.cross(v1, v2))
    
    return jnp.abs(jnp.sum(signed_volumes)) / 6.0


def transform_mesh_to_world(
    vertices_local: jnp.ndarray,
    position: jnp.ndarray,
    rotation: jnp.ndarray,
) -> jnp.ndarray:
    """
    Transform mesh vertices from local to world frame.
    
    Args:
        vertices_local: (n_verts, 3) vertices in local frame
        position: (3,) position of local frame origin in world
        rotation: (3, 3) rotation matrix from local to world
    
    Returns:
        (n_verts, 3) vertices in world frame
    """
    return (rotation @ vertices_local.T).T + position


@partial(jax.jit, static_argnums=())
def compute_mesh_face_data(
    vertices: jnp.ndarray,
    triangles: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute face normals, areas, and centroids for all triangles.
    
    Args:
        vertices: (n_verts, 3) vertex positions
        triangles: (n_triangles, 3) triangle vertex indices
    
    Returns:
        normals: (n_triangles, 3) unit face normals
        areas: (n_triangles,) face areas
        centroids: (n_triangles, 3) face centroids
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    
    # Edges
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Cross product
    crosses = jnp.cross(edge1, edge2)
    cross_norms = jnp.linalg.norm(crosses, axis=1, keepdims=True)
    
    # Normals and areas
    normals = crosses / (cross_norms + 1e-10)
    areas = 0.5 * cross_norms.squeeze()
    
    # Centroids
    centroids = (v0 + v1 + v2) / 3.0
    
    return normals, areas, centroids


def create_box_mesh(
    length: float,
    width: float,
    height: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create a simple box mesh centered at origin.
    
    Args:
        length: Box length (x direction)
        width: Box width (y direction)
        height: Box height (z direction)
    
    Returns:
        vertices: (8, 3) vertex positions
        triangles: (12, 3) triangle indices
    """
    hx, hy, hz = length / 2, width / 2, height / 2
    
    vertices = np.array([
        [-hx, -hy, -hz],  # 0
        [ hx, -hy, -hz],  # 1
        [ hx,  hy, -hz],  # 2
        [-hx,  hy, -hz],  # 3
        [-hx, -hy,  hz],  # 4
        [ hx, -hy,  hz],  # 5
        [ hx,  hy,  hz],  # 6
        [-hx,  hy,  hz],  # 7
    ])
    
    # Two triangles per face, counter-clockwise from outside
    triangles = np.array([
        # Bottom (-z)
        [0, 2, 1], [0, 3, 2],
        # Top (+z)
        [4, 5, 6], [4, 6, 7],
        # Front (-y)
        [0, 1, 5], [0, 5, 4],
        # Back (+y)
        [2, 3, 7], [2, 7, 6],
        # Left (-x)
        [0, 4, 7], [0, 7, 3],
        # Right (+x)
        [1, 2, 6], [1, 6, 5],
    ])
    
    return jnp.array(vertices), jnp.array(triangles)


def create_hull_mesh_from_parameters(
    length: float,
    breadth: float,
    draft: float,
    freeboard: float = 5.0,
    n_length_divisions: int = 20,
    n_breadth_divisions: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create a simplified hull mesh (box-shaped for now).
    
    For a more realistic hull, this would use actual ship lines.
    
    Args:
        length: Ship length [m]
        breadth: Ship breadth [m]
        draft: Ship draft (below waterline) [m]
        freeboard: Height above waterline [m]
        n_length_divisions: Mesh divisions along length
        n_breadth_divisions: Mesh divisions along breadth
    
    Returns:
        vertices: Mesh vertices
        triangles: Triangle indices
    """
    # For simplicity, create a box hull
    # In a full implementation, this would import actual hull geometry
    total_height = draft + freeboard
    
    return create_box_mesh(length, breadth, total_height)
