"""
Polyhedral geometry operations for DPDEM.

Handles extrusion of 2D polygons to 3D polyhedra and computation
of geometric properties (volume, centroid, inertia tensor).
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, List
from .voronoi import compute_polygon_area, compute_polygon_centroid, order_polygon_vertices_ccw


def extrude_polygon_to_polyhedron(
    polygon_2d: np.ndarray,
    thickness: float,
    center_at_centroid: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Extrude a 2D polygon to a 3D polyhedron (prism).
    
    Creates a prism by duplicating the polygon at two z-levels.
    
    Args:
        polygon_2d: (n, 2) array of polygon vertices in order
        thickness: Height of the prism [m]
        center_at_centroid: If True, center vertices at the centroid
    
    Returns:
        vertices: (2*n, 3) array of 3D vertices
        faces: List of face vertex index lists
        face_sizes: Number of vertices per face
    """
    # Ensure counter-clockwise ordering
    polygon_2d = order_polygon_vertices_ccw(polygon_2d)
    n = len(polygon_2d)
    
    # Create top and bottom vertices
    z_bottom = 0.0
    z_top = thickness
    
    bottom_verts = np.hstack([polygon_2d, np.full((n, 1), z_bottom)])
    top_verts = np.hstack([polygon_2d, np.full((n, 1), z_top)])
    
    vertices = np.vstack([bottom_verts, top_verts])
    
    # Center at centroid if requested
    if center_at_centroid:
        centroid_2d = compute_polygon_centroid(polygon_2d)
        centroid_3d = np.array([centroid_2d[0], centroid_2d[1], thickness / 2.0])
        vertices -= centroid_3d
    
    # Create faces
    # Bottom face (indices 0 to n-1, reversed for outward normal)
    bottom_face = list(range(n-1, -1, -1))
    
    # Top face (indices n to 2n-1)
    top_face = list(range(n, 2*n))
    
    # Side faces (quads connecting bottom to top)
    side_faces = []
    for i in range(n):
        j = (i + 1) % n
        # Bottom vertices: i, j
        # Top vertices: n+i, n+j
        # Quad: i, j, n+j, n+i (counter-clockwise from outside)
        side_faces.append([i, j, n+j, n+i])
    
    # Combine all faces
    faces = [bottom_face, top_face] + side_faces
    face_sizes = np.array([len(f) for f in faces])
    
    # Pad faces to uniform length
    max_verts = max(face_sizes)
    faces_padded = np.full((len(faces), max_verts), -1, dtype=np.int32)
    for i, face in enumerate(faces):
        faces_padded[i, :len(face)] = face
    
    return jnp.array(vertices), jnp.array(faces_padded), jnp.array(face_sizes)


def compute_polyhedron_volume(
    vertices: jnp.ndarray,
    faces: jnp.ndarray,
    face_sizes: jnp.ndarray,
) -> float:
    """
    Compute volume of a polyhedron using divergence theorem.
    
    V = (1/6) * Σ_faces (v0 · (v1 × v2)) for triangulated faces
    
    Args:
        vertices: (n_verts, 3) array of vertices
        faces: (n_faces, max_verts) array of face vertex indices
        face_sizes: (n_faces,) array of vertices per face
    
    Returns:
        Volume of the polyhedron [m³]
    """
    volume = 0.0
    n_faces = len(faces)
    
    for i in range(n_faces):
        n_verts = face_sizes[i]
        face_verts = faces[i, :n_verts]
        
        # Triangulate the face using fan triangulation
        v0 = vertices[face_verts[0]]
        for j in range(1, n_verts - 1):
            v1 = vertices[face_verts[j]]
            v2 = vertices[face_verts[j + 1]]
            
            # Signed volume of tetrahedron with origin
            volume += jnp.dot(v0, jnp.cross(v1, v2))
    
    return jnp.abs(volume) / 6.0


def compute_polyhedron_centroid(
    vertices: jnp.ndarray,
    faces: jnp.ndarray,
    face_sizes: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute centroid of a polyhedron.
    
    For a polyhedron with vertices centered at origin, this should be near zero.
    
    Args:
        vertices: (n_verts, 3) array of vertices
        faces: (n_faces, max_verts) array of face vertex indices
        face_sizes: (n_faces,) array of vertices per face
    
    Returns:
        (3,) centroid position
    """
    # Simple approximation: weighted average of face centroids
    total_area = 0.0
    centroid = jnp.zeros(3)
    n_faces = len(faces)
    
    for i in range(n_faces):
        n_verts = face_sizes[i]
        face_verts = faces[i, :n_verts]
        
        # Face centroid
        face_centroid = jnp.mean(vertices[face_verts], axis=0)
        
        # Face area (approximate using triangulation)
        v0 = vertices[face_verts[0]]
        area = 0.0
        for j in range(1, n_verts - 1):
            v1 = vertices[face_verts[j]]
            v2 = vertices[face_verts[j + 1]]
            area += 0.5 * jnp.linalg.norm(jnp.cross(v1 - v0, v2 - v0))
        
        centroid += face_centroid * area
        total_area += area
    
    return centroid / (total_area + 1e-10)


def compute_polyhedron_inertia_tensor(
    vertices: jnp.ndarray,
    faces: jnp.ndarray,
    face_sizes: jnp.ndarray,
    density: float,
) -> jnp.ndarray:
    """
    Compute inertia tensor of a polyhedron assuming uniform density.
    
    Uses the divergence theorem approach for polyhedral inertia.
    
    Args:
        vertices: (n_verts, 3) array of vertices (centered at centroid)
        faces: (n_faces, max_verts) array of face vertex indices
        face_sizes: (n_faces,) array of vertices per face
        density: Material density [kg/m³]
    
    Returns:
        (3, 3) inertia tensor in body frame [kg·m²]
    """
    # Compute using tetrahedra from origin to each face
    Ixx, Iyy, Izz = 0.0, 0.0, 0.0
    Ixy, Ixz, Iyz = 0.0, 0.0, 0.0
    
    n_faces = len(faces)
    
    for i in range(n_faces):
        n_verts = face_sizes[i]
        face_verts = faces[i, :n_verts]
        
        # Triangulate face
        v0 = vertices[face_verts[0]]
        for j in range(1, n_verts - 1):
            v1 = vertices[face_verts[j]]
            v2 = vertices[face_verts[j + 1]]
            
            # Tetrahedron from origin to triangle
            # Volume of tetrahedron
            vol = jnp.abs(jnp.dot(v0, jnp.cross(v1, v2))) / 6.0
            
            # Centroid of tetrahedron
            c = (v0 + v1 + v2) / 4.0
            
            # Inertia of tetrahedron about origin (parallel axis theorem)
            # Using simplified formula for tetrahedron with one vertex at origin
            # I = ρV * (sum of products)
            
            # Diagonal terms
            xx = v0[0]**2 + v1[0]**2 + v2[0]**2 + v0[0]*v1[0] + v0[0]*v2[0] + v1[0]*v2[0]
            yy = v0[1]**2 + v1[1]**2 + v2[1]**2 + v0[1]*v1[1] + v0[1]*v2[1] + v1[1]*v2[1]
            zz = v0[2]**2 + v1[2]**2 + v2[2]**2 + v0[2]*v1[2] + v0[2]*v2[2] + v1[2]*v2[2]
            
            Ixx += density * vol * (yy + zz) / 10.0
            Iyy += density * vol * (xx + zz) / 10.0
            Izz += density * vol * (xx + yy) / 10.0
            
            # Off-diagonal terms
            xy = (v0[0]*v0[1] + v1[0]*v1[1] + v2[0]*v2[1] + 
                  (v0[0]*v1[1] + v1[0]*v0[1] + v0[0]*v2[1] + v2[0]*v0[1] + v1[0]*v2[1] + v2[0]*v1[1]) / 2.0)
            xz = (v0[0]*v0[2] + v1[0]*v1[2] + v2[0]*v2[2] + 
                  (v0[0]*v1[2] + v1[0]*v0[2] + v0[0]*v2[2] + v2[0]*v0[2] + v1[0]*v2[2] + v2[0]*v1[2]) / 2.0)
            yz = (v0[1]*v0[2] + v1[1]*v1[2] + v2[1]*v2[2] + 
                  (v0[1]*v1[2] + v1[1]*v0[2] + v0[1]*v2[2] + v2[1]*v0[2] + v1[1]*v2[2] + v2[1]*v1[2]) / 2.0)
            
            Ixy -= density * vol * xy / 10.0
            Ixz -= density * vol * xz / 10.0
            Iyz -= density * vol * yz / 10.0
    
    return jnp.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])


def compute_face_normals(
    vertices: jnp.ndarray,
    faces: jnp.ndarray,
    face_sizes: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute outward-facing normals for each face.
    
    Args:
        vertices: (n_verts, 3) array of vertices
        faces: (n_faces, max_verts) array of face vertex indices
        face_sizes: (n_faces,) array of vertices per face
    
    Returns:
        (n_faces, 3) array of unit normals
    """
    n_faces = len(faces)
    normals = jnp.zeros((n_faces, 3))
    
    for i in range(n_faces):
        # Use first three vertices to compute normal
        v0 = vertices[faces[i, 0]]
        v1 = vertices[faces[i, 1]]
        v2 = vertices[faces[i, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = jnp.cross(edge1, edge2)
        normal = normal / (jnp.linalg.norm(normal) + 1e-10)
        
        normals = normals.at[i].set(normal)
    
    return normals


def compute_equivalent_sphere_radius(volume: float) -> float:
    """
    Compute radius of sphere with equivalent volume.
    
    R = (3V / 4π)^(1/3)
    
    Args:
        volume: Polyhedron volume [m³]
    
    Returns:
        Equivalent sphere radius [m]
    """
    return jnp.power(3.0 * volume / (4.0 * jnp.pi), 1.0 / 3.0)


def create_polyhedron_from_polygon(
    polygon_2d: np.ndarray,
    thickness: float,
    density: float,
    dilation_radius: float,
) -> 'Polyhedron':
    """
    Create a complete Polyhedron object from a 2D polygon.
    
    Args:
        polygon_2d: (n, 2) array of polygon vertices
        thickness: Ice thickness [m]
        density: Ice density [kg/m³]
        dilation_radius: Dilation sphere radius [m]
    
    Returns:
        Polyhedron NamedTuple with all geometric properties
    """
    from ..state import Polyhedron
    
    # Extrude to 3D
    vertices, faces, face_sizes = extrude_polygon_to_polyhedron(
        polygon_2d, thickness, center_at_centroid=True
    )
    
    # Compute geometric properties
    volume = compute_polyhedron_volume(vertices, faces, face_sizes)
    face_normals = compute_face_normals(vertices, faces, face_sizes)
    equivalent_radius = compute_equivalent_sphere_radius(volume)
    
    # Compute inertia tensor
    inertia_full = compute_polyhedron_inertia_tensor(vertices, faces, face_sizes, density)
    inertia_diag = jnp.diag(inertia_full)  # Assume principal axes aligned with body frame
    
    return Polyhedron(
        vertices=vertices,
        faces=faces,
        face_sizes=face_sizes,
        face_normals=face_normals,
        dilation_radius=dilation_radius,
        volume=float(volume),
        equivalent_radius=float(equivalent_radius),
        inertia_local=inertia_diag,
    )
