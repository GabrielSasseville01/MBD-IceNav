"""
Buoyancy computation for DPDEM.

Implements buoyancy force and moment calculation using tetrahedra
from Eqs. 22-27 in the paper.

Reference: Wang et al. (2024), Metrikin (2014), Liu and Ji (2018)
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from ..kinematics.quaternion import quaternion_to_rotation_matrix


@partial(jax.jit, static_argnums=())
def tetrahedron_volume(
    v0: jnp.ndarray,
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    v3: jnp.ndarray,
) -> float:
    """
    Compute signed volume of a tetrahedron.
    
    V = (1/6) * (v1 - v0) · ((v2 - v0) × (v3 - v0))
    
    Positive if vertices follow right-hand rule.
    
    Args:
        v0, v1, v2, v3: (3,) tetrahedron vertices
    
    Returns:
        Signed volume [m³]
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    edge3 = v3 - v0
    
    return jnp.dot(edge1, jnp.cross(edge2, edge3)) / 6.0


@partial(jax.jit, static_argnums=())
def tetrahedron_centroid(
    v0: jnp.ndarray,
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    v3: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute centroid of a tetrahedron.
    
    x_c = (1/4) * (v0 + v1 + v2 + v3)
    
    Implements Eq. 23.
    
    Args:
        v0, v1, v2, v3: (3,) tetrahedron vertices
    
    Returns:
        (3,) centroid position
    """
    return 0.25 * (v0 + v1 + v2 + v3)


@partial(jax.jit, static_argnums=())
def triangle_area_and_normal(
    v0: jnp.ndarray,
    v1: jnp.ndarray,
    v2: jnp.ndarray,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute area and normal of a triangle.
    
    Args:
        v0, v1, v2: (3,) triangle vertices
    
    Returns:
        area: Triangle area [m²]
        normal: (3,) unit outward normal
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = jnp.cross(edge1, edge2)
    area = 0.5 * jnp.linalg.norm(cross)
    normal = cross / (2.0 * area + 1e-10)
    return area, normal


@partial(jax.jit, static_argnums=())
def clip_triangle_to_waterline(
    v0: jnp.ndarray,
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    waterline_z: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Clip a triangle to the waterline plane.
    
    Implements the waterline intersection cases from Figs. 6-7:
    - Case 0: Triangle fully above waterline (no submerged portion)
    - Case 1: Triangle fully submerged
    - Case 2: One vertex above, two below -> one triangle submerged
    - Case 3: Two vertices above, one below -> one triangle submerged
    
    Args:
        v0, v1, v2: (3,) triangle vertices
        waterline_z: Z-coordinate of waterline [m]
    
    Returns:
        clipped_v0, clipped_v1, clipped_v2: Clipped triangle vertices
        case: 0=above, 1=submerged, 2=one above, 3=two above
    """
    # Check which vertices are below waterline
    below0 = v0[2] < waterline_z
    below1 = v1[2] < waterline_z
    below2 = v2[2] < waterline_z
    
    n_below = below0.astype(int) + below1.astype(int) + below2.astype(int)
    
    def interpolate_to_waterline(va, vb):
        """Find intersection point with waterline plane."""
        t = (waterline_z - va[2]) / (vb[2] - va[2] + 1e-10)
        return va + t * (vb - va)
    
    # Case: fully above
    def case_above():
        return jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), 0
    
    # Case: fully submerged
    def case_submerged():
        return v0, v1, v2, 1
    
    # Case: one vertex above
    def case_one_above():
        # Find the vertex that is above
        # Return triangle formed by two below vertices and intersection points
        # This is simplified - full implementation would handle all orderings
        if not below0:
            e1 = interpolate_to_waterline(v0, v1)
            e2 = interpolate_to_waterline(v0, v2)
            return v1, v2, e1, 2  # Approximate with single triangle
        elif not below1:
            e1 = interpolate_to_waterline(v1, v0)
            e2 = interpolate_to_waterline(v1, v2)
            return v0, v2, e1, 2
        else:  # v2 above
            e1 = interpolate_to_waterline(v2, v0)
            e2 = interpolate_to_waterline(v2, v1)
            return v0, v1, e1, 2
    
    # Case: two vertices above
    def case_two_above():
        # Find the vertex that is below
        if below0:
            e1 = interpolate_to_waterline(v0, v1)
            e2 = interpolate_to_waterline(v0, v2)
            return v0, e1, e2, 3
        elif below1:
            e1 = interpolate_to_waterline(v1, v0)
            e2 = interpolate_to_waterline(v1, v2)
            return v1, e1, e2, 3
        else:  # v2 below
            e1 = interpolate_to_waterline(v2, v0)
            e2 = interpolate_to_waterline(v2, v1)
            return v2, e1, e2, 3
    
    # Select case based on number below
    result = jax.lax.switch(
        n_below,
        [case_above, case_two_above, case_one_above, case_submerged],
    )
    
    return result


@partial(jax.jit, static_argnums=())
def compute_submerged_tetrahedron(
    O_p: jnp.ndarray,
    v0: jnp.ndarray,
    v1: jnp.ndarray,
    v2: jnp.ndarray,
    waterline_z: float,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute submerged volume and centroid for a face.
    
    Forms a tetrahedron from hull center O_p to face vertices,
    clipped at waterline.
    
    Implements Eqs. 23-24.
    
    Args:
        O_p: (3,) hull center point
        v0, v1, v2: (3,) face vertices in world frame
        waterline_z: Z-coordinate of waterline
    
    Returns:
        volume: Submerged volume contribution [m³]
        centroid: (3,) centroid of submerged tetrahedron
    """
    # Clip triangle to waterline
    cv0, cv1, cv2, case = clip_triangle_to_waterline(v0, v1, v2, waterline_z)
    
    # Compute tetrahedron volume and centroid
    # Tetrahedron: O_p, cv0, cv1, cv2
    volume = tetrahedron_volume(O_p, cv0, cv1, cv2)
    centroid = tetrahedron_centroid(O_p, cv0, cv1, cv2)
    
    # Zero out if above waterline
    volume = jnp.where(case == 0, 0.0, volume)
    centroid = jnp.where(case == 0, jnp.zeros(3), centroid)
    
    return volume, centroid


def compute_submerged_volume(
    mesh_vertices: jnp.ndarray,
    mesh_faces: jnp.ndarray,
    position: jnp.ndarray,
    quaternion: jnp.ndarray,
    waterline_z: float = 0.0,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute total submerged volume and center of buoyancy.
    
    Implements Eqs. 25-26:
    V_sub = Σ V_i
    x_buoyancy = Σ(x_i * V_i) / V_sub
    
    Args:
        mesh_vertices: (n_verts, 3) vertices in local frame
        mesh_faces: (n_faces, 3) face vertex indices
        position: (3,) hull position (center of gravity)
        quaternion: (4,) hull orientation
        waterline_z: Z-coordinate of waterline [m]
    
    Returns:
        V_sub: Total submerged volume [m³]
        x_buoyancy: (3,) center of buoyancy in world frame
    """
    # Transform vertices to world frame
    R = quaternion_to_rotation_matrix(quaternion)
    vertices_world = (R @ mesh_vertices.T).T + position
    
    # Hull center (O_p in paper notation)
    O_p = position
    
    # Accumulate volumes and weighted centroids
    total_volume = 0.0
    weighted_centroid = jnp.zeros(3)
    
    n_faces = len(mesh_faces)
    
    for i in range(n_faces):
        face = mesh_faces[i]
        v0 = vertices_world[face[0]]
        v1 = vertices_world[face[1]]
        v2 = vertices_world[face[2]]
        
        vol, cent = compute_submerged_tetrahedron(O_p, v0, v1, v2, waterline_z)
        
        total_volume += vol
        weighted_centroid += cent * vol
    
    # Center of buoyancy
    x_buoyancy = weighted_centroid / (jnp.abs(total_volume) + 1e-10)
    
    return jnp.abs(total_volume), x_buoyancy


@partial(jax.jit, static_argnums=())
def compute_buoyancy_force(
    V_sub: float,
    rho_water: float,
    g: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute buoyancy force.
    
    Implements Eq. 22:
    F_b = -ρ_w * V_sub * g
    
    Args:
        V_sub: Submerged volume [m³]
        rho_water: Water density [kg/m³]
        g: (3,) gravity vector [m/s²]
    
    Returns:
        (3,) buoyancy force [N]
    """
    return -rho_water * V_sub * g


@partial(jax.jit, static_argnums=())
def compute_buoyancy_moment(
    F_b: jnp.ndarray,
    x_buoyancy: jnp.ndarray,
    x_cog: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute buoyancy moment about center of gravity.
    
    Implements Eq. 27:
    M_b = (x_buoyancy - x_cog) × F_b
    
    Args:
        F_b: (3,) buoyancy force [N]
        x_buoyancy: (3,) center of buoyancy [m]
        x_cog: (3,) center of gravity [m]
    
    Returns:
        (3,) buoyancy moment [N·m]
    """
    r = x_buoyancy - x_cog
    return jnp.cross(r, F_b)


def compute_total_buoyancy(
    mesh_vertices: jnp.ndarray,
    mesh_faces: jnp.ndarray,
    position: jnp.ndarray,
    quaternion: jnp.ndarray,
    rho_water: float,
    g: jnp.ndarray,
    waterline_z: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute total buoyancy force and moment on a floating body.
    
    Args:
        mesh_vertices: (n_verts, 3) vertices in local frame
        mesh_faces: (n_faces, 3) face vertex indices
        position: (3,) body position (center of gravity)
        quaternion: (4,) body orientation
        rho_water: Water density [kg/m³]
        g: (3,) gravity vector [m/s²]
        waterline_z: Z-coordinate of waterline [m]
    
    Returns:
        F_b: (3,) buoyancy force [N]
        M_b: (3,) buoyancy moment about CoG [N·m]
    """
    # Compute submerged volume and center of buoyancy
    V_sub, x_buoyancy = compute_submerged_volume(
        mesh_vertices, mesh_faces, position, quaternion, waterline_z
    )
    
    # Buoyancy force
    F_b = compute_buoyancy_force(V_sub, rho_water, g)
    
    # Buoyancy moment
    M_b = compute_buoyancy_moment(F_b, x_buoyancy, position)
    
    return F_b, M_b


@partial(jax.jit, static_argnums=())
def compute_wet_surface_area(
    mesh_vertices: jnp.ndarray,
    mesh_faces: jnp.ndarray,
    position: jnp.ndarray,
    quaternion: jnp.ndarray,
    waterline_z: float = 0.0,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute wet surface area for drag calculation.
    
    Args:
        mesh_vertices: (n_verts, 3) vertices in local frame
        mesh_faces: (n_faces, 3) face vertex indices
        position: (3,) body position
        quaternion: (4,) body orientation
        waterline_z: Z-coordinate of waterline
    
    Returns:
        A_wet: Total wet surface area [m²]
        face_areas: (n_faces,) wet area per face [m²]
    """
    R = quaternion_to_rotation_matrix(quaternion)
    vertices_world = (R @ mesh_vertices.T).T + position
    
    n_faces = len(mesh_faces)
    face_areas = jnp.zeros(n_faces)
    
    for i in range(n_faces):
        face = mesh_faces[i]
        v0 = vertices_world[face[0]]
        v1 = vertices_world[face[1]]
        v2 = vertices_world[face[2]]
        
        # Clip to waterline
        cv0, cv1, cv2, case = clip_triangle_to_waterline(v0, v1, v2, waterline_z)
        
        # Compute area
        area, _ = triangle_area_and_normal(cv0, cv1, cv2)
        
        # Zero if above water
        area = jnp.where(case == 0, 0.0, area)
        face_areas = face_areas.at[i].set(area)
    
    return jnp.sum(face_areas), face_areas
