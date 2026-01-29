"""
Drag force computation for DPDEM.

Implements drag force and moment from Eqs. 28-29 in the paper.

Reference: Wang et al. (2024), Sun and Shen (2012), Liu and Ji (2018)
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple

from ..kinematics.quaternion import quaternion_to_rotation_matrix
from .buoyancy import clip_triangle_to_waterline, triangle_area_and_normal


@partial(jax.jit, static_argnums=())
def compute_drag_force(
    mesh_vertices: jnp.ndarray,
    mesh_faces: jnp.ndarray,
    position: jnp.ndarray,
    quaternion: jnp.ndarray,
    velocity: jnp.ndarray,
    v_water: jnp.ndarray,
    C_d_F: float,
    rho_water: float,
    waterline_z: float = 0.0,
) -> jnp.ndarray:
    """
    Compute drag force on a floating body.
    
    Implements Eq. 28:
    F_d = -0.5 * C_d^F * ρ_w * Σ A_sub^i * (v_s - v_w) * |v_s - v_w|
    
    Args:
        mesh_vertices: (n_verts, 3) vertices in local frame
        mesh_faces: (n_faces, 3) face vertex indices
        position: (3,) body position
        quaternion: (4,) body orientation
        velocity: (3,) body velocity [m/s]
        v_water: (3,) water current velocity [m/s]
        C_d_F: Drag force coefficient [-]
        rho_water: Water density [kg/m³]
        waterline_z: Z-coordinate of waterline [m]
    
    Returns:
        (3,) drag force [N]
    """
    R = quaternion_to_rotation_matrix(quaternion)
    vertices_world = (R @ mesh_vertices.T).T + position
    
    # Relative velocity (body minus water)
    v_rel = velocity - v_water
    v_rel_mag = jnp.linalg.norm(v_rel)
    
    # Accumulate drag from each face
    F_drag = jnp.zeros(3)
    n_faces = len(mesh_faces)
    
    for i in range(n_faces):
        face = mesh_faces[i]
        v0 = vertices_world[face[0]]
        v1 = vertices_world[face[1]]
        v2 = vertices_world[face[2]]
        
        # Clip to waterline
        cv0, cv1, cv2, case = clip_triangle_to_waterline(v0, v1, v2, waterline_z)
        
        # Compute wet area
        area, normal = triangle_area_and_normal(cv0, cv1, cv2)
        area = jnp.where(case == 0, 0.0, area)
        
        # Drag contribution from this face
        # Use the component of velocity normal to the face
        # (simplified model - full model would use more complex drag law)
        F_face = -0.5 * C_d_F * rho_water * area * v_rel * v_rel_mag
        F_drag += F_face
    
    return F_drag


@partial(jax.jit, static_argnums=())
def compute_drag_moment(
    mesh_vertices: jnp.ndarray,
    mesh_faces: jnp.ndarray,
    position: jnp.ndarray,
    quaternion: jnp.ndarray,
    velocity: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    v_water: jnp.ndarray,
    C_d_F: float,
    C_d_M: float,
    rho_water: float,
    r_eff: float,
    waterline_z: float = 0.0,
) -> jnp.ndarray:
    """
    Compute drag moment on a floating body.
    
    Implements Eq. 29:
    M_d = -0.5 * C_d^M * ρ_w * Σ A_sub^i * r_eff³ * ω * |ω| + r_d^i × F_d^i
    
    Args:
        mesh_vertices: (n_verts, 3) vertices in local frame
        mesh_faces: (n_faces, 3) face vertex indices
        position: (3,) body position (center of gravity)
        quaternion: (4,) body orientation
        velocity: (3,) body velocity [m/s]
        angular_velocity: (3,) angular velocity [rad/s]
        v_water: (3,) water current velocity [m/s]
        C_d_F: Drag force coefficient [-]
        C_d_M: Drag moment coefficient [-]
        rho_water: Water density [kg/m³]
        r_eff: Effective radius of submerged hull [m]
        waterline_z: Z-coordinate of waterline [m]
    
    Returns:
        (3,) drag moment about center of gravity [N·m]
    """
    R = quaternion_to_rotation_matrix(quaternion)
    vertices_world = (R @ mesh_vertices.T).T + position
    
    # Relative velocity
    v_rel = velocity - v_water
    v_rel_mag = jnp.linalg.norm(v_rel)
    
    # Angular velocity magnitude
    omega_mag = jnp.linalg.norm(angular_velocity)
    
    # Accumulate moment from each face
    M_drag = jnp.zeros(3)
    total_wet_area = 0.0
    
    n_faces = len(mesh_faces)
    
    for i in range(n_faces):
        face = mesh_faces[i]
        v0 = vertices_world[face[0]]
        v1 = vertices_world[face[1]]
        v2 = vertices_world[face[2]]
        
        # Clip to waterline
        cv0, cv1, cv2, case = clip_triangle_to_waterline(v0, v1, v2, waterline_z)
        
        # Compute wet area and centroid
        area, normal = triangle_area_and_normal(cv0, cv1, cv2)
        area = jnp.where(case == 0, 0.0, area)
        centroid = (cv0 + cv1 + cv2) / 3.0
        
        # Distance from CoG to face centroid
        r_d = centroid - position
        
        # Drag force on this face
        F_face = -0.5 * C_d_F * rho_water * area * v_rel * v_rel_mag
        
        # Moment contribution (r × F)
        M_face = jnp.cross(r_d, F_face)
        M_drag += M_face
        
        total_wet_area += area
    
    # Add rotational damping term
    # M_rot = -0.5 * C_d^M * ρ_w * A_wet * r_eff³ * ω * |ω|
    M_rotational = -0.5 * C_d_M * rho_water * total_wet_area * (r_eff ** 3) * angular_velocity * omega_mag
    
    return M_drag + M_rotational


def compute_total_drag(
    mesh_vertices: jnp.ndarray,
    mesh_faces: jnp.ndarray,
    position: jnp.ndarray,
    quaternion: jnp.ndarray,
    velocity: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    v_water: jnp.ndarray,
    C_d_F: float,
    C_d_M: float,
    rho_water: float,
    r_eff: float,
    waterline_z: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute total drag force and moment on a floating body.
    
    Args:
        mesh_vertices: (n_verts, 3) vertices in local frame
        mesh_faces: (n_faces, 3) face vertex indices
        position: (3,) body position
        quaternion: (4,) body orientation
        velocity: (3,) body velocity [m/s]
        angular_velocity: (3,) angular velocity [rad/s]
        v_water: (3,) water current velocity [m/s]
        C_d_F: Drag force coefficient
        C_d_M: Drag moment coefficient
        rho_water: Water density [kg/m³]
        r_eff: Effective radius [m]
        waterline_z: Z-coordinate of waterline
    
    Returns:
        F_d: (3,) drag force [N]
        M_d: (3,) drag moment [N·m]
    """
    F_d = compute_drag_force(
        mesh_vertices, mesh_faces, position, quaternion,
        velocity, v_water, C_d_F, rho_water, waterline_z
    )
    
    M_d = compute_drag_moment(
        mesh_vertices, mesh_faces, position, quaternion,
        velocity, angular_velocity, v_water,
        C_d_F, C_d_M, rho_water, r_eff, waterline_z
    )
    
    return F_d, M_d


@partial(jax.jit, static_argnums=())
def compute_ice_element_buoyancy(
    position: jnp.ndarray,
    volume: float,
    rho_ice: float,
    rho_water: float,
    g: jnp.ndarray,
    waterline_z: float = 0.0,
) -> jnp.ndarray:
    """
    Compute buoyancy force on an ice element.
    
    Simplified model treating ice element as sphere/box.
    
    Args:
        position: (3,) element center position
        volume: Element volume [m³]
        rho_ice: Ice density [kg/m³]
        rho_water: Water density [kg/m³]
        g: (3,) gravity vector
        waterline_z: Z-coordinate of waterline
    
    Returns:
        (3,) buoyancy force [N]
    """
    # Approximate submerged fraction based on z position
    z = position[2]
    element_radius = jnp.power(3.0 * volume / (4.0 * jnp.pi), 1.0/3.0)
    
    # Fraction submerged (linear approximation)
    depth = waterline_z - z
    submerged_fraction = jnp.clip((depth + element_radius) / (2.0 * element_radius), 0.0, 1.0)
    
    # Submerged volume
    V_sub = volume * submerged_fraction
    
    # Buoyancy force
    F_b = -rho_water * V_sub * g
    
    return F_b


@partial(jax.jit, static_argnums=())
def compute_ice_element_drag(
    position: jnp.ndarray,
    velocity: jnp.ndarray,
    volume: float,
    v_water: jnp.ndarray,
    C_d: float,
    rho_water: float,
    waterline_z: float = 0.0,
) -> jnp.ndarray:
    """
    Compute drag force on an ice element.
    
    Simplified model using equivalent sphere.
    
    Args:
        position: (3,) element center position
        velocity: (3,) element velocity
        volume: Element volume [m³]
        v_water: (3,) water current velocity
        C_d: Drag coefficient
        rho_water: Water density [kg/m³]
        waterline_z: Z-coordinate of waterline
    
    Returns:
        (3,) drag force [N]
    """
    # Equivalent radius
    r_eq = jnp.power(3.0 * volume / (4.0 * jnp.pi), 1.0/3.0)
    
    # Submerged fraction
    z = position[2]
    depth = waterline_z - z
    submerged_fraction = jnp.clip((depth + r_eq) / (2.0 * r_eq), 0.0, 1.0)
    
    # Submerged cross-sectional area
    A_sub = jnp.pi * r_eq ** 2 * submerged_fraction
    
    # Relative velocity
    v_rel = velocity - v_water
    v_rel_mag = jnp.linalg.norm(v_rel)
    
    # Drag force
    F_d = -0.5 * C_d * rho_water * A_sub * v_rel * v_rel_mag
    
    return F_d
