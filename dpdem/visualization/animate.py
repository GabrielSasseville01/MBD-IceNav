"""
Animation utilities for DPDEM simulation.

Creates 2D and 3D animations of ice-platform interactions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection, PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple, Optional, Callable
import jax.numpy as jnp

from ..state import SimulationState, Polyhedron
from ..kinematics.quaternion import quaternion_to_rotation_matrix


def get_polygon_vertices_2d(
    position: np.ndarray,
    quaternion: np.ndarray,
    vertices_local: np.ndarray,
) -> np.ndarray:
    """
    Get 2D polygon vertices in world frame (top-down view).
    
    Args:
        position: (3,) particle position
        quaternion: (4,) particle orientation
        vertices_local: (n, 3) vertices in local frame
    
    Returns:
        (n, 2) vertices in world frame (x, y only)
    """
    R = np.array(quaternion_to_rotation_matrix(jnp.array(quaternion)))
    vertices_world = (R @ vertices_local.T).T + position
    return vertices_world[:, :2]  # Top-down view


def get_polyhedron_faces_3d(
    position: np.ndarray,
    quaternion: np.ndarray,
    vertices_local: np.ndarray,
    faces: np.ndarray,
    face_sizes: np.ndarray,
) -> List[np.ndarray]:
    """
    Get 3D face vertices in world frame.
    
    Args:
        position: (3,) particle position
        quaternion: (4,) particle orientation
        vertices_local: (n_verts, 3) vertices in local frame
        faces: (n_faces, max_verts) face indices
        face_sizes: (n_faces,) vertices per face
    
    Returns:
        List of (n_verts, 3) arrays for each face
    """
    R = np.array(quaternion_to_rotation_matrix(jnp.array(quaternion)))
    vertices_world = (R @ vertices_local.T).T + position
    
    face_verts = []
    for i in range(len(faces)):
        n_verts = int(face_sizes[i])
        face_indices = faces[i, :n_verts]
        face_verts.append(vertices_world[face_indices])
    
    return face_verts


def create_animation_frames(
    states: List[SimulationState],
    polyhedra: tuple,
    frame_skip: int = 1,
) -> List[dict]:
    """
    Extract animation frames from simulation states.
    
    Args:
        states: List of SimulationState objects
        polyhedra: Tuple of Polyhedron objects
        frame_skip: Only use every N-th frame
    
    Returns:
        List of frame dictionaries with positions, orientations, etc.
    """
    frames = []
    
    for i, state in enumerate(states[::frame_skip]):
        particles = state.particles
        platform = state.platform
        
        frame = {
            'time': float(state.time),
            'step': int(state.step),
            'particle_positions': np.array(particles.position),
            'particle_quaternions': np.array(particles.quaternion),
            'particle_polyhedron_idx': np.array(particles.polyhedron_idx),
            'platform_position': np.array(platform.position),
            'platform_quaternion': np.array(platform.quaternion),
            'n_particles': particles.n_particles,
        }
        frames.append(frame)
    
    return frames


def animate_simulation_2d(
    states: List[SimulationState],
    polyhedra: tuple,
    domain_size: Tuple[float, float] = None,
    figsize: Tuple[int, int] = (12, 8),
    interval: int = 50,
    frame_skip: int = 1,
    show_bonds: bool = False,
    ice_color: str = '#87CEEB',
    platform_color: str = '#CD853F',
    water_color: str = '#1E90FF',
    title: str = 'DPDEM Ice-Platform Interaction',
) -> FuncAnimation:
    """
    Create 2D animation of simulation (top-down view).
    
    Args:
        states: List of SimulationState objects
        polyhedra: Tuple of Polyhedron objects
        domain_size: (width, height) of domain, auto-detected if None
        figsize: Figure size in inches
        interval: Milliseconds between frames
        frame_skip: Only animate every N-th frame
        show_bonds: Whether to show bonds between elements
        ice_color: Color for ice elements
        platform_color: Color for platform
        water_color: Background water color
        title: Animation title
    
    Returns:
        matplotlib FuncAnimation object
    """
    frames = create_animation_frames(states, polyhedra, frame_skip)
    
    if len(frames) == 0:
        raise ValueError("No frames to animate")
    
    # Auto-detect domain size
    if domain_size is None:
        all_positions = np.vstack([f['particle_positions'] for f in frames])
        x_min, y_min = all_positions[:, 0].min() - 20, all_positions[:, 1].min() - 20
        x_max, y_max = all_positions[:, 0].max() + 20, all_positions[:, 1].max() + 20
        domain_size = (x_max - x_min, y_max - y_min)
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    else:
        x_center, y_center = domain_size[0] / 2, domain_size[1] / 2
        x_min, y_min = 0, 0
        x_max, y_max = domain_size
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_facecolor(water_color)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    
    # Title with time
    title_text = ax.set_title(f'{title}\nTime: 0.00 s', fontsize=14)
    
    # Ice element patches (will be updated)
    ice_patches = []
    
    # Platform patch
    platform_patch = None
    
    def init():
        """Initialize animation."""
        return []
    
    def update(frame_idx):
        """Update animation for frame."""
        nonlocal ice_patches, platform_patch
        
        frame = frames[frame_idx]
        
        # Clear previous patches
        for patch in ice_patches:
            patch.remove()
        ice_patches = []
        
        if platform_patch is not None:
            platform_patch.remove()
        
        # Draw ice elements
        for i in range(frame['n_particles']):
            poly_idx = int(frame['particle_polyhedron_idx'][i])
            poly = polyhedra[poly_idx]
            
            # Get 2D vertices
            verts_2d = get_polygon_vertices_2d(
                frame['particle_positions'][i],
                frame['particle_quaternions'][i],
                np.array(poly.vertices)
            )
            
            # Create polygon patch
            polygon = plt.Polygon(
                verts_2d, 
                closed=True, 
                facecolor=ice_color,
                edgecolor='#4682B4',
                linewidth=0.5,
                alpha=0.8
            )
            ax.add_patch(polygon)
            ice_patches.append(polygon)
        
        # Draw platform (simplified as rectangle)
        platform = states[0].platform
        p_pos = frame['platform_position']
        
        # Get platform dimensions from mesh
        verts = np.array(platform.mesh_vertices)
        p_width = verts[:, 0].max() - verts[:, 0].min()
        p_height = verts[:, 1].max() - verts[:, 1].min()
        
        # Apply rotation
        R = np.array(quaternion_to_rotation_matrix(jnp.array(frame['platform_quaternion'])))
        
        # Create rotated rectangle corners
        corners = np.array([
            [-p_width/2, -p_height/2],
            [p_width/2, -p_height/2],
            [p_width/2, p_height/2],
            [-p_width/2, p_height/2],
        ])
        # Rotate in 2D (around z-axis)
        cos_theta = R[0, 0]
        sin_theta = R[1, 0]
        R_2d = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        corners_rotated = (R_2d @ corners.T).T + p_pos[:2]
        
        platform_patch = plt.Polygon(
            corners_rotated,
            closed=True,
            facecolor=platform_color,
            edgecolor='#8B4513',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(platform_patch)
        
        # Update title
        title_text.set_text(f'{title}\nTime: {frame["time"]:.2f} s')
        
        return ice_patches + [platform_patch, title_text]
    
    anim = FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, interval=interval, blit=False
    )
    
    return anim


def animate_simulation_3d(
    states: List[SimulationState],
    polyhedra: tuple,
    domain_size: Tuple[float, float, float] = None,
    figsize: Tuple[int, int] = (12, 10),
    interval: int = 50,
    frame_skip: int = 1,
    elev: float = 30,
    azim: float = -60,
    ice_color: str = '#87CEEB',
    platform_color: str = '#CD853F',
    water_color: str = '#1E90FF',
    title: str = 'DPDEM Ice-Platform Interaction (3D)',
) -> FuncAnimation:
    """
    Create 3D animation of simulation.
    
    Args:
        states: List of SimulationState objects
        polyhedra: Tuple of Polyhedron objects
        domain_size: (x, y, z) domain size, auto-detected if None
        figsize: Figure size
        interval: Milliseconds between frames
        frame_skip: Only animate every N-th frame
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
        ice_color: Color for ice elements
        platform_color: Color for platform
        water_color: Color for water plane
        title: Animation title
    
    Returns:
        matplotlib FuncAnimation object
    """
    frames = create_animation_frames(states, polyhedra, frame_skip)
    
    if len(frames) == 0:
        raise ValueError("No frames to animate")
    
    # Auto-detect domain size
    if domain_size is None:
        all_positions = np.vstack([f['particle_positions'] for f in frames])
        x_min, y_min, z_min = all_positions.min(axis=0) - 20
        x_max, y_max, z_max = all_positions.max(axis=0) + 20
        z_min = min(z_min, -15)
        z_max = max(z_max, 10)
    else:
        x_min, y_min, z_min = 0, 0, -domain_size[2]/2
        x_max, y_max, z_max = domain_size[0], domain_size[1], domain_size[2]/2
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.view_init(elev=elev, azim=azim)
    
    title_text = ax.set_title(f'{title}\nTime: 0.00 s', fontsize=14)
    
    # Water plane at z=0
    water_x = np.array([[x_min, x_max], [x_min, x_max]])
    water_y = np.array([[y_min, y_min], [y_max, y_max]])
    water_z = np.array([[0, 0], [0, 0]])
    
    # Store collections for updating
    collections = []
    
    def update(frame_idx):
        nonlocal collections
        
        # Clear previous
        for coll in collections:
            coll.remove()
        collections = []
        
        frame = frames[frame_idx]
        
        # Draw water plane
        water_surf = ax.plot_surface(
            water_x, water_y, water_z,
            color=water_color, alpha=0.3
        )
        collections.append(water_surf)
        
        # Draw ice elements (simplified as boxes)
        for i in range(frame['n_particles']):
            poly_idx = int(frame['particle_polyhedron_idx'][i])
            poly = polyhedra[poly_idx]
            
            pos = frame['particle_positions'][i]
            quat = frame['particle_quaternions'][i]
            
            # Get face vertices
            face_verts = get_polyhedron_faces_3d(
                pos, quat,
                np.array(poly.vertices),
                np.array(poly.faces),
                np.array(poly.face_sizes)
            )
            
            # Create 3D polygon collection
            poly3d = Poly3DCollection(
                face_verts,
                facecolors=ice_color,
                edgecolors='#4682B4',
                linewidths=0.3,
                alpha=0.7
            )
            ax.add_collection3d(poly3d)
            collections.append(poly3d)
        
        # Draw platform
        platform = states[0].platform
        p_pos = frame['platform_position']
        p_quat = frame['platform_quaternion']
        
        # Get platform face vertices
        platform_faces = get_polyhedron_faces_3d(
            p_pos, p_quat,
            np.array(platform.mesh_vertices),
            np.array(platform.mesh_faces),
            np.ones(len(platform.mesh_faces)) * 3  # Triangles
        )
        
        platform_poly = Poly3DCollection(
            platform_faces,
            facecolors=platform_color,
            edgecolors='#8B4513',
            linewidths=0.5,
            alpha=0.9
        )
        ax.add_collection3d(platform_poly)
        collections.append(platform_poly)
        
        title_text.set_text(f'{title}\nTime: {frame["time"]:.2f} s')
        
        return collections
    
    anim = FuncAnimation(
        fig, update, frames=len(frames),
        interval=interval, blit=False
    )
    
    return anim


def save_animation(
    anim: FuncAnimation,
    filename: str,
    fps: int = 20,
    dpi: int = 150,
    writer: str = 'pillow',
):
    """
    Save animation to file.
    
    Args:
        anim: FuncAnimation object
        filename: Output filename (e.g., 'simulation.gif' or 'simulation.mp4')
        fps: Frames per second
        dpi: Resolution
        writer: Animation writer ('pillow' for GIF, 'ffmpeg' for MP4)
    """
    if writer == 'pillow':
        anim.save(filename, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(filename, writer=writer, fps=fps, dpi=dpi)
    
    print(f"Animation saved to {filename}")
