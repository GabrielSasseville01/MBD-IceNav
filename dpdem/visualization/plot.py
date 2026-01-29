"""
Static plotting utilities for DPDEM simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from typing import List, Tuple, Optional
import jax.numpy as jnp

from ..state import SimulationState, Polyhedron
from ..kinematics.quaternion import quaternion_to_rotation_matrix


def plot_ice_field(
    state: SimulationState,
    polyhedra: tuple,
    ax: Optional[plt.Axes] = None,
    ice_color: str = '#87CEEB',
    edge_color: str = '#4682B4',
    alpha: float = 0.8,
    show_bonds: bool = False,
    bond_color: str = 'red',
) -> plt.Axes:
    """
    Plot ice field from simulation state.
    
    Args:
        state: SimulationState
        polyhedra: Tuple of Polyhedron objects
        ax: Matplotlib axes (created if None)
        ice_color: Fill color for ice
        edge_color: Edge color for ice
        alpha: Transparency
        show_bonds: Whether to show bonds
        bond_color: Color for bond lines
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    particles = state.particles
    
    patches = []
    for i in range(particles.n_particles):
        poly_idx = int(particles.polyhedron_idx[i])
        poly = polyhedra[poly_idx]
        
        # Get vertices in world frame
        R = np.array(quaternion_to_rotation_matrix(jnp.array(particles.quaternion[i])))
        verts_world = (R @ np.array(poly.vertices).T).T + np.array(particles.position[i])
        
        # Take top-down view (x, y)
        verts_2d = verts_world[:, :2]
        
        # Create polygon
        polygon = Polygon(verts_2d, closed=True)
        patches.append(polygon)
    
    collection = PatchCollection(
        patches, 
        facecolor=ice_color,
        edgecolor=edge_color,
        linewidth=0.5,
        alpha=alpha
    )
    ax.add_collection(collection)
    
    # Show bonds
    if show_bonds and state.bonds.n_bonds > 0:
        bonds = state.bonds
        for b in range(bonds.n_bonds):
            if bonds.intact[b]:
                i = int(bonds.particle_i[b])
                j = int(bonds.particle_j[b])
                
                pos_i = particles.position[i]
                pos_j = particles.position[j]
                
                ax.plot(
                    [pos_i[0], pos_j[0]],
                    [pos_i[1], pos_j[1]],
                    color=bond_color, linewidth=0.5, alpha=0.5
                )
    
    ax.set_aspect('equal')
    ax.autoscale_view()
    
    return ax


def plot_platform(
    state: SimulationState,
    ax: Optional[plt.Axes] = None,
    color: str = '#CD853F',
    edge_color: str = '#8B4513',
) -> plt.Axes:
    """
    Plot platform from simulation state.
    
    Args:
        state: SimulationState
        ax: Matplotlib axes
        color: Fill color
        edge_color: Edge color
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    platform = state.platform
    
    # Get platform mesh in world frame
    R = np.array(quaternion_to_rotation_matrix(jnp.array(platform.quaternion)))
    verts_world = (R @ np.array(platform.mesh_vertices).T).T + np.array(platform.position)
    
    # Get bounding box in x-y plane
    x_min, x_max = verts_world[:, 0].min(), verts_world[:, 0].max()
    y_min, y_max = verts_world[:, 1].min(), verts_world[:, 1].max()
    
    rect = plt.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=color, edgecolor=edge_color, linewidth=2
    )
    ax.add_patch(rect)
    
    return ax


def plot_forces_time_history(
    states: List[SimulationState],
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot force time history from simulation.
    
    Args:
        states: List of simulation states
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    times = [float(s.time) for s in states]
    
    # Extract platform forces (would need to be computed/stored in simulation)
    # Placeholder for demonstration
    platform_forces_x = np.random.randn(len(states)) * 1e6
    platform_forces_y = np.random.randn(len(states)) * 1e6
    platform_forces_z = np.random.randn(len(states)) * 1e6
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    axes[0].plot(times, platform_forces_x / 1e6, 'b-')
    axes[0].set_ylabel('F_x [MN]')
    axes[0].set_title('Platform Ice Forces')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(times, platform_forces_y / 1e6, 'g-')
    axes[1].set_ylabel('F_y [MN]')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(times, platform_forces_z / 1e6, 'r-')
    axes[2].set_ylabel('F_z [MN]')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_motion_time_history(
    states: List[SimulationState],
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Plot platform motion time history.
    
    Args:
        states: List of simulation states
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    times = [float(s.time) for s in states]
    
    # Extract platform motion
    positions = np.array([np.array(s.platform.position) for s in states])
    velocities = np.array([np.array(s.platform.velocity) for s in states])
    
    # Convert quaternions to Euler angles (simplified)
    roll = []
    pitch = []
    yaw = []
    
    for s in states:
        q = np.array(s.platform.quaternion)
        # Simplified Euler angle extraction
        roll.append(np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)))
        pitch.append(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))
        yaw.append(np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2)))
    
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # Positions
    axes[0, 0].plot(times, positions[:, 0], 'b-')
    axes[0, 0].set_ylabel('Surge [m]')
    axes[0, 0].set_title('Platform Position')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(times, positions[:, 1], 'g-')
    axes[1, 0].set_ylabel('Sway [m]')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[2, 0].plot(times, positions[:, 2], 'r-')
    axes[2, 0].set_ylabel('Heave [m]')
    axes[2, 0].set_xlabel('Time [s]')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Rotations
    axes[0, 1].plot(times, roll, 'b-')
    axes[0, 1].set_ylabel('Roll [deg]')
    axes[0, 1].set_title('Platform Rotation')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(times, pitch, 'g-')
    axes[1, 1].set_ylabel('Pitch [deg]')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 1].plot(times, yaw, 'r-')
    axes[2, 1].set_ylabel('Yaw [deg]')
    axes[2, 1].set_xlabel('Time [s]')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
