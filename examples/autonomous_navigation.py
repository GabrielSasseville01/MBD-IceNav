#!/usr/bin/env python3
"""
Example: Autonomous navigation through ice field.

Demonstrates the integration of lattice-based path planning and
DP control with the DPDEM ice simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dpdem import AutonomousNavigator, NavigationConfig
from dpdem.control import ShipState
from dpdem.planning import CostMap


def create_random_ice_field(
    n_floes: int,
    map_width: float,
    map_height: float,
    min_size: float = 5.0,
    max_size: float = 20.0,
    start_buffer: float = 50.0,
    goal_buffer: float = 50.0,
    seed: int = 42,
):
    """Generate a random ice field for testing."""
    np.random.seed(seed)
    
    positions = []
    polygons = []
    masses = []
    angles = []
    
    for _ in range(n_floes):
        # Random position (avoid start and goal areas)
        x = np.random.uniform(start_buffer + max_size, map_width - goal_buffer - max_size)
        y = np.random.uniform(max_size, map_height - max_size)
        
        # Random size
        size = np.random.uniform(min_size, max_size)
        
        # Random polygon (5-8 vertices)
        n_vertices = np.random.randint(5, 9)
        angles_v = np.sort(np.random.uniform(0, 2*np.pi, n_vertices))
        radii = size * (0.7 + 0.3 * np.random.rand(n_vertices))
        
        vertices = np.column_stack([
            radii * np.cos(angles_v),
            radii * np.sin(angles_v)
        ])
        
        # Random orientation
        angle = np.random.uniform(0, 2*np.pi)
        
        # Mass based on size (assuming 1m thickness, 900 kg/m³)
        area = 0.5 * np.abs(np.sum(
            vertices[:-1, 0] * vertices[1:, 1] - 
            vertices[1:, 0] * vertices[:-1, 1]
        ))
        mass = area * 1.0 * 900  # area * thickness * density
        
        positions.append([x, y])
        polygons.append(vertices)
        masses.append(mass)
        angles.append(angle)
    
    return (
        np.array(positions),
        polygons,
        np.array(masses),
        np.array(angles),
    )


def transform_polygon(vertices, x, y, angle):
    """Transform polygon to world coordinates."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return vertices @ R.T + np.array([x, y])


def create_ship_polygon(x, y, heading, length, width):
    """Create ship polygon for visualization."""
    # Ship shape (pointed bow)
    half_l = length / 2
    half_w = width / 2
    
    # Local vertices (bow at +x)
    vertices = np.array([
        [half_l, 0],           # Bow point
        [half_l * 0.7, half_w],    # Port bow
        [-half_l, half_w],         # Port stern
        [-half_l, -half_w],        # Starboard stern
        [half_l * 0.7, -half_w],   # Starboard bow
    ])
    
    # Transform to world
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    return vertices @ R.T + np.array([x, y])


def run_simulation(
    n_steps: int = 500,
    dt: float = 0.1,
    n_floes: int = 30,
    seed: int = 42,
    animate: bool = True,
    save_animation: bool = False,
):
    """
    Run a simple autonomous navigation simulation.
    
    This is a simplified simulation without full DPDEM physics,
    demonstrating the planner and controller integration.
    """
    # Configuration
    config = NavigationConfig(
        map_width=200.0,
        map_height=100.0,
        map_resolution=1.0,
        ship_length=20.0,
        ship_width=8.0,
        ship_mass=1e5,
        turning_radius=30.0,
        target_speed=2.0,
        replan_interval=2.0,
        goal_tolerance=5.0,
    )
    
    # Create navigator
    navigator = AutonomousNavigator(config)
    
    # Generate ice field
    positions, polygons, masses, angles = create_random_ice_field(
        n_floes=n_floes,
        map_width=config.map_width,
        map_height=config.map_height,
        min_size=3.0,
        max_size=10.0,
        start_buffer=30.0,
        goal_buffer=30.0,
        seed=seed,
    )
    
    # Update cost map
    navigator.update_ice_field(positions, polygons, masses, angles)
    
    # Set start and goal
    start_pos = (20.0, config.map_height / 2)
    goal_pos = (config.map_width - 20.0, config.map_height / 2)
    
    # Initialize ship state
    navigator.ship_state = ShipState(
        x=start_pos[0],
        y=start_pos[1],
        psi=0.0,  # Heading east
        u=0.0,
        v=0.0,
        r=0.0,
        dt=dt,
    )
    
    # Set goal
    navigator.set_goal(goal_pos)
    
    # History for plotting
    history = {
        'x': [],
        'y': [],
        'psi': [],
        'u': [],
        'path': [],
        'setpoint': [],
    }
    
    print(f"Starting autonomous navigation simulation")
    print(f"  Start: {start_pos}")
    print(f"  Goal: {goal_pos}")
    print(f"  Ice floes: {n_floes}")
    
    # Run simulation
    for step in range(n_steps):
        # Get control
        control, nav_state = navigator.step(dt)
        
        # Simple ship dynamics (no ice interaction in this demo)
        # Update velocities based on control forces
        mass = config.ship_mass
        I_zz = (1/12) * mass * (config.ship_length**2 + config.ship_width**2)
        
        # Simple drag
        drag_u = 0.1 * mass * navigator.ship_state.u
        drag_v = 0.2 * mass * navigator.ship_state.v
        drag_r = 0.1 * I_zz * navigator.ship_state.r
        
        # Accelerations
        acc_u = (control[0] - drag_u) / mass
        acc_v = (control[1] - drag_v) / mass
        acc_r = (control[2] - drag_r) / I_zz
        
        # Update velocities
        navigator.ship_state.u += dt * acc_u
        navigator.ship_state.v += dt * acc_v
        navigator.ship_state.r += dt * acc_r
        
        # Integrate position
        navigator.ship_state.integrate()
        
        # Store history
        history['x'].append(navigator.ship_state.x)
        history['y'].append(navigator.ship_state.y)
        history['psi'].append(navigator.ship_state.psi)
        history['u'].append(navigator.ship_state.u)
        history['path'].append(nav_state.path.copy() if nav_state.path is not None else None)
        history['setpoint'].append(nav_state.setpoint.copy())
        
        # Check if complete
        if nav_state.is_complete:
            print(f"Goal reached at step {step}!")
            break
        
        # Progress update
        if step % 100 == 0:
            print(f"Step {step}: pos=({navigator.ship_state.x:.1f}, {navigator.ship_state.y:.1f}), "
                  f"dist_to_goal={nav_state.distance_to_goal:.1f}m, "
                  f"speed={navigator.ship_state.speed:.2f}m/s")
    
    # Visualization
    if animate:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        def init():
            ax.set_xlim(0, config.map_width)
            ax.set_ylim(0, config.map_height)
            ax.set_aspect('equal')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            return []
        
        def animate_frame(frame):
            ax.clear()
            ax.set_xlim(0, config.map_width)
            ax.set_ylim(0, config.map_height)
            ax.set_aspect('equal')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_title(f'Autonomous Navigation (t={frame*dt:.1f}s)')
            
            # Draw ice floes
            ice_patches = []
            for i in range(len(positions)):
                poly_world = transform_polygon(polygons[i], positions[i, 0], positions[i, 1], angles[i])
                ice_patches.append(MplPolygon(poly_world, closed=True))
            
            ice_collection = PatchCollection(ice_patches, facecolor='lightblue', 
                                            edgecolor='blue', alpha=0.6)
            ax.add_collection(ice_collection)
            
            # Draw path
            if history['path'][frame] is not None:
                path = history['path'][frame]
                ax.plot(path[:, 0], path[:, 1], 'g--', linewidth=1.5, label='Planned path')
            
            # Draw trajectory so far
            ax.plot(history['x'][:frame+1], history['y'][:frame+1], 
                   'b-', linewidth=2, label='Trajectory')
            
            # Draw ship
            ship_poly = create_ship_polygon(
                history['x'][frame], history['y'][frame], history['psi'][frame],
                config.ship_length, config.ship_width
            )
            ax.add_patch(MplPolygon(ship_poly, closed=True, facecolor='red', 
                                   edgecolor='darkred', alpha=0.8))
            
            # Draw setpoint
            sp = history['setpoint'][frame]
            ax.plot(sp[0], sp[1], 'mo', markersize=8, label='Setpoint')
            
            # Draw goal
            ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=15, label='Goal')
            
            # Draw start
            ax.plot(start_pos[0], start_pos[1], 'bs', markersize=10, label='Start')
            
            ax.legend(loc='upper right')
            
            return []
        
        n_frames = len(history['x'])
        anim = animation.FuncAnimation(
            fig, animate_frame, init_func=init,
            frames=range(0, n_frames, 2),  # Skip every other frame for speed
            interval=50, blit=True
        )
        
        if save_animation:
            print("Saving animation...")
            anim.save('autonomous_navigation.gif', writer='pillow', fps=20)
            print("Saved to autonomous_navigation.gif")
        
        plt.show()
    
    return history


if __name__ == '__main__':
    run_simulation(
        n_steps=500,
        dt=0.1,
        n_floes=25,
        seed=42,
        animate=True,
        save_animation=False,
    )
