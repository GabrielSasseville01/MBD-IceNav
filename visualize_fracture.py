#!/usr/bin/env python3
"""
Visualization for DPDEM ice fracture simulation.

Shows ice elements connected by bonds drifting toward a ship.
Bonds break on impact, creating fracture patterns.
"""
print("Starting visualization script...", flush=True)

import numpy as np
print("Imported numpy", flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
print("Imported matplotlib", flush=True)

import importlib.util
import sys
import os

# Add current directory to path and import directly
sys.path.insert(0, os.getcwd())
print("Importing demo_fracture...", flush=True)

# Import simulation module directly
spec = importlib.util.spec_from_file_location(
    "demo_fracture", 
    "dpdem/demo_fracture.py"
)
demo_fracture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo_fracture)
print("Loaded demo_fracture", flush=True)

run_fracture_simulation = demo_fracture.run_fracture_simulation
create_ice_sheet = demo_fracture.create_ice_sheet


def animate_simulation(
    positions_history,
    bonds_history,
    bond_pairs,
    ship,
    element_radius,
    output_file='dpdem_fracture.gif',
    fps=20,
):
    """Create animation of ice fracture simulation."""
    
    n_frames = len(positions_history)
    n_particles = positions_history[0].shape[0]
    
    # Domain bounds based on initial positions and ship
    init_pos = positions_history[0]
    x_min = init_pos[:, 0].min() - 5
    x_max = float(ship.center[0]) + float(ship.length)/2 + 10
    y_min = init_pos[:, 1].min() - 5
    y_max = init_pos[:, 1].max() + 5
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors
    water_color = '#1A4D70'
    ice_color = '#B8D8E8'
    ice_edge = '#4080A0'
    bond_color = '#2060A0'
    broken_ice = '#E8E8E8'
    ship_color = '#505050'
    
    def get_bonded_particles(intact_mask):
        """Find which particles still have bonds."""
        bonded = set()
        for b, (i, j) in enumerate(bond_pairs):
            if intact_mask[b]:
                bonded.add(i)
                bonded.add(j)
        return bonded
    
    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_facecolor(water_color)
        
        positions = positions_history[frame]
        intact = bonds_history[frame]
        bonded_particles = get_bonded_particles(intact)
        
        # Draw bonds first (behind particles) - thick visible lines
        bond_lines = []
        for b, (i, j) in enumerate(bond_pairs):
            if intact[b]:
                p1, p2 = positions[i], positions[j]
                # LineCollection expects list of [(x1,y1), (x2,y2)]
                bond_lines.append([(p1[0], p1[1]), (p2[0], p2[1])])
        
        if bond_lines:
            lc = LineCollection(bond_lines, colors='#1040A0', 
                              linewidths=3.0, alpha=0.9, zorder=5)
            ax.add_collection(lc)
        
        # Draw particles
        for i in range(n_particles):
            x, y = positions[i]
            is_bonded = i in bonded_particles
            
            circle = Circle(
                (x, y), element_radius,
                facecolor=ice_color if is_bonded else broken_ice,
                edgecolor=ice_edge,
                linewidth=1.0,
                zorder=3
            )
            ax.add_patch(circle)
        
        # Draw ship (rectangle with rounded corners)
        ship_x = float(ship.center[0]) - float(ship.length)/2
        ship_y = float(ship.center[1]) - float(ship.beam)/2
        
        ship_rect = FancyBboxPatch(
            (ship_x, ship_y),
            float(ship.length), float(ship.beam),
            boxstyle="round,pad=0,rounding_size=2",
            facecolor=ship_color,
            edgecolor='#303030',
            linewidth=2,
            zorder=10
        )
        ax.add_patch(ship_rect)
        
        # Info text
        n_intact = int(intact.sum())
        n_total = len(intact)
        pct = 100 * n_intact / n_total if n_total > 0 else 0
        
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Y [m]', fontsize=11)
        ax.set_title(
            f'DPDEM Ice Fracture Simulation\n'
            f'Frame {frame}/{n_frames-1} | '
            f'Intact bonds: {n_intact}/{n_total} ({pct:.0f}%)',
            fontsize=13, fontweight='bold'
        )
        
        # Grid
        ax.grid(True, alpha=0.2, color='white', linestyle='--')
        
        return []
    
    print(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)
    
    print(f"Saving to {output_file}...")
    anim.save(output_file, writer=PillowWriter(fps=fps), dpi=100)
    print("Done!")
    
    plt.close()


def main():
    print("=" * 60, flush=True)
    print("DPDEM: Ice Fracture Simulation", flush=True)
    print("=" * 60, flush=True)
    
    # Simulation parameters
    n_x, n_y = 10, 6
    spacing = 3.0
    element_radius = 1.2
    
    print("Starting simulation...", flush=True)
    # Run simulation
    positions, bonds_intact, ship, radius = run_fracture_simulation(
        n_x=n_x,
        n_y=n_y,
        spacing=spacing,
        element_radius=element_radius,
        n_steps=1000,
        dt=0.01,
        drift_vel=1.0,
        save_interval=10,
    )
    print(f"Simulation done. {len(positions)} frames saved.", flush=True)
    
    print("Getting bond pairs...", flush=True)
    # Get bond pairs for visualization
    particles, bonds = create_ice_sheet(n_x, n_y, spacing, element_radius, 5.0, 5.0)
    bond_pairs = list(zip(
        np.array(bonds.particle_i),
        np.array(bonds.particle_j)
    ))
    print(f"Creating animation with {len(positions)} frames...", flush=True)
    
    # Create animation
    animate_simulation(
        positions_history=positions,
        bonds_history=bonds_intact,
        bond_pairs=bond_pairs,
        ship=ship,
        element_radius=radius,
        output_file='dpdem_fracture.gif',
        fps=20,
    )


if __name__ == '__main__':
    main()
