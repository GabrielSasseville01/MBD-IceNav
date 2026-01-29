#!/usr/bin/env python3
"""
DPDEM Ice-Ship Interaction - 3D Visualization

3D rendering of the ice-ship simulation showing:
- Ice floes as 3D extruded polygons with thickness
- Ship hull in 3D
- Water surface
- Ice tilting and movement in 3D space
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi

# Import the 2D simulation functions
from visualize_ice_ship import (
    create_voronoi_ice_field,
    simulate_ice_ship_interaction,
    polygon_area,
    polygon_centroid,
)


def create_ship_hull_3d(length, beam, draft, position):
    """Create 3D ship hull vertices and faces."""
    cx, cy = position
    cz = 0  # Waterline at z=0
    
    # Hull cross-section points (simplified)
    n_sections = 10
    hull_verts = []
    
    for i in range(n_sections + 1):
        t = i / n_sections  # 0 at stern, 1 at bow
        
        # X position along ship
        x = cx + length/2 - t * length
        
        # Taper at bow
        x_rel = -(x - cx) / (length/2)
        if x_rel > 0.6:
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
        else:
            taper = 1.0
        taper = max(taper, 0.05)
        
        local_beam = beam * taper
        local_draft = draft * taper
        
        # Cross-section vertices (simplified hull shape)
        # Deck level
        hull_verts.append([x, cy - local_beam/2, cz + draft * 0.3])
        hull_verts.append([x, cy + local_beam/2, cz + draft * 0.3])
        # Waterline
        hull_verts.append([x, cy - local_beam/2 * 0.9, cz])
        hull_verts.append([x, cy + local_beam/2 * 0.9, cz])
        # Keel
        hull_verts.append([x, cy, cz - local_draft])
    
    hull_verts = np.array(hull_verts)
    
    # Create faces
    faces = []
    verts_per_section = 5
    
    for i in range(n_sections):
        base = i * verts_per_section
        next_base = (i + 1) * verts_per_section
        
        # Port side (top)
        faces.append([base, base + 2, next_base + 2, next_base])
        # Starboard side (top)
        faces.append([base + 1, next_base + 1, next_base + 3, base + 3])
        # Port bottom
        faces.append([base + 2, base + 4, next_base + 4, next_base + 2])
        # Starboard bottom
        faces.append([base + 3, next_base + 3, next_base + 4, base + 4])
        # Deck
        faces.append([base, next_base, next_base + 1, base + 1])
    
    return hull_verts, faces


def polygon_to_3d_prism(poly_2d, z_bottom, z_top, center, rotation=0, tilt_x=0, tilt_y=0):
    """Convert 2D polygon to 3D extruded prism with rotation and tilt."""
    n = len(poly_2d)
    
    # Center the polygon
    centered = poly_2d - center
    
    # Apply 2D rotation
    c, s = np.cos(rotation), np.sin(rotation)
    rotated = np.zeros_like(centered)
    rotated[:, 0] = centered[:, 0] * c - centered[:, 1] * s
    rotated[:, 1] = centered[:, 0] * s + centered[:, 1] * c
    
    # Move back to world position
    world_2d = rotated + center
    
    # Create 3D vertices (top and bottom)
    verts = []
    for i in range(n):
        x, y = world_2d[i]
        # Apply tilt (rotation around x and y axes)
        dx = x - center[0]
        dy = y - center[1]
        
        # Bottom vertex
        z_bot = z_bottom + dx * np.sin(tilt_y) - dy * np.sin(tilt_x)
        verts.append([x, y, z_bot])
        
        # Top vertex
        z_tp = z_top + dx * np.sin(tilt_y) - dy * np.sin(tilt_x)
        verts.append([x, y, z_tp])
    
    verts = np.array(verts)
    
    # Create faces
    faces = []
    # Top face
    top_face = [2*i + 1 for i in range(n)]
    faces.append(top_face)
    # Bottom face
    bottom_face = [2*i for i in range(n-1, -1, -1)]
    faces.append(bottom_face)
    # Side faces
    for i in range(n):
        j = (i + 1) % n
        faces.append([2*i, 2*j, 2*j + 1, 2*i + 1])
    
    return verts, faces


def animate_ice_ship_3d(
    polygons, positions_history, rotations_history, bonds_history, bonds,
    ship_hull_verts, ship_hull_faces, ship_pos, initial_centroids,
    ice_thickness=0.5, output_file='dpdem_ice_ship_3d.gif', fps=15
):
    """Create 3D animation of ice-ship interaction."""
    
    n_frames = len(positions_history)
    n_elements = len(polygons)
    
    # Compute local polygons
    local_polys = []
    for poly in polygons:
        centroid = polygon_centroid(poly)
        local_polys.append(poly - centroid)
    
    # Domain bounds
    all_pos = np.concatenate(positions_history)
    x_min = min(all_pos[:, 0].min(), ship_hull_verts[:, 0].min()) - 10
    x_max = max(all_pos[:, 0].max(), ship_hull_verts[:, 0].max()) + 10
    y_min = min(all_pos[:, 1].min(), ship_hull_verts[:, 1].min()) - 10
    y_max = max(all_pos[:, 1].max(), ship_hull_verts[:, 1].max()) + 10
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors - more contrast
    water_color = '#0D3B5C'
    ice_color = '#E8F4F8'
    ice_edge_color = '#4A90A4'
    ship_color = '#2C2C2C'
    
    def update(frame):
        ax.clear()
        
        positions = positions_history[frame]
        rotations = rotations_history[frame]
        intact = bonds_history[frame]
        
        # Set view - tighter bounds on ice/ship area
        view_margin = 5
        ax.set_xlim(x_min + 20, x_max - 100)  # Focus on interaction area
        ax.set_ylim(y_min + 10, y_max - 10)
        ax.set_zlim(-4, 4)
        
        # Viewing angle - higher for better overview
        ax.view_init(elev=35, azim=-50 + frame * 0.15)
        
        # Draw water surface
        water_x = np.array([[x_min, x_max], [x_min, x_max]])
        water_y = np.array([[y_min, y_min], [y_max, y_max]])
        water_z = np.array([[-0.1, -0.1], [-0.1, -0.1]])
        ax.plot_surface(water_x, water_y, water_z, color=water_color, alpha=0.7, zorder=1)
        
        # Build set of broken bonds
        broken_neighbors = {i: set() for i in range(n_elements)}
        for b, (i, j) in enumerate(bonds):
            if not intact[b]:
                broken_neighbors[i].add(j)
                broken_neighbors[j].add(i)
        
        # Draw ice floes as 3D prisms
        for i in range(n_elements):
            # Rotate local polygon
            angle = rotations[i]
            c, s = np.cos(angle), np.sin(angle)
            rotated = np.zeros_like(local_polys[i])
            rotated[:, 0] = local_polys[i][:, 0] * c - local_polys[i][:, 1] * s
            rotated[:, 1] = local_polys[i][:, 0] * s + local_polys[i][:, 1] * c
            world_poly = rotated + positions[i]
            
            has_breaks = len(broken_neighbors[i]) > 0
            
            # Add some tilt for broken floes
            if has_breaks:
                tilt = np.random.uniform(-0.1, 0.1)
                z_offset = np.random.uniform(-0.1, 0.1)
            else:
                tilt = 0
                z_offset = 0
            
            # Create 3D vertices for ice floe
            n_verts = len(world_poly)
            z_bottom = -ice_thickness/2 + z_offset
            z_top = ice_thickness/2 + z_offset
            
            # Top and bottom faces
            top_verts = np.column_stack([world_poly, np.full(n_verts, z_top)])
            bottom_verts = np.column_stack([world_poly, np.full(n_verts, z_bottom)])
            
            # Choose color based on state
            if has_breaks:
                face_color = '#D0E8F0'  # Slightly different for broken
                edge_color = '#3080A0'
                edge_width = 1.0
            else:
                face_color = ice_color
                edge_color = '#A0C8D8'
                edge_width = 0.3
            
            # Draw top face
            ax.add_collection3d(Poly3DCollection(
                [top_verts],
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=0.95
            ))
            
            # Draw side faces (ice thickness visible)
            for j in range(n_verts):
                k = (j + 1) % n_verts
                side_verts = np.array([
                    bottom_verts[j],
                    bottom_verts[k],
                    top_verts[k],
                    top_verts[j]
                ])
                ax.add_collection3d(Poly3DCollection(
                    [side_verts],
                    facecolor='#C0D8E8',
                    edgecolor=edge_color,
                    linewidth=0.2,
                    alpha=0.85
                ))
        
        # Draw ship hull
        hull_faces_verts = [[ship_hull_verts[idx] for idx in face] for face in ship_hull_faces]
        ax.add_collection3d(Poly3DCollection(
            hull_faces_verts,
            facecolor=ship_color,
            edgecolor='#303030',
            linewidth=0.5,
            alpha=1.0,
            zorder=10
        ))
        
        # Labels
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        n_broken = np.sum(~intact)
        n_total = len(intact)
        pct = 100 * n_broken / n_total if n_total > 0 else 0
        
        ax.set_title(
            f'DPDEM 3D: Ice-Ship Interaction\n'
            f'Frame {frame}/{n_frames-1} | Broken: {n_broken}/{n_total} ({pct:.0f}%)',
            fontsize=12, fontweight='bold'
        )
        
        return []
    
    print(f"Creating 3D animation with {n_frames} frames...", flush=True)
    
    # Use fewer frames for 3D (slower to render)
    frame_skip = 3
    frames_to_render = list(range(0, n_frames, frame_skip))
    
    anim = FuncAnimation(fig, update, frames=frames_to_render, interval=1000//fps, blit=False)
    
    print(f"Saving to {output_file}...", flush=True)
    anim.save(output_file, writer=PillowWriter(fps=fps), dpi=80)
    print("Done!", flush=True)
    
    plt.close()


def main():
    print("=" * 60, flush=True)
    print("DPDEM 3D: Ice-Ship Interaction Visualization", flush=True)
    print("=" * 60, flush=True)
    
    # Ice field parameters
    n_elements = 80  # Fewer for faster 3D rendering
    field_width = 45.0
    field_height = 35.0
    ice_thickness = 1.5  # Exaggerated for visibility (paper uses 0.5m)
    
    # Ship parameters
    ship_length = 30.0
    ship_beam = 7.0
    ship_draft = 3.0
    ship_pos = np.array([field_width + 25, field_height / 2])
    
    # Create Voronoi ice field
    print("Generating Voronoi ice field...", flush=True)
    polygons, centroids, bonds = create_voronoi_ice_field(
        n_elements, field_width, field_height, seed=42
    )
    print(f"  Created {len(polygons)} ice elements with {len(bonds)} bonds", flush=True)
    
    # Create 3D ship hull
    print("Creating 3D ship hull...", flush=True)
    ship_hull_verts, ship_hull_faces = create_ship_hull_3d(
        ship_length, ship_beam, ship_draft, ship_pos
    )
    
    # Run simulation
    print("Running simulation...", flush=True)
    positions, rotations, bonds_intact = simulate_ice_ship_interaction(
        polygons=polygons,
        centroids=centroids,
        bonds=bonds,
        ship_pos=ship_pos,
        ship_length=ship_length,
        ship_beam=ship_beam,
        n_frames=250,
        dt=0.05,
        drift_vel=2.0,
    )
    
    n_final = bonds_intact[-1].sum()
    print(f"  Final intact bonds: {n_final}/{len(bonds)}", flush=True)
    
    # Create 3D animation
    animate_ice_ship_3d(
        polygons=polygons,
        positions_history=positions,
        rotations_history=rotations,
        bonds_history=bonds_intact,
        bonds=bonds,
        ship_hull_verts=ship_hull_verts,
        ship_hull_faces=ship_hull_faces,
        ship_pos=ship_pos,
        initial_centroids=centroids,
        ice_thickness=ice_thickness,
        output_file='dpdem_ice_ship_3d.gif',
        fps=12,
    )


if __name__ == '__main__':
    main()
