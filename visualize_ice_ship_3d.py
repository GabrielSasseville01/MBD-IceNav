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
        # Deck level (about 3m above waterline for icebreaker)
        deck_height = 3.0 * taper
        hull_verts.append([x, cy - local_beam/2, cz + deck_height])
        hull_verts.append([x, cy + local_beam/2, cz + deck_height])
        # Waterline
        hull_verts.append([x, cy - local_beam/2 * 0.9, cz])
        hull_verts.append([x, cy + local_beam/2 * 0.9, cz])
        # Keel (draft below waterline)
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


def simulate_3d_ice_physics(positions_history, bonds_history, bonds, ship_pos, 
                            ship_length, ship_beam, n_elements):
    """
    Add 3D physics: vertical movement, tilting, and ice going under hull.
    Based on paper Fig. 17 showing ice bending, breaking, and accumulating under platform.
    """
    n_frames = len(positions_history)
    
    # Initialize 3D state
    z_positions = [np.zeros(n_elements) for _ in range(n_frames)]
    tilts_x = [np.zeros(n_elements) for _ in range(n_frames)]  # Pitch (forward tilt)
    tilts_y = [np.zeros(n_elements) for _ in range(n_frames)]  # Roll (sideways tilt)
    
    ship_bow_x = ship_pos[0] - ship_length / 2
    
    for frame in range(1, n_frames):
        positions = positions_history[frame]
        intact = bonds_history[frame]
        prev_z = z_positions[frame - 1].copy()
        prev_tilt_x = tilts_x[frame - 1].copy()
        prev_tilt_y = tilts_y[frame - 1].copy()
        
        # Check which elements have broken bonds
        broken_count = np.zeros(n_elements)
        for b, (i, j) in enumerate(bonds):
            if not intact[b]:
                broken_count[i] += 1
                broken_count[j] += 1
        
        new_z = prev_z.copy()
        new_tilt_x = prev_tilt_x.copy()
        new_tilt_y = prev_tilt_y.copy()
        
        for i in range(n_elements):
            px, py = positions[i]
            dist_to_bow = px - ship_bow_x
            rel_y = py - ship_pos[1]
            
            # Ice near ship bow gets pushed down and tilts
            if 0 < dist_to_bow < ship_length * 0.3:
                # Close to bow - ice bends down (like Fig 17a,b)
                push_factor = 1 - dist_to_bow / (ship_length * 0.3)
                
                if broken_count[i] > 0:
                    # Broken ice goes under the hull (Fig 17c,d)
                    # Draft is ~10.5m, so ice needs to go deeper
                    target_z = -8.0 - push_factor * 4.0
                    new_z[i] += (target_z - new_z[i]) * 0.1
                    
                    # Tilt forward (nose down) as it slides under
                    new_tilt_x[i] += (0.5 * push_factor - new_tilt_x[i]) * 0.1
                else:
                    # Intact ice bends at contact
                    new_tilt_x[i] += (0.3 * push_factor - new_tilt_x[i]) * 0.05
            
            elif dist_to_bow < 0:
                # Past the bow - ice is under/beside the hull
                if broken_count[i] > 0:
                    # Broken fragments accumulate under hull (ice cushion)
                    # Some float back up on sides
                    if abs(rel_y) > ship_beam / 2:
                        # Outside hull width - float back up
                        new_z[i] += (0 - new_z[i]) * 0.05
                        new_tilt_x[i] *= 0.95
                    else:
                        # Under hull - stay down at keel depth
                        new_z[i] = max(new_z[i], -12.0)  # Below 10.5m draft
                        new_z[i] += (-10.0 - new_z[i]) * 0.02
            
            # Damping
            new_tilt_x[i] *= 0.98
            new_tilt_y[i] *= 0.98
            
            # Buoyancy - ice wants to float
            if new_z[i] < -0.5 and dist_to_bow > ship_length * 0.5:
                new_z[i] += 0.05  # Slowly float back up
        
        z_positions[frame] = new_z
        tilts_x[frame] = new_tilt_x
        tilts_y[frame] = new_tilt_y
    
    return z_positions, tilts_x, tilts_y


def animate_ice_ship_3d(
    polygons, positions_history, rotations_history, bonds_history, bonds,
    ship_hull_verts, ship_hull_faces, ship_pos, ship_length, ship_beam,
    initial_centroids, ice_thickness=0.5, output_file='dpdem_ice_ship_3d.gif', fps=15
):
    """Create 3D animation of ice-ship interaction with vertical physics."""
    
    n_frames = len(positions_history)
    n_elements = len(polygons)
    
    # Compute 3D physics (vertical movement, tilting)
    print("Computing 3D ice physics...", flush=True)
    z_positions, tilts_x, tilts_y = simulate_3d_ice_physics(
        positions_history, bonds_history, bonds, ship_pos,
        ship_length, ship_beam, n_elements
    )
    
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
        z_pos = z_positions[frame]
        tilt_x = tilts_x[frame]
        tilt_y = tilts_y[frame]
        
        # Set view - show full ship and ice field
        # Ship is at x = field_width + 80 = 330, length 158m, so spans ~250 to 410
        ax.set_xlim(0, x_max + 50)  # Full scene
        ax.set_ylim(y_min - 10, y_max + 10)
        ax.set_zlim(-15, 10)  # Deeper for 10.5m draft
        
        # CRITICAL: Set proper aspect ratio so ship looks correct
        # X range ~500m, Y range ~200m, Z range 25m
        # Scale Z up visually so it's visible but proportional
        x_range = x_max + 50
        y_range = (y_max + 10) - (y_min - 10)
        z_range = 25
        # Exaggerate Z by 3x so underwater part visible, but ship still looks long
        ax.set_box_aspect([x_range, y_range, z_range * 3])
        
        # Viewing angle - higher elevation to see over water
        ax.view_init(elev=25, azim=-60 + frame * 0.1)
        
        # Draw water surface - very transparent so ship is visible
        # water_x = np.array([[x_min - 50, x_max + 100], [x_min - 50, x_max + 100]])
        # water_y = np.array([[y_min - 50, y_min - 50], [y_max + 50, y_max + 50]])
        # water_z = np.array([[0, 0], [0, 0]])
        # ax.plot_surface(water_x, water_y, water_z, color=water_color, alpha=0.2, zorder=1)
        
        # Build set of broken bonds
        broken_neighbors = {i: set() for i in range(n_elements)}
        for b, (i, j) in enumerate(bonds):
            if not intact[b]:
                broken_neighbors[i].add(j)
                broken_neighbors[j].add(i)
        
        # Draw ice floes as 3D prisms with tilt
        for i in range(n_elements):
            # Rotate local polygon
            angle = rotations[i]
            c, s = np.cos(angle), np.sin(angle)
            rotated = np.zeros_like(local_polys[i])
            rotated[:, 0] = local_polys[i][:, 0] * c - local_polys[i][:, 1] * s
            rotated[:, 1] = local_polys[i][:, 0] * s + local_polys[i][:, 1] * c
            world_poly = rotated + positions[i]
            
            has_breaks = len(broken_neighbors[i]) > 0
            
            # Get 3D position and orientation
            z_center = z_pos[i]
            tx, ty = tilt_x[i], tilt_y[i]
            
            # Create 3D vertices with tilt
            n_verts = len(world_poly)
            center_xy = positions[i]
            
            top_verts = []
            bottom_verts = []
            for v in world_poly:
                dx = v[0] - center_xy[0]
                dy = v[1] - center_xy[1]
                # Apply tilt
                z_offset = dx * np.sin(tx) + dy * np.sin(ty)
                top_verts.append([v[0], v[1], z_center + ice_thickness/2 + z_offset])
                bottom_verts.append([v[0], v[1], z_center - ice_thickness/2 + z_offset])
            
            top_verts = np.array(top_verts)
            bottom_verts = np.array(bottom_verts)
            
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
        
        # Draw ship hull with different colors for different parts
        # Colors for ship parts
        deck_color = '#8B4513'      # Brown deck
        hull_above = '#C0C0C0'      # Light gray above waterline
        hull_below = '#8B0000'      # Dark red below waterline (anti-fouling paint)
        
        for face_idx, face in enumerate(ship_hull_faces):
            face_verts = np.array([ship_hull_verts[idx] for idx in face])
            
            # Determine face type by checking z coordinates
            z_coords = face_verts[:, 2]
            avg_z = np.mean(z_coords)
            
            # Check if face is mostly horizontal (deck) or vertical (side)
            z_range = np.max(z_coords) - np.min(z_coords)
            is_horizontal = z_range < 1.0
            
            if is_horizontal and avg_z > 1.0:
                # Deck (top horizontal faces above waterline)
                color = deck_color
                edge = '#5D3A1A'
            elif avg_z > 0:
                # Hull above waterline
                color = hull_above
                edge = '#808080'
            else:
                # Hull below waterline
                color = hull_below
                edge = '#5C0000'
            
            ax.add_collection3d(Poly3DCollection(
                [face_verts],
                facecolor=color,
                edgecolor=edge,
                linewidth=0.8,
                alpha=0.95,
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
    
    # Ice field parameters - scaled for ship size
    field_width = 250.0
    field_height = 180.0
    n_elements = 150
    ice_thickness = 2.5  # Typical Arctic multi-year ice thickness
    
    # Ship parameters: CCGS John G. Diefenbaker (Polar Class Icebreaker)
    ship_length = 158.0  # m
    ship_beam = 28.0     # m  
    ship_draft = 10.5    # m (below waterline)
    # ship_depth = 13.5  # m (total hull depth, for reference)
    ship_pos = np.array([field_width + 80, field_height / 2])
    
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
        ship_length=ship_length,
        ship_beam=ship_beam,
        initial_centroids=centroids,
        ice_thickness=ice_thickness,
        output_file='dpdem_ice_ship_3d.gif',
        fps=12,
    )


if __name__ == '__main__':
    main()
