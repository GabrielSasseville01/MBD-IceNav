#!/usr/bin/env python3
"""
DPDEM Ice-Ship Interaction Visualization

Visualizes ice floes (Voronoi polygons) drifting toward and colliding with a ship.
Shows bond fracture when ice impacts the hull.

Based on Wang et al. (2024) - Dilated-polyhedron-based DEM for ice-structure interactions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch
from matplotlib.collections import PatchCollection, LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import sys
import importlib.util

# Import voronoi module directly to avoid broken dpdem/__init__.py
spec = importlib.util.spec_from_file_location(
    "voronoi", 
    "dpdem/geometry/voronoi.py"
)
voronoi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voronoi)

generate_voronoi_polygons = voronoi.generate_voronoi_polygons
compute_polygon_centroid = voronoi.compute_polygon_centroid


# ============================================================================
# Ice Floe Generation (Voronoi)
# ============================================================================

def create_ice_field(
    n_floes: int,
    field_width: float,
    field_height: float,
    seed: int = 42,
) -> tuple:
    """
    Create ice field using Voronoi tessellation.
    
    Returns:
        polygons: List of (N, 2) polygon vertex arrays
        centroids: (n_floes, 2) centroid positions
        bonds: List of (i, j) tuples for adjacent floes
    """
    from scipy.spatial import Voronoi
    
    np.random.seed(seed)
    
    # Generate seed points
    points = np.random.rand(n_floes, 2)
    points[:, 0] *= field_width
    points[:, 1] *= field_height
    
    # Add mirror points to bound all cells
    mirrors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            shifted = points.copy()
            shifted[:, 0] += dx * field_width
            shifted[:, 1] += dy * field_height
            mirrors.append(shifted)
    
    all_points = np.vstack([points] + mirrors)
    vor = Voronoi(all_points)
    
    # Extract polygons for original points
    polygons = []
    centroids = []
    
    for i in range(n_floes):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) < 3:
            continue
        
        vertices = vor.vertices[region]
        
        # Simple clip to domain
        clipped = []
        for v in vertices:
            cx = np.clip(v[0], 0, field_width)
            cy = np.clip(v[1], 0, field_height)
            clipped.append([cx, cy])
        clipped = np.array(clipped)
        
        if len(clipped) >= 3:
            # Order vertices counter-clockwise
            centroid = np.mean(clipped, axis=0)
            angles = np.arctan2(clipped[:, 1] - centroid[1], 
                              clipped[:, 0] - centroid[0])
            order = np.argsort(angles)
            clipped = clipped[order]
            
            polygons.append(clipped)
            centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # Find adjacent floes
    n = len(polygons)
    bonds = []
    bond_threshold = max(field_width, field_height) / np.sqrt(n_floes) * 1.5
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < bond_threshold:
                bonds.append((i, j))
    
    print(f"  Generated {n} Voronoi cells, {len(bonds)} bonds", flush=True)
    return polygons, centroids, bonds


# ============================================================================
# Ship Hull Generation
# ============================================================================

def create_ship_hull(
    length: float,
    beam: float,
    draft: float,
    position: np.ndarray,
    bow_direction: str = 'left',
) -> tuple:
    """
    Create a simple icebreaker hull shape.
    
    Args:
        bow_direction: 'left' or 'right' - which way the bow points
    
    Returns:
        hull_outline: (N, 2) array for 2D top-down view
    """
    cx, cy = position
    
    # 2D hull outline (top-down view) with bow taper
    n_pts = 30
    hull_x = []
    hull_y = []
    
    # Starboard side (positive y)
    for i in range(n_pts + 1):
        t = i / n_pts  # 0 to 1 from stern to bow
        
        if bow_direction == 'left':
            # Bow at negative x (left side)
            x = cx + length/2 - t * length
            x_rel = -(x - cx) / (length/2)  # Flip relative position
        else:
            # Bow at positive x (right side)
            x = cx - length/2 + t * length
            x_rel = (x - cx) / (length/2)
        
        # Taper at bow
        if x_rel > 0.6:  # Bow region
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
        else:
            taper = 1.0
        taper = max(taper, 0.02)
        
        hull_x.append(x)
        hull_y.append(cy + beam/2 * taper)
    
    # Port side (negative y)
    for i in range(n_pts, -1, -1):
        t = i / n_pts
        
        if bow_direction == 'left':
            x = cx + length/2 - t * length
            x_rel = -(x - cx) / (length/2)
        else:
            x = cx - length/2 + t * length
            x_rel = (x - cx) / (length/2)
        
        if x_rel > 0.6:
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
        else:
            taper = 1.0
        taper = max(taper, 0.02)
        
        hull_x.append(x)
        hull_y.append(cy - beam/2 * taper)
    
    hull_outline = np.column_stack([hull_x, hull_y])
    
    return hull_outline


# ============================================================================
# Physics Simulation
# ============================================================================

def simulate_ice_ship_interaction(
    polygons: list,
    centroids: np.ndarray,
    bonds: list,
    ship_pos: np.ndarray,
    ship_length: float,
    ship_beam: float,
    n_frames: int,
    dt: float,
    drift_vel: float,
) -> tuple:
    """
    Simulate ice drifting toward ship with bond fracture.
    
    Returns:
        positions_history: List of (n, 2) position arrays
        rotations_history: List of (n,) rotation angle arrays
        bonds_history: List of intact bond masks
    """
    n = len(polygons)
    
    # State
    positions = centroids.copy()
    velocities = np.zeros_like(positions)
    rotations = np.zeros(n)
    angular_vels = np.zeros(n)
    
    # Bond state
    bond_intact = np.ones(len(bonds), dtype=bool)
    
    # Compute floe masses and radii
    masses = np.array([compute_floe_area(p) * 0.5 * 920 for p in polygons])
    radii = np.array([np.sqrt(compute_floe_area(p) / np.pi) for p in polygons])
    
    # Bond properties
    bond_stiffness = 1e6
    bond_strength = 1e5
    
    positions_history = [positions.copy()]
    rotations_history = [rotations.copy()]
    bonds_history = [bond_intact.copy()]
    
    for frame in range(n_frames):
        forces = np.zeros_like(positions)
        
        # 1. Bond forces
        for b, (i, j) in enumerate(bonds):
            if not bond_intact[b]:
                continue
            
            d = positions[j] - positions[i]
            dist = np.linalg.norm(d)
            rest_len = np.linalg.norm(centroids[j] - centroids[i])
            
            if dist < 1e-6:
                continue
            
            direction = d / dist
            stretch = dist - rest_len
            force_mag = bond_stiffness * stretch
            
            # Check fracture
            if abs(force_mag) > bond_strength:
                bond_intact[b] = False
                continue
            
            forces[i] += force_mag * direction
            forces[j] -= force_mag * direction
        
        # 2. Ship collision
        ship_left = ship_pos[0] - ship_length/2
        ship_right = ship_pos[0] + ship_length/2
        ship_bottom = ship_pos[1] - ship_beam/2
        ship_top = ship_pos[1] + ship_beam/2
        
        for i in range(n):
            px, py = positions[i]
            r = radii[i]
            
            # Simple AABB collision
            if (px + r > ship_left and px - r < ship_right and
                py + r > ship_bottom and py - r < ship_top):
                
                # Find penetration direction
                dx_left = (px + r) - ship_left
                dx_right = ship_right - (px - r)
                dy_bottom = (py + r) - ship_bottom
                dy_top = ship_top - (py - r)
                
                pen = min(dx_left, dx_right, dy_bottom, dy_top)
                
                if pen > 0:
                    # Push away from closest edge
                    if pen == dx_left:
                        forces[i, 0] -= 5e6 * pen
                    elif pen == dx_right:
                        forces[i, 0] += 5e6 * pen
                    elif pen == dy_bottom:
                        forces[i, 1] -= 5e6 * pen
                    else:
                        forces[i, 1] += 5e6 * pen
        
        # 3. Inter-floe contact (simple)
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                overlap = radii[i] + radii[j] - dist
                
                if overlap > 0 and dist > 1e-6:
                    direction = d / dist
                    f = 1e5 * overlap ** 1.5 * direction
                    forces[i] -= f
                    forces[j] += f
        
        # 4. Drift force
        forces[:, 0] += drift_vel * masses * 5
        
        # 5. Damping (water resistance)
        forces -= 0.5 * velocities * masses[:, None]
        
        # 6. Integration
        accel = forces / masses[:, None]
        velocities += accel * dt
        velocities = np.clip(velocities, -20, 20)
        positions += velocities * dt
        
        # Small random rotation for visual interest
        angular_vels += (np.random.randn(n) * 0.01 - angular_vels * 0.1) * dt
        rotations += angular_vels * dt
        
        positions_history.append(positions.copy())
        rotations_history.append(rotations.copy())
        bonds_history.append(bond_intact.copy())
    
    return positions_history, rotations_history, bonds_history


def compute_floe_area(polygon):
    """Compute area of polygon."""
    n = len(polygon)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i, 0] * polygon[j, 1]
        area -= polygon[j, 0] * polygon[i, 1]
    return abs(area) / 2


# ============================================================================
# Visualization
# ============================================================================

def animate_ice_ship(
    polygons: list,
    positions_history: list,
    rotations_history: list,
    bonds_history: list,
    bonds: list,
    ship_outline: np.ndarray,
    ship_pos: np.ndarray,
    output_file: str = 'dpdem_ice_ship.gif',
    fps: int = 20,
):
    """Create animation of ice-ship interaction."""
    
    n_frames = len(positions_history)
    n_floes = len(polygons)
    
    # Compute local polygons (centered at origin)
    local_polys = []
    for poly in polygons:
        centroid = compute_polygon_centroid(poly)
        local_polys.append(poly - centroid)
    
    # Domain bounds
    all_pos = np.concatenate(positions_history)
    x_min = min(all_pos[:, 0].min(), ship_outline[:, 0].min()) - 5
    x_max = max(all_pos[:, 0].max(), ship_outline[:, 0].max()) + 5
    y_min = min(all_pos[:, 1].min(), ship_outline[:, 1].min()) - 5
    y_max = max(all_pos[:, 1].max(), ship_outline[:, 1].max()) + 5
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colors
    water_color = '#1A4D70'
    ice_color = '#C8E4F0'
    ice_edge = '#4080A0'
    broken_ice = '#E0E8EC'
    bond_color = '#2060A0'
    ship_color = '#404040'
    
    def get_bonded_floes(intact):
        bonded = set()
        for b, (i, j) in enumerate(bonds):
            if intact[b]:
                bonded.add(i)
                bonded.add(j)
        return bonded
    
    def rotate_polygon(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return poly @ R.T
    
    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_facecolor(water_color)
        
        positions = positions_history[frame]
        rotations = rotations_history[frame]
        intact = bonds_history[frame]
        bonded_floes = get_bonded_floes(intact)
        
        # Draw bonds
        bond_lines = []
        for b, (i, j) in enumerate(bonds):
            if intact[b]:
                p1, p2 = positions[i], positions[j]
                bond_lines.append([(p1[0], p1[1]), (p2[0], p2[1])])
        
        if bond_lines:
            lc = LineCollection(bond_lines, colors=bond_color,
                              linewidths=2.5, alpha=0.8, zorder=2)
            ax.add_collection(lc)
        
        # Draw ice floes as polygons
        for i in range(n_floes):
            # Rotate and translate polygon
            rotated = rotate_polygon(local_polys[i], rotations[i])
            world_poly = rotated + positions[i]
            
            is_bonded = i in bonded_floes
            color = ice_color if is_bonded else broken_ice
            
            patch = MplPolygon(
                world_poly,
                facecolor=color,
                edgecolor=ice_edge,
                linewidth=1.5,
                zorder=3,
                alpha=0.9
            )
            ax.add_patch(patch)
        
        # Draw ship
        ship_patch = MplPolygon(
            ship_outline,
            facecolor=ship_color,
            edgecolor='#202020',
            linewidth=2,
            zorder=10
        )
        ax.add_patch(ship_patch)
        
        # Stats
        n_intact = int(intact.sum())
        n_total = len(intact)
        pct = 100 * n_intact / n_total if n_total > 0 else 0
        
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Y [m]', fontsize=11)
        ax.set_title(
            f'DPDEM: Ice-Ship Interaction (Voronoi Ice Floes)\n'
            f'Frame {frame}/{n_frames-1} | '
            f'Intact bonds: {n_intact}/{n_total} ({pct:.0f}%)',
            fontsize=13, fontweight='bold'
        )
        ax.grid(True, alpha=0.2, color='white', linestyle='--')
        
        return []
    
    print(f"Creating animation with {n_frames} frames...", flush=True)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)
    
    print(f"Saving to {output_file}...", flush=True)
    anim.save(output_file, writer=PillowWriter(fps=fps), dpi=100)
    print("Done!", flush=True)
    
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60, flush=True)
    print("DPDEM: Ice-Ship Interaction with Voronoi Ice Floes", flush=True)
    print("=" * 60, flush=True)
    
    # Ice field parameters
    n_floes = 80
    field_width = 50.0
    field_height = 35.0
    
    # Ship parameters (similar to R/V Xuelong proportions)
    ship_length = 35.0
    ship_beam = 8.0
    ship_pos = np.array([field_width + ship_length/2 + 5, field_height / 2])
    
    # Create ice field
    print("Generating Voronoi ice field...", flush=True)
    polygons, centroids, bonds = create_ice_field(
        n_floes, field_width, field_height, seed=42
    )
    print(f"  Created {len(polygons)} ice floes with {len(bonds)} bonds", flush=True)
    
    # Create ship hull with bow facing left (toward incoming ice)
    ship_outline = create_ship_hull(ship_length, ship_beam, 3.0, ship_pos, bow_direction='left')
    
    # Run simulation
    print("Running simulation...", flush=True)
    positions, rotations, bonds_intact = simulate_ice_ship_interaction(
        polygons=polygons,
        centroids=centroids,
        bonds=bonds,
        ship_pos=ship_pos,
        ship_length=ship_length,
        ship_beam=ship_beam,
        n_frames=200,
        dt=0.05,
        drift_vel=0.8,
    )
    
    n_final = bonds_intact[-1].sum()
    print(f"  Final bonds: {n_final}/{len(bonds)}", flush=True)
    
    # Create animation
    animate_ice_ship(
        polygons=polygons,
        positions_history=positions,
        rotations_history=rotations,
        bonds_history=bonds_intact,
        bonds=bonds,
        ship_outline=ship_outline,
        ship_pos=ship_pos,
        output_file='dpdem_ice_ship.gif',
        fps=20,
    )


if __name__ == '__main__':
    main()
