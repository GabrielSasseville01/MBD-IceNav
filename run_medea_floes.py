#!/usr/bin/env python3
"""
DPDEM Simulation with MEDEA Fram Strait Satellite Imagery

Uses ice floe data from AUTO-IceNav's MEDEA satellite dataset:
- Location: Arctic Ocean, Fram Strait (85°00'00"N 0°03'00"E)
- Date: 2010-06-29
- Source: Multi-satellite floe size distribution of Arctic sea ice 2000-2020 (Hwang et al. 2022)

This script:
1. Loads pre-extracted ice floe polygons from the MEDEA dataset
2. Converts them to DPDEM's 3D polyhedra representation
3. Visualizes the realistic ice field
4. Runs a simple drift animation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from pathlib import Path
import pickle

# MEDEA dataset parameters (from AUTO-IceNav)
# The original image is ~1651x782 pixels
# Based on typical MEDEA imagery, scale is approximately 1-5 meters per pixel
METERS_PER_PIXEL = 2.5  # Estimated scale factor
ICE_THICKNESS = 1.5     # Typical first-year Arctic ice thickness [m]


def load_medea_floes(pickle_path: str) -> list:
    """
    Load ice floe polygons from MEDEA dataset pickle file.
    
    Args:
        pickle_path: Path to ice_floes_MEDEA_fram_20100629.pkl
    
    Returns:
        List of floe dictionaries with 'vertices', 'area', 'centre', 'radius'
    """
    with open(pickle_path, 'rb') as f:
        floes = pickle.load(f)
    
    print(f"Loaded {len(floes)} ice floes from MEDEA dataset")
    
    # Print statistics
    areas = [f['area'] for f in floes]
    print(f"  Area range: {min(areas):.1f} - {max(areas):.1f} px²")
    print(f"  Mean area: {np.mean(areas):.1f} px²")
    
    return floes


def convert_to_world_coordinates(
    floes: list,
    meters_per_pixel: float,
    center_domain: bool = True,
) -> tuple:
    """
    Convert pixel coordinates to world coordinates (meters).
    
    Args:
        floes: List of floe dictionaries from pickle
        meters_per_pixel: Scale factor
        center_domain: If True, center the domain at origin
    
    Returns:
        Tuple of (polygons_list, metadata_dict)
    """
    polygons = []
    
    for floe in floes:
        vertices = np.array(floe['vertices']) * meters_per_pixel
        polygons.append(vertices)
    
    # Center the domain
    if center_domain and len(polygons) > 0:
        all_verts = np.vstack(polygons)
        center = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2
        polygons = [p - center for p in polygons]
        
        # Recompute bounds
        all_verts = np.vstack(polygons)
        x_range = all_verts[:, 0].max() - all_verts[:, 0].min()
        y_range = all_verts[:, 1].max() - all_verts[:, 1].min()
    else:
        all_verts = np.vstack(polygons) if polygons else np.zeros((1, 2))
        x_range = all_verts[:, 0].max() - all_verts[:, 0].min()
        y_range = all_verts[:, 1].max() - all_verts[:, 1].min()
    
    # Compute areas in m²
    areas_m2 = []
    for p in polygons:
        x, y = p[:, 0], p[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas_m2.append(area)
    
    metadata = {
        'n_floes': len(polygons),
        'domain_size': (x_range, y_range),
        'total_ice_area': sum(areas_m2),
        'mean_floe_area': np.mean(areas_m2) if areas_m2 else 0,
        'meters_per_pixel': meters_per_pixel,
        'source': 'MEDEA Fram Strait 2010-06-29',
        'location': '85°00\'00"N 0°03\'00"E',
    }
    
    return polygons, metadata


def visualize_medea_ice_field(
    polygons: list,
    metadata: dict,
    output_path: str = 'medea_ice_field.png',
    ice_thickness: float = 1.5,
):
    """
    Create a visualization of the MEDEA ice field.
    """
    if len(polygons) == 0:
        print("No polygons to visualize")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Compute floe areas for coloring
    areas = []
    for p in polygons:
        x, y = p[:, 0], p[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas.append(area)
    
    areas = np.array(areas)
    area_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1)
    
    # Plot 1: Top-down view of ice field
    ax1 = axes[0]
    ax1.set_facecolor('#1A4D70')  # Ocean blue
    
    # Color by area (larger = darker blue)
    colors = plt.cm.Blues(0.4 + 0.5 * area_norm)
    
    for i, poly in enumerate(polygons):
        patch = MplPolygon(
            poly,
            facecolor=colors[i],
            edgecolor='#2A5D80',
            linewidth=0.3,
            alpha=0.95,
        )
        ax1.add_patch(patch)
    
    all_verts = np.vstack(polygons)
    margin = 50
    ax1.set_xlim(all_verts[:, 0].min() - margin, all_verts[:, 0].max() + margin)
    ax1.set_ylim(all_verts[:, 1].min() - margin, all_verts[:, 1].max() + margin)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X [m]', fontsize=11)
    ax1.set_ylabel('Y [m]', fontsize=11)
    ax1.set_title(
        f'MEDEA Satellite Ice Field - Fram Strait\n'
        f'{metadata["n_floes"]} floes | {metadata["domain_size"][0]:.0f} x {metadata["domain_size"][1]:.0f} m',
        fontsize=13, fontweight='bold'
    )
    ax1.grid(True, alpha=0.2, color='white', linestyle='--')
    
    # Plot 2: Floe size distribution
    ax2 = axes[1]
    ax2.hist(areas, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(np.mean(areas), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(areas):.0f} m²')
    ax2.axvline(np.median(areas), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(areas):.0f} m²')
    ax2.set_xlabel('Floe Area [m²]', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Floe Size Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    
    # Add text box with statistics
    textstr = '\n'.join([
        f'Location: {metadata["location"]}',
        f'Source: {metadata["source"]}',
        f'Ice thickness: {ice_thickness} m',
        f'Total floes: {metadata["n_floes"]}',
        f'Total ice area: {metadata["total_ice_area"]/1e6:.2f} km²',
        f'Scale: {metadata["meters_per_pixel"]} m/px',
    ])
    ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved ice field visualization to {output_path}")
    plt.close()


def visualize_3d_ice_field(
    polygons: list,
    ice_thickness: float,
    output_path: str = 'medea_ice_field_3d.png',
    max_floes: int = 200,
):
    """
    Create a 3D visualization of the ice field.
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    if len(polygons) == 0:
        print("No polygons for 3D visualization")
        return
    
    # Limit number of floes for performance
    if len(polygons) > max_floes:
        # Sample larger floes
        areas = []
        for p in polygons:
            x, y = p[:, 0], p[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas.append(area)
        
        indices = np.argsort(areas)[-max_floes:]
        polygons = [polygons[i] for i in indices]
        print(f"  Showing top {max_floes} largest floes for 3D view")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by size
    areas = []
    for p in polygons:
        x, y = p[:, 0], p[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas.append(area)
    
    areas = np.array(areas)
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(polygons)))
    
    z_bottom = -ice_thickness * 0.3  # 30% submerged
    z_top = ice_thickness * 0.7      # 70% above water
    
    for i, polygon in enumerate(polygons):
        n = len(polygon)
        if n < 3:
            continue
        
        # Create 3D prism
        bottom = np.column_stack([polygon, np.full(n, z_bottom)])
        top = np.column_stack([polygon, np.full(n, z_top)])
        
        # Draw top face
        ax.add_collection3d(Poly3DCollection(
            [list(top)], alpha=0.9, facecolor=colors[i],
            edgecolor='darkblue', linewidth=0.3
        ))
        
        # Draw side faces
        for j in range(n):
            j_next = (j + 1) % n
            side = np.array([
                bottom[j], bottom[j_next], top[j_next], top[j]
            ])
            ax.add_collection3d(Poly3DCollection(
                [side], alpha=0.7, facecolor=colors[i],
                edgecolor='darkblue', linewidth=0.2
            ))
    
    # Draw water surface
    all_verts = np.vstack(polygons)
    margin = 100
    x_min, y_min = all_verts.min(axis=0) - margin
    x_max, y_max = all_verts.max(axis=0) + margin
    
    water = [[(x_min, y_min, 0), (x_max, y_min, 0), 
              (x_max, y_max, 0), (x_min, y_max, 0)]]
    ax.add_collection3d(Poly3DCollection(
        water, alpha=0.3, facecolor='steelblue', edgecolor='none'
    ))
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(-ice_thickness * 2, ice_thickness * 2)
    
    ax.set_xlabel('X [m]', fontsize=11)
    ax.set_ylabel('Y [m]', fontsize=11)
    ax.set_zlabel('Z [m]', fontsize=11)
    ax.set_title(
        f'MEDEA Fram Strait Ice Field - 3D View\n'
        f'{len(polygons)} floes | thickness={ice_thickness}m',
        fontsize=13, fontweight='bold'
    )
    
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved 3D visualization to {output_path}")
    plt.close()


def animate_ice_drift(
    polygons: list,
    ice_thickness: float,
    output_path: str = 'medea_ice_drift.gif',
    n_frames: int = 300,
    fps: int = 20,
    max_floes: int = 150,
):
    """
    Create an animation of ice floes drifting with ocean current.
    """
    if len(polygons) == 0:
        print("No polygons to animate")
        return
    
    # Limit floes for performance
    if len(polygons) > max_floes:
        areas = []
        for p in polygons:
            x, y = p[:, 0], p[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas.append(area)
        
        indices = np.argsort(areas)[-max_floes:]
        polygons = [polygons[i] for i in indices]
        print(f"  Animating top {max_floes} largest floes")
    
    n = len(polygons)
    
    # Compute centroids and properties
    centroids = []
    areas = []
    local_polys = []
    
    for poly in polygons:
        cx, cy = np.mean(poly, axis=0)
        centroids.append([cx, cy])
        local_polys.append(poly - [cx, cy])
        
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas.append(area)
    
    centroids = np.array(centroids)
    areas = np.array(areas)
    radii = np.sqrt(areas / np.pi)
    masses = areas * ice_thickness * 920  # ice density
    
    # Physics state
    positions = centroids.copy()
    velocities = np.random.randn(n, 2) * 0.2
    rotations = np.zeros(n)
    angular_vels = np.random.randn(n) * 0.01
    
    # Ocean current (simulating Fram Strait currents)
    # East Greenland Current flows southward
    drift = np.array([0.5, -1.5])  # m/s approximate
    
    # Simulate
    dt = 0.1
    positions_history = [positions.copy()]
    rotations_history = [rotations.copy()]
    
    print(f"  Simulating {n_frames} frames...")
    
    for frame in range(n_frames):
        forces = np.zeros_like(positions)
        
        # Ocean current drag
        forces += drift * masses[:, None] * 0.1
        
        # Water drag
        forces -= 0.05 * velocities * masses[:, None]
        
        # Simple collision avoidance
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 0.1:
                    overlap = min_dist - dist
                    direction = d / dist
                    f = 1000 * overlap ** 1.5 * direction
                    forces[i] -= f
                    forces[j] += f
        
        # Update dynamics
        accelerations = forces / masses[:, None]
        velocities += accelerations * dt
        velocities = np.clip(velocities, -3, 3)
        positions += velocities * dt
        
        # Rotations
        angular_vels += np.random.randn(n) * 0.002
        angular_vels *= 0.99
        rotations += angular_vels * dt
        
        positions_history.append(positions.copy())
        rotations_history.append(rotations.copy())
    
    # Create animation
    fig, ax = plt.subplots(figsize=(14, 10))
    
    all_pos = np.concatenate(positions_history)
    x_min = all_pos[:, 0].min() - 50
    x_max = all_pos[:, 0].max() + 50
    y_min = all_pos[:, 1].min() - 50
    y_max = all_pos[:, 1].max() + 50
    
    # Colors by floe size
    area_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1)
    colors = plt.cm.Blues(0.4 + 0.5 * area_norm)
    
    def rotate_poly(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return poly @ R.T
    
    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_facecolor('#1A4D70')
        
        pos = positions_history[frame]
        rots = rotations_history[frame]
        
        for i in range(n):
            rotated = rotate_poly(local_polys[i], rots[i])
            world_poly = rotated + pos[i]
            
            patch = MplPolygon(
                world_poly,
                facecolor=colors[i],
                edgecolor='#2A5D80',
                linewidth=0.5,
                alpha=0.95,
            )
            ax.add_patch(patch)
        
        # Draw current arrow
        arrow_x = x_min + 80
        arrow_y = y_max - 80
        ax.annotate('', xy=(arrow_x + drift[0]*30, arrow_y + drift[1]*30),
                   xytext=(arrow_x, arrow_y),
                   arrowprops=dict(arrowstyle='->', color='yellow', lw=2))
        ax.text(arrow_x, arrow_y + 20, 'Current', color='yellow', fontsize=10)
        
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Y [m]', fontsize=11)
        ax.set_title(
            f'MEDEA Fram Strait Ice Drift Simulation\n'
            f'Frame {frame}/{n_frames} | {n} floes | Location: 85°N 0°E',
            fontsize=13, fontweight='bold'
        )
        ax.grid(True, alpha=0.15, color='white', linestyle='--')
        
        return []
    
    print(f"  Creating animation...")
    anim = FuncAnimation(fig, update, frames=n_frames + 1, interval=1000//fps, blit=False)
    
    print(f"  Saving to {output_path}...")
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=100)
    print(f"  Done!")
    
    plt.close()


def main():
    print("=" * 70)
    print("DPDEM: MEDEA Fram Strait Satellite Ice Field")
    print("=" * 70)
    print()
    print("Location: Arctic Ocean, Fram Strait (85°00'00\"N 0°03'00\"E)")
    print("Date: 2010-06-29")
    print("Source: MEDEA satellite - Multi-satellite floe size distribution")
    print("        (Hwang et al. 2022)")
    print()
    
    # Paths
    data_dir = Path(__file__).parent / 'data'
    pickle_path = data_dir / 'ice_floes_MEDEA_fram_20100629.pkl'
    
    if not pickle_path.exists():
        print(f"ERROR: Pickle file not found: {pickle_path}")
        print("Please download it first with:")
        print("  curl -L 'https://drive.google.com/uc?export=download&id=1aJJdClZHxwOFd7wFO7PW8CDcQ-hsF4f5' -o data/ice_floes_MEDEA_fram_20100629.pkl")
        return
    
    # Load MEDEA floe data
    print("Loading MEDEA ice floe data...")
    floes = load_medea_floes(pickle_path)
    
    # Convert to world coordinates
    print(f"\nConverting to world coordinates (scale: {METERS_PER_PIXEL} m/px)...")
    polygons, metadata = convert_to_world_coordinates(
        floes, 
        meters_per_pixel=METERS_PER_PIXEL,
        center_domain=True
    )
    
    print(f"\n{'='*50}")
    print("Ice Field Statistics:")
    print(f"{'='*50}")
    print(f"  Number of floes: {metadata['n_floes']}")
    print(f"  Domain size: {metadata['domain_size'][0]:.0f} x {metadata['domain_size'][1]:.0f} m")
    print(f"  Total ice area: {metadata['total_ice_area']/1e6:.3f} km²")
    print(f"  Mean floe area: {metadata['mean_floe_area']:.0f} m²")
    print(f"  Ice thickness: {ICE_THICKNESS} m")
    print(f"  Total ice volume: {metadata['total_ice_area'] * ICE_THICKNESS / 1e6:.3f} km³")
    
    # Create visualizations
    print("\n" + "=" * 50)
    print("Generating visualizations...")
    print("=" * 50)
    
    print("\n1. Top-down ice field view...")
    visualize_medea_ice_field(
        polygons, metadata,
        output_path='medea_ice_field.png',
        ice_thickness=ICE_THICKNESS
    )
    
    print("\n2. 3D ice field view...")
    visualize_3d_ice_field(
        polygons,
        ice_thickness=ICE_THICKNESS,
        output_path='medea_ice_field_3d.png',
        max_floes=200
    )
    
    print("\n3. Ice drift animation...")
    animate_ice_drift(
        polygons,
        ice_thickness=ICE_THICKNESS,
        output_path='medea_ice_drift.gif',
        n_frames=300,
        fps=20,
        max_floes=150
    )
    
    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)
    print("\nOutput files:")
    print("  - medea_ice_field.png    : Top-down view with statistics")
    print("  - medea_ice_field_3d.png : 3D visualization")
    print("  - medea_ice_drift.gif    : Ice drift animation")
    
    print("\nTo use with DPDEM simulation:")
    print("-" * 50)
    print("""
from dpdem.geometry.image_floes import load_floes_from_pickle, apply_scale_calibration
from dpdem.simulation import create_ice_field_from_polygons

# Load MEDEA floes
floes = load_floes_from_pickle('data/ice_floes_MEDEA_fram_20100629.pkl')
polygons = apply_scale_calibration(floes, meters_per_pixel=2.5, center_domain=True)

# Create DPDEM ice field
polyhedra, particles = create_ice_field_from_polygons(
    polygons,
    ice_thickness=1.5,
)
print(f"Created {len(polyhedra)} 3D ice floes for simulation")
""")


if __name__ == '__main__':
    main()
