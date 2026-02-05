#!/usr/bin/env python3
"""
DPDEM: MEDEA Satellite Ice Floes with Ship Fracture Simulation

Combines real satellite-derived ice floe shapes from MEDEA Fram Strait
with the same physics model used in visualize_ice_ship.py.

Usage:
    python run_medea_ship_fracture.py           # Full run with animations
    python run_medea_ship_fracture.py --quick   # Simulation only, no animations
    python run_medea_ship_fracture.py --no-3d   # Skip 3D animation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import pickle
import time
import argparse


# ============================================================================
# Ice Material Properties (same as visualize_ice_ship.py)
# ============================================================================

class IceProperties:
    """Physical properties of sea ice."""
    ICE_DENSITY = 920  # kg/m³
    WATER_DENSITY = 1025  # kg/m³
    FLEXURAL_STRENGTH_BASE = 0.3e6  # Pa (reduced for visible fracturing)
    YOUNGS_MODULUS = 5e9  # Pa
    
    @staticmethod
    def flexural_strength(thickness):
        h_ref = 1.0
        return IceProperties.FLEXURAL_STRENGTH_BASE * np.sqrt(thickness / h_ref)
    
    @staticmethod
    def breaking_force(thickness, width):
        """Force required to break ice in bending."""
        sigma_f = IceProperties.flexural_strength(thickness)
        return sigma_f * width * thickness ** 2


class ShipProperties:
    """Physical properties of the ship."""
    def __init__(self, length, beam, draft, displacement=None):
        self.length = length
        self.beam = beam
        self.draft = draft
        
        if displacement is None:
            block_coef = 0.6
            self.displacement = length * beam * draft * block_coef * IceProperties.WATER_DENSITY
        else:
            self.displacement = displacement
        
        self.mass = self.displacement
    
    def impact_force(self, velocity, ice_thickness):
        """Estimate ice impact force."""
        C_force = 5.0
        return C_force * (self.mass ** 0.7) * (velocity ** 0.8) * (ice_thickness ** 0.5)


# ============================================================================
# Data Loading
# ============================================================================

def load_medea_floes(pickle_path: str) -> tuple:
    """Load MEDEA ice floe data from pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Data is a list of floe dicts with 'vertices', 'area', 'centre', 'radius'
    polygons = [np.array(floe['vertices']) for floe in data]
    centroids = np.array([np.array(floe['centre']) for floe in data])
    areas = np.array([floe['area'] for floe in data])
    
    metadata = {
        'n_floes': len(polygons),
        'areas': areas,
        'source': 'MEDEA',
    }
    
    return polygons, centroids, metadata


def scale_floes_to_world(polygons, centroids, meters_per_pixel, center_domain=True, 
                         subsample_region=None):
    """Convert pixel coordinates to world coordinates."""
    scaled_polygons = [p * meters_per_pixel for p in polygons]
    scaled_centroids = centroids * meters_per_pixel
    
    if center_domain:
        all_verts = np.vstack(scaled_polygons)
        center = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2
        scaled_polygons = [p - center for p in scaled_polygons]
        scaled_centroids = scaled_centroids - center
    
    # Optional: filter to a smaller region
    if subsample_region is not None:
        x_min, x_max, y_min, y_max = subsample_region
        keep = []
        for i, c in enumerate(scaled_centroids):
            if x_min <= c[0] <= x_max and y_min <= c[1] <= y_max:
                keep.append(i)
        if len(keep) > 0:
            scaled_polygons = [scaled_polygons[i] for i in keep]
            scaled_centroids = scaled_centroids[keep]
    
    return scaled_polygons, scaled_centroids


# ============================================================================
# Bond Creation
# ============================================================================

def create_proximity_bonds(centroids, radii, threshold_factor=1.5):
    """Create bonds between nearby floes."""
    bonds = []
    n = len(centroids)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            threshold = (radii[i] + radii[j]) * threshold_factor
            if dist < threshold:
                bonds.append((i, j))
    
    return bonds


def assign_thickness_by_area(areas, min_thickness=0.5, max_thickness=3.0, power=0.3):
    """Assign ice thickness correlated with floe area."""
    area_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1)
    thickness_norm = area_norm ** power
    thicknesses = min_thickness + thickness_norm * (max_thickness - min_thickness)
    thicknesses += np.random.randn(len(areas)) * 0.1
    return np.clip(thicknesses, min_thickness, max_thickness)


# ============================================================================
# Geometry Utilities
# ============================================================================

def polygon_area(verts):
    """Compute area using shoelace formula."""
    n = len(verts)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i, 0] * verts[j, 1]
        area -= verts[j, 0] * verts[i, 1]
    return abs(area) / 2


def polygon_centroid(verts):
    return np.mean(verts, axis=0)


def create_ship_hull(length, beam, position, bow_direction='left'):
    """Create ship hull outline."""
    cx, cy = position
    n_pts = 30
    hull_x, hull_y = [], []
    
    for i in range(n_pts + 1):
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
        hull_y.append(cy + beam/2 * taper)
    
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
    
    return np.column_stack([hull_x, hull_y])


# ============================================================================
# Internal Tessellation (convert large floes to small cells)
# ============================================================================

def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside


def clip_cell_to_floe(cell_verts, floe_polygon):
    """
    Clip a Voronoi cell to the floe boundary using shapely.
    
    Args:
        cell_verts: Nx2 array of cell polygon vertices
        floe_polygon: Mx2 array of floe boundary vertices
    
    Returns:
        Clipped polygon vertices as Nx2 array, or None if clipping fails
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    
    try:
        cell = ShapelyPolygon(cell_verts)
        floe = ShapelyPolygon(floe_polygon)
        
        # Fix invalid polygons (self-intersections, etc.)
        if not cell.is_valid:
            cell = cell.buffer(0)
        if not floe.is_valid:
            floe = floe.buffer(0)
        
        # Compute intersection
        clipped = cell.intersection(floe)
        
        if clipped.is_empty:
            return None
        
        # Handle MultiPolygon result (take largest piece)
        if clipped.geom_type == 'MultiPolygon':
            clipped = max(clipped.geoms, key=lambda p: p.area)
        
        # Extract coordinates from valid polygon
        if clipped.geom_type == 'Polygon' and not clipped.is_empty:
            coords = np.array(clipped.exterior.coords[:-1])  # Remove duplicate closing vertex
            if len(coords) >= 3:
                return coords
        
        return None
    except Exception:
        return None


def tessellate_floe_internally(floe_polygon, cells_per_100m2=0.5, min_cells=8, seed=None):
    """
    Tessellate a large satellite floe into small Voronoi cells.
    
    Uses DENSITY-BASED cell count: cells_per_100m2 controls how many cells
    per 100 square meters of floe area.
    
    Args:
        floe_polygon: Nx2 array of polygon vertices
        cells_per_100m2: Number of cells per 100 m² of floe area
        min_cells: Minimum number of cells per floe
        seed: Random seed for reproducibility
    
    Returns:
        List of (cell_polygon, centroid) tuples
    """
    from scipy.spatial import Voronoi
    
    if seed is not None:
        np.random.seed(seed)
    
    # Compute floe properties
    min_xy = floe_polygon.min(axis=0)
    max_xy = floe_polygon.max(axis=0)
    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]
    floe_area = polygon_area(floe_polygon)
    
    # DENSITY-BASED cell count: scale with area
    n_cells = max(min_cells, int(floe_area / 100.0 * cells_per_100m2))
    n_cells = min(n_cells, 150)  # Cap for performance
    
    # Use REJECTION SAMPLING primarily (handles irregular shapes better)
    # Start with some grid points to seed the distribution
    points = []
    
    # Try grid-based for initial coverage
    aspect = max(0.1, width / max(height, 1))
    ny = max(2, int(np.sqrt(n_cells / aspect)))
    nx = max(2, int(n_cells / ny))
    dx = width / nx
    dy = height / ny
    
    for i in range(nx):
        for j in range(ny):
            x = min_xy[0] + (i + 0.5) * dx + np.random.uniform(-0.3 * dx, 0.3 * dx)
            y = min_xy[1] + (j + 0.5) * dy + np.random.uniform(-0.3 * dy, 0.3 * dy)
            if point_in_polygon((x, y), floe_polygon):
                points.append([x, y])
    
    # Fill remaining with rejection sampling (more robust for irregular shapes)
    # Use minimum separation based on sqrt(area/n_cells)
    target_spacing = np.sqrt(floe_area / max(n_cells, 1)) * 0.5
    max_attempts = n_cells * 100  # More attempts for irregular shapes
    attempts = 0
    
    while len(points) < n_cells and attempts < max_attempts:
        x = np.random.uniform(min_xy[0], max_xy[0])
        y = np.random.uniform(min_xy[1], max_xy[1])
        
        if point_in_polygon((x, y), floe_polygon):
            # Check not too close to existing points
            too_close = False
            for px, py in points:
                if (x - px)**2 + (y - py)**2 < target_spacing**2:
                    too_close = True
                    break
            if not too_close:
                points.append([x, y])
        attempts += 1
    
    # If still too few, relax spacing constraint
    if len(points) < min_cells // 2:
        attempts = 0
        while len(points) < min_cells and attempts < max_attempts:
            x = np.random.uniform(min_xy[0], max_xy[0])
            y = np.random.uniform(min_xy[1], max_xy[1])
            if point_in_polygon((x, y), floe_polygon):
                points.append([x, y])
            attempts += 1
    
    if len(points) < 3:
        centroid = polygon_centroid(floe_polygon)
        return [(floe_polygon.copy(), centroid)]
    
    points = np.array(points)
    
    # Add mirror points for bounded Voronoi
    margin = max(width, height) * 2
    mirrors = []
    for dx_m, dy_m in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        shifted = points.copy()
        shifted[:, 0] += dx_m * margin
        shifted[:, 1] += dy_m * margin
        mirrors.append(shifted)
    
    all_points = np.vstack([points] + mirrors)
    
    try:
        vor = Voronoi(all_points)
    except Exception:
        centroid = polygon_centroid(floe_polygon)
        return [(floe_polygon.copy(), centroid)]
    
    cells = []
    
    for i in range(len(points)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) < 3:
            continue
        
        cell_verts = np.array([vor.vertices[v] for v in region])
        
        # PROPER POLYGON CLIPPING: clip cell to actual floe boundary
        clipped_verts = clip_cell_to_floe(cell_verts, floe_polygon)
        
        if clipped_verts is None or len(clipped_verts) < 3:
            continue
        
        # Compute centroid from the clipped polygon for accurate physics
        cell_centroid = polygon_centroid(clipped_verts)
        cells.append((clipped_verts, cell_centroid))
    
    if len(cells) == 0:
        centroid = polygon_centroid(floe_polygon)
        return [(floe_polygon.copy(), centroid)]
    
    return cells


def tessellate_all_floes(polygons, centroids, cells_per_100m2=0.5, seed=42):
    """
    Tessellate all satellite floes into small cells.
    
    Args:
        polygons: List of floe polygon arrays
        centroids: Floe centroids (not used, kept for API compatibility)
        cells_per_100m2: Cell density - number of cells per 100 m²
        seed: Random seed
    
    Returns:
        cell_polygons: List of Nx2 arrays
        cell_centroids: Nx2 array of centroids
        floe_ids: Array mapping each cell to its parent floe
    """
    all_cells = []
    all_centroids = []
    floe_ids = []
    
    np.random.seed(seed)
    
    for floe_idx, poly in enumerate(polygons):
        floe_area = polygon_area(poly)
        cells = tessellate_floe_internally(poly, cells_per_100m2=cells_per_100m2, seed=None)
        
        print(f"    Floe {floe_idx}: area={floe_area:.0f}m², cells={len(cells)}", flush=True)
        
        for cell_poly, cell_centroid in cells:
            all_cells.append(cell_poly)
            all_centroids.append(cell_centroid)
            floe_ids.append(floe_idx)
    
    return all_cells, np.array(all_centroids), np.array(floe_ids)


# ============================================================================
# Physics Simulation (same as visualize_ice_ship.py)
# ============================================================================

def find_connected_components(n, bonds, intact):
    """Find connected components of bonded elements."""
    adj = {i: [] for i in range(n)}
    for b, (i, j) in enumerate(bonds):
        if intact[b]:
            adj[i].append(j)
            adj[j].append(i)
    
    visited = set()
    components = []
    
    for start in range(n):
        if start in visited:
            continue
        component = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)
    
    return components


def point_to_ship_collision(px, py, ship_pos, ship_length, ship_beam, bow_taper_start=0.6):
    """Check collision with ship hull."""
    cx, cy = ship_pos
    half_len = ship_length / 2
    half_beam = ship_beam / 2
    
    rel_x = px - cx
    rel_y = py - cy
    x_rel = rel_x / half_len
    
    if x_rel < -bow_taper_start:
        t = (-x_rel - bow_taper_start) / (1 - bow_taper_start)
        t = min(t, 1.0)
        local_half_beam = half_beam * (1 - 0.9 * t ** 1.5)
        local_half_beam = max(local_half_beam, half_beam * 0.05)
    else:
        local_half_beam = half_beam
    
    if rel_x < -half_len or rel_x > half_len:
        return 0, 0, 0
    
    if abs(rel_y) > local_half_beam:
        return 0, 0, 0
    
    dist_to_bow = rel_x - (-half_len)
    dist_to_side = local_half_beam - abs(rel_y)
    
    if x_rel < -bow_taper_start:
        t = (-x_rel - bow_taper_start) / (1 - bow_taper_start)
        t = min(t, 1.0)
        bow_angle = 0.3 + 0.5 * t
        
        if rel_y >= 0:
            nx = -np.cos(bow_angle)
            ny = np.sin(bow_angle)
        else:
            nx = -np.cos(bow_angle)
            ny = -np.sin(bow_angle)
        
        pen = min(dist_to_bow, dist_to_side) + 0.5
    else:
        if dist_to_bow < dist_to_side:
            nx, ny = -1, 0
            pen = dist_to_bow
        else:
            nx, ny = 0, 1 if rel_y > 0 else -1
            pen = dist_to_side
    
    return max(pen, 0), nx, ny


def simulate_ice_ship_interaction(
    polygons, centroids, bonds, ship_pos, ship_length, ship_beam,
    n_frames, dt, drift_vel, thicknesses, ship_props
):
    """
    Simulate ice drifting toward ship with fracture mechanics.
    Same physics as visualize_ice_ship.py, adapted for satellite floes.
    """
    n = len(polygons)
    
    # Compute masses and radii
    areas = np.array([polygon_area(p) for p in polygons])
    masses = areas * thicknesses * IceProperties.ICE_DENSITY
    radii = np.array([np.sqrt(a / np.pi) for a in areas])
    
    # Precompute local polygons (centered at origin) for collision detection
    local_polys = []
    for poly in polygons:
        centroid = polygon_centroid(poly)
        local_polys.append(poly - centroid)
    
    # State
    positions = centroids.copy()
    velocities = np.zeros_like(positions)
    rotations = np.zeros(n)
    angular_vels = np.zeros(n)
    
    # Bond state
    bond_intact = np.ones(len(bonds), dtype=bool)
    rest_lengths = np.array([np.linalg.norm(centroids[i] - centroids[j]) 
                            for i, j in bonds])
    
    # Compute bond strengths
    # Smaller bond width for more dramatic fracturing
    MAX_BOND_WIDTH = 4.0  # meters - allows easier fracturing
    
    bond_strengths = np.zeros(len(bonds))
    for b, (i, j) in enumerate(bonds):
        h = min(thicknesses[i], thicknesses[j])
        # Cap bond width for realistic strength
        w = min((radii[i] + radii[j]) / 2, MAX_BOND_WIDTH)
        bond_strengths[b] = IceProperties.breaking_force(h, w)
    
    if len(bonds) > 0:
        print(f"  Bond strengths: {bond_strengths.min()/1e6:.2f} - "
              f"{bond_strengths.max()/1e6:.2f} MN", flush=True)
    
    # History
    positions_history = [positions.copy()]
    rotations_history = [rotations.copy()]
    bonds_history = [bond_intact.copy()]
    
    ship_impact_force = ship_props.impact_force(drift_vel, np.mean(thicknesses))
    print(f"  Ship impact force: {ship_impact_force/1e6:.2f} MN", flush=True)
    
    collision_count = 0
    ever_collided = set()  # Track which elements ever collided throughout simulation
    for frame in range(n_frames):
        # Find connected components
        components = find_connected_components(n, bonds, bond_intact)
        
        # Compute forces
        forces = np.zeros_like(positions)
        ship_collision_force = np.zeros(n)
        
        for i in range(n):
            px, py = positions[i]
            r = radii[i]
            
            # Check collision using ACTUAL polygon vertices, not circular approximation
            pen, nx, ny = 0, 0, 0
            
            # Get rotated polygon in world coordinates
            angle = rotations[i]
            c, s = np.cos(angle), np.sin(angle)
            rotated = np.zeros_like(local_polys[i])
            rotated[:, 0] = local_polys[i][:, 0] * c - local_polys[i][:, 1] * s
            rotated[:, 1] = local_polys[i][:, 0] * s + local_polys[i][:, 1] * c
            world_poly = rotated + positions[i]
            
            # Check collision at polygon vertices (actual floe shape)
            for vx, vy in world_poly:
                test_pen, test_nx, test_ny = point_to_ship_collision(
                    vx, vy, ship_pos, ship_length, ship_beam
                )
                if test_pen > pen:
                    pen, nx, ny = test_pen, test_nx, test_ny
            
            # Also check some edge midpoints for better coverage
            n_verts = len(world_poly)
            for j in range(n_verts):
                v1 = world_poly[j]
                v2 = world_poly[(j + 1) % n_verts]
                mid = (v1 + v2) / 2
                test_pen, test_nx, test_ny = point_to_ship_collision(
                    mid[0], mid[1], ship_pos, ship_length, ship_beam
                )
                if test_pen > pen:
                    pen, nx, ny = test_pen, test_nx, test_ny
            
            if pen > 0:
                collision_count += 1
                ever_collided.add(i)
                
                # Collision force (original values for small cells)
                force_mag = (ship_props.mass / 1e6) * 1e5 * pen
                forces[i, 0] += force_mag * nx
                forces[i, 1] += force_mag * ny
                ship_collision_force[i] = force_mag
                
                # Side push for flow around hull
                rel_y = py - ship_pos[1]
                side_direction = 1 if rel_y > 0 else -1
                side_push = (ship_props.mass / 1e6) * 3e4 * pen * side_direction
                forces[i, 1] += side_push
                
                # Angular impulse - very small to prevent spinning
                angular_vels[i] += 0.01 * side_direction * min(pen, 1.0)
        
        # Inter-element repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    direction = d / dist
                    f = 2e5 * overlap ** 1.5 * direction
                    forces[i] -= f
                    forces[j] += f
        
        # Drift force
        for comp in components:
            total_mass = sum(masses[i] for i in comp)
            drift_force = drift_vel * total_mass * 8
            for i in comp:
                forces[i, 0] += drift_force * masses[i] / total_mass
        
        # Water drag
        for i in range(n):
            forces[i] -= 0.5 * velocities[i] * masses[i]
        
        # Move connected components together
        new_velocities = velocities.copy()
        new_positions = positions.copy()
        
        for comp in components:
            if len(comp) == 1:
                i = comp[0]
                accel = forces[i] / masses[i]
                new_velocities[i] += accel * dt
                new_velocities[i] = np.clip(new_velocities[i], -20, 20)
                new_positions[i] += new_velocities[i] * dt
            else:
                total_mass = sum(masses[i] for i in comp)
                com = sum(masses[i] * positions[i] for i in comp) / total_mass
                total_force = sum(forces[i] for i in comp)
                comp_vel = sum(masses[i] * velocities[i] for i in comp) / total_mass
                
                accel = total_force / total_mass
                comp_vel += accel * dt
                comp_vel = np.clip(comp_vel, -20, 20)
                
                for i in comp:
                    new_velocities[i] = comp_vel
                    rel_pos = positions[i] - com
                    new_com = com + comp_vel * dt
                    new_positions[i] = new_com + rel_pos
        
        velocities = new_velocities
        positions = new_positions
        
        # Stress propagation (same as original)
        element_stress = ship_collision_force.copy()
        for _ in range(2):
            new_stress = element_stress.copy()
            for b, (i, j) in enumerate(bonds):
                if bond_intact[b]:
                    stress_transfer = 0.3
                    if element_stress[i] > element_stress[j]:
                        new_stress[j] += element_stress[i] * stress_transfer
                    else:
                        new_stress[i] += element_stress[j] * stress_transfer
            element_stress = new_stress
        
        # Check for bond fracture (same as original)
        for b, (i, j) in enumerate(bonds):
            if not bond_intact[b]:
                continue
            
            d = positions[j] - positions[i]
            dist = np.linalg.norm(d)
            stretch = abs(dist - rest_lengths[b])
            
            avg_thickness = (thicknesses[i] + thicknesses[j]) / 2
            stiffness = IceProperties.YOUNGS_MODULUS * avg_thickness * 0.001
            bond_force = stiffness * stretch
            
            propagated_stress = max(element_stress[i], element_stress[j])
            bond_force += propagated_stress * 0.2
            
            base_strength = bond_strengths[b]
            
            # Direct collision dramatically weakens bonds
            collision_on_bond = ship_collision_force[i] > 0 or ship_collision_force[j] > 0
            if collision_on_bond:
                effective_strength = base_strength * 0.1  # Even weaker for direct collision
            elif propagated_stress > base_strength * 0.5:
                effective_strength = base_strength * 0.6
            else:
                effective_strength = base_strength
            
            # Also break if collision force is strong enough (direct fracture)
            collision_force_on_bond = max(ship_collision_force[i], ship_collision_force[j])
            if collision_on_bond and collision_force_on_bond > effective_strength:
                bond_intact[b] = False
            elif bond_force > effective_strength:
                bond_intact[b] = False
        
        # Small rotation with strong damping to prevent spinning
        angular_vels += np.random.randn(n) * 0.001  # Very small noise
        angular_vels *= 0.85  # Strong damping
        angular_vels = np.clip(angular_vels, -0.2, 0.2)  # Hard cap
        rotations += angular_vels * dt
        
        positions_history.append(positions.copy())
        rotations_history.append(rotations.copy())
        bonds_history.append(bond_intact.copy())
    
    if collision_count > 0:
        print(f"  Total collisions: {collision_count}", flush=True)
        # Debug: which elements collided and do they have bonds?
        bonded_elements = set()
        for i, j in bonds:
            bonded_elements.add(i)
            bonded_elements.add(j)
        collided_with_bonds = ever_collided & bonded_elements
        collided_without_bonds = ever_collided - bonded_elements
        print(f"  Collided elements: {len(ever_collided)} total, "
              f"{len(collided_with_bonds)} bonded, {len(collided_without_bonds)} unbonded", flush=True)
    else:
        print("  WARNING: No collisions detected!", flush=True)
    
    return positions_history, rotations_history, bonds_history


# ============================================================================
# 2D Visualization
# ============================================================================

def animate_ice_ship_2d(
    polygons, positions_history, rotations_history, bonds_history, bonds,
    ship_outline, initial_centroids, thicknesses,
    output_file='medea_ship_fracture.gif', fps=20
):
    """Animate ice-ship interaction in 2D."""
    n_frames = len(positions_history)
    n_elements = len(polygons)
    
    local_polys = []
    for poly in polygons:
        centroid = polygon_centroid(poly)
        local_polys.append(poly - centroid)
    
    all_pos = np.concatenate(positions_history)
    x_min = min(all_pos[:, 0].min(), ship_outline[:, 0].min()) - 20
    x_max = max(all_pos[:, 0].max(), ship_outline[:, 0].max()) + 20
    y_min = min(all_pos[:, 1].min(), ship_outline[:, 1].min()) - 20
    y_max = max(all_pos[:, 1].max(), ship_outline[:, 1].max()) + 20
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    water_color = '#1A4D70'
    ship_color = '#404040'
    
    thickness_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
    ice_colors = []
    for t in thickness_norm:
        r = int(240 - 100 * t)
        g = int(250 - 80 * t)
        b = int(255 - 40 * t)
        ice_colors.append(f'#{r:02x}{g:02x}{b:02x}')
    
    def rotate_polygon(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return poly @ R.T
    
    print(f"Creating 2D animation with {n_frames} frames...", flush=True)
    
    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_facecolor(water_color)
        
        positions = positions_history[frame]
        rotations = rotations_history[frame]
        intact = bonds_history[frame]
        
        # Find broken bonds for edge highlighting
        broken_neighbors = {i: set() for i in range(n_elements)}
        for b, (i, j) in enumerate(bonds):
            if not intact[b]:
                broken_neighbors[i].add(j)
                broken_neighbors[j].add(i)
        
        # Draw ice floes
        for i in range(n_elements):
            rotated = rotate_polygon(local_polys[i], rotations[i])
            world_poly = rotated + positions[i]
            
            has_broken = len(broken_neighbors[i]) > 0
            edge_color = '#1A4D70' if has_broken else ice_colors[i]
            edge_width = 1.5 if has_broken else 0.5
            
            patch = MplPolygon(
                world_poly, closed=True,
                facecolor=ice_colors[i],
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=0.95
            )
            ax.add_patch(patch)
        
        # Draw ship
        ship_patch = MplPolygon(
            ship_outline, closed=True,
            facecolor=ship_color,
            edgecolor='#202020',
            linewidth=2,
            alpha=0.95
        )
        ax.add_patch(ship_patch)
        
        # Stats
        n_broken = np.sum(~intact)
        n_total = len(bonds)
        
        ax.set_title(
            f'DPDEM Ice-Ship Simulation (MEDEA Satellite Floes)\n'
            f'Frame {frame}/{n_frames-1} | Bonds: {n_broken}/{n_total} broken',
            fontsize=14
        )
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        
        return []
    
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    print(f"Saving to {output_file}...", flush=True)
    anim.save(output_file, writer=PillowWriter(fps=fps))
    print("Done!", flush=True)
    plt.close(fig)


# ============================================================================
# 3D Visualization
# ============================================================================

def animate_ice_ship_3d(
    polygons, positions_history, rotations_history, bonds_history, bonds,
    ship_pos, ship_length, ship_beam, ship_draft,
    initial_centroids, thicknesses,
    output_file='medea_ship_fracture_3d.gif', fps=15, max_floes=50
):
    """Create 3D animation."""
    n_frames = len(positions_history)
    n_elements = min(len(polygons), max_floes)
    
    # Select largest floes for 3D visualization
    areas = np.array([polygon_area(p) for p in polygons])
    indices = np.argsort(areas)[-n_elements:]
    
    local_polys = []
    for i in indices:
        centroid = polygon_centroid(polygons[i])
        local_polys.append(polygons[i] - centroid)
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    all_pos = np.concatenate([positions_history[f][indices] for f in range(n_frames)])
    x_range = all_pos[:, 0].max() - all_pos[:, 0].min()
    y_range = all_pos[:, 1].max() - all_pos[:, 1].min()
    
    print(f"Creating 3D animation with {n_frames} frames...", flush=True)
    
    def update(frame):
        ax.clear()
        
        positions = positions_history[frame]
        rotations = rotations_history[frame]
        
        # Draw ice floes
        for idx, i in enumerate(indices):
            h = thicknesses[i]
            angle = rotations[i]
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            
            poly_2d = local_polys[idx] @ R.T + positions[i]
            
            # Top face
            top_verts = np.column_stack([poly_2d, np.full(len(poly_2d), h * 0.1)])
            # Bottom face
            bot_verts = np.column_stack([poly_2d, np.full(len(poly_2d), -h * 0.9)])
            
            # Draw faces
            ax.add_collection3d(Poly3DCollection(
                [top_verts], alpha=0.8, facecolor='#E8F4F8', edgecolor='#A0C0D0', linewidth=0.5
            ))
            ax.add_collection3d(Poly3DCollection(
                [bot_verts], alpha=0.6, facecolor='#B0D0E0', edgecolor='#80A0B0', linewidth=0.5
            ))
            
            # Side faces
            n_verts = len(poly_2d)
            for j in range(n_verts):
                k = (j + 1) % n_verts
                side = np.array([
                    [poly_2d[j, 0], poly_2d[j, 1], h * 0.1],
                    [poly_2d[k, 0], poly_2d[k, 1], h * 0.1],
                    [poly_2d[k, 0], poly_2d[k, 1], -h * 0.9],
                    [poly_2d[j, 0], poly_2d[j, 1], -h * 0.9],
                ])
                ax.add_collection3d(Poly3DCollection(
                    [side], alpha=0.7, facecolor='#D0E8F0', edgecolor='#90B0C0', linewidth=0.3
                ))
        
        # Draw ship
        sx, sy = ship_pos
        ship_verts_top = [
            [sx - ship_length/2, sy - ship_beam/2, ship_draft * 0.3],
            [sx + ship_length/2, sy - ship_beam/2, ship_draft * 0.3],
            [sx + ship_length/2, sy + ship_beam/2, ship_draft * 0.3],
            [sx - ship_length/2, sy + ship_beam/2, ship_draft * 0.3],
        ]
        ship_verts_bot = [
            [sx - ship_length/2, sy - ship_beam/2, -ship_draft * 0.7],
            [sx + ship_length/2, sy - ship_beam/2, -ship_draft * 0.7],
            [sx + ship_length/2, sy + ship_beam/2, -ship_draft * 0.7],
            [sx - ship_length/2, sy + ship_beam/2, -ship_draft * 0.7],
        ]
        
        ax.add_collection3d(Poly3DCollection(
            [ship_verts_top], alpha=0.9, facecolor='#505050', edgecolor='#303030'
        ))
        ax.add_collection3d(Poly3DCollection(
            [ship_verts_bot], alpha=0.9, facecolor='#404040', edgecolor='#303030'
        ))
        
        # Set view
        ax.set_xlim(all_pos[:, 0].min() - 50, all_pos[:, 0].max() + 50)
        ax.set_ylim(all_pos[:, 1].min() - 50, all_pos[:, 1].max() + 50)
        ax.set_zlim(-5, 5)
        ax.view_init(elev=25, azim=-60 + frame * 0.1)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(f'DPDEM 3D Ice-Ship Simulation\nFrame {frame}/{n_frames-1}')
        
        return []
    
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    print(f"Saving to {output_file}...", flush=True)
    anim.save(output_file, writer=PillowWriter(fps=fps))
    print("Done!", flush=True)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MEDEA Ice Ship Fracture Simulation')
    parser.add_argument('--quick', action='store_true', help='Simulation only, no animations')
    parser.add_argument('--no-3d', action='store_true', help='Skip 3D animation')
    args = parser.parse_args()
    
    total_start = time.time()
    
    print("=" * 70)
    print("DPDEM: MEDEA Satellite Ice + Ship Fracture Simulation")
    if args.quick:
        print("  [QUICK MODE - simulation only]")
    elif args.no_3d:
        print("  [NO-3D MODE - skipping 3D animation]")
    print("=" * 70)
    print()
    
    # Parameters
    METERS_PER_PIXEL = 2.5
    
    # Ship parameters - CCGS John G. Diefenbaker style icebreaker (same as visualize_ice_ship_3d.py)
    ship_length = 158.0  # m
    ship_beam = 28.0     # m  
    ship_draft = 10.5    # m
    ship_displacement = 23000e3  # 23,000 tonnes
    
    # Paths
    data_dir = Path(__file__).parent / 'data'
    pickle_path = data_dir / 'ice_floes_MEDEA_fram_20100629.pkl'
    
    if not pickle_path.exists():
        print(f"ERROR: Data file not found: {pickle_path}")
        return
    
    # Load data
    t0 = time.time()
    print("Loading MEDEA ice floe data...", flush=True)
    polygons_px, centroids_px, metadata = load_medea_floes(pickle_path)
    print(f"  Loaded {metadata['n_floes']} floes ({time.time()-t0:.1f}s)", flush=True)
    
    # Scale to world coordinates
    print(f"Scaling to world coordinates ({METERS_PER_PIXEL} m/px)...", flush=True)
    all_polygons, all_centroids = scale_floes_to_world(
        polygons_px, centroids_px, METERS_PER_PIXEL, center_domain=True,
        subsample_region=None  # Use all floes first
    )
    
    # Filter to keep only floes in ship's path (narrow y-band for meaningful collision)
    y_center = (all_centroids[:, 1].min() + all_centroids[:, 1].max()) / 2
    y_band = 80.0  # Keep floes within +/- 80m of ship centerline
    keep_mask = np.abs(all_centroids[:, 1] - y_center) < y_band
    
    sat_polygons = [all_polygons[i] for i in range(len(all_polygons)) if keep_mask[i]]
    sat_centroids = all_centroids[keep_mask]
    
    print(f"  Selected {len(sat_polygons)} satellite floes in ship's path", flush=True)
    
    # Tessellate each satellite floe internally for FRACTURING
    # This allows pieces to break OFF from larger floes
    print("Tessellating floes internally for fracturing...", flush=True)
    # DENSITY-BASED: 0.8 cells per 100m² = ~1 cell per 125m² = ~11m radius cells
    CELLS_PER_100M2 = 0.8  # Higher = more cells = more fracturing
    all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
        sat_polygons, sat_centroids, cells_per_100m2=CELLS_PER_100M2, seed=42
    )
    print(f"  Created {len(all_cells)} total cells", flush=True)
    
    # Filter cells to keep only those in ship's collision path
    ship_y_center = y_center  # Same as satellite floe center
    cell_y_band = 60.0  # Keep cells within +/- 60m of ship (ship beam 28m + cell radius)
    keep_cells = np.abs(all_cell_centroids[:, 1] - ship_y_center) < cell_y_band
    
    polygons = [all_cells[i] for i in range(len(all_cells)) if keep_cells[i]]
    centroids = all_cell_centroids[keep_cells]
    parent_floe_ids = all_parent_ids[keep_cells]
    print(f"  Filtered to {len(polygons)} cells in ship's path (y +/- {cell_y_band}m)", flush=True)
    
    # Compute areas and radii for cells
    areas = np.array([polygon_area(p) for p in polygons])
    radii = np.sqrt(areas / np.pi)
    
    # Assign thickness - cells inherit thickness from parent satellite floe
    print("Assigning ice thickness...", flush=True)
    sat_areas = np.array([polygon_area(p) for p in sat_polygons])
    sat_thicknesses = assign_thickness_by_area(sat_areas, min_thickness=0.8, max_thickness=2.5, power=0.25)
    thicknesses = sat_thicknesses[parent_floe_ids]
    print(f"  Thickness range: {thicknesses.min():.2f} - {thicknesses.max():.2f} m", flush=True)
    
    # Create bonds between adjacent cells (tighter threshold for tessellated cells)
    print("Creating bonds between cells...", flush=True)
    bonds = create_proximity_bonds(centroids, radii, threshold_factor=1.5)
    print(f"  Created {len(bonds)} bonds", flush=True)
    
    # Domain info
    all_verts = np.vstack(polygons)
    domain_width = all_verts[:, 0].max() - all_verts[:, 0].min()
    domain_height = all_verts[:, 1].max() - all_verts[:, 1].min()
    print(f"  Domain: {domain_width:.0f} x {domain_height:.0f} m", flush=True)
    
    # Ship setup
    ship_props = ShipProperties(ship_length, ship_beam, ship_draft, ship_displacement)
    print(f"\nShip: {ship_length}m x {ship_beam}m, mass={ship_props.mass/1e6:.1f} kt", flush=True)
    
    # Debug: print centroid range
    print(f"  Centroids X range: {centroids[:, 0].min():.1f} to {centroids[:, 0].max():.1f}m", flush=True)
    print(f"  Centroids Y range: {centroids[:, 1].min():.1f} to {centroids[:, 1].max():.1f}m", flush=True)
    
    # Position ship very close to ice field for immediate collision
    rightmost_edge = max(centroids[i, 0] + radii[i] for i in range(len(centroids)))
    # Ship bow almost touching the rightmost floe
    ship_bow_x = rightmost_edge + 5  # 5m gap - immediate collision
    # Center ship on ice field's y-center
    y_center = (centroids[:, 1].min() + centroids[:, 1].max()) / 2
    ship_pos = np.array([ship_bow_x + ship_length/2, y_center])
    drift_vel = 6.0  # m/s
    print(f"  Ship bow at x={ship_bow_x:.1f}m (rightmost floe edge at {rightmost_edge:.1f}m)", flush=True)
    
    ship_outline = create_ship_hull(ship_length, ship_beam, ship_pos, bow_direction='left')
    
    # Run simulation - longer to capture all collisions
    t0 = time.time()
    n_frames = 600  # More frames to see full interaction
    print(f"\nRunning fracture simulation ({n_frames} frames)...", flush=True)
    
    positions_history, rotations_history, bonds_history = simulate_ice_ship_interaction(
        polygons=polygons,
        centroids=centroids.copy(),
        bonds=bonds,
        ship_pos=ship_pos,
        ship_length=ship_length,
        ship_beam=ship_beam,
        n_frames=n_frames,
        dt=0.05,
        drift_vel=drift_vel,
        thicknesses=thicknesses,
        ship_props=ship_props,
    )
    sim_time = time.time() - t0
    
    # Stats
    n_initial = len(bonds)
    n_intact = bonds_history[-1].sum() if n_initial > 0 else 0
    n_broken = n_initial - n_intact
    pct = 100 * n_broken / n_initial if n_initial > 0 else 0
    print(f"\n  Final: {n_broken}/{n_initial} bonds broken ({pct:.0f}%)", flush=True)
    print(f"  Simulation time: {sim_time:.1f}s", flush=True)
    
    if args.quick:
        print(f"\nQuick mode complete! Total: {time.time()-total_start:.1f}s")
        return
    
    # 2D animation
    t0 = time.time()
    print("\nGenerating 2D animation...", flush=True)
    animate_ice_ship_2d(
        polygons, positions_history, rotations_history, bonds_history, bonds,
        ship_outline, centroids, thicknesses,
        output_file='medea_ship_fracture.gif', fps=20
    )
    print(f"  2D animation time: {time.time()-t0:.1f}s", flush=True)
    
    # 3D animation
    if not args.no_3d:
        t0 = time.time()
        print("\nGenerating 3D animation...", flush=True)
        animate_ice_ship_3d(
            polygons, positions_history, rotations_history, bonds_history, bonds,
            ship_pos, ship_length, ship_beam, ship_draft,
            centroids, thicknesses,
            output_file='medea_ship_fracture_3d.gif', fps=15, max_floes=50
        )
        print(f"  3D animation time: {time.time()-t0:.1f}s", flush=True)
    
    print(f"\nTotal time: {time.time()-total_start:.1f}s")
    print("\nOutput: medea_ship_fracture.gif" + (" and medea_ship_fracture_3d.gif" if not args.no_3d else ""))


if __name__ == '__main__':
    main()
