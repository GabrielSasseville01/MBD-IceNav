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


def R_xy(psi):
    """Rotation matrix body -> world for 2D (Fossen / AUTO-IceNav). (x_world, y_world)^T = R_xy(psi) @ (u, v)^T."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def ssa(angle):
    """Smallest-signed angle in [-pi, pi) (AUTO-IceNav gnc.ssa)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


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


def create_simple_ice_field(
    seed=42,
):
    """
    Create a minimal synthetic ice field for fast testing: 3 floes of different
    sizes so ship–ice and ice–ice interaction can be observed.
    Returns (sat_polygons, sat_centroids) in world coordinates [m].
    """
    np.random.seed(seed)
    # (cx, cy, half_width, half_height) for each floe — layout along ship path (x) with some y spread
    floes_spec = [
        (0.0, 0.0, 30.0, 20.0),    # main floe: 60x40 m
        (-55.0, 10.0, 15.0, 11.0), # smaller: 30x22 m, left and above
        (50.0, -25.0, 24.0, 16.0), # medium: 48x32 m, right and below
    ]
    sat_polygons = []
    sat_centroids = []
    for cx, cy, w, h in floes_spec:
        poly = np.array([
            [cx - w, cy - h],
            [cx + w, cy - h],
            [cx + w, cy + h],
            [cx - w, cy + h],
        ])
        sat_polygons.append(poly)
        sat_centroids.append(np.array([cx, cy]))
    sat_centroids = np.array(sat_centroids)
    return sat_polygons, sat_centroids


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


def create_ship_hull_at_position_heading(length, beam, position, heading):
    """Create ship hull outline at position with given heading (radians). Bow faces heading direction."""
    cx, cy = position
    half_len = length / 2
    half_beam = beam / 2
    n_pts = 30
    
    # Build hull in local frame (bow at +x, centered at origin)
    local_x = []
    local_y = []
    for i in range(n_pts + 1):
        t = i / n_pts
        x_local = half_len - t * length  # bow at +half_len, stern at -half_len
        x_rel = x_local / half_len
        if x_rel > 0.6:
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
        else:
            taper = 1.0
        taper = max(taper, 0.02)
        local_x.append(x_local)
        local_y.append(half_beam * taper)
    for i in range(n_pts, -1, -1):
        t = i / n_pts
        x_local = half_len - t * length
        x_rel = x_local / half_len
        if x_rel > 0.6:
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
        else:
            taper = 1.0
        taper = max(taper, 0.02)
        local_x.append(x_local)
        local_y.append(-half_beam * taper)
    
    local_verts = np.column_stack([local_x, local_y])
    # Rotate by heading and translate to position
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    world_verts = local_verts @ R.T + np.array([cx, cy])
    return world_verts


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


def _distance_point_to_segment(px, py, ax, ay, bx, by):
    """Signed distance from point (px,py) to segment (A,B). Positive = point to the left of A->B."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c1 = wx * vx + wy * vy
    c2 = vx * vx + vy * vy
    if c2 < 1e-20:
        return np.sqrt(wx * wx + wy * wy), ax, ay
    if c1 <= 0:
        d = np.sqrt(wx * wx + wy * wy)
        return d, ax, ay
    if c1 >= c2:
        dx, dy = px - bx, py - by
        return np.sqrt(dx * dx + dy * dy), bx, by
    b = c1 / c2
    cx, cy = ax + b * vx, ay + b * vy
    dx, dy = px - cx, py - cy
    return np.sqrt(dx * dx + dy * dy), cx, cy


def point_inside_polygon_penetration(px, py, polygon):
    """
    When point (px, py) is inside polygon, return penetration (distance to nearest edge)
    and outward normal from that edge. Assumes polygon is ordered (e.g. CCW).
    If point is not inside, return (0, 0, 0).
    """
    poly = np.asarray(polygon)
    n = len(poly)
    if n < 3:
        return 0.0, 0.0, 0.0
    if not point_in_polygon((px, py), poly):
        return 0.0, 0.0, 0.0
    best_pen = np.inf
    best_nx, best_ny = 0.0, 0.0
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dist, _, _ = _distance_point_to_segment(px, py, ax, ay, bx, by)
        # Outward normal for edge A->B (CCW polygon: left of A->B is inside, so outward is right)
        ex, ey = bx - ax, by - ay
        nx = ey
        ny = -ex
        nlen = np.sqrt(nx * nx + ny * ny) + 1e-20
        nx, ny = nx / nlen, ny / nlen
        # Ensure normal points outward: from centroid to point should be opposite to normal
        cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
        dot = (px - cx) * nx + (py - cy) * ny
        if dot > 0:
            nx, ny = -nx, -ny
        if dist < best_pen:
            best_pen = dist
            best_nx, best_ny = nx, ny
    return best_pen, best_nx, best_ny


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


def point_to_ship_collision(px, py, ship_pos, ship_length, ship_beam, bow_taper_start=0.6, ship_heading=0.0):
    """Check collision with ship hull. ship_heading: angle in radians (0 = bow toward -x)."""
    cx, cy = ship_pos
    half_len = ship_length / 2
    half_beam = ship_beam / 2
    
    # Transform to ship-local frame (ship bow along -x when heading=0)
    dx = px - cx
    dy = py - cy
    c, s = np.cos(-ship_heading), np.sin(-ship_heading)
    rel_x = c * dx - s * dy
    rel_y = s * dx + c * dy
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
            nx_local = -np.cos(bow_angle)
            ny_local = np.sin(bow_angle)
        else:
            nx_local = -np.cos(bow_angle)
            ny_local = -np.sin(bow_angle)
        
        pen = min(dist_to_bow, dist_to_side) + 0.5
    else:
        if dist_to_bow < dist_to_side:
            nx_local, ny_local = -1, 0
            pen = dist_to_bow
        else:
            nx_local = 0
            ny_local = 1 if rel_y > 0 else -1
            pen = dist_to_side
    
    # Rotate normal back to world frame
    nx = np.cos(ship_heading) * nx_local - np.sin(ship_heading) * ny_local
    ny = np.sin(ship_heading) * nx_local + np.cos(ship_heading) * ny_local
    
    return max(pen, 0), nx, ny


def simulate_ice_ship_interaction(
    polygons, centroids, bonds, ship_pos, ship_length, ship_beam,
    n_frames, dt, drift_vel, thicknesses, ship_props,
    ship_trajectory=None,  # Optional: list of (pos, heading) per frame for moving ship
    return_ship_diagnostics=False,  # If True and trajectory given, return per-frame F_ice, vel, pos
):
    """
    Simulate ice-ship interaction with fracture mechanics.
    If ship_trajectory is provided: ship moves through ice (no drift).
    Otherwise: ice drifts toward stationary ship.
    Returns (positions_history, rotations_history, bonds_history, ship_history[, ship_diagnostics]).
    When return_ship_diagnostics=True and ship_trajectory is used, ship_diagnostics is a list of
    dicts per frame: F_ice_ship, ship_vel, ship_pos, desired_pos, desired_vel, t.
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
    MAX_BOND_WIDTH = 4.0  # meters - allows easier fracturing
    bond_strengths = np.zeros(len(bonds))
    for b, (i, j) in enumerate(bonds):
        h = min(thicknesses[i], thicknesses[j])
        w = min((radii[i] + radii[j]) / 2, MAX_BOND_WIDTH)
        bond_strengths[b] = IceProperties.breaking_force(h, w)
    
    if len(bonds) > 0:
        print(f"  Bond strengths: {bond_strengths.min()/1e6:.2f} - "
              f"{bond_strengths.max()/1e6:.2f} MN", flush=True)
    
    # History
    positions_history = [positions.copy()]
    rotations_history = [rotations.copy()]
    bonds_history = [bond_intact.copy()]
    ship_history = []
    
    ship_impact_force = (ship_props.impact_force(drift_vel, np.mean(thicknesses))
                         if (drift_vel > 0 and len(thicknesses) > 0) else 0.0)
    print(f"  Ship impact force: {ship_impact_force/1e6:.2f} MN", flush=True)
    
    collision_count = 0
    ever_collided = set()  # Track which elements ever collided throughout simulation
    
    # Ship dynamics (2.1): 3-DOF body-frame model matching AUTO-IceNav (Fossen notation).
    # State: eta = (x, y, psi), nu = (u, v, r) body-frame (surge, sway, yaw rate).
    # Kinematics: eta_dot from R_xy(psi)*[u,v] and psi_dot = r (State.integrate in sim_dynamics).
    # Dynamics: M*nu_dot + C(nu)*nu + D*nu = tau_control + tau_ice (Fossen 2021 / Supply vessel).
    # tau_ice = [F_ice_body; M_ice] from ice collision reaction; control = body-frame PD (NRC/DPcontrol).
    use_ship_dynamics = ship_trajectory is not None
    ship_diagnostics = [] if (use_ship_dynamics and return_ship_diagnostics) else None
    if use_ship_dynamics:
        ship_pos_dyn = np.array(ship_trajectory[0][0], dtype=float)
        ship_heading_dyn = float(ship_trajectory[0][1])
        # Body-frame velocity nu = [u, v, r] (surge m/s, sway m/s, yaw rate rad/s)
        ship_u = 0.0
        ship_v = 0.0
        ship_r = 0.0
        m = ship_props.mass
        Izz = (1.0 / 12.0) * m * (ship_length**2 + ship_beam**2)
        # Mass matrix (3-DOF, diagonal for simplicity; can add added-mass later)
        M3 = np.diag([m, m, Izz])
        M3inv = np.linalg.inv(M3)
        # Linear damping (Fossen-style; scale with mass/size so ice can still perturb)
        D3 = np.diag([m * 0.02, m * 0.02, Izz * 0.15])
        # Control gains: body-frame PD like NRC/DPcontrol (setpoint = desired pos/heading)
        Kp_xy = ship_props.mass * 0.03
        Kp_psi = 5e9
        Kd_ship = 1e6
    
    for frame in range(n_frames):
        # Ship position and heading this frame (from dynamics state or fixed)
        if use_ship_dynamics:
            ship_pos_frame = ship_pos_dyn.copy()
            ship_heading = ship_heading_dyn
        elif ship_trajectory is not None:
            ship_pos_frame, ship_heading = ship_trajectory[min(frame, len(ship_trajectory) - 1)]
        else:
            ship_pos_frame = ship_pos
            ship_heading = 0.0
        
        # Find connected components
        components = find_connected_components(n, bonds, bond_intact)
        
        # Compute forces
        forces = np.zeros_like(positions)
        ship_collision_force = np.zeros(n)
        F_ice_ship = np.zeros(2)
        M_ice_ship = 0.0
        
        # Ship hull in world frame (once per frame) for ship-to-ice collision (2.3)
        ship_hull_pts = create_ship_hull_at_position_heading(
            ship_length, ship_beam, ship_pos_frame, ship_heading
        )
        N_EDGE_SAMPLES = 4  # sample points along each ice edge (2.3: harder collision detection)
        
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
            
            # Ice -> ship: polygon vertices
            for vx, vy in world_poly:
                test_pen, test_nx, test_ny = point_to_ship_collision(
                    vx, vy, ship_pos_frame, ship_length, ship_beam, ship_heading=ship_heading
                )
                if test_pen > pen:
                    pen, nx, ny = test_pen, test_nx, test_ny
            
            # Ice -> ship: multiple samples along each edge (2.3)
            n_verts = len(world_poly)
            for j in range(n_verts):
                v1 = world_poly[j]
                v2 = world_poly[(j + 1) % n_verts]
                for k in range(1, N_EDGE_SAMPLES + 1):
                    t = k / (N_EDGE_SAMPLES + 1)
                    pt = (1 - t) * v1 + t * v2
                    test_pen, test_nx, test_ny = point_to_ship_collision(
                        pt[0], pt[1], ship_pos_frame, ship_length, ship_beam, ship_heading=ship_heading
                    )
                    if test_pen > pen:
                        pen, nx, ny = test_pen, test_nx, test_ny
            
            # Ship -> ice: hull points inside this ice polygon (2.3)
            for sx, sy in ship_hull_pts:
                if point_in_polygon((sx, sy), world_poly):
                    test_pen, test_nx, test_ny = point_inside_polygon_penetration(sx, sy, world_poly)
                    if test_pen > pen:
                        pen, nx, ny = test_pen, test_nx, test_ny
            
            if pen > 0:
                collision_count += 1
                ever_collided.add(i)
                
                # Collision force (2.2): tied to ice mechanics — breaking force and penetration
                # Effective contact width for this element (same scale as bond width)
                w_contact = min(2.0 * radii[i], 4.0)
                F_break = IceProperties.breaking_force(thicknesses[i], w_contact)
                PEN_CRIT_REF = 0.5  # m — penetration at which force reaches F_break
                force_mag = F_break * min(pen / PEN_CRIT_REF, 1.0)
                forces[i, 0] += force_mag * nx
                forces[i, 1] += force_mag * ny
                ship_collision_force[i] = force_mag
                
                # Side push (flow around hull): fraction of normal force, same ice-mechanics scale
                rel_y = py - ship_pos_frame[1]
                side_direction = 1 if rel_y > 0 else -1
                side_push = 0.2 * force_mag * side_direction
                forces[i, 1] += side_push
                
                # Reaction force and moment on ship (2.1): F_ship = -F_on_ice, M = r × F_ship
                F_ice_ship[0] -= force_mag * nx
                F_ice_ship[1] -= (force_mag * ny + side_push)
                r_vec = positions[i] - ship_pos_frame
                F_ship_i = np.array([force_mag * nx, force_mag * ny + side_push])  # force on ice
                M_ice_ship += r_vec[0] * (-F_ship_i[1]) - r_vec[1] * (-F_ship_i[0])  # r × (-F_on_ice)
                
                # Angular impulse - very small to prevent spinning
                angular_vels[i] += 0.01 * side_direction * min(pen, 1.0)
        
        # 2.4: Bow rubble resistance — extra force opposing surge when ice contacts bow
        RUBBLE_COEF = 0.25  # fraction of bow contact force that acts as rubble resistance
        RUBBLE_CAP_N = 5e6  # cap (N) so rubble doesn't dominate
        bow_contact_force = 0.0
        c_r, s_r = np.cos(ship_heading), np.sin(ship_heading)
        for i in range(n):
            if ship_collision_force[i] <= 0:
                continue
            dx = positions[i, 0] - ship_pos_frame[0]
            dy = positions[i, 1] - ship_pos_frame[1]
            body_x = c_r * dx + s_r * dy  # body-frame forward (bow = +x)
            if body_x > 0:
                bow_contact_force += ship_collision_force[i]
        if bow_contact_force > 0:
            F_rubble_mag = min(RUBBLE_COEF * bow_contact_force, RUBBLE_CAP_N)
            forward = np.array([c_r, s_r])
            F_ice_ship -= F_rubble_mag * forward  # resistance opposes forward motion
        
        # Inter-element contact: normal repulsion (Hertz-like) + Coulomb tangential friction (1.3)
        ICE_ICE_FRICTION = 0.1  # μ, from typical ice–ice friction (e.g. DPDEM config)
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    direction = d / dist
                    # Normal force (repulsion)
                    f_n_mag = 2e5 * overlap ** 1.5
                    f_n = f_n_mag * direction
                    forces[i] -= f_n
                    forces[j] += f_n
                    # Tangential friction (Coulomb): oppose relative sliding, |F_t| ≤ μ |F_n|
                    v_rel = velocities[j] - velocities[i]
                    v_t = v_rel - np.dot(v_rel, direction) * direction
                    v_t_mag = np.linalg.norm(v_t)
                    if v_t_mag > 1e-6:
                        t_hat = v_t / v_t_mag
                        f_t_mag = min(ICE_ICE_FRICTION * f_n_mag, 1e5)  # cap for stability
                        f_t = -f_t_mag * t_hat  # opposes sliding
                        forces[i] += f_t
                        forces[j] -= f_t
        
        # Drift force (only when ship is stationary - ice drifts toward ship)
        if ship_trajectory is None:
            for comp in components:
                total_mass = sum(masses[i] for i in comp)
                drift_force = drift_vel * total_mass * 8
                for i in comp:
                    forces[i, 0] += drift_force * masses[i] / total_mass
        
        # Water drag (1.4): quadratic in speed, depends on submerged/drag area and thickness
        RHO_WATER = IceProperties.WATER_DENSITY
        C_D_DRAG = 0.4  # drag coefficient (order 0.1–0.5 for bluff bodies)
        for i in range(n):
            v = velocities[i]
            v_mag = np.linalg.norm(v)
            # Drag area: top area scaled by thickness so thicker floes have more wetted drag
            A_drag = areas[i] * (thicknesses[i] / 1.0)  # ref thickness 1 m
            if v_mag > 1e-6:
                # Quadratic: F_drag = -0.5 * rho * C_d * A * |v|^2 * v_hat
                drag_mag = 0.5 * RHO_WATER * C_D_DRAG * A_drag * (v_mag ** 2)
                forces[i] -= drag_mag * (v / v_mag)
            else:
                # Linear damping at very low speed for numerical stability
                forces[i] -= 0.5 * RHO_WATER * C_D_DRAG * A_drag * 0.1 * v
        
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
        
        # Stress propagation (1.2): drive from actual bond forces, then diffuse
        element_stress = ship_collision_force.copy()
        for b, (i, j) in enumerate(bonds):
            if not bond_intact[b]:
                continue
            d = positions[j] - positions[i]
            dist = np.linalg.norm(d)
            stretch = abs(dist - rest_lengths[b])
            avg_thickness = (thicknesses[i] + thicknesses[j]) / 2
            stiffness = IceProperties.YOUNGS_MODULUS * avg_thickness * 0.001
            bond_force_mag = stiffness * stretch
            element_stress[i] += bond_force_mag
            element_stress[j] += bond_force_mag
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
        
        # Check for bond fracture
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
            
            collision_on_bond = ship_collision_force[i] > 0 or ship_collision_force[j] > 0
            if collision_on_bond:
                effective_strength = base_strength * 0.1
            elif propagated_stress > base_strength * 0.5:
                effective_strength = base_strength * 0.6
            else:
                effective_strength = base_strength
            
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
        
        # Ship dynamics (2.1): body-frame 3-DOF (AUTO-IceNav / Fossen). M*nu_dot + C*nu + D*nu = tau_control + tau_ice
        if use_ship_dynamics:
            psi = ship_heading_dyn
            nu = np.array([ship_u, ship_v, ship_r])
            # Ice force/moment in body frame (world F_ice_ship, M_ice_ship)
            F_ice_body = R_xy(psi).T @ F_ice_ship
            tau_ice = np.array([F_ice_body[0], F_ice_body[1], M_ice_ship])
            # Control: body-frame error (NRC/DPcontrol style)
            tr = ship_trajectory
            idx = min(frame, len(tr) - 1)
            idx_next = min(frame + 1, len(tr) - 1)
            desired_pos = np.array(tr[idx][0])
            desired_heading = tr[idx][1]
            e_xy_world = desired_pos - ship_pos_dyn
            e_xy_body = R_xy(psi).T @ e_xy_world
            e_psi = ssa(desired_heading - psi)
            desired_vel_world = (np.array(tr[idx_next][0]) - desired_pos) / dt if idx_next > idx else np.zeros(2)
            nu_des_xy = R_xy(psi).T @ desired_vel_world
            nu_des = np.array([nu_des_xy[0], nu_des_xy[1], 0.0])
            tau_control = np.array([
                Kp_xy * e_xy_body[0] + Kd_ship * (nu_des[0] - nu[0]),
                Kp_xy * e_xy_body[1] + Kd_ship * (nu_des[1] - nu[1]),
                Kp_psi * e_psi - Kd_ship * nu[2],
            ])
            # Coriolis (3-DOF, Fossen gnc m2c): C*nu with diagonal M
            C = np.zeros((3, 3))
            C[0, 2] = -m * nu[1]
            C[1, 2] = m * nu[0]
            C[2, 0], C[2, 1] = -C[0, 2], -C[1, 2]
            nu_dot = M3inv @ (tau_control + tau_ice - D3 @ nu - C @ nu)
            nu = nu + nu_dot * dt
            nu[0] = np.clip(nu[0], -15, 15)
            nu[1] = np.clip(nu[1], -8, 8)
            nu[2] = np.clip(nu[2], -0.15, 0.15)
            ship_u, ship_v, ship_r = nu[0], nu[1], nu[2]
            # Kinematics (AUTO-IceNav State.integrate): world velocity from body [u,v], then integrate position/heading
            u_g, v_g = R_xy(psi) @ np.array([ship_u, ship_v])
            ship_pos_dyn[0] += dt * u_g
            ship_pos_dyn[1] += dt * v_g
            ship_heading_dyn = (ship_heading_dyn + dt * ship_r) % (2 * np.pi)
            ship_history.append((ship_pos_dyn.copy(), float(ship_heading_dyn)))
            if ship_diagnostics is not None:
                desired_vel = desired_vel_world
                ship_vel_world = R_xy(ship_heading_dyn) @ np.array([ship_u, ship_v])
                ship_diagnostics.append({
                    't': frame * dt,
                    'F_ice_ship': F_ice_ship.copy(),
                    'ship_vel': ship_vel_world,
                    'ship_pos': ship_pos_dyn.copy(),
                    'desired_pos': desired_pos,
                    'desired_vel': desired_vel,
                })
        else:
            ship_history.append((ship_pos_frame, ship_heading))
        
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
    
    if return_ship_diagnostics:
        return positions_history, rotations_history, bonds_history, ship_history, ship_diagnostics
    return positions_history, rotations_history, bonds_history, ship_history


# ============================================================================
# 2D Visualization
# ============================================================================

def animate_ice_ship_2d(
    polygons, positions_history, rotations_history, bonds_history, bonds,
    ship_history, ship_length, ship_beam, initial_centroids, thicknesses,
    output_file='medea_ship_fracture.gif', fps=20
):
    """Animate ice-ship interaction in 2D. ship_history: list of (pos, heading) per frame.
    Handles n_elements=0 (clear water: ship only)."""
    n_frames = len(positions_history)
    n_elements = len(polygons)
    
    local_polys = []
    for poly in polygons:
        centroid = polygon_centroid(poly)
        local_polys.append(poly - centroid)
    
    if len(ship_history) > 0:
        ship_positions = np.array([h[0] for h in ship_history])
        x_min = ship_positions[:, 0].min() - ship_length - 20
        x_max = ship_positions[:, 0].max() + ship_length + 20
        y_min = ship_positions[:, 1].min() - ship_beam - 20
        y_max = ship_positions[:, 1].max() + ship_beam + 20
    else:
        x_min, x_max, y_min, y_max = -50, 50, -50, 50
    if n_elements > 0:
        all_pos = np.concatenate(positions_history)
        x_min = min(x_min, all_pos[:, 0].min() - 20)
        x_max = max(x_max, all_pos[:, 0].max() + 20)
        y_min = min(y_min, all_pos[:, 1].min() - 20)
        y_max = max(y_max, all_pos[:, 1].max() + 20)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    water_color = '#1A4D70'
    ship_color = '#404040'
    
    if n_elements > 0 and len(thicknesses) > 0:
        thickness_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
        ice_colors = []
        for t in thickness_norm:
            r = int(240 - 100 * t)
            g = int(250 - 80 * t)
            b = int(255 - 40 * t)
            ice_colors.append(f'#{r:02x}{g:02x}{b:02x}')
    else:
        ice_colors = []
    
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
        if frame < len(ship_history):
            ship_pos_frame, ship_heading = ship_history[frame]
            ship_outline = create_ship_hull_at_position_heading(
                ship_length, ship_beam, ship_pos_frame, ship_heading
            )
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

def simulate_3d_ice_physics(positions_history, bonds_history, bonds, ship_history,
                           ship_length, ship_beam, ship_draft, n_elements):
    """
    Add 3D ice dynamics: ice bends and submerges under ship hull when pushed.
    Broken ice goes under the hull; intact ice bends at the bow.
    """
    n_frames = len(positions_history)
    z_positions = [np.zeros(n_elements) for _ in range(n_frames)]
    tilts_x = [np.zeros(n_elements) for _ in range(n_frames)]
    tilts_y = [np.zeros(n_elements) for _ in range(n_frames)]
    
    for frame in range(1, n_frames):
        positions = positions_history[frame]
        intact = bonds_history[frame]
        ship_pos, ship_psi = ship_history[min(frame, len(ship_history) - 1)]
        ship_x, ship_y = ship_pos[0], ship_pos[1]
        
        prev_z = z_positions[frame - 1].copy()
        prev_tilt_x = tilts_x[frame - 1].copy()
        
        broken_count = np.zeros(n_elements)
        for b, (i, j) in enumerate(bonds):
            if not intact[b]:
                broken_count[i] += 1
                broken_count[j] += 1
        
        new_z = prev_z.copy()
        new_tilt_x = prev_tilt_x.copy()
        new_tilt_y = tilts_y[frame - 1].copy()
        
        bow_x = ship_x + (ship_length / 2) * np.cos(ship_psi)
        bow_y = ship_y + (ship_length / 2) * np.sin(ship_psi)
        
        for i in range(n_elements):
            px, py = positions[i]
            dx = px - ship_x
            dy = py - ship_y
            c, s = np.cos(-ship_psi), np.sin(-ship_psi)
            local_x = c * dx - s * dy
            local_y = s * dx + c * dy
            dist_from_bow = ship_length / 2 - local_x
            
            if 0 < dist_from_bow < ship_length * 0.4 and abs(local_y) < ship_beam:
                push_factor = 1 - dist_from_bow / (ship_length * 0.4)
                if broken_count[i] > 0:
                    target_z = -ship_draft - push_factor * 2.0
                    new_z[i] += (target_z - new_z[i]) * 0.15
                    new_tilt_x[i] += (0.4 * push_factor - new_tilt_x[i]) * 0.1
                else:
                    new_tilt_x[i] += (0.2 * push_factor - new_tilt_x[i]) * 0.05
            elif dist_from_bow < 0:
                if broken_count[i] > 0 and abs(local_y) < ship_beam * 0.8:
                    new_z[i] += (-ship_draft * 0.9 - new_z[i]) * 0.05
                elif new_z[i] < -0.5:
                    new_z[i] += (0 - new_z[i]) * 0.03
            
            new_tilt_x[i] *= 0.95
            new_tilt_y[i] *= 0.95
            
            dist_to_ship = np.sqrt((px - ship_x)**2 + (py - ship_y)**2)
            if new_z[i] < -0.3 and dist_to_ship > ship_length:
                new_z[i] += 0.02
        
        z_positions[frame] = new_z
        tilts_x[frame] = new_tilt_x
        tilts_y[frame] = new_tilt_y
    
    return z_positions, tilts_x, tilts_y


def animate_ice_ship_3d(
    polygons, positions_history, rotations_history, bonds_history, bonds,
    ship_history, ship_length, ship_beam, ship_draft,
    initial_centroids, thicknesses,
    output_file='medea_ship_fracture_3d.gif', fps=15, frame_skip=2
):
    """
    Create 3D animation matching 2D: same ice map, ship motion, plus 3D ice dynamics
    (bending, submerging under hull).
    """
    n_frames = len(positions_history)
    n_elements = len(polygons)
    
    local_polys = []
    for poly in polygons:
        centroid = polygon_centroid(poly)
        local_polys.append(poly - centroid)
    
    all_pos = np.concatenate(positions_history)
    x_min, x_max = all_pos[:, 0].min() - 50, all_pos[:, 0].max() + 50
    y_min, y_max = all_pos[:, 1].min() - 50, all_pos[:, 1].max() + 50
    domain_width = x_max - x_min
    domain_height = y_max - y_min
    
    print(f"  Computing 3D ice dynamics (bending, submerging)...", flush=True)
    z_positions, tilts_x, tilts_y = simulate_3d_ice_physics(
        positions_history, bonds_history, bonds, ship_history,
        ship_length, ship_beam, ship_draft, n_elements
    )
    
    thickness_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
    ice_colors = []
    for t in thickness_norm:
        r = int(230 - 80 * t)
        g = int(245 - 60 * t)
        b = int(255 - 30 * t)
        ice_colors.append(f'#{r:02x}{g:02x}{b:02x}')
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    frames_to_render = list(range(0, n_frames, frame_skip))
    if frames_to_render[-1] != n_frames - 1:
        frames_to_render.append(n_frames - 1)
    
    print(f"  Creating 3D animation with {len(frames_to_render)} frames...", flush=True)
    
    def rotate_poly(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return poly @ R.T
    
    def update(frame_idx):
        frame = frames_to_render[frame_idx]
        ax.clear()
        
        ship_pos, ship_psi = ship_history[min(frame, len(ship_history) - 1)]
        ship_x, ship_y = ship_pos[0], ship_pos[1]
        positions = positions_history[frame]
        rotations = rotations_history[frame]
        intact = bonds_history[frame]
        z_pos = z_positions[frame]
        tilt_x = tilts_x[frame]
        tilt_y = tilts_y[frame]
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(-ship_draft - 3, 5)
        ax.set_box_aspect([domain_width, domain_height, (ship_draft + 8) * 2])
        ax.view_init(elev=30, azim=-60 + frame * 0.15)
        
        # Draw ice with 3D dynamics (z offset, tilt)
        for i in range(n_elements):
            rotated = rotate_poly(local_polys[i], rotations[i])
            world_poly = rotated + positions[i]
            z_center = z_pos[i]
            tx, ty = tilt_x[i], tilt_y[i]
            h = thicknesses[i]
            
            n_verts = len(world_poly)
            center_xy = positions[i]
            top_verts = []
            bottom_verts = []
            for v in world_poly:
                dx = v[0] - center_xy[0]
                dy = v[1] - center_xy[1]
                z_offset = dx * np.sin(tx) + dy * np.sin(ty)
                top_verts.append([v[0], v[1], z_center + h * 0.1 + z_offset])
                bottom_verts.append([v[0], v[1], z_center - h * 0.9 + z_offset])
            
            top_verts = np.array(top_verts)
            bottom_verts = np.array(bottom_verts)
            
            has_breaks = any(not intact[b] for b, (pi, pj) in enumerate(bonds) if pi == i or pj == i)
            edge_color = '#2080A0' if has_breaks else '#A0C8D8'
            
            ax.add_collection3d(Poly3DCollection(
                [top_verts], facecolor=ice_colors[i], edgecolor=edge_color,
                linewidth=0.5 if has_breaks else 0.2, alpha=0.9
            ))
            for j in range(n_verts):
                k = (j + 1) % n_verts
                side = np.array([bottom_verts[j], bottom_verts[k], top_verts[k], top_verts[j]])
                ax.add_collection3d(Poly3DCollection(
                    [side], facecolor='#C8E0F0', edgecolor=edge_color, linewidth=0.1, alpha=0.8
                ))
        
        # Draw 3D ship hull at correct position and heading
        c, s = np.cos(ship_psi), np.sin(ship_psi)
        half_l, half_b = ship_length / 2, ship_beam / 2
        deck_height = 1.5
        n_sections = 10
        
        hull_verts = []
        for i in range(n_sections + 1):
            t = i / n_sections
            x_local = half_l - t * ship_length
            x_rel = x_local / half_l
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5 if x_rel > 0.6 else 1.0
            taper = max(taper, 0.05)
            local_beam = ship_beam * taper
            local_draft = ship_draft * taper
            local_deck = deck_height * taper
            hull_verts.extend([
                [x_local, -local_beam/2, local_deck],
                [x_local, local_beam/2, local_deck],
                [x_local, -local_beam/2 * 0.9, 0],
                [x_local, local_beam/2 * 0.9, 0],
                [x_local, 0, -local_draft],
            ])
        
        hull_verts = np.array(hull_verts)
        R_3d = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        hull_verts_world = hull_verts @ R_3d.T + np.array([ship_x, ship_y, 0])
        
        verts_per_section = 5
        deck_color, hull_above, hull_below = '#8B4513', '#C0C0C0', '#8B0000'
        
        for i in range(n_sections):
            base = i * verts_per_section
            next_base = (i + 1) * verts_per_section
            deck_face = [hull_verts_world[base], hull_verts_world[next_base],
                        hull_verts_world[next_base + 1], hull_verts_world[base + 1]]
            ax.add_collection3d(Poly3DCollection(
                [deck_face], facecolor=deck_color, edgecolor='#5D3A1A', linewidth=0.5, alpha=0.95
            ))
            port_top = [hull_verts_world[base], hull_verts_world[base + 2],
                       hull_verts_world[next_base + 2], hull_verts_world[next_base]]
            ax.add_collection3d(Poly3DCollection(
                [port_top], facecolor=hull_above, edgecolor='#808080', linewidth=0.3, alpha=0.95
            ))
            starboard_top = [hull_verts_world[base + 1], hull_verts_world[next_base + 1],
                            hull_verts_world[next_base + 3], hull_verts_world[base + 3]]
            ax.add_collection3d(Poly3DCollection(
                [starboard_top], facecolor=hull_above, edgecolor='#808080', linewidth=0.3, alpha=0.95
            ))
            port_bot = [hull_verts_world[base + 2], hull_verts_world[base + 4],
                       hull_verts_world[next_base + 4], hull_verts_world[next_base + 2]]
            ax.add_collection3d(Poly3DCollection(
                [port_bot], facecolor=hull_below, edgecolor='#5C0000', linewidth=0.3, alpha=0.95
            ))
            starboard_bot = [hull_verts_world[base + 3], hull_verts_world[next_base + 3],
                            hull_verts_world[next_base + 4], hull_verts_world[base + 4]]
            ax.add_collection3d(Poly3DCollection(
                [starboard_bot], facecolor=hull_below, edgecolor='#5C0000', linewidth=0.3, alpha=0.95
            ))
        
        n_broken = np.sum(~intact)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(f'DPDEM 3D Ice-Ship Simulation (MEDEA)\nFrame {frame}/{n_frames-1} | Bonds: {n_broken}/{len(bonds)} broken')
        return []
    
    anim = FuncAnimation(fig, update, frames=len(frames_to_render), blit=False)
    print(f"  Saving to {output_file}...", flush=True)
    anim.save(output_file, writer=PillowWriter(fps=fps))
    print("  Done!", flush=True)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MEDEA Ice Ship Fracture Simulation')
    parser.add_argument('--quick', action='store_true', help='Simulation only, no animations')
    parser.add_argument('--no-3d', action='store_true', help='Skip 3D animation')
    parser.add_argument('--full-map', action='store_true',
                        help='Use full ice map; ship starts bottom-left, moves diagonally to top-right')
    parser.add_argument('--simple', action='store_true',
                        help='Use minimal synthetic ice field (3 floes) for fast testing (no MEDEA data)')
    args = parser.parse_args()
    
    total_start = time.time()
    
    print("=" * 70)
    print("DPDEM: MEDEA Satellite Ice + Ship Fracture Simulation")
    if args.simple:
        print("  [SIMPLE MODE - synthetic ice field for quick testing]")
    if args.full_map:
        print("  [FULL MAP MODE - ship diagonal bottom-left to top-right]")
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
    
    if args.simple:
        # Minimal synthetic ice field: 3 floes of different sizes, no MEDEA
        t0 = time.time()
        print("Creating simple synthetic ice field (3 floes)...", flush=True)
        sat_polygons, sat_centroids = create_simple_ice_field(seed=42)
        print(f"  Created {len(sat_polygons)} floes ({time.time()-t0:.1f}s)", flush=True)
        CELLS_PER_100M2 = 0.25
        all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
            sat_polygons, sat_centroids, cells_per_100m2=CELLS_PER_100M2, seed=42
        )
        polygons = all_cells
        centroids = all_cell_centroids
        parent_floe_ids = all_parent_ids
        print(f"  Tessellated to {len(polygons)} cells", flush=True)
    else:
        # MEDEA data path
        data_dir = Path(__file__).parent / 'data'
        pickle_path = data_dir / 'ice_floes_MEDEA_fram_20100629.pkl'
        if not pickle_path.exists():
            print(f"ERROR: Data file not found: {pickle_path}")
            return
        t0 = time.time()
        print("Loading MEDEA ice floe data...", flush=True)
        polygons_px, centroids_px, metadata = load_medea_floes(pickle_path)
        print(f"  Loaded {metadata['n_floes']} floes ({time.time()-t0:.1f}s)", flush=True)
        print(f"Scaling to world coordinates ({METERS_PER_PIXEL} m/px)...", flush=True)
        all_polygons, all_centroids = scale_floes_to_world(
            polygons_px, centroids_px, METERS_PER_PIXEL, center_domain=True,
            subsample_region=None
        )
        if args.full_map:
            sat_polygons = all_polygons
            sat_centroids = all_centroids
            print(f"  Using full ice map: {len(sat_polygons)} floes", flush=True)
        else:
            y_center = (all_centroids[:, 1].min() + all_centroids[:, 1].max()) / 2
            y_band = 80.0
            keep_mask = np.abs(all_centroids[:, 1] - y_center) < y_band
            sat_polygons = [all_polygons[i] for i in range(len(all_polygons)) if keep_mask[i]]
            sat_centroids = all_centroids[keep_mask]
            print(f"  Selected {len(sat_polygons)} floes in ship's path (y +/- {y_band}m)", flush=True)
        print("Tessellating floes internally for fracturing...", flush=True)
        CELLS_PER_100M2 = 0.8
        all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
            sat_polygons, sat_centroids, cells_per_100m2=CELLS_PER_100M2, seed=42
        )
        print(f"  Created {len(all_cells)} total cells", flush=True)
        if args.full_map:
            polygons = all_cells
            centroids = all_cell_centroids
            parent_floe_ids = all_parent_ids
            print(f"  Using all {len(polygons)} cells (full map)", flush=True)
        else:
            ship_y_center = (sat_centroids[:, 1].min() + sat_centroids[:, 1].max()) / 2
            cell_y_band = 60.0
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
    
    all_verts = np.vstack(polygons)
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
    print(f"  Ice domain X: {x_min:.1f} to {x_max:.1f}m, Y: {y_min:.1f} to {y_max:.1f}m", flush=True)
    
    if args.simple:
        # Simple mode: ship crosses small field along x; enough frames for dynamics to navigate through ice
        n_frames = 250
        ship_start = np.array([x_min - ship_length / 2 - 20.0, (y_min + y_max) / 2])
        ship_end = np.array([x_max + ship_length / 2 + 20.0, (y_min + y_max) / 2])
        ship_heading = 0.0  # bow in +x
        ship_trajectory = []
        for frame in range(n_frames + 1):
            t = frame / n_frames
            pos = ship_start + t * (ship_end - ship_start)
            ship_trajectory.append((pos, ship_heading))
        ship_pos = ship_trajectory[0][0]
        drift_vel = 0.0
        print(f"  Ship path: ({ship_start[0]:.0f}, {ship_start[1]:.0f}) -> ({ship_end[0]:.0f}, {ship_end[1]:.0f}), {n_frames} frames", flush=True)
    elif args.full_map:
        # Ship trajectory: bottom-left to top-right, diagonal straight line
        margin = 50.0
        ship_start = np.array([x_min - ship_length/2 + margin, y_min - ship_beam/2 + margin])
        ship_end = np.array([x_max + ship_length/2 - margin, y_max + ship_beam/2 - margin])
        diagonal_heading = np.arctan2(ship_end[1] - ship_start[1], ship_end[0] - ship_start[0])
        n_frames = 600
        ship_trajectory = []
        for frame in range(n_frames + 1):
            t = frame / n_frames
            pos = ship_start + t * (ship_end - ship_start)
            ship_trajectory.append((pos, diagonal_heading))
        ship_pos = ship_trajectory[0][0]  # For API compatibility
        drift_vel = 0.0
        print(f"  Ship path: bottom-left ({ship_start[0]:.0f}, {ship_start[1]:.0f}) -> "
              f"top-right ({ship_end[0]:.0f}, {ship_end[1]:.0f})", flush=True)
        print(f"  Ship heading: {np.degrees(diagonal_heading):.0f}° (diagonal)", flush=True)
    else:
        # Original: stationary ship, ice drifts toward it
        rightmost_edge = max(centroids[i, 0] + radii[i] for i in range(len(centroids)))
        ship_bow_x = rightmost_edge + 5
        y_center = (centroids[:, 1].min() + centroids[:, 1].max()) / 2
        ship_pos = np.array([ship_bow_x + ship_length/2, y_center])
        ship_trajectory = None
        drift_vel = 6.0
        n_frames = 600
        print(f"  Ship stationary at ({ship_pos[0]:.0f}, {ship_pos[1]:.0f}), ice drifts toward ship", flush=True)
    
    # Run simulation
    t0 = time.time()
    print(f"\nRunning fracture simulation ({n_frames} frames)...", flush=True)
    
    positions_history, rotations_history, bonds_history, ship_history = simulate_ice_ship_interaction(
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
        ship_trajectory=ship_trajectory,
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
    
    out_2d = 'simple_ship_fracture.gif' if args.simple else 'medea_ship_fracture.gif'
    out_3d = 'simple_ship_fracture_3d.gif' if args.simple else 'medea_ship_fracture_3d.gif'
    
    # 2D animation
    t0 = time.time()
    print("\nGenerating 2D animation...", flush=True)
    animate_ice_ship_2d(
        polygons, positions_history, rotations_history, bonds_history, bonds,
        ship_history, ship_length, ship_beam, centroids, thicknesses,
        output_file=out_2d, fps=20
    )
    print(f"  2D animation time: {time.time()-t0:.1f}s", flush=True)
    
    # 3D animation
    if not args.no_3d:
        t0 = time.time()
        print("\nGenerating 3D animation...", flush=True)
        animate_ice_ship_3d(
            polygons, positions_history, rotations_history, bonds_history, bonds,
            ship_history, ship_length, ship_beam, ship_draft,
            centroids, thicknesses,
            output_file=out_3d, fps=15, frame_skip=2
        )
        print(f"  3D animation time: {time.time()-t0:.1f}s", flush=True)
    
    print(f"\nTotal time: {time.time()-total_start:.1f}s")
    print(f"\nOutput: {out_2d}" + (f" and {out_3d}" if not args.no_3d else ""))


if __name__ == '__main__':
    main()
