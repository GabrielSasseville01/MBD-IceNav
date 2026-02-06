#!/usr/bin/env python3
"""
DPDEM: Autonomous Navigation through MEDEA Satellite Ice Field

Combines:
- Real satellite-derived ice floe shapes from MEDEA Fram Strait
- Lattice planner for autonomous path planning
- DPDEM-style fracture physics

Usage:
    python run_medea_autonomous.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi
from pathlib import Path
import pickle
import time

# Import the lattice planner
from dpdem.planning import create_planner


# ============================================================================
# Ice Material Properties
# ============================================================================

class IceProperties:
    ICE_DENSITY = 920  # kg/m³
    WATER_DENSITY = 1025  # kg/m³
    FLEXURAL_STRENGTH_BASE = 0.15e6  # Pa (reduced for visible fracturing)
    YOUNGS_MODULUS = 3e9  # Pa
    
    @staticmethod
    def flexural_strength(thickness):
        h_ref = 1.0
        return IceProperties.FLEXURAL_STRENGTH_BASE * np.sqrt(thickness / h_ref)


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


def ssa(angle):
    """Smallest signed angle."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def create_ship_polygon(cx, cy, heading, length, beam):
    """Create ship polygon at given position and heading."""
    half_l = length / 2
    half_b = beam / 2
    
    n_pts = 20
    hull = []
    
    for i in range(n_pts + 1):
        t = i / n_pts
        x_local = half_l - t * length
        x_rel = x_local / half_l
        
        if x_rel > 0.6:
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
        else:
            taper = 1.0
        taper = max(taper, 0.02)
        
        hull.append([x_local, half_b * taper])
    
    for i in range(n_pts, -1, -1):
        t = i / n_pts
        x_local = half_l - t * length
        x_rel = x_local / half_l
        
        if x_rel > 0.6:
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
        else:
            taper = 1.0
        taper = max(taper, 0.02)
        
        hull.append([x_local, -half_b * taper])
    
    hull = np.array(hull)
    
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    return hull @ R.T + np.array([cx, cy])


def point_in_ship(px, py, ship_x, ship_y, ship_heading, ship_length, ship_beam):
    """Check if point is inside ship hull."""
    c, s = np.cos(-ship_heading), np.sin(-ship_heading)
    dx = px - ship_x
    dy = py - ship_y
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    
    half_l = ship_length / 2
    half_b = ship_beam / 2
    x_rel = local_x / half_l
    
    if x_rel > 0.6:
        taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
    else:
        taper = 1.0
    taper = max(taper, 0.02)
    
    return abs(local_x) < half_l and abs(local_y) < half_b * taper


def ship_ice_penetration(ice_x, ice_y, ship_x, ship_y, ship_heading, ship_length, ship_beam):
    """Calculate penetration depth and normal for ice-ship contact."""
    c, s = np.cos(-ship_heading), np.sin(-ship_heading)
    dx = ice_x - ship_x
    dy = ice_y - ship_y
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    
    half_l = ship_length / 2
    half_b = ship_beam / 2
    x_rel = local_x / half_l
    
    if x_rel > 0.6:
        taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
    else:
        taper = 1.0
    taper = max(taper, 0.02)
    
    dist_to_bow = local_x - half_l
    dist_to_stern = -half_l - local_x
    dist_to_port = local_y - half_b * taper
    dist_to_starboard = -half_b * taper - local_y
    
    if dist_to_bow > 0:
        return 0, 0, 0, 0
    
    pen = 0
    nx_local, ny_local = 0, 0
    
    if abs(local_x) <= half_l and abs(local_y) <= half_b * taper:
        dist_to_side = half_b * taper - abs(local_y)
        dist_to_end = half_l - abs(local_x)
        
        if dist_to_side < dist_to_end:
            pen = dist_to_side
            ny_local = 1 if local_y > 0 else -1
        else:
            pen = dist_to_end
            nx_local = 1 if local_x > 0 else -1
    
    c, s = np.cos(ship_heading), np.sin(ship_heading)
    nx = c * nx_local - s * ny_local
    ny = s * nx_local + c * ny_local
    
    return pen, nx, ny, local_x


# ============================================================================
# MEDEA Data Loading
# ============================================================================

def load_medea_floes(pickle_path):
    """Load MEDEA ice floe data from pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    polygons = [np.array(floe['vertices']) for floe in data]
    centroids = np.array([np.array(floe['centre']) for floe in data])
    areas = np.array([floe['area'] for floe in data])
    
    return polygons, centroids, areas


def scale_floes_to_world(polygons, centroids, areas, meters_per_pixel, min_area_m2=50):
    """Scale to world coords (same as run_medea_floes.py)."""
    # Scale all
    scaled_polygons = [p * meters_per_pixel for p in polygons]
    scaled_centroids = centroids * meters_per_pixel
    scaled_areas = areas * (meters_per_pixel ** 2)
    
    # Center the domain at origin (same as run_medea_floes.py)
    all_verts = np.vstack(scaled_polygons)
    center = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2
    scaled_polygons = [p - center for p in scaled_polygons]
    scaled_centroids = scaled_centroids - center
    
    # Filter by minimum area only
    keep = [i for i in range(len(scaled_areas)) if scaled_areas[i] >= min_area_m2]
    
    filtered_polygons = [scaled_polygons[i] for i in keep]
    filtered_centroids = np.array([scaled_centroids[i] for i in keep])
    filtered_areas = np.array([scaled_areas[i] for i in keep])
    
    # Compute domain bounds
    all_verts = np.vstack(filtered_polygons)
    x_min, y_min = all_verts.min(axis=0)
    x_max, y_max = all_verts.max(axis=0)
    
    # Shift so domain starts at (margin, margin)
    margin = 50.0
    shift = np.array([margin - x_min, margin - y_min])
    filtered_polygons = [p + shift for p in filtered_polygons]
    filtered_centroids = filtered_centroids + shift
    
    field_width = x_max - x_min + 2 * margin
    field_height = y_max - y_min + 2 * margin
    
    return filtered_polygons, filtered_centroids, filtered_areas, field_width, field_height


def tessellate_floe(floe_polygon, cell_size=8.0, seed=None):
    """Tessellate a floe into smaller Voronoi cells."""
    if seed is not None:
        np.random.seed(seed)
    
    # Get floe bounds
    x_min, y_min = floe_polygon.min(axis=0)
    x_max, y_max = floe_polygon.max(axis=0)
    
    # Generate seed points
    nx = max(2, int((x_max - x_min) / cell_size))
    ny = max(2, int((y_max - y_min) / cell_size))
    
    points = []
    for i in range(nx):
        for j in range(ny):
            x = x_min + (i + 0.5 + np.random.uniform(-0.3, 0.3)) * (x_max - x_min) / nx
            y = y_min + (j + 0.5 + np.random.uniform(-0.3, 0.3)) * (y_max - y_min) / ny
            
            # Check if inside floe
            if point_in_polygon([x, y], floe_polygon):
                points.append([x, y])
    
    if len(points) < 3:
        # Return the original floe as a single cell
        centroid = np.mean(floe_polygon, axis=0)
        return [(floe_polygon, centroid)]
    
    points = np.array(points)
    
    # Add mirror points for bounded Voronoi
    width = x_max - x_min
    height = y_max - y_min
    mirrors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            shifted = points.copy()
            shifted[:, 0] += dx * width * 1.5
            shifted[:, 1] += dy * height * 1.5
            mirrors.append(shifted)
    
    all_points = np.vstack([points] + mirrors)
    vor = Voronoi(all_points)
    
    cells = []
    n_actual = len(points)
    
    for i in range(n_actual):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) < 3:
            continue
        
        verts = vor.vertices[region]
        
        # Clip to floe boundary
        clipped = clip_polygon_to_floe(verts, floe_polygon)
        if clipped is not None and len(clipped) >= 3:
            area = polygon_area(clipped)
            if area > 5.0:  # Min cell area
                centroid = np.mean(clipped, axis=0)
                cells.append((clipped, centroid))
    
    return cells if cells else [(floe_polygon, np.mean(floe_polygon, axis=0))]


def point_in_polygon(point, polygon):
    """Ray casting point-in-polygon test."""
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


def clip_polygon_to_floe(cell_verts, floe_polygon):
    """Clip cell to floe boundary using shapely."""
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        
        cell = ShapelyPolygon(cell_verts)
        floe = ShapelyPolygon(floe_polygon)
        
        if not cell.is_valid:
            cell = cell.buffer(0)
        if not floe.is_valid:
            floe = floe.buffer(0)
        
        clipped = cell.intersection(floe)
        
        if clipped.is_empty:
            return None
        
        if clipped.geom_type == 'MultiPolygon':
            clipped = max(clipped.geoms, key=lambda p: p.area)
        
        if clipped.geom_type == 'Polygon' and not clipped.is_empty:
            coords = np.array(clipped.exterior.coords[:-1])
            if len(coords) >= 3:
                return coords
        
        return None
    except Exception:
        return None


def create_bonds_within_floes(cells, floe_ids, max_bond_dist=15.0):
    """Create bonds only between cells from the same original floe."""
    centroids = np.array([c[1] for c in cells])
    n = len(cells)
    bonds = []
    
    for i in range(n):
        for j in range(i + 1, n):
            # Only bond within same floe
            if floe_ids[i] != floe_ids[j]:
                continue
            
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < max_bond_dist:
                bonds.append((i, j))
    
    return bonds


# ============================================================================
# Ship Controller
# ============================================================================

class PathFollowingShip:
    """Ship that follows waypoints from a path planner."""
    
    def __init__(self, length, beam, mass, x, y, heading):
        self.length = length
        self.beam = beam
        self.mass = mass
        self.I_zz = (1/12) * mass * (length**2 + beam**2)
        
        self.x = x
        self.y = y
        self.psi = heading
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0
        
        self.waypoints = None
        self.wp_idx = 0
        self.lookahead = 20.0
        
    def set_path(self, path):
        if path is not None and len(path) > 0:
            self.waypoints = path
            self.wp_idx = 0
    
    def get_target(self):
        if self.waypoints is None or len(self.waypoints) == 0:
            return None
        
        # Find lookahead point
        while self.wp_idx < len(self.waypoints) - 1:
            wp = self.waypoints[self.wp_idx]
            dist = np.sqrt((wp[0] - self.x)**2 + (wp[1] - self.y)**2)
            if dist > self.lookahead:
                break
            self.wp_idx += 1
        
        return self.waypoints[self.wp_idx]
    
    def get_control(self):
        target = self.get_target()
        if target is None:
            return np.zeros(3)
        
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 5.0:
            return np.zeros(3)
        
        psi_desired = np.arctan2(dy, dx)
        heading_error = ssa(psi_desired - self.psi)
        
        # Control gains
        Kp_forward = 5e6
        Kp_heading = 5e7
        Kd_heading = 1e7
        
        tau_surge = Kp_forward
        tau_yaw = Kp_heading * heading_error - Kd_heading * self.r
        
        return np.array([tau_surge, 0.0, tau_yaw])
    
    def update(self, F_surge, F_sway, M_yaw, dt):
        drag_surge = 0.02 * self.mass * self.u
        drag_sway = 0.10 * self.mass * self.v
        drag_yaw = 0.015 * self.I_zz * self.r
        
        acc_u = (F_surge - drag_surge) / self.mass
        acc_v = (F_sway - drag_sway) / self.mass
        acc_r = (M_yaw - drag_yaw) / self.I_zz
        
        self.u += dt * acc_u
        self.v += dt * acc_v
        self.r += dt * acc_r
        
        max_speed = 8.0
        self.u = np.clip(self.u, -max_speed * 0.3, max_speed)
        self.v = np.clip(self.v, -max_speed * 0.3, max_speed * 0.3)
        self.r = np.clip(self.r, -0.5, 0.5)
        
        c, s = np.cos(self.psi), np.sin(self.psi)
        x_dot = c * self.u - s * self.v
        y_dot = s * self.u + c * self.v
        
        self.x += dt * x_dot
        self.y += dt * y_dot
        self.psi += dt * self.r


# ============================================================================
# Main Simulation
# ============================================================================

def run_simulation(
    pickle_path,
    meters_per_pixel=2.5,
    cell_size=8.0,
    ship_length=25.0,
    ship_beam=7.0,
    ship_mass=8e6,
    n_frames=600,
    dt=0.05,
    replan_interval=60,
    seed=42,
    max_floes=50,  # Limit for performance
):
    """Run autonomous navigation through MEDEA ice field."""
    
    np.random.seed(seed)
    
    # Load MEDEA floes
    print(f"\n{'='*60}")
    print("MEDEA Autonomous Navigation")
    print(f"{'='*60}")
    
    print("\nLoading MEDEA ice floes...")
    raw_polygons, raw_centroids, raw_areas = load_medea_floes(pickle_path)
    print(f"  Loaded {len(raw_polygons)} floes from satellite data")
    
    # Scale to world coordinates (same as run_medea_floes.py)
    print("Scaling to world coordinates...")
    polygons, centroids, areas, field_width, field_height = scale_floes_to_world(
        raw_polygons, raw_centroids, raw_areas,
        meters_per_pixel, min_area_m2=200  # Filter small floes
    )
    print(f"  {len(polygons)} floes after filtering")
    print(f"  Domain: {field_width:.0f}m x {field_height:.0f}m")
    
    # Select floes in a compact region for dense ice coverage
    # Filter by maximum area to prevent huge floes that create too many cells
    max_floe_area = 5000  # m² - floes larger than this create too many cells
    
    corridor_x_center = field_width / 2
    corridor_y_center = field_height / 2
    corridor_half_width = 250.0  # X spread
    corridor_half_height = 400.0  # Y spread
    
    # Filter floes within a box region AND with area limit
    corridor_indices = []
    for i in range(len(centroids)):
        cx, cy = centroids[i]
        if abs(cx - corridor_x_center) < corridor_half_width and \
           abs(cy - corridor_y_center) < corridor_half_height and \
           areas[i] <= max_floe_area:
            corridor_indices.append(i)
    
    print(f"  Found {len(corridor_indices)} floes in compact region "
          f"(max_area={max_floe_area}m²)")
    
    if len(corridor_indices) > 0:
        # Take floes from the region
        if len(corridor_indices) > max_floes:
            indices = corridor_indices[:max_floes]
        else:
            indices = list(corridor_indices)
        
        polygons = [polygons[i] for i in indices]
        centroids = centroids[indices]
        areas = areas[indices]
        print(f"  Using {len(indices)} floes (area range: {areas.min():.0f}-{areas.max():.0f} m²)")
    else:
        print("  WARNING: No floes in compact region! Finding small floes nearest to center...")
        # Filter by area first, then find closest
        small_floe_mask = areas <= max_floe_area
        if np.any(small_floe_mask):
            small_indices = np.where(small_floe_mask)[0]
            small_dists = np.sqrt((centroids[small_indices, 0] - corridor_x_center)**2 + 
                                 (centroids[small_indices, 1] - corridor_y_center)**2)
            closest = small_indices[np.argsort(small_dists)[:max_floes]]
        else:
            # No small floes, just take smallest
            closest = np.argsort(areas)[:max_floes]
        
        polygons = [polygons[i] for i in closest]
        centroids = centroids[closest]
        areas = areas[closest]
        print(f"  Using {len(closest)} floes")
    
    ship_draft = 3.5
    # Find domain bounds
    all_x = centroids[:, 0]
    all_y = centroids[:, 1]
    
    # Debug: show ice cell positions
    print(f"\n  Ice cell positions:")
    print(f"    X range: {all_x.min():.1f} to {all_x.max():.1f}")
    print(f"    Y range: {all_y.min():.1f} to {all_y.max():.1f}")
    
    # Place ship at the corridor center, just below ice field
    ice_center_x = all_x.mean()  # Aligned with corridor
    ice_center_y = all_y.mean()
    ice_min_y = all_y.min()
    ice_max_y = all_y.max()
    
    # Start just before the first ice, goal beyond ice
    ship_start = (ice_center_x, ice_min_y - 30, np.pi/2)  # Just below ice
    ship_goal = (ice_center_x, ice_max_y + 50)  # Beyond ice
    
    print(f"  Ship start: ({ship_start[0]:.1f}, {ship_start[1]:.1f})")
    print(f"  Ship goal: ({ship_goal[0]:.1f}, {ship_goal[1]:.1f})")
    
    print(f"\nShip: {ship_length}m x {ship_beam}m, mass={ship_mass/1e6:.1f}t")
    print(f"Ice center: ({ice_center_x:.0f}, {ice_center_y:.0f})")
    print(f"Start: ({ship_start[0]:.0f}, {ship_start[1]:.0f})")
    print(f"Goal: ({ship_goal[0]:.0f}, {ship_goal[1]:.0f})")
    print(f"Using all {len(polygons)} filtered floes (ship placed at ice edge)")
    
    # Tessellate floes for fracture physics
    print(f"Tessellating {len(polygons)} floes (cell_size={cell_size}m)...", flush=True)
    all_cells = []
    floe_ids = []
    
    for floe_idx, floe_poly in enumerate(polygons):
        print(f"  Tessellating floe {floe_idx+1}/{len(polygons)}...", flush=True)
        cells = tessellate_floe(floe_poly, cell_size=cell_size, seed=seed + floe_idx)
        print(f"    -> {len(cells)} cells", flush=True)
        for cell in cells:
            all_cells.append(cell)
            floe_ids.append(floe_idx)
    
    print(f"  Created {len(all_cells)} cells from {len(polygons)} floes")
    
    # Create bonds within each floe
    bonds = create_bonds_within_floes(all_cells, floe_ids, max_bond_dist=cell_size * 1.8)
    print(f"  Created {len(bonds)} bonds")
    
    # Extract cell data
    cell_polygons = [c[0] for c in all_cells]
    cell_centroids = np.array([c[1] for c in all_cells])
    cell_areas = np.array([polygon_area(p) for p in cell_polygons])
    cell_radii = np.sqrt(cell_areas / np.pi)
    
    # Thickness based on floe area
    thickness_mean = 1.0
    cell_thicknesses = np.random.normal(thickness_mean, 0.2, len(all_cells))
    cell_thicknesses = np.clip(cell_thicknesses, 0.5, 2.0)
    
    cell_masses = cell_areas * cell_thicknesses * IceProperties.ICE_DENSITY
    
    n_elements = len(all_cells)
    
    # Create ship
    ship = PathFollowingShip(
        length=ship_length, beam=ship_beam, mass=ship_mass,
        x=ship_start[0], y=ship_start[1], heading=ship_start[2],
    )
    
    # Initialize physics state
    positions = cell_centroids.copy()
    velocities = np.zeros((n_elements, 2))
    rotations = np.zeros(n_elements)
    angular_vels = np.zeros(n_elements)
    bond_intact = np.ones(len(bonds), dtype=bool)
    
    # Initialize planner
    print("\nInitializing lattice planner...")
    planner = create_planner(
        map_width=field_width,
        map_height=field_height,
        ship_length=ship_length,
        ship_width=ship_beam,
        ship_mass=ship_mass,
        turning_radius=ship_length * 0.6,
        resolution=10.0,  # Coarser resolution for large domain
        num_headings=16,
        heuristic_weight=3.0,  # More greedy toward goal
    )
    
    # Set collision cost to 0 - ship ignores ice completely and goes straight
    planner.cost_map.collision_cost = 0.0  # Ice is free to traverse
    planner.cost_map.mass_cost_scale = 0.0  # Mass irrelevant
    
    # Update cost map with ice
    print("Updating cost map with ice...")
    planner.update_cost_map(positions, cell_polygons, cell_masses)
    
    # Plan initial path
    print("Planning initial path...")
    path = planner.plan(
        start=(ship.x, ship.y, ship.psi),
        goal=ship_goal,
        max_iterations=200000,
    )
    
    if path is not None:
        print(f"  Path found: {len(path)} waypoints")
        ship.set_path(path)
    else:
        print("  WARNING: No path found!")
        path = np.array([[ship_start[0], ship_start[1], 0],
                        [ship_goal[0], ship_goal[1], 0]])
        ship.set_path(path)
    
    # History
    ship_history = []
    positions_history = []
    bonds_history = []
    paths_history = []
    
    print(f"\nRunning simulation ({n_frames} frames)...")
    start_time = time.time()
    
    for frame in range(n_frames + 1):
        # Record state
        ship_history.append((ship.x, ship.y, ship.psi, ship.u, ship.v, ship.r))
        positions_history.append(positions.copy())
        bonds_history.append(bond_intact.copy())
        paths_history.append(path.copy() if path is not None else None)
        
        # Replan periodically
        if frame > 0 and frame % replan_interval == 0:
            planner.update_cost_map(positions, cell_polygons, cell_masses)
            new_path = planner.plan(
                start=(ship.x, ship.y, ship.psi),
                goal=ship_goal,
                max_iterations=50000,
            )
            if new_path is not None:
                path = new_path
                ship.set_path(path)
        
        # Get control
        tau = ship.get_control()
        
        # Ship-ice interaction
        ice_forces = np.zeros((n_elements, 2))
        ship_force = np.array([0.0, 0.0])
        ship_moment = 0.0
        ship_ice_hits = 0  # Track ship-ice collisions specifically
        
        for i in range(n_elements):
            # Check collision with polygon vertices (not just centroid)
            pen, nx, ny, local_x = 0, 0, 0, 0
            
            # Get current polygon position (translate from original)
            offset = positions[i] - cell_centroids[i]
            current_poly = cell_polygons[i] + offset
            
            # Check all vertices
            for vx, vy in current_poly:
                test_pen, test_nx, test_ny, test_local_x = ship_ice_penetration(
                    vx, vy,
                    ship.x, ship.y, ship.psi,
                    ship.length, ship.beam
                )
                if test_pen > pen:
                    pen, nx, ny, local_x = test_pen, test_nx, test_ny, test_local_x
            
            # Also check edge midpoints for better coverage
            n_verts = len(current_poly)
            for v_idx in range(n_verts):
                v1 = current_poly[v_idx]
                v2 = current_poly[(v_idx + 1) % n_verts]
                mid = (v1 + v2) / 2
                test_pen, test_nx, test_ny, test_local_x = ship_ice_penetration(
                    mid[0], mid[1],
                    ship.x, ship.y, ship.psi,
                    ship.length, ship.beam
                )
                if test_pen > pen:
                    pen, nx, ny, local_x = test_pen, test_nx, test_ny, test_local_x
            
            if pen > 0.1:
                ship_ice_hits += 1
                    
                contact_stiffness = 2e6
                f_mag = contact_stiffness * pen * cell_thicknesses[i]
                
                ice_forces[i, 0] = f_mag * nx
                ice_forces[i, 1] = f_mag * ny
                
                ship_force -= 0.4 * ice_forces[i]
                
                # Check bond fracture
                if frame >= 10:
                    for b_idx, (pi, pj) in enumerate(bonds):
                        if bond_intact[b_idx] and (pi == i or pj == i):
                            stress = f_mag / (cell_radii[i] * cell_thicknesses[i])
                            strength = IceProperties.flexural_strength(cell_thicknesses[i])
                            
                            if stress > strength * 0.5:
                                bond_intact[b_idx] = False
        
        # Ice-ice interaction
        for i in range(n_elements):
            for j in range(i + 1, n_elements):
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                min_dist = cell_radii[i] + cell_radii[j]
                
                if dist < min_dist * 0.95 and dist > 0.1:
                    overlap = min_dist * 0.95 - dist
                    direction = d / dist
                    f = 5e4 * overlap ** 1.5 * direction
                    ice_forces[i] -= f
                    ice_forces[j] += f
        
        # Bond forces
        for b_idx, (pi, pj) in enumerate(bonds):
            if bond_intact[b_idx]:
                d = positions[pj] - positions[pi]
                dist = np.linalg.norm(d)
                rest_dist = np.linalg.norm(cell_centroids[pj] - cell_centroids[pi])
                
                if dist > 0.1:
                    stretch = dist - rest_dist
                    k_bond = 1e5
                    direction = d / dist
                    f = k_bond * stretch * direction
                    ice_forces[pi] += f
                    ice_forces[pj] -= f
        
        # Update ice dynamics
        drag_coef = 0.1
        for i in range(n_elements):
            acc = ice_forces[i] / cell_masses[i]
            velocities[i] += acc * dt
            velocities[i] *= (1 - drag_coef * dt)
            velocities[i] = np.clip(velocities[i], -5, 5)
            positions[i] += velocities[i] * dt
        
        # Apply control to ship
        c, s = np.cos(ship.psi), np.sin(ship.psi)
        F_world = np.array([c * tau[0] - s * tau[1], s * tau[0] + c * tau[1]])
        F_world += ship_force
        M_total = tau[2] + ship_moment
        
        ship.update(tau[0] + ship_force[0] * c + ship_force[1] * s,
                   tau[1] - ship_force[0] * s + ship_force[1] * c,
                   M_total, dt)
        
        # Progress
        if frame % 100 == 0:
            dist_to_goal = np.sqrt((ship.x - ship_goal[0])**2 + (ship.y - ship_goal[1])**2)
            n_broken = np.sum(~bond_intact)
            speed = np.sqrt(ship.u**2 + ship.v**2)
            
            # Find nearest ice centroid
            min_dist_to_ice = float('inf')
            for i in range(n_elements):
                d = np.sqrt((positions[i, 0] - ship.x)**2 + (positions[i, 1] - ship.y)**2)
                if d < min_dist_to_ice:
                    min_dist_to_ice = d
            
            print(f"  Frame {frame}: pos=({ship.x:.1f}, {ship.y:.1f}), "
                  f"dist={dist_to_goal:.1f}m, speed={speed:.2f}m/s, "
                  f"ice_dist={min_dist_to_ice:.1f}m, ship_hits={ship_ice_hits}, "
                  f"broken={n_broken}/{len(bonds)}")
        
        # Goal check
        dist_to_goal = np.sqrt((ship.x - ship_goal[0])**2 + (ship.y - ship_goal[1])**2)
        if dist_to_goal < 10.0:
            print(f"\nGoal reached at frame {frame}!")
            break
    
    elapsed = time.time() - start_time
    n_broken = np.sum(~bond_intact)
    print(f"\nSimulation: {elapsed:.1f}s")
    print(f"Bonds broken: {n_broken}/{len(bonds)} ({100*n_broken/max(1,len(bonds)):.0f}%)")
    
    return {
        'ship': ship,
        'ship_history': ship_history,
        'positions_history': positions_history,
        'bonds_history': bonds_history,
        'paths_history': paths_history,
        'polygons': cell_polygons,
        'bonds': bonds,
        'thicknesses': cell_thicknesses,
        'masses': cell_masses,
        'field_width': field_width,
        'field_height': field_height,
        'goal': ship_goal,
        'floe_ids': floe_ids,
    }


def create_animation(results, output_file='medea_autonomous.gif', fps=20):
    """Create 2D animation."""
    print("\nCreating animation...")
    
    ship = results['ship']
    ship_history = results['ship_history']
    positions_history = results['positions_history']
    bonds_history = results['bonds_history']
    paths_history = results['paths_history']
    polygons = results['polygons']
    bonds = results['bonds']
    thicknesses = results['thicknesses']
    field_width = results['field_width']
    field_height = results['field_height']
    goal = results['goal']
    
    n_frames = len(ship_history)
    n_elements = len(polygons)
    
    # Local polygons
    local_polys = []
    for i, poly in enumerate(polygons):
        centroid = np.mean(poly, axis=0)
        local_polys.append(poly - centroid)
    
    # Colors by thickness
    t_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
    ice_colors = [plt.cm.Blues(0.4 + 0.5 * t) for t in t_norm]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-10, field_width + 10)
        ax.set_ylim(-10, field_height + 10)
        ax.set_aspect('equal')
        ax.set_facecolor('#1A4D70')
        
        positions = positions_history[frame]
        intact = bonds_history[frame]
        ship_x, ship_y, ship_psi, _, _, _ = ship_history[frame]
        
        # Draw ice
        for i in range(n_elements):
            world_poly = local_polys[i] + positions[i]
            
            has_breaks = any(not intact[b] for b, (pi, pj) in enumerate(bonds)
                           if pi == i or pj == i)
            
            edge_color = '#1A4D70' if has_breaks else ice_colors[i]
            edge_width = 1.5 if has_breaks else 0.5
            
            patch = MplPolygon(
                world_poly, facecolor=ice_colors[i],
                edgecolor=edge_color, linewidth=edge_width,
                alpha=0.95, zorder=3
            )
            ax.add_patch(patch)
        
        # Draw planned path
        for p in paths_history:
            if p is not None and len(p) > 1:
                ax.plot(p[:, 0], p[:, 1], 'cyan', linewidth=1.5, alpha=0.4, zorder=4)
        
        # Draw trajectory
        traj_x = [h[0] for h in ship_history[:frame+1]]
        traj_y = [h[1] for h in ship_history[:frame+1]]
        ax.plot(traj_x, traj_y, 'b-', linewidth=2, alpha=0.7, zorder=5)
        
        # Draw ship
        ship_poly = create_ship_polygon(ship_x, ship_y, ship_psi, ship.length, ship.beam)
        ship_patch = MplPolygon(
            ship_poly, facecolor='#404040', edgecolor='#202020',
            linewidth=2, zorder=10
        )
        ax.add_patch(ship_patch)
        
        # Goal
        ax.plot(goal[0], goal[1], 'g*', markersize=20, zorder=11)
        
        n_broken = np.sum(~intact)
        dist_to_goal = np.sqrt((ship_x - goal[0])**2 + (ship_y - goal[1])**2)
        ax.set_title(
            f'MEDEA Autonomous Navigation\n'
            f'Frame {frame}/{n_frames-1} | Dist: {dist_to_goal:.0f}m | '
            f'Bonds broken: {n_broken}/{len(bonds)}',
            fontsize=12
        )
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        
        return []
    
    print(f"  Rendering {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)
    
    print(f"  Saving to {output_file}...")
    anim.save(output_file, writer=PillowWriter(fps=fps), dpi=80)
    print("  Done!")
    
    plt.close()


def main():
    # Path to MEDEA data
    pickle_path = Path(__file__).parent / 'data' / 'ice_floes_MEDEA_fram_20100629.pkl'
    
    if not pickle_path.exists():
        print(f"ERROR: MEDEA data not found at {pickle_path}")
        return
    
    # Use MEDEA ice field with fracture physics
    # Smaller cells for better collision detection
    results = run_simulation(
        pickle_path=pickle_path,
        meters_per_pixel=2.5,
        cell_size=10.0,  # Smaller cells for reliable collision detection
        ship_length=25.0,
        ship_beam=7.0,
        ship_mass=8e6,
        n_frames=600,  # Run to pass through ice region
        dt=0.05,
        replan_interval=60,
        seed=42,
        max_floes=30,  # More floes for denser coverage
    )
    
    create_animation(results, output_file='medea_autonomous.gif', fps=20)
    print("\nOutput: medea_autonomous.gif")


if __name__ == '__main__':
    main()
