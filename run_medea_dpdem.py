#!/usr/bin/env python3
"""
DPDEM with MEDEA Satellite Ice Field

Merges:
- MEDEA satellite ice floe polygons (realistic ice layout from Fram Strait)
- DPDEM physics (collision forces, ice dynamics, fracturing mechanics)
- Lattice planner for autonomous navigation

This produces results similar to:
- Ice layout from medea_ice_drift.gif (realistic satellite polygons)
- Physics from planned_navigation_sparse.gif (DPDEM dynamics and fracturing)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
import pickle
import time
from scipy.spatial import Voronoi

# Import lattice planner
from dpdem.planning import create_planner


# ============================================================================
# Ice Material Properties (from DPDEM paper)
# ============================================================================

class IceProperties:
    ICE_DENSITY = 920  # kg/m³
    WATER_DENSITY = 1025  # kg/m³
    FLEXURAL_STRENGTH_BASE = 0.15e6  # Pa
    YOUNGS_MODULUS = 3e9  # Pa
    
    @staticmethod
    def flexural_strength(thickness):
        return IceProperties.FLEXURAL_STRENGTH_BASE * np.sqrt(thickness / 1.0)
    
    @staticmethod
    def breaking_force(thickness, width):
        sigma_f = IceProperties.flexural_strength(thickness)
        return sigma_f * width * thickness ** 2


# ============================================================================
# Geometry Utilities
# ============================================================================

def polygon_area(verts):
    """Shoelace formula for polygon area."""
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
    """Create ship hull polygon with tapered bow."""
    half_l = length / 2
    half_b = beam / 2
    
    n_pts = 20
    hull = []
    
    # Port side (bow to stern)
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
    
    # Starboard side (stern to bow)
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
    """Check if point is inside ship hull and return penetration info."""
    c, s = np.cos(-ship_heading), np.sin(-ship_heading)
    dx = px - ship_x
    dy = py - ship_y
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    
    half_l = ship_length / 2
    half_b = ship_beam / 2
    x_rel = local_x / half_l
    
    # Bow taper
    if x_rel > 0.6:
        t = (x_rel - 0.6) / 0.4
        t = min(t, 1.0)
        local_half_beam = half_b * (1 - 0.9 * t ** 1.5)
        local_half_beam = max(local_half_beam, half_b * 0.02)
    else:
        local_half_beam = half_b
    
    # Outside ship
    if local_x < -half_l or local_x > half_l:
        return 0, 0, 0
    if abs(local_y) > local_half_beam:
        return 0, 0, 0
    
    # Inside - compute penetration depth
    dist_to_bow = half_l - local_x
    dist_to_stern = local_x + half_l
    dist_to_side = local_half_beam - abs(local_y)
    
    # Determine normal direction
    if x_rel > 0.6:
        # At bow - angled normal
        t = (x_rel - 0.6) / 0.4
        bow_angle = 0.3 + 0.5 * min(t, 1.0)
        if local_y >= 0:
            nx_local, ny_local = np.cos(bow_angle), np.sin(bow_angle)
        else:
            nx_local, ny_local = np.cos(bow_angle), -np.sin(bow_angle)
        pen = min(dist_to_bow, dist_to_side) + 0.5
    elif dist_to_bow < dist_to_side and dist_to_bow < dist_to_stern:
        nx_local, ny_local = 1, 0
        pen = dist_to_bow
    elif dist_to_stern < dist_to_side:
        nx_local, ny_local = -1, 0
        pen = dist_to_stern
    else:
        nx_local = 0
        ny_local = 1 if local_y > 0 else -1
        pen = dist_to_side
    
    # Rotate normal back to world frame
    c, s = np.cos(ship_heading), np.sin(ship_heading)
    nx = c * nx_local - s * ny_local
    ny = s * nx_local + c * ny_local
    
    return max(pen, 0), nx, ny


# ============================================================================
# MEDEA Ice Field Loading (same as run_medea_floes.py)
# ============================================================================

def load_medea_ice_field(pickle_path, meters_per_pixel=2.5, min_area_m2=50):
    """
    Load MEDEA satellite ice floe polygons.
    Uses same scaling as run_medea_floes.py to match medea_ice_drift.gif.
    """
    with open(pickle_path, 'rb') as f:
        floes = pickle.load(f)
    
    print(f"  Loaded {len(floes)} raw floes from satellite data")
    
    # Scale to world coordinates
    polygons = []
    for floe in floes:
        vertices = np.array(floe['vertices']) * meters_per_pixel
        polygons.append(vertices)
    
    # Center domain (same as run_medea_floes.py)
    all_verts = np.vstack(polygons)
    center = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2
    polygons = [p - center for p in polygons]
    
    # Compute areas and filter
    areas = []
    for p in polygons:
        x, y = p[:, 0], p[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas.append(area)
    areas = np.array(areas)
    
    # Filter by minimum area
    keep = areas >= min_area_m2
    polygons = [polygons[i] for i in range(len(polygons)) if keep[i]]
    areas = areas[keep]
    
    # Compute centroids
    centroids = np.array([np.mean(p, axis=0) for p in polygons])
    
    # Domain bounds
    all_verts = np.vstack(polygons)
    x_range = all_verts[:, 0].max() - all_verts[:, 0].min()
    y_range = all_verts[:, 1].max() - all_verts[:, 1].min()
    
    # Shift to positive coordinates with margin
    margin = 50.0
    shift = np.array([margin - all_verts[:, 0].min(), 
                      margin - all_verts[:, 1].min()])
    polygons = [p + shift for p in polygons]
    centroids = centroids + shift
    
    field_width = x_range + 2 * margin
    field_height = y_range + 2 * margin
    
    print(f"  {len(polygons)} floes after filtering (min_area={min_area_m2}m²)")
    print(f"  Domain: {field_width:.0f}m x {field_height:.0f}m")
    print(f"  Area range: {areas.min():.0f} - {areas.max():.0f} m²")
    
    return {
        'polygons': polygons,
        'centroids': centroids,
        'areas': areas,
        'field_width': field_width,
        'field_height': field_height,
    }


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


def tessellate_floe(floe_polygon, cell_size=8.0, seed=None):
    """Tessellate a floe into smaller Voronoi cells for fracturing."""
    if seed is not None:
        np.random.seed(seed)
    
    x_min, y_min = floe_polygon.min(axis=0)
    x_max, y_max = floe_polygon.max(axis=0)
    
    nx = max(2, int((x_max - x_min) / cell_size))
    ny = max(2, int((y_max - y_min) / cell_size))
    
    points = []
    for i in range(nx):
        for j in range(ny):
            x = x_min + (i + 0.5 + np.random.uniform(-0.3, 0.3)) * (x_max - x_min) / nx
            y = y_min + (j + 0.5 + np.random.uniform(-0.3, 0.3)) * (y_max - y_min) / ny
            if point_in_polygon([x, y], floe_polygon):
                points.append([x, y])
    
    if len(points) < 3:
        centroid = np.mean(floe_polygon, axis=0)
        return [(floe_polygon, centroid)]
    
    points = np.array(points)
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
        clipped = clip_polygon_to_floe(verts, floe_polygon)
        if clipped is not None and len(clipped) >= 3:
            area = polygon_area(clipped)
            if area > 5.0:
                centroid = np.mean(clipped, axis=0)
                cells.append((clipped, centroid))
    
    return cells if cells else [(floe_polygon, np.mean(floe_polygon, axis=0))]


def create_bonds_within_floes(cells, floe_ids, max_bond_dist):
    """Create bonds between cells within the same floe (enables fracturing)."""
    centroids = np.array([c[1] for c in cells])
    n = len(cells)
    bonds = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if floe_ids[i] != floe_ids[j]:
                continue
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < max_bond_dist:
                bonds.append((i, j))
    
    return bonds


def create_bonds_between_floes(centroids, radii, max_bond_dist_factor=2.5):
    """
    Create bonds between adjacent floes.
    Bonds exist between floes whose edges are close together.
    """
    n = len(centroids)
    bonds = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            max_dist = (radii[i] + radii[j]) * max_bond_dist_factor
            
            if dist < max_dist:
                bonds.append((i, j))
    
    return bonds


# ============================================================================
# Ship Controller with Path Following
# ============================================================================

class PathFollowingShip:
    """Ship that follows waypoints using pure pursuit."""
    
    def __init__(self, length, beam, mass, x, y, heading):
        self.length = length
        self.beam = beam
        self.mass = mass
        self.I_zz = (1/12) * mass * (length**2 + beam**2)
        
        self.x = x
        self.y = y
        self.psi = heading
        self.u = 0.0  # Surge velocity
        self.v = 0.0  # Sway velocity
        self.r = 0.0  # Yaw rate
        
        self.waypoints = None
        self.lookahead = 20.0
        
    def set_path(self, path):
        if path is not None and len(path) > 0:
            self.waypoints = path
    
    def get_lookahead_point(self):
        if self.waypoints is None:
            return None
        
        # Find closest point on path
        min_dist = np.inf
        closest_idx = 0
        for i, wp in enumerate(self.waypoints):
            d = np.sqrt((wp[0] - self.x)**2 + (wp[1] - self.y)**2)
            if d < min_dist:
                min_dist = d
                closest_idx = i
        
        # Find lookahead point
        cumulative = 0
        for i in range(closest_idx, len(self.waypoints) - 1):
            seg_len = np.linalg.norm(self.waypoints[i+1, :2] - self.waypoints[i, :2])
            if cumulative + seg_len >= self.lookahead:
                frac = (self.lookahead - cumulative) / max(seg_len, 0.1)
                return self.waypoints[i, :2] + frac * (self.waypoints[i+1, :2] - self.waypoints[i, :2])
            cumulative += seg_len
        
        return self.waypoints[-1, :2]
    
    def get_control(self, dt):
        target = self.get_lookahead_point()
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
    
    def apply_forces(self, control_tau, external_force, external_moment, dt):
        c, s = np.cos(self.psi), np.sin(self.psi)
        
        # Convert external force to body frame
        F_ext_body_x = c * external_force[0] + s * external_force[1]
        F_ext_body_y = -s * external_force[0] + c * external_force[1]
        
        F_surge = control_tau[0] + F_ext_body_x
        F_sway = control_tau[1] + F_ext_body_y
        M_yaw = control_tau[2] + external_moment
        
        # Drag forces
        drag_surge = 0.02 * self.mass * self.u
        drag_sway = 0.10 * self.mass * self.v
        drag_yaw = 0.015 * self.I_zz * self.r
        
        # Accelerations
        acc_u = (F_surge - drag_surge) / self.mass
        acc_v = (F_sway - drag_sway) / self.mass
        acc_r = (M_yaw - drag_yaw) / self.I_zz
        
        # Update velocities
        self.u += dt * acc_u
        self.v += dt * acc_v
        self.r += dt * acc_r
        
        # Speed limits
        max_speed = 8.0
        speed = np.sqrt(self.u**2 + self.v**2)
        if speed > max_speed:
            self.u *= max_speed / speed
            self.v *= max_speed / speed
        self.r = np.clip(self.r, -0.5, 0.5)
        
        # Update position
        vx = c * self.u - s * self.v
        vy = s * self.u + c * self.v
        
        self.x += dt * vx
        self.y += dt * vy
        self.psi = (self.psi + dt * self.r) % (2 * np.pi)
    
    def get_polygon(self):
        return create_ship_polygon(self.x, self.y, self.psi, self.length, self.beam)


# ============================================================================
# Connected Components for Ice Physics
# ============================================================================

def find_connected_components(n, bonds, intact):
    """Find connected components of bonded ice floes."""
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


# ============================================================================
# Main Simulation
# ============================================================================

def run_simulation(
    pickle_path,
    meters_per_pixel=2.5,
    ship_length=25.0,
    ship_beam=7.0,
    ship_mass=8e6,
    ice_thickness=1.0,
    n_frames=500,
    dt=0.05,
    replan_interval=50,
    max_floes=150,  # Limit for performance
    seed=42,
):
    """
    Run DPDEM simulation with MEDEA ice field.
    """
    np.random.seed(seed)
    
    print("=" * 60)
    print("DPDEM with MEDEA Satellite Ice Field")
    print("=" * 60)
    
    # Load MEDEA ice floes
    print("\nLoading MEDEA ice floes...")
    ice_data = load_medea_ice_field(
        pickle_path,
        meters_per_pixel=meters_per_pixel,
        min_area_m2=100,  # Filter very small floes
    )
    
    polygons = ice_data['polygons']
    centroids = ice_data['centroids']
    areas = ice_data['areas']
    field_width = ice_data['field_width']
    field_height = ice_data['field_height']
    
    # Limit floes - use medium-sized ones (large floes create too many cells)
    max_floe_area = 5000  # m² - cap to avoid huge tessellations
    keep = areas <= max_floe_area
    if keep.sum() > max_floes:
        idx = np.where(keep)[0]
        sorted_areas = areas[idx]
        mid = len(idx) // 2
        n_take = min(max_floes, len(idx))
        subset = idx[np.argsort(sorted_areas)[max(0, mid - n_take//2) : mid + n_take//2]]
        polygons = [polygons[i] for i in subset]
        centroids = centroids[subset]
        areas = areas[subset]
        print(f"  Using {len(polygons)} medium-sized floes (max_area={max_floe_area}m²)")
    elif len(polygons) > max_floes:
        indices = np.argsort(areas)[-max_floes:]
        polygons = [polygons[i] for i in indices]
        centroids = centroids[indices]
        areas = areas[indices]
        print(f"  Using {max_floes} largest floes")
    
    # Tessellate floes into cells for fracturing
    cell_size = 18.0  # Larger cells = fewer total = faster
    print(f"\nTessellating {len(polygons)} floes (cell_size={cell_size}m)...")
    all_cells = []
    floe_ids = []
    for floe_idx, floe_poly in enumerate(polygons):
        cells = tessellate_floe(floe_poly, cell_size=cell_size, seed=seed + floe_idx)
        for cell_poly, cell_centroid in cells:
            all_cells.append((cell_poly, cell_centroid))
            floe_ids.append(floe_idx)
    
    cell_polygons = [c[0] for c in all_cells]
    centroids = np.array([c[1] for c in all_cells])
    areas = np.array([polygon_area(p) for p in cell_polygons])
    n = len(all_cells)
    print(f"  Created {n} cells from {len(polygons)} floes")
    
    # Compute physical properties
    radii = np.sqrt(areas / np.pi)
    thicknesses = np.random.normal(ice_thickness, 0.2, n)
    thicknesses = np.clip(thicknesses, 0.5, ice_thickness * 2)
    masses = areas * thicknesses * IceProperties.ICE_DENSITY
    
    # Local polygons (relative to centroid)
    local_polys = []
    for i, poly in enumerate(cell_polygons):
        local_polys.append(poly - centroids[i])
    
    # Create bonds WITHIN floes (enables fracturing when ship pushes)
    bonds = create_bonds_within_floes(all_cells, floe_ids, max_bond_dist=cell_size * 1.8)
    print(f"  Created {len(bonds)} bonds within floes")
    
    # Bond properties
    rest_lengths = np.zeros(len(bonds))
    bond_strengths = np.zeros(len(bonds))
    MAX_BOND_WIDTH = 4.0
    
    for b, (i, j) in enumerate(bonds):
        rest_lengths[b] = np.linalg.norm(centroids[i] - centroids[j])
        h = min(thicknesses[i], thicknesses[j])
        w = min((radii[i] + radii[j]) / 2, MAX_BOND_WIDTH)
        bond_strengths[b] = IceProperties.breaking_force(h, w)
    
    # Ship start/goal positions
    # Start closer to ice center for faster interaction
    ice_min_x = centroids[:, 0].min()
    ice_max_x = centroids[:, 0].max()
    ice_center_x = (ice_min_x + ice_max_x) / 2
    
    ship_start = (ice_min_x - 30.0, field_height / 2, 0.0)  # Just before ice
    ship_goal = (ice_max_x + 30.0, field_height / 2)
    print(f"  Ice x-range: {ice_min_x:.0f} to {ice_max_x:.0f}")
    
    print(f"\nShip: {ship_length}m x {ship_beam}m, mass={ship_mass/1e6:.1f}t")
    print(f"Start: ({ship_start[0]:.0f}, {ship_start[1]:.0f})")
    print(f"Goal: ({ship_goal[0]:.0f}, {ship_goal[1]:.0f})")
    
    # Create ship
    ship = PathFollowingShip(
        length=ship_length, beam=ship_beam, mass=ship_mass,
        x=ship_start[0], y=ship_start[1], heading=ship_start[2],
    )
    
    # Initialize state
    positions = centroids.copy()
    velocities = np.zeros((n, 2))
    rotations = np.zeros(n)
    angular_vels = np.zeros(n)
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
        resolution=5.0,
        num_headings=16,
        heuristic_weight=2.0,
    )
    
    # Cost settings - ship goes THROUGH ice (breaks/fractures it)
    planner.cost_map.collision_cost = 0.0  # Ice is traversable - ship pushes through
    planner.cost_map.mass_cost_scale = 0.0
    planner.goal_tolerance = 15.0
    
    # Update cost map with ice
    print("Updating cost map...")
    planner.update_cost_map(positions, local_polys, masses, rotations)
    
    # Plan initial path
    print("Planning initial path...")
    path = planner.plan(
        (ship.x, ship.y, ship.psi),
        ship_goal,
        max_iterations=30000,
    )
    
    if path is not None:
        print(f"  Path found: {len(path)} waypoints")
        ship.set_path(path)
    else:
        print("  No path found - using direct line")
        path = np.array([
            [ship_start[0], ship_start[1], 0],
            [ship_goal[0], ship_goal[1], 0],
        ])
        ship.set_path(path)
    
    # History storage
    ship_history = [(ship.x, ship.y, ship.psi)]
    positions_history = [positions.copy()]
    rotations_history = [rotations.copy()]
    bonds_history = [bond_intact.copy()]
    paths_history = [path.copy() if path is not None else None]
    
    print(f"\nRunning simulation ({n_frames} frames)...")
    t0 = time.time()
    
    for frame in range(n_frames):
        # Replan periodically
        if frame > 0 and frame % replan_interval == 0:
            planner.update_cost_map(positions, local_polys, masses, rotations)
            new_path = planner.plan(
                (ship.x, ship.y, ship.psi),
                ship_goal,
                max_iterations=10000,
            )
            if new_path is not None:
                ship.set_path(new_path)
                paths_history.append(new_path.copy())
        
        # Get ship control
        control_tau = ship.get_control(dt)
        
        # Find connected components
        components = find_connected_components(n, bonds, bond_intact)
        
        # Compute forces
        forces = np.zeros((n, 2))
        ship_force = np.zeros(2)
        ship_moment = 0.0
        ship_collision_force = np.zeros(n)
        
        # Ship-ice collisions
        for i in range(n):
            angle = rotations[i]
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            world_poly = local_polys[i] @ R.T + positions[i]
            
            max_pen = 0
            best_nx, best_ny = 0, 0
            
            # Check all vertices
            for vx, vy in world_poly:
                pen, nx, ny = point_in_ship(
                    vx, vy, ship.x, ship.y, ship.psi, ship.length, ship.beam
                )
                if pen > max_pen:
                    max_pen, best_nx, best_ny = pen, nx, ny
            
            # Check edge midpoints for better coverage
            n_verts = len(world_poly)
            for j in range(n_verts):
                v1 = world_poly[j]
                v2 = world_poly[(j + 1) % n_verts]
                mid = (v1 + v2) / 2
                pen, nx, ny = point_in_ship(
                    mid[0], mid[1], ship.x, ship.y, ship.psi, ship.length, ship.beam
                )
                if pen > max_pen:
                    max_pen, best_nx, best_ny = pen, nx, ny
            
            if max_pen > 0.05:  # Threshold to avoid spurious contacts
                # Strong contact force - ship PUSHES ice out of the way
                # Stiffness scaled by ship mass and penetration
                force_mag = (ship.mass / 1e6) * 2e5 * max_pen
                forces[i, 0] += force_mag * best_nx
                forces[i, 1] += force_mag * best_ny
                ship_collision_force[i] = force_mag
                
                # Reaction on ship (ice pushes back)
                ship_force[0] -= force_mag * best_nx * 0.5
                ship_force[1] -= force_mag * best_ny * 0.5
                
                # Moment on ship
                rx = positions[i, 0] - ship.x
                ry = positions[i, 1] - ship.y
                ship_moment -= (rx * force_mag * best_ny - ry * force_mag * best_nx) * 0.5
                
                # Lateral clearing - push ice to sides of hull
                rel_y = positions[i, 1] - ship.y
                side_dir = 1 if rel_y > 0 else -1
                forces[i, 1] += (ship.mass / 1e6) * 5e4 * max_pen * side_dir
                angular_vels[i] += 0.02 * side_dir * min(max_pen, 1.0)
        
        # Ice-ice repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                min_dist = (radii[i] + radii[j]) * 0.9
                
                if dist < min_dist and dist > 0.1:
                    overlap = min_dist - dist
                    direction = d / dist
                    f = 1e5 * overlap ** 1.5 * direction
                    forces[i] -= f
                    forces[j] += f
        
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
                new_velocities[i] = np.clip(new_velocities[i], -15, 15)
                new_positions[i] += new_velocities[i] * dt
            else:
                # Move as rigid body
                total_mass = sum(masses[i] for i in comp)
                com = sum(masses[i] * positions[i] for i in comp) / total_mass
                total_force = sum(forces[i] for i in comp)
                comp_vel = sum(masses[i] * velocities[i] for i in comp) / total_mass
                
                accel = total_force / total_mass
                comp_vel += accel * dt
                comp_vel = np.clip(comp_vel, -15, 15)
                
                for i in comp:
                    new_velocities[i] = comp_vel
                    rel_pos = positions[i] - com
                    new_com = com + comp_vel * dt
                    new_positions[i] = new_com + rel_pos
        
        velocities = new_velocities
        positions = new_positions
        
        # Stress propagation for fracture
        element_stress = ship_collision_force.copy()
        for _ in range(2):
            new_stress = element_stress.copy()
            for b, (i, j) in enumerate(bonds):
                if bond_intact[b]:
                    if element_stress[i] > element_stress[j]:
                        new_stress[j] += element_stress[i] * 0.3
                    else:
                        new_stress[i] += element_stress[j] * 0.3
            element_stress = new_stress
        
        # Bond fracture (after warmup)
        if frame >= 10:
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
                
                # Collision weakens bonds
                collision_on_bond = ship_collision_force[i] > 0 or ship_collision_force[j] > 0
                if collision_on_bond:
                    effective_strength = bond_strengths[b] * 0.1
                elif propagated_stress > bond_strengths[b] * 0.5:
                    effective_strength = bond_strengths[b] * 0.6
                else:
                    effective_strength = bond_strengths[b]
                
                collision_force_on_bond = max(ship_collision_force[i], ship_collision_force[j])
                if collision_on_bond and collision_force_on_bond > effective_strength:
                    bond_intact[b] = False
                elif bond_force > effective_strength:
                    bond_intact[b] = False
        
        # Rotations with damping
        angular_vels += np.random.randn(n) * 0.001
        angular_vels *= 0.85
        angular_vels = np.clip(angular_vels, -0.2, 0.2)
        rotations += angular_vels * dt
        
        # Boundary forces
        positions[:, 0] = np.clip(positions[:, 0], 0, field_width)
        positions[:, 1] = np.clip(positions[:, 1], 0, field_height)
        
        # Boundary forces on ship
        y_margin = 15.0
        boundary_force = 0.0
        if ship.y < y_margin + ship.beam/2:
            boundary_force += 5e5 * (y_margin + ship.beam/2 - ship.y)
        if ship.y > field_height - y_margin - ship.beam/2:
            boundary_force -= 5e5 * (ship.y - (field_height - y_margin - ship.beam/2))
        ship_force[1] += boundary_force
        
        # Update ship
        ship.apply_forces(control_tau, ship_force, ship_moment, dt)
        ship.x = np.clip(ship.x, ship.length/2, field_width - ship.length/2)
        ship.y = np.clip(ship.y, ship.beam, field_height - ship.beam)
        
        # Store history
        ship_history.append((ship.x, ship.y, ship.psi))
        positions_history.append(positions.copy())
        rotations_history.append(rotations.copy())
        bonds_history.append(bond_intact.copy())
        
        # Progress
        if frame % 100 == 0:
            dist_to_goal = np.sqrt((ship.x - ship_goal[0])**2 + (ship.y - ship_goal[1])**2)
            n_broken = np.sum(~bond_intact)
            speed = np.sqrt(ship.u**2 + ship.v**2)
            print(f"  Frame {frame}: pos=({ship.x:.1f}, {ship.y:.1f}), "
                  f"dist={dist_to_goal:.1f}m, speed={speed:.2f}m/s, "
                  f"broken={n_broken}/{len(bonds)}")
        
        # Goal check
        dist_to_goal = np.sqrt((ship.x - ship_goal[0])**2 + (ship.y - ship_goal[1])**2)
        if dist_to_goal < 10.0:
            print(f"\nGoal reached at frame {frame}!")
            break
    
    sim_time = time.time() - t0
    n_broken = np.sum(~bonds_history[-1])
    print(f"\nSimulation: {sim_time:.1f}s")
    print(f"Bonds broken: {n_broken}/{len(bonds)} ({100*n_broken/max(1,len(bonds)):.0f}%)")
    
    return {
        'ship_history': ship_history,
        'positions_history': positions_history,
        'rotations_history': rotations_history,
        'bonds_history': bonds_history,
        'paths_history': paths_history,
        'local_polys': local_polys,
        'bonds': bonds,
        'thicknesses': thicknesses,
        'ship': ship,
        'field_width': field_width,
        'field_height': field_height,
        'goal': ship_goal,
    }


# ============================================================================
# Animation
# ============================================================================

def create_animation(results, output_file='medea_dpdem.gif', fps=20):
    """Create animation showing MEDEA ice with DPDEM physics."""
    print("\nCreating animation...")
    
    ship = results['ship']
    ship_history = results['ship_history']
    positions_history = results['positions_history']
    rotations_history = results['rotations_history']
    bonds_history = results['bonds_history']
    paths_history = results['paths_history']
    local_polys = results['local_polys']
    bonds = results['bonds']
    thicknesses = results['thicknesses']
    field_width = results['field_width']
    field_height = results['field_height']
    goal = results['goal']
    
    n_frames = len(ship_history)
    n_elements = len(local_polys)
    
    # Colors by thickness
    t_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
    ice_colors = []
    for t in t_norm:
        r = int(240 - 100 * t)
        g = int(250 - 80 * t)
        b = int(255 - 40 * t)
        ice_colors.append(f'#{r:02x}{g:02x}{b:02x}')
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    def rotate_poly(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return poly @ R.T
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-10, field_width + 10)
        ax.set_ylim(-10, field_height + 10)
        ax.set_aspect('equal')
        ax.set_facecolor('#1A4D70')  # Ocean blue
        
        ship_x, ship_y, ship_psi = ship_history[frame]
        positions = positions_history[frame]
        rotations = rotations_history[frame]
        intact = bonds_history[frame]
        
        # Draw ice floes
        for i in range(n_elements):
            rotated = rotate_poly(local_polys[i], rotations[i])
            world_poly = rotated + positions[i]
            
            # Show broken bonds with different edge color
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
        
        # Draw planned paths
        for p in paths_history:
            if p is not None and len(p) > 1:
                ax.plot(p[:, 0], p[:, 1], 'cyan', linewidth=1.5, alpha=0.4, zorder=4)
        
        # Draw ship trajectory
        traj_x = [h[0] for h in ship_history[:frame+1]]
        traj_y = [h[1] for h in ship_history[:frame+1]]
        ax.plot(traj_x, traj_y, 'yellow', linewidth=2, alpha=0.8, zorder=5)
        
        # Draw ship
        ship_poly = create_ship_polygon(ship_x, ship_y, ship_psi, ship.length, ship.beam)
        ship_patch = MplPolygon(
            ship_poly, facecolor='#404040', edgecolor='#202020',
            linewidth=2, zorder=10
        )
        ax.add_patch(ship_patch)
        
        # Goal marker
        ax.plot(goal[0], goal[1], 'g*', markersize=20, zorder=11)
        
        # Stats
        n_broken = np.sum(~intact)
        dist_to_goal = np.sqrt((ship_x - goal[0])**2 + (ship_y - goal[1])**2)
        
        ax.set_title(
            f'DPDEM with MEDEA Satellite Ice Field\n'
            f'Frame {frame}/{n_frames-1} | Dist: {dist_to_goal:.0f}m | '
            f'Bonds: {n_broken}/{len(bonds)} broken',
            fontsize=13, fontweight='bold'
        )
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True, alpha=0.15, color='white', linestyle='--')
        
        return []
    
    print(f"  Rendering {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)
    
    print(f"  Saving to {output_file}...")
    anim.save(output_file, writer=PillowWriter(fps=fps), dpi=100)
    print("  Done!")
    
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    # Path to MEDEA data
    pickle_path = Path(__file__).parent / 'data' / 'ice_floes_MEDEA_fram_20100629.pkl'
    
    if not pickle_path.exists():
        print(f"ERROR: MEDEA data not found at {pickle_path}")
        return
    
    # Run simulation with MEDEA ice field and DPDEM physics
    # Use smaller scale for faster simulation
    results = run_simulation(
        pickle_path=pickle_path,
        meters_per_pixel=0.5,  # Smaller domain (~320m x 320m)
        ship_length=25.0,
        ship_beam=7.0,
        ship_mass=8e6,
        ice_thickness=1.0,
        n_frames=300,
        dt=0.05,
        replan_interval=150,
        max_floes=12,  # Fewer floes for faster tessellation and simulation
        seed=42,
    )
    
    create_animation(results, output_file='medea_dpdem.gif', fps=20)
    print("\nOutput: medea_dpdem.gif")


if __name__ == '__main__':
    main()
