#!/usr/bin/env python3
"""
DPDEM: Autonomous Ship with Lattice Planner and Ice Fracture

Uses the A* lattice planner from AUTO-IceNav to plan efficient paths
through ice, combined with DPDEM fracture physics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import Voronoi
import time

# Import planning modules
from dpdem.planning import create_planner
from dpdem.control.gnc import ssa


# ============================================================================
# Ice Properties
# ============================================================================

class IceProperties:
    ICE_DENSITY = 920
    WATER_DENSITY = 1025
    FLEXURAL_STRENGTH_BASE = 0.15e6
    YOUNGS_MODULUS = 3e9
    
    @staticmethod
    def flexural_strength(thickness):
        return IceProperties.FLEXURAL_STRENGTH_BASE * np.sqrt(thickness / 1.0)
    
    @staticmethod
    def breaking_force(thickness, width):
        sigma_f = IceProperties.flexural_strength(thickness)
        return sigma_f * width * thickness ** 2


# ============================================================================
# Geometry
# ============================================================================

def polygon_area(verts):
    n = len(verts)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i, 0] * verts[j, 1]
        area -= verts[j, 0] * verts[i, 1]
    return abs(area) / 2


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
        t = (x_rel - 0.6) / 0.4
        t = min(t, 1.0)
        local_half_beam = half_b * (1 - 0.9 * t ** 1.5)
        local_half_beam = max(local_half_beam, half_b * 0.02)
    else:
        local_half_beam = half_b
    
    if local_x < -half_l or local_x > half_l:
        return 0, 0, 0
    if abs(local_y) > local_half_beam:
        return 0, 0, 0
    
    dist_to_bow = half_l - local_x
    dist_to_stern = local_x + half_l
    dist_to_side = local_half_beam - abs(local_y)
    
    if x_rel > 0.6:
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
    
    c, s = np.cos(ship_heading), np.sin(ship_heading)
    nx = c * nx_local - s * ny_local
    ny = s * nx_local + c * ny_local
    
    return max(pen, 0), nx, ny


# ============================================================================
# Ice Field Generation
# ============================================================================

def create_bonded_ice_field(
    field_width, field_height,
    cell_size, ship_start_x, ship_start_y, clear_radius,
    thickness_mean=0.8, thickness_std=0.2, seed=42,
    ice_patches=None,  # Optional: list of (cx, cy, radius) for ice patches
):
    """Create Voronoi ice field with bonds."""
    np.random.seed(seed)
    
    nx = int(field_width / cell_size)
    ny = int(field_height / cell_size)
    dx = field_width / nx
    dy = field_height / ny
    
    points = []
    point_patch_ids = []  # Track which patch each point belongs to
    
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * dx + np.random.uniform(-0.35 * dx, 0.35 * dx)
            y = (j + 0.5) * dy + np.random.uniform(-0.35 * dy, 0.35 * dy)
            
            # Skip near ship start
            dist = np.sqrt((x - ship_start_x)**2 + (y - ship_start_y)**2)
            if dist < clear_radius:
                continue
            
            # If ice_patches specified, only keep points inside patches
            patch_id = -1
            if ice_patches is not None:
                for patch_idx, (cx, cy, radius) in enumerate(ice_patches):
                    if np.sqrt((x - cx)**2 + (y - cy)**2) < radius:
                        patch_id = patch_idx
                        break
                if patch_id < 0:
                    continue
            
            x = np.clip(x, 1, field_width - 1)
            y = np.clip(y, 1, field_height - 1)
            points.append([x, y])
            point_patch_ids.append(patch_id)
    
    points = np.array(points)
    n_actual = len(points)
    
    # Mirror for bounded Voronoi
    mirrors = []
    for dx_m in [-1, 0, 1]:
        for dy_m in [-1, 0, 1]:
            if dx_m == 0 and dy_m == 0:
                continue
            shifted = points.copy()
            shifted[:, 0] += dx_m * field_width
            shifted[:, 1] += dy_m * field_height
            mirrors.append(shifted)
    
    all_points = np.vstack([points] + mirrors)
    vor = Voronoi(all_points)
    
    polygons = []
    centroids = []
    valid_idx = []
    
    for i in range(n_actual):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) < 3:
            continue
        
        verts = vor.vertices[region]
        clipped = np.clip(verts, [0, 0], [field_width, field_height])
        
        if len(clipped) < 3:
            continue
        
        centroid = np.mean(clipped, axis=0)
        angles = np.arctan2(clipped[:, 1] - centroid[1], clipped[:, 0] - centroid[0])
        order = np.argsort(angles)
        clipped = clipped[order]
        
        area = polygon_area(clipped)
        if area < 1.0:
            continue
        
        # If ice_patches specified, only keep polygons whose centroid is inside a patch
        if ice_patches is not None:
            in_patch = False
            for cx, cy, radius in ice_patches:
                if np.sqrt((centroid[0] - cx)**2 + (centroid[1] - cy)**2) < radius:
                    in_patch = True
                    break
            if not in_patch:
                continue
        
        valid_idx.append(i)
        polygons.append(clipped)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    n_valid = len(polygons)
    
    # Create bonds
    idx_to_poly = {valid_idx[i]: i for i in range(n_valid)}
    
    # Map valid poly index to patch ID
    poly_patch_ids = []
    for vi in valid_idx:
        if ice_patches is not None and vi < len(point_patch_ids):
            poly_patch_ids.append(point_patch_ids[vi])
        else:
            poly_patch_ids.append(-1)
    
    bonds = set()
    
    for ridge_points in vor.ridge_points:
        i, j = ridge_points
        if i < n_actual and j < n_actual:
            if i in idx_to_poly and j in idx_to_poly:
                pi, pj = idx_to_poly[i], idx_to_poly[j]
                
                # Only bond cells within the SAME patch (or if no patches defined)
                if ice_patches is None or poly_patch_ids[pi] == poly_patch_ids[pj]:
                    bonds.add((min(pi, pj), max(pi, pj)))
    
    bonds = list(bonds)
    
    thicknesses = np.random.normal(thickness_mean, thickness_std, n_valid)
    thicknesses = np.clip(thicknesses, 0.5, thickness_mean * 2)
    
    areas = np.array([polygon_area(p) for p in polygons])
    masses = areas * thicknesses * IceProperties.ICE_DENSITY
    radii = np.sqrt(areas / np.pi)
    
    MAX_BOND_WIDTH = 4.0
    bond_strengths = np.zeros(len(bonds))
    rest_lengths = np.zeros(len(bonds))
    for b, (i, j) in enumerate(bonds):
        h = min(thicknesses[i], thicknesses[j])
        w = min((radii[i] + radii[j]) / 2, MAX_BOND_WIDTH)
        bond_strengths[b] = IceProperties.breaking_force(h, w)
        rest_lengths[b] = np.linalg.norm(centroids[i] - centroids[j])
    
    local_polys = [p - np.mean(p, axis=0) for p in polygons]
    
    print(f"Created ice field: {n_valid} cells, {len(bonds)} bonds")
    
    return {
        'polygons': polygons,
        'local_polys': local_polys,
        'centroids': centroids,
        'bonds': bonds,
        'bond_strengths': bond_strengths,
        'rest_lengths': rest_lengths,
        'thicknesses': thicknesses,
        'masses': masses,
        'radii': radii,
        'areas': areas,
    }


def find_connected_components(n, bonds, intact):
    """Find connected components."""
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
# Waypoint-Following Ship Controller
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
        self.waypoint_idx = 0
        self.lookahead_dist = 15.0  # Pure pursuit lookahead
        
    def set_path(self, path):
        """Set path to follow (Nx3 array: x, y, heading)."""
        if path is not None and len(path) > 0:
            self.waypoints = path
            self.waypoint_idx = 0
    
    def get_lookahead_point(self):
        """Get lookahead point on path for pure pursuit."""
        if self.waypoints is None or len(self.waypoints) == 0:
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
        cumulative_dist = 0
        for i in range(closest_idx, len(self.waypoints) - 1):
            segment_len = np.linalg.norm(
                self.waypoints[i+1, :2] - self.waypoints[i, :2]
            )
            if cumulative_dist + segment_len >= self.lookahead_dist:
                # Interpolate
                frac = (self.lookahead_dist - cumulative_dist) / segment_len
                return self.waypoints[i, :2] + frac * (
                    self.waypoints[i+1, :2] - self.waypoints[i, :2]
                )
            cumulative_dist += segment_len
        
        # Use last waypoint
        return self.waypoints[-1, :2]
    
    def get_control(self, dt):
        """Pure pursuit controller."""
        lookahead = self.get_lookahead_point()
        if lookahead is None:
            return np.zeros(3)
        
        dx = lookahead[0] - self.x
        dy = lookahead[1] - self.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 5.0:
            return np.zeros(3)
        
        # Desired heading
        psi_desired = np.arctan2(dy, dx)
        heading_error = ssa(psi_desired - self.psi)
        
        # Control gains - high heading gains for responsive turning
        Kp_forward = 5e6
        Kp_heading = 5e7  # Increased significantly for faster turning
        Kd_heading = 1e7  # Increased for better damping
        
        tau_surge = Kp_forward
        tau_yaw = Kp_heading * heading_error - Kd_heading * self.r
        
        return np.array([tau_surge, 0.0, tau_yaw])
    
    def apply_forces(self, control_tau, external_force, external_moment, dt):
        """Update dynamics."""
        c, s = np.cos(self.psi), np.sin(self.psi)
        
        F_ext_body_x = c * external_force[0] + s * external_force[1]
        F_ext_body_y = -s * external_force[0] + c * external_force[1]
        
        F_surge = control_tau[0] + F_ext_body_x
        F_sway = control_tau[1] + F_ext_body_y
        M_yaw = control_tau[2] + external_moment
        
        drag_surge = 0.02 * self.mass * self.u
        drag_sway = 0.10 * self.mass * self.v
        drag_yaw = 0.015 * self.I_zz * self.r  # Reduced for easier turning
        
        acc_u = (F_surge - drag_surge) / self.mass
        acc_v = (F_sway - drag_sway) / self.mass
        acc_r = (M_yaw - drag_yaw) / self.I_zz
        
        self.u += dt * acc_u
        self.v += dt * acc_v
        self.r += dt * acc_r
        
        max_speed = 8.0
        speed = np.sqrt(self.u**2 + self.v**2)
        if speed > max_speed:
            self.u *= max_speed / speed
            self.v *= max_speed / speed
        self.r = np.clip(self.r, -0.5, 0.5)  # Allow faster turning (~28 deg/s max)
        
        vx = c * self.u - s * self.v
        vy = s * self.u + c * self.v
        
        self.x += dt * vx
        self.y += dt * vy
        self.psi = (self.psi + dt * self.r) % (2 * np.pi)
    
    def get_polygon(self):
        return create_ship_polygon(self.x, self.y, self.psi, self.length, self.beam)


# ============================================================================
# Main Simulation
# ============================================================================

def run_simulation(
    field_width=140.0,
    field_height=50.0,
    cell_size=5.5,
    ship_length=28.0,
    ship_beam=8.0,
    ship_mass=10e6,
    n_frames=800,
    dt=0.05,
    replan_interval=50,
    seed=42,
):
    """Run simulation with path planning."""
    
    print("=" * 60)
    print("DPDEM with Lattice Planner Navigation")
    print("=" * 60)
    
    ship_start = (20.0, field_height / 2, 0.0)
    ship_goal = (field_width - 20.0, field_height / 2)
    
    # Create ship
    ship = PathFollowingShip(
        length=ship_length, beam=ship_beam, mass=ship_mass,
        x=ship_start[0], y=ship_start[1], heading=ship_start[2],
    )
    
    print(f"Ship: {ship_length}m x {ship_beam}m, mass={ship_mass/1e6:.1f}t")
    print(f"Start: ({ship_start[0]:.0f}, {ship_start[1]:.0f})")
    print(f"Goal: ({ship_goal[0]:.0f}, {ship_goal[1]:.0f})")
    
    # Create ice field with optional patches
    print("\nGenerating ice field...")
    
    # Define ice patches if field is wide enough (for sparse test)
    ice_patches = None
    if field_height >= 90:
        # Recreate layout similar to autonomous_navigation.gif
        # Scattered individual floes of varying sizes
        ice_patches = [
            # Upper area floes
            (45, 85, 8),
            (95, 90, 7),
            (175, 82, 9),
            
            # Upper-middle area  
            (55, 65, 10),
            (85, 70, 8),
            (110, 65, 9),
            (135, 62, 8),
            (155, 58, 10),
            (175, 65, 7),
            
            # Center area - some clustering
            (50, 45, 12),
            (65, 38, 9),
            (90, 52, 8),
            (105, 48, 9),
            (120, 55, 7),
            (130, 45, 8),
            (150, 50, 6),
            
            # Lower-middle area
            (75, 30, 8),
            (95, 25, 7),
            (140, 35, 9),
            (165, 40, 8),
            
            # Lower area
            (55, 18, 6),
            (85, 10, 8),
            (125, 15, 7),
        ]
        print(f"  Scattered ice field with {len(ice_patches)} individual floes")
    
    ice = create_bonded_ice_field(
        field_width, field_height,
        cell_size=cell_size,
        ship_start_x=ship_start[0],
        ship_start_y=ship_start[1],
        clear_radius=ship_length * 1.2,
        thickness_mean=0.8,
        thickness_std=0.2,
        seed=seed,
        ice_patches=ice_patches,
    )
    
    n = len(ice['polygons'])
    bonds = ice['bonds']
    
    # Create planner with LOW collision cost so it plans THROUGH ice
    print("\nInitializing lattice planner...")
    planner = create_planner(
        map_width=field_width,
        map_height=field_height,
        ship_length=ship_length,
        ship_width=ship_beam,
        ship_mass=ship_mass,
        turning_radius=ship_length * 0.6,  # Tighter turns for maneuvering
        resolution=2.5,  # Finer resolution
        num_headings=16,  # More heading options
        heuristic_weight=1.5,  # Less greedy - find better paths
    )
    
    # Collision cost - higher means planner prefers avoiding ice
    # Set high enough that going around is MUCH better than through
    planner.cost_map.collision_cost = 100.0  # Very high - strongly avoid ice
    planner.cost_map.mass_cost_scale = 1e-5  # Significant mass penalty
    planner.goal_tolerance = 12.0
    
    # Initialize state
    positions = ice['centroids'].copy()
    velocities = np.zeros((n, 2))
    rotations = np.zeros(n)
    angular_vels = np.zeros(n)
    bond_intact = np.ones(len(bonds), dtype=bool)
    
    # Plan initial path
    print("Planning initial path...")
    planner.update_cost_map(
        positions, ice['local_polys'], ice['masses'], rotations
    )
    path = planner.plan(
        (ship.x, ship.y, ship.psi),
        ship_goal,
        max_iterations=50000,
    )
    
    if path is not None:
        print(f"  Path found: {len(path)} waypoints, expanded {planner.nodes_expanded} nodes")
        ship.set_path(path)
    else:
        print("  No path found - using direct approach")
        # Fallback: direct line
        ship.set_path(np.array([
            [ship.x, ship.y, ship.psi],
            [ship_goal[0], ship_goal[1], 0.0],
        ]))
    
    # History
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
            planner.update_cost_map(
                positions, ice['local_polys'], ice['masses'], rotations
            )
            new_path = planner.plan(
                (ship.x, ship.y, ship.psi),
                ship_goal,
                max_iterations=10000,
            )
            if new_path is not None:
                ship.set_path(new_path)
                paths_history.append(new_path.copy())
        
        # Get control
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
            world_poly = ice['local_polys'][i] @ R.T + positions[i]
            
            max_pen = 0
            best_nx, best_ny = 0, 0
            
            for vx, vy in world_poly:
                pen, nx, ny = point_in_ship(vx, vy, ship.x, ship.y, ship.psi,
                                           ship.length, ship.beam)
                if pen > max_pen:
                    max_pen, best_nx, best_ny = pen, nx, ny
            
            n_verts = len(world_poly)
            for j in range(n_verts):
                v1 = world_poly[j]
                v2 = world_poly[(j + 1) % n_verts]
                mid = (v1 + v2) / 2
                pen, nx, ny = point_in_ship(mid[0], mid[1], ship.x, ship.y,
                                           ship.psi, ship.length, ship.beam)
                if pen > max_pen:
                    max_pen, best_nx, best_ny = pen, nx, ny
            
            if max_pen > 0:
                force_mag = (ship.mass / 1e6) * 8e4 * max_pen
                forces[i, 0] += force_mag * best_nx
                forces[i, 1] += force_mag * best_ny
                ship_collision_force[i] = force_mag
                
                ship_force[0] -= force_mag * best_nx * 0.4
                ship_force[1] -= force_mag * best_ny * 0.4
                
                rx = positions[i, 0] - ship.x
                ry = positions[i, 1] - ship.y
                ship_moment -= (rx * force_mag * best_ny - ry * force_mag * best_nx) * 0.4
                
                rel_y = positions[i, 1] - ship.y
                side_dir = 1 if rel_y > 0 else -1
                forces[i, 1] += (ship.mass / 1e6) * 3e4 * max_pen * side_dir
                angular_vels[i] += 0.01 * side_dir * min(max_pen, 1.0)
        
        # Ice-ice repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                min_dist = (ice['radii'][i] + ice['radii'][j]) * 0.9
                
                if dist < min_dist and dist > 0.1:
                    overlap = min_dist - dist
                    direction = d / dist
                    f = 1e5 * overlap ** 1.5 * direction
                    forces[i] -= f
                    forces[j] += f
        
        # Water drag
        for i in range(n):
            forces[i] -= 0.5 * velocities[i] * ice['masses'][i]
        
        # Move connected components
        new_velocities = velocities.copy()
        new_positions = positions.copy()
        
        for comp in components:
            if len(comp) == 1:
                i = comp[0]
                accel = forces[i] / ice['masses'][i]
                new_velocities[i] += accel * dt
                new_velocities[i] = np.clip(new_velocities[i], -15, 15)
                new_positions[i] += new_velocities[i] * dt
            else:
                total_mass = sum(ice['masses'][i] for i in comp)
                com = sum(ice['masses'][i] * positions[i] for i in comp) / total_mass
                total_force = sum(forces[i] for i in comp)
                comp_vel = sum(ice['masses'][i] * velocities[i] for i in comp) / total_mass
                
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
        
        # Stress propagation
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
                stretch = abs(dist - ice['rest_lengths'][b])
                
                avg_thickness = (ice['thicknesses'][i] + ice['thicknesses'][j]) / 2
                stiffness = IceProperties.YOUNGS_MODULUS * avg_thickness * 0.001
                bond_force = stiffness * stretch
                
                propagated_stress = max(element_stress[i], element_stress[j])
                bond_force += propagated_stress * 0.2
                
                collision_on_bond = ship_collision_force[i] > 0 or ship_collision_force[j] > 0
                if collision_on_bond:
                    effective_strength = ice['bond_strengths'][b] * 0.1
                elif propagated_stress > ice['bond_strengths'][b] * 0.5:
                    effective_strength = ice['bond_strengths'][b] * 0.6
                else:
                    effective_strength = ice['bond_strengths'][b]
                
                collision_force_on_bond = max(ship_collision_force[i], ship_collision_force[j])
                if collision_on_bond and collision_force_on_bond > effective_strength:
                    bond_intact[b] = False
                elif bond_force > effective_strength:
                    bond_intact[b] = False
        
        # Rotations
        angular_vels += np.random.randn(n) * 0.001
        angular_vels *= 0.85
        angular_vels = np.clip(angular_vels, -0.2, 0.2)
        rotations += angular_vels * dt
        
        # Boundaries
        positions[:, 0] = np.clip(positions[:, 0], 0, field_width)
        positions[:, 1] = np.clip(positions[:, 1], 0, field_height)
        
        # Boundary forces on ship
        y_margin = 12.0
        boundary_force = 0.0
        dist_to_bottom = ship.y - ship.beam/2
        dist_to_top = field_height - ship.beam/2 - ship.y
        
        if dist_to_bottom < y_margin:
            boundary_force += 5e5 * (y_margin - dist_to_bottom)
        if dist_to_top < y_margin:
            boundary_force -= 5e5 * (y_margin - dist_to_top)
        
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
                  f"dist={dist_to_goal:.1f}m, speed={speed:.2f}m/s, broken={n_broken}/{len(bonds)}")
        
        # Goal check
        dist_to_goal = np.sqrt((ship.x - ship_goal[0])**2 + (ship.y - ship_goal[1])**2)
        if dist_to_goal < 8.0:
            print(f"\nGoal reached at frame {frame}!")
            break
    
    sim_time = time.time() - t0
    n_broken = np.sum(~bonds_history[-1])
    print(f"\nSimulation: {sim_time:.1f}s")
    print(f"Bonds broken: {n_broken}/{len(bonds)} ({100*n_broken/len(bonds):.0f}%)")
    
    return {
        'ship_history': ship_history,
        'positions_history': positions_history,
        'rotations_history': rotations_history,
        'bonds_history': bonds_history,
        'paths_history': paths_history,
        'ice': ice,
        'ship': ship,
        'field_width': field_width,
        'field_height': field_height,
        'goal': ship_goal,
    }


def create_animation(results, output_file='planned_navigation.gif', fps=20):
    """Create animation showing path and fracture."""
    print("\nCreating animation...")
    
    ice = results['ice']
    ship = results['ship']
    ship_history = results['ship_history']
    positions_history = results['positions_history']
    rotations_history = results['rotations_history']
    bonds_history = results['bonds_history']
    paths_history = results['paths_history']
    bonds = ice['bonds']
    field_width = results['field_width']
    field_height = results['field_height']
    goal = results['goal']
    
    n_frames = len(ship_history)
    n_elements = len(ice['polygons'])
    
    thicknesses = ice['thicknesses']
    thickness_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
    ice_colors = []
    for t in thickness_norm:
        r = int(240 - 100 * t)
        g = int(250 - 80 * t)
        b = int(255 - 40 * t)
        ice_colors.append(f'#{r:02x}{g:02x}{b:02x}')
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    current_path = paths_history[0] if paths_history else None
    path_idx = [0]
    
    def rotate_poly(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return poly @ R.T
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-5, field_width + 5)
        ax.set_ylim(-5, field_height + 5)
        ax.set_aspect('equal')
        ax.set_facecolor('#1A4D70')
        
        ship_x, ship_y, ship_psi = ship_history[frame]
        positions = positions_history[frame]
        rotations = rotations_history[frame]
        intact = bonds_history[frame]
        
        # Draw ice
        for i in range(n_elements):
            rotated = rotate_poly(ice['local_polys'][i], rotations[i])
            world_poly = rotated + positions[i]
            
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
        
        # Draw planned path (if available)
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
        ax.plot(goal[0], goal[1], 'g*', markersize=15, zorder=11)
        
        # Stats
        n_broken = np.sum(~intact)
        dist_to_goal = np.sqrt((ship_x - goal[0])**2 + (ship_y - goal[1])**2)
        
        ax.set_title(
            f'DPDEM with Lattice Planner Navigation\n'
            f'Frame {frame}/{n_frames-1} | Dist: {dist_to_goal:.0f}m | '
            f'Bonds: {n_broken}/{len(bonds)} broken',
            fontsize=13, fontweight='bold'
        )
        ax.grid(True, alpha=0.15, color='white', linestyle='--')
        
        return []
    
    print(f"  Rendering {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)
    
    print(f"  Saving to {output_file}...")
    anim.save(output_file, writer=PillowWriter(fps=fps), dpi=100)
    print("  Done!")
    
    plt.close()


def simulate_3d_physics(positions_history, bonds_history, bonds, ship_history, 
                        ship_length, ship_beam, ship_draft, n_elements):
    """
    Add 3D physics: ice goes under ship hull when pushed by bow.
    Based on paper Fig. 17.
    """
    n_frames = len(positions_history)
    
    z_positions = [np.zeros(n_elements) for _ in range(n_frames)]
    tilts_x = [np.zeros(n_elements) for _ in range(n_frames)]
    tilts_y = [np.zeros(n_elements) for _ in range(n_frames)]
    
    for frame in range(1, n_frames):
        positions = positions_history[frame]
        intact = bonds_history[frame]
        ship_x, ship_y, ship_psi = ship_history[frame]
        
        prev_z = z_positions[frame - 1].copy()
        prev_tilt_x = tilts_x[frame - 1].copy()
        
        # Count broken bonds per element
        broken_count = np.zeros(n_elements)
        for b, (i, j) in enumerate(bonds):
            if not intact[b]:
                broken_count[i] += 1
                broken_count[j] += 1
        
        new_z = prev_z.copy()
        new_tilt_x = prev_tilt_x.copy()
        new_tilt_y = tilts_y[frame - 1].copy()
        
        # Ship bow position (in front of ship center)
        bow_x = ship_x + (ship_length / 2) * np.cos(ship_psi)
        bow_y = ship_y + (ship_length / 2) * np.sin(ship_psi)
        
        for i in range(n_elements):
            px, py = positions[i]
            
            # Distance from ship center, in ship frame
            dx = px - ship_x
            dy = py - ship_y
            c, s = np.cos(-ship_psi), np.sin(-ship_psi)
            local_x = c * dx - s * dy  # +x is toward bow
            local_y = s * dx + c * dy  # +y is port side
            
            # Distance from bow in ship frame
            dist_from_bow = ship_length/2 - local_x
            
            # Ice near bow gets pushed down
            if 0 < dist_from_bow < ship_length * 0.4 and abs(local_y) < ship_beam:
                push_factor = 1 - dist_from_bow / (ship_length * 0.4)
                
                if broken_count[i] > 0:
                    # Broken ice dives under hull
                    target_z = -ship_draft - push_factor * 2.0
                    new_z[i] += (target_z - new_z[i]) * 0.15
                    new_tilt_x[i] += (0.4 * push_factor - new_tilt_x[i]) * 0.1
                else:
                    # Intact ice bends at bow
                    new_tilt_x[i] += (0.2 * push_factor - new_tilt_x[i]) * 0.05
            
            elif dist_from_bow < 0:
                # Ice has passed the bow - under or beside hull
                if broken_count[i] > 0 and abs(local_y) < ship_beam * 0.8:
                    # Under hull - stay down
                    new_z[i] += (-ship_draft * 0.9 - new_z[i]) * 0.05
                elif new_z[i] < -0.5:
                    # Outside hull or floating back up
                    new_z[i] += (0 - new_z[i]) * 0.03
            
            # Damping
            new_tilt_x[i] *= 0.95
            new_tilt_y[i] *= 0.95
            
            # Buoyancy - float back if far from ship
            dist_to_ship = np.sqrt((px - ship_x)**2 + (py - ship_y)**2)
            if new_z[i] < -0.3 and dist_to_ship > ship_length:
                new_z[i] += 0.02
        
        z_positions[frame] = new_z
        tilts_x[frame] = new_tilt_x
        tilts_y[frame] = new_tilt_y
    
    return z_positions, tilts_x, tilts_y


def create_3d_animation(results, output_file='planned_navigation_3d.gif', fps=15):
    """Create 3D animation showing ice going under ship."""
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    print("\nCreating 3D animation...")
    
    ice = results['ice']
    ship = results['ship']
    ship_history = results['ship_history']
    positions_history = results['positions_history']
    rotations_history = results['rotations_history']
    bonds_history = results['bonds_history']
    bonds = ice['bonds']
    field_width = results['field_width']
    field_height = results['field_height']
    goal = results['goal']
    
    n_frames = len(ship_history)
    n_elements = len(ice['polygons'])
    
    # Compute 3D physics
    print("  Computing 3D ice physics...")
    ship_draft = 3.0  # Simplified draft
    z_positions, tilts_x, tilts_y = simulate_3d_physics(
        positions_history, bonds_history, bonds, ship_history,
        ship.length, ship.beam, ship_draft, n_elements
    )
    
    # Ice thickness for 3D rendering
    ice_thickness = ice['thicknesses']
    
    # Colors
    thickness_norm = (ice_thickness - ice_thickness.min()) / (ice_thickness.max() - ice_thickness.min() + 0.01)
    ice_colors = []
    for t in thickness_norm:
        r = int(230 - 80 * t)
        g = int(245 - 60 * t)
        b = int(255 - 30 * t)
        ice_colors.append(f'#{r:02x}{g:02x}{b:02x}')
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def rotate_poly(poly, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return poly @ R.T
    
    # Subsample frames for faster 3D rendering
    frame_skip = 3
    frames_to_render = list(range(0, n_frames, frame_skip))
    
    def update(frame_idx):
        frame = frames_to_render[frame_idx]
        ax.clear()
        
        ship_x, ship_y, ship_psi = ship_history[frame]
        positions = positions_history[frame]
        rotations = rotations_history[frame]
        intact = bonds_history[frame]
        z_pos = z_positions[frame]
        tilt_x = tilts_x[frame]
        tilt_y = tilts_y[frame]
        
        ax.set_xlim(-5, field_width + 5)
        ax.set_ylim(-5, field_height + 5)
        ax.set_zlim(-8, 5)
        ax.set_box_aspect([field_width, field_height, 13 * 3])  # Exaggerate Z
        
        ax.view_init(elev=30, azim=-60 + frame * 0.15)
        
        # Draw ice as 3D prisms
        for i in range(n_elements):
            rotated = rotate_poly(ice['local_polys'][i], rotations[i])
            world_poly = rotated + positions[i]
            
            z_center = z_pos[i]
            tx, ty = tilt_x[i], tilt_y[i]
            h = ice_thickness[i]
            
            # Create 3D vertices
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
            
            has_breaks = any(not intact[b] for b, (pi, pj) in enumerate(bonds)
                           if pi == i or pj == i)
            edge_color = '#2080A0' if has_breaks else '#A0C8D8'
            
            # Draw top face
            ax.add_collection3d(Poly3DCollection(
                [top_verts], facecolor=ice_colors[i],
                edgecolor=edge_color, linewidth=0.5 if has_breaks else 0.2,
                alpha=0.9
            ))
            
            # Draw side faces
            for j in range(n_verts):
                k = (j + 1) % n_verts
                side = np.array([bottom_verts[j], bottom_verts[k], 
                               top_verts[k], top_verts[j]])
                ax.add_collection3d(Poly3DCollection(
                    [side], facecolor='#C8E0F0', edgecolor=edge_color,
                    linewidth=0.1, alpha=0.8
                ))
        
        # Draw ship as 3D box
        c, s = np.cos(ship_psi), np.sin(ship_psi)
        half_l, half_b = ship.length / 2, ship.beam / 2
        
        # Ship corners in local frame
        corners_local = np.array([
            [half_l, half_b], [half_l, -half_b],
            [-half_l, -half_b], [-half_l, half_b]
        ])
        
        # Taper bow
        taper_corners = []
        for cl in corners_local:
            if cl[0] > 0:  # Bow end
                taper = max(0.3, 1 - 0.7 * (cl[0] / half_l))
                taper_corners.append([cl[0], cl[1] * taper])
            else:
                taper_corners.append(cl)
        taper_corners = np.array(taper_corners)
        
        # Rotate and translate
        R = np.array([[c, -s], [s, c]])
        ship_corners = taper_corners @ R.T + np.array([ship_x, ship_y])
        
        # Ship 3D (deck at z=1, keel at z=-2)
        deck_z = 1.5
        keel_z = -ship_draft
        
        ship_top = np.column_stack([ship_corners, np.full(4, deck_z)])
        ship_bot = np.column_stack([ship_corners, np.full(4, keel_z)])
        
        ax.add_collection3d(Poly3DCollection(
            [ship_top], facecolor='#606060', edgecolor='#303030',
            linewidth=1, alpha=0.95, zorder=10
        ))
        ax.add_collection3d(Poly3DCollection(
            [ship_bot], facecolor='#800000', edgecolor='#400000',
            linewidth=1, alpha=0.9, zorder=10
        ))
        
        # Ship sides
        for j in range(4):
            k = (j + 1) % 4
            side = np.array([ship_bot[j], ship_bot[k], ship_top[k], ship_top[j]])
            color = '#808080' if side[0, 2] > 0 else '#700000'
            ax.add_collection3d(Poly3DCollection(
                [side], facecolor=color, edgecolor='#303030',
                linewidth=0.5, alpha=0.9, zorder=10
            ))
        
        # Goal marker
        ax.scatter([goal[0]], [goal[1]], [2], c='green', s=100, marker='*', zorder=11)
        
        n_broken = np.sum(~intact)
        ax.set_title(
            f'DPDEM 3D with Lattice Planner\n'
            f'Frame {frame}/{n_frames-1} | Bonds broken: {n_broken}/{len(bonds)}',
            fontsize=12
        )
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        return []
    
    print(f"  Rendering {len(frames_to_render)} frames...")
    anim = FuncAnimation(fig, update, frames=len(frames_to_render), 
                        interval=1000//fps, blit=False)
    
    print(f"  Saving to {output_file}...")
    anim.save(output_file, writer=PillowWriter(fps=fps), dpi=80)
    print("  Done!")
    
    plt.close()


def main():
    # Scattered ice floes similar to autonomous_navigation.gif layout
    results = run_simulation(
        field_width=200.0,
        field_height=100.0,
        cell_size=8.0,  # Medium cells for varied floe sizes
        ship_length=25.0,
        ship_beam=7.0,
        ship_mass=8e6,
        n_frames=500,
        dt=0.05,
        replan_interval=50,
        seed=42,
    )
    
    create_animation(results, output_file='planned_navigation_sparse.gif', fps=20)
    
    print("\nOutput: planned_navigation_sparse.gif")


if __name__ == '__main__':
    main()
