#!/usr/bin/env python3
"""
DPDEM Autonomous Ship Navigation Through Ice Field

Combines:
- DPDEM physics simulation (ice contact, fracture, bond mechanics)
- Autonomous navigation (lattice planner + DP controller)

The ship actively navigates through the ice field using path planning
and trajectory tracking, while the DPDEM physics handles ice-ship interactions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi

# Import DPDEM navigation modules
from dpdem.control import ShipState, DPController
from dpdem.control.gnc import Rxy, ssa
from dpdem.planning import create_planner


# ============================================================================
# Ice Material Properties
# ============================================================================

class IceProperties:
    """Physical properties of sea ice."""
    ICE_DENSITY = 920  # kg/m³
    WATER_DENSITY = 1025  # kg/m³
    FLEXURAL_STRENGTH_BASE = 0.5e6  # Pa
    YOUNGS_MODULUS = 5e9  # Pa
    
    @staticmethod
    def flexural_strength(thickness):
        h_ref = 1.0
        return IceProperties.FLEXURAL_STRENGTH_BASE * np.sqrt(thickness / h_ref)
    
    @staticmethod
    def breaking_force(thickness, width):
        sigma_f = IceProperties.flexural_strength(thickness)
        return sigma_f * width * thickness ** 2


# ============================================================================
# Geometry Utilities
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
    """Create ship polygon with given pose."""
    # Ship shape (pointed bow)
    half_l = length / 2
    half_b = beam / 2
    
    # Local vertices (bow at +x in body frame)
    local = np.array([
        [half_l, 0],                    # Bow point
        [half_l * 0.7, half_b],         # Port bow
        [-half_l, half_b],              # Port stern
        [-half_l, -half_b],             # Starboard stern
        [half_l * 0.7, -half_b],        # Starboard bow
    ])
    
    # Rotate and translate
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    world = local @ R.T + np.array([cx, cy])
    
    return world


def point_in_ship(px, py, ship_cx, ship_cy, ship_heading, ship_length, ship_beam):
    """Check if point is inside ship hull and compute penetration."""
    # Transform point to ship's local frame
    c, s = np.cos(-ship_heading), np.sin(-ship_heading)
    dx = px - ship_cx
    dy = py - ship_cy
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    
    half_l = ship_length / 2
    half_b = ship_beam / 2
    
    # Relative position in ship frame
    x_rel = local_x / half_l  # -1 = stern, +1 = bow
    
    # Compute local beam (tapers at bow)
    if x_rel > 0.6:  # Bow region
        t = (x_rel - 0.6) / 0.4
        t = min(t, 1.0)
        local_half_beam = half_b * (1 - 0.9 * t ** 1.5)
        local_half_beam = max(local_half_beam, half_b * 0.05)
    else:
        local_half_beam = half_b
    
    # Check bounds
    if local_x < -half_l or local_x > half_l:
        return 0, 0, 0
    if abs(local_y) > local_half_beam:
        return 0, 0, 0
    
    # Inside ship - compute penetration and normal
    dist_to_bow = half_l - local_x
    dist_to_stern = local_x + half_l
    dist_to_side = local_half_beam - abs(local_y)
    
    # Push to nearest edge (in local frame)
    if x_rel > 0.6:
        # Bow region - push along bow angle
        t = (x_rel - 0.6) / 0.4
        bow_angle = 0.3 + 0.5 * min(t, 1.0)
        if local_y >= 0:
            nx_local, ny_local = np.cos(bow_angle), np.sin(bow_angle)
        else:
            nx_local, ny_local = np.cos(bow_angle), -np.sin(bow_angle)
        pen = min(dist_to_bow, dist_to_side) + 0.5
    elif dist_to_bow < dist_to_side and dist_to_bow < dist_to_stern:
        nx_local, ny_local = 1, 0  # Push toward bow
        pen = dist_to_bow
    elif dist_to_stern < dist_to_side:
        nx_local, ny_local = -1, 0  # Push toward stern
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
# Ice Field Generation
# ============================================================================

def create_voronoi_ice_field(
    n_elements, field_width, field_height,
    ship_start_x, ship_start_y, clear_radius,
    seed=42, thickness_mean=1.5, thickness_std=0.3,
    ice_band_start=None, ice_band_end=None,
):
    """Create dense Voronoi ice field with solid ridge that ship must break through."""
    np.random.seed(seed)
    
    # Ice band covers the middle - includes a solid ridge
    if ice_band_start is None:
        ice_band_start = field_width * 0.30
    if ice_band_end is None:
        ice_band_end = field_width * 0.85
    
    points = []
    
    # Create a solid ridge at a specific x position - very tight vertical spacing
    ridge_x = field_width * 0.45  # Ridge position
    ridge_width = 8.0  # Width of the ridge zone
    n_ridge = int(field_height / 2.5)  # Dense vertical coverage
    
    for j in range(n_ridge):
        y = (j + 0.5) * field_height / n_ridge
        # Multiple columns in the ridge
        for dx_offset in [-3, 0, 3]:
            x = ridge_x + dx_offset + np.random.uniform(-0.5, 0.5)
            points.append([x, y])
    
    # Create a second ridge
    ridge_x2 = field_width * 0.65
    for j in range(n_ridge):
        y = (j + 0.5) * field_height / n_ridge
        for dx_offset in [-2, 0, 2]:
            x = ridge_x2 + dx_offset + np.random.uniform(-0.5, 0.5)
            points.append([x, y])
    
    # Add scattered ice in between ridges
    n_scattered = n_elements - len(points)
    for _ in range(max(0, n_scattered)):
        x = np.random.uniform(ice_band_start, ice_band_end)
        y = np.random.uniform(3, field_height - 3)
        
        # Skip near ship start
        dist = np.sqrt((x - ship_start_x)**2 + (y - ship_start_y)**2)
        if dist < clear_radius:
            continue
        
        points.append([x, y])
    
    points = np.array(points)
    n_actual = len(points)
    
    # Mirror points for bounded Voronoi
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
    
    # Extract polygons
    polygons = []
    centroids = []
    valid_idx = []
    
    for i in range(n_actual):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) < 3:
            continue
        
        verts = vor.vertices[region]
        
        # Clip to domain
        clipped = np.clip(verts, [0, 0], [field_width, field_height])
        
        if len(clipped) < 3:
            continue
        
        # Order vertices CCW
        centroid = np.mean(clipped, axis=0)
        angles = np.arctan2(clipped[:, 1] - centroid[1], clipped[:, 0] - centroid[0])
        order = np.argsort(angles)
        clipped = clipped[order]
        
        area = polygon_area(clipped)
        if area < 0.5:
            continue
        
        valid_idx.append(i)
        polygons.append(clipped)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    n_valid = len(polygons)
    
    # Create bonds
    idx_to_poly = {valid_idx[i]: i for i in range(n_valid)}
    bonds = set()
    
    for ridge_points in vor.ridge_points:
        i, j = ridge_points
        if i < n_actual and j < n_actual:
            if i in idx_to_poly and j in idx_to_poly:
                pi, pj = idx_to_poly[i], idx_to_poly[j]
                bonds.add((min(pi, pj), max(pi, pj)))
    
    bonds = list(bonds)
    
    # Variable thickness
    thicknesses = np.random.normal(thickness_mean, thickness_std, n_valid)
    thicknesses = np.clip(thicknesses, 0.5, thickness_mean * 2)
    
    print(f"Created ice field: {n_valid} elements, {len(bonds)} bonds")
    
    return polygons, centroids, bonds, thicknesses


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


# ============================================================================
# Autonomous Ship with DPDEM Physics
# ============================================================================

class AutonomousShip:
    """Ship with autonomous navigation and physics."""
    
    def __init__(
        self,
        length, beam, mass,
        start_x, start_y, start_heading,
        controller_wn=None, controller_zeta=None,
    ):
        self.length = length
        self.beam = beam
        self.mass = mass
        
        # Inertia
        self.I_zz = (1/12) * mass * (length**2 + beam**2)
        
        # State
        self.x = start_x
        self.y = start_y
        self.psi = start_heading
        self.u = 0.0  # surge velocity
        self.v = 0.0  # sway velocity
        self.r = 0.0  # yaw rate
        
        # Controller - higher gains for faster response
        if controller_wn is None:
            controller_wn = np.array([0.3, 0.3, 0.25])
        if controller_zeta is None:
            controller_zeta = np.array([0.7, 0.7, 0.7])
        
        self.controller = DPController.from_ship_params(
            mass=mass, length=length, breadth=beam,
            wn=controller_wn, zeta=controller_zeta,
        )
        
        # Planner (will be set up later)
        self.planner = None
        self.path = None
        self.setpoint = np.array([start_x, start_y, start_heading])
        self.goal = None
    
    @property
    def eta(self):
        return np.array([self.x, self.y, self.psi])
    
    @property
    def nu(self):
        return np.array([self.u, self.v, self.r])
    
    def setup_planner(self, map_width, map_height, turning_radius=30.0):
        """Initialize the path planner."""
        self.planner = create_planner(
            map_width=map_width,
            map_height=map_height,
            ship_length=self.length,
            ship_width=self.beam,
            ship_mass=self.mass,
            turning_radius=turning_radius,
            resolution=2.0,
            num_headings=16,
            heuristic_weight=2.0,  # More aggressive toward goal
        )
        # Lower collision cost so ship is willing to break through thin ice
        self.planner.cost_map.collision_cost = 30.0
    
    def update_planner_costmap(self, positions, polygons, masses, angles):
        """Update the planner's cost map with current ice positions."""
        if self.planner is not None:
            self.planner.update_cost_map(positions, polygons, masses, angles)
    
    def plan_path(self, goal):
        """Plan a path to the goal."""
        self.goal = np.array(goal)
        
        if self.planner is None:
            return
        
        start = (self.x, self.y, self.psi)
        path = self.planner.plan(start, tuple(goal), max_iterations=5000)
        
        if path is not None and len(path) > 1:
            self.path = path
            print(f"  Path found: {len(path)} waypoints, {self.planner.nodes_expanded} nodes expanded")
        else:
            # Direct path
            self.path = np.array([
                [self.x, self.y, self.psi],
                [goal[0], goal[1], np.arctan2(goal[1] - self.y, goal[0] - self.x)]
            ])
    
    def get_control(self, dt):
        """Compute control forces from controller."""
        if self.goal is None:
            return np.zeros(3)
        
        # Distance to goal
        dist_to_goal = np.sqrt((self.x - self.goal[0])**2 + (self.y - self.goal[1])**2)
        
        # Update setpoint from path
        if self.path is not None and len(self.path) > 1:
            # Find closest point on path ahead of us
            min_dist = np.inf
            best_idx = 0
            
            for i, wp in enumerate(self.path):
                dist = np.sqrt((wp[0] - self.x)**2 + (wp[1] - self.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            # Look further ahead on path for smoother tracking
            lookahead_idx = min(best_idx + 5, len(self.path) - 1)
            self.setpoint = self.path[lookahead_idx].copy()
        else:
            # Direct to goal
            psi_desired = np.arctan2(self.goal[1] - self.y, self.goal[0] - self.x)
            self.setpoint = np.array([self.goal[0], self.goal[1], psi_desired])
        
        # Compute control
        tau = self.controller.compute_control(self.eta, self.nu, self.setpoint, dt)
        
        # Add constant forward thrust to maintain momentum
        if dist_to_goal > 10.0:
            # Desired heading toward setpoint
            psi_desired = np.arctan2(self.setpoint[1] - self.y, self.setpoint[0] - self.x)
            heading_error = abs(ssa(self.psi - psi_desired))
            
            # Forward thrust (reduce if heading is off)
            if heading_error < 0.5:  # ~30 degrees
                forward_boost = 3e5 * (1.0 - heading_error)
                tau[0] += forward_boost
        
        # Saturate - allow higher forces for icebreaking
        max_force = 3e6
        max_moment = 8e6
        tau[0] = np.clip(tau[0], -max_force, max_force)
        tau[1] = np.clip(tau[1], -max_force, max_force)
        tau[2] = np.clip(tau[2], -max_moment, max_moment)
        
        return tau
    
    def apply_forces(self, control_tau, external_force, external_moment, dt):
        """Apply forces and update dynamics."""
        # Transform control from body to global
        c, s = np.cos(self.psi), np.sin(self.psi)
        
        # Total forces in body frame
        F_body = np.array([control_tau[0], control_tau[1]])
        
        # Add external forces (already in global frame) - transform to body
        R_inv = np.array([[c, s], [-s, c]])
        F_external_body = R_inv @ external_force[:2]
        F_body += F_external_body
        
        # Drag
        drag_u = 0.1 * self.mass * self.u
        drag_v = 0.2 * self.mass * self.v
        drag_r = 0.1 * self.I_zz * self.r
        
        # Accelerations in body frame
        acc_u = (F_body[0] - drag_u) / self.mass
        acc_v = (F_body[1] - drag_v) / self.mass
        acc_r = (control_tau[2] + external_moment - drag_r) / self.I_zz
        
        # Update velocities
        self.u += dt * acc_u
        self.v += dt * acc_v
        self.r += dt * acc_r
        
        # Clamp velocities
        max_speed = 8.0
        speed = np.sqrt(self.u**2 + self.v**2)
        if speed > max_speed:
            self.u *= max_speed / speed
            self.v *= max_speed / speed
        self.r = np.clip(self.r, -0.5, 0.5)
        
        # Update position (body to global)
        vx = c * self.u - s * self.v
        vy = s * self.u + c * self.v
        
        self.x += dt * vx
        self.y += dt * vy
        self.psi = (self.psi + dt * self.r) % (2 * np.pi)
    
    def get_polygon(self):
        """Get ship polygon for visualization."""
        return create_ship_polygon(self.x, self.y, self.psi, self.length, self.beam)


# ============================================================================
# Simulation
# ============================================================================

def run_simulation(
    field_width=150.0,
    field_height=80.0,
    n_ice_elements=80,
    ship_length=25.0,
    ship_beam=6.0,
    ship_mass=5e6,
    n_frames=600,
    dt=0.05,
    seed=42,
    replan_interval=50,
):
    """Run the autonomous ship navigation simulation."""
    
    print("=" * 60)
    print("DPDEM Autonomous Ship Navigation")
    print("=" * 60)
    
    # Ship start and goal
    ship_start = (20.0, field_height / 2)
    ship_goal = (field_width - 20.0, field_height / 2)
    
    # Create ship
    ship = AutonomousShip(
        length=ship_length,
        beam=ship_beam,
        mass=ship_mass,
        start_x=ship_start[0],
        start_y=ship_start[1],
        start_heading=0.0,
    )
    ship.setup_planner(field_width, field_height, turning_radius=ship_length * 1.5)
    
    print(f"Ship: {ship_length}m x {ship_beam}m, mass={ship_mass/1e6:.1f}t")
    print(f"Start: {ship_start}, Goal: {ship_goal}")
    
    # Create ice field (cleared area around ship start for initial movement)
    print("\nGenerating ice field...")
    polygons, centroids, bonds, thicknesses = create_voronoi_ice_field(
        n_ice_elements, field_width, field_height,
        ship_start[0], ship_start[1], clear_radius=ship_length * 1.2,
        seed=seed, thickness_mean=0.8, thickness_std=0.15,
    )
    
    n = len(polygons)
    
    # Compute ice properties
    areas = np.array([polygon_area(p) for p in polygons])
    masses = areas * thicknesses * IceProperties.ICE_DENSITY
    radii = np.sqrt(areas / np.pi)
    
    # Local polygons (centered)
    local_polys = [p - np.mean(p, axis=0) for p in polygons]
    
    # Ice state
    positions = centroids.copy()
    velocities = np.zeros((n, 2))
    rotations = np.zeros(n)
    angular_vels = np.zeros(n)
    
    # Bond state
    bond_intact = np.ones(len(bonds), dtype=bool)
    rest_lengths = np.array([np.linalg.norm(centroids[i] - centroids[j]) for i, j in bonds])
    
    # Bond strengths
    bond_strengths = np.zeros(len(bonds))
    for b, (i, j) in enumerate(bonds):
        h = min(thicknesses[i], thicknesses[j])
        w = (radii[i] + radii[j]) / 2
        bond_strengths[b] = IceProperties.breaking_force(h, w)
    
    # History
    ship_history = [(ship.x, ship.y, ship.psi)]
    positions_history = [positions.copy()]
    rotations_history = [rotations.copy()]
    bonds_history = [bond_intact.copy()]
    path_history = [None]
    
    # Initial planning
    print("\nPlanning initial path...")
    ship.update_planner_costmap(positions, local_polys, masses, rotations)
    ship.plan_path(ship_goal)
    
    print("\nRunning simulation...")
    
    for frame in range(n_frames):
        # Replan periodically
        if frame > 0 and frame % replan_interval == 0:
            ship.update_planner_costmap(positions, local_polys, masses, rotations)
            ship.plan_path(ship_goal)
        
        # Get ship control
        control_tau = ship.get_control(dt)
        
        # Compute ice-ship and ice-ice forces
        ice_forces = np.zeros((n, 2))
        ship_force = np.zeros(2)
        ship_moment = 0.0
        ship_collision_force = np.zeros(n)
        
        # Ship-ice collisions
        for i in range(n):
            px, py = positions[i]
            
            # Check collision
            pen, nx, ny = point_in_ship(
                px, py, ship.x, ship.y, ship.psi, ship.length, ship.beam
            )
            
            # Also check element boundary points
            if pen <= 0:
                for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    test_x = px + radii[i] * 0.7 * np.cos(angle)
                    test_y = py + radii[i] * 0.7 * np.sin(angle)
                    pen, nx, ny = point_in_ship(
                        test_x, test_y, ship.x, ship.y, ship.psi, ship.length, ship.beam
                    )
                    if pen > 0:
                        break
            
            if pen > 0:
                # Contact force magnitude
                force_mag = 1e5 * pen ** 1.5
                
                # Force on ice (push away from ship)
                ice_forces[i, 0] += force_mag * nx
                ice_forces[i, 1] += force_mag * ny
                
                # Reaction on ship
                ship_force[0] -= force_mag * nx
                ship_force[1] -= force_mag * ny
                
                # Moment on ship from off-center collision
                rx = px - ship.x
                ry = py - ship.y
                ship_moment -= rx * force_mag * ny - ry * force_mag * nx
                
                ship_collision_force[i] = force_mag
                
                # Angular kick to ice
                angular_vels[i] += 0.1 * np.sign(ny) * pen
        
        # Ice-ice repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 0.1:
                    overlap = min_dist - dist
                    direction = d / dist
                    f = 1e5 * overlap ** 1.5 * direction
                    ice_forces[i] -= f
                    ice_forces[j] += f
        
        # Water drag on ice
        ice_forces -= 0.3 * velocities * masses[:, None]
        
        # Stress propagation through bonds
        element_stress = ship_collision_force.copy()
        for _ in range(2):
            new_stress = element_stress.copy()
            for b, (i, j) in enumerate(bonds):
                if bond_intact[b]:
                    transfer = 0.3
                    if element_stress[i] > element_stress[j]:
                        new_stress[j] += element_stress[i] * transfer
                    else:
                        new_stress[i] += element_stress[j] * transfer
            element_stress = new_stress
        
        # Bond fracture
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
            
            # Effective strength
            if ship_collision_force[i] > 0 or ship_collision_force[j] > 0:
                effective_strength = bond_strengths[b] * 0.25
            elif propagated_stress > bond_strengths[b] * 0.5:
                effective_strength = bond_strengths[b] * 0.6
            else:
                effective_strength = bond_strengths[b]
            
            if bond_force > effective_strength:
                bond_intact[b] = False
        
        # Update ice dynamics
        components = find_connected_components(n, bonds, bond_intact)
        
        new_velocities = velocities.copy()
        new_positions = positions.copy()
        
        for comp in components:
            if len(comp) == 1:
                i = comp[0]
                accel = ice_forces[i] / masses[i]
                new_velocities[i] += accel * dt
                new_velocities[i] = np.clip(new_velocities[i], -10, 10)
                new_positions[i] += new_velocities[i] * dt
            else:
                total_mass = sum(masses[i] for i in comp)
                com = sum(masses[i] * positions[i] for i in comp) / total_mass
                total_force = sum(ice_forces[i] for i in comp)
                comp_vel = sum(masses[i] * velocities[i] for i in comp) / total_mass
                
                accel = total_force / total_mass
                comp_vel += accel * dt
                comp_vel = np.clip(comp_vel, -10, 10)
                
                for i in comp:
                    new_velocities[i] = comp_vel
                    rel_pos = positions[i] - com
                    new_com = com + comp_vel * dt
                    new_positions[i] = new_com + rel_pos
        
        velocities = new_velocities
        positions = new_positions
        
        # Update ice rotations
        angular_vels += np.random.randn(n) * 0.003
        angular_vels *= 0.97
        rotations += angular_vels * dt
        
        # Clamp ice positions to domain
        positions[:, 0] = np.clip(positions[:, 0], 0, field_width)
        positions[:, 1] = np.clip(positions[:, 1], 0, field_height)
        
        # Update ship
        ship.apply_forces(control_tau, ship_force, ship_moment, dt)
        
        # Clamp ship to domain
        ship.x = np.clip(ship.x, ship.length/2, field_width - ship.length/2)
        ship.y = np.clip(ship.y, ship.beam/2, field_height - ship.beam/2)
        
        # Store history
        ship_history.append((ship.x, ship.y, ship.psi))
        positions_history.append(positions.copy())
        rotations_history.append(rotations.copy())
        bonds_history.append(bond_intact.copy())
        path_history.append(ship.path.copy() if ship.path is not None else None)
        
        # Progress
        if frame % 100 == 0:
            dist_to_goal = np.sqrt((ship.x - ship_goal[0])**2 + (ship.y - ship_goal[1])**2)
            n_broken = np.sum(~bond_intact)
            speed = np.sqrt(ship.u**2 + ship.v**2)
            print(f"  Frame {frame}: pos=({ship.x:.1f}, {ship.y:.1f}), "
                  f"dist={dist_to_goal:.1f}m, speed={speed:.2f}m/s, broken={n_broken}/{len(bonds)}")
        
        # Check if goal reached
        dist_to_goal = np.sqrt((ship.x - ship_goal[0])**2 + (ship.y - ship_goal[1])**2)
        if dist_to_goal < 5.0:
            print(f"\nGoal reached at frame {frame}!")
            break
    
    return (ship_history, positions_history, rotations_history, bonds_history,
            path_history, polygons, local_polys, bonds, thicknesses,
            field_width, field_height, ship_goal, ship)


def create_animation(
    ship_history, positions_history, rotations_history, bonds_history,
    path_history, polygons, local_polys, bonds, thicknesses,
    field_width, field_height, ship_goal, ship,
    output_file='autonomous_dpdem.gif', fps=20,
):
    """Create animation of the simulation."""
    print("\nCreating animation...")
    
    n_frames = len(ship_history)
    n_elements = len(polygons)
    
    # Color by thickness
    thickness_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
    ice_colors = []
    for t in thickness_norm:
        r = int(220 - 80 * t)
        g = int(235 - 60 * t)
        b = int(255 - 30 * t)
        ice_colors.append(f'#{r:02x}{g:02x}{b:02x}')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
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
        path = path_history[frame]
        
        # Draw ice
        for i in range(n_elements):
            rotated = rotate_poly(local_polys[i], rotations[i])
            world_poly = rotated + positions[i]
            
            # Check if this element has broken bonds
            has_breaks = any(not intact[b] for b, (pi, pj) in enumerate(bonds) 
                           if pi == i or pj == i)
            
            edge_color = '#2A5D80' if has_breaks else ice_colors[i]
            edge_width = 1.0 if has_breaks else 0.3
            
            patch = MplPolygon(
                world_poly, facecolor=ice_colors[i],
                edgecolor=edge_color, linewidth=edge_width,
                alpha=0.95, zorder=3
            )
            ax.add_patch(patch)
        
        # Draw planned path
        if path is not None and len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], 'g--', linewidth=1.5, 
                   alpha=0.6, zorder=5, label='Planned path')
        
        # Draw ship trajectory so far
        traj_x = [h[0] for h in ship_history[:frame+1]]
        traj_y = [h[1] for h in ship_history[:frame+1]]
        ax.plot(traj_x, traj_y, 'yellow', linewidth=2, alpha=0.7, zorder=6)
        
        # Draw ship
        ship_poly = create_ship_polygon(ship_x, ship_y, ship_psi, ship.length, ship.beam)
        ship_patch = MplPolygon(
            ship_poly, facecolor='#404040',
            edgecolor='#202020', linewidth=2,
            zorder=10
        )
        ax.add_patch(ship_patch)
        
        # Draw goal
        ax.plot(ship_goal[0], ship_goal[1], 'g*', markersize=15, zorder=11, label='Goal')
        
        # Stats
        n_intact = int(intact.sum())
        n_broken = len(bonds) - n_intact
        dist_to_goal = np.sqrt((ship_x - ship_goal[0])**2 + (ship_y - ship_goal[1])**2)
        
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Y [m]', fontsize=11)
        ax.set_title(
            f'DPDEM Autonomous Ship Navigation\n'
            f'Frame {frame}/{n_frames-1} | '
            f'Dist to goal: {dist_to_goal:.1f}m | '
            f'Broken bonds: {n_broken}/{len(bonds)}',
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


def main():
    # Run simulation - narrow field forces ship through ice
    results = run_simulation(
        field_width=100.0,
        field_height=35.0,  # Narrow field
        n_ice_elements=150,
        ship_length=12.0,
        ship_beam=3.5,
        ship_mass=1.5e6,
        n_frames=800,  # Longer to reach goal
        dt=0.05,
        seed=456,
        replan_interval=40,
    )
    
    # Create animation
    create_animation(*results, output_file='autonomous_dpdem.gif', fps=20)
    
    print("\nSimulation complete!")
    print("Output: autonomous_dpdem.gif")


if __name__ == '__main__':
    main()
