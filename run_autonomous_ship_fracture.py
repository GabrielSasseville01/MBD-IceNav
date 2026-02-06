#!/usr/bin/env python3
"""
DPDEM: Autonomous Ship Navigation with Ice Fracture

Based on run_medea_ship_fracture.py physics, but with autonomous ship navigation.
The ship actively plans and navigates through ice, breaking bonds as needed.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import Voronoi
import time

# Import navigation modules
from dpdem.control import ShipState
from dpdem.control.gnc import ssa


# ============================================================================
# Ice Properties (from run_medea_ship_fracture.py)
# ============================================================================

class IceProperties:
    ICE_DENSITY = 920
    WATER_DENSITY = 1025
    FLEXURAL_STRENGTH_BASE = 0.15e6  # Further reduced for easier fracturing
    YOUNGS_MODULUS = 3e9  # Softer ice
    
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


def polygon_centroid(verts):
    return np.mean(verts, axis=0)


def create_ship_polygon(cx, cy, heading, length, beam):
    """Create ship polygon at given position and heading."""
    half_l = length / 2
    half_b = beam / 2
    
    # Local vertices (bow at +x)
    n_pts = 20
    hull = []
    
    # Starboard side (positive y in local frame)
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
    
    # Port side (negative y in local frame)
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
    
    # Rotate and translate
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    world_hull = hull @ R.T + np.array([cx, cy])
    
    return world_hull


def point_in_ship(px, py, ship_x, ship_y, ship_heading, ship_length, ship_beam):
    """Check if point is inside ship and compute penetration/normal."""
    # Transform to ship local frame
    c, s = np.cos(-ship_heading), np.sin(-ship_heading)
    dx = px - ship_x
    dy = py - ship_y
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    
    half_l = ship_length / 2
    half_b = ship_beam / 2
    x_rel = local_x / half_l
    
    # Compute local beam at this x (tapered bow at +x_rel > 0.6)
    if x_rel > 0.6:
        t = (x_rel - 0.6) / 0.4
        t = min(t, 1.0)
        local_half_beam = half_b * (1 - 0.9 * t ** 1.5)
        local_half_beam = max(local_half_beam, half_b * 0.02)
    else:
        local_half_beam = half_b
    
    # Check bounds
    if local_x < -half_l or local_x > half_l:
        return 0, 0, 0
    if abs(local_y) > local_half_beam:
        return 0, 0, 0
    
    # Inside ship - compute penetration
    dist_to_bow = half_l - local_x
    dist_to_stern = local_x + half_l
    dist_to_side = local_half_beam - abs(local_y)
    
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
        nx_local, ny_local = 1, 0
        pen = dist_to_bow
    elif dist_to_stern < dist_to_side:
        nx_local, ny_local = -1, 0
        pen = dist_to_stern
    else:
        nx_local = 0
        ny_local = 1 if local_y > 0 else -1
        pen = dist_to_side
    
    # Rotate normal back to world
    c, s = np.cos(ship_heading), np.sin(ship_heading)
    nx = c * nx_local - s * ny_local
    ny = s * nx_local + c * ny_local
    
    return max(pen, 0), nx, ny


# ============================================================================
# Ice Field Generation (Voronoi with bonds)
# ============================================================================

def create_bonded_ice_field(
    field_width, field_height,
    n_elements, cell_size,
    ship_start_x, ship_start_y, clear_radius,
    thickness_mean=1.0, thickness_std=0.2,
    seed=42,
):
    """Create ice field with Voronoi cells and bonds."""
    np.random.seed(seed)
    
    # Generate seed points on grid with jitter
    nx = int(field_width / cell_size)
    ny = int(field_height / cell_size)
    dx = field_width / nx
    dy = field_height / ny
    
    points = []
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * dx + np.random.uniform(-0.35 * dx, 0.35 * dx)
            y = (j + 0.5) * dy + np.random.uniform(-0.35 * dy, 0.35 * dy)
            
            # Skip near ship start
            dist = np.sqrt((x - ship_start_x)**2 + (y - ship_start_y)**2)
            if dist < clear_radius:
                continue
            
            x = np.clip(x, 1, field_width - 1)
            y = np.clip(y, 1, field_height - 1)
            points.append([x, y])
    
    if len(points) < 10:
        raise ValueError("Too few ice elements generated")
    
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
        clipped = np.clip(verts, [0, 0], [field_width, field_height])
        
        if len(clipped) < 3:
            continue
        
        # Order CCW
        centroid = np.mean(clipped, axis=0)
        angles = np.arctan2(clipped[:, 1] - centroid[1], clipped[:, 0] - centroid[0])
        order = np.argsort(angles)
        clipped = clipped[order]
        
        area = polygon_area(clipped)
        if area < 1.0:
            continue
        
        valid_idx.append(i)
        polygons.append(clipped)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    n_valid = len(polygons)
    
    # Create bonds between adjacent cells
    idx_to_poly = {valid_idx[i]: i for i in range(n_valid)}
    bonds = set()
    
    for ridge_points in vor.ridge_points:
        i, j = ridge_points
        if i < n_actual and j < n_actual:
            if i in idx_to_poly and j in idx_to_poly:
                pi, pj = idx_to_poly[i], idx_to_poly[j]
                bonds.add((min(pi, pj), max(pi, pj)))
    
    bonds = list(bonds)
    
    # Thickness
    thicknesses = np.random.normal(thickness_mean, thickness_std, n_valid)
    thicknesses = np.clip(thicknesses, 0.5, thickness_mean * 2)
    
    # Compute properties
    areas = np.array([polygon_area(p) for p in polygons])
    masses = areas * thicknesses * IceProperties.ICE_DENSITY
    radii = np.sqrt(areas / np.pi)
    
    # Bond strengths
    MAX_BOND_WIDTH = 4.0
    bond_strengths = np.zeros(len(bonds))
    rest_lengths = np.zeros(len(bonds))
    for b, (i, j) in enumerate(bonds):
        h = min(thicknesses[i], thicknesses[j])
        w = min((radii[i] + radii[j]) / 2, MAX_BOND_WIDTH)
        bond_strengths[b] = IceProperties.breaking_force(h, w)
        rest_lengths[b] = np.linalg.norm(centroids[i] - centroids[j])
    
    # Local polygons
    local_polys = [p - np.mean(p, axis=0) for p in polygons]
    
    print(f"Created ice field: {n_valid} cells, {len(bonds)} bonds")
    print(f"  Thickness: {thicknesses.min():.2f} - {thicknesses.max():.2f} m")
    
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
# Autonomous Ship Controller
# ============================================================================

class AutonomousShip:
    """Ship with simple proportional-derivative control toward goal."""
    
    def __init__(self, length, beam, mass, x, y, heading):
        self.length = length
        self.beam = beam
        self.mass = mass
        self.I_zz = (1/12) * mass * (length**2 + beam**2)
        
        self.x = x
        self.y = y
        self.psi = heading
        self.u = 0.0  # surge velocity
        self.v = 0.0  # sway velocity
        self.r = 0.0  # yaw rate
        
        self.goal = None
        
    def set_goal(self, gx, gy):
        self.goal = np.array([gx, gy])
    
    def get_control(self, dt):
        """Simple PD control toward goal."""
        if self.goal is None:
            return np.zeros(3)
        
        # Heading toward goal
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 5.0:
            return np.zeros(3)  # At goal
        
        psi_desired = np.arctan2(dy, dx)
        heading_error = ssa(psi_desired - self.psi)
        
        # Control gains - need strong thrust to break ice
        Kp_forward = 6e6  # Forward thrust - stronger icebreaker
        Kp_heading = 8e6   # Heading correction
        Kd_heading = 2e6   # Heading rate damping
        
        # Forward thrust - always push forward
        tau_surge = Kp_forward
        
        # Yaw control - keep heading toward goal
        tau_yaw = Kp_heading * heading_error - Kd_heading * self.r
        
        return np.array([tau_surge, 0.0, tau_yaw])
    
    def apply_forces(self, control_tau, external_force, external_moment, dt):
        """Update ship dynamics."""
        # Body frame forces
        c, s = np.cos(self.psi), np.sin(self.psi)
        
        # External force in body frame
        F_ext_body_x = c * external_force[0] + s * external_force[1]
        F_ext_body_y = -s * external_force[0] + c * external_force[1]
        
        # Total body forces
        F_surge = control_tau[0] + F_ext_body_x
        F_sway = control_tau[1] + F_ext_body_y
        M_yaw = control_tau[2] + external_moment
        
        # Drag - reduced for better icebreaking momentum
        drag_surge = 0.02 * self.mass * self.u
        drag_sway = 0.10 * self.mass * self.v
        drag_yaw = 0.03 * self.I_zz * self.r
        
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
        self.r = np.clip(self.r, -0.3, 0.3)
        
        # Update position
        vx = c * self.u - s * self.v
        vy = s * self.u + c * self.v
        
        self.x += dt * vx
        self.y += dt * vy
        self.psi = (self.psi + dt * self.r) % (2 * np.pi)
    
    def get_polygon(self):
        return create_ship_polygon(self.x, self.y, self.psi, self.length, self.beam)


# ============================================================================
# Simulation
# ============================================================================

def run_simulation(
    field_width=200.0,
    field_height=80.0,
    cell_size=6.0,
    ship_length=30.0,
    ship_beam=8.0,
    ship_mass=8e6,
    n_frames=800,
    dt=0.05,
    seed=42,
):
    """Run autonomous ship navigation with ice fracture."""
    
    print("=" * 60)
    print("DPDEM Autonomous Ship with Ice Fracture")
    print("=" * 60)
    
    # Ship start and goal
    ship_start = (25.0, field_height / 2, 0.0)  # x, y, heading
    ship_goal = (field_width - 25.0, field_height / 2)
    
    # Create ship
    ship = AutonomousShip(
        length=ship_length,
        beam=ship_beam,
        mass=ship_mass,
        x=ship_start[0],
        y=ship_start[1],
        heading=ship_start[2],
    )
    ship.set_goal(*ship_goal)
    
    print(f"Ship: {ship_length}m x {ship_beam}m, mass={ship_mass/1e6:.1f}t")
    print(f"Start: ({ship_start[0]:.0f}, {ship_start[1]:.0f}), Goal: ({ship_goal[0]:.0f}, {ship_goal[1]:.0f})")
    
    # Create ice field
    print("\nGenerating ice field...")
    ice = create_bonded_ice_field(
        field_width, field_height,
        n_elements=500,
        cell_size=cell_size,
        ship_start_x=ship_start[0],
        ship_start_y=ship_start[1],
        clear_radius=ship_length * 1.5,  # More room to accelerate
        thickness_mean=0.8,  # Thinner ice = easier to break
        thickness_std=0.2,
        seed=seed,
    )
    
    n = len(ice['polygons'])
    bonds = ice['bonds']
    
    # State
    positions = ice['centroids'].copy()
    velocities = np.zeros((n, 2))
    rotations = np.zeros(n)
    angular_vels = np.zeros(n)
    bond_intact = np.ones(len(bonds), dtype=bool)
    
    # History
    ship_history = [(ship.x, ship.y, ship.psi)]
    positions_history = [positions.copy()]
    rotations_history = [rotations.copy()]
    bonds_history = [bond_intact.copy()]
    
    print(f"\nRunning simulation ({n_frames} frames)...")
    t0 = time.time()
    
    for frame in range(n_frames):
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
            # Check multiple points on the ice element
            angle = rotations[i]
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            world_poly = ice['local_polys'][i] @ R.T + positions[i]
            
            max_pen = 0
            best_nx, best_ny = 0, 0
            
            # Check vertices
            for vx, vy in world_poly:
                pen, nx, ny = point_in_ship(vx, vy, ship.x, ship.y, ship.psi, 
                                           ship.length, ship.beam)
                if pen > max_pen:
                    max_pen, best_nx, best_ny = pen, nx, ny
            
            # Check edge midpoints
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
                # Collision force - strong enough to fracture but not stop ship
                force_mag = (ship.mass / 1e6) * 8e4 * max_pen
                forces[i, 0] += force_mag * best_nx
                forces[i, 1] += force_mag * best_ny
                ship_collision_force[i] = force_mag
                
                # Reaction on ship (asymmetric - ice yields more than ship)
                ship_force[0] -= force_mag * best_nx * 0.4
                ship_force[1] -= force_mag * best_ny * 0.4
                
                # Moment on ship
                rx = positions[i, 0] - ship.x
                ry = positions[i, 1] - ship.y
                ship_moment -= rx * force_mag * best_ny - ry * force_mag * best_nx
                
                # Side push for flow around hull
                rel_y = positions[i, 1] - ship.y
                side_dir = 1 if rel_y > 0 else -1
                forces[i, 1] += (ship.mass / 1e6) * 3e4 * max_pen * side_dir
                
                # Small angular kick
                angular_vels[i] += 0.01 * side_dir * min(max_pen, 1.0)
        
        # Ice-ice repulsion (soft contact, doesn't break bonds)
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                min_dist = (ice['radii'][i] + ice['radii'][j]) * 0.9  # Allow slight overlap
                
                if dist < min_dist and dist > 0.1:
                    overlap = min_dist - dist
                    direction = d / dist
                    f = 1e5 * overlap ** 1.5 * direction
                    forces[i] -= f
                    forces[j] += f
        
        # Water drag on ice
        for i in range(n):
            forces[i] -= 0.5 * velocities[i] * ice['masses'][i]
        
        # Move connected components together
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
        
        # Bond fracture (skip first 10 frames to let ice settle)
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
                
                # Effective strength
                collision_on_bond = ship_collision_force[i] > 0 or ship_collision_force[j] > 0
                if collision_on_bond:
                    effective_strength = ice['bond_strengths'][b] * 0.1
                elif propagated_stress > ice['bond_strengths'][b] * 0.5:
                    effective_strength = ice['bond_strengths'][b] * 0.6
                else:
                    effective_strength = ice['bond_strengths'][b]
                
                # Direct fracture from collision
                collision_force_on_bond = max(ship_collision_force[i], ship_collision_force[j])
                if collision_on_bond and collision_force_on_bond > effective_strength:
                    bond_intact[b] = False
                elif bond_force > effective_strength:
                    bond_intact[b] = False
        
        # Update ice rotations (with strong damping)
        angular_vels += np.random.randn(n) * 0.001
        angular_vels *= 0.85
        angular_vels = np.clip(angular_vels, -0.2, 0.2)
        rotations += angular_vels * dt
        
        # Clamp positions
        positions[:, 0] = np.clip(positions[:, 0], 0, field_width)
        positions[:, 1] = np.clip(positions[:, 1], 0, field_height)
        
        # Soft boundary forces on ship (push back to center)
        y_center = field_height / 2
        y_margin = 15.0  # Start pushing when this close to boundary
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
        
        # Check goal
        dist_to_goal = np.sqrt((ship.x - ship_goal[0])**2 + (ship.y - ship_goal[1])**2)
        if dist_to_goal < 10.0:
            print(f"\nGoal reached at frame {frame}!")
            break
    
    sim_time = time.time() - t0
    n_broken = np.sum(~bonds_history[-1])
    print(f"\nSimulation complete: {sim_time:.1f}s")
    print(f"Final: {n_broken}/{len(bonds)} bonds broken ({100*n_broken/len(bonds):.0f}%)")
    
    return {
        'ship_history': ship_history,
        'positions_history': positions_history,
        'rotations_history': rotations_history,
        'bonds_history': bonds_history,
        'ice': ice,
        'ship': ship,
        'field_width': field_width,
        'field_height': field_height,
        'goal': ship_goal,
    }


def create_animation(results, output_file='autonomous_ship_fracture.gif', fps=20):
    """Create animation."""
    print("\nCreating animation...")
    
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
    
    # Colors by thickness
    thicknesses = ice['thicknesses']
    thickness_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
    ice_colors = []
    for t in thickness_norm:
        r = int(240 - 100 * t)
        g = int(250 - 80 * t)
        b = int(255 - 40 * t)
        ice_colors.append(f'#{r:02x}{g:02x}{b:02x}')
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
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
            
            # Check for broken bonds
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
        
        # Draw ship trajectory
        traj_x = [h[0] for h in ship_history[:frame+1]]
        traj_y = [h[1] for h in ship_history[:frame+1]]
        ax.plot(traj_x, traj_y, 'yellow', linewidth=2, alpha=0.7, zorder=5)
        
        # Draw ship
        ship_poly = create_ship_polygon(ship_x, ship_y, ship_psi, ship.length, ship.beam)
        ship_patch = MplPolygon(
            ship_poly, facecolor='#404040', edgecolor='#202020',
            linewidth=2, zorder=10
        )
        ax.add_patch(ship_patch)
        
        # Draw goal
        ax.plot(goal[0], goal[1], 'g*', markersize=15, zorder=11)
        
        # Stats
        n_broken = np.sum(~intact)
        dist_to_goal = np.sqrt((ship_x - goal[0])**2 + (ship_y - goal[1])**2)
        
        ax.set_title(
            f'DPDEM Autonomous Ship Navigation with Fracture\n'
            f'Frame {frame}/{n_frames-1} | Dist: {dist_to_goal:.0f}m | '
            f'Bonds broken: {n_broken}/{len(bonds)}',
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
    results = run_simulation(
        field_width=160.0,
        field_height=50.0,  # Narrower so ship stays centered
        cell_size=5.5,
        ship_length=28.0,
        ship_beam=8.0,
        ship_mass=12e6,  # Heavier icebreaker
        n_frames=1200,
        dt=0.05,
        seed=42,
    )
    
    create_animation(results, output_file='autonomous_ship_fracture.gif', fps=20)
    
    print("\nOutput: autonomous_ship_fracture.gif")


if __name__ == '__main__':
    main()
