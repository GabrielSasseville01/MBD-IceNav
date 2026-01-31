#!/usr/bin/env python3
"""
DPDEM Ice-Ship Interaction Visualization

Based on Wang et al. (2024) - Dilated-polyhedron-based DEM for ice-structure interactions.

Key concepts from the paper:
- Sea ice elements are irregular polygons generated via Voronoi tessellation (Fig. 2)
- Adjacent elements are bonded together to represent frozen state (Section 2.1, Fig. 3)
- Bonded elements move together as one mass until bonds break under stress
- Fracture occurs when stress exceeds strength criterion (Eq. 14)

Enhanced physics:
- Variable ice thickness per floe (affects fracture resistance)
- Fracture depends on: ice thickness, ship mass, ship speed
- Flexural strength scales with thickness^1.5 (bending mechanics)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection
from scipy.spatial import Voronoi
from scipy.ndimage import label


# ============================================================================
# Ice Material Properties (from literature)
# ============================================================================

class IceProperties:
    """Physical properties of sea ice."""
    # Density
    ICE_DENSITY = 920  # kg/m³
    WATER_DENSITY = 1025  # kg/m³
    
    # Mechanical properties (typical values from Timco & Weeks 2010)
    FLEXURAL_STRENGTH_BASE = 0.5e6  # Pa (0.5 MPa base flexural strength)
    COMPRESSIVE_STRENGTH = 2.5e6   # Pa
    YOUNGS_MODULUS = 5e9           # Pa
    
    # Fracture energy (from paper Table 1)
    FRACTURE_ENERGY_MODE_I = 15    # J/m² (tensile)
    FRACTURE_ENERGY_MODE_II = 45   # J/m² (shear)
    
    @staticmethod
    def flexural_strength(thickness):
        """
        Flexural strength varies with thickness.
        Thicker ice is stronger but also more brittle.
        σ_f ∝ h^0.5 approximately (empirical relationship)
        """
        h_ref = 1.0  # reference thickness
        return IceProperties.FLEXURAL_STRENGTH_BASE * np.sqrt(thickness / h_ref)
    
    @staticmethod
    def breaking_force(thickness, width):
        """
        Force required to break ice in bending.
        From beam theory: F_break ∝ σ_f * w * h²
        """
        sigma_f = IceProperties.flexural_strength(thickness)
        return sigma_f * width * thickness ** 2


# ============================================================================
# Ship Properties
# ============================================================================

class ShipProperties:
    """Physical properties of the ship."""
    def __init__(self, length, beam, draft, displacement=None):
        self.length = length  # m
        self.beam = beam      # m
        self.draft = draft    # m
        
        # Displacement (mass of water displaced = ship mass)
        if displacement is None:
            # Estimate from hull dimensions (simplified box approximation)
            # Real ships have ~0.5-0.7 block coefficient
            block_coef = 0.6
            self.displacement = (length * beam * draft * 
                                block_coef * IceProperties.WATER_DENSITY)
        else:
            self.displacement = displacement
        
        self.mass = self.displacement  # kg
    
    def impact_force(self, velocity, ice_thickness):
        """
        Estimate ice impact force using momentum-impulse approach.
        F = m * dv / dt, where dt depends on ice crushing
        
        From Daley (2000) ice load model:
        F ∝ (displacement)^0.7 * (velocity)^0.8 * (thickness)^0.5
        """
        # Empirical coefficients (calibrated)
        C_force = 0.1  # Force coefficient
        
        F = C_force * (self.mass ** 0.7) * (velocity ** 0.8) * (ice_thickness ** 0.5)
        return F
    
    def kinetic_energy(self, velocity):
        """Ship kinetic energy."""
        return 0.5 * self.mass * velocity ** 2


# ============================================================================
# Voronoi Ice Field Generation (as per paper Section 2.1, Fig. 2)
# ============================================================================

def create_voronoi_ice_field(
    n_elements: int,
    field_width: float,
    field_height: float,
    seed: int = 42,
    thickness_mean: float = 1.5,
    thickness_std: float = 0.5,
) -> tuple:
    """
    Create ice field using Voronoi tessellation as described in the paper.
    
    From paper Section 2.1:
    "To replicate arbitrary morphological features of natural sea ice, 
    a Voronoi cutting algorithm was employed to parametrically divide 
    a two-dimensional plane into polygons."
    
    Args:
        n_elements: Target number of ice floes
        field_width: Width of ice field [m]
        field_height: Height of ice field [m]
        seed: Random seed
        thickness_mean: Mean ice thickness [m]
        thickness_std: Standard deviation of thickness [m]
    
    Returns:
        polygons: List of (N, 2) polygon vertex arrays
        centroids: (n, 2) array of element centers
        bonds: List of (i, j) tuples for adjacent elements
        thicknesses: (n,) array of ice thickness per element [m]
    """
    np.random.seed(seed)
    
    # Generate seed points with random distribution for varied floe sizes
    # Mix of clustered and sparse regions creates natural size variation
    points = []
    
    # Some points from jittered grid (creates medium-sized floes)
    nx = int(np.sqrt(n_elements * 0.5 * field_width / field_height))
    ny = int(n_elements * 0.5 / nx)
    dx = field_width / nx
    dy = field_height / ny
    
    for i in range(nx):
        for j in range(ny):
            # Large jitter for varied spacing
            x = (i + 0.5) * dx + np.random.uniform(-0.45*dx, 0.45*dx)
            y = (j + 0.5) * dy + np.random.uniform(-0.45*dy, 0.45*dy)
            x = np.clip(x, 0.1, field_width - 0.1)
            y = np.clip(y, 0.1, field_height - 0.1)
            points.append([x, y])
    
    # Add random points (creates varied sizes - some large, some small)
    n_random = n_elements - len(points)
    for _ in range(n_random):
        x = np.random.uniform(0.5, field_width - 0.5)
        y = np.random.uniform(0.5, field_height - 0.5)
        points.append([x, y])
    
    points = np.array(points)
    n_actual = len(points)
    
    # Add mirror points to bound the Voronoi cells
    mirrors = []
    for dx_mult in [-1, 0, 1]:
        for dy_mult in [-1, 0, 1]:
            if dx_mult == 0 and dy_mult == 0:
                continue
            shifted = points.copy()
            shifted[:, 0] += dx_mult * field_width
            shifted[:, 1] += dy_mult * field_height
            mirrors.append(shifted)
    
    all_points = np.vstack([points] + mirrors)
    vor = Voronoi(all_points)
    
    # Extract polygons for original points only
    polygons = []
    centroids = []
    valid_idx = []  # Map from original index to polygon list index
    
    for i in range(n_actual):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) < 3:
            continue
        
        verts = vor.vertices[region]
        
        # Clip to domain
        clipped_verts = []
        for v in verts:
            cx = np.clip(v[0], 0, field_width)
            cy = np.clip(v[1], 0, field_height)
            clipped_verts.append([cx, cy])
        clipped_verts = np.array(clipped_verts)
        
        if len(clipped_verts) < 3:
            continue
        
        # Order vertices counter-clockwise
        centroid = np.mean(clipped_verts, axis=0)
        angles = np.arctan2(clipped_verts[:, 1] - centroid[1],
                          clipped_verts[:, 0] - centroid[0])
        order = np.argsort(angles)
        clipped_verts = clipped_verts[order]
        
        # Check polygon area is reasonable
        area = polygon_area(clipped_verts)
        if area < 0.5:  # Skip tiny polygons
            continue
        
        valid_idx.append(i)
        polygons.append(clipped_verts)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    n_valid = len(polygons)
    
    # Create bonds between adjacent elements (using ridge info from Voronoi)
    # Two elements are adjacent if they share a Voronoi ridge
    idx_to_poly = {valid_idx[i]: i for i in range(n_valid)}
    bonds = set()
    
    for ridge_points in vor.ridge_points:
        i, j = ridge_points
        if i < n_actual and j < n_actual:
            if i in idx_to_poly and j in idx_to_poly:
                pi, pj = idx_to_poly[i], idx_to_poly[j]
                bonds.add((min(pi, pj), max(pi, pj)))
    
    bonds = list(bonds)
    
    # Generate variable ice thickness for each floe
    # Thickness varies spatially - can have ridges, thin spots, etc.
    thicknesses = np.random.normal(thickness_mean, thickness_std, n_valid)
    thicknesses = np.clip(thicknesses, 0.3, thickness_mean * 2.5)  # Physical bounds
    
    # Add some spatial correlation - nearby floes have similar thickness
    for _ in range(3):
        new_thick = thicknesses.copy()
        for i in range(n_valid):
            neighbors = [j for pi, pj in bonds if pi == i or pj == i 
                        for j in [pi, pj] if j != i]
            if neighbors:
                neighbor_avg = np.mean([thicknesses[j] for j in neighbors])
                new_thick[i] = 0.7 * thicknesses[i] + 0.3 * neighbor_avg
        thicknesses = new_thick
    
    print(f"  Generated {n_valid} Voronoi elements, {len(bonds)} bonds", flush=True)
    print(f"  Ice thickness: {thicknesses.min():.2f} - {thicknesses.max():.2f} m "
          f"(mean: {thicknesses.mean():.2f} m)", flush=True)
    
    return polygons, centroids, bonds, thicknesses


def polygon_area(verts):
    """Compute signed area of polygon using shoelace formula."""
    n = len(verts)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i, 0] * verts[j, 1]
        area -= verts[j, 0] * verts[i, 1]
    return abs(area) / 2


def polygon_centroid(verts):
    """Compute centroid of polygon."""
    return np.mean(verts, axis=0)


# ============================================================================
# Ship Hull
# ============================================================================

def create_ship_hull(length, beam, position, bow_direction='left'):
    """Create ship hull outline (2D top-down view)."""
    cx, cy = position
    n_pts = 30
    hull_x, hull_y = [], []
    
    # Starboard side
    for i in range(n_pts + 1):
        t = i / n_pts
        if bow_direction == 'left':
            x = cx + length/2 - t * length
            x_rel = -(x - cx) / (length/2)
        else:
            x = cx - length/2 + t * length
            x_rel = (x - cx) / (length/2)
        
        # Taper at bow
        if x_rel > 0.6:
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
        else:
            taper = 1.0
        taper = max(taper, 0.02)
        
        hull_x.append(x)
        hull_y.append(cy + beam/2 * taper)
    
    # Port side
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
# Physics Simulation
# ============================================================================

def find_connected_components(n, bonds, intact):
    """Find connected components of bonded elements."""
    # Build adjacency list for intact bonds only
    adj = {i: [] for i in range(n)}
    for b, (i, j) in enumerate(bonds):
        if intact[b]:
            adj[i].append(j)
            adj[j].append(i)
    
    # BFS to find components
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
    """
    Compute collision with ship hull including tapered bow.
    Bow is at LEFT (negative x direction from ship center).
    
    Returns: (penetration_depth, normal_x, normal_y) or (0, 0, 0) if no collision
    """
    cx, cy = ship_pos
    half_len = ship_length / 2
    half_beam = ship_beam / 2
    
    # Ship spans from cx - half_len (bow tip) to cx + half_len (stern)
    bow_x = cx - half_len
    stern_x = cx + half_len
    
    # Relative position from ship center
    rel_x = px - cx
    rel_y = py - cy
    
    # x_rel: -1 at bow, +1 at stern
    x_rel = rel_x / half_len
    
    # Compute local beam at this x position (taper at bow = left side = negative x_rel)
    if x_rel < -bow_taper_start:  # Bow region (left side, x_rel < -0.6)
        # t goes from 0 at taper start to 1 at bow tip
        t = (-x_rel - bow_taper_start) / (1 - bow_taper_start)
        t = min(t, 1.0)
        local_half_beam = half_beam * (1 - 0.9 * t ** 1.5)
        local_half_beam = max(local_half_beam, half_beam * 0.05)
    else:
        local_half_beam = half_beam
    
    # Check if outside ship x-bounds
    if rel_x < -half_len or rel_x > half_len:
        return 0, 0, 0
    
    # Check if outside ship y-bounds (accounting for taper)
    if abs(rel_y) > local_half_beam:
        return 0, 0, 0
    
    # Inside ship - compute penetration and push direction
    # Distance to bow (left edge)
    dist_to_bow = rel_x - (-half_len)  # How far from bow tip
    # Distance to side
    dist_to_side = local_half_beam - abs(rel_y)
    
    if x_rel < -bow_taper_start:
        # In bow region - push along angled bow surface
        t = (-x_rel - bow_taper_start) / (1 - bow_taper_start)
        t = min(t, 1.0)
        # Bow surface angle (steeper toward tip)
        bow_angle = 0.3 + 0.5 * t  # Radians, roughly 20-45 degrees
        
        # Normal points outward from bow surface
        if rel_y >= 0:
            nx = -np.cos(bow_angle)
            ny = np.sin(bow_angle)
        else:
            nx = -np.cos(bow_angle)
            ny = -np.sin(bow_angle)
        
        pen = min(dist_to_bow, dist_to_side) + 0.5  # Add small offset
    else:
        # In main hull body - push to nearest edge
        if dist_to_bow < dist_to_side:
            nx, ny = -1, 0  # Push toward bow (left)
            pen = dist_to_bow
        else:
            nx, ny = 0, 1 if rel_y > 0 else -1  # Push to side
            pen = dist_to_side
    
    return max(pen, 0), nx, ny


def simulate_ice_ship_interaction(
    polygons, centroids, bonds, ship_pos, ship_length, ship_beam,
    n_frames, dt, drift_vel, thicknesses=None, ship_props=None
):
    """
    Simulate ice drifting toward ship with realistic fracture mechanics.
    
    Key physics:
    - Bonded elements move together as rigid bodies
    - Fracture depends on: ice thickness, ship mass, ship speed
    - Flexural strength scales with thickness (bending mechanics)
    - Impact force from ship momentum transfer
    
    Args:
        polygons: List of ice floe polygons
        centroids: Floe centroids
        bonds: Bond pairs
        ship_pos: Ship position
        ship_length, ship_beam: Ship dimensions
        n_frames: Number of frames
        dt: Time step
        drift_vel: Ice drift velocity (m/s)
        thicknesses: Ice thickness per floe [m]
        ship_props: ShipProperties object (if None, uses defaults)
    """
    n = len(polygons)
    
    # Default thicknesses if not provided
    if thicknesses is None:
        thicknesses = np.ones(n) * 1.5  # 1.5m default
    
    # Default ship properties
    if ship_props is None:
        ship_props = ShipProperties(ship_length, ship_beam, draft=ship_beam/3)
    
    # Compute masses (area * thickness * density) and effective radii
    areas = np.array([polygon_area(p) for p in polygons])
    masses = areas * thicknesses * IceProperties.ICE_DENSITY
    radii = np.array([np.sqrt(a / np.pi) for a in areas])
    
    # State
    positions = centroids.copy()
    velocities = np.zeros_like(positions)
    rotations = np.zeros(n)
    angular_vels = np.zeros(n)
    
    # Bond state - all start intact (frozen ice)
    bond_intact = np.ones(len(bonds), dtype=bool)
    
    # Initial rest lengths for bonds
    rest_lengths = np.array([np.linalg.norm(centroids[i] - centroids[j]) 
                            for i, j in bonds])
    
    # Compute bond strengths based on ice thickness at each bond
    # Bond strength depends on the thinner ice of the two bonded elements
    # and the width of the bond (approximated by element size)
    bond_strengths = np.zeros(len(bonds))
    for b, (i, j) in enumerate(bonds):
        # Use minimum thickness (weakest link)
        h = min(thicknesses[i], thicknesses[j])
        # Bond width approximated by average element radius
        w = (radii[i] + radii[j]) / 2
        # Breaking force from flexural strength
        bond_strengths[b] = IceProperties.breaking_force(h, w)
    
    print(f"  Bond strengths: {bond_strengths.min()/1e6:.2f} - "
          f"{bond_strengths.max()/1e6:.2f} MN", flush=True)
    
    # History
    positions_history = [positions.copy()]
    rotations_history = [rotations.copy()]
    bonds_history = [bond_intact.copy()]
    
    # Ship impact force (depends on mass and speed)
    ship_impact_force = ship_props.impact_force(drift_vel, np.mean(thicknesses))
    print(f"  Ship impact force: {ship_impact_force/1e6:.2f} MN", flush=True)
    
    for frame in range(n_frames):
        # 1. Find connected components (groups that move together)
        components = find_connected_components(n, bonds, bond_intact)
        
        # 2. Compute forces on each element
        forces = np.zeros_like(positions)
        bond_stresses = np.zeros(len(bonds))
        
        ship_collision_force = np.zeros(n)
        
        for i in range(n):
            px, py = positions[i]
            r = radii[i]
            
            # Check collision with ship using actual hull shape
            # Check multiple points around the element
            pen, nx, ny = point_to_ship_collision(
                px, py, ship_pos, ship_length, ship_beam
            )
            
            # Also check element boundary
            if pen <= 0:
                pen, nx, ny = point_to_ship_collision(
                    px + r, py, ship_pos, ship_length, ship_beam
                )
            
            if pen > 0:
                # Collision force scales with ship mass and penetration
                # Heavier ships exert more force
                force_mag = (ship_props.mass / 1e6) * 1e5 * pen
                forces[i, 0] += force_mag * nx
                forces[i, 1] += force_mag * ny
                ship_collision_force[i] = force_mag
                
                # Extra sideways force to encourage flow around hull
                # Push ice toward nearest side of ship
                rel_y = py - ship_pos[1]
                side_direction = 1 if rel_y > 0 else -1
                side_push = (ship_props.mass / 1e6) * 3e4 * pen * side_direction
                forces[i, 1] += side_push
                
                # Apply torque to encourage rotation/flow
                angular_vels[i] += 0.2 * side_direction * pen
        
        # Inter-element repulsion (prevents overlap, allows flow around obstacles)
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    overlap = min_dist - dist
                    direction = d / dist
                    # Repulsion force
                    f = 2e5 * overlap ** 1.5 * direction
                    forces[i] -= f
                    forces[j] += f
        
        # Drift force (ice drifting toward ship)
        for comp in components:
            # Apply drift force to component center of mass
            total_mass = sum(masses[i] for i in comp)
            drift_force = drift_vel * total_mass * 8
            
            # Distribute to elements
            for i in comp:
                forces[i, 0] += drift_force * masses[i] / total_mass
        
        # Water drag (higher to slow down pileup)
        for i in range(n):
            forces[i] -= 0.5 * velocities[i] * masses[i]
        
        # 3. Move connected components together
        new_velocities = velocities.copy()
        new_positions = positions.copy()
        
        for comp in components:
            if len(comp) == 1:
                # Single element - simple integration
                i = comp[0]
                accel = forces[i] / masses[i]
                new_velocities[i] += accel * dt
                new_velocities[i] = np.clip(new_velocities[i], -20, 20)
                new_positions[i] += new_velocities[i] * dt
            else:
                # Multi-element component - move as rigid body
                # Compute center of mass
                total_mass = sum(masses[i] for i in comp)
                com = sum(masses[i] * positions[i] for i in comp) / total_mass
                
                # Total force on component
                total_force = sum(forces[i] for i in comp)
                
                # Component velocity (average weighted by mass)
                comp_vel = sum(masses[i] * velocities[i] for i in comp) / total_mass
                
                # Update component velocity
                accel = total_force / total_mass
                comp_vel += accel * dt
                comp_vel = np.clip(comp_vel, -20, 20)
                
                # Move all elements in component together
                for i in comp:
                    new_velocities[i] = comp_vel
                    # Maintain relative position within component
                    rel_pos = positions[i] - com
                    new_com = com + comp_vel * dt
                    new_positions[i] = new_com + rel_pos
        
        velocities = new_velocities
        positions = new_positions
        
        # 4. Propagate stress through ice sheet (limited propagation)
        # Elements under compression from ship transfer stress to neighbors
        element_stress = ship_collision_force.copy()
        
        # Propagate stress backward through bonds (2 iterations, lower transfer)
        for _ in range(2):
            new_stress = element_stress.copy()
            for b, (i, j) in enumerate(bonds):
                if bond_intact[b]:
                    # Stress propagates through intact bonds (reduced rate)
                    stress_transfer = 0.3  # 30% of stress transfers
                    if element_stress[i] > element_stress[j]:
                        new_stress[j] += element_stress[i] * stress_transfer
                    else:
                        new_stress[i] += element_stress[j] * stress_transfer
            element_stress = new_stress
        
        # 5. Check for bond fracture with physics-based criterion
        # Bonds break when applied force exceeds bond strength
        # Strength depends on ice thickness (thinner ice breaks easier)
        for b, (i, j) in enumerate(bonds):
            if not bond_intact[b]:
                continue
            
            # Compute stress on bond
            d = positions[j] - positions[i]
            dist = np.linalg.norm(d)
            
            # Change in distance from rest (strain)
            stretch = abs(dist - rest_lengths[b])
            
            # Force from bond deformation (elastic)
            # Stiffness scales with ice thickness
            avg_thickness = (thicknesses[i] + thicknesses[j]) / 2
            stiffness = IceProperties.YOUNGS_MODULUS * avg_thickness * 0.001
            bond_force = stiffness * stretch
            
            # Add force from ship collision (direct impact)
            propagated_stress = max(element_stress[i], element_stress[j])
            bond_force += propagated_stress * 0.2
            
            # Get this bond's strength (based on thickness)
            base_strength = bond_strengths[b]
            
            # Strength reduction factors:
            # - Direct ship contact: stress concentration at impact
            # - High propagated stress: fatigue/pre-weakening
            if ship_collision_force[i] > 0 or ship_collision_force[j] > 0:
                # Direct impact - significant strength reduction
                effective_strength = base_strength * 0.25
            elif propagated_stress > base_strength * 0.5:
                # High stress propagation - some weakening
                effective_strength = base_strength * 0.6
            else:
                effective_strength = base_strength
            
            bond_stresses[b] = bond_force
            
            # Fracture criterion: force exceeds strength
            if bond_force > effective_strength:
                bond_intact[b] = False
        
        # 5. Small rotation for visual effect
        angular_vels += np.random.randn(n) * 0.005
        angular_vels *= 0.95  # Damping
        rotations += angular_vels * dt
        
        positions_history.append(positions.copy())
        rotations_history.append(rotations.copy())
        bonds_history.append(bond_intact.copy())
    
    return positions_history, rotations_history, bonds_history


# ============================================================================
# Visualization
# ============================================================================

def animate_ice_ship(
    polygons, positions_history, rotations_history, bonds_history, bonds,
    ship_outline, initial_centroids, thicknesses=None, 
    output_file='dpdem_ice_ship.gif', fps=20
):
    """
    Animate ice-ship interaction.
    
    Visualization approach:
    - Draw actual Voronoi polygon shapes
    - Color ice by thickness (thicker = darker blue)
    - Bonded elements appear as one continuous mass (no visible boundaries)
    - Broken bonds show as visible cracks/edges
    """
    n_frames = len(positions_history)
    n_elements = len(polygons)
    
    # Default thickness if not provided
    if thicknesses is None:
        thicknesses = np.ones(n_elements) * 1.5
    
    # Compute local polygons (centered at origin)
    local_polys = []
    for poly in polygons:
        centroid = polygon_centroid(poly)
        local_polys.append(poly - centroid)
    
    # Domain bounds
    all_pos = np.concatenate(positions_history)
    x_min = min(all_pos[:, 0].min(), ship_outline[:, 0].min()) - 5
    x_max = max(all_pos[:, 0].max(), ship_outline[:, 0].max()) + 5
    y_min = min(all_pos[:, 1].min(), ship_outline[:, 1].min()) - 5
    y_max = max(all_pos[:, 1].max(), ship_outline[:, 1].max()) + 5
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colors - ice color varies by thickness
    water_color = '#1A4D70'
    crack_color = '#2A5D80'
    ship_color = '#404040'
    
    # Color map for thickness: thin ice = lighter/whiter, thick ice = darker blue
    thickness_norm = (thicknesses - thicknesses.min()) / (thicknesses.max() - thicknesses.min() + 0.01)
    ice_colors = []
    for t in thickness_norm:
        # Interpolate from white/light cyan (thin) to deep blue (thick)
        # More dramatic color range for better visualization
        r = int(240 - 100 * t)  # 240 -> 140
        g = int(250 - 80 * t)   # 250 -> 170
        b = int(255 - 40 * t)   # 255 -> 215 (stays blueish)
        ice_colors.append(f'#{r:02x}{g:02x}{b:02x}')
    
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
        
        # Build set of broken bonds for each element
        broken_neighbors = {i: set() for i in range(n_elements)}
        for b, (i, j) in enumerate(bonds):
            if not intact[b]:
                broken_neighbors[i].add(j)
                broken_neighbors[j].add(i)
        
        # Draw all ice elements
        # Color by thickness, edges shown for broken bonds
        for i in range(n_elements):
            rotated = rotate_polygon(local_polys[i], rotations[i])
            world_poly = rotated + positions[i]
            
            has_breaks = len(broken_neighbors[i]) > 0
            
            # Color based on thickness
            floe_color = ice_colors[i]
            
            if has_breaks:
                # Show this element with visible edge
                edge_color = crack_color
                edge_width = 1.0
            else:
                # Part of intact ice sheet - subtle edge
                edge_color = floe_color
                edge_width = 0.3
            
            patch = MplPolygon(
                world_poly,
                facecolor=floe_color,
                edgecolor=edge_color,
                linewidth=edge_width,
                zorder=3,
                alpha=1.0
            )
            ax.add_patch(patch)
        
        # Draw crack lines where bonds have broken
        n_broken = np.sum(~intact)
        if n_broken > 0:
            crack_lines = []
            for b, (i, j) in enumerate(bonds):
                if not intact[b]:
                    # Draw line between centers of broken pair
                    p1, p2 = positions[i], positions[j]
                    dist = np.linalg.norm(p1 - p2)
                    init_dist = np.linalg.norm(initial_centroids[i] - initial_centroids[j])
                    
                    # Only show crack if elements haven't drifted too far apart
                    if dist < init_dist * 2.5:
                        crack_lines.append([(p1[0], p1[1]), (p2[0], p2[1])])
            
            if crack_lines:
                lc = LineCollection(crack_lines, colors=water_color,
                                  linewidths=2.0, alpha=0.8, zorder=4)
                ax.add_collection(lc)
        
        # Draw ship (on top)
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
        pct_broken = 100 * (n_total - n_intact) / n_total if n_total > 0 else 0
        
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Y [m]', fontsize=11)
        ax.set_title(
            f'DPDEM: Level Ice Fracture on Ship Impact\n'
            f'Frame {frame}/{n_frames-1} | '
            f'Broken bonds: {n_total - n_intact}/{n_total} ({pct_broken:.0f}%)',
            fontsize=13, fontweight='bold'
        )
        ax.grid(True, alpha=0.15, color='white', linestyle='--')
        
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

def create_multi_patch_ice_field(
    n_patches: int,
    field_width: float,
    field_height: float,
    seed: int = 42,
):
    """
    Create multiple separate ice patches/floes of different sizes and thicknesses.
    
    Each patch is an independent ice mass that can have its own:
    - Size (number of elements)
    - Thickness
    - Shape (from Voronoi tessellation)
    
    Returns:
        polygons: List of all polygon vertex arrays
        centroids: All element centers
        bonds: All bonds (only within patches, not between)
        thicknesses: Thickness per element
        patch_ids: Which patch each element belongs to
    """
    np.random.seed(seed)
    
    all_polygons = []
    all_centroids = []
    all_bonds = []
    all_thicknesses = []
    all_patch_ids = []
    
    element_offset = 0
    
    # Generate random patch properties
    for patch_idx in range(n_patches):
        # Random patch size (20-80 elements per patch - larger patches)
        patch_n_elements = np.random.randint(20, 80)
        
        # Random patch position (leave room at right for ship)
        patch_cx = np.random.uniform(10, field_width - 10)
        patch_cy = np.random.uniform(10, field_height - 10)
        
        # Random patch dimensions - much larger
        patch_width = np.random.uniform(15, 35)
        patch_height = np.random.uniform(12, 28)
        
        # Random thickness for this patch (uniform across entire patch)
        # Thicker patches = multi-year ice, thinner = first-year ice
        patch_thickness = np.random.uniform(0.5, 3.5)
        
        # Generate Voronoi cells for this patch
        points = []
        nx = int(np.sqrt(patch_n_elements * patch_width / patch_height))
        ny = max(1, int(patch_n_elements / max(nx, 1)))
        dx = patch_width / max(nx, 1)
        dy = patch_height / max(ny, 1)
        
        for i in range(nx):
            for j in range(ny):
                x = patch_cx - patch_width/2 + (i + 0.5) * dx + np.random.uniform(-0.4*dx, 0.4*dx)
                y = patch_cy - patch_height/2 + (j + 0.5) * dy + np.random.uniform(-0.4*dy, 0.4*dy)
                points.append([x, y])
        
        if len(points) < 3:
            continue
            
        points = np.array(points)
        
        # Add mirror points for bounded Voronoi
        mirrors = []
        for dx_m in [-1, 0, 1]:
            for dy_m in [-1, 0, 1]:
                if dx_m == 0 and dy_m == 0:
                    continue
                shifted = points.copy()
                shifted[:, 0] += dx_m * patch_width
                shifted[:, 1] += dy_m * patch_height
                mirrors.append(shifted)
        
        all_points = np.vstack([points] + mirrors)
        
        try:
            vor = Voronoi(all_points)
        except:
            continue
        
        # Extract polygons for this patch
        patch_polygons = []
        patch_centroids = []
        valid_idx = []
        n_orig = len(points)
        
        for i in range(n_orig):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) < 3:
                continue
            
            verts = vor.vertices[region]
            
            # Clip to patch bounds
            clipped_verts = []
            for v in verts:
                cx = np.clip(v[0], patch_cx - patch_width/2, patch_cx + patch_width/2)
                cy = np.clip(v[1], patch_cy - patch_height/2, patch_cy + patch_height/2)
                clipped_verts.append([cx, cy])
            clipped_verts = np.array(clipped_verts)
            
            if len(clipped_verts) < 3:
                continue
            
            # Order vertices
            centroid = np.mean(clipped_verts, axis=0)
            angles = np.arctan2(clipped_verts[:, 1] - centroid[1],
                              clipped_verts[:, 0] - centroid[0])
            order = np.argsort(angles)
            clipped_verts = clipped_verts[order]
            
            area = polygon_area(clipped_verts)
            if area < 0.3:
                continue
            
            valid_idx.append(i)
            patch_polygons.append(clipped_verts)
            patch_centroids.append(centroid)
        
        if len(patch_polygons) < 2:
            continue
        
        patch_centroids = np.array(patch_centroids)
        n_valid = len(patch_polygons)
        
        # Create bonds within this patch only
        idx_to_poly = {valid_idx[i]: i for i in range(n_valid)}
        patch_bonds = set()
        
        for ridge_points in vor.ridge_points:
            i, j = ridge_points
            if i < n_orig and j < n_orig:
                if i in idx_to_poly and j in idx_to_poly:
                    pi, pj = idx_to_poly[i], idx_to_poly[j]
                    # Add element_offset to make global indices
                    patch_bonds.add((min(pi, pj) + element_offset, 
                                   max(pi, pj) + element_offset))
        
        # All elements in this patch have the same thickness
        patch_thicknesses = np.full(n_valid, patch_thickness)
        
        # Add to global lists
        all_polygons.extend(patch_polygons)
        all_centroids.extend(patch_centroids)
        all_bonds.extend(list(patch_bonds))
        all_thicknesses.extend(patch_thicknesses)
        all_patch_ids.extend([patch_idx] * n_valid)
        
        element_offset += n_valid
    
    all_centroids = np.array(all_centroids)
    all_thicknesses = np.array(all_thicknesses)
    all_patch_ids = np.array(all_patch_ids)
    
    print(f"  Created {n_patches} patches with {len(all_polygons)} total elements", flush=True)
    print(f"  {len(all_bonds)} bonds (within patches only)", flush=True)
    print(f"  Thickness range: {all_thicknesses.min():.2f} - {all_thicknesses.max():.2f} m", flush=True)
    
    return all_polygons, all_centroids, all_bonds, all_thicknesses, all_patch_ids


def main():
    print("=" * 60, flush=True)
    print("DPDEM: Multiple Ice Patches with Variable Properties", flush=True)
    print("=" * 60, flush=True)
    
    # Field parameters - larger field for bigger patches
    field_width = 80.0
    field_height = 60.0
    n_patches = 5  # Fewer but larger patches
    
    # Ship parameters
    ship_length = 30.0
    ship_beam = 7.0
    ship_draft = 3.0
    ship_displacement = 8000e3
    
    ship_props = ShipProperties(
        length=ship_length,
        beam=ship_beam, 
        draft=ship_draft,
        displacement=ship_displacement
    )
    print(f"Ship: {ship_length}m x {ship_beam}m, mass={ship_props.mass/1e6:.1f} kt", 
          flush=True)
    
    ship_pos = np.array([field_width + 30, field_height / 2])
    drift_vel = 2.5  # Slightly faster to reach patches
    
    # Create multiple separate ice patches
    print("\nGenerating multiple ice patches...", flush=True)
    polygons, centroids, bonds, thicknesses, patch_ids = create_multi_patch_ice_field(
        n_patches, field_width, field_height, seed=42
    )
    print(f"  Created {len(polygons)} ice elements with {len(bonds)} bonds", flush=True)
    
    # Create ship hull (bow facing left toward ice)
    ship_outline = create_ship_hull(ship_length, ship_beam, ship_pos, bow_direction='left')
    
    # Run simulation with physics-based fracture
    print("\nRunning simulation...", flush=True)
    positions, rotations, bonds_intact = simulate_ice_ship_interaction(
        polygons=polygons,
        centroids=centroids,
        bonds=bonds,
        ship_pos=ship_pos,
        ship_length=ship_length,
        ship_beam=ship_beam,
        n_frames=400,
        dt=0.05,
        drift_vel=drift_vel,
        thicknesses=thicknesses,
        ship_props=ship_props,
    )
    
    n_final = bonds_intact[-1].sum() if len(bonds) > 0 else 0
    print(f"  Final intact bonds: {n_final}/{len(bonds)}", flush=True)
    
    # Create animation
    animate_ice_ship(
        polygons=polygons,
        positions_history=positions,
        rotations_history=rotations,
        bonds_history=bonds_intact,
        bonds=bonds,
        ship_outline=ship_outline,
        initial_centroids=centroids,
        thicknesses=thicknesses,
        output_file='dpdem_ice_ship.gif',
        fps=20,
    )


if __name__ == '__main__':
    main()
