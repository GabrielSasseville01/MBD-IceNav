#!/usr/bin/env python3
"""
DPDEM Ice-Ship Interaction Visualization

Based on Wang et al. (2024) - Dilated-polyhedron-based DEM for ice-structure interactions.

Key concepts from the paper:
- Sea ice elements are irregular polygons generated via Voronoi tessellation (Fig. 2)
- Adjacent elements are bonded together to represent frozen state (Section 2.1, Fig. 3)
- Bonded elements move together as one mass until bonds break under stress
- Fracture occurs when stress exceeds strength criterion (Eq. 14)
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
# Voronoi Ice Field Generation (as per paper Section 2.1, Fig. 2)
# ============================================================================

def create_voronoi_ice_field(
    n_elements: int,
    field_width: float,
    field_height: float,
    seed: int = 42,
) -> tuple:
    """
    Create ice field using Voronoi tessellation as described in the paper.
    
    From paper Section 2.1:
    "To replicate arbitrary morphological features of natural sea ice, 
    a Voronoi cutting algorithm was employed to parametrically divide 
    a two-dimensional plane into polygons."
    
    Returns:
        polygons: List of (N, 2) polygon vertex arrays
        centroids: (n, 2) array of element centers
        bonds: List of (i, j) tuples for adjacent elements
    """
    np.random.seed(seed)
    
    # Generate seed points with some regularity (jittered grid)
    nx = int(np.sqrt(n_elements * field_width / field_height))
    ny = int(n_elements / nx)
    
    points = []
    dx = field_width / nx
    dy = field_height / ny
    
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * dx + np.random.uniform(-0.3*dx, 0.3*dx)
            y = (j + 0.5) * dy + np.random.uniform(-0.3*dy, 0.3*dy)
            x = np.clip(x, 0.1, field_width - 0.1)
            y = np.clip(y, 0.1, field_height - 0.1)
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
    
    print(f"  Generated {n_valid} Voronoi elements, {len(bonds)} bonds", flush=True)
    return polygons, centroids, bonds


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
    n_frames, dt, drift_vel
):
    """
    Simulate ice drifting toward ship with bond fracture.
    
    Key physics from paper:
    - Bonded elements move together as rigid bodies
    - When stress exceeds strength (Eq. 14), bonds break
    - Broken elements then move independently
    """
    n = len(polygons)
    
    # Compute masses and effective radii
    masses = np.array([polygon_area(p) * 0.5 * 920 for p in polygons])
    radii = np.array([np.sqrt(polygon_area(p) / np.pi) for p in polygons])
    
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
    
    # History
    positions_history = [positions.copy()]
    rotations_history = [rotations.copy()]
    bonds_history = [bond_intact.copy()]
    
    # Bond strength (from paper Table 1: σ = 0.6 MPa, C = 1.8 MPa)
    # Lowered for demo to show more visible fracture
    bond_strength = 5e5  # Threshold force for fracture
    
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
                # Strong repulsion force along hull normal
                force_mag = 1e7 * pen
                forces[i, 0] += force_mag * nx
                forces[i, 1] += force_mag * ny
                ship_collision_force[i] = force_mag
                
                # Extra sideways force to encourage flow around hull
                # Push ice toward nearest side of ship
                rel_y = py - ship_pos[1]
                side_direction = 1 if rel_y > 0 else -1
                side_push = 5e6 * pen * side_direction
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
        
        # 4. Propagate stress through ice sheet
        # Elements under compression from ship transfer stress to neighbors
        element_stress = ship_collision_force.copy()
        
        # Propagate stress backward through bonds (multiple iterations)
        for _ in range(3):
            new_stress = element_stress.copy()
            for b, (i, j) in enumerate(bonds):
                if bond_intact[b]:
                    # Stress propagates through intact bonds
                    stress_transfer = 0.6  # 60% of stress transfers
                    if element_stress[i] > element_stress[j]:
                        new_stress[j] += element_stress[i] * stress_transfer
                    else:
                        new_stress[i] += element_stress[j] * stress_transfer
            element_stress = new_stress
        
        # 5. Check for bond fracture with propagated stress
        # Bonds break when stress exceeds strength (Eq. 14 in paper)
        for b, (i, j) in enumerate(bonds):
            if not bond_intact[b]:
                continue
            
            # Compute stress on bond
            d = positions[j] - positions[i]
            dist = np.linalg.norm(d)
            
            # Change in distance from rest (strain)
            stretch = abs(dist - rest_lengths[b])
            
            # Force from bond deformation
            bond_force = 1e6 * stretch
            
            # Add propagated stress from ship collision
            propagated_stress = max(element_stress[i], element_stress[j])
            bond_force += propagated_stress * 0.3
            
            # Determine effective strength
            # Direct ship contact = much weaker
            # Propagated stress = somewhat weaker
            if ship_collision_force[i] > 0 or ship_collision_force[j] > 0:
                effective_strength = bond_strength * 0.1
            elif propagated_stress > bond_strength * 0.5:
                effective_strength = bond_strength * 0.5
            else:
                effective_strength = bond_strength
            
            bond_stresses[b] = bond_force
            
            # Fracture criterion (simplified version of Eq. 14)
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
    ship_outline, initial_centroids, output_file='dpdem_ice_ship.gif', fps=20
):
    """
    Animate ice-ship interaction.
    
    Visualization approach:
    - Draw actual Voronoi polygon shapes
    - Bonded elements appear as one continuous mass (no visible boundaries)
    - Broken bonds show as visible cracks/edges
    """
    n_frames = len(positions_history)
    n_elements = len(polygons)
    
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
    
    # Colors
    water_color = '#1A4D70'
    ice_color = '#D8EAF3'
    crack_color = '#2A5D80'
    ship_color = '#404040'
    
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
        # Elements with no broken bonds: no visible edge (part of solid sheet)
        # Elements with broken bonds: show edges where breaks occurred
        for i in range(n_elements):
            rotated = rotate_polygon(local_polys[i], rotations[i])
            world_poly = rotated + positions[i]
            
            has_breaks = len(broken_neighbors[i]) > 0
            
            if has_breaks:
                # Show this element with visible edge
                edge_color = crack_color
                edge_width = 1.0
            else:
                # Part of intact ice sheet - edge same as fill (invisible)
                edge_color = ice_color
                edge_width = 0.5
            
            patch = MplPolygon(
                world_poly,
                facecolor=ice_color,
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

def main():
    print("=" * 60, flush=True)
    print("DPDEM: Ice-Ship Interaction with Voronoi Ice Elements", flush=True)
    print("=" * 60, flush=True)
    
    # Ice field parameters
    n_elements = 100
    field_width = 45.0
    field_height = 35.0
    
    # Ship parameters
    ship_length = 30.0
    ship_beam = 7.0
    # Start ship further away so we see approach
    ship_pos = np.array([field_width + 25, field_height / 2])
    
    # Create Voronoi ice field
    print("Generating Voronoi ice field...", flush=True)
    polygons, centroids, bonds = create_voronoi_ice_field(
        n_elements, field_width, field_height, seed=42
    )
    print(f"  Created {len(polygons)} ice elements with {len(bonds)} bonds", flush=True)
    
    # Create ship hull (bow facing left toward ice)
    ship_outline = create_ship_hull(ship_length, ship_beam, ship_pos, bow_direction='left')
    
    # Run simulation (longer to see full interaction)
    print("Running simulation...", flush=True)
    positions, rotations, bonds_intact = simulate_ice_ship_interaction(
        polygons=polygons,
        centroids=centroids,
        bonds=bonds,
        ship_pos=ship_pos,
        ship_length=ship_length,
        ship_beam=ship_beam,
        n_frames=350,
        dt=0.05,
        drift_vel=2.0,  # Faster drift
    )
    
    n_final = bonds_intact[-1].sum()
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
        output_file='dpdem_ice_ship.gif',
        fps=20,
    )


if __name__ == '__main__':
    main()
