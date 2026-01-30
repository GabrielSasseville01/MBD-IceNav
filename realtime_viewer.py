#!/usr/bin/env python3
"""
Real-time DPDEM viewer using Polyscope + JAX.

Combines JAX-accelerated physics with Polyscope's interactive 3D rendering.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import polyscope as ps
from scipy.spatial import Voronoi

# JAX configuration
jax.config.update("jax_enable_x64", False)


# =============================================================================
# JAX Physics Kernels
# =============================================================================

@jax.jit
def compute_bond_forces(
    positions: jnp.ndarray,      # (N, 3)
    velocities: jnp.ndarray,     # (N, 3)
    bond_i: jnp.ndarray,         # (M,)
    bond_j: jnp.ndarray,         # (M,)
    bond_rest_length: jnp.ndarray,  # (M,)
    bond_intact: jnp.ndarray,    # (M,) bool
    stiffness: float,
    damping: float,
) -> tuple:
    """Compute forces from bonds between elements."""
    # Get positions of bonded pairs
    pos_i = positions[bond_i]  # (M, 3)
    pos_j = positions[bond_j]  # (M, 3)
    vel_i = velocities[bond_i]
    vel_j = velocities[bond_j]
    
    # Bond vectors
    d = pos_j - pos_i  # (M, 3)
    dist = jnp.linalg.norm(d, axis=1, keepdims=True)  # (M, 1)
    dist_safe = jnp.maximum(dist, 1e-6)
    direction = d / dist_safe  # (M, 3)
    
    # Stretch and strain rate
    stretch = dist[:, 0] - bond_rest_length  # (M,)
    rel_vel = vel_j - vel_i
    strain_rate = jnp.sum(rel_vel * direction, axis=1)  # (M,)
    
    # Force magnitude (spring + damping)
    force_mag = stiffness * stretch + damping * strain_rate  # (M,)
    
    # Mask out broken bonds
    force_mag = jnp.where(bond_intact, force_mag, 0.0)
    
    # Force vectors
    forces = force_mag[:, None] * direction  # (M, 3)
    
    # Compute bond stress for fracture check
    bond_stress = jnp.abs(force_mag)
    
    return forces, bond_stress


@jax.jit
def compute_ship_collision(
    positions: jnp.ndarray,      # (N, 3)
    velocities: jnp.ndarray,     # (N, 3)
    radii: jnp.ndarray,          # (N,)
    ship_pos: jnp.ndarray,       # (3,)
    ship_length: float,
    ship_beam: float,
    ship_draft: float,
) -> tuple:
    """Compute collision forces with ship hull."""
    N = positions.shape[0]
    
    # Ship bounds
    bow_x = ship_pos[0] - ship_length / 2
    stern_x = ship_pos[0] + ship_length / 2
    
    # Relative position to ship center
    rel = positions - ship_pos  # (N, 3)
    
    # Check if inside ship bounding box (simplified)
    in_x = (positions[:, 0] > bow_x - radii) & (positions[:, 0] < stern_x + radii)
    in_y = jnp.abs(rel[:, 1]) < (ship_beam / 2 + radii)
    in_z = rel[:, 2] > -ship_draft - radii
    
    in_ship = in_x & in_y & in_z
    
    # Compute penetration and normal for bow collision
    x_rel = rel[:, 0] / (ship_length / 2)  # -1 at bow, +1 at stern
    
    # Bow taper
    taper = jnp.where(
        x_rel < -0.6,
        1.0 - 0.9 * jnp.power(jnp.clip((-x_rel - 0.6) / 0.4, 0, 1), 1.5),
        1.0
    )
    local_beam = ship_beam * taper / 2
    
    # Penetration from bow
    pen_bow = radii - (positions[:, 0] - bow_x)
    
    # Penetration from sides
    pen_side = radii + local_beam - jnp.abs(rel[:, 1])
    
    # Choose dominant collision
    is_bow = pen_bow > pen_side
    penetration = jnp.where(is_bow, pen_bow, pen_side)
    penetration = jnp.maximum(penetration, 0.0)
    
    # Normal direction
    nx = jnp.where(is_bow, -1.0, 0.0)
    ny = jnp.where(is_bow, 0.0, -jnp.sign(rel[:, 1]))
    nz = jnp.zeros(N)
    
    # Add slight downward push at bow (ice goes under)
    nz = jnp.where(is_bow & (x_rel < -0.7), -0.3, nz)
    
    normal = jnp.stack([nx, ny, nz], axis=1)
    normal_len = jnp.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / jnp.maximum(normal_len, 1e-6)
    
    # Force
    force_mag = 1e7 * penetration * jnp.where(in_ship, 1.0, 0.0)
    forces = force_mag[:, None] * normal
    
    # Collision flag
    colliding = in_ship & (penetration > 0.1)
    
    return forces, colliding


@jax.jit
def simulation_step(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    masses: jnp.ndarray,
    radii: jnp.ndarray,
    bond_i: jnp.ndarray,
    bond_j: jnp.ndarray,
    bond_rest_length: jnp.ndarray,
    bond_intact: jnp.ndarray,
    ship_pos: jnp.ndarray,
    ship_length: float,
    ship_beam: float,
    ship_draft: float,
    bond_stiffness: float,
    bond_strength: float,
    dt: float,
    drift_force: float,
) -> tuple:
    """Single simulation step."""
    N = positions.shape[0]
    
    # Initialize forces with drift
    forces = jnp.zeros_like(positions)
    forces = forces.at[:, 0].set(drift_force * masses)
    
    # Bond forces
    bond_forces, bond_stress = compute_bond_forces(
        positions, velocities, bond_i, bond_j, bond_rest_length, bond_intact,
        bond_stiffness, bond_stiffness * 0.1
    )
    
    # Accumulate bond forces (scatter add)
    forces = forces.at[bond_i].add(bond_forces)
    forces = forces.at[bond_j].add(-bond_forces)
    
    # Ship collision
    ship_forces, colliding = compute_ship_collision(
        positions, velocities, radii, ship_pos,
        ship_length, ship_beam, ship_draft
    )
    forces = forces + ship_forces
    
    # Damping (water drag)
    forces = forces - 0.5 * velocities * masses[:, None]
    
    # Gravity on Z (buoyancy approximation)
    forces = forces.at[:, 2].add(-positions[:, 2] * masses * 0.5)
    
    # Integration
    accel = forces / masses[:, None]
    new_velocities = velocities + accel * dt
    new_velocities = jnp.clip(new_velocities, -50, 50)
    new_positions = positions + new_velocities * dt
    
    # Bond fracture check
    # Propagate collision stress through bonds
    collision_stress = jnp.where(colliding, 1e6, 0.0)
    stress_i = collision_stress[bond_i]
    stress_j = collision_stress[bond_j]
    total_stress = bond_stress + jnp.maximum(stress_i, stress_j) * 0.5
    
    new_bond_intact = bond_intact & (total_stress < bond_strength)
    
    return new_positions, new_velocities, new_bond_intact


# =============================================================================
# Geometry Creation
# =============================================================================

def create_voronoi_ice_field(n_elements, field_width, field_height, ice_thickness, seed=42):
    """Create Voronoi ice field with 3D mesh data for polyscope."""
    np.random.seed(seed)
    
    # Generate seed points
    points = []
    nx = int(np.sqrt(n_elements * field_width / field_height))
    ny = int(n_elements / nx)
    dx, dy = field_width / nx, field_height / ny
    
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * dx + np.random.uniform(-0.4*dx, 0.4*dx)
            y = (j + 0.5) * dy + np.random.uniform(-0.4*dy, 0.4*dy)
            points.append([x, y])
    
    # Add random points
    n_random = max(0, n_elements - len(points))
    for _ in range(n_random):
        points.append([np.random.uniform(1, field_width-1), 
                      np.random.uniform(1, field_height-1)])
    
    points = np.array(points)
    
    # Add mirror points for bounded Voronoi
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
    
    # Extract valid polygons
    polygons = []
    centroids = []
    n_orig = len(points)
    
    for i in range(n_orig):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) < 3:
            continue
        
        verts = vor.vertices[region]
        
        # Clip to domain
        verts = np.clip(verts, [0, 0], [field_width, field_height])
        
        if len(verts) >= 3:
            centroid = np.mean(verts, axis=0)
            # Order vertices
            angles = np.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0])
            order = np.argsort(angles)
            verts = verts[order]
            
            polygons.append(verts)
            centroids.append(centroid)
    
    centroids = np.array(centroids)
    n_elements = len(polygons)
    
    # Create 3D meshes for polyscope
    all_vertices = []
    all_faces = []
    element_indices = []  # Which vertices belong to which element
    vertex_offset = 0
    
    for i, poly in enumerate(polygons):
        n_verts = len(poly)
        cx, cy = centroids[i]
        
        # Create 3D vertices (top and bottom faces)
        z_top = ice_thickness / 2
        z_bot = -ice_thickness / 2
        
        # Top vertices, then bottom
        for v in poly:
            all_vertices.append([v[0], v[1], z_top])
        for v in poly:
            all_vertices.append([v[0], v[1], z_bot])
        
        # Top face (fan triangulation)
        for j in range(1, n_verts - 1):
            all_faces.append([vertex_offset, vertex_offset + j, vertex_offset + j + 1])
        
        # Bottom face
        bot_offset = vertex_offset + n_verts
        for j in range(1, n_verts - 1):
            all_faces.append([bot_offset, bot_offset + j + 1, bot_offset + j])
        
        # Side faces
        for j in range(n_verts):
            j_next = (j + 1) % n_verts
            v0 = vertex_offset + j
            v1 = vertex_offset + j_next
            v2 = bot_offset + j_next
            v3 = bot_offset + j
            all_faces.append([v0, v1, v2])
            all_faces.append([v0, v2, v3])
        
        element_indices.append((vertex_offset, vertex_offset + 2 * n_verts))
        vertex_offset += 2 * n_verts
    
    vertices = np.array(all_vertices, dtype=np.float32)
    faces = np.array(all_faces, dtype=np.int32)
    
    # Compute masses and radii
    masses = np.array([np.pi * 2**2 * ice_thickness * 920 for _ in polygons], dtype=np.float32)
    radii = np.array([2.0 for _ in polygons], dtype=np.float32)
    
    # Create bonds (adjacent elements)
    idx_map = {tuple(centroids[i]): i for i in range(n_elements)}
    bonds = []
    threshold = max(field_width, field_height) / np.sqrt(n_elements) * 1.5
    
    for i in range(n_elements):
        for j in range(i + 1, n_elements):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < threshold:
                bonds.append((i, j))
    
    bonds = np.array(bonds, dtype=np.int32) if bonds else np.zeros((0, 2), dtype=np.int32)
    
    return {
        'vertices': vertices,
        'faces': faces,
        'centroids': centroids,
        'polygons': polygons,
        'element_indices': element_indices,
        'masses': masses,
        'radii': radii,
        'bonds': bonds,
    }


def create_ship_mesh(length, beam, draft, position):
    """Create ship hull mesh for polyscope."""
    cx, cy, cz = position
    
    vertices = []
    faces = []
    
    n_sections = 20
    for i in range(n_sections + 1):
        t = i / n_sections
        x = cx + length/2 - t * length
        
        # Taper at bow
        x_rel = -(x - cx) / (length/2)
        if x_rel > 0.6:
            taper = 1.0 - 0.9 * ((x_rel - 0.6) / 0.4) ** 1.5
        else:
            taper = 1.0
        taper = max(taper, 0.05)
        
        local_beam = beam * taper
        local_draft = draft * taper
        deck_h = 3.0 * taper
        
        # Section vertices: deck, waterline, keel
        vertices.extend([
            [x, cy - local_beam/2, cz + deck_h],   # 0: deck port
            [x, cy + local_beam/2, cz + deck_h],   # 1: deck starboard
            [x, cy - local_beam/2 * 0.9, cz],      # 2: waterline port
            [x, cy + local_beam/2 * 0.9, cz],      # 3: waterline starboard
            [x, cy, cz - local_draft],             # 4: keel
        ])
    
    vertices = np.array(vertices, dtype=np.float32)
    
    # Create faces
    vps = 5  # vertices per section
    for i in range(n_sections):
        base = i * vps
        next_base = (i + 1) * vps
        
        # Deck
        faces.append([base, next_base, next_base + 1])
        faces.append([base, next_base + 1, base + 1])
        
        # Port side (deck to waterline)
        faces.append([base, base + 2, next_base + 2])
        faces.append([base, next_base + 2, next_base])
        
        # Starboard side
        faces.append([base + 1, next_base + 1, next_base + 3])
        faces.append([base + 1, next_base + 3, base + 3])
        
        # Port bottom
        faces.append([base + 2, base + 4, next_base + 4])
        faces.append([base + 2, next_base + 4, next_base + 2])
        
        # Starboard bottom
        faces.append([base + 3, next_base + 3, next_base + 4])
        faces.append([base + 3, next_base + 4, base + 4])
    
    faces = np.array(faces, dtype=np.int32)
    
    return vertices, faces


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("DPDEM Real-time Viewer (Polyscope + JAX)")
    print("=" * 60)
    
    # Parameters
    field_width = 200.0
    field_height = 150.0
    n_elements = 100
    ice_thickness = 2.5
    
    ship_length = 158.0
    ship_beam = 28.0
    ship_draft = 10.5
    ship_pos = np.array([field_width + 60, field_height / 2, 0.0], dtype=np.float32)
    
    # Create ice field
    print("Creating ice field...")
    ice_data = create_voronoi_ice_field(n_elements, field_width, field_height, ice_thickness)
    n_actual = len(ice_data['centroids'])
    print(f"  {n_actual} ice elements, {len(ice_data['bonds'])} bonds")
    
    # Create ship
    print("Creating ship...")
    ship_verts, ship_faces = create_ship_mesh(ship_length, ship_beam, ship_draft, ship_pos)
    
    # Initialize JAX arrays
    positions_3d = np.zeros((n_actual, 3), dtype=np.float32)
    positions_3d[:, :2] = ice_data['centroids']
    positions = jnp.array(positions_3d)
    velocities = jnp.zeros_like(positions)
    masses = jnp.array(ice_data['masses'])
    radii = jnp.array(ice_data['radii'])
    
    bond_i = jnp.array(ice_data['bonds'][:, 0])
    bond_j = jnp.array(ice_data['bonds'][:, 1])
    bond_rest_length = jnp.linalg.norm(
        positions[bond_i] - positions[bond_j], axis=1
    )
    bond_intact = jnp.ones(len(bond_i), dtype=bool)
    
    ship_pos_jax = jnp.array(ship_pos)
    
    # Simulation parameters
    dt = 0.02
    bond_stiffness = 5e6
    bond_strength = 2e6
    drift_force = 50.0
    
    # Initialize Polyscope
    print("Initializing Polyscope...")
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    
    # Register meshes
    ice_mesh = ps.register_surface_mesh("ice", ice_data['vertices'], ice_data['faces'])
    ice_mesh.set_color((0.85, 0.92, 0.97))
    ice_mesh.set_edge_width(1)
    ice_mesh.set_edge_color((0.4, 0.6, 0.7))
    
    ship_mesh = ps.register_surface_mesh("ship", ship_verts, ship_faces)
    ship_mesh.set_color((0.6, 0.3, 0.2))
    
    # Water plane
    water_verts = np.array([
        [-50, -50, -0.5],
        [field_width + 200, -50, -0.5],
        [field_width + 200, field_height + 50, -0.5],
        [-50, field_height + 50, -0.5],
    ], dtype=np.float32)
    water_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    water_mesh = ps.register_surface_mesh("water", water_verts, water_faces)
    water_mesh.set_color((0.1, 0.3, 0.5))
    water_mesh.set_transparency(0.7)
    
    # Set camera
    ps.look_at((field_width/2, -100, 80), (field_width/2 + 50, field_height/2, 0))
    
    print("Starting simulation... Close window to exit.")
    
    frame = 0
    while True:
        # Physics step (JAX)
        positions, velocities, bond_intact = simulation_step(
            positions, velocities, masses, radii,
            bond_i, bond_j, bond_rest_length, bond_intact,
            ship_pos_jax, ship_length, ship_beam, ship_draft,
            bond_stiffness, bond_strength, dt, drift_force
        )
        
        # Update mesh positions
        pos_np = np.array(positions)
        new_verts = ice_data['vertices'].copy()
        
        for i, (start, end) in enumerate(ice_data['element_indices']):
            dx = pos_np[i, 0] - ice_data['centroids'][i, 0]
            dy = pos_np[i, 1] - ice_data['centroids'][i, 1]
            dz = pos_np[i, 2]
            new_verts[start:end, 0] += dx
            new_verts[start:end, 1] += dy
            new_verts[start:end, 2] += dz
        
        ice_mesh.update_vertex_positions(new_verts)
        
        # Color by bond status
        n_intact = int(jnp.sum(bond_intact))
        n_total = len(bond_intact)
        
        # Render frame
        ps.frame_tick()
        
        if not ps.is_initialized():
            break
        
        frame += 1
        if frame % 50 == 0:
            print(f"Frame {frame}: {n_intact}/{n_total} bonds intact")


if __name__ == "__main__":
    main()
