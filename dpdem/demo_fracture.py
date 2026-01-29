"""
DPDEM Ice Fracture Simulation - JAX Implementation

Simulates bonded ice elements drifting toward a ship/platform.
Bonds break when stress exceeds strength threshold upon impact.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple
import numpy as np


# ============================================================================
# State Structures
# ============================================================================

class ParticleState(NamedTuple):
    """Ice element state."""
    position: jnp.ndarray       # (N, 2) positions
    velocity: jnp.ndarray       # (N, 2) velocities
    radius: jnp.ndarray         # (N,) radii
    mass: jnp.ndarray           # (N,) masses


class BondState(NamedTuple):
    """Bond connectivity and state."""
    particle_i: jnp.ndarray     # (M,) indices
    particle_j: jnp.ndarray     # (M,) indices
    rest_length: jnp.ndarray    # (M,) initial lengths
    stiffness: jnp.ndarray      # (M,) bond stiffness
    strength: jnp.ndarray       # (M,) breaking threshold
    intact: jnp.ndarray         # (M,) bool


class ShipGeometry(NamedTuple):
    """Ship hull for collision."""
    center: jnp.ndarray     # (2,) position
    length: float
    beam: float


class SimState(NamedTuple):
    """Full simulation state."""
    particles: ParticleState
    bonds: BondState
    time: float


# ============================================================================
# Bond Force Computation
# ============================================================================

def compute_bond_forces_and_check_fracture(
    particles: ParticleState,
    bonds: BondState,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute spring forces from bonds and check for fracture.
    
    Returns:
        forces: (N, 2) total force on each particle
        should_break: (M,) bool mask of bonds exceeding strength
    """
    N = particles.position.shape[0]
    
    # Get positions of bonded particles
    pos_i = particles.position[bonds.particle_i]
    pos_j = particles.position[bonds.particle_j]
    
    # Bond vectors and lengths
    bond_vec = pos_j - pos_i
    current_length = jnp.linalg.norm(bond_vec, axis=1) + 1e-10
    bond_dir = bond_vec / current_length[:, None]
    
    # Elongation (positive = tension)
    elongation = current_length - bonds.rest_length
    
    # Spring force magnitude (positive pulls toward j)
    force_mag = bonds.stiffness * elongation
    
    # Check fracture: |force| > strength
    should_break = jnp.abs(force_mag) > bonds.strength
    
    # Zero out forces for broken bonds
    force_mag = jnp.where(bonds.intact, force_mag, 0.0)
    
    # Force vectors
    forces_on_i = force_mag[:, None] * bond_dir  # Pulls i toward j
    
    # Scatter to particles
    forces = jnp.zeros((N, 2))
    forces = forces.at[bonds.particle_i].add(forces_on_i)
    forces = forces.at[bonds.particle_j].add(-forces_on_i)
    
    return forces, should_break


# ============================================================================
# Ship Collision
# ============================================================================

def compute_ship_collision(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    radius: float,
    ship: ShipGeometry,
) -> Tuple[jnp.ndarray, float]:
    """
    Compute collision force and penetration for one particle.
    Simple axis-aligned box collision.
    
    Returns:
        force: (2,) collision force
        penetration: scalar penetration depth
    """
    # Ship bounds
    ship_left = ship.center[0] - ship.length / 2
    ship_right = ship.center[0] + ship.length / 2
    ship_bottom = ship.center[1] - ship.beam / 2
    ship_top = ship.center[1] + ship.beam / 2
    
    # Particle bounds
    p_left = pos[0] - radius
    p_right = pos[0] + radius
    p_bottom = pos[1] - radius
    p_top = pos[1] + radius
    
    # Check overlap on each axis
    overlap_x = jnp.minimum(p_right, ship_right) - jnp.maximum(p_left, ship_left)
    overlap_y = jnp.minimum(p_top, ship_top) - jnp.maximum(p_bottom, ship_bottom)
    
    # Both must be positive for collision
    is_colliding = (overlap_x > 0) & (overlap_y > 0)
    
    # Penetration is the smaller overlap
    penetration = jnp.where(
        is_colliding,
        jnp.minimum(overlap_x, overlap_y),
        0.0
    )
    
    # Normal direction - push in direction of smaller penetration
    # If overlap_x < overlap_y, push in x direction
    rel_x = pos[0] - ship.center[0]
    rel_y = pos[1] - ship.center[1]
    
    normal_x = jnp.where(overlap_x < overlap_y, jnp.sign(rel_x), 0.0)
    normal_y = jnp.where(overlap_x >= overlap_y, jnp.sign(rel_y), 0.0)
    normal = jnp.array([normal_x, normal_y])
    normal = normal / (jnp.linalg.norm(normal) + 1e-10)
    
    # Contact force
    stiffness = 5e6
    damping = 5e3
    
    v_normal = jnp.dot(vel, normal)
    force_mag = jnp.where(
        is_colliding,
        stiffness * penetration - damping * jnp.minimum(v_normal, 0),
        0.0
    )
    
    return force_mag * normal, penetration


def compute_all_ship_collisions(
    particles: ParticleState,
    ship: ShipGeometry,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute ship collision for all particles."""
    collision_vmap = jax.vmap(
        compute_ship_collision,
        in_axes=(0, 0, 0, None)
    )
    forces, penetrations = collision_vmap(
        particles.position,
        particles.velocity,
        particles.radius,
        ship
    )
    return forces, penetrations


# ============================================================================
# Particle-Particle Contact (for broken bonds)
# ============================================================================

def compute_pair_contact(
    pos_i: jnp.ndarray,
    pos_j: jnp.ndarray,
    vel_i: jnp.ndarray,
    vel_j: jnp.ndarray,
    r_i: float,
    r_j: float,
) -> jnp.ndarray:
    """Contact force between two particles."""
    d = pos_j - pos_i
    dist = jnp.linalg.norm(d) + 1e-10
    overlap = r_i + r_j - dist
    
    normal = d / dist
    v_rel = vel_j - vel_i
    v_n = jnp.dot(v_rel, normal)
    
    # Hertz contact
    stiffness = 1e6
    damping = 1e3
    
    force_mag = jnp.where(
        overlap > 0,
        stiffness * overlap ** 1.5 - damping * v_n,
        0.0
    )
    
    return -force_mag * normal  # Force on i


# ============================================================================
# Time Integration
# ============================================================================

def simulation_step(
    state: SimState,
    dt: float,
    drift_vel: float,
    ship: ShipGeometry,
) -> SimState:
    """Single simulation step."""
    particles = state.particles
    bonds = state.bonds
    N = particles.position.shape[0]
    
    # 1. Bond forces and fracture check
    bond_forces, should_break = compute_bond_forces_and_check_fracture(
        particles, bonds
    )
    
    # Update intact bonds (break those exceeding strength)
    new_intact = bonds.intact & ~should_break
    new_bonds = bonds._replace(intact=new_intact)
    
    # 2. Ship collision forces
    ship_forces, _ = compute_all_ship_collisions(particles, ship)
    
    # 3. Drift force (gentle push toward ship)
    # Keep it gentle so bonds don't break during drift
    drift_force = jnp.array([drift_vel * particles.mass[0] * 5, 0.0])
    drift_forces = jnp.broadcast_to(drift_force, (N, 2))
    
    # 4. Damping (water resistance)
    damping_coef = 0.5
    damping_forces = -damping_coef * particles.velocity * particles.mass[:, None]
    
    # 5. Total forces and acceleration
    total_forces = bond_forces + ship_forces + drift_forces + damping_forces
    accel = total_forces / particles.mass[:, None]
    
    # 6. Semi-implicit Euler integration
    new_vel = particles.velocity + accel * dt
    new_pos = particles.position + new_vel * dt
    
    # Clamp velocities
    new_vel = jnp.clip(new_vel, -20, 20)
    
    new_particles = particles._replace(
        position=new_pos,
        velocity=new_vel
    )
    
    return SimState(
        particles=new_particles,
        bonds=new_bonds,
        time=state.time + dt
    )


# ============================================================================
# Initialization
# ============================================================================

def create_ice_sheet(
    n_x: int,
    n_y: int,
    spacing: float,
    element_radius: float,
    start_x: float = 0.0,
    start_y: float = 0.0,
) -> Tuple[ParticleState, BondState]:
    """
    Create a bonded grid of ice particles.
    """
    # Grid positions
    positions = []
    for j in range(n_y):
        for i in range(n_x):
            x = start_x + i * spacing
            y = start_y + j * spacing
            positions.append([x, y])
    
    positions = np.array(positions)
    N = len(positions)
    
    # Particle properties
    ice_thickness = 0.5
    density = 920.0
    volume = np.pi * element_radius**2 * ice_thickness
    mass = density * volume
    
    particles = ParticleState(
        position=jnp.array(positions),
        velocity=jnp.zeros((N, 2)),
        radius=jnp.full(N, element_radius),
        mass=jnp.full(N, mass),
    )
    
    # Create bonds between adjacent particles
    bond_threshold = spacing * 1.5
    bonds_i, bonds_j = [], []
    rest_lengths = []
    
    for i in range(N):
        for j in range(i + 1, N):
            d = positions[j] - positions[i]
            dist = np.linalg.norm(d)
            if dist < bond_threshold:
                bonds_i.append(i)
                bonds_j.append(j)
                rest_lengths.append(dist)
    
    M = len(bonds_i)
    
    # Bond properties - stiff enough to keep ice together during drift
    # but will break under ship collision forces
    bond_stiffness = 5e6  # N/m - reduced to allow some flex
    bond_strength = 2e5   # N - force threshold for breaking (increased)
    
    bonds = BondState(
        particle_i=jnp.array(bonds_i, dtype=jnp.int32),
        particle_j=jnp.array(bonds_j, dtype=jnp.int32),
        rest_length=jnp.array(rest_lengths),
        stiffness=jnp.full(M, bond_stiffness),
        strength=jnp.full(M, bond_strength),
        intact=jnp.ones(M, dtype=bool),
    )
    
    return particles, bonds


def run_fracture_simulation(
    n_x: int = 12,
    n_y: int = 8,
    spacing: float = 3.0,
    element_radius: float = 1.2,
    n_steps: int = 2000,
    dt: float = 0.005,
    drift_vel: float = 0.5,
    save_interval: int = 10,
) -> Tuple[list, list, ShipGeometry, float]:
    """
    Run the fracture simulation.
    
    Returns:
        positions: List of position arrays
        bonds: List of intact masks
        ship: Ship geometry
        element_radius: For visualization
    """
    print(f"Creating ice sheet: {n_x} x {n_y} = {n_x * n_y} particles")
    
    # Ice sheet starts on the left
    particles, bonds = create_ice_sheet(
        n_x, n_y, spacing, element_radius,
        start_x=5.0, start_y=5.0
    )
    
    # Ship on the right
    ice_width = n_x * spacing
    ice_height = n_y * spacing
    ship = ShipGeometry(
        center=jnp.array([ice_width + 25, ice_height / 2 + 5]),
        length=25.0,
        beam=10.0,
    )
    
    print(f"Created {bonds.particle_i.shape[0]} bonds")
    print(f"Ship at ({float(ship.center[0]):.1f}, {float(ship.center[1]):.1f})")
    
    state = SimState(particles=particles, bonds=bonds, time=0.0)
    
    # JIT compile step function
    @jax.jit
    def step(s):
        return simulation_step(s, dt, drift_vel, ship)
    
    # Run simulation
    positions_history = [np.array(state.particles.position)]
    bonds_history = [np.array(state.bonds.intact)]
    
    print(f"Running {n_steps} steps...")
    
    for i in range(n_steps):
        state = step(state)
        
        if (i + 1) % save_interval == 0:
            positions_history.append(np.array(state.particles.position))
            bonds_history.append(np.array(state.bonds.intact))
        
        if (i + 1) % (n_steps // 10) == 0:
            n_intact = int(state.bonds.intact.sum())
            total = len(state.bonds.intact)
            print(f"  Step {i+1}/{n_steps}, Time: {state.time:.2f}s, "
                  f"Bonds: {n_intact}/{total}")
    
    print("Done!")
    return positions_history, bonds_history, ship, element_radius


if __name__ == "__main__":
    positions, bonds, ship, radius = run_fracture_simulation()
    print(f"Saved {len(positions)} frames")
    print(f"Final intact bonds: {bonds[-1].sum()}/{len(bonds[-1])}")
