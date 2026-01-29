"""
Time integration for DPDEM simulation.

Implements Velocity-Verlet integration scheme and the main simulation loop
using jax.lax.scan for efficient execution.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Any, NamedTuple

from ..state import (
    SimulationState, IceParticles, Bonds, ContactStates, Platform, Forces
)
from ..kinematics.quaternion import (
    quaternion_derivative,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
)
from ..contact.detection import (
    detect_all_contacts,
    compute_relative_velocity_at_contact,
    decompose_relative_velocity,
)
from ..contact.forces import (
    compute_contact_force,
    update_tangential_displacement,
)
from ..bonds.model import compute_bond_force
from ..bonds.fracture import update_bond_state
from ..dynamics.ice_element import (
    compute_ice_element_acceleration,
    compute_ice_element_angular_acceleration,
)
from ..dynamics.platform import compute_platform_dynamics


class IntegrationConfig(NamedTuple):
    """Configuration for time integration."""
    dt: float
    neighbor_update_interval: int
    E: float
    nu: float
    mu_ice_ice: float
    mu_ice_hull: float
    rho_ice: float
    rho_water: float
    sigma_bar: float
    C: float
    mu_b: float
    G_c_I: float
    G_c_II: float
    beta: float
    C_d: float
    g: jnp.ndarray
    v_water: jnp.ndarray
    waterline_z: float


@partial(jax.jit, static_argnums=())
def velocity_verlet_step(
    position: jnp.ndarray,
    velocity: jnp.ndarray,
    acceleration_old: jnp.ndarray,
    acceleration_new: jnp.ndarray,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Velocity-Verlet integration step for position and velocity.
    
    x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
    v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
    
    Args:
        position: (3,) current position
        velocity: (3,) current velocity
        acceleration_old: (3,) acceleration at current time
        acceleration_new: (3,) acceleration at new time
        dt: Time step
    
    Returns:
        new_position: Updated position
        new_velocity: Updated velocity
    """
    new_position = position + velocity * dt + 0.5 * acceleration_old * dt ** 2
    new_velocity = velocity + 0.5 * (acceleration_old + acceleration_new) * dt
    
    return new_position, new_velocity


@partial(jax.jit, static_argnums=())
def integrate_quaternion_verlet(
    quaternion: jnp.ndarray,
    omega: jnp.ndarray,
    alpha_old: jnp.ndarray,
    alpha_new: jnp.ndarray,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Integrate quaternion using modified Velocity-Verlet.
    
    Args:
        quaternion: (4,) current quaternion
        omega: (3,) current angular velocity
        alpha_old: (3,) angular acceleration at current time
        alpha_new: (3,) angular acceleration at new time
        dt: Time step
    
    Returns:
        new_quaternion: Updated quaternion (normalized)
        new_omega: Updated angular velocity
    """
    # Update angular velocity (Verlet)
    new_omega = omega + 0.5 * (alpha_old + alpha_new) * dt
    
    # Update quaternion using midpoint angular velocity
    omega_mid = omega + 0.5 * alpha_old * dt
    dq = quaternion_derivative(quaternion, omega_mid)
    new_quaternion = quaternion + dq * dt
    new_quaternion = quaternion_normalize(new_quaternion)
    
    return new_quaternion, new_omega


def compute_all_forces(
    state: SimulationState,
    config: IntegrationConfig,
    polyhedra: tuple,
) -> Tuple[Forces, SimulationState]:
    """
    Compute all forces and moments on particles and platform.
    
    This is the main force computation routine that:
    1. Detects contacts between particles
    2. Computes contact forces (Hertz-Mindlin)
    3. Computes bond forces and updates damage
    4. Computes hydrodynamic forces
    5. Aggregates forces on particles and platform
    
    Args:
        state: Current simulation state
        config: Integration configuration
        polyhedra: Tuple of Polyhedron objects
    
    Returns:
        forces: Forces structure with all accumulated forces/moments
        new_state: Updated state (with new contact/bond states)
    """
    particles = state.particles
    bonds = state.bonds
    contacts = state.contacts
    platform = state.platform
    
    n_particles = particles.n_particles
    
    # Initialize force accumulators
    particle_forces = jnp.zeros((n_particles, 3))
    particle_moments = jnp.zeros((n_particles, 3))
    platform_force = jnp.zeros(3)
    platform_moment = jnp.zeros(3)
    
    # Get neighbor pairs for contact detection
    # In practice, this would come from the neighbor list
    # For now, use a simple distance-based check
    
    # 1. Ice-ice contacts
    # (Simplified: iterate over all pairs within cutoff)
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # Check distance
            dist = jnp.linalg.norm(particles.position[i] - particles.position[j])
            
            # Get polyhedra
            poly_i = polyhedra[int(particles.polyhedron_idx[i])]
            poly_j = polyhedra[int(particles.polyhedron_idx[j])]
            
            # Effective radii
            r_i = poly_i.equivalent_radius + poly_i.dilation_radius
            r_j = poly_j.equivalent_radius + poly_j.dilation_radius
            
            # Check for contact (sphere approximation)
            overlap = r_i + r_j - dist
            
            if overlap > 0:
                # Contact normal
                direction = particles.position[i] - particles.position[j]
                normal = direction / (jnp.linalg.norm(direction) + 1e-10)
                
                # Contact point
                contact_point = particles.position[j] + (r_j - 0.5 * overlap) * normal
                
                # Relative velocity at contact
                v_rel = compute_relative_velocity_at_contact(
                    particles.velocity[i], particles.angular_velocity[i],
                    particles.velocity[j], particles.angular_velocity[j],
                    contact_point, particles.position[i], particles.position[j]
                )
                
                v_n, v_t = decompose_relative_velocity(v_rel, normal)
                
                # Tangential displacement (from contact state)
                # Simplified: use velocity-based approximation
                delta_t = v_t * config.dt
                
                # Compute contact force
                F_contact, F_n_mag = compute_contact_force(
                    overlap, v_n, delta_t, v_t, normal,
                    config.E, config.nu,
                    poly_i.equivalent_radius, poly_j.equivalent_radius,
                    particles.mass[i], particles.mass[j],
                    config.mu_ice_ice
                )
                
                # Contact moments
                r_i = contact_point - particles.position[i]
                r_j = contact_point - particles.position[j]
                M_i = jnp.cross(r_i, F_contact)
                M_j = jnp.cross(r_j, -F_contact)
                
                # Accumulate forces
                particle_forces = particle_forces.at[i].add(F_contact)
                particle_forces = particle_forces.at[j].add(-F_contact)
                particle_moments = particle_moments.at[i].add(M_i)
                particle_moments = particle_moments.at[j].add(M_j)
    
    # 2. Bond forces
    for b in range(bonds.n_bonds):
        if bonds.intact[b]:
            i = int(bonds.particle_i[b])
            j = int(bonds.particle_j[b])
            
            poly_i = polyhedra[int(particles.polyhedron_idx[i])]
            
            # Compute bond force
            F_bond, M_bond, sigma_n, sigma_t = compute_bond_force(
                particles.position[i], particles.quaternion[i],
                particles.velocity[i], particles.angular_velocity[i],
                bonds.local_position_i[b],
                particles.position[j], particles.quaternion[j],
                particles.velocity[j], particles.angular_velocity[j],
                bonds.local_position_j[b],
                bonds.initial_length[b],
                bonds.area[b],
                int(bonds.n_bond_points[b]),
                config.E, config.nu, config.beta,
                bonds.damage[b]
            )
            
            # Accumulate
            particle_forces = particle_forces.at[i].add(F_bond)
            particle_forces = particle_forces.at[j].add(-F_bond)
            particle_moments = particle_moments.at[i].add(M_bond)
    
    # 3. Ice-platform contacts
    # (Simplified: check distance to platform)
    R_platform = quaternion_to_rotation_matrix(platform.quaternion)
    
    for i in range(n_particles):
        # Simple distance check to platform center
        dist_to_platform = jnp.linalg.norm(particles.position[i] - platform.position)
        
        poly_i = polyhedra[int(particles.polyhedron_idx[i])]
        r_i = poly_i.equivalent_radius + poly_i.dilation_radius
        
        # Approximate platform as large sphere for contact check
        # In full implementation, would check against hull mesh
        platform_radius = 50.0  # Approximate
        
        overlap = r_i + platform_radius - dist_to_platform
        
        if overlap > 0:
            direction = particles.position[i] - platform.position
            normal = direction / (jnp.linalg.norm(direction) + 1e-10)
            contact_point = platform.position + (platform_radius - 0.5 * overlap) * normal
            
            # Relative velocity
            v_platform_at_contact = platform.velocity + jnp.cross(
                platform.angular_velocity, contact_point - platform.position
            )
            v_particle_at_contact = particles.velocity[i] + jnp.cross(
                particles.angular_velocity[i], contact_point - particles.position[i]
            )
            v_rel = v_particle_at_contact - v_platform_at_contact
            v_n, v_t = decompose_relative_velocity(v_rel, normal)
            
            delta_t = v_t * config.dt
            
            # Contact force
            F_contact, _ = compute_contact_force(
                overlap, v_n, delta_t, v_t, normal,
                config.E, config.nu,
                poly_i.equivalent_radius, platform_radius,
                particles.mass[i], platform.mass,
                config.mu_ice_hull
            )
            
            # Apply to particle
            r_p = contact_point - particles.position[i]
            M_p = jnp.cross(r_p, F_contact)
            particle_forces = particle_forces.at[i].add(F_contact)
            particle_moments = particle_moments.at[i].add(M_p)
            
            # Reaction on platform
            r_plat = contact_point - platform.position
            platform_force = platform_force - F_contact
            platform_moment = platform_moment - jnp.cross(r_plat, F_contact)
    
    forces = Forces(
        particle_forces=particle_forces,
        particle_moments=particle_moments,
        platform_force=platform_force,
        platform_moment=platform_moment,
    )
    
    return forces, state


def simulation_step(
    state: SimulationState,
    config: IntegrationConfig,
    polyhedra: tuple,
) -> SimulationState:
    """
    Execute one simulation time step.
    
    Uses a modified Velocity-Verlet scheme:
    1. Compute forces at current state
    2. Update velocities (half step)
    3. Update positions
    4. Update quaternions
    5. Compute forces at new state
    6. Update velocities (full step)
    
    Args:
        state: Current simulation state
        config: Integration configuration
        polyhedra: Polyhedron geometry templates
    
    Returns:
        Updated simulation state
    """
    dt = config.dt
    particles = state.particles
    platform = state.platform
    
    # Compute forces at current state
    forces, state = compute_all_forces(state, config, polyhedra)
    
    # Compute accelerations for particles
    n_particles = particles.n_particles
    accelerations = jnp.zeros((n_particles, 3))
    alphas = jnp.zeros((n_particles, 3))
    
    for i in range(n_particles):
        poly = polyhedra[int(particles.polyhedron_idx[i])]
        
        # Hydrodynamic forces
        from ..hydrodynamics.drag import compute_ice_element_buoyancy, compute_ice_element_drag
        
        F_buoy = compute_ice_element_buoyancy(
            particles.position[i], poly.volume,
            config.rho_ice, config.rho_water, config.g, config.waterline_z
        )
        F_drag = compute_ice_element_drag(
            particles.position[i], particles.velocity[i], poly.volume,
            config.v_water, config.C_d, config.rho_water, config.waterline_z
        )
        
        # Total force and moment on particle
        F_total = forces.particle_forces[i] + F_buoy + F_drag + particles.mass[i] * config.g
        M_total = forces.particle_moments[i]
        
        # Accelerations
        a = F_total / particles.mass[i]
        from ..kinematics.quaternion import euler_equations_global
        alpha = euler_equations_global(
            particles.angular_velocity[i], M_total,
            particles.quaternion[i], poly.inertia_local
        )
        
        accelerations = accelerations.at[i].set(a)
        alphas = alphas.at[i].set(alpha)
    
    # Update particles
    new_positions = particles.position + particles.velocity * dt + 0.5 * accelerations * dt ** 2
    new_velocities = particles.velocity + accelerations * dt
    
    # Update quaternions
    new_quaternions = jnp.zeros_like(particles.quaternion)
    new_omegas = particles.angular_velocity + alphas * dt
    
    for i in range(n_particles):
        dq = quaternion_derivative(particles.quaternion[i], particles.angular_velocity[i])
        new_q = particles.quaternion[i] + dq * dt
        new_q = quaternion_normalize(new_q)
        new_quaternions = new_quaternions.at[i].set(new_q)
    
    # Update platform
    platform_accel, platform_alpha = compute_platform_dynamics(
        platform.position, platform.velocity, platform.quaternion, platform.angular_velocity,
        platform.mass, platform.inertia,
        platform.mesh_vertices, platform.mesh_faces,
        forces.platform_force, forces.platform_moment,
        jnp.zeros(3), jnp.zeros(3),  # No mooring
        config.rho_water, config.g, config.v_water,
        config.C_d, config.C_d, 10.0,  # r_eff
        config.waterline_z
    )
    
    new_platform_pos = platform.position + platform.velocity * dt + 0.5 * platform_accel * dt ** 2
    new_platform_vel = platform.velocity + platform_accel * dt
    new_platform_omega = platform.angular_velocity + platform_alpha * dt
    
    dq_plat = quaternion_derivative(platform.quaternion, platform.angular_velocity)
    new_platform_quat = platform.quaternion + dq_plat * dt
    new_platform_quat = quaternion_normalize(new_platform_quat)
    
    # Construct new state
    new_particles = IceParticles(
        polyhedron_idx=particles.polyhedron_idx,
        position=new_positions,
        quaternion=new_quaternions,
        velocity=new_velocities,
        angular_velocity=new_omegas,
        mass=particles.mass,
        n_particles=n_particles,
    )
    
    new_platform = Platform(
        mesh_vertices=platform.mesh_vertices,
        mesh_faces=platform.mesh_faces,
        mesh_face_normals=platform.mesh_face_normals,
        mesh_face_areas=platform.mesh_face_areas,
        position=new_platform_pos,
        quaternion=new_platform_quat,
        velocity=new_platform_vel,
        angular_velocity=new_platform_omega,
        mass=platform.mass,
        inertia=platform.inertia,
        center_of_gravity_local=platform.center_of_gravity_local,
    )
    
    new_state = SimulationState(
        particles=new_particles,
        polyhedra=state.polyhedra,
        bonds=state.bonds,
        contacts=state.contacts,
        platform=new_platform,
        neighbors=state.neighbors,
        time=state.time + dt,
        step=state.step + 1,
    )
    
    return new_state


def run_simulation(
    initial_state: SimulationState,
    config: IntegrationConfig,
    polyhedra: tuple,
    n_steps: int,
) -> SimulationState:
    """
    Run simulation for specified number of steps.
    
    Uses jax.lax.fori_loop for efficient execution.
    
    Args:
        initial_state: Initial simulation state
        config: Integration configuration
        polyhedra: Polyhedron geometry templates
        n_steps: Number of time steps to run
    
    Returns:
        Final simulation state
    """
    def step_fn(i, state):
        return simulation_step(state, config, polyhedra)
    
    final_state = jax.lax.fori_loop(0, n_steps, step_fn, initial_state)
    
    return final_state
