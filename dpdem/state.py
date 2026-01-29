"""
JAX pytree state structures for DPDEM simulation.

All state classes are NamedTuples which are automatically registered as JAX pytrees.
This enables seamless use with jax.jit, jax.vmap, and jax.lax.scan.
"""

from typing import NamedTuple, Any
import jax.numpy as jnp


class Polyhedron(NamedTuple):
    """
    Polyhedron geometry template.
    
    Represents a dilated polyhedron (Minkowski sum of polyhedron + sphere).
    Vertices are in the local (body) coordinate frame centered at the centroid.
    
    Attributes:
        vertices: Vertex positions in local frame, shape (n_vertices, 3)
        faces: Face vertex indices, shape (n_faces, max_verts_per_face), padded with -1
        face_sizes: Number of vertices per face, shape (n_faces,)
        face_normals: Outward face normals, shape (n_faces, 3)
        dilation_radius: Sphere radius for Minkowski sum [m]
        volume: Polyhedron volume [m³]
        equivalent_radius: Radius of volume-equivalent sphere [m]
        inertia_local: Diagonal inertia tensor in body frame [kg·m²], shape (3,)
    """
    vertices: jnp.ndarray           # (n_vertices, 3)
    faces: jnp.ndarray              # (n_faces, max_verts_per_face)
    face_sizes: jnp.ndarray         # (n_faces,)
    face_normals: jnp.ndarray       # (n_faces, 3)
    dilation_radius: float
    volume: float
    equivalent_radius: float
    inertia_local: jnp.ndarray      # (3,) - I_xx, I_yy, I_zz


class IceParticle(NamedTuple):
    """
    State of a single sea ice element.
    
    Each ice particle is a rigid body with 6-DOF motion described by
    position, orientation (quaternion), and their time derivatives.
    
    Attributes:
        polyhedron_idx: Index into array of Polyhedron templates
        position: Center of mass position in global frame [m], shape (3,)
        quaternion: Orientation as [q0, q1, q2, q3] where q0 is scalar, shape (4,)
        velocity: Linear velocity in global frame [m/s], shape (3,)
        angular_velocity: Angular velocity in global frame [rad/s], shape (3,)
        mass: Particle mass [kg]
    """
    polyhedron_idx: int
    position: jnp.ndarray           # (3,)
    quaternion: jnp.ndarray         # (4,) - [q0, q1, q2, q3], scalar first
    velocity: jnp.ndarray           # (3,)
    angular_velocity: jnp.ndarray   # (3,) - in global frame
    mass: float


class IceParticles(NamedTuple):
    """
    Batched state of all ice particles.
    
    All arrays have leading dimension N (number of particles).
    
    Attributes:
        polyhedron_idx: Indices into Polyhedron templates, shape (N,)
        position: Positions, shape (N, 3)
        quaternion: Orientations, shape (N, 4)
        velocity: Velocities, shape (N, 3)
        angular_velocity: Angular velocities, shape (N, 3)
        mass: Masses, shape (N,)
        n_particles: Number of active particles
    """
    polyhedron_idx: jnp.ndarray     # (N,)
    position: jnp.ndarray           # (N, 3)
    quaternion: jnp.ndarray         # (N, 4)
    velocity: jnp.ndarray           # (N, 3)
    angular_velocity: jnp.ndarray   # (N, 3)
    mass: jnp.ndarray               # (N,)
    n_particles: int


class Bond(NamedTuple):
    """
    State of a single bond between two ice elements.
    
    Bonds connect adjacent ice elements and can fail according to
    the mixed-mode fracture criterion.
    
    Attributes:
        particle_i: Index of first particle
        particle_j: Index of second particle
        area: Bond area [m²]
        n_bond_points: Number of bond points at the interface
        local_position_i: Bond attachment point in particle i's local frame, shape (3,)
        local_position_j: Bond attachment point in particle j's local frame, shape (3,)
        initial_length: Initial bond length [m]
        intact: Whether bond is still intact
        damage: Damage variable D from Eq. 16, 0 = undamaged, 1 = failed
        delta_m_max: Maximum equivalent deformation during loading history [m]
        delta_m_0: Deformation at damage initiation [m]
    """
    particle_i: int
    particle_j: int
    area: float
    n_bond_points: int
    local_position_i: jnp.ndarray   # (3,)
    local_position_j: jnp.ndarray   # (3,)
    initial_length: float
    intact: bool
    damage: float
    delta_m_max: float
    delta_m_0: float


class Bonds(NamedTuple):
    """
    Batched state of all bonds.
    
    All arrays have leading dimension M (number of bonds).
    
    Attributes:
        particle_i: First particle indices, shape (M,)
        particle_j: Second particle indices, shape (M,)
        area: Bond areas, shape (M,)
        n_bond_points: Number of bond points, shape (M,)
        local_position_i: Attachment points in particle i frames, shape (M, 3)
        local_position_j: Attachment points in particle j frames, shape (M, 3)
        initial_length: Initial lengths, shape (M,)
        intact: Intact flags, shape (M,)
        damage: Damage variables, shape (M,)
        delta_m_max: Maximum deformations, shape (M,)
        delta_m_0: Damage initiation deformations, shape (M,)
        n_bonds: Number of bonds (including broken ones)
    """
    particle_i: jnp.ndarray         # (M,)
    particle_j: jnp.ndarray         # (M,)
    area: jnp.ndarray               # (M,)
    n_bond_points: jnp.ndarray      # (M,)
    local_position_i: jnp.ndarray   # (M, 3)
    local_position_j: jnp.ndarray   # (M, 3)
    initial_length: jnp.ndarray     # (M,)
    intact: jnp.ndarray             # (M,) bool
    damage: jnp.ndarray             # (M,)
    delta_m_max: jnp.ndarray        # (M,)
    delta_m_0: jnp.ndarray          # (M,)
    n_bonds: int


class ContactState(NamedTuple):
    """
    State tracking for a single contact pair.
    
    Tracks tangential displacement history for Mindlin contact model.
    
    Attributes:
        tangential_displacement: Accumulated tangential displacement [m], shape (3,)
        active: Whether contact is currently active
        particle_i: Index of first particle
        particle_j: Index of second particle
    """
    tangential_displacement: jnp.ndarray  # (3,)
    active: bool
    particle_i: int
    particle_j: int


class ContactStates(NamedTuple):
    """
    Batched contact state for all potential contacts.
    
    Uses fixed-size arrays with validity masks for JAX compatibility.
    
    Attributes:
        tangential_displacement: Tangential displacements, shape (N, max_contacts, 3)
        active: Active flags, shape (N, max_contacts)
        partner_idx: Partner particle index for each contact slot, shape (N, max_contacts)
    """
    tangential_displacement: jnp.ndarray  # (N, max_contacts, 3)
    active: jnp.ndarray                   # (N, max_contacts) bool
    partner_idx: jnp.ndarray              # (N, max_contacts) int


class Platform(NamedTuple):
    """
    State of the floating platform structure.
    
    The platform is a rigid body with 6-DOF motion, represented as a
    triangular mesh for buoyancy and contact calculations.
    
    Attributes:
        mesh_vertices: Vertex positions in local frame [m], shape (n_verts, 3)
        mesh_faces: Triangle vertex indices, shape (n_faces, 3)
        mesh_face_normals: Outward face normals, shape (n_faces, 3)
        mesh_face_areas: Face areas [m²], shape (n_faces,)
        position: Center of gravity in global frame [m], shape (3,)
        quaternion: Orientation [q0, q1, q2, q3], shape (4,)
        velocity: Linear velocity [m/s], shape (3,)
        angular_velocity: Angular velocity [rad/s], shape (3,)
        mass: Platform mass [kg]
        inertia: Inertia tensor in body frame [kg·m²], shape (3, 3)
        center_of_gravity_local: CoG offset in body frame [m], shape (3,)
    """
    mesh_vertices: jnp.ndarray      # (n_verts, 3)
    mesh_faces: jnp.ndarray         # (n_faces, 3)
    mesh_face_normals: jnp.ndarray  # (n_faces, 3)
    mesh_face_areas: jnp.ndarray    # (n_faces,)
    position: jnp.ndarray           # (3,)
    quaternion: jnp.ndarray         # (4,)
    velocity: jnp.ndarray           # (3,)
    angular_velocity: jnp.ndarray   # (3,)
    mass: float
    inertia: jnp.ndarray            # (3, 3)
    center_of_gravity_local: jnp.ndarray  # (3,)


class Forces(NamedTuple):
    """
    Accumulated forces and moments on all particles.
    
    Attributes:
        particle_forces: Total force on each particle [N], shape (N, 3)
        particle_moments: Total moment on each particle [N·m], shape (N, 3)
        platform_force: Total force on platform [N], shape (3,)
        platform_moment: Total moment on platform [N·m], shape (3,)
    """
    particle_forces: jnp.ndarray    # (N, 3)
    particle_moments: jnp.ndarray   # (N, 3)
    platform_force: jnp.ndarray     # (3,)
    platform_moment: jnp.ndarray    # (3,)


class SimulationState(NamedTuple):
    """
    Complete state of the DPDEM simulation.
    
    This is the main state object passed through the simulation loop.
    
    Attributes:
        particles: State of all ice particles
        polyhedra: Array of polyhedron geometry templates
        bonds: State of all inter-element bonds
        contacts: Contact tracking state
        platform: State of the floating platform
        neighbors: JAX-MD neighbor list (opaque type)
        time: Current simulation time [s]
        step: Current simulation step
    """
    particles: IceParticles
    polyhedra: tuple                # Tuple of Polyhedron (heterogeneous shapes)
    bonds: Bonds
    contacts: ContactStates
    platform: Platform
    neighbors: Any                  # JAX-MD NeighborList
    time: float
    step: int


# Factory functions for creating initial states

def create_ice_particles(
    n_particles: int,
    polyhedron_idx: jnp.ndarray,
    positions: jnp.ndarray,
    masses: jnp.ndarray,
) -> IceParticles:
    """
    Create initial ice particles state.
    
    Args:
        n_particles: Number of particles
        polyhedron_idx: Polyhedron template indices, shape (N,)
        positions: Initial positions, shape (N, 3)
        masses: Particle masses, shape (N,)
    
    Returns:
        IceParticles state with zero velocities and identity orientation
    """
    return IceParticles(
        polyhedron_idx=polyhedron_idx,
        position=positions,
        quaternion=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (n_particles, 1)),
        velocity=jnp.zeros((n_particles, 3)),
        angular_velocity=jnp.zeros((n_particles, 3)),
        mass=masses,
        n_particles=n_particles,
    )


def create_bonds(
    particle_i: jnp.ndarray,
    particle_j: jnp.ndarray,
    areas: jnp.ndarray,
    n_bond_points: jnp.ndarray,
    local_positions_i: jnp.ndarray,
    local_positions_j: jnp.ndarray,
    initial_lengths: jnp.ndarray,
) -> Bonds:
    """
    Create initial bonds state.
    
    Args:
        particle_i: First particle indices, shape (M,)
        particle_j: Second particle indices, shape (M,)
        areas: Bond areas, shape (M,)
        n_bond_points: Number of bond points, shape (M,)
        local_positions_i: Attachment points in particle i frames, shape (M, 3)
        local_positions_j: Attachment points in particle j frames, shape (M, 3)
        initial_lengths: Initial bond lengths, shape (M,)
    
    Returns:
        Bonds state with all bonds intact and zero damage
    """
    n_bonds = len(particle_i)
    return Bonds(
        particle_i=particle_i,
        particle_j=particle_j,
        area=areas,
        n_bond_points=n_bond_points,
        local_position_i=local_positions_i,
        local_position_j=local_positions_j,
        initial_length=initial_lengths,
        intact=jnp.ones(n_bonds, dtype=bool),
        damage=jnp.zeros(n_bonds),
        delta_m_max=jnp.zeros(n_bonds),
        delta_m_0=jnp.zeros(n_bonds),
        n_bonds=n_bonds,
    )


def create_contact_states(
    n_particles: int,
    max_contacts_per_particle: int,
) -> ContactStates:
    """
    Create initial contact tracking state.
    
    Args:
        n_particles: Number of particles
        max_contacts_per_particle: Maximum contacts per particle
    
    Returns:
        ContactStates with no active contacts
    """
    return ContactStates(
        tangential_displacement=jnp.zeros((n_particles, max_contacts_per_particle, 3)),
        active=jnp.zeros((n_particles, max_contacts_per_particle), dtype=bool),
        partner_idx=jnp.full((n_particles, max_contacts_per_particle), -1, dtype=jnp.int32),
    )


def create_platform(
    mesh_vertices: jnp.ndarray,
    mesh_faces: jnp.ndarray,
    mass: float,
    inertia: jnp.ndarray,
    initial_position: jnp.ndarray,
) -> Platform:
    """
    Create initial platform state.
    
    Args:
        mesh_vertices: Vertex positions in local frame, shape (n_verts, 3)
        mesh_faces: Triangle indices, shape (n_faces, 3)
        mass: Platform mass [kg]
        inertia: Inertia tensor [kg·m²], shape (3, 3)
        initial_position: Initial position of CoG, shape (3,)
    
    Returns:
        Platform state with zero velocity
    """
    # Compute face normals and areas
    v0 = mesh_vertices[mesh_faces[:, 0]]
    v1 = mesh_vertices[mesh_faces[:, 1]]
    v2 = mesh_vertices[mesh_faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = jnp.cross(edge1, edge2)
    areas = 0.5 * jnp.linalg.norm(cross, axis=1)
    normals = cross / (2.0 * areas[:, None] + 1e-10)
    
    return Platform(
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        mesh_face_normals=normals,
        mesh_face_areas=areas,
        position=initial_position,
        quaternion=jnp.array([1.0, 0.0, 0.0, 0.0]),
        velocity=jnp.zeros(3),
        angular_velocity=jnp.zeros(3),
        mass=mass,
        inertia=inertia,
        center_of_gravity_local=jnp.zeros(3),
    )
