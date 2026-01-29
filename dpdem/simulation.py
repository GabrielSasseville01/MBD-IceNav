"""
Main simulation module for DPDEM.

Provides high-level interface for setting up and running simulations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Optional

from .state import (
    SimulationState, IceParticles, Bonds, ContactStates, Platform,
    Polyhedron, create_ice_particles, create_bonds, create_contact_states, create_platform
)
from .config import (
    DEFAULT_ICE_PROPERTIES, DEFAULT_FLUID_PROPERTIES, DEFAULT_SIMULATION_CONFIG,
    IceProperties, FluidProperties, SimulationConfig
)
from .geometry.voronoi import generate_voronoi_polygons
from .geometry.polyhedra import create_polyhedron_from_polygon
from .geometry.mesh import create_hull_mesh_from_parameters
from .bonds.model import initialize_bonds
from .integration.timestepper import IntegrationConfig, run_simulation, simulation_step


def create_ice_field(
    n_elements: int,
    domain_size: Tuple[float, float],
    ice_thickness: float,
    ice_properties: IceProperties = DEFAULT_ICE_PROPERTIES,
    dilation_radius: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[Polyhedron], IceParticles]:
    """
    Create an ice field with random Voronoi-based elements.
    
    Args:
        n_elements: Number of ice elements
        domain_size: (width, height) of ice field [m]
        ice_thickness: Ice thickness [m]
        ice_properties: Physical properties of ice
        dilation_radius: Dilation radius for contact [m]
        seed: Random seed for reproducibility
    
    Returns:
        polyhedra: List of Polyhedron objects
        particles: IceParticles state
    """
    # Generate Voronoi polygons
    polygons, centroids = generate_voronoi_polygons(n_elements, domain_size, seed)
    
    n_actual = len(polygons)
    
    # Create polyhedra from polygons
    polyhedra = []
    for polygon in polygons:
        poly = create_polyhedron_from_polygon(
            polygon, ice_thickness, ice_properties.density, dilation_radius
        )
        polyhedra.append(poly)
    
    # Create particle positions (centroids at waterline)
    positions = np.zeros((n_actual, 3))
    positions[:, 0] = centroids[:, 0]
    positions[:, 1] = centroids[:, 1]
    positions[:, 2] = 0.0  # At waterline
    
    # Compute masses
    masses = np.array([poly.volume * ice_properties.density for poly in polyhedra])
    
    # Create particles state
    particles = create_ice_particles(
        n_actual,
        jnp.arange(n_actual),  # Each particle uses its own polyhedron
        jnp.array(positions),
        jnp.array(masses),
    )
    
    return polyhedra, particles


def create_platform_from_params(
    length: float,
    breadth: float,
    draft: float,
    displacement: float,
    initial_position: jnp.ndarray,
) -> Platform:
    """
    Create a platform with simplified box geometry.
    
    Args:
        length: Platform length [m]
        breadth: Platform breadth [m]
        draft: Platform draft [m]
        displacement: Platform displacement [kg]
        initial_position: (3,) initial position of CoG
    
    Returns:
        Platform state
    """
    # Create hull mesh
    vertices, faces = create_hull_mesh_from_parameters(
        length, breadth, draft, freeboard=5.0
    )
    
    # Compute inertia (box approximation)
    mass = displacement
    I_xx = (1.0/12.0) * mass * (breadth**2 + draft**2)
    I_yy = (1.0/12.0) * mass * (length**2 + draft**2)
    I_zz = (1.0/12.0) * mass * (length**2 + breadth**2)
    inertia = jnp.diag(jnp.array([I_xx, I_yy, I_zz]))
    
    return create_platform(vertices, faces, mass, inertia, initial_position)


def setup_simulation(
    n_ice_elements: int,
    ice_field_size: Tuple[float, float],
    ice_thickness: float,
    platform_length: float,
    platform_breadth: float,
    platform_draft: float,
    platform_displacement: float,
    ice_properties: IceProperties = DEFAULT_ICE_PROPERTIES,
    fluid_properties: FluidProperties = DEFAULT_FLUID_PROPERTIES,
    sim_config: SimulationConfig = DEFAULT_SIMULATION_CONFIG,
    seed: Optional[int] = None,
) -> Tuple[SimulationState, IntegrationConfig, tuple]:
    """
    Set up a complete simulation.
    
    Args:
        n_ice_elements: Number of ice elements
        ice_field_size: (width, height) of ice field [m]
        ice_thickness: Ice thickness [m]
        platform_length: Platform length [m]
        platform_breadth: Platform breadth [m]
        platform_draft: Platform draft [m]
        platform_displacement: Platform mass [kg]
        ice_properties: Ice physical properties
        fluid_properties: Water physical properties
        sim_config: Simulation configuration
        seed: Random seed
    
    Returns:
        state: Initial simulation state
        config: Integration configuration
        polyhedra: Polyhedron geometry templates
    """
    # Create ice field
    polyhedra, particles = create_ice_field(
        n_ice_elements, ice_field_size, ice_thickness,
        ice_properties, dilation_radius=0.1, seed=seed
    )
    
    # Create platform (centered in domain)
    platform_position = jnp.array([
        ice_field_size[0] / 2,
        ice_field_size[1] / 2,
        -platform_draft / 2  # CoG below waterline
    ])
    platform = create_platform_from_params(
        platform_length, platform_breadth, platform_draft,
        platform_displacement, platform_position
    )
    
    # Initialize bonds between adjacent elements
    adjacency_threshold = sim_config.element_size * 1.5
    bonds_data = initialize_bonds(
        np.array(particles.position),
        np.array(particles.quaternion),
        [np.array(p.vertices) for p in polyhedra],
        adjacency_threshold
    )
    
    bonds = create_bonds(*[jnp.array(x) for x in bonds_data])
    
    # Create contact states
    contacts = create_contact_states(
        particles.n_particles,
        sim_config.max_contacts_per_particle
    )
    
    # Create initial simulation state
    state = SimulationState(
        particles=particles,
        polyhedra=tuple(polyhedra),
        bonds=bonds,
        contacts=contacts,
        platform=platform,
        neighbors=None,  # Will be initialized on first step
        time=0.0,
        step=0,
    )
    
    # Create integration config
    config = IntegrationConfig(
        dt=sim_config.dt,
        neighbor_update_interval=sim_config.neighbor_update_interval,
        E=ice_properties.elastic_modulus,
        nu=ice_properties.poisson_ratio,
        mu_ice_ice=ice_properties.ice_ice_friction,
        mu_ice_hull=ice_properties.ice_hull_friction,
        rho_ice=ice_properties.density,
        rho_water=fluid_properties.density,
        sigma_bar=ice_properties.normal_strength,
        C=ice_properties.cohesion,
        mu_b=ice_properties.internal_friction_coef,
        G_c_I=ice_properties.fracture_energy_mode_I,
        G_c_II=ice_properties.fracture_energy_mode_II,
        beta=sim_config.bond_damping_beta,
        C_d=fluid_properties.drag_coefficient_force,
        g=sim_config.gravity,
        v_water=jnp.zeros(3),  # No current by default
        waterline_z=0.0,
    )
    
    return state, config, tuple(polyhedra)


def run_dpdem_simulation(
    state: SimulationState,
    config: IntegrationConfig,
    polyhedra: tuple,
    n_steps: int,
    save_interval: int = 100,
) -> List[SimulationState]:
    """
    Run DPDEM simulation and optionally save intermediate states.
    
    Args:
        state: Initial simulation state
        config: Integration configuration
        polyhedra: Polyhedron geometry templates
        n_steps: Total number of steps
        save_interval: Save state every N steps (0 = don't save intermediates)
    
    Returns:
        List of saved states (at least initial and final)
    """
    saved_states = [state]
    
    if save_interval > 0:
        n_intervals = n_steps // save_interval
        
        for i in range(n_intervals):
            state = run_simulation(state, config, polyhedra, save_interval)
            saved_states.append(state)
            
            print(f"Step {(i+1) * save_interval}/{n_steps}, "
                  f"Time: {float(state.time):.4f} s")
        
        # Run remaining steps
        remaining = n_steps % save_interval
        if remaining > 0:
            state = run_simulation(state, config, polyhedra, remaining)
            saved_states.append(state)
    else:
        state = run_simulation(state, config, polyhedra, n_steps)
        saved_states.append(state)
    
    return saved_states


# Convenience function for quick setup
def create_irspp_simulation(
    ice_thickness: float = 0.5,
    ice_velocity: float = 0.15,
    n_elements: int = 100,
    seed: int = 42,
) -> Tuple[SimulationState, IntegrationConfig, tuple]:
    """
    Create IRSPP bilateral ice pressure simulation (Section 4 of paper).
    
    Args:
        ice_thickness: Ice floe thickness [m]
        ice_velocity: Ice drift velocity [m/s]
        n_elements: Number of ice elements
        seed: Random seed
    
    Returns:
        state, config, polyhedra
    """
    from .config import IRSPP_PARAMS
    
    # Domain size based on IRSPP length
    domain_size = (IRSPP_PARAMS.length * 3, IRSPP_PARAMS.breadth * 3)
    
    state, config, polyhedra = setup_simulation(
        n_ice_elements=n_elements,
        ice_field_size=domain_size,
        ice_thickness=ice_thickness,
        platform_length=IRSPP_PARAMS.length,
        platform_breadth=IRSPP_PARAMS.breadth,
        platform_draft=IRSPP_PARAMS.draft,
        platform_displacement=IRSPP_PARAMS.displacement,
        seed=seed,
    )
    
    # Set ice drift velocity
    config = config._replace(v_water=jnp.array([ice_velocity, 0.0, 0.0]))
    
    return state, config, polyhedra
