"""
Main simulation module for DPDEM.

Provides high-level interface for setting up and running simulations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, Optional, Union, Dict
from pathlib import Path

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


def create_ice_field_from_polygons(
    polygons: List[np.ndarray],
    ice_thickness: float,
    ice_properties: IceProperties = DEFAULT_ICE_PROPERTIES,
    dilation_radius: float = 0.1,
    initial_positions: Optional[np.ndarray] = None,
) -> Tuple[List[Polyhedron], IceParticles]:
    """
    Create an ice field from a list of 2D polygon shapes.
    
    This function allows creating ice fields from arbitrary polygon shapes,
    such as those extracted from satellite imagery. Each polygon is extruded
    to create a 3D polyhedron.
    
    Args:
        polygons: List of (n_vertices, 2) arrays defining 2D polygon shapes.
            Vertices should be in meters in a consistent coordinate frame.
        ice_thickness: Ice thickness [m]
        ice_properties: Physical properties of ice
        dilation_radius: Dilation radius for contact [m]
        initial_positions: Optional (n_polygons, 3) array of initial positions.
            If None, positions are computed from polygon centroids at z=0.
    
    Returns:
        polyhedra: List of Polyhedron objects
        particles: IceParticles state
    
    Example:
        >>> from dpdem.geometry.image_floes import process_image_to_polygons
        >>> polygons, metadata = process_image_to_polygons("satellite.png", 2.5)
        >>> polyhedra, particles = create_ice_field_from_polygons(
        ...     polygons, ice_thickness=1.2
        ... )
    """
    n_elements = len(polygons)
    
    if n_elements == 0:
        raise ValueError("No polygons provided")
    
    # Create polyhedra from polygons
    polyhedra = []
    centroids = []
    
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError(f"Polygon must have at least 3 vertices, got {len(polygon)}")
        
        poly = create_polyhedron_from_polygon(
            polygon, ice_thickness, ice_properties.density, dilation_radius
        )
        polyhedra.append(poly)
        
        # Compute centroid for positioning
        centroid = np.mean(polygon, axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # Create particle positions
    if initial_positions is not None:
        positions = np.array(initial_positions)
        if positions.shape != (n_elements, 3):
            raise ValueError(
                f"initial_positions shape {positions.shape} doesn't match "
                f"number of polygons ({n_elements}, 3)"
            )
    else:
        # Position at centroids, at waterline (z=0)
        positions = np.zeros((n_elements, 3))
        positions[:, 0] = centroids[:, 0]
        positions[:, 1] = centroids[:, 1]
        positions[:, 2] = 0.0
    
    # Compute masses
    masses = np.array([poly.volume * ice_properties.density for poly in polyhedra])
    
    # Create particles state
    particles = create_ice_particles(
        n_elements,
        jnp.arange(n_elements),
        jnp.array(positions),
        jnp.array(masses),
    )
    
    return polyhedra, particles


def create_ice_field_from_image(
    image_path: Union[str, Path],
    meters_per_pixel: float,
    ice_thickness: float,
    ice_properties: IceProperties = DEFAULT_ICE_PROPERTIES,
    dilation_radius: float = 0.1,
    crop_bounds: Optional[Tuple[int, int, int, int]] = None,
    min_floe_area: float = 50.0,
    dt_threshold: float = 0.2,
    epsilon_factor: float = 0.005,
    center_domain: bool = True,
) -> Tuple[List[Polyhedron], IceParticles, Dict]:
    """
    Create an ice field directly from a satellite/overhead image.
    
    This function runs the complete image processing pipeline:
    1. Load and optionally crop the image
    2. Segment ice floes using watershed algorithm
    3. Extract polygon contours from segments
    4. Scale to world coordinates
    5. Create 3D polyhedra for each floe
    
    Args:
        image_path: Path to the satellite/overhead image file.
        meters_per_pixel: Scale factor - how many meters each pixel represents.
        ice_thickness: Ice thickness [m]
        ice_properties: Physical properties of ice
        dilation_radius: Dilation radius for contact detection [m]
        crop_bounds: Optional (x, y, width, height) to crop the image.
        min_floe_area: Minimum floe area in square meters to keep.
        dt_threshold: Distance transform threshold for watershed (0-1).
            Higher values require more separation between floes.
        epsilon_factor: Polygon simplification factor. Smaller values
            preserve more detail but create more complex polygons.
        center_domain: If True, centers the ice field at the origin.
    
    Returns:
        polyhedra: List of Polyhedron objects
        particles: IceParticles state
        metadata: Dict containing domain info, number of floes, etc.
    
    Raises:
        ImportError: If OpenCV is not installed.
        FileNotFoundError: If image file doesn't exist.
    
    Example:
        >>> polyhedra, particles, meta = create_ice_field_from_image(
        ...     "satellite.png",
        ...     meters_per_pixel=2.5,
        ...     ice_thickness=1.2,
        ...     min_floe_area=100.0,
        ... )
        >>> print(f"Created {len(polyhedra)} floes over {meta['domain_size']} meters")
    """
    # Import image processing functions (requires OpenCV)
    from .geometry.image_floes import process_image_to_polygons
    
    # Convert min_floe_area from m² to pixels²
    min_area_pixels = min_floe_area / (meters_per_pixel ** 2)
    
    # Process image to polygons
    polygons, metadata = process_image_to_polygons(
        image_path=image_path,
        meters_per_pixel=meters_per_pixel,
        crop_bounds=crop_bounds,
        dt_threshold=dt_threshold,
        min_area_pixels=min_area_pixels,
        epsilon_factor=epsilon_factor,
        center_domain=center_domain,
    )
    
    if len(polygons) == 0:
        raise ValueError(
            f"No ice floes found in image. Try adjusting dt_threshold "
            f"(currently {dt_threshold}) or min_floe_area (currently {min_floe_area} m²)"
        )
    
    # Create polyhedra and particles from polygons
    polyhedra, particles = create_ice_field_from_polygons(
        polygons=polygons,
        ice_thickness=ice_thickness,
        ice_properties=ice_properties,
        dilation_radius=dilation_radius,
    )
    
    # Add ice thickness to metadata
    metadata['ice_thickness'] = ice_thickness
    metadata['total_ice_volume'] = sum(p.volume for p in polyhedra)
    
    return polyhedra, particles, metadata


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


def setup_simulation_from_image(
    image_path: Union[str, Path],
    meters_per_pixel: float,
    ice_thickness: float,
    platform_length: float,
    platform_breadth: float,
    platform_draft: float,
    platform_displacement: float,
    platform_position: Optional[np.ndarray] = None,
    ice_properties: IceProperties = DEFAULT_ICE_PROPERTIES,
    fluid_properties: FluidProperties = DEFAULT_FLUID_PROPERTIES,
    sim_config: SimulationConfig = DEFAULT_SIMULATION_CONFIG,
    crop_bounds: Optional[Tuple[int, int, int, int]] = None,
    min_floe_area: float = 50.0,
    dt_threshold: float = 0.2,
    epsilon_factor: float = 0.005,
    bonded: bool = False,
    adjacency_threshold: Optional[float] = None,
) -> Tuple[SimulationState, IntegrationConfig, tuple, Dict]:
    """
    Set up a complete simulation using ice floes from a satellite image.
    
    This function provides a complete pipeline from satellite imagery to
    a ready-to-run DPDEM simulation:
    1. Extracts ice floe shapes from the image
    2. Creates 3D polyhedra for each floe
    3. Sets up the platform/vessel
    4. Optionally creates bonds between adjacent floes
    5. Initializes all simulation state
    
    Args:
        image_path: Path to the satellite/overhead image file.
        meters_per_pixel: Scale factor - meters per pixel.
        ice_thickness: Ice thickness [m]
        platform_length: Platform length [m]
        platform_breadth: Platform breadth [m]
        platform_draft: Platform draft [m]
        platform_displacement: Platform mass [kg]
        platform_position: Optional (3,) array for platform position.
            If None, placed at domain center.
        ice_properties: Ice physical properties
        fluid_properties: Water physical properties
        sim_config: Simulation configuration
        crop_bounds: Optional (x, y, width, height) to crop the image.
        min_floe_area: Minimum floe area in m² to keep.
        dt_threshold: Distance transform threshold for watershed.
        epsilon_factor: Polygon simplification factor.
        bonded: If True, creates bonds between adjacent floes.
        adjacency_threshold: Distance threshold for bond creation.
            If None, uses 1.5 * average floe equivalent radius.
    
    Returns:
        state: Initial simulation state
        config: Integration configuration
        polyhedra: Polyhedron geometry templates
        metadata: Dict with domain info, floe statistics, etc.
    
    Example:
        >>> state, config, polyhedra, meta = setup_simulation_from_image(
        ...     "satellite.png",
        ...     meters_per_pixel=2.5,
        ...     ice_thickness=1.2,
        ...     platform_length=80.0,
        ...     platform_breadth=20.0,
        ...     platform_draft=6.0,
        ...     platform_displacement=10000.0,
        ...     bonded=False,  # Loose pack ice
        ... )
        >>> print(f"Setup {meta['n_floes']} floes")
    """
    # Create ice field from image
    polyhedra, particles, metadata = create_ice_field_from_image(
        image_path=image_path,
        meters_per_pixel=meters_per_pixel,
        ice_thickness=ice_thickness,
        ice_properties=ice_properties,
        crop_bounds=crop_bounds,
        min_floe_area=min_floe_area,
        dt_threshold=dt_threshold,
        epsilon_factor=epsilon_factor,
        center_domain=True,  # Center for platform placement
    )
    
    # Determine platform position
    domain_size = metadata['domain_size']
    if platform_position is None:
        # Place platform at domain center
        platform_pos = jnp.array([
            0.0,  # Domain is centered
            0.0,
            -platform_draft / 2  # CoG below waterline
        ])
    else:
        platform_pos = jnp.array(platform_position)
    
    # Create platform
    platform = create_platform_from_params(
        platform_length, platform_breadth, platform_draft,
        platform_displacement, platform_pos
    )
    
    # Initialize bonds if requested
    if bonded:
        if adjacency_threshold is None:
            # Use 1.5 * average equivalent radius
            avg_radius = np.mean([p.equivalent_radius for p in polyhedra])
            adjacency_threshold = 1.5 * avg_radius
        
        bonds_data = initialize_bonds(
            np.array(particles.position),
            np.array(particles.quaternion),
            [np.array(p.vertices) for p in polyhedra],
            adjacency_threshold
        )
        bonds = create_bonds(*[jnp.array(x) for x in bonds_data])
        metadata['n_bonds'] = bonds.n_bonds
        metadata['bonded'] = True
    else:
        # Create empty bonds structure
        empty_bonds_data = (
            np.array([], dtype=np.int32),  # particle_i
            np.array([], dtype=np.int32),  # particle_j
            np.array([]),  # areas
            np.array([], dtype=np.int32),  # n_bond_points
            np.zeros((0, 3)),  # local_positions_i
            np.zeros((0, 3)),  # local_positions_j
            np.array([]),  # initial_lengths
        )
        bonds = create_bonds(*[jnp.array(x) for x in empty_bonds_data])
        metadata['n_bonds'] = 0
        metadata['bonded'] = False
    
    # Create contact states
    contacts = create_contact_states(
        particles.n_particles,
        sim_config.max_contacts_per_particle
    )
    
    # Create simulation state
    state = SimulationState(
        particles=particles,
        polyhedra=tuple(polyhedra),
        bonds=bonds,
        contacts=contacts,
        platform=platform,
        neighbors=None,
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
        v_water=jnp.zeros(3),
        waterline_z=0.0,
    )
    
    # Add platform info to metadata
    metadata['platform'] = {
        'length': platform_length,
        'breadth': platform_breadth,
        'draft': platform_draft,
        'displacement': platform_displacement,
        'position': list(platform_pos),
    }
    
    return state, config, tuple(polyhedra), metadata


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
