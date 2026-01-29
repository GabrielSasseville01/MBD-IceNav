"""
Simulation parameters from Table 1 of Wang et al. (2024).

All parameters use SI units unless otherwise specified.
"""

import jax.numpy as jnp
from typing import NamedTuple


class IceProperties(NamedTuple):
    """Physical properties of sea ice."""
    density: float                  # ρ_i [kg/m³]
    elastic_modulus: float          # E [Pa]
    poisson_ratio: float            # ν [-]
    normal_strength: float          # σ̄ [Pa] - tensile strength
    cohesion: float                 # C [Pa] - cohesive strength
    internal_friction_coef: float   # μ_b [-] - internal friction coefficient
    ice_ice_friction: float         # μ_ii [-]
    ice_hull_friction: float        # μ_ih [-]
    fracture_energy_mode_I: float   # G_c^I [J/m²]
    fracture_energy_mode_II: float  # G_c^II [J/m²]


class FluidProperties(NamedTuple):
    """Physical properties of seawater."""
    density: float                  # ρ_w [kg/m³]
    drag_coefficient_force: float   # C_d^F [-]
    drag_coefficient_moment: float  # C_d^M [-]


class SimulationConfig(NamedTuple):
    """Global simulation configuration."""
    gravity: jnp.ndarray            # g [m/s²] - gravity vector
    dt: float                       # Time step [s]
    element_size: float             # Default element size [m]
    neighbor_update_interval: int   # Steps between neighbor list updates
    max_contacts_per_particle: int  # Maximum contacts for array allocation
    bond_damping_beta: float        # β [-] - viscous damping coefficient


# Default parameters from Table 1
DEFAULT_ICE_PROPERTIES = IceProperties(
    density=920.0,                      # kg/m³
    elastic_modulus=2.4e9,              # 2.4 GPa
    poisson_ratio=0.3,
    normal_strength=0.6e6,              # 0.6 MPa
    cohesion=1.8e6,                      # 1.8 MPa
    internal_friction_coef=0.7,         # Typical value for ice
    ice_ice_friction=0.1,
    ice_hull_friction=0.1,
    fracture_energy_mode_I=12.0,        # J/m²
    fracture_energy_mode_II=12.0,       # J/m²
)

DEFAULT_FLUID_PROPERTIES = FluidProperties(
    density=1035.0,                     # kg/m³
    drag_coefficient_force=0.4,
    drag_coefficient_moment=0.4,
)

DEFAULT_SIMULATION_CONFIG = SimulationConfig(
    gravity=jnp.array([0.0, 0.0, -9.8]),  # m/s²
    dt=1e-4,                              # 0.1 ms timestep (typical for DEM)
    element_size=2.5,                     # m (from Section 3.3)
    neighbor_update_interval=10,
    max_contacts_per_particle=32,
    bond_damping_beta=0.1,
)


# Derived parameters (computed from base properties)
def compute_effective_modulus(E: float, nu: float) -> float:
    """
    Compute effective elastic modulus E* for Hertz contact.
    
    E* = E / (2 * (1 - ν²))
    
    Args:
        E: Elastic modulus [Pa]
        nu: Poisson's ratio [-]
    
    Returns:
        Effective modulus E* [Pa]
    """
    return E / (2.0 * (1.0 - nu ** 2))


def compute_effective_radius(R_i: float, R_j: float) -> float:
    """
    Compute effective radius R* for Hertz contact.
    
    R* = (R_i * R_j) / (R_i + R_j)
    
    Args:
        R_i: Radius of particle i [m]
        R_j: Radius of particle j [m]
    
    Returns:
        Effective radius R* [m]
    """
    return (R_i * R_j) / (R_i + R_j)


def compute_normal_stiffness(E: float, nu: float, R_i: float, R_j: float) -> float:
    """
    Compute normal contact stiffness k_n for Hertz contact.
    
    k_n = (4/3) * E* * sqrt(R*)
    
    Args:
        E: Elastic modulus [Pa]
        nu: Poisson's ratio [-]
        R_i: Equivalent radius of particle i [m]
        R_j: Equivalent radius of particle j [m]
    
    Returns:
        Normal stiffness k_n [N/m^(3/2)]
    """
    E_star = compute_effective_modulus(E, nu)
    R_star = compute_effective_radius(R_i, R_j)
    return (4.0 / 3.0) * E_star * jnp.sqrt(R_star)


def compute_max_tangential_overlap(mu: float, delta_n: float, nu: float) -> float:
    """
    Compute maximum tangential overlap δ_t^max.
    
    δ_t^max = μ * δ_n * (2 - ν) / (2 - 2ν)
    
    Args:
        mu: Friction coefficient [-]
        delta_n: Normal overlap [m]
        nu: Poisson's ratio [-]
    
    Returns:
        Maximum tangential overlap [m]
    """
    return mu * delta_n * (2.0 - nu) / (2.0 - 2.0 * nu)


def compute_equivalent_mass(m_A: float, m_B: float) -> float:
    """
    Compute equivalent mass m_AB for contact.
    
    1/m_AB = 1/m_A + 1/m_B
    
    Args:
        m_A: Mass of particle A [kg]
        m_B: Mass of particle B [kg]
    
    Returns:
        Equivalent mass m_AB [kg]
    """
    return (m_A * m_B) / (m_A + m_B)


# Fracture mechanics constants
ETA_BK = 1.75  # Benzeggagh-Kenane exponent for brittle materials


# Ship/Platform parameters from Table 2
class VesselParameters(NamedTuple):
    """Parameters for vessel/platform structures."""
    length: float       # L [m]
    breadth: float      # B [m]
    draft: float        # T_d [m]
    displacement: float # Δ [kg]
    stem_angle: float   # φ [rad] (optional, for icebreakers)
    waterline_angle: float  # α [rad] (optional, for icebreakers)


# R/V Xuelong parameters (for validation)
XUELONG_PARAMS = VesselParameters(
    length=147.2,
    breadth=22.6,
    draft=9.0,
    displacement=21000e3,  # 21000 tonnes in kg
    stem_angle=jnp.deg2rad(24.0),
    waterline_angle=jnp.deg2rad(20.0),
)

# IRSPP parameters
IRSPP_PARAMS = VesselParameters(
    length=85.0,
    breadth=22.5,
    draft=8.5,
    displacement=10000e3,  # 10000 tonnes in kg
    stem_angle=0.0,  # Not applicable
    waterline_angle=0.0,  # Not applicable
)
