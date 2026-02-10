"""
Centralized parameters for MEDEA ice–ship simulation (Section 5.4).

Tune simulation, run, and visualization from here. Use get_default_simulation_config()
and get_default_run_config() for defaults; pass to simulate_ice_ship_interaction and
to the main script as needed.
"""

from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Ice material (used by IceProperties when config is applied)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IceMaterialConfig:
    """Ice and water material constants (for future use with IceProperties)."""
    # kg/m³ — density of sea ice used for mass and buoyancy.
    ice_density: float = 920.0
    # kg/m³ — seawater density for buoyancy and displacement.
    water_density: float = 1025.0
    # Pa — reference flexural strength at thickness_ref; scaled by sqrt(h/h_ref) for breaking force.
    flexural_strength_base: float = 0.3e6
    # Pa — Young's modulus used for bond stiffness and ice mechanics.
    youngs_modulus: float = 5e9
    # m — reference thickness for flexural strength scaling.
    thickness_ref: float = 1.0
    # Dimensionless coefficient in impact force model (F ∝ C × mass^α × v^β × h^γ).
    impact_C_force: float = 5.0
    # Exponent for ice element mass in impact force expression.
    impact_mass_exp: float = 0.7
    # Exponent for relative velocity in impact force expression.
    impact_velocity_exp: float = 0.8
    # Exponent for ice thickness in impact force expression.
    impact_thickness_exp: float = 0.5
    # Block coefficient: displacement = L×B×D × water_density × ship_block_coef when not given.
    ship_block_coef: float = 0.6


# ---------------------------------------------------------------------------
# Simulation (collision, bonds, submergence, rubble, ice-ice, stress, ship PD)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationConfig:
    """All tunable simulation parameters."""
    # --- Time ---
    # s — fixed time step for integration.
    dt: float = 0.05

    # --- Collision (ice–ship) ---
    # m — reference penetration depth at which normal force reaches F_break (force = min(pen/pen_crit, 1) × F_break).
    pen_crit_ref: float = 0.5
    # Number of sample points along each ice edge for collision detection (reduces missed contacts).
    n_edge_samples: int = 4
    # Contact width for breaking force: w_contact = min(this × sqrt(area), w_contact_cap).
    w_contact_radius_factor: float = 2.0
    # m — maximum effective contact width used in breaking force.
    w_contact_cap: float = 4.0
    # Fraction of normal collision force applied as tangential “flow around hull” force on ice.
    side_push_factor: float = 0.2
    # Scale factor for angular impulse imparted to ice elements by ship contact.
    angular_impulse_scale: float = 0.01
    # m — penetration cap used when computing angular impulse.
    angular_impulse_pen_cap: float = 1.0

    # --- Submergence / buoyancy (3.1) ---
    # m/s — effective submergence increase per metre of penetration while under the ship.
    submergence_rate: float = 0.4
    # m — cap on penetration used to compute submergence (limits sink depth).
    submergence_pen_cap: float = 0.5
    # s — time constant for submergence decay when ice is no longer in contact with the ship.
    rise_time_const: float = 0.5

    # --- Bow rubble (2.4) ---
    # Fraction of bow contact force applied as extra resistance (surge opposition) on the ship.
    rubble_coef: float = 0.25
    # N — cap on bow rubble resistance force.
    rubble_cap_n: float = 5e6

    # --- Bonds ---
    # m — maximum distance between cell centroids for a bond to be created.
    max_bond_width: float = 4.0
    # Bond stiffness = youngs_modulus × avg_thickness × this factor.
    bond_stiffness_factor: float = 0.001

    # --- Ice–ice contact ---
    # Coefficient of friction μ between ice elements.
    ice_ice_friction: float = 0.1
    # Coefficient for Hertz-like normal repulsion force between overlapping ice elements.
    hertz_factor: float = 2e5
    # Exponent in normal repulsion (force ∝ overlap^hertz_exponent).
    hertz_exponent: float = 1.5
    # N — cap on tangential (friction) force between ice elements.
    friction_cap: float = 1e5

    # --- Drift (ship stationary) ---
    # Scale factor for ice drift velocity when ship is not moving (e.g. wind/current proxy).
    drift_factor: float = 8.0

    # --- Water drag on ice ---
    # Quadratic drag coefficient (F_drag ∝ c_d × area × v² at reference thickness).
    c_d_drag: float = 0.4
    # m — reference thickness for drag force scaling.
    drag_ref_thickness: float = 1.0
    # Linear drag term at low speed for numerical stability.
    linear_drag_factor: float = 0.1

    # --- Ice velocity clamp ---
    # (v_min, v_max) in m/s — ice element velocity clamped to this range (x and y).
    velocity_clip: Tuple[float, float] = (-20.0, 20.0)

    # --- Stress propagation and fracture ---
    # Fraction of stress transferred to neighbors when a bond breaks.
    stress_transfer: float = 0.3
    # Number of diffusion steps for stress propagation after breakage.
    stress_diffusion_steps: int = 2
    # Weight of propagated stress in effective strength calculation.
    propagated_stress_factor: float = 0.2
    # Multiplier of base strength when bond has recent ship collision (reduces strength).
    effective_strength_collision: float = 0.1
    # Stress threshold (× base) above which effective_strength_stress is used.
    effective_strength_stress_threshold: float = 0.5
    # Multiplier of base strength when propagated stress is above threshold.
    effective_strength_stress: float = 0.6

    # --- Angular motion (ice elements) ---
    # Small random angular velocity added per step for numerical diversity.
    angular_noise: float = 0.001
    # Per-step damping factor for ice element angular velocity.
    angular_damping: float = 0.85
    # (ω_min, ω_max) in rad/s — clamp on ice element angular velocity.
    angular_cap: Tuple[float, float] = (-0.2, 0.2)

    # --- Ship dynamics (body-frame PD) ---
    # Linear damping on surge/sway: force = -this × mass × velocity (xy).
    ship_damping_xy: float = 0.02
    # Linear damping on yaw: torque = -this × Izz × angular_velocity.
    ship_damping_psi: float = 0.15
    # Proportional gain for position error (xy): Kp = this × ship mass.
    Kp_xy_scale: float = 0.03
    # Proportional gain for heading error (yaw) in PD controller.
    Kp_psi: float = 5e9
    # Derivative gain in ship PD controller (velocity error).
    Kd_ship: float = 1e6
    # (|surge|_max, |sway|_max, |yaw_rate|_max) in m/s and rad/s — clamp on ship body velocity.
    ship_velocity_clip: Tuple[float, float, float] = (15.0, 8.0, 0.15)


# ---------------------------------------------------------------------------
# Run / script (domain, ship size, tessellation, thickness, frames)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    """Parameters for run_medea_ship_fracture.py: domain, ship, tessellation, thickness, frames."""
    # m/px — scale from MEDEA pixel coordinates to world metres.
    meters_per_pixel: float = 2.5
    # m — ship length overall (e.g. MEDEA vessel).
    ship_length: float = 158.0
    # m — ship beam (width).
    ship_beam: float = 28.0
    # m — ship draft (depth below waterline).
    ship_draft: float = 10.5
    # kg — ship mass (e.g. 23 kt); used for dynamics when not inferred from hull.
    ship_displacement: float = 23000e3
    # s — time step passed to simulation (should match SimulationConfig.dt if changed).
    dt: float = 0.05
    # Number of simulation frames in simple (synthetic) ice field mode.
    n_frames_simple: int = 600
    # Number of simulation frames in MEDEA (satellite) ice field mode.
    n_frames_medea: int = 600
    # Target Voronoi cells per 100 m² in simple mode (controls floe tessellation density).
    cells_per_100m2_simple: float = 0.25
    # Target Voronoi cells per 100 m² in MEDEA mode (denser for larger floes).
    cells_per_100m2_medea: float = 0.8
    # Bond created if centroid distance < (r_i + r_j) × this factor (r = equivalent radius).
    bond_threshold_factor: float = 1.5
    # m — minimum ice thickness in area-based thickness assignment.
    thickness_min: float = 0.8
    # m — maximum ice thickness in area-based thickness assignment.
    thickness_max: float = 2.5
    # Exponent in thickness–area relation: thickness ∝ area^power, then clamped to min/max.
    thickness_power: float = 0.25
    # m — standard deviation of optional noise added to thickness (if used).
    thickness_noise_std: float = 0.1
    # m — half-width of band in y around map centre; floes outside are excluded (MEDEA path filter).
    y_band: float = 80.0
    # m — half-width of band in y for filtering cells in ship path (MEDEA).
    cell_y_band: float = 60.0
    # m — margin from domain edges for ship start/end in full-map diagonal run.
    domain_margin: float = 50.0
    # m — extra distance beyond domain for ship path start/end in simple mode.
    path_margin: float = 20.0
    # Random seed for reproducibility (tessellation, thickness, etc.) when --seed not set.
    default_seed: int = 42
    # Frames per second for 2D GIF animation.
    fps_2d: int = 20
    # Frames per second for 3D GIF animation.
    fps_3d: int = 15
    # Render every Nth frame in 3D to reduce file size (1 = all frames).
    frame_skip_3d: int = 2


# ---------------------------------------------------------------------------
# Visualization (colors, thresholds, animation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VisualizationConfig:
    """Parameters for 2D/3D animation (3D uses these; 2D uses RunConfig for fps)."""
    # m — ice with centre z below -this is drawn as submerged (blue); above as surfaced (light).
    z_surface_threshold: float = 0.05
    # Hex colour for top face of ice elements at or above surface in 3D.
    surfaced_face: str = '#D8ECF8'
    # Hex colour for side faces of ice elements at or above surface in 3D.
    surfaced_side: str = '#C0E0F0'
    # Hex colour for top face of submerged ice in 3D.
    submerged_face: str = '#2E6A9E'
    # Hex colour for side faces of submerged ice in 3D.
    submerged_side: str = '#4A8AC4'
    # Frames per second for 2D animation (duplicated here for standalone viz use).
    fps_2d: int = 20
    # Frames per second for 3D animation.
    fps_3d: int = 15
    # Render every Nth frame in 3D (1 = all frames).
    frame_skip_3d: int = 2
    # 3D view elevation angle in degrees (vertical).
    view_elev: float = 30.0
    # 3D view azimuth start angle in degrees (horizontal).
    view_azim_start: float = -60.0
    # 3D view azimuth rotation per frame (degrees) for rotating camera.
    view_azim_per_frame: float = 0.15


# ---------------------------------------------------------------------------
# Default instances
# ---------------------------------------------------------------------------

def get_default_ice_material_config() -> IceMaterialConfig:
    return IceMaterialConfig()


def get_default_simulation_config() -> SimulationConfig:
    return SimulationConfig()


def get_default_run_config() -> RunConfig:
    return RunConfig()


def get_default_visualization_config() -> VisualizationConfig:
    return VisualizationConfig()
