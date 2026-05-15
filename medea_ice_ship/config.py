"""
Centralized parameters for MEDEA ice–ship simulation (Section 5.4).

Tune simulation, run, and visualization from here. Use get_default_simulation_config()
and get_default_run_config() for defaults; pass to simulate_ice_ship_interaction and
to the main script as needed.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


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
    # s — fixed time step for integration (run script passes RunConfig.dt; this is fallback).
    dt: float = 0.05
    # If True, cap dt at Rayleigh-based stability limit (Eq. 11). Otherwise dt is unchanged (advisory only).
    use_rayleigh_dt_cap: bool = False
    # Safety factor b in Eq. 11 (e.g. 0.1 = 10% of theoretical max).
    rayleigh_safety_factor: float = 0.1

    # --- Collision (ice–ship) ---
    # m — reference penetration depth at which normal force reaches F_break (at pen_crit_thickness_ref).
    pen_crit_ref: float = 0.5
    # m — thickness at which pen_crit = pen_crit_ref; pen_crit scales with thickness so thick ice needs more penetration to reach F_break (avoids fracturing 10 m ice with a few cm).
    pen_crit_thickness_ref: float = 1.0
    # m — ice thicker than this cannot fracture (ship collision or DPDEM); treat as unbreakable / land. Prevents unrealistic fracturing of e.g. 100 m "land-like" ice.
    max_fracture_thickness: float = 15.0
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
    # If True, add bond forces Fb to equations of motion and integrate per-particle (Eq. 7). If False, rigid-body component motion (legacy).
    use_bond_forces_in_dynamics: bool = True
    # If True, use Verlet-style position update: x_new = x + v*dt + 0.5*a*dt^2 (then v_new = v + a*dt). Full velocity Verlet would require a second force evaluation.
    use_verlet_position_update: bool = True

    # --- DPDEM-style bond fracture (hybrid criterion, damage, Mohr–Coulomb) ---
    # If True, use hybrid failure (⟨σ⟩/σ̄)²+(τ/τ̄)²≥1, damage variable D, stiffness softening. Default True; set False for legacy force-vs-strength bonds.
    use_dpdem_style_bonds: bool = True
    # Poisson ratio for bond stiffness split (k_n, k_s). Used when use_dpdem_style_bonds.
    poisson_ratio: float = 0.3
    # Pa — Mohr–Coulomb cohesion C (shear strength intercept).
    bond_cohesion: float = 1.8e6
    # Internal friction coefficient μ_b (shear strength τ̄ = C - μ_b·σ).
    bond_internal_friction: float = 0.7
    # J/m² — Mode I and Mode II critical fracture energies (Benzeggagh–Kenane).
    bond_G_c_I: float = 12.0
    bond_G_c_II: float = 12.0
    # Benzeggagh–Kenane exponent η (1.75 for brittle materials).
    bond_eta_bk: float = 1.75

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

    # --- Ocean currents ---
    # Current mode: "none" (default), "uniform", "learned".
    ocean_current_mode: str = "none"
    # Current speed [m/s] (uniform mode, or peak of synthetic initial field for learned mode).
    ocean_current_speed: float = 0.3
    # Current direction [radians] for uniform mode (0 = +x, pi/2 = +y).
    ocean_current_direction: float = 0.0
    # Path to initial current field (.npy / .pkl / .pth) for learned mode. If None a synthetic divergence-free field is generated.
    ocean_current_field_path: Optional[str] = None
    # Path to trained SeaCurrentRNN weights (learned mode only). Train with: python scripts/train_seanet.py --synthetic
    ocean_current_seanet_weights: Optional[str] = None
    # Spatial grid resolution when generating synthetic initial field.
    ocean_current_grid_resolution: int = 64
    # Evolve the learned field every N simulation frames.
    ocean_current_update_every: int = 1000
    # Multiplicative scale applied to the learned field velocities.
    ocean_current_noise_strength: float = 0.5

    # --- Water drag on ice ---
    # Quadratic drag coefficient (F_drag ∝ c_d × area × v² at reference thickness).
    c_d_drag: float = 0.4
    # m — reference thickness for drag force scaling.
    drag_ref_thickness: float = 1.0
    # Linear drag term at low speed for numerical stability.
    linear_drag_factor: float = 0.1
    # Rotational drag (Eq. 10): Md = 0.5·CMd·ρw·r³eff·ω·|ω|·Asub. Coefficient for moment [-].
    drag_coefficient_moment: float = 0.4

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
    # Use MSS-style 3-DOF dynamics (Fossen: M = M_RB + M_A, D, C(nu)). If False, use simple diagonal M, D.
    use_mss_style_dynamics: bool = True
    # Linear damping on surge/sway: force = -this × mass × velocity (xy).
    ship_damping_xy: float = 0.02
    # Linear damping on yaw: torque = -this × Izz × angular_velocity.
    ship_damping_psi: float = 0.15
    # MSS-style: added mass in surge as fraction of ship mass (typical 0.05–0.2).
    mss_added_mass_surge_frac: float = 0.05
    # MSS-style: added mass in sway as fraction of ship mass (typical 0.3–0.8).
    mss_added_mass_sway_frac: float = 0.5
    # MSS-style: added inertia in yaw as fraction of Izz (typical 0.1–0.3).
    mss_added_mass_yaw_frac: float = 0.2
    # Proportional gain for position error (xy): Kp = this × ship mass.
    Kp_xy_scale: float = 0.03
    # Proportional gain for heading error (yaw) in PD controller.
    Kp_psi: float = 5e9
    # Derivative gain in ship PD controller (velocity error).
    Kd_ship: float = 1e6
    # (|surge|_max, |sway|_max, |yaw_rate|_max) in m/s and rad/s — clamp on ship body velocity.
    ship_velocity_clip: Tuple[float, float, float] = (15.0, 8.0, 0.15)
    # N — cap on magnitude of ice-on-ship force from collision (thrust limit). Default 25 MN is order-of-magnitude for typical icebreaker bollard pull / installed thrust (e.g. 1–50 MN range); set to None to disable (unlimited thrust).
    max_ship_ice_force: Optional[float] = 25e6
    # N — ice brake threshold for setpoint blending.  When the surge ice reaction force exceeds
    # this value the DP setpoint is blended back toward the current ship position, reducing the
    # position error that drives the controller and preventing the ship from over-thrusting
    # against heavy ice.  Formula: brake_factor = clip(threshold / F_surge, 0.1, 1.0),
    # effective_setpoint = ship_pos + brake_factor * (setpoint - ship_pos).
    # Set to 0.0 (default) to disable — existing behaviour is unchanged.
    ice_brake_threshold: float = 0.0
    # If True, cap ice-on-ship impulse so the ship loses kinetic energy (stops) but does not bounce (no elastic pushback beyond full stop).
    collision_impulse_dissipation_cap: bool = True
    # When collision_impulse_dissipation_cap: cap impulse at this fraction of ship momentum along collision direction (1.0 = at most full stop, no reversal).
    collision_impulse_cap_factor: float = 1.0
    # If True, after ship integration zero out any velocity component away from the ice (restitution = 0). Prevents bounce from PD controller or numerics.
    collision_zero_restitution: bool = True

    # --- JAX simulation ---
    # If True, use the vectorized JAX simulation backend (jax_simulation.py) instead of NumPy.
    # Enables GPU execution and JIT compilation. Enable float64 with jax_enable_x64 for validation.
    use_jax_simulation: bool = False
    # If True, enable float64 in JAX (jax.config.update("jax_enable_x64", True)). Slower on GPU
    # but required for numerical validation against NumPy baseline. Auto-enabled when use_jax_simulation.
    jax_enable_x64: bool = True


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
    # s — time step passed to simulation. Finer (e.g. 0.05) gives smoother, more realistic ship dynamics.
    dt: float = 0.05
    # Number of simulation frames in simple (synthetic) ice field mode.
    n_frames_simple: int = 300
    # Number of simulation frames in MEDEA (satellite) ice field mode.
    n_frames_medea: int = 300
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
    # --- Planning (lattice A* or Diffusion MPC) ---
    # Planner type: "lattice" (A* on motion primitives; matches AUTO-IceNav) or "diffusion_mpc" (sampling-based MPC, alternative).
    planning_planner: str = "lattice"
    # m/cell — resolution used for planning grid and motion primitives (coarser than ice mesh).
    planning_resolution: float = 2.0
    # m — extra margin around ice domain in planning costmap (should be large enough to contain ship start/end).
    planning_margin: float = 200.0
    # m — minimum turning radius for motion primitives (on order of a few ship lengths).
    planning_turning_radius: float = 350.0
    # Number of discrete heading bins in lattice (e.g. 16 = 22.5° increments).
    planning_num_headings: int = 16
    # Weight for heuristic in weighted A* (1 = admissible A*, >1 = faster but less optimal).
    planning_heuristic_weight: float = 5.0
    # m — distance to goal within which planner terminates (ship considered “arrived”).
    planning_goal_tolerance: float = 150.0
    # Maximum number of A* node expansions before giving up (prevents runaway search).
    planning_max_iterations: int = 200000
    # If True, use greedy best-first search (f = h) instead of weighted A*.
    # Much faster (DFS-like depth-first toward goal) but paths may be suboptimal.
    planning_greedy: bool = False
    # Receding horizon (AUTO-IceNav style): replan every this many simulated seconds (converted to frames at runtime via dt).
    planning_replan_every_n_seconds: float = 30.0
    # If True, refine A* path with NLP (path length + costmap integral + curvature); requires casadi.
    planning_use_path_optimizer: bool = False
    # Path optimizer: number of waypoints (including start/goal).
    planning_optimizer_waypoints: int = 41
    # Path optimizer: weight for costmap integral in objective.
    planning_optimizer_cost_weight: float = 0.1
    # Path optimizer: weight for curvature penalty (delta_psi)^2.
    planning_optimizer_curvature_weight: float = 10.0
    # Path optimizer: max Ipopt iterations.
    planning_optimizer_max_iter: int = 500
    # KE cost weight (collision_cost_weight in AUTO-IceNav model).
    planning_ice_collision_cost: float = 5.0
    # Ridge costmap (AUTO-IceNav): path to pickle from generate_ridge_costmap.py; None = disabled.
    planning_ridge_costmap_file: Optional[str] = None
    # Ridge cost weight (additive; 0 = disabled). Higher = planner avoids ridges more.
    planning_ridge_cost_weight: float = 0.0
    # Ship speed [m/s] for KE cost.
    planning_ship_speed: float = 4.0
    # Target ship speed [m/s] when planning. If planning_use_target_speed, n_frames is derived so
    # path is traversed at this speed (realistic icebreaker ~4 m/s). Also used in KE collision cost.
    planning_target_speed: float = 4.0
    # Extra clearance [m] added to costmap dilation so planned path stays farther from ice (avoids collision when tracking at 4 m/s).
    planning_clearance_margin: float = 15.0
    # If True and --plan, derive n_frames from path length / target_speed so ship travels at target speed.
    planning_use_target_speed: bool = True
    # If True and --plan, optimize speed per segment: slower in thick ice, faster in open water.
    # Balances impact cost (KE ∝ v²) vs time. Overrides fixed target speed when enabled.
    planning_optimize_speed: bool = False
    # Speed optimization: unit-free scale factor for the auto-calibrated time weight.
    # Effective λ = planning_speed_optimize_time_weight × 2 × base_cost × v_ref, which
    # gives v_opt = v_ref in open water when scale=1.0. >1 biases toward higher speeds; <1 is more conservative.
    planning_speed_optimize_time_weight: float = 1.0
    # Speed optimization: min/max speed [m/s].
    planning_speed_optimize_v_min: float = 1.0
    planning_speed_optimize_v_max: float = 8.0
    # Scale factor for KE cost (raw J -> planning units). With per-cell mass, 1e-9 yields
    # costs 0.1–0.6 (all appear purple). 1e-6 yields ~0.1–100 so ice is clearly visible.
    planning_ke_cost_scale: float = 1e-6
    # Polygon swell factor for collision cost (AUTO-IceNav: 0.1).
    planning_obs_cost_buffer_factor: float = 0.1
    # SIC exponent: cost *= SIC^weight (0 = no SIC; AUTO-IceNav: 1).
    planning_ice_resistance_weight: float = 1.0
    # Diffusion MPC only (sampling-based): number of primitive steps per rollout.
    planning_diffusion_horizon_steps: int = 12
    # Diffusion MPC only: number of candidate rollouts to score (sampling) or samples per diffusion step (model-based).
    planning_diffusion_num_samples: int = 64
    # Diffusion MPC only: weight for goal distance in score (sampling-based).
    planning_diffusion_goal_weight: float = 2.0
    # Diffusion MPC only: random seed (None = no fixed seed).
    planning_diffusion_seed: Optional[int] = None
    # Model-based diffusion (AUTO-IceNav): controller horizon (time steps); trajectory length = horizon+1.
    planning_diffusion_controller_horizon: int = 300
    # Model-based diffusion: number of reverse diffusion steps.
    planning_diffusion_num_diffusion_steps: int = 30
    # Model-based diffusion: softmax temperature for reward weighting.
    planning_diffusion_temperature: float = 0.1
    # Model-based diffusion: noise schedule (beta0, betaT).
    planning_diffusion_beta0: float = 1e-4
    planning_diffusion_betaT: float = 1e-2
    # --- JAX Model-Based Diffusion (used when --planner diffusion-mpc) ---
    # Planning horizon (number of action steps; trajectory length = horizon+1).
    # With dt_action = dt_substep × n_substeps, total trajectory covers H × dt_action × speed.
    # At 4 m/s with dt_action=1s, H=300 covers ~1200 m (enough for a 1400 m map).
    planning_jax_mbd_horizon: int = 700
    # Monte Carlo samples per diffusion step (higher = better coverage of action space).
    planning_jax_mbd_nsample: int = 2048
    # Number of reverse diffusion steps.
    planning_jax_mbd_ndiffuse: int = 50
    # Softmax temperature for importance weighting (lower = greedier).
    planning_jax_mbd_temperature: float = 0.01
    # Noise schedule endpoints.
    planning_jax_mbd_beta0: float = 1e-5
    planning_jax_mbd_betaT: float = 0.05
    # Cost weight: accumulated costmap traversal (hull-aware, segment-length weighted).
    # Mirrors the A* lattice planner's costmap integral.
    planning_jax_mbd_costmap_weight: float = 1.0
    # Cost weight: goal-reaching (small nudge; replaces A*'s heuristic guidance).
    planning_jax_mbd_goal_weight: float = 0.0#0.1
    # Cost weight: total arc length (matches A*'s base distance cost).
    planning_jax_mbd_length_weight: float = 0.0#1.0
    # Cost weight: control smoothness (penalise sudden changes).
    planning_jax_mbd_smoothness_weight: float = 0#20268.0
    # Cost weight: x-deviation at goal-line crossing (soft goal x alignment).
    planning_jax_mbd_goal_x_weight: float = 0.0
    # Normalises the raw costmap-traversal cost before applying costmap_weight.
    # The costmap integral (sum of KE cell values × segment lengths) is divided
    # by this value so that costmap_weight=1.0 gives equal importance to the
    # costmap and path-length terms at the reference ice density.
    # Use calibration/calibrate_from_mbd_debug.py to compute the right value
    # for your scenario (≈ mean_costmap_integral / mean_path_length).
    # Default 1.0 = no normalisation (backward-compatible).
    planning_jax_mbd_costmap_normalizer: float = 1.0#108.2
    # Max surge force [N].
    planning_jax_mbd_surge_max: float = 2e6
    # Max sway force [N].
    planning_jax_mbd_sway_max: float = 5e5
    # Max yaw torque [N·m].
    planning_jax_mbd_yaw_max: float = 5e8
    # Euler sub-step size [s] for dynamics integration within each action step.
    planning_jax_mbd_dt_substep: float = 1.0
    # Number of Euler sub-steps per action step (dt_action = dt_substep × n_substeps).
    # Lower values give finer trajectory resolution comparable to lattice planner.
    planning_jax_mbd_n_substeps: int = 1
    # Initial normalised surge bias (0.7 ≈ strong forward thrust at start of optimisation).
    planning_jax_mbd_init_surge_bias: float = 0.7
    # Warm-start MBD from the lattice A* path (seeds initial actions from path headings).
    # When False, MBD starts from a constant forward-surge bias (pure diffusion, no A* prior).
    planning_jax_mbd_warm_start: bool = True
    # When True, the plan output is the best individual sample action sequence seen across
    # ALL denoising steps (lowest cost_fn(Y0) encountered), rather than the iterated
    # weighted-mean Ybar_final.  Useful for diagnosing mean-aggregation failures in
    # non-convex cost landscapes.
    planning_jax_mbd_use_best_sample: bool = False
    # --- Homotopy-class MBD seeding and classification ---
    # When True, each homotopy class is warm-started from a geometric path that routes
    # through that class's passage sequence (Hernandez et al. reference-frame seeding).
    # Fixes exploration failure for topologically complex classes (e.g. sequential gates).
    planning_jax_mbd_homotopy_per_class_seeds: bool = True
    # When True, use the Hernandez reference-frame classifier (ray cross-product side test)
    # instead of the default mean-x proximity classifier.
    planning_jax_mbd_use_rf_classifier: bool = False
    # --- EB-MBD (Emerging-Barrier MBD): constraint enforcement via log-barrier ---
    # When True, replaces big-M OOB/goal penalties with an emerging log-barrier.
    # See: Mishra & Manchester (2026) "EB-MBD: Emerging-Barrier Model-Based Diffusion".
    planning_jax_mbd_use_eb_mbd: bool = False
    # Barrier offset at the start of reverse diffusion (c_S = c_max, relaxed).
    # Should be ≥ max expected constraint violation in metres.
    # For the toy bimodal scenario: goal miss ~1350 m, so c_max ≥ 1500.
    planning_jax_mbd_eb_c_max: float = 1500.0
    # Barrier hardness in cost units.  Must be on the same scale as J(x).
    # With J(x) ~ 1e4–1e5 and log(c_max) ~ 7, mu ~ 1e4 puts the barrier
    # at ~10–30% of J to guide without dominating.  Tune via the alive-fraction
    # diagnostic: at s=S it should be ≈ 1.0; at s=0 it should be < 0.5.
    planning_jax_mbd_eb_mu: float = 50000.0
    # Barrier schedule shape.  κ=1 → linear; κ>1 → slow then fast tightening.
    planning_jax_mbd_eb_kappa: float = 1.0
    # --- JAX MPPI (used when --planner mppi) ---
    planning_mppi_horizon: int = 300
    # More samples than MBD because MPPI relies on breadth per iteration.
    planning_mppi_nsample: int = 2048
    # More iterations so the nominal sequence converges (noise shrinks effectively
    # because samples cluster around the improving nominal).
    planning_mppi_n_iterations: int = 10
    planning_mppi_temperature: float = 0.1
    planning_mppi_noise_sigma: float = 0.3
    planning_mppi_warm_start: bool = True
    # --- JAX SQP multiple shooting (used when --planner sqp) ---
    planning_sqp_horizon: int = 150
    planning_sqp_max_iters: int = 20
    planning_sqp_reg: float = 1e-3
    planning_sqp_armijo_c: float = 1e-4
    planning_sqp_armijo_beta: float = 0.5
    planning_sqp_merit_mu: float = 100.0
    planning_sqp_tol_grad: float = 1e-4
    planning_sqp_tol_constraint: float = 1e-5
    planning_sqp_warm_start: bool = True
    planning_sqp_dt_substep: float = 1.0
    planning_sqp_n_substeps: int = 5
    # --- JAX Random Single Shooting (used when --planner random-shooting) ---
    planning_rss_horizon: int = 300
    planning_rss_nsample: int = 4096
    planning_rss_warm_start: bool = True
    # --- JAX Box-iLQR (used when --planner ilqr) ---
    planning_ilqr_horizon: int = 150
    planning_ilqr_max_iters: int = 30
    planning_ilqr_reg: float = 1e-4
    planning_ilqr_sigma0: float = 0.1
    planning_ilqr_barrier_tol: float = 1e-4
    planning_ilqr_barrier_reduction: float = 0.1
    planning_ilqr_armijo_c: float = 1e-4
    planning_ilqr_warm_start: bool = True
    # --- JAX GradCEM (used when --planner gradcem) ---
    planning_gradcem_horizon: int = 300
    planning_gradcem_nsample: int = 512
    planning_gradcem_top_samples: int = 64
    planning_gradcem_opt_iters: int = 10
    planning_gradcem_lr: float = 0.1
    planning_gradcem_grad_clip: bool = True
    planning_gradcem_max_grad_norm: float = 1.0
    planning_gradcem_noise_sigma: float = 1.0
    planning_gradcem_warm_start: bool = True
    # Random seed for reproducibility (tessellation, thickness, etc.) when --seed not set.
    default_seed: int = 42
    # Frames per second for 2D animation.
    fps_2d: int = 20
    # Frames per second for 3D animation.
    fps_3d: int = 15
    # Render every Nth frame in 2D (1 = all frames; 2 = half the frames, faster).
    frame_skip_2d: int = 2
    # Render every Nth frame in 3D (2 = 301 frames for 600 sim; 3 = 201 frames, faster).
    frame_skip_3d: int = 3
    # (width, height) inches for 3D figure; height 8.96 gives 896 px at 100 DPI (divisible by 16 for FFmpeg).
    figsize_3d: Tuple[float, float] = (12.0, 8.96)
    # Parallel animation workers (1 = sequential; 4+ = parallel render for MP4). Only used for MP4 output.
    n_workers: int = 4


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
