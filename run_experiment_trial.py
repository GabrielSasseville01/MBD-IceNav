#!/usr/bin/env python3
"""
Single-trial experiment runner for the ice-navigation benchmark.

Each invocation runs exactly one (block, scenario, planner, seed) trial,
writes a JSON result file, and exits.  Designed to be submitted as one
MILA/SLURM job per trial.

Usage:
    uv run python run_experiment_trial.py \
        --block A \
        --scenario A_conc30_thin \
        --planner mbd \
        --seed 0 \
        --output-dir outputs/experiments

Planners:
    astar       A* lattice planner (standalone, no trajectory optimizer)
    mbd         Model-Based Diffusion (cold start, no A* warm-start)
    eb-mbd      EB-MBD with emerging-barrier constraint (eb_c_max=800,
                eb_mu=50000, eb_kappa=1.0)
    mppi        Model Predictive Path Integral
    rss         Random Single Shooting
    ilqr        Box-iLQR

All planners share the same KE-based costmap and cost function:
  costmap integral (KE collision) + path length + x-arrival at goal line
  + goal-line enforcement (big-M=1e6 for sampling-based; smooth terminal
    cost with goal_weight=100 for iLQR) + OOB wall penalty (A*).
Replan every 30 s.  Max simulated time: 1800 s.  Seeds: 0-4.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

# ── Project imports ────────────────────────────────────────────────────────
from medea_ice_ship.synthetic_map import (
    GOAL_POINT,
    GOAL_Y_LINE,
    SHIP_START,
    START_HEADING,
    X_HI,
    X_LO,
    Y_HI,
    Y_LO,
    get_scenario,
    make_current_model,
    make_ice_field,
    list_scenarios,
)
from medea_ice_ship.config import get_default_run_config
from medea_ice_ship.properties import ShipProperties
from medea_ice_ship.bonds import create_proximity_bonds
from medea_ice_ship.geometry import polygon_area
from medea_ice_ship.tessellation import tessellate_all_floes
from medea_ice_ship.jax_costmap import build_costmap_jax
from medea_ice_ship.costmap_build import build_costmap_from_ice_state
from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan
from medea_ice_ship.planning import (
    create_medea_lattice_planner,
    optimize_with_diffusion,
    optimize_with_ilqr,
    optimize_with_mppi,
    optimize_with_rss,
    path_to_ship_trajectory_arr,
    plan_path,
)

# ── Constants ──────────────────────────────────────────────────────────────
PLANNERS = ["astar", "mbd", "eb-mbd", "eb-mbd-ice", "mppi", "rss", "ilqr"]
SEEDS    = [0, 1, 2, 3, 4]

DT            = 0.05      # s — physics timestep (matches RunConfig default)
MAX_SIM_TIME  = 1800.0    # s
REPLAN_EVERY  = 30.0      # s
GOAL_TOL      = 15.0      # m — how close to goal_y_line counts as success
DEFAULT_HORIZON = 200     # action steps shared by all trajopt planners

# Costmap margin for the synthetic 300×700 m map
COSTMAP_MARGIN = 30.0     # m  (default 200 m is designed for 1400×1400 m maps)

# EB-MBD barrier parameters (from the reference command)
EB_C_MAX         = 600.0
EB_MU            = 50000.0
EB_KAPPA         = 1.0
EB_GOAL_X_WEIGHT = 1.0
EB_LENGTH_WEIGHT = 1.0


# ── Config builder ─────────────────────────────────────────────────────────

def _make_plan_cfg(planner: str, horizon: int = DEFAULT_HORIZON,
                   overrides: Optional[dict] = None):
    """Build a RunConfig with experiment-standard overrides.

    Extra per-trial overrides (e.g. cost weights for a tuning grid point)
    are applied LAST, after the experiment defaults, so they win.
    """
    cfg = get_default_run_config()

    # AUTO-IceNav PSV ship (FossenSupply3DOF defaults: 76.2 m, 6 000 t)
    cfg = replace(cfg,
        ship_length=76.2,
        ship_beam=18.0,
        ship_draft=5.5,
        ship_displacement=6000e3,
    )

    # Shared horizon for all trajectory optimisers (A* ignores this)
    cfg = replace(cfg,
        planning_jax_mbd_horizon=horizon,
        planning_mppi_horizon=horizon,
        planning_rss_horizon=horizon,
        planning_ilqr_horizon=horizon,
        # All optimisers: cold start (no A* warm-start)
        planning_jax_mbd_warm_start=False,
        planning_mppi_warm_start=False,
        planning_rss_warm_start=False,
        planning_ilqr_warm_start=False,
        # RSS: override to N=2048
        planning_rss_nsample=2048,
        # Replanning uses goal_y_line, not a soft goal weight (overridden for iLQR below)
        planning_jax_mbd_goal_weight=0.0,
        # Shared cost weights: path length + x-arrival at goal line (same as EB-MBD)
        planning_jax_mbd_goal_x_weight=EB_GOAL_X_WEIGHT,
        planning_jax_mbd_length_weight=EB_LENGTH_WEIGHT,
        # Smaller margin for the synthetic map
        planning_margin=COSTMAP_MARGIN,
        # A* must plan paths that cross the goal line, not merely approach it
        planning_goal_tolerance=0.0,
        # Fine-grained A* for the 300×750 m synthetic map
        planning_resolution=1.0,
        planning_num_headings=32,
        planning_turning_radius=50.0,
        planning_max_iterations=500_000,
    )

    if planner == "ilqr":
        # iLQR uses a smooth differentiable goal-line terminal cost instead of
        # the non-differentiable big-M, so a large goal_weight is needed to provide
        # gradient toward the goal line throughout the backward pass.
        cfg = replace(cfg, planning_jax_mbd_goal_weight=100.0)

    if overrides:
        unknown = [k for k in overrides if not hasattr(cfg, k)]
        if unknown:
            raise ValueError(f"Unknown RunConfig fields in overrides: {unknown}")
        cfg = replace(cfg, **overrides)

    return cfg


# ── Costmap builder (shared across planners) ───────────────────────────────

def _build_cost_map(centroids, thicknesses, radii, plan_cfg):
    cost_map, _ = build_costmap_jax(
        centroids, thicknesses, radii,
        X_LO, X_HI, Y_LO, Y_HI,
        resolution=plan_cfg.planning_resolution,
        margin=plan_cfg.planning_margin,
        base_cost=0.0,
        collision_cost=plan_cfg.planning_ice_collision_cost,
        obs_cost_buffer_factor=plan_cfg.planning_obs_cost_buffer_factor,
        ship_length=plan_cfg.ship_length,
        ship_beam=plan_cfg.ship_beam,
        planning_clearance_margin=plan_cfg.planning_clearance_margin,
        ship_speed=plan_cfg.planning_ship_speed,
        ship_mass=float(plan_cfg.ship_displacement),
        ke_cost_scale=plan_cfg.planning_ke_cost_scale,
        ice_resistance_weight=plan_cfg.planning_ice_resistance_weight,
    )
    return cost_map


def _build_cost_map_astar(world_polygons, centroids, thicknesses, plan_cfg):
    """Build a dpdem CostMap (has world_to_grid) for the A* lattice planner."""
    cost_map, bounds = build_costmap_from_ice_state(
        world_polygons, centroids, thicknesses,
        X_LO, X_HI, Y_LO, Y_HI,
        resolution=plan_cfg.planning_resolution,
        margin=plan_cfg.planning_margin,
        base_cost=0.0,
        ship_mass=float(plan_cfg.ship_displacement),
        collision_cost=plan_cfg.planning_ice_collision_cost * 100.0,
        ship_length=plan_cfg.ship_length,
        ship_beam=plan_cfg.ship_beam,
        planning_clearance_margin=plan_cfg.planning_clearance_margin,
        ship_speed=plan_cfg.planning_ship_speed,
        ke_cost_scale=plan_cfg.planning_ke_cost_scale,
        obs_cost_buffer_factor=plan_cfg.planning_obs_cost_buffer_factor,
        ice_resistance_weight=plan_cfg.planning_ice_resistance_weight,
        wall_bounds=(X_LO, X_HI, Y_LO, Y_HI),
    )
    return cost_map, bounds


# ── Straight-line seed path ────────────────────────────────────────────────

def _straight_line_path(start_pose):
    """Two-point straight-line path from current pose to goal."""
    x0, y0, psi0 = start_pose
    return np.array([
        [x0,         y0,         psi0],
        [GOAL_POINT[0], GOAL_POINT[1], START_HEADING],
    ], dtype=np.float64)


# ── Reject-sampling filter (standard MBD) ─────────────────────────────────

def _make_reject_sampling_filter(plan_cfg):
    """
    Build a sample_filter_fn for standard MBD: at each denoising step, drop
    samples that are OOB (any hull corner outside [X_LO, X_HI] in x or ship
    centre below Y_LO in y) OR whose last waypoint does not cross the goal
    line. No replacement — invalid samples are simply discarded.

    Mirrors examples/toy_mbd_bimodal.py:1987-2017, parameterised with the
    synthetic-map bounds.
    """
    _x_lo = float(X_LO)
    _x_hi = float(X_HI)
    _y_lo = float(Y_LO)
    _goal = float(GOAL_Y_LINE)
    _half_l = plan_cfg.ship_length / 2.0
    _half_w = plan_cfg.ship_beam / 2.0
    _dx = np.array([0.0,  _half_l,  _half_l, -_half_l, -_half_l], dtype=np.float32)
    _dy = np.array([0.0,  _half_w, -_half_w,  _half_w, -_half_w], dtype=np.float32)

    def sample_filter_fn(Y0s_norm: np.ndarray, trajs: np.ndarray) -> np.ndarray:
        xs  = trajs[:, :, 0]
        ys  = trajs[:, :, 1]
        psi = trajs[:, :, 2]
        c = np.cos(psi)
        s = np.sin(psi)
        hull_x = (xs[:, :, None]
                  + c[:, :, None] * _dx[None, None, :]
                  - s[:, :, None] * _dy[None, None, :])
        hull_oob = ((hull_x < _x_lo) | (hull_x > _x_hi)).any(axis=(1, 2))
        y_oob = (ys < _y_lo).any(axis=1)
        oob = hull_oob | y_oob
        reaches_goal = ys[:, -1] >= _goal
        return (~oob) & reaches_goal

    return sample_filter_fn


# ── MPC truncation: only consume one replan window of any plan ─────────────

def _truncate_path_to_window(path: np.ndarray, n_remaining: int,
                              ship_speed: float, dt: float) -> np.ndarray:
    """
    Truncate *path* to at most ``n_remaining * ship_speed * dt`` meters of arc
    length, interpolating the final segment so the cut is exact.

    Trajectory optimisers (MBD/MPPI/RSS/iLQR) plan over a long horizon (e.g.
    300 s at H=300), but the simulator only consumes the first ``replan_every``
    seconds of any plan before replanning. Without this truncation, the
    replan callback hands ``n_remaining`` frames the *entire* horizon path,
    making path_to_ship_trajectory_arr command an artificially high speed.
    """
    if path is None or len(path) < 2:
        return path
    max_dist = float(n_remaining) * float(ship_speed) * float(dt)
    seg_lens = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    if cum[-1] <= max_dist:
        return path
    cut = int(np.searchsorted(cum, max_dist, side="right"))
    cut = max(1, min(cut, len(path) - 1))
    over = cum[cut] - max_dist
    seg = float(seg_lens[cut - 1])
    frac = 1.0 - (over / seg) if seg > 1e-9 else 1.0
    frac = float(np.clip(frac, 0.0, 1.0))
    p0 = path[cut - 1]
    p1 = path[cut]
    # Interpolate heading on the shorter arc
    dpsi = (p1[2] - p0[2] + np.pi) % (2.0 * np.pi) - np.pi
    interp = np.array([
        p0[0] + frac * (p1[0] - p0[0]),
        p0[1] + frac * (p1[1] - p0[1]),
        (p0[2] + frac * dpsi + np.pi) % (2.0 * np.pi) - np.pi,
    ], dtype=path.dtype)
    return np.vstack([path[:cut], interp])


# ── Replan callback factory ────────────────────────────────────────────────

def _make_replan_callback(planner: str, plan_cfg, radii, planning_times: list):
    """
    Return a replan callback compatible with simulate_jax_chunked_replan.

    Signature: (frame, world_polygons, centroids, thicknesses,
                ship_pos_f, ship_heading_f, n_remaining) -> trajectory array | None
    """
    goal   = GOAL_POINT
    goal_y = GOAL_Y_LINE

    def _callback(frame, world_polygons, centroids, thicknesses,
                  ship_pos_f, ship_heading_f, n_remaining):

        t0 = time.perf_counter()

        start_pose = (float(ship_pos_f[0]), float(ship_pos_f[1]), float(ship_heading_f))
        init_path  = _straight_line_path(start_pose)

        if planner == "astar":
            # A* needs a dpdem CostMap (has world_to_grid); JaxCostMapResult does not.
            try:
                astar_map, astar_bounds = _build_cost_map_astar(
                    world_polygons, centroids, thicknesses, plan_cfg)
                lattice = create_medea_lattice_planner(
                    astar_map, astar_bounds, start_pose, goal, plan_cfg,
                    goal_y_line=goal_y,
                )
                path = plan_path(lattice, start_pose, goal, plan_cfg,
                                 cost_map=astar_map)
            except Exception:
                path = None

        else:
            # JAX-based planners use JaxCostMapResult.
            cost_map = _build_cost_map(centroids, thicknesses, radii, plan_cfg)

            if planner == "mbd":
                # Standard MBD: same cost terms as EB-MBD (KE + length + x-goal,
                # no big-M), constraints enforced by reject-sampling instead of
                # log barriers. Drop samples that are OOB or fail to reach the
                # goal line; no replacement.
                path = optimize_with_diffusion(
                    init_path, start_pose, goal, cost_map, plan_cfg,
                    start_nu=(2.0, 0.0, 0.0),
                    warm_start=False,
                    goal_y_line=goal_y,
                    goal_y_line_penalty=0.0,
                    oob_penalty=0.0,
                    sample_filter_fn=_make_reject_sampling_filter(plan_cfg),
                )

            elif planner == "eb-mbd":
                path = optimize_with_diffusion(
                    init_path, start_pose, goal, cost_map, plan_cfg,
                    start_nu=(2.0, 0.0, 0.0),
                    warm_start=False,
                    goal_y_line=goal_y,
                    use_eb_mbd=True,
                    eb_x_min=float(cost_map.x_origin),
                    eb_x_max=float(cost_map.x_origin + cost_map.width),
                    eb_y_min=float(cost_map.y_origin),
                    eb_c_max=EB_C_MAX,
                    eb_mu=EB_MU,
                    eb_kappa=EB_KAPPA,
                )

            elif planner == "eb-mbd-ice":
                path = optimize_with_diffusion(
                    init_path, start_pose, goal, cost_map, plan_cfg,
                    start_nu=(2.0, 0.0, 0.0),
                    warm_start=False,
                    goal_y_line=goal_y,
                    use_eb_mbd=True,
                    eb_x_min=float(cost_map.x_origin),
                    eb_x_max=float(cost_map.x_origin + cost_map.width),
                    eb_y_min=float(cost_map.y_origin),
                    eb_c_max=EB_C_MAX,
                    eb_mu=EB_MU,
                    eb_kappa=EB_KAPPA,
                    eb_ice_as_constraint=True,
                )

            elif planner == "mppi":
                path = optimize_with_mppi(
                    init_path, start_pose, goal, cost_map, plan_cfg,
                    start_nu=(2.0, 0.0, 0.0),
                    warm_start=False,
                    goal_y_line=goal_y,
                )

            elif planner == "rss":
                path = optimize_with_rss(
                    init_path, start_pose, goal, cost_map, plan_cfg,
                    start_nu=(2.0, 0.0, 0.0),
                    warm_start=False,
                    goal_y_line=goal_y,
                )

            elif planner == "ilqr":
                path = optimize_with_ilqr(
                    init_path, start_pose, goal, cost_map, plan_cfg,
                    start_nu=(2.0, 0.0, 0.0),
                    warm_start=False,
                    goal_y_line=goal_y,
                )

            else:
                raise ValueError(f"Unknown planner: {planner!r}")

        planning_times.append(time.perf_counter() - t0)

        if path is None or len(path) < 2 or not np.all(np.isfinite(path)):
            return None

        # MPC: only execute the next replan-window's worth of the plan at 4 m/s.
        path = _truncate_path_to_window(
            path, n_remaining, plan_cfg.planning_ship_speed, DT)
        path_len = float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
        n_path_frames = max(1, int(path_len / float(plan_cfg.planning_ship_speed) / DT))
        return path_to_ship_trajectory_arr(path, min(n_remaining, n_path_frames))

    return _callback


# ── Metrics ────────────────────────────────────────────────────────────────

def _compute_metrics(ship_diagnostics, ship_history, goal_y_line, dt):
    """
    Compute F_max, F_mean, F_95, path_length, success, simulated_time,
    fractional progress, and full final pose from the simulation outputs.
    """
    forces = []
    for d in ship_diagnostics:
        f = np.asarray(d["F_ice_ship"])
        forces.append(float(np.linalg.norm(f)))

    forces = np.array(forces) if forces else np.array([0.0])

    f_max  = float(forces.max())
    f_mean = float(forces.mean())
    f_95   = float(np.percentile(forces, 95))

    positions  = np.array([h[0] for h in ship_history])
    headings   = np.array([h[1] for h in ship_history]) if ship_history else np.array([])
    if len(positions) > 1:
        path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    else:
        path_length = 0.0

    final_x       = float(positions[-1, 0]) if len(positions) > 0 else float(SHIP_START[0])
    final_y       = float(positions[-1, 1]) if len(positions) > 0 else float(SHIP_START[1])
    final_heading = float(headings[-1])     if len(headings)  > 0 else float(START_HEADING)
    success       = bool(final_y >= goal_y_line - GOAL_TOL)
    # Fractional progress toward goal_y_line, capped at 1.0. Drives the
    # tuning score's primary term when binary success has no gradient.
    fractional_progress = float(min(max(final_y, 0.0), goal_y_line) / goal_y_line)
    sim_time      = len(ship_history) * dt

    return {
        "f_max":               f_max,
        "f_mean":              f_mean,
        "f_95":                f_95,
        "path_length":         path_length,
        "final_x":             final_x,
        "final_y":             final_y,
        "final_heading":       final_heading,
        "fractional_progress": fractional_progress,
        "success":             success,
        "simulated_time":      sim_time,
    }


def _ice_field_stats(field: dict, thick_threshold: float = 1.0) -> dict:
    """Summary statistics of the generated ice field (pre-tessellation)."""
    thicknesses = np.asarray(field["thicknesses"])
    return {
        "actual_concentration": float(field["concentration"]),
        "n_floes":              int(len(field["polygons"])),
        "mean_thickness":       float(thicknesses.mean()) if len(thicknesses) else 0.0,
        "max_thickness":        float(thicknesses.max())  if len(thicknesses) else 0.0,
        "n_thick_floes":        int(np.sum(thicknesses >= thick_threshold)),
    }


# ── Main trial runner ──────────────────────────────────────────────────────

def run_trial(block: str, scenario_id: str, planner: str, seed: int,
              output_dir: Path, horizon: int = DEFAULT_HORIZON,
              overrides: Optional[dict] = None,
              output_subpath: Optional[str] = None) -> dict:
    """
    Parameters
    ----------
    overrides
        Optional dict of `RunConfig` field overrides applied after the
        experiment-standard defaults. Used by the tuning grid driver to
        sweep cost weights without rewriting this file.
    output_subpath
        If given, the JSON is written to
        ``output_dir / output_subpath / {block} / {scenario} / {planner} / seed_{seed}.json``
        instead of the default ``output_dir / {block} / ...``. Used so
        tuning runs can land under ``outputs/tuning/{group}/grid_{idx}/...``.
    """

    spec         = get_scenario(block, scenario_id)
    plan_cfg     = _make_plan_cfg(planner, horizon=horizon, overrides=overrides)
    current_model = make_current_model(spec)

    print(f"\n=== Trial: block={block}  scenario={scenario_id}  "
          f"planner={planner}  seed={seed} ===", flush=True)

    # ── Generate ice field ─────────────────────────────────────────────────
    print("Generating ice field...", flush=True)
    field = make_ice_field(spec, seed)
    sat_polygons    = field["polygons"]
    sat_centroids   = field["centroids"]
    sat_thicknesses = field["thicknesses"]
    ice_stats       = _ice_field_stats(field)
    print(f"  {len(sat_polygons)} floes, actual conc={field['concentration']:.1%}",
          flush=True)

    # ── Tessellate + bonds ─────────────────────────────────────────────────
    print("Tessellating...", flush=True)
    polygons, centroids, parent_ids = tessellate_all_floes(
        sat_polygons, sat_centroids,
        cells_per_100m2=plan_cfg.cells_per_100m2_simple,
        seed=seed,
    )
    thicknesses = sat_thicknesses[parent_ids]
    areas       = np.array([polygon_area(p) for p in polygons])
    radii       = np.sqrt(areas / np.pi)
    bonds       = create_proximity_bonds(
        centroids, radii,
        threshold_factor=plan_cfg.bond_threshold_factor,
    )
    print(f"  {len(polygons)} cells, {len(bonds)} bonds", flush=True)

    # ── Ship properties ────────────────────────────────────────────────────
    ship_props = ShipProperties(
        plan_cfg.ship_length,
        plan_cfg.ship_beam,
        plan_cfg.ship_draft,
        plan_cfg.ship_displacement,
    )

    # ── Simulation parameters ──────────────────────────────────────────────
    n_frames_max      = int(MAX_SIM_TIME / DT)
    replan_n_frames   = max(1, round(REPLAN_EVERY / DT))

    # Initial straight-line trajectory (all planners start cold)
    start_pose  = (float(SHIP_START[0]), float(SHIP_START[1]), START_HEADING)
    init_path   = _straight_line_path(start_pose)
    init_traj   = path_to_ship_trajectory_arr(init_path, n_frames_max)

    # ── Replan callback ────────────────────────────────────────────────────
    planning_times: list = []
    replan_callback = _make_replan_callback(
        planner, plan_cfg, radii, planning_times,
    )

    # ── Run simulation ─────────────────────────────────────────────────────
    print("Running simulation...", flush=True)
    t_sim_start = time.perf_counter()

    result = simulate_jax_chunked_replan(
        polygons, centroids, bonds,
        np.array(SHIP_START), plan_cfg.ship_length, plan_cfg.ship_beam,
        n_frames_max, DT,
        drift_vel=np.array([0.0, 0.0]),
        thicknesses=thicknesses,
        ship_props=ship_props,
        ship_trajectory=init_traj,
        return_ship_diagnostics=True,
        replan_every_n_frames=replan_n_frames,
        replan_callback=replan_callback,
        parent_ids=parent_ids,
        ocean_current=current_model,
        goal_y_line=GOAL_Y_LINE,
        goal_tolerance=GOAL_TOL,
        use_auto_icenav_ship=True,
        verbose=True,
    )

    wall_time = time.perf_counter() - t_sim_start
    print(f"  Simulation complete in {wall_time:.1f} s", flush=True)

    # result = (positions_history, rotations_history, bonds_history,
    #           ship_history, submergence_history, ship_diagnostics)
    _, _, _, ship_history, _, ship_diagnostics = result

    # ── Compute metrics ────────────────────────────────────────────────────
    metrics = _compute_metrics(ship_diagnostics, ship_history, GOAL_Y_LINE, DT)
    metrics["n_replans"]      = len(planning_times)
    metrics["planning_times"] = planning_times
    metrics["wall_time_s"]    = wall_time

    print(f"  success={metrics['success']}  "
          f"f_max={metrics['f_max']:.0f} N  "
          f"f_mean={metrics['f_mean']:.0f} N  "
          f"path={metrics['path_length']:.0f} m  "
          f"sim_t={metrics['simulated_time']:.0f} s",
          flush=True)

    # ── Save result ────────────────────────────────────────────────────────
    out = {
        "block":             block,
        "scenario_id":       scenario_id,
        "concentration":     spec.concentration,
        "thickness_mode":    spec.thickness_mode,
        "with_currents":     spec.with_currents,
        "planner":           planner,
        "seed":              seed,
        "horizon":           horizon,
        "overrides_applied": overrides or {},
        **ice_stats,
        **metrics,
    }

    if output_subpath is not None:
        out_dir = output_dir / output_subpath / block / scenario_id / planner
    else:
        out_dir = output_dir / block / scenario_id / planner
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{seed}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Result saved → {out_path}", flush=True)

    return out


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Run a single ice-navigation experiment trial."
    )
    p.add_argument("--block",       required=True, choices=["A", "B"],
                   help="Experiment block (A or B)")
    p.add_argument("--scenario",
                   help="Scenario ID, e.g. A_conc30_thin or B_conc50_bimodal")
    p.add_argument("--planner",     choices=PLANNERS,
                   help="Planner to evaluate")
    p.add_argument("--seed",        type=int, choices=SEEDS,
                   help="Random seed (0–4)")
    p.add_argument("--output-dir",  default="outputs/experiments",
                   help="Root directory for JSON result files")
    p.add_argument("--output-subpath", default=None,
                   help=("Optional nested subpath under --output-dir, e.g. "
                         "'tuning/shared_trajopt/grid_0042'. Used by the "
                         "tuning driver so tuning runs don't collide with "
                         "the main experiments tree."))
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                   help=("Planning horizon (action steps) shared by all trajopt "
                         "planners (MBD/EB-MBD/MPPI/RSS/iLQR). A* ignores it. "
                         f"Default: {DEFAULT_HORIZON}."))
    p.add_argument("--overrides-json", default=None,
                   help=("Optional JSON string of RunConfig field overrides "
                         "applied AFTER the experiment-standard defaults, e.g. "
                         "'{\"planning_jax_mbd_costmap_weight\": 10.0, "
                         "\"planning_jax_mbd_length_weight\": 0.1}'. Unknown "
                         "fields raise. Used by submit_tuning.py."))
    p.add_argument("--overrides-file", default=None,
                   help=("Optional path to a JSON file of RunConfig field "
                         "overrides. Merged with --overrides-json; CLI string "
                         "wins on key collisions."))
    p.add_argument("--list-scenarios", action="store_true",
                   help="Print all scenario IDs for the given block and exit")
    return p.parse_args()


def _load_overrides(args) -> Optional[dict]:
    overrides: dict = {}
    if args.overrides_file:
        path = Path(args.overrides_file)
        if not path.exists():
            raise SystemExit(f"--overrides-file not found: {path}")
        overrides.update(json.loads(path.read_text()))
    if args.overrides_json:
        overrides.update(json.loads(args.overrides_json))
    return overrides if overrides else None


def main():
    args = _parse_args()

    if args.list_scenarios:
        for spec in list_scenarios(args.block):
            print(spec.scenario_id)
        return

    missing = [f"--{f}" for f, v in [("scenario", args.scenario),
                                       ("planner",  args.planner),
                                       ("seed",     args.seed)] if v is None]
    if missing:
        raise SystemExit(
            "the following arguments are required for a trial run: "
            + ", ".join(missing)
        )

    overrides = _load_overrides(args)

    run_trial(
        block          = args.block,
        scenario_id    = args.scenario,
        planner        = args.planner,
        seed           = args.seed,
        output_dir     = Path(args.output_dir),
        horizon        = args.horizon,
        overrides      = overrides,
        output_subpath = args.output_subpath,
    )


if __name__ == "__main__":
    main()
