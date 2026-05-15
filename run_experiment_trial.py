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
    eb-mbd      EB-MBD with emerging-barrier constraint (eb_c_max=2000,
                eb_mu=50000, eb_kappa=1.0, goal_x_weight=1.0, length_weight=1.0)
    mppi        Model Predictive Path Integral
    rss         Random Single Shooting
    ilqr        Box-iLQR

All planners share the same KE-based costmap and replan every 30 s.
Max simulated time: 1800 s.  Seeds: 0-4.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

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
PLANNERS = ["astar", "mbd", "eb-mbd", "mppi", "rss", "ilqr"]
SEEDS    = [0, 1, 2, 3, 4]

DT            = 0.05      # s — physics timestep (matches RunConfig default)
MAX_SIM_TIME  = 1800.0    # s
REPLAN_EVERY  = 30.0      # s
GOAL_TOL      = 15.0      # m — how close to goal_y_line counts as success

# Costmap margin for the synthetic 300×700 m map
COSTMAP_MARGIN = 30.0     # m  (default 200 m is designed for 1400×1400 m maps)

# EB-MBD barrier parameters (from the reference command)
EB_C_MAX         = 2000.0
EB_MU            = 50000.0
EB_KAPPA         = 1.0
EB_GOAL_X_WEIGHT = 1.0
EB_LENGTH_WEIGHT = 1.0


# ── Config builder ─────────────────────────────────────────────────────────

def _make_plan_cfg(planner: str):
    """Build a RunConfig with experiment-standard overrides."""
    cfg = get_default_run_config()

    # Horizon = 800 steps for all trajectory optimisers
    cfg = replace(cfg,
        planning_jax_mbd_horizon=800,
        planning_mppi_horizon=800,
        planning_rss_horizon=800,
        planning_ilqr_horizon=800,
        # All optimisers: cold start (no A* warm-start)
        planning_jax_mbd_warm_start=False,
        planning_mppi_warm_start=False,
        planning_rss_warm_start=False,
        planning_ilqr_warm_start=False,
        # RSS: override to N=2048
        planning_rss_nsample=2048,
        # Replanning uses goal_y_line, not a soft goal weight
        planning_jax_mbd_goal_weight=0.0,
        # Smaller margin for the synthetic map
        planning_margin=COSTMAP_MARGIN,
    )

    if planner == "eb-mbd":
        cfg = replace(cfg,
            planning_jax_mbd_goal_x_weight=EB_GOAL_X_WEIGHT,
            planning_jax_mbd_length_weight=EB_LENGTH_WEIGHT,
        )

    return cfg


# ── Costmap builder (shared across planners) ───────────────────────────────

def _build_cost_map(centroids, thicknesses, radii, plan_cfg):
    cost_map, _ = build_costmap_jax(
        centroids, thicknesses, radii,
        X_LO, X_HI, Y_LO, Y_HI,
        resolution=plan_cfg.planning_resolution,
        margin=plan_cfg.planning_margin,
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


# ── Straight-line seed path ────────────────────────────────────────────────

def _straight_line_path(start_pose):
    """Two-point straight-line path from current pose to goal."""
    x0, y0, psi0 = start_pose
    return np.array([
        [x0,         y0,         psi0],
        [GOAL_POINT[0], GOAL_POINT[1], START_HEADING],
    ], dtype=np.float64)


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

        cost_map = _build_cost_map(centroids, thicknesses, radii, plan_cfg)

        start_pose = (float(ship_pos_f[0]), float(ship_pos_f[1]), float(ship_heading_f))
        init_path  = _straight_line_path(start_pose)

        bounds_f = {
            "x_origin": cost_map.x_origin,
            "y_origin": cost_map.y_origin,
            "width":    cost_map.width,
            "height":   cost_map.height,
        }

        if planner == "astar":
            try:
                lattice = create_medea_lattice_planner(
                    cost_map, bounds_f, start_pose, goal, plan_cfg,
                    goal_y_line=goal_y,
                )
                path = plan_path(lattice, start_pose, goal, plan_cfg,
                                 cost_map=cost_map)
            except Exception:
                path = None
            if path is None:
                planning_times.append(time.perf_counter() - t0)
                return None

        elif planner == "mbd":
            path = optimize_with_diffusion(
                init_path, start_pose, goal, cost_map, plan_cfg,
                start_nu=(2.0, 0.0, 0.0),
                warm_start=False,
                goal_y_line=goal_y,
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

        elif planner == "mppi":
            path = optimize_with_mppi(
                init_path, start_pose, goal, cost_map, plan_cfg,
                start_nu=(2.0, 0.0, 0.0),
                warm_start=False,
            )

        elif planner == "rss":
            path = optimize_with_rss(
                init_path, start_pose, goal, cost_map, plan_cfg,
                start_nu=(2.0, 0.0, 0.0),
                warm_start=False,
            )

        elif planner == "ilqr":
            path = optimize_with_ilqr(
                init_path, start_pose, goal, cost_map, plan_cfg,
                start_nu=(2.0, 0.0, 0.0),
                warm_start=False,
            )

        else:
            raise ValueError(f"Unknown planner: {planner!r}")

        planning_times.append(time.perf_counter() - t0)

        if path is None or len(path) < 2:
            return None

        return path_to_ship_trajectory_arr(path, max(1, n_remaining))

    return _callback


# ── Metrics ────────────────────────────────────────────────────────────────

def _compute_metrics(ship_diagnostics, ship_history, goal_y_line, dt):
    """
    Compute F_max, F_mean, F_95, path_length, success, simulated_time
    from the simulation outputs.
    """
    forces = []
    for d in ship_diagnostics:
        f = np.asarray(d["F_ice_ship"])
        forces.append(float(np.linalg.norm(f)))

    forces = np.array(forces) if forces else np.array([0.0])

    f_max  = float(forces.max())
    f_mean = float(forces.mean())
    f_95   = float(np.percentile(forces, 95))

    # Path length from ship_history (list of (pos_2d, heading))
    positions  = np.array([h[0] for h in ship_history])
    if len(positions) > 1:
        path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    else:
        path_length = 0.0

    final_y    = float(positions[-1, 1]) if len(positions) > 0 else 0.0
    success    = bool(final_y >= goal_y_line - GOAL_TOL)
    sim_time   = len(ship_history) * dt

    return {
        "f_max":        f_max,
        "f_mean":       f_mean,
        "f_95":         f_95,
        "path_length":  path_length,
        "final_y":      final_y,
        "success":      success,
        "simulated_time": sim_time,
    }


# ── Main trial runner ──────────────────────────────────────────────────────

def run_trial(block: str, scenario_id: str, planner: str, seed: int,
              output_dir: Path) -> dict:

    spec         = get_scenario(block, scenario_id)
    plan_cfg     = _make_plan_cfg(planner)
    current_model = make_current_model(spec)

    print(f"\n=== Trial: block={block}  scenario={scenario_id}  "
          f"planner={planner}  seed={seed} ===", flush=True)

    # ── Generate ice field ─────────────────────────────────────────────────
    print("Generating ice field...", flush=True)
    field = make_ice_field(spec, seed)
    sat_polygons    = field["polygons"]
    sat_centroids   = field["centroids"]
    sat_thicknesses = field["thicknesses"]
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
        "block":          block,
        "scenario_id":    scenario_id,
        "concentration":  spec.concentration,
        "thickness_mode": spec.thickness_mode,
        "with_currents":  spec.with_currents,
        "planner":        planner,
        "seed":           seed,
        **metrics,
    }

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
    p.add_argument("--list-scenarios", action="store_true",
                   help="Print all scenario IDs for the given block and exit")
    return p.parse_args()


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
        p.error(f"the following arguments are required for a trial run: {', '.join(missing)}")

    run_trial(
        block       = args.block,
        scenario_id = args.scenario,
        planner     = args.planner,
        seed        = args.seed,
        output_dir  = Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
