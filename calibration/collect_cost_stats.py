#!/usr/bin/env python3
"""
Calibration step 1: collect raw MBD cost component statistics.

Runs the diffusion-MPC planner at multiple positions along the full-map
route, using the real MEDEA satellite ice field.  At each probe point the
planner is called with ``return_cost_components=True`` so the four raw
(unweighted) cost terms are recorded before any weight is applied.

The resulting CSV is consumed by ``analyze_cost_stats.py`` which computes
balanced weights such that no single term dominates the objective.

Usage
-----
    uv run python calibration/collect_cost_stats.py
    uv run python calibration/collect_cost_stats.py --n-probes 20 --output calibration/results/my_run.csv

The script intentionally mirrors the scenario parameters used by::

    uv run python run_medea_ship_fracture.py --full-map --plan --no-3d \\
        --no-lattice-search-movie --replan-every 10

so that the collected statistics reflect real replanning conditions.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root so the script can be run from any working directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Scenario constants — must match run_medea_ship_fracture.py --full-map
# ---------------------------------------------------------------------------
SHIP_START = np.array([-545.0, -600.0])
SHIP_END   = np.array([-350.0,  600.0])   # goal_y_line = 600
GOAL_Y_LINE = 600.0
DIAGONAL_HEADING = np.pi / 4.0            # 45 deg

# AUTO-IceNav Fossen supply vessel (matches run_medea_ship_fracture.py defaults)
SHIP_LENGTH      = 76.2    # m
SHIP_BEAM        = 17.0    # m
SHIP_DRAFT       = 7.0     # m
SHIP_MASS        = 6_000e3 # kg

DOMAIN_LIMIT = 700.0       # m  (clamped ice domain half-extent)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect raw MBD cost-component statistics for weight calibration."
    )
    p.add_argument(
        "--n-probes", type=int, default=15,
        help="Number of equally-spaced positions along the route to probe (default: 15).",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path for the output CSV.  Defaults to calibration/results/cost_stats_<timestamp>.csv",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the ice field and JAX MBD sampler (default: 42).",
    )
    p.add_argument(
    p.add_argument(
        "--float32", action="store_true",
        help="Run JAX in float32 (default: float64).",
    )
    return p


# ---------------------------------------------------------------------------
# Ice-data loading
# ---------------------------------------------------------------------------

def _load_ice_field(seed: int):
    """Load the MEDEA satellite ice field (mirrors run_medea_ship_fracture.py --full-map)."""
    from medea_ice_ship.data import load_medea_floes, scale_floes_to_world
    from medea_ice_ship.tessellation import tessellate_all_floes
    from medea_ice_ship.bonds import assign_thickness_by_area
    from medea_ice_ship.geometry import polygon_area
    from medea_ice_ship.config import RunConfig

    cfg = RunConfig()

    pickle_path = _PROJECT_ROOT / "data" / "ice_floes_MEDEA_fram_20100629.pkl"
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"MEDEA data file not found: {pickle_path}\n"
            "Make sure the data directory is present under the project root."
        )

    print("Loading MEDEA satellite ice field...", flush=True)
    t0 = time.time()
    polygons_px, centroids_px, metadata = load_medea_floes(pickle_path)
    print(f"  Loaded {metadata['n_floes']} floes ({time.time()-t0:.1f}s)", flush=True)

    sat_polygons, sat_centroids = scale_floes_to_world(
        polygons_px, centroids_px,
        meters_per_pixel=cfg.meters_per_pixel,
        center_domain=True,
        subsample_region=None,
    )
    print(f"  Scaled to world coordinates ({cfg.meters_per_pixel} m/px)", flush=True)

    t0 = time.time()
    all_cells, all_centroids, all_parent_ids = tessellate_all_floes(
        sat_polygons, sat_centroids,
        cells_per_100m2=cfg.cells_per_100m2_medea,
        seed=seed,
    )
    print(f"  Tessellated to {len(all_cells)} cells ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    sat_areas = np.array([polygon_area(p) for p in sat_polygons])
    sat_thicknesses = assign_thickness_by_area(
        sat_areas,
        min_thickness=cfg.thickness_min,
        max_thickness=cfg.thickness_max,
        power=cfg.thickness_power,
    )
    thicknesses = sat_thicknesses[all_parent_ids]
    print(f"  Assigned thicknesses (min={thicknesses.min():.2f}, max={thicknesses.max():.2f} m) "
          f"({time.time()-t0:.1f}s)", flush=True)

    return all_cells, np.array(all_centroids), thicknesses


# ---------------------------------------------------------------------------
# Costmap construction
# ---------------------------------------------------------------------------

def _build_costmap(polygons, centroids, thicknesses, plan_cfg):
    """Build the initial costmap over the clamped domain."""
    from medea_ice_ship.costmap_build import build_costmap_from_ice_state

    x_min, x_max = -DOMAIN_LIMIT, DOMAIN_LIMIT
    y_min, y_max = -DOMAIN_LIMIT, DOMAIN_LIMIT

    t0 = time.time()
    cost_map, bounds = build_costmap_from_ice_state(
        polygons, centroids, thicknesses,
        x_min, x_max, y_min, y_max,
        resolution=plan_cfg.planning_resolution,
        margin=plan_cfg.planning_margin,
        ship_mass=SHIP_MASS,
        collision_cost=plan_cfg.planning_ice_collision_cost,
        wall_bounds=(x_min, x_max, y_min, y_max),
        ship_length=SHIP_LENGTH,
        ship_beam=SHIP_BEAM,
        planning_clearance_margin=getattr(plan_cfg, "planning_clearance_margin", 15.0),
        ridge_costmap_file=plan_cfg.planning_ridge_costmap_file,
        ridge_cost_weight=plan_cfg.planning_ridge_cost_weight,
        ship_speed=getattr(plan_cfg, "planning_ship_speed", 4.0),
        ke_cost_scale=getattr(plan_cfg, "planning_ke_cost_scale", 1e-6),
        obs_cost_buffer_factor=getattr(plan_cfg, "planning_obs_cost_buffer_factor", 0.1),
        ice_resistance_weight=getattr(plan_cfg, "planning_ice_resistance_weight", 1.0),
    )
    print(f"  Costmap built in {time.time()-t0:.1f}s "
          f"({cost_map.cost_map.shape[0]}×{cost_map.cost_map.shape[1]} cells)", flush=True)
    return cost_map, bounds


# ---------------------------------------------------------------------------
# Probe-point generation
# ---------------------------------------------------------------------------

def _make_probe_points(n_probes: int):
    """
    Return n_probes equally-spaced positions along the nominal route.

    We exclude the first 5 % and last 5 % of the route so every probe has a
    meaningful remaining distance to the goal.  Each probe is a
    (x, y, heading) tuple in world coordinates.
    """
    alphas = np.linspace(0.05, 0.85, n_probes)
    poses = []
    for a in alphas:
        pos = SHIP_START + a * (SHIP_END - SHIP_START)
        poses.append((float(pos[0]), float(pos[1]), DIAGONAL_HEADING))
    return poses


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def run_collection(args) -> Path:
    import jax

    # JAX precision
    if not args.float32:
        jax.config.update("jax_enable_x64", True)
    print("=" * 60)
    print(f"JAX backend: {[str(d) for d in jax.devices()]}")
    print(f"Precision: {'float32' if args.float32 else 'float64'}")
    print("=" * 60)

    np.random.seed(args.seed)

    # Config
    from dataclasses import replace
    from medea_ice_ship.config import RunConfig
    base_cfg = RunConfig()
    plan_cfg = replace(
        base_cfg,
        planning_turning_radius=50.0,
        planning_jax_mbd_warm_start=False,  # cold start (matches CLI default)
    )
    # Load ice and build costmap
    polygons, centroids, thicknesses = _load_ice_field(args.seed)
    print("\nBuilding costmap...", flush=True)
    cost_map, _ = _build_costmap(polygons, centroids, thicknesses, plan_cfg)

    # Probe points along the route
    probe_poses = _make_probe_points(args.n_probes)
    goal = (float(SHIP_END[0]), GOAL_Y_LINE)

    print(f"\nProbing {args.n_probes} positions along route...", flush=True)
    print(f"  Start: {SHIP_START}, Goal: {goal}", flush=True)
    print(f"  Weights: costmap={plan_cfg.planning_jax_mbd_costmap_weight}, "
          f"length={plan_cfg.planning_jax_mbd_length_weight}, "
          f"goal={plan_cfg.planning_jax_mbd_goal_weight}, "
          f"smoothness={plan_cfg.planning_jax_mbd_smoothness_weight}", flush=True)
    print()

    from medea_ice_ship.planning import optimize_with_diffusion

    records = []
    for i, start_pose in enumerate(probe_poses):
        alpha = 0.05 + i * (0.80 / max(args.n_probes - 1, 1))
        dist_to_goal = float(np.hypot(start_pose[0] - goal[0], start_pose[1] - goal[1]))
        print(f"  [{i+1:02d}/{args.n_probes}] pos=({start_pose[0]:.0f}, {start_pose[1]:.0f})  "
              f"dist_to_goal={dist_to_goal:.0f} m", flush=True)

        # Straight-line initial path (cold start)
        initial_path = np.array([
            [start_pose[0], start_pose[1], start_pose[2]],
            [goal[0],       goal[1],       start_pose[2]],
        ])

        t0 = time.time()
        path, comps = optimize_with_diffusion(
            continuous_path=initial_path,
            start_pose=start_pose,
            goal=goal,
            cost_map=cost_map,
            run_cfg=plan_cfg,
            start_nu=(2.0, 0.0, 0.0),
            replan_idx=i,
            return_cost_components=True,
        )
        elapsed = time.time() - t0

        if comps is None:
            print(f"    WARNING: cost components not returned (path too short?)", flush=True)
            continue

        record = {
            "probe_idx":      i,
            "route_alpha":    round(alpha, 4),
            "pos_x":          round(start_pose[0], 2),
            "pos_y":          round(start_pose[1], 2),
            "dist_to_goal_m": round(dist_to_goal, 1),
            "elapsed_s":      round(elapsed, 2),
            # Raw (unweighted) cost components
            "costmap_cost_raw":  round(comps["costmap_cost"], 4),
            "path_length_raw":   round(comps["path_length"], 2),
            "goal_dist_raw":     round(comps["goal_dist"], 2),
            "smoothness_raw":    round(comps["smoothness"], 6),
            # Weighted total (for sanity check)
            "total_weighted":    round(comps["total_weighted"], 4),
            # Weights used
            "w_costmap":    comps["weights"]["costmap"],
            "w_length":     comps["weights"]["length"],
            "w_goal":       comps["weights"]["goal"],
            "w_smoothness": comps["weights"]["smoothness"],
        }
        records.append(record)

        print(f"    costmap={comps['costmap_cost']:.2f}  "
              f"length={comps['path_length']:.1f} m  "
              f"goal_dist={comps['goal_dist']:.1f} m  "
              f"smoothness={comps['smoothness']:.5f}  "
              f"(total_weighted={comps['total_weighted']:.2f}, {elapsed:.1f}s)", flush=True)

    # Write CSV
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = _PROJECT_ROOT / "calibration" / "results" / f"cost_stats_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(records[0].keys()) if records else []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved {len(records)} records to {out_path}", flush=True)

    # Quick summary
    if records:
        import statistics
        print("\nRaw component summary (across all probes):")
        print(f"  {'Term':<20} {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
        print("  " + "-" * 64)
        for key, label in [
            ("costmap_cost_raw", "costmap_cost"),
            ("path_length_raw",  "path_length [m]"),
            ("goal_dist_raw",    "goal_dist [m]"),
            ("smoothness_raw",   "smoothness"),
        ]:
            vals = [r[key] for r in records]
            print(f"  {label:<20} {statistics.mean(vals):>10.4f}  "
                  f"{statistics.stdev(vals) if len(vals) > 1 else 0:>10.4f}  "
                  f"{min(vals):>10.4f}  {max(vals):>10.4f}")
        print()
        print("Run analyze_cost_stats.py to compute balanced weights:")
        print(f"  uv run python calibration/analyze_cost_stats.py {out_path}")

    return out_path


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_collection(args)


if __name__ == "__main__":
    main()
