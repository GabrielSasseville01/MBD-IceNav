#!/usr/bin/env python3
"""
Planner benchmark: generate random ice fields at various concentrations,
run every available planner, simulate, and save planned-path and actual-path
overlay images with ``_<planner_name>`` suffix.

Usage:
  uv run python examples/benchmark_planners.py
  uv run python examples/benchmark_planners.py --concentrations 0.2 0.4
  uv run python examples/benchmark_planners.py --planners lattice diffusion-mpc
  uv run python examples/benchmark_planners.py --no-sim          # plan-only (no physics sim)
  uv run python examples/benchmark_planners.py --n-frames 300    # shorter sim
  uv run python examples/benchmark_planners.py --replan-every 120

Output (default: outputs/benchmark/):
  conc20_ice_field.png                   — ice field visualisation
  conc20_planned_lattice.png             — planned path on costmap
  conc20_result_lattice.png              — planned + actual path after sim
  conc20_planned_diffusion-mpc.png
  conc20_result_diffusion-mpc.png
  ...
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if "--live" in sys.argv:
    os.environ["DPDEM_LIVE_VIEW"] = "1"
    import matplotlib
    matplotlib.use("TkAgg")

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from dataclasses import replace

from medea_ice_ship import (
    ShipProperties,
    polygon_area,
    polygon_centroid,
    create_proximity_bonds,
    tessellate_all_floes,
    build_costmap_from_ice_state,
    create_medea_lattice_planner,
    plan_path,
    optimize_with_diffusion,
    optimize_with_mppi,
    optimize_with_sqp,
    optimize_with_gradcem,
    optimize_with_rss,
    optimize_with_ilqr,
    path_to_ship_trajectory,
    simulate_ice_ship_interaction,
    make_live_2d_callback,
)
from medea_ice_ship.config import get_default_run_config, get_default_simulation_config
from medea_ice_ship.ice_field_generator import generate_ice_field

ALL_PLANNERS = ["lattice", "diffusion-mpc", "mppi", "sqp", "gradcem", "random-shooting", "ilqr"]


# ── metrics ─────────────────────────────────────────────────────────────────

def compute_path_metrics(path, cost_map, label="planned"):
    """Compute path length and costmap integral for a waypoint path (Nx2+ array)."""
    if path is None or len(path) < 2:
        return {}
    pts = np.asarray(path)[:, :2]
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total_length = float(seg_lens.sum())
    costmap_integral = float(cost_map.get_path_cost(np.asarray(path)))
    return {
        f"{label}_length_m": total_length,
        f"{label}_costmap_cost": costmap_integral,
    }


def compute_actual_path_metrics(ship_history, cost_map):
    """Compute metrics from the actual ship trajectory after simulation."""
    if not ship_history or len(ship_history) < 2:
        return {}
    xs = np.array([sh[0][0] for sh in ship_history])
    ys = np.array([sh[0][1] for sh in ship_history])
    seg_lens = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    total_length = float(seg_lens.sum())

    # sample costmap cost along actual path
    costmap_cost = 0.0
    for i in range(len(xs)):
        costmap_cost += cost_map.get_cost_world(float(xs[i]), float(ys[i]))

    return {
        "actual_length_m": total_length,
        "actual_costmap_cost": costmap_cost,
    }


def compute_fuel_proxy(path_length, costmap_cost, impulse_total,
                       length_weight=1.0, ice_weight=0.01, impulse_weight=1e-6):
    """
    Fuel consumption proxy combining path length, ice traversal cost, and damage.

    fuel ≈ length_weight × path_length
         + ice_weight × costmap_integral
         + impulse_weight × cumulative_impulse

    All three contribute: longer paths burn more fuel, ice breaking uses energy
    proportional to thickness (via costmap), and collisions add drag/damage cost.
    """
    return (length_weight * path_length
            + ice_weight * costmap_cost
            + impulse_weight * impulse_total)


# ── visualisation helpers ───────────────────────────────────────────────────

def plot_ice_field(
    polygons, thicknesses, domain, title="Ice Field", out_path=None,
    ship_start=None, ship_end=None, ship_length=None, ship_beam=None,
    goal_y_line=None,
):
    """Save a static image of the ice field with optional ship markers."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    patches = []
    colors = []
    for i, poly in enumerate(polygons):
        patches.append(MplPolygon(poly, closed=True))
        colors.append(thicknesses[i])
    pc = PatchCollection(patches, cmap="Blues", edgecolor="#555555", linewidth=0.3)
    pc.set_array(np.array(colors))
    if len(colors) > 0:
        pc.set_clim(min(colors), max(colors))
    ax.add_collection(pc)

    x_lo, x_hi, y_lo, y_hi = domain
    pad = 30.0
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_aspect("equal")

    if ship_start is not None:
        ax.plot(ship_start[0], ship_start[1], "gs", ms=12, zorder=5, label="Start")
    if ship_end is not None:
        ax.plot(ship_end[0], ship_end[1], "rx", ms=14, mew=3, zorder=5, label="Goal")
    if goal_y_line is not None:
        ax.axhline(goal_y_line, color="magenta", ls="--", lw=2, label=f"Goal line y={goal_y_line:.0f}")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    fig.colorbar(pc, ax=ax, shrink=0.7, label="Thickness [m]")
    if ship_start is not None or ship_end is not None or goal_y_line is not None:
        ax.legend(loc="upper left")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {out_path}")
    plt.close(fig)


def plot_planned_path(
    cost_map, path, start, goal, domain, planner_name,
    title=None, out_path=None,
    goal_y_line=None, ship_length=None, ship_beam=None,
):
    """Save planned path overlaid on the costmap."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    cm = cost_map.cost_map.copy()
    cm[cm > 500] = 500  # cap for visualisation
    extent = [
        cost_map.x_origin,
        cost_map.x_origin + cost_map.width,
        cost_map.y_origin,
        cost_map.y_origin + cost_map.height,
    ]
    ax.imshow(
        cm, origin="lower", extent=extent,
        cmap="YlOrRd", alpha=0.85, aspect="equal",
    )
    if path is not None and len(path) > 0:
        ax.plot(path[:, 0], path[:, 1], "b-", lw=2, label="Planned path", zorder=4)
    ax.plot(start[0], start[1], "gs", ms=12, zorder=5, label="Start")
    if goal_y_line is not None:
        ax.axhline(goal_y_line, color="magenta", ls="--", lw=2, label=f"Goal line y={goal_y_line:.0f}")
    else:
        ax.plot(goal[0], goal[1], "rx", ms=14, mew=3, zorder=5, label="Goal")

    x_lo, x_hi, y_lo, y_hi = domain
    pad = 30.0
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title or f"Planned Path — {planner_name}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {out_path}")
    plt.close(fig)


def plot_result_paths(
    cost_map, planned_path, ship_history, domain,
    planner_name, title=None, out_path=None,
    goal_y_line=None,
):
    """Save planned + actual ship path overlaid on costmap."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    cm = cost_map.cost_map.copy()
    cm[cm > 500] = 500
    extent = [
        cost_map.x_origin,
        cost_map.x_origin + cost_map.width,
        cost_map.y_origin,
        cost_map.y_origin + cost_map.height,
    ]
    ax.imshow(
        cm, origin="lower", extent=extent,
        cmap="YlOrRd", alpha=0.6, aspect="equal",
    )

    if planned_path is not None and len(planned_path) > 0:
        ax.plot(planned_path[:, 0], planned_path[:, 1], "b--", lw=1.5,
                label="Planned path", zorder=4, alpha=0.7)

    actual_x = [sh[0][0] for sh in ship_history]
    actual_y = [sh[0][1] for sh in ship_history]
    ax.plot(actual_x, actual_y, "r-", lw=2, label="Actual path", zorder=5)

    ax.plot(actual_x[0], actual_y[0], "gs", ms=12, zorder=6, label="Start")
    ax.plot(actual_x[-1], actual_y[-1], "m^", ms=10, zorder=6, label="End")
    if goal_y_line is not None:
        ax.axhline(goal_y_line, color="magenta", ls="--", lw=2,
                   label=f"Goal line y={goal_y_line:.0f}")

    x_lo, x_hi, y_lo, y_hi = domain
    pad = 30.0
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title or f"Result — {planner_name}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"    Saved: {out_path}")
    plt.close(fig)


# ── core benchmark logic ───────────────────────────────────────────────────

def run_planner(
    planner_name: str,
    cost_map,
    bounds,
    start,
    goal,
    run_cfg,
    goal_y_line=None,
):
    """Run a single planner, return path (Nx3 array) or None."""
    use_sampling = planner_name in ("diffusion-mpc", "mppi", "sqp", "gradcem", "random-shooting", "ilqr")
    warm_start_flags = {
        "diffusion-mpc": "planning_jax_mbd_warm_start",
        "mppi": "planning_mppi_warm_start",
        "sqp": "planning_sqp_warm_start",
        "gradcem": "planning_gradcem_warm_start",
        "random-shooting": "planning_rss_warm_start",
        "ilqr": "planning_ilqr_warm_start",
    }
    if use_sampling:
        ws_attr = warm_start_flags[planner_name]
        skip_astar = not getattr(run_cfg, ws_attr, True)
    else:
        skip_astar = False

    if skip_astar:
        sx, sy, sh = start
        gx, gy = goal
        path = np.array([
            [sx, sy, sh],
            [gx, gy, np.arctan2(gy - sy, gx - sx)],
        ])
    else:
        planner = create_medea_lattice_planner(
            cost_map=cost_map, bounds=bounds,
            start=start, goal=goal, run_cfg=run_cfg,
            goal_y_line=goal_y_line,
        )
        path = plan_path(planner, start, goal, run_cfg, cost_map=cost_map)

    if path is None:
        return None

    if use_sampling:
        opt_fn = {
            "diffusion-mpc": optimize_with_diffusion,
            "mppi": optimize_with_mppi,
            "sqp": optimize_with_sqp,
            "gradcem": optimize_with_gradcem,
            "random-shooting": optimize_with_rss,
            "ilqr": optimize_with_ilqr,
        }[planner_name]
        path = opt_fn(path, start, goal, cost_map, run_cfg, start_nu=(2.0, 0.0, 0.0))

    return path


def run_benchmark_for_field(
    field: dict,
    planners: list[str],
    out_dir: Path,
    conc_tag: str,
    run_cfg,
    sim_cfg,
    n_frames: int,
    no_sim: bool = False,
    replan_every: int | None = None,
    goal_y_line_enabled: bool = True,
    kinematic: bool = False,
    live: bool = False,
):
    """Run all planners on a single ice field and save images."""
    sat_polygons = field["polygons"]
    sat_centroids = field["centroids"]
    sat_thicknesses = field["thicknesses"]
    x_lo, x_hi, y_lo, y_hi = field["domain"]

    # tessellate
    polygons, centroids, parent_ids = tessellate_all_floes(
        sat_polygons, sat_centroids,
        cells_per_100m2=run_cfg.cells_per_100m2_medea,
        seed=42,
    )
    thicknesses = sat_thicknesses[parent_ids]
    areas = np.array([polygon_area(p) for p in polygons])
    radii = np.sqrt(areas / np.pi)
    bonds = create_proximity_bonds(centroids, radii, threshold_factor=run_cfg.bond_threshold_factor)
    print(f"  Tessellated: {len(polygons)} cells, {len(bonds)} bonds", flush=True)

    ship_length = run_cfg.ship_length
    ship_beam = run_cfg.ship_beam
    ship_props = ShipProperties(
        ship_length, ship_beam, run_cfg.ship_draft, run_cfg.ship_displacement,
    )

    inset = ship_length / 2 + run_cfg.path_margin
    ship_start = np.array([(x_lo + x_hi) / 2, y_lo + inset])
    ship_end = np.array([(x_lo + x_hi) / 2, y_hi - inset])
    goal_y_line = float(y_hi) if goal_y_line_enabled else None

    start_heading = np.arctan2(ship_end[1] - ship_start[1], ship_end[0] - ship_start[0])
    start_pose = (float(ship_start[0]), float(ship_start[1]), float(start_heading))
    goal_pt = (float(ship_end[0]), float(ship_end[1]))

    # ice field image
    plot_ice_field(
        sat_polygons, sat_thicknesses, field["domain"],
        title=f"Ice Field — conc {field['concentration']:.0%} ({len(sat_polygons)} floes)",
        out_path=str(out_dir / f"{conc_tag}_ice_field.png"),
        ship_start=ship_start, ship_end=ship_end,
        ship_length=ship_length, ship_beam=ship_beam,
        goal_y_line=goal_y_line,
    )

    # build costmap
    wall_bounds = (x_lo, x_hi, y_lo, y_hi)
    cost_map, bounds = build_costmap_from_ice_state(
        sat_polygons, sat_centroids, sat_thicknesses,
        x_lo, x_hi, y_lo, y_hi,
        resolution=run_cfg.planning_resolution,
        margin=run_cfg.planning_margin,
        ship_mass=run_cfg.ship_displacement,
        collision_cost=run_cfg.planning_ice_collision_cost,
        wall_bounds=wall_bounds,
    )

    results = []  # list of metric dicts per planner

    for planner_name in planners:
        print(f"\n  ── Planner: {planner_name} ──", flush=True)
        t0 = time.time()

        try:
            path = run_planner(
                planner_name, cost_map, bounds,
                start_pose, goal_pt, run_cfg,
                goal_y_line=goal_y_line,
            )
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            continue

        if path is None:
            print("    No path found.", flush=True)
            continue

        plan_time = time.time() - t0

        # ── planned path metrics ───────────────────────────────────────
        planned_metrics = compute_path_metrics(path, cost_map, label="planned")
        print(
            f"    Path: {len(path)} wpts, length={planned_metrics.get('planned_length_m', 0):.0f}m, "
            f"costmap={planned_metrics.get('planned_costmap_cost', 0):.0f}, "
            f"plan time: {plan_time:.1f}s",
            flush=True,
        )

        plot_planned_path(
            cost_map, path, start_pose, goal_pt, field["domain"],
            planner_name=planner_name,
            title=(
                f"Planned — {planner_name} (conc {field['concentration']:.0%})\n"
                f"length={planned_metrics.get('planned_length_m', 0):.0f}m, "
                f"costmap={planned_metrics.get('planned_costmap_cost', 0):.0f}"
            ),
            out_path=str(out_dir / f"{conc_tag}_planned_{planner_name}.png"),
            goal_y_line=goal_y_line,
        )

        row = {
            "concentration": field["concentration"],
            "planner": planner_name,
            "plan_time_s": plan_time,
            **planned_metrics,
        }

        if no_sim:
            results.append(row)
            continue

        # simulate
        ship_trajectory = path_to_ship_trajectory(path, n_frames)
        ship_pos = ship_trajectory[0][0]

        # replanning setup (disabled when kinematic or replan_every=0)
        replan_cb = None
        replan_n = None
        if not kinematic and replan_every is not None and replan_every > 0:
            replan_n = replan_every

            def _make_replan_cb(pname):
                def _cb(frame, world_polys, cents, thicks, spos, shead, n_rem):
                    cm_f, _ = build_costmap_from_ice_state(
                        world_polys, cents, thicks,
                        x_lo, x_hi, y_lo, y_hi,
                        resolution=run_cfg.planning_resolution,
                        margin=run_cfg.planning_margin,
                        ship_mass=run_cfg.ship_displacement,
                        collision_cost=run_cfg.planning_ice_collision_cost,
                                        wall_bounds=wall_bounds,
                    )
                    st = (float(spos[0]), float(spos[1]), float(shead))
                    try:
                        p = run_planner(pname, cm_f, bounds, st, goal_pt, run_cfg, goal_y_line)
                    except Exception:
                        return None
                    if p is None:
                        return None
                    return path_to_ship_trajectory(p, n_rem)
                return _cb

            replan_cb = _make_replan_cb(planner_name)
            print(f"    Replanning every {replan_n} frames", flush=True)

        frame_callback = None
        if live:
            local_polys_live = [np.asarray(p) - polygon_centroid(p) for p in polygons]
            margin_live = 50.0
            extent_live = (
                x_lo - margin_live, x_hi + margin_live,
                y_lo - margin_live, y_hi + margin_live,
            )
            _fig_live, _ax_live, frame_callback = make_live_2d_callback(
                bonds, local_polys_live, ship_length, ship_beam,
                thicknesses, extent_live, n_frames,
            )
            print(f"    Live 2D view enabled", flush=True)

        mode = "kinematic" if kinematic else "dynamics"
        t0 = time.time()
        print(f"    Simulating ({n_frames} frames, {mode})...", flush=True)
        sim_result = simulate_ice_ship_interaction(
            polygons=polygons,
            centroids=centroids.copy(),
            bonds=bonds,
            ship_pos=ship_pos,
            ship_length=ship_length,
            ship_beam=ship_beam,
            n_frames=n_frames,
            dt=run_cfg.dt,
            drift_vel=0.0,
            thicknesses=thicknesses,
            ship_props=ship_props,
            ship_trajectory=ship_trajectory,
            config=sim_cfg,
            replan_every_n_frames=replan_n,
            replan_callback=replan_cb,
            parent_ids=parent_ids,
            kinematic_ship=kinematic,
            frame_callback=frame_callback,
        )
        ship_history = sim_result[3]
        damage_summary = sim_result[5] if len(sim_result) > 5 and isinstance(sim_result[5], dict) else None
        sim_time = time.time() - t0

        # ── actual path metrics ────────────────────────────────────────
        actual_metrics = compute_actual_path_metrics(ship_history, cost_map)
        impulse_total = 0.0
        bonds_broken = 0
        if damage_summary is not None:
            impulse_total = damage_summary.get("impulse_total", 0.0)
            dps = damage_summary.get("dpdem_bond_stats")
            if dps:
                bonds_broken = dps.get("n_failed_by_damage", 0)

        fuel = compute_fuel_proxy(
            actual_metrics.get("actual_length_m", 0),
            actual_metrics.get("actual_costmap_cost", 0),
            impulse_total,
        )

        print(
            f"    Sim time: {sim_time:.1f}s | "
            f"actual length={actual_metrics.get('actual_length_m', 0):.0f}m, "
            f"costmap={actual_metrics.get('actual_costmap_cost', 0):.0f}, "
            f"impulse={impulse_total/1e6:.2f} MN·s, "
            f"fuel proxy={fuel:.0f}",
            flush=True,
        )

        row.update({
            "sim_time_s": sim_time,
            **actual_metrics,
            "impulse_MNs": impulse_total / 1e6,
            "bonds_broken": bonds_broken,
            "fuel_proxy": fuel,
        })
        results.append(row)

        if live:
            print("    Close the live window to continue...", flush=True)
            plt.show()

        plot_result_paths(
            cost_map, path, ship_history, field["domain"],
            planner_name=planner_name,
            title=(
                f"Result — {planner_name} (conc {field['concentration']:.0%})\n"
                f"length={actual_metrics.get('actual_length_m', 0):.0f}m, "
                f"costmap={actual_metrics.get('actual_costmap_cost', 0):.0f}, "
                f"impulse={impulse_total/1e6:.2f} MN·s, fuel={fuel:.0f}"
            ),
            out_path=str(out_dir / f"{conc_tag}_result_{planner_name}.png"),
            goal_y_line=goal_y_line,
        )

    return results


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all planners on random ice fields at various concentrations",
    )
    parser.add_argument(
        "--concentrations", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5],
        help="Ice field concentrations to test (default: 0.2 0.3 0.4 0.5)",
    )
    parser.add_argument(
        "--planners", nargs="+", type=str, default=ALL_PLANNERS,
        choices=ALL_PLANNERS,
        help="Planners to benchmark (default: all)",
    )
    parser.add_argument(
        "--map-size", type=float, default=1400.0,
        help="Square map side length in metres (default: 1400)",
    )
    parser.add_argument(
        "--n-frames", type=int, default=300,
        help="Number of simulation frames per run (default: 300)",
    )
    parser.add_argument(
        "--no-sim", action="store_true",
        help="Plan-only mode: save planned-path images without running physics simulation",
    )
    parser.add_argument(
        "--replan-every", type=int, default=120,
        help="Replan every N frames during simulation (default: 120; 0 to disable)",
    )
    parser.add_argument(
        "--kinematic", action="store_true",
        help="Place ship exactly on planned path (no physics); fast way to compare planned vs reference",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Open a real-time 2D window during each simulation (close window to continue to next planner)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for ice field generation (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/benchmark",
        help="Output directory (default: outputs/benchmark)",
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use greedy best-first search for lattice planner (faster, suboptimal)",
    )
    parser.add_argument(
        "--no-warm-start", action="store_true",
        help="Disable A* warm-start for sampling planners (diffusion-mpc, mppi, sqp, gradcem)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = get_default_run_config()
    run_cfg = replace(
        run_cfg,
        planning_turning_radius=50.0,
        planning_greedy=args.greedy,
    )
    if args.no_warm_start:
        run_cfg = replace(
            run_cfg,
            planning_jax_mbd_warm_start=False,
            planning_mppi_warm_start=False,
            planning_sqp_warm_start=False,
            planning_gradcem_warm_start=False,
        )
    sim_cfg = get_default_simulation_config()

    print("=" * 60)
    print("Planner Benchmark")
    print(f"  Concentrations: {args.concentrations}")
    print(f"  Planners:       {args.planners}")
    print(f"  Map size:       {args.map_size:.0f} x {args.map_size:.0f} m")
    print(f"  Sim frames:     {args.n_frames}")
    print(f"  No sim:         {args.no_sim}")
    print(f"  Kinematic:      {args.kinematic}")
    print(f"  Replan every:   {args.replan_every if args.replan_every else 'disabled'}")
    print(f"  No warm-start:  {args.no_warm_start}")
    print(f"  Output:         {out_dir.resolve()}")
    print("=" * 60)

    total_t0 = time.time()
    all_results = []

    for conc in args.concentrations:
        print(f"\n{'─'*60}")
        print(f"Generating ice field at {conc:.0%} concentration...", flush=True)
        field = generate_ice_field(
            concentration=conc,
            map_shape=(args.map_size, args.map_size),
            seed=args.seed + int(conc * 1000),
        )
        print(
            f"  {len(field['polygons'])} floes, actual concentration {field['concentration']:.1%}",
            flush=True,
        )

        conc_tag = f"conc{int(conc * 100)}"
        field_results = run_benchmark_for_field(
            field=field,
            planners=args.planners,
            out_dir=out_dir,
            conc_tag=conc_tag,
            run_cfg=run_cfg,
            sim_cfg=sim_cfg,
            n_frames=args.n_frames,
            no_sim=args.no_sim,
            replan_every=args.replan_every if args.replan_every else None,
            kinematic=args.kinematic,
            live=args.live,
        )
        all_results.extend(field_results)

    total_time = time.time() - total_t0

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("BENCHMARK RESULTS")
    print(f"{'='*100}")

    if all_results:
        has_sim = any("actual_length_m" in r for r in all_results)

        if has_sim:
            header = (
                f"{'Conc':>5} │ {'Planner':<15} │ {'Plan(s)':>7} │ "
                f"{'Pln Len':>8} │ {'Act Len':>8} │ {'Pln Cost':>9} │ "
                f"{'Act Cost':>9} │ {'Impulse':>9} │ {'Fuel':>9}"
            )
            print(header)
            print("─" * len(header))
            for r in all_results:
                print(
                    f"{r.get('concentration', 0):>5.0%} │ "
                    f"{r['planner']:<15} │ "
                    f"{r.get('plan_time_s', 0):>7.1f} │ "
                    f"{r.get('planned_length_m', 0):>7.0f}m │ "
                    f"{r.get('actual_length_m', 0):>7.0f}m │ "
                    f"{r.get('planned_costmap_cost', 0):>9.0f} │ "
                    f"{r.get('actual_costmap_cost', 0):>9.0f} │ "
                    f"{r.get('impulse_MNs', 0):>7.2f}MN │ "
                    f"{r.get('fuel_proxy', 0):>9.0f}"
                )
        else:
            header = (
                f"{'Conc':>5} │ {'Planner':<15} │ {'Plan(s)':>7} │ "
                f"{'Pln Len':>8} │ {'Pln Cost':>9}"
            )
            print(header)
            print("─" * len(header))
            for r in all_results:
                print(
                    f"{r.get('concentration', 0):>5.0%} │ "
                    f"{r['planner']:<15} │ "
                    f"{r.get('plan_time_s', 0):>7.1f} │ "
                    f"{r.get('planned_length_m', 0):>7.0f}m │ "
                    f"{r.get('planned_costmap_cost', 0):>9.0f}"
                )

        # Save CSV
        csv_path = out_dir / "benchmark_results.csv"
        all_keys = []
        for r in all_results:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(csv_path, "w") as f:
            f.write(",".join(all_keys) + "\n")
            for r in all_results:
                vals = [str(r.get(k, "")) for k in all_keys]
                f.write(",".join(vals) + "\n")
        print(f"\nResults saved to: {csv_path}")

    print(f"\nBenchmark complete in {total_time:.1f}s")
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
