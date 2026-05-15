"""
Quick local benchmark: 30% concentration uniform-thin ice, no currents.

Runs all planners on the standard 300×700 m synthetic map, then simulates
each planned trajectory through the ice physics and reports:
  - cumulated planning time
  - actual path length (from simulation)
  - actual costmap cost (integrated over the ship's true trajectory)

Output: outputs/local_scenario/
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medea_ice_ship import (
    build_costmap_from_ice_state,
    create_medea_lattice_planner,
    plan_path,
    optimize_with_diffusion,
    optimize_with_mppi,
    optimize_with_rss,
    optimize_with_ilqr,
    optimize_with_gradcem,
    create_proximity_bonds,
    polygon_area,
    path_to_ship_trajectory_arr,
    ShipProperties,
)
from medea_ice_ship.config import get_default_run_config, get_default_simulation_config
from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan
from medea_ice_ship.synthetic_map import (
    ScenarioSpec,
    make_ice_field,
    SHIP_START,
    GOAL_Y_LINE,
    START_HEADING,
    X_LO, X_HI, Y_LO, Y_HI,
)

OUT_DIR = ROOT / "outputs" / "local_scenario"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Simulation: dt=0.05 s, ship at ~4 m/s, map is 600 m → ~150 s → 3000 frames.
# 6000 frames (300 s) gives a generous margin; goal_y_line stops it early.
N_SIM_FRAMES   = 6000
SIM_GOAL_TOL   = 10.0   # [m] simulation stops when ship y >= GOAL_Y_LINE - this

# ── Scenario ──────────────────────────────────────────────────────────────────

SPEC = ScenarioSpec(
    block="A",
    scenario_id="A_conc30_thin",
    concentration=0.30,
    thickness_mode="uniform_thin",
    with_currents=False,
)

START = (float(SHIP_START[0]), float(SHIP_START[1]), float(START_HEADING))
GOAL  = (0.0, GOAL_Y_LINE)

# ── Reduced horizons for a quick local run ────────────────────────────────────

def make_run_cfg():
    cfg = get_default_run_config()
    return dataclasses.replace(
        cfg,
        # MBD
        planning_jax_mbd_horizon=400,
        planning_jax_mbd_nsample=512,
        planning_jax_mbd_ndiffuse=20,
        # MPPI
        planning_mppi_horizon=300,
        planning_mppi_nsample=512,
        planning_mppi_n_iterations=5,
        # RSS
        planning_rss_horizon=300,
        planning_rss_nsample=1024,
        # iLQR
        planning_ilqr_horizon=200,
        planning_ilqr_max_iters=15,
        # GradCEM
        planning_gradcem_horizon=300,
        planning_gradcem_nsample=256,
        planning_gradcem_top_samples=32,
        planning_gradcem_opt_iters=5,
    )

# ── Plotting helpers ──────────────────────────────────────────────────────────

PLANNER_COLORS = {
    "lattice":  "#1f77b4",
    "rss":      "#d62728",
    "ilqr":     "#9467bd",
    "mppi":     "#ff7f0e",
    "mbd":      "#2ca02c",
    "eb-mbd":   "#17becf",
    "gradcem":  "#8c564b",
}

def _costmap_extent(cost_map):
    return [cost_map.x_origin, cost_map.x_origin + cost_map.width,
            cost_map.y_origin, cost_map.y_origin + cost_map.height]


def _draw_costmap(ax, cost_map):
    cm = cost_map.cost_map.copy()
    ice_mask = (cm > 0) & (cm < 1e6)   # exclude wall cells (1e8)
    vmax = float(np.percentile(cm[ice_mask], 99)) if ice_mask.any() else 1.0
    ax.imshow(np.clip(cm, 0, vmax), origin="lower", extent=_costmap_extent(cost_map),
              cmap="YlOrRd", vmin=0, vmax=vmax, alpha=0.85, aspect="equal")


def _draw_ice_field(ax, polygons, thicknesses):
    patches = [MplPolygon(p, closed=True) for p in polygons]
    coll = PatchCollection(patches, array=thicknesses, cmap="Blues",
                           edgecolors="steelblue", linewidths=0.3, alpha=0.7)
    ax.add_collection(coll)


def _draw_path(ax, path, label, color, lw=2.0, ls="-", zorder=4):
    if path is not None and len(path) >= 2:
        p = np.asarray(path)
        ax.plot(p[:, 0], p[:, 1], ls=ls, color=color, lw=lw, label=label, zorder=zorder)


def _ax_limits(ax):
    pad = 20.0
    ax.set_xlim(X_LO - pad, X_HI + pad)
    ax.set_ylim(Y_LO - pad, Y_HI + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")


def _mark_start_goal(ax):
    ax.plot(START[0], START[1], "gs", ms=12, zorder=8, label="Start")
    ax.axhline(GOAL_Y_LINE, color="red", ls="--", lw=1.5, zorder=7,
               label=f"Goal y={GOAL_Y_LINE:.0f} m")


def _extend_to_goal_line(path, y_line=GOAL_Y_LINE):
    """Append one waypoint at y_line if the path falls short."""
    if path is None or len(path) < 2:
        return path
    p = np.asarray(path)
    if p[-1, 1] >= y_line:
        return p
    last = p[-1]
    psi  = last[2]
    dy   = y_line - last[1]
    dx   = dy * np.tan(psi - np.pi / 2) if abs(np.cos(psi)) > 1e-6 else 0.0
    return np.vstack([p, [[last[0] + dx, y_line, psi]]])


def _truncate_at_goal_line(path, y_line=GOAL_Y_LINE):
    """Clip path at the first waypoint that reaches or crosses y_line."""
    if path is None or len(path) < 2:
        return path
    p = np.asarray(path)
    crossed = np.where(p[:, 1] >= y_line)[0]
    return p if len(crossed) == 0 else p[: crossed[0] + 1]


def _path_length(path):
    if path is None or len(path) < 2:
        return float("nan")
    p = np.asarray(path)
    return float(np.sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)))


def _ship_history_to_array(ship_history):
    """Convert list of (pos_2d, heading) to (N, 3) [x, y, psi] array."""
    positions = np.array([pos for pos, _ in ship_history])
    headings  = np.array([h   for _, h   in ship_history])
    return np.column_stack([positions, headings])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("Local scenario benchmark (planning + simulation)")
    print(f"  Scenario : {SPEC.scenario_id}")
    print(f"  Start    : {START}")
    print(f"  Goal     : y = {GOAL_Y_LINE} m")
    print("=" * 64)

    run_cfg = make_run_cfg()
    sim_cfg = get_default_simulation_config()

    # ── 1. Ice field ──────────────────────────────────────────────────
    print("\n[1] Generating ice field …")
    field       = make_ice_field(SPEC, seed=SEED)
    polygons    = field["polygons"]
    centroids   = field["centroids"]
    thicknesses = field["thicknesses"]
    print(f"    {len(polygons)} floes, concentration={SPEC.concentration:.0%}")

    fig, ax = plt.subplots(figsize=(5, 11))
    _draw_ice_field(ax, polygons, thicknesses)
    _mark_start_goal(ax)
    _ax_limits(ax)
    ax.set_title(f"Ice field — {SPEC.concentration:.0%} conc., {SPEC.thickness_mode}")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "00_ice_field.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("    saved 00_ice_field.png")

    # ── 2. Physics bonds + costmap ────────────────────────────────────
    print("\n[2] Building bonds & costmap …")
    areas = np.array([polygon_area(p) for p in polygons])
    radii = np.sqrt(areas / np.pi)
    bonds = create_proximity_bonds(
        centroids, radii, threshold_factor=run_cfg.bond_threshold_factor
    )
    print(f"    {len(bonds)} bonds between {len(polygons)} floes")

    ship_props = ShipProperties(
        length=run_cfg.ship_length,
        beam=run_cfg.ship_beam,
        draft=run_cfg.ship_draft,
        displacement=run_cfg.ship_displacement,
    )

    t0 = time.perf_counter()
    cost_map, bounds = build_costmap_from_ice_state(
        polygons, centroids, thicknesses,
        X_LO, X_HI, Y_LO, Y_HI,
        resolution=run_cfg.planning_resolution,
        margin=run_cfg.planning_margin,
        ship_mass=run_cfg.ship_displacement,
        collision_cost=run_cfg.planning_ice_collision_cost,
        wall_bounds=(X_LO, X_HI, Y_LO, Y_HI),
        ship_length=run_cfg.ship_length,
        ship_beam=run_cfg.ship_beam,
    )
    print(f"    costmap in {time.perf_counter()-t0:.2f} s — "
          f"grid ({cost_map.rows}×{cost_map.cols}), res={cost_map.resolution} m/cell")

    fig, ax = plt.subplots(figsize=(5, 11))
    _draw_costmap(ax, cost_map)
    _mark_start_goal(ax)
    _ax_limits(ax)
    ax.set_title("Costmap (KE collision cost)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_costmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("    saved 01_costmap.png")

    # ── Simulation helper ─────────────────────────────────────────────
    def run_simulation(planned_path):
        """
        Follow planned_path through the ice physics and return:
          actual_arr : (M, 3) [x, y, psi] actual ship trajectory
          damage     : dict with 'impulse_total'
          sim_elapsed: wall-clock time [s]
        """
        traj_arr = path_to_ship_trajectory_arr(planned_path, N_SIM_FRAMES)
        t_sim = time.perf_counter()
        out = simulate_jax_chunked_replan(
            polygons=polygons,
            centroids=centroids,
            bonds=bonds,
            ship_pos=SHIP_START,
            ship_length=run_cfg.ship_length,
            ship_beam=run_cfg.ship_beam,
            n_frames=N_SIM_FRAMES,
            dt=sim_cfg.dt,
            drift_vel=np.zeros(2),
            thicknesses=thicknesses,
            ship_props=ship_props,
            ship_trajectory=traj_arr,
            replan_every_n_frames=None,   # single-shot, no replanning
            config=sim_cfg,
            goal_y_line=GOAL_Y_LINE,
            goal_tolerance=SIM_GOAL_TOL,
            verbose=False,
        )
        sim_elapsed = time.perf_counter() - t_sim
        # out = (positions_history, rotations_history, bonds_history,
        #        ship_history, submergence_history, damage_summary)
        ship_history  = out[3]
        damage_summary = out[5]
        actual_arr = _ship_history_to_array(ship_history)
        return actual_arr, damage_summary, sim_elapsed

    # ── 3. A* lattice ─────────────────────────────────────────────────
    print("\n[3] A* lattice planner …")
    t0 = time.perf_counter()
    lattice_planner = create_medea_lattice_planner(
        cost_map=cost_map, bounds=bounds,
        start=START, goal=GOAL, run_cfg=run_cfg,
        goal_y_line=GOAL_Y_LINE,
    )
    lattice_path = plan_path(lattice_planner, START, GOAL, run_cfg, cost_map=cost_map)
    lattice_elapsed = time.perf_counter() - t0
    if lattice_path is None:
        print("  WARNING: A* found no path — using straight-line fallback")
        lattice_path = np.array([[START[0], START[1], START[2]],
                                  [GOAL[0],  GOAL[1],  np.pi / 2]])
    lattice_path   = _extend_to_goal_line(lattice_path)
    lattice_path   = _truncate_at_goal_line(lattice_path)
    lattice_cost   = float(cost_map.get_path_cost(lattice_path))
    lattice_length = _path_length(lattice_path)
    print(f"    plan: {lattice_elapsed:.2f} s | cost={lattice_cost:.0f} | len={lattice_length:.0f} m")

    # ── 4. JAX optimizers ─────────────────────────────────────────────
    plan_results = {
        "lattice": (lattice_path, lattice_elapsed, lattice_cost, lattice_length),
    }

    jax_planners = [
        ("rss",     optimize_with_rss,       {}),
        ("ilqr",    optimize_with_ilqr,      {}),
        ("mppi",    optimize_with_mppi,      {}),
        ("mbd",     optimize_with_diffusion, {"goal_y_line": GOAL_Y_LINE}),
        ("eb-mbd",  optimize_with_diffusion, {"goal_y_line": GOAL_Y_LINE, "use_eb_mbd": True}),
        ("gradcem", optimize_with_gradcem,   {}),
    ]

    for name, opt_fn, extra_kwargs in jax_planners:
        print(f"\n[{name.upper()}] planning …")
        t0 = time.perf_counter()
        try:
            path = opt_fn(lattice_path, START, GOAL, cost_map, run_cfg,
                          start_nu=(2.0, 0.0, 0.0), **extra_kwargs)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}"); traceback.print_exc()
            plan_results[name] = (None, 0.0, float("inf"), float("nan"))
            continue
        elapsed    = time.perf_counter() - t0
        path_trunc = _truncate_at_goal_line(path)
        cost       = float(cost_map.get_path_cost(np.asarray(path_trunc)))
        length     = _path_length(path_trunc)
        print(f"    plan: {elapsed:.2f} s | cost={cost:.0f} | len={length:.0f} m")
        plan_results[name] = (path_trunc, elapsed, cost, length)

    # ── 5. Simulate every planner ─────────────────────────────────────
    print("\n[5] Running simulations …")
    print("    (first call triggers XLA JIT compilation — may take 10–30 s)")
    sim_results = {}   # name -> (actual_arr, actual_cost, actual_length, sim_elapsed)

    for name, (planned_path, plan_elapsed, _, _) in plan_results.items():
        if planned_path is None:
            sim_results[name] = (None, float("inf"), float("nan"), 0.0)
            continue
        print(f"\n  [{name}] simulating …", flush=True)
        actual_arr, damage, sim_elapsed = run_simulation(planned_path)
        actual_trunc = _truncate_at_goal_line(actual_arr)
        actual_cost  = float(cost_map.get_path_cost(actual_trunc))
        actual_len   = _path_length(actual_trunc)
        print(f"    sim: {sim_elapsed:.1f} s | actual cost={actual_cost:.0f} | "
              f"actual len={actual_len:.0f} m | impulse={damage['impulse_total']:.1f} N·s")
        sim_results[name] = (actual_arr, actual_cost, actual_len, sim_elapsed)

        # Individual plot: planned (dashed) + actual (solid)
        color = PLANNER_COLORS.get(name, "black")
        fig, ax = plt.subplots(figsize=(6, 12))
        _draw_costmap(ax, cost_map)
        _draw_path(ax, planned_path, "planned", color, lw=1.5, ls="--")
        _draw_path(ax, actual_trunc, "actual",  color, lw=2.5, ls="-")
        _mark_start_goal(ax)
        _ax_limits(ax)
        ax.set_title(f"{name} | plan={plan_elapsed:.1f}s | "
                     f"actual cost={actual_cost:.0f} | len={actual_len:.0f} m")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(OUT_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {name}.png")

    # ── 6. Combined comparison plots ──────────────────────────────────
    print("\n[6] Saving comparison figures …")

    # 6a. Planned paths
    fig, ax = plt.subplots(figsize=(7, 14))
    _draw_costmap(ax, cost_map)
    for name, (path, elapsed, cost, length) in plan_results.items():
        if path is None: continue
        lbl = f"{name}  [plan {elapsed:.1f}s, cost={cost:.0f}, {length:.0f}m]"
        _draw_path(ax, path, lbl, PLANNER_COLORS.get(name, "black"),
                   lw=1.5 if name == "lattice" else 2.0)
    _mark_start_goal(ax)
    _ax_limits(ax)
    ax.set_title(f"Planned paths — {SPEC.scenario_id}")
    ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "08_planned_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  saved 08_planned_comparison.png")

    # 6b. Actual trajectories
    fig, ax = plt.subplots(figsize=(7, 14))
    _draw_costmap(ax, cost_map)
    for name, (actual_arr, actual_cost, actual_len, _) in sim_results.items():
        if actual_arr is None: continue
        actual_trunc = _truncate_at_goal_line(actual_arr)
        lbl = f"{name}  [cost={actual_cost:.0f}, {actual_len:.0f}m]"
        _draw_path(ax, actual_trunc, lbl, PLANNER_COLORS.get(name, "black"),
                   lw=1.5 if name == "lattice" else 2.0)
    _mark_start_goal(ax)
    _ax_limits(ax)
    ax.set_title(f"Actual trajectories — {SPEC.scenario_id}")
    ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "09_actual_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  saved 09_actual_comparison.png")

    # ── 7. Summary table ──────────────────────────────────────────────
    W = 86
    print("\n" + "=" * W)
    print(f"{'Planner':<10}  {'Plan(s)':>7}  {'Planned cost':>13}  {'Planned len':>12}"
          f"  {'Actual cost':>12}  {'Actual len':>11}")
    print("  " + "-"*8 + "  " + "-"*7 + "  " + "-"*13 + "  " + "-"*12
          + "  " + "-"*12 + "  " + "-"*11)
    for name in plan_results:
        _, plan_t, plan_cost, plan_len = plan_results[name]
        actual_arr, act_cost, act_len, _ = sim_results.get(
            name, (None, float("inf"), float("nan"), 0.0)
        )
        p_c = f"{plan_cost:.0f}"  if plan_cost < 1e9 else "FAILED"
        p_l = f"{plan_len:.0f}"   if not np.isnan(plan_len) else "—"
        a_c = f"{act_cost:.0f}"   if act_cost  < 1e9 else "FAILED"
        a_l = f"{act_len:.0f}"    if not np.isnan(act_len)  else "—"
        print(f"  {name:<10}  {plan_t:>7.2f}  {p_c:>13}  {p_l:>12}  {a_c:>12}  {a_l:>11}")
    print("=" * W)

    # CSV
    csv_lines = ["planner,plan_time_s,planned_cost,planned_length_m,"
                 "actual_cost,actual_length_m\n"]
    for name in plan_results:
        _, plan_t, plan_cost, plan_len = plan_results[name]
        _, act_cost, act_len, _ = sim_results.get(name, (None, float("nan"), float("nan"), 0.0))
        csv_lines.append(
            f"{name},{plan_t:.3f},{plan_cost:.2f},{plan_len:.1f},"
            f"{act_cost:.2f},{act_len:.1f}\n"
        )
    csv_path = OUT_DIR / "summary.csv"
    csv_path.write_text("".join(csv_lines))
    print(f"\nCSV: {csv_path}")
    print(f"All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
