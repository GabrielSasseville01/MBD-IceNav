#!/usr/bin/env python3
"""
Toy MBD scenario suite — four environments showcasing failure modes.

Scenarios
---------
  thin_corridor     80 m corridors on each side of a 640 m wide floe.  Most
                    random trajectories hit ice → ESS collapse for EB-MBD.
  sequential_gates  Two offset rectangular blocks forcing an S-curve path.
                    Tests per-constraint adaptivity of AL-MBD vs the single
                    merged-constraint barrier of EB-MBD.
  asymmetric        Floe leaves a wide (200 m) left corridor and a narrow
                    (80 m) right corridor; goal biased toward the narrow side.
                    Tests goal-directed vs constraint-satisfying trade-off.
  cluttered         12 randomly placed ice disks (seed=42).  Tests
                    many-constraint performance.

Output structure
----------------
  outputs/scenarios/{scenario}/{method}/
      costmap.png
      result.png
      denoising/          ← per-step trajectory movies from jax_mbd
      trajectory_fan.mp4  ← sample fan + crossing histogram
      eb_barrier_schedule.mp4     (EB-MBD only)
      al_multiplier_evolution.png (AL-MBD only)
  outputs/scenarios/{scenario}/comparison_table.txt

Usage
-----
  # Run all methods on thin_corridor
  uv run python examples/toy_mbd_scenarios.py --scenario thin_corridor --run-all-methods

  # Run a single method
  uv run python examples/toy_mbd_scenarios.py --scenario cluttered --method al_mbd

  # Custom params
  uv run python examples/toy_mbd_scenarios.py --scenario sequential_gates --method eb_mbd \\
      --eb-c-max 2000 --eb-mu 60000
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from scipy.ndimage import maximum_filter

from medea_ice_ship.config import get_default_run_config
from medea_ice_ship.jax_costmap import JaxCostMapResult, build_costmap_jax
from medea_ice_ship.jax_mbd_planner import jax_mbd_optimize, rrt_path_on_costmap, astar_path_on_costmap
from medea_ice_ship.planning import create_medea_lattice_planner, plan_path
from medea_ice_ship.visualization import _animation_writer
from dpdem.planning.cost_map import CostMap as DpdemCostMap


# ── Shared map geometry ─────────────────────────────────────────────────────────

X_MIN, X_MAX = 0.0,   800.0
Y_MIN, Y_MAX = 0.0,  1400.0
GOAL_LINE    = 1350.0
WALL_W       = 40.0   # OOB shading width for plots


# ── Scenario config ─────────────────────────────────────────────────────────────

@dataclass
class ScenarioCfg:
    name: str
    description: str
    start: Tuple[float, float, float]  # (x, y, psi)
    goal: Tuple[float, float]          # (x, y)
    costmap_fn: Callable               # (run_cfg) -> (JaxCostMapResult, geo_info)
    crossing_y: float                  # y-level for crossing-x histogram
    obstacle_x_spans: List[Tuple[float, float]]   # (x_lo, x_hi) — red in histogram
    corridor_x_spans: List[Tuple[float, float]]   # (x_lo, x_hi) — green in histogram
    goal_x_weight: float = 1.0
    # Hard rectangular obstacles passed as constraints to EB/AL-MBD (not as cost)
    obstacle_rects: Optional[List[Tuple[float, float, float, float]]] = None  # (x0,x1,y0,y1)


# ── Costmap builders ─────────────────────────────────────────────────────────────

def _disk_costmap(centroids, radii, run_cfg, inflate=True):
    cost_map, _ = build_costmap_jax(
        centroids, np.full(len(centroids), 2.0), radii,
        X_MIN, X_MAX, Y_MIN, Y_MAX,
        resolution=run_cfg.planning_resolution,
        margin=20.0,
        base_cost=0.0,
        ship_length=run_cfg.ship_length if inflate else None,
        ship_beam=run_cfg.ship_beam   if inflate else None,
        ship_mass=run_cfg.ship_displacement,
        obs_cost_buffer_factor=0.05 if inflate else 0.0,
        verbose=True,
    )
    return cost_map


def _rect_costmap(rects, run_cfg, inflate=True):
    """Build costmap with rectangular high-cost obstacles."""
    resolution = run_cfg.planning_resolution
    margin = 20.0
    x_origin = X_MIN - margin
    y_origin = Y_MIN - margin
    width  = (X_MAX - X_MIN) + 2 * margin
    height = (Y_MAX - Y_MIN) + 2 * margin
    cols = max(1, int(np.ceil(width  / resolution)))
    rows = max(1, int(np.ceil(height / resolution)))

    grid = np.zeros((rows, cols), dtype=np.float32)
    ICE_COST = 1e5
    WALL_COST = 1e8
    for (rx0, rx1, ry0, ry1) in rects:
        c0 = max(0, int((rx0 - x_origin) / resolution))
        c1 = min(cols, int(np.ceil((rx1 - x_origin) / resolution)))
        r0 = max(0, int((ry0 - y_origin) / resolution))
        r1 = min(rows, int(np.ceil((ry1 - y_origin) / resolution)))
        grid[r0:r1, c0:c1] = ICE_COST

    if inflate:
        infl_cells = max(1, int(np.ceil(run_cfg.ship_beam / 2.0 / resolution)))
        grid = maximum_filter(grid, size=2 * infl_cells + 1).astype(np.float32)

    # Side walls prevent x-axis escape; EB barrier (eb_y_min) handles the bottom so
    # hull corners extending below Y_MIN don't hit WALL_COST and drown the ice signal.
    col_x = x_origin + (np.arange(cols) + 0.5) * resolution
    grid[:, (col_x < X_MIN) | (col_x > X_MAX)] = WALL_COST

    return JaxCostMapResult(
        cost_map=grid,
        x_origin=x_origin, y_origin=y_origin,
        resolution=resolution,
        rows=rows, cols=cols,
        width=width, height=height,
        base_cost=0.0,
    )


def _build_thin_corridor(run_cfg, inflate=True):
    """640 m wide floe — 80 m corridors left and right."""
    FX0, FX1 = 80.0, 720.0
    FY0, FY1 = 560.0, 720.0
    cx, cy = (FX0 + FX1) / 2, (FY0 + FY1) / 2
    r = 0.5 * math.sqrt((FX1 - FX0) ** 2 + (FY1 - FY0) ** 2)
    cm = _disk_costmap(np.array([[cx, cy]]), np.array([r]), run_cfg, inflate=inflate)
    return cm, {"rects": [(FX0, FX1, FY0, FY1)], "disks": [], "cx": cx, "cy": cy}


def _build_sequential_gates(run_cfg, inflate=True):
    """Two rectangular blocks: right gate at lower level, left gate at upper."""
    BLOCK1 = (0.0, 600.0, 400.0, 550.0)    # leaves right gate x=[600,800] (200 m)
    BLOCK2 = (200.0, 800.0, 800.0, 950.0)  # leaves left gate  x=[0,200]   (200 m)
    cm = _rect_costmap([BLOCK1, BLOCK2], run_cfg, inflate=inflate)
    return cm, {"rects": [BLOCK1, BLOCK2], "disks": []}


def _build_asymmetric(run_cfg, inflate=True):
    """Floe x=[200,720]: wide left corridor (200 m), narrow right (80 m)."""
    FX0, FX1 = 200.0, 720.0
    FY0, FY1 = 560.0, 720.0
    cx, cy = (FX0 + FX1) / 2, (FY0 + FY1) / 2
    r = 0.5 * math.sqrt((FX1 - FX0) ** 2 + (FY1 - FY0) ** 2)
    cm = _disk_costmap(np.array([[cx, cy]]), np.array([r]), run_cfg, inflate=inflate)
    return cm, {"rects": [(FX0, FX1, FY0, FY1)], "disks": [], "cx": cx, "cy": cy}


def _build_cluttered(run_cfg, inflate=True):
    """12 randomly placed ice disks (seed=42)."""
    rng = np.random.default_rng(42)
    n = 12
    cx = rng.uniform(100.0, 700.0, n)
    cy = rng.uniform(200.0, 1100.0, n)
    r  = rng.uniform(50.0, 90.0, n)
    centroids = np.column_stack([cx, cy])
    cm = _disk_costmap(centroids, r, run_cfg, inflate=inflate)
    return cm, {"rects": [], "disks": list(zip(cx.tolist(), cy.tolist(), r.tolist()))}


def _build_open_water(run_cfg):
    """Uniform open-water costmap — no ice cost anywhere."""
    resolution = run_cfg.planning_resolution
    margin = 20.0
    x_origin = X_MIN - margin
    y_origin = Y_MIN - margin
    width  = (X_MAX - X_MIN) + 2 * margin
    height = (Y_MAX - Y_MIN) + 2 * margin
    cols = max(1, int(np.ceil(width  / resolution)))
    rows = max(1, int(np.ceil(height / resolution)))
    grid = np.zeros((rows, cols), dtype=np.float32)
    return JaxCostMapResult(
        cost_map=grid,
        x_origin=x_origin, y_origin=y_origin,
        resolution=resolution,
        rows=rows, cols=cols,
        width=width, height=height,
        base_cost=0.0,
    )


# ── Scenario registry ───────────────────────────────────────────────────────────

SCENARIOS: dict[str, ScenarioCfg] = {
    "thin_corridor": ScenarioCfg(
        name="thin_corridor",
        description="Wide floe leaving 80 m corridors — stresses ESS collapse",
        start=(400.0, 50.0, math.pi / 2),
        goal=(400.0, 1350.0),
        costmap_fn=_build_thin_corridor,
        crossing_y=640.0,
        obstacle_x_spans=[(80.0, 720.0)],
        corridor_x_spans=[(X_MIN, 80.0), (720.0, X_MAX)],
        goal_x_weight=1.0,
    ),
    "sequential_gates": ScenarioCfg(
        name="sequential_gates",
        description="Two offset gates requiring an S-curve path",
        start=(400.0, 50.0, math.pi / 2),
        goal=(400.0, 1350.0),
        costmap_fn=_build_sequential_gates,
        crossing_y=475.0,               # centre of lower block
        obstacle_x_spans=[(0.0, 600.0)],   # lower block spans (for histogram at y=475)
        corridor_x_spans=[(600.0, X_MAX)], # right gate of lower block
        goal_x_weight=1.0,
    ),
    "asymmetric": ScenarioCfg(
        name="asymmetric",
        description="Wide left / narrow right corridor — goal on narrow side",
        start=(400.0, 50.0, math.pi / 2),
        goal=(760.0, 1350.0),           # goal biased toward narrow right corridor
        costmap_fn=_build_asymmetric,
        crossing_y=640.0,
        obstacle_x_spans=[(200.0, 720.0)],
        corridor_x_spans=[(X_MIN, 200.0), (720.0, X_MAX)],
        goal_x_weight=1.0,
    ),
    "cluttered": ScenarioCfg(
        name="cluttered",
        description="12 scattered ice disks — many-constraint performance",
        start=(400.0, 50.0, math.pi / 2),
        goal=(400.0, 1350.0),
        costmap_fn=_build_cluttered,
        crossing_y=700.0,               # mid-map measurement level
        obstacle_x_spans=[],
        corridor_x_spans=[],
        goal_x_weight=1.0,
    ),
    "sequential_gates_hard": ScenarioCfg(
        name="sequential_gates_hard",
        description="Gates as hard constraints on open water — EB/AL only",
        start=(400.0, 50.0, math.pi / 2),
        goal=(400.0, 1350.0),
        costmap_fn=lambda run_cfg, inflate=True: (
            _build_open_water(run_cfg),
            {"rects": [(0.0, 600.0, 400.0, 550.0), (200.0, 800.0, 800.0, 950.0)], "disks": []}
        ),
        crossing_y=475.0,
        obstacle_x_spans=[(0.0, 600.0)],
        corridor_x_spans=[(600.0, X_MAX)],
        goal_x_weight=1.0,
        obstacle_rects=[
            (0.0, 600.0, 400.0, 550.0),    # BLOCK1: leaves right gate x=[600,800]
            (200.0, 800.0, 800.0, 950.0),  # BLOCK2: leaves left gate  x=[0,200]
        ],
    ),
}


# ── Plotting helpers ─────────────────────────────────────────────────────────────

def _draw_costmap_background(ax, cost_map, geo_info):
    grid   = cost_map.cost_map
    extent = [cost_map.x_origin, cost_map.x_origin + cost_map.width,
              cost_map.y_origin, cost_map.y_origin + cost_map.height]
    # Clip wall-cost cells before scaling so ice costs are visible
    WALL_CLIP = 1e7
    display_grid = np.where(grid >= WALL_CLIP, 0.0, grid)
    vmax = float(np.percentile(display_grid[display_grid > 0], 95)) if np.any(display_grid > 0) else 1.0
    ax.imshow(display_grid, origin="lower", extent=extent, cmap="YlOrRd",
              alpha=0.65, vmin=0, vmax=vmax)

    for (rx0, rx1, ry0, ry1) in geo_info.get("rects", []):
        ax.add_patch(mpatches.FancyBboxPatch(
            (rx0, ry0), rx1 - rx0, ry1 - ry0,
            linewidth=1.5, edgecolor="steelblue", facecolor="none",
            boxstyle="square,pad=0",
        ))
    for (dcx, dcy, dr) in geo_info.get("disks", []):
        ax.add_patch(plt.Circle((dcx, dcy), dr,
                                linewidth=1, edgecolor="steelblue",
                                facecolor="none"))

    for patch_xy, pw, ph in [
        ((X_MIN - WALL_W, Y_MIN), WALL_W, Y_MAX - Y_MIN + WALL_W),
        ((X_MAX,          Y_MIN), WALL_W, Y_MAX - Y_MIN + WALL_W),
        ((X_MIN - WALL_W, Y_MIN - WALL_W), X_MAX - X_MIN + 2*WALL_W, WALL_W),
    ]:
        ax.add_patch(mpatches.Rectangle(patch_xy, pw, ph,
                                        facecolor="black", alpha=0.25, zorder=2))

    ax.axhline(GOAL_LINE, color="magenta", lw=2, ls="--", alpha=0.8, zorder=3,
               label=f"Goal  y={GOAL_LINE:.0f} m")
    ax.set_xlim(X_MIN - WALL_W - 5, X_MAX + WALL_W + 5)
    ax.set_ylim(Y_MIN - WALL_W - 5, Y_MAX + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")


def plot_costmap(cost_map, geo_info, sc: ScenarioCfg, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 16))
    _draw_costmap_background(ax, cost_map, geo_info)
    ax.plot(*sc.start[:2], "go",  ms=10, zorder=6, label="Start")
    ax.plot(*sc.goal,      "m*",  ms=14, zorder=6, label="Goal")
    ax.set_title(f"{sc.name}\n{sc.description}", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def plot_result(cost_map, geo_info, sc: ScenarioCfg,
                mean_traj, best_traj, mean_cost, best_cost,
                algo_name: str, out_path: str,
                rrt_path=None, warmstart_traj=None) -> None:
    fig, ax = plt.subplots(figsize=(6, 16))
    _draw_costmap_background(ax, cost_map, geo_info)
    if rrt_path is not None:
        _path_label = "A* waypoints" if "A*" in algo_name else "RRT waypoints"
        ax.plot(rrt_path[:, 0], rrt_path[:, 1], "w-",  lw=5, zorder=3)
        ax.plot(rrt_path[:, 0], rrt_path[:, 1], "--", color="cyan", lw=2.5,
                zorder=4, label=_path_label)
        ax.scatter(rrt_path[:, 0], rrt_path[:, 1], c="cyan", s=40, zorder=5)
    if warmstart_traj is not None:
        ax.plot(warmstart_traj[:, 0], warmstart_traj[:, 1], "w-",  lw=3, zorder=5)
        ax.plot(warmstart_traj[:, 0], warmstart_traj[:, 1], "-", color="yellow", lw=1.8,
                alpha=0.8, zorder=6, label="Warm-start rollout (Ybar₀)")
    ax.plot(mean_traj[:, 0], mean_traj[:, 1], "w-",    lw=4, zorder=7)
    ax.plot(mean_traj[:, 0], mean_traj[:, 1], "-",  color="tomato", lw=2.5,
            zorder=8, label=f"Mean traj  (cost={mean_cost:.0f})")
    ax.plot(best_traj[:, 0], best_traj[:, 1], "w-",    lw=4, zorder=9)
    ax.plot(best_traj[:, 0], best_traj[:, 1], "-",  color="lime",  lw=2.5,
            zorder=10, label=f"Best sample  (cost={best_cost:.0f})")
    ax.plot(*sc.start[:2], "go", ms=10, zorder=11, label="Start")
    ax.plot(*sc.goal,      "m*", ms=14, zorder=11, label="Goal")
    ratio = mean_cost / max(best_cost, 1.0)
    verdict = "COLLAPSE" if ratio > 1.5 else "OK"
    ax.set_title(
        f"{sc.name}  [{algo_name}]\n"
        f"mean={mean_cost:.0f}  best={best_cost:.0f}  "
        f"ratio={ratio:.2f}× [{verdict}]",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ── Crossing-x helper ────────────────────────────────────────────────────────────

def _crossing_x_at_y(traj: np.ndarray, y_target: float) -> float:
    ys, xs = traj[:, 1], traj[:, 0]
    for i in range(1, len(ys)):
        if ys[i - 1] < y_target <= ys[i]:
            dy = ys[i] - ys[i - 1]
            frac = (y_target - ys[i - 1]) / max(dy, 1e-8)
            return float(xs[i - 1] + frac * (xs[i] - xs[i - 1]))
    return float("nan")


# ── Trajectory fan movie ─────────────────────────────────────────────────────────

def generate_trajectory_fan_movie(
    diag: dict, cost_map, geo_info, sc: ScenarioCfg,
    algo_name: str, out_path: str,
) -> None:
    """
    Side-by-side movie:
      Left  — sample trajectory fan + mean/best on costmap background
      Right — crossing-x histogram at sc.crossing_y
    """
    n_steps = len(diag["sample_trajectories"])
    if n_steps == 0:
        print("  No sample data — skipping trajectory fan movie.", flush=True)
        return

    # Pre-compute per step
    crossing_xs_per_step: list[np.ndarray] = []
    mean_crossing_per_step: list[float] = []
    global_min_traj_per_step: list[np.ndarray] = []
    _best_cost = float("inf")
    _best_traj = None
    for step in range(n_steps):
        trajs = diag["sample_trajectories"][step]
        cxs   = np.array([_crossing_x_at_y(trajs[i], sc.crossing_y) for i in range(len(trajs))])
        crossing_xs_per_step.append(cxs)
        mt_idx = min(step + 1, len(diag["mean_trajectories"]) - 1)
        mean_crossing_per_step.append(_crossing_x_at_y(diag["mean_trajectories"][mt_idx], sc.crossing_y))
        sc_best = diag["cost_mins"][step]
        if sc_best < _best_cost:
            _best_cost = sc_best
            _best_traj = trajs[0]
        global_min_traj_per_step.append(_best_traj)

    bins        = np.linspace(X_MIN, X_MAX, 41)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width   = bins[1] - bins[0]
    max_count   = 1
    for cxs in crossing_xs_per_step:
        valid = cxs[~np.isnan(cxs)]
        if len(valid):
            max_count = max(max_count, int(np.histogram(valid, bins=bins)[0].max()))

    def _bar_color(bx: float) -> str:
        for (lo, hi) in sc.obstacle_x_spans:
            if lo <= bx <= hi:
                return "tomato"
        for (lo, hi) in sc.corridor_x_spans:
            if lo <= bx <= hi:
                return "limegreen" if bx < X_MAX / 2 else "deepskyblue"
        return "gold"

    bar_colors = [_bar_color(bx) for bx in bin_centers]

    fig = plt.figure(figsize=(18, 11), facecolor="k")
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.08)
    ax_map  = fig.add_subplot(gs[0], facecolor="#111111")
    ax_hist = fig.add_subplot(gs[1], facecolor="#111111")

    for ax in (ax_map, ax_hist):
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555555")

    # Left: static costmap background
    grid   = cost_map.cost_map
    extent = [cost_map.x_origin, cost_map.x_origin + cost_map.width,
              cost_map.y_origin, cost_map.y_origin + cost_map.height]
    WALL_CLIP = 1e7
    display_grid = np.where(grid >= WALL_CLIP, 0.0, grid)
    vmax   = float(np.percentile(display_grid[display_grid > 0], 95)) if np.any(display_grid > 0) else 1.0
    ax_map.imshow(display_grid, origin="lower", extent=extent, cmap="YlOrRd",
                  alpha=0.55, vmin=0, vmax=vmax, zorder=1)
    for (rx0, rx1, ry0, ry1) in geo_info.get("rects", []):
        ax_map.add_patch(mpatches.FancyBboxPatch(
            (rx0, ry0), rx1 - rx0, ry1 - ry0,
            linewidth=1.5, edgecolor="white", facecolor="none",
            boxstyle="square,pad=0", zorder=2,
        ))
    for (dcx, dcy, dr) in geo_info.get("disks", []):
        ax_map.add_patch(plt.Circle((dcx, dcy), dr,
                                    linewidth=1, edgecolor="white",
                                    facecolor="none", zorder=2))
    ax_map.axhline(sc.crossing_y, color="cyan", lw=1.2, ls="--", alpha=0.7, zorder=3,
                   label=f"Crossing y={sc.crossing_y:.0f} m")
    ax_map.plot(*sc.start[:2], "go", ms=8, zorder=5)
    ax_map.plot(*sc.goal,      "m*", ms=12, zorder=5)
    ax_map.set_xlim(X_MIN, X_MAX)
    ax_map.set_ylim(Y_MIN, Y_MAX)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.set_title("Sample trajectories", color="white", fontsize=11)
    ax_map.legend(loc="upper right", fontsize=7, facecolor="#222222",
                  labelcolor="white", framealpha=0.7)

    sample_lc        = LineCollection([], linewidths=0.5, alpha=0.30, zorder=3)
    ax_map.add_collection(sample_lc)
    mean_outline_map, = ax_map.plot([], [], "w-",  lw=4,  zorder=4)
    mean_line_map,    = ax_map.plot([], [], "-",   color="orange", lw=2, zorder=5)
    best_outline_map, = ax_map.plot([], [], "w-",  lw=4,  zorder=6)
    best_line_map,    = ax_map.plot([], [], "-",   color="lime",   lw=2, zorder=7)
    gmin_outline_map, = ax_map.plot([], [], "w-",  lw=4,  zorder=8)
    gmin_line_map,    = ax_map.plot([], [], "-",   color="gold",   lw=2, zorder=9)
    step_text = ax_map.text(0.02, 0.98, "", transform=ax_map.transAxes,
                             color="white", fontsize=9, va="top",
                             bbox=dict(facecolor="#00000088", edgecolor="none", pad=3))

    # Right: histogram region shading
    for (lo, hi) in sc.corridor_x_spans:
        col = "limegreen" if lo < X_MAX / 2 else "deepskyblue"
        ax_hist.axvspan(lo, hi, alpha=0.10, color=col, zorder=1)
    for (lo, hi) in sc.obstacle_x_spans:
        ax_hist.axvspan(lo, hi, alpha=0.18, color="red", zorder=1)
        ax_hist.axvline(lo, color="red", lw=1.2, ls="--", alpha=0.8, zorder=2)
        ax_hist.axvline(hi, color="red", lw=1.2, ls="--", alpha=0.8, zorder=2)

    mean_vline = ax_hist.axvline(np.nan, color="orange", lw=2.5, zorder=5)
    best_vline = ax_hist.axvline(np.nan, color="lime",   lw=2.5, zorder=5, ls="--")
    gmin_vline = ax_hist.axvline(np.nan, color="gold",   lw=2.5, zorder=5, ls=":")
    ax_hist.set_xlim(X_MIN, X_MAX)
    ax_hist.set_ylim(0, max_count * 1.18)
    ax_hist.set_xlabel(f"x when trajectory crosses y={sc.crossing_y:.0f} m  [m]")
    ax_hist.set_ylabel("Sample count")
    ax_hist.set_title(f"Crossing-x distribution — {algo_name}",
                      color="white", fontsize=10)
    legend_handles = [
        plt.Line2D([0], [0], color="orange", lw=2.5,        label="Mean traj"),
        plt.Line2D([0], [0], color="lime",   lw=2.5, ls="--", label="Step best"),
        plt.Line2D([0], [0], color="gold",   lw=2.5, ls=":",  label="Global best"),
    ]
    ax_hist.legend(handles=legend_handles, loc="upper right", fontsize=8,
                   facecolor="#222222", labelcolor="white", framealpha=0.8)
    bar_rects = ax_hist.bar(bin_centers, np.zeros(len(bin_centers)),
                             width=bin_width * 0.88, color=bar_colors,
                             alpha=0.80, zorder=3)

    cmap_traj  = plt.cm.coolwarm
    _accept_rates = diag.get("accept_rates", [])

    def _update(frame: int):
        step = frame
        trajs = diag["sample_trajectories"][step]
        n     = len(trajs)
        sample_lc.set_segments([trajs[i, :, :2] for i in range(n)])
        sample_lc.set_color([cmap_traj(i / max(n - 1, 1)) for i in range(n)])

        mt_idx = min(step + 1, len(diag["mean_trajectories"]) - 1)
        mt = diag["mean_trajectories"][mt_idx]
        mean_outline_map.set_data(mt[:, 0], mt[:, 1])
        mean_line_map.set_data(mt[:, 0], mt[:, 1])

        bt = trajs[0]
        best_outline_map.set_data(bt[:, 0], bt[:, 1])
        best_line_map.set_data(bt[:, 0], bt[:, 1])

        gt = global_min_traj_per_step[step]
        gmin_outline_map.set_data(gt[:, 0], gt[:, 1])
        gmin_line_map.set_data(gt[:, 0], gt[:, 1])

        com  = diag["cost_of_mean"][step] if step < len(diag["cost_of_mean"]) else float("nan")
        cmin = diag["cost_mins"][step]
        n_reached = int(np.sum(~np.isnan(crossing_xs_per_step[step])))
        _ar = f"\naccept: {_accept_rates[step]*100:.0f}%" if step < len(_accept_rates) else ""
        step_text.set_text(
            f"Step {step+1}/{n_steps}\n"
            f"cost(mean)={com:,.0f}\n"
            f"cost(best)={cmin:,.0f}\n"
            f"reached level: {n_reached}/{n}"
            + _ar
        )

        cxs    = crossing_xs_per_step[step]
        valid  = cxs[~np.isnan(cxs)]
        counts = np.histogram(valid, bins=bins)[0] if len(valid) else np.zeros(len(bin_centers))
        for rect, h in zip(bar_rects, counts):
            rect.set_height(h)

        mcx = mean_crossing_per_step[step]
        mean_vline.set_xdata([mcx, mcx] if not np.isnan(mcx) else [np.nan, np.nan])
        bcx = _crossing_x_at_y(bt, sc.crossing_y)
        best_vline.set_xdata([bcx, bcx] if not np.isnan(bcx) else [np.nan, np.nan])
        gcx = _crossing_x_at_y(gt, sc.crossing_y)
        gmin_vline.set_xdata([gcx, gcx] if not np.isnan(gcx) else [np.nan, np.nan])

        fig.suptitle(
            f"{sc.name}  [{algo_name}] — step {step+1}/{n_steps}",
            color="white", fontsize=12, y=0.995,
        )
        return (sample_lc, mean_outline_map, mean_line_map,
                best_outline_map, best_line_map,
                gmin_outline_map, gmin_line_map,
                mean_vline, best_vline, gmin_vline, step_text, *bar_rects)

    ani = animation.FuncAnimation(fig, _update, frames=n_steps, interval=200, blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer, out_path = _animation_writer(out_path, fps=5)
    ani.save(out_path, writer=writer, dpi=100, savefig_kwargs={"facecolor": "k"})
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ── EB-MBD barrier movie ─────────────────────────────────────────────────────────

def generate_eb_barrier_movie(diag: dict, out_path: str) -> None:
    if "barrier_offsets" not in diag or not diag["barrier_offsets"]:
        print("  No EB-MBD barrier data — skipping.", flush=True)
        return
    n_steps     = len(diag["barrier_offsets"])
    offsets     = np.array(diag["barrier_offsets"])
    alive_fracs = np.array(diag["alive_fracs"])
    g_vals_list = diag["constraint_values"]
    step_xs     = np.arange(n_steps)
    g_p10 = np.array([np.percentile(g, 10) for g in g_vals_list])
    g_p25 = np.array([np.percentile(g, 25) for g in g_vals_list])
    g_p50 = np.array([np.percentile(g, 50) for g in g_vals_list])
    g_p75 = np.array([np.percentile(g, 75) for g in g_vals_list])
    g_p90 = np.array([np.percentile(g, 90) for g in g_vals_list])
    g_min = np.array([np.min(g)             for g in g_vals_list])
    g_max = np.array([np.max(g)             for g in g_vals_list])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="k")
    ax_c, ax_alive, ax_g = axes
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")

    ax_c.plot(step_xs, offsets, color="cyan", lw=2)
    ax_c.set_xlabel("Denoising step"); ax_c.set_ylabel("Barrier offset  c_s  [m]")
    ax_c.set_title("Barrier offset  c_s"); ax_c.set_xlim(-0.5, n_steps - 0.5)
    ax_c.set_ylim(bottom=0)
    vline_c = ax_c.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)

    ax_alive.plot(step_xs, alive_fracs * 100, color="limegreen", lw=2)
    ax_alive.axhline(100, color="#444444", lw=1, ls=":")
    ax_alive.axhline(50,  color="#444444", lw=1, ls=":")
    ax_alive.set_xlabel("Denoising step"); ax_alive.set_ylabel("Alive samples [%]")
    ax_alive.set_title("Fraction with g(x)+c_s > 0")
    ax_alive.set_xlim(-0.5, n_steps - 0.5); ax_alive.set_ylim(0, 105)
    vline_alive = ax_alive.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)

    ax_g.fill_between(step_xs, g_min,  g_p10,  color="tomato",    alpha=0.15)
    ax_g.fill_between(step_xs, g_p10,  g_p25,  color="gold",      alpha=0.20)
    ax_g.fill_between(step_xs, g_p25,  g_p75,  color="limegreen", alpha=0.30)
    ax_g.fill_between(step_xs, g_p75,  g_p90,  color="gold",      alpha=0.20)
    ax_g.fill_between(step_xs, g_p90,  g_max,  color="tomato",    alpha=0.15)
    ax_g.plot(step_xs, g_p50, color="white", lw=1.5, label="median g(x)")
    ax_g.plot(step_xs, -offsets, color="cyan", lw=1.5, ls="--", label="-c_s")
    ax_g.axhline(0, color="#888888", lw=1, ls=":")
    ax_g.set_xlabel("Denoising step"); ax_g.set_ylabel("g(x) [m]")
    ax_g.set_title("Constraint distribution"); ax_g.set_xlim(-0.5, n_steps - 0.5)
    ax_g.legend(fontsize=7, labelcolor="white", facecolor="#1a1a1a", framealpha=0.6)
    vline_g = ax_g.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)

    fig.suptitle("EB-MBD barrier schedule diagnostics", color="white", fontsize=13)

    def _upd(frame):
        vline_c.set_xdata([frame, frame])
        vline_alive.set_xdata([frame, frame])
        vline_g.set_xdata([frame, frame])
        return vline_c, vline_alive, vline_g

    ani = animation.FuncAnimation(fig, _upd, frames=n_steps, interval=200, blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer, out_path = _animation_writer(out_path, fps=5)
    ani.save(out_path, writer=writer, dpi=100, savefig_kwargs={"facecolor": "k"})
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ── AL-MBD multiplier plot ───────────────────────────────────────────────────────

def generate_al_multiplier_plot(diag: dict, out_path: str) -> None:
    mul_means  = diag.get("multiplier_means", [])
    mul_maxes  = diag.get("multiplier_maxes", [])
    violations = diag.get("constraint_violations", [])
    if not mul_means:
        print("  No AL multiplier data — skipping.", flush=True)
        return
    steps = np.arange(len(mul_means))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(steps, mul_means, label="Mean multiplier", color="steelblue")
    axes[0].plot(steps, mul_maxes, label="Max multiplier",  color="tomato", ls="--")
    axes[0].set_ylabel("Multiplier M")
    axes[0].set_title("AL-MBD: Multiplier evolution across diffusion steps")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, violations, color="darkorange", label="Mean constraint violation")
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_ylabel("Violation (m, mean)")
    axes[1].set_xlabel("Diffusion step (N → 1)")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def generate_homotopy_class_plot(diag: dict, out_path: str) -> None:
    """Plot per-class sample counts and mean costs across diffusion steps."""
    class_counts = diag.get("class_counts", [])
    class_costs  = diag.get("class_costs", [])
    best_class   = diag.get("best_class", [])
    n_classes    = diag.get("n_classes", 1)
    if not class_counts:
        return

    counts_arr = np.array(class_counts)   # (steps, n_classes)
    costs_arr  = np.array(class_costs)    # (steps, n_classes)
    steps      = np.arange(len(class_counts))

    cmap   = plt.cm.get_cmap("tab10", n_classes)
    colors = [cmap(c) for c in range(n_classes)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for c in range(n_classes):
        axes[0].plot(steps, counts_arr[:, c], color=colors[c], label=f"cls {c}")
    axes[0].set_ylabel("Sample count")
    axes[0].set_title(f"H-MBD: per-class sample counts ({n_classes} classes)")
    axes[0].legend(fontsize=7, ncol=min(n_classes, 4)); axes[0].grid(True, alpha=0.3)

    finite_costs = np.where(np.isfinite(costs_arr), costs_arr, np.nan)
    for c in range(n_classes):
        axes[1].plot(steps, finite_costs[:, c], color=colors[c], label=f"cls {c}")
    # Mark which class was leading at each step
    if best_class:
        bc_arr = np.array(best_class)
        best_costs = np.array([finite_costs[s, bc_arr[s]] for s in steps])
        axes[1].plot(steps, best_costs, "k--", lw=1.5, label="best class", zorder=5)
    axes[1].set_ylabel("Mean sample cost")
    axes[1].set_xlabel("Diffusion step (N → 1)")
    axes[1].legend(fontsize=7, ncol=min(n_classes, 4)); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ── Reject-sampling filter (same for all scenarios) ──────────────────────────────

def _make_reject_filter(cost_map, run_cfg):
    _x_lo   = float(cost_map.x_origin)
    _x_hi   = float(cost_map.x_origin) + float(cost_map.cols) * float(cost_map.resolution)
    _half_l = run_cfg.ship_length / 2.0
    _half_w = run_cfg.ship_beam   / 2.0
    _dx = np.array([0.0, _half_l,  _half_l, -_half_l, -_half_l], dtype=np.float32)
    _dy = np.array([0.0, _half_w, -_half_w,  _half_w, -_half_w], dtype=np.float32)

    def _filter(Y0s_norm: np.ndarray, trajs: np.ndarray) -> np.ndarray:
        xs  = trajs[:, :, 0]
        ys  = trajs[:, :, 1]
        psi = trajs[:, :, 2]
        c, s = np.cos(psi), np.sin(psi)
        hull_x = (xs[:, :, None]
                  + c[:, :, None] * _dx[None, None, :]
                  - s[:, :, None] * _dy[None, None, :])
        oob = ((hull_x < _x_lo) | (hull_x > _x_hi)).any(axis=(1, 2))
        oob |= (ys < Y_MIN).any(axis=1)
        reaches_goal = ys[:, -1] >= GOAL_LINE
        return (~oob) & reaches_goal
    return _filter


# ── Single-method run ────────────────────────────────────────────────────────────

def run_one_method(
    method: str,
    sc: ScenarioCfg,
    cost_map,
    geo_info: dict,
    run_cfg,
    args,
    out_dir: Path,
) -> dict:
    """
    Run one planning method on the given scenario and produce all outputs.
    Returns a metrics dict.

    method: "mbd" | "eb_mbd" | "eb_rrt_mbd" | "eb_astar_mbd" | "reject_mbd" |
            "al_mbd" | "sa_mbd" | "homotopy_mbd" | "eb_homotopy_mbd" | "al_homotopy_mbd"
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Method flags
    use_eb          = method == "eb_mbd"
    use_eb_rrt      = method == "eb_rrt_mbd"
    use_eb_astar    = method == "eb_astar_mbd"
    use_al          = method in ("al_mbd", "al_homotopy_mbd")
    use_rej         = method == "reject_mbd"
    use_sa          = method == "sa_mbd"
    use_homotopy    = method == "homotopy_mbd"
    use_eb_homotopy = method == "eb_homotopy_mbd"
    use_al_homotopy = method == "al_homotopy_mbd"
    _algo_map = {
        "mbd":             "MBD",
        "eb_mbd":          "EB-MBD",
        "eb_rrt_mbd":      "EB-RRT-MBD",
        "eb_astar_mbd":    "EB-A*-MBD",
        "reject_mbd":      "Reject-MBD",
        "sa_mbd":          "SA-MBD",
        "homotopy_mbd":    "H-MBD",
        "eb_homotopy_mbd": "EB-H-MBD",
        "al_homotopy_mbd": "AL-H-MBD",
    }
    algo_name = _algo_map.get(method, "AL-MBD")

    sample_filter_fn = _make_reject_filter(cost_map, run_cfg) if use_rej else None

    # Geometric warm start: RRT or A* path for warm-started methods
    rrt_lattice_path = None
    if use_eb_rrt:
        print(f"  RRT: searching for feasible path ...", flush=True)
        _rrt_t0 = time.perf_counter()
        rrt_lattice_path = rrt_path_on_costmap(
            cost_map,
            start_xy=(sc.start[0], sc.start[1]),
            goal_xy=sc.goal,
            max_iters=8000,
            step_size=20.0,
            goal_bias=0.15,
            cost_threshold_frac=0.1,
            smooth_iters=300,
            seed=42,
        )
        _rrt_dt = time.perf_counter() - _rrt_t0
        if rrt_lattice_path is not None:
            print(f"  RRT: found path ({len(rrt_lattice_path)} waypoints, {_rrt_dt:.2f}s)",
                  flush=True)
        else:
            print(f"  RRT: no path found ({_rrt_dt:.2f}s) — using cold start", flush=True)

    if use_eb_astar:
        print(f"  A*: searching for feasible path ...", flush=True)
        _astar_t0 = time.perf_counter()
        rrt_lattice_path = astar_path_on_costmap(
            cost_map,
            start_xy=(sc.start[0], sc.start[1]),
            goal_xy=sc.goal,
            cost_threshold_frac=0.1,
            smooth_iters=300,
            seed=42,
        )
        _astar_dt = time.perf_counter() - _astar_t0
        if rrt_lattice_path is not None:
            print(f"  A*: found path ({len(rrt_lattice_path)} waypoints, {_astar_dt:.2f}s)",
                  flush=True)
        else:
            print(f"  A*: no path found ({_astar_dt:.2f}s) — using cold start", flush=True)

    print(f"\n[{sc.name}] Running {algo_name} ...", flush=True)
    _t0 = time.perf_counter()
    traj, diag = jax_mbd_optimize(
        cost_map=cost_map,
        start_pose=sc.start,
        goal=sc.goal,
        run_cfg=run_cfg,
        start_nu=(2.0, 0.0, 0.0),
        lattice_path=rrt_lattice_path,
        warm_start=(use_eb_rrt or use_eb_astar) and rrt_lattice_path is not None,
        return_diagnostics=True,
        goal_y_line=GOAL_LINE,
        goal_y_line_penalty=1e6,
        denoising_movie_dir=str(out_dir / "denoising"),
        cost_truncate_y=Y_MAX,
        sample_filter_fn=sample_filter_fn,
        max_resample_iters=None,
        # EB-MBD (also active for EB-H-MBD, EB-RRT-MBD, EB-A*-MBD)
        use_eb_mbd=use_eb or use_eb_homotopy or use_eb_rrt or use_eb_astar,
        eb_x_min=X_MIN, eb_x_max=X_MAX, eb_y_min=Y_MIN,
        eb_c_max=args.eb_c_max, eb_mu=args.eb_mu, eb_kappa=args.eb_kappa,
        eb_oob_cost=args.eb_oob_cost,
        # AL-MBD
        use_al_mbd=use_al,
        al_x_min=X_MIN, al_x_max=X_MAX, al_y_min=Y_MIN,
        al_rho=args.al_rho, al_M_max=args.al_M_max,
        # SA-MBD
        use_sa_mbd=use_sa,
        sa_lambda=args.sa_lambda,
        # Homotopy-MBD (also active for EB-H-MBD and AL-H-MBD)
        use_homotopy_mbd=use_homotopy or use_eb_homotopy or use_al_homotopy,
        homotopy_n_obs_max=args.homotopy_n_obs_max,
        # Hard rectangular obstacles (EB/AL only)
        obstacle_rects=sc.obstacle_rects,
    )
    planning_time = time.perf_counter() - _t0

    # Extract trajectories
    mean_traj = np.array(diag["mean_trajectories"][-1])
    mean_cost = diag["cost_of_mean"][-1] if diag["cost_of_mean"] else float("nan")

    if (use_eb or use_eb_rrt or use_eb_astar or use_al or use_homotopy or use_eb_homotopy or use_al_homotopy) and diag.get("best_base_cost_trajectory") is not None:
        best_cost = diag["best_base_cost"]
        best_traj = diag["best_base_cost_trajectory"]
    else:
        cost_mins_arr = np.array(diag["cost_mins"])
        global_min_step = int(cost_mins_arr.argmin())
        best_cost = float(cost_mins_arr[global_min_step])
        best_traj = diag["sample_trajectories"][global_min_step][0]

    # Metrics
    path_length = float(np.sum(np.linalg.norm(np.diff(mean_traj[:, :2], axis=0), axis=1)))
    traj_cost   = float(sum(cost_map.get_cost_world(float(x), float(y))
                            for x, y in mean_traj[:, :2]))
    _ys       = mean_traj[:, 1]
    _xs       = mean_traj[:, 0]
    _cross    = int(np.argmax(_ys >= GOAL_LINE)) if np.any(_ys >= GOAL_LINE) else -1
    x_at_goal = float(mean_traj[_cross, 0]) if _cross >= 0 else float("nan")
    # OOB check: any waypoint outside [X_MIN, X_MAX] × [Y_MIN, ∞)
    _oob_xs   = (_xs < X_MIN) | (_xs > X_MAX)
    _oob_ys   = (_ys < Y_MIN)
    traj_oob  = bool(np.any(_oob_xs | _oob_ys))
    # Goal reached only if crossing is in-bounds
    _goal_x_ok = _cross >= 0 and X_MIN <= x_at_goal <= X_MAX
    reaches_goal_clean = _cross >= 0 and _goal_x_ok and not traj_oob

    metrics = {
        "method":        algo_name,
        "planning_time": planning_time,
        "traj_cost":     traj_cost,
        "path_length":   path_length,
        "reaches_goal":  reaches_goal_clean,
        "traj_oob":      traj_oob,
        "x_at_goal":     x_at_goal,
        "goal_x":        sc.goal[0],
    }

    # Console table
    print(f"\n{'─'*58}", flush=True)
    print(f"  PLANNING METRICS  ({algo_name})  —  {sc.name}", flush=True)
    print(f"{'─'*58}", flush=True)
    print(f"  Planning time  : {planning_time:.1f} s", flush=True)
    print(f"  Traj cost      : {traj_cost:.1f}  (raw costmap)", flush=True)
    print(f"  Path length    : {path_length:.1f} m", flush=True)
    print(f"  Reaches goal   : {'yes' if reaches_goal_clean else 'NO'}"
          f"{' (OOB!)' if _cross >= 0 and not reaches_goal_clean else ''}", flush=True)
    if _cross >= 0:
        print(f"  x at goal      : {x_at_goal:.1f} m  (goal x = {sc.goal[0]:.0f} m)", flush=True)
    print(f"  OOB violation  : {'YES' if traj_oob else 'no'}", flush=True)
    print(f"{'─'*58}\n", flush=True)

    # Plots and movies
    plot_costmap(cost_map, geo_info, sc, str(out_dir / "costmap.png"))
    plot_result(cost_map, geo_info, sc, mean_traj, best_traj,
                mean_cost, best_cost, algo_name, str(out_dir / "result.png"),
                rrt_path=rrt_lattice_path,
                warmstart_traj=diag.get("warmstart_trajectory") if diag else None)

    print(f"  Generating trajectory fan movie...", flush=True)
    generate_trajectory_fan_movie(
        diag, cost_map, geo_info, sc, algo_name,
        str(out_dir / "trajectory_fan.mp4"),
    )

    if use_eb or use_eb_rrt or use_eb_astar or use_eb_homotopy:
        print("  Generating EB barrier movie...", flush=True)
        generate_eb_barrier_movie(diag, str(out_dir / "eb_barrier_schedule.mp4"))

    if use_al:
        print("  Generating AL-MBD multiplier plot...", flush=True)
        generate_al_multiplier_plot(diag, str(out_dir / "al_multiplier_evolution.png"))
        m_means = np.array(diag.get("multiplier_means", []))
        m_maxes = np.array(diag.get("multiplier_maxes", []))
        cviol   = np.array(diag.get("constraint_violations", []))
        if len(m_means):
            print(f"  AL summary: mult_mean {m_means[0]:.4f}→{m_means[-1]:.4f}"
                  f"  mult_max {m_maxes[0]:.4f}→{m_maxes[-1]:.4f}"
                  f"  c_viol {cviol[0]:.3f}→{cviol[-1]:.3f}", flush=True)

    if (use_homotopy or use_eb_homotopy or use_al_homotopy) and diag.get("class_counts"):
        print("  Generating homotopy class evolution plot...", flush=True)
        generate_homotopy_class_plot(diag, str(out_dir / "homotopy_class_evolution.png"))

    print(f"  Done. Outputs in {out_dir}", flush=True)
    return metrics


# ── Comparison table ─────────────────────────────────────────────────────────────

def print_comparison_table(all_metrics: list[dict], sc: ScenarioCfg,
                            out_path: Optional[str] = None) -> None:
    lines = [
        "",
        f"{'='*70}",
        f"  COMPARISON TABLE — {sc.name}",
        f"  {sc.description}",
        f"{'='*70}",
        f"  {'Method':<14}  {'Time (s)':>10}  {'Cost':>12}  {'Length (m)':>12}  "
        f"{'Goal?':>6}  {'OOB?':>5}  {'x@goal (m)':>12}",
        f"  {'─'*14}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*6}  {'─'*5}  {'─'*12}",
    ]
    for m in all_metrics:
        g   = "yes" if m["reaches_goal"] else "NO"
        oob = "YES" if m.get("traj_oob", False) else "no"
        x   = f"{m['x_at_goal']:.1f}" if m.get("x_at_goal") and not np.isnan(m["x_at_goal"]) else "—"
        lines.append(
            f"  {m['method']:<14}  {m['planning_time']:>10.1f}  "
            f"{m['traj_cost']:>12.1f}  {m['path_length']:>12.1f}  "
            f"{g:>6}  {oob:>5}  {x:>12}"
        )
    lines.append(f"{'='*70}")
    lines.append(f"  Goal x = {sc.goal[0]:.0f} m")
    lines.append(f"{'='*70}")
    lines.append("")
    text = "\n".join(lines)
    print(text, flush=True)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text)
        print(f"  Comparison table saved: {out_path}", flush=True)


# ── Lattice A* planner ───────────────────────────────────────────────────────────

def _to_dpdem_costmap(jax_cm: JaxCostMapResult) -> DpdemCostMap:
    """
    Wrap a JaxCostMapResult into the dpdem CostMap used by LatticePlanner.

    Adds 100 m of extra open-water padding at the south edge so the ship's
    stern (half-length ~79 m below center) stays inside the map when the
    ship starts near y=50.  Without this, every primitive's corner check
    returns inf and A* generates 0 successors.
    """
    extra_m  = 100.0
    extra_r  = int(np.ceil(extra_m / jax_cm.resolution))
    pad      = np.full((extra_r, jax_cm.cols), max(float(jax_cm.base_cost), 0.1),
                       dtype=np.float64)
    padded   = np.vstack([pad, jax_cm.cost_map.astype(np.float64)])

    new_y_origin = jax_cm.y_origin - extra_r * jax_cm.resolution
    new_height   = jax_cm.height   + extra_r * jax_cm.resolution
    new_rows     = jax_cm.rows     + extra_r

    cm = DpdemCostMap(
        width=jax_cm.width,
        height=new_height,
        resolution=jax_cm.resolution,
        x_origin=jax_cm.x_origin,
        y_origin=new_y_origin,
        base_cost=float(jax_cm.base_cost),
    )
    cm.cost_map = padded
    cm.rows     = new_rows
    return cm


def generate_lattice_search_movie(
    search_history: list,
    final_path: Optional[np.ndarray],
    cost_map: JaxCostMapResult,
    geo_info: dict,
    sc: ScenarioCfg,
    out_path: str,
    steps_per_frame: int = 3,
    fps: int = 8,
) -> None:
    """
    Animate the A* lattice search expansion.

    Each frame accumulates ``steps_per_frame`` expansion steps.  Valid arcs
    (added to the open set) are drawn in cyan; rejected arcs (OOB /
    impassable / dominated) in dim-gray; the node being expanded is marked
    with an orange dot.  The final path is overlaid in the last frame.
    """
    n_steps = len(search_history)
    if n_steps == 0:
        print("  [search movie] no search history — skipping.", flush=True)
        return

    # Pre-accumulate line segments so we can draw with LineCollection
    # Each entry: (segments_Nx2x2, color, alpha, lw, zorder)
    valid_segs:   List[np.ndarray] = []   # (N, 2, 2) per arc
    invalid_segs: List[np.ndarray] = []

    # Build per-step lists so we can reveal them frame by frame
    step_valid:   List[List[np.ndarray]] = []  # list over steps, each a list of arcs
    step_invalid: List[List[np.ndarray]] = []
    step_nodes:   List[Tuple[float, float]] = []

    for (expanded_xyh, g, h, f, valid_edges, invalid_edges) in search_history:
        step_nodes.append((expanded_xyh[0], expanded_xyh[1]))
        v_arcs, iv_arcs = [], []
        for arc, _end, _cost in valid_edges:
            pts = arc[:, :2]
            segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (K-1, 2, 2)
            v_arcs.append(segs)
        for arc, _end, _reason in invalid_edges:
            pts = arc[:, :2]
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            iv_arcs.append(segs)
        step_valid.append(v_arcs)
        step_invalid.append(iv_arcs)

    n_frames = math.ceil(n_steps / steps_per_frame) + 1  # +1 for final-path frame

    fig, ax = plt.subplots(figsize=(6, 16))
    _draw_costmap_background(ax, cost_map, geo_info)
    ax.plot(*sc.start[:2], "go", ms=10, zorder=9, label="Start")
    ax.set_title(f"{sc.name}  [Lattice A* search]", fontsize=9)

    # Persistent artists we update each frame
    valid_lc   = LineCollection([], colors="cyan",    linewidths=1.2, alpha=0.6, zorder=5)
    invalid_lc = LineCollection([], colors="dimgray", linewidths=0.7, alpha=0.35, zorder=4)
    node_dot,  = ax.plot([], [], "o", color="orange", ms=8, zorder=10)
    path_line, = ax.plot([], [], "-", color="deepskyblue", lw=3, zorder=11)
    path_line_w,= ax.plot([], [], "w-", lw=5, zorder=10)
    info_txt   = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                         fontsize=8, va="top", ha="left",
                         color="white",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.55))

    ax.add_collection(invalid_lc)
    ax.add_collection(valid_lc)

    all_valid_segs:   List[np.ndarray] = []
    all_invalid_segs: List[np.ndarray] = []

    def _update(frame_idx: int):
        if frame_idx < n_frames - 1:
            # Reveal steps_per_frame expansion steps
            lo = frame_idx * steps_per_frame
            hi = min(lo + steps_per_frame, n_steps)
            for si in range(lo, hi):
                all_invalid_segs.extend(step_invalid[si])
                all_valid_segs.extend(step_valid[si])
            # Node dot = last expanded in this batch
            nx, ny = step_nodes[hi - 1]
            node_dot.set_data([nx], [ny])

            # Flatten segments for LineCollection
            if all_valid_segs:
                valid_lc.set_segments(np.vstack(all_valid_segs))
            if all_invalid_segs:
                invalid_lc.set_segments(np.vstack(all_invalid_segs))

            expanded = hi
            info_txt.set_text(
                f"Step {expanded}/{n_steps}\n"
                f"Arcs shown: {sum(len(v) for v in step_valid[:hi])}"
            )
        else:
            # Final frame: show complete path
            node_dot.set_data([], [])
            if final_path is not None:
                path_line_w.set_data(final_path[:, 0], final_path[:, 1])
                path_line.set_data(final_path[:, 0], final_path[:, 1])
            info_txt.set_text(f"Done  ({n_steps} nodes expanded)\nFinal path shown")

        return valid_lc, invalid_lc, node_dot, path_line, path_line_w, info_txt

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000 // fps, blit=False
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer, out_path = _animation_writer(out_path, fps=fps)
    ani.save(out_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def plot_lattice_result(cost_map, geo_info, sc: ScenarioCfg,
                        path: np.ndarray, path_cost: float,
                        path_length: float, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 16))
    _draw_costmap_background(ax, cost_map, geo_info)
    ax.plot(path[:, 0], path[:, 1], "w-",  lw=4, zorder=4)
    ax.plot(path[:, 0], path[:, 1], "-", color="deepskyblue", lw=2.5,
            zorder=5, label=f"A* path  (cost={path_cost:.0f}  len={path_length:.0f} m)")
    ax.plot(*sc.start[:2], "go", ms=10, zorder=6, label="Start")
    ax.plot(*sc.goal,      "m*", ms=14, zorder=6, label="Goal")
    ax.set_title(f"{sc.name}  [Lattice A*]\ncost={path_cost:.0f}  length={path_length:.0f} m",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def run_lattice_method(
    sc: ScenarioCfg,
    cost_map: JaxCostMapResult,
    geo_info: dict,
    run_cfg,
    out_dir: Path,
) -> dict:
    """Run A* lattice planner on the scenario and produce a result plot."""
    out_dir.mkdir(parents=True, exist_ok=True)

    dpdem_cm = _to_dpdem_costmap(cost_map)
    bounds   = {"x_origin": cost_map.x_origin, "y_origin": cost_map.y_origin,
                "width": cost_map.width, "height": cost_map.height}

    # Use tight goal tolerance so the path actually reaches y=GOAL_LINE.
    # Default 150 m stops at y=1200, which our metric (y >= 1350) rejects.
    # Use reduced turning radius: the real ship value (350 m) produces primitives
    # of 275–1100 m — too coarse for an 800 × 1400 m toy map.  100 m gives
    # 79–314 m steps, which is fine-grained enough to navigate 80–200 m corridors.
    run_cfg_lat = replace(run_cfg,
                          planning_goal_tolerance=10.0,
                          planning_turning_radius=100.0)

    print(f"\n[{sc.name}] Running Lattice A* ...", flush=True)
    _t0 = time.perf_counter()
    try:
        lattice_planner = create_medea_lattice_planner(
            dpdem_cm, bounds, sc.start, sc.goal, run_cfg_lat,
            goal_y_line=GOAL_LINE,
        )
        result = plan_path(lattice_planner, sc.start, sc.goal, run_cfg_lat,
                           cost_map=dpdem_cm, return_search_history=True)
        path, search_history = result if result[0] is not None or isinstance(result, tuple) else (result, [])
    except Exception as e:
        print(f"  Lattice planner error: {e}", flush=True)
        path, search_history = None, []
    planning_time = time.perf_counter() - _t0

    if path is None:
        print("  Lattice planner returned no path.", flush=True)
        metrics = {
            "method": "Lattice A*", "planning_time": planning_time,
            "traj_cost": float("nan"), "path_length": float("nan"),
            "reaches_goal": False, "x_at_goal": float("nan"),
            "goal_x": sc.goal[0],
        }
        return metrics

    path = np.asarray(path)

    # Metrics — use JaxCostMapResult.get_cost_world to be consistent with MBD
    path_length = float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
    traj_cost   = float(sum(cost_map.get_cost_world(float(x), float(y))
                            for x, y in path[:, :2]))
    _ys = path[:, 1]
    # Lattice planner stops at GOAL_LINE - goal_tolerance; accept last waypoint
    # within tolerance as reaching the goal.
    _goal_thresh = GOAL_LINE - run_cfg_lat.planning_goal_tolerance - 5.0
    _cross    = int(np.argmax(_ys >= _goal_thresh)) if np.any(_ys >= _goal_thresh) else -1
    x_at_goal = float(path[_cross, 0]) if _cross >= 0 else float("nan")

    metrics = {
        "method": "Lattice A*", "planning_time": planning_time,
        "traj_cost": traj_cost, "path_length": path_length,
        "reaches_goal": _cross >= 0, "x_at_goal": x_at_goal,
        "goal_x": sc.goal[0],
    }

    print(f"\n{'─'*58}", flush=True)
    print(f"  PLANNING METRICS  (Lattice A*)  —  {sc.name}", flush=True)
    print(f"{'─'*58}", flush=True)
    print(f"  Planning time  : {planning_time:.1f} s", flush=True)
    print(f"  Traj cost      : {traj_cost:.1f}  (raw costmap)", flush=True)
    print(f"  Path length    : {path_length:.1f} m", flush=True)
    print(f"  Reaches goal   : {'yes' if _cross >= 0 else 'NO'}", flush=True)
    if _cross >= 0:
        print(f"  x at goal      : {x_at_goal:.1f} m  (goal x = {sc.goal[0]:.0f} m)", flush=True)
    print(f"{'─'*58}\n", flush=True)

    # Result plot + search movie
    plot_costmap(cost_map, geo_info, sc, str(out_dir / "costmap.png"))
    plot_lattice_result(cost_map, geo_info, sc, path, traj_cost, path_length,
                        str(out_dir / "result.png"))
    print("  Generating A* search movie ...", flush=True)
    generate_lattice_search_movie(
        search_history, path, cost_map, geo_info, sc,
        out_path=str(out_dir / "lattice_search.mp4"),
    )

    print(f"  Done. Outputs in {out_dir}", flush=True)
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Four toy MBD scenarios demonstrating method failure modes"
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        required=True,
        help="Scenario to run",
    )
    _method_choices = ["mbd", "eb_mbd", "eb_rrt_mbd", "eb_astar_mbd", "reject_mbd",
                       "al_mbd", "sa_mbd", "homotopy_mbd", "eb_homotopy_mbd",
                       "al_homotopy_mbd", "lattice"]
    parser.add_argument(
        "--method",
        choices=_method_choices,
        nargs="+",
        default=None,
        help="One or more methods to run (incompatible with --run-all-methods)",
    )
    parser.add_argument(
        "--run-all-methods", action="store_true",
        help="Run all methods (lattice + MBD variants including H-MBD) and print a comparison table",
    )
    parser.add_argument("--output-root", default="outputs/scenarios")
    parser.add_argument("--nsample",     type=int,   default=512)
    parser.add_argument("--ndiffuse",    type=int,   default=50)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--horizon",     type=int,   default=1200)
    # EB-MBD
    parser.add_argument("--eb-c-max",  type=float, default=1500.0)
    parser.add_argument("--eb-mu",     type=float, default=50000.0)
    parser.add_argument("--eb-kappa",  type=float, default=1.0)
    # EB/AL OOB cost (linear cost per m of OOB violation)
    parser.add_argument("--eb-oob-cost", type=float, default=5.0)
    # AL-MBD
    parser.add_argument("--al-rho",   type=float, default=50.0)
    parser.add_argument("--al-M-max", type=float, default=1000.0)
    # SA-MBD
    parser.add_argument("--sa-lambda", type=float, default=0.05)
    # H-MBD
    parser.add_argument("--no-inflation", action="store_true",
                        help="Disable ship-beam dilation of obstacles in the costmap")
    parser.add_argument("--homotopy-n-obs-max", type=int, default=3,
                        help="Max obstacles to track for H-signature (2^n classes)")
    args = parser.parse_args()

    if args.method is None and not args.run_all_methods:
        parser.error("Specify --method or --run-all-methods")
    if args.method is not None and args.run_all_methods:
        parser.error("--method and --run-all-methods are mutually exclusive")

    sc      = SCENARIOS[args.scenario]
    out_root = Path(args.output_root) / sc.name

    run_cfg = get_default_run_config()
    run_cfg = replace(
        run_cfg,
        planning_jax_mbd_nsample=args.nsample,
        planning_jax_mbd_ndiffuse=args.ndiffuse,
        planning_jax_mbd_temperature=args.temperature,
        planning_jax_mbd_use_best_sample=False,
        planning_jax_mbd_horizon=args.horizon,
        planning_jax_mbd_n_substeps=1,
        planning_jax_mbd_dt_substep=1.0,
        planning_jax_mbd_goal_x_weight=sc.goal_x_weight,
        planning_jax_mbd_length_weight=0.0,
    )

    inflate = not args.no_inflation
    print(f"\nBuilding costmap for '{sc.name}' "
          f"({'no inflation' if not inflate else 'with ship-beam inflation'}) ...", flush=True)
    cost_map, geo_info = sc.costmap_fn(run_cfg, inflate=inflate)

    all_methods = ["lattice", "mbd", "eb_mbd", "eb_rrt_mbd", "eb_astar_mbd",
                   "reject_mbd", "al_mbd", "sa_mbd", "homotopy_mbd", "eb_homotopy_mbd", "al_homotopy_mbd"]
    methods = all_methods if args.run_all_methods else args.method
    all_metrics: list[dict] = []

    for method in methods:
        method_dir = out_root / method
        if method == "lattice":
            m = run_lattice_method(sc, cost_map, geo_info, run_cfg, method_dir)
        else:
            m = run_one_method(method, sc, cost_map, geo_info, run_cfg, args, method_dir)
        all_metrics.append(m)

    if len(all_metrics) > 1:
        table_path = str(out_root / "comparison_table.txt") if args.run_all_methods else None
        print_comparison_table(all_metrics, sc, table_path)
    elif all_metrics:
        print_comparison_table(all_metrics, sc)

    print(f"\nALL DONE — outputs in {out_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
