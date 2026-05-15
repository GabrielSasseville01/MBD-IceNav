#!/usr/bin/env python3
"""
Minimal example demonstrating MBD multimodal collapse.

Scenario
--------
  - Map: 400 m × 1400 m  (ship travels north, +Y direction)
  - Ship starts at (200, 50, π/2) heading north, goal line at y=1350
  - A large symmetric ice floe (160 m wide, 160 m tall) sits centered on
    the straight-line path, leaving two equivalent corridors of equal
    width (120 m) to the left and right.
  - The two optimal paths (go left / go right) are mirror images and have
    equal cost.  Their action-space average goes straight through the floe.
  - Boundary: left (x<0), right (x>400), and bottom (y<0) are OOB;
    top is open (no OOB penalty above the goal line).

Outputs (default: outputs/toy_mbd_bimodal/)
-------------------------------------------
  bimodal_costmap.png          — costmap + the two ideal bypass routes
  denoising/replan_0000_denoising.mp4   — MBD denoising movie (green = running global best)
  bimodal_result.png           — final mean trajectory vs global best sample

Run:
  uv run python examples/toy_mbd_bimodal.py
  uv run python examples/toy_mbd_bimodal.py --use-best-sample
  uv run python examples/toy_mbd_bimodal.py --nsample 256 --ndiffuse 30
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from dataclasses import replace

from medea_ice_ship.config import get_default_run_config
from medea_ice_ship.jax_costmap import build_costmap_jax
from medea_ice_ship.jax_mbd_planner import jax_mbd_optimize


# ── Scene geometry ─────────────────────────────────────────────────────────────

X_MIN, X_MAX = 0.0,    800.0
Y_MIN, Y_MAX = 0.0,   1400.0
X_MID = (X_MIN + X_MAX) / 2.0   # 400.0

# Start in the middle, heading north (π/2 rad); goal line at y=1350
START     = (X_MID, 50.0, math.pi / 2)
GOAL      = (X_MID, 1350.0)     # used for soft goal-distance cost
GOAL_LINE = 1350.0               # hard constraint: must reach y ≥ this

# Central ice floe: 160 m wide (X) × 160 m tall (Y), centred on the straight
# path.  Leaves two symmetric 320 m corridors (left and right).
FLOE_X_LO, FLOE_X_HI = 320.0, 480.0
FLOE_Y_LO, FLOE_Y_HI = 560.0, 720.0

FLOE_THICKNESS = 2.0   # m

# Represent the floe as a single disk element in the KE costmap model.
_FLOE_CX = (FLOE_X_LO + FLOE_X_HI) / 2.0
_FLOE_CY = (FLOE_Y_LO + FLOE_Y_HI) / 2.0
_FLOE_W  = FLOE_X_HI - FLOE_X_LO
_FLOE_H  = FLOE_Y_HI - FLOE_Y_LO
_FLOE_R  = 0.5 * np.sqrt(_FLOE_W**2 + _FLOE_H**2)


# ── Costmap ────────────────────────────────────────────────────────────────────

def _build_costmap(run_cfg):
    centroids   = np.array([[_FLOE_CX, _FLOE_CY]])
    thicknesses = np.array([FLOE_THICKNESS])
    radii       = np.array([_FLOE_R])

    cost_map, _ = build_costmap_jax(
        centroids, thicknesses, radii,
        X_MIN, X_MAX, Y_MIN, Y_MAX,
        resolution=run_cfg.planning_resolution,
        margin=20.0,
        base_cost=0.0,          # open water = no cost (toy demo: only floe has cost)
        ship_length=run_cfg.ship_length,
        ship_beam=run_cfg.ship_beam,
        ship_mass=run_cfg.ship_displacement,
        obs_cost_buffer_factor=0.05,
        verbose=True,
    )

    return cost_map


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _costmap_background(ax, cost_map):
    grid = cost_map.cost_map
    extent = [
        cost_map.x_origin, cost_map.x_origin + cost_map.width,
        cost_map.y_origin, cost_map.y_origin + cost_map.height,
    ]
    vmax = float(np.percentile(grid[grid > 0], 95)) if np.any(grid > 0) else 1.0
    ax.imshow(grid, origin="lower", extent=extent, cmap="YlOrRd",
              alpha=0.65, vmin=0, vmax=vmax)
    rect = mpatches.FancyBboxPatch(
        (FLOE_X_LO, FLOE_Y_LO), _FLOE_W, _FLOE_H,
        linewidth=2, edgecolor="navy", facecolor="steelblue", alpha=0.2,
        boxstyle="square,pad=0",
    )
    ax.add_patch(rect)

    # OOB shading: left, right, bottom walls
    oob_alpha = 0.25
    wall_w = 40.0
    for patch_xy, patch_w, patch_h in [
        ((X_MIN - wall_w, Y_MIN), wall_w, Y_MAX - Y_MIN + wall_w),  # left wall
        ((X_MAX,          Y_MIN), wall_w, Y_MAX - Y_MIN + wall_w),  # right wall
        ((X_MIN - wall_w, Y_MIN - wall_w), X_MAX - X_MIN + 2 * wall_w, wall_w),  # bottom
    ]:
        ax.add_patch(mpatches.Rectangle(patch_xy, patch_w, patch_h,
                                        facecolor="black", alpha=oob_alpha, zorder=2))

    # Goal line
    ax.axhline(GOAL_LINE, color="magenta", lw=2, ls="--", alpha=0.8, zorder=3,
               label=f"Goal line  y={GOAL_LINE:.0f} m")

    ax.set_xlim(X_MIN - wall_w - 5, X_MAX + wall_w + 5)
    ax.set_ylim(Y_MIN - wall_w - 5, Y_MAX + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")


def plot_costmap(cost_map, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 16))
    _costmap_background(ax, cost_map)

    # Ideal bypass routes (schematic)
    bypass_x_lo = (X_MIN + FLOE_X_LO) / 2.0
    bypass_x_hi = (X_MAX + FLOE_X_HI) / 2.0
    for bx, label, col in [
        (bypass_x_lo, "Left bypass  (optimal)",  "limegreen"),
        (bypass_x_hi, "Right bypass (optimal)", "deepskyblue"),
    ]:
        ys = [START[1], _FLOE_CY - _FLOE_H * 0.7, _FLOE_CY, _FLOE_CY + _FLOE_H * 0.7, GOAL[1]]
        xs = [START[0], bx, bx, bx, GOAL[0]]
        ax.plot(xs, ys, "--", color=col, lw=2.5, label=label)

    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], "--", color="red",
            lw=1.5, alpha=0.7, label="Action-space mean (through floe)")

    ax.plot(*START[:2], "go", ms=10, zorder=6, label="Start")
    ax.plot(*GOAL,      "m*", ms=14, zorder=6, label="Goal")
    ax.set_title(
        "Bimodal scenario — symmetric ice block\n"
        "Two equally-good bypasses; their action-space average goes straight through"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def plot_result(cost_map, mean_traj, best_traj, mean_cost, best_cost, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 16))
    _costmap_background(ax, cost_map)

    ax.plot(mean_traj[:, 0], mean_traj[:, 1], "w-", lw=4, zorder=4)
    ax.plot(mean_traj[:, 0], mean_traj[:, 1], "-", color="tomato", lw=2.5,
            zorder=5, label=f"MBD mean  (cost={mean_cost:.0f})")

    ax.plot(best_traj[:, 0], best_traj[:, 1], "w-", lw=4, zorder=6)
    ax.plot(best_traj[:, 0], best_traj[:, 1], "-", color="lime", lw=2.5,
            zorder=7, label=f"Global best sample  (cost={best_cost:.0f})")

    ax.plot(*START[:2], "go", ms=10, zorder=8, label="Start")
    ax.plot(*GOAL,      "m*", ms=14, zorder=8, label="Goal")

    ratio = mean_cost / max(best_cost, 1.0)
    verdict = "COLLAPSE" if ratio > 1.5 else "OK"
    ax.set_title(
        f"MBD result  |  mean cost={mean_cost:.0f}  vs  best sample cost={best_cost:.0f}\n"
        f"Collapse ratio = {ratio:.2f}×  [{verdict}]"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ── Bimodal histogram movie ────────────────────────────────────────────────────

def _crossing_x_at_y(traj: np.ndarray, y_target: float) -> float:
    """Return the x-position where *traj* first crosses y=y_target (linear interp).
    Returns NaN if the trajectory never reaches y_target."""
    ys = traj[:, 1]
    xs = traj[:, 0]
    for i in range(1, len(ys)):
        if ys[i - 1] < y_target <= ys[i]:
            dy = ys[i] - ys[i - 1]
            frac = (y_target - ys[i - 1]) / max(dy, 1e-8)
            return float(xs[i - 1] + frac * (xs[i] - xs[i - 1]))
    return float("nan")


def generate_bimodal_histogram_movie(diag: dict, cost_map, out_path: str) -> None:
    """
    Side-by-side animated movie showing how the sample distribution evolves
    across MBD denoising steps.

      Left panel  — sample trajectory fan (blue=best, red=worst cost rank).
      Right panel — histogram of the x-position where each sample crosses
                    y=_FLOE_CY (the floe centre-line), coloured by corridor:
                      green  = left corridor  (x < FLOE_X_LO)
                      red    = through floe   (FLOE_X_LO ≤ x ≤ FLOE_X_HI)
                      cyan   = right corridor (x > FLOE_X_HI)
                    Orange vertical line = where the *mean* trajectory crosses.

    The histogram directly reveals whether the sample population is bimodal
    (two corridor peaks) or has collapsed toward the floe centre.
    """
    n_steps = len(diag["sample_trajectories"])
    if n_steps == 0:
        print("  No sample data — skipping bimodal histogram movie.", flush=True)
        return

    # ── Pre-compute crossing-x for all samples at every step ───────────────
    crossing_xs_per_step: list[np.ndarray] = []
    mean_crossing_per_step: list[float] = []
    global_min_traj_per_step: list[np.ndarray] = []   # running global best trajectory

    _running_best_cost = float("inf")
    _running_best_traj = None

    for step in range(n_steps):
        trajs = diag["sample_trajectories"][step]   # (N, H+1, 3)
        cxs = np.array([_crossing_x_at_y(trajs[i], _FLOE_CY) for i in range(len(trajs))])
        crossing_xs_per_step.append(cxs)

        # Mean trajectory at this step (index step+1 because index 0 is pre-loop snapshot)
        mt_idx = min(step + 1, len(diag["mean_trajectories"]) - 1)
        mean_crossing_per_step.append(_crossing_x_at_y(diag["mean_trajectories"][mt_idx], _FLOE_CY))

        # Running global minimum (carry forward if this step is not better)
        step_best_cost = diag["cost_mins"][step]
        if step_best_cost < _running_best_cost:
            _running_best_cost = step_best_cost
            _running_best_traj = trajs[0]   # sorted best-first
        global_min_traj_per_step.append(_running_best_traj)

    # ── Histogram bins & stable y-limit ────────────────────────────────────
    bins = np.linspace(X_MIN, X_MAX, 41)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    max_count = 1
    for cxs in crossing_xs_per_step:
        valid = cxs[~np.isnan(cxs)]
        if len(valid):
            max_count = max(max_count, int(np.histogram(valid, bins=bins)[0].max()))

    # Bar colour: left corridor = limegreen, through floe = tomato, right = deepskyblue
    bar_colors = [
        "limegreen" if bx < FLOE_X_LO else ("tomato" if bx <= FLOE_X_HI else "deepskyblue")
        for bx in bin_centers
    ]

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor="k")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.08)
    ax_map  = fig.add_subplot(gs[0], facecolor="#111111")
    ax_hist = fig.add_subplot(gs[1], facecolor="#111111")

    for ax in (ax_map, ax_hist):
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

    # ── Left panel: static costmap background ──────────────────────────────
    grid = cost_map.cost_map
    extent = [
        cost_map.x_origin, cost_map.x_origin + cost_map.width,
        cost_map.y_origin, cost_map.y_origin + cost_map.height,
    ]
    vmax_val = float(np.percentile(grid[grid > 0], 95)) if np.any(grid > 0) else 1.0
    ax_map.imshow(grid, origin="lower", extent=extent, cmap="YlOrRd",
                  alpha=0.55, vmin=0, vmax=vmax_val, zorder=1)
    ax_map.add_patch(mpatches.FancyBboxPatch(
        (FLOE_X_LO, FLOE_Y_LO), _FLOE_W, _FLOE_H,
        linewidth=2, edgecolor="white", facecolor="steelblue", alpha=0.2,
        boxstyle="square,pad=0", zorder=2,
    ))
    # Dashed line at floe centre — this is where we measure crossing-x
    ax_map.axhline(_FLOE_CY, color="cyan", lw=1.2, ls="--", alpha=0.7, zorder=3,
                   label=f"Crossing level  y={_FLOE_CY:.0f} m")
    ax_map.plot(*START[:2], "go", ms=8, zorder=5)
    ax_map.plot(*GOAL,      "m*", ms=12, zorder=5)
    ax_map.set_xlim(X_MIN, X_MAX)
    ax_map.set_ylim(Y_MIN, Y_MAX)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.set_title("Sample trajectories", color="white", fontsize=11)
    ax_map.legend(loc="upper right", fontsize=7, facecolor="#222222",
                  labelcolor="white", framealpha=0.7)

    # Dynamic map artists
    sample_lc = LineCollection([], linewidths=0.5, alpha=0.30, zorder=3)
    ax_map.add_collection(sample_lc)
    mean_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=4)
    mean_line_map,    = ax_map.plot([], [], "-", color="orange", lw=2, zorder=5)
    best_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=6)
    best_line_map,    = ax_map.plot([], [], "-", color="lime",   lw=2, zorder=7)
    gmin_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=8)
    gmin_line_map,    = ax_map.plot([], [], "-", color="gold",   lw=2, zorder=9)
    step_text = ax_map.text(0.02, 0.98, "", transform=ax_map.transAxes,
                             color="white", fontsize=9, va="top", ha="left",
                             bbox=dict(facecolor="#00000088", edgecolor="none", pad=3))

    # ── Right panel: static shading ────────────────────────────────────────
    ax_hist.axvspan(X_MIN,     FLOE_X_LO, alpha=0.10, color="limegreen", zorder=1)
    ax_hist.axvspan(FLOE_X_LO, FLOE_X_HI, alpha=0.18, color="red",       zorder=1)
    ax_hist.axvspan(FLOE_X_HI, X_MAX,     alpha=0.10, color="deepskyblue", zorder=1)
    ax_hist.axvline(FLOE_X_LO, color="red", lw=1.2, ls="--", alpha=0.8, zorder=2)
    ax_hist.axvline(FLOE_X_HI, color="red", lw=1.2, ls="--", alpha=0.8, zorder=2)

    mean_vline = ax_hist.axvline(np.nan, color="orange", lw=2.5, zorder=5)
    best_vline = ax_hist.axvline(np.nan, color="lime",   lw=2.5, zorder=5, ls="--")
    gmin_vline = ax_hist.axvline(np.nan, color="gold",   lw=2.5, zorder=5, ls=":")
    legend_handles = [
        mpatches.Patch(color="limegreen",   alpha=0.7, label="Left corridor"),
        mpatches.Patch(color="tomato",      alpha=0.7, label="Through floe (bad)"),
        mpatches.Patch(color="deepskyblue", alpha=0.7, label="Right corridor"),
        plt.Line2D([0], [0], color="orange", lw=2.5,        label="Mean traj crossing"),
        plt.Line2D([0], [0], color="lime",   lw=2.5, ls="--", label="Step best crossing"),
        plt.Line2D([0], [0], color="gold",   lw=2.5, ls=":",  label="Global best crossing"),
    ]
    ax_hist.legend(handles=legend_handles, loc="upper right", fontsize=8,
                   facecolor="#222222", labelcolor="white", framealpha=0.8)
    ax_hist.set_xlim(X_MIN, X_MAX)
    ax_hist.set_ylim(0, max_count * 1.18)
    ax_hist.set_xlabel(f"x when trajectory crosses y={_FLOE_CY:.0f} m  [m]")
    ax_hist.set_ylabel("Number of samples")
    ax_hist.set_title("Crossing-x distribution  (bimodal = two bypass routes coexist)",
                      color="white", fontsize=10)

    # Dynamic bar artists
    bar_rects = ax_hist.bar(bin_centers, np.zeros(len(bin_centers)),
                             width=bin_width * 0.88, color=bar_colors,
                             alpha=0.80, zorder=3)

    # ── Animation update ────────────────────────────────────────────────────
    cmap_traj = plt.cm.coolwarm

    _accept_rates = diag.get("accept_rates", [])

    def _update(frame):
        step = frame

        # Left panel — trajectory fan
        trajs = diag["sample_trajectories"][step]   # (N, H+1, 3), sorted best→worst
        n = len(trajs)
        segs   = [trajs[i, :, :2] for i in range(n)]
        colors = [cmap_traj(i / max(n - 1, 1)) for i in range(n)]
        sample_lc.set_segments(segs)
        sample_lc.set_color(colors)

        mt_idx = min(step + 1, len(diag["mean_trajectories"]) - 1)
        mt = diag["mean_trajectories"][mt_idx]
        mean_outline_map.set_data(mt[:, 0], mt[:, 1])
        mean_line_map.set_data(mt[:, 0], mt[:, 1])

        # Step best sample = index 0 (sorted best-first)
        bt = trajs[0]
        best_outline_map.set_data(bt[:, 0], bt[:, 1])
        best_line_map.set_data(bt[:, 0], bt[:, 1])

        # Global min trajectory (running best across all steps so far)
        gt = global_min_traj_per_step[step]
        gmin_outline_map.set_data(gt[:, 0], gt[:, 1])
        gmin_line_map.set_data(gt[:, 0], gt[:, 1])

        com  = diag["cost_of_mean"][step] if step < len(diag["cost_of_mean"]) else float("nan")
        cmin = diag["cost_mins"][step]
        n_reached = int(np.sum(~np.isnan(crossing_xs_per_step[step])))
        _ar_str = ""
        if step < len(_accept_rates):
            _ar_str = f"\naccept rate: {_accept_rates[step]*100:.0f}%"
        step_text.set_text(
            f"Step {step + 1}/{n_steps}\n"
            f"cost(mean) = {com:,.0f}\n"
            f"cost(best) = {cmin:,.0f}\n"
            f"reached floe level: {n_reached}/{n}"
            + _ar_str
        )

        # Right panel — histogram bars
        cxs   = crossing_xs_per_step[step]
        valid = cxs[~np.isnan(cxs)]
        counts = np.histogram(valid, bins=bins)[0] if len(valid) else np.zeros(len(bin_centers))
        for rect, h in zip(bar_rects, counts):
            rect.set_height(h)

        # Crossing vlines: mean, step best, global min
        mcx = mean_crossing_per_step[step]
        mean_vline.set_xdata([mcx, mcx] if not np.isnan(mcx) else [np.nan, np.nan])
        bcx = _crossing_x_at_y(bt, _FLOE_CY)
        best_vline.set_xdata([bcx, bcx] if not np.isnan(bcx) else [np.nan, np.nan])
        gcx = _crossing_x_at_y(gt, _FLOE_CY)
        gmin_vline.set_xdata([gcx, gcx] if not np.isnan(gcx) else [np.nan, np.nan])

        fig.suptitle(
            f"MBD denoising — step {step + 1}/{n_steps}  |  "
            f"σ decreasing →  samples concentrate",
            color="white", fontsize=12, y=0.995,
        )

        return (sample_lc, mean_outline_map, mean_line_map,
                best_outline_map, best_line_map,
                gmin_outline_map, gmin_line_map,
                mean_vline, best_vline, gmin_vline, step_text, *bar_rects)

    ani = animation.FuncAnimation(
        fig, _update, frames=n_steps, interval=200, blit=False,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=5, bitrate=2000)
    ani.save(out_path, writer=writer, dpi=100, savefig_kwargs={"facecolor": "k"})
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ── Bimodal weight-histogram movie ────────────────────────────────────────────

def generate_bimodal_weight_movie(diag: dict, cost_map, out_path: str) -> None:
    """
    Same layout as generate_bimodal_histogram_movie, but the right-panel bars
    show the *sum of softmax importance weights* for samples that cross y=_FLOE_CY
    at each x-bin, rather than the raw count of samples.

    Because the softmax weights sum to 1 across all samples at each step, the
    bar heights tell you what fraction of the planner's "attention" is directed
    at each corridor — more informative than the raw sample count when temperature
    skews the weight distribution.

    Note: diag["weights"] is in the *original* (unsorted) sample order, while
    diag["sample_trajectories"] is sorted best-first.  We recover the alignment
    via argsort(sample_costs) so the weight assigned to each stored trajectory is
    correctly identified.
    """
    n_steps = len(diag["sample_trajectories"])
    if n_steps == 0:
        print("  No sample data — skipping bimodal weight movie.", flush=True)
        return

    if "weights" not in diag or len(diag["weights"]) == 0:
        print("  No weight data in diagnostics — skipping bimodal weight movie.", flush=True)
        return

    # ── Pre-compute per-step crossing-x and matching weights ──────────────
    crossing_xs_per_step: list[np.ndarray] = []
    sample_weights_per_step: list[np.ndarray] = []
    mean_crossing_per_step: list[float] = []
    global_min_traj_per_step: list[np.ndarray] = []

    _running_best_cost = float("inf")
    _running_best_traj = None

    for step in range(n_steps):
        trajs   = diag["sample_trajectories"][step]   # (N, H+1, 3), sorted best→worst
        costs_np = diag["sample_costs"][step]          # (N,) in original order
        w_np     = diag["weights"][step]               # (N,) in original order, sums to 1

        # Map sorted trajectory indices back to original sample indices.
        # Use the stored sort index when available (AL-MBD sorts by total AL cost,
        # not base cost, so np.argsort(costs_np) would misalign weights).
        n_stored = len(trajs)
        _sort_indices = diag.get("sample_sort_indices", [])
        sort_idx = _sort_indices[step] if _sort_indices else np.argsort(costs_np)
        sorted_weights = w_np[sort_idx[:n_stored]]     # weights aligned with stored trajs

        cxs = np.array([_crossing_x_at_y(trajs[i], _FLOE_CY) for i in range(n_stored)])
        crossing_xs_per_step.append(cxs)
        sample_weights_per_step.append(sorted_weights)

        mt_idx = min(step + 1, len(diag["mean_trajectories"]) - 1)
        mean_crossing_per_step.append(_crossing_x_at_y(diag["mean_trajectories"][mt_idx], _FLOE_CY))

        step_best_cost = diag["cost_mins"][step]
        if step_best_cost < _running_best_cost:
            _running_best_cost = step_best_cost
            _running_best_traj = trajs[0]
        global_min_traj_per_step.append(_running_best_traj)

    # ── Histogram bins & stable y-limit ────────────────────────────────────
    bins = np.linspace(X_MIN, X_MAX, 41)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    max_weight_sum = 1e-6
    for cxs, ws in zip(crossing_xs_per_step, sample_weights_per_step):
        valid_mask = ~np.isnan(cxs)
        if valid_mask.any():
            cxs_v = cxs[valid_mask]
            ws_v  = ws[valid_mask]
            for j in range(len(bin_centers)):
                in_bin = (cxs_v >= bins[j]) & (cxs_v < bins[j + 1])
                if in_bin.any():
                    max_weight_sum = max(max_weight_sum, float(ws_v[in_bin].max()))

    bar_colors = [
        "limegreen" if bx < FLOE_X_LO else ("tomato" if bx <= FLOE_X_HI else "deepskyblue")
        for bx in bin_centers
    ]

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor="k")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.08)
    ax_map  = fig.add_subplot(gs[0], facecolor="#111111")
    ax_hist = fig.add_subplot(gs[1], facecolor="#111111")

    for ax in (ax_map, ax_hist):
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

    # Left panel: static costmap background
    grid = cost_map.cost_map
    extent = [
        cost_map.x_origin, cost_map.x_origin + cost_map.width,
        cost_map.y_origin, cost_map.y_origin + cost_map.height,
    ]
    vmax_val = float(np.percentile(grid[grid > 0], 95)) if np.any(grid > 0) else 1.0
    ax_map.imshow(grid, origin="lower", extent=extent, cmap="YlOrRd",
                  alpha=0.55, vmin=0, vmax=vmax_val, zorder=1)
    ax_map.add_patch(mpatches.FancyBboxPatch(
        (FLOE_X_LO, FLOE_Y_LO), _FLOE_W, _FLOE_H,
        linewidth=2, edgecolor="white", facecolor="steelblue", alpha=0.2,
        boxstyle="square,pad=0", zorder=2,
    ))
    ax_map.axhline(_FLOE_CY, color="cyan", lw=1.2, ls="--", alpha=0.7, zorder=3,
                   label=f"Crossing level  y={_FLOE_CY:.0f} m")
    ax_map.plot(*START[:2], "go", ms=8, zorder=5)
    ax_map.plot(*GOAL,      "m*", ms=12, zorder=5)
    ax_map.set_xlim(X_MIN, X_MAX)
    ax_map.set_ylim(Y_MIN, Y_MAX)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.set_title("Sample trajectories", color="white", fontsize=11)
    ax_map.legend(loc="upper right", fontsize=7, facecolor="#222222",
                  labelcolor="white", framealpha=0.7)

    sample_lc = LineCollection([], linewidths=0.5, alpha=0.30, zorder=3)
    ax_map.add_collection(sample_lc)
    mean_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=4)
    mean_line_map,    = ax_map.plot([], [], "-", color="orange", lw=2, zorder=5)
    best_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=6)
    best_line_map,    = ax_map.plot([], [], "-", color="lime",   lw=2, zorder=7)
    gmin_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=8)
    gmin_line_map,    = ax_map.plot([], [], "-", color="gold",   lw=2, zorder=9)
    step_text = ax_map.text(0.02, 0.98, "", transform=ax_map.transAxes,
                             color="white", fontsize=9, va="top", ha="left",
                             bbox=dict(facecolor="#00000088", edgecolor="none", pad=3))

    # Right panel: static shading
    ax_hist.axvspan(X_MIN,     FLOE_X_LO, alpha=0.10, color="limegreen",   zorder=1)
    ax_hist.axvspan(FLOE_X_LO, FLOE_X_HI, alpha=0.18, color="red",         zorder=1)
    ax_hist.axvspan(FLOE_X_HI, X_MAX,     alpha=0.10, color="deepskyblue", zorder=1)
    ax_hist.axvline(FLOE_X_LO, color="red", lw=1.2, ls="--", alpha=0.8, zorder=2)
    ax_hist.axvline(FLOE_X_HI, color="red", lw=1.2, ls="--", alpha=0.8, zorder=2)

    mean_vline = ax_hist.axvline(np.nan, color="orange", lw=2.5, zorder=5)
    best_vline = ax_hist.axvline(np.nan, color="lime",   lw=2.5, zorder=5, ls="--")
    gmin_vline = ax_hist.axvline(np.nan, color="gold",   lw=2.5, zorder=5, ls=":")
    legend_handles = [
        mpatches.Patch(color="limegreen",   alpha=0.7, label="Left corridor"),
        mpatches.Patch(color="tomato",      alpha=0.7, label="Through floe (bad)"),
        mpatches.Patch(color="deepskyblue", alpha=0.7, label="Right corridor"),
        plt.Line2D([0], [0], color="orange", lw=2.5,        label="Mean traj crossing"),
        plt.Line2D([0], [0], color="lime",   lw=2.5, ls="--", label="Step best crossing"),
        plt.Line2D([0], [0], color="gold",   lw=2.5, ls=":",  label="Global best crossing"),
    ]
    ax_hist.legend(handles=legend_handles, loc="upper right", fontsize=8,
                   facecolor="#222222", labelcolor="white", framealpha=0.8)
    ax_hist.set_xlim(X_MIN, X_MAX)
    ax_hist.set_ylim(0, 0.1)
    ax_hist.set_xlabel(f"x when trajectory crosses y={_FLOE_CY:.0f} m  [m]")
    ax_hist.set_ylabel("Max softmax weight in bin")
    ax_hist.set_title(
        "Crossing-x max weight  (height = best single trajectory weight per corridor)",
        color="white", fontsize=10,
    )

    bar_rects = ax_hist.bar(bin_centers, np.zeros(len(bin_centers)),
                             width=bin_width * 0.88, color=bar_colors,
                             alpha=0.80, zorder=3)

    # Animation update
    cmap_traj = plt.cm.coolwarm

    def _update(frame):
        step = frame

        trajs = diag["sample_trajectories"][step]
        n = len(trajs)
        segs   = [trajs[i, :, :2] for i in range(n)]
        colors = [cmap_traj(i / max(n - 1, 1)) for i in range(n)]
        sample_lc.set_segments(segs)
        sample_lc.set_color(colors)

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
        step_text.set_text(
            f"Step {step + 1}/{n_steps}\n"
            f"cost(mean) = {com:,.0f}\n"
            f"cost(best) = {cmin:,.0f}\n"
            f"reached floe level: {n_reached}/{n}"
        )

        # Max-weight histogram: for each bin, the highest weight among
        # all samples whose crossing-x falls in that bin.
        cxs = crossing_xs_per_step[step]
        ws  = sample_weights_per_step[step]
        valid_mask = ~np.isnan(cxs)
        w_hist = np.zeros(len(bin_centers))
        if valid_mask.any():
            cxs_v = cxs[valid_mask]
            ws_v  = ws[valid_mask]
            for j in range(len(bin_centers)):
                in_bin = (cxs_v >= bins[j]) & (cxs_v < bins[j + 1])
                if in_bin.any():
                    w_hist[j] = float(ws_v[in_bin].max())
        for rect, h in zip(bar_rects, w_hist):
            rect.set_height(h)

        mcx = mean_crossing_per_step[step]
        mean_vline.set_xdata([mcx, mcx] if not np.isnan(mcx) else [np.nan, np.nan])
        bcx = _crossing_x_at_y(bt, _FLOE_CY)
        best_vline.set_xdata([bcx, bcx] if not np.isnan(bcx) else [np.nan, np.nan])
        gcx = _crossing_x_at_y(gt, _FLOE_CY)
        gmin_vline.set_xdata([gcx, gcx] if not np.isnan(gcx) else [np.nan, np.nan])

        fig.suptitle(
            f"MBD denoising — step {step + 1}/{n_steps}  |  "
            f"σ decreasing →  samples concentrate",
            color="white", fontsize=12, y=0.995,
        )

        return (sample_lc, mean_outline_map, mean_line_map,
                best_outline_map, best_line_map,
                gmin_outline_map, gmin_line_map,
                mean_vline, best_vline, gmin_vline, step_text, *bar_rects)

    ani = animation.FuncAnimation(
        fig, _update, frames=n_steps, interval=200, blit=False,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=5, bitrate=2000)
    ani.save(out_path, writer=writer, dpi=100, savefig_kwargs={"facecolor": "k"})
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ── Bimodal mean-cost histogram movie ─────────────────────────────────────────

def generate_bimodal_cost_movie(diag: dict, cost_map, out_path: str) -> None:
    """
    Diagnostic movie: right panel shows the *mean cost* of samples crossing
    y=_FLOE_CY at each x-bin.  Bins with no samples are shown at 0.

    Use this to verify whether the cost landscape is actually differentiated
    across corridors.  If costs are flat here too, the scenario is not
    penalising the floe enough.  If costs are clearly differentiated but
    softmax weights are still flat, there is a weight computation bug.
    """
    n_steps = len(diag["sample_trajectories"])
    if n_steps == 0:
        print("  No sample data — skipping bimodal cost movie.", flush=True)
        return

    # ── Pre-compute crossing-x and per-sample costs per step ──────────────
    crossing_xs_per_step: list[np.ndarray] = []
    sample_costs_per_step: list[np.ndarray] = []
    mean_crossing_per_step: list[float] = []
    global_min_traj_per_step: list[np.ndarray] = []

    _running_best_cost = float("inf")
    _running_best_traj = None

    for step in range(n_steps):
        trajs    = diag["sample_trajectories"][step]   # (N, H+1, 3), sorted best→worst
        costs_np = diag["sample_costs"][step]          # (N,) in original order

        n_stored      = len(trajs)
        _sort_indices = diag.get("sample_sort_indices", [])
        sort_idx      = _sort_indices[step] if _sort_indices else np.argsort(costs_np)
        sorted_costs  = costs_np[sort_idx[:n_stored]]

        cxs = np.array([_crossing_x_at_y(trajs[i], _FLOE_CY) for i in range(n_stored)])
        crossing_xs_per_step.append(cxs)
        sample_costs_per_step.append(sorted_costs)

        mt_idx = min(step + 1, len(diag["mean_trajectories"]) - 1)
        mean_crossing_per_step.append(_crossing_x_at_y(diag["mean_trajectories"][mt_idx], _FLOE_CY))

        step_best_cost = diag["cost_mins"][step]
        if step_best_cost < _running_best_cost:
            _running_best_cost = step_best_cost
            _running_best_traj = trajs[0]
        global_min_traj_per_step.append(_running_best_traj)

    # ── Histogram bins & stable y-limit ────────────────────────────────────
    bins        = np.linspace(X_MIN, X_MAX, 41)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width   = bins[1] - bins[0]

    max_mean_cost = 1.0
    for cxs, cs in zip(crossing_xs_per_step, sample_costs_per_step):
        valid = ~np.isnan(cxs)
        if valid.any():
            for j in range(len(bin_centers)):
                in_bin = valid & (cxs >= bins[j]) & (cxs < bins[j + 1])
                if in_bin.any():
                    max_mean_cost = max(max_mean_cost, float(cs[in_bin].mean()))

    bar_colors = [
        "limegreen" if bx < FLOE_X_LO else ("tomato" if bx <= FLOE_X_HI else "deepskyblue")
        for bx in bin_centers
    ]

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor="k")
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.08)
    ax_map  = fig.add_subplot(gs[0], facecolor="#111111")
    ax_hist = fig.add_subplot(gs[1], facecolor="#111111")

    for ax in (ax_map, ax_hist):
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

    grid     = cost_map.cost_map
    extent   = [cost_map.x_origin, cost_map.x_origin + cost_map.width,
                cost_map.y_origin, cost_map.y_origin + cost_map.height]
    vmax_val = float(np.percentile(grid[grid > 0], 95)) if np.any(grid > 0) else 1.0
    ax_map.imshow(grid, origin="lower", extent=extent, cmap="YlOrRd",
                  alpha=0.55, vmin=0, vmax=vmax_val, zorder=1)
    ax_map.add_patch(mpatches.FancyBboxPatch(
        (FLOE_X_LO, FLOE_Y_LO), _FLOE_W, _FLOE_H,
        linewidth=2, edgecolor="white", facecolor="steelblue", alpha=0.2,
        boxstyle="square,pad=0", zorder=2,
    ))
    ax_map.axhline(_FLOE_CY, color="cyan", lw=1.2, ls="--", alpha=0.7, zorder=3,
                   label=f"Crossing level  y={_FLOE_CY:.0f} m")
    ax_map.plot(*START[:2], "go", ms=8, zorder=5)
    ax_map.plot(*GOAL,      "m*", ms=12, zorder=5)
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
    mean_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=4)
    mean_line_map,    = ax_map.plot([], [], "-", color="orange", lw=2, zorder=5)
    best_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=6)
    best_line_map,    = ax_map.plot([], [], "-", color="lime",   lw=2, zorder=7)
    gmin_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=8)
    gmin_line_map,    = ax_map.plot([], [], "-", color="gold",   lw=2, zorder=9)
    step_text = ax_map.text(0.02, 0.98, "", transform=ax_map.transAxes,
                             color="white", fontsize=9, va="top", ha="left",
                             bbox=dict(facecolor="#00000088", edgecolor="none", pad=3))

    ax_hist.axvspan(X_MIN,     FLOE_X_LO, alpha=0.10, color="limegreen",   zorder=1)
    ax_hist.axvspan(FLOE_X_LO, FLOE_X_HI, alpha=0.18, color="red",         zorder=1)
    ax_hist.axvspan(FLOE_X_HI, X_MAX,     alpha=0.10, color="deepskyblue", zorder=1)
    ax_hist.axvline(FLOE_X_LO, color="red", lw=1.2, ls="--", alpha=0.8, zorder=2)
    ax_hist.axvline(FLOE_X_HI, color="red", lw=1.2, ls="--", alpha=0.8, zorder=2)

    mean_vline = ax_hist.axvline(np.nan, color="orange", lw=2.5, zorder=5)
    best_vline = ax_hist.axvline(np.nan, color="lime",   lw=2.5, zorder=5, ls="--")
    gmin_vline = ax_hist.axvline(np.nan, color="gold",   lw=2.5, zorder=5, ls=":")
    legend_handles = [
        mpatches.Patch(color="limegreen",   alpha=0.7, label="Left corridor"),
        mpatches.Patch(color="tomato",      alpha=0.7, label="Through floe (bad)"),
        mpatches.Patch(color="deepskyblue", alpha=0.7, label="Right corridor"),
        plt.Line2D([0], [0], color="orange", lw=2.5,         label="Mean traj crossing"),
        plt.Line2D([0], [0], color="lime",   lw=2.5, ls="--", label="Step best crossing"),
        plt.Line2D([0], [0], color="gold",   lw=2.5, ls=":",  label="Global best crossing"),
    ]
    ax_hist.legend(handles=legend_handles, loc="upper right", fontsize=8,
                   facecolor="#222222", labelcolor="white", framealpha=0.8)
    ax_hist.set_xlim(X_MIN, X_MAX)
    ax_hist.set_ylim(0, max_mean_cost * 1.18)
    ax_hist.set_xlabel(f"x when trajectory crosses y={_FLOE_CY:.0f} m  [m]")
    ax_hist.set_ylabel("Mean cost of samples in bin")
    ax_hist.set_title(
        "Crossing-x mean cost  (diagnostic: are costs actually differentiated?)",
        color="white", fontsize=10,
    )

    bar_rects = ax_hist.bar(bin_centers, np.zeros(len(bin_centers)),
                             width=bin_width * 0.88, color=bar_colors,
                             alpha=0.80, zorder=3)

    cmap_traj = plt.cm.coolwarm

    def _update(frame):
        step = frame

        trajs = diag["sample_trajectories"][step]
        n     = len(trajs)
        segs   = [trajs[i, :, :2] for i in range(n)]
        colors = [cmap_traj(i / max(n - 1, 1)) for i in range(n)]
        sample_lc.set_segments(segs)
        sample_lc.set_color(colors)

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
        step_text.set_text(
            f"Step {step + 1}/{n_steps}\n"
            f"cost(mean) = {com:,.0f}\n"
            f"cost(best) = {cmin:,.0f}\n"
            f"reached floe level: {n_reached}/{n}"
        )

        cxs   = crossing_xs_per_step[step]
        cs    = sample_costs_per_step[step]
        valid = ~np.isnan(cxs)
        cost_hist = np.zeros(len(bin_centers))
        if valid.any():
            for j in range(len(bin_centers)):
                in_bin = valid & (cxs >= bins[j]) & (cxs < bins[j + 1])
                if in_bin.any():
                    cost_hist[j] = float(cs[in_bin].mean())
        for rect, h in zip(bar_rects, cost_hist):
            rect.set_height(h)

        mcx = mean_crossing_per_step[step]
        mean_vline.set_xdata([mcx, mcx] if not np.isnan(mcx) else [np.nan, np.nan])
        bcx = _crossing_x_at_y(bt, _FLOE_CY)
        best_vline.set_xdata([bcx, bcx] if not np.isnan(bcx) else [np.nan, np.nan])
        gcx = _crossing_x_at_y(gt, _FLOE_CY)
        gmin_vline.set_xdata([gcx, gcx] if not np.isnan(gcx) else [np.nan, np.nan])

        fig.suptitle(
            f"MBD denoising — step {step + 1}/{n_steps}  |  "
            f"σ decreasing →  samples concentrate",
            color="white", fontsize=12, y=0.995,
        )

        return (sample_lc, mean_outline_map, mean_line_map,
                best_outline_map, best_line_map,
                gmin_outline_map, gmin_line_map,
                mean_vline, best_vline, gmin_vline, step_text, *bar_rects)

    ani = animation.FuncAnimation(
        fig, _update, frames=n_steps, interval=200, blit=False,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=5, bitrate=2000)
    ani.save(out_path, writer=writer, dpi=100, savefig_kwargs={"facecolor": "k"})
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ── Bimodal min-cost histogram movie ──────────────────────────────────────────

def generate_bimodal_min_cost_movie(diag: dict, cost_map, out_path: str) -> None:
    """
    Diagnostic movie: right panel shows the *minimum cost* of samples crossing
    y=_FLOE_CY at each x-bin.  Bins with no samples are shown at 0.

    Complements the mean-cost movie: if even the best sample in a bin has high
    cost, the cost landscape truly does not reward that corridor.  If min-cost
    is low for corridors but mean-cost is high, outlier/OOB trajectories are
    inflating the mean.
    """
    n_steps = len(diag["sample_trajectories"])
    if n_steps == 0:
        print("  No sample data — skipping bimodal min-cost movie.", flush=True)
        return

    # ── Pre-compute crossing-x and per-sample costs per step ──────────────
    crossing_xs_per_step: list[np.ndarray] = []
    sample_costs_per_step: list[np.ndarray] = []
    mean_crossing_per_step: list[float] = []
    global_min_traj_per_step: list[np.ndarray] = []

    _running_best_cost = float("inf")
    _running_best_traj = None

    for step in range(n_steps):
        trajs    = diag["sample_trajectories"][step]   # (N, H+1, 3), sorted best→worst
        costs_np = diag["sample_costs"][step]          # (N,) in original order

        n_stored      = len(trajs)
        _sort_indices = diag.get("sample_sort_indices", [])
        sort_idx      = _sort_indices[step] if _sort_indices else np.argsort(costs_np)
        sorted_costs  = costs_np[sort_idx[:n_stored]]

        cxs = np.array([_crossing_x_at_y(trajs[i], _FLOE_CY) for i in range(n_stored)])
        crossing_xs_per_step.append(cxs)
        sample_costs_per_step.append(sorted_costs)

        mt_idx = min(step + 1, len(diag["mean_trajectories"]) - 1)
        mean_crossing_per_step.append(_crossing_x_at_y(diag["mean_trajectories"][mt_idx], _FLOE_CY))

        step_best_cost = diag["cost_mins"][step]
        if step_best_cost < _running_best_cost:
            _running_best_cost = step_best_cost
            _running_best_traj = trajs[0]
        global_min_traj_per_step.append(_running_best_traj)

    # ── Histogram bins & stable y-limit ────────────────────────────────────
    bins        = np.linspace(X_MIN, X_MAX, 41)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width   = bins[1] - bins[0]

    max_min_cost = 1.0
    for cxs, cs in zip(crossing_xs_per_step, sample_costs_per_step):
        valid = ~np.isnan(cxs)
        if valid.any():
            for j in range(len(bin_centers)):
                in_bin = valid & (cxs >= bins[j]) & (cxs < bins[j + 1])
                if in_bin.any():
                    max_min_cost = max(max_min_cost, float(cs[in_bin].min()))

    bar_colors = [
        "limegreen" if bx < FLOE_X_LO else ("tomato" if bx <= FLOE_X_HI else "deepskyblue")
        for bx in bin_centers
    ]

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor="k")
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.08)
    ax_map  = fig.add_subplot(gs[0], facecolor="#111111")
    ax_hist = fig.add_subplot(gs[1], facecolor="#111111")

    for ax in (ax_map, ax_hist):
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

    grid     = cost_map.cost_map
    extent   = [cost_map.x_origin, cost_map.x_origin + cost_map.width,
                cost_map.y_origin, cost_map.y_origin + cost_map.height]
    vmax_val = float(np.percentile(grid[grid > 0], 95)) if np.any(grid > 0) else 1.0
    ax_map.imshow(grid, origin="lower", extent=extent, cmap="YlOrRd",
                  alpha=0.55, vmin=0, vmax=vmax_val, zorder=1)
    ax_map.add_patch(mpatches.FancyBboxPatch(
        (FLOE_X_LO, FLOE_Y_LO), _FLOE_W, _FLOE_H,
        linewidth=2, edgecolor="white", facecolor="steelblue", alpha=0.2,
        boxstyle="square,pad=0", zorder=2,
    ))
    ax_map.axhline(_FLOE_CY, color="cyan", lw=1.2, ls="--", alpha=0.7, zorder=3,
                   label=f"Crossing level  y={_FLOE_CY:.0f} m")
    ax_map.plot(*START[:2], "go", ms=8, zorder=5)
    ax_map.plot(*GOAL,      "m*", ms=12, zorder=5)
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
    mean_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=4)
    mean_line_map,    = ax_map.plot([], [], "-", color="orange", lw=2, zorder=5)
    best_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=6)
    best_line_map,    = ax_map.plot([], [], "-", color="lime",   lw=2, zorder=7)
    gmin_outline_map, = ax_map.plot([], [], "w-", lw=4, zorder=8)
    gmin_line_map,    = ax_map.plot([], [], "-", color="gold",   lw=2, zorder=9)
    step_text = ax_map.text(0.02, 0.98, "", transform=ax_map.transAxes,
                             color="white", fontsize=9, va="top", ha="left",
                             bbox=dict(facecolor="#00000088", edgecolor="none", pad=3))

    ax_hist.axvspan(X_MIN,     FLOE_X_LO, alpha=0.10, color="limegreen",   zorder=1)
    ax_hist.axvspan(FLOE_X_LO, FLOE_X_HI, alpha=0.18, color="red",         zorder=1)
    ax_hist.axvspan(FLOE_X_HI, X_MAX,     alpha=0.10, color="deepskyblue", zorder=1)
    ax_hist.axvline(FLOE_X_LO, color="red", lw=1.2, ls="--", alpha=0.8, zorder=2)
    ax_hist.axvline(FLOE_X_HI, color="red", lw=1.2, ls="--", alpha=0.8, zorder=2)

    mean_vline = ax_hist.axvline(np.nan, color="orange", lw=2.5, zorder=5)
    best_vline = ax_hist.axvline(np.nan, color="lime",   lw=2.5, zorder=5, ls="--")
    gmin_vline = ax_hist.axvline(np.nan, color="gold",   lw=2.5, zorder=5, ls=":")
    legend_handles = [
        mpatches.Patch(color="limegreen",   alpha=0.7, label="Left corridor"),
        mpatches.Patch(color="tomato",      alpha=0.7, label="Through floe (bad)"),
        mpatches.Patch(color="deepskyblue", alpha=0.7, label="Right corridor"),
        plt.Line2D([0], [0], color="orange", lw=2.5,         label="Mean traj crossing"),
        plt.Line2D([0], [0], color="lime",   lw=2.5, ls="--", label="Step best crossing"),
        plt.Line2D([0], [0], color="gold",   lw=2.5, ls=":",  label="Global best crossing"),
    ]
    ax_hist.legend(handles=legend_handles, loc="upper right", fontsize=8,
                   facecolor="#222222", labelcolor="white", framealpha=0.8)
    ax_hist.set_xlim(X_MIN, X_MAX)
    ax_hist.set_ylim(0, 1.25*1e6)
    ax_hist.set_xlabel(f"x when trajectory crosses y={_FLOE_CY:.0f} m  [m]")
    ax_hist.set_ylabel("Min cost of samples in bin")
    ax_hist.set_title(
        "Crossing-x min cost  (diagnostic: best-case cost per corridor)",
        color="white", fontsize=10,
    )

    bar_rects = ax_hist.bar(bin_centers, np.zeros(len(bin_centers)),
                             width=bin_width * 0.88, color=bar_colors,
                             alpha=0.80, zorder=3)

    cmap_traj = plt.cm.coolwarm

    def _update(frame):
        step = frame

        trajs = diag["sample_trajectories"][step]
        n     = len(trajs)
        segs   = [trajs[i, :, :2] for i in range(n)]
        colors = [cmap_traj(i / max(n - 1, 1)) for i in range(n)]
        sample_lc.set_segments(segs)
        sample_lc.set_color(colors)

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
        step_text.set_text(
            f"Step {step + 1}/{n_steps}\n"
            f"cost(mean) = {com:,.0f}\n"
            f"cost(best) = {cmin:,.0f}\n"
            f"reached floe level: {n_reached}/{n}"
        )

        cxs   = crossing_xs_per_step[step]
        cs    = sample_costs_per_step[step]
        valid = ~np.isnan(cxs)
        cost_hist = np.zeros(len(bin_centers))
        if valid.any():
            for j in range(len(bin_centers)):
                in_bin = valid & (cxs >= bins[j]) & (cxs < bins[j + 1])
                if in_bin.any():
                    cost_hist[j] = float(cs[in_bin].min())
        for rect, h in zip(bar_rects, cost_hist):
            rect.set_height(h)

        mcx = mean_crossing_per_step[step]
        mean_vline.set_xdata([mcx, mcx] if not np.isnan(mcx) else [np.nan, np.nan])
        bcx = _crossing_x_at_y(bt, _FLOE_CY)
        best_vline.set_xdata([bcx, bcx] if not np.isnan(bcx) else [np.nan, np.nan])
        gcx = _crossing_x_at_y(gt, _FLOE_CY)
        gmin_vline.set_xdata([gcx, gcx] if not np.isnan(gcx) else [np.nan, np.nan])

        fig.suptitle(
            f"MBD denoising — step {step + 1}/{n_steps}  |  "
            f"σ decreasing →  samples concentrate",
            color="white", fontsize=12, y=0.995,
        )

        return (sample_lc, mean_outline_map, mean_line_map,
                best_outline_map, best_line_map,
                gmin_outline_map, gmin_line_map,
                mean_vline, best_vline, gmin_vline, step_text, *bar_rects)

    ani = animation.FuncAnimation(
        fig, _update, frames=n_steps, interval=200, blit=False,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=5, bitrate=2000)
    ani.save(out_path, writer=writer, dpi=100, savefig_kwargs={"facecolor": "k"})
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ── Cost-category + softmax-analysis movie ─────────────────────────────────────

def generate_cost_category_movie(
    diag: dict,
    temperature: float,
    out_path: str,
    goal_penalty: float = 1e6,
    oob_thresh: float = 1e8,
    fps: int = 5,
) -> None:
    """
    Diagnostic movie that makes the "OOB inflates rew_std → uniform weights"
    mechanism directly visible.

    Left panel (animated):
        Bar chart of sample counts in 5 cost categories at each denoising step,
        with percentage labels and a text box showing the key softmax numbers:
        rew_std, Δlogp0, per-sample weight ratio, total corridor weight, and
        the crowd advantage factor.

    Right panels (static time-series + animated vertical cursor):
        Top    — Fraction of samples in each category vs. denoising step.
        Middle — Mean logp0 per category vs. step (shows how OOB pulls all
                 non-OOB logp0 values close together).
        Bottom — Δlogp0(corridor − miss-goal), exp(Δlogp0) per-sample weight
                 ratio, and total corridor weight fraction on a twin axis.

    Cost categories:
        ~0         cost < 1e4          goal reached, open-water corridor
        (0, 1e6)   1e4 ≤ cost < 1e6   goal reached, through ice
        ~1e6       1e6 ≤ cost < 1.1e6 missed goal, open-water (just goal penalty)
        (1e6,1e8)  1.1e6 ≤ cost < 1e8 missed goal, through ice
        >1e8       cost ≥ 1e8         out-of-bounds (OOB)
    """
    n_steps = len(diag["sample_costs"])
    if n_steps == 0:
        print("  No sample data — skipping cost category movie.", flush=True)
        return
    if "weights" not in diag or len(diag["weights"]) == 0:
        print("  No weight data — skipping cost category movie.", flush=True)
        return

    # ── Category definitions ──────────────────────────────────────────────────
    THRESH_CORR      = 1e4
    THRESH_MISS_CLEAN = goal_penalty * 1.1   # separates "~1e6 no-ice" from "ice+miss"

    CAT_LABELS = [
        "~0\n(corr+goal)",
        "(0, 1e6)\n(ice+goal)",
        "~1e6\n(open+miss)",
        "(1e6,1e8)\n(ice+miss)",
        ">1e8\n(OOB)",
    ]
    CAT_COLORS = ["limegreen", "royalblue", "gold", "tomato", "mediumpurple"]
    CAT_BOUNDS = [
        (0,                THRESH_CORR),
        (THRESH_CORR,      goal_penalty),
        (goal_penalty,     THRESH_MISS_CLEAN),
        (THRESH_MISS_CLEAN, oob_thresh),
        (oob_thresh,       np.inf),
    ]
    N_CATS = len(CAT_LABELS)

    def _categorize(costs: np.ndarray) -> np.ndarray:
        cats = np.full(len(costs), N_CATS - 1, dtype=int)
        for k, (lo, hi) in enumerate(CAT_BOUNDS):
            cats[(costs >= lo) & (costs < hi)] = k
        return cats

    # ── Pre-compute per-step statistics ──────────────────────────────────────
    Nsample = len(diag["sample_costs"][0])
    counts_hist      = np.zeros((n_steps, N_CATS), dtype=float)
    fracs_hist       = np.zeros((n_steps, N_CATS), dtype=float)
    rew_std_hist     = np.zeros(n_steps)
    rew_mean_hist    = np.zeros(n_steps)
    logp0_cat_hist   = np.full((n_steps, N_CATS), np.nan)   # mean logp0 per cat
    delta_logp0_hist = np.full(n_steps, np.nan)              # logp0[corr] − logp0[miss]
    w_ratio_hist     = np.full(n_steps, np.nan)              # exp(Δlogp0)
    corr_wfrac_hist  = np.zeros(n_steps)                     # sum w for cat-0
    crowd_hist       = np.full(n_steps, np.nan)              # n_miss / (n_corr × exp(Δlogp0))

    for step in range(n_steps):
        costs_np = np.array(diag["sample_costs"][step], dtype=float)
        w_np     = np.array(diag["weights"][step], dtype=float)

        cats = _categorize(costs_np)
        for k in range(N_CATS):
            counts_hist[step, k] = np.sum(cats == k)
        fracs_hist[step] = counts_hist[step] / Nsample

        # Replicate the planner's z-score normalisation exactly
        rews     = -costs_np
        rew_mean = float(rews.mean())
        rew_std  = max(float(rews.std()), 1e-6)
        rew_std_hist[step] = rew_std
        rew_mean_hist[step] = rew_mean

        logp0 = (rews - rew_mean) / rew_std / temperature
        for k in range(N_CATS):
            mask = cats == k
            if mask.any():
                logp0_cat_hist[step, k] = float(logp0[mask].mean())

        # Δlogp0: category 0 (corridor+goal) vs category 2 (~1e6 open water+miss)
        lp0 = logp0_cat_hist[step, 0]
        lp2 = logp0_cat_hist[step, 2]
        if np.isfinite(lp0) and np.isfinite(lp2):
            delta = lp0 - lp2
            delta_logp0_hist[step] = delta
            w_ratio_hist[step] = float(np.exp(np.clip(delta, -50, 50)))

        mask0 = cats == 0
        corr_wfrac_hist[step] = float(w_np[mask0].sum()) if mask0.any() else 0.0

        # Crowd advantage: how many times does miss-goal outweigh corridor in
        # total weight, despite corridor's per-sample weight advantage?
        n_corr  = counts_hist[step, 0]
        n_miss  = counts_hist[step, 2] + counts_hist[step, 3]
        wr      = w_ratio_hist[step]
        if n_corr > 0 and np.isfinite(wr) and wr > 0:
            crowd_hist[step] = n_miss / (n_corr * wr)

    step_xs = np.arange(n_steps)

    # ── Figure ────────────────────────────────────────────────────────────────
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20, 10), facecolor="k")
    gs_outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.4], wspace=0.14)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=gs_outer[1], hspace=0.55,
    )
    ax_bars  = fig.add_subplot(gs_outer[0])
    ax_frac  = fig.add_subplot(gs_right[0])
    ax_logp0 = fig.add_subplot(gs_right[1])
    ax_delta = fig.add_subplot(gs_right[2])

    _white_style_ax = [ax_bars, ax_frac, ax_logp0, ax_delta]
    for ax in _white_style_ax:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # ── Left panel: category bar chart (animated) ─────────────────────────────
    x_pos = np.arange(N_CATS)
    bar_rects = ax_bars.bar(
        x_pos, counts_hist[0], color=CAT_COLORS, width=0.65,
        edgecolor="white", linewidth=0.5,
    )
    ax_bars.set_xticks(x_pos)
    ax_bars.set_xticklabels(CAT_LABELS, color="white", fontsize=8)
    ax_bars.set_ylim(0, Nsample * 1.25)
    ax_bars.set_ylabel("Sample count", color="white")
    ax_bars_title = ax_bars.set_title(
        f"Step 1/{n_steps}  —  cost-category distribution",
        color="white", fontsize=10,
    )
    bar_pct_texts = [
        ax_bars.text(
            x_pos[k], counts_hist[0, k] + Nsample * 0.015,
            f"{fracs_hist[0, k]*100:.1f}%",
            ha="center", va="bottom", color="white", fontsize=8,
        )
        for k in range(N_CATS)
    ]
    stats_text = ax_bars.text(
        0.03, 0.98, "", transform=ax_bars.transAxes,
        color="white", fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a1a", alpha=0.9),
        fontfamily="monospace",
    )

    def _fmt_stats(step: int) -> str:
        rew_std_v = rew_std_hist[step]
        rew_mean_v = rew_mean_hist[step]
        d_lp      = delta_logp0_hist[step]
        wr        = w_ratio_hist[step]
        wf        = corr_wfrac_hist[step]
        crowd     = crowd_hist[step]
        n_c = int(counts_hist[step, 0])
        n_m = int(counts_hist[step, 2] + counts_hist[step, 3])
        return (
            f"rew_std          = {rew_std_v:.3e}\n"
            f"rew_mean          = {rew_mean_v:.3e}\n"
            f"Δlogp0 (corr−miss) = {d_lp:+.3f}\n"
            f"per-sample w ratio = {wr:.3f}×\n"
            f"corridor total w%  = {wf*100:.2f}%\n"
            f"n_corr / n_miss    = {n_c} / {n_m}\n"
            f"crowd advantage    = {crowd:.1f}×  (miss outweighs corr)"
        )

    stats_text.set_text(_fmt_stats(0))

    # ── Right top: category fractions over all steps ──────────────────────────
    for k in range(N_CATS):
        ax_frac.plot(
            step_xs, fracs_hist[:, k],
            color=CAT_COLORS[k], lw=1.5,
            label=CAT_LABELS[k].replace("\n", " "),
        )
    ax_frac.set_ylabel("Fraction of samples", color="white", fontsize=8)
    ax_frac.set_title("Category fractions vs. denoising step", color="white", fontsize=9)
    ax_frac.set_xlim(-0.5, n_steps - 0.5)
    ax_frac.set_ylim(0, 1.05)
    ax_frac.legend(fontsize=6, ncol=2, framealpha=0.35, labelcolor="white",
                   facecolor="#1a1a1a", loc="upper right")
    frac_vline = ax_frac.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)

    # ── Right middle: mean logp0 per category ─────────────────────────────────
    for k in range(N_CATS):
        ax_logp0.plot(
            step_xs, logp0_cat_hist[:, k],
            color=CAT_COLORS[k], lw=1.5,
        )
    ax_logp0.set_ylabel("Mean logp0", color="white", fontsize=8)
    ax_logp0.set_title(
        "Mean logp0 per category  (higher = more weight; note how close ~0 and ~1e6 are)",
        color="white", fontsize=9,
    )
    ax_logp0.set_xlim(-0.5, n_steps - 0.5)
    logp0_vline = ax_logp0.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)

    # ── Right bottom: Δlogp0, weight ratio, and corridor weight fraction ───────
    ax_delta_twin = ax_delta.twinx()
    ax_delta_twin.set_facecolor("#111111")
    ax_delta_twin.tick_params(colors="limegreen", labelsize=8)
    ax_delta_twin.yaxis.label.set_color("limegreen")
    for spine in ax_delta_twin.spines.values():
        spine.set_edgecolor("#444444")

    ln1, = ax_delta.plot(
        step_xs, delta_logp0_hist, color="cyan", lw=1.5,
        label="Δlogp0 (corr − miss-goal)",
    )
    ln2, = ax_delta.plot(
        step_xs, w_ratio_hist, color="yellow", lw=1.5, ls="--",
        label="per-sample w ratio  exp(Δlogp0)",
    )
    ln3, = ax_delta_twin.plot(
        step_xs, corr_wfrac_hist * 100, color="limegreen", lw=1.5,
        label="corridor total weight %",
    )
    # Also plot crowd advantage on the primary axis
    finite_mask = np.isfinite(crowd_hist)
    ln4, = ax_delta.plot(
        step_xs[finite_mask], crowd_hist[finite_mask],
        color="tomato", lw=1.5, ls=":",
        label="crowd advantage  n_miss / (n_corr × ratio)",
    )

    ax_delta.set_ylabel("Δlogp0  /  w-ratio  /  crowd", color="white", fontsize=8)
    ax_delta_twin.set_ylabel("Corridor total w%", color="limegreen", fontsize=8)
    ax_delta.set_xlabel("Denoising step", color="white", fontsize=8)
    ax_delta.set_title(
        "Per-sample advantage vs. total corridor weight  (the crowd effect)",
        color="white", fontsize=9,
    )
    ax_delta.set_xlim(-0.5, n_steps - 0.5)
    ax_delta.legend(
        handles=[ln1, ln2, ln3, ln4],
        fontsize=6, framealpha=0.35, labelcolor="white",
        facecolor="#1a1a1a", loc="upper right",
    )
    delta_vline = ax_delta.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)

    # ── Animation ─────────────────────────────────────────────────────────────
    def _update(frame: int):
        step = frame

        # Left: bar heights + percentage labels
        for k, rect in enumerate(bar_rects):
            rect.set_height(counts_hist[step, k])
        for k, txt in enumerate(bar_pct_texts):
            txt.set_y(counts_hist[step, k] + Nsample * 0.015)
            txt.set_text(f"{fracs_hist[step, k]*100:.1f}%")
        ax_bars_title.set_text(
            f"Step {step + 1}/{n_steps}  —  cost-category distribution"
        )
        stats_text.set_text(_fmt_stats(step))

        # Right: advance vertical cursor
        frac_vline.set_xdata([step, step])
        logp0_vline.set_xdata([step, step])
        delta_vline.set_xdata([step, step])

        return []

    ani = animation.FuncAnimation(fig, _update, frames=n_steps, interval=200, blit=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(out_path, writer=writer, dpi=110, savefig_kwargs={"facecolor": "k"})
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ── Spatial weight amplification movie ────────────────────────────────────────

def generate_spatial_weight_movie(
    diag: dict,
    temperature: float,
    out_path: str,
    fps: int = 5,
) -> None:
    """
    The key diagnostic for the crowd-effect question.

    Samples are categorised by x-crossing at y=_FLOE_CY into three spatial
    regions:
        Left corridor   x < FLOE_X_LO
        Middle          FLOE_X_LO ≤ x ≤ FLOE_X_HI
        Right corridor  x > FLOE_X_HI
        No crossing     trajectory doesn't cross y=_FLOE_CY (shown separately)

    For each denoising step the movie shows four panels:

    Top-left  (animated bars):
        For each region, two side-by-side bars:
          light bar  = fraction of total samples in that region
          filled bar = fraction of total softmax weight in that region
        If the softmax is amplifying corridor samples, the weight bar will be
        taller than the sample bar.  If they are equal, the softmax is blind
        to spatial location (crowd effect dominates).

    Bottom-left (animated):
        Amplification ratio = weight_fraction / sample_fraction per region.
        Ratio = 1 → softmax not amplifying; >> 1 → softmax is helping corridor.

    Top-right (static + cursor):
        Sample fraction per region over all denoising steps.

    Bottom-right (static + cursor):
        Weight fraction per region AND rew_std (twin axis) over all steps.
        Shows whether corridor weight ever rises meaningfully.
    """
    n_steps = len(diag["sample_costs"])
    if n_steps == 0:
        print("  No sample data — skipping spatial weight movie.", flush=True)
        return
    if "weights" not in diag or len(diag["weights"]) == 0:
        print("  No weight data — skipping spatial weight movie.", flush=True)
        return

    # ── Spatial regions ───────────────────────────────────────────────────────
    REG_NAMES  = ["Left\ncorridor", "Middle\n(ice)", "Right\ncorridor", "No\ncrossing"]
    REG_COLORS = ["limegreen", "tomato", "deepskyblue", "#888888"]

    def _region(cx: float) -> int:
        if np.isnan(cx):
            return 3
        if cx < FLOE_X_LO:
            return 0
        if cx > FLOE_X_HI:
            return 2
        return 1

    # ── Pre-compute per-step stats ─────────────────────────────────────────────
    Nsample = len(diag["sample_costs"][0])
    N_REG = 4

    sample_frac_hist = np.zeros((n_steps, N_REG))   # fraction of samples
    weight_frac_hist = np.zeros((n_steps, N_REG))   # fraction of total weight
    amp_ratio_hist   = np.full((n_steps, N_REG), np.nan)  # weight_frac / sample_frac
    rew_std_hist     = np.zeros(n_steps)

    for step in range(n_steps):
        trajs    = diag["sample_trajectories"][step]        # sorted best→worst
        costs_np = np.array(diag["sample_costs"][step], dtype=float)
        w_np     = np.array(diag["weights"][step], dtype=float)

        # Align weights to sorted trajectories (same logic as weight movie)
        n_stored      = len(trajs)
        _sort_indices = diag.get("sample_sort_indices", [])
        sort_idx      = _sort_indices[step] if _sort_indices else np.argsort(costs_np)
        sorted_w      = w_np[sort_idx[:n_stored]]

        # x-crossing for each stored trajectory
        cxs = np.array([_crossing_x_at_y(trajs[i], _FLOE_CY) for i in range(n_stored)])
        regions = np.array([_region(cx) for cx in cxs])

        for r in range(N_REG):
            mask = regions == r
            sample_frac_hist[step, r] = mask.sum() / n_stored
            weight_frac_hist[step, r] = float(sorted_w[mask].sum())

        # Amplification ratio (only for regions with samples)
        for r in range(N_REG):
            sf = sample_frac_hist[step, r]
            wf = weight_frac_hist[step, r]
            if sf > 1e-9:
                amp_ratio_hist[step, r] = wf / sf

        # rew_std (replicate planner's normalisation)
        rews = -costs_np
        rew_std_hist[step] = max(float(rews.std()), 1e-6)

    step_xs = np.arange(n_steps)

    # ── Figure ────────────────────────────────────────────────────────────────
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20, 10), facecolor="k")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.14)
    ax_bars  = fig.add_subplot(gs[0, 0])   # sample vs weight bars (animated)
    ax_amp   = fig.add_subplot(gs[1, 0])   # amplification ratio bars (animated)
    ax_sfrac = fig.add_subplot(gs[0, 1])   # sample fraction time-series
    ax_wfrac = fig.add_subplot(gs[1, 1])   # weight fraction + rew_std time-series

    for ax in (ax_bars, ax_amp, ax_sfrac, ax_wfrac):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # ── Top-left: sample fraction vs weight fraction bars ─────────────────────
    # For 4 regions × 2 bar types: group by region, offset within group
    bar_w  = 0.35
    x_base = np.arange(N_REG, dtype=float)
    x_samp = x_base - bar_w / 2
    x_wgt  = x_base + bar_w / 2

    bars_samp = ax_bars.bar(
        x_samp, sample_frac_hist[0], width=bar_w,
        color=[(*plt.matplotlib.colors.to_rgb(c), 0.35) for c in REG_COLORS],
        edgecolor=REG_COLORS, linewidth=1.2, label="sample fraction",
    )
    bars_wgt = ax_bars.bar(
        x_wgt, weight_frac_hist[0], width=bar_w,
        color=REG_COLORS, edgecolor="white", linewidth=0.5, label="weight fraction",
    )
    ax_bars.set_xticks(x_base)
    ax_bars.set_xticklabels(REG_NAMES, color="white", fontsize=8)
    ax_bars.set_ylim(0, 1.15)
    ax_bars.set_ylabel("Fraction of total", color="white")
    bars_title = ax_bars.set_title(
        "Step 1/%d  —  sample (hollow) vs weight (filled) fraction" % n_steps,
        color="white", fontsize=9,
    )
    ax_bars.legend(fontsize=7, framealpha=0.3, labelcolor="white",
                   facecolor="#1a1a1a", loc="upper right")
    # Stats text: show rew_std and corridor weight total
    bars_stats = ax_bars.text(
        0.02, 0.97, "", transform=ax_bars.transAxes,
        color="white", fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", alpha=0.9),
        fontfamily="monospace",
    )

    # ── Bottom-left: amplification ratio per region ────────────────────────────
    amp_bars = ax_amp.bar(
        x_base, amp_ratio_hist[0], width=0.6,
        color=REG_COLORS, edgecolor="white", linewidth=0.5,
    )
    ax_amp.axhline(1.0, color="white", lw=1.0, ls="--", alpha=0.5,
                   label="ratio = 1  (no amplification)")
    ax_amp.set_xticks(x_base)
    ax_amp.set_xticklabels(REG_NAMES, color="white", fontsize=8)
    ax_amp.set_ylabel("weight_frac / sample_frac", color="white")
    ax_amp.set_title(
        "Amplification ratio  (> 1 means softmax boosts that region)",
        color="white", fontsize=9,
    )
    ax_amp.legend(fontsize=7, framealpha=0.3, labelcolor="white", facecolor="#1a1a1a")
    amp_title = ax_amp.set_title(
        "Amplification ratio  (> 1 = softmax boosts that region, = 1 = crowd-blind)",
        color="white", fontsize=9,
    )
    # y-limit set dynamically per frame (capped to keep it readable)
    amp_ylim_pad = 1.2

    # ── Top-right: sample fraction time-series ────────────────────────────────
    for r in range(N_REG):
        ax_sfrac.plot(
            step_xs, sample_frac_hist[:, r],
            color=REG_COLORS[r], lw=1.5, label=REG_NAMES[r].replace("\n", " "),
        )
    ax_sfrac.set_xlim(-0.5, n_steps - 0.5)
    ax_sfrac.set_ylim(0, 1.05)
    ax_sfrac.set_ylabel("Sample fraction", color="white", fontsize=8)
    ax_sfrac.set_title("Sample fraction per region over denoising steps",
                        color="white", fontsize=9)
    ax_sfrac.legend(fontsize=7, ncol=2, framealpha=0.3, labelcolor="white",
                    facecolor="#1a1a1a")
    sfrac_vline = ax_sfrac.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)

    # ── Bottom-right: weight fraction + rew_std ────────────────────────────────
    ax_wfrac_twin = ax_wfrac.twinx()
    ax_wfrac_twin.set_facecolor("#111111")
    ax_wfrac_twin.tick_params(colors="#aaaaaa", labelsize=8)
    ax_wfrac_twin.yaxis.label.set_color("#aaaaaa")
    for spine in ax_wfrac_twin.spines.values():
        spine.set_edgecolor("#444444")

    for r in range(N_REG - 1):   # skip "no crossing"
        ax_wfrac.plot(
            step_xs, weight_frac_hist[:, r],
            color=REG_COLORS[r], lw=1.5, label=REG_NAMES[r].replace("\n", " "),
        )
    rew_std_line, = ax_wfrac_twin.semilogy(
        step_xs, rew_std_hist, color="#aaaaaa", lw=1.2, ls=":",
        label="rew_std (right, log)",
    )
    ax_wfrac.set_xlim(-0.5, n_steps - 0.5)
    ax_wfrac.set_ylim(0, 1.05)
    ax_wfrac.set_ylabel("Weight fraction", color="white", fontsize=8)
    ax_wfrac_twin.set_ylabel("rew_std (log scale)", color="#aaaaaa", fontsize=8)
    ax_wfrac.set_xlabel("Denoising step", color="white", fontsize=8)
    ax_wfrac.set_title(
        "Weight fraction per region  +  rew_std  (key: do corridors ever get weight?)",
        color="white", fontsize=9,
    )
    ax_wfrac.legend(fontsize=7, framealpha=0.3, labelcolor="white",
                    facecolor="#1a1a1a", loc="upper right")
    wfrac_vline = ax_wfrac.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)

    # ── Animation ─────────────────────────────────────────────────────────────
    def _update(frame: int):
        step = frame

        # Top-left: update bars
        for r, (bs, bw) in enumerate(zip(bars_samp, bars_wgt)):
            bs.set_height(sample_frac_hist[step, r])
            bw.set_height(weight_frac_hist[step, r])
        bars_title.set_text(
            f"Step {step+1}/{n_steps}  —  sample (hollow) vs weight (filled) fraction"
        )

        # Stats text
        corr_sf  = sample_frac_hist[step, 0] + sample_frac_hist[step, 2]
        corr_wf  = weight_frac_hist[step, 0] + weight_frac_hist[step, 2]
        mid_sf   = sample_frac_hist[step, 1]
        mid_wf   = weight_frac_hist[step, 1]
        rew_std  = rew_std_hist[step]
        amp_corr = amp_ratio_hist[step, 0]  # left corridor
        amp_mid  = amp_ratio_hist[step, 1]
        bars_stats.set_text(
            f"rew_std         = {rew_std:.2e}\n"
            f"corridor sample = {corr_sf*100:.1f}%   weight = {corr_wf*100:.1f}%\n"
            f"middle   sample = {mid_sf*100:.1f}%   weight = {mid_wf*100:.1f}%\n"
            f"amp(left corr)  = {amp_corr:.3f}×\n"
            f"amp(middle)     = {amp_mid:.3f}×"
        )

        # Bottom-left: amplification ratio bars
        valid_amps = amp_ratio_hist[step][np.isfinite(amp_ratio_hist[step])]
        ylim_top = max(float(valid_amps.max()) * amp_ylim_pad, 2.0) if len(valid_amps) else 2.0
        ax_amp.set_ylim(0, ylim_top)
        for r, bar in enumerate(amp_bars):
            val = amp_ratio_hist[step, r]
            bar.set_height(val if np.isfinite(val) else 0.0)

        # Right panels: advance cursor
        sfrac_vline.set_xdata([step, step])
        wfrac_vline.set_xdata([step, step])

        return []

    ani = animation.FuncAnimation(fig, _update, frames=n_steps, interval=200, blit=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(out_path, writer=writer, dpi=110, savefig_kwargs={"facecolor": "k"})
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def _generate_al_multiplier_plot(diag: dict, out_path: str) -> None:
    """
    Static plot showing how AL-MBD multipliers and constraint violations evolve
    over the reverse diffusion steps.
    """
    mul_means = diag.get("multiplier_means", [])
    mul_maxes = diag.get("multiplier_maxes", [])
    violations = diag.get("constraint_violations", [])

    if not mul_means:
        print("  [AL-MBD plot] no multiplier data — skipping.", flush=True)
        return

    steps = np.arange(len(mul_means))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax0 = axes[0]
    ax0.plot(steps, mul_means, label="Mean multiplier", color="steelblue")
    ax0.plot(steps, mul_maxes, label="Max multiplier",  color="tomato", ls="--")
    ax0.set_ylabel("Multiplier M")
    ax0.set_title("AL-MBD: Multiplier evolution across diffusion steps")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(steps, violations, color="darkorange", label="Mean constraint violation")
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_ylabel("Violation (m, mean)")
    ax1.set_xlabel("Diffusion step (N → 1)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_eb_barrier_movie(diag: dict, out_path: str) -> None:
    """
    Diagnostic movie for EB-MBD showing the barrier schedule evolution.

    Three panels (static time-series + animated cursor):
      Left   — c_s (barrier offset) vs denoising step
      Middle — alive fraction (fraction of samples with g(x)+c_s > 0) vs step
      Right  — constraint value distribution (box-and-whiskers) per step

    Only generated when diagnostics include EB-MBD extras (barrier_offsets key).
    """
    if "barrier_offsets" not in diag or not diag["barrier_offsets"]:
        print("  No EB-MBD barrier data — skipping barrier movie.", flush=True)
        return

    n_steps       = len(diag["barrier_offsets"])
    offsets       = np.array(diag["barrier_offsets"])           # (n_steps,)
    alive_fracs   = np.array(diag["alive_fracs"])               # (n_steps,)
    g_vals_list   = diag["constraint_values"]                   # list of (Nsample,)

    step_xs = np.arange(n_steps)

    # Pre-compute percentiles for constraint distribution
    g_p10  = np.array([np.percentile(g, 10)  for g in g_vals_list])
    g_p25  = np.array([np.percentile(g, 25)  for g in g_vals_list])
    g_p50  = np.array([np.percentile(g, 50)  for g in g_vals_list])
    g_p75  = np.array([np.percentile(g, 75)  for g in g_vals_list])
    g_p90  = np.array([np.percentile(g, 90)  for g in g_vals_list])
    g_min  = np.array([np.min(g)             for g in g_vals_list])
    g_max  = np.array([np.max(g)             for g in g_vals_list])

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

    # Left: c_s schedule
    ax_c.plot(step_xs, offsets, color="cyan", lw=2)
    ax_c.set_xlabel("Denoising step (S → 0)")
    ax_c.set_ylabel("Barrier offset  c_s  [m]")
    ax_c.set_title("Barrier offset schedule  c_s")
    ax_c.set_xlim(-0.5, n_steps - 0.5)
    ax_c.set_ylim(bottom=0)
    vline_c = ax_c.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)

    # Middle: alive fraction
    ax_alive.plot(step_xs, alive_fracs * 100, color="limegreen", lw=2)
    ax_alive.axhline(100, color="#444444", lw=1, ls=":")
    ax_alive.axhline(50,  color="#444444", lw=1, ls=":")
    ax_alive.set_xlabel("Denoising step (S → 0)")
    ax_alive.set_ylabel("Alive samples  [%]")
    ax_alive.set_title("Fraction with  g(x) + c_s > 0")
    ax_alive.set_xlim(-0.5, n_steps - 0.5)
    ax_alive.set_ylim(0, 105)
    vline_alive = ax_alive.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)

    # Right: constraint value distribution
    ax_g.fill_between(step_xs, g_min,  g_p10,  color="tomato",     alpha=0.15)
    ax_g.fill_between(step_xs, g_p10,  g_p25,  color="gold",       alpha=0.20)
    ax_g.fill_between(step_xs, g_p25,  g_p75,  color="limegreen",  alpha=0.30)
    ax_g.fill_between(step_xs, g_p75,  g_p90,  color="gold",       alpha=0.20)
    ax_g.fill_between(step_xs, g_p90,  g_max,  color="tomato",     alpha=0.15)
    ax_g.plot(step_xs, g_p50, color="white", lw=1.5, label="median g(x)")
    # Overlay −c_s to show where the "alive" threshold is
    ax_g.plot(step_xs, -offsets, color="cyan", lw=1.5, ls="--",
              label=r"−c_s  (dead threshold)")
    ax_g.axhline(0, color="#888888", lw=1, ls=":")
    ax_g.set_xlabel("Denoising step (S → 0)")
    ax_g.set_ylabel("g(x)  [m]")
    ax_g.set_title("Constraint distribution  g(x) = min(g_OOB, g_goal)")
    ax_g.set_xlim(-0.5, n_steps - 0.5)
    ax_g.legend(fontsize=7, labelcolor="white", facecolor="#1a1a1a",
                framealpha=0.6)
    vline_g = ax_g.axvline(0, color="white", lw=1.5, ls="--", alpha=0.7)

    fig.suptitle("EB-MBD barrier schedule diagnostics", color="white",
                 fontsize=13, y=1.01)

    def _update(frame: int):
        vline_c.set_xdata([frame, frame])
        vline_alive.set_xdata([frame, frame])
        vline_g.set_xdata([frame, frame])
        return vline_c, vline_alive, vline_g

    ani = animation.FuncAnimation(fig, _update, frames=n_steps, interval=200,
                                   blit=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=5, bitrate=2000)
    ani.save(out_path, writer=writer, dpi=100, savefig_kwargs={"facecolor": "k"})
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="MBD bimodal collapse demo")
    parser.add_argument("--output-dir", default="outputs/toy_mbd_bimodal")
    parser.add_argument("--nsample",     type=int,   default=512)
    parser.add_argument("--ndiffuse",    type=int,   default=50)
    parser.add_argument("--temperature",    type=float, default=0.01)
    parser.add_argument("--goal-x-weight", type=float, default=0.0,
                        help="Weight for soft x-deviation cost at goal-line crossing (default: 0)")
    parser.add_argument("--length-weight", type=float, default=0.0,
                        help="Weight for path-length soft cost (default: 0)")
    parser.add_argument("--use-best-sample", action="store_true",
                        help="Return global best sample instead of Ybar_final")
    parser.add_argument(
        "--reject-sampling", action="store_true",
        help=(
            "Enable reject sampling: at each denoising step, filter out samples "
            "that are OOB or fail to reach the goal line. Only the remaining "
            "feasible samples are used for cost/weight computation. No resampling "
            "with replacement — invalid samples are simply discarded."
        ),
    )
    parser.add_argument(
        "--max-resample-iters", type=int, default=None,
        help="Ignored (kept for CLI compatibility; filter-once mode does not resample).",
    )
    # ── EB-MBD args ────────────────────────────────────────────────────────
    parser.add_argument(
        "--eb-mbd", action="store_true",
        help=(
            "Enable Emerging-Barrier MBD (EB-MBD).  Replaces big-M OOB and "
            "goal-line penalties with a progressively tightening log-barrier. "
            "See Mishra & Manchester (2026)."
        ),
    )
    parser.add_argument(
        "--eb-c-max", type=float, default=1500.0,
        help=(
            "EB-MBD barrier offset at the start of the reverse diffusion [m]. "
            "Should be ≥ max expected constraint violation (goal miss ~1350 m). "
            "Default: 1500."
        ),
    )
    parser.add_argument(
        "--eb-mu", type=float, default=50000.0,
        help=(
            "EB-MBD barrier hardness in cost units.  Scales the log-barrier "
            "relative to the base cost J(x).  Default: 50000."
        ),
    )
    parser.add_argument(
        "--eb-kappa", type=float, default=1.0,
        help=(
            "EB-MBD barrier schedule shape.  κ=1 → linear; κ>1 → slow then "
            "fast tightening towards the end.  Default: 1.0."
        ),
    )
    # ── AL-MBD args ────────────────────────────────────────────────────────
    parser.add_argument(
        "--al-mbd", action="store_true",
        help=(
            "Enable Augmented Lagrangian MBD (AL-MBD).  Replaces the hard "
            "indicator weight with a smooth AL surrogate and updates Lagrange "
            "multipliers across diffusion steps via dual ascent."
        ),
    )
    parser.add_argument(
        "--al-rho", type=float, default=0.1,
        help=(
            "AL-MBD penalty weight ρ.  Controls how fast constraint pressure "
            "builds per dual step.  Default: 0.1."
        ),
    )
    parser.add_argument(
        "--al-M-max", type=float, default=100.0,
        help=(
            "AL-MBD multiplier clip value.  Prevents blow-up during noisy "
            "early diffusion.  Default: 100."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────
    run_cfg = get_default_run_config()
    run_cfg = replace(
        run_cfg,
        planning_jax_mbd_nsample=args.nsample,
        planning_jax_mbd_ndiffuse=args.ndiffuse,
        planning_jax_mbd_temperature=args.temperature,
        planning_jax_mbd_use_best_sample=args.use_best_sample,
        # Horizon: 1300 m at ~4 m/s ≈ 325 s → H=650 with dt=1s gives comfortable coverage
        planning_jax_mbd_horizon=750,
        planning_jax_mbd_n_substeps=1,
        planning_jax_mbd_dt_substep=1.0,
        planning_jax_mbd_goal_x_weight=args.goal_x_weight,
        planning_jax_mbd_length_weight=args.length_weight,
    )

    # ── Costmap ───────────────────────────────────────────────────────────
    print("Building costmap...", flush=True)
    cost_map = _build_costmap(run_cfg)
    plot_costmap(cost_map, str(out_dir / "bimodal_costmap.png"))

    # ── Optional reject-sampling filter ──────────────────────────────────
    # Validity mirrors exactly what the cost function penalises:
    #   OOB  — any hull-corner x exits the costmap grid (the _make_query
    #           function returns 1e8 for x outside the grid x-extent, which
    #           accumulates to ~1e10 when many waypoints are affected).
    #           Checking only the ship *centre* against [X_MIN, X_MAX] is
    #           insufficient: with half_l=79 m a centre at x=50 heading west
    #           puts the bow at x=-29, outside the costmap (x_origin=-20).
    #   Goal  — last waypoint y >= GOAL_LINE, matching make_goal_line_cost_fn
    #           which checks y_final = traj[-1, 1] (not any-waypoint).
    sample_filter_fn = None
    if args.reject_sampling:
        # Costmap x-extent (the _make_query OOB boundary)
        _x_cost_lo = float(cost_map.x_origin)
        _x_cost_hi = (float(cost_map.x_origin)
                      + float(cost_map.cols) * float(cost_map.resolution))
        _y_min = float(Y_MIN)
        _goal  = float(GOAL_LINE)

        # Hull corners used by _make_raw_components_fn (ship_length=158m, beam=28m)
        _half_l = run_cfg.ship_length / 2.0   # 79 m
        _half_w = run_cfg.ship_beam   / 2.0   # 14 m
        _dx = np.array([0.0, _half_l, _half_l, -_half_l, -_half_l], dtype=np.float32)
        _dy = np.array([0.0, _half_w, -_half_w, _half_w, -_half_w], dtype=np.float32)

        def sample_filter_fn(Y0s_norm: np.ndarray, trajs: np.ndarray) -> np.ndarray:
            """
            Return bool mask: True = non-OOB (hull corners inside costmap)
            AND goal-achieving (last waypoint y >= GOAL_LINE).

            OOB uses hull corners, not just ship centre, to match the
            bilinear-interpolation query in _make_raw_components_fn which
            samples the costmap at 5 hull-corner points per waypoint.
            Goal uses y_final (last step) to match make_goal_line_cost_fn.
            """
            xs  = trajs[:, :, 0]   # (N, H+1) ship centre x
            ys  = trajs[:, :, 1]   # (N, H+1) ship centre y
            psi = trajs[:, :, 2]   # (N, H+1) heading

            # Hull corner x in world frame: (N, H+1, 5)
            c = np.cos(psi)        # (N, H+1)
            s = np.sin(psi)        # (N, H+1)
            hull_x = (xs[:, :, None]
                      + c[:, :, None] * _dx[None, None, :]
                      - s[:, :, None] * _dy[None, None, :])

            # OOB: any hull corner outside the costmap x-extent
            hull_oob = ((hull_x < _x_cost_lo) | (hull_x > _x_cost_hi)).any(axis=(1, 2))
            # OOB: ship centre goes below y floor
            y_oob = (ys < _y_min).any(axis=1)
            oob = hull_oob | y_oob

            # Goal: last-waypoint y (consistent with make_goal_line_cost_fn)
            reaches_goal = ys[:, -1] >= _goal

            return (~oob) & reaches_goal

        print(
            f"\n  [reject-sampling] enabled (filter-once, no replacement)  "
            f"(hull OOB check: x∈[{_x_cost_lo:.0f},{_x_cost_hi:.0f}], centre y≥{Y_MIN}  "
            f"|  goal: y_final≥{GOAL_LINE})",
            flush=True,
        )

    # ── MBD / EB-MBD (cold start — warm-start would bias one side) ──────────
    denoising_dir = str(out_dir / "denoising")
    _rs_label = " + reject-sampling" if args.reject_sampling else ""
    _eb_label = " + EB-MBD barrier" if args.eb_mbd else ""
    _al_label = " + AL-MBD" if args.al_mbd else ""
    if args.al_mbd:
        _algo_name = "AL-MBD"
    elif args.eb_mbd:
        _algo_name = "EB-MBD"
    else:
        _algo_name = "MBD"
    print(
        f"\nRunning {_algo_name} "
        f"(cold start{_rs_label}{_eb_label}{_al_label}, diagnostics + denoising movie)...",
        flush=True,
    )
    _t0 = time.perf_counter()
    traj, diag = jax_mbd_optimize(
        cost_map=cost_map,
        start_pose=START,
        goal=GOAL,
        run_cfg=run_cfg,
        start_nu=(2.0, 0.0, 0.0),   # 2 m/s surge (forward = north), no sway
        lattice_path=None,
        warm_start=False,            # cold start: no A* bias toward either side
        return_diagnostics=True,
        goal_y_line=GOAL_LINE,
        goal_y_line_penalty=1e6,
        denoising_movie_dir=denoising_dir,
        cost_truncate_y=Y_MAX,       # no cost/OOB past the top of the map
        sample_filter_fn=sample_filter_fn,
        max_resample_iters=args.max_resample_iters,
        # EB-MBD
        use_eb_mbd=args.eb_mbd,
        eb_x_min=X_MIN,
        eb_x_max=X_MAX,
        eb_y_min=Y_MIN,
        eb_c_max=args.eb_c_max,
        eb_mu=args.eb_mu,
        eb_kappa=args.eb_kappa,
        # AL-MBD
        use_al_mbd=args.al_mbd,
        al_x_min=X_MIN,
        al_x_max=X_MAX,
        al_y_min=Y_MIN,
        al_rho=args.al_rho,
        al_M_max=args.al_M_max,
    )
    planning_time = time.perf_counter() - _t0

    # ── Extract trajectories from diagnostics ─────────────────────────────
    # Mean trajectory = final Ybar_c rolled out
    mean_traj = np.array(diag["mean_trajectories"][-1])   # (H+1, 3)

    # Cost of the mean = cost_of_mean at the last step
    mean_cost = diag["cost_of_mean"][-1] if diag["cost_of_mean"] else float("nan")

    # Global best sample trajectory.
    # For AL-MBD / EB-MBD: use the sample tracked by minimum BASE cost so the
    # displayed cost matches the plotted trajectory exactly (sample_trajectories
    # is sorted by total AL/barrier cost and can differ from min-base-cost).
    # For vanilla MBD: keep the existing step-based selection.
    if (args.al_mbd or args.eb_mbd) and diag.get("best_base_cost_trajectory") is not None:
        best_cost = diag["best_base_cost"]
        best_traj = diag["best_base_cost_trajectory"]   # (H+1, 3)
        global_min_step = -1   # not step-based for AL-MBD
    else:
        cost_mins_arr = np.array(diag["cost_mins"])
        global_min_step = int(cost_mins_arr.argmin())
        best_cost = float(cost_mins_arr[global_min_step])
        best_traj = diag["sample_trajectories"][global_min_step][0]   # (H+1, 3)

    # ── Console summary ───────────────────────────────────────────────────
    ratio = mean_cost / max(best_cost, 1.0)
    print(f"\n{'='*60}", flush=True)
    print(f"  MBD mean trajectory cost : {mean_cost:.1f}", flush=True)
    _step_label = "global best-base-cost" if global_min_step == -1 else f"step {global_min_step}/{len(diag['cost_mins'])-1}"
    print(f"  Global best sample cost  : {best_cost:.2f}  ({_step_label})", flush=True)
    print(f"  Collapse ratio           : {ratio:.2f}×", flush=True)
    if ratio > 1.5:
        print("  --> MULTIMODAL COLLAPSE: mean is >1.5× worse than best sample", flush=True)
    else:
        print("  --> No severe collapse (ratio close to 1)", flush=True)
    if args.reject_sampling and diag.get("accept_rates"):
        _ar = np.array(diag["accept_rates"])
        print(
            f"  Reject-filter keep rate: "
            f"min={_ar.min()*100:.1f}%  mean={_ar.mean()*100:.1f}%  max={_ar.max()*100:.1f}%",
            flush=True,
        )
    print(f"{'='*60}\n", flush=True)

    # ── Result plot ───────────────────────────────────────────────────────
    plot_result(
        cost_map, mean_traj, best_traj,
        mean_cost, best_cost,
        str(out_dir / "bimodal_result.png"),
    )

    # ── Bimodal histogram movie ───────────────────────────────────────────
    print("\nGenerating bimodal histogram movie (sample counts)...", flush=True)
    generate_bimodal_histogram_movie(
        diag, cost_map,
        str(out_dir / "bimodal_histogram.mp4"),
    )

    # ── Bimodal weight movie ──────────────────────────────────────────────
    print("\nGenerating bimodal weight movie (softmax weights)...", flush=True)
    generate_bimodal_weight_movie(
        diag, cost_map,
        str(out_dir / "bimodal_weight_histogram.mp4"),
    )

    # ── Bimodal cost movie (diagnostic) ──────────────────────────────────
    print("\nGenerating bimodal cost movie (mean cost per bin)...", flush=True)
    generate_bimodal_cost_movie(
        diag, cost_map,
        str(out_dir / "bimodal_cost_histogram.mp4"),
    )

    # ── Bimodal min-cost movie (diagnostic) ──────────────────────────────
    print("\nGenerating bimodal min-cost movie (min cost per bin)...", flush=True)
    generate_bimodal_min_cost_movie(
        diag, cost_map,
        str(out_dir / "bimodal_min_cost_histogram.mp4"),
    )

    # ── Cost-category + softmax analysis movie ───────────────────────────────
    print("\nGenerating cost-category movie (softmax analysis)...", flush=True)
    generate_cost_category_movie(
        diag,
        temperature=args.temperature,
        out_path=str(out_dir / "cost_category_analysis.mp4"),
        goal_penalty=1e6,
        oob_thresh=1e8,
    )

    # ── Spatial weight amplification movie ───────────────────────────────────
    print("\nGenerating spatial weight movie (corridor vs middle amplification)...", flush=True)
    generate_spatial_weight_movie(
        diag,
        temperature=args.temperature,
        out_path=str(out_dir / "spatial_weight_amplification.mp4"),
    )

    # ── EB-MBD barrier schedule movie (only when EB-MBD was used) ────────────
    if args.eb_mbd:
        print("\nGenerating EB-MBD barrier schedule movie...", flush=True)
        generate_eb_barrier_movie(
            diag,
            out_path=str(out_dir / "eb_barrier_schedule.mp4"),
        )

    # ── AL-MBD multiplier evolution plot (only when AL-MBD was used) ─────────
    if args.al_mbd:
        print("\nGenerating AL-MBD multiplier evolution plot...", flush=True)
        _generate_al_multiplier_plot(
            diag,
            out_path=str(out_dir / "al_multiplier_evolution.png"),
        )
        # Brief summary of multiplier and constraint stats
        m_means = np.array(diag.get("multiplier_means", []))
        m_maxes = np.array(diag.get("multiplier_maxes", []))
        cviol   = np.array(diag.get("constraint_violations", []))
        if len(m_means):
            print(f"  AL-MBD summary: mult_mean {m_means[0]:.4f}->{m_means[-1]:.4f}"
                  f"  mult_max {m_maxes[0]:.4f}->{m_maxes[-1]:.4f}"
                  f"  c_viol {cviol[0]:.3f}->{cviol[-1]:.3f}", flush=True)
        _mt = np.array(diag["mean_trajectories"][-1])
        print(f"  Final mean traj: y∈[{_mt[:,1].min():.1f}, {_mt[:,1].max():.1f}]"
              f"  y_final={_mt[-1,1]:.1f}  x@floe={_mt[int(len(_mt)*590/1400),0]:.1f}"
              f"  (floe x: [{FLOE_X_LO:.0f}, {FLOE_X_HI:.0f}])", flush=True)

    # ── Planning metrics table ────────────────────────────────────────────
    _mt = np.array(diag["mean_trajectories"][-1])   # (H+1, 3)  [x, y, psi]
    path_length = float(np.sum(np.linalg.norm(np.diff(_mt[:, :2], axis=0), axis=1)))
    # Raw costmap cost along the trajectory — same evaluation for all methods.
    # Sum get_cost_world at each waypoint (step ≈ resolution, no interpolation needed).
    traj_cost = float(sum(cost_map.get_cost_world(float(x), float(y))
                          for x, y in _mt[:, :2]))
    # x position when the trajectory first crosses GOAL_LINE (NaN if it never does)
    _ys = _mt[:, 1]
    _cross_idx = int(np.argmax(_ys >= GOAL_LINE)) if np.any(_ys >= GOAL_LINE) else -1
    x_at_goal = float(_mt[_cross_idx, 0]) if _cross_idx >= 0 else float("nan")
    reaches_goal = _cross_idx >= 0

    print(f"\n{'─'*56}", flush=True)
    print(f"  PLANNING METRICS  ({_algo_name})", flush=True)
    print(f"{'─'*56}", flush=True)
    print(f"  Planning time     : {planning_time:.1f} s", flush=True)
    print(f"  Trajectory cost   : {traj_cost:.1f}  (raw costmap, method-agnostic)", flush=True)
    print(f"  Path length       : {path_length:.1f} m", flush=True)
    print(f"  Reaches goal      : {'yes' if reaches_goal else 'NO'}", flush=True)
    if reaches_goal:
        print(f"  x at goal line    : {x_at_goal:.1f} m  (goal x = {GOAL[0]:.0f} m)", flush=True)
    print(f"{'─'*56}\n", flush=True)

    print(f"Done. Outputs in {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
