"""
Test and comparison of baseline planners: RSS, iLQR, MPPI, and MBD.

Builds a synthetic ice field, constructs a costmap, runs A* lattice planning
to obtain a reference path, then runs each trajectory optimizer. Plots all
trajectories overlaid on the same costmap and prints timing/cost summaries.

Output images saved to outputs/test_baseline_planners/.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# ── project root on path ──────────────────────────────────────────────────────
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
    tessellate_all_floes,
    create_proximity_bonds,
    polygon_area,
    polygon_centroid,
)
from medea_ice_ship.config import get_default_run_config, get_default_simulation_config
from medea_ice_ship.ice_field_generator import generate_ice_field

OUT_DIR = ROOT / "outputs" / "test_baseline_planners"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── build synthetic ice field ─────────────────────────────────────────────────

def build_test_field(concentration: float = 0.35, seed: int = 7):
    """Generate a small ice field suitable for fast planner testing."""
    field = generate_ice_field(
        concentration=concentration,
        map_shape=(800.0, 800.0),
        ice_start_y=50.0,
        min_radius=15.0,
        max_radius=80.0,
        seed=seed,
    )
    return field


# ── helpers ───────────────────────────────────────────────────────────────────

def _plot_costmap_ax(ax, cost_map, domain):
    """Render costmap as background image on ax."""
    cm = cost_map.cost_map.copy()
    cm = np.clip(cm, 0, np.percentile(cm[cm > 0], 95) if (cm > 0).any() else 1)
    extent = [
        cost_map.x_origin,
        cost_map.x_origin + cost_map.width,
        cost_map.y_origin,
        cost_map.y_origin + cost_map.height,
    ]
    ax.imshow(cm, origin="lower", extent=extent, cmap="YlOrRd", alpha=0.85, aspect="equal")


def _plot_path(ax, path, label, color, lw=2.0, zorder=4):
    if path is not None and len(path) >= 2:
        p = np.asarray(path)
        ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=lw, label=label, zorder=zorder)


def plot_comparison(cost_map, domain, start, goal, paths_dict, out_path):
    """
    Plot all planner trajectories overlaid on the costmap.

    paths_dict: {planner_name: (path_array or None, elapsed_s, metric)}
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    _plot_costmap_ax(ax, cost_map, domain)

    colors = {
        "lattice":          "#1f77b4",
        "mppi":             "#ff7f0e",
        "diffusion-mpc":    "#2ca02c",
        "random-shooting":  "#d62728",
        "ilqr":             "#9467bd",
    }
    lws = {
        "lattice": 1.5,
        "mppi": 2.0,
        "diffusion-mpc": 2.0,
        "random-shooting": 2.0,
        "ilqr": 2.0,
    }

    for name, (path, elapsed, metric) in paths_dict.items():
        clr = colors.get(name, "black")
        lw = lws.get(name, 2.0)
        label = f"{name}  [{elapsed:.1f}s, cost={metric:.1f}]"
        _plot_path(ax, path, label=label, color=clr, lw=lw)

    ax.plot(start[0], start[1], "gs", ms=12, zorder=8, label="Start")
    ax.plot(goal[0], goal[1], "rx", ms=14, mew=3, zorder=8, label="Goal")

    x_lo, x_hi, y_lo, y_hi = domain
    pad = 30.0
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Planner comparison — costmap background")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison figure: {out_path}")


# ── main test ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Baseline planner test")
    print("=" * 60)

    # ── 1. Build ice field ────────────────────────────────────────────
    print("\n[1] Generating ice field (concentration=0.35)...")
    field = build_test_field(concentration=0.35, seed=7)
    polygons_sat = field["polygons"]
    centroids_sat = field["centroids"]
    thicknesses_sat = field["thicknesses"]
    domain = field["domain"]
    x_lo, x_hi, y_lo, y_hi = domain
    print(f"    {len(polygons_sat)} polygons, domain [{x_lo:.0f},{x_hi:.0f}] x [{y_lo:.0f},{y_hi:.0f}] m")

    # ── 2. Tessellate ─────────────────────────────────────────────────
    print("\n[2] Tessellating floes...")
    run_cfg = get_default_run_config()
    sim_cfg = get_default_simulation_config()

    # Use low tessellation density for speed
    import dataclasses
    run_cfg = dataclasses.replace(
        run_cfg,
        cells_per_100m2_medea=0.5,
        # Reduce horizons for fast test
        planning_rss_horizon=100,
        planning_rss_nsample=1024,
        planning_ilqr_horizon=50,
        planning_ilqr_max_iters=10,
        planning_jax_mbd_horizon=100,
        planning_mppi_horizon=100,
        planning_mppi_nsample=512,
    )

    polygons, centroids, parent_ids = tessellate_all_floes(
        polygons_sat, centroids_sat,
        cells_per_100m2=run_cfg.cells_per_100m2_medea,
        seed=42,
    )
    areas = np.array([polygon_area(p) for p in polygons])
    radii = np.sqrt(areas / np.pi)
    bonds = create_proximity_bonds(centroids, radii, threshold_factor=run_cfg.bond_threshold_factor)
    thicknesses = np.array([thicknesses_sat[pid] for pid in parent_ids])
    print(f"    {len(polygons)} tessellated cells")

    # ── 3. Build costmap ──────────────────────────────────────────────
    print("\n[3] Building costmap...")
    t0 = time.perf_counter()
    wall_bounds = (x_lo, x_hi, y_lo, y_hi)
    cost_map, bounds = build_costmap_from_ice_state(
        polygons_sat, centroids_sat, thicknesses_sat,
        x_lo, x_hi, y_lo, y_hi,
        resolution=run_cfg.planning_resolution,
        margin=run_cfg.planning_margin,
        ship_mass=run_cfg.ship_displacement,
        collision_cost=run_cfg.planning_ice_collision_cost,
        wall_bounds=wall_bounds,
        ship_length=run_cfg.ship_length,
        ship_beam=run_cfg.ship_beam,
    )
    print(f"    Costmap built in {time.perf_counter() - t0:.2f}s — "
          f"shape ({cost_map.rows}, {cost_map.cols}), res={cost_map.resolution}m")

    # ── 4. Define start / goal ────────────────────────────────────────
    start = (0.0, y_lo + 20.0, np.pi / 2)   # (x, y, psi) — heading north
    goal  = (0.0, y_hi - 20.0)               # (x, y)
    print(f"\n[4] Start: {start}, Goal: {goal}")

    # ── 5. A* lattice path ────────────────────────────────────────────
    print("\n[5] Running A* lattice planner...")
    t0 = time.perf_counter()
    planner = create_medea_lattice_planner(
        cost_map=cost_map, bounds=bounds,
        start=start, goal=goal, run_cfg=run_cfg,
    )
    lattice_path = plan_path(planner, start, goal, run_cfg, cost_map=cost_map)
    lattice_elapsed = time.perf_counter() - t0
    if lattice_path is None:
        print("  WARNING: A* failed to find a path — using straight line fallback")
        lattice_path = np.array([
            [start[0], start[1], start[2]],
            [goal[0],  goal[1],  np.pi / 2],
        ])
    lattice_cost = float(cost_map.get_path_cost(lattice_path))
    print(f"    A* done in {lattice_elapsed:.2f}s, {len(lattice_path)} waypoints, "
          f"costmap integral={lattice_cost:.1f}")

    # Save lattice plot
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_costmap_ax(ax, cost_map, domain)
    _plot_path(ax, lattice_path, "A* lattice", "#1f77b4", lw=2.0)
    ax.plot(start[0], start[1], "gs", ms=12, zorder=8, label="Start")
    ax.plot(goal[0], goal[1], "rx", ms=14, mew=3, zorder=8, label="Goal")
    ax.set_xlim(x_lo - 30, x_hi + 30); ax.set_ylim(y_lo - 30, y_hi + 30)
    ax.set_aspect("equal"); ax.set_title("A* Lattice Path"); ax.legend()
    fig.savefig(OUT_DIR / "01_lattice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {OUT_DIR / '01_lattice.png'}")

    # ── 6. Run each optimizer ─────────────────────────────────────────
    results = {"lattice": (lattice_path, lattice_elapsed, lattice_cost)}

    def run_optimizer(name, opt_fn):
        print(f"\n[{name}] Running optimizer...")
        t0 = time.perf_counter()
        try:
            path = opt_fn(
                lattice_path, start, goal, cost_map, run_cfg,
                start_nu=(2.0, 0.0, 0.0),
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
            return None, 0.0, float("inf")
        elapsed = time.perf_counter() - t0
        if path is not None and len(path) >= 2:
            cost = float(cost_map.get_path_cost(np.asarray(path)))
        else:
            cost = float("inf")
        print(f"  {name} done in {elapsed:.2f}s, {len(path) if path is not None else 0} pts, "
              f"costmap integral={cost:.1f}")
        return path, elapsed, cost

    # Random Single Shooting
    rss_path, rss_elapsed, rss_cost = run_optimizer("random-shooting", optimize_with_rss)
    results["random-shooting"] = (rss_path, rss_elapsed, rss_cost)

    if rss_path is not None:
        fig, ax = plt.subplots(figsize=(10, 10))
        _plot_costmap_ax(ax, cost_map, domain)
        _plot_path(ax, lattice_path, "A* lattice", "#1f77b4", lw=1.5)
        _plot_path(ax, rss_path, "RSS", "#d62728", lw=2.5)
        ax.plot(start[0], start[1], "gs", ms=12, zorder=8, label="Start")
        ax.plot(goal[0], goal[1], "rx", ms=14, mew=3, zorder=8, label="Goal")
        ax.set_xlim(x_lo - 30, x_hi + 30); ax.set_ylim(y_lo - 30, y_hi + 30)
        ax.set_aspect("equal")
        ax.set_title(f"Random Single Shooting  ({rss_elapsed:.1f}s, cost={rss_cost:.1f})")
        ax.legend()
        fig.savefig(OUT_DIR / "02_rss.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {OUT_DIR / '02_rss.png'}")

    # iLQR
    ilqr_path, ilqr_elapsed, ilqr_cost = run_optimizer("ilqr", optimize_with_ilqr)
    results["ilqr"] = (ilqr_path, ilqr_elapsed, ilqr_cost)

    if ilqr_path is not None:
        fig, ax = plt.subplots(figsize=(10, 10))
        _plot_costmap_ax(ax, cost_map, domain)
        _plot_path(ax, lattice_path, "A* lattice", "#1f77b4", lw=1.5)
        _plot_path(ax, ilqr_path, "iLQR", "#9467bd", lw=2.5)
        ax.plot(start[0], start[1], "gs", ms=12, zorder=8, label="Start")
        ax.plot(goal[0], goal[1], "rx", ms=14, mew=3, zorder=8, label="Goal")
        ax.set_xlim(x_lo - 30, x_hi + 30); ax.set_ylim(y_lo - 30, y_hi + 30)
        ax.set_aspect("equal")
        ax.set_title(f"Box-iLQR  ({ilqr_elapsed:.1f}s, cost={ilqr_cost:.1f})")
        ax.legend()
        fig.savefig(OUT_DIR / "03_ilqr.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {OUT_DIR / '03_ilqr.png'}")

    # MPPI (comparison)
    mppi_path, mppi_elapsed, mppi_cost = run_optimizer("mppi", optimize_with_mppi)
    results["mppi"] = (mppi_path, mppi_elapsed, mppi_cost)

    if mppi_path is not None:
        fig, ax = plt.subplots(figsize=(10, 10))
        _plot_costmap_ax(ax, cost_map, domain)
        _plot_path(ax, lattice_path, "A* lattice", "#1f77b4", lw=1.5)
        _plot_path(ax, mppi_path, "MPPI", "#ff7f0e", lw=2.5)
        ax.plot(start[0], start[1], "gs", ms=12, zorder=8, label="Start")
        ax.plot(goal[0], goal[1], "rx", ms=14, mew=3, zorder=8, label="Goal")
        ax.set_xlim(x_lo - 30, x_hi + 30); ax.set_ylim(y_lo - 30, y_hi + 30)
        ax.set_aspect("equal")
        ax.set_title(f"MPPI  ({mppi_elapsed:.1f}s, cost={mppi_cost:.1f})")
        ax.legend()
        fig.savefig(OUT_DIR / "04_mppi.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {OUT_DIR / '04_mppi.png'}")

    # MBD (comparison)
    mbd_path, mbd_elapsed, mbd_cost = run_optimizer("diffusion-mpc", optimize_with_diffusion)
    results["diffusion-mpc"] = (mbd_path, mbd_elapsed, mbd_cost)

    if mbd_path is not None:
        fig, ax = plt.subplots(figsize=(10, 10))
        _plot_costmap_ax(ax, cost_map, domain)
        _plot_path(ax, lattice_path, "A* lattice", "#1f77b4", lw=1.5)
        _plot_path(ax, mbd_path, "MBD (diffusion-mpc)", "#2ca02c", lw=2.5)
        ax.plot(start[0], start[1], "gs", ms=12, zorder=8, label="Start")
        ax.plot(goal[0], goal[1], "rx", ms=14, mew=3, zorder=8, label="Goal")
        ax.set_xlim(x_lo - 30, x_hi + 30); ax.set_ylim(y_lo - 30, y_hi + 30)
        ax.set_aspect("equal")
        ax.set_title(f"MBD diffusion-mpc  ({mbd_elapsed:.1f}s, cost={mbd_cost:.1f})")
        ax.legend()
        fig.savefig(OUT_DIR / "05_mbd.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {OUT_DIR / '05_mbd.png'}")

    # ── 7. Comparison figure ──────────────────────────────────────────
    print("\n[7] Saving comparison figure...")
    plot_comparison(
        cost_map, domain, start, goal, results,
        out_path=OUT_DIR / "06_comparison.png",
    )

    # ── 8. Print summary table ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  {'Planner':<20}  {'Elapsed (s)':>12}  {'Costmap integral':>17}  {'Pts':>6}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*17}  {'-'*6}")
    for name, (path, elapsed, cost) in results.items():
        npts = len(path) if path is not None else 0
        cost_str = f"{cost:.1f}" if cost < float("inf") else "FAILED"
        print(f"  {name:<20}  {elapsed:>12.2f}  {cost_str:>17}  {npts:>6}")
    print("=" * 60)
    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
