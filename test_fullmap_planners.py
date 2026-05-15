"""
Run all baseline planners on the full MEDEA map scenario (same setup as
``run_medea_ship_fracture.py --full-map --plan-only``).

Loads real MEDEA ice floe data, builds the costmap, runs A* lattice planning,
then runs RSS / iLQR / MPPI / MBD optimizers and overlays all paths on the
costmap for a side-by-side comparison.

Output: outputs/test_fullmap_planners/
"""

from __future__ import annotations

import sys
import time
import dataclasses
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
    polygon_area,
    create_proximity_bonds,
    tessellate_all_floes,
    build_costmap_from_ice_state,
    create_medea_lattice_planner,
    plan_path,
    load_medea_floes,
    scale_floes_to_world,
    assign_thickness_by_area,
    create_ship_hull_at_position_heading,
)
from medea_ice_ship.jax_rss_planner import jax_rss_optimize
from medea_ice_ship.jax_ilqr_planner import jax_ilqr_optimize
from medea_ice_ship.jax_mppi_planner import jax_mppi_optimize
from medea_ice_ship.jax_mbd_planner import jax_mbd_optimize
from medea_ice_ship.config import get_default_run_config

OUT_DIR = ROOT / "outputs" / "test_fullmap_planners"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = ROOT / "data" / "ice_floes_MEDEA_fram_20100629.pkl"

# ── Full-map scenario constants (mirror run_medea_ship_fracture.py --full-map) ─
DOMAIN_LIMIT  = 700.0
SHIP_START    = np.array([-545.0, -600.0])
GOAL_Y_LINE   = 600.0
SHIP_GOAL     = np.array([-350.0, GOAL_Y_LINE])  # ship_end in run_medea_ship_fracture.py line 644
START_HEADING = np.pi / 4.0   # 45° diagonal


# ── plan-only style plot helper ───────────────────────────────────────────────

def make_plan_only_figure(
    cost_map,
    polygons,
    thicknesses,
    ship_length,
    ship_beam,
    paths_dict,          # {name: (path, color, lw)}
    title,
    x_min, x_max, y_min, y_max,
    figsize=(12, 12),
    goal_xy=None,        # (x, y) of the actual goal; if None, falls back to last path point
):
    """
    Reproduce the --plan-only figure style:
      - Costmap as translucent background (YlOrRd, alpha=0.5)
      - Ice polygons coloured by thickness (Blues, alpha=0.6) with colorbar
      - Ship hull at start (orange)
      - All paths overlaid
      - Goal line (lime dashed)
      - Start marker (green circle) and end marker (red star) from first path
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Costmap background
    grid = cost_map.cost_map
    vmax = float(np.percentile(grid[grid > 0], 95)) if np.any(grid > 0) else 1.0
    extent = [
        cost_map.x_origin, cost_map.x_origin + cost_map.width,
        cost_map.y_origin, cost_map.y_origin + cost_map.height,
    ]
    ax.imshow(grid, origin="lower", extent=extent, cmap="YlOrRd",
              alpha=0.5, vmin=0, vmax=vmax, aspect="equal")

    # Ice polygons coloured by thickness
    patches = [MplPolygon(p, closed=True) for p in polygons]
    pc = PatchCollection(patches, cmap="Blues", edgecolor="#555555",
                         linewidth=0.3, alpha=0.6)
    pc.set_array(thicknesses)
    pc.set_clim(thicknesses.min(), thicknesses.max())
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, shrink=0.5, label="Thickness [m]", location="left")

    # Paths
    for name, (path, color, lw) in paths_dict.items():
        if path is not None and len(path) >= 2:
            p = np.asarray(path)
            ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=lw,
                    label=name, zorder=5)

    # Start / end markers from the first valid path
    first_path = next(
        (v[0] for v in paths_dict.values() if v[0] is not None and len(v[0]) >= 2),
        None,
    )
    if first_path is not None:
        fp = np.asarray(first_path)
        ax.plot(fp[0, 0], fp[0, 1], "go", ms=12, zorder=6, label="Start")
        if goal_xy is not None:
            ax.plot(goal_xy[0], goal_xy[1], "r*", ms=16, zorder=6, label="Goal")
        else:
            ax.plot(fp[-1, 0], fp[-1, 1], "r*", ms=16, zorder=6, label="End")

    # Goal line
    ax.axhline(GOAL_Y_LINE, color="lime", ls="--", lw=2,
               label=f"Goal line y={GOAL_Y_LINE:.0f}", zorder=4)

    # Ship hull at start
    hull = create_ship_hull_at_position_heading(ship_length, ship_beam,
                                                SHIP_START, START_HEADING)
    ax.add_patch(MplPolygon(hull, closed=True, fc="orange", ec="black",
                            lw=1.5, alpha=0.8, zorder=7))

    pad = 30.0
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Full-map planner comparison (MEDEA ice data)")
    print("=" * 65)

    # ── 1. Load MEDEA data ────────────────────────────────────────────
    print(f"\n[1] Loading MEDEA ice floe data: {DATA_PATH.name} ...")
    if not DATA_PATH.exists():
        print(f"  ERROR: data file not found at {DATA_PATH}")
        return
    t0 = time.perf_counter()
    polygons_px, centroids_px, metadata = load_medea_floes(DATA_PATH)
    run_cfg = get_default_run_config()
    all_polygons, all_centroids = scale_floes_to_world(
        polygons_px, centroids_px,
        run_cfg.meters_per_pixel,
        center_domain=True,
        subsample_region=None,
    )
    sat_polygons, sat_centroids = all_polygons, all_centroids
    print(f"    {len(sat_polygons)} floes loaded in {time.perf_counter()-t0:.2f}s")

    # ── 2. Tessellate ─────────────────────────────────────────────────
    print("\n[2] Tessellating floes ...")
    t0 = time.perf_counter()
    all_cells, all_centroids_c, all_parent_ids = tessellate_all_floes(
        sat_polygons, sat_centroids,
        cells_per_100m2=run_cfg.cells_per_100m2_medea,
        seed=42,
    )
    polygons   = all_cells
    centroids  = all_centroids_c
    parent_ids = all_parent_ids
    areas = np.array([polygon_area(p) for p in polygons])
    radii = np.sqrt(areas / np.pi)
    print(f"    {len(polygons)} cells in {time.perf_counter()-t0:.2f}s")

    # ── 3. Thickness & bonds ──────────────────────────────────────────
    print("\n[3] Assigning thickness and creating bonds ...")
    sat_areas = np.array([polygon_area(p) for p in sat_polygons])
    sat_thicknesses = assign_thickness_by_area(
        sat_areas,
        min_thickness=run_cfg.thickness_min,
        max_thickness=run_cfg.thickness_max,
        power=run_cfg.thickness_power,
    )
    thicknesses = sat_thicknesses[parent_ids]
    bonds = create_proximity_bonds(centroids, radii,
                                   threshold_factor=run_cfg.bond_threshold_factor)
    print(f"    Thickness {thicknesses.min():.2f}–{thicknesses.max():.2f} m, "
          f"{len(bonds)} bonds")

    # ── 4. Domain ─────────────────────────────────────────────────────
    x_min, x_max = -DOMAIN_LIMIT, DOMAIN_LIMIT
    y_min, y_max = -DOMAIN_LIMIT, DOMAIN_LIMIT

    ship_length = run_cfg.ship_length
    ship_beam   = run_cfg.ship_beam

    # ── 5. Build costmap ──────────────────────────────────────────────
    print("\n[4] Building costmap ...")
    # Distance from start (-545,-600) to goal (-350,600) is ~1215 m.
    # At 4 m/s with dt_substep=1.0 s and n_substeps=1, each action step covers ~4 m,
    # so we need at least ceil(1215/4) ≈ 304 steps.  Use 450 to leave headroom for
    # non-straight paths through ice.
    plan_cfg = dataclasses.replace(
        run_cfg,
        planning_resolution=4.0,
        planning_num_headings=8,
        planning_heuristic_weight=8.0,
        planning_turning_radius=25.0,
        planning_rss_horizon=1000,
        planning_rss_nsample=1024,
        planning_ilqr_horizon=450,
        planning_ilqr_max_iters=15,
        planning_jax_mbd_horizon=450,
        planning_mppi_horizon=450,
        planning_mppi_nsample=512,
    )
    t0 = time.perf_counter()
    cost_map, bounds = build_costmap_from_ice_state(
        sat_polygons, sat_centroids, sat_thicknesses,
        x_min, x_max, y_min, y_max,
        resolution=plan_cfg.planning_resolution,
        margin=plan_cfg.planning_margin,
        ship_mass=run_cfg.ship_displacement,
        collision_cost=plan_cfg.planning_ice_collision_cost,
        wall_bounds=(x_min, x_max, y_min, y_max),
        ship_length=ship_length,
        ship_beam=ship_beam,
        planning_clearance_margin=plan_cfg.planning_clearance_margin,
    )
    print(f"    Costmap ({cost_map.rows}×{cost_map.cols}), "
          f"res={cost_map.resolution}m — built in {time.perf_counter()-t0:.2f}s")

    # ── 6. Start / goal ───────────────────────────────────────────────
    start = (float(SHIP_START[0]), float(SHIP_START[1]), START_HEADING)
    goal  = (float(SHIP_GOAL[0]),  float(SHIP_GOAL[1]))
    print(f"\n[5] Start: ({start[0]:.0f}, {start[1]:.0f}, "
          f"{np.degrees(start[2]):.0f}°)  Goal: ({goal[0]:.0f}, {goal[1]:.0f})")

    # ── 7. A* lattice ─────────────────────────────────────────────────
    print("\n[6] Running A* lattice planner ...")
    t0 = time.perf_counter()
    planner = create_medea_lattice_planner(
        cost_map=cost_map, bounds=bounds,
        start=start, goal=goal,
        run_cfg=plan_cfg,
        goal_y_line=GOAL_Y_LINE,
    )
    lattice_path = plan_path(planner, start, goal, plan_cfg, cost_map=cost_map)
    lattice_elapsed = time.perf_counter() - t0
    if lattice_path is None:
        print("  WARNING: A* failed — using straight-line fallback")
        lattice_path = np.array([
            [start[0], start[1], start[2]],
            [goal[0],  goal[1],  START_HEADING],
        ])
    lattice_cost = float(cost_map.get_path_cost(lattice_path))
    print(f"    A* done in {lattice_elapsed:.2f}s, {len(lattice_path)} waypoints, "
          f"costmap integral={lattice_cost:.0f}")

    # Save standalone lattice figure (plan-only style)
    fig = make_plan_only_figure(
        cost_map, polygons, thicknesses, ship_length, ship_beam,
        paths_dict={"A* lattice": (lattice_path, "#1f77b4", 2.5)},
        title=(f"Full-map — A* Lattice ({lattice_elapsed:.1f}s, "
               f"cost={lattice_cost:.0f}) — {len(lattice_path)} waypoints\n"
               f"{len(polygons)} cells, domain [{x_min:.0f},{x_max:.0f}]×"
               f"[{y_min:.0f},{y_max:.0f}] m"),
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        goal_xy=goal,
    )
    fig.savefig(OUT_DIR / "01_lattice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {OUT_DIR / '01_lattice.png'}")

    # ── 8. Run trajectory optimizers ──────────────────────────────────
    results = {"lattice": (lattice_path, lattice_elapsed, lattice_cost)}

    COLORS = {
        "lattice":         "#1f77b4",
        "random-shooting": "#d62728",
        "ilqr":            "#9467bd",
        "mppi":            "#ff7f0e",
        "diffusion-mpc":   "#2ca02c",
    }
    NAMES = {
        "random-shooting": "Random Single Shooting",
        "ilqr":            "Box-iLQR",
        "mppi":            "MPPI",
        "diffusion-mpc":   "MBD diffusion-MPC",
    }
    # Call the underlying jax_*_optimize functions directly — no lattice_path,
    # no warm-start, no A*-tail stitching — so each planner is fully standalone.
    OPT_FNS = {
        "random-shooting": jax_rss_optimize,
        "ilqr":            jax_ilqr_optimize,
        "mppi":            jax_mppi_optimize,
        "diffusion-mpc":   jax_mbd_optimize,
    }
    FIG_IDX = {
        "random-shooting": 2, "ilqr": 3, "mppi": 4, "diffusion-mpc": 5,
    }

    for key in ("random-shooting", "ilqr", "mppi", "diffusion-mpc"):
        print(f"\n[{NAMES[key]}] Running optimizer ...")
        t0 = time.perf_counter()
        try:
            opt_path = OPT_FNS[key](
                cost_map=cost_map,
                start_pose=start,
                goal=goal,
                run_cfg=plan_cfg,
                start_nu=(2.0, 0.0, 0.0),
                warm_start=False,
            )
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            opt_path = None
        elapsed = time.perf_counter() - t0
        cost = (float(cost_map.get_path_cost(np.asarray(opt_path)))
                if opt_path is not None and len(opt_path) >= 2 else float("inf"))
        npts = len(opt_path) if opt_path is not None else 0
        print(f"    done in {elapsed:.2f}s, {npts} pts, costmap integral={cost:.0f}")
        results[key] = (opt_path, elapsed, cost)

        if opt_path is not None:
            fig = make_plan_only_figure(
                cost_map, polygons, thicknesses, ship_length, ship_beam,
                paths_dict={
                    "A* lattice":  (lattice_path, COLORS["lattice"],  1.5),
                    NAMES[key]:    (opt_path,      COLORS[key],        2.5),
                },
                title=(f"Full-map — {NAMES[key]} ({elapsed:.1f}s, "
                       f"cost={cost:.0f}) — {npts} waypoints\n"
                       f"{len(polygons)} cells, domain [{x_min:.0f},{x_max:.0f}]×"
                       f"[{y_min:.0f},{y_max:.0f}] m"),
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                goal_xy=goal,
            )
            out = OUT_DIR / f"{FIG_IDX[key]:02d}_{key.replace('-','_')}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: {out}")

    # ── 8b. MBD variant: goal-line hard constraint, no soft goal term ─
    print("\n[MBD goal-line constraint] goal_weight=0, big-M penalty for y<600 ...")
    t0 = time.perf_counter()
    try:
        mbd_gl_cfg = dataclasses.replace(plan_cfg, planning_jax_mbd_goal_weight=0.0)
        mbd_gl_path = jax_mbd_optimize(
            cost_map=cost_map,
            start_pose=start,
            goal=goal,
            run_cfg=mbd_gl_cfg,
            start_nu=(2.0, 0.0, 0.0),
            warm_start=False,
            goal_y_line=GOAL_Y_LINE,
            goal_y_line_penalty=1e4,
        )
    except Exception as exc:
        import traceback
        print(f"  ERROR: {exc}")
        traceback.print_exc()
        mbd_gl_path = None
    mbd_gl_elapsed = time.perf_counter() - t0
    mbd_gl_cost = (float(cost_map.get_path_cost(np.asarray(mbd_gl_path)))
                   if mbd_gl_path is not None and len(mbd_gl_path) >= 2 else float("inf"))
    mbd_gl_npts = len(mbd_gl_path) if mbd_gl_path is not None else 0
    print(f"    done in {mbd_gl_elapsed:.2f}s, {mbd_gl_npts} pts, "
          f"costmap integral={mbd_gl_cost:.0f}")
    results["diffusion-mpc-goalline"] = (mbd_gl_path, mbd_gl_elapsed, mbd_gl_cost)

    if mbd_gl_path is not None:
        fig = make_plan_only_figure(
            cost_map, polygons, thicknesses, ship_length, ship_beam,
            paths_dict={
                "A* lattice":           (lattice_path,  COLORS["lattice"],        1.5),
                "MBD (soft goal)":      (results["diffusion-mpc"][0], COLORS["diffusion-mpc"], 1.5),
                "MBD (goal-line y=600)":(mbd_gl_path,   "#e377c2",                2.5),
            },
            title=(f"Full-map — MBD goal-line constraint (goal_weight=0, big-M y≥600)\n"
                   f"({mbd_gl_elapsed:.1f}s, cost={mbd_gl_cost:.0f}) — {mbd_gl_npts} waypoints"),
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            goal_xy=goal,
        )
        fig.savefig(OUT_DIR / "05b_mbd_goalline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {OUT_DIR / '05b_mbd_goalline.png'}")

    # ── 8c. MBD variant: EB-MBD (emerging barrier, soft goal-x + length costs) ─
    print("\n[EB-MBD] Emerging-barrier MBD — OOB+goal via barrier, soft goal-x + length ...")
    t0 = time.perf_counter()
    try:
        eb_mbd_cfg = dataclasses.replace(
            plan_cfg,
            planning_jax_mbd_goal_weight=0.0,    # no soft goal-distance term
            planning_jax_mbd_goal_x_weight=1.0,  # soft x-deviation at goal crossing
            planning_jax_mbd_length_weight=1.0,  # soft path-length cost
        )
        eb_mbd_result = jax_mbd_optimize(
            cost_map=cost_map,
            start_pose=start,
            goal=goal,
            run_cfg=eb_mbd_cfg,
            start_nu=(2.0, 0.0, 0.0),
            warm_start=False,
            goal_y_line=GOAL_Y_LINE,
            # EB-MBD: emerging barrier enforces OOB and goal line (no big-M)
            use_eb_mbd=True,
            eb_x_min=x_min,
            eb_x_max=x_max,
            eb_y_min=y_min,
            eb_c_max=1500.0,
            eb_mu=50000.0,
            eb_kappa=1.0,
            return_diagnostics=True,
            denoising_movie_dir=str(OUT_DIR / "eb_mbd_denoising"),
        )
        eb_mbd_path, eb_mbd_diag = eb_mbd_result
    except Exception as exc:
        import traceback
        print(f"  ERROR: {exc}")
        traceback.print_exc()
        eb_mbd_path, eb_mbd_diag = None, None
    eb_mbd_elapsed = time.perf_counter() - t0
    eb_mbd_cost = (float(cost_map.get_path_cost(np.asarray(eb_mbd_path)))
                   if eb_mbd_path is not None and len(eb_mbd_path) >= 2 else float("inf"))
    eb_mbd_npts = len(eb_mbd_path) if eb_mbd_path is not None else 0
    print(f"    done in {eb_mbd_elapsed:.2f}s, {eb_mbd_npts} pts, "
          f"costmap integral={eb_mbd_cost:.0f}")
    results["diffusion-mpc-eb"] = (eb_mbd_path, eb_mbd_elapsed, eb_mbd_cost)

    if eb_mbd_path is not None:
        fig = make_plan_only_figure(
            cost_map, polygons, thicknesses, ship_length, ship_beam,
            paths_dict={
                "A* lattice":           (lattice_path,  COLORS["lattice"],        1.5),
                "MBD (goal-line y=600)":(mbd_gl_path,   "#e377c2",                1.5),
                "EB-MBD":               (eb_mbd_path,   "#17becf",                2.5),
            },
            title=(f"Full-map — EB-MBD (emerging barrier, goal-x + length soft costs)\n"
                   f"({eb_mbd_elapsed:.1f}s, cost={eb_mbd_cost:.0f}) — {eb_mbd_npts} waypoints"),
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            goal_xy=goal,
        )
        fig.savefig(OUT_DIR / "05c_mbd_eb.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {OUT_DIR / '05c_mbd_eb.png'}")

    # ── 9. Comparison overlay (all planners) ──────────────────────────
    print("\n[7] Saving comparison overlay ...")
    paths_dict_all = {}
    lws = {"lattice": 1.5, "random-shooting": 2.0,
           "ilqr": 2.0, "mppi": 2.0, "diffusion-mpc": 2.0,
           "diffusion-mpc-goalline": 2.0, "diffusion-mpc-eb": 2.0}
    display_names = {
        "lattice":                  "A* lattice",
        "random-shooting":          "RSS",
        "ilqr":                     "iLQR",
        "mppi":                     "MPPI",
        "diffusion-mpc":            "MBD",
        "diffusion-mpc-goalline":   "MBD (goal-line)",
        "diffusion-mpc-eb":         "EB-MBD",
    }
    COLORS["diffusion-mpc-goalline"] = "#e377c2"
    COLORS["diffusion-mpc-eb"] = "#17becf"
    for key, (path, elapsed, cost) in results.items():
        cost_str = f"{cost:.0f}" if cost < float("inf") else "FAILED"
        label = f"{display_names[key]}  [{elapsed:.1f}s, cost={cost_str}]"
        paths_dict_all[label] = (path, COLORS[key], lws[key])

    fig = make_plan_only_figure(
        cost_map, polygons, thicknesses, ship_length, ship_beam,
        paths_dict=paths_dict_all,
        title=(f"Full-map — all planners comparison\n"
               f"{len(polygons)} cells, domain [{x_min:.0f},{x_max:.0f}]×"
               f"[{y_min:.0f},{y_max:.0f}] m"),
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        goal_xy=goal,
    )
    fig.savefig(OUT_DIR / "06_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {OUT_DIR / '06_comparison.png'}")

    # ── 10. Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"  {'Planner':<22}  {'Elapsed (s)':>12}  {'Costmap integral':>17}  {'Pts':>6}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*17}  {'-'*6}")
    for name, (path, elapsed, cost) in results.items():
        npts = len(path) if path is not None else 0
        cost_str = f"{cost:.0f}" if cost < float("inf") else "FAILED"
        print(f"  {name:<22}  {elapsed:>12.2f}  {cost_str:>17}  {npts:>6}")
    print("=" * 65)
    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
