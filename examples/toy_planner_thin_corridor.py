#!/usr/bin/env python3
"""
Toy example: dense ice band with one thin-ice corridor (thickness test).

The ship must cross from start to goal. Between them, ice spans the full width
of the domain in y (no way around). Most ice is thick (e.g. 10 m); one horizontal
strip at a fixed y is thin (e.g. 1 m). This tests whether the lattice planner
prefers the thin-ice corridor when damage-aware cost is enabled.

  - Domain: x ∈ [-200, 800], y ∈ [0, 400] (ice band extended in y to block bypass).
  - Ice: band x ∈ [150, 450], full width in y (no bypass).
  - Thickness: 10 m everywhere except one strip at y = 280 (y ∈ [240, 320], 80 m wide) = 1 m.
  - Start: (-200, 400); goal: (800, 400). Corridor is offset (y=280) so ship must turn into it.

Run:
  uv run python examples/toy_planner_thin_corridor.py
  uv run python examples/toy_planner_thin_corridor.py --live
  uv run python examples/toy_planner_thin_corridor.py --no-damage-cost   # baseline: no thickness preference
  uv run python examples/toy_planner_thin_corridor.py --no-replan        # single initial path (no receding horizon)
  uv run python examples/toy_planner_thin_corridor.py --no-dpdem-bonds  # legacy bond model (force vs strength)

Output (default: outputs/toy_planner_thin_corridor/):
  - toy_thin_corridor_lattice_search.mp4
  - toy_thin_corridor.mp4
  - toy_thin_corridor_3d.mp4
  - toy_thin_corridor_costmap.mp4
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

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
import argparse

from medea_ice_ship import (
    ShipProperties,
    polygon_area,
    polygon_centroid,
    create_proximity_bonds,
    tessellate_all_floes,
    build_costmap_from_ice_state,
    create_medea_lattice_planner,
    plan_path,
    path_to_ship_trajectory,
    simulate_ice_ship_interaction,
    animate_ice_ship_2d,
    animate_ice_ship_3d,
    animate_costmap_movie,
    make_live_2d_callback,
    plot_hull_impact_vectors,
    animate_lattice_search,
)
from medea_ice_ship.config import get_default_run_config, get_default_simulation_config


def make_thick_ice_with_thin_corridor(
    x_ice_lo: float,
    x_ice_hi: float,
    y_lo: float,
    y_hi: float,
    y_corridor_center: float,
    corridor_half_width: float,
    thick_m: float = 10.0,
    thin_m: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Create three large rectangular ice floes: bottom (thick), middle strip (thin), top (thick).
    Ice spans [x_ice_lo, x_ice_hi] × [y_lo, y_hi] with no gaps. The middle strip at
    y ∈ [y_corridor_center - corridor_half_width, y_corridor_center + corridor_half_width]
    is thin_m; the rest is thick_m. Large contiguous floes ensure 10 m ice behaves like
    a solid wall (ship bounces) rather than a grid of small cells that can be peeled through.
    Returns (polygons, centroids, thicknesses) in world coordinates.
    """
    def rect(x0, y0, w, h):
        return np.array([
            [x0, y0],
            [x0 + w, y0],
            [x0 + w, y0 + h],
            [x0, y0 + h],
        ], dtype=float)

    width = x_ice_hi - x_ice_lo
    y_corridor_lo = y_corridor_center - corridor_half_width
    y_corridor_hi = y_corridor_center + corridor_half_width

    polygons = []
    centroids = []
    thicknesses = []

    # Bottom thick floe: y_lo to y_corridor_lo
    if y_corridor_lo > y_lo:
        h_bot = y_corridor_lo - y_lo
        polygons.append(rect(x_ice_lo, y_lo, width, h_bot))
        centroids.append([x_ice_lo + width / 2, y_lo + h_bot / 2])
        thicknesses.append(thick_m)

    # Middle thin strip (1 m corridor)
    h_mid = y_corridor_hi - y_corridor_lo
    polygons.append(rect(x_ice_lo, y_corridor_lo, width, h_mid))
    centroids.append([x_ice_lo + width / 2, y_corridor_center])
    thicknesses.append(thin_m)

    # Top thick floe: y_corridor_hi to y_hi
    if y_hi > y_corridor_hi:
        h_top = y_hi - y_corridor_hi
        polygons.append(rect(x_ice_lo, y_corridor_hi, width, h_top))
        centroids.append([x_ice_lo + width / 2, y_corridor_hi + h_top / 2])
        thicknesses.append(thick_m)

    return polygons, np.array(centroids), np.array(thicknesses)


def main():
    parser = argparse.ArgumentParser(
        description="Toy planner: dense ice band with one thin-ice corridor (thickness test)"
    )
    parser.add_argument("--live", action="store_true", help="Open real-time 2D window during simulation")
    parser.add_argument(
        "--no-damage-cost",
        action="store_true",
        help="Disable damage-aware cost (planner will not prefer thin ice)",
    )
    parser.add_argument(
        "--damage-cost-scale",
        type=float,
        default=100.0,
        metavar="SCALE",
        help="Damage cost scale when damage cost is enabled (default 100).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/toy_planner_thin_corridor",
        help="Output directory for movies and plots",
    )
    parser.add_argument(
        "--no-lattice-search-movie",
        action="store_true",
        help="Skip A* lattice search visualization movie",
    )
    parser.add_argument(
        "--no-replan",
        action="store_true",
        help="Disable receding-horizon replanning (use single initial path for whole run)",
    )
    parser.add_argument(
        "--no-dpdem-bonds",
        action="store_true",
        help="Use legacy bond model (force vs strength) instead of default DPDEM-style",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Domain and ice band
    x_min, x_max = -400.0, 800.0
    y_min, y_max = 0.0, 400.0
    start_x, goal_x = -400.0, 800.0
    x_ice_lo, x_ice_hi = 150.0, 450.0
    # Thin strip offset from start/goal so ship must turn into the corridor
    y_corridor_center = 280.0   # thin strip at y ∈ [240, 320]
    y_start_goal = 400.0        # start and goal at y=400 (above corridor)
    corridor_half_width = 40.0  # 80 m wide

    run_cfg = get_default_run_config()
    # Use smaller margin so costmap is smaller and A* can complete (long path -200 to 800).
    planning_margin = 80.0
    run_cfg = replace(run_cfg, planning_margin=planning_margin)
    # Extend ice band vertically to cover full costmap (y_min - margin to y_max + margin).
    # Otherwise planner can "go around" through open water at y < 0 or y > 400.
    y_ice_lo = y_min - planning_margin
    y_ice_hi = y_max + planning_margin

    sat_polygons, sat_centroids, sat_thicknesses = make_thick_ice_with_thin_corridor(
        x_ice_lo, x_ice_hi, y_ice_lo, y_ice_hi,
        y_corridor_center, corridor_half_width,
        thick_m=10.0, thin_m=1.0,
    )
    n_thin = int(np.sum(sat_thicknesses < 2.0))
    print(f"Ice: {len(sat_polygons)} floes ({n_thin} thin 1 m, rest 10 m); band y ∈ [{y_ice_lo:.0f}, {y_ice_hi:.0f}] (no bypass)", flush=True)

    run_cfg = replace(
        run_cfg,
        planning_turning_radius=50.0,
        planning_resolution=4.0,           # coarser grid → smaller state space, longer steps
        planning_max_iterations=200000,   # long path (-200 to 800)
        planning_goal_tolerance=80.0,     # accept goal within 80 m
    )
    if not args.no_damage_cost:
        run_cfg = replace(
            run_cfg,
            planning_damage_cost_scale=float(args.damage_cost_scale),
        )
        print("Damage-aware cost enabled (scale={})".format(run_cfg.planning_damage_cost_scale), flush=True)
    else:
        print("Damage-aware cost disabled (baseline).", flush=True)

    print("Tessellating ice into cells...", flush=True)
    polygons, centroids, parent_floe_ids = tessellate_all_floes(
        sat_polygons, sat_centroids,
        cells_per_100m2=run_cfg.cells_per_100m2_simple,
        seed=42,
    )
    thicknesses = sat_thicknesses[parent_floe_ids]
    print(f"  {len(polygons)} cells", flush=True)

    areas = np.array([polygon_area(p) for p in polygons])
    radii = np.sqrt(areas / np.pi)
    bonds = create_proximity_bonds(centroids, radii, threshold_factor=run_cfg.bond_threshold_factor)
    print(f"  {len(bonds)} bonds", flush=True)

    ship_length = run_cfg.ship_length
    ship_beam = run_cfg.ship_beam
    ship_draft = run_cfg.ship_draft
    ship_displacement = run_cfg.ship_displacement
    ship_props = ShipProperties(ship_length, ship_beam, ship_draft, ship_displacement)

    margin = 30.0
    start = (start_x, y_start_goal, 0.0)
    goal = (goal_x, y_start_goal)

    costmap_x_max = max(x_max, goal_x + margin)
    cost_map, bounds = build_costmap_from_ice_state(
        sat_polygons, sat_centroids, sat_thicknesses,
        x_min, costmap_x_max, y_min, y_max,
        resolution=run_cfg.planning_resolution,
        margin=run_cfg.planning_margin,
        ship_mass=run_cfg.ship_displacement,
        damage_cost_scale=run_cfg.planning_damage_cost_scale,
        damage_thickness_exponent=run_cfg.planning_damage_thickness_exponent,
    )

    planner = create_medea_lattice_planner(
        cost_map=cost_map, bounds=bounds, start=start, goal=goal, run_cfg=run_cfg,
    )
    if not args.no_lattice_search_movie:
        result = plan_path(planner, start, goal, run_cfg, cost_map=cost_map, return_search_history=True)
        path, search_history = result
        if path is None:
            print("Planner did not find a path. Aborting.")
            sys.exit(1)
        print("Generating lattice search visualization...", flush=True)
        animate_lattice_search(
            cost_map, search_history, path, start, goal,
            output_file=str(out_dir / "toy_thin_corridor_lattice_search.mp4"),
            frame_stride=max(1, planner.nodes_expanded // 200),
            fps=15,
        )
        print(f"  Saved: {out_dir / 'toy_thin_corridor_lattice_search.mp4'}", flush=True)
    else:
        path = plan_path(planner, start, goal, run_cfg, cost_map=cost_map)

    if path is None:
        print("Planner did not find a path. Aborting.")
        sys.exit(1)

    print(f"Path found: {len(path)} waypoints (nodes expanded: {planner.nodes_expanded})")

    # Set n_frames so reference speed (path_length / (n_frames * dt)) is at or below ship max surge.
    # Otherwise the ship cannot keep up and lags even in open water.
    seg_lens = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    path_arc_length = float(np.sum(seg_lens))
    sim_cfg_for_limits = get_default_simulation_config()
    max_surge = sim_cfg_for_limits.ship_velocity_clip[0]  # m/s
    n_frames = max(1000, int(np.ceil(path_arc_length / (max_surge * run_cfg.dt))))
    print(f"  Path arc length: {path_arc_length:.0f} m -> n_frames={n_frames} (ref speed ~{path_arc_length/(n_frames*run_cfg.dt):.1f} m/s)", flush=True)
    ship_trajectory = path_to_ship_trajectory(path, n_frames)
    ship_pos = ship_trajectory[0][0]

    # Receding-horizon replanning (AUTO-IceNav style): rebuild costmap and replan every N frames
    costmap_snapshots = [(0, cost_map)]
    replan_every_n_frames = None
    replan_callback = None
    if not args.no_replan:
        def _replan_callback(frame, world_polygons, centroids_f, thicknesses_f, ship_pos_f, ship_heading_f, n_remaining):
            all_verts = np.vstack(world_polygons)
            x_min_f = float(all_verts[:, 0].min())
            x_max_f = float(all_verts[:, 0].max())
            y_min_f = float(all_verts[:, 1].min())
            y_max_f = float(all_verts[:, 1].max())
            cost_map_f, _ = build_costmap_from_ice_state(
                world_polygons, centroids_f, thicknesses_f,
                x_min_f, x_max_f, y_min_f, y_max_f,
                resolution=run_cfg.planning_resolution,
                margin=run_cfg.planning_margin,
                ship_mass=run_cfg.ship_displacement,
                damage_cost_scale=run_cfg.planning_damage_cost_scale,
                damage_thickness_exponent=run_cfg.planning_damage_thickness_exponent,
            )
            costmap_snapshots.append((frame, cost_map_f))
            start_f = (float(ship_pos_f[0]), float(ship_pos_f[1]), float(ship_heading_f))
            bounds_f = {
                "x_origin": cost_map_f.x_origin, "y_origin": cost_map_f.y_origin,
                "width": cost_map_f.width, "height": cost_map_f.height,
            }
            try:
                planner_f = create_medea_lattice_planner(
                    cost_map_f, bounds_f, start_f, goal, run_cfg,
                )
            except ValueError:
                return None
            path_f = plan_path(planner_f, start_f, goal, run_cfg, cost_map=cost_map_f)
            if path_f is None:
                return None
            return path_to_ship_trajectory(path_f, n_remaining)

        replan_every_n_frames = run_cfg.planning_replan_every_n_frames
        replan_callback = _replan_callback
        print(f"  Receding horizon: replan every {replan_every_n_frames} frames", flush=True)

    frame_callback = None
    if args.live:
        local_polys_live = [np.asarray(p) - polygon_centroid(p) for p in polygons]
        margin_live = 50.0
        extent_live = (x_min - margin_live, goal_x + margin_live, y_min - margin_live, y_max + margin_live)
        _fig_live, _ax_live, frame_callback = make_live_2d_callback(
            bonds, local_polys_live, ship_length, ship_beam, thicknesses, extent_live, n_frames,
        )
        print("Live 2D view enabled.", flush=True)
    print("Running fracture simulation...", flush=True)
    sim_cfg = get_default_simulation_config()
    if args.no_dpdem_bonds:
        sim_cfg = replace(sim_cfg, use_dpdem_style_bonds=False)
        print("  Bond model: legacy (force vs strength)", flush=True)
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
        replan_every_n_frames=replan_every_n_frames,
        replan_callback=replan_callback,
        frame_callback=frame_callback,
        parent_ids=parent_floe_ids,
    )
    positions_history, rotations_history, bonds_history, ship_history = sim_result[0], sim_result[1], sim_result[2], sim_result[3]
    submergence_history = sim_result[4]
    damage_summary = sim_result[5] if len(sim_result) > 5 and isinstance(sim_result[5], dict) else None
    print("Simulation done.")
    if damage_summary is not None:
        print(f"  Ship damage proxy (cumulative impulse): {damage_summary.get('impulse_total', 0)/1e6:.2f} MN·s", flush=True)
        dps = damage_summary.get("dpdem_bond_stats")
        if dps is not None:
            print(
                f"  DPDEM bond stats: damage>0.01: {dps['n_damaged']}/{dps['n_bonds']}, "
                f"mean: {dps['mean_damage']:.4f}, max: {dps['max_damage']:.4f}, failed by D>=1: {dps['n_failed_by_damage']}",
                flush=True,
            )
        segs = damage_summary.get("hull_impulse_segments")
        contacts = damage_summary.get("hull_impulse_contacts")
        if contacts or segs:
            hull_plot_path = out_dir / "toy_thin_corridor_hull_impact.png"
            plot_hull_impact_vectors(
                ship_length, ship_beam,
                hull_impulse_segments=segs, hull_impulse_contacts=contacts,
                output_file=str(hull_plot_path),
            )
            print(f"  Saved hull impact: {hull_plot_path.resolve()}", flush=True)
    if args.live:
        plt.show()

    view_bounds = (start_x - 50.0, goal_x + 50.0, y_min, max(y_max, y_start_goal + 50))
    out_mp4 = out_dir / "toy_thin_corridor.mp4"
    print(f"Generating 2D animation -> {out_mp4.name}...", flush=True)
    animate_ice_ship_2d(
        polygons,
        positions_history,
        rotations_history,
        bonds_history,
        bonds,
        ship_history,
        ship_length,
        ship_beam,
        centroids,
        thicknesses,
        output_file=str(out_mp4),
        fps=run_cfg.fps_2d,
        frame_skip=2,
        n_workers=run_cfg.n_workers,
        view_bounds=view_bounds,
    )
    print(f"Saved: {out_mp4.resolve()}")

    out_3d_mp4 = out_dir / "toy_thin_corridor_3d.mp4"
    print(f"Generating 3D animation -> {out_3d_mp4.name}...", flush=True)
    animate_ice_ship_3d(
        polygons,
        positions_history,
        rotations_history,
        bonds_history,
        bonds,
        ship_history,
        ship_length,
        ship_beam,
        ship_draft,
        centroids,
        thicknesses,
        output_file=str(out_3d_mp4),
        fps=run_cfg.fps_3d,
        frame_skip=run_cfg.frame_skip_3d,
        submergence_history=submergence_history,
        figsize=run_cfg.figsize_3d,
        n_workers=run_cfg.n_workers,
    )
    print(f"Saved: {out_3d_mp4.resolve()}")

    local_polygons = [np.asarray(p) - polygon_centroid(p) for p in polygons]
    out_costmap_mp4 = out_dir / "toy_thin_corridor_costmap.mp4"
    print(f"Generating costmap movie -> {out_costmap_mp4.name}...", flush=True)
    animate_costmap_movie(
        cost_map,
        bounds,
        ship_history,
        path=path,
        positions_history=positions_history,
        rotations_history=rotations_history,
        local_polygons=local_polygons,
        costmap_snapshots=costmap_snapshots,
        ship_length=ship_length,
        ship_beam=ship_beam,
        output_file=str(out_costmap_mp4),
        fps=run_cfg.fps_2d,
        frame_skip=2,
        n_workers=run_cfg.n_workers,
        figsize=(16, 12),
    )
    print(f"Saved: {out_costmap_mp4.resolve()}")
    print(f"Done. Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
