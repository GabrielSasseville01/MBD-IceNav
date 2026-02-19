#!/usr/bin/env python3
"""
Toy example: densely packed ice between start and goal with one opening.

Creates a wall of ice in the middle of the domain and leaves a single gap (opening).
Plans a path through the gap, runs the fracture simulation with the ship following
that path, and generates a 2D movie of the ship navigating (as in the simple example).

  - Domain: 600 m × 400 m (x × y)
  - Start: left side, middle y; goal: right side, middle y
  - Start and goal at y=200; opening at y=300 so the ship must detour (not straight ahead).
  - Ice: wall at x ∈ [200, 400], gap at y ∈ [260, 340] (80 m tall, centered at y=300).
  - Output: PNG (planned path), MP4 (ship navigating), costmap MP4 (AUTO-IceNav style).

Run:
  uv run python examples/toy_planner_opening.py
  uv run python examples/toy_planner_opening.py --live          # real-time 2D window during simulation
  uv run python examples/toy_planner_opening.py --damage-cost    # damage-aware cost (favours safer path)
  uv run python examples/toy_planner_opening.py --damage-cost 200  # custom damage cost scale
  uv run python examples/toy_planner_opening.py --no-replan        # single initial path (no receding horizon)
  uv run python examples/toy_planner_opening.py --no-dpdem-bonds  # legacy bond model (force vs strength)

  Damage proxy (cumulative impulse) is printed after the simulation. Use --damage-cost to make the
  planner penalise thick/heavy ice in the costmap so the chosen path tends to reduce exposure.

Output (default: outputs/toy_planner/):
  - toy_planner_lattice_search.mp4  (A* lattice graph search visualization; skip with --no-lattice-search-movie)
  - toy_planner_opening.mp4   (2D animation of ship navigating through the opening)
  - toy_planner_opening_costmap.mp4  (costmap movie: costmap background + ship trail + planned path)
  - toy_planner_hull_impact.png  (accumulated impact vectors on hull)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Optional real-time view: set backend before importing medea_ice_ship
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
    animate_costmap_movie,
    make_live_2d_callback,
    plot_hull_impact_vectors,
    animate_lattice_search,
)
from medea_ice_ship.config import get_default_run_config, get_default_simulation_config


def make_wall_with_opening(
    x_wall_lo: float,
    x_wall_hi: float,
    y_domain_lo: float,
    y_domain_hi: float,
    y_gap_lo: float,
    y_gap_hi: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Create two rectangular ice floes that form a wall with a gap.

    Wall spans x ∈ [x_wall_lo, x_wall_hi]. Gap is y ∈ [y_gap_lo, y_gap_hi].
    Returns polygons (world coords), centroids (Nx2), thicknesses (N,) for the two floes.
    """
    thickness = 1.5  # m, uniform

    def rect(x0, y0, w, h):
        """Rectangle as 4 vertices (x, y), counterclockwise."""
        return np.array([
            [x0, y0],
            [x0 + w, y0],
            [x0 + w, y0 + h],
            [x0, y0 + h],
        ], dtype=float)

    width = x_wall_hi - x_wall_lo
    h_bottom = y_gap_lo - y_domain_lo
    h_top = y_domain_hi - y_gap_hi

    polygons = []
    centroids = []
    thicknesses = []

    if h_bottom > 0:
        p_bot = rect(x_wall_lo, y_domain_lo, width, h_bottom)
        polygons.append(p_bot)
        centroids.append([x_wall_lo + width / 2, y_domain_lo + h_bottom / 2])
        thicknesses.append(thickness)
    if h_top > 0:
        p_top = rect(x_wall_lo, y_gap_hi, width, h_top)
        polygons.append(p_top)
        centroids.append([x_wall_lo + width / 2, y_gap_hi + h_top / 2])
        thicknesses.append(thickness)

    return polygons, np.array(centroids), np.array(thicknesses)


def main():
    parser = argparse.ArgumentParser(description="Toy planner: ice wall with opening")
    parser.add_argument("--live", action="store_true", help="Open real-time 2D window during simulation")
    parser.add_argument(
        "--damage-cost",
        nargs="?",
        type=float,
        default=None,
        const=100.0,
        metavar="SCALE",
        help="Enable damage-aware cost (favours safer path). Optional SCALE (default 100).",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/toy_planner",
                        help="Output directory for movies and plots (default: outputs/toy_planner)")
    parser.add_argument("--no-lattice-search-movie", action="store_true",
                        help="Skip generating the A* lattice search visualization movie (generated by default)")
    parser.add_argument("--no-replan", action="store_true",
                        help="Disable receding-horizon replanning (use single initial path for whole run)")
    parser.add_argument("--no-dpdem-bonds", action="store_true",
                        help="Use legacy bond model (force vs strength) instead of default DPDEM-style")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Domain (ice wall in middle; goal further in x to test longer path)
    x_min, x_max = 0.0, 600.0
    y_min, y_max = 0.0, 400.0
    goal_x = 800.0  # goal further right than ice domain
    # Wall with opening (opening at y=300 so ship at y=200 must detour)
    x_wall_lo, x_wall_hi = 200.0, 400.0
    y_gap_lo, y_gap_hi = 260.0, 340.0  # opening 80 m tall, centered at y=300

    # Two rectangular "satellite" floes (wall with gap)
    sat_polygons, sat_centroids, sat_thicknesses = make_wall_with_opening(
        x_wall_lo, x_wall_hi, y_min, y_max, y_gap_lo, y_gap_hi,
    )

    run_cfg = get_default_run_config()
    run_cfg = replace(run_cfg, planning_turning_radius=50.0)
    if args.damage_cost is not None:
        run_cfg = replace(run_cfg, planning_damage_cost_scale=float(args.damage_cost))
        print("  Damage-aware cost enabled (scale={})".format(run_cfg.planning_damage_cost_scale), flush=True)

    # Tessellate for fracture simulation (same pipeline as simple example)
    print("Tessellating ice wall into cells...", flush=True)
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

    all_verts = np.vstack(polygons)
    x_min_ice = float(all_verts[:, 0].min())
    x_max_ice = float(all_verts[:, 0].max())
    y_min_ice = float(all_verts[:, 1].min())
    y_max_ice = float(all_verts[:, 1].max())

    ship_length = run_cfg.ship_length
    ship_beam = run_cfg.ship_beam
    ship_draft = run_cfg.ship_draft
    ship_displacement = run_cfg.ship_displacement
    ship_props = ShipProperties(ship_length, ship_beam, ship_draft, ship_displacement)

    y_mid = (y_min + y_max) / 2.0
    margin = 30.0
    start = (x_min + margin, y_mid, 0.0)
    goal = (goal_x, y_mid)

    # Build costmap so it covers start to goal (extend x to goal_x so planner can reach it)
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
            output_file=str(out_dir / "toy_planner_lattice_search.mp4"),
            frame_stride=max(1, planner.nodes_expanded // 200),
            fps=15,
        )
        print(f"  Saved: {out_dir / 'toy_planner_lattice_search.mp4'}", flush=True)
    else:
        path = plan_path(planner, start, goal, run_cfg, cost_map=cost_map)

    if path is None:
        print("Planner did not find a path. Aborting.")
        sys.exit(1)

    print(f"Path found: {len(path)} waypoints (nodes expanded: {planner.nodes_expanded})")

    n_frames = 800  # longer run so ship reaches goal at x=800
    ship_trajectory = path_to_ship_trajectory(path, n_frames)
    ship_pos = ship_trajectory[0][0]

    # Receding-horizon replanning (AUTO-IceNav style)
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

    # Run simulation (optionally with real-time 2D view)
    frame_callback = None
    if args.live:
        local_polys_live = [np.asarray(p) - polygon_centroid(p) for p in polygons]
        margin_live = 50.0
        extent_live = (x_min - margin_live, goal_x + margin_live, y_min - margin_live, y_max + margin_live)
        _fig_live, _ax_live, frame_callback = make_live_2d_callback(
            bonds, local_polys_live, ship_length, ship_beam, thicknesses, extent_live, n_frames,
        )
        print("  Live 2D view enabled; window will update each frame.", flush=True)
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
    positions_history = sim_result[0]
    rotations_history = sim_result[1]
    bonds_history = sim_result[2]
    ship_history = sim_result[3]
    submergence_history = sim_result[4]
    damage_summary = sim_result[5] if len(sim_result) > 5 and isinstance(sim_result[5], dict) else None
    print("  Simulation done.")
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
            hull_plot_path = out_dir / "toy_planner_hull_impact.png"
            plot_hull_impact_vectors(
                ship_length, ship_beam,
                hull_impulse_segments=segs, hull_impulse_contacts=contacts,
                output_file=str(hull_plot_path),
            )
            print(f"  Saved hull impact vectors: {hull_plot_path.resolve()}", flush=True)
    if args.live:
        print("  Close the live window to continue.", flush=True)
        plt.show()

    # 2D movie: ship navigating through the opening
    out_mp4 = out_dir / "toy_planner_opening.mp4"
    print(f"Generating 2D animation -> {out_mp4.name}...", flush=True)
    # Fixed view so the whole grid (start to goal) is visible in the 2D movie
    view_bounds = (0.0, goal_x + 50.0, 0.0, 400.0)
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

    # Costmap movie: same view as 2D simulation, moving ice (fracture/displacement) and same figsize
    local_polygons = [np.asarray(p) - polygon_centroid(p) for p in polygons]
    out_costmap_mp4 = out_dir / "toy_planner_opening_costmap.mp4"
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
