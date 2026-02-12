#!/usr/bin/env python3
"""
DPDEM: MEDEA Satellite Ice Floes with Ship Fracture Simulation

Entry point: builds ice field (simple or MEDEA), runs simulation via medea_ice_ship package,
optionally saves/loads state and generates 2D/3D animations.

Usage (with uv):
    uv run python run_medea_ship_fracture.py           # Full run with animations
    uv run python run_medea_ship_fracture.py --quick   # Simulation only, no animations
    uv run python run_medea_ship_fracture.py --no-3d   # Skip 3D animation

  Plan then simulate (lattice planner path instead of straight line):
    uv run python run_medea_ship_fracture.py --simple --plan   # Planned path through ice (use with --simple or --full-map)
  Real-time 2D view (debugging; opens a window that updates each frame):
    uv run python run_medea_ship_fracture.py --simple --live   # Optional: --quick to skip writing movies
  Save/replay (run simulation once, generate animations later):
    uv run python run_medea_ship_fracture.py --simple --quick --save-state scenario.pkl   # Simulate + save, no animations
    uv run python run_medea_ship_fracture.py --load-state scenario.pkl                    # Generate 2D and 3D from saved state
    uv run python run_medea_ship_fracture.py --load-state scenario.pkl --no-3d             # Replay 2D only
"""

import os
import numpy as np
from pathlib import Path
import time
import argparse
import multiprocessing

# Use spawn to avoid fork + multithreaded libs (e.g. JAX) deadlock when parallelizing animation
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def main():
    parser = argparse.ArgumentParser(description='MEDEA Ice Ship Fracture Simulation')
    parser.add_argument('--live', action='store_true',
                        help='Open a 2D window and update it in real time during simulation (for debugging)')
    parser.add_argument('--quick', action='store_true', help='Simulation only, no animations')
    parser.add_argument('--no-3d', action='store_true', help='Skip 3D animation')
    parser.add_argument('--full-map', action='store_true',
                        help='Use full ice map; ship diagonal bottom-left to top-right')
    parser.add_argument('--simple', action='store_true',
                        help='Use minimal synthetic ice field (3 floes) for fast testing (no MEDEA data)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default from config)')
    parser.add_argument('--save-state', type=str, metavar='FILE', default=None,
                        help='Save full simulation state to FILE for replay (pickle)')
    parser.add_argument('--load-state', type=str, metavar='FILE', default=None,
                        help='Load state from FILE and run visualization only (no re-simulation)')
    parser.add_argument('--thick-ice', type=float, metavar='M', default=None,
                        help='Override ice thickness to M metres for all cells (e.g. 10 for thick-ice test)')
    parser.add_argument('--thrust-cap', type=float, metavar='N', default=None,
                        help='Cap ice-on-ship force to N Newtons (e.g. 25e6 for 25 MN)')
    parser.add_argument('--no-thrust-cap', action='store_true',
                        help='Disable thrust cap (unlimited ship thrust)')
    parser.add_argument('--gif', action='store_true',
                        help='Output GIF instead of MP4 (slower, larger files)')
    parser.add_argument('--plan', action='store_true',
                        help='Use lattice planner to plan path (costmap + A*); only with --simple or --full-map')
    parser.add_argument('--optimize-path', action='store_true',
                        help='Refine planned path with NLP (path length + costmap + curvature); requires casadi; use with --plan')
    parser.add_argument('--no-mss-dynamics', action='store_true',
                        help='Use simple diagonal M/D instead of MSS-style 3-DOF (Fossen added mass + damping)')
    parser.add_argument('--output-dir', type=str, default='outputs/default',
                        help='Output directory for movies and plots (default: outputs/default)')
    args = parser.parse_args()

    # Set interactive matplotlib backend before importing medea_ice_ship (so visualization uses it for --live)
    if args.live:
        os.environ['DPDEM_LIVE_VIEW'] = '1'
        import matplotlib
        matplotlib.use('TkAgg')
    from medea_ice_ship import (
        ShipProperties,
        load_medea_floes,
        scale_floes_to_world,
        create_simple_ice_field,
        polygon_area,
        polygon_centroid,
        create_proximity_bonds,
        assign_thickness_by_area,
        tessellate_all_floes,
        simulate_ice_ship_interaction,
        animate_ice_ship_2d,
        animate_ice_ship_3d,
    animate_costmap_movie,
    make_live_2d_callback,
    plot_hull_impact_vectors,
        save_simulation_state,
        load_simulation_state,
        build_costmap_from_ice_state,
        create_medea_lattice_planner,
        plan_path,
        path_to_ship_trajectory,
    )
    from medea_ice_ship.config import RunConfig, get_default_run_config, get_default_simulation_config
    from dataclasses import replace as replace_config
    if args.live:
        import matplotlib.pyplot as plt

    run_cfg = get_default_run_config()
    if args.seed is None:
        args.seed = run_cfg.default_seed
    video_ext = '.gif' if args.gif else '.mp4'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    np.random.seed(args.seed)

    if args.load_state:
        state = load_simulation_state(args.load_state)
        out_2d = state.get('output_2d') or ('replay_2d.gif' if args.gif else 'replay_2d.mp4')
        out_3d = state.get('output_3d') or ('replay_3d.gif' if args.gif else 'replay_3d.mp4')
        print("\nGenerating 2D animation (replay)...", flush=True)
        animate_ice_ship_2d(
            state['polygons'], state['positions_history'], state['rotations_history'],
            state['bonds_history'], state['bonds'],
            state['ship_history'], state['ship_length'], state['ship_beam'],
            state['centroids'], state['thicknesses'],
            output_file=out_2d, fps=run_cfg.fps_2d, frame_skip=run_cfg.frame_skip_2d,
            n_workers=run_cfg.n_workers,
        )
        if not args.no_3d:
            print("\nGenerating 3D animation (replay)...", flush=True)
            animate_ice_ship_3d(
                state['polygons'], state['positions_history'], state['rotations_history'],
                state['bonds_history'], state['bonds'],
                state['ship_history'], state['ship_length'], state['ship_beam'], state['ship_draft'],
                state['centroids'], state['thicknesses'],
                output_file=out_3d, fps=run_cfg.fps_3d, frame_skip=run_cfg.frame_skip_3d,
                submergence_history=state.get('submergence_history'),
                figsize=run_cfg.figsize_3d,
                n_workers=run_cfg.n_workers,
            )
        print(f"\nReplay complete. Output: {out_2d}" + (f", {out_3d}" if not args.no_3d else ""))
        return

    print("=" * 70)
    print("DPDEM: MEDEA Satellite Ice + Ship Fracture Simulation")
    if args.simple:
        print("  [SIMPLE MODE - synthetic ice field for quick testing]")
    if args.full_map:
        print("  [FULL MAP MODE - ship diagonal bottom-left to top-right]")
    if args.quick:
        print("  [QUICK MODE - simulation only]")
    elif args.no_3d:
        print("  [NO-3D MODE - skipping 3D animation]")
    print("=" * 70)
    print()

    ship_length = run_cfg.ship_length
    ship_beam = run_cfg.ship_beam
    ship_draft = run_cfg.ship_draft
    ship_displacement = run_cfg.ship_displacement

    if args.simple:
        t0 = time.time()
        print("Creating simple synthetic ice field (3 floes)...", flush=True)
        sat_polygons, sat_centroids = create_simple_ice_field(seed=args.seed)
        print(f"  Created {len(sat_polygons)} floes ({time.time()-t0:.1f}s)", flush=True)
        all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
            sat_polygons, sat_centroids, cells_per_100m2=run_cfg.cells_per_100m2_simple, seed=args.seed
        )
        polygons = all_cells
        centroids = all_cell_centroids
        parent_floe_ids = all_parent_ids
        print(f"  Tessellated to {len(polygons)} cells", flush=True)
    else:
        data_dir = Path(__file__).parent / 'data'
        pickle_path = data_dir / 'ice_floes_MEDEA_fram_20100629.pkl'
        if not pickle_path.exists():
            print(f"ERROR: Data file not found: {pickle_path}")
            return
        t0 = time.time()
        print("Loading MEDEA ice floe data...", flush=True)
        polygons_px, centroids_px, metadata = load_medea_floes(pickle_path)
        print(f"  Loaded {metadata['n_floes']} floes ({time.time()-t0:.1f}s)", flush=True)
        print(f"Scaling to world coordinates ({run_cfg.meters_per_pixel} m/px)...", flush=True)
        all_polygons, all_centroids = scale_floes_to_world(
            polygons_px, centroids_px, run_cfg.meters_per_pixel, center_domain=True, subsample_region=None
        )
        if args.full_map:
            sat_polygons, sat_centroids = all_polygons, all_centroids
            print(f"  Using full ice map: {len(sat_polygons)} floes", flush=True)
        else:
            y_center = (all_centroids[:, 1].min() + all_centroids[:, 1].max()) / 2
            keep_mask = np.abs(all_centroids[:, 1] - y_center) < run_cfg.y_band
            sat_polygons = [all_polygons[i] for i in range(len(all_polygons)) if keep_mask[i]]
            sat_centroids = all_centroids[keep_mask]
            print(f"  Selected {len(sat_polygons)} floes in ship's path (y +/- {run_cfg.y_band}m)", flush=True)
        print("Tessellating floes internally for fracturing...", flush=True)
        all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
            sat_polygons, sat_centroids, cells_per_100m2=run_cfg.cells_per_100m2_medea, seed=args.seed
        )
        print(f"  Created {len(all_cells)} total cells", flush=True)
        if args.full_map:
            polygons, centroids, parent_floe_ids = all_cells, all_cell_centroids, all_parent_ids
            print(f"  Using all {len(polygons)} cells (full map)", flush=True)
        else:
            ship_y_center = (sat_centroids[:, 1].min() + sat_centroids[:, 1].max()) / 2
            keep_cells = np.abs(all_cell_centroids[:, 1] - ship_y_center) < run_cfg.cell_y_band
            polygons = [all_cells[i] for i in range(len(all_cells)) if keep_cells[i]]
            centroids = all_cell_centroids[keep_cells]
            parent_floe_ids = all_parent_ids[keep_cells]
            print(f"  Filtered to {len(polygons)} cells in ship's path (y +/- {run_cfg.cell_y_band}m)", flush=True)

    areas = np.array([polygon_area(p) for p in polygons])
    radii = np.sqrt(areas / np.pi)
    print("Assigning ice thickness...", flush=True)
    sat_areas = np.array([polygon_area(p) for p in sat_polygons])
    sat_thicknesses = assign_thickness_by_area(
        sat_areas, min_thickness=run_cfg.thickness_min, max_thickness=run_cfg.thickness_max, power=run_cfg.thickness_power
    )
    thicknesses = sat_thicknesses[parent_floe_ids]
    if args.thick_ice is not None:
        thicknesses = np.full_like(thicknesses, float(args.thick_ice))
        print(f"  Thickness overridden to {args.thick_ice} m (thick-ice test)", flush=True)
    else:
        print(f"  Thickness range: {thicknesses.min():.2f} - {thicknesses.max():.2f} m", flush=True)
    print("Creating bonds between cells...", flush=True)
    bonds = create_proximity_bonds(centroids, radii, threshold_factor=run_cfg.bond_threshold_factor)
    print(f"  Created {len(bonds)} bonds", flush=True)
    all_verts = np.vstack(polygons)
    domain_width = all_verts[:, 0].max() - all_verts[:, 0].min()
    domain_height = all_verts[:, 1].max() - all_verts[:, 1].min()
    print(f"  Domain: {domain_width:.0f} x {domain_height:.0f} m", flush=True)

    ship_props = ShipProperties(ship_length, ship_beam, ship_draft, ship_displacement)
    print(f"\nShip: {ship_length}m x {ship_beam}m, mass={ship_props.mass/1e6:.1f} kt", flush=True)
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
    print(f"  Ice domain X: {x_min:.1f} to {x_max:.1f}m, Y: {y_min:.1f} to {y_max:.1f}m", flush=True)

    if args.simple:
        n_frames = run_cfg.n_frames_simple
        ship_start = np.array([x_min - ship_length / 2 - run_cfg.path_margin, (y_min + y_max) / 2])
        ship_end = np.array([x_max + ship_length / 2 + run_cfg.path_margin, (y_min + y_max) / 2])
        ship_heading = 0.0
        ship_trajectory = [(ship_start + (frame / n_frames) * (ship_end - ship_start), ship_heading) for frame in range(n_frames + 1)]
        ship_pos = ship_trajectory[0][0]
        drift_vel = 0.0
        print(f"  Ship path: ({ship_start[0]:.0f}, {ship_start[1]:.0f}) -> ({ship_end[0]:.0f}, {ship_end[1]:.0f}), {n_frames} frames", flush=True)
    elif args.full_map:
        margin = run_cfg.domain_margin
        ship_start = np.array([x_min - ship_length/2 + margin, y_min - ship_beam/2 + margin])
        ship_end = np.array([x_max + ship_length/2 - margin, y_max + ship_beam/2 - margin])
        diagonal_heading = np.arctan2(ship_end[1] - ship_start[1], ship_end[0] - ship_start[0])
        n_frames = run_cfg.n_frames_medea
        ship_trajectory = [(ship_start + (frame / n_frames) * (ship_end - ship_start), diagonal_heading) for frame in range(n_frames + 1)]
        ship_pos = ship_trajectory[0][0]
        drift_vel = 0.0
        print(f"  Ship path: bottom-left -> top-right, heading {np.degrees(diagonal_heading):.0f}°", flush=True)
    else:
        rightmost_edge = max(centroids[i, 0] + radii[i] for i in range(len(centroids)))
        ship_bow_x = rightmost_edge + 5
        y_center = (centroids[:, 1].min() + centroids[:, 1].max()) / 2
        ship_pos = np.array([ship_bow_x + ship_length/2, y_center])
        ship_trajectory = None
        drift_vel = 6.0
        n_frames = run_cfg.n_frames_medea
        print(f"  Ship stationary at ({ship_pos[0]:.0f}, {ship_pos[1]:.0f}), ice drifts toward ship", flush=True)
        if args.plan:
            print("  --plan is ignored when not using --simple or --full-map (ship stationary, ice drifts).", flush=True)

    planned_path = None
    planned_cost_map = None
    planned_bounds = None
    replan_callback = None
    replan_every_n_frames = None
    costmap_snapshots = None  # (frame_idx, cost_map) for costmap movie (AUTO-IceNav: costmap updated at replan)
    # Plan-then-simulate: replace fixed trajectory with planned path (Section 5); enable receding-horizon replan (AUTO-IceNav style)
    if args.plan and (args.simple or args.full_map):
        plan_cfg = replace_config(run_cfg, planning_turning_radius=50.0)  # so primitive step length fits map
        if args.optimize_path:
            plan_cfg = replace_config(plan_cfg, planning_use_path_optimizer=True)
            print("  Path refinement: NLP optimizer enabled (path length + costmap + curvature)", flush=True)
        cost_map, bounds = build_costmap_from_ice_state(
            polygons, centroids, thicknesses,
            x_min, x_max, y_min, y_max,
            resolution=plan_cfg.planning_resolution,
            margin=plan_cfg.planning_margin,
            ship_mass=plan_cfg.ship_displacement,
            damage_cost_scale=plan_cfg.planning_damage_cost_scale,
            damage_thickness_exponent=plan_cfg.planning_damage_thickness_exponent,
        )
        start_heading = (ship_heading if args.simple else diagonal_heading)
        start = (float(ship_start[0]), float(ship_start[1]), float(start_heading))
        goal = (float(ship_end[0]), float(ship_end[1]))
        planner = create_medea_lattice_planner(cost_map, bounds, start, goal, plan_cfg)
        path = plan_path(planner, start, goal, plan_cfg, cost_map=cost_map)
        if path is None:
            print("  Planner did not find a path; aborting. Try without --plan for fixed trajectory.", flush=True)
            return
        ship_trajectory = path_to_ship_trajectory(path, n_frames)
        ship_pos = ship_trajectory[0][0]
        planned_path = path
        planned_cost_map = cost_map
        planned_bounds = bounds
        costmap_snapshots = [(0, cost_map)]  # initial costmap for movie (AUTO-IceNav style)
        print(f"  Planned path: {len(path)} waypoints -> trajectory length {len(ship_trajectory)}", flush=True)

        # Receding-horizon callback (AUTO-IceNav style): rebuild costmap from current ice state and replan
        def _replan_callback(frame, world_polygons, centroids, thicknesses, ship_pos_f, ship_heading_f, n_remaining):
            all_verts = np.vstack(world_polygons)
            x_min_f = float(all_verts[:, 0].min())
            x_max_f = float(all_verts[:, 0].max())
            y_min_f = float(all_verts[:, 1].min())
            y_max_f = float(all_verts[:, 1].max())
            cost_map_f, _ = build_costmap_from_ice_state(
                world_polygons, centroids, thicknesses,
                x_min_f, x_max_f, y_min_f, y_max_f,
                resolution=plan_cfg.planning_resolution,
                margin=plan_cfg.planning_margin,
                ship_mass=plan_cfg.ship_displacement,
                damage_cost_scale=plan_cfg.planning_damage_cost_scale,
                damage_thickness_exponent=plan_cfg.planning_damage_thickness_exponent,
            )
            costmap_snapshots.append((frame, cost_map_f))  # for costmap movie: update every replan
            start_f = (float(ship_pos_f[0]), float(ship_pos_f[1]), float(ship_heading_f))
            try:
                planner_f = create_medea_lattice_planner(cost_map_f, {"x_origin": cost_map_f.x_origin, "y_origin": cost_map_f.y_origin, "width": cost_map_f.width, "height": cost_map_f.height}, start_f, goal, plan_cfg)
            except ValueError:
                return None
            path_f = plan_path(planner_f, start_f, goal, plan_cfg, cost_map=cost_map_f)
            if path_f is None:
                return None
            return path_to_ship_trajectory(path_f, n_remaining)

        replan_callback = _replan_callback
        replan_every_n_frames = run_cfg.planning_replan_every_n_frames
        print(f"  Receding horizon: replan every {replan_every_n_frames} frames", flush=True)

    sim_cfg = get_default_simulation_config()
    if args.no_mss_dynamics:
        sim_cfg = replace_config(sim_cfg, use_mss_style_dynamics=False)
        print("  Ship dynamics: simple diagonal M/D (MSS-style disabled)", flush=True)
    if args.no_thrust_cap:
        from dataclasses import replace
        sim_cfg = replace(sim_cfg, max_ship_ice_force=None)
        print("  Thrust cap: disabled (unlimited)", flush=True)
    elif args.thrust_cap is not None:
        from dataclasses import replace
        sim_cfg = replace(sim_cfg, max_ship_ice_force=args.thrust_cap)
        print(f"  Thrust cap: {args.thrust_cap/1e6:.1f} MN", flush=True)
    frame_callback = None
    if args.live:
        local_polys_live = [np.asarray(p) - polygon_centroid(p) for p in polygons]
        margin = 50.0
        extent_live = (float(x_min) - margin, float(x_max) + margin, float(y_min) - margin, float(y_max) + margin)
        _fig_live, _ax_live, frame_callback = make_live_2d_callback(
            bonds, local_polys_live, ship_length, ship_beam, thicknesses, extent_live, n_frames,
        )
        print("  Live 2D view: window will update each frame", flush=True)
    t0 = time.time()
    print(f"\nRunning fracture simulation ({n_frames} frames)...", flush=True)
    sim_result = simulate_ice_ship_interaction(
        polygons=polygons, centroids=centroids.copy(), bonds=bonds,
        ship_pos=ship_pos, ship_length=ship_length, ship_beam=ship_beam,
        n_frames=n_frames, dt=run_cfg.dt, drift_vel=drift_vel, thicknesses=thicknesses,
        ship_props=ship_props, ship_trajectory=ship_trajectory,
        config=sim_cfg,
        replan_every_n_frames=replan_every_n_frames,
        replan_callback=replan_callback,
        frame_callback=frame_callback,
    )
    positions_history = sim_result[0]
    rotations_history = sim_result[1]
    bonds_history = sim_result[2]
    ship_history = sim_result[3]
    submergence_history = sim_result[4] if len(sim_result) > 4 else None
    damage_summary = sim_result[5] if len(sim_result) > 5 and isinstance(sim_result[5], dict) else None
    sim_time = time.time() - t0

    n_initial = len(bonds)
    n_intact = bonds_history[-1].sum() if n_initial > 0 else 0
    n_broken = n_initial - n_intact
    pct = 100 * n_broken / n_initial if n_initial > 0 else 0
    print(f"\n  Final: {n_broken}/{n_initial} bonds broken ({pct:.0f}%)", flush=True)
    if damage_summary is not None:
        J = damage_summary.get('impulse_total', 0.0)
        print(f"  Ship damage proxy (cumulative impulse): {J/1e6:.2f} MN·s", flush=True)
        segs = damage_summary.get("hull_impulse_segments")
        contacts = damage_summary.get("hull_impulse_contacts")
        if contacts or segs:
            hull_out = str(out_dir / (('hull_impact_simple' if args.simple else 'hull_impact_medea') + '.png'))
            plot_hull_impact_vectors(
                ship_length, ship_beam,
                hull_impulse_segments=segs, hull_impulse_contacts=contacts,
                output_file=hull_out,
            )
            print(f"  Saved hull impact vectors: {hull_out}", flush=True)
    print(f"  Simulation time: {sim_time:.1f}s", flush=True)

    if args.live:
        print("\n  Live view: close the window to continue (or to exit).", flush=True)
        plt.show()

    if args.save_state:
        save_simulation_state(
            polygons, centroids, bonds, thicknesses,
            positions_history, rotations_history, bonds_history, ship_history, submergence_history,
            ship_length, ship_beam, ship_draft, n_frames, dt=run_cfg.dt,
            output_2d=('simple_ship_fracture' if args.simple else 'medea_ship_fracture') + video_ext,
            output_3d=('simple_ship_fracture_3d' if args.simple else 'medea_ship_fracture_3d') + video_ext,
            simple_mode=args.simple, filepath=args.save_state,
        )

    if args.quick:
        print(f"\nQuick mode complete! Total: {time.time()-total_start:.1f}s")
        return

    out_2d = str(out_dir / (('simple_ship_fracture' if args.simple else 'medea_ship_fracture') + video_ext))
    out_3d = str(out_dir / (('simple_ship_fracture_3d' if args.simple else 'medea_ship_fracture_3d') + video_ext))
    t0 = time.time()
    print("\nGenerating 2D animation...", flush=True)
    animate_ice_ship_2d(
        polygons, positions_history, rotations_history, bonds_history, bonds,
        ship_history, ship_length, ship_beam, centroids, thicknesses,
        output_file=out_2d, fps=run_cfg.fps_2d, frame_skip=run_cfg.frame_skip_2d,
        n_workers=run_cfg.n_workers,
    )
    print(f"  2D animation time: {time.time()-t0:.1f}s", flush=True)
    if not args.no_3d:
        t0 = time.time()
        print("\nGenerating 3D animation...", flush=True)
        animate_ice_ship_3d(
            polygons, positions_history, rotations_history, bonds_history, bonds,
            ship_history, ship_length, ship_beam, ship_draft, centroids, thicknesses,
            output_file=out_3d, fps=run_cfg.fps_3d, frame_skip=run_cfg.frame_skip_3d,
            submergence_history=submergence_history,
            figsize=run_cfg.figsize_3d,
            n_workers=run_cfg.n_workers,
        )
        print(f"  3D animation time: {time.time()-t0:.1f}s", flush=True)
    if args.plan and planned_path is not None and planned_cost_map is not None:
        out_costmap = str(out_dir / (('costmap_simple_ship_fracture' if args.simple else 'costmap_medea_ship_fracture') + video_ext))
        t0 = time.time()
        print("\nGenerating costmap movie (AUTO-IceNav style)...", flush=True)
        local_polygons = [np.asarray(p) - polygon_centroid(p) for p in polygons]
        animate_costmap_movie(
            planned_cost_map, planned_bounds, ship_history,
            path=planned_path,
            positions_history=positions_history,
            rotations_history=rotations_history,
            local_polygons=local_polygons,
            costmap_snapshots=costmap_snapshots,
            ship_length=ship_length, ship_beam=ship_beam,
            output_file=out_costmap, fps=run_cfg.fps_2d, frame_skip=run_cfg.frame_skip_2d,
            n_workers=run_cfg.n_workers,
        )
        print(f"  Costmap movie time: {time.time()-t0:.1f}s", flush=True)
    print(f"\nTotal time: {time.time()-total_start:.1f}s")
    print(f"\nOutput directory: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
