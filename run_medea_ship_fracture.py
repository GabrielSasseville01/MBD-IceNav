#!/usr/bin/env python3
"""
DPDEM: MEDEA Satellite Ice Floes with Ship Fracture Simulation

Entry point: builds ice field (simple or MEDEA), runs simulation via medea_ice_ship package,
optionally saves/loads state and generates 2D/3D animations.

Usage:
    python run_medea_ship_fracture.py           # Full run with animations
    python run_medea_ship_fracture.py --quick   # Simulation only, no animations
    python run_medea_ship_fracture.py --no-3d   # Skip 3D animation
"""

import numpy as np
from pathlib import Path
import time
import argparse

from medea_ice_ship import (
    ShipProperties,
    load_medea_floes,
    scale_floes_to_world,
    create_simple_ice_field,
    polygon_area,
    create_proximity_bonds,
    assign_thickness_by_area,
    tessellate_all_floes,
    simulate_ice_ship_interaction,
    animate_ice_ship_2d,
    animate_ice_ship_3d,
    save_simulation_state,
    load_simulation_state,
)
from medea_ice_ship.config import RunConfig, get_default_run_config, get_default_simulation_config

# 5.1 / 5.4: Default seed and run parameters from config
def _get_run_config():
    return get_default_run_config()


def main():
    parser = argparse.ArgumentParser(description='MEDEA Ice Ship Fracture Simulation')
    parser.add_argument('--quick', action='store_true', help='Simulation only, no animations')
    parser.add_argument('--no-3d', action='store_true', help='Skip 3D animation')
    parser.add_argument('--full-map', action='store_true',
                        help='Use full ice map; ship diagonal bottom-left to top-right')
    parser.add_argument('--simple', action='store_true',
                        help='Use minimal synthetic ice field (3 floes) for fast testing (no MEDEA data)')
    default_seed = get_default_run_config().default_seed
    parser.add_argument('--seed', type=int, default=default_seed,
                        help=f'Random seed for reproducibility (default: {default_seed})')
    parser.add_argument('--save-state', type=str, metavar='FILE', default=None,
                        help='Save full simulation state to FILE for replay (pickle)')
    parser.add_argument('--load-state', type=str, metavar='FILE', default=None,
                        help='Load state from FILE and run visualization only (no re-simulation)')
    args = parser.parse_args()
    run_cfg = _get_run_config()

    total_start = time.time()
    np.random.seed(args.seed)

    if args.load_state:
        state = load_simulation_state(args.load_state)
        out_2d = state.get('output_2d') or 'replay_2d.gif'
        out_3d = state.get('output_3d') or 'replay_3d.gif'
        print("\nGenerating 2D animation (replay)...", flush=True)
        animate_ice_ship_2d(
            state['polygons'], state['positions_history'], state['rotations_history'],
            state['bonds_history'], state['bonds'],
            state['ship_history'], state['ship_length'], state['ship_beam'],
            state['centroids'], state['thicknesses'],
            output_file=out_2d, fps=run_cfg.fps_2d
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

    sim_cfg = get_default_simulation_config()
    t0 = time.time()
    print(f"\nRunning fracture simulation ({n_frames} frames)...", flush=True)
    sim_result = simulate_ice_ship_interaction(
        polygons=polygons, centroids=centroids.copy(), bonds=bonds,
        ship_pos=ship_pos, ship_length=ship_length, ship_beam=ship_beam,
        n_frames=n_frames, dt=run_cfg.dt, drift_vel=drift_vel, thicknesses=thicknesses,
        ship_props=ship_props, ship_trajectory=ship_trajectory,
        config=sim_cfg,
    )
    positions_history = sim_result[0]
    rotations_history = sim_result[1]
    bonds_history = sim_result[2]
    ship_history = sim_result[3]
    submergence_history = sim_result[4] if len(sim_result) > 4 else None
    sim_time = time.time() - t0

    n_initial = len(bonds)
    n_intact = bonds_history[-1].sum() if n_initial > 0 else 0
    n_broken = n_initial - n_intact
    pct = 100 * n_broken / n_initial if n_initial > 0 else 0
    print(f"\n  Final: {n_broken}/{n_initial} bonds broken ({pct:.0f}%)", flush=True)
    print(f"  Simulation time: {sim_time:.1f}s", flush=True)

    if args.save_state:
        save_simulation_state(
            polygons, centroids, bonds, thicknesses,
            positions_history, rotations_history, bonds_history, ship_history, submergence_history,
            ship_length, ship_beam, ship_draft, n_frames, dt=run_cfg.dt,
            output_2d='simple_ship_fracture.gif' if args.simple else 'medea_ship_fracture.gif',
            output_3d='simple_ship_fracture_3d.gif' if args.simple else 'medea_ship_fracture_3d.gif',
            simple_mode=args.simple, filepath=args.save_state,
        )

    if args.quick:
        print(f"\nQuick mode complete! Total: {time.time()-total_start:.1f}s")
        return

    out_2d = 'simple_ship_fracture.gif' if args.simple else 'medea_ship_fracture.gif'
    out_3d = 'simple_ship_fracture_3d.gif' if args.simple else 'medea_ship_fracture_3d.gif'
    t0 = time.time()
    print("\nGenerating 2D animation...", flush=True)
    animate_ice_ship_2d(
        polygons, positions_history, rotations_history, bonds_history, bonds,
        ship_history, ship_length, ship_beam, centroids, thicknesses,
        output_file=out_2d, fps=run_cfg.fps_2d
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
        )
        print(f"  3D animation time: {time.time()-t0:.1f}s", flush=True)
    print(f"\nTotal time: {time.time()-total_start:.1f}s")
    print(f"\nOutput: {out_2d}" + (f" and {out_3d}" if not args.no_3d else ""))


if __name__ == '__main__':
    main()
