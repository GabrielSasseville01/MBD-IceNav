#!/usr/bin/env python3
"""
Ship dynamics demo: run the simple ice-field simulation with ship dynamics (2.1)
and plot diagnostics to verify that ice loads affect the ship.

Run from project root:
    python examples/ship_dynamics_demo.py
    uv run python examples/ship_dynamics_demo.py

Optional: compare with clear water (no ice):
    python examples/ship_dynamics_demo.py --compare

Outputs (with --compare):
    ship_dynamics_with_ice_plot.png   - Four-panel diagnostics (with ice)
    ship_dynamics_clear_water_plot.png - Four-panel diagnostics (clear water)
    ship_dynamics_with_ice.gif        - 2D animation (with ice)
    ship_dynamics_clear_water.gif     - 2D animation (clear water)

What to look for (ship dynamics working correctly):
  - With --compare: clear water (green) should track desired speed/path better than with ice (blue).
  - Path: cross-track deviations from zero when in ice; smaller in clear water.
  - Speed: dips or plateau below desired when in ice; closer to desired in clear water.
"""

import argparse
import sys
from pathlib import Path

# Run from project root so run_medea_ship_fracture is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_medea_ship_fracture import (
    create_simple_ice_field,
    tessellate_all_floes,
    create_proximity_bonds,
    assign_thickness_by_area,
    polygon_area,
    ShipProperties,
    simulate_ice_ship_interaction,
    animate_ice_ship_2d,
)


def run_simulation(use_ice: bool, ship_start, ship_end, n_frames, dt, ship_props,
                   ship_length, ship_beam):
    """Run one simulation; use_ice=False means clear water (no floes)."""
    ship_heading = 0.0
    ship_trajectory = []
    for frame in range(n_frames + 1):
        t = frame / n_frames
        pos = ship_start + t * (ship_end - ship_start)
        ship_trajectory.append((pos, ship_heading))
    ship_pos = ship_trajectory[0][0]

    if use_ice:
        sat_polygons, sat_centroids = create_simple_ice_field(seed=42)
        all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
            sat_polygons, sat_centroids, cells_per_100m2=0.25, seed=42
        )
        polygons = all_cells
        centroids = all_cell_centroids
        areas = np.array([polygon_area(p) for p in polygons])
        radii = np.sqrt(areas / np.pi)
        bonds = create_proximity_bonds(centroids, radii, threshold_factor=1.5)
        sat_areas = np.array([polygon_area(p) for p in sat_polygons])
        thicknesses = assign_thickness_by_area(
            sat_areas, min_thickness=0.8, max_thickness=2.5, power=0.25
        )[all_parent_ids]
    else:
        polygons = []
        centroids = np.zeros((0, 2))
        bonds = []
        thicknesses = np.array([])

    result = simulate_ice_ship_interaction(
        polygons=polygons,
        centroids=centroids.copy(),
        bonds=bonds,
        ship_pos=ship_pos,
        ship_length=ship_length,
        ship_beam=ship_beam,
        n_frames=n_frames,
        dt=dt,
        drift_vel=0.0,
        thicknesses=thicknesses,
        ship_props=ship_props,
        ship_trajectory=ship_trajectory,
        return_ship_diagnostics=True,
    )
    geometry = {'polygons': polygons, 'centroids': centroids, 'bonds': bonds, 'thicknesses': thicknesses}
    return result, ship_start, ship_end, geometry


def build_plot_arrays(ship_diagnostics, ship_start, ship_end):
    """Extract arrays for plotting from diagnostics."""
    t = np.array([d['t'] for d in ship_diagnostics])
    F_mag = np.array([np.linalg.norm(d['F_ice_ship']) for d in ship_diagnostics])
    speed = np.array([np.linalg.norm(d['ship_vel']) for d in ship_diagnostics])
    desired_speed = np.array([np.linalg.norm(d['desired_vel']) for d in ship_diagnostics])
    actual_x = np.array([d['ship_pos'][0] for d in ship_diagnostics])
    actual_y = np.array([d['ship_pos'][1] for d in ship_diagnostics])
    desired_x = np.array([d['desired_pos'][0] for d in ship_diagnostics])
    desired_y = np.array([d['desired_pos'][1] for d in ship_diagnostics])
    pos_error = np.sqrt((actual_x - desired_x) ** 2 + (actual_y - desired_y) ** 2)
    path_vec = ship_end - ship_start
    path_len = np.linalg.norm(path_vec)
    u_path = path_vec / path_len
    u_perp = np.array([-u_path[1], u_path[0]])
    along_path_desired = (desired_x - ship_start[0]) * u_path[0] + (desired_y - ship_start[1]) * u_path[1]
    along_path_actual = (actual_x - ship_start[0]) * u_path[0] + (actual_y - ship_start[1]) * u_path[1]
    cross_track_actual = (actual_x - ship_start[0]) * u_perp[0] + (actual_y - ship_start[1]) * u_perp[1]
    return {
        't': t, 'F_mag': F_mag, 'speed': speed, 'desired_speed': desired_speed,
        'pos_error': pos_error, 'along_path_desired': along_path_desired,
        'along_path_actual': along_path_actual, 'cross_track_actual': cross_track_actual,
        'ship_start': ship_start, 'ship_end': ship_end,
    }


def save_four_panel(data, suptitle, output_path, show_ice_force=True, data_other=None, other_label='other'):
    """Save one 2x2 four-panel plot from a data dict (from build_plot_arrays).
    If data_other is provided, overlay it in light gray so comparison is visible."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    t = data['t']

    ax = axes[0, 0]
    ax.axhline(0, color='k', linestyle='--', alpha=0.5, label='Desired path')
    if data_other is not None:
        ax.plot(data_other['along_path_actual'], data_other['cross_track_actual'],
                color='gray', linestyle='--', alpha=0.6, label=other_label)
    ax.plot(data['along_path_actual'], data['cross_track_actual'], 'b-', label='This scenario')
    ax.set_xlabel('Along-path distance (m)')
    ax.set_ylabel('Cross-track error (m)')
    ax.set_title('Path: lateral deviation from desired')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if data_other is not None:
        ax.plot(data_other['t'], data_other['pos_error'], color='gray', linestyle='--', alpha=0.6, label=other_label)
    ax.plot(t, data['pos_error'], 'g-', label='This scenario')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.set_title('Distance from desired path')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, data['desired_speed'], 'k--', alpha=0.7, label='Desired speed')
    if data_other is not None:
        ax.plot(data_other['t'], data_other['speed'], color='gray', linestyle='--', alpha=0.6, label=other_label)
    ax.plot(t, data['speed'], 'b-', label='This scenario')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)', color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_title('Ship speed vs ice force')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    if show_ice_force:
        ax2 = ax.twinx()
        ax2.plot(t, data['F_mag'] / 1e6, 'C1-', alpha=0.8, label='|F_ice| (MN)')
        ax2.set_ylabel('Ice force on ship (MN)', color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        ax2.legend(loc='upper right')

    ax = axes[1, 1]
    ax.plot(t, data['along_path_desired'], 'k--', alpha=0.7, label='Desired along-path')
    if data_other is not None:
        ax.plot(data_other['t'], data_other['along_path_actual'], color='gray', linestyle='--', alpha=0.6, label=other_label)
    ax.plot(t, data['along_path_actual'], 'b-', label='This scenario')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Along-path position (m)')
    ax.set_title('Progress along path')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # If comparing, add text box with max differences (so user can see effect size)
    if data_other is not None:
        n = min(len(data['t']), len(data_other['t']))
        diff_speed = np.abs(data['speed'][:n] - data_other['speed'][:n])
        diff_pos_err = np.abs(data['pos_error'][:n] - data_other['pos_error'][:n])
        diff_along = np.abs(data['along_path_actual'][:n] - data_other['along_path_actual'][:n])
        diff_cross = np.abs(data['cross_track_actual'][:n] - data_other['cross_track_actual'][:n])
        msg = (
            f"Max |diff| vs other scenario:\n"
            f"  Speed: {np.max(diff_speed):.4f} m/s\n"
            f"  Pos error: {np.max(diff_pos_err):.3f} m\n"
            f"  Along-path: {np.max(diff_along):.3f} m\n"
            f"  Cross-track: {np.max(diff_cross):.3f} m"
        )
        fig.text(0.99, 0.01, msg, fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), family='monospace')
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout(rect=[0, 0.10, 1, 0.96] if data_other is not None else None)
    plt.savefig(output_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Ship dynamics demo (with optional clear-water comparison)')
    parser.add_argument('--compare', action='store_true',
                        help='Run with ice and clear water, then plot comparison')
    args = parser.parse_args()

    ship_length = 158.0
    ship_beam = 28.0
    ship_draft = 10.5
    ship_displacement = 23000e3
    dt = 0.05
    n_frames = 250
    ship_props = ShipProperties(ship_length, ship_beam, ship_draft, ship_displacement)

    # Use same path extent for both runs so desired speed is comparable
    ship_start = np.array([-169.0, -10.0])
    ship_end = np.array([173.0, -10.0])

    if args.compare:
        print("Ship dynamics demo: WITH ICE vs CLEAR WATER comparison")
        print("=" * 60)
        print("Running with ice...")
        result_ice, s0, s1, geom_ice = run_simulation(
            use_ice=True, ship_start=ship_start, ship_end=ship_end,
            n_frames=n_frames, dt=dt, ship_props=ship_props,
            ship_length=ship_length, ship_beam=ship_beam,
        )
        print("Running clear water (no ice)...")
        result_water, _, _, geom_water = run_simulation(
            use_ice=False, ship_start=ship_start, ship_end=ship_end,
            n_frames=n_frames, dt=dt, ship_props=ship_props,
            ship_length=ship_length, ship_beam=ship_beam,
        )
        pos_hist_ice, rot_hist_ice, bond_hist_ice, ship_hist_ice, diag_ice = result_ice
        pos_hist_water, rot_hist_water, bond_hist_water, ship_hist_water, diag_water = result_water
        if not diag_ice or not diag_water:
            print("No diagnostics; exiting.")
            return
        data_ice = build_plot_arrays(diag_ice, ship_start, ship_end)
        data_water = build_plot_arrays(diag_water, ship_start, ship_end)

        # Diagnose: are the two runs actually different? (helps explain identical-looking panels)
        n = min(len(data_ice['t']), len(data_water['t']))
        max_diff_speed = np.max(np.abs(data_ice['speed'][:n] - data_water['speed'][:n]))
        max_diff_pos_err = np.max(np.abs(data_ice['pos_error'][:n] - data_water['pos_error'][:n]))
        max_diff_along = np.max(np.abs(data_ice['along_path_actual'][:n] - data_water['along_path_actual'][:n]))
        max_diff_cross = np.max(np.abs(data_ice['cross_track_actual'][:n] - data_water['cross_track_actual'][:n]))
        print("\nMax |ice − clear water| over time:")
        print(f"  Speed:        {max_diff_speed:.6f} m/s")
        print(f"  Position err: {max_diff_pos_err:.4f} m")
        print(f"  Along-path:   {max_diff_along:.4f} m")
        print(f"  Cross-track:  {max_diff_cross:.4f} m")
        if max_diff_speed < 0.01 and max_diff_along < 0.1:
            print("  → Ice has negligible effect on surge (speed/progress); only lateral (cross-track) differs.")

        # 2D GIFs for visual comparison
        print("Creating 2D GIF: with ice...")
        animate_ice_ship_2d(
            geom_ice['polygons'], pos_hist_ice, rot_hist_ice, bond_hist_ice, geom_ice['bonds'],
            ship_hist_ice, ship_length, ship_beam, geom_ice['centroids'], geom_ice['thicknesses'],
            output_file=str(ROOT / 'ship_dynamics_with_ice.gif'), fps=20,
        )
        print("Creating 2D GIF: clear water...")
        animate_ice_ship_2d(
            geom_water['polygons'], pos_hist_water, rot_hist_water, bond_hist_water, geom_water['bonds'],
            ship_hist_water, ship_length, ship_beam, geom_water['centroids'], geom_water['thicknesses'],
            output_file=str(ROOT / 'ship_dynamics_clear_water.gif'), fps=20,
        )

        # Separate four-panel plots for each scenario (overlay other scenario in gray + diff box)
        print("Saving four-panel plot: with ice...")
        save_four_panel(
            data_ice,
            suptitle='Ship dynamics: with ice (same path and control)',
            output_path=ROOT / 'ship_dynamics_with_ice_plot.png',
            show_ice_force=True,
            data_other=data_water,
            other_label='Clear water',
        )
        print(f"  Saved: ship_dynamics_with_ice_plot.png")
        print("Saving four-panel plot: clear water...")
        save_four_panel(
            data_water,
            suptitle='Ship dynamics: clear water (same path and control)',
            output_path=ROOT / 'ship_dynamics_clear_water_plot.png',
            show_ice_force=True,  # will show zero force
            data_other=data_ice,
            other_label='With ice',
        )
        print(f"  Saved: ship_dynamics_clear_water_plot.png")
        print("GIFs: ship_dynamics_with_ice.gif, ship_dynamics_clear_water.gif")
        return

    # Single run with ice (original behaviour)
    print("Ship dynamics demo: simple ice field + diagnostics")
    print("=" * 60)
    print("Creating simple ice field...")
    result, ship_start, ship_end, _ = run_simulation(
        use_ice=True, ship_start=ship_start, ship_end=ship_end,
        n_frames=n_frames, dt=dt, ship_props=ship_props,
        ship_length=ship_length, ship_beam=ship_beam,
    )
    positions_history, rotations_history, bonds_history, ship_history, ship_diagnostics = result

    if not ship_diagnostics:
        print("No diagnostics (ship_trajectory was not used?). Exiting.")
        return

    data = build_plot_arrays(ship_diagnostics, ship_start, ship_end)
    t = data['t']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.axhline(0, color='k', linestyle='--', alpha=0.5, label='Desired path')
    ax.plot(data['along_path_actual'], data['cross_track_actual'], 'b-', label='Actual path (dynamics)')
    ax.set_xlabel('Along-path distance (m)')
    ax.set_ylabel('Cross-track error (m)')
    ax.set_title('Path: lateral deviation from desired')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, data['pos_error'], 'g-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.set_title('Distance from desired path')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, data['desired_speed'], 'k--', alpha=0.7, label='Desired speed')
    ax.plot(t, data['speed'], 'b-', label='Actual speed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)', color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_title('Ship speed vs ice force')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, data['F_mag'] / 1e6, 'C1-', alpha=0.8, label='|F_ice| (MN)')
    ax2.set_ylabel('Ice force on ship (MN)', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    ax2.legend(loc='upper right')

    ax = axes[1, 1]
    ax.plot(t, data['along_path_desired'], 'k--', alpha=0.7, label='Desired along-path')
    ax.plot(t, data['along_path_actual'], 'b-', label='Actual along-path')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Along-path position (m)')
    ax.set_title('Progress along path (lag when in ice)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Ship dynamics demo: ice loads affect the ship (2.1)', fontsize=12)
    plt.tight_layout()
    out = ROOT / 'ship_dynamics_demo.png'
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")
    print("\nRun with --compare to see side-by-side with clear water.")


if __name__ == '__main__':
    main()
