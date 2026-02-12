#!/usr/bin/env python3
"""
Thick-ice (unbreakable) demo: one very thick floe in the ship path to test that
the ship reacts realistically when ice is too strong to break.

The floe is given a constant high thickness (e.g. 10 m) so that:
- Breaking force F_break = σ_f(h) * w * h² is very large (ice won't fracture).
- Collision force is capped at F_break (2.2), so the ship experiences strong
  resistance and should slow, stop, or be deflected instead of plowing through.

Run from project root:
    python examples/thick_ice_unbreakable_demo.py
    uv run python examples/thick_ice_unbreakable_demo.py

Outputs:
    thick_ice_unbreakable.gif   - 2D animation
    thick_ice_unbreakable_plot.png - Four-panel diagnostics
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_medea_ship_fracture import (
    tessellate_all_floes,
    create_proximity_bonds,
    polygon_area,
    polygon_centroid,
    ShipProperties,
    IceProperties,
    simulate_ice_ship_interaction,
    animate_ice_ship_2d,
)


# One large floe: rectangle in the ship path (centered on y = -10, ship path)
# Size 100 m x 100 m, 10 m thickness — effectively unbreakable
FLOE_HALF_WIDTH = 50.0   # m → 100 m total width
FLOE_HALF_HEIGHT = 50.0  # m → 100 m total length
FLOE_CENTER = np.array([0.0, -10.0])  # on ship path
THICK_ICE_M = 10.0       # m — "unbreakable" thickness (very high F_break)


def create_thick_floe():
    """Single rectangular floe, centered on ship path."""
    cx, cy = FLOE_CENTER[0], FLOE_CENTER[1]
    poly = np.array([
        [cx - FLOE_HALF_WIDTH, cy - FLOE_HALF_HEIGHT],
        [cx + FLOE_HALF_WIDTH, cy - FLOE_HALF_HEIGHT],
        [cx + FLOE_HALF_WIDTH, cy + FLOE_HALF_HEIGHT],
        [cx - FLOE_HALF_WIDTH, cy + FLOE_HALF_HEIGHT],
    ])
    centroid = np.array([cx, cy])
    return [poly], np.array([centroid])


def main():
    print("Thick-ice (unbreakable) demo: ship vs very thick floe")
    print("=" * 60)

    ship_length = 158.0
    ship_beam = 28.0
    ship_draft = 10.5
    ship_displacement = 23000e3
    dt = 0.05
    n_frames = 350
    ship_props = ShipProperties(ship_length, ship_beam, ship_draft, ship_displacement)

    sat_polygons, sat_centroids = create_thick_floe()
    print(f"Creating single floe {2*FLOE_HALF_WIDTH:.0f}×{2*FLOE_HALF_HEIGHT:.0f} m at {FLOE_CENTER}")
    print(f"  Thickness set to {THICK_ICE_M} m (effectively unbreakable)")

    all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
        sat_polygons, sat_centroids, cells_per_100m2=0.3, seed=42
    )
    polygons = all_cells
    centroids = all_cell_centroids
    areas = np.array([polygon_area(p) for p in polygons])
    radii = np.sqrt(areas / np.pi)
    bonds = create_proximity_bonds(centroids, radii, threshold_factor=1.5)
    # All cells get the same high thickness — ice is too strong to break
    thicknesses = np.full(len(polygons), THICK_ICE_M)

    all_verts = np.vstack(polygons)
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
    ship_start = np.array([x_min - ship_length - 30.0, (y_min + y_max) / 2])
    ship_end = np.array([x_max + ship_length + 30.0, (y_min + y_max) / 2])
    ship_heading = 0.0
    ship_trajectory = []
    for frame in range(n_frames + 1):
        t = frame / n_frames
        pos = ship_start + t * (ship_end - ship_start)
        ship_trajectory.append((pos, ship_heading))
    ship_pos = ship_trajectory[0][0]

    # Sanity: breaking force for one cell
    w_contact = min(2.0 * radii[0], 4.0)
    F_break_sample = IceProperties.breaking_force(THICK_ICE_M, w_contact)
    print(f"  Sample F_break (one cell): {F_break_sample/1e6:.1f} MN")

    print("\nRunning simulation (ship dynamics + ice)...")
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
    pos_hist, rot_hist, bond_hist, ship_hist, diag = result
    if not diag:
        print("No diagnostics.")
        return

    # Four-panel plot
    t = np.array([d['t'] for d in diag])
    path_vec = ship_end - ship_start
    u_path = path_vec / np.linalg.norm(path_vec)
    u_perp = np.array([-u_path[1], u_path[0]])
    desired_x = np.array([d['desired_pos'][0] for d in diag])
    desired_y = np.array([d['desired_pos'][1] for d in diag])
    actual_x = np.array([d['ship_pos'][0] for d in diag])
    actual_y = np.array([d['ship_pos'][1] for d in diag])
    along_path_desired = (desired_x - ship_start[0]) * u_path[0] + (desired_y - ship_start[1]) * u_path[1]
    along_path_actual = (actual_x - ship_start[0]) * u_path[0] + (actual_y - ship_start[1]) * u_path[1]
    cross_track = (actual_x - ship_start[0]) * u_perp[0] + (actual_y - ship_start[1]) * u_perp[1]
    pos_error = np.sqrt((actual_x - desired_x)**2 + (actual_y - desired_y)**2)
    speed = np.array([np.linalg.norm(d['ship_vel']) for d in diag])
    desired_speed = np.array([np.linalg.norm(d['desired_vel']) for d in diag])
    F_mag = np.array([np.linalg.norm(d['F_ice_ship']) for d in diag])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.plot(along_path_actual, cross_track, 'b-')
    ax.set_xlabel('Along-path distance (m)')
    ax.set_ylabel('Cross-track error (m)')
    ax.set_title('Path: lateral deviation')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, pos_error, 'g-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.set_title('Distance from desired path')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, desired_speed, 'k--', alpha=0.7, label='Desired speed')
    ax.plot(t, speed, 'b-', label='Actual speed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, F_mag / 1e6, 'C1-', alpha=0.8, label='|F_ice| (MN)')
    ax2.set_ylabel('Ice force (MN)', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    ax2.legend(loc='upper right')

    ax = axes[1, 1]
    ax.plot(t, along_path_desired, 'k--', alpha=0.7, label='Desired')
    ax.plot(t, along_path_actual, 'b-', label='Actual')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Along-path position (m)')
    ax.set_title('Progress along path')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Thick ice ({THICK_ICE_M} m) — ship vs unbreakable floe', fontsize=12)
    plt.tight_layout()
    out_plot = ROOT / 'thick_ice_unbreakable_plot.png'
    plt.savefig(out_plot, dpi=120)
    plt.close()
    print(f"Saved: {out_plot}")

    print("Creating 2D GIF...")
    animate_ice_ship_2d(
        polygons, pos_hist, rot_hist, bond_hist, bonds,
        ship_hist, ship_length, ship_beam, centroids, thicknesses,
        output_file=str(ROOT / 'thick_ice_unbreakable.gif'), fps=20,
    )
    print("Done. Check: ship should slow/stop when hitting the thick floe; ice force should spike.")


if __name__ == '__main__':
    main()
