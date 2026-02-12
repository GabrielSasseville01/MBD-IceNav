#!/usr/bin/env python3
"""
Example: run the lattice planner on the simple ice field and visualize the path.

Pipeline:
  simple ice (3 floes) -> costmap -> planner -> plan(start, goal)
  -> plot costmap + ice outlines + planned path (start/goal marked).

Uses a smaller planning_turning_radius (50 m) so that primitive step length
fits the costmap; with the default 350 m, step length would exceed the map
and no path would be found.

Run:
  uv run python examples/visualize_planned_path.py

Output:
  planned_path_visualization.png  (and .pdf) in project root.
  If no path is found, exits with code 1 and a message.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

from medea_ice_ship import (
    create_simple_ice_field,
    tessellate_all_floes,
    polygon_area,
    assign_thickness_by_area,
    build_costmap_from_ice_state,
    create_medea_lattice_planner,
)
from medea_ice_ship.config import get_default_run_config
from dataclasses import replace


def main():
    seed = 42
    np.random.seed(seed)
    run_cfg = get_default_run_config()
    # Primitives use length = turning_radius * resolution; default 350*2 = 700 m is longer
    # than the map, so no path is found. Use a smaller turning radius so step length ~ 100 m.
    run_cfg = replace(run_cfg, planning_turning_radius=50.0)

    # Same ice and costmap as planner-creation test
    sat_polygons, sat_centroids = create_simple_ice_field(seed=seed)
    all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
        sat_polygons,
        sat_centroids,
        cells_per_100m2=run_cfg.cells_per_100m2_simple,
        seed=seed,
    )
    polygons = all_cells
    centroids = all_cell_centroids
    sat_areas = np.array([polygon_area(p) for p in sat_polygons])
    sat_thicknesses = assign_thickness_by_area(
        sat_areas,
        min_thickness=run_cfg.thickness_min,
        max_thickness=run_cfg.thickness_max,
        power=run_cfg.thickness_power,
    )
    thicknesses = sat_thicknesses[all_parent_ids]

    all_verts = np.vstack(polygons)
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()

    resolution = run_cfg.planning_resolution
    margin = run_cfg.planning_margin
    cost_map, bounds = build_costmap_from_ice_state(
        polygons,
        centroids,
        thicknesses,
        x_min,
        x_max,
        y_min,
        y_max,
        resolution=resolution,
        margin=margin,
        ship_mass=run_cfg.ship_displacement,
    )

    y_mid = (y_min + y_max) / 2.0
    ship_length = run_cfg.ship_length
    path_margin = run_cfg.path_margin
    start = (x_min - ship_length / 2.0 - path_margin, y_mid, 0.0)
    goal = (x_max + ship_length / 2.0 + path_margin, y_mid)

    planner = create_medea_lattice_planner(
        cost_map=cost_map,
        bounds=bounds,
        start=start,
        goal=goal,
        run_cfg=run_cfg,
    )

    path = planner.plan(
        start=start,
        goal=goal,
        max_iterations=run_cfg.planning_max_iterations,
    )

    if path is None:
        print("Planner did not find a path.", file=sys.stderr)
        print(f"  Nodes expanded: {planner.nodes_expanded}", file=sys.stderr)
        sys.exit(1)

    print(f"Path found: {len(path)} waypoints")
    print(f"  Nodes expanded: {planner.nodes_expanded}, generated: {planner.nodes_generated}")
    print(f"  Start: ({path[0, 0]:.1f}, {path[0, 1]:.1f}), goal: ({path[-1, 0]:.1f}, {path[-1, 1]:.1f})")

    # Plot: costmap + ice + path
    cols, rows = cost_map.cols, cost_map.rows
    x_origin = bounds["x_origin"]
    y_origin = bounds["y_origin"]
    X = x_origin + np.arange(cols + 1) * resolution
    Y = y_origin + np.arange(rows + 1) * resolution

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cost = cost_map.cost_map
    norm = colors.LogNorm(vmin=max(0.5, cost.min()), vmax=cost.max())
    ax.pcolormesh(X, Y, cost, norm=norm, cmap="viridis", shading="flat")

    for poly in polygons:
        closed = np.vstack([poly, poly[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=0.4, alpha=0.9)

    rect = Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False, edgecolor="cyan", linewidth=1.5, linestyle="--", label="Ice domain",
    )
    ax.add_patch(rect)

    # Planned path
    ax.plot(path[:, 0], path[:, 1], "r-", linewidth=2, label="Planned path", zorder=5)
    ax.plot(start[0], start[1], "go", markersize=10, label="Start", zorder=6)
    ax.plot(goal[0], goal[1], "rs", markersize=10, label="Goal", zorder=6)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title("Lattice planner: simple ice field\nPath should avoid high-cost (ice) regions")
    ax.grid(True, alpha=0.3)

    out_dir = Path(__file__).resolve().parent.parent
    png_path = out_dir / "planned_path_visualization.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")
    fig.savefig(out_dir / "planned_path_visualization.pdf", bbox_inches="tight")
    print(f"Saved: {out_dir / 'planned_path_visualization.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
