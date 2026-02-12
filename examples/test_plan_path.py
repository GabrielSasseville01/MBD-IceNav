#!/usr/bin/env python3
"""
Test planning a path on the simple ice field (Section 3 of PLANNER_INTEGRATION_CHECKLIST).

Pipeline: simple ice → costmap → planner → plan_path(start, goal).

Checks:
  - Path is not None.
  - path.shape[1] == 3 ([x, y, heading]).
  - path[0] near start, path[-1] near goal (within goal_tolerance).
  - x increases roughly from start to goal (path[-1, 0] >= path[0, 0] for left-to-right).

Uses a smaller planning_turning_radius so the planner finds a path (same as visualize_planned_path).
Run: uv run python examples/test_plan_path.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from dataclasses import replace

from medea_ice_ship import (
    create_simple_ice_field,
    tessellate_all_floes,
    polygon_area,
    assign_thickness_by_area,
    build_costmap_from_ice_state,
    create_medea_lattice_planner,
    plan_path,
)
from medea_ice_ship.config import get_default_run_config


def main():
    seed = 42
    np.random.seed(seed)
    run_cfg = get_default_run_config()
    run_cfg = replace(run_cfg, planning_turning_radius=50.0)

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

    cost_map, bounds = build_costmap_from_ice_state(
        polygons,
        centroids,
        thicknesses,
        x_min,
        x_max,
        y_min,
        y_max,
        resolution=run_cfg.planning_resolution,
        margin=run_cfg.planning_margin,
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

    path = plan_path(planner, start, goal, run_cfg, cost_map=cost_map)

    assert path is not None, "Planner returned None (no path found)"
    print("path is not None [OK]")

    assert path.ndim == 2 and path.shape[1] == 3, (
        f"Expected path shape (N, 3), got {path.shape}"
    )
    print(f"path.shape = {path.shape} [OK]")

    tol = run_cfg.planning_goal_tolerance
    assert np.hypot(path[0, 0] - start[0], path[0, 1] - start[1]) < tol + 1.0, (
        f"path[0] ({path[0, 0]:.1f}, {path[0, 1]:.1f}) not near start ({start[0]:.1f}, {start[1]:.1f})"
    )
    assert np.hypot(path[-1, 0] - goal[0], path[-1, 1] - goal[1]) < tol + 1.0, (
        f"path[-1] ({path[-1, 0]:.1f}, {path[-1, 1]:.1f}) not near goal ({goal[0]:.1f}, {goal[1]:.1f})"
    )
    print("path[0] near start, path[-1] near goal [OK]")

    assert path[-1, 0] >= path[0, 0] - 10.0, (
        f"Expected x to increase (start-to-goal left-to-right); path x: {path[0, 0]:.1f} -> {path[-1, 0]:.1f}"
    )
    print("path x increases roughly start to goal [OK]")

    print("\nSection 3 plan-path checks passed.")


if __name__ == "__main__":
    main()
