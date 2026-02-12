#!/usr/bin/env python3
"""
Test creating a MEDEA lattice planner (Section 2.3 of PLANNER_INTEGRATION_CHECKLIST).

Pipeline:
  simple ice (3 floes) -> tessellated cells -> thicknesses
  -> costmap (Section 1 helper) -> lattice planner (Section 2 helper)

Checks:
  - Planner is created without error.
  - CostMap shape matches expected (rows, cols) from world size and resolution.
  - Start/goal (same as simple run) are inside costmap bounds.

Run:
  uv run python examples/test_planner_creation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from medea_ice_ship import (
    create_simple_ice_field,
    tessellate_all_floes,
    polygon_area,
    assign_thickness_by_area,
    build_costmap_from_ice_state,
    create_medea_lattice_planner,
)
from medea_ice_ship.config import get_default_run_config


def main():
    seed = 42
    np.random.seed(seed)
    run_cfg = get_default_run_config()

    # Mirror simple ice construction from run_medea_ship_fracture.py
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

    # Use planning-specific resolution and a generous margin so the simple-mode
    # start/goal lie inside the planning costmap.
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

    # Simple-mode start/goal as in run_medea_ship_fracture.py
    y_mid = (y_min + y_max) / 2.0
    ship_length = run_cfg.ship_length
    path_margin = run_cfg.path_margin
    start = np.array(
        [x_min - ship_length / 2.0 - path_margin, y_mid, 0.0]
    )  # heading 0 rad
    goal = np.array(
        [x_max + ship_length / 2.0 + path_margin, y_mid]
    )

    # Ensure start/goal are within costmap bounds (helper will also validate).
    ox, oy = cost_map.x_origin, cost_map.y_origin
    wx, wy = cost_map.width, cost_map.height
    assert ox <= start[0] < ox + wx and oy <= start[1] < oy + wy, "Start outside costmap bounds"
    assert ox <= goal[0] < ox + wx and oy <= goal[1] < oy + wy, "Goal outside costmap bounds"
    print("Start and goal lie inside costmap bounds [OK]")

    # Create planner
    planner = create_medea_lattice_planner(
        cost_map=cost_map,
        bounds=bounds,
        start=(float(start[0]), float(start[1]), float(start[2])),
        goal=(float(goal[0]), float(goal[1])),
        run_cfg=run_cfg,
    )
    assert planner is not None
    print("Lattice planner created successfully [OK]")

    # CostMap shape sanity check
    expected_width = (x_max - x_min) + 2.0 * margin
    expected_height = (y_max - y_min) + 2.0 * margin
    expected_cols = int(np.ceil(expected_width / resolution))
    expected_rows = int(np.ceil(expected_height / resolution))
    assert cost_map.shape == (expected_rows, expected_cols), (
        f"Expected cost_map.shape=({expected_rows},{expected_cols}), got {cost_map.shape}"
    )
    print(f"CostMap shape {cost_map.shape} (expected ({expected_rows}, {expected_cols})) [OK]")

    print("\nSection 2.3 planner-creation checks passed.")


if __name__ == "__main__":
    main()

