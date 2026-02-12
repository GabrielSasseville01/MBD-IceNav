#!/usr/bin/env python3
"""
Test building a costmap from MEDEA simple ice state (Section 1.3 of PLANNER_INTEGRATION_CHECKLIST).

- Creates a simple ice field (3 floes), tessellates, assigns thickness.
- Builds costmap via build_costmap_from_ice_state.
- Checks: (1) cost at ice centroids is higher than base_cost; (2) cost in open water is base_cost;
  (3) costmap shape and resolution match expected (width/height in world units / resolution).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from medea_ice_ship import (
    create_simple_ice_field,
    tessellate_all_floes,
    polygon_area,
    create_proximity_bonds,
    assign_thickness_by_area,
)
from medea_ice_ship.costmap_build import build_costmap_from_ice_state
from medea_ice_ship.config import get_default_run_config


def main():
    seed = 42
    np.random.seed(seed)
    run_cfg = get_default_run_config()

    # Mirror run_medea_ship_fracture --simple ice construction
    sat_polygons, sat_centroids = create_simple_ice_field(seed=seed)
    all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
        sat_polygons, sat_centroids,
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

    resolution = 2.0
    margin = 20.0
    cost_map, bounds = build_costmap_from_ice_state(
        polygons, centroids, thicknesses,
        x_min, x_max, y_min, y_max,
        resolution=resolution,
        margin=margin,
        ship_mass=run_cfg.ship_displacement,
    )

    # (1) Cost at ice centroids should be > base_cost
    base = cost_map.base_cost
    for i in range(min(5, len(centroids))):
        c = centroids[i]
        cost_at = cost_map.get_cost_world(c[0], c[1])
        assert cost_at > base, f"Expected cost at centroid {i} ({c}) to be > base_cost {base}, got {cost_at}"
    print(f"Cost at sample ice centroids > base_cost ({base}) [OK]")

    # (2) Cost far from ice (e.g. outside domain) or in a known open-water cell
    # Use a point well outside the ice domain but inside the costmap (within margin)
    open_x = x_min - margin / 2
    open_y = y_min - margin / 2
    cost_open = cost_map.get_cost_world(open_x, open_y)
    assert cost_open == base, f"Expected open-water cost {base}, got {cost_open}"
    print(f"Cost at open-water point ({open_x}, {open_y}) == base_cost [OK]")

    # (3) Costmap shape: rows/cols = ceil(height/resolution), ceil(width/resolution)
    expected_width_world = (x_max - x_min) + 2 * margin
    expected_height_world = (y_max - y_min) + 2 * margin
    expected_cols = int(np.ceil(expected_width_world / resolution))
    expected_rows = int(np.ceil(expected_height_world / resolution))
    assert cost_map.shape == (expected_rows, expected_cols), (
        f"Expected shape (rows, cols) = ({expected_rows}, {expected_cols}), got {cost_map.shape}"
    )
    print(f"Costmap shape (rows, cols) = {cost_map.shape} (expected ({expected_rows}, {expected_cols})) [OK]")

    # Bounds returned
    assert bounds["x_origin"] == x_min - margin
    assert bounds["y_origin"] == y_min - margin
    assert bounds["width"] == expected_width_world
    assert bounds["height"] == expected_height_world
    print(f"Bounds: x_origin={bounds['x_origin']}, y_origin={bounds['y_origin']}, "
          f"width={bounds['width']}, height={bounds['height']} [OK]")

    print("\nAll Section 1.3 costmap-from-ice checks passed.")


if __name__ == "__main__":
    main()
