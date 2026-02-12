#!/usr/bin/env python3
"""
Minimal test for CostMap origin support (Section 0.3 of PLANNER_INTEGRATION_CHECKLIST).

Verifies that:
- world_to_grid / grid_to_world round-trip correctly when origin is non-zero
- A point at (x_min, y_min) maps to grid cell (0, 0)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dpdem.planning.cost_map import CostMap


def main():
    # Domain like MEDEA simple: e.g. x_min=-70, y_min=-41 (not origin-aligned)
    x_min, y_min = -70.0, -41.0
    width, height = 150.0, 65.0
    resolution = 2.0
    x_origin = x_min
    y_origin = y_min

    cm = CostMap(
        width=width,
        height=height,
        resolution=resolution,
        x_origin=x_origin,
        y_origin=y_origin,
    )

    # (x_min, y_min) should map to grid (0, 0)
    row, col = cm.world_to_grid(x_min, y_min)
    assert row == 0 and col == 0, f"Expected (0,0), got (row,col)=({row},{col})"
    print(f"world_to_grid({x_min}, {y_min}) = (row={row}, col={col}) [OK]")

    # grid_to_world(0, 0) should be cell center near (x_min, y_min)
    x0, y0 = cm.grid_to_world(0, 0)
    expected_x = x_origin + 0.5 * resolution
    expected_y = y_origin + 0.5 * resolution
    assert abs(x0 - expected_x) < 1e-9 and abs(y0 - expected_y) < 1e-9
    print(f"grid_to_world(0, 0) = ({x0}, {y0}) [OK]")

    # Round-trip: world -> grid -> world (cell center)
    x_w, y_w = -50.0, -10.0
    r, c = cm.world_to_grid(x_w, y_w)
    x_back, y_back = cm.grid_to_world(r, c)
    # Cell center may not equal (x_w, y_w); check we're in the right cell
    assert cm.in_bounds(r, c), "Grid indices out of bounds"
    assert abs((x_back - x_w) / resolution) <= 0.6 and abs((y_back - y_w) / resolution) <= 0.6
    print(f"Round-trip world ({x_w},{y_w}) -> grid ({r},{c}) -> world ({x_back},{y_back}) [OK]")

    # Bounds: point just inside and just outside
    assert cm.get_cost_world(x_origin + 0.1, y_origin + 0.1) == cm.base_cost
    print("Cost at interior point [OK]")

    print("\nAll Section 0.3 costmap origin checks passed.")


if __name__ == "__main__":
    main()
