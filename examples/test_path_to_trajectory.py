#!/usr/bin/env python3
"""
Test path → ship_trajectory conversion in isolation (Section 4.3 of PLANNER_INTEGRATION_CHECKLIST).

Uses a dummy path: 10 waypoints from (0, 0) to (100, 0) with heading 0.
n_frames=100 → trajectory must have length 101, first/last match path start/goal,
and positions must progress along the path.

Run: uv run python examples/test_path_to_trajectory.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from medea_ice_ship import path_to_ship_trajectory


def main():
    # Dummy path: 10 waypoints (0,0) to (100,0), heading 0
    path = np.zeros((10, 3))
    path[:, 0] = np.linspace(0, 100, 10)
    path[:, 1] = 0.0
    path[:, 2] = 0.0

    n_frames = 100
    trajectory = path_to_ship_trajectory(path, n_frames)

    assert len(trajectory) == n_frames + 1, (
        f"Expected len(trajectory) = n_frames + 1 = {n_frames + 1}, got {len(trajectory)}"
    )
    print(f"len(trajectory) == n_frames + 1 == {len(trajectory)} [OK]")

    tol = 1e-6
    assert np.allclose(trajectory[0][0], path[0, :2], atol=tol), (
        f"trajectory[0] position {trajectory[0][0]} != path[0,:2] {path[0, :2]}"
    )
    assert abs(trajectory[0][1] - path[0, 2]) < tol, (
        f"trajectory[0] heading {trajectory[0][1]} != path[0,2] {path[0, 2]}"
    )
    assert np.allclose(trajectory[-1][0], path[-1, :2], atol=tol), (
        f"trajectory[-1] position {trajectory[-1][0]} != path[-1,:2] {path[-1, :2]}"
    )
    assert abs(trajectory[-1][1] - path[-1, 2]) < tol, (
        f"trajectory[-1] heading {trajectory[-1][1]} != path[-1,2] {path[-1, 2]}"
    )
    print("trajectory[0] ≈ path[0], trajectory[-1] ≈ path[-1] [OK]")

    # Positions should progress along path (x increasing)
    xs = np.array([trajectory[k][0][0] for k in range(len(trajectory))])
    assert np.all(np.diff(xs) >= -1e-6), "Positions should progress (x non-decreasing)"
    assert xs[0] <= 0.0 + tol and xs[-1] >= 100.0 - tol, "Path should span [0, 100] in x"
    print("Positions progress along path (x from 0 to 100) [OK]")

    print("\nSection 4.3 path-to-trajectory checks passed.")


if __name__ == "__main__":
    main()
