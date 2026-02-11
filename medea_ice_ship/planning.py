"""Planning helpers for MEDEA ice–ship simulation.

Section 2–4 of PLANNER_INTEGRATION_CHECKLIST: create a lattice planner,
run A* to get a path, and convert the path to ship_trajectory for the simulator.
"""

from typing import Tuple, Dict, Optional, List

import numpy as np

from dpdem.planning.lattice_planner import LatticePlanner, Primitives
from dpdem.planning.cost_map import CostMap
from medea_ice_ship.config import RunConfig


def _point_in_costmap_bounds(cost_map: CostMap, x: float, y: float) -> bool:
    """Return True if (x, y) is inside the costmap world bounds."""
    ox, oy = cost_map.x_origin, cost_map.y_origin
    return (ox <= x < ox + cost_map.width) and (oy <= y < oy + cost_map.height)


def create_medea_lattice_planner(
    cost_map: CostMap,
    bounds: Dict[str, float],
    start: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg: RunConfig,
) -> LatticePlanner:
    """
    Create a DPDEM LatticePlanner configured for the MEDEA costmap.

    This helper:
      - Validates that start and goal lie inside the costmap world bounds.
      - Creates motion primitives with MEDEA planning parameters.
      - Wraps them in a LatticePlanner that uses the given CostMap.

    Args:
        cost_map: CostMap built from MEDEA ice state (Section 1 helper).
        bounds: Dict from build_costmap_from_ice_state with at least:
            x_origin, y_origin, width, height (all in world units).
        start: Ship start pose (x, y, heading) in world coordinates [m, m, rad].
        goal: Ship goal position (x, y) in world coordinates [m, m].
        run_cfg: RunConfig providing ship size/mass and planning parameters.

    Returns:
        Configured LatticePlanner ready for planning (see Section 3).
    """
    sx, sy, _ = start
    gx, gy = goal

    # Ensure start/goal are inside the costmap bounds (same frame as simulation).
    if not _point_in_costmap_bounds(cost_map, sx, sy):
        raise ValueError(
            f"Start ({sx:.1f}, {sy:.1f}) is outside costmap bounds "
            f"[x={cost_map.x_origin:.1f}..{cost_map.x_origin + cost_map.width:.1f}, "
            f"y={cost_map.y_origin:.1f}..{cost_map.y_origin + cost_map.height:.1f}]"
        )
    if not _point_in_costmap_bounds(cost_map, gx, gy):
        raise ValueError(
            f"Goal ({gx:.1f}, {gy:.1f}) is outside costmap bounds "
            f"[x={cost_map.x_origin:.1f}..{cost_map.x_origin + cost_map.width:.1f}, "
            f"y={cost_map.y_origin:.1f}..{cost_map.y_origin + cost_map.height:.1f}]"
        )

    # Motion primitives: configured from planning parameters (resolution, turning radius, headings).
    primitives = Primitives(
        turning_radius=run_cfg.planning_turning_radius,
        num_headings=run_cfg.planning_num_headings,
        scale=run_cfg.planning_resolution,
    )

    # Planner uses the existing MEDEA cost map (with correct origin and resolution).
    planner = LatticePlanner(
        primitives=primitives,
        cost_map=cost_map,
        ship_length=run_cfg.ship_length,
        ship_width=run_cfg.ship_beam,
        heuristic_weight=run_cfg.planning_heuristic_weight,
        goal_tolerance=run_cfg.planning_goal_tolerance,
    )

    return planner


def plan_path(
    planner: LatticePlanner,
    start: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg: RunConfig,
) -> Optional[np.ndarray]:
    """
    Run the lattice planner from start to goal (Section 3).

    Uses start and goal in the same world frame as the costmap/simulation.
    **Behaviour when no path is found:** returns `None`. Callers should check
    and either abort with a clear message or fall back to a fixed trajectory.

    Args:
        planner: Configured LatticePlanner (from create_medea_lattice_planner).
        start: (x, y, heading) in world coordinates [m, m, rad].
        goal: (x, y) in world coordinates [m, m].
        run_cfg: RunConfig for planning_max_iterations.

    Returns:
        Path as Nx3 array [x, y, heading] in world coordinates, or None if no path found.
    """
    return planner.plan(
        start=start,
        goal=goal,
        max_iterations=run_cfg.planning_max_iterations,
    )


def path_to_ship_trajectory(
    path: np.ndarray,
    n_frames: int,
) -> List[Tuple[np.ndarray, float]]:
    """
    Convert a planned path (N×3) to ship_trajectory for simulate_ice_ship_interaction (Section 4).

    Produces a list of (position_2d, heading) of length n_frames + 1. First entry is path
    start; last entry is path goal. Intermediate points are interpolated along the path by
    arc length so the ship progresses evenly over the frames. Heading is interpolated
    between waypoints (linear in angle, with wrap).

    Args:
        path: N×3 array [x, y, heading] in world coordinates.
        n_frames: Number of simulation frames (trajectory will have n_frames + 1 entries).

    Returns:
        List of (np.array([x, y]), heading), length n_frames + 1. trajectory[0] matches
        path[0]; trajectory[-1] matches path[-1].
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError(f"path must have shape (N, 3), got {path.shape}")
    n_pts = path.shape[0]
    if n_pts < 2:
        single = (np.array([path[0, 0], path[0, 1]], dtype=float), float(path[0, 2]))
        return [single] * (n_frames + 1)

    # Cumulative arc length along path
    seg_lens = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    s = np.zeros(n_pts)
    s[1:] = np.cumsum(seg_lens)
    S = s[-1]
    if S <= 0:
        single = (path[0, :2].copy(), float(path[0, 2]))
        return [single] * (n_frames + 1)

    def interpolate_angle(a0: float, a1: float, frac: float) -> float:
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        if da > np.pi:
            da -= 2 * np.pi
        elif da < -np.pi:
            da += 2 * np.pi
        out = a0 + frac * da
        return (out + np.pi) % (2 * np.pi) - np.pi

    trajectory = []
    for k in range(n_frames + 1):
        if k == 0:
            trajectory.append((path[0, :2].copy(), float(path[0, 2])))
            continue
        if k == n_frames:
            trajectory.append((path[-1, :2].copy(), float(path[-1, 2])))
            continue
        t = k / n_frames
        target_s = t * S
        # Segment i such that s[i] <= target_s < s[i+1]
        i = np.searchsorted(s, target_s, side="right") - 1
        i = max(0, min(i, n_pts - 2))
        if seg_lens[i] > 0:
            frac = (target_s - s[i]) / seg_lens[i]
        else:
            frac = 1.0
        frac = np.clip(frac, 0.0, 1.0)
        x = (1 - frac) * path[i, 0] + frac * path[i + 1, 0]
        y = (1 - frac) * path[i, 1] + frac * path[i + 1, 1]
        heading = interpolate_angle(path[i, 2], path[i + 1, 2], frac)
        trajectory.append((np.array([x, y], dtype=float), float(heading)))

    return trajectory

