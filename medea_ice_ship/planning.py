"""Planning helpers for MEDEA ice–ship simulation.

Section 2–4 of PLANNER_INTEGRATION_CHECKLIST: create a lattice planner,
run A* to get a path, and convert the path to ship_trajectory for the simulator.

Planner dynamics and footprint:
  - Lattice: motion primitives respect minimum turning_radius (Dubins-like arcs);
    path cost uses ship center + four hull corners (ship_length × ship_beam) so
    the plan avoids ice for the full ship footprint.
  - Diffusion MPC (JAX): reverse diffusion over body-frame control sequences,
    rolled out through 3-DOF Fossen dynamics (same M3/D3 as simulation).
    Cost is hull-aware (center + corners), with costmap, goal, effort,
    smoothness, and optional path-following from lattice A*.
"""

from typing import Tuple, Dict, Optional, List, Union

import numpy as np

from dpdem.planning.lattice_planner import LatticePlanner, Primitives
from dpdem.planning.cost_map import CostMap
from dpdem.planning.path_optimizer import optimize_path as refine_path_nlp
from dpdem.planning.diffusion_mpc import DiffusionMPCPlanner
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
    goal_y_line: Optional[float] = None,
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
        goal_y_line: If set, goal is the horizontal line y = goal_y_line (any x).

    Returns:
        Configured LatticePlanner ready for planning (see Section 3).
    """
    sx, sy, _ = start
    gx, gy = goal

    if not _point_in_costmap_bounds(cost_map, sx, sy):
        raise ValueError(
            f"Start ({sx:.1f}, {sy:.1f}) is outside costmap bounds "
            f"[x={cost_map.x_origin:.1f}..{cost_map.x_origin + cost_map.width:.1f}, "
            f"y={cost_map.y_origin:.1f}..{cost_map.y_origin + cost_map.height:.1f}]"
        )
    if goal_y_line is None and not _point_in_costmap_bounds(cost_map, gx, gy):
        raise ValueError(
            f"Goal ({gx:.1f}, {gy:.1f}) is outside costmap bounds "
            f"[x={cost_map.x_origin:.1f}..{cost_map.x_origin + cost_map.width:.1f}, "
            f"y={cost_map.y_origin:.1f}..{cost_map.y_origin + cost_map.height:.1f}]"
        )

    primitives = Primitives(
        turning_radius=run_cfg.planning_turning_radius,
        num_headings=run_cfg.planning_num_headings,
        scale=run_cfg.planning_resolution,
    )

    planner = LatticePlanner(
        primitives=primitives,
        cost_map=cost_map,
        ship_length=run_cfg.ship_length,
        ship_width=run_cfg.ship_beam,
        heuristic_weight=run_cfg.planning_heuristic_weight,
        goal_tolerance=run_cfg.planning_goal_tolerance,
        goal_y_line=goal_y_line,
        greedy=getattr(run_cfg, "planning_greedy", True),
    )

    return planner


def create_medea_diffusion_mpc_planner(
    cost_map: CostMap,
    bounds: Dict[str, float],
    start: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg: RunConfig,
) -> DiffusionMPCPlanner:
    """
    Create a Diffusion MPC planner configured for the MEDEA costmap (AUTO-IceNav port).

    Uses the same costmap and motion primitives as the lattice planner; generates
    candidate paths by sampling primitive rollouts and returns the best by cost + goal distance.
    """
    sx, sy, _ = start
    gx, gy = goal
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
    primitives = Primitives(
        turning_radius=run_cfg.planning_turning_radius,
        num_headings=run_cfg.planning_num_headings,
        scale=run_cfg.planning_resolution,
    )
    planner = DiffusionMPCPlanner(
        primitives=primitives,
        cost_map=cost_map,
        ship_length=run_cfg.ship_length,
        ship_width=run_cfg.ship_beam,
        goal_tolerance=run_cfg.planning_goal_tolerance,
        horizon_steps=run_cfg.planning_diffusion_horizon_steps,
        num_samples=run_cfg.planning_diffusion_num_samples,
        goal_weight=run_cfg.planning_diffusion_goal_weight,
        seed=run_cfg.planning_diffusion_seed,
    )
    return planner


def plan_path(
    planner: Union[LatticePlanner, DiffusionMPCPlanner],
    start: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg: RunConfig,
    cost_map: Optional[CostMap] = None,
    return_search_history: bool = False,
):
    """
    Run the planner (lattice A* or Diffusion MPC) from start to goal (Section 3).

    Uses start and goal in the same world frame as the costmap/simulation.
    If planning_use_path_optimizer is True and cost_map is provided, refines the path
    with an NLP (path length + costmap integral + curvature penalty).
    return_search_history is only used by LatticePlanner (Diffusion MPC returns empty history).

    **Behaviour when no path is found:** returns `None`. Callers should check
    and either abort with a clear message or fall back to a fixed trajectory.

    Args:
        planner: Configured LatticePlanner or DiffusionMPCPlanner (from create_medea_*).
        start: (x, y, heading) in world coordinates [m, m, rad].
        goal: (x, y) in world coordinates [m, m].
        run_cfg: RunConfig for planning_max_iterations and optional path optimizer.
        cost_map: CostMap (required when planning_use_path_optimizer is True).
        return_search_history: If True, return (path, search_history) for visualization.

    Returns:
        Path as Nx3 array [x, y, heading], or None if no path found.
        If return_search_history is True, returns (path, search_history).
    """
    result = planner.plan(
        start=start,
        goal=goal,
        max_iterations=run_cfg.planning_max_iterations,
        return_search_history=return_search_history,
    )
    if return_search_history:
        path, search_history = result
    else:
        path = result
        search_history = None
    if path is None:
        return (None, search_history) if return_search_history else None
    if getattr(run_cfg, "planning_use_path_optimizer", False) and cost_map is not None:
        refined = refine_path_nlp(
            path,
            cost_map,
            start=start,
            goal=goal,
            ship_length=run_cfg.ship_length,
            ship_beam=run_cfg.ship_beam,
            n_waypoints=run_cfg.planning_optimizer_waypoints,
            cost_weight=run_cfg.planning_optimizer_cost_weight,
            curvature_weight=run_cfg.planning_optimizer_curvature_weight,
            max_iter=run_cfg.planning_optimizer_max_iter,
        )
        if refined is not None:
            path = refined
    return (path, search_history) if return_search_history else path


def _align_headings_to_path_direction(path: np.ndarray) -> np.ndarray:
    """Set each point's heading to the direction of travel (path tangent) so heading aligns with trajectory."""
    path = np.asarray(path, dtype=float)
    if path.shape[0] < 2:
        return path
    out = path.copy()
    dx = np.diff(path[:, 0])
    dy = np.diff(path[:, 1])
    psi = np.arctan2(dy, dx)
    out[:-1, 2] = psi
    out[-1, 2] = psi[-1]
    return out


def optimize_with_diffusion(
    continuous_path: np.ndarray,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    cost_map: CostMap,
    run_cfg: RunConfig,
    start_nu: Optional[Tuple[float, float, float]] = None,
    warm_start: Optional[bool] = None,
    viz_dir: Optional[str] = None,
    replan_idx: int = 0,
    viz_world_polygons=None,
    viz_thicknesses=None,
) -> np.ndarray:
    """
    Optimize a path using JAX Model-Based Diffusion (3-DOF dynamics + costmap).

    The MBD optimises body-frame control sequences in normalised action space,
    rolling them out through Fossen-style 3-DOF vessel dynamics to produce
    physically smooth trajectories with hull-aware costmap evaluation.

    The entire diffusion loop and parallel sampling are JIT-compiled via JAX
    and will run on GPU when available.

    Args:
        continuous_path: (N, 3) path [x, y, heading] in world coordinates.
        start_pose: (x, y, psi) current pose.
        goal: (x, y) goal position.
        cost_map: CostMap for collision/path cost.
        run_cfg: RunConfig for ship params and planning_jax_mbd_* settings.
        start_nu: Optional (u, v, r) initial velocities; default (2, 0, 0).
        warm_start: If True, seed MBD from the A* path; if False, cold start.
            Defaults to run_cfg.planning_jax_mbd_warm_start.

    Returns:
        Optimized path (K, 3) [x, y, heading] in world coordinates.
    """
    from medea_ice_ship.jax_mbd_planner import jax_mbd_optimize

    continuous_path = np.asarray(continuous_path, dtype=float)
    if continuous_path.shape[0] < 2:
        return continuous_path

    optimized_traj = jax_mbd_optimize(
        cost_map=cost_map,
        start_pose=start_pose,
        goal=goal,
        run_cfg=run_cfg,
        start_nu=start_nu,
        lattice_path=continuous_path,
        warm_start=warm_start,
        viz_dir=viz_dir,
        replan_idx=replan_idx,
        viz_world_polygons=viz_world_polygons,
        viz_thicknesses=viz_thicknesses,
    )

    optimized_traj = _align_headings_to_path_direction(optimized_traj)

    # Extend with remaining lattice path beyond MBD coverage
    end_pos = optimized_traj[-1, :2]
    dists = np.linalg.norm(continuous_path[:, :2] - end_pos, axis=1)
    nearest_idx = int(np.argmin(dists))
    if nearest_idx < len(continuous_path) - 1:
        optimized_traj = np.vstack([optimized_traj, continuous_path[nearest_idx + 1:]])

    return optimized_traj


def optimize_with_mppi(
    continuous_path: np.ndarray,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    cost_map: CostMap,
    run_cfg: RunConfig,
    start_nu: Optional[Tuple[float, float, float]] = None,
    warm_start: Optional[bool] = None,
) -> np.ndarray:
    """
    Optimize a path using JAX MPPI (3-DOF dynamics + costmap).

    Same interface as optimize_with_diffusion but uses MPPI instead of MBD:
    importance-weighted sampling with fixed-variance noise, multiple warm-start
    iterations. Reuses the same cost function (accumulated costmap + arc length + goal).
    """
    from medea_ice_ship.jax_mppi_planner import jax_mppi_optimize

    continuous_path = np.asarray(continuous_path, dtype=float)
    if continuous_path.shape[0] < 2:
        return continuous_path

    optimized_traj = jax_mppi_optimize(
        cost_map=cost_map,
        start_pose=start_pose,
        goal=goal,
        run_cfg=run_cfg,
        start_nu=start_nu,
        lattice_path=continuous_path,
        warm_start=warm_start,
    )

    optimized_traj = _align_headings_to_path_direction(optimized_traj)

    end_pos = optimized_traj[-1, :2]
    dists = np.linalg.norm(continuous_path[:, :2] - end_pos, axis=1)
    nearest_idx = int(np.argmin(dists))
    if nearest_idx < len(continuous_path) - 1:
        optimized_traj = np.vstack([optimized_traj, continuous_path[nearest_idx + 1:]])

    return optimized_traj


def optimize_with_sqp(
    continuous_path: np.ndarray,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    cost_map: CostMap,
    run_cfg: RunConfig,
    start_nu: Optional[Tuple[float, float, float]] = None,
    warm_start: Optional[bool] = None,
) -> np.ndarray:
    """
    Optimize a path using JAX multiple-shooting SQP (jaxproxqp).

    Same interface as optimize_with_diffusion/optimize_with_mppi but uses
    gradient-based SQP with QP subproblems. Reuses the same cost function.
    """
    from medea_ice_ship.jax_sqp_planner import jax_sqp_optimize

    continuous_path = np.asarray(continuous_path, dtype=float)
    if continuous_path.shape[0] < 2:
        return continuous_path

    optimized_traj = jax_sqp_optimize(
        cost_map=cost_map,
        start_pose=start_pose,
        goal=goal,
        run_cfg=run_cfg,
        start_nu=start_nu,
        lattice_path=continuous_path,
        warm_start=warm_start,
    )

    optimized_traj = _align_headings_to_path_direction(optimized_traj)

    end_pos = optimized_traj[-1, :2]
    dists = np.linalg.norm(continuous_path[:, :2] - end_pos, axis=1)
    nearest_idx = int(np.argmin(dists))
    if nearest_idx < len(continuous_path) - 1:
        optimized_traj = np.vstack([optimized_traj, continuous_path[nearest_idx + 1:]])

    return optimized_traj


def optimize_with_gradcem(
    continuous_path: np.ndarray,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    cost_map: CostMap,
    run_cfg: RunConfig,
    start_nu: Optional[Tuple[float, float, float]] = None,
    warm_start: Optional[bool] = None,
) -> np.ndarray:
    """
    Optimize a path using JAX GradCEM (gradient + Cross-Entropy Method).

    Same interface as optimize_with_diffusion/optimize_with_mppi but uses
    hybrid gradient + CEM elite selection. Reuses the same cost function.
    """
    from medea_ice_ship.jax_gradcem_planner import jax_gradcem_optimize

    continuous_path = np.asarray(continuous_path, dtype=float)
    if continuous_path.shape[0] < 2:
        return continuous_path

    optimized_traj = jax_gradcem_optimize(
        cost_map=cost_map,
        start_pose=start_pose,
        goal=goal,
        run_cfg=run_cfg,
        start_nu=start_nu,
        lattice_path=continuous_path,
        warm_start=warm_start,
    )

    optimized_traj = _align_headings_to_path_direction(optimized_traj)

    end_pos = optimized_traj[-1, :2]
    dists = np.linalg.norm(continuous_path[:, :2] - end_pos, axis=1)
    nearest_idx = int(np.argmin(dists))
    if nearest_idx < len(continuous_path) - 1:
        optimized_traj = np.vstack([optimized_traj, continuous_path[nearest_idx + 1:]])

    return optimized_traj


def optimize_speed_profile(
    path: np.ndarray,
    cost_map: CostMap,
    v_ref: float = 4.0,
    v_min: float = 1.0,
    v_max: float = 8.0,
    time_weight: float = 1.0,
    ship_length: float = 158.0,
    ship_beam: float = 28.0,
) -> Tuple[np.ndarray, float]:
    """
    Optimize speed along path segments to balance impact cost (KE ∝ v²) vs time.

    In thick ice (high costmap cost): optimal speed is lower (reduce impact).
    In open water (low cost): optimal speed is higher (save time).

    Cost model: cost through segment i = c_i * (v_i/v_ref)² * L_i (KE scales with v²).
    Time through segment i = L_i / v_i.
    Objective: minimize sum_i [ c_i*(v_i/v_ref)²*L_i + time_weight * L_i/v_i ].
    Per-segment optimum: v_i = (time_weight * v_ref² / (2*c_i))^(1/3), clamped to [v_min, v_max].

    Args:
        path: N×3 array [x, y, heading].
        cost_map: CostMap for sampling cost along path.
        v_ref: Reference speed [m/s] used when building costmap.
        v_min, v_max: Speed bounds [m/s].
        time_weight: Weight for time in objective (higher = prefer faster).
        ship_length, ship_beam: For sampling ship corners along path.

    Returns:
        speeds: (N-1) array of optimal speed per segment [m/s].
        total_time: Total traversal time [s].
    """
    path = np.asarray(path, dtype=float)
    n_pts = path.shape[0]
    if n_pts < 2:
        return np.array([v_ref]), 0.0

    seg_lens = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    n_seg = len(seg_lens)
    v_ref = max(v_ref, 1e-6)

    def _sample_cost(p: np.ndarray, heading: float) -> float:
        cost = cost_map.get_cost_world(p[0], p[1])
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        half_l, half_w = ship_length / 2, ship_beam / 2
        for dx, dy in [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]:
            cx = p[0] + cos_h * dx - sin_h * dy
            cy = p[1] + sin_h * dx + cos_h * dy
            cost = max(cost, cost_map.get_cost_world(cx, cy))
        return cost

    costs = np.zeros(n_seg)
    for i in range(n_seg):
        n_samp = max(3, int(seg_lens[i] / cost_map.resolution) + 1)
        for t in np.linspace(0, 1, n_samp):
            p = (1 - t) * path[i, :2] + t * path[i + 1, :2]
            h = (1 - t) * path[i, 2] + t * path[i + 1, 2]
            costs[i] += _sample_cost(p, h)
        costs[i] /= n_samp
        costs[i] = max(costs[i], cost_map.base_cost * 0.5)  # avoid div by zero

    # Optimal v_i = (λ * v_ref² / (2*c_i))^(1/3)
    lam = time_weight
    v_opt = np.power(lam * v_ref * v_ref / (2.0 * costs), 1.0 / 3.0)
    v_opt = np.clip(v_opt, v_min, v_max)

    total_time = np.sum(seg_lens / v_opt)
    return v_opt, float(total_time)


def path_to_ship_trajectory(
    path: np.ndarray,
    n_frames: int,
    speed_profile: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, float]]:
    """
    Convert a planned path (N×3) to ship_trajectory for simulate_ice_ship_interaction (Section 4).

    Produces a list of (position_2d, heading) of length n_frames + 1. First entry is path
    start; last entry is path goal. Intermediate points are interpolated along the path.
    By default uses uniform arc-length (constant speed). If speed_profile is provided,
    uses variable speed: more time in high-cost (ice) segments, less in open water.
    Heading is interpolated between waypoints (linear in angle, with wrap).

    Args:
        path: N×3 array [x, y, heading] in world coordinates.
        n_frames: Number of simulation frames (trajectory will have n_frames + 1 entries).
        speed_profile: Optional (N-1) array of speed per segment [m/s]. When provided,
            trajectory is time-parametrized so ship spends more time in low-speed segments.

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

    # Time parametrization: uniform arc-length (default) or variable speed
    if speed_profile is not None and len(speed_profile) == len(seg_lens):
        v = np.maximum(speed_profile, 1e-6)
        seg_times = seg_lens / v
        T_total = np.sum(seg_times)
        T_cum = np.zeros(n_pts)
        T_cum[1:] = np.cumsum(seg_times)
    else:
        T_total = 1.0
        T_cum = s / S if S > 0 else np.linspace(0, 1, n_pts)

    trajectory = []
    for k in range(n_frames + 1):
        if k == 0:
            trajectory.append((path[0, :2].copy(), float(path[0, 2])))
            continue
        if k == n_frames:
            trajectory.append((path[-1, :2].copy(), float(path[-1, 2])))
            continue
        t = k / n_frames
        if speed_profile is not None and len(speed_profile) == len(seg_lens):
            target_t = t * T_total
            i = np.searchsorted(T_cum, target_t, side="right") - 1
            i = max(0, min(i, n_pts - 2))
            if seg_times[i] > 0:
                frac = (target_t - T_cum[i]) / seg_times[i]
            else:
                frac = 1.0
            frac = np.clip(frac, 0.0, 1.0)
            target_s = s[i] + frac * seg_lens[i]
        else:
            target_s = t * S
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

