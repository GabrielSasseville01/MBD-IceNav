"""Planning helpers for MEDEA ice–ship simulation.

Section 2–4 of PLANNER_INTEGRATION_CHECKLIST: create a lattice planner,
run A* to get a path, and convert the path to ship_trajectory for the simulator.

Planner dynamics and footprint:
  - Lattice: motion primitives respect minimum turning_radius (Dubins-like arcs);
    path cost uses ship center + four hull corners (ship_length × ship_beam) so
    the plan avoids ice for the full ship footprint.
  - Diffusion MPC: rollouts use 3-DOF dynamics (same M3/D3 as simulation) and
    control limits; cost is evaluated at ship center and hull corners when
    ship_length/ship_beam are passed, so optimized paths account for hull size.
"""

from typing import Tuple, Dict, Optional, List, Union

import numpy as np

from dpdem.planning.lattice_planner import LatticePlanner, Primitives
from dpdem.planning.cost_map import CostMap
from dpdem.planning.path_optimizer import optimize_path as refine_path_nlp
from dpdem.planning.diffusion_mpc import DiffusionMPCPlanner
from medea_ice_ship.config import RunConfig, get_default_simulation_config
from medea_ice_ship.vessel_dynamics import get_mss_style_3dof
from medea_ice_ship.diffusion_controller import ModelBasedDiffusionController
from medea_ice_ship.properties import R_xy


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
) -> np.ndarray:
    """
    Optimize a path using model-based diffusion (AUTO-IceNav style).

    Uses the continuous path (e.g. from lattice A*) as reference. Builds waypoint
    anchors from a downsampled path, warm-starts controls from path tracking,
    runs ModelBasedDiffusionController to get an optimized trajectory.

    Args:
        continuous_path: (N, 3) path [x, y, heading] in world coordinates.
        start_pose: (x, y, psi) current pose.
        goal: (x, y) goal position.
        cost_map: CostMap for collision/path cost.
        run_cfg: RunConfig for dt, ship params, and diffusion_* settings.
        start_nu: Optional (u, v, r) initial velocities; default (2, 0, 0).

    Returns:
        Optimized path (K, 3) [x, y, heading] in world coordinates.
    """
    continuous_path = np.asarray(continuous_path, dtype=float)
    if continuous_path.shape[0] < 2:
        return continuous_path
    horizon = min(
        getattr(run_cfg, "planning_diffusion_controller_horizon", 80),
        continuous_path.shape[0],
    )
    local_path = continuous_path[:horizon].copy()
    if continuous_path.shape[0] < horizon:
        last = local_path[-1:]
        local_path = np.vstack([local_path, np.tile(last, (horizon - len(local_path), 1))])
    goal_y = goal[1]
    # Discrete waypoint anchors from path
    n_discrete = min(10, horizon)
    discrete_indices = np.linspace(0, horizon - 1, n_discrete).astype(int)
    waypoint_anchors = {}
    for idx in discrete_indices:
        if idx < len(continuous_path):
            waypoint_anchors[int(idx)] = continuous_path[idx].copy()
    # Ship dynamics: use same Izz and M3/D3 as simulation so diffusion rollouts match sim behavior
    mass = run_cfg.ship_displacement
    L = run_cfg.ship_length
    B = run_cfg.ship_beam
    Izz = (1.0 / 12.0) * mass * (L ** 2 + B ** 2)
    sim_cfg = get_default_simulation_config()
    if getattr(sim_cfg, "use_mss_style_dynamics", True):
        M3, D3 = get_mss_style_3dof(
            mass, Izz,
            ship_damping_xy=getattr(sim_cfg, "ship_damping_xy", 0.02),
            ship_damping_psi=getattr(sim_cfg, "ship_damping_psi", 0.15),
            added_mass_surge_frac=getattr(sim_cfg, "mss_added_mass_surge_frac", 0.05),
            added_mass_sway_frac=getattr(sim_cfg, "mss_added_mass_sway_frac", 0.5),
            added_mass_yaw_frac=getattr(sim_cfg, "mss_added_mass_yaw_frac", 0.2),
        )
    else:
        M3 = np.diag([mass, mass, Izz])
        D3 = np.diag([
            mass * getattr(sim_cfg, "ship_damping_xy", 0.02),
            mass * getattr(sim_cfg, "ship_damping_xy", 0.02),
            Izz * getattr(sim_cfg, "ship_damping_psi", 0.15),
        ])
    dt = run_cfg.dt
    target_speed = 2.0
    # Controller
    controller = ModelBasedDiffusionController(
        horizon=horizon,
        M3=M3,
        D3=D3,
        diffusion_noise_clipping=1e6,
        num_samples=getattr(run_cfg, "planning_diffusion_num_samples", 64),
        num_diffusion_steps=getattr(run_cfg, "planning_diffusion_num_diffusion_steps", 30),
        temperature=getattr(run_cfg, "planning_diffusion_temperature", 0.1),
        beta0=getattr(run_cfg, "planning_diffusion_beta0", 1e-4),
        betaT=getattr(run_cfg, "planning_diffusion_betaT", 1e-2),
        seed=run_cfg.planning_diffusion_seed or 0,
        ship_length=L,
        ship_beam=B,
    )
    controller.waypoint_anchors = waypoint_anchors
    controller.enforce_forward_motion = True
    controller.target_surge_speed = target_speed
    # Warm start: bias toward turning to align heading then thrusting forward (minimize sway / crabbing)
    eta = np.array(start_pose, dtype=float)
    nu = np.array(start_nu if start_nu is not None else (target_speed, 0.0, 0.0), dtype=float)
    warm_start = np.zeros((horizon, 3))
    for t in range(horizon):
        target = local_path[t] if t < len(local_path) else local_path[-1]
        pos_err = target[:2] - eta[:2]
        ang_err = (target[2] - eta[2] + np.pi) % (2 * np.pi) - np.pi
        R = R_xy(eta[2])
        body_err = R.T @ pos_err
        warm_start[t, 0] = max(1e5, 9e5)
        # Prefer yaw to align heading; use only modest sway so diffusion does not start from a crab
        warm_start[t, 1] = np.clip(body_err[1] * 1.5e5, -2e5, 2e5)
        warm_start[t, 2] = np.clip(ang_err * 4e6, -5e6, 5e6)
        vel_global = R @ np.array([nu[0], nu[1]])
        eta[0] += dt * vel_global[0] * 0.6
        eta[1] += dt * vel_global[1] * 0.6
        eta[2] = (eta[2] + dt * nu[2] * 0.6 + np.pi) % (2 * np.pi) - np.pi
    # Run diffusion
    pose = np.array(start_pose, dtype=float)
    nu_start = np.array(start_nu if start_nu is not None else (target_speed, 0.0, 0.0), dtype=float)
    optimized_traj = controller.run_diffusion(
        pose=pose,
        nu=nu_start,
        local_path=local_path,
        dt=dt,
        costmap=cost_map,
        goal_y=goal_y,
        warm_start_controls=warm_start,
    )
    # Align headings with path direction (tangent) so commanded heading matches trajectory like lattice.
    # Diffusion rollout gives psi = actual yaw from dynamics (can crab); we want heading = direction of travel.
    optimized_traj = _align_headings_to_path_direction(optimized_traj)
    # Extend with remaining continuous path if needed
    if len(continuous_path) > horizon:
        optimized_traj = np.vstack([optimized_traj, continuous_path[horizon:]])
    return optimized_traj


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

