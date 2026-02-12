"""
Path refinement via nonlinear optimization (AUTO-IceNav style).

Takes an A* path and refines it by minimizing path length + costmap integral + curvature
penalty. Uses CasADi + Ipopt when available; otherwise returns the path unchanged.
"""

from typing import Optional, Tuple
import numpy as np

from dpdem.planning.cost_map import CostMap


def _resample_path(path: np.ndarray, n_waypoints: int) -> np.ndarray:
    """Resample path to n_waypoints (N+1) points with roughly uniform arc length."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] < 2:
        return path
    n_pts = path.shape[0]
    if n_pts < 2 or n_waypoints < 2:
        return path
    # Arc lengths
    segs = np.diff(path[:, :2], axis=0)
    seg_lens = np.linalg.norm(segs, axis=1)
    s = np.zeros(n_pts)
    s[1:] = np.cumsum(seg_lens)
    S = s[-1]
    if S <= 0:
        return path
    # Headings from segments
    head = np.zeros(n_pts)
    head[:-1] = np.arctan2(segs[:, 1], segs[:, 0])
    head[-1] = head[-2]
    # Sample at uniform s
    s_samp = np.linspace(0, S, n_waypoints, endpoint=True)
    out = np.zeros((n_waypoints, 3))
    for k, sk in enumerate(s_samp):
        i = np.searchsorted(s, sk, side="right") - 1
        i = max(0, min(i, n_pts - 2))
        if seg_lens[i] > 0:
            frac = (sk - s[i]) / seg_lens[i]
        else:
            frac = 1.0
        frac = np.clip(frac, 0.0, 1.0)
        out[k, 0] = (1 - frac) * path[i, 0] + frac * path[i + 1, 0]
        out[k, 1] = (1 - frac) * path[i, 1] + frac * path[i + 1, 1]
        out[k, 2] = (1 - frac) * head[i] + frac * head[i + 1]
    out[:, 2] = np.arctan2(np.sin(out[:, 2]), np.cos(out[:, 2]))
    return out


def optimize_path(
    path: np.ndarray,
    cost_map: CostMap,
    start: Tuple[float, float, float],
    goal: Tuple[float, float],
    ship_length: float,
    ship_beam: float,
    n_waypoints: int = 41,
    cost_weight: float = 0.1,
    curvature_weight: float = 10.0,
    max_iter: int = 500,
) -> Optional[np.ndarray]:
    """
    Refine path by NLP: minimize path length + costmap cost integral + curvature penalty.

    Waypoints (x, y, psi) with first and last fixed at start/goal. Cost is evaluated
    via a linear interpolant of the costmap in world coordinates. If CasADi is not
    available, returns the path unchanged.

    Args:
        path: Nx3 A* path [x, y, heading] in world coordinates.
        cost_map: CostMap in world coordinates.
        start: (x, y, psi) start pose.
        goal: (x, y) goal position (heading at goal is free).
        ship_length: Ship length [m] (for bounds margin).
        ship_beam: Ship beam [m].
        n_waypoints: Number of waypoints (including start and goal).
        cost_weight: Weight for costmap integral term.
        curvature_weight: Weight for (delta_psi)^2 penalty.
        max_iter: Maximum Ipopt iterations.

    Returns:
        Refined path (n_waypoints x 3) or None on failure; if CasADi missing, returns
        resampled path (no optimization).
    """
    try:
        import casadi as ca
    except ImportError:
        return _resample_path(path, n_waypoints)

    path = np.asarray(path, dtype=float)
    if path.shape[0] < 2:
        return path

    # Resample to n_waypoints for warm start
    warm = _resample_path(path, n_waypoints)
    warm[0, :] = [start[0], start[1], start[2]]
    warm[-1, :2] = [goal[0], goal[1]]
    # Goal heading: align with last segment
    if n_waypoints >= 2:
        dx = warm[-1, 0] - warm[-2, 0]
        dy = warm[-1, 1] - warm[-2, 1]
        warm[-1, 2] = np.arctan2(dy, dx)

    # Cost interpolant in world coordinates (grid = cell centers)
    rows, cols = cost_map.rows, cost_map.cols
    res = cost_map.resolution
    xo, yo = cost_map.x_origin, cost_map.y_origin
    x_grid = np.array([xo + (j + 0.5) * res for j in range(cols)], dtype=float)
    y_grid = np.array([yo + (i + 0.5) * res for i in range(rows)], dtype=float)
    # Value at (x_grid[j], y_grid[i]) = cost_map[i, j] -> 2D array shape (cols, rows)
    V = cost_map.cost_map.T  # (cols, rows)
    # Linear interpolant: ca.interpolant(name, 'linear', [x_grid, y_grid], V.ravel(order='F'))
    lut = ca.interpolant("cost_lut", "linear", [x_grid, y_grid], V.ravel(order="F"))

    # Bounds (keep inside costmap with small margin)
    margin = max(ship_length, ship_beam) * 0.5
    x_lo, x_hi = xo + margin, xo + cost_map.width - margin
    y_lo, y_hi = yo + margin, yo + cost_map.height - margin
    if x_lo >= x_hi or y_lo >= y_hi:
        return warm

    N = n_waypoints
    opti = ca.Opti()
    # Decision variables: (x_i, y_i, psi_i) for i = 1..N-2 (first and last fixed)
    x = opti.variable(N - 2)
    y = opti.variable(N - 2)
    psi = opti.variable(N - 2)

    # Build full waypoints (symbolic)
    x_all = ca.vertcat(ca.MX(start[0]), x, ca.MX(goal[0]))
    y_all = ca.vertcat(ca.MX(start[1]), y, ca.MX(goal[1]))
    psi_all = ca.vertcat(ca.MX(start[2]), psi, ca.MX(warm[-1, 2]))

    # Bounds on variables
    opti.subject_to(opti.bounded(x_lo, x, x_hi))
    opti.subject_to(opti.bounded(y_lo, y, y_hi))
    opti.subject_to(opti.bounded(-np.pi, psi, np.pi))

    # Objective: path length + cost integral + curvature penalty
    path_len = 0.0
    cost_int = 0.0
    curv_pen = 0.0
    for i in range(N - 1):
        dx_ = x_all[i + 1] - x_all[i]
        dy_ = y_all[i + 1] - y_all[i]
        seg_len = ca.sqrt(dx_**2 + dy_**2 + 1e-12)
        path_len += seg_len
        c_i = lut(ca.vertcat(x_all[i], y_all[i]))
        c_i1 = lut(ca.vertcat(x_all[i + 1], y_all[i + 1]))
        cost_int += (c_i + c_i1) * 0.5 * seg_len
        dpsi = psi_all[i + 1] - psi_all[i]
        curv_pen += dpsi**2
    J = path_len + cost_weight * cost_int + curvature_weight * curv_pen
    opti.minimize(J)

    # Warm start
    opti.set_initial(x, warm[1:-1, 0])
    opti.set_initial(y, warm[1:-1, 1])
    opti.set_initial(psi, warm[1:-1, 2])

    # Solver
    opti.solver("ipopt", {"expand": True, "max_iter": max_iter, "print_level": 0})
    try:
        sol = opti.solve()
        x_sol = np.array(sol.value(x)).ravel()
        y_sol = np.array(sol.value(y)).ravel()
        psi_sol = np.array(sol.value(psi)).ravel()
        out = np.zeros((N, 3))
        out[0] = [start[0], start[1], start[2]]
        out[1:-1, 0] = x_sol
        out[1:-1, 1] = y_sol
        out[1:-1, 2] = psi_sol
        out[-1] = [goal[0], goal[1], float(sol.value(psi_all[-1]))]
        return out
    except Exception:
        return warm
