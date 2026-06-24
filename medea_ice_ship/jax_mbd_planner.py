"""
JAX Model-Based Diffusion (MBD) planner for 3-DOF vessel trajectory optimisation.

Optimises body-frame control sequences [X, Y, N] (surge force, sway force, yaw
torque) via MBD reverse diffusion.  Controls are rolled out through Fossen-style
3-DOF dynamics to produce physically smooth trajectories, evaluated against
a costmap for ice avoidance + goal reaching + optional path following.

Used by ``medea_ice_ship.planning.optimize_with_diffusion`` for the
``--planner diffusion-mpc`` pipeline in ``run_medea_ship_fracture.py``.

Algorithm (MBD paper):
  Forward:  p(Y_i | Y_0) = N(sqrt(abar_i) Y_0,  (1−abar_i) I)
  Score:    ∇ log p_i ≈ −Y_i/(1−abar_i) + sqrt(abar_i)/(1−abar_i) Ȳ_0
  Reverse:  Y_{i−1} = 1/√α_i  (Y_i + (1−abar_i) score)
  Weights:  w_k = softmax(−J(Y0_k) / λ)
  Ȳ_0     = Σ w_k Y0_k
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Any, List, Union

from medea_ice_ship.config import get_default_simulation_config


# ── JAX 3-DOF vessel dynamics (Fossen: M ν̇ + C(ν)ν + D ν = τ) ───────────

def make_dynamics_rollout(
    mass: float,
    Izz: float,
    dt_substep: float,
    n_substeps: int,
    ship_damping_xy: float = 0.02,
    ship_damping_psi: float = 0.15,
    added_mass_surge_frac: float = 0.05,
    added_mass_sway_frac: float = 0.5,
    added_mass_yaw_frac: float = 0.2,
):
    """
    Build a pure-JAX dynamics rollout (JIT-traceable, vmap-compatible).

    Each action step holds control constant for *n_substeps* Euler sub-steps
    of size *dt_substep* → effective action period = dt_substep × n_substeps.

    Returns
    -------
    rollout_fn : callable(eta0, nu0, u_seq) → trajectory (H+1, 3)
    M3_np, D3_np : np.ndarray  — mass / damping (numpy, diagnostics only)
    """
    m11 = float(mass * (1.0 + added_mass_surge_frac))
    m22 = float(mass * (1.0 + added_mass_sway_frac))
    m33 = float(Izz * (1.0 + added_mass_yaw_frac))

    M3_inv_diag = jnp.array([1.0 / m11, 1.0 / m22, 1.0 / m33], dtype=jnp.float32)
    D3_diag = jnp.array([
        mass * ship_damping_xy,
        mass * ship_damping_xy,
        Izz * ship_damping_psi,
    ], dtype=jnp.float32)

    _m11 = jnp.float32(m11)
    _m22 = jnp.float32(m22)
    _dt = jnp.float32(dt_substep)
    # Explicit float32 constants so that Python floats (which become float64
    # when jax_enable_x64=True) do not promote the heading arithmetic to float64,
    # which would change the scan carry dtype and corrupt the rollout.
    _pi  = jnp.float32(3.141592653589793)
    _2pi = jnp.float32(6.283185307179586)
    # Velocity clipping to prevent float32 overflow from the Coriolis instability.
    # Under large yaw torques the coupled surge/sway/yaw Euler ODE has an
    # unstable mode (eigenvalue > 0) that causes velocities to grow
    # exponentially, producing NaN via inf−inf.  The clips must also be
    # physically realistic: with ±30 m/s clips the rollout quickly hits 30 m/s,
    # producing a ~23 km arc-length path; the goal sits at only 1.6% of that
    # arc, so the simulation traverses it far faster than the intended 4 m/s.
    # Physically:
    #   terminal surge  = surge_max / (mass × damping_xy) ≈ 4.35 m/s
    #   terminal sway   = sway_max  / (mass × damping_xy) ≈ 1.09 m/s
    #   terminal yaw    = yaw_max   / (Izz  × damping_psi) ≈ 0.07 rad/s
    # Clips are set to 1.5–7× terminal so the planner can still explore
    # fast and sharp-turn trajectories while keeping arc-lengths realistic.
    _nu_lo = jnp.array([-3.0, -2.0, -0.5], dtype=jnp.float32)
    _nu_hi = jnp.array([ 6.0,  2.0,  0.5], dtype=jnp.float32)

    def _euler_step(state, tau):
        """Single Euler sub-step.  state = [eta(3); nu(3)]."""
        tau = tau.astype(state.dtype)  # guard against float64 leaking from diffusion loop
        eta, nu = state[:3], state[3:6]
        C_nu = jnp.stack([
            -_m22 * nu[1] * nu[2],
             _m11 * nu[0] * nu[2],
            (_m22 - _m11) * nu[0] * nu[1],
        ])
        nu_dot = M3_inv_diag * (tau - C_nu - D3_diag * nu)
        nu_new = jnp.clip(nu + _dt * nu_dot, _nu_lo, _nu_hi)
        c = jnp.cos(eta[2])
        s = jnp.sin(eta[2])
        vx = c * nu_new[0] - s * nu_new[1]
        vy = s * nu_new[0] + c * nu_new[1]
        psi_new = (eta[2] + _dt * nu_new[2] + _pi) % _2pi - _pi
        eta_new = jnp.stack([eta[0] + _dt * vx, eta[1] + _dt * vy, psi_new])
        return jnp.concatenate([eta_new, nu_new]), None

    def _action_step(state, tau):
        """Hold one action for n_substeps Euler sub-steps."""
        state_new, _ = jax.lax.scan(
            lambda s, _: _euler_step(s, tau), state, None, length=n_substeps,
        )
        return state_new, state_new[:3]

    def _action_step_full(state, tau):
        """Same as _action_step but emits full 6-D state (η, ν) per action step."""
        state_new, _ = jax.lax.scan(
            lambda s, _: _euler_step(s, tau), state, None, length=n_substeps,
        )
        return state_new, state_new

    def rollout(eta0, nu0, u_seq):
        """Forward-roll dynamics.  u_seq: (H, 3) → traj: (H+1, 3)."""
        state0 = jnp.concatenate([eta0, nu0])
        _, traj = jax.lax.scan(_action_step, state0, u_seq)
        return jnp.concatenate([eta0[None, :], traj], axis=0)

    def rollout_with_nu(eta0, nu0, u_seq):
        """Forward-roll dynamics emitting both pose and body-frame velocity.

        u_seq: (H, 3) → (eta_traj, nu_traj) each of shape (H+1, 3).
        Used by the adaptive-velocity cost path, which scales the KE costmap
        by v²(t) = u_t² + v_t² from the rollout's emergent ν.
        """
        state0 = jnp.concatenate([eta0, nu0])
        _, states = jax.lax.scan(_action_step_full, state0, u_seq)
        eta_traj = jnp.concatenate([eta0[None, :], states[:, :3]], axis=0)
        nu_traj  = jnp.concatenate([nu0[None, :],  states[:, 3:6]], axis=0)
        return eta_traj, nu_traj

    rollout.with_nu = rollout_with_nu

    M3_np = np.diag([m11, m22, m33])
    D3_np = np.diag([float(d) for d in D3_diag])
    return rollout, M3_np, D3_np


# ── Costmap bilinear interpolation (JAX) ──────────────────────────────────

def _make_query(grid: jnp.ndarray, meta: dict, oob_penalty: float = 1e8):
    """Return a bilinear-interpolation query for a JAX costmap grid.

    Parameters
    ----------
    oob_penalty : float
        Cost added when the queried x-coordinate is outside the costmap
        x-extent.  Set to 0.0 when using EB-MBD (the barrier replaces
        the hard OOB penalty).  Default 1e8 preserves existing behaviour.
    """
    xo = float(meta["x_origin"])
    yo = float(meta["y_origin"])
    res = float(meta["resolution"])
    nrows = int(meta["rows"])
    ncols = int(meta["cols"])
    _oob_penalty = jnp.float32(oob_penalty)

    def query(x, y):
        cf = (x - xo) / res - 0.5
        rf = (y - yo) / res - 0.5
        r0 = jnp.floor(rf).astype(jnp.int32)
        c0 = jnp.floor(cf).astype(jnp.int32)
        fr = rf - r0
        fc = cf - c0
        r0c = jnp.clip(r0, 0, nrows - 1)
        r1c = jnp.clip(r0 + 1, 0, nrows - 1)
        c0c = jnp.clip(c0, 0, ncols - 1)
        c1c = jnp.clip(c0 + 1, 0, ncols - 1)
        oob = jnp.where(
            (c0 < 0) | (c0 + 1 >= ncols),
            _oob_penalty, 0.0,
        )
        val = (
            grid[r0c, c0c] * (1 - fr) * (1 - fc)
            + grid[r0c, c1c] * (1 - fr) * fc
            + grid[r1c, c0c] * fr * (1 - fc)
            + grid[r1c, c1c] * fr * fc
        )
        return val + oob

    return query


def _make_query_split(grid: jnp.ndarray, meta: dict, oob_penalty: float = 1e8):
    """Bilinear query that returns (ke_value, oob_value) separately.

    Used by the adaptive-velocity cost path: the KE component must be
    multiplied by v²(t) from the rollout before the base_cost floor is
    applied, while the OOB penalty must remain v-independent.
    """
    xo = float(meta["x_origin"])
    yo = float(meta["y_origin"])
    res = float(meta["resolution"])
    nrows = int(meta["rows"])
    ncols = int(meta["cols"])
    _oob_penalty = jnp.float32(oob_penalty)

    def query(x, y):
        cf = (x - xo) / res - 0.5
        rf = (y - yo) / res - 0.5
        r0 = jnp.floor(rf).astype(jnp.int32)
        c0 = jnp.floor(cf).astype(jnp.int32)
        fr = rf - r0
        fc = cf - c0
        r0c = jnp.clip(r0, 0, nrows - 1)
        r1c = jnp.clip(r0 + 1, 0, nrows - 1)
        c0c = jnp.clip(c0, 0, ncols - 1)
        c1c = jnp.clip(c0 + 1, 0, ncols - 1)
        oob = jnp.where(
            (c0 < 0) | (c0 + 1 >= ncols),
            _oob_penalty, 0.0,
        )
        ke = (
            grid[r0c, c0c] * (1 - fr) * (1 - fc)
            + grid[r0c, c1c] * (1 - fr) * fc
            + grid[r1c, c0c] * fr * (1 - fc)
            + grid[r1c, c1c] * fr * fc
        )
        return ke, oob

    return query


# ── Path resampling (numpy, called before JIT) ───────────────────────────

def _resample_path_by_arclength(path: np.ndarray, n_points: int) -> np.ndarray:
    """Resample (N, 3) path [x, y, heading] to (n_points, 3) by uniform arc length."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2 or n_points < 1:
        return np.tile(path[0] if len(path) else np.zeros(3), (n_points, 1))

    seg_lens = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    s = np.zeros(len(path))
    s[1:] = np.cumsum(seg_lens)
    S = s[-1]
    if S <= 0:
        return np.tile(path[0], (n_points, 1))

    s_target = np.linspace(0, S, n_points)
    result = np.zeros((n_points, 3))
    for k, st in enumerate(s_target):
        i = np.searchsorted(s, st, side="right") - 1
        i = max(0, min(i, len(path) - 2))
        seg = max(seg_lens[i], 1e-8)
        frac = np.clip((st - s[i]) / seg, 0.0, 1.0)
        result[k, :2] = (1 - frac) * path[i, :2] + frac * path[i + 1, :2]
        da = (path[i + 1, 2] - path[i, 2] + np.pi) % (2 * np.pi) - np.pi
        result[k, 2] = path[i, 2] + frac * da
    return result


# ── RRT path planner (numpy, costmap-based, for MBD warm-starting) ───────

def rrt_path_on_costmap(
    cost_map,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    *,
    max_iters: int = 8000,
    step_size: float = 20.0,
    goal_bias: float = 0.15,
    cost_threshold_frac: float = 0.1,
    smooth_iters: int = 300,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """
    RRT path planner on a 2-D costmap grid.

    Cells with cost > max_cost * cost_threshold_frac are treated as obstacles.
    Uses goal-biased sampling and random-shortcut smoothing.

    Returns (N, 3) float64 array [x, y, heading], or None if no path is found.
    The returned path can be passed directly to _warm_start_from_path.
    """
    grid = np.array(cost_map.cost_map)
    # Treat wall-cost cells (>=1e7) as obstacles regardless of the threshold fraction.
    # Clip them for max-cost estimation so ice cells remain detectable.
    WALL_CLIP = 1e7
    _det_grid = np.where(grid >= WALL_CLIP, 0.0, grid)
    max_cost = float(_det_grid.max())
    res    = float(cost_map.resolution)
    x_orig = float(cost_map.x_origin)
    y_orig = float(cost_map.y_origin)
    nrows, ncols = grid.shape

    if max_cost > 0.0:
        occ = (grid >= WALL_CLIP) | (_det_grid > max_cost * cost_threshold_frac)
    else:
        occ = (grid >= WALL_CLIP)

    x_lo, x_hi = x_orig, x_orig + ncols * res
    y_lo, y_hi = y_orig, y_orig + nrows * res

    def _free(x, y):
        ri = int((y - y_orig) / res)
        ci = int((x - x_orig) / res)
        if ri < 0 or ri >= nrows or ci < 0 or ci >= ncols:
            return False
        return not occ[ri, ci]

    def _segment_free(p1, p2, n=12):
        ts = np.linspace(0.0, 1.0, n)
        xs = p1[0] + ts * (p2[0] - p1[0])
        ys = p1[1] + ts * (p2[1] - p1[1])
        for x, y in zip(xs, ys):
            if not _free(x, y):
                return False
        return True

    rng     = np.random.default_rng(seed)
    nodes   = [np.array(start_xy, dtype=np.float64)]
    parent  = [-1]
    goal_np = np.array(goal_xy,   dtype=np.float64)
    found   = False

    for _ in range(max_iters):
        if rng.random() < goal_bias:
            q_rand = goal_np.copy()
        else:
            q_rand = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)])

        arr    = np.array(nodes)
        idx_nn = int(np.argmin(np.linalg.norm(arr - q_rand, axis=1)))
        q_near = nodes[idx_nn]

        d  = q_rand - q_near
        dn = float(np.linalg.norm(d))
        if dn < 1e-6:
            continue
        q_new = q_near + (d / dn) * min(step_size, dn)

        if not _segment_free(q_near, q_new):
            continue

        nodes.append(q_new)
        parent.append(idx_nn)

        if float(np.linalg.norm(q_new - goal_np)) <= step_size:
            if _segment_free(q_new, goal_np):
                nodes.append(goal_np.copy())
                parent.append(len(nodes) - 2)
                found = True
                break

    if not found:
        return None

    # Trace path from goal back to start
    path: list = []
    idx = len(nodes) - 1
    while idx >= 0:
        path.append(nodes[idx])
        idx = parent[idx]
    path.reverse()
    path_xy = np.array(path)

    # Random shortcutting
    for _ in range(smooth_iters):
        n = len(path_xy)
        if n <= 2:
            break
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(i + 1, n))
        if _segment_free(path_xy[i], path_xy[j]):
            path_xy = np.vstack([path_xy[: i + 1], path_xy[j:]])

    # Attach headings
    n       = len(path_xy)
    result  = np.zeros((n, 3), dtype=np.float64)
    result[:, :2] = path_xy
    for i in range(n - 1):
        dx, dy = path_xy[i + 1] - path_xy[i]
        result[i, 2] = np.arctan2(dy, dx)
    result[-1, 2] = result[-2, 2]
    return result


# ── A* path planner (numpy, costmap-based, for MBD warm-starting) ────────

def astar_path_on_costmap(
    cost_map,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    *,
    cost_threshold_frac: float = 0.1,
    smooth_iters: int = 300,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """
    A* path planner on a 2-D costmap grid.

    Uses the same occupancy mask as rrt_path_on_costmap (cells with cost >
    max_cost * cost_threshold_frac are obstacles, plus wall-cost cells).
    8-connected grid search with Euclidean heuristic.  Followed by the same
    random-shortcut smoothing as the RRT planner.

    Returns (N, 3) float64 array [x, y, heading], or None if no path found.
    """
    import heapq

    grid   = np.array(cost_map.cost_map)
    WALL_CLIP = 1e7
    _det   = np.where(grid >= WALL_CLIP, 0.0, grid)
    max_cost = float(_det.max())
    res    = float(cost_map.resolution)
    x_orig = float(cost_map.x_origin)
    y_orig = float(cost_map.y_origin)
    nrows, ncols = grid.shape

    if max_cost > 0.0:
        occ = (grid >= WALL_CLIP) | (_det > max_cost * cost_threshold_frac)
    else:
        occ = (grid >= WALL_CLIP)

    def _world_to_cell(x, y):
        ci = int(np.clip((x - x_orig) / res, 0, ncols - 1))
        ri = int(np.clip((y - y_orig) / res, 0, nrows - 1))
        return ri, ci

    def _cell_to_world(ri, ci):
        return x_orig + (ci + 0.5) * res, y_orig + (ri + 0.5) * res

    r_start, c_start = _world_to_cell(*start_xy)
    r_goal,  c_goal  = _world_to_cell(*goal_xy)

    # 8-connected neighbours
    moves = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    move_costs = [np.sqrt(2), 1.0, np.sqrt(2), 1.0, 1.0, np.sqrt(2), 1.0, np.sqrt(2)]

    def _h(r, c):
        return np.sqrt((r - r_goal)**2 + (c - c_goal)**2)

    g = np.full((nrows, ncols), np.inf)
    g[r_start, c_start] = 0.0
    came_from = {}
    open_heap = [(float(_h(r_start, c_start)), r_start, c_start)]

    found = False
    while open_heap:
        f, r, c = heapq.heappop(open_heap)
        if r == r_goal and c == c_goal:
            found = True
            break
        if f > g[r, c] + _h(r, c) + 1e-6:
            continue
        for (dr, dc), mc in zip(moves, move_costs):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                continue
            if occ[nr, nc]:
                continue
            ng = g[r, c] + mc
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (ng + _h(nr, nc), nr, nc))

    if not found:
        return None

    # Trace path
    path_cells = []
    node = (r_goal, c_goal)
    while node in came_from:
        path_cells.append(node)
        node = came_from[node]
    path_cells.append((r_start, c_start))
    path_cells.reverse()

    path_xy = np.array([_cell_to_world(r, c) for r, c in path_cells])

    # Same random-shortcut smoothing as RRT
    def _free(x, y):
        ri = int((y - y_orig) / res)
        ci = int((x - x_orig) / res)
        if ri < 0 or ri >= nrows or ci < 0 or ci >= ncols:
            return False
        return not occ[ri, ci]

    def _segment_free(p1, p2, n=12):
        ts = np.linspace(0.0, 1.0, n)
        for t in ts:
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if not _free(x, y):
                return False
        return True

    rng = np.random.default_rng(seed)
    for _ in range(smooth_iters):
        n = len(path_xy)
        if n <= 2:
            break
        i = int(rng.integers(0, n - 1))
        j = int(rng.integers(i + 1, n))
        if _segment_free(path_xy[i], path_xy[j]):
            path_xy = np.vstack([path_xy[:i + 1], path_xy[j:]])

    # Attach headings
    n = len(path_xy)
    result = np.zeros((n, 3), dtype=np.float64)
    result[:, :2] = path_xy
    for i in range(n - 1):
        dx, dy = path_xy[i + 1] - path_xy[i]
        result[i, 2] = np.arctan2(dy, dx)
    result[-1, 2] = result[-2, 2]
    return result


# ── Warm-start helpers (numpy, pre-JIT) ──────────────────────────────────

def _warm_start_from_path(
    lattice_path: Optional[np.ndarray],
    H: int,
    init_surge_bias: float = 0.7,
    eta0: Optional[np.ndarray] = None,
    nu0: Optional[np.ndarray] = None,
    M3_np: Optional[np.ndarray] = None,
    D3_np: Optional[np.ndarray] = None,
    dt_substep: float = 1.0,
    n_substeps: int = 1,
    action_center_np: Optional[np.ndarray] = None,
    action_scale_np: Optional[np.ndarray] = None,
    lookahead_dist: float = 60.0,
) -> np.ndarray:
    """
    Compute initial normalised actions (H, 3) from a lattice path.

    When ship dynamics parameters are provided (eta0, nu0, M3_np, D3_np,
    action_center_np, action_scale_np), uses a closed-loop pure-pursuit
    simulation: the ship state is propagated forward at each step and the yaw
    command is computed from the actual ship heading vs. a lookahead point on
    the path.  This produces a warm-start rollout that genuinely follows the
    RRT path through dynamics.

    Without dynamics parameters, falls back to the original open-loop geometric
    approach (path headings only).
    """
    Ybar = np.zeros((H, 3))
    Ybar[:, 0] = init_surge_bias

    if lattice_path is None or len(lattice_path) < 3:
        return Ybar

    path_ref = _resample_path_by_arclength(lattice_path, H + 1)

    # ── Closed-loop pure-pursuit (preferred when dynamics are available) ──
    if (eta0 is not None and M3_np is not None
            and action_center_np is not None and action_scale_np is not None):

        eta = np.array(eta0, dtype=np.float64)
        nu  = np.array(nu0 if nu0 is not None else [2.0, 0.0, 0.0], dtype=np.float64)
        M3_inv = np.linalg.inv(M3_np.astype(np.float64))
        D3     = D3_np.astype(np.float64) if D3_np is not None else np.zeros((3, 3))
        ac     = np.array(action_center_np, dtype=np.float64)
        asc    = np.array(action_scale_np,  dtype=np.float64)
        m11    = float(M3_np[0, 0])
        m22    = float(M3_np[1, 1])

        # Physical surge matching the normalised bias
        surge_phys = float(init_surge_bias * asc[0] + ac[0])

        # Cumulative arc-length along path_ref for lookahead queries
        segs   = np.linalg.norm(np.diff(path_ref[:, :2], axis=0), axis=1)
        cum_s  = np.concatenate([[0.0], np.cumsum(segs)])
        s_total = float(cum_s[-1])

        def _lookahead_xy(pos_xy: np.ndarray) -> np.ndarray:
            """World-coordinate target: path point `lookahead_dist` ahead."""
            dists   = np.linalg.norm(path_ref[:, :2] - pos_xy[None, :], axis=1)
            nearest = int(np.argmin(dists))
            target_s = min(float(cum_s[nearest]) + lookahead_dist, s_total)
            idx = int(np.searchsorted(cum_s, target_s))
            return path_ref[min(idx, len(path_ref) - 1), :2]

        for t in range(H):
            tgt = _lookahead_xy(eta[:2])
            dx  = float(tgt[0] - eta[0])
            dy  = float(tgt[1] - eta[1])
            desired = np.arctan2(dy, dx) if (abs(dx) + abs(dy)) > 0.5 else float(eta[2])
            err_psi = (desired - float(eta[2]) + np.pi) % (2 * np.pi) - np.pi

            # Proportional-derivative yaw command (physical torque)
            tau_r   = float(np.clip(err_psi * 2.0 * asc[2], -asc[2], asc[2]))
            tau     = np.array([surge_phys, 0.0, tau_r])
            u_norm  = np.clip((tau - ac) / (asc + 1e-8), -1.0, 1.0)
            Ybar[t] = u_norm

            # Euler substeps with actual ship state
            for _ in range(n_substeps):
                C_nu = np.array([
                    -m22 * nu[1] * nu[2],
                     m11 * nu[0] * nu[2],
                    (m22 - m11) * nu[0] * nu[1],
                ])
                nu_dot = M3_inv @ (tau - C_nu - D3 @ nu)
                nu     = nu + dt_substep * nu_dot
                c, s   = np.cos(eta[2]), np.sin(eta[2])
                vx     = c * nu[0] - s * nu[1]
                vy     = s * nu[0] + c * nu[1]
                eta    = np.array([
                    eta[0] + dt_substep * vx,
                    eta[1] + dt_substep * vy,
                    (eta[2] + dt_substep * nu[2] + np.pi) % (2 * np.pi) - np.pi,
                ])
        return Ybar

    # ── Fallback: open-loop geometric (no dynamics info) ──────────────────
    for t in range(H):
        dx = path_ref[t + 1, 0] - path_ref[t, 0]
        dy = path_ref[t + 1, 1] - path_ref[t, 1]
        desired = np.arctan2(dy, dx)
        current = path_ref[t, 2]
        err = (desired - current + np.pi) % (2 * np.pi) - np.pi
        Ybar[t, 2] = np.clip(err * 2.0, -0.8, 0.8)

    return Ybar


# ── Cost function (normalised actions → dynamics → costmap + goal) ────────

def _make_raw_components_fn(
    grid: jnp.ndarray,
    meta: dict,
    goal_world,
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    ship_length: float = 0.0,
    ship_beam: float = 0.0,
    cost_truncate_y: Optional[float] = None,
    oob_penalty: float = 1e8,
):
    """
    Shared implementation used by both make_cost_fn and make_cost_components_fn.

    Returns a function raw_fn(u_norm) -> jnp.array([costmap, length, goal, jerk])
    where each element is the *unweighted* cost component, using exactly the same
    formulas as cost_fn so that weighted = raw * weight_vector is always consistent
    with the total cost returned by cost_fn.

    cost_truncate_y : float, optional
        If set, waypoints with y > cost_truncate_y contribute zero to the
        costmap integral (including any OOB penalties from _make_query).
        Useful for open-ended scenarios where the trajectory is allowed to
        exit the map from the top without penalty.
    """
    # Adaptive-velocity mode: cells store per-v² KE coefficient (no v baked,
    # no base_cost floor). The cost integral multiplies KE by v²(t) from the
    # rollout's emergent ν, applies max(base_cost, ...) floor at query, and
    # keeps OOB penalty unscaled. Meta carries the flag and floor value.
    _adaptive_v = bool(meta.get("adaptive_velocity", False))
    _base_cost  = jnp.float32(meta.get("base_cost", 0.0))

    goal_xy = jnp.array(goal_world[:2], dtype=jnp.float32)
    _has_truncate = cost_truncate_y is not None
    _truncate_y = jnp.float32(cost_truncate_y if cost_truncate_y is not None else 0.0)

    half_l = float(ship_length) / 2.0
    half_w = float(ship_beam) / 2.0
    use_hull = ship_length > 0 and ship_beam > 0

    if _adaptive_v:
        query_split = _make_query_split(grid, meta, oob_penalty=oob_penalty)
        if use_hull:
            _dx_body = jnp.array([0.0, half_l, half_l, -half_l, -half_l], dtype=jnp.float32)
            _dy_body = jnp.array([0.0, half_w, -half_w, half_w, -half_w], dtype=jnp.float32)

            def _point_cost_split(point):
                c, s = jnp.cos(point[2]), jnp.sin(point[2])
                wx = point[0] + c * _dx_body - s * _dy_body
                wy = point[1] + s * _dx_body + c * _dy_body
                kes, oobs = jax.vmap(query_split)(wx, wy)
                return jnp.max(kes), jnp.max(oobs)
        else:
            def _point_cost_split(point):
                return query_split(point[0], point[1])
    else:
        query = _make_query(grid, meta, oob_penalty=oob_penalty)
        if use_hull:
            _dx_body = jnp.array([0.0, half_l, half_l, -half_l, -half_l], dtype=jnp.float32)
            _dy_body = jnp.array([0.0, half_w, -half_w, half_w, -half_w], dtype=jnp.float32)

            def _point_cost(point):
                c, s = jnp.cos(point[2]), jnp.sin(point[2])
                wx = point[0] + c * _dx_body - s * _dy_body
                wy = point[1] + s * _dx_body + c * _dy_body
                return jnp.max(jax.vmap(query)(wx, wy))
        else:
            def _point_cost(point):
                return query(point[0], point[1])

    def raw_fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        if _adaptive_v:
            traj, nu_traj = rollout_fn.with_nu(eta0, nu0, u_phys)
        else:
            traj = rollout_fn(eta0, nu0, u_phys)

        diffs = jnp.diff(traj[:, :2], axis=0)
        seg_lens = jnp.sqrt(jnp.sum(diffs ** 2, axis=1) + 1e-8)

        if _adaptive_v:
            wp_kes, wp_oobs = jax.vmap(_point_cost_split)(traj)
            # v²(t) = u_t² + v_t² in body frame (equal to vx² + vy² in world
            # frame). Used to scale the ice-cell KE coefficient per waypoint.
            v_sq = nu_traj[:, 0] ** 2 + nu_traj[:, 1] ** 2
            wp_costs = jnp.maximum(_base_cost, wp_kes * v_sq) + wp_oobs
        else:
            wp_costs = jax.vmap(_point_cost)(traj)
        # Zero out cost contributions for waypoints AFTER the goal line is first
        # crossed.  We use a cumulative-max so the mask respects trajectory order:
        # every step at or after the first crossing of _truncate_y is suppressed,
        # but steps before (including the crossing step itself) are kept.
        # _has_truncate is a Python bool evaluated at trace time, so the branch
        # is resolved statically and never touches traced JAX values.
        if _has_truncate:
            y_cummax = jnp.maximum.accumulate(traj[:, 1])
            # prev_cummax[t] = max y seen strictly before step t
            prev_cummax = jnp.concatenate(
                [jnp.array([jnp.float32(-1e9)]), y_cummax[:-1]]
            )
            y_mask = (prev_cummax < _truncate_y).astype(jnp.float32)
            wp_costs = wp_costs * y_mask
            # Mask path-length the same way: only count segments before goal crossing.
            # Without this, a long horizon (e.g. H=800, dt=5s → 4000s rollout) makes
            # path_length ~56 km regardless of trajectory quality, collapsing reward
            # variance to zero and causing softmax overflow (NaN weights).
            path_length = (seg_lens * y_mask[:-1]).sum()
        else:
            path_length = seg_lens.sum()
        seg_costmap = 0.5 * (wp_costs[:-1] + wp_costs[1:]) * seg_lens
        costmap_cost = seg_costmap.sum()

        # Exponentially-discounted running mean goal distance (same as cost_fn).
        H_plus1 = traj.shape[0]
        t_idx = jnp.arange(H_plus1, dtype=jnp.float32)
        tau = float(H_plus1) / 3.0
        discounts = jnp.exp(-t_idx / tau)
        discounts = discounts / discounts.sum()
        dists_all = jnp.sqrt(jnp.sum((traj[:, :2] - goal_xy) ** 2, axis=1) + 1e-4)
        goal_dist = jnp.dot(discounts, dists_all)

        du = jnp.diff(u_norm, axis=0)
        jerk = jnp.mean(du ** 2)

        return jnp.array([costmap_cost, path_length, goal_dist, jerk])

    return raw_fn


def make_cost_fn(
    grid: jnp.ndarray,
    meta: dict,
    goal_world,
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    costmap_weight: float = 1.0,
    costmap_normalizer: float = 1.0,
    goal_weight: float = 0.1,
    length_weight: float = 1.0,
    smoothness_weight: float = 0.005,
    ship_length: float = 0.0,
    ship_beam: float = 0.0,
    cost_truncate_y: Optional[float] = None,
    oob_penalty: float = 1e8,
):
    """
    Build cost J(u_norm) matching the A* lattice planner's cost structure.

    For each trajectory segment the cost is:
        arc_length  +  max_hull_costmap × segment_length
    summed (accumulated) over the full trajectory — identical to the lattice
    planner's ``_compute_path_cost``.  A small goal-distance term provides
    directional guidance (the lattice planner gets this from the A* heuristic).

    costmap_normalizer scales down the raw costmap integral so that
    ``costmap_weight=1.0`` gives comparable magnitude to the path-length term.
    Effective costmap weight = costmap_weight / costmap_normalizer.
    Use ``calibration/calibrate_from_mbd_debug.py`` to determine the right value.
    """
    raw_fn = _make_raw_components_fn(
        grid, meta, goal_world, rollout_fn, eta0, nu0,
        action_center, action_scale, ship_length, ship_beam,
        cost_truncate_y=cost_truncate_y,
        oob_penalty=oob_penalty,
    )
    weights = jnp.array(
        [costmap_weight / costmap_normalizer, length_weight, goal_weight, smoothness_weight],
        dtype=jnp.float32,
    )

    def cost_fn(u_norm):
        return jnp.dot(weights, raw_fn(u_norm))

    return cost_fn


# ── Waypoint-mode MBD ────────────────────────────────────────────────────────────
# Decision variable is the sequence of interior (x, y) waypoints in a chord-frame
# parameterization (origin at start, x-axis along start→goal). Endpoints are
# fixed by construction, so the planner cannot fail to reach the goal. There is
# no dynamics rollout — the trajectory is the polyline, scored by integrating
# the KE costmap along densely-sampled sub-segments.

def make_waypoint_cost_fn(
    grid: jnp.ndarray,
    meta: dict,
    start_xy: Tuple[float, float],
    goal_xy:  Tuple[float, float],
    *,
    n_interior: int,
    along_scale: float,
    lat_scale:   float,
    n_dense_per_seg: int = 20,
    costmap_weight: float = 1.0,
    length_weight:  float = 0.0,
    oob_penalty:    float = 0.0,
):
    """Build a cost over interior waypoints expressed in a chord frame.

    Decision variable ``Y ∈ [-1, 1]^{n_interior × 2}``:
        Y[i, 0] → along-chord offset (scaled by ``along_scale``)
        Y[i, 1] → lateral  offset    (scaled by ``lat_scale``)

    World waypoint ``i`` = ``start + (i / (n_interior + 1)) * (goal - start)
    + Y[i, 0] * along_scale * e_along + Y[i, 1] * lat_scale * e_lat``.
    ``Ybar = 0`` therefore corresponds to a uniformly-spaced straight line from
    start to goal — a sensible diffusion prior.

    Cost = ``costmap_weight · ∫ KE(s) ds  +  length_weight · ∫ ds`` evaluated
    by densely interpolating ``n_dense_per_seg`` points per polyline segment
    and trapezoidal integration of the KE costmap (queried via ``_make_query``).
    """
    start = jnp.asarray(start_xy, dtype=jnp.float32)
    goal  = jnp.asarray(goal_xy,  dtype=jnp.float32)
    chord = goal - start
    chord_len = jnp.sqrt(jnp.sum(chord ** 2) + 1e-8)
    e_along = chord / chord_len
    e_lat   = jnp.stack([-e_along[1], e_along[0]])

    frac = jnp.arange(1, n_interior + 1, dtype=jnp.float32) / float(n_interior + 1)
    P0   = start[None, :] + frac[:, None] * chord[None, :]   # (n_interior, 2)

    along_s = jnp.float32(along_scale)
    lat_s   = jnp.float32(lat_scale)

    query = _make_query(grid, meta, oob_penalty=oob_penalty)

    K = int(n_dense_per_seg)
    # Skip α=0 so each sub-segment endpoint is hit exactly once; α=1 is the
    # next vertex itself.  Concatenating ``start`` once at the front makes the
    # dense sequence tile cleanly with no duplicates.
    alphas_dense = jnp.linspace(0.0, 1.0, K + 1, dtype=jnp.float32)[1:]

    def cost_fn(Y_norm):
        offset = (
            Y_norm[:, 0:1] * along_s * e_along[None, :]
            + Y_norm[:, 1:2] * lat_s * e_lat[None, :]
        )
        interior = P0 + offset                                          # (n_interior, 2)
        path = jnp.concatenate([start[None], interior, goal[None]], axis=0)

        seg_starts = path[:-1]
        seg_diffs  = path[1:] - path[:-1]
        dense = seg_starts[:, None, :] + alphas_dense[None, :, None] * seg_diffs[:, None, :]
        dense = dense.reshape(-1, 2)
        dense = jnp.concatenate([start[None], dense], axis=0)           # (S*K + 1, 2)

        costs = jax.vmap(query)(dense[:, 0], dense[:, 1])
        seg = jnp.diff(dense, axis=0)
        seg_lens = jnp.sqrt(jnp.sum(seg ** 2, axis=1) + 1e-8)
        cost_integral = jnp.sum(0.5 * (costs[:-1] + costs[1:]) * seg_lens)
        length = seg_lens.sum()
        return costmap_weight * cost_integral + length_weight * length

    return cost_fn


def run_waypoint_mbd(
    cost_fn,
    *,
    n_interior: int,
    D: int = 2,
    Nsample: int,
    Ndiffuse: int,
    temperature: float,
    beta0: float,
    betaT: float,
    seed: int,
    Ybar_init: Optional[jnp.ndarray] = None,
):
    """Reverse-diffusion optimisation over normalised waypoint offsets.

    Mirrors :func:`run_mbd` (sample → softmax(−cost/τ) → weighted mean → reverse
    step) on a ``(n_interior, D)`` decision variable instead of ``(H, 3)``
    controls.  No rollout, no dynamics — the cost function is responsible for
    turning ``Y`` into a polyline and scoring it.

    Returns
    -------
    Ybar_final : (n_interior, D) float32 — optimised normalised offsets
    cost_means : (Ndiffuse - 1,) float32 — mean cost per diffusion step
    cost_mins  : (Ndiffuse - 1,) float32 — min  cost per diffusion step
    """
    if Ybar_init is None:
        Ybar_i = jnp.zeros((n_interior, D), dtype=jnp.float32)
    else:
        Ybar_i = jnp.clip(jnp.asarray(Ybar_init, dtype=jnp.float32), -1.0, 1.0)

    if Ndiffuse <= 1:
        return Ybar_i, jnp.zeros((0,), dtype=jnp.float32), jnp.zeros((0,), dtype=jnp.float32)

    rng = jax.random.PRNGKey(seed)
    betas = jnp.linspace(beta0, betaT, Ndiffuse, dtype=jnp.float32)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1.0 - alphas_bar)

    cost_batch = jax.vmap(cost_fn)

    def _reverse_step(carry, _):
        i, rng_c, Ybar_c = carry
        Yi = Ybar_c * jnp.sqrt(alphas_bar[i])

        rng_c, rng_s = jax.random.split(rng_c)
        eps = jax.random.normal(rng_s, (Nsample, n_interior, D), dtype=jnp.float32)
        Y0s = eps * sigmas[i] + Ybar_c
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        costs = cost_batch(Y0s).astype(jnp.float32)
        rews = -costs
        logp0 = (rews - rews.max()) / jnp.float32(temperature)
        weights = jax.nn.softmax(logp0)
        Ybar_new = jnp.einsum("n,nij->ij", weights, Y0s)

        score = (
            1.0 / (1.0 - alphas_bar[i])
            * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar_new)
        )
        Yim1 = (
            1.0 / jnp.sqrt(alphas[i])
            * (Yi + (1.0 - alphas_bar[i]) * score)
        )
        i_prev = jnp.maximum(i - 1, 0)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i_prev])
        Ybar_im1 = jnp.clip(Ybar_im1, -1.0, 1.0).astype(jnp.float32)
        return (i - 1, rng_c, Ybar_im1), (costs.mean(), costs.min())

    @jax.jit
    def _run(Ybar_start, rng_start):
        init_carry = (jnp.int32(Ndiffuse - 1), rng_start, Ybar_start)
        final_carry, metrics = jax.lax.scan(
            _reverse_step, init_carry, None, length=Ndiffuse - 1,
        )
        return final_carry[2], metrics[0], metrics[1]

    Ybar_final, cost_means, cost_mins = _run(Ybar_i, rng)
    return Ybar_final, cost_means, cost_mins


def make_cost_components_fn(
    grid: jnp.ndarray,
    meta: dict,
    goal_world,
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    ship_length: float = 0.0,
    ship_beam: float = 0.0,
):
    """
    Like make_cost_fn but returns individual (unweighted) cost components.

    Returns a function components_fn(u_norm) -> jnp.array([costmap, length, goal, jerk])
    using the same formulas as make_cost_fn so that weighted = raw * weight_vector
    is always consistent with the total cost.
    """
    return _make_raw_components_fn(
        grid, meta, goal_world, rollout_fn, eta0, nu0,
        action_center, action_scale, ship_length, ship_beam,
    )


def make_goal_line_cost_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    goal_y_line: float,
    penalty: float = 1e6,
):
    """
    Big-M penalty for trajectories that never reach y >= goal_y_line.

    Returns 0 if any waypoint satisfies y >= goal_y_line, else `penalty`.
    Intended to replace the soft goal-distance term with a hard constraint.
    """
    _goal_y = jnp.float32(goal_y_line)
    _penalty = jnp.float32(penalty)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)
        # max_y = jnp.max(traj[:, 1])
        y_final = traj[-1, 1]
        return jnp.where(y_final >= _goal_y, jnp.float32(0.0), _penalty)

    return fn


def make_goal_progress_cost_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    goal_y_line: float,
    weight: float,
):
    """Per-step "shortfall to goal-line" cost — incentivises forward motion.

    Returns ``weight · Σ_{t=0..H} max(0, goal_y_line − y_t)``.  Each waypoint
    below ``goal_y_line`` contributes its shortfall to the cost; waypoints at
    or above the line contribute zero.  This gives the diffusion samples a
    progress gradient at every step (cf. terminal-only Euclidean), and the
    clipping keeps the incentive bounded once the ship has crossed the line.
    """
    _goal_y = jnp.float32(goal_y_line)
    _w = jnp.float32(weight)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)
        shortfall = jnp.maximum(_goal_y - traj[:, 1], 0.0)
        return _w * jnp.sum(shortfall)

    return fn


def make_y_bottom_cost_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    y_min: float,
    penalty: float = 1e8,
):
    """
    Big-M penalty for trajectories whose ship-centre dips below y_min at any
    waypoint.  Mirrors the toy reject filter's y-bottom check, used by planners
    that have no native OOB-constraint mechanism (MPPI/RSS/iLQR).
    """
    _y_min = jnp.float32(y_min)
    _penalty = jnp.float32(penalty)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)
        below = jnp.any(traj[:, 1] < _y_min)
        return jnp.where(below, _penalty, jnp.float32(0.0))

    return fn


def make_terminal_goal_distance_cost_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    goal_x: float,
    goal_y_line: float,
    weight: float = 1.0,
):
    """
    Distance-to-goal cost that ignores anything past the goal line.

    Two regimes (selected by whether any waypoint crosses ``goal_y_line``):

      • Trajectory reaches/crosses the goal line:
            cost = weight · |x_crossing - goal_x|
        ``x_crossing`` is linearly interpolated between the two waypoints
        straddling the first ``y >= goal_y_line`` transition.  This way
        anything the trajectory does *after* crossing the line (sideways
        drift, overshoot in y) is not penalised.

      • Trajectory misses the goal (no waypoint reaches goal_y_line):
            cost = weight · sqrt( (x_final - goal_x)² + (goal_y_line - y_final)² )
        Full Euclidean distance from the final waypoint to the goal point.
        Provides smooth gradient information in BOTH x and y so misses
        still feel a pull toward the goal.

    A tiny ε is added inside the sqrt on the miss branch for differentiability
    at the goal.
    """
    _goal_x = jnp.float32(goal_x)
    _goal_y = jnp.float32(goal_y_line)
    _weight = jnp.float32(weight)
    _eps_sq = jnp.float32(1e-6)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)            # (H+1, 3)

        # ── First crossing of goal_y_line ────────────────────────────────
        crossed = (traj[:, 1] >= _goal_y).astype(jnp.int32)
        any_crossed = crossed.sum() > 0
        i1 = jnp.argmax(crossed)                        # 0 if never crossed
        i0 = jnp.maximum(i1 - 1, 0)
        y0 = traj[i0, 1]
        y1 = traj[i1, 1]
        x0 = traj[i0, 0]
        x1 = traj[i1, 0]
        dy_seg = y1 - y0
        frac = jnp.where(dy_seg > 1e-6, (_goal_y - y0) / dy_seg, 0.0)
        frac = jnp.clip(frac, 0.0, 1.0)
        x_crossing = x0 + frac * (x1 - x0)

        # ── Two regimes ──────────────────────────────────────────────────
        cost_reach = jnp.abs(x_crossing - _goal_x)
        dx_miss    = traj[-1, 0] - _goal_x
        dy_miss    = _goal_y - traj[-1, 1]
        cost_miss  = jnp.sqrt(dx_miss * dx_miss + dy_miss * dy_miss + _eps_sq)

        return _weight * jnp.where(any_crossed, cost_reach, cost_miss)

    return fn


def make_goal_arrival_x_cost_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    goal_x: float,
    goal_y_line: float,
    weight: float = 1.0,
):
    """
    Soft cost penalising x-deviation at the first crossing of goal_y_line.

    Linearly interpolates the crossing x between the waypoint before and after
    the first y >= goal_y_line transition.  If the trajectory never reaches the
    goal line the last waypoint x is used (the goal-line big-M penalty handles
    the infeasible case more strongly).

    Returns weight * |x_crossing - goal_x|.
    """
    _goal_x = jnp.float32(goal_x)
    _goal_y = jnp.float32(goal_y_line)
    _weight = jnp.float32(weight)
    H_plus1 = eta0.shape[0] if hasattr(eta0, "shape") else None  # resolved at trace time

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)            # (H+1, 3)

        crossed = (traj[:, 1] >= _goal_y).astype(jnp.int32)
        first_idx = jnp.argmax(crossed)                 # 0 if never crossed
        any_crossed = crossed.sum() > 0

        # Linear interpolation between step before and step of crossing
        i1 = first_idx
        i0 = jnp.maximum(i1 - 1, 0)
        y0 = traj[i0, 1]
        y1 = traj[i1, 1]
        x0 = traj[i0, 0]
        x1 = traj[i1, 0]
        dy = y1 - y0
        frac = jnp.where(dy > 1e-6, (_goal_y - y0) / dy, 0.0)
        frac = jnp.clip(frac, 0.0, 1.0)
        x_crossing_interp = x0 + frac * (x1 - x0)

        # Fall back to last waypoint x if goal never reached
        x_crossing = jnp.where(any_crossed, x_crossing_interp, traj[-1, 0])
        return _weight * jnp.abs(x_crossing - _goal_x)

    return fn


def make_terminal_goal_euclidean_cost_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    goal_xy,
    weight: float = 1.0,
):
    """
    Soft cost: weight * || traj[-1, :2] - goal_xy ||.

    Penalises only the trajectory's final waypoint by its 2-D Euclidean
    distance to the goal point.  Intended to *replace* both the goal-line
    big-M penalty and the goal-x crossing soft cost with a single smooth
    terminal pull toward (goal_x, goal_y).
    """
    _gx = jnp.float32(goal_xy[0])
    _gy = jnp.float32(goal_xy[1])
    _weight = jnp.float32(weight)
    _eps_sq = jnp.float32(1e-6)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)             # (H+1, 3)
        dx = traj[-1, 0] - _gx
        dy = traj[-1, 1] - _gy
        return _weight * jnp.sqrt(dx * dx + dy * dy + _eps_sq)

    return fn


def make_almbd_reference_cost_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    goal_xy,
    stage_action_weight: float = 0.1,
    terminal_weight: float = 20.0,
):
    """
    Mirror of the AL-MBD reference J(Y) (al_mbd_zscore_reference.md):

        stage_cost(x_t, u_t) = ||(x_t, y_t) - goal||_2  +  stage_action_weight * ||u_t||_2
        terminal_cost(x_T)   = terminal_weight * ||(x_T, y_T) - goal||_2
        J(Y) = sum_t stage_cost(x_t, u_t) + terminal_cost(x_T)

    Adapted for the ship: state is (x, y, psi) so goal-distance uses the 2D
    position only; action is the 3-vector of NORMALISED (surge, sway, yaw)
    in [-1, 1] (stage_action_weight=0.1 matches the reference "u/10").
    """
    _gx = jnp.float32(goal_xy[0])
    _gy = jnp.float32(goal_xy[1])
    _a_w = jnp.float32(stage_action_weight)
    _t_w = jnp.float32(terminal_weight)
    _eps = jnp.float32(1e-6)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)             # (H+1, 3)
        dx = traj[:, 0] - _gx
        dy = traj[:, 1] - _gy
        dists = jnp.sqrt(dx * dx + dy * dy + _eps)        # (H+1,)
        stage_dist = dists[:-1].sum()
        terminal = _t_w * dists[-1]
        # ||u_t||_2 over the normalised action 3-vector
        u_mags = jnp.sqrt(jnp.sum(u_norm ** 2, axis=-1) + _eps)
        stage_action = _a_w * u_mags.sum()
        return stage_dist + stage_action + terminal

    return fn


# ── EB-MBD constraint functions ──────────────────────────────────────────

def make_oob_constraint_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    cost_truncate_y: Optional[float] = None,
):
    """
    OOB constraint function for EB-MBD.

    Returns g_OOB(u_norm) = min over active waypoints of the signed distance
    to the nearest OOB boundary (left, right, bottom walls only — no top wall).

      g_OOB > 0  ⟺  all active waypoints are strictly inside [x_min, x_max] × [y_min, ∞)
      g_OOB ≤ 0  ⟺  at least one active waypoint is OOB

    When cost_truncate_y is set (typically = goal_y_line), waypoints after the
    first crossing of that y value are excluded from the min computation.
    """
    _x_min = jnp.float32(x_min)
    _x_max = jnp.float32(x_max)
    _y_min = jnp.float32(y_min)
    _has_truncate = cost_truncate_y is not None
    _truncate_y = jnp.float32(cost_truncate_y if cost_truncate_y is not None else 0.0)
    _large = jnp.float32(1e9)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)   # (H+1, 3)
        xs = traj[:, 0]
        ys = traj[:, 1]
        # Signed distance to each wall: positive inside, negative outside
        dist_left   = xs - _x_min    # positive when x > x_min
        dist_right  = _x_max - xs    # positive when x < x_max
        dist_bottom = ys - _y_min    # positive when y > y_min
        if _has_truncate:
            # Mask out waypoints after the goal line is first crossed.
            # Inactive waypoints contribute +inf so they never affect the min.
            y_cummax = jnp.maximum.accumulate(ys)
            prev_cummax = jnp.concatenate(
                [jnp.array([jnp.float32(-1e9)]), y_cummax[:-1]]
            )
            active = (prev_cummax < _truncate_y)
            dist_left   = jnp.where(active, dist_left,   _large)
            dist_right  = jnp.where(active, dist_right,  _large)
            dist_bottom = jnp.where(active, dist_bottom, _large)
        return jnp.minimum(
            jnp.minimum(jnp.min(dist_left), jnp.min(dist_right)),
            jnp.min(dist_bottom),
        )

    return fn


def make_oob_cost_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    oob_cost_per_m: float = 5.0,
    cost_truncate_y: Optional[float] = None,
):
    """
    Linear OOB cost added to the base costmap cost for EB-MBD.

    Returns oob_cost_per_m × Σ_t max(0, violation_t), where violation_t is
    the sum of boundary violations (x < x_min, x > x_max, y < y_min) at each
    waypoint.  Zero for fully in-bounds trajectories; proportional to distance
    OOB otherwise — providing a smooth gradient within an all-OOB class without
    the wall-avoidance bias of a two-sided log barrier.
    """
    _x_min = jnp.float32(x_min)
    _x_max = jnp.float32(x_max)
    _y_min = jnp.float32(y_min)
    _scale = jnp.float32(oob_cost_per_m)
    _has_truncate = cost_truncate_y is not None
    _truncate_y = jnp.float32(cost_truncate_y if cost_truncate_y is not None else 0.0)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)   # (H+1, 3)
        xs = traj[:, 0]
        ys = traj[:, 1]
        viol = (jnp.maximum(0.0, _x_min - xs)
                + jnp.maximum(0.0, xs - _x_max)
                + jnp.maximum(0.0, _y_min - ys))
        if _has_truncate:
            y_cummax = jnp.maximum.accumulate(ys)
            prev_cummax = jnp.concatenate(
                [jnp.array([jnp.float32(-1e9)]), y_cummax[:-1]]
            )
            y_mask = (prev_cummax < _truncate_y).astype(jnp.float32)
            viol = viol * y_mask
        return _scale * jnp.sum(viol)

    return fn


def make_goal_constraint_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    goal_y_line: float,
):
    """
    Goal-reaching constraint function for EB-MBD.

    Returns g_goal(u_norm) = y_final - goal_y_line.

      g_goal ≥ 0  ⟺  the final waypoint reaches or exceeds the goal line
      g_goal < 0  ⟺  trajectory ends short of the goal by |g_goal| metres
    """
    _goal_y = jnp.float32(goal_y_line)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)
        return traj[-1, 1] - _goal_y

    return fn


def make_goal_point_constraint_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    goal_xy: Tuple[float, float],
    tolerance: float = 20.0,
):
    """
    Goal-point constraint for EB-MBD.

    Returns g_goal(u_norm) = tolerance - ‖x_final - goal_xy‖.

      g_goal ≥ 0  ⟺  final waypoint is within `tolerance` metres of goal_xy
      g_goal < 0  ⟺  final waypoint is `|g_goal|` metres outside the ball

    Use this instead of make_goal_constraint_fn (which only enforces the
    goal y-line) when the planner must reach the actual goal *point*.
    """
    _gx = jnp.float32(goal_xy[0])
    _gy = jnp.float32(goal_xy[1])
    _tol = jnp.float32(tolerance)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)
        dx = traj[-1, 0] - _gx
        dy = traj[-1, 1] - _gy
        dist = jnp.sqrt(dx * dx + dy * dy + 1e-8)
        return _tol - dist

    return fn


def make_ice_cost_constraint_fn(
    grid: jnp.ndarray,
    meta: dict,
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    ice_threshold: float = 0.0,
    cost_truncate_y: Optional[float] = None,
    ship_length: float = 0.0,
    ship_beam: float = 0.0,
):
    """
    Ice-avoidance constraint for EB-MBD.

    Returns g_ice(u_norm) = ice_threshold - total_costmap_integral_along_traj.

      g_ice >= 0  ⟺  total ice cost <= ice_threshold  (trajectory avoids ice)
      g_ice <  0  ⟺  trajectory hits ice (infeasible under the EB barrier)

    With ice_threshold=0.0 (default), any ice contact is infeasible.
    Uses the same trapezoid-rule costmap integral as make_cost_fn, with
    optional hull footprint sampling and goal-line truncation.
    """
    _adaptive_v = bool(meta.get("adaptive_velocity", False))
    _threshold = jnp.float32(ice_threshold)
    _has_truncate = cost_truncate_y is not None
    _truncate_y = jnp.float32(cost_truncate_y if cost_truncate_y is not None else 0.0)

    half_l = float(ship_length) / 2.0
    half_w = float(ship_beam) / 2.0
    use_hull = ship_length > 0 and ship_beam > 0

    if _adaptive_v:
        query_split = _make_query_split(grid, meta, oob_penalty=0.0)
        if use_hull:
            _dx_body = jnp.array([0.0, half_l, half_l, -half_l, -half_l], dtype=jnp.float32)
            _dy_body = jnp.array([0.0, half_w, -half_w, half_w, -half_w], dtype=jnp.float32)

            def _point_ke(point):
                c, s = jnp.cos(point[2]), jnp.sin(point[2])
                wx = point[0] + c * _dx_body - s * _dy_body
                wy = point[1] + s * _dx_body + c * _dy_body
                kes, _ = jax.vmap(query_split)(wx, wy)
                return jnp.max(kes)
        else:
            def _point_ke(point):
                ke, _ = query_split(point[0], point[1])
                return ke
    else:
        query = _make_query(grid, meta, oob_penalty=0.0)
        if use_hull:
            _dx_body = jnp.array([0.0, half_l, half_l, -half_l, -half_l], dtype=jnp.float32)
            _dy_body = jnp.array([0.0, half_w, -half_w, half_w, -half_w], dtype=jnp.float32)

            def _point_cost(point):
                c, s = jnp.cos(point[2]), jnp.sin(point[2])
                wx = point[0] + c * _dx_body - s * _dy_body
                wy = point[1] + s * _dx_body + c * _dy_body
                return jnp.max(jax.vmap(query)(wx, wy))
        else:
            def _point_cost(point):
                return query(point[0], point[1])

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        if _adaptive_v:
            traj, nu_traj = rollout_fn.with_nu(eta0, nu0, u_phys)
        else:
            traj = rollout_fn(eta0, nu0, u_phys)

        diffs = jnp.diff(traj[:, :2], axis=0)
        seg_lens = jnp.sqrt(jnp.sum(diffs ** 2, axis=1) + 1e-8)

        if _adaptive_v:
            wp_kes = jax.vmap(_point_ke)(traj)
            v_sq = nu_traj[:, 0] ** 2 + nu_traj[:, 1] ** 2
            wp_costs = wp_kes * v_sq
        else:
            wp_costs = jax.vmap(_point_cost)(traj)
        if _has_truncate:
            y_cummax = jnp.maximum.accumulate(traj[:, 1])
            prev_cummax = jnp.concatenate(
                [jnp.array([jnp.float32(-1e9)]), y_cummax[:-1]]
            )
            y_mask = (prev_cummax < _truncate_y).astype(jnp.float32)
            wp_costs = wp_costs * y_mask

        seg_costmap = 0.5 * (wp_costs[:-1] + wp_costs[1:]) * seg_lens
        total_ice_cost = seg_costmap.sum()
        return _threshold - total_ice_cost

    return fn


def make_combined_constraint_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    goal_y_line: float,
):
    """
    Combined OOB + goal constraint for EB-MBD.

    Returns g(u_norm) = min(g_OOB(u_norm), g_goal(u_norm)).

    A trajectory is fully feasible when g ≥ 0 (all waypoints inside
    bounds AND final waypoint reaches the goal line).  The barrier
    progressively enforces this as c_s decreases to 0.
    """
    oob_fn  = make_oob_constraint_fn(
        rollout_fn, eta0, nu0, action_center, action_scale,
        x_min, x_max, y_min,
    )
    goal_fn = make_goal_constraint_fn(
        rollout_fn, eta0, nu0, action_center, action_scale,
        goal_y_line,
    )

    def fn(u_norm):
        return jnp.minimum(oob_fn(u_norm), goal_fn(u_norm))

    return fn


def make_rect_obstacles_eb_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    rects: list,
):
    """
    EB-MBD hard-obstacle constraint for rectangular exclusion zones.

    Each rect is (x0, x1, y0, y1) in world coordinates.  Returns a scalar g
    where g > 0 iff ALL trajectory waypoints are OUTSIDE all rectangles.

    Uses the box signed-distance function: SDF > 0 outside, < 0 inside.
    """
    # Pre-compute centres and half-extents as JAX constants
    _boxes = [(jnp.float32((x0 + x1) / 2), jnp.float32((x1 - x0) / 2),
               jnp.float32((y0 + y1) / 2), jnp.float32((y1 - y0) / 2))
              for x0, x1, y0, y1 in rects]

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)   # (H+1, 3)
        xs, ys = traj[:, 0], traj[:, 1]
        g = jnp.inf
        for cx, ax, cy, ay in _boxes:
            sdf = jnp.maximum(jnp.abs(xs - cx) - ax, jnp.abs(ys - cy) - ay)
            g = jnp.minimum(g, jnp.min(sdf))   # min over waypoints, then rects
        return g

    return fn


def make_polygon_obstacles_al_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    polygons: list,
):
    """AL-MBD per-timestep polygon-obstacle constraint (exact polygon SDF).

    Each ``polygons[j]`` is an ``(N_j, 2)`` array of CCW vertices (no repeat
    of the first vertex at the end).  Returns ``g`` of shape ``(H, n_polys)``
    using the AL sign convention (g > 0 ⇒ violated): ``g_{t,j} = −SDF_j(p_t)``
    where the SDF is positive outside the polygon and negative inside.

    Implementation notes
    --------------------
    *   Polygons are padded to a common ``V_max`` vertex count by repeating
        the last vertex.  Padded edges have zero length and are masked out of
        both the unsigned-distance min and the even-odd point-in-polygon test.
    *   The unsigned distance is the standard point-to-segment formula
        (project onto segment, clip parametric coordinate to ``[0, 1]``).
    *   Inside-test uses a horizontal ray going +x from the query point and
        counts how many real (unmasked) edges it crosses; an odd count = inside.
    *   Designed to ``jax.vmap`` cleanly over (waypoint, polygon) — the
        per-step cost is ``O(H · n_polys · V_max)``.
    """
    if not polygons:
        raise ValueError("polygons list is empty")
    V_max = max(len(p) for p in polygons)
    n_polys = len(polygons)
    padded = np.zeros((n_polys, V_max, 2), dtype=np.float32)
    masks  = np.zeros((n_polys, V_max), dtype=bool)
    for j, p in enumerate(polygons):
        arr = np.asarray(p, dtype=np.float32)
        V = len(arr)
        padded[j, :V] = arr
        # Pad with the FIRST vertex so the closing edge V-1 (which connects
        # the last real vertex to poly[V] = poly[0]) is geometrically correct.
        # Padded edges V..V_max-1 collapse to v0→v0 (zero length) and are
        # masked out below.  Closing edge V-1 itself is kept as real.
        if V < V_max:
            padded[j, V:] = arr[0]
        masks[j, :V] = True
    polys_jnp = jnp.asarray(padded)        # (n_polys, V_max, 2)
    edge_mask = jnp.asarray(masks)         # (n_polys, V_max)
    _inf = jnp.float32(1e9)
    _eps = jnp.float32(1e-8)

    def _sdf_one_poly(poly, mask, point):
        # Edge k goes from poly[k] to poly[(k+1) mod V_max]
        a = poly
        b = jnp.roll(poly, -1, axis=0)
        ab = b - a                                       # (V_max, 2)
        ap = point[None, :] - a                          # (V_max, 2)
        ab_len_sq = jnp.sum(ab * ab, axis=-1) + _eps     # (V_max,)
        t = jnp.sum(ap * ab, axis=-1) / ab_len_sq        # (V_max,)
        t_clip = jnp.clip(t, 0.0, 1.0)
        closest = a + t_clip[:, None] * ab               # (V_max, 2)
        dist = jnp.linalg.norm(point[None, :] - closest, axis=-1)
        dist = jnp.where(mask, dist, _inf)
        d_min = jnp.min(dist)
        # Even-odd inside test using horizontal +x ray from the query point.
        # An edge from (x1,y1) to (x2,y2) crosses the ray iff
        # (y1 > py) != (y2 > py)  AND  px < x_intersect, where x_intersect
        # is the edge's x-coordinate at y = py.
        py = point[1]; px = point[0]
        y1 = a[:, 1]; y2 = b[:, 1]
        x1 = a[:, 0]; x2 = b[:, 0]
        cond1 = (y1 > py) != (y2 > py)
        x_isect = (x2 - x1) * (py - y1) / (y2 - y1 + _eps) + x1
        cond2 = px < x_isect
        crosses = cond1 & cond2 & mask
        n_crosses = jnp.sum(crosses.astype(jnp.int32))
        inside = (n_crosses % 2) == 1
        return jnp.where(inside, -d_min, d_min)

    def _sdf_all_polys_at_point(point):
        return jax.vmap(_sdf_one_poly, in_axes=(0, 0, None))(
            polys_jnp, edge_mask, point
        )

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)             # (H+1, 3)
        points = traj[1:, :2]                            # (H, 2)
        sdfs = jax.vmap(_sdf_all_polys_at_point)(points) # (H, n_polys)
        return -sdfs                                     # AL: g > 0 ⇔ inside

    return fn


def make_disk_obstacles_al_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    disks: list,
):
    """AL-MBD per-timestep disk-obstacle constraint.

    Each disk is ``(cx, cy, r)``.  Returns ``g`` of shape ``(H, n_disks)``
    using AL sign convention (g > 0 ⇒ violated): g_{t, j} = r_j - dist(p_t, c_j),
    positive when waypoint t penetrates disk j.

    Mirrors ``make_rect_obstacles_al_fn`` but for circular ice floes — used by
    the scenarios whose costmap is built from disk obstacles (thin_corridor,
    asymmetric, cluttered).
    """
    cxs = jnp.array([float(d[0]) for d in disks], dtype=jnp.float32)
    cys = jnp.array([float(d[1]) for d in disks], dtype=jnp.float32)
    rs  = jnp.array([float(d[2]) for d in disks], dtype=jnp.float32)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)            # (H+1, 3)
        xs = traj[1:, 0]                                 # (H,)
        ys = traj[1:, 1]
        dx = xs[:, None] - cxs[None, :]                  # (H, n_disks)
        dy = ys[:, None] - cys[None, :]
        dist = jnp.sqrt(dx * dx + dy * dy + 1e-8)
        return rs[None, :] - dist                        # (H, n_disks)

    return fn


def make_rect_obstacles_al_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    rects: list,
):
    """
    AL-MBD hard-obstacle constraint for rectangular exclusion zones.

    Returns g of shape (H, n_rects) using AL sign convention (g > 0 = violated).
    Violated at step t when the corresponding waypoint is inside any rectangle.
    """
    _boxes = [(jnp.float32((x0 + x1) / 2), jnp.float32((x1 - x0) / 2),
               jnp.float32((y0 + y1) / 2), jnp.float32((y1 - y0) / 2))
              for x0, x1, y0, y1 in rects]

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)   # (H+1, 3)
        xs = traj[1:, 0]    # (H,)
        ys = traj[1:, 1]    # (H,)
        cols = []
        for cx, ax, cy, ay in _boxes:
            sdf = jnp.maximum(jnp.abs(xs - cx) - ax, jnp.abs(ys - cy) - ay)
            cols.append((-sdf)[:, None])    # positive inside rect (violated)
        return jnp.concatenate(cols, axis=1)    # (H, n_rects)

    return fn


def make_goal_reach_al_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    goal_y_line: float,
    H: int,
):
    """
    Dedicated AL constraint column for terminal goal-reach.

    Returns g of shape (H, 1) using AL sign convention (g > 0 = violated):

        g[t, 0] = 0                          for t < H - 1
        g[H-1, 0] = goal_y_line - y_H        (positive ⟹ ship didn't reach)

    Kept as a SEPARATE column from OOB so the goal-reach can carry its own
    ρ (much larger than the per-floe ice ρ).
    """
    _goal_y = jnp.float32(goal_y_line)
    _H = int(H)

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)   # (H+1, 3)
        ys = traj[1:, 1]                       # (H,)
        # Zeros for t < H-1, goal-reach residual at t = H-1.
        # jnp.zeros_like keeps gradient/dtype semantics; we use `at[].set` to
        # avoid Python list construction inside the traced function.
        col = jnp.zeros((_H,), dtype=jnp.float32)
        col = col.at[-1].set(_goal_y - ys[-1])
        return col[:, None]                    # (H, 1)

    return fn


def make_al_per_step_constraints_fn(
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: Optional[float] = None,
    goal_y_line: Optional[float] = None,
):
    """
    Per-timestep constraint function for AL-MBD (AL paper convention: g ≤ 0 = feasible).

    Returns g(u_norm) of shape (H, 1).

    At each control step t the constraint is the maximum OOB penetration
    of the resulting waypoint (x_{t+1}, y_{t+1}):

        g_t = max(x_min − x_{t+1},  x_{t+1} − x_max,  y_min − y_{t+1},  y_{t+1} − y_max)

    At the last control step (t = H−1) the goal constraint is also included:

        g_{H-1} = max(g_OOB_{H-1},  goal_y_line − y_H)

    Positive value ⟹ violated; negative ⟹ satisfied (margin = |g_t|).
    """
    _x_min  = jnp.float32(x_min)
    _x_max  = jnp.float32(x_max)
    _y_min  = jnp.float32(y_min)
    _y_max  = jnp.float32(y_max) if y_max is not None else None
    _goal_y = jnp.float32(goal_y_line) if goal_y_line is not None else None

    def fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)   # (H+1, 3)
        xs = traj[1:, 0]    # (H,) — x after each control step
        ys = traj[1:, 1]    # (H,) — y after each control step

        # OOB: positive when outside bounds
        g_oob = jnp.maximum(
            jnp.maximum(_x_min - xs, xs - _x_max),
            _y_min - ys,
        )   # (H,)
        if _y_max is not None:
            g_oob = jnp.maximum(g_oob, ys - _y_max)

        if _goal_y is not None:
            # Goal: positive when below the goal line (only relevant at last step)
            g_goal_last = jnp.array([_goal_y]) - ys[-1:]   # (1,)
            g_last = jnp.maximum(g_oob[-1:], g_goal_last)  # (1,)
            g = jnp.concatenate([g_oob[:-1], g_last], axis=0)  # (H,)
        else:
            g = g_oob

        return g[:, None]   # (H, 1)

    return fn


# ── MBD reverse diffusion process (action space) ─────────────────────────

def run_mbd(
    cost_fn,
    H: int,
    Nsample: int,
    Ndiffuse: int,
    temperature: float,
    beta0: float,
    betaT: float,
    seed: int,
    Ybar_init: Optional[jnp.ndarray] = None,
    init_surge_bias: float = 0.7,
) -> Tuple[jnp.ndarray, float, float]:
    """
    Run MBD reverse diffusion in normalised action space.

    The entire diffusion loop is compiled into a single JAX computation via
    ``jax.lax.scan``, eliminating Python-loop overhead and enabling full GPU
    acceleration when a GPU backend is available.  All ``Nsample`` candidate
    trajectories are evaluated in parallel via ``jax.vmap``.

    Parameters
    ----------
    Ybar_init : (H, 3), optional
        Warm-start for the mean action sequence (e.g. derived from a lattice
        path).  If None, defaults to constant forward surge bias.

    Returns
    -------
    Ybar_final : (H, 3) optimised normalised actions in [-1, 1]
    cost_mean  : float  — mean cost at final step
    cost_min   : float  — min cost at final step
    """
    if Ybar_init is not None:
        Ybar_i = jnp.array(Ybar_init, dtype=jnp.float32)
        Ybar_i = jnp.clip(Ybar_i, -1.0, 1.0)
    else:
        Ybar_i = jnp.zeros((H, 3), dtype=jnp.float32)
        Ybar_i = Ybar_i.at[:, 0].set(init_surge_bias)

    if Ndiffuse <= 1:
        return Ybar_i, 0.0, 0.0

    rng = jax.random.PRNGKey(seed)

    betas = jnp.linspace(beta0, betaT, Ndiffuse, dtype=jnp.float32)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1.0 - alphas_bar)

    cost_batch = jax.vmap(cost_fn)

    def _reverse_step(carry, _unused):
        """Single reverse diffusion step (scan body — NOT individually JIT'd)."""
        i, rng_c, Ybar_c = carry
        Yi = Ybar_c * jnp.sqrt(alphas_bar[i])

        rng_c, rng_s = jax.random.split(rng_c)
        eps = jax.random.normal(rng_s, (Nsample, H, 3), dtype=jnp.float32)
        Y0s = eps * sigmas[i] + Ybar_c
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        costs = cost_batch(Y0s).astype(jnp.float32)
        rews = -costs
        # MBD paper: w_k = softmax(rews / temperature).
        # Max-subtract is a no-op on softmax output and keeps exp() stable.
        logp0 = (rews - rews.max()) / jnp.float32(temperature)
        weights = jax.nn.softmax(logp0)

        Ybar_new = jnp.einsum("n,nij->ij", weights, Y0s)

        score = (
            1.0
            / (1.0 - alphas_bar[i])
            * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar_new)
        )
        Yim1 = (
            1.0
            / jnp.sqrt(alphas[i])
            * (Yi + (1.0 - alphas_bar[i]) * score)
        )
        i_prev = jnp.maximum(i - 1, 0)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i_prev])
        Ybar_im1 = jnp.clip(Ybar_im1, -1.0, 1.0).astype(jnp.float32)

        return (i - 1, rng_c, Ybar_im1), (costs.mean(), costs.min())

    @jax.jit
    def _run_diffusion_loop(Ybar_start, rng_start):
        """Full reverse diffusion compiled as a single JAX computation."""
        init_carry = (jnp.int32(Ndiffuse - 1), rng_start, Ybar_start)
        final_carry, metrics = jax.lax.scan(
            _reverse_step, init_carry, None, length=Ndiffuse - 1,
        )
        return final_carry[2], metrics[0][-1], metrics[1][-1]

    Ybar_final, cost_mean_val, cost_min_val = _run_diffusion_loop(Ybar_i, rng)

    return Ybar_final, float(cost_mean_val), float(cost_min_val)


def run_mbd_with_diagnostics(
    cost_fn,
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    H: int,
    Nsample: int,
    Ndiffuse: int,
    temperature: float,
    beta0: float,
    betaT: float,
    seed: int,
    Ybar_init: Optional[jnp.ndarray] = None,
    init_surge_bias: float = 0.7,
    n_top_samples: int = 10,
    sample_filter_fn=None,
    max_resample_iters=None,
) -> dict:
    """
    Run MBD reverse diffusion and capture per-step diagnostics for visualisation.

    Unlike ``run_mbd`` (which uses ``jax.lax.scan`` for speed), this version
    runs the diffusion steps in Python so we can snapshot trajectories and
    samples at every iteration.

    Parameters
    ----------
    sample_filter_fn : callable, optional
        ``filter_fn(Y0s_norm, trajs) -> bool_mask`` where *Y0s_norm* is
        ``(Nsample, H, 3)`` normalised actions and *trajs* is
        ``(Nsample, H+1, 3)`` rolled-out trajectories.  Returns a boolean
        mask of shape ``(Nsample,)`` where ``True`` means the sample is
        *valid* (e.g., non-OOB and goal-achieving).  When provided, invalid
        samples are filtered out once (no resampling); only the remaining
        valid samples are used for cost evaluation and the weighted mean
        (i.e. invalid samples receive zero weight — indicator-function
        rejection).  If no valid samples remain at a step, ``Ybar_c`` is
        left unchanged for that step (rather than letting the rejected
        pool drive the mean).
    max_resample_iters : int or None
        Ignored — kept for API compatibility only (filter-once mode does not
        resample).

    Returns dict with:
        mean_trajectories  : list of (H+1, 3) arrays — Ybar rolled out at each step
        sample_trajectories: list of (n_top_samples, H+1, 3) — top samples at each step
        sample_costs       : list of (Nsample,) — all sample costs at each step
        cost_means         : list of floats
        cost_mins          : list of floats
        weights            : list of (Nsample,) — importance weights at each step
        accept_rates       : list of floats — fraction of accepted samples (1.0 when
                             sample_filter_fn is None)
        n_total_sampled    : list of ints — total draws per step (≥ Nsample; equals
                             Nsample when sample_filter_fn is None)
    """
    if Ybar_init is not None:
        Ybar_c = jnp.array(Ybar_init, dtype=jnp.float32)
        Ybar_c = jnp.clip(Ybar_c, -1.0, 1.0)
    else:
        Ybar_c = jnp.zeros((H, 3))
        Ybar_c = Ybar_c.at[:, 0].set(init_surge_bias)

    rng = jax.random.PRNGKey(seed)
    betas = jnp.linspace(beta0, betaT, Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1.0 - alphas_bar)

    cost_batch = jax.jit(jax.vmap(cost_fn))
    rollout_batch = jax.jit(jax.vmap(rollout_fn, in_axes=(None, None, 0)))
    rollout_single = jax.jit(rollout_fn)

    diag = {
        "mean_trajectories": [],
        "sample_trajectories": [],
        "sample_costs": [],
        "cost_means": [],
        "cost_mins": [],
        "cost_of_mean": [],   # cost_fn(Ybar_c) evaluated at each step
        "weights": [],
        "Ybar_final": None,   # filled at end of loop
        "best_sample_actions": None,   # (H, 3) normalized actions of global best sample
        "best_sample_cost": float("inf"),
        "second_best_sample_actions": None,  # (H, 3) normalized actions of 2nd-best (same step as best)
        "second_best_sample_cost": float("inf"),
        "accept_rates": [],      # fraction of valid samples per step (1.0 when no filter)
        "n_total_sampled": [],   # total draws per step (>= Nsample)
        # Top-N trajectories from the rejected pool per step. Same shape as
        # sample_trajectories. Empty list entry when no filter or all samples
        # accepted. Used by the denoising movie to render rejected samples
        # in a distinct colour.
        "rejected_sample_trajectories": [],
    }

    def _rollout_actions(u_norm):
        u_phys = u_norm * action_scale + action_center
        return rollout_single(eta0, nu0, u_phys)

    # Snapshot initial
    traj0 = np.array(_rollout_actions(Ybar_c))
    diag["mean_trajectories"].append(traj0)

    for i_rev in range(Ndiffuse - 1, 0, -1):
        i = i_rev
        Yi = Ybar_c * jnp.sqrt(alphas_bar[i])

        rng, rng_s = jax.random.split(rng)
        eps = jax.random.normal(rng_s, (Nsample, H, 3), dtype=jnp.float32)
        Y0s = eps * sigmas[i] + Ybar_c
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # ── Reject sampling (filter-once, no replacement) ────────────────────
        _n_valid = Nsample
        _valid_mask = None
        _trajs_pre_filter = None
        _Y0s_pre_filter = None
        if sample_filter_fn is not None:
            _u_check = Y0s * action_scale[None] + action_center[None]
            _trajs_pre_filter = np.array(rollout_batch(eta0, nu0, _u_check))
            _valid = np.array(sample_filter_fn(np.array(Y0s), _trajs_pre_filter))
            _n_valid = int(_valid.sum())
            _valid_mask = _valid
            _Y0s_pre_filter = Y0s
            if _n_valid > 0:
                Y0s = Y0s[_valid]   # keep only valid samples (no resample)
        # ─────────────────────────────────────────────────────────────────────

        if sample_filter_fn is not None and _n_valid == 0:
            # Indicator-function weighting: all weights are zero, so no sample
            # contributes to the mean. Keep Ybar_c unchanged this step rather
            # than dragging it toward the (constraint-violating) full sample
            # pool. Build placeholder diagnostics over the pre-filter pool.
            costs = cost_batch(_Y0s_pre_filter)
            w = jnp.zeros((_Y0s_pre_filter.shape[0],), dtype=jnp.float32)
            Ybar_new = Ybar_c
        else:
            costs = cost_batch(Y0s)
            rews = -costs
            # MBD paper: w_k = softmax(rews / temperature).  Max-subtract only.
            logp0 = (rews - rews.max()) / temperature
            w = jax.nn.softmax(logp0)

            Ybar_new = jnp.einsum("n,nij->ij", w, Y0s)

        score = (
            1.0 / (1.0 - alphas_bar[i])
            * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar_new)
        )
        Yim1 = (
            1.0 / jnp.sqrt(alphas[i])
            * (Yi + (1.0 - alphas_bar[i]) * score)
        )
        i_prev = max(i - 1, 0)
        Ybar_c = Yim1 / jnp.sqrt(alphas_bar[i_prev])
        Ybar_c = jnp.clip(Ybar_c, -1.0, 1.0)

        costs_np = np.array(costs)
        w_np = np.array(w)
        diag["sample_costs"].append(costs_np)
        diag["cost_means"].append(float(costs_np.mean()))
        diag["cost_mins"].append(float(costs_np.min()))
        diag["weights"].append(w_np)
        diag["n_total_sampled"].append(Nsample)
        diag["accept_rates"].append(_n_valid / max(Nsample, 1))
        # Cost of the mean action sequence itself (not any sample)
        diag["cost_of_mean"].append(float(cost_fn(Ybar_c)))
        # Track global best (and second-best from the same step) across all steps
        step_sort_idx = np.argsort(costs_np)
        step_best_idx = int(step_sort_idx[0])
        step_best_cost = float(costs_np[step_best_idx])
        if step_best_cost < diag["best_sample_cost"]:
            diag["best_sample_cost"] = step_best_cost
            diag["best_sample_actions"] = np.array(Y0s[step_best_idx])
            if len(step_sort_idx) > 1:
                step_2nd_idx = int(step_sort_idx[1])
                diag["second_best_sample_cost"] = float(costs_np[step_2nd_idx])
                diag["second_best_sample_actions"] = np.array(Y0s[step_2nd_idx])

        # Roll out mean trajectory
        mean_traj = np.array(_rollout_actions(Ybar_c))
        diag["mean_trajectories"].append(mean_traj)

        # Store top-N accepted and top-N rejected trajectories separately.
        # The "accepted" pool is exactly the set the filter approved
        # (_valid_mask True); the "rejected" pool is its complement. We do NOT
        # use the post-fallback Y0s for "accepted" — when _n_valid == 0 the
        # softmax falls back to the full set to avoid NaN, but every sample
        # was rejected so the accepted pool is empty.
        if _valid_mask is not None and _trajs_pre_filter is not None and _Y0s_pre_filter is not None:
            # Cost on every pre-filter sample (cheap: rollouts already cached)
            _all_costs = np.array(cost_batch(_Y0s_pre_filter))

            _acc_idx = np.where(_valid_mask)[0]
            _rej_idx = np.where(~_valid_mask)[0]

            if len(_acc_idx) > 0:
                _acc_top = _acc_idx[np.argsort(_all_costs[_acc_idx])[:n_top_samples]]
                _acc_trajs = _trajs_pre_filter[_acc_top]
            else:
                _acc_trajs = np.zeros((0, H + 1, 3), dtype=np.float32)

            if len(_rej_idx) > 0:
                _rej_top = _rej_idx[np.argsort(_all_costs[_rej_idx])[:n_top_samples]]
                _rej_trajs = _trajs_pre_filter[_rej_top]
            else:
                _rej_trajs = np.zeros((0, H + 1, 3), dtype=np.float32)

            diag["sample_trajectories"].append(_acc_trajs)
            diag["rejected_sample_trajectories"].append(_rej_trajs)
        else:
            # No filter: store top-N by post-softmax cost as before
            _n_avail = len(costs_np)
            n_store = min(n_top_samples, _n_avail)
            sort_idx = np.argsort(costs_np)[:n_store]
            store_actions = Y0s[sort_idx]
            store_phys = store_actions * action_scale[None, :] + action_center[None, :]
            store_trajs = np.array(rollout_batch(eta0, nu0, store_phys))
            diag["sample_trajectories"].append(store_trajs)
            diag["rejected_sample_trajectories"].append(
                np.zeros((0, H + 1, 3), dtype=np.float32)
            )

    diag["Ybar_final"] = Ybar_c
    return diag


def run_eb_mbd_with_diagnostics(
    cost_fn,
    constraint_fn,
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    H: int,
    Nsample: int,
    Ndiffuse: int,
    temperature: float,
    beta0: float,
    betaT: float,
    seed: int,
    eb_c_max: float,
    eb_mu: float,
    eb_kappa: float = 1.0,
    Ybar_init: Optional[jnp.ndarray] = None,
    init_surge_bias: float = 0.7,
    n_top_samples: int = 10,
    oob_constraint_fn=None,
    eb_c_max_oob: float = 0.0,
    # When None, the OOB barrier reuses ``eb_kappa``.  Set this when you want
    # the OOB schedule to clamp to zero on a different time-scale than the
    # goal/main barrier — typically with a negative ``eb_c_max_oob`` (safety
    # margin) and a steeper κ_oob so the margin disappears in the first few
    # diffusion steps while the goal barrier keeps relaxing slowly.
    eb_kappa_oob: Optional[float] = None,
    ice_constraint_fn=None,
    eb_c_max_ice: float = 0.0,
    # ── Optional AL-ice column (hybrid EB+AL): when al_ice_constraint_fn is
    #    provided, the reward is augmented with a per-sample AL penalty
    #    -(ρ/2) Σ_{t,j} (M[t,j]/ρ + g_ice[t,j])₊² and the multipliers M are
    #    updated by dual ascent on the mean trajectory each diffusion step.
    al_ice_constraint_fn=None,
    al_rho_ice: float = 0.0,
    al_M_max_ice: float = 1.0e6,
    # Optional hard reject filter (e.g. OOB).  filter_fn(Y0s_norm, trajs) →
    # bool_mask of shape (Nsample,) where True ⇒ valid (kept) and False ⇒
    # rejected (zero softmax weight).  When all samples are rejected at a
    # step, Ybar_c is left unchanged.
    sample_filter_fn=None,
) -> dict:
    """
    Emerging-Barrier MBD reverse diffusion with per-step diagnostics.

    Implements Algorithm 1 from "EB-MBD: Emerging-Barrier Model-Based
    Diffusion for Safe Trajectory Optimization in Highly Constrained
    Environments" (Mishra & Manchester, 2026).

    The effective reward uses **additive** barrier terms — one per
    constraint type — so each constraint gets an independent c_max::

        r(x,s) = −J(x)
               + μ·log(max(g_goal(x) + c_s,     ε))   ← goal / rects
               + μ·log(max(g_oob(x)  + c_s_oob, ε))   ← domain walls

    Using separate c_max values prevents the large goal-infeasibility
    slack (c_max~1500 m) from also permitting far-OOB exploration that
    bypasses high-cost costmap regions.  Set eb_c_max_oob=0 for a strict
    domain-wall constraint from step 0.

    Parameters
    ----------
    cost_fn : callable
        Base cost J(u_norm) → scalar.  Should NOT include big-M OOB or
        goal-line penalties (pass oob_penalty=0 to make_cost_fn and
        omit make_goal_line_cost_fn).  The barrier enforces those constraints.
    constraint_fn : callable
        g_goal(u_norm) → scalar.  Encodes goal-reaching (and optionally
        hard obstacle) constraints.  Build with make_goal_constraint_fn or
        make_combined_constraint_fn.  Uses eb_c_max.
    oob_constraint_fn : callable or None
        g_oob(u_norm) → scalar.  Encodes domain-wall OOB constraints.
        Build with make_oob_constraint_fn.  Uses eb_c_max_oob.
        When None, OOB is folded into constraint_fn (legacy behaviour).
    eb_c_max : float
        Barrier offset for the goal/combined constraint at the start of
        reverse diffusion.  Should be ≥ max expected goal distance (m).
    eb_c_max_oob : float
        Barrier offset for the OOB constraint.  Set to 0 to enforce
        domain walls strictly from step 0 (recommended when the costmap
        contains soft high-cost regions that trajectories could bypass
        by going OOB).
    eb_mu : float
        Barrier hardness in cost units.  Must be on the same scale as J(x)
        for the barrier to influence the softmax weights.  A good starting
        point: eb_mu ≈ 0.1 × E[J(x)] / log(eb_c_max).
    eb_kappa : float
        Shape of the barrier schedule.  κ = 1 → linear;  κ > 1 → slow
        tightening at first, fast at the end;  κ < 1 → fast then slow.

    Returns
    -------
    dict — same structure as run_mbd_with_diagnostics, plus:
        ``barrier_offsets``  : list of floats  — c_s at each step
        ``constraint_values``: list of (Nsample,) arrays — g(x) per step
        ``alive_fracs``      : list of floats  — fraction of samples with
                               g(x) + c_s > 0 at each step
    """
    if Ybar_init is not None:
        Ybar_c = jnp.array(Ybar_init, dtype=jnp.float32)
        Ybar_c = jnp.clip(Ybar_c, -1.0, 1.0)
    else:
        Ybar_c = jnp.zeros((H, 3))
        Ybar_c = Ybar_c.at[:, 0].set(init_surge_bias)

    rng = jax.random.PRNGKey(seed)
    betas = jnp.linspace(beta0, betaT, Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1.0 - alphas_bar)

    cost_batch       = jax.jit(jax.vmap(cost_fn))
    constraint_batch = jax.jit(jax.vmap(constraint_fn))
    oob_constraint_batch = (
        jax.jit(jax.vmap(oob_constraint_fn)) if oob_constraint_fn is not None else None
    )
    ice_constraint_batch = (
        jax.jit(jax.vmap(ice_constraint_fn)) if ice_constraint_fn is not None else None
    )
    # Hybrid EB+AL: ice as Lagrange multipliers. al_ice_constraint_fn returns
    # (H, n_ice_cols) per sample (same shape as make_rect_obstacles_al_fn).
    al_ice_constraint_batch = (
        jax.jit(jax.vmap(al_ice_constraint_fn))
        if al_ice_constraint_fn is not None else None
    )
    use_al_ice = al_ice_constraint_batch is not None
    if use_al_ice:
        _al_g_probe = al_ice_constraint_fn(jnp.zeros((H, 3), dtype=jnp.float32))
        _n_al_ice_cols = int(_al_g_probe.shape[1])
        M_ice = jnp.zeros((H, _n_al_ice_cols), dtype=jnp.float32)
    else:
        _n_al_ice_cols = 0
        M_ice = None
    rollout_batch    = jax.jit(jax.vmap(rollout_fn, in_axes=(None, None, 0)))
    rollout_single   = jax.jit(rollout_fn)

    diag = {
        "mean_trajectories":         [],
        "sample_trajectories":       [],
        "sample_sort_indices":       [],   # indices into original sample order used when storing trajectories
        "sample_costs":              [],
        "cost_means":                [],
        "cost_mins":                 [],
        "reward_maxes":              [],   # per-step max reward (EB-MBD only)
        "cost_of_mean":              [],
        "weights":                   [],
        "Ybar_final":                None,
        "sample_sort_by_reward":     False,   # EB-MBD stores samples sorted by min cost (same as standard MBD)
        "best_sample_actions":       None,
        "best_sample_cost":          float("-inf"),   # tracks best effective reward (higher = better)
        "second_best_sample_actions": None,
        "second_best_sample_cost":   float("-inf"),
        # EB-MBD: best sample by BASE cost (for consistent gold trajectory display)
        "best_base_cost_actions":    None,
        "best_base_cost_trajectory": None,
        "best_base_cost":            float("inf"),
        "accept_rates":              [],
        "n_total_sampled":           [],
        # EB-MBD extras
        "barrier_offsets":           [],   # c_s at each step (main constraint)
        "constraint_values":         [],   # g(x) per sample per step (main)
        "alive_fracs":               [],   # fraction with g+c_s > 0 (main)
        # Per-constraint schedules / live samples — populated when the matching
        # *_constraint_fn argument is set.  Used by the toy script to plot
        # OOB vs Gate barrier evolution side by side.
        "barrier_offsets_oob":       [],
        "barrier_offsets_ice":       [],
        "alive_fracs_oob":           [],
        "alive_fracs_ice":           [],
        # Hybrid EB+AL: AL-ice diagnostics
        "al_ice_multiplier_means":      [],
        "al_ice_multiplier_maxes":      [],
        "al_ice_constraint_violations": [],
    }

    def _rollout_actions(u_norm):
        u_phys = u_norm * action_scale + action_center
        return rollout_single(eta0, nu0, u_phys)

    # Snapshot initial mean trajectory
    traj0 = np.array(_rollout_actions(Ybar_c))
    diag["mean_trajectories"].append(traj0)

    S = float(Ndiffuse - 1)  # total diffusion steps (for barrier schedule)
    # OOB uses its own κ when provided so its safety-margin schedule can clamp
    # to zero independently of the (slower) goal/main barrier schedule.
    _kappa_oob = float(eb_kappa_oob) if eb_kappa_oob is not None else float(eb_kappa)

    for i_rev in range(Ndiffuse - 1, 0, -1):
        i = i_rev

        # ── Barrier offset for this step ──────────────────────────────────
        # c_s = c_max × (s/S)^κ: relaxed at s=S, tight at s=0
        s_frac = float(i) / S           # 1.0 at start, → 0 at end
        c_s     = float(eb_c_max)     * (s_frac ** float(eb_kappa))
        c_s_oob = float(eb_c_max_oob) * (s_frac ** _kappa_oob)
        c_s_ice = float(eb_c_max_ice) * (s_frac ** float(eb_kappa))

        Yi = Ybar_c * jnp.sqrt(alphas_bar[i])

        rng, rng_s = jax.random.split(rng)
        eps = jax.random.normal(rng_s, (Nsample, H, 3), dtype=jnp.float32)
        Y0s = eps * sigmas[i] + Ybar_c
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # ── Compute costs and constraints ─────────────────────────────────
        costs   = cost_batch(Y0s)             # (Nsample,) — J(x)
        g_vals  = constraint_batch(Y0s)       # (Nsample,) — g_goal in metres

        # ── Barrier term(s): additive per constraint type ─────────────────
        # Goal/combined constraint with large c_max (allows infeasible starts)
        barrier = jnp.float32(eb_mu) * jnp.log(
            jnp.maximum(g_vals + jnp.float32(c_s), jnp.float32(1e-6))
        )
        # OOB constraint: ONE-SIDED barrier. In-bounds trajectories contribute 0
        # so they add no variance to the reward signal. OOB trajectories are
        # immediately and severely penalized (mu*log(1e-6)) to preserve a clean
        # goal gradient after normalization. c_s_oob=0 enforces this strictly.
        if oob_constraint_batch is not None:
            g_oob_vals = oob_constraint_batch(Y0s)
            effective_g_oob = g_oob_vals + jnp.float32(c_s_oob)
            oob_barrier = jnp.where(
                effective_g_oob >= 0.0,
                jnp.float32(0.0),
                jnp.float32(eb_mu) * jnp.log(
                    jnp.maximum(effective_g_oob, jnp.float32(1e-6))
                ),
            )
            barrier = barrier + oob_barrier
        # Ice constraint: SYMMETRIC log barrier with its own c_s schedule.
        # g_ice = ice_threshold - swath_integral (typically threshold = 0).
        # Early (s_frac=1) the budget c_s_ice ≈ eb_c_max_ice allows paths
        # through ice; late (s_frac→0) c_s_ice → 0 so the barrier tightens
        # toward strict ice avoidance. Independent from c_s_oob/c_s.
        if ice_constraint_batch is not None:
            g_ice_vals = ice_constraint_batch(Y0s)
            ice_barrier = jnp.float32(eb_mu) * jnp.log(
                jnp.maximum(g_ice_vals + jnp.float32(c_s_ice),
                            jnp.float32(1e-6))
            )
            barrier = barrier + ice_barrier

        # Hybrid EB+AL: per-sample AL-ice penalty using current M, ρ.
        # penalty_k = (ρ/2) Σ_{t,j} (M[t,j]/ρ + g_ice_AL[k,t,j])₊²
        # Subtracting from reward is equivalent to adding to cost in the
        # softmax that selects samples.
        if use_al_ice:
            g_al_ice = al_ice_constraint_batch(Y0s)         # (N, H, n_g)
            _rho = jnp.float32(al_rho_ice)
            _arg = M_ice[None] / jnp.maximum(_rho, 1e-12) + g_al_ice
            _arg_pos = jnp.maximum(_arg, 0.0)
            al_ice_penalty = 0.5 * _rho * jnp.sum(_arg_pos ** 2, axis=(1, 2))
        else:
            al_ice_penalty = jnp.float32(0.0)

        rews = -costs + barrier - al_ice_penalty

        # Hard reject filter (e.g. OOB): rolled-out trajectories that fail the
        # filter get -inf reward → zero softmax weight, identical to dropping
        # them from the pool but vectorised.  When all samples fail, we keep
        # Ybar_c unchanged at the bottom of the step.
        if sample_filter_fn is not None:
            _u_check = Y0s * action_scale[None] + action_center[None]
            _trajs_pre_filter = rollout_batch(eta0, nu0, _u_check)
            _valid_np = np.array(
                sample_filter_fn(np.array(Y0s), np.array(_trajs_pre_filter)),
                dtype=bool,
            )
            _n_valid = int(_valid_np.sum())
            _valid_jax = jnp.asarray(_valid_np, dtype=jnp.float32)
            rews = rews + (1.0 - _valid_jax) * jnp.float32(-1.0e30)
        else:
            _n_valid = int(Nsample)
        # EB-MBD paper: w_k = softmax((-J + barrier) / temperature).
        # Max-subtract only — does not affect softmax output, just keeps exp() stable.
        logp0 = (rews - rews.max()) / jnp.float32(temperature)
        w     = jax.nn.softmax(logp0)

        Ybar_new = jnp.einsum("n,nij->ij", w, Y0s)
        # If the reject filter eliminated every sample, the softmax above is
        # over a uniform -1e30 reward field and Ybar_new is the average of
        # rejected (infeasible) samples.  Setting Ybar_new = Ybar_c gives a
        # zero score → Ybar_c is unchanged that step.
        if sample_filter_fn is not None and _n_valid == 0:
            Ybar_new = Ybar_c

        score = (
            1.0 / (1.0 - alphas_bar[i])
            * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar_new)
        )
        Yim1 = (
            1.0 / jnp.sqrt(alphas[i])
            * (Yi + (1.0 - alphas_bar[i]) * score)
        )
        i_prev = max(i - 1, 0)
        Ybar_c = Yim1 / jnp.sqrt(alphas_bar[i_prev])
        Ybar_c = jnp.clip(Ybar_c, -1.0, 1.0)

        # ── Record diagnostics ────────────────────────────────────────────
        costs_np  = np.array(costs)
        rews_np   = np.array(rews)
        g_vals_np = np.array(g_vals)
        w_np      = np.array(w)

        alive_frac = float(np.mean((g_vals_np + c_s) > 0))

        diag["sample_costs"].append(costs_np)
        diag["cost_means"].append(float(costs_np.mean()))
        diag["cost_mins"].append(float(costs_np.min()))
        diag["reward_maxes"].append(float(rews_np.max()))
        diag["weights"].append(w_np)
        diag["n_total_sampled"].append(Nsample)
        diag["accept_rates"].append(_n_valid / max(Nsample, 1))
        diag["cost_of_mean"].append(float(cost_fn(Ybar_c)))
        diag["barrier_offsets"].append(c_s)
        diag["constraint_values"].append(g_vals_np)
        diag["alive_fracs"].append(alive_frac)
        # Per-constraint schedule + alive fraction (only populated when the
        # matching *_constraint_fn exists).  These mirror the AL per-column
        # diagnostics so the toy script can plot OOB vs Gate evolution.
        if oob_constraint_batch is not None:
            _g_oob_np = np.array(g_oob_vals)
            diag["barrier_offsets_oob"].append(c_s_oob)
            diag["alive_fracs_oob"].append(float(np.mean((_g_oob_np + c_s_oob) > 0)))
        if ice_constraint_batch is not None:
            _g_ice_np = np.array(g_ice_vals)
            diag["barrier_offsets_ice"].append(c_s_ice)
            diag["alive_fracs_ice"].append(float(np.mean((_g_ice_np + c_s_ice) > 0)))

        # Hybrid EB+AL: dual ascent on M_ice using the mean trajectory's g.
        # M ← clip( M + ρ·ᾱ_i·g₊ , 0 , M_max ).  ᾱ_i (cumulative product of
        # 1−β) is the same diffusion-weight schedule used by AL-MBD so dual
        # steps are suppressed in the high-noise regime.
        if use_al_ice:
            g_al_mean = al_ice_constraint_fn(Ybar_c)               # (H, n_g)
            alpha_bar_i = alphas_bar[i]
            M_ice = jnp.maximum(
                M_ice + jnp.float32(al_rho_ice) * alpha_bar_i
                * jnp.maximum(g_al_mean, 0.0),
                0.0,
            )
            M_ice = jnp.minimum(M_ice, jnp.float32(al_M_max_ice))
            _M_np = np.array(M_ice)
            _g_np = np.array(g_al_mean)
            diag["al_ice_multiplier_means"].append(float(_M_np.mean()))
            diag["al_ice_multiplier_maxes"].append(float(_M_np.max()))
            diag["al_ice_constraint_violations"].append(
                float(np.maximum(_g_np, 0.0).mean())
            )

        # Track global best (and second-best) by effective reward (not raw cost).
        step_sort_idx  = np.argsort(-rews_np)   # descending reward
        step_best_idx  = int(step_sort_idx[0])
        step_best_rew  = float(rews_np[step_best_idx])
        if step_best_rew > diag["best_sample_cost"]:
            diag["best_sample_cost"]    = step_best_rew
            diag["best_sample_actions"] = np.array(Y0s[step_best_idx])
            if len(step_sort_idx) > 1:
                step_2nd_idx = int(step_sort_idx[1])
                diag["second_best_sample_cost"]    = float(rews_np[step_2nd_idx])
                diag["second_best_sample_actions"] = np.array(Y0s[step_2nd_idx])

        # Roll out mean trajectory
        mean_traj = np.array(_rollout_actions(Ybar_c))
        diag["mean_trajectories"].append(mean_traj)

        # Track best sample by BASE cost AMONG GOAL-REACHING SAMPLES ONLY so the
        # gold trajectory in the denoising movie always shows a feasible path.
        # Non-goal-reaching samples (g < 0) can have low base cost (they don't
        # need to travel the full distance), so filtering is essential.
        goal_mask = g_vals_np >= 0.0
        if goal_mask.any():
            masked_costs = np.where(goal_mask, costs_np, np.inf)
            base_best_idx = int(np.argmin(masked_costs))
        else:
            base_best_idx = int(np.argmin(costs_np))  # fallback: no feasible sample yet
        base_best_cost = float(costs_np[base_best_idx])
        if base_best_cost < diag["best_base_cost"]:
            diag["best_base_cost"]     = base_best_cost
            diag["best_base_cost_actions"] = np.array(Y0s[base_best_idx])
            _bb_phys = Y0s[base_best_idx] * action_scale + action_center
            diag["best_base_cost_trajectory"] = np.array(rollout_single(eta0, nu0, _bb_phys))

        # Roll out top samples sorted by min base cost (best first), same as standard MBD.
        n_store  = min(n_top_samples, Nsample)
        sort_idx = np.argsort(costs_np)[:n_store]
        store_phys  = (Y0s[sort_idx] * action_scale[None, :] + action_center[None, :])
        store_trajs = np.array(rollout_batch(eta0, nu0, store_phys))
        diag["sample_trajectories"].append(store_trajs)
        diag["sample_sort_indices"].append(sort_idx)

    diag["Ybar_final"] = Ybar_c
    return diag


def run_al_mbd_with_diagnostics(
    cost_fn,
    al_constraint_fn,
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    H: int,
    Nsample: int,
    Ndiffuse: int,
    temperature: float,
    beta0: float,
    betaT: float,
    seed: int,
    al_rho: float,
    al_M_max: float,
    Ybar_init: Optional[jnp.ndarray] = None,
    init_surge_bias: float = 0.7,
    n_top_samples: int = 10,
    al_M_init_oob: float = 0.0,
    al_rho_oob: Optional[float] = None,
    al_rho_goal: Optional[float] = None,
    use_zscore_weighting: bool = False,
    zscore_temperature: Optional[float] = None,
    # ── Emerging-barrier schedule on the goal-reach AL column (last col) ──
    # When `al_goal_eb_c_max` is set, the goal-reach column's value gets a
    # step-dependent offset c_s(i) = al_goal_eb_c_max · (i / (Ndiffuse − 1))^κ
    # SUBTRACTED before clipping.  Effect: the constraint floor
    # `y_T ≥ goal_y_line − c_s(i)` retracts from `goal_y_line − c_max` at
    # i = N−1 up to `goal_y_line` at i = 0.  Same algebra as the EB-MBD
    # log-barrier schedule but expressed as an AL threshold shift, so the
    # weighting stays in the AL framework (legacy/z-score softmax — no
    # log-barrier term added to L).
    al_goal_eb_c_max: Optional[float] = None,
    al_goal_eb_kappa: float = 1.0,
) -> dict:
    """
    Augmented Lagrangian MBD (AL-MBD) reverse diffusion with per-step diagnostics.

    Implements Algorithm 1 from the AL-MBD research note.  The hard-indicator
    constraint weight of vanilla MBD is replaced by a smooth AL-derived weight::

        w^AL(Y) ∝ exp(−L(Y, M, ρ) / λ)

    where the trajectory AL cost is::

        L(Y, M, ρ) = J(Y) + (ρ/2) Σ_{t,j} [M_{t,j}/ρ + g_{t,j}(x_t, u_t)]²₊

    Lagrange multipliers M are updated at every diffusion step via dual ascent::

        M_t ← max(0,  M_t + ρ γ_i g_t(x*_t, u*_t))

    with γ_i = ᾱ_i (suppresses dual steps in the high-noise regime).

    Parameters
    ----------
    cost_fn : callable
        Base cost J(u_norm) → scalar.  Should NOT include OOB or goal-line
        big-M penalties; the AL constraint handles those.
    al_constraint_fn : callable
        g(u_norm) → (H, n_g).  g_{t,j} ≤ 0 means constraint j is satisfied
        at timestep t.  Build with make_al_per_step_constraints_fn.
    al_rho : float
        AL penalty weight ρ.  Controls how fast constraints are enforced per
        dual step.  Practical starting point: 0.1–1.0.
    al_M_max : float
        Maximum multiplier value (clipping for numerical stability).

    Returns
    -------
    dict — same keys as run_mbd_with_diagnostics, plus:
        ``multiplier_means``      : list of floats  — mean M across (t, j) per step
        ``multiplier_maxes``      : list of floats  — max M per step
        ``constraint_violations`` : list of floats  — mean max(0, g_t) of mean traj per step
        ``al_penalties``          : list of (Nsample,) arrays — per-sample AL penalty
    """
    if Ybar_init is not None:
        Ybar_c = jnp.array(Ybar_init, dtype=jnp.float32)
        Ybar_c = jnp.clip(Ybar_c, -1.0, 1.0)
    else:
        Ybar_c = jnp.zeros((H, 3))
        Ybar_c = Ybar_c.at[:, 0].set(init_surge_bias)

    rng = jax.random.PRNGKey(seed)
    betas      = jnp.linspace(beta0, betaT, Ndiffuse)
    alphas     = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas     = jnp.sqrt(1.0 - alphas_bar)

    # Detect n_g from a single constraint evaluation
    _g_init = al_constraint_fn(Ybar_c)   # (H, n_g)
    n_g = int(_g_init.shape[1])

    # Multipliers M: (H, n_g), initialised at zero (cold start).
    # Optional warm-start on column 0 (OOB) to make the OOB constraint hard
    # from the first diffusion step: the linear AL term α̅ᵢ·M·g₊ then bites
    # immediately, even at the high-noise classes where the α̅ᵢ schedule is
    # small.  Column convention: build_al_constraint_fn places OOB at col 0
    # and rect/ice constraints at cols 1+ (see jax_mbd_optimize).
    M = jnp.zeros((H, n_g), dtype=jnp.float32)
    if al_M_init_oob > 0.0:
        M = M.at[:, 0].set(jnp.float32(al_M_init_oob))

    # Per-column ρ: OOB column gets al_rho_oob (defaults to al_rho if None),
    # rect/ice columns get al_rho.  Larger al_rho_oob makes the OOB constraint
    # dominate sample selection even in the all-infeasible regime where the
    # softmax is one-hot — without that asymmetry, scaling ρ uniformly only
    # rescales L and preserves sample ordering, so the winner never changes.
    _rho_oob = float(al_rho_oob) if al_rho_oob is not None else float(al_rho)
    rho_vec = jnp.full((n_g,), jnp.float32(al_rho), dtype=jnp.float32).at[0].set(
        jnp.float32(_rho_oob)
    )
    # Goal-reach column (when used) is appended LAST by jax_mbd_optimize.
    # Override its ρ so the goal constraint can be made arbitrarily harder
    # than the per-floe ice constraints.
    if al_rho_goal is not None:
        rho_vec = rho_vec.at[-1].set(jnp.float32(float(al_rho_goal)))

    cost_batch            = jax.jit(jax.vmap(cost_fn))
    al_constraint_batch   = jax.jit(jax.vmap(al_constraint_fn))
    rollout_batch         = jax.jit(jax.vmap(rollout_fn, in_axes=(None, None, 0)))
    rollout_single        = jax.jit(rollout_fn)

    diag = {
        "mean_trajectories":          [],
        "sample_trajectories":        [],
        "sample_sort_indices":        [],   # indices into original sample order used when storing trajectories
        "sample_costs":               [],
        "cost_means":                 [],
        "cost_mins":                  [],
        "cost_of_mean":               [],
        "weights":                    [],
        "Ybar_final":                 None,
        "best_sample_actions":        None,
        "best_sample_cost":           float("-inf"),
        "second_best_sample_actions": None,
        "second_best_sample_cost":    float("-inf"),
        # AL-MBD: best sample by BASE cost (not total AL cost)
        "best_base_cost_actions":     None,
        "best_base_cost_trajectory":  None,
        "best_base_cost":             float("inf"),
        "accept_rates":               [],
        "n_total_sampled":            [],
        # AL-MBD extras
        "multiplier_means":           [],
        "multiplier_maxes":           [],
        "constraint_violations":      [],
        "al_penalties":               [],
        # Per-constraint-column stats (n_g entries per step).  Column 0 is the
        # OOB constraint built by make_al_per_step_constraints_fn; columns 1+
        # are rect/ice constraints appended by make_rect_obstacles_al_fn.
        "multiplier_means_per_col":      [],
        "multiplier_maxes_per_col":      [],
        "constraint_violations_per_col": [],
        # Emerging-barrier schedule on the goal-reach AL column (only
        # populated when al_goal_eb_c_max is not None).
        "goal_eb_barrier_offsets":       [],
    }

    # ── Emerging-barrier schedule for the goal-reach AL column ────────────
    # When enabled, the LAST column of g (the goal-reach column produced by
    # make_goal_reach_al_fn) gets a step-dependent offset subtracted before
    # clipping into g_+.  Effect: `effective_g_goal = g_goal − c_s(i)`,
    # where c_s starts at al_goal_eb_c_max and decays to 0 under the κ
    # schedule.  At step i = Ndiffuse − 1 the floor is goal_y_line − c_max
    # (typically below the start so the constraint is trivially satisfied);
    # at i = 0 the floor is goal_y_line (full strictness).
    _use_eb_goal = al_goal_eb_c_max is not None
    if _use_eb_goal:
        _eb_goal_S = float(max(Ndiffuse - 1, 1))
        _eb_goal_offset_vec = (
            jnp.zeros((n_g,), dtype=jnp.float32).at[-1].set(1.0)
        )  # selector for the LAST column

    def _rollout_actions(u_norm):
        u_phys = u_norm * action_scale + action_center
        return rollout_single(eta0, nu0, u_phys)

    diag["mean_trajectories"].append(np.array(_rollout_actions(Ybar_c)))

    for i_rev in range(Ndiffuse - 1, 0, -1):
        i = i_rev
        i_prev = i - 1
        alpha_bar_m = alphas_bar[i_prev]   # ᾱ_{i-1}, one step cleaner
        alpha_i_val = alphas[i]            # α_i = ᾱ_i / ᾱ_{i-1}

        Yi = Ybar_c * jnp.sqrt(alphas_bar[i])

        rng, rng_s = jax.random.split(rng)
        eps = jax.random.normal(rng_s, (Nsample, H, 3), dtype=jnp.float32)
        # Reference sampling (al_mbd_zscore_reference.md §"Diffusion reverse step"):
        #   mean = Y_i / sqrt(ᾱ_{i-1}),  std = sqrt(1/ᾱ_{i-1} − 1)
        # With carry Ybar_c = Y_i / sqrt(ᾱ_i): mean = Ybar_c · sqrt(α_i).
        # Previous code used `Ybar_c + sqrt(1−ᾱ_i)·eps`, which under-explores at
        # high noise (std too small by factor 1/sqrt(ᾱ_i)).
        std_samp  = jnp.sqrt(1.0 / alpha_bar_m - 1.0)
        mean_samp = Ybar_c * jnp.sqrt(alpha_i_val)
        Y0s = mean_samp + std_samp * eps
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # Base cost and per-step constraint values
        costs  = cost_batch(Y0s)               # (Nsample,)
        g_vals = al_constraint_batch(Y0s)      # (Nsample, H, n_g)

        # Emerging-barrier offset for the goal-reach AL column (last col).
        # `effective_g_goal = g_goal − c_s(i)` makes the floor
        # `y_T ≥ goal_y_line − c_s(i)` retract from `goal_y_line − c_max`
        # at i = N−1 up to `goal_y_line` at i = 0.  Applied via a vector
        # whose only nonzero entry is on column −1, then broadcast over
        # (Nsample, H, n_g).
        if _use_eb_goal:
            _c_s_goal = float(al_goal_eb_c_max) * (
                (float(i) / _eb_goal_S) ** float(al_goal_eb_kappa)
            )
            g_vals = g_vals - (jnp.float32(_c_s_goal) * _eb_goal_offset_vec)[None, None, :]

        # AL penalty: scheduled (linear + quadratic) form.
        #
        # Two improvements over the standard quadratic AL:
        #   1. Use max(g, 0) not max(M/rho + g, 0) — the M/rho offset creates a
        #      spurious penalty zone for FEASIBLE trajectories (g ∈ [-M/rho, 0]),
        #      pushing the trajectory mean away from constraint boundaries even when
        #      it's already inside, which causes it to miss narrow gates.
        #   2. Scale by alphas_bar[i] (same as dual-ascent gamma_i) — near-0 at
        #      early diffusion steps (high noise) so samples are weighted by base cost
        #      (ice + goal), allowing lateral exploration toward low-cost corridors;
        #      grows to ~0.8 at late steps to enforce constraints.
        #
        # Penalty form for g > 0: schedule * (M*g + (rho/2)*g^2)
        # This equals the linearized AL: the first-order Lagrangian term M*g plus
        # the quadratic penalty (rho/2)*g^2.  Combined, they have the same gradient
        # structure as standard AL but without the feasibility-zone penalization.
        penalty_schedule = float(alphas_bar[i])
        g_pos   = jnp.maximum(g_vals, jnp.float32(0.0))               # (Nsample, H, n_g)
        al_pen  = penalty_schedule * (
            M[None, :, :] * g_pos
            + (rho_vec[None, None, :] / 2.0) * g_pos ** 2
        ).sum(axis=(-2, -1))                                           # (Nsample,)

        L = costs + al_pen   # (Nsample,) total AL cost

        if use_zscore_weighting:
            # AL-MBD z-score variant with finite-only stats + all-infeasible
            # NaN guard, matching al_mbd_zscore_reference.md §"Z-score softmax
            # weighting (with inf-rejection masking)".  For pure AL-MBD L is
            # always finite, so finite_mask is all-True and this collapses to
            # the original MBD softmax — but the guard is kept so that mixing
            # in a hard reject sentinel (+inf) wouldn't collapse the weights
            # over feasible samples.
            _lambda = jnp.float32(
                zscore_temperature if zscore_temperature is not None else temperature
            )
            finite_mask = jnp.isfinite(L)
            L_safe   = jnp.nan_to_num(L, posinf=1e9, neginf=-1e9)
            n_finite = jnp.sum(finite_mask.astype(jnp.float32))
            safe_L   = jnp.where(finite_mask, L_safe, jnp.float32(0.0))
            L_mean   = safe_L.sum() / jnp.maximum(n_finite, jnp.float32(1.0))
            diff_sq  = jnp.where(finite_mask, (L_safe - L_mean) ** 2, jnp.float32(0.0))
            L_std    = jnp.sqrt(diff_sq.sum() / jnp.maximum(n_finite, jnp.float32(1.0)))
            L_std    = jnp.where(L_std < jnp.float32(1e-4), jnp.float32(1.0), L_std)
            scores   = -(L_safe - L_mean) / L_std / _lambda
            scores   = jnp.where(finite_mask, scores, jnp.float32(-jnp.inf))
            w        = jax.nn.softmax(scores)
            # If every sample is infeasible (scores all -inf), softmax → NaN.
            # Fall back to uniform weights to keep diffusion moving.
            w = jax.lax.cond(
                jnp.all(jnp.isnan(w)),
                lambda _: jnp.ones_like(w) / jnp.float32(w.shape[0]),
                lambda x: x,
                w,
            )
        else:
            rews     = -L
            # AL-MBD paper: w_k = softmax(-L_k / temperature).  Max-subtract only.
            logp0    = (rews - rews.max()) / jnp.float32(temperature)
            w        = jax.nn.softmax(logp0)

        Ybar_new = jnp.einsum("n,nij->ij", w, Y0s)

        score = (
            1.0 / (1.0 - alphas_bar[i])
            * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar_new)
        )
        Yim1 = (
            1.0 / jnp.sqrt(alphas[i])
            * (Yi + (1.0 - alphas_bar[i]) * score)
        )
        i_prev = max(i - 1, 0)
        Ybar_c = Yim1 / jnp.sqrt(alphas_bar[i_prev])
        Ybar_c = jnp.clip(Ybar_c, -1.0, 1.0)

        # Dual ascent on M using current best estimate Y^{(i-1)} = Ybar_c
        # Step size γ_i = ᾱ_i (suppresses updates in high-noise regime).
        # Per-column ρ flows into the dual step too, so the OOB column grows
        # (and decays) at its own rate.
        g_cur   = al_constraint_fn(Ybar_c)   # (H, n_g)
        # Same emerging-barrier offset as the penalty branch above, so the
        # multiplier on the goal column accumulates against the SAME
        # effective constraint that drove sample weighting.
        if _use_eb_goal:
            g_cur = g_cur - (jnp.float32(_c_s_goal) * _eb_goal_offset_vec)[None, :]
        gamma_i = float(alphas_bar[i])
        M = jnp.maximum(
            jnp.float32(0.0),
            M + (rho_vec[None, :] * jnp.float32(gamma_i)) * g_cur,
        )
        M = jnp.minimum(M, jnp.float32(al_M_max))

        # Diagnostics
        costs_np  = np.array(costs)
        L_np      = np.array(L)
        al_pen_np = np.array(al_pen)
        g_cur_np  = np.array(g_cur)
        w_np      = np.array(w)
        M_np      = np.array(M)

        step_sort_idx = np.argsort(L_np)          # ascending total AL cost
        step_best_idx = int(step_sort_idx[0])
        step_best_rew = -float(L_np[step_best_idx])

        if step_best_rew > diag["best_sample_cost"]:
            diag["best_sample_cost"]    = step_best_rew
            diag["best_sample_actions"] = np.array(Y0s[step_best_idx])
            if len(step_sort_idx) > 1:
                step_2nd_idx = int(step_sort_idx[1])
                diag["second_best_sample_cost"]    = -float(L_np[step_2nd_idx])
                diag["second_best_sample_actions"] = np.array(Y0s[step_2nd_idx])

        # Track best sample by BASE cost (excludes AL penalty) so the
        # diagnostic gold trajectory matches the reported base cost exactly.
        base_best_idx = int(np.argmin(costs_np))
        base_best_cost = float(costs_np[base_best_idx])
        if base_best_cost < diag["best_base_cost"]:
            diag["best_base_cost"]     = base_best_cost
            diag["best_base_cost_actions"] = np.array(Y0s[base_best_idx])
            _bb_phys = Y0s[base_best_idx] * action_scale + action_center
            diag["best_base_cost_trajectory"] = np.array(rollout_single(eta0, nu0, _bb_phys))

        diag["sample_costs"].append(costs_np)
        diag["cost_means"].append(float(costs_np.mean()))
        diag["cost_mins"].append(float(costs_np.min()))
        diag["weights"].append(w_np)
        diag["n_total_sampled"].append(Nsample)
        diag["accept_rates"].append(1.0)
        diag["cost_of_mean"].append(float(cost_fn(Ybar_c)))
        diag["multiplier_means"].append(float(M_np.mean()))
        diag["multiplier_maxes"].append(float(M_np.max()))
        diag["constraint_violations"].append(float(np.maximum(g_cur_np, 0.0).mean()))
        diag["al_penalties"].append(al_pen_np)
        # Per-column: average/max across H timesteps, one value per constraint col.
        diag["multiplier_means_per_col"].append(M_np.mean(axis=0))
        diag["multiplier_maxes_per_col"].append(M_np.max(axis=0))
        diag["constraint_violations_per_col"].append(
            np.maximum(g_cur_np, 0.0).mean(axis=0)
        )
        if _use_eb_goal:
            diag["goal_eb_barrier_offsets"].append(_c_s_goal)

        diag["mean_trajectories"].append(np.array(_rollout_actions(Ybar_c)))

        n_store    = min(n_top_samples, Nsample)
        sort_idx   = np.argsort(L_np)[:n_store]
        store_phys = Y0s[sort_idx] * action_scale[None, :] + action_center[None, :]
        diag["sample_trajectories"].append(np.array(rollout_batch(eta0, nu0, store_phys)))
        diag["sample_sort_indices"].append(sort_idx)  # maps stored traj index → original sample index

    diag["Ybar_final"] = Ybar_c
    return diag


# ── Replan visualization helpers ─────────────────────────────────────────

def _sample_final_step(
    Ybar_final: jnp.ndarray,
    cost_fn,
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    H: int,
    Nsample: int,
    beta0: float,
    betaT: float,
    Ndiffuse: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample Nsample trajectories around Ybar_final using the final diffusion step's noise level.

    This is a single cheap batch forward pass (not a full re-run of MBD).
    Returns:
        trajs   : (Nsample, H+1, 3) rolled-out trajectories [x, y, psi]
        costs   : (Nsample,) total cost per sample
        Y0s_norm: (Nsample, H, 3) normalised action sequences
    """
    betas_np = np.linspace(float(beta0), float(betaT), Ndiffuse)
    alphas_bar_np = np.cumprod(1.0 - betas_np)
    # Use sigma at step i=1 (last reverse step) — small but non-zero
    sigma = float(np.sqrt(max(1.0 - alphas_bar_np[1], 0.0)))

    rng = jax.random.PRNGKey(seed + 9999)
    eps = jax.random.normal(rng, (Nsample, H, 3), dtype=jnp.float32)
    Y0s = jnp.clip(eps * sigma + Ybar_final[None], -1.0, 1.0)

    cost_batch = jax.jit(jax.vmap(cost_fn))
    costs = np.array(cost_batch(Y0s))

    rollout_batch = jax.jit(jax.vmap(rollout_fn, in_axes=(None, None, 0)))
    u_phys = Y0s * action_scale[None] + action_center[None]
    trajs = np.array(rollout_batch(eta0, nu0, u_phys))

    return trajs, costs, np.array(Y0s)


def _generate_denoising_movie_from_diagnostics(
    diag: dict,
    cost_map,
    world_polygons,
    thicknesses,
    goal_y_line,
    goal,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    replan_idx: int,
    output_path: str,
    ship_length: Optional[float] = None,
    ship_beam: Optional[float] = None,
    fps: int = 10,
) -> None:
    """
    Generate a denoising-evolution MP4 for one replan step from diagnostics data.

    Each animation frame corresponds to one diffusion step and shows the sample
    trajectory distribution alongside the current mean trajectory.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection, LineCollection
    from matplotlib import animation
    import os

    mean_trajs = diag["mean_trajectories"]     # list of (H+1, 3), length Ndiffuse
    sample_trajs = diag["sample_trajectories"] # list of (Nsample, H+1, 3), length Ndiffuse-1
    cost_mins = diag.get("cost_mins", [])
    cost_means = diag.get("cost_means", [])

    n_frames = len(mean_trajs)
    n_samples = len(sample_trajs[0]) if sample_trajs else 0

    print(
        f"  [replan {replan_idx:04d}] Generating denoising movie: "
        f"{n_frames} steps, {n_samples} samples -> {output_path}",
        flush=True,
    )

    # Start pose: first waypoint of the first mean trajectory (== eta0).
    _start_pose = np.asarray(mean_trajs[0])[0, :3]
    _start_xy = _start_pose[:2]
    _start_psi = float(_start_pose[2])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # ── Static background ──────────────────────────────────────────────
    # Ice polygons only — no costmap heatmap underlay. Initial-plan and replan
    # use different costmap builders (polygon vs disk-rasterised), so the
    # underlay would render differently each call. The polygons themselves are
    # builder-independent, so they alone keep the style consistent.
    if world_polygons is not None and thicknesses is not None and len(world_polygons) > 0:
        patches = [MplPolygon(poly, closed=True) for poly in world_polygons]
        pc = PatchCollection(patches, cmap="Blues", edgecolor="#555555",
                              linewidth=0.3, alpha=0.6)
        t_arr = np.asarray(thicknesses, dtype=float)
        pc.set_array(t_arr)
        pc.set_clim(float(t_arr.min()), float(t_arr.max()))
        ax.add_collection(pc)

    if goal_y_line is not None:
        ax.axhline(goal_y_line, color="lime", ls="--", lw=2, zorder=4,
                   label=f"Goal y={goal_y_line:.0f}")
    else:
        ax.plot(goal[0], goal[1], "mx", ms=14, mew=3, zorder=6, label="Goal")
    if ship_length is not None and ship_beam is not None:
        from medea_ice_ship.geometry import create_ship_hull_at_position_heading
        _hull = create_ship_hull_at_position_heading(
            ship_length, ship_beam, _start_xy, _start_psi,
        )
        # alpha=1.0: fully opaque so the fuchsia sample trajectories (zorder=4)
        # passing through the hull at the start pose don't bleed through and
        # cause the hull to visually flicker frame-to-frame.
        ax.add_patch(MplPolygon(_hull, closed=True, fc="orange", ec="black",
                                lw=1.5, alpha=1.0, zorder=6))

    pad = 30.0
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    # ── Dynamic artists ────────────────────────────────────────────────
    sample_norm = plt.Normalize(vmin=0, vmax=1)
    sample_lc = LineCollection([], cmap="cool", norm=sample_norm,
                                linewidths=0.6, alpha=0.25, zorder=4)
    ax.add_collection(sample_lc)

    mean_outline, = ax.plot([], [], "w-", lw=4, zorder=5)
    mean_line, = ax.plot([], [], "-", color="tomato", lw=2, zorder=5,
                         label="Mean trajectory")

    title_obj = ax.set_title(f"Replan {replan_idx:04d} — denoising step 0 / {n_frames - 1}")
    ax.legend(loc="upper right", fontsize=8)

    def _update(frame):
        mt = mean_trajs[frame]
        mean_outline.set_data(mt[:, 0], mt[:, 1])
        mean_line.set_data(mt[:, 0], mt[:, 1])

        if frame < len(sample_trajs) and sample_trajs:
            st = sample_trajs[frame]
            segments = [st[k, :, :2] for k in range(len(st))]
            colors = np.linspace(0, 1, len(st))
            sample_lc.set_segments(segments)
            sample_lc.set_array(colors)
        else:
            sample_lc.set_segments([])

        cost_str = ""
        if frame < len(cost_mins):
            cost_str = f"  |  cost_min={cost_mins[frame]:.1f}, cost_mean={cost_means[frame]:.1f}"
        title_obj.set_text(
            f"Replan {replan_idx:04d} — denoising step {frame} / {n_frames - 1}{cost_str}"
        )

    ani = animation.FuncAnimation(fig, _update, frames=n_frames, interval=1000 // fps)

    from medea_ice_ship.visualization import _animation_writer
    writer, output_path = _animation_writer(output_path, fps)
    ani.save(output_path, writer=writer, dpi=120)
    plt.close(fig)


def _save_mbd_replan_viz(
    viz_dir: str,
    replan_idx: int,
    trajs: np.ndarray,
    costs: np.ndarray,
    Y0s_norm: np.ndarray,
    Ybar_final: jnp.ndarray,
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    components_fn,
    cost_map,
    goal_world,
    costmap_weight: float,
    costmap_normalizer: float = 1.0,
    length_weight: float = 1.0,
    goal_weight: float = 0.1,
    smoothness_weight: float = 0.005,
    world_polygons=None,
    thicknesses=None,
    lattice_path=None,
    n_top: int = 5,
) -> None:
    """
    Generate and save three MBD debug outputs for one replan:
      1. Density plot: all Nsample trajectories with low alpha
      2. Top-N plot: top n_top trajectories by total cost, with score labels
      3. Text file: cost decomposition (costmap / length / goal / jerk) for top-N paths

    Axes always match the full costmap extent.  The lattice path (full A* path
    from ship to goal, including the portion beyond the MBD horizon) is drawn
    as a dashed background line for context.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    import os

    os.makedirs(viz_dir, exist_ok=True)

    grid = cost_map.cost_map
    extent = [
        cost_map.x_origin, cost_map.x_origin + cost_map.width,
        cost_map.y_origin, cost_map.y_origin + cost_map.height,
    ]
    vmax = float(np.percentile(grid[grid > 0], 95)) if np.any(grid > 0) else 1.0

    # Full costmap domain — same as the simulation movie
    xlim = (cost_map.x_origin, cost_map.x_origin + cost_map.width)
    ylim = (cost_map.y_origin, cost_map.y_origin + cost_map.height)

    def _draw_background(ax):
        ax.imshow(grid, origin="lower", extent=extent, cmap="YlOrRd",
                  alpha=0.5, vmin=0, vmax=vmax)
        if world_polygons is not None and thicknesses is not None and len(world_polygons) > 0:
            patches = [MplPolygon(poly, closed=True) for poly in world_polygons]
            pc = PatchCollection(patches, cmap="Blues", edgecolor="#555555",
                                 linewidth=0.3, alpha=0.5)
            pc.set_array(np.array(thicknesses))
            pc.set_clim(float(thicknesses.min()), float(thicknesses.max()))
            ax.add_collection(pc)
        # Full lattice path (context for what happens beyond MBD horizon)
        if lattice_path is not None and len(lattice_path) >= 2:
            lp = np.asarray(lattice_path)
            ax.plot(lp[:, 0], lp[:, 1], "--", color="gray", lw=1.2, alpha=0.7,
                    label="Lattice path (full)", zorder=3)

    goal_xy = np.array(goal_world[:2])
    ship_xy = np.array([float(eta0[0]), float(eta0[1])])

    # MBD horizon coverage in metres (for title)
    mean_traj_len = float(np.sum(np.linalg.norm(np.diff(trajs.mean(axis=0)[:, :2], axis=0), axis=1)))

    sort_idx = np.argsort(costs)
    top_idx = sort_idx[:n_top]

    # ── Plot 1: Density (all Nsample samples) ─────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 12))
    _draw_background(ax1)
    for traj in trajs:
        ax1.plot(traj[:, 0], traj[:, 1], "b-", alpha=0.06, lw=0.7)
    # Mean trajectory from Ybar_final
    u_mean_phys = np.array(Ybar_final * action_scale + action_center)
    mean_traj = np.array(jax.jit(rollout_fn)(eta0, nu0, jnp.array(u_mean_phys)))
    ax1.plot(mean_traj[:, 0], mean_traj[:, 1], "r-", lw=2.5, label="Mean (Ybar_final)", zorder=5)
    ax1.plot(*ship_xy, "go", ms=10, zorder=6, label="Ship")
    ax1.plot(*goal_xy, "mx", ms=12, mew=2.5, zorder=6, label="Goal")
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_title(
        f"MBD Replan #{replan_idx} — {len(trajs)} samples at final denoising step\n"
        f"MBD horizon ≈{mean_traj_len:.0f} m  |  cost range [{costs.min():.1f}, {costs.max():.1f}]"
    )
    ax1.legend(fontsize=9)
    plt.tight_layout()
    density_path = os.path.join(viz_dir, f"replan_{replan_idx:04d}_density.png")
    fig1.savefig(density_path, dpi=120, bbox_inches="tight")
    plt.close(fig1)
    print(f"  [MBD viz] saved density plot → {density_path}", flush=True)

    # ── Plot 2: Top-N paths ────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 12))
    _draw_background(ax2)
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, n_top))
    for rank, idx in enumerate(top_idx):
        traj = trajs[idx]
        label = f"#{rank + 1}  cost={costs[idx]:.1f}"
        ax2.plot(traj[:, 0], traj[:, 1], color=colors[rank], lw=2.0, label=label, zorder=5)
        ax2.plot(traj[-1, 0], traj[-1, 1], ".", color=colors[rank], ms=8, zorder=6)
    ax2.plot(*ship_xy, "go", ms=10, zorder=7, label="Ship")
    ax2.plot(*goal_xy, "mx", ms=12, mew=2.5, zorder=7, label="Goal")
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_title(
        f"MBD Replan #{replan_idx} — Top {n_top} paths by total cost\n"
        f"MBD horizon ≈{mean_traj_len:.0f} m  (dashed = full lattice path to goal)"
    )
    ax2.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    top_path = os.path.join(viz_dir, f"replan_{replan_idx:04d}_top{n_top}.png")
    fig2.savefig(top_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [MBD viz] saved top-{n_top} plot  → {top_path}", flush=True)

    # ── Plot 3: Terminal position scatter ──────────────────────────────────
    term_xy = trajs[:, -1, :2]  # (Nsample, 2)
    fig3, ax3 = plt.subplots(figsize=(12, 12))
    _draw_background(ax3)
    sc = ax3.scatter(
        term_xy[:, 0], term_xy[:, 1],
        c=costs, cmap="RdYlGn_r", s=12, alpha=0.6,
        vmin=costs.min(), vmax=np.percentile(costs, 90),
        zorder=4,
    )
    plt.colorbar(sc, ax=ax3, label="Total cost")
    ax3.plot(*ship_xy, "go", ms=10, zorder=6, label="Ship (start)")
    ax3.plot(*goal_xy, "mx", ms=12, mew=2.5, zorder=6, label="Goal")
    ax3.set_xlim(*xlim)
    ax3.set_ylim(*ylim)
    ax3.set_aspect("equal")
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.set_title(
        f"MBD Replan #{replan_idx} — Terminal positions of {len(trajs)} samples\n"
        f"colour = total cost  |  goal at ({goal_xy[0]:.0f}, {goal_xy[1]:.0f})"
    )
    ax3.legend(fontsize=9)
    plt.tight_layout()
    scatter_path = os.path.join(viz_dir, f"replan_{replan_idx:04d}_terminal_scatter.png")
    fig3.savefig(scatter_path, dpi=120, bbox_inches="tight")
    plt.close(fig3)
    print(f"  [MBD viz] saved terminal scatter → {scatter_path}", flush=True)

    # ── Text: cost decomposition for top-N ────────────────────────────────
    # Effective costmap weight = costmap_weight / costmap_normalizer
    effective_costmap_weight = costmap_weight / costmap_normalizer
    weights_arr = np.array([effective_costmap_weight, length_weight, goal_weight, smoothness_weight])
    comp_names = ["costmap", "length ", "goal   ", "jerk   "]
    comp_jit = jax.jit(components_fn)

    # Geometry summary — helps spot OOB issues immediately
    cm_x0, cm_y0 = cost_map.x_origin, cost_map.y_origin
    cm_x1, cm_y1 = cm_x0 + cost_map.width, cm_y0 + cost_map.height
    traj_len_est = float(np.mean([
        np.sum(np.linalg.norm(np.diff(trajs[i, :, :2], axis=0), axis=1))
        for i in sort_idx[:min(5, len(sort_idx))]
    ]))
    lines = [
        f"MBD Replan #{replan_idx} — Top {n_top} cost breakdown",
        f"Weights: costmap={costmap_weight:.3f}  normalizer={costmap_normalizer:.4g}"
        f"  effective_costmap={effective_costmap_weight:.5g}"
        f"  length={length_weight:.3f}  goal={goal_weight:.4f}  smoothness={smoothness_weight:.5f}",
        "-" * 72,
        f"Geometry: costmap [{cm_x0:.0f},{cm_x1:.0f}] x [{cm_y0:.0f},{cm_y1:.0f}]  "
        f"({cost_map.width:.0f}x{cost_map.height:.0f} m)",
        f"  ship=({ship_xy[0]:.0f},{ship_xy[1]:.0f})  goal=({goal_xy[0]:.0f},{goal_xy[1]:.0f})  "
        f"avg traj length={traj_len_est:.0f} m",
        "=" * 72,
    ]
    for rank, idx in enumerate(top_idx):
        u_norm_j = jnp.array(Y0s_norm[idx], dtype=jnp.float32)
        raw = np.array(comp_jit(u_norm_j))          # [costmap, length, goal, jerk] unweighted
        weighted = raw * weights_arr                  # contribution to total cost
        total_cost = float(costs[idx])
        lines.append(f"\nRank #{rank + 1}  total_cost = {total_cost:.3f}")
        for name, w_val, raw_val in zip(comp_names, weighted, raw):
            pct = 100.0 * w_val / total_cost if total_cost > 0 else 0.0
            lines.append(
                f"  {name}  weighted={w_val:9.3f}  ({pct:5.1f}%)  raw={raw_val:.3f}"
            )
        # OOB analysis for this trajectory
        traj_i = trajs[idx]  # (H+1, 3)
        oob_mask = (
            (traj_i[:, 0] < cm_x0) | (traj_i[:, 0] > cm_x1) |
            (traj_i[:, 1] < cm_y0)
        )
        n_oob = int(oob_mask.sum())
        if n_oob > 0:
            seg_lens_i = np.linalg.norm(np.diff(traj_i[:, :2], axis=0), axis=1)
            # Estimate OOB cost contribution: each OOB waypoint adds 1e4 × avg(adj seg lens)
            oob_cost_est = 0.0
            for t in range(len(traj_i)):
                if oob_mask[t]:
                    adj_lens = []
                    if t > 0:
                        adj_lens.append(seg_lens_i[t - 1])
                    if t < len(seg_lens_i):
                        adj_lens.append(seg_lens_i[t])
                    oob_cost_est += 1e4 * (sum(adj_lens) / max(len(adj_lens), 1))
            oob_pct = 100.0 * oob_cost_est / total_cost if total_cost > 0 else 0.0
            lines.append(
                f"  *** OOB: {n_oob} waypoints outside costmap  "
                f"est. OOB cost={oob_cost_est:.0f}  ({oob_pct:.1f}% of total) ***"
            )
        else:
            lines.append(f"  OOB: 0 waypoints outside costmap")

    txt_path = os.path.join(viz_dir, f"replan_{replan_idx:04d}_costs.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    # Also echo to stdout
    print("\n".join(lines), flush=True)
    print(f"  [MBD viz] saved cost breakdown → {txt_path}", flush=True)


# ── High-level entry point ────────────────────────────────────────────────

def _jax_backend_info() -> str:
    """Return a short description of the active JAX backend."""
    try:
        dev = jax.devices()[0]
        platform = dev.platform.upper()
        name = getattr(dev, "device_kind", str(dev))
        return f"{platform} ({name})"
    except Exception:
        return "unknown"


def jax_mbd_optimize(
    cost_map: Any,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg,
    start_nu: Optional[Tuple[float, float, float]] = None,
    lattice_path: Optional[np.ndarray] = None,
    warm_start: Optional[bool] = None,
    viz_dir: Optional[str] = None,
    replan_idx: int = 0,
    viz_world_polygons=None,
    viz_thicknesses=None,
    return_diagnostics: bool = False,
    return_cost_components: bool = False,
    goal_y_line: Optional[float] = None,
    goal_y_line_penalty: float = 1e6,
    goal_x_weight: Optional[float] = None,
    terminal_goal_distance_weight: float = 0.0,
    # Per-step "shortfall to goal-line" cost (clipped above goal_progress_y_line).
    # When goal_progress_weight > 0, adds  w · Σ_t max(0, goal_progress_y_line − y_t)
    # to the trajectory cost.  Provides forward-progress gradient at every
    # waypoint instead of an end-of-horizon goal constraint.
    goal_progress_y_line: Optional[float] = None,
    goal_progress_weight: float = 0.0,
    denoising_movie_dir: Optional[str] = None,
    cost_truncate_y: Optional[float] = None,
    sample_filter_fn=None,
    max_resample_iters=None,
    # ── EB-MBD parameters ──────────────────────────────────────────────────
    use_eb_mbd: bool = False,
    eb_x_min: Optional[float] = None,
    eb_x_max: Optional[float] = None,
    eb_y_min: Optional[float] = None,
    eb_c_max: Optional[float] = None,
    eb_mu: Optional[float] = None,
    eb_kappa: Optional[float] = None,
    eb_oob_cost: float = 5.0,
    eb_oob_as_constraint: bool = False,
    eb_ice_as_constraint: bool = False,
    eb_c_max_ice: Optional[float] = None,
    eb_decouple_oob: bool = False,
    eb_c_max_oob: float = 0.0,
    # When None, OOB reuses ``eb_kappa``. Set to a higher value when using a
    # negative ``eb_c_max_oob`` (safety margin) so the margin clamps to zero
    # in the early diffusion steps independently of the goal schedule.
    eb_kappa_oob: Optional[float] = None,
    # ── AL-MBD parameters ──────────────────────────────────────────────────
    use_al_mbd: bool = False,
    al_x_min: Optional[float] = None,
    al_x_max: Optional[float] = None,
    al_y_min: Optional[float] = None,
    al_y_max: Optional[float] = None,
    al_rho: Optional[float] = None,
    al_M_max: Optional[float] = None,
    al_M_init_oob: float = 0.0,
    al_rho_oob: Optional[float] = None,
    al_rho_goal: Optional[float] = None,
    # ── Emerging-barrier schedule on the goal-reach AL column ──────────────
    # When `al_goal_eb_c_max` is set (positive metres), an EB-style retracting
    # threshold is applied to the goal-reach AL column: the constraint floor
    # `y_T ≥ goal_y_line − c_s(i)` retracts from `goal_y_line − c_max` at
    # i = Ndiffuse − 1 up to `goal_y_line` at i = 0, with
    # c_s(i) = c_max · (i / (Ndiffuse − 1))^κ.  This is the "barrier retracts
    # upward" semantics from the EB-MBD paper applied as an AL threshold
    # shift, so the rest of the AL-MBD machinery (softmax weighting, dual
    # ascent, multipliers) is unchanged.  Requires goal_y_line to be set;
    # if al_rho_goal is None it defaults to al_rho (uniform ρ across columns).
    al_goal_eb_c_max: Optional[float] = None,
    al_goal_eb_kappa: float = 1.0,
    # ── AL-MBD z-score weighting (al_mbd_zscore_reference.md) ──────────────
    # When use_al_zscore=True, replaces the standard softmax(−L/λ) with
    # softmax(−(L − mean(L)) / std(L) / λ).  al_zscore_temperature overrides
    # the temperature inside the z-score (the reference warns λ must shrink
    # ~proportionally to L_std for the weights to discriminate).
    # When use_al_reference_cost=True, swaps the costmap-based base cost J
    # for the reference J(Y) = Σ_t [||x_t − goal|| + 0.1·||u_t||] +
    #                           20·||x_T − goal|| (no costmap, no length).
    use_al_zscore: bool = False,
    al_zscore_temperature: Optional[float] = None,
    use_al_reference_cost: bool = False,
    # ── Hybrid EB + AL: goal=EB(slow), OOB=EB(hard early), ice=AL ──────────
    use_hybrid_eb_al_mbd: bool = False,
    al_rho_ice: float = 0.0,
    al_M_max_ice: float = 1.0e6,
    # Tolerance (m) for the goal-POINT EB constraint in hybrid mode: feasible
    # when ‖x_T − goal_xy‖ ≤ hybrid_goal_tolerance. Set to None to fall back
    # to the legacy y-line constraint (g = y_T − goal_y_line).
    hybrid_goal_tolerance: Optional[float] = 20.0,
    # ── Hard rectangular obstacles (for EB/AL only) ────────────────────────
    obstacle_rects: Optional[list] = None,
    obstacle_disks: Optional[list] = None,
    obstacle_polygons: Optional[list] = None,
    # ── OOB big-M override (non-EB branch). When None, defaults to 1e8.
    #    Pass 0.0 to disable the big-M (used by reject-sampling MBD).
    oob_penalty: Optional[float] = None,
) -> np.ndarray:
    """
    Optimize trajectory using JAX Model-Based Diffusion with 3-DOF dynamics.

    The entire reverse-diffusion loop is compiled into a single JAX
    computation (``jax.lax.scan``).  All ``Nsample`` candidate trajectories
    are evaluated in parallel via ``jax.vmap``.  When a GPU is available JAX
    will automatically dispatch both the parallel sampling and the scan loop
    to the GPU — no code changes needed.

    Parameters
    ----------
    cost_map : CostMap
        Costmap object (from ``build_costmap_from_ice_state``).
    start_pose : (x, y, psi)
        Starting pose [m, m, rad].
    goal : (x, y)
        Goal position [m, m].
    run_cfg : RunConfig
        Configuration (provides ship params and MBD tuning knobs).
    start_nu : (u, v, r), optional
        Initial body-frame velocities [m/s, m/s, rad/s].  Default (2, 0, 0).
    lattice_path : (N, 3), optional
        Lattice A* path [x, y, heading] used to warm-start the action
        sequence (only when warm-start is enabled).
    warm_start : bool, optional
        If True, seed the initial action sequence from ``lattice_path``.
        If False, start from constant forward surge (pure diffusion).
        Defaults to ``run_cfg.planning_jax_mbd_warm_start``.

    Returns
    -------
    trajectory : (H+1, 3) array of [x, y, psi] in world frame.
    """
    # ── Read parameters from config (with safe defaults) ──────────────
    H = getattr(run_cfg, "planning_jax_mbd_horizon", 80)
    Nsample = getattr(run_cfg, "planning_jax_mbd_nsample", 256)
    Ndiffuse = getattr(run_cfg, "planning_jax_mbd_ndiffuse", 50)
    temperature = getattr(run_cfg, "planning_jax_mbd_temperature", 0.1)
    beta0 = getattr(run_cfg, "planning_jax_mbd_beta0", 1e-4)
    betaT = getattr(run_cfg, "planning_jax_mbd_betaT", 0.05)
    costmap_weight = getattr(run_cfg, "planning_jax_mbd_costmap_weight", 1.0)
    costmap_normalizer = getattr(run_cfg, "planning_jax_mbd_costmap_normalizer", 1.0)
    goal_weight = getattr(run_cfg, "planning_jax_mbd_goal_weight", 0.1)
    length_weight = getattr(run_cfg, "planning_jax_mbd_length_weight", 1.0)
    smoothness_weight = getattr(run_cfg, "planning_jax_mbd_smoothness_weight", 0.005)
    _cfg_goal_x_weight = getattr(run_cfg, "planning_jax_mbd_goal_x_weight", 0.0)
    # Explicit argument overrides config; None means "use config value"
    goal_x_weight = goal_x_weight if goal_x_weight is not None else _cfg_goal_x_weight
    surge_max = getattr(run_cfg, "planning_jax_mbd_surge_max", 2e6)
    sway_max = getattr(run_cfg, "planning_jax_mbd_sway_max", 5e5)
    yaw_max = getattr(run_cfg, "planning_jax_mbd_yaw_max", 5e8)
    dt_substep = getattr(run_cfg, "planning_jax_mbd_dt_substep", 1.0)
    n_substeps = getattr(run_cfg, "planning_jax_mbd_n_substeps", 5)
    init_surge_bias = getattr(run_cfg, "planning_jax_mbd_init_surge_bias", 0.7)
    seed = getattr(run_cfg, "planning_diffusion_seed", None) or 42
    use_best_sample = getattr(run_cfg, "planning_jax_mbd_use_best_sample", False)

    if warm_start is None:
        warm_start = getattr(run_cfg, "planning_jax_mbd_warm_start", True)

    # ── EB-MBD parameters (args override config) ──────────────────────
    if not use_eb_mbd:
        use_eb_mbd = getattr(run_cfg, "planning_jax_mbd_use_eb_mbd", False)

    # Hybrid EB+AL: activates EB branch (for goal + OOB barriers) and routes
    # obstacle_rects through an AL-ice constraint instead of an EB rect barrier.
    if use_hybrid_eb_al_mbd:
        use_eb_mbd = True
    if eb_c_max is None:
        eb_c_max = getattr(run_cfg, "planning_jax_mbd_eb_c_max", 500.0)
    if eb_mu is None:
        eb_mu = getattr(run_cfg, "planning_jax_mbd_eb_mu", 50000.0)
    if eb_kappa is None:
        eb_kappa = getattr(run_cfg, "planning_jax_mbd_eb_kappa", 1.0)
    # ── AL-MBD parameters (args override config) ──────────────────────
    if not use_al_mbd:
        use_al_mbd = getattr(run_cfg, "planning_jax_mbd_use_al_mbd", False)
    if al_rho is None:
        al_rho = getattr(run_cfg, "planning_jax_mbd_al_rho", 0.1)
    if al_M_max is None:
        al_M_max = getattr(run_cfg, "planning_jax_mbd_al_M_max", 1000.0)

    # ── Convert costmap to JAX ────────────────────────────────────────
    grid = jnp.array(cost_map.cost_map, dtype=jnp.float32)
    meta = {
        "x_origin": float(cost_map.x_origin),
        "y_origin": float(cost_map.y_origin),
        "resolution": float(cost_map.resolution),
        "rows": int(cost_map.rows),
        "cols": int(cost_map.cols),
        "adaptive_velocity": bool(getattr(cost_map, "adaptive_velocity", False)),
        "base_cost":         float(getattr(cost_map, "base_cost", 0.0)),
    }

    # ── Ship dynamics (consistent with simulation) ────────────────────
    mass = run_cfg.ship_displacement
    L = run_cfg.ship_length
    B = run_cfg.ship_beam
    Izz = (1.0 / 12.0) * mass * (L ** 2 + B ** 2)

    sim_cfg = get_default_simulation_config()
    rollout_fn, M3_np, D3_np = make_dynamics_rollout(
        mass=mass,
        Izz=Izz,
        dt_substep=dt_substep,
        n_substeps=n_substeps,
        ship_damping_xy=sim_cfg.ship_damping_xy,
        ship_damping_psi=sim_cfg.ship_damping_psi,
        added_mass_surge_frac=sim_cfg.mss_added_mass_surge_frac,
        added_mass_sway_frac=sim_cfg.mss_added_mass_sway_frac,
        added_mass_yaw_frac=sim_cfg.mss_added_mass_yaw_frac,
    )

    # ── Initial state ─────────────────────────────────────────────────
    eta0 = jnp.array(start_pose, dtype=jnp.float32)
    nu0 = jnp.array(start_nu or (2.0, 0.0, 0.0), dtype=jnp.float32)

    # ── Action normalisation ──────────────────────────────────────────
    action_center = jnp.array([surge_max / 2, 0.0, 0.0], dtype=jnp.float32)
    action_scale = jnp.array([surge_max / 2, sway_max, yaw_max], dtype=jnp.float32)

    # ── Warm-start from lattice path (seeds MBD, not used in cost) ────
    dt_action = dt_substep * n_substeps
    Ybar_init = None
    if warm_start and lattice_path is not None and len(lattice_path) >= 3:
        lattice_path = np.asarray(lattice_path, dtype=float)
        speed_estimate = 4.0
        coverage_distance = H * dt_action * speed_estimate

        seg_lens = np.linalg.norm(
            np.diff(lattice_path[:, :2], axis=0), axis=1
        )
        s = np.zeros(len(lattice_path))
        s[1:] = np.cumsum(seg_lens)

        cutoff_idx = int(np.searchsorted(s, coverage_distance, side="right"))
        cutoff_idx = max(min(cutoff_idx, len(lattice_path)), 2)

        truncated = lattice_path[:cutoff_idx]
        Ybar_init = _warm_start_from_path(
            truncated, H, init_surge_bias,
            eta0=np.array(start_pose, dtype=np.float64),
            nu0=np.array(start_nu or (2.0, 0.0, 0.0), dtype=np.float64),
            M3_np=M3_np,
            D3_np=D3_np,
            dt_substep=dt_substep,
            n_substeps=n_substeps,
            action_center_np=np.array([surge_max / 2, 0.0, 0.0]),
            action_scale_np=np.array([surge_max / 2, sway_max, yaw_max]),
        )

    # ── Cost function and constraint function ─────────────────────────
    # When a goal line is used, automatically suppress all cost contributions
    # (costmap and OOB penalties) for waypoints that have already crossed it.
    # The simulation stops at the goal line, so trajectory portions beyond it
    # are irrelevant and should not be penalised.
    effective_truncate_y = cost_truncate_y
    if effective_truncate_y is None and goal_y_line is not None:
        effective_truncate_y = goal_y_line

    oob_constraint_fn = None  # set in EB branch below; None elsewhere

    if use_eb_mbd:
        # EB-MBD: wall cells in the costmap handle OOB for rect-grid scenarios;
        # the EB linear OOB cost handles disk-costmap scenarios.
        # Keep oob_penalty=0.0 to avoid penalising hull corners near the grid edge.
        # When eb_ice_as_constraint=True the costmap is removed from J(x) entirely
        # and enforced as a hard EB barrier constraint (g_ice = 0 - costmap_integral).
        _cmap_weight = 0.0 if eb_ice_as_constraint else costmap_weight
        cost_fn = make_cost_fn(
            grid, meta, goal, rollout_fn, eta0, nu0,
            action_center, action_scale,
            costmap_weight=_cmap_weight,
            costmap_normalizer=costmap_normalizer,
            goal_weight=goal_weight,
            length_weight=length_weight,
            smoothness_weight=smoothness_weight,
            ship_length=L,
            ship_beam=B,
            cost_truncate_y=effective_truncate_y,
            oob_penalty=0.0,
        )
        if goal_y_line is not None and goal_x_weight > 0.0:
            _gx_fn = make_terminal_goal_distance_cost_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                goal_x=float(goal[0]), goal_y_line=goal_y_line, weight=goal_x_weight,
            )
            _base_fn = cost_fn
            cost_fn = lambda u_norm: _base_fn(u_norm) + _gx_fn(u_norm)  # noqa: E731
        if terminal_goal_distance_weight > 0.0:
            _td_fn = make_terminal_goal_euclidean_cost_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                goal_xy=(float(goal[0]), float(goal[1])),
                weight=terminal_goal_distance_weight,
            )
            _base_td = cost_fn
            cost_fn = lambda u_norm: _base_td(u_norm) + _td_fn(u_norm)  # noqa: E731
        if goal_progress_y_line is not None and goal_progress_weight > 0.0:
            _gp_fn = make_goal_progress_cost_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                goal_y_line=float(goal_progress_y_line),
                weight=float(goal_progress_weight),
            )
            _base_gp = cost_fn
            cost_fn = lambda u_norm: _base_gp(u_norm) + _gp_fn(u_norm)  # noqa: E731
        # Resolve EB-MBD boundary parameters (default to costmap extent)
        _x_min = eb_x_min if eb_x_min is not None else float(cost_map.x_origin)
        _x_max = (eb_x_max if eb_x_max is not None
                  else float(cost_map.x_origin) + float(cost_map.width))
        _y_min = eb_y_min if eb_y_min is not None else float(cost_map.y_origin)
        if goal_y_line is not None:
            _goal_y = float(goal_y_line)
            if use_hybrid_eb_al_mbd and hybrid_goal_tolerance is not None:
                # Hybrid mode: enforce reaching the actual goal POINT, not just
                # crossing the y-line. The EB barrier then pulls samples whose
                # endpoint lands inside the tolerance ball around goal_xy.
                constraint_fn = make_goal_point_constraint_fn(
                    rollout_fn, eta0, nu0, action_center, action_scale,
                    goal_xy=(float(goal[0]), float(goal[1])),
                    tolerance=float(hybrid_goal_tolerance),
                )
            else:
                constraint_fn = make_goal_constraint_fn(
                    rollout_fn, eta0, nu0, action_center, action_scale,
                    goal_y_line=_goal_y,
                )
        elif use_hybrid_eb_al_mbd:
            # Hybrid with no goal barrier: trivial constraint (always
            # feasible) so the EB barrier μ·log(1 + c_s) is constant across
            # samples and drops out of the softmax. Goal pull lives entirely
            # in the cost function (terminal Euclidean weight).
            def constraint_fn(u_norm):  # noqa: E306
                return jnp.float32(1.0)
        elif eb_decouple_oob and obstacle_rects:
            # AL-parity setup: rects go through the main constraint_fn (with the
            # eb_c_max schedule) so gates emerge late; OOB is wired below as a
            # separate barrier with eb_c_max_oob=0 so it is hard from step 0.
            constraint_fn = make_rect_obstacles_eb_fn(
                rollout_fn, eta0, nu0, action_center, action_scale, obstacle_rects,
            )
        else:
            # No goal-line hard constraint: use OOB (bottom/left/right) as the EB
            # barrier constraint so the framework still has a constraint to enforce.
            constraint_fn = make_oob_constraint_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                x_min=_x_min, x_max=_x_max, y_min=_y_min,
                cost_truncate_y=effective_truncate_y,
            )
        # Ice constraint. When eb_c_max_ice is not None we wire it as a
        # SEPARATE additive barrier with its own c_s schedule (parallel to
        # the OOB barrier); otherwise we fall back to the legacy combined-min
        # form so existing callers behave identically.
        ice_constraint_fn = None
        if eb_ice_as_constraint:
            _ice_c_fn = make_ice_cost_constraint_fn(
                grid, meta, rollout_fn, eta0, nu0, action_center, action_scale,
                ice_threshold=0.0,
                cost_truncate_y=effective_truncate_y,
                ship_length=L,
                ship_beam=B,
            )
            if eb_c_max_ice is not None:
                # Decoupled: ice gets its own barrier with eb_c_max_ice schedule.
                ice_constraint_fn = _ice_c_fn
            else:
                # Legacy: combine ice with goal/walls under shared eb_c_max.
                _goal_c_fn = constraint_fn
                constraint_fn = lambda u: jnp.minimum(_goal_c_fn(u), _ice_c_fn(u))  # noqa: E731
        if use_hybrid_eb_al_mbd and sample_filter_fn is not None:
            # Hybrid mode with reject filter: OOB is handled by the filter,
            # not by an EB barrier. Setting oob_constraint_fn=None ensures
            # the EB runner doesn't double-count OOB.
            oob_constraint_fn = None
        elif eb_oob_as_constraint or eb_decouple_oob:
            # OOB as a hard EB barrier constraint (combined with goal constraint).
            # `eb_decouple_oob` is the AL-parity path: OOB gets its own barrier
            # with eb_c_max_oob (default 0 → hard from step 0).
            oob_constraint_fn = make_oob_constraint_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                x_min=_x_min, x_max=_x_max, y_min=_y_min,
                cost_truncate_y=effective_truncate_y,
            )
        else:
            # OOB enforcement via linear cost (distance × steps), not a barrier.
            if eb_oob_cost > 0.0:
                _oob_cost_fn = make_oob_cost_fn(
                    rollout_fn, eta0, nu0, action_center, action_scale,
                    x_min=_x_min, x_max=_x_max, y_min=_y_min,
                    oob_cost_per_m=eb_oob_cost,
                    cost_truncate_y=effective_truncate_y,
                )
                _base_cost_fn = cost_fn
                cost_fn = lambda u: _base_cost_fn(u) + _oob_cost_fn(u)  # noqa: E731
            oob_constraint_fn = None
        if obstacle_rects and not eb_decouple_oob and not use_hybrid_eb_al_mbd:
            _rect_fn = make_rect_obstacles_eb_fn(
                rollout_fn, eta0, nu0, action_center, action_scale, obstacle_rects)
            _base_c = constraint_fn
            # Hard obstacle rects combined with goal constraint (share large c_max)
            constraint_fn = lambda u: jnp.minimum(_base_c(u), _rect_fn(u))  # noqa: E731

        # Hybrid EB+AL: build the AL-ice constraint from obstacle_rects (and/or
        # disks) so the EB runner can apply the AL penalty + dual ascent.
        _al_ice_fn = None
        if use_hybrid_eb_al_mbd and obstacle_rects:
            _al_ice_fn = make_rect_obstacles_al_fn(
                rollout_fn, eta0, nu0, action_center, action_scale, obstacle_rects)
        if use_hybrid_eb_al_mbd and obstacle_disks:
            _disk_al_fn = make_disk_obstacles_al_fn(
                rollout_fn, eta0, nu0, action_center, action_scale, obstacle_disks)
            if _al_ice_fn is None:
                _al_ice_fn = _disk_al_fn
            else:
                _base_al = _al_ice_fn
                _al_ice_fn = lambda u: jnp.concatenate(  # noqa: E731
                    [_base_al(u), _disk_al_fn(u)], axis=1)
    else:
        _oob_pen = 1e8 if oob_penalty is None else float(oob_penalty)
        cost_fn = make_cost_fn(
            grid, meta, goal, rollout_fn, eta0, nu0,
            action_center, action_scale,
            costmap_weight=costmap_weight,
            costmap_normalizer=costmap_normalizer,
            goal_weight=goal_weight,
            length_weight=length_weight,
            smoothness_weight=smoothness_weight,
            ship_length=L,
            ship_beam=B,
            cost_truncate_y=effective_truncate_y,
            oob_penalty=_oob_pen,
        )
        if goal_y_line is not None and float(goal_y_line_penalty) > 0.0:
            _gl_fn = make_goal_line_cost_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                goal_y_line, goal_y_line_penalty,
            )
            _base_fn = cost_fn
            cost_fn = lambda u_norm: _base_fn(u_norm) + _gl_fn(u_norm)  # noqa: E731
        if goal_y_line is not None and goal_x_weight > 0.0:
            _gx_fn = make_terminal_goal_distance_cost_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                goal_x=float(goal[0]), goal_y_line=goal_y_line, weight=goal_x_weight,
            )
            _prev_fn = cost_fn
            cost_fn = lambda u_norm: _prev_fn(u_norm) + _gx_fn(u_norm)  # noqa: E731
        if terminal_goal_distance_weight > 0.0:
            _td_fn = make_terminal_goal_euclidean_cost_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                goal_xy=(float(goal[0]), float(goal[1])),
                weight=terminal_goal_distance_weight,
            )
            _prev_td = cost_fn
            cost_fn = lambda u_norm: _prev_td(u_norm) + _td_fn(u_norm)  # noqa: E731
        if goal_progress_y_line is not None and goal_progress_weight > 0.0:
            _gp_fn = make_goal_progress_cost_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                goal_y_line=float(goal_progress_y_line),
                weight=float(goal_progress_weight),
            )
            _prev_gp = cost_fn
            cost_fn = lambda u_norm: _prev_gp(u_norm) + _gp_fn(u_norm)  # noqa: E731
        constraint_fn = None

    # ── AL-MBD: build base cost (no OOB penalty) + per-step constraint fn ─
    al_constraint_fn = None
    if use_al_mbd:
        if use_eb_mbd:
            raise ValueError("use_al_mbd and use_eb_mbd are mutually exclusive.")
        if use_al_reference_cost:
            # AL-MBD reference J(Y) (al_mbd_zscore_reference.md).  Replaces
            # the costmap+length+terminal mix with the per-step goal-pull +
            # action-magnitude + terminal-distance form used in the navigator
            # task that the z-score variant was tuned on.  Skips every other
            # add-on cost (goal-line big-M, goal-x, terminal_goal_distance)
            # to keep the trajectory cost L of comparable magnitude to the
            # 2D navigator's, so the reference's z-score λ behaviour transfers.
            cost_fn = make_almbd_reference_cost_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                goal_xy=(float(goal[0]), float(goal[1])),
                stage_action_weight=0.1,
                terminal_weight=20.0,
            )
        else:
            # AL handles OOB via per-step augmented Lagrangian constraints.
            # The Big-M is opt-in via the explicit `oob_penalty` arg; default off.
            _al_oob_pen = 1e8 if oob_penalty is None else float(oob_penalty)
            cost_fn = make_cost_fn(
                grid, meta, goal, rollout_fn, eta0, nu0,
                action_center, action_scale,
                costmap_weight=costmap_weight,
                costmap_normalizer=costmap_normalizer,
                goal_weight=goal_weight,
                length_weight=length_weight,
                smoothness_weight=smoothness_weight,
                ship_length=L,
                ship_beam=B,
                cost_truncate_y=effective_truncate_y,
                oob_penalty=_al_oob_pen,
            )
            # Soft goal-line penalty provides gradient signal from step 0; the AL
            # dual ascent progressively tightens the hard constraint.  Use a smaller
            # penalty than vanilla MBD (1e4 vs 1e6) so the base cost stays well-
            # conditioned alongside the AL penalty.
            #
            # Suppressed when an emerging goal barrier (`al_goal_eb_c_max`) is
            # in use: that mode wants the early diffusion steps to explore
            # freely below goal_y_line, and the 1e4·max(0, goal_y_line − y_T)
            # term would overwhelm everything else and pin trajectories near
            # the goal line from step 0, defeating the schedule.
            if goal_y_line is not None and al_goal_eb_c_max is None:
                _gl_fn = make_goal_line_cost_fn(
                    rollout_fn, eta0, nu0, action_center, action_scale,
                    goal_y_line, penalty=1e4,
                )
                _base_fn = cost_fn
                cost_fn = lambda u_norm: _base_fn(u_norm) + _gl_fn(u_norm)  # noqa: E731
            if goal_y_line is not None and goal_x_weight > 0.0:
                _gx_fn = make_terminal_goal_distance_cost_fn(
                    rollout_fn, eta0, nu0, action_center, action_scale,
                    goal_x=float(goal[0]), goal_y_line=goal_y_line, weight=goal_x_weight,
                )
                _prev_fn = cost_fn
                cost_fn = lambda u_norm: _prev_fn(u_norm) + _gx_fn(u_norm)  # noqa: E731
            if terminal_goal_distance_weight > 0.0:
                _td_fn = make_terminal_goal_euclidean_cost_fn(
                    rollout_fn, eta0, nu0, action_center, action_scale,
                    goal_xy=(float(goal[0]), float(goal[1])),
                    weight=terminal_goal_distance_weight,
                )
                _prev_td = cost_fn
                cost_fn = lambda u_norm: _prev_td(u_norm) + _td_fn(u_norm)  # noqa: E731
            if goal_progress_y_line is not None and goal_progress_weight > 0.0:
                _gp_fn = make_goal_progress_cost_fn(
                    rollout_fn, eta0, nu0, action_center, action_scale,
                    goal_y_line=float(goal_progress_y_line),
                    weight=float(goal_progress_weight),
                )
                _prev_gp = cost_fn
                cost_fn = lambda u_norm: _prev_gp(u_norm) + _gp_fn(u_norm)  # noqa: E731
        _al_x_min = al_x_min if al_x_min is not None else float(cost_map.x_origin)
        _al_x_max = (al_x_max if al_x_max is not None
                     else float(cost_map.x_origin) + float(cost_map.width))
        _al_y_min = al_y_min if al_y_min is not None else float(cost_map.y_origin)
        _al_y_max = (al_y_max if al_y_max is not None
                     else float(cost_map.y_origin) + float(cost_map.height))
        # Linear OOB cost: same as EB-MBD approach — provides gradient at the
        # domain boundary so the class mean never drifts slightly OOB.
        # The AL dual-ascent constraint is an additional enforcement layer.
        if eb_oob_cost > 0.0:
            _oob_cost_fn = make_oob_cost_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                x_min=_al_x_min, x_max=_al_x_max, y_min=_al_y_min,
                oob_cost_per_m=eb_oob_cost,
                cost_truncate_y=effective_truncate_y,
            )
            _base_fn_al = cost_fn
            cost_fn = lambda u: _base_fn_al(u) + _oob_cost_fn(u)  # noqa: E731
        # When the goal-reach has its own AL column (al_rho_goal set), DON'T
        # also fold it into column 0; otherwise the goal-line residual would
        # be double-counted (once in column 0 via OOB-merge, once in the
        # dedicated last column).
        _al_goal_y = float(goal_y_line) if (goal_y_line is not None
                                            and al_rho_goal is None) else None
        al_constraint_fn = make_al_per_step_constraints_fn(
            rollout_fn, eta0, nu0, action_center, action_scale,
            x_min=_al_x_min, x_max=_al_x_max, y_min=_al_y_min, y_max=_al_y_max,
            goal_y_line=_al_goal_y,
        )
        if obstacle_rects:
            _rect_al_fn = make_rect_obstacles_al_fn(
                rollout_fn, eta0, nu0, action_center, action_scale, obstacle_rects)
            _base_al = al_constraint_fn
            al_constraint_fn = lambda u: jnp.concatenate(  # noqa: E731
                [_base_al(u), _rect_al_fn(u)], axis=1)
        if obstacle_disks:
            _disk_al_fn = make_disk_obstacles_al_fn(
                rollout_fn, eta0, nu0, action_center, action_scale, obstacle_disks)
            _base_al = al_constraint_fn
            al_constraint_fn = lambda u: jnp.concatenate(  # noqa: E731
                [_base_al(u), _disk_al_fn(u)], axis=1)
        if obstacle_polygons:
            _poly_al_fn = make_polygon_obstacles_al_fn(
                rollout_fn, eta0, nu0, action_center, action_scale, obstacle_polygons)
            _base_al = al_constraint_fn
            al_constraint_fn = lambda u: jnp.concatenate(  # noqa: E731
                [_base_al(u), _poly_al_fn(u)], axis=1)
        # Dedicated goal-reach column (appended LAST so rho_vec[-1] =
        # al_rho_goal in run_al_mbd_with_diagnostics).  Triggered when
        # goal_y_line is set AND either:
        #   (a) al_rho_goal is provided                       (goalhard)
        #   (b) al_goal_eb_c_max is provided (emerging barrier — defaults
        #       al_rho_goal to al_rho if the caller didn't override).
        # Otherwise the legacy OOB-merged form handles the goal-line residual.
        if (
            goal_y_line is not None
            and (al_rho_goal is not None or al_goal_eb_c_max is not None)
        ):
            if al_rho_goal is None:
                al_rho_goal = al_rho
            _goal_al_fn = make_goal_reach_al_fn(
                rollout_fn, eta0, nu0, action_center, action_scale,
                float(goal_y_line), int(H),
            )
            _base_al_g = al_constraint_fn
            al_constraint_fn = lambda u: jnp.concatenate(  # noqa: E731
                [_base_al_g(u), _goal_al_fn(u)], axis=1)

    # ── Run MBD (importance-weighted score ascent) ────────────────────
    if use_al_mbd:
        _algo = "AL-MBD"
    elif use_eb_mbd:
        _algo = "EB-MBD"
    else:
        _algo = "MBD"
    ws_label = "warm-started from A*" if Ybar_init is not None else "cold start (no A* prior)"
    backend = _jax_backend_info()
    print(
        f"  JAX {_algo} [{backend}]: H={H}, Nsample={Nsample}, Ndiffuse={Ndiffuse}, "
        f"temp={temperature}, dt_action={dt_action}s, {ws_label}",
        flush=True,
    )
    if use_eb_mbd:
        print(
            f"  EB-MBD barrier: c_max={eb_c_max:.1f}, mu={eb_mu:.0f}, kappa={eb_kappa:.2f}",
            flush=True,
        )
    if use_al_mbd:
        _zs_tag = ""
        if use_al_zscore:
            _zs_lam = al_zscore_temperature if al_zscore_temperature is not None else temperature
            _zs_tag = f", zscore=ON (lambda={_zs_lam:.4g})"
        _refc_tag = ", reference-cost" if use_al_reference_cost else ""
        print(
            f"  AL-MBD: rho={al_rho:.4g}, M_max={al_M_max:.4g}{_zs_tag}{_refc_tag}",
            flush=True,
        )
    import time as _time
    _t0 = _time.perf_counter()
    _diagnostics = None
    # Run diagnostics mode when explicitly requested OR when a denoising movie
    # is wanted OR when a sample_filter_fn is provided (the fast-path `run_mbd`
    # has no filter parameter and silently drops it).
    _need_diagnostics = (
        return_diagnostics
        or (denoising_movie_dir is not None)
        or (sample_filter_fn is not None and not use_eb_mbd and not use_al_mbd)
    )
    if _need_diagnostics:
        if use_al_mbd:
            _diagnostics = run_al_mbd_with_diagnostics(
                cost_fn, al_constraint_fn, rollout_fn, eta0, nu0,
                action_center, action_scale,
                H, Nsample, Ndiffuse, temperature, beta0, betaT, seed,
                al_rho=al_rho,
                al_M_max=al_M_max,
                Ybar_init=Ybar_init,
                init_surge_bias=init_surge_bias,
                n_top_samples=Nsample,
                al_M_init_oob=al_M_init_oob,
                al_rho_oob=al_rho_oob,
                al_rho_goal=al_rho_goal,
                use_zscore_weighting=use_al_zscore,
                zscore_temperature=al_zscore_temperature,
                al_goal_eb_c_max=al_goal_eb_c_max,
                al_goal_eb_kappa=al_goal_eb_kappa,
            )
        elif use_eb_mbd:
            _diagnostics = run_eb_mbd_with_diagnostics(
                cost_fn, constraint_fn, rollout_fn, eta0, nu0,
                action_center, action_scale,
                H, Nsample, Ndiffuse, temperature, beta0, betaT, seed,
                eb_c_max=eb_c_max,
                eb_mu=eb_mu,
                eb_kappa=eb_kappa,
                Ybar_init=Ybar_init,
                init_surge_bias=init_surge_bias,
                n_top_samples=Nsample,
                oob_constraint_fn=oob_constraint_fn,
                eb_c_max_oob=float(eb_c_max_oob),
                eb_kappa_oob=(float(eb_kappa_oob)
                              if eb_kappa_oob is not None else None),
                ice_constraint_fn=ice_constraint_fn,
                eb_c_max_ice=float(eb_c_max_ice) if eb_c_max_ice is not None else 0.0,
                al_ice_constraint_fn=(_al_ice_fn if use_hybrid_eb_al_mbd else None),
                al_rho_ice=float(al_rho_ice),
                al_M_max_ice=float(al_M_max_ice),
                sample_filter_fn=(sample_filter_fn if use_hybrid_eb_al_mbd else None),
            )
        else:
            _diagnostics = run_mbd_with_diagnostics(
                cost_fn, rollout_fn, eta0, nu0,
                action_center, action_scale,
                H, Nsample, Ndiffuse, temperature, beta0, betaT, seed,
                Ybar_init=Ybar_init,
                init_surge_bias=init_surge_bias,
                n_top_samples=Nsample,
                sample_filter_fn=sample_filter_fn,
                max_resample_iters=max_resample_iters,
            )
        # Use cost stats from final diagnostics step
        cost_mean = _diagnostics["cost_means"][-1] if _diagnostics["cost_means"] else float("nan")
        cost_min = _diagnostics["cost_mins"][-1] if _diagnostics["cost_mins"] else float("nan")
        Ybar_final = _diagnostics["Ybar_final"]  # normalized action sequence (H, 3)
    else:
        if use_al_mbd:
            _al_diag = run_al_mbd_with_diagnostics(
                cost_fn, al_constraint_fn, rollout_fn, eta0, nu0,
                action_center, action_scale,
                H, Nsample, Ndiffuse, temperature, beta0, betaT, seed,
                al_rho=al_rho,
                al_M_max=al_M_max,
                Ybar_init=Ybar_init,
                init_surge_bias=init_surge_bias,
                n_top_samples=1,
                al_M_init_oob=al_M_init_oob,
                al_rho_oob=al_rho_oob,
                al_rho_goal=al_rho_goal,
                use_zscore_weighting=use_al_zscore,
                zscore_temperature=al_zscore_temperature,
                al_goal_eb_c_max=al_goal_eb_c_max,
                al_goal_eb_kappa=al_goal_eb_kappa,
            )
            Ybar_final = _al_diag["Ybar_final"]
            cost_mean  = _al_diag["cost_means"][-1] if _al_diag["cost_means"] else float("nan")
            cost_min   = _al_diag["cost_mins"][-1]  if _al_diag["cost_mins"]  else float("nan")
        elif use_eb_mbd:
            _eb_diag = run_eb_mbd_with_diagnostics(
                cost_fn, constraint_fn, rollout_fn, eta0, nu0,
                action_center, action_scale,
                H, Nsample, Ndiffuse, temperature, beta0, betaT, seed,
                eb_c_max=eb_c_max,
                eb_mu=eb_mu,
                eb_kappa=eb_kappa,
                Ybar_init=Ybar_init,
                init_surge_bias=init_surge_bias,
                n_top_samples=1,
                oob_constraint_fn=oob_constraint_fn,
                eb_c_max_oob=float(eb_c_max_oob),
                eb_kappa_oob=(float(eb_kappa_oob)
                              if eb_kappa_oob is not None else None),
                ice_constraint_fn=ice_constraint_fn,
                eb_c_max_ice=float(eb_c_max_ice) if eb_c_max_ice is not None else 0.0,
                al_ice_constraint_fn=(_al_ice_fn if use_hybrid_eb_al_mbd else None),
                al_rho_ice=float(al_rho_ice),
                al_M_max_ice=float(al_M_max_ice),
                sample_filter_fn=(sample_filter_fn if use_hybrid_eb_al_mbd else None),
            )
            Ybar_final = _eb_diag["Ybar_final"]
            cost_mean  = _eb_diag["cost_means"][-1] if _eb_diag["cost_means"] else float("nan")
            cost_min   = _eb_diag["cost_mins"][-1]  if _eb_diag["cost_mins"]  else float("nan")
        else:
            Ybar_final, cost_mean, cost_min = run_mbd(
                cost_fn, H, Nsample, Ndiffuse, temperature,
                beta0, betaT, seed,
                Ybar_init=Ybar_init,
                init_surge_bias=init_surge_bias,
            )
    _elapsed = _time.perf_counter() - _t0
    print(
        f"  JAX {_algo} done ({_elapsed:.1f}s): cost_mean={cost_mean:.1f}, cost_min={cost_min:.1f}",
        flush=True,
    )
    if _diagnostics is not None and _diagnostics["cost_mins"]:
        _cost_mins_arr = np.array(_diagnostics["cost_mins"])
        _cost_of_mean_arr = np.array(_diagnostics["cost_of_mean"])
        _global_min_cost = float(_cost_mins_arr.min())
        _global_min_step = int(_cost_mins_arr.argmin())
        _n_steps = len(_cost_mins_arr)
        _last_step_min = float(_cost_mins_arr[-1])
        _last_step_std = float(np.std(_diagnostics["sample_costs"][-1]))
        _final_cost = float(jax.jit(cost_fn)(Ybar_final))
        # Best cost_of_mean ever achieved (closest the mean got to the good basin)
        _mean_best_cost = float(_cost_of_mean_arr.min())
        _mean_best_step = int(_cost_of_mean_arr.argmin())
        print(
            f"  [MBD debug] "
            f"final_traj={_final_cost:.1f} | "
            f"last_step: best={_last_step_min:.1f}, std={_last_step_std:.1f} | "
            f"global_min={_global_min_cost:.1f} at step {_global_min_step}/{_n_steps - 1} | "
            f"mean_best={_mean_best_cost:.1f} at step {_mean_best_step}/{_n_steps - 1}",
            flush=True,
        )
        # Per-step table: step | cost_of_mean | cost_min | gap | accept
        _ar = _diagnostics.get("accept_rates") or []
        if _ar:
            _ar_arr = np.array(_ar)
            print(
                f"  [MBD debug] reject-sampling accept: "
                f"min={_ar_arr.min()*100:.1f}%  mean={_ar_arr.mean()*100:.1f}%  "
                f"max={_ar_arr.max()*100:.1f}%  zero-accept steps={int((_ar_arr == 0.0).sum())}/{len(_ar_arr)}",
                flush=True,
            )
        print(f"  [MBD debug] step-by-step (cost_of_mean | cost_min | gap | accept):", flush=True)
        for _s in range(_n_steps):
            _com = _cost_of_mean_arr[_s]
            _cmin = _cost_mins_arr[_s]
            _acc = _ar[_s] if _s < len(_ar) else None
            _acc_str = f" | accept={_acc*100:5.1f}%" if _acc is not None else ""
            _marker = " <-- global min" if _s == _global_min_step else ""
            _marker2 = " <-- mean best" if _s == _mean_best_step and _mean_best_step != _global_min_step else ""
            print(f"    step {_s:3d}: mean={_com:10.1f} | best_sample={_cmin:10.1f} | gap={_com - _cmin:10.1f}{_acc_str}{_marker}{_marker2}", flush=True)

        # ── Action-space convexity probe ──────────────────────────────────────
        # Interpolate between the global best and its same-step 2nd-best sample
        # across alpha in [0, 1].  If the midpoint cost >> both endpoints, the
        # cost landscape is non-convex in action space (the "banana" problem).
        _Ya = _diagnostics["best_sample_actions"]
        _Yb = _diagnostics["second_best_sample_actions"]
        if _Ya is not None and _Yb is not None:
            _cost_fn_jit = jax.jit(cost_fn)
            _alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            _interp_costs = []
            for _a in _alphas:
                _Ymix = jnp.array((1.0 - _a) * _Ya + _a * _Yb)
                _interp_costs.append(float(_cost_fn_jit(_Ymix)))
            print(
                f"  [MBD convexity] best={_diagnostics['best_sample_cost']:.1f}, "
                f"2nd_best={_diagnostics['second_best_sample_cost']:.1f}",
                flush=True,
            )
            print(
                f"  [MBD convexity] action-space interpolation (alpha=0 is best, alpha=1 is 2nd-best):",
                flush=True,
            )
            for _a, _c in zip(_alphas, _interp_costs):
                _excess = _c - min(_interp_costs[0], _interp_costs[-1])
                print(f"    alpha={_a:.2f}: cost={_c:10.1f}  (+{_excess:.1f} above lower endpoint)", flush=True)

    # ── Optional: generate replan visualizations ──────────────────────
    # Static plots (density / top-N / cost breakdown): use final-step samples.
    # When diagnostics are available (denoising movie path), derive from them.
    # Otherwise fall back to the cheap single-step resample.
    if viz_dir is not None:
        components_fn = make_cost_components_fn(
            grid, meta, goal, rollout_fn, eta0, nu0,
            action_center, action_scale,
            ship_length=L, ship_beam=B,
        )
        if _need_diagnostics and _diagnostics is not None:
            # Last step samples are already in diagnostics — use for density/scatter plots.
            trajs_viz = _diagnostics["sample_trajectories"][-1]
            costs_viz = _diagnostics["sample_costs"][-1]
            Ybar_for_viz = _diagnostics["Ybar_final"]  # normalized action sequence (H, 3)
            # Y0s_norm not stored in diagnostics; resample around Ybar_final to get real
            # action sequences for the cost-breakdown text file (cheap single forward pass).
            _, _, Y0s_viz = _sample_final_step(
                Ybar_for_viz, cost_fn, rollout_fn, eta0, nu0,
                action_center, action_scale,
                H, Nsample, beta0, betaT, Ndiffuse, seed,
            )
        else:
            trajs_viz, costs_viz, Y0s_viz = _sample_final_step(
                Ybar_final, cost_fn, rollout_fn, eta0, nu0,
                action_center, action_scale,
                H, Nsample, beta0, betaT, Ndiffuse, seed,
            )
            Ybar_for_viz = Ybar_final
        _save_mbd_replan_viz(
            viz_dir=viz_dir,
            replan_idx=replan_idx,
            trajs=trajs_viz,
            costs=costs_viz,
            Y0s_norm=Y0s_viz,
            Ybar_final=Ybar_for_viz,
            rollout_fn=rollout_fn,
            eta0=eta0,
            nu0=nu0,
            action_center=action_center,
            action_scale=action_scale,
            components_fn=components_fn,
            cost_map=cost_map,
            goal_world=goal,
            costmap_weight=costmap_weight,
            costmap_normalizer=costmap_normalizer,
            length_weight=length_weight,
            goal_weight=goal_weight,
            smoothness_weight=smoothness_weight,
            world_polygons=viz_world_polygons,
            thicknesses=viz_thicknesses,
        )

    # ── Optional: denoising movie ─────────────────────────────────────
    if denoising_movie_dir is not None and _diagnostics is not None:
        import os
        os.makedirs(denoising_movie_dir, exist_ok=True)
        movie_path = os.path.join(denoising_movie_dir, f"replan_{replan_idx:04d}_denoising.gif")
        _xlim = (cost_map.x_origin, cost_map.x_origin + cost_map.width)
        _ylim = (cost_map.y_origin, cost_map.y_origin + cost_map.height)
        _generate_denoising_movie_from_diagnostics(
            diag=_diagnostics,
            cost_map=cost_map,
            world_polygons=viz_world_polygons,
            thicknesses=viz_thicknesses,
            goal_y_line=goal_y_line,
            goal=goal,
            x_min=_xlim[0], x_max=_xlim[1],
            y_min=_ylim[0], y_max=_ylim[1],
            replan_idx=replan_idx,
            output_path=movie_path,
            ship_length=L,
            ship_beam=B,
        )

    # ── Roll out final trajectory ─────────────────────────────────────
    rollout_jit = jax.jit(rollout_fn)
    if use_best_sample and _diagnostics is not None and _diagnostics["best_sample_actions"] is not None:
        # Use the best individual sample action sequence found across all denoising steps
        best_actions_norm = jnp.array(_diagnostics["best_sample_actions"])
        u_phys_best = best_actions_norm * action_scale + action_center
        traj = np.array(rollout_jit(eta0, nu0, u_phys_best))
        print(
            f"  [MBD best-sample] using best sample (cost={_diagnostics['best_sample_cost']:.1f}) "
            f"instead of Ybar_final (cost={float(cost_fn(Ybar_final)):.1f})",
            flush=True,
        )
    elif return_diagnostics:
        # Final trajectory is already the last mean trajectory captured in diagnostics
        traj = _diagnostics["mean_trajectories"][-1]
    else:
        u_phys = Ybar_final * action_scale + action_center
        traj = np.array(rollout_jit(eta0, nu0, u_phys))

    # ── Optionally compute raw (unweighted) cost components ───────────
    # Only supported on the fast path (not return_diagnostics) since we need
    # Ybar_final as an action sequence.
    _cost_components = None
    if return_cost_components and not return_diagnostics:
        comp_fn = jax.jit(make_cost_components_fn(
            grid, meta, goal, rollout_fn, eta0, nu0,
            action_center, action_scale,
            ship_length=L, ship_beam=B,
        ))
        comps = np.array(comp_fn(Ybar_final))
        _cost_components = {
            "costmap_cost": float(comps[0]),
            "path_length": float(comps[1]),
            "goal_dist": float(comps[2]),
            "smoothness": float(comps[3]),
            "costmap_normalizer": costmap_normalizer,
            "weights": {
                "costmap": costmap_weight,
                "length": length_weight,
                "goal": goal_weight,
                "smoothness": smoothness_weight,
            },
            # total_weighted uses the *effective* costmap weight (divided by normalizer)
            # so it matches the actual cost evaluated by make_cost_fn.
            "total_weighted": float(
                (costmap_weight / costmap_normalizer) * comps[0]
                + length_weight * comps[1]
                + goal_weight * comps[2]
                + smoothness_weight * comps[3]
            ),
        }

    if return_diagnostics:
        # Attach the warm-start rollout so callers can compare it against the
        # RRT path and the final MBD trajectory.
        if _diagnostics is not None and Ybar_init is not None:
            try:
                _ws_arr = jnp.array(Ybar_init, dtype=jnp.float32)
                if _ws_arr.ndim == 2:
                    _ws_phys = _ws_arr * action_scale + action_center
                    _ws_traj = np.array(jax.jit(rollout_fn)(eta0, nu0, _ws_phys))
                    _diagnostics["warmstart_trajectory"] = _ws_traj
            except Exception:
                pass
        return traj, _diagnostics
    if return_cost_components:
        return traj, _cost_components
    return traj
