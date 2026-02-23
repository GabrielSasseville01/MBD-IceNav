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
from typing import Optional, Tuple, Any

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

    M3_inv_diag = jnp.array([1.0 / m11, 1.0 / m22, 1.0 / m33])
    D3_diag = jnp.array([
        mass * ship_damping_xy,
        mass * ship_damping_xy,
        Izz * ship_damping_psi,
    ])

    _m11 = jnp.float32(m11)
    _m22 = jnp.float32(m22)
    _dt = jnp.float32(dt_substep)

    def _euler_step(state, tau):
        """Single Euler sub-step.  state = [eta(3); nu(3)]."""
        eta, nu = state[:3], state[3:6]
        C_nu = jnp.stack([
            -_m22 * nu[1] * nu[2],
             _m11 * nu[0] * nu[2],
            (_m22 - _m11) * nu[0] * nu[1],
        ])
        nu_dot = M3_inv_diag * (tau - C_nu - D3_diag * nu)
        nu_new = nu + _dt * nu_dot
        c = jnp.cos(eta[2])
        s = jnp.sin(eta[2])
        vx = c * nu_new[0] - s * nu_new[1]
        vy = s * nu_new[0] + c * nu_new[1]
        psi_new = (eta[2] + _dt * nu_new[2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        eta_new = jnp.stack([eta[0] + _dt * vx, eta[1] + _dt * vy, psi_new])
        return jnp.concatenate([eta_new, nu_new]), None

    def _action_step(state, tau):
        """Hold one action for n_substeps Euler sub-steps."""
        state_new, _ = jax.lax.scan(
            lambda s, _: _euler_step(s, tau), state, None, length=n_substeps,
        )
        return state_new, state_new[:3]

    def rollout(eta0, nu0, u_seq):
        """Forward-roll dynamics.  u_seq: (H, 3) → traj: (H+1, 3)."""
        state0 = jnp.concatenate([eta0, nu0])
        _, traj = jax.lax.scan(_action_step, state0, u_seq)
        return jnp.concatenate([eta0[None, :], traj], axis=0)

    M3_np = np.diag([m11, m22, m33])
    D3_np = np.diag([float(d) for d in D3_diag])
    return rollout, M3_np, D3_np


# ── Costmap bilinear interpolation (JAX) ──────────────────────────────────

def _make_query(grid: jnp.ndarray, meta: dict):
    """Return a bilinear-interpolation query for a JAX costmap grid."""
    xo = float(meta["x_origin"])
    yo = float(meta["y_origin"])
    res = float(meta["resolution"])
    nrows = int(meta["rows"])
    ncols = int(meta["cols"])

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
            (r0 < 0) | (r0 + 1 >= nrows) | (c0 < 0) | (c0 + 1 >= ncols),
            1e4, 0.0,
        )
        val = (
            grid[r0c, c0c] * (1 - fr) * (1 - fc)
            + grid[r0c, c1c] * (1 - fr) * fc
            + grid[r1c, c0c] * fr * (1 - fc)
            + grid[r1c, c1c] * fr * fc
        )
        return val + oob

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


# ── Warm-start from lattice path (numpy, pre-JIT) ────────────────────────

def _warm_start_from_path(
    lattice_path: Optional[np.ndarray],
    H: int,
    init_surge_bias: float = 0.7,
) -> np.ndarray:
    """
    Compute initial normalised actions (H, 3) from a lattice A* path.

    Provides an informed starting point for MBD by setting:
      - surge: constant forward bias
      - yaw:   proportional to heading changes along the path
      - sway:  zero (let MBD optimise lateral forces)

    The lattice path only seeds the optimisation; it is NOT used as a cost
    penalty — MBD is free to deviate if costmap + goal suggest a better route.
    """
    Ybar = np.zeros((H, 3))
    Ybar[:, 0] = init_surge_bias

    if lattice_path is None or len(lattice_path) < 3:
        return Ybar

    path_ref = _resample_path_by_arclength(lattice_path, H + 1)

    for t in range(H):
        dx = path_ref[t + 1, 0] - path_ref[t, 0]
        dy = path_ref[t + 1, 1] - path_ref[t, 1]
        desired = np.arctan2(dy, dx)
        current = path_ref[t, 2]
        err = (desired - current + np.pi) % (2 * np.pi) - np.pi
        Ybar[t, 2] = np.clip(err * 2.0, -0.8, 0.8)

    return Ybar


# ── Cost function (normalised actions → dynamics → costmap + goal) ────────

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
    goal_weight: float = 0.1,
    length_weight: float = 1.0,
    smoothness_weight: float = 0.005,
    ship_length: float = 0.0,
    ship_beam: float = 0.0,
):
    """
    Build cost J(u_norm) matching the A* lattice planner's cost structure.

    For each trajectory segment the cost is:
        arc_length  +  max_hull_costmap × segment_length
    summed (accumulated) over the full trajectory — identical to the lattice
    planner's ``_compute_path_cost``.  A small goal-distance term provides
    directional guidance (the lattice planner gets this from the A* heuristic).
    """
    query = _make_query(grid, meta)
    goal_xy = jnp.array(goal_world[:2], dtype=jnp.float32)

    half_l = float(ship_length) / 2.0
    half_w = float(ship_beam) / 2.0
    use_hull = ship_length > 0 and ship_beam > 0

    if use_hull:
        _dx_body = jnp.array([0.0, half_l, half_l, -half_l, -half_l])
        _dy_body = jnp.array([0.0, half_w, -half_w, half_w, -half_w])

        def _point_cost(point):
            c, s = jnp.cos(point[2]), jnp.sin(point[2])
            wx = point[0] + c * _dx_body - s * _dy_body
            wy = point[1] + s * _dx_body + c * _dy_body
            return jnp.max(jax.vmap(query)(wx, wy))
    else:
        def _point_cost(point):
            return query(point[0], point[1])

    def cost_fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)

        # Segment lengths (arc-length weighting, like A*)
        diffs = jnp.diff(traj[:, :2], axis=0)  # (H, 2)
        seg_lens = jnp.sqrt(jnp.sum(diffs ** 2, axis=1) + 1e-8)  # (H,)

        # Hull-aware costmap at waypoints (max over center + 4 corners)
        wp_costs = jax.vmap(_point_cost)(traj)  # (H+1,)

        # Accumulated costmap cost weighted by segment length (matches A*)
        # Use average of endpoints per segment × segment_length
        seg_costmap = 0.5 * (wp_costs[:-1] + wp_costs[1:]) * seg_lens
        costmap_cost = seg_costmap.sum()

        # Arc length (A*'s base distance cost)
        path_length = seg_lens.sum()

        # Small goal nudge (replaces A*'s heuristic-guided search)
        goal_dist = jnp.sqrt(jnp.sum((traj[-1, :2] - goal_xy) ** 2) + 1e-4)

        # Control smoothness
        du = jnp.diff(u_norm, axis=0)
        jerk = jnp.mean(du ** 2)

        return (
            costmap_weight * costmap_cost
            + length_weight * path_length
            + goal_weight * goal_dist
            + smoothness_weight * jerk
        )

    return cost_fn


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
        Ybar_i = jnp.zeros((H, 3))
        Ybar_i = Ybar_i.at[:, 0].set(init_surge_bias)

    if Ndiffuse <= 1:
        return Ybar_i, 0.0, 0.0

    rng = jax.random.PRNGKey(seed)

    betas = jnp.linspace(beta0, betaT, Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1.0 - alphas_bar)

    cost_batch = jax.vmap(cost_fn)

    def _reverse_step(carry, _unused):
        """Single reverse diffusion step (scan body — NOT individually JIT'd)."""
        i, rng_c, Ybar_c = carry
        Yi = Ybar_c * jnp.sqrt(alphas_bar[i])

        rng_c, rng_s = jax.random.split(rng_c)
        eps = jax.random.normal(rng_s, (Nsample, H, 3))
        Y0s = eps * sigmas[i] + Ybar_c
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        costs = cost_batch(Y0s)
        rews = -costs
        rew_mean = rews.mean()
        rew_std = jnp.maximum(rews.std(), 1e-6)
        logp0 = (rews - rew_mean) / rew_std / temperature
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
        Ybar_im1 = jnp.clip(Ybar_im1, -1.0, 1.0)

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
    goal_weight = getattr(run_cfg, "planning_jax_mbd_goal_weight", 0.1)
    length_weight = getattr(run_cfg, "planning_jax_mbd_length_weight", 1.0)
    smoothness_weight = getattr(run_cfg, "planning_jax_mbd_smoothness_weight", 0.005)
    surge_max = getattr(run_cfg, "planning_jax_mbd_surge_max", 2e6)
    sway_max = getattr(run_cfg, "planning_jax_mbd_sway_max", 5e5)
    yaw_max = getattr(run_cfg, "planning_jax_mbd_yaw_max", 5e8)
    dt_substep = getattr(run_cfg, "planning_jax_mbd_dt_substep", 1.0)
    n_substeps = getattr(run_cfg, "planning_jax_mbd_n_substeps", 5)
    init_surge_bias = getattr(run_cfg, "planning_jax_mbd_init_surge_bias", 0.7)
    seed = getattr(run_cfg, "planning_diffusion_seed", None) or 42

    if warm_start is None:
        warm_start = getattr(run_cfg, "planning_jax_mbd_warm_start", True)

    # ── Convert costmap to JAX ────────────────────────────────────────
    grid = jnp.array(cost_map.cost_map, dtype=jnp.float32)
    meta = {
        "x_origin": float(cost_map.x_origin),
        "y_origin": float(cost_map.y_origin),
        "resolution": float(cost_map.resolution),
        "rows": int(cost_map.rows),
        "cols": int(cost_map.cols),
    }

    # ── Ship dynamics (consistent with simulation) ────────────────────
    mass = run_cfg.ship_displacement
    L = run_cfg.ship_length
    B = run_cfg.ship_beam
    Izz = (1.0 / 12.0) * mass * (L ** 2 + B ** 2)

    sim_cfg = get_default_simulation_config()
    rollout_fn, _, _ = make_dynamics_rollout(
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
        Ybar_init = _warm_start_from_path(truncated, H, init_surge_bias)

    # ── Cost function (costmap + goal + effort + smoothness only) ─────
    cost_fn = make_cost_fn(
        grid, meta, goal, rollout_fn, eta0, nu0,
        action_center, action_scale,
        costmap_weight=costmap_weight,
        goal_weight=goal_weight,
        length_weight=length_weight,
        smoothness_weight=smoothness_weight,
        ship_length=L,
        ship_beam=B,
    )

    # ── Run MBD (importance-weighted score ascent) ────────────────────
    ws_label = "warm-started from A*" if Ybar_init is not None else "cold start (no A* prior)"
    backend = _jax_backend_info()
    print(
        f"  JAX MBD [{backend}]: H={H}, Nsample={Nsample}, Ndiffuse={Ndiffuse}, "
        f"temp={temperature}, dt_action={dt_action}s, {ws_label}",
        flush=True,
    )
    import time as _time
    _t0 = _time.perf_counter()
    Ybar_final, cost_mean, cost_min = run_mbd(
        cost_fn, H, Nsample, Ndiffuse, temperature,
        beta0, betaT, seed,
        Ybar_init=Ybar_init,
        init_surge_bias=init_surge_bias,
    )
    _elapsed = _time.perf_counter() - _t0
    print(
        f"  JAX MBD done ({_elapsed:.1f}s): cost_mean={cost_mean:.1f}, cost_min={cost_min:.1f}",
        flush=True,
    )

    # ── Roll out final trajectory ─────────────────────────────────────
    rollout_jit = jax.jit(rollout_fn)
    u_phys = Ybar_final * action_scale + action_center
    traj = rollout_jit(eta0, nu0, u_phys)

    return np.array(traj)
