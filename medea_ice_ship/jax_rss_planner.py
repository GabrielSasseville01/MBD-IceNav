"""
JAX Random Single Shooting (RSS) planner for 3-DOF vessel trajectory optimisation.

Samples Nsample action sequences uniformly from [-1, 1]^(H×3), evaluates all
via vmap, and returns the minimum-cost trajectory. No iterative updates.

Serves as the simplest possible baseline: a single-shot Monte Carlo search.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Any

from medea_ice_ship.config import get_default_simulation_config
from medea_ice_ship.jax_mbd_planner import (
    make_dynamics_rollout,
    make_cost_fn,
    make_goal_line_cost_fn,
    make_goal_arrival_x_cost_fn,
    make_terminal_goal_euclidean_cost_fn,
    make_y_bottom_cost_fn,
    _warm_start_from_path,
    _jax_backend_info,
)


def run_rss(
    cost_fn,
    H: int,
    Nsample: int,
    seed: int,
    v_init: Optional[jnp.ndarray] = None,
    init_surge_bias: float = 0.7,
) -> Tuple[jnp.ndarray, float, float]:
    """
    Random Single Shooting: sample Nsample action sequences, return the best.

    Draws (Nsample - 1) action sequences uniformly from [-1, 1] for sway/yaw
    and [0, 1] for surge (always forward). If v_init is given, it is included
    as one of the candidates.

    Parameters
    ----------
    cost_fn : callable
        J(u_norm) where u_norm has shape (H, 3), returns scalar cost.
    H : int
        Control horizon.
    Nsample : int
        Total number of candidates (including v_init if provided).
    seed : int
        Random seed.
    v_init : (H, 3), optional
        Warm-start nominal sequence included as one candidate.
    init_surge_bias : float
        Not used (kept for API consistency with other planners).

    Returns
    -------
    best_u : (H, 3) best normalised action sequence
    cost_mean : float — mean cost across all candidates
    cost_min : float — minimum cost
    """
    rng = jax.random.PRNGKey(seed)

    # Sample sway and yaw uniformly from [-1, 1]; surge from [0, 1] (always forward)
    rng, rng_surge, rng_sway, rng_yaw = jax.random.split(rng, 4)
    surge = jax.random.uniform(rng_surge, (Nsample, H, 1), minval=0.0, maxval=1.0,
                               dtype=jnp.float32)
    sway = jax.random.uniform(rng_sway, (Nsample, H, 1), minval=-1.0, maxval=1.0,
                              dtype=jnp.float32)
    yaw = jax.random.uniform(rng_yaw, (Nsample, H, 1), minval=-1.0, maxval=1.0,
                             dtype=jnp.float32)
    u_norms = jnp.concatenate([surge, sway, yaw], axis=2)  # (Nsample, H, 3)

    # Replace first sample with warm start if provided
    if v_init is not None:
        v = jnp.array(v_init, dtype=jnp.float32)
        v = jnp.clip(v, -1.0, 1.0)
        u_norms = u_norms.at[0].set(v)

    cost_batch = jax.jit(jax.vmap(cost_fn))
    costs = cost_batch(u_norms)

    best_idx = jnp.argmin(costs)
    best_u = u_norms[best_idx]
    return best_u, float(costs.mean()), float(costs.min())


def jax_rss_optimize(
    cost_map: Any,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg,
    start_nu: Optional[Tuple[float, float, float]] = None,
    lattice_path: Optional[np.ndarray] = None,
    warm_start: Optional[bool] = None,
    goal_y_line: Optional[float] = None,
    goal_y_line_penalty: float = 1e6,
    terminal_goal_distance_weight: float = 0.0,
    y_bottom: Optional[float] = None,
    y_bottom_penalty: float = 1e8,
) -> np.ndarray:
    """
    Optimize trajectory using Random Single Shooting.

    Samples Nsample random action sequences, evaluates all via JAX vmap,
    returns the minimum-cost trajectory rolled out through Fossen 3-DOF dynamics.

    Parameters
    ----------
    cost_map : CostMap
        Costmap object (from ``build_costmap_from_ice_state``).
    start_pose : (x, y, psi)
        Starting pose [m, m, rad].
    goal : (x, y)
        Goal position [m, m].
    run_cfg : RunConfig
        Configuration (provides ship params and RSS tuning knobs).
    start_nu : (u, v, r), optional
        Initial body-frame velocities. Default (2, 0, 0).
    lattice_path : (N, 3), optional
        Lattice A* path used for warm-start candidate (when warm_start=True).
    warm_start : bool, optional
        If True, include lattice warm-start as one candidate. Default from config.

    Returns
    -------
    trajectory : (H+1, 3) array of [x, y, psi] in world frame.
    """
    # ── Read parameters from config ───────────────────────────────────
    H = getattr(run_cfg, "planning_rss_horizon", 300)
    Nsample = getattr(run_cfg, "planning_rss_nsample", 4096)
    init_surge_bias = getattr(run_cfg, "planning_jax_mbd_init_surge_bias", 0.7)
    seed = getattr(run_cfg, "planning_diffusion_seed", None) or 42

    costmap_weight = getattr(run_cfg, "planning_jax_mbd_costmap_weight", 1.0)
    costmap_normalizer = getattr(run_cfg, "planning_jax_mbd_costmap_normalizer", 1.0)
    goal_weight = getattr(run_cfg, "planning_jax_mbd_goal_weight", 0.1)
    length_weight = getattr(run_cfg, "planning_jax_mbd_length_weight", 1.0)
    smoothness_weight = getattr(run_cfg, "planning_jax_mbd_smoothness_weight", 0.005)
    goal_x_weight = getattr(run_cfg, "planning_jax_mbd_goal_x_weight", 0.0)
    surge_max = getattr(run_cfg, "planning_jax_mbd_surge_max", 2e6)
    sway_max = getattr(run_cfg, "planning_jax_mbd_sway_max", 5e5)
    yaw_max = getattr(run_cfg, "planning_jax_mbd_yaw_max", 5e8)
    dt_substep = getattr(run_cfg, "planning_jax_mbd_dt_substep", 1.0)
    n_substeps = getattr(run_cfg, "planning_jax_mbd_n_substeps", 1)

    if warm_start is None:
        warm_start = getattr(run_cfg, "planning_rss_warm_start", True)

    # ── Convert costmap to JAX ────────────────────────────────────────
    grid = jnp.array(cost_map.cost_map, dtype=jnp.float32)
    meta = {
        "x_origin": float(cost_map.x_origin),
        "y_origin": float(cost_map.y_origin),
        "resolution": float(cost_map.resolution),
        "rows": int(cost_map.rows),
        "cols": int(cost_map.cols),
    }

    # ── Ship dynamics ─────────────────────────────────────────────────
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

    eta0 = jnp.array(start_pose, dtype=jnp.float32)
    nu0 = jnp.array(start_nu or (2.0, 0.0, 0.0), dtype=jnp.float32)

    action_center = jnp.array([surge_max / 2, 0.0, 0.0], dtype=jnp.float32)
    action_scale = jnp.array([surge_max / 2, sway_max, yaw_max], dtype=jnp.float32)

    # ── Warm-start candidate from lattice path ────────────────────────
    dt_action = dt_substep * n_substeps
    v_init = None
    if warm_start and lattice_path is not None and len(lattice_path) >= 3:
        lattice_path = np.asarray(lattice_path, dtype=float)
        speed_estimate = 4.0
        coverage_distance = H * dt_action * speed_estimate

        seg_lens = np.linalg.norm(np.diff(lattice_path[:, :2], axis=0), axis=1)
        s = np.zeros(len(lattice_path))
        s[1:] = np.cumsum(seg_lens)

        cutoff_idx = int(np.searchsorted(s, coverage_distance, side="right"))
        cutoff_idx = max(min(cutoff_idx, len(lattice_path)), 2)

        truncated = lattice_path[:cutoff_idx]
        v_init = _warm_start_from_path(truncated, H, init_surge_bias)
        v_init = jnp.array(v_init, dtype=jnp.float32)

    # ── Cost function ─────────────────────────────────────────────────
    cost_truncate_y = goal_y_line  # suppress cost after goal line crossing
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
        cost_truncate_y=cost_truncate_y,
    )
    if goal_y_line is not None:
        _gl_fn = make_goal_line_cost_fn(
            rollout_fn, eta0, nu0, action_center, action_scale,
            goal_y_line, goal_y_line_penalty,
        )
        _base_fn = cost_fn
        cost_fn = lambda u_norm: _base_fn(u_norm) + _gl_fn(u_norm)  # noqa: E731
    if goal_y_line is not None and goal_x_weight > 0.0:
        _gx_fn = make_goal_arrival_x_cost_fn(
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
    if y_bottom is not None and y_bottom_penalty > 0.0:
        _yb_fn = make_y_bottom_cost_fn(
            rollout_fn, eta0, nu0, action_center, action_scale,
            y_min=float(y_bottom), penalty=y_bottom_penalty,
        )
        _prev_yb = cost_fn
        cost_fn = lambda u_norm: _prev_yb(u_norm) + _yb_fn(u_norm)  # noqa: E731

    # ── Run RSS ───────────────────────────────────────────────────────
    ws_label = "with A* warm-start candidate" if v_init is not None else "cold start"
    backend = _jax_backend_info()
    print(
        f"  JAX RSS [{backend}]: H={H}, Nsample={Nsample}, "
        f"dt_action={dt_action}s, {ws_label}",
        flush=True,
    )
    import time as _time
    _t0 = _time.perf_counter()
    best_u, cost_mean, cost_min = run_rss(
        cost_fn, H, Nsample, seed,
        v_init=v_init,
        init_surge_bias=init_surge_bias,
    )
    _elapsed = _time.perf_counter() - _t0
    print(
        f"  JAX RSS done ({_elapsed:.1f}s): cost_mean={cost_mean:.1f}, cost_min={cost_min:.1f}",
        flush=True,
    )

    # ── Roll out final trajectory ─────────────────────────────────────
    rollout_jit = jax.jit(rollout_fn)
    u_phys = best_u * action_scale + action_center
    traj = rollout_jit(eta0, nu0, u_phys)

    return np.array(traj)
