"""
JAX Model Predictive Path Integral (MPPI) planner for 3-DOF vessel trajectory optimisation.

Optimises body-frame control sequences [X, Y, N] via importance-weighted sampling.
Controls are rolled out through Fossen-style 3-DOF dynamics, evaluated against
a costmap for ice avoidance + goal reaching. Reuses cost function and dynamics
from the MBD planner but uses standard MPPI update (fixed-variance sampling +
exponential importance weighting).

Used by ``medea_ice_ship.planning.optimize_with_mppi`` for the
``--planner mppi`` pipeline in ``run_medea_ship_fracture.py``.
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
    _warm_start_from_path,
    _jax_backend_info,
)


def run_mppi(
    cost_fn,
    H: int,
    Nsample: int,
    n_iterations: int,
    temperature: float,
    noise_sigma: float,
    seed: int,
    v_init: Optional[jnp.ndarray] = None,
    init_surge_bias: float = 0.7,
) -> Tuple[jnp.ndarray, float, float]:
    """
    Run MPPI in normalised action space.

    At each iteration: sample us = v + noise, evaluate costs, update
    v = weighted_average(us) with omega = softmax(-(costs - min_cost) / temperature).

    Parameters
    ----------
    cost_fn : callable
        J(u_norm) where u_norm has shape (H, 3).
    H : int
        Control horizon (number of action steps).
    Nsample : int
        Number of trajectory samples per iteration.
    n_iterations : int
        Number of MPPI warm-start iterations.
    temperature : float
        Inverse temperature (lambda); higher = sharper weighting.
    noise_sigma : float
        Std dev of exploration noise added to nominal control.
    seed : int
        Random seed.
    v_init : (H, 3), optional
        Initial nominal control sequence in [-1, 1]. If None, use constant surge bias.
    init_surge_bias : float
        Surge bias when v_init is None.

    Returns
    -------
    v_final : (H, 3) optimised nominal actions in [-1, 1]
    cost_mean : float — mean cost at final iteration
    cost_min : float — min cost at final iteration
    """
    if v_init is not None:
        v = jnp.array(v_init, dtype=jnp.float32)
        v = jnp.clip(v, -1.0, 1.0)
    else:
        v = jnp.zeros((H, 3))
        v = v.at[:, 0].set(init_surge_bias)

    rng = jax.random.PRNGKey(seed)
    cost_batch = jax.vmap(cost_fn)

    def _mppi_step(carry, _unused):
        v_c, rng_c = carry
        rng_next, rng_noise = jax.random.split(rng_c)
        noise = jax.random.normal(rng_noise, (Nsample, H, 3)) * noise_sigma
        us = v_c[None, :, :] + noise
        us = jnp.clip(us, -1.0, 1.0)

        costs = cost_batch(us)
        beta = jnp.min(costs)
        eta = jnp.sum(jnp.exp(-(costs - beta) / temperature))
        omega = (1.0 / eta) * jnp.exp(-(costs - beta) / temperature)

        v_new = jnp.einsum("n,nij->ij", omega, us)
        v_new = jnp.clip(v_new, -1.0, 1.0)

        return (v_new, rng_next), (costs.mean(), costs.min())

    @jax.jit
    def _run_mppi_loop(v_start, rng_start):
        carry = (v_start, rng_start)
        (v_final, _), metrics = jax.lax.scan(
            _mppi_step, carry, None, length=n_iterations
        )
        return v_final, metrics[0][-1], metrics[1][-1]

    v_final, cost_mean_val, cost_min_val = _run_mppi_loop(v, rng)

    return v_final, float(cost_mean_val), float(cost_min_val)


def jax_mppi_optimize(
    cost_map: Any,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg,
    start_nu: Optional[Tuple[float, float, float]] = None,
    lattice_path: Optional[np.ndarray] = None,
    warm_start: Optional[bool] = None,
    goal_y_line: Optional[float] = None,
    goal_y_line_penalty: float = 1e6,
) -> np.ndarray:
    """
    Optimize trajectory using JAX MPPI with 3-DOF dynamics.

    Uses the same cost function (accumulated costmap + arc length + goal)
    and dynamics as the MBD planner. The MPPI loop is compiled via
    ``jax.lax.scan`` for GPU efficiency.

    Parameters
    ----------
    cost_map : CostMap
        Costmap object (from ``build_costmap_from_ice_state``).
    start_pose : (x, y, psi)
        Starting pose [m, m, rad].
    goal : (x, y)
        Goal position [m, m].
    run_cfg : RunConfig
        Configuration (provides ship params and MPPI tuning).
    start_nu : (u, v, r), optional
        Initial body-frame velocities. Default (2, 0, 0).
    lattice_path : (N, 3), optional
        Lattice A* path used to warm-start (when warm_start=True).
    warm_start : bool, optional
        If True, seed initial control from lattice_path. Defaults to config.

    Returns
    -------
    trajectory : (H+1, 3) array of [x, y, psi] in world frame.
    """
    # ── Read parameters from config ───────────────────────────────────
    H = getattr(run_cfg, "planning_mppi_horizon", 80)
    Nsample = getattr(run_cfg, "planning_mppi_nsample", 1024)
    n_iterations = getattr(run_cfg, "planning_mppi_n_iterations", 5)
    temperature = getattr(run_cfg, "planning_mppi_temperature", 0.1)
    noise_sigma = getattr(run_cfg, "planning_mppi_noise_sigma", 0.3)
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
    n_substeps = getattr(run_cfg, "planning_jax_mbd_n_substeps", 5)

    if warm_start is None:
        warm_start = getattr(run_cfg, "planning_mppi_warm_start", True)

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

    # ── Warm-start from lattice path ──────────────────────────────────
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

    # ── Run MPPI ──────────────────────────────────────────────────────
    ws_label = "warm-started from A*" if v_init is not None else "cold start (no A* prior)"
    backend = _jax_backend_info()
    print(
        f"  JAX MPPI [{backend}]: H={H}, Nsample={Nsample}, n_iter={n_iterations}, "
        f"temp={temperature}, sigma={noise_sigma}, dt_action={dt_action}s, {ws_label}",
        flush=True,
    )
    import time as _time
    _t0 = _time.perf_counter()
    v_final, cost_mean, cost_min = run_mppi(
        cost_fn, H, Nsample, n_iterations,
        temperature, noise_sigma, seed,
        v_init=v_init,
        init_surge_bias=init_surge_bias,
    )
    _elapsed = _time.perf_counter() - _t0
    print(
        f"  JAX MPPI done ({_elapsed:.1f}s): cost_mean={cost_mean:.1f}, cost_min={cost_min:.1f}",
        flush=True,
    )

    # ── Roll out final trajectory ─────────────────────────────────────
    rollout_jit = jax.jit(rollout_fn)
    u_phys = v_final * action_scale + action_center
    traj = rollout_jit(eta0, nu0, u_phys)

    return np.array(traj)
