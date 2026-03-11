"""
JAX gradient-based trajectory optimizer for 3-DOF vessel planning.

Optimises body-frame control sequences [surge, sway, yaw] via gradient descent
(Adam) on the cost J = costmap_integral + path_length + goal_distance.
Uses the same Fossen 3-DOF dynamics and costmap as the diffusion planner.

Standalone: works from a simple initial trajectory (e.g. straight line toward
goal) without needing A* warm start. Optionally accepts A* path for warm start.
"""

from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Any

from medea_ice_ship.config import get_default_simulation_config
from medea_ice_ship.jax_mbd_planner import (
    make_dynamics_rollout,
    make_cost_fn,
    _warm_start_from_path,
    _resample_path_by_arclength,
)


def _adam_step(
    u: jnp.ndarray,
    grad: jnp.ndarray,
    m: jnp.ndarray,
    v: jnp.ndarray,
    t: int,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single Adam update step."""
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m_new / (1 - beta1 ** (t + 1))
    v_hat = v_new / (1 - beta2 ** (t + 1))
    u_new = u - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return jnp.clip(u_new, -1.0, 1.0), m_new, v_new


def jax_gradient_optimize(
    cost_map: Any,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg,
    start_nu: Optional[Tuple[float, float, float]] = None,
    lattice_path: Optional[np.ndarray] = None,
    warm_start: Optional[bool] = None,
) -> np.ndarray:
    """
    Optimize trajectory using gradient descent (Adam) on control sequence.

    Uses JAX autodiff to compute d(cost)/d(actions) and iteratively refines
    the trajectory. Costmap gradients pull the path away from high-cost ice.
    Works standalone from a straight-line init, or with optional A* warm start.

    Parameters
    ----------
    cost_map : CostMap
        Costmap from build_costmap_from_ice_state.
    start_pose : (x, y, psi)
        Starting pose [m, m, rad].
    goal : (x, y)
        Goal position [m, m].
    run_cfg : RunConfig
        Config for ship params and planning_gradient_* / planning_jax_mbd_*.
    start_nu : (u, v, r), optional
        Initial body velocities. Default (2, 0, 0).
    lattice_path : (N, 3), optional
        A* path for warm start. If None, cold start from straight line.
    warm_start : bool, optional
        Use lattice_path to initialize controls. Default True when path provided.

    Returns
    -------
    trajectory : (H+1, 3) [x, y, psi] in world frame.
    """
    H = getattr(run_cfg, "planning_jax_mbd_horizon", 300)
    n_iters = getattr(run_cfg, "planning_gradient_n_iters", 100)
    lr = getattr(run_cfg, "planning_gradient_lr", 0.05)
    init_surge_bias = getattr(run_cfg, "planning_jax_mbd_init_surge_bias", 0.7)
    costmap_weight = getattr(run_cfg, "planning_jax_mbd_costmap_weight", 1.0)
    goal_weight = getattr(run_cfg, "planning_jax_mbd_goal_weight", 0.1)
    length_weight = getattr(run_cfg, "planning_jax_mbd_length_weight", 1.0)
    # Gradient planner: optional overrides so costmap can dominate (avoids straight-line local minimum).
    costmap_weight = getattr(run_cfg, "planning_gradient_costmap_weight", costmap_weight)
    goal_weight = getattr(run_cfg, "planning_gradient_goal_weight", goal_weight)
    length_weight = getattr(run_cfg, "planning_gradient_length_weight", length_weight)
    smoothness_weight = getattr(run_cfg, "planning_jax_mbd_smoothness_weight", 0.005)
    surge_max = getattr(run_cfg, "planning_jax_mbd_surge_max", 2e6)
    sway_max = getattr(run_cfg, "planning_jax_mbd_sway_max", 5e5)
    yaw_max = getattr(run_cfg, "planning_jax_mbd_yaw_max", 5e8)
    dt_substep = getattr(run_cfg, "planning_jax_mbd_dt_substep", 1.0)
    n_substeps = getattr(run_cfg, "planning_jax_mbd_n_substeps", 5)

    if warm_start is None:
        warm_start = lattice_path is not None and len(lattice_path) >= 3

    # Smooth the costmap so gradients are non-zero (raw grid is piecewise, giving zero gradient in cell interiors).
    grid_np = np.asarray(cost_map.cost_map, dtype=np.float32)
    try:
        from scipy.ndimage import gaussian_filter
        sigma = getattr(run_cfg, "planning_gradient_costmap_smooth_sigma", 3.0)
        grid_np = gaussian_filter(grid_np, sigma=sigma, mode="nearest")
    except Exception:
        pass
    grid = jnp.array(grid_np, dtype=jnp.float32)
    meta = {
        "x_origin": float(cost_map.x_origin),
        "y_origin": float(cost_map.y_origin),
        "resolution": float(cost_map.resolution),
        "rows": int(cost_map.rows),
        "cols": int(cost_map.cols),
    }

    mass = run_cfg.ship_displacement
    L = run_cfg.ship_length
    B = run_cfg.ship_beam
    Izz = (1.0 / 12.0) * mass * (L ** 2 + B ** 2)

    sim_cfg = get_default_simulation_config()
    rollout_fn, _, _ = make_dynamics_rollout(
        mass=mass, Izz=Izz,
        dt_substep=dt_substep, n_substeps=n_substeps,
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

    dt_action = dt_substep * n_substeps
    u_init = np.zeros((H, 3))
    u_init[:, 0] = init_surge_bias

    if warm_start and lattice_path is not None and len(lattice_path) >= 3:
        lattice_path = np.asarray(lattice_path, dtype=float)
        coverage_distance = H * dt_action * 4.0
        seg_lens = np.linalg.norm(np.diff(lattice_path[:, :2], axis=0), axis=1)
        s = np.zeros(len(lattice_path))
        s[1:] = np.cumsum(seg_lens)
        cutoff_idx = max(min(int(np.searchsorted(s, coverage_distance, side="right")), len(lattice_path)), 2)
        truncated = lattice_path[:cutoff_idx]
        u_init = _warm_start_from_path(truncated, H, init_surge_bias)
        init_label = "warm-started from A*"
    else:
        goal_dx = goal[0] - start_pose[0]
        goal_dy = goal[1] - start_pose[1]
        desired_heading = np.arctan2(goal_dy, goal_dx)
        # Small random perturbation to yaw so we don't start in a straight-line local minimum
        # where costmap gradient can be zero (e.g. in flat high-cost regions).
        rng = np.random.default_rng(getattr(run_cfg, "default_seed", 42))
        perturb = 0.15 * (rng.random(H, dtype=np.float32) - 0.5)
        for t in range(H):
            err = (desired_heading - 0.0 + np.pi) % (2 * np.pi) - np.pi
            u_init[t, 2] = np.clip(err * 0.5 + perturb[t], -0.8, 0.8)
        init_label = "cold start (straight toward goal + small yaw perturbation)"

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

    grad_fn = jax.jit(jax.grad(cost_fn))
    cost_fn_jit = jax.jit(cost_fn)

    u = jnp.array(u_init, dtype=jnp.float32)
    m = jnp.zeros_like(u)
    v = jnp.zeros_like(u)

    try:
        dev = jax.devices()[0]
        backend = f"{dev.platform.upper()} ({getattr(dev, 'device_kind', '')})"
    except Exception:
        backend = "unknown"

    print(
        f"  JAX Gradient [{backend}]: H={H}, n_iters={n_iters}, lr={lr}, "
        f"dt_action={dt_action}s, {init_label}",
        flush=True,
    )
    print(
        f"  Cost weights: costmap={costmap_weight}, length={length_weight}, goal={goal_weight}",
        flush=True,
    )
    try:
        from scipy.ndimage import gaussian_filter
        sigma = getattr(run_cfg, "planning_gradient_costmap_smooth_sigma", 3.0)
        print(f"  Costmap smoothed with Gaussian sigma={sigma} cells", flush=True)
    except Exception:
        pass
    t0 = time.perf_counter()

    for i in range(n_iters):
        g = grad_fn(u)
        u, m, v = _adam_step(u, g, m, v, i, lr=lr)
        if (i + 1) % 20 == 0 or i == 0:
            c = float(cost_fn_jit(u))
            print(f"    iter {i+1}/{n_iters} cost={c:.1f}", flush=True)

    elapsed = time.perf_counter() - t0
    cost_final = float(cost_fn_jit(u))
    print(f"  JAX Gradient done ({elapsed:.1f}s): final cost={cost_final:.1f}", flush=True)

    u_phys = u * action_scale + action_center
    traj = rollout_fn(eta0, nu0, u_phys)
    return np.array(traj)
