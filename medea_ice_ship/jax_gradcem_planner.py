"""
JAX GradCEM (Gradient + Cross-Entropy Method) planner for 3-DOF vessel trajectory optimisation.

Port of the L4DC 2020 algorithm: combines gradient-based optimization (RMSprop) with
CEM elite selection and resampling. Optimises body-frame control sequences [X, Y, N]
through Fossen-style 3-DOF dynamics, evaluated against a costmap. Reuses cost function
and dynamics from the MBD planner.

Used by ``medea_ice_ship.planning.optimize_with_gradcem`` for the
``--planner gradcem`` pipeline in ``run_medea_ship_fracture.py``.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Optional, Tuple, Any

from medea_ice_ship.config import get_default_simulation_config
from medea_ice_ship.jax_mbd_planner import (
    make_dynamics_rollout,
    make_cost_fn,
    _warm_start_from_path,
    _jax_backend_info,
)


def run_gradcem(
    cost_fn,
    H: int,
    K: int,
    top_K: int,
    opt_iters: int,
    lr: float,
    grad_clip: bool,
    max_grad_norm: float,
    sigma_init: float,
    seed: int,
    mu_init: Optional[jnp.ndarray] = None,
    init_surge_bias: float = 0.7,
) -> Tuple[jnp.ndarray, float, float]:
    """
    Run GradCEM in normalised action space.

    Each iteration: sample K trajectories from N(mu, sigma), evaluate costs, gradient
    step (RMSprop), select top_K elites, refit belief, resample bottom (K - top_K).

    Parameters
    ----------
    cost_fn : callable
        J(u_norm) where u_norm has shape (H, 3). Lower is better.
    H : int
        Control horizon (action steps).
    K : int
        Number of trajectory samples per iteration.
    top_K : int
        Number of elites for CEM belief update.
    opt_iters : int
        Number of optimization iterations.
    lr : float
        RMSprop learning rate.
    grad_clip : bool
        Whether to clip per-sample gradient norms.
    max_grad_norm : float
        Max gradient norm when grad_clip=True.
    sigma_init : float
        Initial std for sampling (belief starts as N(mu_init, sigma_init)).
    seed : int
        Random seed.
    mu_init : (H, 3), optional
        Initial mean for belief. If None, use constant surge bias.
    init_surge_bias : float
        Surge bias when mu_init is None.

    Returns
    -------
    best_actions : (H, 3) best action sequence in [-1, 1]
    cost_mean : float
    cost_min : float
    """
    if mu_init is not None:
        mu = jnp.array(mu_init, dtype=jnp.float32)
        mu = jnp.clip(mu, -1.0, 1.0)
    else:
        mu = jnp.zeros((H, 3))
        mu = mu.at[:, 0].set(init_surge_bias)
    sigma = jnp.ones((H, 3)) * sigma_init

    rng = jax.random.PRNGKey(seed)
    cost_batch = jax.vmap(cost_fn)

    def loss_fn(actions: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(cost_batch(actions))

    grad_fn = jax.grad(loss_fn)
    optimizer = optax.rmsprop(learning_rate=lr)

    # Sample initial actions
    rng, rng_s = jax.random.split(rng)
    noise = jax.random.normal(rng_s, (K, H, 3))
    actions = mu[None, :, :] + sigma[None, :, :] * noise
    actions = jnp.clip(actions, -1.0, 1.0)

    opt_state = optimizer.init(actions)

    for _ in range(opt_iters):
        # Gradient step
        grad = grad_fn(actions)
        if grad_clip:
            epsilon = 1e-6
            grad_norm = jnp.linalg.norm(grad, axis=(1, 2), keepdims=True) + epsilon
            grad = grad / grad_norm * jnp.minimum(grad_norm, max_grad_norm)
        updates, opt_state = optimizer.update(grad, opt_state)
        actions = optax.apply_updates(actions, updates)
        actions = jnp.clip(actions, -1.0, 1.0)

        # Evaluate costs (lower is better)
        costs = cost_batch(actions)

        # CEM: select top_K elites (lowest cost)
        elite_idx = jnp.argsort(costs)[:top_K]
        elite_actions = actions[elite_idx]
        mu = jnp.mean(elite_actions, axis=0)
        sigma = jnp.std(elite_actions, axis=0, ddof=0) + 1e-6

        # Resample bottom (K - top_K) from updated belief
        k_resamp = K - top_K
        if k_resamp > 0:
            rng, rng_s = jax.random.split(rng)
            resamp_noise = jax.random.normal(rng_s, (k_resamp, H, 3))
            resamp_actions = mu[None, :, :] + sigma[None, :, :] * resamp_noise
            resamp_actions = jnp.clip(resamp_actions, -1.0, 1.0)
            worst_idx = jnp.argsort(costs)[-k_resamp:]
            actions = actions.at[worst_idx].set(resamp_actions)

    costs = cost_batch(actions)
    best_idx = jnp.argmin(costs)
    best_actions = actions[best_idx]
    return best_actions, float(jnp.mean(costs)), float(jnp.min(costs))


def jax_gradcem_optimize(
    cost_map: Any,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg: Any,
    start_nu: Optional[Tuple[float, float, float]] = None,
    lattice_path: Optional[np.ndarray] = None,
    warm_start: Optional[bool] = None,
) -> np.ndarray:
    """
    Optimize trajectory using JAX GradCEM with 3-DOF dynamics.

    Uses the same cost function (accumulated costmap + arc length + goal) and dynamics
    as MBD/MPPI. Combines gradient descent with CEM elite selection.

    Returns
    -------
    trajectory : (H+1, 3) array of [x, y, psi] in world frame.
    """
    H = getattr(run_cfg, "planning_gradcem_horizon", 80)
    K = getattr(run_cfg, "planning_gradcem_nsample", 512)
    top_K = getattr(run_cfg, "planning_gradcem_top_samples", 64)
    opt_iters = getattr(run_cfg, "planning_gradcem_opt_iters", 10)
    lr = getattr(run_cfg, "planning_gradcem_lr", 0.1)
    grad_clip = getattr(run_cfg, "planning_gradcem_grad_clip", True)
    max_grad_norm = getattr(run_cfg, "planning_gradcem_max_grad_norm", 1.0)
    sigma_init = getattr(run_cfg, "planning_gradcem_noise_sigma", 1.0)
    init_surge_bias = getattr(run_cfg, "planning_jax_mbd_init_surge_bias", 0.7)
    seed = getattr(run_cfg, "planning_diffusion_seed", None) or 42

    costmap_weight = getattr(run_cfg, "planning_jax_mbd_costmap_weight", 1.0)
    goal_weight = getattr(run_cfg, "planning_jax_mbd_goal_weight", 0.1)
    length_weight = getattr(run_cfg, "planning_jax_mbd_length_weight", 1.0)
    smoothness_weight = getattr(run_cfg, "planning_jax_mbd_smoothness_weight", 0.005)
    surge_max = getattr(run_cfg, "planning_jax_mbd_surge_max", 2e6)
    sway_max = getattr(run_cfg, "planning_jax_mbd_sway_max", 5e5)
    yaw_max = getattr(run_cfg, "planning_jax_mbd_yaw_max", 5e8)
    dt_substep = getattr(run_cfg, "planning_jax_mbd_dt_substep", 1.0)
    n_substeps = getattr(run_cfg, "planning_jax_mbd_n_substeps", 5)

    if warm_start is None:
        warm_start = getattr(run_cfg, "planning_gradcem_warm_start", True)

    grid = jnp.array(cost_map.cost_map, dtype=jnp.float32)
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

    mu_init = None
    if warm_start and lattice_path is not None and len(lattice_path) >= 3:
        lattice_path = np.asarray(lattice_path, dtype=float)
        dt_action = dt_substep * n_substeps
        speed_estimate = 4.0
        coverage_distance = H * dt_action * speed_estimate

        seg_lens = np.linalg.norm(np.diff(lattice_path[:, :2], axis=0), axis=1)
        s = np.zeros(len(lattice_path))
        s[1:] = np.cumsum(seg_lens)

        cutoff_idx = int(np.searchsorted(s, coverage_distance, side="right"))
        cutoff_idx = max(min(cutoff_idx, len(lattice_path)), 2)

        truncated = lattice_path[:cutoff_idx]
        mu_init = _warm_start_from_path(truncated, H, init_surge_bias)
        mu_init = jnp.array(mu_init, dtype=jnp.float32)

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

    ws_label = "warm-started from A*" if mu_init is not None else "cold start"
    backend = _jax_backend_info()
    print(
        f"  JAX GradCEM [{backend}]: H={H}, K={K}, top_K={top_K}, opt_iters={opt_iters}, "
        f"lr={lr}, sigma_init={sigma_init}, {ws_label}",
        flush=True,
    )
    import time as _time
    _t0 = _time.perf_counter()
    best_actions, cost_mean, cost_min = run_gradcem(
        cost_fn, H, K, top_K, opt_iters,
        lr=lr,
        grad_clip=grad_clip,
        max_grad_norm=max_grad_norm,
        sigma_init=sigma_init,
        seed=seed,
        mu_init=mu_init,
        init_surge_bias=init_surge_bias,
    )
    _elapsed = _time.perf_counter() - _t0
    print(
        f"  JAX GradCEM done ({_elapsed:.1f}s): cost_mean={cost_mean:.1f}, cost_min={cost_min:.1f}",
        flush=True,
    )

    rollout_jit = jax.jit(rollout_fn)
    u_phys = best_actions * action_scale + action_center
    traj = rollout_jit(eta0, nu0, u_phys)

    return np.array(traj)
