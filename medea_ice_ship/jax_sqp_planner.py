"""
JAX multiple-shooting SQP planner for 3-DOF vessel trajectory optimisation.

Uses Sequential Quadratic Programming with jaxproxqp to solve the QP subproblems.
Optimises over state and control sequences directly with dynamics continuity
constraints. Same cost structure (costmap + arc length + goal + smoothness)
as MBD/MPPI.

Used by ``medea_ice_ship.planning.optimize_with_sqp`` for the
``--planner sqp`` pipeline in ``run_medea_ship_fracture.py``.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Any, Callable

from medea_ice_ship.config import get_default_simulation_config
from medea_ice_ship.jax_mbd_planner import (
    _make_query,
    _warm_start_from_path,
    _jax_backend_info,
    make_dynamics_rollout,
)


# ── Dynamics: single action step (6D state in, 6D state out) ─────────────────

def make_dynamics_step(
    mass: float,
    Izz: float,
    dt_substep: float,
    n_substeps: int,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    ship_damping_xy: float = 0.02,
    ship_damping_psi: float = 0.15,
    added_mass_surge_frac: float = 0.05,
    added_mass_sway_frac: float = 0.5,
    added_mass_yaw_frac: float = 0.2,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Build single action-step dynamics f(state_6d, u_norm_3d) -> state_6d.

    Maps normalised control to physical forces via action_center/action_scale,
    integrates Fossen 3-DOF over n_substeps Euler substeps.
    JAX-differentiable for jacfwd/jacrev.
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
    _m11 = m11
    _m22 = m22
    _dt = dt_substep

    def _euler_step(state: jnp.ndarray, tau: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
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

    def _action_step(state: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        state_new, _ = jax.lax.scan(
            lambda s, _: _euler_step(s, tau), state, None, length=n_substeps,
        )
        return state_new

    def step(state_6d: jnp.ndarray, u_norm_3d: jnp.ndarray) -> jnp.ndarray:
        tau = u_norm_3d * action_scale + action_center
        return _action_step(state_6d, tau)

    return step


# ── Pack / unpack decision variables z ──────────────────────────────────────

def _pack_z(states: jnp.ndarray, controls: jnp.ndarray) -> jnp.ndarray:
    """
    Pack states (N+1, 6) and controls (N, 3) into interleaved z (9N+6,).
    z = [x_0(6), u_0(3), x_1(6), u_1(3), ..., x_{N-1}(6), u_{N-1}(3), x_N(6)]
    """
    N = controls.shape[0]
    parts = []
    for k in range(N):
        parts.append(states[k])
        parts.append(controls[k])
    parts.append(states[N])
    return jnp.concatenate(parts)


def _unpack_z(z: jnp.ndarray, N: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Unpack z into states (N+1, 6) and controls (N, 3)."""
    states = []
    controls = []
    idx = 0
    for k in range(N):
        states.append(z[idx : idx + 6])
        idx += 6
        controls.append(z[idx : idx + 3])
        idx += 3
    states.append(z[idx : idx + 6])
    return jnp.stack(states), jnp.stack(controls)


# ── Cost function on flattened z ────────────────────────────────────────────

def make_sqp_cost(
    grid: jnp.ndarray,
    meta: dict,
    goal_xy: jnp.ndarray,
    costmap_weight: float = 1.0,
    goal_weight: float = 0.1,
    length_weight: float = 1.0,
    smoothness_weight: float = 0.005,
    ship_length: float = 0.0,
    ship_beam: float = 0.0,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Build cost J(z) on flattened z. Same terms as MBD/MPPI.
    """
    query = _make_query(grid, meta)
    half_l = float(ship_length) / 2.0
    half_w = float(ship_beam) / 2.0
    use_hull = ship_length > 0 and ship_beam > 0

    if use_hull:
        _dx_body = jnp.array([0.0, half_l, half_l, -half_l, -half_l])
        _dy_body = jnp.array([0.0, half_w, -half_w, half_w, -half_w])

        def _point_cost(eta3: jnp.ndarray) -> jnp.ndarray:
            c, s = jnp.cos(eta3[2]), jnp.sin(eta3[2])
            wx = eta3[0] + c * _dx_body - s * _dy_body
            wy = eta3[1] + s * _dx_body + c * _dy_body
            return jnp.max(jax.vmap(query)(wx, wy))
    else:
        def _point_cost(eta3: jnp.ndarray) -> jnp.ndarray:
            return query(eta3[0], eta3[1])

    def cost_fn(z: jnp.ndarray) -> jnp.ndarray:
        N = (z.shape[0] - 6) // 9
        states, controls = _unpack_z(z, N)

        etas = states[:, :3]
        diffs = jnp.diff(etas[:, :2], axis=0)
        seg_lens = jnp.sqrt(jnp.sum(diffs ** 2, axis=1) + 1e-8)

        wp_costs = jax.vmap(_point_cost)(etas)
        seg_costmap = 0.5 * (wp_costs[:-1] + wp_costs[1:]) * seg_lens
        costmap_cost = seg_costmap.sum()

        path_length = seg_lens.sum()
        goal_dist = jnp.sqrt(jnp.sum((etas[-1, :2] - goal_xy) ** 2) + 1e-4)

        du = jnp.diff(controls, axis=0)
        jerk = jnp.mean(du ** 2)

        return (
            costmap_weight * costmap_cost
            + length_weight * path_length
            + goal_weight * goal_dist
            + smoothness_weight * jerk
        )

    return cost_fn


# ── QP assembly and SQP loop ────────────────────────────────────────────────

def _defect(z: jnp.ndarray, f_step: Callable, x_init: jnp.ndarray, N: int) -> jnp.ndarray:
    """Equality constraint violation: [x_0 - x_init; f(x_k,u_k) - x_{k+1} for k=0..N-1]."""
    states, controls = _unpack_z(z, N)
    defects = []
    defects.append(states[0] - x_init)
    for k in range(N):
        x_next_pred = f_step(states[k], controls[k])
        defects.append(x_next_pred - states[k + 1])
    return jnp.concatenate(defects)


def build_qp_data(
    z_bar: jnp.ndarray,
    cost_fn: Callable,
    f_step: Callable,
    x_init: jnp.ndarray,
    N: int,
    reg: float,
) -> Tuple[Any, jnp.ndarray, jnp.ndarray]:
    """
    Build QP subproblem in delta-space for SQP.
    Returns (QPModel, l_box, u_box) for the step dz.
    """
    _, QPModel, _ = _import_jaxproxqp()

    n = z_bar.shape[0]
    states_bar, controls_bar = _unpack_z(z_bar, N)

    grad_J = jax.grad(cost_fn)(z_bar)
    H_cost = jax.hessian(cost_fn)(z_bar)
    H_qp = jnp.array(H_cost) + reg * jnp.eye(n)
    H_qp = 0.5 * (H_qp + H_qp.T)
    g_qp = jnp.array(grad_J)

    defect = _defect(z_bar, f_step, x_init, N)
    b_eq = -defect

    jac_defect_x = jax.jacfwd(lambda z: _defect(z, f_step, x_init, N))(z_bar)
    A_eq = jnp.array(jac_defect_x)

    l_box = np.full(n, -1e9, dtype=np.float64)
    u_box = np.full(n, 1e9, dtype=np.float64)
    for k in range(N):
        idx = 6 + 9 * k + 6
        for j in range(3):
            l_box[idx + j] = max(-1.0 - float(controls_bar[k, j]), -1e9)
            u_box[idx + j] = min(1.0 - float(controls_bar[k, j]), 1e9)

    C_empty = np.zeros((0, n), dtype=np.float64)
    l_empty = np.zeros(0, dtype=np.float64)
    u_empty = np.zeros(0, dtype=np.float64)

    qp = QPModel.create(
        H=np.array(H_qp, dtype=np.float64),
        g=np.array(g_qp, dtype=np.float64),
        A=np.array(A_eq, dtype=np.float64),
        b=np.array(b_eq, dtype=np.float64),
        C=C_empty,
        u=u_empty,
        l=l_empty,
        l_box=np.array(l_box, dtype=np.float64),
        u_box=np.array(u_box, dtype=np.float64),
    )
    return qp, jnp.array(l_box), jnp.array(u_box)


def _import_jaxproxqp():
    """Import jaxproxqp; raise ImportError with helpful message on failure."""
    try:
        from jaxproxqp.jaxproxqp import JaxProxQP
        from jaxproxqp.qp_problems import QPModel
        from jaxproxqp.settings import Settings
        return JaxProxQP, QPModel, Settings
    except ImportError as e:
        raise ImportError(
            "jaxproxqp is required for --planner sqp. Install with: pip install -e /path/to/jaxproxqp\n"
            "If you see flax/KeyError with Python 3.14, use Python 3.11 or 3.12."
        ) from e


def run_sqp(
    f_step: Callable,
    cost_fn: Callable,
    x_init: jnp.ndarray,
    N: int,
    max_iters: int,
    reg: float,
    armijo_c: float = 1e-4,
    armijo_beta: float = 0.5,
    merit_mu: float = 100.0,
    tol_grad: float = 1e-4,
    tol_constraint: float = 1e-5,
    z_init: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, int, float, float]:
    """
    Run SQP loop. Returns (z_final, n_iters, cost_final, constraint_violation).
    """
    JaxProxQP, _, Settings = _import_jaxproxqp()

    if z_init is not None:
        z = jnp.array(z_init, dtype=jnp.float64)
    else:
        states = jnp.zeros((N + 1, 6), dtype=jnp.float64)
        states = states.at[0].set(jnp.array(x_init, dtype=jnp.float64))
        u_default = jnp.array([0.7, 0.0, 0.0], dtype=jnp.float64)
        for k in range(N):
            states = states.at[k + 1].set(f_step(states[k], u_default))
        controls = jnp.tile(u_default[None, :], (N, 1))
        z = _pack_z(states, controls)

    settings = Settings.default_float64()

    def merit(z_val: jnp.ndarray) -> jnp.ndarray:
        c = _defect(z_val, f_step, x_init, N)
        return cost_fn(z_val) + merit_mu * jnp.sum(jnp.abs(c))

    for it in range(max_iters):
        defect = _defect(z, f_step, x_init, N)
        constraint_viol = float(jnp.sqrt(jnp.sum(defect ** 2)))
        grad_norm = float(jnp.linalg.norm(jax.grad(cost_fn)(z)))

        if grad_norm < tol_grad and constraint_viol < tol_constraint:
            return z, it, float(cost_fn(z)), constraint_viol

        qp, l_box, u_box = build_qp_data(z, cost_fn, f_step, x_init, N, reg)
        solver = JaxProxQP(qp, settings)
        sol = solver.solve()
        dz = np.array(sol.x)

        alpha = 1.0
        phi0 = float(merit(z))
        dphi = float(np.dot(grad_J_ := np.array(jax.grad(cost_fn)(z)), dz))
        z_np = np.array(z)

        for _ in range(20):
            z_new = jnp.array(z_np + alpha * dz)
            phi_new = float(merit(z_new))
            if phi_new <= phi0 + armijo_c * alpha * dphi:
                break
            alpha *= armijo_beta

        z = jnp.array(z_np + alpha * dz)

    return z, max_iters, float(cost_fn(z)), float(jnp.sqrt(jnp.sum(_defect(z, f_step, x_init, N) ** 2)))


# ── Top-level optimizer ─────────────────────────────────────────────────────

def jax_sqp_optimize(
    cost_map: Any,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg: Any,
    start_nu: Optional[Tuple[float, float, float]] = None,
    lattice_path: Optional[np.ndarray] = None,
    warm_start: Optional[bool] = None,
) -> np.ndarray:
    """
    Optimize trajectory using JAX multiple-shooting SQP with jaxproxqp.
    Same interface as jax_mppi_optimize / jax_mbd_optimize.
    """
    _x64_prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        return _jax_sqp_optimize_impl(
            cost_map, start_pose, goal, run_cfg,
            start_nu, lattice_path, warm_start,
        )
    finally:
        jax.config.update("jax_enable_x64", _x64_prev)


def _jax_sqp_optimize_impl(
    cost_map: Any,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg: Any,
    start_nu: Optional[Tuple[float, float, float]],
    lattice_path: Optional[np.ndarray],
    warm_start: Optional[bool],
) -> np.ndarray:
    H = getattr(run_cfg, "planning_sqp_horizon", 40)
    max_iters = getattr(run_cfg, "planning_sqp_max_iters", 20)
    reg = getattr(run_cfg, "planning_sqp_reg", 1e-3)
    armijo_c = getattr(run_cfg, "planning_sqp_armijo_c", 1e-4)
    armijo_beta = getattr(run_cfg, "planning_sqp_armijo_beta", 0.5)
    merit_mu = getattr(run_cfg, "planning_sqp_merit_mu", 100.0)
    tol_grad = getattr(run_cfg, "planning_sqp_tol_grad", 1e-4)
    tol_constraint = getattr(run_cfg, "planning_sqp_tol_constraint", 1e-5)
    init_surge_bias = getattr(run_cfg, "planning_jax_mbd_init_surge_bias", 0.7)
    dt_substep = getattr(run_cfg, "planning_sqp_dt_substep", 1.0)
    n_substeps = getattr(run_cfg, "planning_sqp_n_substeps", 5)

    costmap_weight = getattr(run_cfg, "planning_jax_mbd_costmap_weight", 1.0)
    goal_weight = getattr(run_cfg, "planning_jax_mbd_goal_weight", 0.1)
    length_weight = getattr(run_cfg, "planning_jax_mbd_length_weight", 1.0)
    smoothness_weight = getattr(run_cfg, "planning_jax_mbd_smoothness_weight", 0.005)
    surge_max = getattr(run_cfg, "planning_jax_mbd_surge_max", 2e6)
    sway_max = getattr(run_cfg, "planning_jax_mbd_sway_max", 5e5)
    yaw_max = getattr(run_cfg, "planning_jax_mbd_yaw_max", 5e8)

    if warm_start is None:
        warm_start = getattr(run_cfg, "planning_sqp_warm_start", True)

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
    action_center = jnp.array([surge_max / 2, 0.0, 0.0], dtype=jnp.float64)
    action_scale = jnp.array([surge_max / 2, sway_max, yaw_max], dtype=jnp.float64)

    f_step = make_dynamics_step(
        mass=mass,
        Izz=Izz,
        dt_substep=dt_substep,
        n_substeps=n_substeps,
        action_center=action_center,
        action_scale=action_scale,
        ship_damping_xy=sim_cfg.ship_damping_xy,
        ship_damping_psi=sim_cfg.ship_damping_psi,
        added_mass_surge_frac=sim_cfg.mss_added_mass_surge_frac,
        added_mass_sway_frac=sim_cfg.mss_added_mass_sway_frac,
        added_mass_yaw_frac=sim_cfg.mss_added_mass_yaw_frac,
    )

    eta0 = jnp.array(start_pose, dtype=jnp.float64)
    nu0 = jnp.array(start_nu or (2.0, 0.0, 0.0), dtype=jnp.float64)
    x_init = jnp.concatenate([eta0, nu0])

    goal_xy = jnp.array(goal[:2], dtype=jnp.float32)
    cost_fn = make_sqp_cost(
        grid, meta, goal_xy,
        costmap_weight=costmap_weight,
        goal_weight=goal_weight,
        length_weight=length_weight,
        smoothness_weight=smoothness_weight,
        ship_length=L,
        ship_beam=B,
    )

    z_init = None
    if warm_start and lattice_path is not None and len(lattice_path) >= 3:
        lattice_path = np.asarray(lattice_path, dtype=float)
        u_init = _warm_start_from_path(lattice_path, H, init_surge_bias)
        state = jnp.array(x_init)
        states_list = [state]
        for k in range(H):
            state = f_step(state, jnp.array(u_init[k]))
            states_list.append(state)
        states_init = jnp.stack(states_list)
        z_init = _pack_z(
            jnp.array(states_init, dtype=jnp.float64),
            jnp.array(u_init, dtype=jnp.float64),
        )

    backend = _jax_backend_info()
    ws_label = "warm-started from A*" if z_init is not None else "cold start"
    print(
        f"  JAX SQP [{backend}]: H={H}, max_iter={max_iters}, reg={reg}, {ws_label}",
        flush=True,
    )
    import time as _time
    _t0 = _time.perf_counter()
    z_final, n_iters, cost_final, viol = run_sqp(
        f_step, cost_fn, x_init, H,
        max_iters=max_iters, reg=reg,
        armijo_c=armijo_c, armijo_beta=armijo_beta, merit_mu=merit_mu,
        tol_grad=tol_grad, tol_constraint=tol_constraint,
        z_init=z_init,
    )
    _elapsed = _time.perf_counter() - _t0
    print(
        f"  JAX SQP done ({_elapsed:.1f}s): iter={n_iters}, cost={cost_final:.1f}, viol={viol:.2e}",
        flush=True,
    )

    states_final, _ = _unpack_z(z_final, H)
    traj = np.array(states_final[:, :3])
    return traj
