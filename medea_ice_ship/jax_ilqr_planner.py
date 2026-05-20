"""
JAX Box-iLQR planner for 3-DOF vessel trajectory optimisation.

Implements the Box-constrained Iterative Linear Quadratic Regulator (Box-iLQR)
from: "Safe Optimal Control using Log Barrier Constrained iLQR"
(Abhijeet & Chakravorty, 2026, arXiv:2602.05046).

Uses logarithmic barrier functions to enforce control box constraints
u ∈ (-1, 1)^3 (normalised action space). Per-step cost uses the same
costmap/goal/length structure as MBD/MPPI. Dynamics Jacobians and cost
Hessians are computed via JAX automatic differentiation.

Algorithm:
  Outer loop: reduce barrier parameter sigma -> 0.
    Inner loop (up to max_iters):
      1. Linearise dynamics and quadratise cost along current trajectory.
      2. Backward pass (LQR sweep with barrier terms, jax.lax.scan).
      3. Forward pass with backtracking line search (Armijo condition).
    Until convergence (||k||_inf < tol).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Any
import time as _time

from medea_ice_ship.config import get_default_simulation_config
from medea_ice_ship.jax_mbd_planner import (
    _make_query,
    _warm_start_from_path,
    _jax_backend_info,
)
from medea_ice_ship.jax_sqp_planner import make_dynamics_step


# ── Per-step cost functions ───────────────────────────────────────────────

def make_stage_cost(
    grid: jnp.ndarray,
    meta: dict,
    dt_action: float,
    costmap_weight: float = 1.0,
    costmap_normalizer: float = 1.0,
    length_weight: float = 1.0,
    smoothness_weight: float = 0.005,
    ship_length: float = 0.0,
    ship_beam: float = 0.0,
):
    """
    Build per-step running cost l(x, u) for iLQR.

    l(x, u) = (costmap_w * costmap(eta) + length_w) * ||nu[:2]|| * dt_action
             + smooth_w * ||u||^2

    where x = [eta (3), nu (3)] is the 6D state and u is the normalised
    3D control.  This cost is differentiable via JAX autodiff.
    """
    query = _make_query(grid, meta)
    cw = jnp.float32(costmap_weight / costmap_normalizer)
    lw = jnp.float32(length_weight)
    sw = jnp.float32(smoothness_weight)
    dt = jnp.float32(dt_action)

    half_l = float(ship_length) / 2.0
    half_w = float(ship_beam) / 2.0
    use_hull = ship_length > 0 and ship_beam > 0

    if use_hull:
        _dx = jnp.array([0.0, half_l, half_l, -half_l, -half_l], dtype=jnp.float32)
        _dy = jnp.array([0.0, half_w, -half_w, half_w, -half_w], dtype=jnp.float32)

        def _point_cost(eta):
            c, s = jnp.cos(eta[2]), jnp.sin(eta[2])
            wx = eta[0] + c * _dx - s * _dy
            wy = eta[1] + s * _dx + c * _dy
            return jnp.max(jax.vmap(query)(wx, wy))
    else:
        def _point_cost(eta):
            return query(eta[0], eta[1])

    def stage_cost(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        eta = x[:3]
        nu = x[3:6]
        speed = jnp.sqrt(nu[0] ** 2 + nu[1] ** 2 + jnp.float32(1e-8))
        seg_len = speed * dt
        cm = _point_cost(eta)
        return (cw * cm + lw) * seg_len + sw * jnp.sum(u ** 2)

    return stage_cost


def make_terminal_cost(
    goal_xy: jnp.ndarray,
    goal_weight: float = 1.0,
    goal_y_line: Optional[float] = None,
):
    """
    Build terminal cost phi(x_T).

    When goal_y_line is None: phi = goal_weight * ||eta_T[:2] - goal||.
    When goal_y_line is set: phi = goal_weight * ||(dx, max(0, goal_y - y_T))||,
    which drives the ship to reach y >= goal_y_line while centering x on goal_x.
    This smooth formulation is differentiable everywhere, suitable for iLQR.
    """
    gw = jnp.float32(goal_weight)
    goal = jnp.array(goal_xy[:2], dtype=jnp.float32)
    _use_line = goal_y_line is not None
    _goal_y = jnp.float32(goal_y_line) if goal_y_line is not None else jnp.float32(0.0)

    def terminal_cost(x: jnp.ndarray) -> jnp.ndarray:
        eta = x[:3]
        dx = eta[0] - goal[0]
        if _use_line:
            dy = jnp.maximum(jnp.float32(0.0), _goal_y - eta[1])
        else:
            dy = eta[1] - goal[1]
        return gw * jnp.sqrt(dx * dx + dy * dy + jnp.float32(1e-4))

    return terminal_cost


def make_barrier(sigma: float):
    """
    Log-barrier for control box constraint u ∈ (-1, 1).

    B(u) = -sigma * sum_i [ log(1 - u_i) + log(1 + u_i) ]
    """
    sig = jnp.float32(sigma)

    def barrier(u: jnp.ndarray) -> jnp.ndarray:
        eps = jnp.float32(1e-8)
        return -sig * jnp.sum(
            jnp.log(jnp.maximum(1.0 - u, eps))
            + jnp.log(jnp.maximum(1.0 + u, eps))
        )

    return barrier


# ── Rollout helpers ───────────────────────────────────────────────────────

def _rollout_from_step(
    step_fn,
    x0: jnp.ndarray,
    u_seq: jnp.ndarray,
) -> jnp.ndarray:
    """Roll out step_fn from x0 given u_seq (H, 3) -> states (H+1, 6)."""
    def _scan_step(x, u):
        x_new = step_fn(x, u)
        return x_new, x_new

    _, xs = jax.lax.scan(_scan_step, x0, u_seq)
    return jnp.concatenate([x0[None], xs], axis=0)


# ── iLQR core ─────────────────────────────────────────────────────────────

def run_ilqr(
    step_fn,
    stage_cost_fn,
    terminal_cost_fn,
    x0: jnp.ndarray,
    u_init: jnp.ndarray,
    H: int,
    max_iters: int = 30,
    reg: float = 1e-4,
    sigma0: float = 0.1,
    barrier_tol: float = 1e-4,
    barrier_reduction: float = 0.1,
    armijo_c: float = 1e-4,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Run Box-iLQR (Algorithm 1 from Abhijeet & Chakravorty 2026).

    Parameters
    ----------
    step_fn : callable
        x_new = step_fn(x, u), differentiable (JAX).
    stage_cost_fn : callable
        l(x, u) -> scalar, WITHOUT barrier (barrier added internally).
    terminal_cost_fn : callable
        phi(x) -> scalar.
    x0 : (6,) initial state [eta (3), nu (3)].
    u_init : (H, 3) initial normalised control sequence, in (-0.99, 0.99).
    H : int
        Control horizon.
    max_iters : int
        Maximum iLQR inner iterations per barrier level.
    reg : float
        Tikhonov regularisation for Q_uu.
    sigma0 : float
        Initial log-barrier coefficient.
    barrier_tol : float
        Stop outer loop when sigma < this.
    barrier_reduction : float
        Multiplicative factor to reduce sigma each outer step.
    armijo_c : float
        Armijo condition constant for line search.

    Returns
    -------
    u_opt : (H, 3) optimised normalised actions
    x_traj : (H+1, 6) trajectory
    cost_final : float
    """
    u = jnp.clip(jnp.array(u_init, dtype=jnp.float32), -0.99, 0.99)

    # ── JIT-compiled helpers (built once, reused across outer/inner loops) ──
    _rollout_jit = jax.jit(_rollout_from_step, static_argnums=(0,))
    x_traj = _rollout_jit(step_fn, x0, u)

    # JIT-compiled forward pass with feedback using jax.lax.scan
    def _fwd_pass(x0_fp, u_nominal, k, K, x_bar, alpha_fp):
        """Full forward pass with LQR feedback, compiled as single XLA op."""
        def scan_step(x_cur, slices):
            u_t, k_t, K_t, xbar_t = slices
            dx = x_cur - xbar_t
            u_new = jnp.clip(
                u_t + alpha_fp * k_t + K_t @ dx,
                jnp.float32(-0.99), jnp.float32(0.99),
            )
            x_next = step_fn(x_cur, u_new)
            return x_next, (x_next, u_new)
        _, (xs, us_new) = jax.lax.scan(scan_step, x0_fp, (u_nominal, k, K, x_bar))
        return jnp.concatenate([x0_fp[None], xs], axis=0), us_new

    _fwd_pass_jit = jax.jit(_fwd_pass)

    # Static terminal cost derivatives (don't depend on sigma)
    _term_grad = jax.jit(jax.grad(terminal_cost_fn))
    _term_hess = jax.jit(jax.hessian(terminal_cost_fn))
    # Static dynamics Jacobians (don't depend on sigma)
    _dyn_jac_x = jax.jit(jax.jacobian(step_fn, argnums=0))
    _dyn_jac_u = jax.jit(jax.jacobian(step_fn, argnums=1))
    _batch_Fx = jax.jit(jax.vmap(_dyn_jac_x))
    _batch_Fu = jax.jit(jax.vmap(_dyn_jac_u))

    def _total_cost(xs, us, sigma):
        # Inline log-barrier so sigma stays as a Python float (no abstract tracing)
        sig = jnp.float32(sigma)
        eps = jnp.float32(1e-8)
        b_sum = -sig * jnp.sum(
            jnp.log(jnp.maximum(1.0 - us, eps))
            + jnp.log(jnp.maximum(1.0 + us, eps))
        )
        l_sum = jnp.sum(jax.vmap(stage_cost_fn)(xs[:-1], us))
        phi = terminal_cost_fn(xs[-1])
        return l_sum + b_sum + phi

    # static_argnums=(2,) so sigma is a compile-time constant (recompiles per barrier level)
    _total_cost_jit = jax.jit(_total_cost, static_argnums=(2,))

    reg_m = jnp.float32(reg)
    sigma = float(sigma0)

    while sigma >= barrier_tol:
        barrier_fn = make_barrier(sigma)

        def aug_stage(x, u):
            return stage_cost_fn(x, u) + barrier_fn(u)

        # JIT + vmap cost derivatives for this barrier level
        _batch_lx = jax.jit(jax.vmap(jax.grad(aug_stage, argnums=0)))
        _batch_lu = jax.jit(jax.vmap(jax.grad(aug_stage, argnums=1)))
        _batch_lxx = jax.jit(jax.vmap(jax.hessian(aug_stage, argnums=0)))
        _batch_luu = jax.jit(jax.vmap(jax.hessian(aug_stage, argnums=1)))
        _batch_lxu = jax.jit(jax.vmap(
            jax.jacobian(jax.grad(aug_stage, argnums=0), argnums=1)
        ))

        J_cur = float(_total_cost_jit(x_traj, u, sigma))

        for _iter in range(max_iters):
            xs_now = x_traj[:-1]  # (H, 6)

            # ── 1. Batch linearisation (vectorised) ───────────────────
            l_x = _batch_lx(xs_now, u)    # (H, 6)
            l_u = _batch_lu(xs_now, u)    # (H, 3)
            l_xx = _batch_lxx(xs_now, u)  # (H, 6, 6)
            l_uu = _batch_luu(xs_now, u)  # (H, 3, 3)
            l_xu = _batch_lxu(xs_now, u)  # (H, 6, 3)
            F_x = _batch_Fx(xs_now, u)    # (H, 6, 6)
            F_u = _batch_Fu(xs_now, u)    # (H, 6, 3)

            V_x_T = _term_grad(x_traj[-1])   # (6,)
            V_xx_T = _term_hess(x_traj[-1])  # (6, 6)

            # ── 2. Backward pass (jax.lax.scan, reversed) ─────────────
            def bwd_step(carry, t_data):
                V_x, V_xx = carry
                lx, lu, lxx, luu, lxu, Fx, Fu = t_data

                Q_x = lx + Fx.T @ V_x
                Q_u = lu + Fu.T @ V_x
                Q_xx = lxx + Fx.T @ V_xx @ Fx
                Q_ux = lxu.T + Fu.T @ V_xx @ Fx   # (3, 6)
                Q_uu = luu + Fu.T @ V_xx @ Fu + reg_m * jnp.eye(3, dtype=jnp.float32)

                k_t = -jnp.linalg.solve(Q_uu, Q_u)    # (3,)
                K_t = -jnp.linalg.solve(Q_uu, Q_ux)   # (3, 6)

                V_x_new = Q_x + K_t.T @ Q_uu @ k_t + K_t.T @ Q_u + Q_ux.T @ k_t
                V_xx_new = Q_xx + K_t.T @ Q_uu @ K_t + K_t.T @ Q_ux + Q_ux.T @ K_t
                V_xx_new = 0.5 * (V_xx_new + V_xx_new.T)

                dJ_t = -Q_u.T @ jnp.linalg.solve(Q_uu, Q_u)
                return (V_x_new, V_xx_new), (k_t, K_t, dJ_t)

            t_data_rev = (
                l_x[::-1], l_u[::-1], l_xx[::-1], l_uu[::-1], l_xu[::-1],
                F_x[::-1], F_u[::-1],
            )
            (_, _), (k_rev, K_rev, dJ_rev) = jax.lax.scan(
                bwd_step, (V_x_T, V_xx_T), t_data_rev,
            )
            k = k_rev[::-1]   # (H, 3)
            K = K_rev[::-1]   # (H, 3, 6)
            expected_decrease = float(jnp.sum(dJ_rev))

            # ── 3. Forward pass with backtracking line search ──────────
            alpha = 1.0
            accepted = False
            for _ in range(10):
                x_new, u_new = _fwd_pass_jit(x0, u, k, K, x_traj[:-1], jnp.float32(alpha))
                J_new = float(_total_cost_jit(x_new, u_new, sigma))

                if J_new < J_cur + armijo_c * alpha * expected_decrease:
                    accepted = True
                    break
                alpha *= 0.5

            if accepted:
                u = u_new
                x_traj = x_new
                J_cur = J_new
            else:
                break  # line search failed; keep current iterate

            # Convergence: feedforward gain norm
            k_inf = float(jnp.max(jnp.abs(k)))
            if k_inf < 1e-4:
                break

        sigma = max(sigma * barrier_reduction, barrier_tol * 0.1)
        if sigma < barrier_tol:
            break

    return u, x_traj, J_cur


# ── High-level entry point ────────────────────────────────────────────────

def jax_ilqr_optimize(
    cost_map: Any,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg,
    start_nu: Optional[Tuple[float, float, float]] = None,
    lattice_path: Optional[np.ndarray] = None,
    warm_start: Optional[bool] = None,
    goal_y_line: Optional[float] = None,
) -> np.ndarray:
    """
    Optimize trajectory using Box-iLQR (3-DOF dynamics + costmap).

    Uses log-barrier functions to handle control box constraints u ∈ (-1, 1)^3.
    Dynamics Jacobians and cost Hessians computed via JAX automatic differentiation.

    Parameters
    ----------
    cost_map : CostMap
        Costmap object (from ``build_costmap_from_ice_state``).
    start_pose : (x, y, psi)
        Starting pose [m, m, rad].
    goal : (x, y)
        Goal position [m, m].
    run_cfg : RunConfig
        Configuration.
    start_nu : (u, v, r), optional
        Initial body-frame velocities. Default (2, 0, 0).
    lattice_path : (N, 3), optional
        Lattice A* path used to warm-start the control sequence.
    warm_start : bool, optional
        If True, seed initial controls from lattice_path. Default from config.

    Returns
    -------
    trajectory : (H+1, 3) array of [x, y, psi] in world frame.
    """
    # ── Read parameters from config ───────────────────────────────────
    H = getattr(run_cfg, "planning_ilqr_horizon", 150)
    max_iters = getattr(run_cfg, "planning_ilqr_max_iters", 30)
    reg = getattr(run_cfg, "planning_ilqr_reg", 1e-4)
    sigma0 = getattr(run_cfg, "planning_ilqr_sigma0", 0.1)
    barrier_tol = getattr(run_cfg, "planning_ilqr_barrier_tol", 1e-4)
    barrier_reduction = getattr(run_cfg, "planning_ilqr_barrier_reduction", 0.1)
    armijo_c = getattr(run_cfg, "planning_ilqr_armijo_c", 1e-4)
    init_surge_bias = getattr(run_cfg, "planning_jax_mbd_init_surge_bias", 0.7)
    seed = getattr(run_cfg, "planning_diffusion_seed", None) or 42

    costmap_weight = getattr(run_cfg, "planning_jax_mbd_costmap_weight", 1.0)
    costmap_normalizer = getattr(run_cfg, "planning_jax_mbd_costmap_normalizer", 1.0)
    goal_weight = getattr(run_cfg, "planning_jax_mbd_goal_weight", 0.1)
    length_weight = getattr(run_cfg, "planning_jax_mbd_length_weight", 1.0)
    smoothness_weight = getattr(run_cfg, "planning_jax_mbd_smoothness_weight", 0.005)
    surge_max = getattr(run_cfg, "planning_jax_mbd_surge_max", 2e6)
    sway_max = getattr(run_cfg, "planning_jax_mbd_sway_max", 5e5)
    yaw_max = getattr(run_cfg, "planning_jax_mbd_yaw_max", 5e8)
    dt_substep = getattr(run_cfg, "planning_jax_mbd_dt_substep", 1.0)
    n_substeps = getattr(run_cfg, "planning_jax_mbd_n_substeps", 1)

    if warm_start is None:
        warm_start = getattr(run_cfg, "planning_ilqr_warm_start", True)

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
    action_center = jnp.array([surge_max / 2, 0.0, 0.0], dtype=jnp.float32)
    action_scale = jnp.array([surge_max / 2, sway_max, yaw_max], dtype=jnp.float32)

    step_fn = make_dynamics_step(
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

    # ── Initial state ─────────────────────────────────────────────────
    eta0 = jnp.array(start_pose, dtype=jnp.float32)
    nu0 = jnp.array(start_nu or (2.0, 0.0, 0.0), dtype=jnp.float32)
    x0 = jnp.concatenate([eta0, nu0])

    # ── Cost functions ────────────────────────────────────────────────
    dt_action = dt_substep * n_substeps

    stage_cost_fn = make_stage_cost(
        grid, meta, dt_action,
        costmap_weight=costmap_weight,
        costmap_normalizer=costmap_normalizer,
        length_weight=length_weight,
        smoothness_weight=smoothness_weight,
        ship_length=L,
        ship_beam=B,
    )

    goal_xy = jnp.array(goal[:2], dtype=jnp.float32)
    terminal_cost_fn = make_terminal_cost(goal_xy, goal_weight=goal_weight, goal_y_line=goal_y_line)

    # ── Warm-start ────────────────────────────────────────────────────
    u_init = np.zeros((H, 3), dtype=np.float32)
    u_init[:, 0] = init_surge_bias
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
        u_init = _warm_start_from_path(truncated, H, init_surge_bias)

    u_init = jnp.array(np.clip(u_init, -0.99, 0.99), dtype=jnp.float32)

    # ── Run Box-iLQR ─────────────────────────────────────────────────
    ws_label = "warm-started from A*" if (warm_start and lattice_path is not None) else "cold start"
    backend = _jax_backend_info()
    print(
        f"  JAX iLQR [{backend}]: H={H}, max_iters={max_iters}, "
        f"sigma0={sigma0}, dt_action={dt_action}s, {ws_label}",
        flush=True,
    )
    _t0 = _time.perf_counter()
    u_opt, x_traj, cost_final = run_ilqr(
        step_fn=step_fn,
        stage_cost_fn=stage_cost_fn,
        terminal_cost_fn=terminal_cost_fn,
        x0=x0,
        u_init=u_init,
        H=H,
        max_iters=max_iters,
        reg=reg,
        sigma0=sigma0,
        barrier_tol=barrier_tol,
        barrier_reduction=barrier_reduction,
        armijo_c=armijo_c,
    )
    _elapsed = _time.perf_counter() - _t0
    print(
        f"  JAX iLQR done ({_elapsed:.1f}s): cost_final={cost_final:.3f}",
        flush=True,
    )

    # ── Extract [x, y, psi] trajectory ───────────────────────────────
    traj = np.array(x_traj[:, :3])  # (H+1, 3)
    return traj
