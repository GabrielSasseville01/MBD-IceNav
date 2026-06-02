"""
JAX DIAL-MPC planner for 3-DOF vessel trajectory optimisation.

Reference
---------
H. Xue, C. Pan, Z. Yi, G. Qu, G. Shi.
"Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via
Diffusion-Style Annealing", ICRA 2025 (arXiv 2409.15610).
Authors' code: https://github.com/LeCar-Lab/dial-mpc

Algorithm 1 inner loop (no receding horizon by default): runs ``N_diffuse``
iterations of MPPI score-ascent over the control sequence, with the Gaussian
sampling kernel annealed both across iterations (trajectory-level) and across
the horizon (action-level), per eqs. (5)-(7) of the paper:

    sigma^i_h  =  exp(-(N-i)/(beta1*N) - (H-h)/(beta2*H))
               =  horizon_diffuse_factor^(H_node - h) * sigma_scale
                  * traj_diffuse_factor^i        (i = 0 .. N-1)

The control sequence is parameterised as ``H_node + 1`` spline control nodes
in [-1, 1]^3 (surge / sway / yaw normalised), interpolated to ``H`` fine
actions via a quadratic interpolating spline.  The spline matrix is
precomputed once via ``scipy.interpolate.InterpolatedUnivariateSpline(k=2)``
to match the reference (which uses ``jax_cosmo`` for the same spline) without
adding ``jax_cosmo`` as a dependency.

At each inner iteration the MPPI update (eq. (2)) is::

    rew_Ybar = cost(Ybar)                      # (negated cost, no rollout reduction here)
    logp     = (rews - rew_Ybar) / rews.std() / temp_sample
    weights  = softmax(logp)
    Ybar     = sum_n weights_n * Y0s_n

Ybar itself is appended to the sample set (so the current incumbent is one of
the candidates), and samples are clipped to ``[-1, 1]``.

This file reuses ``make_dynamics_rollout``, ``make_cost_fn``, and
``make_terminal_goal_euclidean_cost_fn`` from ``jax_mbd_planner.py`` so the
cost being optimised is identical to the other MBD variants in the toy
scenario suite.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import InterpolatedUnivariateSpline

from medea_ice_ship.config import get_default_simulation_config
from medea_ice_ship.jax_mbd_planner import (
    _jax_backend_info,
    make_cost_fn,
    make_dynamics_rollout,
    make_terminal_goal_euclidean_cost_fn,
)


# ── Quadratic spline matrix (precomputed once, applied as constant matmul) ──

def _build_quadratic_spline_matrix(
    src_knots: np.ndarray, dst_eval: np.ndarray
) -> np.ndarray:
    """
    Return matrix ``A`` of shape ``(len(dst_eval), len(src_knots))`` such that
    ``A @ y`` evaluates the quadratic interpolating spline through
    ``(src_knots, y)`` at the points ``dst_eval``.

    Built by feeding the unit basis vectors through scipy's
    ``InterpolatedUnivariateSpline(k=2)`` — equivalent to the
    ``jax_cosmo.scipy.interpolate.InterpolatedUnivariateSpline(k=2)``
    used by the reference DIAL-MPC implementation.
    """
    n_src = len(src_knots)
    A = np.zeros((len(dst_eval), n_src), dtype=np.float64)
    for j in range(n_src):
        e_j = np.zeros(n_src)
        e_j[j] = 1.0
        spline = InterpolatedUnivariateSpline(src_knots, e_j, k=2)
        A[:, j] = spline(dst_eval)
    return A.astype(np.float32)


# ── Single inner iteration (one MPPI score-ascent step) ─────────────────────

def _make_reverse_once(
    cost_fn,
    rollout_fn,
    eta0: jnp.ndarray,
    nu0: jnp.ndarray,
    action_center: jnp.ndarray,
    action_scale: jnp.ndarray,
    node2u_mat: jnp.ndarray,   # (H, H_node + 1)
    H_node: int,
    Nsample: int,
    nu: int,
    temp_sample: float,
    pin_first_action: bool,
):
    """
    Build a JIT-compiled MPPI score-ascent step matching DIAL-MPC's
    ``reverse_once``.  Returns ``(Ybar_new, info_dict)``.

    ``noise_scale`` is the per-node sigma vector of length ``H_node + 1``
    (already incorporating both the action-level and trajectory-level
    annealing factors for the current inner iteration).
    """
    _eps = jnp.float32(1e-8)
    _temp = jnp.float32(temp_sample)

    def _rollout_from_nodes(Y_nodes):
        """Y_nodes: (H_node+1, nu) normalized → traj: (H+1, 3)."""
        u_norm = node2u_mat @ Y_nodes                # (H, nu)
        u_phys = u_norm * action_scale + action_center
        return rollout_fn(eta0, nu0, u_phys)

    def _cost_from_nodes(Y_nodes):
        """Same cost as the other MBD variants (defined on normalized u_seq)."""
        u_norm = node2u_mat @ Y_nodes
        return cost_fn(u_norm)

    _cost_batch = jax.vmap(_cost_from_nodes)
    _rollout_batch = jax.vmap(_rollout_from_nodes)

    def reverse_once(rng, Ybar_i, noise_scale):
        rng, rng_s = jax.random.split(rng)
        eps_Y = jax.random.normal(
            rng_s, (Nsample, H_node + 1, nu), dtype=jnp.float32
        )
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i[None]
        if pin_first_action:
            # Receding-horizon semantics: cannot change the action already
            # being executed.  Single-shot mode keeps this off so u[0] is
            # optimised too.
            Y0s = Y0s.at[:, 0].set(Ybar_i[0, :])
        # Append the incumbent so it always participates in the softmax.
        Y0s = jnp.concatenate([Y0s, Ybar_i[None]], axis=0)
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        costs = _cost_batch(Y0s)                     # (Nsample + 1,)
        rews = -costs
        # rews of the incumbent (last sample) plays the role of rew_Ybar_i
        # in the reference, which subtracts a baseline before normalising
        # by rews.std for numerical stability of the softmax.
        rew_Ybar_i = rews[-1]
        rew_std = rews.std() + _eps
        logp = (rews - rew_Ybar_i) / rew_std / _temp
        weights = jax.nn.softmax(logp)

        Ybar_new = jnp.einsum("n,nij->ij", weights, Y0s)

        info = {
            "Y0s": Y0s,
            "rews": rews,
            "costs": costs,
            "weights": weights,
        }
        return rng, Ybar_new, info

    return jax.jit(reverse_once), jax.jit(_rollout_from_nodes), jax.jit(
        _rollout_batch
    ), jax.jit(_cost_from_nodes)


# ── Main entry point ────────────────────────────────────────────────────────

def dial_mpc_optimize(
    cost_map: Any,
    start_pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    run_cfg,
    *,
    start_nu: Optional[Tuple[float, float, float]] = None,
    return_diagnostics: bool = True,
    # ── DIAL-MPC hyperparameters (paper-named) ─────────────────────────────
    H_node: int = 25,
    Nsample: int = 2048,
    Ndiffuse: int = 10,
    temp_sample: float = 0.05,
    horizon_diffuse_factor: float = 0.9,
    traj_diffuse_factor: float = 0.5,
    sigma_scale: float = 1.0,
    pin_first_action: bool = False,
    # ── Cost-function passthroughs (match other MBD variants) ──────────────
    terminal_goal_distance_weight: float = 0.0,
    cost_truncate_y: Optional[float] = None,
    oob_penalty: float = 1e8,
    n_top_samples: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, dict]:
    """
    Single-shot DIAL-MPC trajectory optimization.

    Returns
    -------
    mean_traj : (H+1, 3) ndarray
        World-frame trajectory rolled out from the final ``Ybar``.
    diag : dict
        Diagnostics dict with the same keys as ``run_mbd_with_diagnostics``
        (``mean_trajectories``, ``sample_trajectories``, ``cost_means``,
        ``cost_mins``, ``cost_of_mean``, ``weights``, ``accept_rates``,
        ``n_total_sampled``, ``Ybar_final``, ``rejected_sample_trajectories``,
        ``best_sample_actions`` / ``best_sample_cost``), so the existing
        ``toy_mbd_scenarios`` plotting / fan-movie helpers work unchanged.

        Additional DIAL-specific keys:
        ``noise_scales`` (per-iteration per-node sigma), ``temp_sample``.
    """
    # ── Read shared MBD knobs from run_cfg (dynamics + cost shape) ────────
    H = getattr(run_cfg, "planning_jax_mbd_horizon", 80)
    costmap_weight = getattr(run_cfg, "planning_jax_mbd_costmap_weight", 1.0)
    costmap_normalizer = getattr(run_cfg, "planning_jax_mbd_costmap_normalizer", 1.0)
    goal_weight = getattr(run_cfg, "planning_jax_mbd_goal_weight", 0.1)
    length_weight = getattr(run_cfg, "planning_jax_mbd_length_weight", 1.0)
    smoothness_weight = getattr(run_cfg, "planning_jax_mbd_smoothness_weight", 0.005)
    surge_max = getattr(run_cfg, "planning_jax_mbd_surge_max", 2e6)
    sway_max = getattr(run_cfg, "planning_jax_mbd_sway_max", 5e5)
    yaw_max = getattr(run_cfg, "planning_jax_mbd_yaw_max", 5e8)
    dt_substep = getattr(run_cfg, "planning_jax_mbd_dt_substep", 1.0)
    n_substeps = getattr(run_cfg, "planning_jax_mbd_n_substeps", 5)

    if H_node + 1 > H:
        raise ValueError(
            f"H_node + 1 ({H_node + 1}) must be <= H ({H}); not enough fine "
            "actions to interpolate that many nodes."
        )

    # ── Costmap → JAX ─────────────────────────────────────────────────────
    grid = jnp.array(cost_map.cost_map, dtype=jnp.float32)
    meta = {
        "x_origin": float(cost_map.x_origin),
        "y_origin": float(cost_map.y_origin),
        "resolution": float(cost_map.resolution),
        "rows": int(cost_map.rows),
        "cols": int(cost_map.cols),
    }

    # ── Dynamics rollout (shared with other MBD variants) ─────────────────
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
    nu = 3

    # ── Cost function (identical structure to the other MBD methods) ─────
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
        oob_penalty=oob_penalty,
    )
    if terminal_goal_distance_weight > 0.0:
        _td_fn = make_terminal_goal_euclidean_cost_fn(
            rollout_fn, eta0, nu0, action_center, action_scale,
            goal_xy=(float(goal[0]), float(goal[1])),
            weight=terminal_goal_distance_weight,
        )
        _base = cost_fn
        cost_fn = lambda u_norm: _base(u_norm) + _td_fn(u_norm)  # noqa: E731

    # ── Quadratic spline node→u matrix (precomputed) ─────────────────────
    # step_nodes / step_us in normalised time units; only the relative
    # spacing matters for the spline matrix.
    step_us = np.linspace(0.0, 1.0, H, dtype=np.float64)
    step_nodes = np.linspace(0.0, 1.0, H_node + 1, dtype=np.float64)
    node2u_np = _build_quadratic_spline_matrix(step_nodes, step_us)
    node2u_mat = jnp.array(node2u_np, dtype=jnp.float32)

    # ── DIAL annealing schedule (eqs. 5-7) ───────────────────────────────
    # Action-level: largest sigma at the back of the horizon (h = H_node);
    # smallest at the front (h = 0).
    sigma_control_np = (horizon_diffuse_factor ** np.arange(H_node + 1)[::-1]).astype(
        np.float32
    ) * sigma_scale
    # Trajectory-level multiplier per inner iteration i=0..Ndiffuse-1.
    # i=0 -> full sigma_control; i=Ndiffuse-1 -> smallest sigma.
    traj_factor_np = (traj_diffuse_factor ** np.arange(Ndiffuse)).astype(np.float32)
    # Combined: shape (Ndiffuse, H_node + 1).
    noise_schedule_np = traj_factor_np[:, None] * sigma_control_np[None, :]
    noise_schedule = jnp.array(noise_schedule_np, dtype=jnp.float32)

    # ── Build JIT-ed reverse_once + helpers ──────────────────────────────
    reverse_once_jit, rollout_from_nodes, rollout_from_nodes_batch, cost_from_nodes = (
        _make_reverse_once(
            cost_fn=cost_fn,
            rollout_fn=rollout_fn,
            eta0=eta0,
            nu0=nu0,
            action_center=action_center,
            action_scale=action_scale,
            node2u_mat=node2u_mat,
            H_node=H_node,
            Nsample=Nsample,
            nu=nu,
            temp_sample=temp_sample,
            pin_first_action=pin_first_action,
        )
    )

    # ── Run the inner loop with per-step diagnostics ─────────────────────
    backend = _jax_backend_info()
    print(
        f"  JAX DIAL-MPC [{backend}]: H={H}, H_node={H_node}, "
        f"Nsample={Nsample}, Ndiffuse={Ndiffuse}, temp_sample={temp_sample}, "
        f"horizon_diffuse={horizon_diffuse_factor}, "
        f"traj_diffuse={traj_diffuse_factor}, sigma_scale={sigma_scale}",
        flush=True,
    )
    t0 = time.perf_counter()

    Ybar = jnp.zeros((H_node + 1, nu), dtype=jnp.float32)
    rng = jax.random.PRNGKey(seed)

    diag = {
        "mean_trajectories": [],
        "sample_trajectories": [],
        "sample_costs": [],
        "cost_means": [],
        "cost_mins": [],
        "cost_of_mean": [],
        "weights": [],
        "accept_rates": [],
        "n_total_sampled": [],
        "rejected_sample_trajectories": [],
        "noise_scales": [],
        "best_sample_actions": None,
        "best_sample_cost": float("inf"),
        "second_best_sample_actions": None,
        "second_best_sample_cost": float("inf"),
        "Ybar_final": None,
        "temp_sample": float(temp_sample),
    }

    # Initial mean trajectory (rolled out from zero-action sequence).
    traj0 = np.array(rollout_from_nodes(Ybar))
    diag["mean_trajectories"].append(traj0)

    for i in range(Ndiffuse):
        noise_scale = noise_schedule[i]
        rng, Ybar, info = reverse_once_jit(rng, Ybar, noise_scale)
        Ybar = jnp.clip(Ybar, -1.0, 1.0)

        Y0s_np = np.array(info["Y0s"])
        costs_np = np.array(info["costs"])
        w_np = np.array(info["weights"])
        # Drop the appended incumbent (last entry) when reporting sample
        # statistics — keeps shapes consistent with the other MBD variants
        # (Nsample, not Nsample+1).
        Y0s_samples = Y0s_np[:-1]
        costs_samples = costs_np[:-1]
        w_samples = w_np[:-1]

        diag["sample_costs"].append(costs_samples)
        diag["cost_means"].append(float(costs_samples.mean()))
        diag["cost_mins"].append(float(costs_samples.min()))
        diag["weights"].append(w_samples)
        diag["accept_rates"].append(1.0)
        diag["n_total_sampled"].append(Nsample)
        diag["cost_of_mean"].append(float(cost_from_nodes(Ybar)))
        diag["noise_scales"].append(np.array(noise_scale))

        # Track global best sample (lowest-cost individual sample seen so
        # far) for the result plot.
        sort_idx = np.argsort(costs_samples)
        best_idx = int(sort_idx[0])
        best_cost = float(costs_samples[best_idx])
        if best_cost < diag["best_sample_cost"]:
            diag["best_sample_cost"] = best_cost
            diag["best_sample_actions"] = Y0s_samples[best_idx]
            if len(sort_idx) > 1:
                second_idx = int(sort_idx[1])
                diag["second_best_sample_cost"] = float(costs_samples[second_idx])
                diag["second_best_sample_actions"] = Y0s_samples[second_idx]

        # Roll out the updated mean.
        mean_traj = np.array(rollout_from_nodes(Ybar))
        diag["mean_trajectories"].append(mean_traj)

        # Top-N sample trajectories for the fan movie.
        n_store = min(n_top_samples, len(costs_samples))
        top_idx = np.argsort(costs_samples)[:n_store]
        top_nodes = jnp.array(Y0s_samples[top_idx])
        top_trajs = np.array(rollout_from_nodes_batch(top_nodes))
        diag["sample_trajectories"].append(top_trajs)
        # No reject filter in DIAL — same shape contract as vanilla MBD.
        diag["rejected_sample_trajectories"].append(
            np.zeros((0, H + 1, 3), dtype=np.float32)
        )

    diag["Ybar_final"] = np.array(Ybar)
    final_traj = np.array(rollout_from_nodes(Ybar))

    elapsed = time.perf_counter() - t0
    print(
        f"  DIAL-MPC done: {Ndiffuse} inner iterations in {elapsed:.2f} s, "
        f"final cost={diag['cost_of_mean'][-1]:.1f}",
        flush=True,
    )

    if not return_diagnostics:
        return final_traj, {}
    return final_traj, diag
