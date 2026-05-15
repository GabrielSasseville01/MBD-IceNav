#!/usr/bin/env python3
"""
Model-Based Diffusion (MBD) trajectory planner with 3-DOF vessel dynamics (JAX).

Optimises body-frame control sequences [X, Y, N] (surge force, sway force, yaw
torque) via the MBD reverse diffusion process.  Controls are rolled out through
Fossen-style 3-DOF vessel dynamics to produce physically consistent ship
trajectories, which are evaluated against a 2D costmap.

Algorithm (MBD paper):
  Forward:  p(Y_i | Y_0) = N(sqrt(alpha_bar_i) Y_0, (1-alpha_bar_i) I)
  Score:    nabla log p_i(Y_i) ~ -Y_i/(1-abar_i) + sqrt(abar_i)/(1-abar_i) Ybar_0
  Reverse:  Y_{i-1} = 1/sqrt(alpha_i) (Y_i + (1-abar_i) score)
  Weights:  w_k = softmax(-J(Y0_k) / lambda)
  Ybar_0  = sum(w_k Y0_k)

Decision variable Y = action sequence u_norm (H, 3) in [-1, 1]^3.
Trajectories are obtained by forward-rolling denormalised controls through
3-DOF vessel dynamics (M nu_dot + C(nu) nu + D nu = tau).

Run:
  uv run python examples/mbd_costmap_planner.py
  uv run python examples/mbd_costmap_planner.py --Nsample 1024 --Ndiffuse 150

Output (default: outputs/mbd_costmap_planner/):
  - mbd_denoising_snapshots.png  (static multi-panel figure)
  - mbd_denoising_animation.mp4  (animation of reverse process)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm
from tqdm import tqdm
import argparse

from medea_ice_ship import build_costmap_from_ice_state
from medea_ice_ship.config import get_default_run_config


# ─────────────────────────────────────────────────────────────────────────────
# Ice corridor setup (reused from toy_planner_thin_corridor.py)
# ─────────────────────────────────────────────────────────────────────────────

def make_thick_ice_with_thin_corridor(
    x_ice_lo, x_ice_hi, y_lo, y_hi,
    y_corridor_center, corridor_half_width,
    thick_m=10.0, thin_m=1.0,
):
    """Three rectangular ice floes: bottom thick, middle thin corridor, top thick."""
    def rect(x0, y0, w, h):
        return np.array([[x0, y0], [x0+w, y0], [x0+w, y0+h], [x0, y0+h]], dtype=float)

    width = x_ice_hi - x_ice_lo
    y_lo_corr = y_corridor_center - corridor_half_width
    y_hi_corr = y_corridor_center + corridor_half_width

    polygons, centroids, thicknesses = [], [], []
    if y_lo_corr > y_lo:
        h = y_lo_corr - y_lo
        polygons.append(rect(x_ice_lo, y_lo, width, h))
        centroids.append([x_ice_lo + width / 2, y_lo + h / 2])
        thicknesses.append(thick_m)
    h_mid = y_hi_corr - y_lo_corr
    polygons.append(rect(x_ice_lo, y_lo_corr, width, h_mid))
    centroids.append([x_ice_lo + width / 2, y_corridor_center])
    thicknesses.append(thin_m)
    if y_hi > y_hi_corr:
        h = y_hi - y_hi_corr
        polygons.append(rect(x_ice_lo, y_hi_corr, width, h))
        centroids.append([x_ice_lo + width / 2, y_hi_corr + h / 2])
        thicknesses.append(thick_m)

    return polygons, np.array(centroids), np.array(thicknesses)


# ─────────────────────────────────────────────────────────────────────────────
# JAX 3-DOF vessel dynamics  (Fossen: M nu_dot + C(nu) nu + D nu = tau)
# ─────────────────────────────────────────────────────────────────────────────

def make_dynamics_rollout(
    mass, Izz, dt_substep, n_substeps,
    ship_damping_xy=0.02, ship_damping_psi=0.15,
    added_mass_surge_frac=0.05, added_mass_sway_frac=0.5,
    added_mass_yaw_frac=0.2,
):
    """
    Build a pure-JAX dynamics rollout function (JIT-traceable, vmap-compatible).

    Each action step holds the control constant for *n_substeps* Euler sub-steps
    of size *dt_substep*, giving an effective action period of
    dt_action = dt_substep * n_substeps.

    Returns:
        rollout_fn(eta0, nu0, u_seq) -> trajectory (H+1, 3) of [x, y, psi]
        M3_np, D3_np: mass / damping matrices (numpy, for diagnostics)
    """
    m11 = float(mass * (1.0 + added_mass_surge_frac))
    m22 = float(mass * (1.0 + added_mass_sway_frac))
    m33 = float(Izz  * (1.0 + added_mass_yaw_frac))

    M3_inv_diag = jnp.array([1.0 / m11, 1.0 / m22, 1.0 / m33])
    D3_diag     = jnp.array([
        mass * ship_damping_xy,
        mass * ship_damping_xy,
        Izz  * ship_damping_psi,
    ])

    _m11 = jnp.float32(m11)
    _m22 = jnp.float32(m22)
    _dt  = jnp.float32(dt_substep)

    def _euler_step(state, tau):
        """Single Euler integration sub-step.  state = [eta(3); nu(3)]."""
        eta, nu = state[:3], state[3:6]

        # C(nu) nu  (only non-zero off-diagonal products)
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
        """Forward-roll dynamics.  u_seq: (H, 3) -> traj: (H+1, 3)."""
        state0 = jnp.concatenate([eta0, nu0])
        _, traj = jax.lax.scan(_action_step, state0, u_seq)
        return jnp.concatenate([eta0[None, :], traj], axis=0)

    M3_np = np.diag([m11, m22, m33])
    D3_np = np.diag([float(d) for d in D3_diag])
    return rollout, M3_np, D3_np


# ─────────────────────────────────────────────────────────────────────────────
# Costmap bilinear interpolation (JAX)
# ─────────────────────────────────────────────────────────────────────────────

def _make_query(grid, meta):
    """Return a bilinear-interpolation query function for the costmap."""
    xo   = float(meta["x_origin"])
    yo   = float(meta["y_origin"])
    res  = float(meta["resolution"])
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


# ─────────────────────────────────────────────────────────────────────────────
# Cost function  (normalised actions -> dynamics rollout -> costmap + goal)
# ─────────────────────────────────────────────────────────────────────────────

def make_cost_fn(
    grid, meta, goal_world, rollout_fn, eta0, nu0,
    action_center, action_scale,
    costmap_weight=1.0, goal_weight=0.5,
    control_weight=0.01, smoothness_weight=0.02,
):
    """
    Build cost function J(u_norm).

    u_norm (H, 3) in [-1, 1] -> denormalise -> dynamics rollout -> evaluate.
    Cost = costmap traversal + goal distance + control effort + control jerk.
    """
    query = _make_query(grid, meta)
    goal_xy = jnp.array(goal_world[:2], dtype=jnp.float32)

    def cost_fn(u_norm):
        u_phys = u_norm * action_scale + action_center
        traj = rollout_fn(eta0, nu0, u_phys)  # (H+1, 3)

        # Costmap cost: mean over waypoints + segment midpoints
        wp_costs = jax.vmap(lambda p: query(p[0], p[1]))(traj[:, :2])
        mids = (traj[:-1, :2] + traj[1:, :2]) / 2.0
        mid_costs = jax.vmap(lambda p: query(p[0], p[1]))(mids)
        costmap_cost = (wp_costs.sum() + mid_costs.sum()) / (
            wp_costs.shape[0] + mid_costs.shape[0]
        )

        # Goal reaching (linear distance for constant gradient magnitude)
        goal_dist = jnp.sqrt(jnp.sum((traj[-1, :2] - goal_xy) ** 2) + 1e-4)

        # Control effort (in normalised action space)
        effort = jnp.mean(u_norm ** 2)

        # Control smoothness (penalise sudden changes between actions)
        du = jnp.diff(u_norm, axis=0)
        jerk = jnp.mean(du ** 2)

        return (
            costmap_weight * costmap_cost
            + goal_weight * goal_dist
            + control_weight * effort
            + smoothness_weight * jerk
        )

    return cost_fn


# ─────────────────────────────────────────────────────────────────────────────
# MBD reverse diffusion process  (action space)
# ─────────────────────────────────────────────────────────────────────────────

def run_mbd(
    cost_fn, rollout_fn, eta0, nu0, action_center, action_scale,
    H, Nsample, Ndiffuse, temperature,
    beta0, betaT, seed, n_vis_samples=30, vis_every=5,
):
    """
    Model-Based Diffusion reverse process in **action space**.

    Decision variable Y = u_norm (H, 3) in [-1, 1].
    Trajectories obtained by rolling out denormalised controls through dynamics.

    Math (identical to mbd_planner.py):
        Yi       = Ybar_i * sqrt(alpha_bar[i])
        Y0s      = Ybar_i + eps * sigma[i]             (sample from phi_i)
        weights  = softmax((rews - mean) / std / T)    (importance weights)
        Ybar     = einsum("n,nij->ij", weights, Y0s)   (weighted mean)
        score    = (-Yi + sqrt(abar) Ybar) / (1 - abar)
        Y_{i-1}  = 1/sqrt(alpha) (Yi + (1 - abar) score)
        Ybar_{i-1} = Y_{i-1} / sqrt(abar[i-1])
    """
    rng = jax.random.PRNGKey(seed)

    betas = jnp.linspace(beta0, betaT, Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1.0 - alphas_bar)
    print(
        f"  Noise schedule: sigma[1]={float(sigmas[1]):.4f}, "
        f"sigma[N-1]={float(sigmas[-1]):.4f}, "
        f"alpha_bar[N-1]={float(alphas_bar[-1]):.4f}"
    )

    # Initialise with strong forward thrust and a gentle yaw schedule
    # that steers from start toward the goal
    Ybar_i = jnp.zeros((H, 3))
    Ybar_i = Ybar_i.at[:, 0].set(0.7)

    cost_batch = jax.vmap(cost_fn)

    @jax.jit
    def reverse_once(carry, _unused):
        """Single MBD reverse step (Eq. reverse + Eq. Y0_bar)."""
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        rng, rng_s = jax.random.split(rng)
        eps = jax.random.normal(rng_s, (Nsample, H, 3))
        Y0s = eps * sigmas[i] + Ybar_i
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        costs = cost_batch(Y0s)

        rews = -costs
        rew_mean = rews.mean()
        rew_std = jnp.maximum(rews.std(), 1e-6)
        logp0 = (rews - rew_mean) / rew_std / temperature
        weights = jax.nn.softmax(logp0)

        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)

        score = (
            1.0
            / (1.0 - alphas_bar[i])
            * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        )
        Yim1 = (
            1.0
            / jnp.sqrt(alphas[i])
            * (Yi + (1.0 - alphas_bar[i]) * score)
        )
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])
        Ybar_im1 = jnp.clip(Ybar_im1, -1.0, 1.0)

        return (i - 1, rng, Ybar_im1), (costs.mean(), costs.min())

    # Rollout helpers for visualisation (JIT once)
    @jax.jit
    def _rollout_single(u_norm):
        return rollout_fn(eta0, nu0, u_norm * action_scale + action_center)

    @jax.jit
    def _rollout_batch(u_norms):
        return jax.vmap(lambda u: rollout_fn(
            eta0, nu0, u * action_scale + action_center
        ))(u_norms)

    history = {
        "traj_snapshots": [],
        "cost_means": [],
        "cost_mins": [],
        "sample_traj_snapshots": [],
        "step_indices": [],
    }

    step_count = 0
    with tqdm(range(Ndiffuse - 1, 0, -1), desc="MBD Reverse") as pbar:
        for i in pbar:
            carry = (i, rng, Ybar_i)
            (_, rng, Ybar_i), (c_mean, c_min) = reverse_once(carry, None)

            traj_mean = _rollout_single(Ybar_i)
            history["traj_snapshots"].append(np.array(traj_mean))
            history["cost_means"].append(float(c_mean))
            history["cost_mins"].append(float(c_min))

            if step_count % vis_every == 0:
                rng_v, _ = jax.random.split(rng)
                eps_v = jax.random.normal(rng_v, (n_vis_samples, H, 3))
                Y0s_v = eps_v * float(sigmas[max(i - 1, 1)]) + Ybar_i
                Y0s_v = jnp.clip(Y0s_v, -1.0, 1.0)
                sample_trajs = _rollout_batch(Y0s_v)
                history["sample_traj_snapshots"].append(np.array(sample_trajs))
                history["step_indices"].append(step_count)

            step_count += 1
            pbar.set_postfix({"cost": f"{c_mean:.1f}", "min": f"{c_min:.1f}"})

    final_traj = np.array(_rollout_single(Ybar_i))
    final_actions = np.array(Ybar_i * action_scale + action_center)
    return Ybar_i, final_traj, final_actions, history


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _costmap_extent(meta):
    return [
        meta["x_origin"],
        meta["x_origin"] + meta["cols"] * meta["resolution"],
        meta["y_origin"],
        meta["y_origin"] + meta["rows"] * meta["resolution"],
    ]


def _plot_costmap(ax, costmap_np, meta, start_w, goal_w, title=""):
    extent = _costmap_extent(meta)
    data = np.maximum(costmap_np, 0.5)
    ax.imshow(
        data, origin="lower", extent=extent, cmap="YlOrRd",
        norm=LogNorm(vmin=1, vmax=data.max()), aspect="equal",
    )
    ax.plot(start_w[0], start_w[1], "go", ms=10, zorder=5, label="Start")
    ax.plot(goal_w[0], goal_w[1], "r*", ms=14, zorder=5, label="Goal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)


def _plot_traj_with_heading(ax, traj, color="b", lw=2, label=None, arrow_every=5):
    """Plot trajectory with heading arrows."""
    ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=lw, label=label, zorder=4)
    idx = np.arange(0, len(traj), max(1, arrow_every))
    arrow_len = 20.0
    for k in idx:
        dx = arrow_len * np.cos(traj[k, 2])
        dy = arrow_len * np.sin(traj[k, 2])
        ax.annotate(
            "", xy=(traj[k, 0] + dx, traj[k, 1] + dy),
            xytext=(traj[k, 0], traj[k, 1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            zorder=5,
        )


def generate_static_figure(
    costmap_np, meta, start_w, goal_w, final_traj, history, output_path,
):
    """Three-panel figure: (a) costmap, (b) denoising snapshots, (c) convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # (a) Costmap + final trajectory
    _plot_costmap(axes[0], costmap_np, meta, start_w, goal_w, "(a) Costmap + Final")
    _plot_traj_with_heading(axes[0], final_traj, color="b", lw=2.5, label="Final")
    axes[0].legend(fontsize=9)

    # (b) Trajectory denoising snapshots
    _plot_costmap(axes[1], costmap_np, meta, start_w, goal_w, "(b) Trajectory Denoising")
    n = len(history["traj_snapshots"])
    snap_idx = np.linspace(0, n - 1, min(n, 10), dtype=int)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(snap_idx)))
    for j, si in enumerate(snap_idx):
        tw = history["traj_snapshots"][si]
        lbl = f"step {si}" if j in [0, len(snap_idx) - 1] else None
        axes[1].plot(
            tw[:, 0], tw[:, 1], "-", color=colors[j], lw=1.5, alpha=0.7, label=lbl,
        )
    _plot_traj_with_heading(axes[1], final_traj, color="b", lw=3, label="Final")
    axes[1].legend(fontsize=8)

    # (c) Cost convergence
    axes[2].plot(history["cost_means"], "b-", label="Mean cost", alpha=0.8)
    axes[2].plot(history["cost_mins"], "g--", label="Min cost", alpha=0.8)
    axes[2].set_xlabel("Diffusion step")
    axes[2].set_ylabel("Cost")
    axes[2].set_title("(c) Cost Convergence")
    axes[2].set_yscale("log")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_animation(
    costmap_np, meta, start_w, goal_w, history, output_path, fps=15,
):
    """MP4/GIF animation of the denoising process."""
    fig, ax = plt.subplots(figsize=(14, 8))
    extent = _costmap_extent(meta)
    data = np.maximum(costmap_np, 0.5)
    ax.imshow(
        data, origin="lower", extent=extent, cmap="YlOrRd",
        norm=LogNorm(vmin=1, vmax=data.max()), aspect="equal",
    )
    ax.plot(start_w[0], start_w[1], "go", ms=12, zorder=5)
    ax.plot(goal_w[0], goal_w[1], "r*", ms=16, zorder=5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    n = len(history["traj_snapshots"])

    snap_trajs = {}
    for j, si in enumerate(history["step_indices"]):
        snap_trajs[si] = history["sample_traj_snapshots"][j]

    (mean_line,) = ax.plot([], [], "b-", lw=2.5, zorder=4)
    n_vis = min(
        20,
        history["sample_traj_snapshots"][0].shape[0]
        if history["sample_traj_snapshots"] else 0,
    )
    sample_lines = [
        ax.plot([], [], "-", color="cyan", alpha=0.15, lw=0.7)[0]
        for _ in range(n_vis)
    ]
    title_text = ax.set_title("", fontsize=14)

    def _init():
        mean_line.set_data([], [])
        for sl in sample_lines:
            sl.set_data([], [])
        return [mean_line] + sample_lines

    def _update(frame):
        tw = history["traj_snapshots"][frame]
        mean_line.set_data(tw[:, 0], tw[:, 1])
        best_si = min(history["step_indices"], key=lambda s: abs(s - frame))
        if best_si in snap_trajs:
            sw = snap_trajs[best_si]
            for j, sl in enumerate(sample_lines):
                if j < len(sw):
                    sl.set_data(sw[j, :, 0], sw[j, :, 1])
                else:
                    sl.set_data([], [])
        title_text.set_text(
            f"MBD Step {frame + 1}/{n}  |  "
            f"Mean Cost: {history['cost_means'][frame]:.1f}"
        )
        return [mean_line, title_text] + sample_lines

    anim = animation.FuncAnimation(
        fig, _update, init_func=_init, frames=n, interval=1000 // fps, blit=False,
    )
    try:
        anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=100)
        print(f"  Saved: {output_path}")
    except Exception as exc:
        gif_path = output_path.with_suffix(".gif")
        print(f"  ffmpeg unavailable ({exc}), saving GIF instead...")
        anim.save(str(gif_path), writer="pillow", fps=fps, dpi=80)
        print(f"  Saved: {gif_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MBD trajectory planner with 3-DOF vessel dynamics (JAX)"
    )
    # MBD parameters
    parser.add_argument("--Nsample", type=int, default=512,
                        help="Monte Carlo samples per diffusion step")
    parser.add_argument("--Ndiffuse", type=int, default=200,
                        help="Reverse diffusion steps")
    parser.add_argument("--Horizon", type=int, default=100,
                        help="Number of control action steps")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Softmax temperature for importance weighting")
    parser.add_argument("--beta0", type=float, default=1e-4)
    parser.add_argument("--betaT", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    # Dynamics
    parser.add_argument("--dt-substep", type=float, default=1.0,
                        help="Euler integration sub-step [s]")
    parser.add_argument("--n-substeps", type=int, default=5,
                        help="Sub-steps per action step")
    parser.add_argument("--initial-speed", type=float, default=2.0,
                        help="Initial surge speed [m/s]")
    parser.add_argument("--initial-heading", type=float, default=-10.0,
                        help="Initial heading [deg] (0 = east, negative = south)")

    # Action bounds
    parser.add_argument("--surge-max", type=float, default=2e6,
                        help="Max surge force [N]")
    parser.add_argument("--sway-max", type=float, default=5e5,
                        help="Max sway force [N]")
    parser.add_argument("--yaw-max", type=float, default=5e8,
                        help="Max yaw torque [N*m]")

    # Cost weights
    parser.add_argument("--costmap-weight", type=float, default=0.1)
    parser.add_argument("--goal-weight", type=float, default=5.0)
    parser.add_argument("--control-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--damage-cost-scale", type=float, default=100.0)

    # Output
    parser.add_argument("--output-dir", type=str,
                        default="outputs/mbd_costmap_planner")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--vis-every", type=int, default=5,
                        help="Store visualisation samples every N diffusion steps")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build costmap ────────────────────────────────────────────────────────
    print("Building costmap...", flush=True)
    x_min, x_max = -400.0, 800.0
    y_min, y_max = 0.0, 400.0
    planning_margin = 80.0

    sat_polys, sat_cents, sat_thicks = make_thick_ice_with_thin_corridor(
        x_ice_lo=150.0, x_ice_hi=450.0,
        y_lo=y_min - planning_margin, y_hi=y_max + planning_margin,
        y_corridor_center=280.0, corridor_half_width=40.0,
        thick_m=10.0, thin_m=1.0,
    )
    print(
        f"  Ice: {len(sat_polys)} floes "
        f"({int(np.sum(sat_thicks < 2))} thin 1 m, rest 10 m)"
    )

    run_cfg = get_default_run_config()
    cost_map, bounds = build_costmap_from_ice_state(
        sat_polys, sat_cents, sat_thicks,
        x_min, max(x_max, 830.0), y_min, y_max,
        resolution=4.0, margin=planning_margin,
        ship_mass=run_cfg.ship_displacement,
    )
    print(
        f"  Costmap: {cost_map.rows} x {cost_map.cols} cells, "
        f"res={cost_map.resolution} m"
    )

    grid = jnp.array(cost_map.cost_map, dtype=jnp.float32)
    meta = {
        "x_origin": float(cost_map.x_origin),
        "y_origin": float(cost_map.y_origin),
        "resolution": float(cost_map.resolution),
        "rows": int(cost_map.rows),
        "cols": int(cost_map.cols),
    }
    costmap_np = np.array(cost_map.cost_map)

    # ── Ship dynamics ────────────────────────────────────────────────────────
    ship_mass = run_cfg.ship_displacement
    ship_length = run_cfg.ship_length
    ship_beam = run_cfg.ship_beam
    k_zz = 0.25 * ship_length
    Izz = ship_mass * k_zz ** 2

    print(
        f"  Ship: mass={ship_mass/1e6:.0f} t, L={ship_length} m, "
        f"B={ship_beam} m, Izz={Izz:.2e}"
    )

    rollout_fn, M3, D3 = make_dynamics_rollout(
        mass=ship_mass, Izz=Izz,
        dt_substep=args.dt_substep, n_substeps=args.n_substeps,
    )
    dt_action = args.dt_substep * args.n_substeps
    total_time = args.Horizon * dt_action
    print(
        f"  Dynamics: dt_sub={args.dt_substep} s, n_sub={args.n_substeps}, "
        f"dt_action={dt_action} s, total={total_time} s"
    )

    # Terminal velocity at max surge (for reference)
    v_term = args.surge_max / (ship_mass * 0.02)
    print(f"  Terminal speed at max surge: {v_term:.1f} m/s")

    # ── Initial state ────────────────────────────────────────────────────────
    start_w = np.array([-400.0, 400.0])
    goal_w  = np.array([800.0, 400.0])

    psi0 = np.deg2rad(args.initial_heading)
    eta0 = jnp.array([start_w[0], start_w[1], psi0], dtype=jnp.float32)
    nu0  = jnp.array([args.initial_speed, 0.0, 0.0], dtype=jnp.float32)

    print(
        f"  Start: ({start_w[0]:.0f}, {start_w[1]:.0f}), "
        f"heading={args.initial_heading} deg, speed={args.initial_speed} m/s"
    )
    print(f"  Goal:  ({goal_w[0]:.0f}, {goal_w[1]:.0f})")

    # ── Action normalisation ─────────────────────────────────────────────────
    # u_norm in [-1, 1] maps to:
    #   surge: [0, surge_max]        (always forward)
    #   sway:  [-sway_max, sway_max]
    #   yaw:   [-yaw_max, yaw_max]
    action_center = jnp.array(
        [args.surge_max / 2, 0.0, 0.0], dtype=jnp.float32
    )
    action_scale = jnp.array(
        [args.surge_max / 2, args.sway_max, args.yaw_max], dtype=jnp.float32
    )
    print(
        f"  Actions: surge=[0, {args.surge_max:.0e}] N, "
        f"sway=[+/-{args.sway_max:.0e}] N, yaw=[+/-{args.yaw_max:.0e}] N*m"
    )

    # ── Cost function ────────────────────────────────────────────────────────
    cost_fn = make_cost_fn(
        grid, meta, goal_w, rollout_fn, eta0, nu0,
        action_center, action_scale,
        costmap_weight=args.costmap_weight,
        goal_weight=args.goal_weight,
        control_weight=args.control_weight,
        smoothness_weight=args.smoothness_weight,
    )

    # ── Run MBD ──────────────────────────────────────────────────────────────
    print(
        f"\nRunning MBD reverse diffusion: "
        f"Nsample={args.Nsample}, Ndiffuse={args.Ndiffuse}, "
        f"H={args.Horizon}, temp={args.temperature}",
        flush=True,
    )
    final_u_norm, final_traj, final_actions, history = run_mbd(
        cost_fn, rollout_fn, eta0, nu0, action_center, action_scale,
        H=args.Horizon, Nsample=args.Nsample, Ndiffuse=args.Ndiffuse,
        temperature=args.temperature,
        beta0=args.beta0, betaT=args.betaT, seed=args.seed,
        vis_every=args.vis_every,
    )

    print(f"\nOptimised trajectory:")
    print(
        f"  Start: ({final_traj[0, 0]:.1f}, {final_traj[0, 1]:.1f}), "
        f"psi={np.rad2deg(final_traj[0, 2]):.1f} deg"
    )
    print(
        f"  End:   ({final_traj[-1, 0]:.1f}, {final_traj[-1, 1]:.1f}), "
        f"psi={np.rad2deg(final_traj[-1, 2]):.1f} deg"
    )
    goal_dist = np.linalg.norm(final_traj[-1, :2] - goal_w)
    print(f"  Goal distance: {goal_dist:.1f} m")
    print(
        f"  Cost: {history['cost_means'][-1]:.1f} (mean), "
        f"{history['cost_mins'][-1]:.1f} (min)"
    )
    print(
        f"  Surge force: mean={final_actions[:, 0].mean():.0f} N, "
        f"max={final_actions[:, 0].max():.0f} N"
    )
    print(
        f"  Yaw torque: mean={np.abs(final_actions[:, 2]).mean():.0f} N*m, "
        f"max={np.abs(final_actions[:, 2]).max():.0f} N*m"
    )

    # ── Visualisation ────────────────────────────────────────────────────────
    print("\nGenerating visualisations...", flush=True)

    generate_static_figure(
        costmap_np, meta, start_w, goal_w, final_traj, history,
        out_dir / "mbd_denoising_snapshots.png",
    )

    if not args.no_animation:
        generate_animation(
            costmap_np, meta, start_w, goal_w, history,
            out_dir / "mbd_denoising_animation.mp4",
        )

    print(f"\nDone. Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
