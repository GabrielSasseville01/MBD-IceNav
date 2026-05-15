#!/usr/bin/env python3
"""
Visualise how MBD (Model-Based Diffusion) iterations evolve.

Generates a multi-panel figure showing:
  1. Top-left: Cost convergence (mean and min) over diffusion steps.
  2. Top-right: Weight entropy over diffusion steps (how concentrated the
     importance weights become).
  3. Bottom (wide): Trajectory evolution on the costmap — the mean trajectory
     at several diffusion steps overlaid on the costmap, with top-K sample
     trajectories shown as thin lines.

Usage:
  uv run python examples/visualize_diffusion.py
  uv run python examples/visualize_diffusion.py --concentration 0.3
  uv run python examples/visualize_diffusion.py --no-warm-start
  uv run python examples/visualize_diffusion.py --ndiffuse 100 --nsample 1024

Output:  outputs/diffusion_viz/diffusion_evolution.png
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from dataclasses import replace

from medea_ice_ship import (
    ShipProperties,
    polygon_area,
    polygon_centroid,
    create_proximity_bonds,
    tessellate_all_floes,
    build_costmap_from_ice_state,
    create_medea_lattice_planner,
    plan_path,
    path_to_ship_trajectory,
)
from medea_ice_ship.config import get_default_run_config, get_default_simulation_config
from medea_ice_ship.ice_field_generator import generate_ice_field
from medea_ice_ship.jax_mbd_planner import (
    make_dynamics_rollout,
    make_cost_fn,
    run_mbd_with_diagnostics,
    _warm_start_from_path,
    _jax_backend_info,
)


def main():
    parser = argparse.ArgumentParser(
        description="Visualise MBD diffusion iteration evolution",
    )
    parser.add_argument("--concentration", type=float, default=0.3,
                        help="Ice concentration for generated field (default: 0.3)")
    parser.add_argument("--map-size", type=float, default=1400.0,
                        help="Square map side length [m] (default: 1400)")
    parser.add_argument("--ndiffuse", type=int, default=None,
                        help="Override number of diffusion steps")
    parser.add_argument("--nsample", type=int, default=None,
                        help="Override number of MC samples")
    parser.add_argument("--no-warm-start", action="store_true",
                        help="Disable A* warm-start (cold start)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-top", type=int, default=15,
                        help="Number of top-K sample trajectories to show per step")
    parser.add_argument("--output-dir", type=str, default="outputs/diffusion_viz")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy lattice for warm-start (faster A*)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = get_default_run_config()
    run_cfg = replace(run_cfg, planning_turning_radius=50.0, planning_greedy=args.greedy)
    if args.ndiffuse is not None:
        run_cfg = replace(run_cfg, planning_jax_mbd_ndiffuse=args.ndiffuse)
    if args.nsample is not None:
        run_cfg = replace(run_cfg, planning_jax_mbd_nsample=args.nsample)
    if args.no_warm_start:
        run_cfg = replace(run_cfg, planning_jax_mbd_warm_start=False)

    # ── Generate ice field ─────────────────────────────────────────────
    print("Generating ice field...", flush=True)
    field = generate_ice_field(
        concentration=args.concentration,
        map_shape=(args.map_size, args.map_size),
        seed=args.seed,
    )
    sat_polys = field["polygons"]
    sat_cents = field["centroids"]
    sat_thick = field["thicknesses"]
    x_lo, x_hi, y_lo, y_hi = field["domain"]
    print(f"  {len(sat_polys)} floes, actual {field['concentration']:.1%}", flush=True)

    polys, cents, pids = tessellate_all_floes(
        sat_polys, sat_cents,
        cells_per_100m2=run_cfg.cells_per_100m2_medea, seed=42,
    )
    thicknesses = sat_thick[pids]
    areas = np.array([polygon_area(p) for p in polys])
    radii = np.sqrt(areas / np.pi)
    bonds = create_proximity_bonds(cents, radii, threshold_factor=run_cfg.bond_threshold_factor)
    print(f"  Tessellated: {len(polys)} cells, {len(bonds)} bonds", flush=True)

    ship_length = run_cfg.ship_length
    ship_beam = run_cfg.ship_beam
    inset = ship_length / 2 + run_cfg.path_margin
    ship_start = np.array([(x_lo + x_hi) / 2, y_lo + inset])
    ship_end = np.array([(x_lo + x_hi) / 2, y_hi - inset])
    goal_y_line = float(y_hi)
    start_heading = np.arctan2(ship_end[1] - ship_start[1], ship_end[0] - ship_start[0])
    start_pose = (float(ship_start[0]), float(ship_start[1]), float(start_heading))
    goal_pt = (float(ship_end[0]), float(ship_end[1]))

    # ── Build costmap ──────────────────────────────────────────────────
    wall_bounds = (x_lo, x_hi, y_lo, y_hi)
    cost_map, bounds = build_costmap_from_ice_state(
        sat_polys, sat_cents, sat_thick,
        x_lo, x_hi, y_lo, y_hi,
        resolution=run_cfg.planning_resolution,
        margin=run_cfg.planning_margin,
        ship_mass=run_cfg.ship_displacement,
        collision_cost=run_cfg.planning_ice_collision_cost,
        wall_bounds=wall_bounds,
    )
    print("  Costmap built.", flush=True)

    # ── Lattice A* path (warm-start seed) ──────────────────────────────
    warm_start = run_cfg.planning_jax_mbd_warm_start
    lattice_path = None
    if warm_start:
        print("  Running lattice A* for warm-start...", flush=True)
        planner = create_medea_lattice_planner(
            cost_map=cost_map, bounds=bounds,
            start=start_pose, goal=goal_pt, run_cfg=run_cfg,
            goal_y_line=goal_y_line,
        )
        lattice_path = plan_path(planner, start_pose, goal_pt, run_cfg, cost_map=cost_map)
        if lattice_path is not None:
            print(f"    A* path: {len(lattice_path)} waypoints", flush=True)
        else:
            print("    A* found no path; cold start.", flush=True)

    # ── Set up MBD components (mirror jax_mbd_optimize) ────────────────
    H = run_cfg.planning_jax_mbd_horizon
    Nsample = run_cfg.planning_jax_mbd_nsample
    Ndiffuse = run_cfg.planning_jax_mbd_ndiffuse
    temperature = run_cfg.planning_jax_mbd_temperature
    beta0 = run_cfg.planning_jax_mbd_beta0
    betaT = run_cfg.planning_jax_mbd_betaT
    dt_substep = run_cfg.planning_jax_mbd_dt_substep
    n_substeps = run_cfg.planning_jax_mbd_n_substeps
    init_surge_bias = run_cfg.planning_jax_mbd_init_surge_bias
    seed_val = run_cfg.planning_diffusion_seed or 42

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
    nu0 = jnp.array((2.0, 0.0, 0.0), dtype=jnp.float32)

    surge_max = run_cfg.planning_jax_mbd_surge_max
    sway_max = run_cfg.planning_jax_mbd_sway_max
    yaw_max = run_cfg.planning_jax_mbd_yaw_max
    action_center = jnp.array([surge_max / 2, 0.0, 0.0], dtype=jnp.float32)
    action_scale = jnp.array([surge_max / 2, sway_max, yaw_max], dtype=jnp.float32)

    grid = jnp.array(cost_map.cost_map, dtype=jnp.float32)
    meta = {
        "x_origin": float(cost_map.x_origin),
        "y_origin": float(cost_map.y_origin),
        "resolution": float(cost_map.resolution),
        "rows": int(cost_map.rows),
        "cols": int(cost_map.cols),
    }

    Ybar_init = None
    if warm_start and lattice_path is not None and len(lattice_path) >= 3:
        dt_action = dt_substep * n_substeps
        speed_est = 4.0
        coverage = H * dt_action * speed_est
        seg_lens = np.linalg.norm(np.diff(lattice_path[:, :2], axis=0), axis=1)
        s = np.zeros(len(lattice_path))
        s[1:] = np.cumsum(seg_lens)
        cutoff = max(min(int(np.searchsorted(s, coverage, side="right")), len(lattice_path)), 2)
        Ybar_init = _warm_start_from_path(lattice_path[:cutoff], H, init_surge_bias)

    cost_fn = make_cost_fn(
        grid, meta, goal_pt, rollout_fn, eta0, nu0,
        action_center, action_scale,
        costmap_weight=run_cfg.planning_jax_mbd_costmap_weight,
        goal_weight=run_cfg.planning_jax_mbd_goal_weight,
        length_weight=run_cfg.planning_jax_mbd_length_weight,
        smoothness_weight=run_cfg.planning_jax_mbd_smoothness_weight,
        ship_length=L, ship_beam=B,
    )

    # ── Run MBD with diagnostics ───────────────────────────────────────
    backend = _jax_backend_info()
    ws_label = "warm-started" if Ybar_init is not None else "cold start"
    print(f"\nRunning MBD [{backend}]: H={H}, N={Nsample}, "
          f"Ndiffuse={Ndiffuse}, temp={temperature}, {ws_label}", flush=True)
    t0 = time.perf_counter()
    diag = run_mbd_with_diagnostics(
        cost_fn=cost_fn,
        rollout_fn=rollout_fn,
        eta0=eta0, nu0=nu0,
        action_center=action_center,
        action_scale=action_scale,
        H=H, Nsample=Nsample, Ndiffuse=Ndiffuse,
        temperature=temperature,
        beta0=beta0, betaT=betaT,
        seed=seed_val,
        Ybar_init=Ybar_init,
        init_surge_bias=init_surge_bias,
        n_top_samples=args.n_top,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done ({elapsed:.1f}s)", flush=True)

    # ── Plot ───────────────────────────────────────────────────────────
    n_steps = len(diag["cost_means"])
    mean_trajs = diag["mean_trajectories"]   # len = n_steps + 1 (includes initial)
    sample_trajs = diag["sample_trajectories"]  # len = n_steps
    cost_means = diag["cost_means"]
    cost_mins = diag["cost_mins"]
    all_weights = diag["weights"]

    # Select ~8 evenly spaced steps for the trajectory panel
    n_show = min(8, n_steps)
    show_idx = np.linspace(0, n_steps - 1, n_show, dtype=int)

    cmap_traj = cm.get_cmap("viridis", n_show)

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], hspace=0.25, wspace=0.25)

    # ── Panel 1: Cost convergence ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    steps = np.arange(1, n_steps + 1)
    ax1.plot(steps, cost_means, "b-", lw=1.5, label="Mean cost")
    ax1.plot(steps, cost_mins, "r-", lw=1.5, label="Min cost")
    ax1.set_xlabel("Diffusion step")
    ax1.set_ylabel("Cost")
    ax1.set_title("Cost Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # ── Panel 2: Weight entropy ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    entropies = []
    max_weights = []
    for w in all_weights:
        w_clip = np.clip(w, 1e-12, None)
        entropies.append(float(-np.sum(w_clip * np.log(w_clip))))
        max_weights.append(float(w.max()))
    ax2.plot(steps, entropies, "g-", lw=1.5, label="Entropy")
    ax2.set_xlabel("Diffusion step")
    ax2.set_ylabel("Weight entropy (nats)")
    ax2.set_title("Importance Weight Concentration")
    ax2.grid(True, alpha=0.3)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(steps, max_weights, "m--", lw=1, alpha=0.7, label="Max weight")
    ax2_twin.set_ylabel("Max weight", color="m")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")

    # ── Panel 3: Trajectory evolution on costmap ───────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    cm_data = cost_map.cost_map.copy()
    cm_data[cm_data > 500] = 500
    extent = [
        cost_map.x_origin,
        cost_map.x_origin + cost_map.width,
        cost_map.y_origin,
        cost_map.y_origin + cost_map.height,
    ]
    ax3.imshow(cm_data, origin="lower", extent=extent,
               cmap="YlOrRd", alpha=0.5, aspect="equal")

    # Show lattice path if available
    if lattice_path is not None:
        ax3.plot(lattice_path[:, 0], lattice_path[:, 1],
                 "k--", lw=1, alpha=0.4, label="A* warm-start", zorder=2)

    # Show sample trajectories at selected steps (thin, transparent)
    for panel_i, step_i in enumerate(show_idx):
        color = cmap_traj(panel_i / max(n_show - 1, 1))
        if step_i < len(sample_trajs):
            top_t = sample_trajs[step_i]
            for k in range(len(top_t)):
                ax3.plot(top_t[k, :, 0], top_t[k, :, 1],
                         color=color, lw=0.3, alpha=0.25, zorder=3)

    # Show mean trajectories at selected steps (thick)
    for panel_i, step_i in enumerate(show_idx):
        color = cmap_traj(panel_i / max(n_show - 1, 1))
        # step_i in diag refers to after that many reverse steps
        # mean_trajectories[0] is initial, mean_trajectories[step_i+1] is after step_i
        traj_i = mean_trajs[step_i + 1] if step_i + 1 < len(mean_trajs) else mean_trajs[-1]
        label = f"Step {step_i + 1}/{n_steps}"
        ax3.plot(traj_i[:, 0], traj_i[:, 1],
                 color=color, lw=2.5, alpha=0.85, zorder=4, label=label)

    # Show initial mean trajectory (before any diffusion)
    ax3.plot(mean_trajs[0][:, 0], mean_trajs[0][:, 1],
             "k-", lw=1.5, alpha=0.5, zorder=3, label="Initial (step 0)")

    # Final trajectory (bold)
    final_traj = mean_trajs[-1]
    ax3.plot(final_traj[:, 0], final_traj[:, 1],
             "w-", lw=4, alpha=0.9, zorder=5)
    ax3.plot(final_traj[:, 0], final_traj[:, 1],
             color=cmap_traj(1.0), lw=2.5, zorder=6, label="Final")

    ax3.plot(start_pose[0], start_pose[1], "gs", ms=14, zorder=7, label="Start")
    ax3.axhline(goal_y_line, color="magenta", ls="--", lw=2, label=f"Goal y={goal_y_line:.0f}")

    pad = 30.0
    ax3.set_xlim(x_lo - pad, x_hi + pad)
    ax3.set_ylim(y_lo - pad, y_hi + pad)
    ax3.set_aspect("equal")
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.set_title(f"MBD Trajectory Evolution — {n_steps} diffusion steps, {Nsample} samples")
    ax3.legend(loc="upper left", fontsize=8, ncol=2)

    fig.suptitle(
        f"MBD Diffusion Evolution — conc {args.concentration:.0%}, "
        f"H={H}, N={Nsample}, Ndiffuse={Ndiffuse}, temp={temperature}\n"
        f"Backend: {backend}, {ws_label}, elapsed: {elapsed:.1f}s",
        fontsize=13,
    )

    out_path = out_dir / "diffusion_evolution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path.resolve()}")
    plt.close(fig)

    # ── Panel-per-step figure (detailed view) ──────────────────────────
    n_detail = min(12, n_steps)
    detail_idx = np.linspace(0, n_steps - 1, n_detail, dtype=int)
    ncols = min(4, n_detail)
    nrows = int(np.ceil(n_detail / ncols))
    fig2, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    for panel_i, step_i in enumerate(detail_idx):
        r, c = divmod(panel_i, ncols)
        ax = axes[r, c]
        ax.imshow(cm_data, origin="lower", extent=extent,
                  cmap="YlOrRd", alpha=0.5, aspect="equal")

        # Sample trajectories
        if step_i < len(sample_trajs):
            top_t = sample_trajs[step_i]
            # Color by rank (best = blue, worst = red)
            for k in range(len(top_t)):
                frac = k / max(len(top_t) - 1, 1)
                c_sample = cm.get_cmap("coolwarm")(frac)
                ax.plot(top_t[k, :, 0], top_t[k, :, 1],
                        color=c_sample, lw=0.8, alpha=0.5)

        # Mean trajectory
        traj_i = mean_trajs[step_i + 1] if step_i + 1 < len(mean_trajs) else mean_trajs[-1]
        ax.plot(traj_i[:, 0], traj_i[:, 1], "b-", lw=2, zorder=5)

        ax.plot(start_pose[0], start_pose[1], "gs", ms=8, zorder=6)
        ax.axhline(goal_y_line, color="magenta", ls="--", lw=1)
        ax.set_xlim(x_lo - pad, x_hi + pad)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_aspect("equal")

        cost_str = f"cost: {cost_mins[step_i]:.0f}" if step_i < len(cost_mins) else ""
        ax.set_title(f"Step {step_i + 1}/{n_steps}\n{cost_str}", fontsize=9)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for panel_i in range(n_detail, nrows * ncols):
        r, c = divmod(panel_i, ncols)
        axes[r, c].set_visible(False)

    fig2.suptitle(
        f"MBD Per-Step Detail — samples colored blue (best) → red (worst)",
        fontsize=12,
    )
    fig2.tight_layout()
    out_path2 = out_dir / "diffusion_per_step.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path2.resolve()}")
    plt.close(fig2)

    print(f"\nDone. Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
