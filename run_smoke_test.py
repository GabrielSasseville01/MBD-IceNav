#!/usr/bin/env python3
"""
Local smoke test: run a planner (default: astar) seed 0 on every scenario.

Results land in outputs/smoke_test/ (separate from the main experiment tree).
Already-completed trials are skipped automatically, so the script is safe to
interrupt and re-run.

Usage:
    uv run python run_smoke_test.py [--planner PLANNER] [--block A|B]
                                    [--scenario ID] [--with-plots] [--with-movies]
                                    [--with-search-movie] [--output-dir DIR]

--planner            Which planner to run: astar (default), mbd, eb-mbd, astar-eb-mbd, mppi, rss, ilqr.

--with-plots         Generate a map PNG per trial showing the ice field at t=0,
                     the ship's actual trajectory, and the first planned path.
                     Trials whose PNG already exists are skipped.
                     Trials whose JSON exists but PNG does not are re-run.

--with-movies        Generate an MP4 animation per trial (top-down 2D view, moving ice + ship).
                     Implies --with-plots (needs the full trajectory runner).
                     Trials whose MP4 already exists are skipped.
                     frame_skip=10 (DT=0.05s → every 0.5s simulated per rendered frame).

--with-search-movie      Generate an MP4 of A* node-expansion steps from the first replan.
                         Only meaningful for --planner astar. Implies --with-plots.
                         Output: seed_0_search.mp4.

--with-denoising-movie   Generate a denoising-evolution MP4 for the first replan.
                         Only meaningful for --planner mbd or eb-mbd. Implies --with-plots.
                         Output: replan_0000_denoising.mp4 (inside the planner output dir).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from medea_ice_ship.synthetic_map import (
    GOAL_POINT, GOAL_Y_LINE, SHIP_START, START_HEADING,
    X_LO, X_HI, Y_LO, Y_HI,
    list_scenarios, make_current_model, make_ice_field,
)
from medea_ice_ship.properties import ShipProperties
from medea_ice_ship.bonds import create_proximity_bonds
from medea_ice_ship.geometry import polygon_area
from medea_ice_ship.tessellation import tessellate_all_floes
from medea_ice_ship.geometry import create_ship_hull_at_position_heading
from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan
from medea_ice_ship import animate_ice_ship_2d, animate_lattice_search
from medea_ice_ship.visualization import WATER_COLOR, ICE_CMAP, ICE_EDGE_COLOR
from medea_ice_ship.planning import (
    create_medea_lattice_planner, plan_path, path_to_ship_trajectory_arr,
    optimize_with_diffusion, optimize_with_mppi, optimize_with_rss, optimize_with_ilqr,
)
from run_experiment_trial import (
    _make_plan_cfg, _build_cost_map, _build_cost_map_astar,
    _straight_line_path, _compute_metrics, _truncate_path_to_window,
    _make_reject_sampling_filter,
    run_trial,
    DT, MAX_SIM_TIME, REPLAN_EVERY, GOAL_TOL,
    DEFAULT_HORIZON,
    EB_C_MAX, EB_MU, EB_KAPPA, PLANNERS,
    AL_RHO, AL_M_MAX, AL_M_INIT_OOB, AL_RHO_GOAL,
    AL_RHO_ICE, AL_M_MAX_ICE,
    TERMINAL_GOAL_DISTANCE_WEIGHT,
)

DEFAULT_PLANNER = "astar"
SEED            = 0
OUTPUT_DIR      = Path("outputs/smoke_test")


# ── Map plot ───────────────────────────────────────────────────────────────

def _plot_map(spec, polygons_world, thicknesses, ship_history,
              first_plan_path, out_path, plan_cfg, planner):
    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    ax.set_facecolor(WATER_COLOR)

    patches = [MplPolygon(p, closed=True) for p in polygons_world]
    pc = PatchCollection(patches, cmap=ICE_CMAP, edgecolor=ICE_EDGE_COLOR,
                         linewidth=0.3, alpha=0.95)
    pc.set_array(thicknesses)
    pc.set_clim(float(thicknesses.min()), float(thicknesses.max()))
    ax.add_collection(pc)
    fig.colorbar(pc, ax=ax, shrink=0.5, label="Thickness [m]", location="left")

    if first_plan_path is not None and len(first_plan_path) >= 2:
        fp = np.asarray(first_plan_path)
        ax.plot(fp[:, 0], fp[:, 1], "r-", lw=2.5,
                label=f"{planner} plan ({len(fp)} wp)", zorder=5)

    positions = np.array([h[0] for h in ship_history])
    if len(positions) >= 2:
        ax.plot(positions[:, 0], positions[:, 1], "-",
                color="#FFD966", lw=2.0, label="Ship trajectory", zorder=5)

    ax.plot(*SHIP_START, "go", ms=12, zorder=6, label="Start")
    ax.plot(*GOAL_POINT, "r*", ms=16, zorder=6, label="Goal")
    ax.axhline(GOAL_Y_LINE, color="lime", ls="--", lw=2,
               label=f"Goal line y={GOAL_Y_LINE:.0f}", zorder=4)

    # Ship at its final pose (we're showing the executed trajectory)
    if len(ship_history) > 0:
        final_pos, final_heading = ship_history[-1]
        hull_pos = np.array(final_pos)
        hull_heading = float(final_heading)
    else:
        hull_pos = np.array(SHIP_START)
        hull_heading = float(START_HEADING)
    hull = create_ship_hull_at_position_heading(
        plan_cfg.ship_length, plan_cfg.ship_beam, hull_pos, hull_heading)
    ax.add_patch(MplPolygon(hull, closed=True, fc="orange", ec="black",
                            lw=1.5, alpha=0.85, zorder=7))

    pad = 5.0
    ax.set_xlim(X_LO - pad, X_HI + pad)
    ax.set_ylim(Y_LO - pad, Y_HI + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="upper right", fontsize=9)

    success = len(positions) > 0 and bool(positions[-1, 1] >= GOAL_Y_LINE - GOAL_TOL)
    conc_pct = int(round(spec.concentration * 100))
    n_cells = len(polygons_world)
    ax.set_title(
        f"{planner} — {spec.scenario_id} (seed={SEED})  success={success}\n"
        f"{n_cells} cells, conc={conc_pct}%  {spec.thickness_mode}  "
        f"curr={spec.with_currents}  "
        f"domain [{X_LO:.0f},{X_HI:.0f}]×[{Y_LO:.0f},{Y_HI:.0f}] m",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot  → {out_path}", flush=True)


# ── Trial runner with trajectory + plot ───────────────────────────────────

MOVIE_FRAME_SKIP  = 10
MOVIE_FPS         = 20
SEARCH_MOVIE_FPS  = 15


def _run_trial_for_plot(
    spec, output_dir: Path,
    planner: str = DEFAULT_PLANNER,
    with_movie: bool = False,
    with_search_movie: bool = False,
    with_denoising_movie: bool = False,
    horizon: int = DEFAULT_HORIZON,
):
    """
    Run a trial, capture ship trajectory and first planned path, save JSON + PNG.
    Optionally also saves a 2D MP4 and/or (for astar only) a search-expansion MP4.
    Returns (metrics_dict, png_path, mp4_path|None, search_mp4_path|None).
    """
    plan_cfg      = _make_plan_cfg(planner, horizon=horizon)
    current_model = make_current_model(spec)
    goal          = GOAL_POINT
    goal_y        = GOAL_Y_LINE

    print("Generating ice field...", flush=True)
    field = make_ice_field(spec, SEED)
    sat_polygons    = field["polygons"]
    sat_centroids   = field["centroids"]
    sat_thicknesses = field["thicknesses"]
    print(f"  {len(sat_polygons)} floes, conc={field['concentration']:.1%}", flush=True)

    if spec.no_fracture:
        print("Skipping tessellation (rigid floes, no fracture)...", flush=True)
        polygons    = list(sat_polygons)
        centroids   = np.asarray(sat_centroids)
        parent_ids  = np.arange(len(polygons))
        thicknesses = np.asarray(sat_thicknesses)
        radii       = np.sqrt(np.array([polygon_area(p) for p in polygons]) / np.pi)
        bonds       = []
        print(f"  {len(polygons)} rigid floes, 0 bonds", flush=True)
    else:
        print("Tessellating...", flush=True)
        polygons, centroids, parent_ids = tessellate_all_floes(
            sat_polygons, sat_centroids,
            cells_per_100m2=plan_cfg.cells_per_100m2_simple,
            seed=SEED,
        )
        thicknesses = sat_thicknesses[parent_ids]
        radii = np.sqrt(np.array([polygon_area(p) for p in polygons]) / np.pi)
        bonds = create_proximity_bonds(centroids, radii,
                                       threshold_factor=plan_cfg.bond_threshold_factor)
        print(f"  {len(polygons)} cells, {len(bonds)} bonds", flush=True)

    ship_props = ShipProperties(
        plan_cfg.ship_length, plan_cfg.ship_beam,
        plan_cfg.ship_draft, plan_cfg.ship_displacement,
    )

    n_frames_max    = int(MAX_SIM_TIME / DT)
    replan_n_frames = max(1, round(REPLAN_EVERY / DT))
    start_pose = (float(SHIP_START[0]), float(SHIP_START[1]), START_HEADING)
    init_traj  = path_to_ship_trajectory_arr(_straight_line_path(start_pose),
                                             n_frames_max)

    out_dir = output_dir / spec.block / spec.scenario_id / planner

    planning_times: list = []
    path_snapshots: list = []
    first_search_state: dict = {}   # astar only
    first_denoising_done = [False]  # mbd/eb-mbd only

    def _callback(frame, world_polygons, centroids_cb, thicknesses_cb,
                  ship_pos_f, ship_heading_f, n_remaining):
        t0   = time.perf_counter()
        sp   = (float(ship_pos_f[0]), float(ship_pos_f[1]), float(ship_heading_f))
        init = _straight_line_path(sp)

        try:
            if planner == "astar":
                astar_map, astar_bounds = _build_cost_map_astar(
                    world_polygons, centroids_cb, thicknesses_cb, plan_cfg)
                lattice = create_medea_lattice_planner(
                    astar_map, astar_bounds, sp, goal, plan_cfg,
                    goal_y_line=goal_y)
                capture_search = with_search_movie and not first_search_state
                if capture_search:
                    path, search_history = plan_path(
                        lattice, sp, goal, plan_cfg,
                        cost_map=astar_map, return_search_history=True)
                    first_search_state.update(
                        astar_map=astar_map, lattice=lattice,
                        search_history=search_history, path=path, start=sp,
                        world_polygons=list(world_polygons),
                        thicknesses=np.asarray(thicknesses_cb).copy())
                else:
                    path = plan_path(lattice, sp, goal, plan_cfg, cost_map=astar_map)

            elif planner == "astar-eb-mbd":
                # EB-MBD warm-started from the A* lattice path. Falls back to
                # cold start from the straight-line if A* fails.
                try:
                    astar_map, astar_bounds = _build_cost_map_astar(
                        world_polygons, centroids_cb, thicknesses_cb, plan_cfg)
                    lattice = create_medea_lattice_planner(
                        astar_map, astar_bounds, sp, goal, plan_cfg,
                        goal_y_line=goal_y)
                    astar_path = plan_path(lattice, sp, goal, plan_cfg,
                                           cost_map=astar_map)
                except Exception:
                    astar_path = None

                if (astar_path is not None and len(astar_path) >= 2
                        and np.all(np.isfinite(astar_path))):
                    seed_path  = astar_path
                    warm_start = True
                else:
                    seed_path  = init
                    warm_start = False

                cost_map = _build_cost_map(centroids_cb, thicknesses_cb, radii, plan_cfg)
                _dmovie = None
                _viz_polys = None
                _viz_thicks = None
                if with_denoising_movie and not first_denoising_done[0]:
                    _dmovie = str(out_dir)
                    _viz_polys = list(world_polygons)
                    _viz_thicks = np.asarray(thicknesses_cb).copy()
                    first_denoising_done[0] = True
                path = optimize_with_diffusion(
                    seed_path, sp, goal, cost_map, plan_cfg,
                    start_nu=(2.0, 0.0, 0.0), warm_start=warm_start,
                    goal_y_line=None,
                    terminal_goal_distance_weight=1.0,
                    use_eb_mbd=True,
                    eb_x_min=float(cost_map.x_origin),
                    eb_x_max=float(cost_map.x_origin + cost_map.width),
                    eb_y_min=float(cost_map.y_origin),
                    eb_c_max=EB_C_MAX, eb_mu=EB_MU, eb_kappa=EB_KAPPA,
                    denoising_movie_dir=_dmovie,
                    viz_world_polygons=_viz_polys,
                    viz_thicknesses=_viz_thicks,
                )

            else:
                cost_map = _build_cost_map(centroids_cb, thicknesses_cb, radii, plan_cfg)
                _dmovie = None
                _viz_polys = None
                _viz_thicks = None
                if with_denoising_movie and not first_denoising_done[0]:
                    _dmovie = str(out_dir)
                    _viz_polys = list(world_polygons)
                    _viz_thicks = np.asarray(thicknesses_cb).copy()
                    first_denoising_done[0] = True
                # MBD: ice in costmap, goal-line enforced by reject filter
                #      (drops samples with y_T < goal_y_line)  +  terminal
                #      Euclidean pull.
                # AL-MBD: ice as AL disk constraints; goal-line as AL
                #      goal-reach column  +  terminal Euclidean pull.
                # EB-MBD: hybrid EB+AL — EB barrier on goal-line + OOB,
                #      AL multipliers on ice disks  +  terminal Euclidean.
                # MPPI/RSS keep Big-M goal_y_line (no constraint mechanism).
                # iLQR uses make_terminal_cost with goal_y_line=None.
                if planner == "mbd":
                    path = optimize_with_diffusion(
                        init, sp, goal, cost_map, plan_cfg,
                        start_nu=(2.0, 0.0, 0.0), warm_start=False,
                        goal_y_line=None,
                        goal_y_line_penalty=0.0,
                        terminal_goal_distance_weight=TERMINAL_GOAL_DISTANCE_WEIGHT,
                        oob_penalty=0.0,
                        sample_filter_fn=_make_reject_sampling_filter(
                            plan_cfg, goal_y_line=goal_y),
                        denoising_movie_dir=_dmovie,
                        viz_world_polygons=_viz_polys,
                        viz_thicknesses=_viz_thicks,
                    )
                elif planner == "eb-mbd":
                    disks = [
                        (float(c[0]), float(c[1]), float(r))
                        for c, r in zip(centroids_cb, radii)
                    ]
                    eb_cfg = replace(plan_cfg, planning_jax_mbd_costmap_weight=0.0)
                    path = optimize_with_diffusion(
                        init, sp, goal, cost_map, eb_cfg,
                        start_nu=(2.0, 0.0, 0.0), warm_start=False,
                        goal_y_line=goal_y,
                        goal_y_line_penalty=0.0,
                        terminal_goal_distance_weight=TERMINAL_GOAL_DISTANCE_WEIGHT,
                        use_eb_mbd=True,
                        eb_x_min=float(cost_map.x_origin),
                        eb_x_max=float(cost_map.x_origin + cost_map.width),
                        eb_y_min=float(cost_map.y_origin),
                        eb_c_max=EB_C_MAX, eb_mu=EB_MU, eb_kappa=EB_KAPPA,
                        eb_oob_as_constraint=True,
                        use_hybrid_eb_al_mbd=True,
                        hybrid_goal_tolerance=None,
                        al_rho_ice=AL_RHO_ICE,
                        al_M_max_ice=AL_M_MAX_ICE,
                        obstacle_disks=disks,
                        denoising_movie_dir=_dmovie,
                        viz_world_polygons=_viz_polys,
                        viz_thicknesses=_viz_thicks,
                    )
                elif planner == "al-mbd":
                    disks = [
                        (float(c[0]), float(c[1]), float(r))
                        for c, r in zip(centroids_cb, radii)
                    ]
                    al_cfg = replace(plan_cfg, planning_jax_mbd_costmap_weight=0.0)
                    path = optimize_with_diffusion(
                        init, sp, goal, cost_map, al_cfg,
                        start_nu=(2.0, 0.0, 0.0), warm_start=False,
                        goal_y_line=goal_y,
                        goal_y_line_penalty=0.0,
                        terminal_goal_distance_weight=TERMINAL_GOAL_DISTANCE_WEIGHT,
                        oob_penalty=0.0,
                        use_al_mbd=True,
                        al_x_min=float(cost_map.x_origin),
                        al_x_max=float(cost_map.x_origin + cost_map.width),
                        al_y_min=float(cost_map.y_origin),
                        al_rho=AL_RHO,
                        al_M_max=AL_M_MAX,
                        al_M_init_oob=AL_M_INIT_OOB,
                        al_rho_goal=AL_RHO_GOAL,
                        obstacle_disks=disks,
                        denoising_movie_dir=_dmovie,
                        viz_world_polygons=_viz_polys,
                        viz_thicknesses=_viz_thicks,
                    )
                elif planner == "mppi":
                    path = optimize_with_mppi(
                        init, sp, goal, cost_map, plan_cfg,
                        start_nu=(2.0, 0.0, 0.0), warm_start=False,
                        goal_y_line=None,
                        terminal_goal_distance_weight=1.0,
                        y_bottom=float(Y_LO),
                    )
                elif planner == "rss":
                    path = optimize_with_rss(
                        init, sp, goal, cost_map, plan_cfg,
                        start_nu=(2.0, 0.0, 0.0), warm_start=False,
                        goal_y_line=None,
                        terminal_goal_distance_weight=1.0,
                        y_bottom=float(Y_LO),
                    )
                elif planner == "ilqr":
                    path = optimize_with_ilqr(
                        init, sp, goal, cost_map, plan_cfg,
                        start_nu=(2.0, 0.0, 0.0), warm_start=False,
                        goal_y_line=None,
                        y_bottom=float(Y_LO),
                    )
                else:
                    raise ValueError(f"Unknown planner: {planner!r}")

        except Exception:
            path = None

        planning_times.append(time.perf_counter() - t0)
        if path is None or len(path) < 2 or not np.all(np.isfinite(path)):
            return None

        # MPC: only execute the next replan-window's worth of the plan at 4 m/s.
        path = _truncate_path_to_window(
            path, n_remaining, plan_cfg.planning_ship_speed, DT)
        path_snapshots.append((frame, np.array(path, dtype=np.float64)))
        path_len = float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
        n_path_frames = max(1, int(path_len / float(plan_cfg.planning_ship_speed) / DT))
        return path_to_ship_trajectory_arr(path, min(n_remaining, n_path_frames))

    print("Running simulation...", flush=True)
    t0 = time.perf_counter()
    result = simulate_jax_chunked_replan(
        polygons, centroids, bonds,
        np.array(SHIP_START), plan_cfg.ship_length, plan_cfg.ship_beam,
        n_frames_max, DT,
        drift_vel=np.array([0.0, 0.0]),
        thicknesses=thicknesses,
        ship_props=ship_props,
        ship_trajectory=init_traj,
        return_ship_diagnostics=True,
        replan_every_n_frames=replan_n_frames,
        replan_callback=_callback,
        parent_ids=parent_ids,
        ocean_current=current_model,
        goal_y_line=GOAL_Y_LINE,
        goal_tolerance=GOAL_TOL,
        use_auto_icenav_ship=True,
        verbose=True,
    )
    wall_time = time.perf_counter() - t0
    print(f"  Simulation done in {wall_time:.1f} s", flush=True)

    positions_history, rotations_history, bonds_history, ship_history, _, ship_diagnostics = result

    metrics = _compute_metrics(ship_diagnostics, ship_history, GOAL_Y_LINE, DT)
    metrics["n_replans"]      = len(planning_times)
    metrics["planning_times"] = planning_times
    metrics["wall_time_s"]    = wall_time

    out_rec = {
        "block": spec.block, "scenario_id": spec.scenario_id,
        "concentration": spec.concentration, "thickness_mode": spec.thickness_mode,
        "with_currents": spec.with_currents, "no_fracture": spec.no_fracture,
        "planner": planner, "seed": SEED,
        **metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"seed_{SEED}.json"
    with open(json_path, "w") as f:
        json.dump(out_rec, f, indent=2)
    print(f"  JSON  → {json_path}", flush=True)

    first_plan = path_snapshots[0][1] if path_snapshots else None
    png_path   = out_dir / f"seed_{SEED}_map.png"
    _plot_map(spec, polygons, thicknesses, ship_history, first_plan,
              png_path, plan_cfg, planner)

    mp4_path = None
    if with_movie:
        mp4_path = out_dir / f"seed_{SEED}_movie.mp4"
        print("Generating 2D movie...", flush=True)
        t0 = time.perf_counter()
        animate_ice_ship_2d(
            polygons, positions_history, rotations_history, bonds_history, bonds,
            ship_history, plan_cfg.ship_length, plan_cfg.ship_beam,
            centroids, thicknesses,
            output_file=str(mp4_path),
            fps=MOVIE_FPS,
            frame_skip=MOVIE_FRAME_SKIP,
            n_workers=1,
            view_bounds=(X_LO, X_HI, Y_LO, Y_HI),
            ocean_current=current_model,
            path_snapshots=path_snapshots if path_snapshots else None,
            goal_y_line=GOAL_Y_LINE,
            initial_floe_of=parent_ids,
            dt=DT,
        )
        print(f"  Movie → {mp4_path}  ({time.perf_counter() - t0:.1f} s)", flush=True)

    search_mp4_path = None
    if with_search_movie and planner == "astar":
        if first_search_state:
            search_mp4_path = out_dir / f"seed_{SEED}_search.mp4"
            print("Generating A* search expansion movie...", flush=True)
            t0 = time.perf_counter()
            n_exp = max(1, first_search_state["lattice"].nodes_expanded)
            frame_stride = max(1, n_exp // 300)
            animate_lattice_search(
                first_search_state["astar_map"],
                first_search_state["search_history"],
                first_search_state["path"],
                first_search_state["start"],
                GOAL_POINT,
                output_file=str(search_mp4_path),
                frame_stride=frame_stride,
                fps=SEARCH_MOVIE_FPS,
                world_polygons=first_search_state.get("world_polygons"),
                thicknesses=first_search_state.get("thicknesses"),
            )
            print(f"  Search movie → {search_mp4_path}  ({time.perf_counter() - t0:.1f} s)",
                  flush=True)
        else:
            print("  No search history captured (first plan may have raised an exception).",
                  flush=True)

    denoising_mp4_path = None
    if with_denoising_movie and planner in ("mbd", "eb-mbd", "astar-eb-mbd"):
        _dp = out_dir / "replan_0000_denoising.mp4"
        if _dp.exists():
            denoising_mp4_path = _dp

    return out_rec, png_path, mp4_path, search_mp4_path, denoising_mp4_path


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--planner", choices=PLANNERS, default=DEFAULT_PLANNER,
                   help=f"Planner to run (default: {DEFAULT_PLANNER})")
    p.add_argument("--block", choices=["A", "B", "C"],
                   help="Restrict to one block (default: all A, B, C). "
                        "C is the rigid-floe twin of A (no tessellation).")
    p.add_argument("--scenario",
                   help="Run only this scenario ID, e.g. A_conc05_thin (implies --block)")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help="Root output directory (default: outputs/smoke_test)")
    p.add_argument("--with-plots", action="store_true",
                   help="Generate a map PNG per trial (re-runs trials that lack a PNG)")
    p.add_argument("--with-movies", action="store_true",
                   help="Generate a 2D MP4 movie per trial (implies --with-plots)")
    p.add_argument("--with-search-movie", action="store_true",
                   help="Generate A* node-expansion MP4 from first replan (astar only, implies --with-plots)")
    p.add_argument("--with-denoising-movie", action="store_true",
                   help="Generate denoising-evolution MP4 for first replan (mbd/eb-mbd only, implies --with-plots)")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                   help=("Planning horizon (action steps) for trajopt planners "
                         "(mbd/eb-mbd/mppi/rss/ilqr). A* ignores it. "
                         f"Default: {DEFAULT_HORIZON}."))
    args = p.parse_args()
    if args.with_movies or args.with_search_movie or args.with_denoising_movie:
        args.with_plots = True

    planner               = args.planner
    output_dir            = Path(args.output_dir)
    with_plots            = args.with_plots
    with_movies           = args.with_movies
    with_search_movie     = args.with_search_movie
    with_denoising_movie  = args.with_denoising_movie
    horizon               = args.horizon

    # Build scenario list
    if args.scenario:
        inferred_block = args.scenario[0] if args.scenario[0] in ("A", "B") else None
        if inferred_block is None:
            p.error(f"Cannot infer block from scenario {args.scenario!r}; use --block too")
        block = args.block or inferred_block
        from medea_ice_ship.synthetic_map import get_scenario
        scenarios = [get_scenario(block, args.scenario)]
    else:
        blocks    = [args.block] if args.block else ["A", "B", "C"]
        scenarios = [spec for block in blocks for spec in list_scenarios(block)]

    n_total   = len(scenarios)
    n_done    = 0
    n_skipped = 0
    n_failed  = 0
    results_log = []

    flags = []
    if with_movies:          flags.append("with-movies")
    if with_search_movie:    flags.append("with-search-movie")
    if with_denoising_movie: flags.append("with-denoising-movie")
    if not flags and with_plots: flags.append("with-plots")
    scenario_ids = ", ".join(s.scenario_id for s in scenarios[:3])
    if len(scenarios) > 3: scenario_ids += f" ... ({len(scenarios)} total)"
    print(f"\n{'='*70}")
    print(f"  Smoke test: {planner}  seed={SEED}  ({n_total} scenarios: {scenario_ids})"
          + (f"  [{', '.join(flags)}]" if flags else ""))
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    for i, spec in enumerate(scenarios, 1):
        out_dir         = output_dir / spec.block / spec.scenario_id / planner
        json_path            = out_dir / f"seed_{SEED}.json"
        png_path             = out_dir / f"seed_{SEED}_map.png"
        mp4_path             = out_dir / f"seed_{SEED}_movie.mp4"
        search_mp4_path      = out_dir / f"seed_{SEED}_search.mp4"
        denoising_mp4_path   = out_dir / "replan_0000_denoising.mp4"

        json_exists          = json_path.exists()
        png_exists           = png_path.exists()
        mp4_exists           = mp4_path.exists()
        search_mp4_exists    = search_mp4_path.exists()
        denoising_mp4_exists = denoising_mp4_path.exists()

        need_plots    = with_plots  and not (json_exists and png_exists)
        need_movie    = with_movies and not mp4_exists
        need_search   = with_search_movie and planner == "astar" and not search_mp4_exists
        need_denoising = with_denoising_movie and planner in ("mbd", "eb-mbd", "astar-eb-mbd") and not denoising_mp4_exists

        if json_exists and not need_plots and not need_movie and not need_search and not need_denoising:
            artifacts = "JSON"
            if png_exists: artifacts += " + PNG"
            if with_movies and mp4_exists: artifacts += " + MP4"
            if with_search_movie and search_mp4_exists: artifacts += " + search-MP4"
            if with_denoising_movie and denoising_mp4_exists: artifacts += " + denoising-MP4"
            print(f"[{i:02d}/{n_total}] SKIP  {spec.scenario_id}  ({artifacts} exist)")
            n_skipped += 1
            results_log.append((spec.scenario_id, "skip", None))
            continue

        print(f"\n[{i:02d}/{n_total}] START {spec.scenario_id}", flush=True)
        try:
            if with_plots:
                out, png, mp4, search_mp4, denoising_mp4 = _run_trial_for_plot(
                    spec, output_dir,
                    planner=planner,
                    with_movie=with_movies,
                    with_search_movie=with_search_movie,
                    with_denoising_movie=with_denoising_movie,
                    horizon=horizon,
                )
            else:
                out = run_trial(
                    block       = spec.block,
                    scenario_id = spec.scenario_id,
                    planner     = planner,
                    seed        = SEED,
                    output_dir  = output_dir,
                    horizon     = horizon,
                )
                png        = None
                mp4          = None
                search_mp4   = None
                denoising_mp4 = None

            status = "OK  " if out["success"] else "FAIL"
            line = (f"[{i:02d}/{n_total}] {status}  {spec.scenario_id}  "
                    f"success={out['success']}  "
                    f"f_max={out['f_max']:.0f} N  "
                    f"path={out['path_length']:.0f} m  "
                    f"wall={out['wall_time_s']:.1f} s")
            if png:           line += f"  → {png.name}"
            if mp4:           line += f"  → {mp4.name}"
            if search_mp4:    line += f"  → {search_mp4.name}"
            if denoising_mp4: line += f"  → {denoising_mp4.name}"
            print(line)
            results_log.append((spec.scenario_id, status.strip(), out))
            n_done += 1

        except Exception:
            print(f"[{i:02d}/{n_total}] ERROR {spec.scenario_id}")
            traceback.print_exc()
            results_log.append((spec.scenario_id, "error", None))
            n_failed += 1

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {planner}  seed={SEED}")
    print(f"{'='*70}")
    hdr = f"  {'Scenario':<32} {'Status':<8} {'Success':<9} {'f_max (N)':<12} {'Path (m)':<10} {'Wall (s)'}"
    print(hdr)
    print(f"  {'-'*32} {'-'*8} {'-'*9} {'-'*12} {'-'*10} {'-'*8}")
    for scenario_id, status, out in results_log:
        if out is not None:
            print(f"  {scenario_id:<32} {status:<8} {str(out['success']):<9} "
                  f"{out['f_max']:<12.0f} {out['path_length']:<10.0f} {out['wall_time_s']:.1f}")
        else:
            print(f"  {scenario_id:<32} {status:<8}")

    n_success = sum(1 for _, _, o in results_log if o is not None and o["success"])
    print(f"\n  Completed: {n_done}  Skipped: {n_skipped}  Errors: {n_failed}  Total: {n_total}")
    if n_done:
        print(f"  Success rate: {n_success}/{n_done} = {n_success/n_done:.0%}")
    print()


if __name__ == "__main__":
    main()
