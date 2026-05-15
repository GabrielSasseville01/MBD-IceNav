#!/usr/bin/env python3
"""
DPDEM: MEDEA Satellite Ice Floes with Ship Fracture Simulation

Entry point: builds ice field (simple or MEDEA), runs simulation via medea_ice_ship package,
optionally saves/loads state and generates 2D/3D animations.

Usage (with uv):
    uv run python run_medea_ship_fracture.py           # Full run with animations
    uv run python run_medea_ship_fracture.py --quick   # Simulation only, no animations
    uv run python run_medea_ship_fracture.py --no-3d   # Skip 3D animation

  Plan then simulate (lattice planner path instead of straight line):
    uv run python run_medea_ship_fracture.py --simple --plan   # Planned path (use with --simple, --toy-sparse, or --full-map)
  Real-time 2D view (debugging; opens a window that updates each frame):
    uv run python run_medea_ship_fracture.py --simple --live   # Optional: --quick to skip writing movies
  Save/replay (run simulation once, generate animations later):
    uv run python run_medea_ship_fracture.py --simple --quick --save-state scenario.pkl   # Simulate + save, no animations
    uv run python run_medea_ship_fracture.py --load-state scenario.pkl                    # Generate 2D and 3D from saved state
    uv run python run_medea_ship_fracture.py --load-state scenario.pkl --no-3d             # Replay 2D only
"""

import os
import numpy as np
from pathlib import Path
import time
import argparse
import multiprocessing
import matplotlib.pyplot as plt

# Use spawn to avoid fork + multithreaded libs (e.g. JAX) deadlock when parallelizing animation
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def _clip_path_to_goal(path_arr, goal_pt, goal_y_line_val=None):
    """Ensure path ends at the goal.

    For goal_y_line: truncates at the first waypoint crossing y >= goal_y_line,
    interpolates the exact crossing, and appends it so the path ends on the goal line.
    If the path never reaches goal_y_line, appends the goal point as a fallback.

    For point goals: always appends the goal point if the path does not already end
    there (removes the old distance-tolerance restriction).
    """
    if path_arr is None or len(path_arr) == 0:
        return path_arr
    path_arr = np.asarray(path_arr, dtype=float)
    if goal_y_line_val is not None:
        for i in range(len(path_arr)):
            if path_arr[i, 1] >= goal_y_line_val:
                if i == 0:
                    return path_arr[:1]
                y0, y1 = path_arr[i - 1, 1], path_arr[i, 1]
                t = (goal_y_line_val - y0) / (y1 - y0) if y1 > y0 else 0.0
                cx = path_arr[i - 1, 0] + t * (path_arr[i, 0] - path_arr[i - 1, 0])
                da = (path_arr[i, 2] - path_arr[i - 1, 2] + np.pi) % (2 * np.pi) - np.pi
                cpsi = path_arr[i - 1, 2] + t * da
                return np.vstack([path_arr[:i], [[cx, goal_y_line_val, cpsi]]])
        # Path doesn't reach goal_y_line — append goal point as fallback
        gp = np.asarray(goal_pt, dtype=float)
        if np.hypot(path_arr[-1, 0] - gp[0], path_arr[-1, 1] - gp[1]) > 1e-3:
            return np.vstack([path_arr, [[gp[0], gp[1], path_arr[-1, 2]]]])
        return path_arr
    else:
        gp = np.asarray(goal_pt, dtype=float)
        if np.hypot(path_arr[-1, 0] - gp[0], path_arr[-1, 1] - gp[1]) > 1e-3:
            return np.vstack([path_arr, [[gp[0], gp[1], path_arr[-1, 2]]]])
        return path_arr


def _make_rejection_sample_filter(cost_map, ship_length, ship_beam, goal_y_line=None, goal=None):
    """Build a sample_filter_fn for reject-sampling in jax_mbd_optimize.

    Mirrors the toy_mbd_bimodal logic: rejects samples whose hull corners exit
    the costmap x-extent or whose centre dips below the y floor, and (when
    goal_y_line is set) that never reach the goal line on the last waypoint.
    """
    _x_lo = float(cost_map.x_origin)
    _x_hi = float(cost_map.x_origin) + float(cost_map.cols) * float(cost_map.resolution)
    _y_lo = float(cost_map.y_origin)

    _half_l = ship_length / 2.0
    _half_w = ship_beam   / 2.0
    _dx = np.array([0.0,  _half_l,  _half_l, -_half_l, -_half_l], dtype=np.float32)
    _dy = np.array([0.0,  _half_w, -_half_w,  _half_w, -_half_w], dtype=np.float32)

    _goal_y = float(goal_y_line) if goal_y_line is not None else None
    _goal_pt = (float(goal[0]), float(goal[1])) if goal is not None else None

    def _filter(Y0s_norm: np.ndarray, trajs: np.ndarray) -> np.ndarray:
        xs  = trajs[:, :, 0]   # (N, H+1)
        ys  = trajs[:, :, 1]
        psi = trajs[:, :, 2]

        c = np.cos(psi)
        s = np.sin(psi)
        hull_x = (xs[:, :, None]
                  + c[:, :, None] * _dx[None, None, :]
                  - s[:, :, None] * _dy[None, None, :])

        oob = (
            ((hull_x < _x_lo) | (hull_x > _x_hi)).any(axis=(1, 2))
            | (ys < _y_lo).any(axis=1)
        )

        if _goal_y is not None:
            reaches = ys[:, -1] >= _goal_y
        elif _goal_pt is not None:
            dist = np.hypot(xs[:, -1] - _goal_pt[0], ys[:, -1] - _goal_pt[1])
            reaches = dist < 50.0  # within 50 m of goal
        else:
            reaches = np.ones(trajs.shape[0], dtype=bool)

        return (~oob) & reaches

    return _filter


def _generate_diffusion_denoising_movie(
    diag: dict,
    cost_map,
    polygons,
    thicknesses,
    start,
    goal,
    goal_y_line,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    output_path: str,
    fps: int = 10,
    no_show: bool = True,
):
    """
    Generate an MP4 animation of the diffusion denoising process.

    Each frame shows the distribution of sampled trajectories (top-K by cost)
    overlaid on the costmap, evolving from the initial noisy state to the
    refined final trajectory across Ndiffuse denoising steps.

    Args:
        diag: Diagnostics dict from run_mbd_with_diagnostics (via optimize_with_diffusion
              with return_diagnostics=True). Keys: mean_trajectories, sample_trajectories,
              sample_costs, cost_means, cost_mins, weights.
        cost_map: CostMap used for planning.
        polygons: List of ice polygon vertex arrays [(N,2), ...].
        thicknesses: Array of ice thickness values aligned with polygons.
        start: (x, y, psi) start pose.
        goal: (gx, gy) goal position.
        goal_y_line: If not None, draw a horizontal goal line at this y.
        x_min, x_max, y_min, y_max: World bounds for axis limits.
        output_path: Path to save the MP4 file.
        fps: Frames per second for the movie (default 10).
        no_show: If True, close the figure without displaying.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection, LineCollection
    from matplotlib import animation

    mean_trajs = diag["mean_trajectories"]   # list of (H+1, 3), length Ndiffuse
    sample_trajs = diag["sample_trajectories"]  # list of (Nsample, H+1, 3), length Ndiffuse-1
    cost_mins = diag.get("cost_mins", [])
    cost_means = diag.get("cost_means", [])

    # Frame k shows mean_trajs[k] alongside sample_trajs[k] (aligned: same iteration).
    # sample_trajs has Ndiffuse-1 entries (no samples for the last frame).
    n_frames = len(mean_trajs)  # = Ndiffuse
    n_samples = len(sample_trajs[0]) if sample_trajs else 0

    print(
        f"  Generating diffusion movie: {n_frames} frames, {n_samples} samples/frame -> {output_path}",
        flush=True,
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # ── Static background ──────────────────────────────────────────────
    grid = cost_map.cost_map
    extent = [
        cost_map.x_origin, cost_map.x_origin + cost_map.width,
        cost_map.y_origin, cost_map.y_origin + cost_map.height,
    ]
    vmax_grid = float(np.percentile(grid[grid > 0], 95)) if np.any(grid > 0) else 1.0
    ax.imshow(grid, origin='lower', extent=extent, cmap='YlOrRd', alpha=0.5,
              vmin=0, vmax=vmax_grid)

    patches = []
    colors_th = []
    for poly, th in zip(polygons, thicknesses):
        patches.append(MplPolygon(poly, closed=True))
        colors_th.append(th)
    if patches:
        pc = PatchCollection(patches, cmap='Blues', edgecolor='#555555', linewidth=0.3, alpha=0.6)
        pc.set_array(np.array(colors_th))
        pc.set_clim(float(thicknesses.min()), float(thicknesses.max()))
        ax.add_collection(pc)

    if goal_y_line is not None:
        ax.axhline(goal_y_line, color='lime', ls='--', lw=2, zorder=4, label=f'Goal y={goal_y_line:.0f}')
    else:
        ax.plot(goal[0], goal[1], 'mx', ms=14, mew=3, zorder=6, label='Goal')
    ax.plot(start[0], start[1], 'go', ms=12, zorder=6, label='Start')

    pad = 30.0
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    # ── Dynamic artists (updated each frame) ──────────────────────────
    # All sample trajectories via LineCollection (efficient for large Nsample).
    # Sorted best→worst by cost; colored blue (best) → magenta (worst).
    sample_norm = plt.Normalize(vmin=0, vmax=1)
    sample_lc = LineCollection([], cmap='cool', norm=sample_norm,
                                linewidths=0.6, alpha=0.25, zorder=4)
    ax.add_collection(sample_lc)

    # Mean trajectory: bold white outline + colored line
    mean_outline, = ax.plot([], [], 'w-', lw=4, zorder=5)
    mean_line, = ax.plot([], [], '-', color='tomato', lw=2, zorder=5, label='Mean trajectory')

    title_obj = ax.set_title('Diffusion denoising — step 0 / ?')
    ax.legend(loc='upper right', fontsize=8)

    def _update(frame):
        # Mean trajectory for this frame
        mt = mean_trajs[frame]
        mean_outline.set_data(mt[:, 0], mt[:, 1])
        mean_line.set_data(mt[:, 0], mt[:, 1])

        # Sample trajectories: frame k → sample_trajs[k] (same iteration as mean)
        if frame < len(sample_trajs) and sample_trajs:
            st = sample_trajs[frame]  # (Nsample, H+1, 3), sorted best→worst
            segments = [st[k, :, :2] for k in range(len(st))]
            colors = np.linspace(0, 1, len(st))
            sample_lc.set_segments(segments)
            sample_lc.set_array(colors)
        else:
            sample_lc.set_segments([])

        # Title with cost info
        cost_str = ""
        if frame < len(cost_mins):
            cost_str = f"  |  cost_min={cost_mins[frame]:.1f}, cost_mean={cost_means[frame]:.1f}"
        title_obj.set_text(
            f"Diffusion denoising — step {frame} / {n_frames - 1}{cost_str}"
        )

    ani = animation.FuncAnimation(fig, _update, frames=n_frames, interval=1000 // fps)

    from medea_ice_ship.visualization import _animation_writer
    writer, output_path = _animation_writer(output_path, fps)
    ani.save(output_path, writer=writer, dpi=120)

    if not no_show:
        plt.show()
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='MEDEA Ice Ship Fracture Simulation')
    parser.add_argument('--live', action='store_true',
                        help='Open a 2D window and update it in real time during simulation (for debugging)')
    parser.add_argument('--quick', action='store_true', help='Simulation only, no animations')
    parser.add_argument('--no-3d', action='store_true', help='Skip 3D animation')
    parser.add_argument('--full-map', action='store_true',
                        help='Use full ice map; ship diagonal bottom-left to top-right')
    parser.add_argument('--simple', action='store_true',
                        help='Use minimal synthetic ice field (3 floes) for fast testing (no MEDEA data)')
    parser.add_argument('--toy-sparse', action='store_true',
                        help='Use sparse toy ice field: varying thickness (e.g. a few ~10 m icebergs, rest 0.5–2 m)')
    parser.add_argument('--ice-tank-toy', action='store_true',
                        help='Use NRC ice-tank-style toy field: 75 m × 32 m basin with floes, best with --ice-tank-psv')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default from config)')
    parser.add_argument('--save-state', type=str, metavar='FILE', default=None,
                        help='Save full simulation state to FILE for replay (pickle)')
    parser.add_argument('--load-state', type=str, metavar='FILE', default=None,
                        help='Load state from FILE and run visualization only (no re-simulation)')
    parser.add_argument('--thick-ice', type=float, metavar='M', default=None,
                        help='Override ice thickness to M metres for all cells (e.g. 10 for thick-ice test)')
    parser.add_argument('--thrust-cap', type=float, metavar='N', default=None,
                        help='Cap ice-on-ship force to N Newtons (e.g. 25e6 for 25 MN)')
    parser.add_argument('--no-thrust-cap', action='store_true',
                        help='Disable thrust cap (unlimited ship thrust)')
    parser.add_argument('--ice-tank-psv', action='store_true',
                        help='Use AUTO-IceNav ice-tank PSV-scale vessel (L≈83 m, B≈17 m, displacement≈8 kt) instead of default Fossen supply vessel (L=76.2 m, B=17 m, 6 kt)')
    parser.add_argument('--icebreaker', action='store_true',
                        help='Use the large icebreaker vessel (L=158 m, B=28 m, displacement≈23 kt) from config instead of default Fossen supply vessel (L=76.2 m, B=17 m, 6 kt)')
    parser.add_argument('--gif', action='store_true',
                        help='Output GIF instead of MP4 (slower, larger files)')
    parser.add_argument('--plan', action='store_true',
                        help='Use lattice planner to plan path (costmap + A*); only with --simple, --toy-sparse, or --full-map')
    parser.add_argument('--fast-lattice', action='store_true',
                        help='Use faster A* params (coarser resolution, fewer headings) — speeds up full-map planning while keeping path quality')
    parser.add_argument('--no-lattice-search-movie', action='store_true', dest='no_lattice_search_movie',
                        help='Skip A* lattice search visualization movie (generated by default when --plan)')
    parser.add_argument('--planner', type=str, choices=['lattice', 'diffusion-mpc', 'mppi', 'sqp', 'gradcem'], default='diffusion-mpc',
                        help='Planner to use with --plan: lattice (A*), diffusion-mpc (default), mppi, sqp, or gradcem')
    parser.add_argument('--optimize-path', action='store_true',
                        help='Refine planned path with NLP (path length + costmap + curvature); requires casadi; use with --plan')
    parser.add_argument('--warm-start', action='store_true',
                        help='With diffusion-mpc/mppi/sqp/gradcem: warm-start from A* lattice path (default is cold start)')
    parser.add_argument('--no-mss-dynamics', action='store_true',
                        help='Use simple diagonal M/D instead of MSS-style 3-DOF (Fossen added mass + damping)')
    parser.add_argument('--kinematic-ship', action='store_true',
                        help='With --plan: place ship exactly on planned path each frame (no dynamics). Use to verify plan vs actual.')
    parser.add_argument('--no-dpdem-bonds', action='store_true',
                        help='Use legacy bond model (force vs strength) instead of default DPDEM-style (hybrid criterion, damage, Mohr-Coulomb)')
    parser.add_argument('--view-map', action='store_true',
                        help='Show the ice map in a matplotlib window and exit (no simulation or planning)')
    parser.add_argument('--plan-only', action='store_true',
                        help='Plan path, save costmap+path to file, then exit (no simulation). Implies --plan')
    parser.add_argument('--compare-planners', action='store_true',
                        help='With --plan-only: run both lattice A* and diffusion-mpc, plot both paths for comparison')
    parser.add_argument('--no-show', action='store_true',
                        help='With --plan-only: save to file only, do not open matplotlib window (headless)')
    parser.add_argument('--diffusion-movie', action='store_true',
                        help='With --plan-only and diffusion-mpc planner: also generate an MP4 movie of the denoising distribution evolving across diffusion steps')
    parser.add_argument('--replan-every', type=float, metavar='SECS', default=None,
                        help='With --plan: replan every SECS simulated seconds (override config default of 30 s; larger = fewer replans, faster run)')
    parser.add_argument('--ridge-costmap', type=str, metavar='FILE', default=None,
                        help='Ridge costmap pickle for ridge-aware planning (from generate_ridge_costmap.py)')
    parser.add_argument('--ridge-weight', type=float, default=None, metavar='W',
                        help='Ridge cost weight (default 2.0 when --ridge-costmap set)')
    parser.add_argument('--ice-brake-force', type=float, default=None, metavar='N',
                        help='Ice brake threshold [N]: when surge ice force exceeds this, '
                             'setpoint is blended toward ship position to reduce DP thrust in ice. '
                             'Default: disabled (0). Example: 1e6 (1 MN)')
    parser.add_argument('--no-replan', action='store_true',
                        help='With --plan: use single initial plan for whole run (no receding-horizon replanning)')
    parser.add_argument('--output-dir', type=str, default='outputs/default',
                        help='Output directory for movies and plots (default: outputs/default)')
    parser.add_argument('--current', type=str, default='none',
                        choices=['none', 'uniform', 'learned'],
                        help='Ocean current mode: none (default), uniform, learned (SeaCurrentRNN)')
    parser.add_argument('--current-speed', type=float, default=None,
                        help='Current speed [m/s] (uniform), or peak speed of synthetic field (learned). Default from config.')
    parser.add_argument('--current-direction', type=float, default=None,
                        help='Current direction [degrees, 0=+x] (uniform mode only)')
    parser.add_argument('--current-field', type=str, default=None,
                        help='Initial current field (.npy/.pkl/.pth) for learned mode. If omitted a synthetic field is generated.')
    parser.add_argument('--seanet-weights', type=str, default=None,
                        help='Path to trained seanet.pth (learned mode). Train with: python scripts/train_seanet.py --synthetic')
    parser.add_argument('--current-only', action='store_true',
                        help='With --simple: ship stays far left (no collision). Use to verify ice drifts from current alone.')
    parser.add_argument('--frames', type=int, default=None, metavar='N',
                        help='Early-stop: run only the first N frames of the simulation. Does not affect ship speed, trajectory interpolation, or planning. Simulated time = N × dt.')
    parser.add_argument('--target-speed', type=float, default=None, metavar='M/S',
                        help='Target ship speed [m/s] when --plan. Derives n_frames so path is traversed at this speed. Also used in KE collision cost. Default 4.0 (realistic icebreaker).')
    parser.add_argument('--optimize-speed', action='store_true',
                        help='Optimize speed per path segment: slower in thick ice, faster in open water. Balances impact (KE ∝ v²) vs time.')
    parser.add_argument('--float32', action='store_true',
                        help='Use float32 instead of float64 for JAX computation (faster on GPU, less accurate). '
                             'Default is float64 for numerical accuracy.')
    parser.add_argument('--mbd-viz-dir', type=str, default=None, metavar='DIR',
                        help='With --planner diffusion-mpc: save per-replan debug plots (density, top-5, cost breakdown) to DIR.')
    parser.add_argument('--mbd-goal-line', type=float, default=None, metavar='Y',
                        help='With --planner diffusion-mpc: replace soft goal cost with a big-M penalty for trajectories that never reach y >= Y. Sets goal_weight=0.')
    parser.add_argument('--polygon-replan', action='store_true',
                        help='Use polygon-based costmap (same as initial plan) for replanning instead of JAX disk approximation. Slower but matches AUTO-IceNav exactly.')
    parser.add_argument('--mbd-denoising-movie', action='store_true',
                        help='With --mbd-viz-dir: also save a denoising-evolution MP4 (replan_XXXX_denoising.mp4) for each replan step.')
    # ── Reject-sampling filter ────────────────────────────────────────────
    parser.add_argument('--reject-sampling', action='store_true',
                        help='With diffusion-mpc: at each denoising step resample until all Nsample trajectories are non-OOB and reach the goal. '
                             'Uses hull-corner OOB check (matching costmap query) and goal-line check.')
    parser.add_argument('--max-resample-iters', type=int, default=None, metavar='N',
                        help='Maximum resampling rounds per denoising step (only with --reject-sampling). Default: unlimited.')
    # ── EB-MBD args ───────────────────────────────────────────────────────
    parser.add_argument('--eb-mbd', action='store_true',
                        help='Enable Emerging-Barrier MBD (EB-MBD): replaces big-M OOB/goal penalties with a '
                             'progressively tightening log-barrier over the reverse diffusion.')
    parser.add_argument('--eb-c-max', type=float, default=1500.0, metavar='M',
                        help='EB-MBD barrier offset at the start of reverse diffusion [m]. '
                             'Should be >= max expected constraint violation. Default: 1500.')
    parser.add_argument('--eb-mu', type=float, default=50000.0, metavar='F',
                        help='EB-MBD barrier hardness (cost units). Default: 50000.')
    parser.add_argument('--eb-kappa', type=float, default=1.0, metavar='K',
                        help='EB-MBD barrier schedule shape: kappa=1 -> linear, kappa>1 -> slow then fast. Default: 1.0.')
    parser.add_argument('--eb-goal-x-weight', type=float, default=0.0, metavar='W',
                        help='EB-MBD soft cost weight for x-deviation at goal-line crossing. Default: 0 (disabled).')
    parser.add_argument('--eb-oob-constraint', action='store_true',
                        help='EB-MBD: enforce OOB as a hard barrier constraint instead of the default linear soft cost.')
    parser.add_argument('--eb-length-weight', type=float, default=0.0, metavar='W',
                        help='EB-MBD soft path-length cost weight. Default: 0 (disabled).')
    args = parser.parse_args()

    if args.plan_only:
        args.plan = True
    if getattr(args, 'compare_planners', False):
        args.plan_only = True
        args.plan = True

    # Set interactive matplotlib backend before importing medea_ice_ship (so visualization uses it for --live / --view-map / --plan-only)
    if args.plan_only and getattr(args, 'no_show', False):
        import matplotlib
        matplotlib.use('Agg')  # Headless: save to file only
    elif args.live or args.view_map or args.plan_only:
        os.environ['DPDEM_LIVE_VIEW'] = '1'
        import matplotlib
        matplotlib.use('TkAgg')
    from medea_ice_ship import (
        ShipProperties,
        load_medea_floes,
        scale_floes_to_world,
        create_simple_ice_field,
        create_sparse_toy_ice_field,
        create_ice_tank_basin,
        create_ship_hull_at_position_heading,
        polygon_area,
        polygon_centroid,
        create_proximity_bonds,
        assign_thickness_by_area,
        tessellate_all_floes,
        animate_ice_ship_2d,
        animate_ice_ship_3d,
        animate_costmap_movie,
        make_live_2d_callback,
        plot_hull_impact_vectors,
        animate_lattice_search,
        plot_costmap_with_grid,
        plot_lattice_search_costs,
        animate_lattice_primitives_stepwise,
        save_simulation_state,
        load_simulation_state,
        build_costmap_from_ice_state,
        create_medea_lattice_planner,
        create_medea_diffusion_mpc_planner,
        plan_path,
        optimize_with_diffusion,
        optimize_with_mppi,
        optimize_with_sqp,
        optimize_with_gradcem,
        optimize_speed_profile,
        path_to_ship_trajectory_arr,
        create_ocean_current_model,
    )
    from medea_ice_ship.config import RunConfig, get_default_run_config, get_default_simulation_config
    from dataclasses import replace as replace_config
    if args.live:
        import matplotlib.pyplot as plt

    run_cfg = get_default_run_config()
    # Optional: override ship parameters to match AUTO-IceNav ice-tank PSV-scale vessel
    if getattr(args, 'ice_tank_psv', False):
        # 1:45 scale model is 1.84 m × 0.38 m, 90 kg → full-scale ≈ 83 m × 17 m, ~8,200 t
        run_cfg = replace_config(
            run_cfg,
            ship_length=83.0,
            ship_beam=17.0,
            ship_draft=7.0,
            ship_displacement=8_200e3,
        )
        print("  [Vessel] Using AUTO-IceNav ice-tank PSV-scale ship (L=83 m, B=17 m, mass≈8.2 kt)", flush=True)
    if args.seed is None:
        args.seed = run_cfg.default_seed
    video_ext = '.gif' if args.gif else '.mp4'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    np.random.seed(args.seed)

    if args.load_state:
        state = load_simulation_state(args.load_state)
        out_2d = state.get('output_2d') or ('replay_2d.gif' if args.gif else 'replay_2d.mp4')
        out_3d = state.get('output_3d') or ('replay_3d.gif' if args.gif else 'replay_3d.mp4')
        print("\nGenerating 2D animation (replay)...", flush=True)
        animate_ice_ship_2d(
            state['polygons'], state['positions_history'], state['rotations_history'],
            state['bonds_history'], state['bonds'],
            state['ship_history'], state['ship_length'], state['ship_beam'],
            state['centroids'], state['thicknesses'],
            output_file=out_2d, fps=run_cfg.fps_2d, frame_skip=run_cfg.frame_skip_2d,
            n_workers=run_cfg.n_workers,
            dt=state.get('dt', run_cfg.dt),
        )
        if not args.no_3d:
            print("\nGenerating 3D animation (replay)...", flush=True)
            animate_ice_ship_3d(
                state['polygons'], state['positions_history'], state['rotations_history'],
                state['bonds_history'], state['bonds'],
                state['ship_history'], state['ship_length'], state['ship_beam'], state['ship_draft'],
                state['centroids'], state['thicknesses'],
                output_file=out_3d, fps=run_cfg.fps_3d, frame_skip=run_cfg.frame_skip_3d,
                submergence_history=state.get('submergence_history'),
                figsize=run_cfg.figsize_3d,
                n_workers=run_cfg.n_workers,
                dt=state.get('dt', run_cfg.dt),
            )
        print(f"\nReplay complete. Output: {out_2d}" + (f", {out_3d}" if not args.no_3d else ""))
        return

    # ── JAX backend setup ─────────────────────────────────────────────────────
    import jax
    _use_float32 = getattr(args, 'float32', False)
    if not _use_float32:
        jax.config.update("jax_enable_x64", True)
    print("=" * 70)
    print(f"JAX backend: {[str(d) for d in jax.devices()]}")
    print(f"  Precision: {'float32' if _use_float32 else 'float64'}")
    print("=" * 70)

    t_setup_start = time.time()
    print("=" * 70)
    print("DPDEM: MEDEA Satellite Ice + Ship Fracture Simulation")
    synthetic_mode = args.simple or args.toy_sparse or args.ice_tank_toy
    if args.simple:
        print("  [SIMPLE MODE - synthetic ice field for quick testing]")
    if args.toy_sparse:
        print("  [TOY SPARSE - sparse ice field with varying thickness (icebergs + moderate)]")
    if args.full_map:
        print("  [FULL MAP MODE - ship diagonal bottom-left to top-right]")
    if args.quick:
        print("  [QUICK MODE - simulation only]")
    elif args.no_3d:
        print("  [NO-3D MODE - skipping 3D animation]")
    print("=" * 70)
    print()

    ship_length = run_cfg.ship_length
    ship_beam = run_cfg.ship_beam
    ship_draft = run_cfg.ship_draft
    ship_displacement = run_cfg.ship_displacement

    # Use AUTO-IceNav vessel geometry by default so DPDEM ice forces and moments (hull, lever arms)
    # match the simulated dynamics (Fossen supply 76.2 m, 6 kt). Override if --ice-tank-psv or --icebreaker was set.
    if getattr(args, 'icebreaker', False):
        print(f"  [Vessel] Large icebreaker: L={ship_length:.1f} m, B={ship_beam:.1f} m, mass={ship_displacement/1e6:.0f} kt", flush=True)
    elif not getattr(args, 'ice_tank_psv', False):
        ship_length = 76.2
        ship_beam = 17.0
        ship_draft = 7.0
        ship_displacement = 6000.0e3
        print("  [Vessel] AUTO-IceNav Fossen supply: L=76.2 m, B=17 m, mass=6 kt (ice forces use this geometry)", flush=True)

    sat_thicknesses_override = None  # when set, use instead of assign_thickness_by_area
    if args.ice_tank_toy:
        t0 = time.time()
        basin_x, basin_y = (0.0, 1100.0), (0.0, 200.0)
        ice_coverage = 0.2
        print(f"Creating ice-tank basin {basin_x[1]:.0f}×{basin_y[1]:.0f} m with {ice_coverage*100:.0f}% ice (no overlapping floes)...", flush=True)
        sat_polygons, sat_centroids, sat_thicknesses_override = create_ice_tank_basin(
            seed=args.seed,
            domain_x=basin_x,
            domain_y=basin_y,
            ice_coverage_fraction=ice_coverage,
            floe_size_range=(8.0, 45.0),
            thickness_range=(0.5, 2.0),
        )
        print(f"  Created {len(sat_polygons)} floes ({time.time()-t0:.1f}s)", flush=True)
        all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
            sat_polygons, sat_centroids, cells_per_100m2=run_cfg.cells_per_100m2_simple, seed=args.seed
        )
        polygons = all_cells
        centroids = all_cell_centroids
        parent_floe_ids = all_parent_ids
        print(f"  Tessellated to {len(polygons)} cells", flush=True)
    elif args.toy_sparse:
        t0 = time.time()
        print("Creating sparse toy ice field (varying thickness: icebergs ~10 m, moderate 0.5–2 m)...", flush=True)
        sat_polygons, sat_centroids, sat_thicknesses_override = create_sparse_toy_ice_field(seed=args.seed)
        print(f"  Created {len(sat_polygons)} floes ({time.time()-t0:.1f}s)", flush=True)
        all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
            sat_polygons, sat_centroids, cells_per_100m2=run_cfg.cells_per_100m2_simple, seed=args.seed
        )
        polygons = all_cells
        centroids = all_cell_centroids
        parent_floe_ids = all_parent_ids
        print(f"  Tessellated to {len(polygons)} cells", flush=True)
    elif args.simple:
        t0 = time.time()
        print("Creating simple synthetic ice field (3 floes)...", flush=True)
        sat_polygons, sat_centroids = create_simple_ice_field(seed=args.seed)
        print(f"  Created {len(sat_polygons)} floes ({time.time()-t0:.1f}s)", flush=True)
        all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
            sat_polygons, sat_centroids, cells_per_100m2=run_cfg.cells_per_100m2_simple, seed=args.seed
        )
        polygons = all_cells
        centroids = all_cell_centroids
        parent_floe_ids = all_parent_ids
        print(f"  Tessellated to {len(polygons)} cells", flush=True)
    else:
        data_dir = Path(__file__).parent / 'data'
        pickle_path = data_dir / 'ice_floes_MEDEA_fram_20100629.pkl'
        if not pickle_path.exists():
            print(f"ERROR: Data file not found: {pickle_path}")
            return
        t0 = time.time()
        print("Loading MEDEA ice floe data...", flush=True)
        polygons_px, centroids_px, metadata = load_medea_floes(pickle_path)
        print(f"  Loaded {metadata['n_floes']} floes ({time.time()-t0:.1f}s)", flush=True)
        print(f"Scaling to world coordinates ({run_cfg.meters_per_pixel} m/px)...", flush=True)
        all_polygons, all_centroids = scale_floes_to_world(
            polygons_px, centroids_px, run_cfg.meters_per_pixel, center_domain=True, subsample_region=None
        )
        if args.full_map:
            sat_polygons, sat_centroids = all_polygons, all_centroids
            print(f"  Using full ice map: {len(sat_polygons)} floes", flush=True)
        else:
            y_center = (all_centroids[:, 1].min() + all_centroids[:, 1].max()) / 2
            keep_mask = np.abs(all_centroids[:, 1] - y_center) < run_cfg.y_band
            sat_polygons = [all_polygons[i] for i in range(len(all_polygons)) if keep_mask[i]]
            sat_centroids = all_centroids[keep_mask]
            print(f"  Selected {len(sat_polygons)} floes in ship's path (y +/- {run_cfg.y_band}m)", flush=True)
        print("Tessellating floes internally for fracturing...", flush=True)
        all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
            sat_polygons, sat_centroids, cells_per_100m2=run_cfg.cells_per_100m2_medea, seed=args.seed
        )
        print(f"  Created {len(all_cells)} total cells", flush=True)
        if args.full_map:
            polygons, centroids, parent_floe_ids = all_cells, all_cell_centroids, all_parent_ids
            print(f"  Using all {len(polygons)} cells (full map)", flush=True)
        else:
            ship_y_center = (sat_centroids[:, 1].min() + sat_centroids[:, 1].max()) / 2
            keep_cells = np.abs(all_cell_centroids[:, 1] - ship_y_center) < run_cfg.cell_y_band
            polygons = [all_cells[i] for i in range(len(all_cells)) if keep_cells[i]]
            centroids = all_cell_centroids[keep_cells]
            parent_floe_ids = all_parent_ids[keep_cells]
            print(f"  Filtered to {len(polygons)} cells in ship's path (y +/- {run_cfg.cell_y_band}m)", flush=True)

    areas = np.array([polygon_area(p) for p in polygons])
    radii = np.sqrt(areas / np.pi)
    print("Assigning ice thickness...", flush=True)
    if sat_thicknesses_override is not None:
        thicknesses = sat_thicknesses_override[parent_floe_ids]
        print(f"  Using per-floe thickness from sparse toy field", flush=True)
    else:
        sat_areas = np.array([polygon_area(p) for p in sat_polygons])
        sat_thicknesses = assign_thickness_by_area(
            sat_areas, min_thickness=run_cfg.thickness_min, max_thickness=run_cfg.thickness_max, power=run_cfg.thickness_power
        )
        thicknesses = sat_thicknesses[parent_floe_ids]
    if args.thick_ice is not None:
        thicknesses = np.full_like(thicknesses, float(args.thick_ice))
        print(f"  Thickness overridden to {args.thick_ice} m (thick-ice test)", flush=True)
    else:
        print(f"  Thickness range: {thicknesses.min():.2f} - {thicknesses.max():.2f} m", flush=True)
    print("Creating bonds between cells...", flush=True)
    bonds = create_proximity_bonds(centroids, radii, threshold_factor=run_cfg.bond_threshold_factor)
    print(f"  Created {len(bonds)} bonds", flush=True)
    all_verts = np.vstack(polygons)
    domain_width = all_verts[:, 0].max() - all_verts[:, 0].min()
    domain_height = all_verts[:, 1].max() - all_verts[:, 1].min()
    print(f"  Domain: {domain_width:.0f} x {domain_height:.0f} m", flush=True)

    ship_props = ShipProperties(ship_length, ship_beam, ship_draft, ship_displacement)
    print(f"\nShip: {ship_length}m x {ship_beam}m, mass={ship_props.mass/1e6:.1f} kt", flush=True)
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
    print(f"  Ice domain X: {x_min:.1f} to {x_max:.1f}m, Y: {y_min:.1f} to {y_max:.1f}m", flush=True)

    DOMAIN_LIMIT = 700.0
    if args.full_map:
        x_min = -DOMAIN_LIMIT #max(x_min, -DOMAIN_LIMIT)
        x_max = DOMAIN_LIMIT #min(x_max,  DOMAIN_LIMIT)
        y_min = -DOMAIN_LIMIT #max(y_min, -DOMAIN_LIMIT)
        y_max = DOMAIN_LIMIT #min(y_max,  DOMAIN_LIMIT)
        print(f"  Clamped domain: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]", flush=True)
    if getattr(args, 'ice_tank_toy', False):
        x_min, x_max = 0.0, 1100.0
        y_min, y_max = 0.0, 200.0
        print(f"  Ice-tank basin domain: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}] m", flush=True)

    if args.view_map:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        patches = []
        colors = []
        for i, poly in enumerate(polygons):
            patches.append(MplPolygon(poly, closed=True))
            colors.append(thicknesses[i])
        pc = PatchCollection(patches, cmap='Blues', edgecolor='#555555', linewidth=0.3)
        pc.set_array(np.array(colors))
        pc.set_clim(thicknesses.min(), thicknesses.max())
        ax.add_collection(pc)
        for i, j in bonds:
            ax.plot([centroids[i, 0], centroids[j, 0]],
                    [centroids[i, 1], centroids[j, 1]],
                    'k-', lw=0.15, alpha=0.3)
        pad = 30.0
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        mode = 'simple' if synthetic_mode else ('full-map' if args.full_map else 'band')
        ax.set_title(f'Ice Map ({mode}) — {len(polygons)} cells, {len(bonds)} bonds\n'
                     f'Domain: [{x_min:.0f}, {x_max:.0f}] x [{y_min:.0f}, {y_max:.0f}] m')
        cb = fig.colorbar(pc, ax=ax, shrink=0.7, label='Thickness [m]')
        plt.tight_layout()
        print(f"\n  Showing ice map ({len(polygons)} cells). Close the window to exit.", flush=True)
        plt.show()
        return

    goal_y_line = None
    if synthetic_mode:
        n_frames = run_cfg.n_frames_simple
        ship_start = np.array([x_min - ship_length / 2 - run_cfg.path_margin, (y_min + y_max) / 2])
        ship_end = np.array([x_max + ship_length / 2 + run_cfg.path_margin, (y_min + y_max) / 2])
        if getattr(args, 'current_only', False):
            # Ship stays far left so it never reaches the ice; ice drifts under current only.
            ship_end = ship_start.copy()
            print(f"  [--current-only] Ship stationary at ({ship_start[0]:.0f}, {ship_start[1]:.0f}); ice drifts under current only", flush=True)
        ship_heading = 0.0
        ship_trajectory = [(ship_start + (frame / n_frames) * (ship_end - ship_start), ship_heading) for frame in range(n_frames + 1)]
        ship_pos = ship_trajectory[0][0]
        drift_vel = 0.0
        print(f"  Ship path: ({ship_start[0]:.0f}, {ship_start[1]:.0f}) -> ({ship_end[0]:.0f}, {ship_end[1]:.0f}), {n_frames} frames", flush=True)
    elif args.full_map:
        inset = ship_length / 2 + run_cfg.path_margin
        # Original full-map setup: start near bottom-left, goal near top edge (y=600)
        ship_start = np.array([-545.0, -600.0])
        ship_end   = np.array([-350.0,  600.0])
        goal_y_line = 600.0
        diagonal_heading = np.pi / 4.0            # 45° diagonal (bow toward +x and +y)
        n_frames = run_cfg.n_frames_medea
        ship_trajectory = [(ship_start + (frame / n_frames) * (ship_end - ship_start), diagonal_heading) for frame in range(n_frames + 1)]
        ship_pos = ship_trajectory[0][0]
        drift_vel = 0.0
        print(f"  Ship path: ({ship_start[0]:.0f}, {ship_start[1]:.0f}) -> top edge (y={goal_y_line:.0f}), heading {np.degrees(diagonal_heading):.0f}°, {n_frames} frames", flush=True)
    else:
        rightmost_edge = max(centroids[i, 0] + radii[i] for i in range(len(centroids)))
        ship_bow_x = rightmost_edge + 5
        y_center = (centroids[:, 1].min() + centroids[:, 1].max()) / 2
        ship_pos = np.array([ship_bow_x + ship_length/2, y_center])
        ship_trajectory = None
        drift_vel = 6.0
        n_frames = run_cfg.n_frames_medea
        if getattr(args, 'frames', None) is not None:
            n_frames = args.frames
        print(f"  Ship stationary at ({ship_pos[0]:.0f}, {ship_pos[1]:.0f}), ice drifts toward ship, {n_frames} frames", flush=True)
        if args.plan:
            print("  --plan is ignored when not using --simple, --toy-sparse, or --full-map (ship stationary, ice drifts).", flush=True)

    # Target-speed mode: derive n_frames so ship travels at realistic speed (path_length / (n_frames*dt) = target_speed)
    # Skip when --frames is set (user requested exact frame count).
    use_target_speed = (
        args.plan and (synthetic_mode or args.full_map)
        and getattr(run_cfg, 'planning_use_target_speed', True)
    )
    if use_target_speed:
        target_speed = getattr(args, 'target_speed', None) or getattr(run_cfg, 'planning_target_speed', 4.0)
        dt_sim = run_cfg.dt
        # Estimate path length (straight-line; actual path will be longer due to ice avoidance)
        if goal_y_line is not None:
            goal_pt = np.array([ship_start[0], goal_y_line])
        else:
            goal_pt = np.array(ship_end[:2])
        est_len = float(np.hypot(goal_pt[0] - ship_start[0], goal_pt[1] - ship_start[1]))
        n_frames = max(10, int(np.ceil(est_len / (target_speed * dt_sim))))
        print(f"  Target speed: {target_speed:.1f} m/s → n_frames={n_frames} (est. path {est_len:.0f} m, sim time {n_frames*dt_sim:.2f} s)", flush=True)
        # Rebuild ship_trajectory with new n_frames for fixed trajectory (will be replaced by planned path)
        if synthetic_mode:
            ship_trajectory = [(ship_start + (frame / n_frames) * (ship_end - ship_start), ship_heading) for frame in range(n_frames + 1)]
        elif args.full_map:
            ship_trajectory = [(ship_start + (frame / n_frames) * (ship_end - ship_start), diagonal_heading) for frame in range(n_frames + 1)]

    planned_path = None
    planned_cost_map = None
    planned_bounds = None
    replan_callback = None
    replan_every_n_frames = None
    costmap_snapshots = None  # (frame_idx, cost_map) for costmap movie (AUTO-IceNav: costmap updated at replan)
    # Diagnostics: store timing for each phase
    diag = {"setup": 0.0, "costmap": 0.0, "astar": 0.0, "lattice_movie": 0.0, "sampling_mpc": 0.0, "simulation": 0.0, "anim_2d": 0.0, "anim_3d": 0.0, "anim_costmap": 0.0}

    t_setup_end = time.time()
    diag["setup"] = t_setup_end - t_setup_start

    # Plan-then-simulate: replace fixed trajectory with planned path (Section 5); enable receding-horizon replan (AUTO-IceNav style)
    if args.plan and (synthetic_mode or args.full_map):
        plan_cfg = replace_config(run_cfg, planning_turning_radius=50.0)
        if getattr(args, 'fast_lattice', False) and args.full_map:
            # Tuned for full-map: coarser grid (4x fewer cells), fewer headings (2x), stronger heuristic
            # Keeps similar path quality while significantly speeding up A*
            plan_cfg = replace_config(
                plan_cfg,
                planning_resolution=4.0,
                planning_num_headings=8,
                planning_heuristic_weight=8.0,
                planning_turning_radius=25.0,  # step length 25*4=100m, same as 50*2
            )
            print("  Fast lattice: resolution=4m, 8 headings, heuristic_weight=8 (optimized for full-map)", flush=True)
        if args.ridge_costmap:
            plan_cfg = replace_config(
                plan_cfg,
                planning_ridge_costmap_file=args.ridge_costmap,
                planning_ridge_cost_weight=args.ridge_weight if args.ridge_weight is not None else 2.0,
            )
        # Toy-sparse has thick (10 m) ice → very high cost; use shorter primitives so first step stays in open water
        if args.toy_sparse:
            plan_cfg = replace_config(plan_cfg, planning_turning_radius=25.0)
        # For the simple example, use a tighter goal tolerance than the default 150 m
        # so the lattice path ends reasonably close to the goal without being too strict.
        if args.simple:
            plan_cfg = replace_config(plan_cfg, planning_goal_tolerance=50.0)
        if not args.warm_start:
            plan_cfg = replace_config(
                plan_cfg,
                planning_jax_mbd_warm_start=False,
                planning_mppi_warm_start=False,
                planning_sqp_warm_start=False,
                planning_gradcem_warm_start=False,
            )
        if args.optimize_path:
            plan_cfg = replace_config(plan_cfg, planning_use_path_optimizer=True)
            print("  Path refinement: NLP optimizer enabled (path length + costmap + curvature)", flush=True)
        _wall = (x_min, x_max, y_min, y_max) if args.full_map else None
        # For synthetic (simple/toy-sparse), extend costmap domain to include start and goal
        if synthetic_mode:
            plan_x_min = min(x_min, float(ship_start[0]), float(ship_end[0]))
            plan_x_max = max(x_max, float(ship_start[0]), float(ship_end[0]))
            plan_y_min = min(y_min, float(ship_start[1]), float(ship_end[1]))
            plan_y_max = max(y_max, float(ship_start[1]), float(ship_end[1]))
        else:
            plan_x_min, plan_x_max = x_min, x_max
            plan_y_min, plan_y_max = y_min, y_max
        t0 = time.time()
        _ship_speed = target_speed if use_target_speed else getattr(plan_cfg, "planning_ship_speed", 4.0)
        cost_map, bounds = build_costmap_from_ice_state(
            polygons, centroids, thicknesses,
            plan_x_min, plan_x_max, plan_y_min, plan_y_max,
            resolution=plan_cfg.planning_resolution,
            margin=plan_cfg.planning_margin,
            ship_mass=ship_displacement,
            collision_cost=plan_cfg.planning_ice_collision_cost,
            wall_bounds=_wall,
            ship_length=ship_length,
            ship_beam=ship_beam,
            planning_clearance_margin=getattr(plan_cfg, "planning_clearance_margin", 15.0),
            ridge_costmap_file=plan_cfg.planning_ridge_costmap_file,
            ridge_cost_weight=plan_cfg.planning_ridge_cost_weight,
            ship_speed=_ship_speed,
            ke_cost_scale=getattr(plan_cfg, "planning_ke_cost_scale", 1e-6),
            obs_cost_buffer_factor=getattr(plan_cfg, "planning_obs_cost_buffer_factor", 0.1),
            ice_resistance_weight=getattr(plan_cfg, "planning_ice_resistance_weight", 1.0),
        )
        diag["costmap"] = time.time() - t0
        start_heading = (ship_heading if synthetic_mode else diagonal_heading)
        start = (float(ship_start[0]), float(ship_start[1]), float(start_heading))
        goal = (float(ship_end[0]), float(ship_end[1]))
        use_diffusion_mpc = (args.planner == 'diffusion-mpc')
        use_mppi = (args.planner == 'mppi')
        use_sqp = (args.planner == 'sqp')
        use_gradcem = (args.planner == 'gradcem')
        use_sampling_mpc = use_diffusion_mpc or use_mppi or use_sqp or use_gradcem
        skip_astar = use_sampling_mpc and not args.warm_start
        planner_type = (
            "model-based diffusion" if use_diffusion_mpc
            else ("MPPI" if use_mppi else ("SQP" if use_sqp else "GradCEM"))
        ) if use_sampling_mpc else "Lattice A*"

        if skip_astar:
            print(f"  Planner: {planner_type} (cold start, no A*)", flush=True)
            lattice_path_for_opt = np.array([
                [start[0], start[1], start[2]],
                [goal[0], goal[1], start[2]],
            ])
        else:
            planner = create_medea_lattice_planner(cost_map, bounds, start, goal, plan_cfg, goal_y_line=goal_y_line)
            if use_sampling_mpc:
                print(f"  Planner: Lattice A* (warm-start) then {planner_type} refinement", flush=True)
            else:
                print("  Planner: Lattice A*", flush=True)

        if skip_astar:
            path = lattice_path_for_opt
        elif not use_sampling_mpc and not getattr(args, 'no_lattice_search_movie', False):
            t0 = time.time()
            result = plan_path(planner, start, goal, plan_cfg, cost_map=cost_map, return_search_history=True)
            path, search_history = result
            diag["astar"] = time.time() - t0
            if path is None:
                print("  Planner did not find a path; aborting. Try without --plan for fixed trajectory.", flush=True)
                return
            # Extend path to exact goal so lattice movie and costmap show path reaching the goal (planner stops within tolerance).
            if goal_y_line is None and len(path) > 0:
                last_pt = path[-1, :2]
                dist = np.hypot(last_pt[0] - goal[0], last_pt[1] - goal[1])
                tol = getattr(plan_cfg, "planning_goal_tolerance", 150.0)
                if 1e-6 < dist <= tol:
                    path = np.vstack([path, [goal[0], goal[1], path[-1, 2]]])
            lattice_name = 'lattice_search_simple' if synthetic_mode else 'lattice_search_medea'
            lattice_out = str(out_dir / (lattice_name + video_ext))
            print("  Generating lattice search visualization...", flush=True)
            t0 = time.time()
            animate_lattice_search(
                cost_map, search_history, path, start, goal,
                output_file=lattice_out,
                frame_stride=max(1, planner.nodes_expanded // 200),
                fps=15,
                show_edge_costs=True,
            )
            diag["lattice_movie"] = time.time() - t0
            print(f"  Saved: {lattice_out}", flush=True)
            # Primitive-by-primitive movie: show one primitive per frame (current in black, previous in blue)
            primitives_out = str(out_dir / (("lattice_primitives_simple" if synthetic_mode else "lattice_primitives_medea") + video_ext))
            print("  Generating lattice primitives (stepwise) visualization...", flush=True)
            animate_lattice_primitives_stepwise(
                cost_map,
                search_history,
                output_file=primitives_out,
                fps=15,
            )
            print(f"  Saved: {primitives_out}", flush=True)
            # Understanding plots: costmap with grid + cell costs, g(n)/h(n) decomposition
            plot_costmap_with_grid(
                cost_map,
                path=path,
                start=(start[0], start[1]),
                goal=goal,
                output_file=str(out_dir / "costmap_with_grid.png"),
                cell_label_stride=4,
                max_labels=60,
            )
            print(f"  Saved: {out_dir / 'costmap_with_grid.png'}", flush=True)
            plot_lattice_search_costs(
                search_history,
                output_file=str(out_dir / "lattice_search_costs.png"),
            )
            print(f"  Saved: {out_dir / 'lattice_search_costs.png'}", flush=True)
        else:
            if not use_sampling_mpc and getattr(args, 'no_lattice_search_movie', False):
                print("  Skipping lattice search movie (--no-lattice-search-movie)", flush=True)
            t0 = time.time()
            path = plan_path(planner, start, goal, plan_cfg, cost_map=cost_map)
            diag["astar"] = time.time() - t0
            if path is None:
                print("  Planner did not find a path; aborting. Try without --plan for fixed trajectory.", flush=True)
                return

        # Clip/extend the path to end exactly at the goal (handles both point goals and goal_y_line).
        if path is not None and len(path) > 0:
            path = _clip_path_to_goal(path, goal, goal_y_line_val=goal_y_line)

        path_lattice = path.copy() if (args.plan_only and getattr(args, 'compare_planners', False)) else None

        run_diffusion_for_compare = args.plan_only and getattr(args, 'compare_planners', False)
        _diffusion_diag = None
        if use_sampling_mpc or run_diffusion_for_compare:
            t0 = time.time()
            if use_diffusion_mpc or run_diffusion_for_compare:
                print("  Running model-based diffusion optimization...", flush=True)
                _want_diag = (args.plan_only and getattr(args, 'diffusion_movie', False)
                              and (use_diffusion_mpc or run_diffusion_for_compare))
                _init_filter = (
                    _make_rejection_sample_filter(
                        cost_map, ship_length, ship_beam,
                        goal_y_line=goal_y_line, goal=goal,
                    )
                    if getattr(args, 'reject_sampling', False)
                    else None
                )
                _eb_kwargs = {}
                if getattr(args, 'eb_mbd', False):
                    # Mirror examples/toy_mbd_bimodal.py: hard EB log-barrier on
                    # goal-line reaching + linear soft OOB cost on x∈[x_min,x_max]
                    # and y≥y_min.  cost_truncate_y is set above the goal line to
                    # the costmap's top edge so cost/OOB beyond the map are zero
                    # (matches Y_MAX truncation in the toy).
                    _eb_kwargs = dict(
                        use_eb_mbd=True,
                        eb_x_min=float(cost_map.x_origin),
                        eb_x_max=float(cost_map.x_origin + cost_map.width),
                        eb_y_min=float(cost_map.y_origin),
                        eb_c_max=args.eb_c_max,
                        eb_mu=args.eb_mu,
                        eb_kappa=args.eb_kappa,
                        eb_oob_as_constraint=getattr(args, 'eb_oob_constraint', False),
                        goal_y_line=goal_y_line,
                        cost_truncate_y=float(cost_map.y_origin + cost_map.height),
                    )
                _init_viz_kwargs = dict(
                    viz_dir=getattr(args, 'mbd_viz_dir', None),
                    replan_idx=0,
                    viz_world_polygons=polygons,
                    viz_thicknesses=thicknesses,
                    denoising_movie_dir=(getattr(args, 'mbd_viz_dir', None)
                                         if getattr(args, 'mbd_denoising_movie', False) else None),
                )
                if _want_diag:
                    path, _diffusion_diag = optimize_with_diffusion(
                        path, start, goal, cost_map, plan_cfg, start_nu=(2.0, 0.0, 0.0),
                        return_diagnostics=True,
                        sample_filter_fn=_init_filter,
                        max_resample_iters=getattr(args, 'max_resample_iters', None),
                        **_init_viz_kwargs,
                        **_eb_kwargs,
                    )
                else:
                    path = optimize_with_diffusion(
                        path, start, goal, cost_map, plan_cfg, start_nu=(2.0, 0.0, 0.0),
                        sample_filter_fn=_init_filter,
                        max_resample_iters=getattr(args, 'max_resample_iters', None),
                        **_init_viz_kwargs,
                        **_eb_kwargs,
                    )
            elif use_mppi:
                print("  Running MPPI optimization...", flush=True)
                path = optimize_with_mppi(
                    path, start, goal, cost_map, plan_cfg, start_nu=(2.0, 0.0, 0.0)
                )
            elif use_sqp:
                print("  Running SQP optimization...", flush=True)
                path = optimize_with_sqp(
                    path, start, goal, cost_map, plan_cfg, start_nu=(2.0, 0.0, 0.0)
                )
            else:
                print("  Running GradCEM optimization...", flush=True)
                path = optimize_with_gradcem(
                    path, start, goal, cost_map, plan_cfg, start_nu=(2.0, 0.0, 0.0)
                )
            diag["sampling_mpc"] = time.time() - t0
            # Clip/extend the optimized path to end at the goal (MBD/etc may overshoot).
            if path is not None and len(path) > 0:
                path = _clip_path_to_goal(path, goal, goal_y_line_val=goal_y_line)
        # Speed optimization: explicit flag or config flag only (not auto-enabled with KE cost).
        # The KE cost scale and time_weight=1.0 together push v_opt to v_min for most ice
        # conditions, causing n_frames to balloon and the ship to barely move.
        use_speed_opt = (
            getattr(args, 'optimize_speed', False)
            or getattr(plan_cfg, 'planning_optimize_speed', False)
        )
        speed_profile = None
        if use_speed_opt:
            v_ref = _ship_speed
            v_min = getattr(plan_cfg, 'planning_speed_optimize_v_min', 1.0)
            v_max = getattr(plan_cfg, 'planning_speed_optimize_v_max', 8.0)
            # Auto-calibrate time_weight so that v_opt = v_ref in open water (c = base_cost).
            # From v_ref = (lam * v_ref^2 / (2 * c_open))^(1/3) → lam = 2 * c_open * v_ref.
            # planning_speed_optimize_time_weight is a unit-free scale factor (1.0 = open-water
            # speed equals target speed; >1 = faster overall; <1 = more conservative).
            c_open = getattr(cost_map, 'base_cost', 0.1)
            tw_scale = getattr(plan_cfg, 'planning_speed_optimize_time_weight', 1.0)
            tw = tw_scale * 2.0 * c_open * v_ref
            speed_profile, total_time = optimize_speed_profile(
                path, cost_map, v_ref=v_ref, v_min=v_min, v_max=v_max,
                time_weight=tw, ship_length=ship_length, ship_beam=ship_beam,
            )
            n_frames = max(10, int(np.ceil(total_time / run_cfg.dt)))
            avg_v = float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)) / total_time) if total_time > 0 else v_ref
            print(f"  Speed optimization: {len(speed_profile)} segments, total time {total_time:.1f}s, n_frames={n_frames}, avg speed {avg_v:.2f} m/s (range {speed_profile.min():.2f}-{speed_profile.max():.2f})", flush=True)
        ship_trajectory = path_to_ship_trajectory_arr(path, n_frames, speed_profile=speed_profile)
        ship_pos = ship_trajectory[0, :2]
        planned_path = path
        planned_cost_map = cost_map
        planned_bounds = bounds
        costmap_snapshots = [(0, cost_map)]  # initial costmap for movie (AUTO-IceNav style)
        path_snapshots = [(0, path.copy())]  # (frame_idx, path) so costmap movie can show path at each replan
        print(f"  Planned path: {len(path)} waypoints -> trajectory length {len(ship_trajectory)}", flush=True)

        if args.plan_only:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon
            from matplotlib.collections import PatchCollection
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            # Costmap as background
            grid = cost_map.cost_map
            extent = [cost_map.x_origin, cost_map.x_origin + cost_map.width,
                      cost_map.y_origin, cost_map.y_origin + cost_map.height]
            ax.imshow(grid, origin='lower', extent=extent, cmap='YlOrRd', alpha=0.5,
                      vmin=0, vmax=np.percentile(grid[grid > 0], 95) if np.any(grid > 0) else 1.0)
            # Ice polygons
            patches = []
            colors_th = []
            for i, poly in enumerate(polygons):
                patches.append(MplPolygon(poly, closed=True))
                colors_th.append(thicknesses[i])
            pc = PatchCollection(patches, cmap='Blues', edgecolor='#555555', linewidth=0.3, alpha=0.6)
            pc.set_array(np.array(colors_th))
            pc.set_clim(thicknesses.min(), thicknesses.max())
            ax.add_collection(pc)
            fig.colorbar(pc, ax=ax, shrink=0.5, label='Thickness [m]', location='left')
            # Planned path(s)
            compare = getattr(args, 'compare_planners', False) and path_lattice is not None
            if compare:
                ax.plot(path_lattice[:, 0], path_lattice[:, 1], 'b-', lw=2, label='Lattice A*', zorder=5, alpha=0.9)
                ax.plot(path[:, 0], path[:, 1], 'r-', lw=2.5, label='Diffusion MPC', zorder=5)
            else:
                ax.plot(path[:, 0], path[:, 1], 'r-', lw=2.5, label='Planned path', zorder=5)
            ax.plot(path[0, 0], path[0, 1], 'go', ms=12, zorder=6, label='Start')
            ax.plot(path[-1, 0], path[-1, 1], 'r*', ms=16, zorder=6, label='End')
            # Goal line
            if goal_y_line is not None:
                ax.axhline(goal_y_line, color='lime', ls='--', lw=2, label=f'Goal line y={goal_y_line:.0f}', zorder=4)
            else:
                ax.plot(goal[0], goal[1], 'mx', ms=14, mew=3, zorder=6, label='Goal')
            # Ship at start (use same tapered hull as costmap movie for consistent proportions)
            hull = create_ship_hull_at_position_heading(ship_length, ship_beam, ship_start, start_heading)
            ax.add_patch(MplPolygon(hull, closed=True, fc='orange', ec='black', lw=1.5, alpha=0.8, zorder=7))

            pad = 30.0
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_aspect('equal')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            mode = 'simple' if synthetic_mode else ('full-map' if args.full_map else 'band')
            compare = getattr(args, 'compare_planners', False) and path_lattice is not None
            if compare:
                ax.set_title(f'Planner Comparison ({mode}) — Lattice: {len(path_lattice)} wp, Diffusion: {len(path)} wp\n'
                             f'{len(polygons)} cells, domain [{x_min:.0f}, {x_max:.0f}] x [{y_min:.0f}, {y_max:.0f}] m')
            else:
                planner_name = args.planner
                ax.set_title(f'Planned Path ({planner_name}, {mode}) — {len(path)} waypoints\n'
                             f'{len(polygons)} cells, domain [{x_min:.0f}, {x_max:.0f}] x [{y_min:.0f}, {y_max:.0f}] m')
            ax.legend(loc='upper right', fontsize=9)
            plt.tight_layout()
            compare = getattr(args, 'compare_planners', False) and path_lattice is not None
            plan_only_name = 'plan_only_medea' if args.full_map else ('plan_only_simple' if args.simple else 'plan_only_toy_sparse')
            if compare:
                plan_only_name = plan_only_name + '_comparison'
            plan_only_out = str(out_dir / (plan_only_name + '.png'))
            plt.savefig(plan_only_out, dpi=150, bbox_inches='tight')
            print(f"\n  --plan-only: saved costmap + planned path to {plan_only_out}", flush=True)
            if not getattr(args, 'no_show', False):
                print("  Close window to exit.", flush=True)
                plt.show()
            else:
                plt.close()

            # ── Optional: diffusion denoising movie ───────────────────
            if _diffusion_diag is not None:
                _movie_name = plan_only_name.replace('_comparison', '') + '_diffusion_denoising.mp4'
                _movie_out = str(out_dir / _movie_name)
                _movie_out = _generate_diffusion_denoising_movie(
                    diag=_diffusion_diag,
                    cost_map=cost_map,
                    polygons=polygons,
                    thicknesses=thicknesses,
                    start=start,
                    goal=goal,
                    goal_y_line=goal_y_line,
                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                    output_path=_movie_out,
                    no_show=getattr(args, 'no_show', False),
                )
                print(f"  Diffusion movie saved to {_movie_out}", flush=True)
            return

        # Receding-horizon callback (AUTO-IceNav style): rebuild costmap from current ice state and replan
        _mbd_viz_dir = getattr(args, 'mbd_viz_dir', None)
        _mbd_goal_line = getattr(args, 'mbd_goal_line', None)
        if _mbd_goal_line is not None:
            plan_cfg = replace_config(plan_cfg, planning_jax_mbd_goal_weight=0.0)
        if getattr(args, 'eb_mbd', False):
            plan_cfg = replace_config(
                plan_cfg,
                planning_jax_mbd_goal_x_weight=args.eb_goal_x_weight,
                planning_jax_mbd_length_weight=args.eb_length_weight,
            )
        _mbd_replan_counter = [0]  # mutable counter accessible inside closure

        def _replan_callback(frame, world_polygons, centroids, thicknesses, ship_pos_f, ship_heading_f, n_remaining):
            # Rebuild costmap from current ice state (counted as additional planning time)
            t0_cost = time.time()
            if args.full_map:
                x_min_f, x_max_f = -DOMAIN_LIMIT, DOMAIN_LIMIT
                y_min_f, y_max_f = -DOMAIN_LIMIT, DOMAIN_LIMIT
            else:
                _r_max = float(radii.max())
                x_min_f = float(centroids[:, 0].min()) - _r_max
                x_max_f = float(centroids[:, 0].max()) + _r_max
                y_min_f = float(centroids[:, 1].min()) - _r_max
                y_max_f = float(centroids[:, 1].max()) + _r_max
            if getattr(args, 'polygon_replan', False) and world_polygons is not None:
                # Polygon-based costmap: same method as initial plan (matches AUTO-IceNav exactly).
                from medea_ice_ship.costmap_build import build_costmap_from_ice_state
                cost_map_f, _ = build_costmap_from_ice_state(
                    world_polygons, centroids, thicknesses,
                    x_min_f, x_max_f, y_min_f, y_max_f,
                    resolution=plan_cfg.planning_resolution,
                    margin=plan_cfg.planning_margin,
                    ship_mass=float(ship_displacement),
                    collision_cost=plan_cfg.planning_ice_collision_cost,
                    wall_bounds=(x_min_f, x_max_f, y_min_f, y_max_f) if args.full_map else None,
                    ship_length=ship_length,
                    ship_beam=ship_beam,
                    planning_clearance_margin=getattr(plan_cfg, "planning_clearance_margin", 15.0),
                    ship_speed=getattr(plan_cfg, "planning_ship_speed", 4.0),
                    ke_cost_scale=getattr(plan_cfg, "planning_ke_cost_scale", 1e-6),
                    obs_cost_buffer_factor=getattr(plan_cfg, "planning_obs_cost_buffer_factor", 0.1),
                    ice_resistance_weight=getattr(plan_cfg, "planning_ice_resistance_weight", 1.0),
                )
            else:
                # GPU-native costmap: JAX disk-rasterization directly from centroids+radii.
                # `radii` from outer scope is constant (fracture only breaks bonds, not geometry).
                from medea_ice_ship.jax_costmap import build_costmap_jax
                cost_map_f, _ = build_costmap_jax(
                    centroids, thicknesses, radii,
                    x_min_f, x_max_f, y_min_f, y_max_f,
                    resolution=plan_cfg.planning_resolution,
                    margin=plan_cfg.planning_margin,
                    collision_cost=plan_cfg.planning_ice_collision_cost,
                    obs_cost_buffer_factor=getattr(plan_cfg, "planning_obs_cost_buffer_factor", 0.1),
                    ship_length=ship_length,
                    ship_beam=ship_beam,
                    planning_clearance_margin=getattr(plan_cfg, "planning_clearance_margin", 15.0),
                    ship_speed=getattr(plan_cfg, "planning_ship_speed", 4.0),
                    ship_mass=float(ship_displacement),
                    ke_cost_scale=getattr(plan_cfg, "planning_ke_cost_scale", 1e-6),
                    ice_resistance_weight=getattr(plan_cfg, "planning_ice_resistance_weight", 0.0),
                )
            diag["costmap"] += time.time() - t0_cost
            costmap_snapshots.append((frame, cost_map_f))
            start_f = (float(ship_pos_f[0]), float(ship_pos_f[1]), float(ship_heading_f))
            bounds_f = {"x_origin": cost_map_f.x_origin, "y_origin": cost_map_f.y_origin, "width": cost_map_f.width, "height": cost_map_f.height}
            t0_plan = time.time()
            if skip_astar:
                path_f = np.array([
                    [start_f[0], start_f[1], start_f[2]],
                    [goal[0], goal[1], start_f[2]],
                ])
            else:
                try:
                    planner_f = create_medea_lattice_planner(cost_map_f, bounds_f, start_f, goal, plan_cfg, goal_y_line=goal_y_line)
                except ValueError:
                    return None
                path_f = plan_path(planner_f, start_f, goal, plan_cfg, cost_map=cost_map_f)
                if path_f is None:
                    return None
            # Count all replanning work (lattice + optional sampling MPC refinement) as planning time
            diag["astar"] += time.time() - t0_plan
            if use_sampling_mpc:
                if use_diffusion_mpc:
                    _replan_filter = (
                        _make_rejection_sample_filter(
                            cost_map_f, ship_length, ship_beam,
                            goal_y_line=_mbd_goal_line or goal_y_line, goal=goal,
                        )
                        if getattr(args, 'reject_sampling', False)
                        else None
                    )
                    _replan_eb_kwargs = {}
                    _replan_goal_line = _mbd_goal_line
                    if getattr(args, 'eb_mbd', False):
                        # Mirror examples/toy_mbd_bimodal.py: hard EB log-barrier
                        # on goal-line reaching + linear soft OOB cost.  Override
                        # any --mbd-goal-line by the scenario's goal_y_line so
                        # the EB barrier targets the full-map's goal.
                        _replan_eb_kwargs = dict(
                            use_eb_mbd=True,
                            eb_x_min=float(cost_map_f.x_origin),
                            eb_x_max=float(cost_map_f.x_origin + cost_map_f.width),
                            eb_y_min=float(cost_map_f.y_origin),
                            eb_c_max=args.eb_c_max,
                            eb_mu=args.eb_mu,
                            eb_kappa=args.eb_kappa,
                            eb_oob_as_constraint=getattr(args, 'eb_oob_constraint', False),
                            cost_truncate_y=float(cost_map_f.y_origin + cost_map_f.height),
                        )
                        if goal_y_line is not None:
                            _replan_goal_line = goal_y_line
                    path_f = optimize_with_diffusion(
                        path_f, start_f, goal, cost_map_f, plan_cfg, start_nu=(2.0, 0.0, 0.0),
                        viz_dir=_mbd_viz_dir,
                        replan_idx=_mbd_replan_counter[0],
                        viz_world_polygons=world_polygons,
                        viz_thicknesses=thicknesses,
                        goal_y_line=_replan_goal_line,
                        denoising_movie_dir=_mbd_viz_dir if getattr(args, 'mbd_denoising_movie', False) else None,
                        sample_filter_fn=_replan_filter,
                        max_resample_iters=getattr(args, 'max_resample_iters', None),
                        **_replan_eb_kwargs,
                    )
                    _mbd_replan_counter[0] += 1
                elif use_mppi:
                    path_f = optimize_with_mppi(
                        path_f, start_f, goal, cost_map_f, plan_cfg, start_nu=(2.0, 0.0, 0.0)
                    )
                elif use_sqp:
                    path_f = optimize_with_sqp(
                        path_f, start_f, goal, cost_map_f, plan_cfg, start_nu=(2.0, 0.0, 0.0)
                    )
                else:
                    path_f = optimize_with_gradcem(
                        path_f, start_f, goal, cost_map_f, plan_cfg, start_nu=(2.0, 0.0, 0.0)
                    )
            # Clip/extend path to end exactly at the goal (handles both point goals and goal_y_line).
            if path_f is not None and len(path_f) > 0:
                path_f = _clip_path_to_goal(path_f, goal, goal_y_line_val=goal_y_line)
            path_snapshots.append((frame, path_f.copy()))  # for costmap movie: show this plan from this frame on
            sp_f = None
            T_raw = None
            if use_speed_opt and path_f is not None and len(path_f) >= 2:
                sp_f, T_raw = optimize_speed_profile(
                    path_f, cost_map_f, v_ref=_ship_speed,
                    v_min=getattr(plan_cfg, 'planning_speed_optimize_v_min', 1.0),
                    v_max=getattr(plan_cfg, 'planning_speed_optimize_v_max', 8.0),
                    time_weight=getattr(plan_cfg, 'planning_speed_optimize_time_weight', 1.0) * 2.0 * getattr(cost_map_f, 'base_cost', 0.1) * _ship_speed,
                    ship_length=ship_length, ship_beam=ship_beam,
                )
            # Determine n_frames for the replanned trajectory:
            # - With speed optimization: use ceil(T_raw / dt) so each sim frame advances by exactly
            #   one dt of traversal time, matching the speed profile calibration.
            # - Without speed optimization: use n_frames - frame (natural remaining budget from the
            #   initial target-speed computation) so ship speed is consistent regardless of --frames.
            if use_speed_opt and T_raw is not None:
                n_frames_f = max(1, int(np.ceil(T_raw / run_cfg.dt)))
            else:
                n_frames_f = max(1, n_frames - frame)
            return path_to_ship_trajectory_arr(path_f, n_frames_f, speed_profile=sp_f)

        if args.no_replan:
            replan_callback = None
            replan_every_n_frames = None
            print("  Receding horizon: disabled (single initial plan)", flush=True)
        else:
            replan_callback = _replan_callback
            replan_every_n_seconds = (
                args.replan_every
                if args.replan_every is not None
                else run_cfg.planning_replan_every_n_seconds
            )
            replan_every_n_frames = max(1, round(replan_every_n_seconds / run_cfg.dt))
            print(f"  Receding horizon: replan every {replan_every_n_seconds:.1f} s ({replan_every_n_frames} frames at dt={run_cfg.dt} s)", flush=True)

    sim_cfg = get_default_simulation_config()
    if args.no_mss_dynamics:
        sim_cfg = replace_config(sim_cfg, use_mss_style_dynamics=False)
        print("  Ship dynamics: simple diagonal M/D (MSS-style disabled)", flush=True)
    if args.no_thrust_cap:
        from dataclasses import replace
        sim_cfg = replace(sim_cfg, max_ship_ice_force=None)
        print("  Thrust cap: disabled (unlimited)", flush=True)
    elif args.thrust_cap is not None:
        from dataclasses import replace
        sim_cfg = replace(sim_cfg, max_ship_ice_force=args.thrust_cap)
        print(f"  Thrust cap: {args.thrust_cap/1e6:.1f} MN", flush=True)
    if getattr(args, 'ice_brake_force', None) is not None:
        sim_cfg = replace_config(sim_cfg, ice_brake_threshold=float(args.ice_brake_force))
        print(f"  Ice brake: threshold {args.ice_brake_force/1e6:.2f} MN "
              f"(setpoint blended toward ship pos when ice surge force exceeds threshold)", flush=True)
    if args.no_dpdem_bonds:
        sim_cfg = replace_config(sim_cfg, use_dpdem_style_bonds=False)
        print("  Bond model: legacy (force vs strength)", flush=True)
    # --frames: run only the first N frames (early stop only; trajectory kept at full length so
    # replanning can write new segments beyond run_n_frames without shape errors).
    _n_frames_arg = getattr(args, 'frames', None)
    run_n_frames = min(n_frames, _n_frames_arg) if _n_frames_arg is not None else n_frames
    if _n_frames_arg is not None and run_n_frames < n_frames:
        print(f"  Running first {run_n_frames} of {n_frames} frames (--frames {_n_frames_arg}); ship speed unchanged.", flush=True)
    frame_callback = None
    if args.live:
        local_polys_live = [np.asarray(p) - polygon_centroid(p) for p in polygons]
        margin = 50.0
        extent_live = (float(x_min) - margin, float(x_max) + margin, float(y_min) - margin, float(y_max) + margin)
        _fig_live, _ax_live, frame_callback = make_live_2d_callback(
            bonds, local_polys_live, ship_length, ship_beam, thicknesses, extent_live, run_n_frames,
            dt=run_cfg.dt,
        )
        print("  Live 2D view: window will update each frame", flush=True)
    # ── Ocean current model ──
    _oc_speed = args.current_speed if args.current_speed is not None else sim_cfg.ocean_current_speed
    _oc_dir = np.radians(args.current_direction) if args.current_direction is not None else sim_cfg.ocean_current_direction
    ocean_current_model = create_ocean_current_model(
        mode=args.current,
        domain=(x_min, x_max, y_min, y_max),
        speed=_oc_speed,
        direction_rad=_oc_dir,
        field_path=args.current_field or sim_cfg.ocean_current_field_path,
        seanet_weights_path=args.seanet_weights or sim_cfg.ocean_current_seanet_weights,
        grid_resolution=sim_cfg.ocean_current_grid_resolution,
        update_every_n_steps=sim_cfg.ocean_current_update_every,
        noise_strength=sim_cfg.ocean_current_noise_strength,
        seed=args.seed,
    )
    if ocean_current_model.is_active:
        print(f"  Ocean current: mode={args.current}, speed={_oc_speed:.2f} m/s", flush=True)

    t0 = time.time()
    print(f"\nRunning fracture simulation ({run_n_frames} frames) [JAX backend]...", flush=True)
    _goal_point = None
    if goal_y_line is None and ship_trajectory is not None and len(ship_trajectory) > 0:
        last = ship_trajectory[-1]
        if isinstance(last, np.ndarray) and last.ndim == 1:
            _goal_point = tuple(last[:2])
        else:
            _goal_point = tuple(last[0])
    _goal_tol = 15.0  # metres for early-stop (goal reached) check
    _kinematic = args.kinematic_ship
    _has_replan = replan_callback is not None
    if _has_replan and not _kinematic:
        # ── GPU-accelerated chunked replan path ────────────────────────────
        # Runs physics in fixed-length jax.lax.scan segments between replan
        # events, reducing Python overhead from O(N_frames) → O(N_replans).
        from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan
        print("  [JAX-chunked] Using chunked-replan path "
              f"(chunk={replan_every_n_frames} frames)...", flush=True)
        sim_result = simulate_jax_chunked_replan(
            polygons=polygons,
            centroids=centroids.copy(),
            bonds=bonds,
            ship_pos=ship_pos,
            ship_length=ship_length,
            ship_beam=ship_beam,
            n_frames=run_n_frames,
            dt=run_cfg.dt,
            drift_vel=drift_vel,
            thicknesses=thicknesses,
            ship_props=ship_props,
            ship_trajectory=ship_trajectory,
            return_ship_diagnostics=True,
            config=sim_cfg,
            replan_every_n_frames=replan_every_n_frames,
            replan_callback=replan_callback,
            frame_callback=frame_callback,
            parent_ids=parent_floe_ids,
            ocean_current=ocean_current_model,
            goal_point=_goal_point,
            goal_y_line=goal_y_line,
            goal_tolerance=_goal_tol,
            use_auto_icenav_ship=True,
            # --polygon-replan needs world polygons; disk-based JAX path only needs centroids
            skip_polygon_extraction=not getattr(args, 'polygon_replan', False),
        )
    else:
        # ── Full JAX scan path (no replanning or kinematic ship) ─────────
        # Uses jax.lax.scan over ALL frames in a single XLA program,
        # eliminating per-frame Python overhead and GPU sync entirely.
        from medea_ice_ship.jax_simulation import simulate_full_jax_scan
        _n_ice = len(centroids)
        # Pre-compute ocean current: sample_chunk_jax(n_frames, n) gives (n_frames, n, 2).
        if ocean_current_model is not None and hasattr(ocean_current_model, "sample_chunk_jax") and ocean_current_model.is_active:
            _v_water_all = ocean_current_model.sample_chunk_jax(
                run_n_frames, _n_ice, pos_jax=None, dtype=None,
            )
            # Ship-position current: sample at ship's initial position (exact for
            # uniform current; approximate for spatially-varying fields).
            import numpy as _np
            _ship_pos_np = _np.array([[float(ship_pos[0]), float(ship_pos[1])]])
            _v_ship_1    = ocean_current_model.sample_batch(_ship_pos_np)[0]  # (2,)
            _v_water_ship_all = _np.tile(_v_ship_1, (run_n_frames, 1))        # (n_frames, 2)
        else:
            _v_water_all      = None  # simulate_full_jax_scan defaults to zeros
            _v_water_ship_all = None
        print("  [JAX-scan] Compiling full-scan program "
              "(first call ~30-60s, subsequent calls fast)...", flush=True)
        sim_result = simulate_full_jax_scan(
            polygons=polygons,
            centroids=centroids.copy(),
            bonds=bonds,
            ship_pos=ship_pos,
            ship_length=ship_length,
            ship_beam=ship_beam,
            n_frames=run_n_frames,
            dt=run_cfg.dt,
            drift_vel=drift_vel,
            thicknesses=thicknesses,
            ship_props=ship_props,
            ship_trajectory=ship_trajectory,
            config=sim_cfg,
            parent_ids=parent_floe_ids,
            v_water_all=_v_water_all,
            v_water_ship_all=_v_water_ship_all,
            use_auto_icenav_ship=True,
        )
        # Post-process: trim to goal if early-stopping was requested
        if goal_y_line is not None or _goal_point is not None:
            _sh_hist = sim_result[3]
            for _f, (_spos, _spsi) in enumerate(_sh_hist):
                _stop = False
                if goal_y_line is not None and _spos[1] >= goal_y_line - _goal_tol:
                    _stop = True
                elif _goal_point is not None:
                    if np.linalg.norm(np.array(_spos) - np.array(_goal_point)) <= _goal_tol:
                        _stop = True
                if _stop:
                    _trim = _f + 2
                    sim_result = (
                        sim_result[0][:_trim],
                        sim_result[1][:_trim],
                        sim_result[2][:_trim],
                        sim_result[3][:_f + 1],
                        sim_result[4][:_trim] if len(sim_result) > 4 else [],
                        sim_result[5] if len(sim_result) > 5 else {},
                    )
                    break
    positions_history = sim_result[0]
    rotations_history = sim_result[1]
    bonds_history = sim_result[2]
    ship_history = sim_result[3]
    submergence_history = sim_result[4] if len(sim_result) > 4 else None
    ship_diagnostics = sim_result[5] if len(sim_result) > 5 and isinstance(sim_result[5], list) else None
    damage_summary = None
    if ship_diagnostics and isinstance(ship_diagnostics[-1], dict) and "damage_summary" in ship_diagnostics[-1]:
        damage_summary = ship_diagnostics[-1]["damage_summary"]
    sim_time = time.time() - t0
    diag["simulation"] = sim_time
    run_n_frames = len(positions_history)  # actual frames (may be less if goal reached early)

    n_initial = len(bonds)
    n_intact = bonds_history[-1].sum() if n_initial > 0 else 0
    n_broken = n_initial - n_intact
    pct = 100 * n_broken / n_initial if n_initial > 0 else 0
    print(f"\n  Final: {n_broken}/{n_initial} bonds broken ({pct:.0f}%)", flush=True)
    if damage_summary is not None:
        J = damage_summary.get('impulse_total', 0.0)
        print(f"  Ship damage proxy (cumulative impulse): {J/1e6:.2f} MN·s", flush=True)
        dps = damage_summary.get("dpdem_bond_stats")
        if dps is not None:
            print(
                f"  DPDEM bond stats: damage>0.01: {dps['n_damaged']}/{dps['n_bonds']}, "
                f"mean: {dps['mean_damage']:.4f}, max: {dps['max_damage']:.4f}, failed by D>=1: {dps['n_failed_by_damage']}",
                flush=True,
            )
        segs = damage_summary.get("hull_impulse_segments")
        contacts = damage_summary.get("hull_impulse_contacts")
        if contacts or segs:
            hull_out = str(out_dir / (('hull_impact_simple' if synthetic_mode else 'hull_impact_medea') + '.png'))
            plot_hull_impact_vectors(
                ship_length, ship_beam,
                hull_impulse_segments=segs, hull_impulse_contacts=contacts,
                output_file=hull_out,
            )
            print(f"  Saved hull impact vectors: {hull_out}", flush=True)
    if ship_diagnostics:
        # Diagnostic plot: ice force vs ship speed along ship heading.
        # Filter out the trailing damage_summary-only entry and any malformed diagnostics.
        import matplotlib.pyplot as plt
        diag_frames = [
            d for d in ship_diagnostics
            if isinstance(d, dict)
            and ("t" in d)
            and ("F_ice_ship" in d)
            and ("ship_vel" in d)
        ]
        if len(diag_frames) >= 2:
            t = np.array([d["t"] for d in diag_frames])
            F_ice = np.stack([d["F_ice_ship"] for d in diag_frames])   # (N, 2)
            v_ship = np.stack([d["ship_vel"] for d in diag_frames])    # (N, 2)
            headings = np.array([h[1] for h in ship_history[:len(diag_frames)]])
            forward = np.column_stack([np.cos(headings), np.sin(headings)])
            F_ice_along = np.einsum("ij,ij->i", F_ice, forward)
            v_along = np.einsum("ij,ij->i", v_ship, forward)
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_title("Ice force vs ship speed along heading")
            ax1.plot(t, F_ice_along / 1e6, color="C0")
            ax1.set_xlabel("t [s]")
            ax1.set_ylabel("F_ice · heading [MN]", color="C0")
            ax1.tick_params(axis="y", labelcolor="C0")
            ax2 = ax1.twinx()
            ax2.plot(t, v_along, color="C1")
            ax2.set_ylabel("Ship speed · heading [m/s]", color="C1")
            ax2.tick_params(axis="y", labelcolor="C1")
            fig.tight_layout()
            out_plot = out_dir / ("ship_force_vs_speed_simple_thick_ice.png" if synthetic_mode else "ship_force_vs_speed_medea_thick_ice.png")
            plt.savefig(out_plot, dpi=150)
            plt.close(fig)
            print(f"  Saved ship force/speed plot: {out_plot}", flush=True)
    print(f"  Simulation time: {sim_time:.1f}s", flush=True)

    if args.live:
        print("\n  Live view: close the window to continue (or to exit).", flush=True)
        plt.show()

    if args.save_state:
        save_simulation_state(
            polygons, centroids, bonds, thicknesses,
            positions_history, rotations_history, bonds_history, ship_history, submergence_history,
            ship_length, ship_beam, ship_draft, run_n_frames, dt=run_cfg.dt,
            output_2d=('simple_ship_fracture' if synthetic_mode else 'medea_ship_fracture') + video_ext,
            output_3d=('simple_ship_fracture_3d' if synthetic_mode else 'medea_ship_fracture_3d') + video_ext,
            simple_mode=synthetic_mode, filepath=args.save_state,
        )

    if args.quick:
        total = time.time() - total_start
        print(f"\n{'='*60}")
        print("TIMING DIAGNOSTICS (quick mode)")
        print("=" * 60)
        for name, val in [
            ("Setup", diag["setup"]),
            ("Costmap build", diag["costmap"]),
            ("A* search", diag["astar"]),
            ("Lattice search movie", diag["lattice_movie"]),
            ("Diffusion/MPPI/etc", diag["sampling_mpc"]),
            ("Simulation", diag["simulation"]),
        ]:
            if val > 0:
                print(f"  {name}: {val:.3f}s ({100*val/total:.1f}%)")
        print(f"{'='*60}\nQuick mode complete! Total: {total:.3f}s")
        return

    out_2d = str(out_dir / (('simple_ship_fracture' if synthetic_mode else 'medea_ship_fracture') + ('_thick_ice' if args.thick_ice is not None else '') + video_ext))
    out_3d = str(out_dir / (('simple_ship_fracture_3d' if synthetic_mode else 'medea_ship_fracture_3d') + ('_thick_ice' if args.thick_ice is not None else '') + video_ext))
    t0 = time.time()
    print("\nGenerating 2D animation...", flush=True)
    if args.thick_ice is not None:
        print(f"  Thick-ice run: writing to {out_2d} (rigid-display mode)", flush=True)
    animate_ice_ship_2d(
        polygons, positions_history, rotations_history, bonds_history, bonds,
        ship_history, ship_length, ship_beam, centroids, thicknesses,
        output_file=out_2d, fps=run_cfg.fps_2d, frame_skip=run_cfg.frame_skip_2d,
        n_workers=run_cfg.n_workers,
        ocean_current=ocean_current_model,
        path_snapshots=path_snapshots if (args.plan and planned_path is not None) else None,
        goal_y_line=goal_y_line,
        max_fracture_thickness=getattr(sim_cfg, "max_fracture_thickness", 15.0),
        initial_floe_of=parent_floe_ids,
        dt=run_cfg.dt,
    )
    diag["anim_2d"] = time.time() - t0
    print(f"  2D animation time: {diag['anim_2d']:.1f}s", flush=True)
    if not args.no_3d:
        t0 = time.time()
        print("\nGenerating 3D animation...", flush=True)
        animate_ice_ship_3d(
            polygons, positions_history, rotations_history, bonds_history, bonds,
            ship_history, ship_length, ship_beam, ship_draft, centroids, thicknesses,
            output_file=out_3d, fps=run_cfg.fps_3d, frame_skip=run_cfg.frame_skip_3d,
            submergence_history=submergence_history,
            figsize=run_cfg.figsize_3d,
            n_workers=run_cfg.n_workers,
            path_snapshots=path_snapshots if (args.plan and planned_path is not None) else None,
            max_fracture_thickness=getattr(sim_cfg, "max_fracture_thickness", 15.0),
            initial_floe_of=parent_floe_ids,
            dt=run_cfg.dt,
        )
        diag["anim_3d"] = time.time() - t0
        print(f"  3D animation time: {diag['anim_3d']:.1f}s", flush=True)
    if args.plan and planned_path is not None and planned_cost_map is not None:
        out_costmap = str(out_dir / (('costmap_simple_ship_fracture' if synthetic_mode else 'costmap_medea_ship_fracture') + video_ext))
        t0 = time.time()
        print("\nGenerating costmap movie (AUTO-IceNav style)...", flush=True)
        local_polygons = [np.asarray(p) - polygon_centroid(p) for p in polygons]
        # Red = planned path (updates at replan times if path_snapshots given). Blue = actual ship path. Goal = fixed.
        animate_costmap_movie(
            planned_cost_map, planned_bounds, ship_history,
            path=planned_path,
            path_snapshots=path_snapshots if costmap_snapshots else None,
            goal=goal,
            goal_y_line=goal_y_line,
            polygons=polygons,
            thicknesses=thicknesses,
            positions_history=positions_history,
            rotations_history=rotations_history,
            local_polygons=local_polygons,
            costmap_snapshots=costmap_snapshots,
            ship_length=ship_length, ship_beam=ship_beam,
            output_file=out_costmap, fps=run_cfg.fps_2d, frame_skip=run_cfg.frame_skip_2d,
            n_workers=run_cfg.n_workers,
            dt=run_cfg.dt,
        )
        diag["anim_costmap"] = time.time() - t0
        print(f"  Costmap movie time: {diag['anim_costmap']:.1f}s", flush=True)

    # ── Timing diagnostics summary ──
    total = time.time() - total_start
    print(f"\n{'='*60}")
    print("TIMING DIAGNOSTICS")
    print("=" * 60)
    for name, val in [
        ("Setup (load/tessellate/bonds)", diag["setup"]),
        ("Costmap build", diag["costmap"]),
        ("A* search", diag["astar"]),
        ("Lattice search movie", diag["lattice_movie"]),
        ("Diffusion/MPPI/SQP/GradCEM", diag["sampling_mpc"]),
        ("Simulation", diag["simulation"]),
        ("2D animation", diag["anim_2d"]),
        ("3D animation", diag["anim_3d"]),
        ("Costmap movie", diag["anim_costmap"]),
    ]:
        if val > 0:
            pct = 100 * val / total
            print(f"  {name}: {val:.1f}s ({pct:.0f}%)")
    print(f"{'='*60}")
    print(f"Total time: {total:.1f}s")
    print(f"\nOutput directory: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
