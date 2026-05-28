"""
Pin down: why does the "best sample" in eb_astar_mbd report cost 13.8M
when its centroid trajectory is visibly ice-free?

Reproduces the exact toy_mbd_scenarios.py pipeline for eb_astar_mbd on
sequential_gates, then inspects every trajectory the planner reports:
  - warmstart Ybar           (what A* hands to MBD)
  - best_sample_actions      (the "green" sample in result.png)
  - second_best_sample       (used in the convexity probe)
  - Ybar_final               (the red mean trajectory)

For each, we report centroid vs hull ice contact AND the full cost
component breakdown.
"""
from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import maximum_filter

from medea_ice_ship.config import get_default_run_config, get_default_simulation_config
from medea_ice_ship.jax_mbd_planner import (
    _make_raw_components_fn, _warm_start_from_path,
    astar_path_on_costmap, make_dynamics_rollout, make_cost_fn,
    make_terminal_goal_euclidean_cost_fn, jax_mbd_optimize,
)
from toy_mbd_scenarios import SCENARIOS, Y_MAX, X_MIN, X_MAX, Y_MIN

sc = SCENARIOS["sequential_gates"]
run_cfg = get_default_run_config()
run_cfg = replace(
    run_cfg,
    planning_jax_mbd_nsample=512, planning_jax_mbd_ndiffuse=50,
    planning_jax_mbd_temperature=0.01,
    planning_jax_mbd_use_best_sample=False,
    planning_jax_mbd_horizon=500,    # MATCH toy_mbd_scenarios.py default
    planning_jax_mbd_n_substeps=1, planning_jax_mbd_dt_substep=1.0,
    planning_jax_mbd_goal_x_weight=0.0,
    planning_jax_mbd_length_weight=0.0,
)

# ── Split-costmap setup (matches toy_mbd_scenarios.py main()) ────────────────
cost_map, _ = sc.costmap_fn(run_cfg, inflate=False)
_infl_cells = max(1, int(np.ceil(run_cfg.ship_beam / cost_map.resolution)))
search_grid = maximum_filter(np.asarray(cost_map.cost_map),
                              size=2*_infl_cells + 1).astype(np.float32)
cost_map_search = replace(cost_map, cost_map=search_grid)

# ── A* path on inflated map ─────────────────────────────────────────────────
astar_path = astar_path_on_costmap(cost_map_search, start_xy=(sc.start[0], sc.start[1]),
                                    goal_xy=sc.goal, cost_threshold_frac=0.1,
                                    smooth_iters=300, seed=42)
print(f"A* path: {len(astar_path)} waypoints, y∈[{astar_path[:,1].min():.0f},{astar_path[:,1].max():.0f}]")

# ── Dynamics rollout (matches jax_mbd_optimize) ─────────────────────────────
mass = run_cfg.ship_displacement
L = run_cfg.ship_length
B = run_cfg.ship_beam
Izz = (1.0/12.0) * mass * (L**2 + B**2)
sim_cfg = get_default_simulation_config()
rollout_fn, M3_np, D3_np = make_dynamics_rollout(
    mass=mass, Izz=Izz, dt_substep=1.0, n_substeps=1,
    ship_damping_xy=sim_cfg.ship_damping_xy, ship_damping_psi=sim_cfg.ship_damping_psi,
    added_mass_surge_frac=sim_cfg.mss_added_mass_surge_frac,
    added_mass_sway_frac=sim_cfg.mss_added_mass_sway_frac,
    added_mass_yaw_frac=sim_cfg.mss_added_mass_yaw_frac,
)

H = run_cfg.planning_jax_mbd_horizon   # 500
print(f"H = {H}")
eta0 = jnp.array(sc.start, dtype=jnp.float32)
nu0 = jnp.array((2.0, 0.0, 0.0), dtype=jnp.float32)
surge_max = run_cfg.planning_jax_mbd_surge_max
sway_max  = run_cfg.planning_jax_mbd_sway_max
yaw_max   = run_cfg.planning_jax_mbd_yaw_max
action_center = jnp.array([surge_max/2, 0.0, 0.0], dtype=jnp.float32)
action_scale  = jnp.array([surge_max/2, sway_max, yaw_max], dtype=jnp.float32)

# ── Build the EXACT same cost_fn the EB branch uses ─────────────────────────
grid = jnp.array(cost_map.cost_map, dtype=jnp.float32)
meta = {"x_origin": float(cost_map.x_origin), "y_origin": float(cost_map.y_origin),
        "resolution": float(cost_map.resolution),
        "rows": int(cost_map.rows), "cols": int(cost_map.cols)}

cost_fn_base = make_cost_fn(
    grid, meta, sc.goal, rollout_fn, eta0, nu0,
    action_center, action_scale,
    costmap_weight=run_cfg.planning_jax_mbd_costmap_weight,
    costmap_normalizer=run_cfg.planning_jax_mbd_costmap_normalizer,
    goal_weight=run_cfg.planning_jax_mbd_goal_weight,
    length_weight=run_cfg.planning_jax_mbd_length_weight,
    smoothness_weight=run_cfg.planning_jax_mbd_smoothness_weight,
    ship_length=L, ship_beam=B,
    cost_truncate_y=Y_MAX,
    oob_penalty=0.0,   # eb branch uses 0
)
td_fn = make_terminal_goal_euclidean_cost_fn(
    rollout_fn, eta0, nu0, action_center, action_scale,
    goal_xy=(float(sc.goal[0]), float(sc.goal[1])), weight=1.0,
)
def full_cost(u_norm):
    return cost_fn_base(u_norm) + td_fn(u_norm)
cost_jit = jax.jit(full_cost)

raw_fn = _make_raw_components_fn(
    grid, meta, sc.goal, rollout_fn, eta0, nu0,
    action_center, action_scale, L, B,
    cost_truncate_y=Y_MAX, oob_penalty=0.0,
)
raw_jit = jax.jit(raw_fn)

# Hull-collision counter (matches _make_raw_components_fn's hull sampling)
half_l, half_w = L/2.0, B/2.0
dx_body = np.array([0.0, half_l, half_l, -half_l, -half_l])
dy_body = np.array([0.0, half_w, -half_w, half_w, -half_w])

def diag_traj(name, u_norm):
    """Print centroid / hull ice contact + cost breakdown for one action seq."""
    u_phys = jnp.array(u_norm) * action_scale + action_center
    traj = np.array(rollout_fn(eta0, nu0, u_phys))
    print(f"\n=== {name} ===")
    print(f"  traj: {len(traj)} pts, x∈[{traj[:,0].min():.0f},{traj[:,0].max():.0f}], y∈[{traj[:,1].min():.0f},{traj[:,1].max():.0f}], end=({traj[-1,0]:.0f},{traj[-1,1]:.0f})")
    # Centroid
    cent_costs = np.array([cost_map.get_cost_world(float(x), float(y))
                            for x, y in traj[:,:2]])
    print(f"  centroid:  hits={int((cent_costs>0).sum())}/{len(cent_costs)}  sum={cent_costs.sum():.1e}")
    # Hull
    hits = 0
    cells_in_ice = 0
    for x, y, psi in traj:
        c, s = np.cos(psi), np.sin(psi)
        wx = x + c*dx_body - s*dy_body
        wy = y + s*dx_body + c*dy_body
        cs = np.array([cost_map.get_cost_world(float(_x), float(_y)) for _x, _y in zip(wx, wy)])
        if cs.max() > 0:
            hits += 1
            cells_in_ice += int((cs > 0).sum())
    print(f"  hull(5-pt): hits={hits}/{len(traj)}  cells_in_ice={cells_in_ice}/{5*len(traj)}")
    # Cost components
    raw = np.array(raw_jit(jnp.array(u_norm)))   # [costmap, length, goal_dist, jerk]
    full = float(cost_jit(jnp.array(u_norm)))
    td = float(td_fn(jnp.array(u_norm)))
    print(f"  components: costmap_int={raw[0]:.1f}  length={raw[1]:.1f}  goal_dist={raw[2]:.1f}  jerk={raw[3]:.3e}")
    print(f"              terminal_dist={td:.1f}  ||  TOTAL cost_fn = {full:.1f}")
    return traj

# ── 1. Warmstart Ybar ───────────────────────────────────────────────────────
seg = np.linalg.norm(np.diff(astar_path[:,:2], axis=0), axis=1)
s = np.concatenate([[0.0], np.cumsum(seg)])
cov = H * 1.0 * 4.0
cut = max(min(int(np.searchsorted(s, cov, side="right")), len(astar_path)), 2)
Ybar_warm = _warm_start_from_path(
    astar_path[:cut], H, init_surge_bias=0.7,
    eta0=np.array(sc.start, dtype=np.float64), nu0=np.array((2.0, 0.0, 0.0)),
    M3_np=M3_np, D3_np=D3_np, dt_substep=1.0, n_substeps=1,
    action_center_np=np.array([surge_max/2, 0.0, 0.0]),
    action_scale_np=np.array([surge_max/2, sway_max, yaw_max]),
)
diag_traj("WARMSTART Ybar (cold start to MBD)", Ybar_warm)

# ── 2. Run the actual eb_astar_mbd via jax_mbd_optimize ────────────────────
print("\n\n>>> Running eb_astar_mbd through jax_mbd_optimize ...\n")
traj_final, diag = jax_mbd_optimize(
    cost_map=cost_map,
    start_pose=sc.start,
    goal=sc.goal,
    run_cfg=run_cfg,
    start_nu=(2.0, 0.0, 0.0),
    lattice_path=astar_path,
    warm_start=True,
    return_diagnostics=True,
    goal_y_line=None, goal_y_line_penalty=0.0,
    terminal_goal_distance_weight=1.0,
    oob_penalty=0.0,
    cost_truncate_y=Y_MAX,
    sample_filter_fn=None, max_resample_iters=None,
    use_eb_mbd=True,
    eb_x_min=X_MIN, eb_x_max=X_MAX, eb_y_min=Y_MIN,
    eb_c_max=1500.0, eb_mu=50000.0, eb_kappa=1.0,
    eb_oob_cost=0.0,
)

# ── 3. Inspect the trajectories MBD produced ───────────────────────────────
diag_traj("BEST_SAMPLE (highest reward; the green sample in result.png)",
          np.asarray(diag["best_sample_actions"]))
diag_traj("SECOND_BEST_SAMPLE",
          np.asarray(diag["second_best_sample_actions"]))
diag_traj("FINAL Ybar (the red mean trajectory)",
          np.asarray(diag["Ybar_final"]))
if diag.get("best_base_cost_trajectory") is not None:
    print(f"\nBEST_BASE_COST sample (best by raw cost among in-bounds): cost={diag['best_base_cost']:.1f}")
    diag_traj("BEST_BASE_COST sample actions",
              np.asarray(diag["best_base_cost_actions"]))
