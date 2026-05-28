"""
Diagnose eb_astar_mbd collapse on sequential_gates.

Reproduces the same costmap, A* warmstart, and cost-fn construction as the
toy_mbd_scenarios.py pipeline, but evaluates each piece in isolation to
localise where the trajectory cost actually comes from.
"""
from __future__ import annotations
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from medea_ice_ship.config import get_default_run_config
from medea_ice_ship.jax_mbd_planner import (
    _make_raw_components_fn,
    _warm_start_from_path,
    astar_path_on_costmap,
    make_cost_fn,
    make_dynamics_rollout,
    make_terminal_goal_euclidean_cost_fn,
)
from medea_ice_ship.config import get_default_simulation_config

# Match toy_mbd_scenarios.py layout
sys.path.insert(0, str(ROOT / "examples"))
from toy_mbd_scenarios import _build_sequential_gates, SCENARIOS, Y_MAX

X_MIN, X_MAX = 0.0, 800.0
Y_MIN = 0.0

sc = SCENARIOS["sequential_gates"]

# ── Match the run_cfg as used by toy_mbd_scenarios.py ───────────────────────
run_cfg = get_default_run_config()
run_cfg = replace(
    run_cfg,
    planning_jax_mbd_nsample=512,
    planning_jax_mbd_ndiffuse=50,
    planning_jax_mbd_temperature=0.01,
    planning_jax_mbd_use_best_sample=False,
    planning_jax_mbd_horizon=300,
    planning_jax_mbd_n_substeps=1,
    planning_jax_mbd_dt_substep=1.0,
    planning_jax_mbd_goal_x_weight=0.0,
    planning_jax_mbd_length_weight=0.0,   # length penalty currently disabled
)

cost_map, geo_info = sc.costmap_fn(run_cfg, inflate=True)

print("\n=== Scenario ===")
print(f"  start = {sc.start}")
print(f"  goal  = {sc.goal}")
print(f"  costmap rows×cols = {cost_map.rows}×{cost_map.cols}  (res={cost_map.resolution} m)")
print(f"  grid min/max     = {cost_map.cost_map.min():.1f} / {cost_map.cost_map.max():.1e}")
print(f"  rect obstacles   = {geo_info['rects']}")

# ── A* path on the costmap (same call as eb_astar_mbd uses) ───────────────
print("\n=== A* warm-start path ===")
astar_path = astar_path_on_costmap(
    cost_map,
    start_xy=(sc.start[0], sc.start[1]),
    goal_xy=sc.goal,
    cost_threshold_frac=0.1,
    smooth_iters=300,
    seed=42,
)
if astar_path is None:
    print("  A* found NO path — diagnostic cannot continue.")
    sys.exit(1)
print(f"  A* path waypoints  : {len(astar_path)}")
print(f"  A* x range         : [{astar_path[:,0].min():.1f}, {astar_path[:,0].max():.1f}]")
print(f"  A* y range         : [{astar_path[:,1].min():.1f}, {astar_path[:,1].max():.1f}]")

# Quick check: cost along the A* waypoints (ignore segments, just point-sample)
costs_along_path = np.array([cost_map.get_cost_world(float(x), float(y))
                              for x, y in astar_path[:, :2]])
print(f"  cost along A* pts  : min={costs_along_path.min():.1f}  max={costs_along_path.max():.1f}  mean={costs_along_path.mean():.1f}")
print(f"  pts with cost>0    : {int((costs_along_path > 0).sum())} / {len(costs_along_path)}")

# ── Build the dynamics rollout (same as jax_mbd_optimize) ─────────────────
mass = run_cfg.ship_displacement
L = run_cfg.ship_length
B = run_cfg.ship_beam
Izz = (1.0 / 12.0) * mass * (L**2 + B**2)
sim_cfg = get_default_simulation_config()
rollout_fn, M3_np, D3_np = make_dynamics_rollout(
    mass=mass, Izz=Izz,
    dt_substep=1.0, n_substeps=1,
    ship_damping_xy=sim_cfg.ship_damping_xy,
    ship_damping_psi=sim_cfg.ship_damping_psi,
    added_mass_surge_frac=sim_cfg.mss_added_mass_surge_frac,
    added_mass_sway_frac=sim_cfg.mss_added_mass_sway_frac,
    added_mass_yaw_frac=sim_cfg.mss_added_mass_yaw_frac,
)

H = run_cfg.planning_jax_mbd_horizon
eta0 = jnp.array(sc.start, dtype=jnp.float32)
nu0 = jnp.array((2.0, 0.0, 0.0), dtype=jnp.float32)
surge_max = run_cfg.planning_jax_mbd_surge_max
sway_max = run_cfg.planning_jax_mbd_sway_max
yaw_max = run_cfg.planning_jax_mbd_yaw_max
action_center = jnp.array([surge_max / 2, 0.0, 0.0], dtype=jnp.float32)
action_scale = jnp.array([surge_max / 2, sway_max, yaw_max], dtype=jnp.float32)

# ── Build the warm-start Ybar from the A* path (same code path) ────────────
dt_action = 1.0
speed_estimate = 4.0
coverage_distance = H * dt_action * speed_estimate
seg_lens = np.linalg.norm(np.diff(astar_path[:, :2], axis=0), axis=1)
s = np.zeros(len(astar_path))
s[1:] = np.cumsum(seg_lens)
cutoff_idx = int(np.searchsorted(s, coverage_distance, side="right"))
cutoff_idx = max(min(cutoff_idx, len(astar_path)), 2)
truncated = astar_path[:cutoff_idx]
print(f"  truncated to {len(truncated)} pts (coverage={coverage_distance:.0f} m, s_max={s[-1]:.0f} m)")

Ybar_warm = _warm_start_from_path(
    truncated, H, init_surge_bias=0.7,
    eta0=np.array(sc.start, dtype=np.float64),
    nu0=np.array((2.0, 0.0, 0.0), dtype=np.float64),
    M3_np=M3_np, D3_np=D3_np,
    dt_substep=1.0, n_substeps=1,
    action_center_np=np.array([surge_max / 2, 0.0, 0.0]),
    action_scale_np=np.array([surge_max / 2, sway_max, yaw_max]),
)
print(f"\n  Ybar warm-start  shape = {Ybar_warm.shape}  range = [{Ybar_warm.min():.3f}, {Ybar_warm.max():.3f}]")

# ── Roll out the warm-start to get the actual trajectory ──────────────────
u_phys = jnp.array(Ybar_warm) * action_scale + action_center
warm_traj = np.array(rollout_fn(eta0, nu0, u_phys))
print(f"  rolled-out traj  shape = {warm_traj.shape}")
print(f"    x range = [{warm_traj[:,0].min():.1f}, {warm_traj[:,0].max():.1f}]")
print(f"    y range = [{warm_traj[:,1].min():.1f}, {warm_traj[:,1].max():.1f}]")
print(f"    end pos = ({warm_traj[-1,0]:.1f}, {warm_traj[-1,1]:.1f})")
costs_along_traj = np.array([cost_map.get_cost_world(float(x), float(y))
                              for x, y in warm_traj[:, :2]])
print(f"    pts hitting ice (cost>0)  : {int((costs_along_traj > 0).sum())} / {len(costs_along_traj)}")
print(f"    pts hitting WALL (cost>1e7): {int((costs_along_traj > 1e7).sum())} / {len(costs_along_traj)}")

# ── Cost the warm-start with the exact cost_fn used by eb_astar_mbd ───────
grid = jnp.array(cost_map.cost_map, dtype=jnp.float32)
meta = {"x_origin": float(cost_map.x_origin), "y_origin": float(cost_map.y_origin),
        "resolution": float(cost_map.resolution),
        "rows": int(cost_map.rows), "cols": int(cost_map.cols)}

costmap_weight = run_cfg.planning_jax_mbd_costmap_weight
costmap_normalizer = run_cfg.planning_jax_mbd_costmap_normalizer
goal_weight = run_cfg.planning_jax_mbd_goal_weight
length_weight = run_cfg.planning_jax_mbd_length_weight
smoothness_weight = run_cfg.planning_jax_mbd_smoothness_weight

raw_fn = _make_raw_components_fn(
    grid, meta, sc.goal, rollout_fn, eta0, nu0,
    action_center, action_scale, L, B,
    cost_truncate_y=Y_MAX, oob_penalty=0.0,  # eb path uses oob_penalty=0
)
raw_components = np.array(raw_fn(jnp.array(Ybar_warm)))
print("\n=== Cost components of the WARM-START Ybar ===")
print(f"  raw [costmap, length, goal_dist, jerk] = {raw_components}")
print(f"  weighted contributions:")
print(f"    costmap = {raw_components[0] * costmap_weight / costmap_normalizer:.1f}")
print(f"    length  = {raw_components[1] * length_weight:.1f}  (weight={length_weight})")
print(f"    goal    = {raw_components[2] * goal_weight:.1f}  (weight={goal_weight})")
print(f"    jerk    = {raw_components[3] * smoothness_weight:.1f}  (weight={smoothness_weight})")

# Terminal Euclidean component
td_fn = make_terminal_goal_euclidean_cost_fn(
    rollout_fn, eta0, nu0, action_center, action_scale,
    goal_xy=(float(sc.goal[0]), float(sc.goal[1])),
    weight=1.0,
)
td_val = float(td_fn(jnp.array(Ybar_warm)))
print(f"    terminal_dist = {td_val:.1f}  (weight=1.0 default)")

# Full cost_fn (matching the eb branch construction)
cost_fn = make_cost_fn(
    grid, meta, sc.goal, rollout_fn, eta0, nu0,
    action_center, action_scale,
    costmap_weight=costmap_weight,
    costmap_normalizer=costmap_normalizer,
    goal_weight=goal_weight,
    length_weight=length_weight,
    smoothness_weight=smoothness_weight,
    ship_length=L, ship_beam=B,
    cost_truncate_y=Y_MAX,
    oob_penalty=0.0,
)
full_cost = float(cost_fn(jnp.array(Ybar_warm)) + td_fn(jnp.array(Ybar_warm)))
print(f"\n  TOTAL cost(warm-start Ybar)         = {full_cost:.1f}")

# ── Now: cost when we add noise (simulating step 1 of diffusion) ──────────
print("\n=== Cost stats of NOISE-PERTURBED warmstart at various sigmas ===")
print(f"  (mirrors the noise levels the diffusion loop sees from end → start)")
rng = jax.random.PRNGKey(42)
for sigma in [0.05, 0.1, 0.2, 0.5, 0.8, 1.0]:
    rng, sub = jax.random.split(rng)
    eps = jax.random.normal(sub, (256, H, 3), dtype=jnp.float32)
    Y0s = jnp.array(Ybar_warm)[None] + sigma * eps   # (256, H, 3)
    Y0s = jnp.clip(Y0s, -1.0, 1.0)
    batched_cost = jax.vmap(cost_fn)
    batched_td = jax.vmap(td_fn)
    costs = np.array(batched_cost(Y0s) + batched_td(Y0s))
    print(f"  sigma={sigma:4.2f}: cost  min={costs.min():10.1f}  median={float(np.median(costs)):10.1f}  mean={costs.mean():10.1f}  max={costs.max():10.1e}")
