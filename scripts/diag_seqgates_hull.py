"""
Drill into WHERE the 14M costmap cost actually comes from.

Hypothesis: hull-corner sampling queries the inflated costmap, while inflation
already accounts for the ship beam. The corners (at length/2 from centroid)
land in the inflated buffer even when the centroid is ice-free.
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

from medea_ice_ship.config import get_default_run_config, get_default_simulation_config
from medea_ice_ship.jax_mbd_planner import (
    _make_raw_components_fn, _warm_start_from_path,
    astar_path_on_costmap, make_dynamics_rollout,
)
from toy_mbd_scenarios import SCENARIOS, Y_MAX

sc = SCENARIOS["sequential_gates"]
run_cfg = get_default_run_config()
run_cfg = replace(run_cfg, planning_jax_mbd_horizon=300,
                  planning_jax_mbd_n_substeps=1, planning_jax_mbd_dt_substep=1.0)

print(f"\nShip dims: L={run_cfg.ship_length:.0f}  B={run_cfg.ship_beam:.0f}")
print(f"  half_l={run_cfg.ship_length/2:.0f}  half_w={run_cfg.ship_beam/2:.0f}")
print(f"  longest hull point from centroid = sqrt((L/2)² + (B/2)²) = {np.hypot(run_cfg.ship_length/2, run_cfg.ship_beam/2):.1f}")

# Split-costmap setup (matches toy_mbd_scenarios.py post-fix):
#   - search map: dilated by SHIP_BEAM (28 m) for A*/RRT
#   - eval map: un-inflated for cost reporting
from scipy.ndimage import maximum_filter
from dataclasses import replace as _replace
cost_map, _ = sc.costmap_fn(run_cfg, inflate=False)
_infl_cells = max(1, int(np.ceil(run_cfg.ship_beam / cost_map.resolution)))
cost_map_search = _replace(
    cost_map,
    cost_map=maximum_filter(np.asarray(cost_map.cost_map),
                            size=2 * _infl_cells + 1).astype(np.float32),
)
print(f"\n*** A* runs on map dilated by ship_beam={run_cfg.ship_beam:.0f} m;"
      f" cost evaluated on UN-INFLATED map ***")

# Compare hull-sampling vs centroid-only sampling along the warm-start traj.
astar_path = astar_path_on_costmap(cost_map_search, start_xy=(sc.start[0], sc.start[1]),
                                    goal_xy=sc.goal, cost_threshold_frac=0.1,
                                    smooth_iters=300, seed=42)
mass = run_cfg.ship_displacement
L = run_cfg.ship_length
B = run_cfg.ship_beam
Izz = (1.0 / 12.0) * mass * (L**2 + B**2)
sim_cfg = get_default_simulation_config()
rollout_fn, M3_np, D3_np = make_dynamics_rollout(
    mass=mass, Izz=Izz, dt_substep=1.0, n_substeps=1,
    ship_damping_xy=sim_cfg.ship_damping_xy, ship_damping_psi=sim_cfg.ship_damping_psi,
    added_mass_surge_frac=sim_cfg.mss_added_mass_surge_frac,
    added_mass_sway_frac=sim_cfg.mss_added_mass_sway_frac,
    added_mass_yaw_frac=sim_cfg.mss_added_mass_yaw_frac,
)

H = 300
eta0 = jnp.array(sc.start, dtype=jnp.float32)
nu0 = jnp.array((2.0, 0.0, 0.0), dtype=jnp.float32)
surge_max = run_cfg.planning_jax_mbd_surge_max
sway_max = run_cfg.planning_jax_mbd_sway_max
yaw_max = run_cfg.planning_jax_mbd_yaw_max
action_center = jnp.array([surge_max/2, 0.0, 0.0], dtype=jnp.float32)
action_scale = jnp.array([surge_max/2, sway_max, yaw_max], dtype=jnp.float32)

coverage = H * 1.0 * 4.0
seg = np.linalg.norm(np.diff(astar_path[:,:2], axis=0), axis=1)
s = np.concatenate([[0.0], np.cumsum(seg)])
cutoff = max(min(int(np.searchsorted(s, coverage, side="right")), len(astar_path)), 2)
Ybar_warm = _warm_start_from_path(
    astar_path[:cutoff], H, init_surge_bias=0.7,
    eta0=np.array(sc.start, dtype=np.float64), nu0=np.array((2.0, 0.0, 0.0)),
    M3_np=M3_np, D3_np=D3_np, dt_substep=1.0, n_substeps=1,
    action_center_np=np.array([surge_max/2, 0.0, 0.0]),
    action_scale_np=np.array([surge_max/2, sway_max, yaw_max]),
)
u_phys = jnp.array(Ybar_warm) * action_scale + action_center
traj = np.array(rollout_fn(eta0, nu0, u_phys))   # (301, 3)
print(f"\nWarmstart traj: {len(traj)} pts, x∈[{traj[:,0].min():.0f},{traj[:,0].max():.0f}], y∈[{traj[:,1].min():.0f},{traj[:,1].max():.0f}]")

# Sample CENTROID cost
cent_costs = np.array([cost_map.get_cost_world(float(x), float(y)) for x, y in traj[:,:2]])
print(f"\nCENTROID-only query along warmstart:")
print(f"  pts with cost>0      : {int((cent_costs > 0).sum())} / {len(cent_costs)}")
print(f"  sum(centroid_costs)  = {cent_costs.sum():.1f}")

# Sample HULL cost (the 5-point hull sampling used in cost_fn)
half_l = L / 2.0
half_w = B / 2.0
dx_body = np.array([0.0, half_l, half_l, -half_l, -half_l])
dy_body = np.array([0.0, half_w, -half_w, half_w, -half_w])

hull_costs_max = np.zeros(len(traj))
n_hull_hit = np.zeros(len(traj), dtype=int)
for k, (x, y, psi) in enumerate(traj):
    c, s_psi = np.cos(psi), np.sin(psi)
    wx = x + c * dx_body - s_psi * dy_body
    wy = y + s_psi * dx_body + c * dy_body
    cs = np.array([cost_map.get_cost_world(float(_x), float(_y)) for _x, _y in zip(wx, wy)])
    hull_costs_max[k] = cs.max()
    n_hull_hit[k] = int((cs > 0).sum())

print(f"\nHULL (5-pt) query along warmstart:")
print(f"  pts with max_hull_cost>0  : {int((hull_costs_max > 0).sum())} / {len(traj)}")
print(f"  total hull samples >0     : {int(n_hull_hit.sum())} / {len(traj)*5}")
print(f"  sum(max_hull_costs)       = {hull_costs_max.sum():.1f}")

# Trapezoid integral as in _make_raw_components_fn
seg_lens = np.linalg.norm(np.diff(traj[:,:2], axis=0), axis=1)
seg_int = 0.5 * (hull_costs_max[:-1] + hull_costs_max[1:]) * seg_lens
print(f"\nTrapezoidal integral (matches _make_raw_components_fn):")
print(f"  sum(0.5·(c_t+c_{{t+1}})·seg_len) = {seg_int.sum():.1f}")

# Which y-ranges are taking the hit?
mask = hull_costs_max > 0
if mask.any():
    print(f"\n  Hull-cost hits at y in: [{traj[mask,1].min():.0f}, {traj[mask,1].max():.0f}]")
    print(f"  Hull-cost hits at x in: [{traj[mask,0].min():.0f}, {traj[mask,0].max():.0f}]")
    # First few hits
    idxs = np.where(mask)[0][:8]
    for k in idxs:
        print(f"    step {k:3d}: pos=({traj[k,0]:6.1f},{traj[k,1]:6.1f},psi={traj[k,2]:+.2f})  hull_max={hull_costs_max[k]:.1e}  hull_hits={n_hull_hit[k]}/5")

# Distance from each hit to nearest obstacle edge
print(f"\nObstacle BLOCK1: x∈[0,600], y∈[400,550]")
print(f"Obstacle BLOCK2: x∈[200,800], y∈[800,950]")
