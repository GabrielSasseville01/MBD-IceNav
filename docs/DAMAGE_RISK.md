# Ship damage measure and damage-aware cost (safe travels)

This note describes the damage proxy and how it is used in the cost function for path planning.

## 1. Damage measure in simulation (AUTO-IceNav style)

**Cumulative impulse** from ice on the ship is used as a scalar damage proxy:

- **J = Σ_t |F_ice_ship(t)| × dt** over the simulation (force magnitude × time step).
- Units: N·s (or MN·s when reported). Correlates with structural load exposure (e.g. Sun et al. polar ship risk; impulse tracking in AUTO-IceNav).

**Where it is computed**

- In `simulate_ice_ship_interaction`, `F_ice_ship` is the total force from ice on the ship each frame (collision + buoyancy reaction + rubble). After applying rubble, we accumulate `J_cumul += |F_ice_ship| * dt`.

**How to get it**

- When `ship_trajectory` is not None, the simulation returns a 6th element **damage_summary** = `{'impulse_total': J_cumul}`.
- With `return_ship_diagnostics=True`, the last entry in `ship_diagnostics` includes `'damage_summary'`, and each frame has `'impulse_cumul'`.

**Using it**

- Log or constrain total impulse per run.
- Compare scenarios (e.g. different paths or speeds) by impulse.
- Later: impulse budget in receding-horizon (replan if cumulative damage exceeds a threshold).

## 2. Damage-aware cost (planning)

To favour **safer paths** (lower expected damage), the **costmap** can include a **damage-potential** term so that the planner naturally avoids thick/heavy ice.

**Cost per cell**

- Usual: `cost = collision_cost × (1 + relative_mass × mass_cost_scale)` (mass-based).
- If **damage_cost_scale > 0** and **thicknesses** are provided when building the costmap:
  - **damage_cost = damage_cost_scale × (mass / ship_mass) × (thickness / thickness_ref)^exponent**
  - Cell cost = max(existing cost, damage_cost).
- So thicker and heavier ice gets higher cost; the A* path minimizes the integral of cost along the path, so it tends to avoid high-damage regions.

**Parameters**

- **planning_damage_cost_scale** (RunConfig): default 0 (off). Set to a positive value (e.g. 50–200) to enable damage-aware cost.
- **planning_damage_thickness_exponent**: default 0.5. Larger values penalize thick ice more.
- **thickness_ref**: 1.0 m (in CostMap / build_costmap_from_ice_state).

**Wiring**

- `build_costmap_from_ice_state` accepts `damage_cost_scale`, `damage_thickness_exponent`, `thickness_ref` and passes **thicknesses** into `CostMap.update_from_ice_field`. The run script and toy example pass these from `plan_cfg` / `run_cfg`.

## 3. References (conceptually)

- **AUTO-IceNav**: impulse vectors on the hull from collisions.
- **Sun et al. (Dalian)**: risk framework for polar ship under ice loads (fatigue/ultimate, FOSM, DEM–FEM, risk matrix). We do not run FEM; we use impulse as a proxy and thickness in the costmap to approximate “expected load” for planning.

## 4. Summary

| What | Where | Purpose |
|------|--------|--------|
| **Impulse total** | Simulation return / diagnostics | Measure of damage exposure per run; logging and future constraints. |
| **Damage potential in costmap** | CostMap + build_costmap_from_ice_state | Thickness- and mass-aware cost so the planner prefers safer paths. |

With **planning_damage_cost_scale > 0**, planned paths tend to reduce exposure to thick/heavy ice, supporting safer travels while still using a single cost function (no separate “damage constraint” in the planner yet).
