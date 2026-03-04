# A* Lattice Planner – MEDEA Integration Checklist

Integrate the DPDEM A* lattice planner with the MEDEA ship–ice fracture simulation so the ship follows a **planned path** instead of a fixed straight or diagonal trajectory.  
**Test each step on the simple ice field** (`--simple`) first; the design must work for **any** simulation mode (simple, full-map, or default MEDEA band).  
**Ridging / SIC cost** is out of scope here and can be added later.

---

## 0. Prerequisites and coordinate frame

Ensure the planner’s world frame matches the simulation’s world frame so planned waypoints can be used directly as ship positions.

- [x] **0.1** Document the MEDEA simulation world frame: origin and axes used for `all_verts`, `centroids`, and `ship_trajectory` in `run_medea_ship_fracture.py` (e.g. “x right, y up; domain from `x_min` to `x_max`, `y_min` to `y_max`”).
- [x] **0.2** Confirm DPDEM `CostMap` in [dpdem/planning/cost_map.py](dpdem/planning/cost_map.py) uses the same convention: `world_to_grid(x, y)` and `grid_to_world(row, col)` assume map origin at `(0, 0)` and extent `[0, width]` × `[0, height]`. If the MEDEA domain is not origin-aligned (e.g. `x_min` ≠ 0), add an **origin** `(x_origin, y_origin)` to the costmap so that world coords `(x, y)` map to grid via `(x - x_origin, y - y_origin)` and the planner’s path is returned in the same world coords as the simulation.
- [x] **0.3** Add a minimal test or script that builds a tiny costmap with a known origin and extent and checks `world_to_grid` / `grid_to_world` round-trip and that a point at `(x_min, y_min)` maps to the correct grid cell.  
  **Done:** `examples/test_costmap_origin.py` – run with `uv run python examples/test_costmap_origin.py`.

**How to test:** Run the small test; inspect that costmap grid indices align with world coords for the MEDEA domain (e.g. print `world_to_grid(x_min, y_min)` and `grid_to_world(0, 0)`).

---

## 1. Build costmap from MEDEA state

Compute a costmap from the same ice state used at simulation start (polygons, centroids, thicknesses). No planner or simulation run yet.

- [x] **1.1** Add a helper (e.g. in `medea_ice_ship` or `dpdem/planning`) that takes: `polygons`, `centroids`, `thicknesses`, domain bounds `(x_min, x_max, y_min, y_max)`, and options `resolution`, `margin`, `ship_mass`, `base_cost`, `collision_cost`, `mass_cost_scale`. Compute **masses** = area(polygon) × thickness × ice_density (e.g. 900 kg/m³). Build a DPDEM `CostMap` with width/height = domain + 2×margin and origin = (x_min - margin, y_min - margin) if supported, or document that the caller must translate coordinates. Call `cost_map.update_from_ice_field(centroids, polygons, masses, angles=None)` (angles zero if not needed).
  **Done:** [medea_ice_ship/costmap_build.py](medea_ice_ship/costmap_build.py) — `build_costmap_from_ice_state()`.
- [x] **1.2** Return the costmap and the world bounds (e.g. `(x_origin, y_origin, width, height)` or `(x_min, x_max, y_min, y_max)`) so the planner and trajectory conversion use the same frame.
  **Done:** Helper returns `(cost_map, bounds)` with `bounds` containing `x_min`, `x_max`, `y_min`, `y_max`, `x_origin`, `y_origin`, `width`, `height`.
- [x] **1.3** Add a small test or script that: loads or creates a **simple** ice field (e.g. 3 floes), calls the helper to build the costmap, and checks that (1) the costmap has higher cost where ice cells are (e.g. sample a few cells at centroids vs open water), and (2) the costmap shape and resolution are as expected.
  **Done:** [examples/test_costmap_from_ice.py](examples/test_costmap_from_ice.py) — run with `uv run python examples/test_costmap_from_ice.py`.

**How to test:** Run the test script; optionally plot or print cost at ice centroids vs at a point far from ice; confirm costmap dimensions match `(width, height)` in world units divided by resolution.

---

## 2. Planning config and planner creation

Add planning parameters and a way to create the lattice planner for the MEDEA domain (no path execution yet).

- [x] **2.1** Add planning-related config (e.g. in [medea_ice_ship/config.py](medea_ice_ship/config.py) or a small `PlanningConfig`): `planning_resolution`, `planning_margin`, `turning_radius`, `num_headings`, `heuristic_weight`, `goal_tolerance`, `max_iterations`. Reuse ship length, beam, and mass from `RunConfig`.
  **Done:** Added planning fields to `RunConfig` in `medea_ice_ship/config.py` (`planning_resolution`, `planning_margin`, `planning_turning_radius`, `planning_num_headings`, `planning_heuristic_weight`, `planning_goal_tolerance`, `planning_max_iterations`).
- [x] **2.2** Add a function that, given domain bounds, ship start/goal (and optionally heading), and the costmap from §1, creates a DPDEM `LatticePlanner` (or uses `create_planner` from [dpdem/planning/lattice_planner.py](dpdem/planning/lattice_planner.py)) with map width/height, resolution, ship dimensions, and tuning params. Ensure start and goal are inside the costmap bounds (use the same origin/bounds as the costmap).
  **Done:** `medea_ice_ship/planning.py` – `create_medea_lattice_planner(cost_map, bounds, start, goal, run_cfg)` validates start/goal inside costmap bounds and returns a configured `LatticePlanner`.
- [x] **2.3** Test that the planner is created without error for the **simple** scenario: build ice → build costmap (§1) → create planner with start = left-of-domain, goal = right-of-domain (same as current simple mode). No need to call `plan()` yet.
  **Done:** `examples/test_planner_creation.py` – run with `uv run python examples/test_planner_creation.py`; confirms start/goal inside bounds, planner creation succeeds, and costmap shape matches expectations.

**How to test:** Run a script that builds simple ice, builds costmap, creates planner; assert planner is not None and cost_map has correct shape.

---

## 3. Plan from start to goal

Run the planner and obtain a path (list of waypoints). Still no simulation.

- [x] **3.1** Call `planner.plan(start=(x, y, heading), goal=(goal_x, goal_y))` with start and goal in the same world frame as the costmap. Handle the case when no path is found (e.g. return `None` or raise; document behaviour).
  **Done:** `plan_path(planner, start, goal, run_cfg)` in [medea_ice_ship/planning.py](medea_ice_ship/planning.py) calls `planner.plan()` with `max_iterations` from config. Planner returns `None` when no path; docstring states callers should check and abort or fall back.
- [x] **3.2** For the **simple** scenario, verify that a path is found: start at left, goal at right, heading 0 (or aligned with goal). Check that the returned path has shape (N, 3) with [x, y, heading], and that x increases roughly from start to goal (or waypoints lie between start and goal).
  **Done:** [examples/test_plan_path.py](examples/test_plan_path.py) — asserts path not None, shape (N, 3), path[0] near start, path[-1] near goal, path[-1,0] >= path[0,0]. Uses `planning_turning_radius=50` so a path is found.
- [x] **3.3** Optionally log or plot the path on top of the costmap (e.g. save a figure) to visually confirm it avoids high-cost regions.
  **Done:** [examples/visualize_planned_path.py](examples/visualize_planned_path.py) saves `planned_path_visualization.png` (and .pdf) with path over costmap.

**How to test:** Script: simple ice → costmap → planner → plan(start, goal) → assert path is not None, path.shape[1] == 3, path[0] near start, path[-1] near goal.

---

## 4. Convert path to ship_trajectory

Turn the planned path (N×3 waypoints) into the format expected by `simulate_ice_ship_interaction`: a list of `(position_2d, heading)` of length `n_frames + 1`.

- [x] **4.1** Implement conversion: given `path` (N×3) and `n_frames`, produce `ship_trajectory` = list of `(np.array([x, y]), heading)` with length `n_frames + 1`. First entry = start position and heading; last entry = goal position and heading. Interpolate along the path by arc length (or by index) so waypoints are evenly spaced over the frames. Take heading at each frame from the path (interpolate heading between waypoints if needed).
  **Done:** [medea_ice_ship/planning.py](medea_ice_ship/planning.py) — `path_to_ship_trajectory(path, n_frames)`. Arc-length parametrization; heading interpolated with angle wrap.
- [x] **4.2** Ensure the first and last trajectory entries match the planner start and goal (within a small tolerance) so the simulation does not jump at the ends.
  **Done:** First and last entries are set to `path[0]` and `path[-1]` exactly.
- [x] **4.3** Test in isolation: for a dummy path (e.g. 10 waypoints from (0,0) to (100,0)) and n_frames=100, check that `ship_trajectory` has length 101, first and last positions/headings match, and positions progress along the path.
  **Done:** [examples/test_path_to_trajectory.py](examples/test_path_to_trajectory.py) — run with `uv run python examples/test_path_to_trajectory.py`.

**How to test:** Unit test or small script: path → trajectory → assert len(trajectory) == n_frames + 1, trajectory[0][0] ≈ path[0,:2], trajectory[-1][0] ≈ path[-1,:2].

---

## 5. Wire plan-then-simulate in the run script

Integrate planning into the MEDEA run so that with a flag (e.g. `--plan` or `--lattice`), the ship follows the planned path.

- [x] **5.1** In [run_medea_ship_fracture.py](run_medea_ship_fracture.py), add a CLI flag (e.g. `--plan`) that enables “plan then simulate” mode. When set, after building ice and determining ship start/goal/heading (same logic as today for simple, full-map, or default), do not set `ship_trajectory` from the fixed path; instead: build costmap (§1), create planner (§2), call plan (§3), convert path to `ship_trajectory` (§4), then call `simulate_ice_ship_interaction(..., ship_trajectory=ship_trajectory)`.
  **Done:** `--plan` added; when `--plan` and (`--simple` or `--full-map`), costmap → planner → plan_path → path_to_ship_trajectory → simulate. Uses `planning_turning_radius=50` so a path is found.
- [x] **5.2** When planning fails (no path), either abort with a clear message or fall back to the original fixed trajectory; document the behaviour.
  **Done:** On no path, script prints a message and returns (abort); no fallback.
- [ ] **5.3** Test with **simple** mode: `uv run python run_medea_ship_fracture.py --simple --plan`. Confirm that (1) the run completes, (2) 2D and 3D outputs are produced, and (3) the ship path in the animation follows a non-straight path where it avoids ice (visually reasonable). Compare with `--simple` without `--plan` (straight line) to see the difference.

**How to test:**
```bash
uv run python run_medea_ship_fracture.py --simple --plan
# Inspect simple_ship_fracture.mp4 (and 3D) – path should avoid dense ice where possible.
```

---

## 6. Full-map and default MEDEA

Ensure the same flow works for non-simple scenarios (no code changes that tie logic to “simple” only).

- [ ] **6.1** Run with **full-map**: `uv run python run_medea_ship_fracture.py --full-map --plan --quick --save-state scenario_full.pkl`. Confirm planning and simulation run without error; optionally generate animations from the saved state.
- [ ] **6.2** Run with **default MEDEA** (no `--simple`, no `--full-map`): ship stationary, ice drifts. For this mode, start/goal may differ (e.g. goal downstream); ensure costmap extent and planner start/goal are set correctly for the chosen scenario. If the default mode uses a single “goal” (e.g. a line or point), document and test that `--plan` still produces a valid trajectory (or clearly document that `--plan` is only for simple/full-map if that is the design choice).

**How to test:** Run the commands above; confirm no crashes and that trajectory length and simulation frame count match.

---

## 7. Optional improvements (later)

- [x] **7.1** Ridging / SIC cost: add an optional ridge costmap or sea-ice concentration layer to the costmap (as in AUTO-IceNav); align with MEDEA domain and feed into the same `CostMap` or an extended cost function.
   **Done:** `CostMap.apply_ridge_cost()`, `build_costmap_from_ice_state(ridge_costmap_file=..., ridge_cost_weight=...)`, `--ridge-costmap` / `--ridge-weight` CLI, `planning_ridge_costmap_file` / `planning_ridge_cost_weight` in `RunConfig`.
- [x] **7.2** Receding horizon: periodically update the costmap from current ice state and replan during simulation (more invasive; requires coupling simulator with planner at multiple time steps).
  **Done:** `simulate_ice_ship_interaction` accepts `replan_every_n_frames` and `replan_callback`. When triggered, the callback receives current ice state (world polygons, centroids, thicknesses) and ship state; it rebuilds the costmap, plans from current pose to goal, and returns a new trajectory segment. Run script passes a callback when `--plan`; config `planning_replan_every_n_frames` (default 60) controls replan interval (AUTO-IceNav style).
- [ ] **7.3** AUTO-IceNav-style cost: optional kinetic-energy collision cost (ship_speed, ship_mass, mass_ice) and/or swath-based edge cost and Dubins + obstacles-only heuristic for closer parity and potentially better paths.

---

## Order of work

1. **§0** – Coordinate frame and costmap origin (so path waypoints are in simulation coords).
2. **§1** – Build costmap from MEDEA state; test on simple ice.
3. **§2** – Planning config and planner creation; test on simple.
4. **§3** – Run planner and check path; test on simple.
5. **§4** – Path → ship_trajectory; test with dummy path.
6. **§5** – Integrate into run script with `--plan`; test `--simple --plan`.
7. **§6** – Test `--full-map --plan` and default MEDEA if applicable.
8. **§7** – Optional enhancements (ridging, receding horizon, AUTO-IceNav cost/heuristic) as needed.
