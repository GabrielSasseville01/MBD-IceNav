# DPDEM: Dilated-Polyhedron-based Discrete Element Method

Implementation of ice–ship interaction simulation based on:

> Liu et al. (2021) — *Dilated-polyhedron-based DEM analysis of the ice resistance on ship hulls in escort operations in level ice* — Marine Structures 80

> Wang et al. (2024) — *Interaction between ice cover and floating platform simulated by dilated-polyhedron-based DEM* — Ocean Engineering 301

The main entry point is **MEDEA ship–ice fracture**: satellite-derived ice floes (MEDEA Fram Strait) with DPDEM-style bond fracture and ship–ice interaction. The run script supports several toy scenarios, two planners (lattice A* and diffusion-MPC), and multiple visualizations.

---

## Quick start

```bash
# Install (uv recommended)
uv sync
# or: pip install -r requirements.txt

# Simple synthetic ice field (3 floes, straight path, fast)
uv run python run_medea_ship_fracture.py --simple

# Same with planned path through ice (lattice A*)
uv run python run_medea_ship_fracture.py --simple --plan
```

Outputs go to `outputs/default/` by default (override with `--output-dir`). Use `--quick` to run simulation only (no animations).

---

## Toy examples and commands

All commands use `run_medea_ship_fracture.py` unless noted. Use `uv run python ...` to run with the project environment.

### 1. Simple straight (no planning)

Minimal synthetic ice field (3 floes), ship on a straight path left → right. No MEDEA data required.

```bash
uv run python run_medea_ship_fracture.py --simple
```

- **Output dir:** `outputs/default/` (or `--output-dir DIR`)
- **Generated:** 2D animation, 3D animation; hull impact plot if there are collisions.

### 2. Simple straight with thick ice

Same as above but override ice thickness (e.g. 10 m) for all cells. Useful to test fracture vs unbreakable ice.

```bash
uv run python run_medea_ship_fracture.py --simple --thick-ice 10
```

Optional: `--thrust-cap 25e6` (25 MN) to cap ship thrust; `--no-thrust-cap` for unlimited.

### 3. Lattice planner (A* on motion primitives)

Plan a path through the simple ice field with **lattice A*** (costmap + Dubins-like primitives), then simulate the ship following that path with receding-horizon replanning.

```bash
uv run python run_medea_ship_fracture.py --simple --plan --planner lattice
```

- **Planner:** A* on motion primitives (AUTO-IceNav style).
- **Generated:** 2D animation, 3D animation, **lattice search movie** (A* expansion), **costmap movie** (costmap + planned path in red, actual path in blue), hull impact plot when applicable.
- Skip lattice search movie: add `--no-lattice-search-movie`.

### 4. Diffusion planner (lattice + model-based diffusion)

Same scenario but refine the lattice path with **model-based diffusion** (control optimization with 3-DOF dynamics and costmap).

```bash
uv run python run_medea_ship_fracture.py --simple --plan --planner diffusion-mpc
```

- **Planner:** Lattice A* (initial path) then model-based diffusion refinement.
- **Generated:** Same as lattice (2D, 3D, costmap movie, hull impact) but **no** lattice search movie (search is not visualized for diffusion runs).

### 5. Kinematic ship (path verification)

Ship is placed exactly on the planned path each frame (no dynamics). Use to check that the planned path is feasible and that red (plan) and blue (actual) coincide.

```bash
# Lattice plan, kinematic execution
uv run python run_medea_ship_fracture.py --simple --plan --planner lattice --kinematic-ship

# Diffusion plan, kinematic execution
uv run python run_medea_ship_fracture.py --simple --plan --planner diffusion-mpc --kinematic-ship
```

Use a dedicated output dir to compare: e.g. `--output-dir outputs/simple_lattice` or `outputs/simple_diffusion`.

### 6. Full MEDEA map (diagonal ship)

Use the full satellite ice map; ship goes diagonal bottom-left → top-right. Requires MEDEA data: `data/ice_floes_MEDEA_fram_20100629.pkl`.

```bash
uv run python run_medea_ship_fracture.py --full-map
uv run python run_medea_ship_fracture.py --full-map --plan --planner lattice
```

Output filenames use `medea_` instead of `simple_` (e.g. `medea_ship_fracture.mp4`, `costmap_medea_ship_fracture.mp4`).

### 7. Toy planner: wall with opening

Standalone example: a wall of ice with one gap; the planner finds a path through the gap. Different scenario from the simple 3-floe field.

```bash
uv run python examples/toy_planner_opening.py
```

- **Output dir:** `outputs/toy_planner/`
- **Generated:** `toy_planner_lattice_search.mp4`, `toy_planner_opening.mp4` (2D ship animation), `toy_planner_opening_costmap.mp4`, `toy_planner_hull_impact.png`.
- Options: `--live` (real-time 2D window), `--damage-cost` (damage-aware costmap), `--no-lattice-search-movie`, `--no-replan` (disable receding-horizon replanning).

### 8. Toy planner: thin-ice corridor (thickness test)

Dense ice band spanning the full width in y (ship must traverse). Most ice is 10 m thick; one horizontal strip at center y is 1 m. Tests whether the lattice planner prefers the thin-ice corridor when damage-aware cost is enabled.

```bash
uv run python examples/toy_planner_thin_corridor.py
```

- **Output dir:** `outputs/toy_planner_thin_corridor/`
- **Generated:** `toy_thin_corridor_lattice_search.mp4`, `toy_thin_corridor.mp4`, `toy_thin_corridor_3d.mp4`, `toy_thin_corridor_costmap.mp4`, `toy_thin_corridor_hull_impact.png`.
- Options: `--live`, `--no-damage-cost` (baseline: no thickness preference), `--damage-cost-scale SCALE`, `--no-lattice-search-movie`, `--no-replan` (disable receding-horizon replanning).

---

## Generated visualizations

| Output | When | Description |
|--------|------|--------------|
| **2D animation** | Always (unless `--quick`) | Top-down view: ice floes, bonds, ship hull; fracture and motion over time. `simple_ship_fracture.mp4` or `medea_ship_fracture.mp4`. |
| **3D animation** | Always unless `--no-3d` | Same run with z/tilt and submergence; 3D hull. `simple_ship_fracture_3d.mp4` or `medea_ship_fracture_3d.mp4`. |
| **Costmap movie** | When `--plan` | Costmap as background, ice outline, **red** = initial planned path, **blue** = actual ship path. `costmap_simple_ship_fracture.mp4` or `costmap_medea_ship_fracture.mp4`. |
| **Lattice search movie** | When `--plan --planner lattice` and not `--no-lattice-search-movie` | A* search expansion and chosen path on the costmap. `lattice_search_simple.mp4` or `lattice_search_medea.mp4`. |
| **Hull impact plot** | When ship has collisions (and not `--quick`) | Impulse vectors on the hull (AUTO-IceNav style). `hull_impact_simple.png` or `hull_impact_medea.png`. |

File format: **MP4** by default; use `--gif` for GIF (slower, larger, no parallel workers). All paths are under the directory given by `--output-dir` (default `outputs/default`).

---

## Common flags

| Flag | Effect |
|------|--------|
| `--simple` | Synthetic 3-floe ice field (no MEDEA data). |
| `--full-map` | Full MEDEA map; ship diagonal. |
| `--plan` | Plan path (costmap + planner); only with `--simple` or `--full-map`. |
| `--planner lattice` | Use lattice A* (default). |
| `--planner diffusion-mpc` | Use lattice A* then model-based diffusion. |
| `--kinematic-ship` | With `--plan`: ship follows path exactly (no dynamics). |
| `--thick-ice M` | Override ice thickness to M metres. |
| `--no-dpdem-bonds` | Use legacy bond model (force vs strength) instead of default DPDEM-style (hybrid criterion, damage, Mohr-Coulomb). |
| `--quick` | Simulation only, no animations. |
| `--no-3d` | Skip 3D animation. |
| `--no-lattice-search-movie` | With `--plan --planner lattice`: skip A* search movie. |
| `--output-dir DIR` | Write outputs to `DIR` (default `outputs/default`). |
| `--seed N` | Random seed for reproducibility. |
| `--gif` | Output GIF instead of MP4. |
| `--live` | Real-time 2D window during simulation. |
| `--save-state FILE` | Save simulation state to pickle for replay. |
| `--load-state FILE` | Replay: load state and generate animations only. |

Optional tuning: `--thrust-cap N`, `--no-thrust-cap`, `--no-mss-dynamics`, `--no-dpdem-bonds` (legacy bonds), `--optimize-path` (NLP path refinement; requires CasADi).

---

## Reproducibility and replay

- **Seed:** The script uses a single random seed (config default or `--seed N`) for tessellation, thickness noise, and angular noise. Same seed ⇒ same ice layout and outcome.
- **Save state:** Run with `--simple --quick --save-state scenario.pkl` to save the full state (positions, rotations, bonds, ship history, etc.) after the simulation.
- **Replay:** Run with `--load-state scenario.pkl` to skip simulation and generate only 2D and 3D animations from the saved state. Add `--no-3d` for 2D only.

---

## Planning and control

With `--plan`, the run script uses **receding-horizon planning** (AUTO-IceNav style): the costmap is rebuilt from the current ice state every N frames (config `planning_replan_every_n_frames`), and the trajectory is updated from the new plan.

- **Lattice (`--planner lattice`):** A* on motion primitives (Dubins-like), costmap, ship footprint and turning radius respected.
- **Diffusion-MPC (`--planner diffusion-mpc`):** Lattice A* for an initial path, then model-based diffusion over control sequences (3-DOF dynamics, costmap, path/waypoint rewards). See [docs/DIFFUSION_MPC.md](docs/DIFFUSION_MPC.md).

The `dpdem` library also provides planning and control modules used in examples:

- **Planning:** `dpdem.planning` — lattice planner, cost map, primitives.
- **Control:** `dpdem.control` — DP controller, trajectory setpoints.
- **Navigation:** `dpdem.navigation` — `AutonomousNavigator`.

Example scripts:

```bash
uv run python examples/visualize_costmap.py      # Costmap from simple ice
uv run python examples/visualize_planned_path.py # Plan and plot path on costmap
```

---

## Data and dependencies

- **MEDEA runs** (`--full-map` or default non-simple): require `data/ice_floes_MEDEA_fram_20100629.pkl` (see `data/` or AUTO-IceNav MEDEA dataset).
- **Simple / toy planner:** No MEDEA file needed.
- Parallel rendering uses multiple workers (config `n_workers`; default 4). GIF output does not use parallel workers.

---

## Features

- **Voronoi ice elements:** Irregular polygon shapes (paper Fig. 2); MEDEA floes tessellated for fracture.
- **Bond mechanics:** Adjacent elements bonded; fracture under stress. **Default: DPDEM-style** (hybrid criterion (⟨σ⟩/σ̄)²+(τ/τ̄)², Mohr–Coulomb shear strength, damage variable D, Benzeggagh–Kenane Gc, paper Eqs. 14–19). Use `--no-dpdem-bonds` (run script or toy examples) or `SimulationConfig(use_dpdem_style_bonds=False)` for the legacy force-vs-strength model.
- **Contact:** Hertz–Mindlin–style contact; in the MEDEA script, simplified repulsion and ship collision.
- **3D visualization:** Ice z-offset and tilt from ship-relative position; tapered 3D hull.
- **Ship dynamics:** Optional 3-DOF (Fossen-style M, D, Coriolis) with PD tracking; or kinematic mode for path verification.

---

## References

- Wang, X., Liu, L., Wang, S., & Ji, S. (2024). Interaction between ice cover and floating platform simulated by dilated-polyhedron-based DEM. *Ocean Engineering*, 301, 117413.
- MEDEA ice data: Hwang et al. (2022), multi-satellite floe size distribution (2000–2020); Fram Strait 2010-06-29.
