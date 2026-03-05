# DPDEM: Dilated-Polyhedron-based Discrete Element Method

Implementation of ice–ship interaction simulation based on:

> Liu et al. (2021) — *Dilated-polyhedron-based DEM analysis of the ice resistance on ship hulls in escort operations in level ice* — Marine Structures 80

> Wang et al. (2024) — *Interaction between ice cover and floating platform simulated by dilated-polyhedron-based DEM* — Ocean Engineering 301

The main entry point is **MEDEA ship–ice fracture**: satellite-derived ice floes (MEDEA Fram Strait) with DPDEM-style bond fracture and ship–ice interaction. The run script supports several toy scenarios, two planners (lattice A* and diffusion-MPC), and multiple visualizations.

---

## Physics and equations of motion

The simulation implements the equations of motion from the DPDEM papers (Liu et al. 2021; Wang et al. 2024). All physics are active by default.

### Translational motion (Eq. 7)

**m · du/dt = Fc + Fb + Fh + G**

Each ice element is driven by four force categories:

- **Fc** — contact forces from ship collision (penetration-based, with thrust cap) and ice-ice interaction (Hertz-like repulsion with Coulomb friction).
- **Fb** — bond forces between bonded neighbours. Spring stiffness scales with ice thickness and Young's modulus; for DPDEM bonds the force is softened by the damage variable: `F = k · Δ · (1 − D)`.
- **Fh** — hydrodynamic forces: buoyancy (`Ff = ρ_w · g · A · z_sub`) when pushed underwater by the ship, and quadratic drag (`Fd = 0.5 · C_Fd · ρ_w · A · (v_ice − v_water) · |v_ice − v_water|`, Eq. 9). When ocean currents are enabled, drag uses the *relative* velocity between ice and local water so ice can drift with the current.
- **G** — gravity, handled implicitly (2D horizontal plane; vertical buoyancy restores ice to the surface).

Integration uses the Verlet algorithm (leapfrog/symplectic Euler form: `v += a·dt`, `x += v·dt`). Bonded components share a single centre-of-mass velocity (rigid-body component integration) for stability, while solo elements integrate independently.

### Rotational motion (Eq. 8)

**I · dω/dt = Mc + Mb + Mh**

Torques from contact forces (`r × Fc`), bond forces (`r × Fb`), buoyancy moment (`rf × Ff`, Eq. 10), and rotational drag (`Md = 0.5 · C_Md · ρ_w · r³_eff · ω · |ω| · A_sub`, Eq. 10). The simulation is 2D, so scalar angular velocity and rotation angle are used (quaternions are unnecessary without out-of-plane rotation).

### Rayleigh time step (Eq. 11)

**Δt = b · πr / (0.163ν + 0.877) · √(ρ/G)**

Computed at startup based on the smallest element radius, ice density, and shear modulus `G = E / (2(1+ν))`. Printed as an advisory reference; the rigid-body component integration does not require the Rayleigh cap for stability.

### DPDEM bond fracture (Eqs. 14–19)

Enabled by default (`use_dpdem_style_bonds = True`). Bonds accumulate damage through a hybrid failure criterion combining tensile and shear stress:

- **Hybrid criterion:** `(⟨σ_n⟩/σ̄)² + (σ_t/τ̄)² ≥ 1` with Mohr–Coulomb shear strength `τ̄ = C + σ_n · tan(φ)`.
- **Damage variable D:** Evolves from 0 → 1 based on Benzeggagh–Kenane mixed-mode fracture energy (G_c_I, G_c_II, η_BK).
- **Spatial locality:** Damage only accumulates at the crack front — bonds adjacent to the ship contact or to already-broken bonds.
- **Per-floe eligibility:** DPDEM damage is only enabled for floes where the ship's thrust-capped collision force exceeds the bond strength threshold (prevents thick ice from breaking unrealistically).

Use `--no-dpdem-bonds` to fall back to the legacy force-vs-strength model.

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

### 4b. MPPI planner (lattice + Model Predictive Path Integral)

Same pipeline as diffusion-MPC but uses **MPPI** (importance-weighted sampling with fixed-variance noise) instead of diffusion annealing. Same cost function (accumulated costmap + arc length + goal).

```bash
uv run python run_medea_ship_fracture.py --simple --plan --planner mppi
```

- **Planner:** Lattice A* (initial path) then MPPI refinement (JAX, GPU-friendly).

### 5. Kinematic ship (path verification)

Ship is placed exactly on the planned path each frame (no dynamics). Use to check that the planned path is feasible and that red (plan) and blue (actual) coincide.

```bash
# Lattice plan, kinematic execution
uv run python run_medea_ship_fracture.py --simple --plan --planner lattice --kinematic-ship

# Diffusion or MPPI plan, kinematic execution
uv run python run_medea_ship_fracture.py --simple --plan --planner diffusion-mpc --kinematic-ship
uv run python run_medea_ship_fracture.py --simple --plan --planner mppi --kinematic-ship
```

Use a dedicated output dir to compare: e.g. `--output-dir outputs/simple_lattice` or `outputs/simple_diffusion`.

### 6. Full MEDEA map (diagonal ship)

Use the full satellite ice map; ship goes diagonal bottom-left → top-right. Requires MEDEA data: `data/ice_floes_MEDEA_fram_20100629.pkl`.

```bash
uv run python run_medea_ship_fracture.py --full-map
uv run python run_medea_ship_fracture.py --full-map --plan --planner lattice
uv run python run_medea_ship_fracture.py --full-map --plan --planner diffusion-mpc
uv run python run_medea_ship_fracture.py --full-map --plan --planner mppi
```

Output filenames use `medea_` instead of `simple_` (e.g. `medea_ship_fracture.mp4`, `costmap_medea_ship_fracture.mp4`).

### 6b. Plan-only (costmap + path, no simulation)

Plan the trajectory and save the costmap with planned path to a PNG, then exit without running the simulation. Useful for debugging costmaps and planning parameters.

```bash
# Interactive: show plot, save to outputs/default/plan_only_medea.png
uv run python run_medea_ship_fracture.py --full-map --plan-only

# Headless: save only, no window (e.g. for CI or batch runs)
uv run python run_medea_ship_fracture.py --full-map --plan-only --no-show

# With ridge costmap
uv run python run_medea_ship_fracture.py --full-map --plan-only --no-show \
  --ridge-costmap data/ridge_costmap_MEDEA_fram_20100629_scale05_fixed.pkl --ridge-weight 2.0
```

- **Output:** `plan_only_medea.png` (or `plan_only_simple` / `plan_only_toy_sparse` for other modes).
- **Options:** `--no-lattice-search-movie` to skip A* search movie when running with `--plan` (plan-only implies `--plan`).

### 7. Toy planner: wall with opening

Standalone example: a wall of ice with one gap; the planner finds a path through the gap. Different scenario from the simple 3-floe field.

```bash
uv run python examples/toy_planner_opening.py
uv run python examples/toy_planner_opening.py --planner diffusion-mpc
uv run python examples/toy_planner_opening.py --planner mppi
```

- **Output dir:** `outputs/toy_planner/`
- **Generated:** `toy_planner_lattice_search.mp4`, `toy_planner_opening.mp4` (2D ship animation), `toy_planner_opening_costmap.mp4`, `toy_planner_hull_impact.png`.
- Options: `--planner diffusion-mpc` or `--planner mppi` (sampling-based refinement), `--live` (real-time 2D window), `--damage-cost` (damage-aware costmap), `--no-lattice-search-movie`, `--no-replan` (disable receding-horizon replanning).

### 8. Toy planner: thin-ice corridor (thickness test)

Dense ice band spanning the full width in y (ship must traverse). Most ice is 10 m thick; one horizontal strip at center y is 1 m. Tests whether the lattice planner prefers the thin-ice corridor when damage-aware cost is enabled.

```bash
uv run python examples/toy_planner_thin_corridor.py
uv run python examples/toy_planner_thin_corridor.py --planner diffusion-mpc
uv run python examples/toy_planner_thin_corridor.py --planner mppi
```

- **Output dir:** `outputs/toy_planner_thin_corridor/`
- **Generated:** `toy_thin_corridor_lattice_search.mp4`, `toy_thin_corridor.mp4`, `toy_thin_corridor_3d.mp4`, `toy_thin_corridor_costmap.mp4`, `toy_thin_corridor_hull_impact.png`.
- Options: `--planner diffusion-mpc` or `--planner mppi` (sampling-based refinement), `--live`, `--no-damage-cost` (baseline: no thickness preference), `--damage-cost-scale SCALE`, `--no-lattice-search-movie`, `--no-replan` (disable receding-horizon replanning).

### 9. Planner benchmark (random ice fields, all planners)

The benchmark script generates **random ice fields** at several concentrations (AUTO-IceNav style), runs every selected planner on each field, and saves planned-path and result (planned + actual path) images with a **planner name suffix**, plus a CSV of metrics (path length, costmap cost, impulse, fuel proxy).

```bash
# Full benchmark: all planners, all concentrations, plan + simulate (will take long - instead try one planner at a time)
uv run python examples/benchmark_planners.py

# Plan-only (no physics simulation) — fastest, for comparing planned paths (good for vis)
uv run python examples/benchmark_planners.py --no-sim

# Specific planners and concentrations
uv run python examples/benchmark_planners.py --planners lattice diffusion-mpc --concentrations 0.2 0.4

# Cold start for sampling planners (no A* warm-start)
uv run python examples/benchmark_planners.py --no-warm-start --planners diffusion-mpc mppi gradcem

# Full simulation with live 2D window (watch ship move)
uv run python examples/benchmark_planners.py --live --planners lattice

# Kinematic execution (ship follows plan exactly; no dynamics, faster)
uv run python examples/benchmark_planners.py --kinematic

# Fewer frames / less replanning for quicker runs
uv run python examples/benchmark_planners.py --n-frames 200 --replan-every 150
```

**Output directory:** `outputs/benchmark/` (or `--output-dir DIR`).

**Generated files (per concentration, per planner):**

| File | Description |
|------|-------------|
| `conc20_ice_field.png` | Ice field and start/goal. |
| `conc20_planned_<planner>.png` | Planned path on costmap (e.g. `conc20_planned_lattice.png`, `conc20_planned_diffusion-mpc.png`). |
| `conc20_result_<planner>.png` | Planned (dashed) + actual ship path on costmap (only when not `--no-sim`). |
| `benchmark_results.csv` | Table of metrics: concentration, planner, plan time, planned/actual length, costmap cost, impulse [MN·s], fuel proxy. |

**Benchmark arguments:**

| Flag | Default | Description |
|------|--------|-------------|
| `--no-sim` | off | Plan only; no simulation. Saves planned-path images only, no result images or metrics. |
| `--planners` | all | Planners to run: `lattice`, `diffusion-mpc`, `mppi`, `sqp`, `gradcem` (space-separated). |
| `--no-warm-start` | off | For diffusion-mpc/mppi/sqp/gradcem: do not warm-start from A* (cold start). |
| `--concentrations` | 0.2 0.3 0.4 0.5 | Ice concentrations to test. |
| `--kinematic` | off | Place ship on planned path each frame (no dynamics); faster, for path-vs-reference comparison. |
| `--live` | off | Open a real-time 2D window during each simulation; close to continue. |
| `--replan-every` | 120 | Replan every N simulation frames (0 = no replanning). |
| `--n-frames` | 300 | Number of simulation frames per run. |
| `--map-size` | 1400 | Square map side length [m]. |
| `--seed` | 42 | Random seed for ice field generation. |
| `--greedy` | off | Use greedy best-first search for lattice (faster, suboptimal). |
| `--output-dir` | outputs/benchmark | Output directory. |

**Metrics (when not `--no-sim`):** planned/actual path length [m], costmap integral (ice/thickness cost), cumulative impulse [MN·s], bonds broken, and a **fuel proxy** (path length + ice cost + impulse) for comparing planners on fuel consumption.

---

## Generated visualizations

| Output | When | Description |
|--------|------|--------------|
| **2D animation** | Always (unless `--quick`) | Top-down view: ice floes, bonds, ship hull; fracture and motion over time. `simple_ship_fracture.mp4` or `medea_ship_fracture.mp4`. |
| **3D animation** | Always unless `--no-3d` | Same run with z/tilt and submergence; 3D hull. `simple_ship_fracture_3d.mp4` or `medea_ship_fracture_3d.mp4`. |
| **Costmap movie** | When `--plan` | Costmap as background, ice outline, **red** = initial planned path, **blue** = actual ship path. `costmap_simple_ship_fracture.mp4` or `costmap_medea_ship_fracture.mp4`. |
| **Lattice search movie** | When `--plan --planner lattice` (only for pure lattice; not for diffusion-mpc/mppi) and not `--no-lattice-search-movie` | A* search expansion and chosen path on the costmap. `lattice_search_simple.mp4` or `lattice_search_medea.mp4`. |
| **Plan-only PNG** | When `--plan-only` | Costmap + planned path + ice thickness; saved to `plan_only_medea.png` (or `plan_only_simple` / `plan_only_toy_sparse`). No simulation. |
| **Hull impact plot** | When ship has collisions (and not `--quick`) | Impulse vectors on the hull (AUTO-IceNav style). `hull_impact_simple.png` or `hull_impact_medea.png`. |

File format: **MP4** by default; use `--gif` for GIF (slower, larger, no parallel workers). All paths are under the directory given by `--output-dir` (default `outputs/default`).

---

## Common flags

| Flag | Effect |
|------|--------|
| `--simple` | Synthetic 3-floe ice field (no MEDEA data). |
| `--full-map` | Full MEDEA map; ship diagonal. |
| `--plan` | Plan path (costmap + planner); only with `--simple` or `--full-map`. |
| `--plan-only` | Plan and save costmap+path to PNG, then exit (no simulation). Implies `--plan`. |
| `--no-show` | With `--plan-only`: save to file only, do not open window (headless). |
| `--planner lattice` | Use lattice A* (default). |
| `--planner diffusion-mpc` | Use lattice A* then model-based diffusion. |
| `--planner mppi` | Use lattice A* then MPPI (importance-weighted sampling). |
| `--planner sqp` | Use lattice A* then multiple-shooting SQP (jaxproxqp). |
| `--planner gradcem` | Use lattice A* then GradCEM (gradient + Cross-Entropy Method). |
| `--no-warm-start` | With diffusion-mpc/mppi/sqp/gradcem: disable A* warm-start (cold start). |
| `--kinematic-ship` | With `--plan`: ship follows path exactly (no dynamics). |
| `--thick-ice M` | Override ice thickness to M metres. |
| `--no-dpdem-bonds` | Use legacy bond model (force vs strength) instead of default DPDEM-style (hybrid criterion, damage, Mohr-Coulomb). |
| `--quick` | Simulation only, no animations. |
| `--no-3d` | Skip 3D animation. |
| `--no-lattice-search-movie` | With `--plan --planner lattice`: skip A* search movie. |
| `--output-dir DIR` | Write outputs to `DIR` (default `outputs/default`). |
| `--current none\|uniform\|learned` | Ocean current mode (default `none`). See [Ocean currents](#ocean-currents). |
| `--current-speed S` | Current speed [m/s] for uniform mode (default from config). |
| `--current-direction D` | Current direction [degrees, 0=+x] for uniform mode. |
| `--seanet-weights PATH` | Path to trained `seanet.pth` for learned mode. |
| `--current-field PATH` | Initial current field (e.g. `ecco4_x0.pth` or `seanet_x0.npy`) for learned mode. |
| `--current-only` | With `--simple`: ship stays put; ice drifts under current only (for testing). |
| `--seed N` | Random seed for reproducibility. |
| `--gif` | Output GIF instead of MP4. |
| `--live` | Real-time 2D window during simulation. |
| `--save-state FILE` | Save simulation state to pickle for replay. |
| `--load-state FILE` | Replay: load state and generate animations only. |

Optional tuning: `--thrust-cap N`, `--no-thrust-cap`, `--no-mss-dynamics`, `--no-dpdem-bonds` (legacy bonds), `--optimize-path` (NLP path refinement; requires CasADi).

---

## Ocean currents

Three modes control the water velocity field used in drag (Eq. 9) and ship damping:

| Mode | Description |
|------|-------------|
| **none** (default) | Zero current; drag uses ice/ship velocity as before. |
| **uniform** | Constant speed and direction everywhere. Ice and ship experience relative-velocity drag; ice can drift with the current. |
| **learned** | Spatially varying field evolved by **SeaCurrentRNN** (physics-informed shallow-water RNN). Requires PyTorch and trained weights. |

**CLI (run_medea_ship_fracture.py):** `--current uniform`, `--current-speed 0.4`, `--current-direction 90` (degrees; 90 = +y). For learned: `--current learned`, `--seanet-weights data/seanet.pth`, `--current-field data/ecco4_x0.pth` (or `data/seanet_x0.npy` from synthetic training).

**Learned model setup (no external data):**

```bash
# Train on synthetic divergence-free fields (no download)
uv run python scripts/train_seanet.py --synthetic

# Run simulation with learned currents
uv run python run_medea_ship_fracture.py --simple --current learned \
    --seanet-weights data/seanet.pth --current-field data/seanet_x0.npy
```

**Learned model with ECCO4 data:** Run `scripts/preprocess_ecco4.py` (requires NASA ECCO WebDAV credentials and `xarray`, `netcdf4`), then `scripts/train_seanet.py --data data/ECCO4.pkl`. Saves `data/seanet.pth` and `data/ecco4_x0.pth`.

**Toy examples** (`toy_planner_opening.py`, `toy_planner_thin_corridor.py`) support `--current uniform`, `--current-speed`, `--current-direction`; default is no current.

**Visualization:** When a current is active, the 2D animation draws a quiver plot of the current field behind the ice and ship.

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
- **MPPI (`--planner mppi`):** Lattice A* for initial path, then Model Predictive Path Integral (importance-weighted sampling) with the same cost function.
- **SQP (`--planner sqp`):** Lattice A* for initial path, then multiple-shooting Sequential Quadratic Programming (jaxproxqp) with 3-DOF dynamics and costmap.
- **GradCEM (`--planner gradcem`):** Lattice A* for initial path, then GradCEM (gradient + Cross-Entropy Method) combining RMSprop with CEM elite selection and resampling.

The `dpdem` library also provides planning and control modules used in examples:

- **Planning:** `dpdem.planning` — lattice planner, cost map, primitives.
- **Control:** `dpdem.control` — DP controller, trajectory setpoints.
- **Navigation:** `dpdem.navigation` — `AutonomousNavigator`.

Example scripts:

```bash
uv run python examples/visualize_costmap.py      # Costmap from simple ice
uv run python examples/visualize_planned_path.py # Plan and plot path on costmap
uv run python examples/benchmark_planners.py     # Benchmark all planners on random ice (see §9)
uv run python examples/visualize_diffusion.py   # MBD diffusion iteration evolution
```

---

## Data and dependencies

### MEDEA data generation pipeline

Full MEDEA runs (`--full-map`) and ridge-aware planning require several generated files. Follow this pipeline:

**1. Segmented PNG** (required for ridge costmap; also used to extract polygons):

```bash
curl -L -o data/MEDEA_fram_20100629.png \
  "https://ramadda.data.bas.ac.uk/repository/entry/get/MEDEA_fram_20100629.png?entryid=synth%3A7dc6e19d-79fa-41f1-99a7-da408592382f%3AL0ZyYWN0dXJpbmcvZnNkL01FREVBX2ZyYW1fMjAxMDA2MjkucG5n"
```

**2. Ice floe polygons (pkl)** — required for simulation:

- From [AUTO-IceNav](https://github.com/GabrielSasseville01/AUTO-IceNav) MEDEA dataset, or
- Generated from the PNG: `dpdem.geometry.image_floes.process_image_to_polygons("data/MEDEA_fram_20100629.png", meters_per_pixel=2.5)` (save to `data/ice_floes_MEDEA_fram_20100629.pkl`).

**3. Ridge costmap (pkl)** — optional, for ridge-aware planning:

```bash
# Download satellite TIF (optional; improves ridge detection)
uv add gdown
uv run gdown 1jV6xLCtFoe4qCWu_pOy-OnXI583lJ3Q3 --output data/
# Rename to MEDEA_fram_20100629_satellite.tif if needed

# Generate ridge costmap + visualization PNG
uv run python scripts/generate_ridge_costmap.py \
  --input data/MEDEA_fram_20100629.png \
  --satellite data/MEDEA_fram_20100629_satellite.tif \
  --output data/ridge_costmap_MEDEA_fram_20100629_scale05_fixed.pkl \
  --crop_size 600 --costmap_scale 0.5 --method combined --window_size 25 --visualize
```

- **Output:** `ridge_costmap_MEDEA_fram_20100629_scale05_fixed.pkl` (planning) and `*_visualization.png` (with `--visualize`).
- See [docs/RIDGE_COSTMAP.md](docs/RIDGE_COSTMAP.md) for details and `--synthetic` for testing without real data.

**4. Use in planning:**

```bash
uv run python run_medea_ship_fracture.py --full-map --plan \
  --ridge-costmap data/ridge_costmap_MEDEA_fram_20100629_scale05_fixed.pkl --ridge-weight 2.0
```

- **Simple / toy planner:** No MEDEA file needed.
- Parallel rendering uses multiple workers (config `n_workers`; default 4). GIF output does not use parallel workers.

---

## Features

- **Paper-accurate physics:** Full equations of motion (Eqs. 7–11) from the DPDEM papers — translational and rotational dynamics, quadratic hydrodynamic drag, buoyancy moments, rotational drag, Rayleigh time step. Active by default; no flags needed.
- **DPDEM bond fracture (default):** Hybrid criterion (⟨σ⟩/σ̄)²+(τ/τ̄)², Mohr–Coulomb shear strength, damage variable D, Benzeggagh–Kenane mixed-mode fracture energy (Eqs. 14–19). Spatially-local crack propagation. Use `--no-dpdem-bonds` for the legacy force-vs-strength model.
- **Voronoi ice elements:** Irregular polygon shapes (paper Fig. 2); MEDEA floes tessellated for fracture.
- **Contact:** Hertz-like normal repulsion with Coulomb friction for ice-ice contact; penetration-based ship collision with thrust cap.
- **3D visualization:** Ice z-offset and tilt from ship-relative position; tapered 3D hull.
- **Ship dynamics:** Optional 3-DOF (Fossen-style M, D, Coriolis) with PD tracking; or kinematic mode for path verification.

---

## Test suite

Run all tests from the project root. Each generates 2D/3D animations and diagnostic output confirming the DPDEM physics are active.

```bash
# 1. Simple physics (DPDEM fracture, 3 floes, dynamic ship)
uv run python run_medea_ship_fracture.py --simple --output-dir outputs/simple_physics

# 2. Thick ice (10 m — ship cannot break, 0 bonds broken)
uv run python run_medea_ship_fracture.py --simple --thick-ice 10 --output-dir outputs/simple_physics_thick

# 3. Kinematic ship (straight path, no dynamics)
uv run python run_medea_ship_fracture.py --simple --kinematic-ship --output-dir outputs/simple_straight

# 4. Lattice planner (A* plans path around ice)
uv run python run_medea_ship_fracture.py --simple --plan --output-dir outputs/simple_lattice

# 5. Toy planner: wall with opening
uv run python examples/toy_planner_opening.py --output-dir outputs/toy_planner

# 6. Toy thin corridor (damage-aware planner routes through thin ice)
uv run python examples/toy_planner_thin_corridor.py --output-dir outputs/toy_thin
```

Expected results:

| Test | Bonds broken | Key behaviour |
|------|-------------|---------------|
| Simple physics | ~84% (49/58) | Progressive DPDEM fracture on ship contact |
| Thick ice (10 m) | 0% (0/58) | Ship cannot break thick ice (force gating) |
| Kinematic ship | ~91% (53/58) | More fracture (ship plows through at constant speed) |
| Lattice planner | 0% (0/58) | Planner avoids all ice (no collisions) |
| Toy opening | ~10% (49/512) | Ship navigates through gap, minor edge contact |
| Toy thin corridor | ~20% (233/1164) | Planner routes through thin ice; thick ice unbroken |

All tests print `DPDEM-style bond fracture enabled (hybrid criterion, damage, Mohr-Coulomb)` confirming the paper physics are active.

---

## References

- Wang, X., Liu, L., Wang, S., & Ji, S. (2024). Interaction between ice cover and floating platform simulated by dilated-polyhedron-based DEM. *Ocean Engineering*, 301, 117413.
- MEDEA ice data: Hwang et al. (2022), multi-satellite floe size distribution (2000–2020); Fram Strait 2010-06-29.
