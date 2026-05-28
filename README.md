# DPDEM: Dilated-Polyhedron-based Discrete Element Method

Ice–ship interaction simulator + ice-navigation planner benchmark. Physics from:

> Liu et al. (2021) — *Dilated-polyhedron-based DEM analysis of the ice resistance on ship hulls in escort operations in level ice* — Marine Structures 80

> Wang et al. (2024) — *Interaction between ice cover and floating platform simulated by dilated-polyhedron-based DEM* — Ocean Engineering 301

The main entry point is **MEDEA ship–ice fracture**: satellite-derived ice floes (MEDEA Fram Strait) with DPDEM-style bond fracture and ship–ice interaction. The physics are fully **JAX-accelerated** (JIT-compiled, GPU/TPU-ready via `jax.lax.scan`). The run script supports several toy scenarios, five planners (lattice A*, diffusion-MPC, MPPI, SQP, GradCEM), and multiple visualizations.

Example outputs from a single planned run on the full MEDEA map (lattice planner, `outputs/my_run/`):

| 2D fracture | 3D fracture | Costmap + planned/actual path |
|:-:|:-:|:-:|
| <video src="outputs/my_run/medea_ship_fracture.mp4" controls width="280"></video> | <video src="outputs/my_run/medea_ship_fracture_3d.mp4" controls width="280"></video> | <video src="outputs/my_run/costmap_medea_ship_fracture.mp4" controls width="280"></video> |
| [`medea_ship_fracture.mp4`](outputs/my_run/medea_ship_fracture.mp4) | [`medea_ship_fracture_3d.mp4`](outputs/my_run/medea_ship_fracture_3d.mp4) | [`costmap_medea_ship_fracture.mp4`](outputs/my_run/costmap_medea_ship_fracture.mp4) |

> The HTML5 `<video>` tags above render in most local markdown viewers (VS Code preview, MkDocs, etc.) but **not in GitHub's rendered README** — on GitHub, use the file links below each cell instead.

---

## TL;DR — first five minutes

```bash
# 1. Install (uv is the supported toolchain).
uv sync

# 2. Smallest planned run: 3 synthetic floes, lattice A* through the field, ~1 min.
#    --polygon-replan is required for the lattice planner during receding-horizon replans
#    (otherwise the JAX costmap is missing a method the legacy lattice planner needs).
uv run python run_medea_ship_fracture.py --simple --plan --planner lattice \
    --polygon-replan --output-dir outputs/hello

# 3. Full MEDEA Fram-Strait map with the default planner (diffusion-MPC).
uv run python run_medea_ship_fracture.py --full-map --plan \
    --output-dir outputs/medea_diffusion

# 4. Same map, lattice A*. --fast-lattice tunes A* for the 1.5 km map; --polygon-replan is
#    again needed for replans.
uv run python run_medea_ship_fracture.py --full-map --plan --planner lattice \
    --fast-lattice --polygon-replan --output-dir outputs/medea_lattice
```

After (3) or (4), look in the output directory for `medea_ship_fracture.mp4`, `medea_ship_fracture_3d.mp4`, `costmap_medea_ship_fracture.mp4`, and (for lattice runs) `lattice_search_medea.mp4` + `hull_impact_medea.png`. Steps (3) and (4) need `data/ice_floes_MEDEA_fram_20100629.pkl` — see [Data and dependencies](#data-and-dependencies).

> **Known sharp edges in the current code, worth knowing before you experiment.**
> - **Lattice + replanning requires `--polygon-replan`.** During receding-horizon replans the costmap is rebuilt via `build_costmap_jax` (returns `JaxCostMapResult`, no `world_to_grid`); the legacy lattice planner needs `world_to_grid` and crashes. `--polygon-replan` switches replans to `build_costmap_from_ice_state` (legacy `CostMap`) and the lattice path works. Sampling planners (diffusion-MPC, MPPI, SQP, GradCEM) consume `JaxCostMapResult` directly and don't need this flag. Alternative: `--no-replan` (a single up-front plan, no replans).
> - **`--simple` without `--plan` looks stationary.** The bare `--simple` mode hard-codes `n_frames_simple = 300` (15 s) and the AUTO-IceNav 3-DOF ship can't traverse the ~260 m reference path in that time — `--plan` recomputes `n_frames` from `planning_target_speed`. `--kinematic-ship` is a silent no-op on the current JAX backend, so it does **not** rescue this case.
> - **Diffusion-MPC on `--simple`** uses the full-map-sized horizon (~2.8 km) and over-shoots the 150 m field. Use `--planner lattice` for `--simple --plan`.
> - **`outputs/my_run/*.mp4`** at the top of this README were generated 2026-03-05, before the recent JAX refactor. They're representative of the kind of output to expect but won't reproduce bit-for-bit against current `main`.

---

## Repository tour

```
DPDEM/
├── run_medea_ship_fracture.py   ← main entry point (simulate + plan + animate)
├── run_experiment_trial.py      ← single (scenario, planner, seed) trial for the benchmark
├── run_smoke_test.py            ← runs one planner against every scenario locally
├── submit_experiments.py        ← submit the full benchmark to SLURM
├── submit_tuning.py             ← hyper-parameter tuning over SLURM job arrays
├── medea_ice_ship/              ← simulation + planners library (JAX backend)
│   ├── jax_simulation.py        ← JIT'd physics (chunked-replan and full-scan)
│   ├── jax_costmap.py           ← GPU-native costmap rasterisation
│   ├── jax_mbd_planner.py       ← model-based diffusion MPC + EB-MBD
│   ├── jax_mppi_planner.py / jax_sqp_planner.py / jax_gradcem_planner.py
│   ├── planning.py              ← lattice A* (motion primitives, Dubins-like)
│   ├── ice_field_generator.py   ← random convex-polygon ice fields (AUTO-IceNav style)
│   ├── synthetic_map.py         ← fixed 300×750 m synthetic scenarios used by the benchmark
│   ├── ocean_current.py / seanet.py     ← uniform & learned currents
│   ├── config.py                ← all simulation + run parameters (dataclasses)
│   └── …                        ← bonds, geometry, vessel dynamics, visualization
├── dpdem/                       ← lower-level DPDEM modules (planning, control, navigation, …)
├── examples/                    ← self-contained toy demos (see "Toy problems" below)
├── scripts/                     ← data preprocessing + SeaNet training
├── slurm_scripts/               ← pre-generated SLURM sbatch files (Block A grid)
├── data/                        ← MEDEA PNG, polygons pkl, ridge costmap, SeaNet weights
├── docs/                        ← deeper notes: DIFFUSION_MPC, RIDGE_COSTMAP, EXPERIMENTS, …
└── outputs/                     ← all generated artefacts land here (one sub-dir per run)
```

### Workflow at a glance

```
                                ┌──────────────────────────────┐
                                │ Ice field source             │
                                │  • --simple    (3 floes)     │
                                │  • --full-map  (MEDEA pkl)   │
                                │  • --toy-sparse / examples/* │
                                │  • synthetic_map (benchmark) │
                                └──────────────┬───────────────┘
                                               │ polygons, centroids, thicknesses
                                               ▼
┌──────────────────┐   costmap   ┌──────────────────────────────┐
│ Planner          │ ◄────────── │ jax_costmap.build_costmap_jax │
│  lattice A*      │             │  (KE/disk rasterisation, GPU) │
│  diffusion-MPC   │             └──────────────────────────────┘
│  MPPI            │
│  SQP             │   plan      ┌──────────────────────────────┐
│  GradCEM         │ ──────────► │ Ship reference trajectory     │
└──────────────────┘             └──────────────┬───────────────┘
                                                ▼
                                ┌──────────────────────────────┐
                                │ jax_simulation               │
                                │  • DPDEM physics (Eqs. 7–11) │
                                │  • bond fracture (Eqs. 14–19)│
                                │  • 3-DOF ship + PD tracking  │
                                │  • receding-horizon replan   │
                                └──────────────┬───────────────┘
                                               ▼
                                ┌──────────────────────────────┐
                                │ visualization                │
                                │  2D / 3D / costmap / lattice │
                                │  / hull impact / denoising   │
                                └──────────────────────────────┘
```

Every run writes to one **`--output-dir`** (default `outputs/default/`). The simulation is deterministic from `--seed`; rerunning with the same flags reproduces every artefact bit-for-bit.

---

## Install

```bash
# Recommended: uv (https://docs.astral.sh/uv/)
uv sync

# Alternative: pip
pip install -r requirements.txt
```

All commands in this README assume `uv run python ...`. With plain `pip` install, drop the `uv run` prefix.

The first JAX-backed run JIT-compiles the physics kernel (~30–60 s); subsequent runs reuse the cached compilation.

---

## Running the full MEDEA-map simulation

The MEDEA Fram-Strait satellite scene is the headline scenario. The ship travels diagonally from bottom-left to top-right through a real ice field, and the planner has to find a navigable path. **Requires `data/ice_floes_MEDEA_fram_20100629.pkl`** (see [Data and dependencies](#data-and-dependencies)).

```bash
# 1. Straight diagonal, no planning — fixed reference trajectory through the ice field.
uv run python run_medea_ship_fracture.py --full-map --output-dir outputs/medea_straight

# 2. Lattice A* — needs --fast-lattice (A* tuning for the 1.5 km map) and --polygon-replan
#    (legacy CostMap during replans; see "sharp edges" note in TL;DR).
uv run python run_medea_ship_fracture.py --full-map --plan --planner lattice \
    --fast-lattice --polygon-replan --output-dir outputs/medea_lattice

# 3. Diffusion-MPC (default planner — fully GPU).
uv run python run_medea_ship_fracture.py --full-map --plan \
    --output-dir outputs/medea_diffusion

# 4. MPPI (importance-weighted sampling).
uv run python run_medea_ship_fracture.py --full-map --plan --planner mppi \
    --output-dir outputs/medea_mppi

# 5. SQP (multiple-shooting) or GradCEM (gradient + cross-entropy).
uv run python run_medea_ship_fracture.py --full-map --plan --planner sqp     --output-dir outputs/medea_sqp
uv run python run_medea_ship_fracture.py --full-map --plan --planner gradcem --output-dir outputs/medea_gradcem
```

Each command produces, in `--output-dir`:

| File | Description |
|------|-------------|
| `medea_ship_fracture.mp4` | 2D top-down animation (ice floes, bonds, ship hull, fracture). |
| `medea_ship_fracture_3d.mp4` | Same simulation with z-tilt, submergence, 3D hull. |
| `costmap_medea_ship_fracture.mp4` | Costmap background, **red** = initial plan, **blue** = actual ship path. |
| `lattice_search_medea.mp4` | A* node expansion + chosen path (only with `--planner lattice`). |
| `hull_impact_medea.png` | Accumulated impact vectors on the ship hull (AUTO-IceNav style). |

Useful add-on flags for these commands:
- `--warm-start` — sampling/gradient planners seed from an A* path instead of cold-starting.
- `--no-replan` — single up-front plan, no receding-horizon replanning. Side-effect: also avoids the lattice + `JaxCostMapResult` issue, so it can be used in place of `--polygon-replan`.
- `--plan-only` — plan once, save costmap+path PNG, exit (no simulation; fastest way to debug a planner).
- `--no-3d` — skip the 3D MP4 (saves ~half the rendering time).
- `--quick` — simulation only, no animations.
- `--gif` — write GIFs instead of MP4 (slower, larger files).

Open-water example, no planning:

<video src="outputs/my_run/medea_ship_fracture.mp4" controls width="380"></video>

Costmap movie (red = plan, blue = actual):

<video src="outputs/my_run/costmap_medea_ship_fracture.mp4" controls width="380"></video>

---

## Generating your own synthetic maps

There are three independent ways to build a custom ice scene:

### 1. `--simple` / `--toy-sparse` (zero code)

`run_medea_ship_fracture.py` ships with two built-in synthetic scenes:

| Flag | Scene |
|------|-------|
| `--simple` | 3 hand-placed floes, straight left→right path, no MEDEA data required. |
| `--toy-sparse` | Sparser synthetic field for plan-vs-execute comparisons. |

Combine with `--plan`, `--planner …`, `--thick-ice M`, `--seed N`, etc. as for `--full-map`.

### 2. Random convex-polygon fields — `ice_field_generator.generate_ice_field`

`medea_ice_ship/ice_field_generator.py` produces AUTO-IceNav-style random convex-polygon ice fields at a target concentration. The simplest way to use it for a one-off scene:

```python
from medea_ice_ship.ice_field_generator import generate_ice_field

field = generate_ice_field(
    concentration=0.35,            # 35% area coverage
    map_shape=(1400.0, 1400.0),    # (length_y, width_x) in metres
    ice_start_y=0.0,               # leave clear water below this y
    min_radius=5.0, max_radius=80.0,
    thickness_range=(0.3, 3.5),    # m
    seed=42,
)
# field keys: polygons, centroids, thicknesses, concentration, domain
```

`examples/benchmark_planners.py` is the canonical end-to-end example — it generates random fields at a list of concentrations, runs each selected planner, and saves planned/actual paths + a metrics CSV:

```bash
# Plan-only (no physics) — fastest path comparison across planners
uv run python examples/benchmark_planners.py --no-sim \
    --concentrations 0.2 0.3 0.4 0.5 --output-dir outputs/my_benchmark

# Full sim (slower) — adds result_*.png and benchmark_results.csv
uv run python examples/benchmark_planners.py \
    --planners lattice diffusion-mpc mppi --concentrations 0.3 \
    --output-dir outputs/my_benchmark
```

### 3. Fixed-grid synthetic benchmark — `medea_ice_ship.synthetic_map`

For the formal benchmark we use a **fixed 300 × 750 m corridor**, ship at (150, 50) heading +y, goal line y = 700. Scenarios vary concentration (5–70 %), thickness mode (`uniform_thin` / `uniform_thick` / `bimodal`), and presence of a uniform current. The scenario grid is enumerated by `synthetic_map.list_scenarios('A')`, `list_scenarios('B')`, and `list_scenarios('C')`.

The benchmark is split into three blocks:

| Block | Grid | Notes |
|-------|------|-------|
| **A** | 36 scenarios × 7 planners × 5 seeds = 1 260 trials | Fracturable floes (tessellated, DPDEM bonds). |
| **B** | 21 scenarios × 7 planners × 5 seeds =   735 trials | Adds ocean currents on top of selected A configurations. |
| **C** | 36 scenarios × 7 planners × 5 seeds = 1 260 trials | Same grid as A but **rigid floes** (no tessellation, no fracture). |

Run a single trial:

```bash
uv run python run_experiment_trial.py \
    --block A \
    --scenario A_conc30_thin \
    --planner mbd \
    --seed 0 \
    --output-dir outputs/my_experiment
```

Run one planner against every scenario locally (smoke test):

```bash
uv run python run_smoke_test.py --planner astar --with-plots
uv run python run_smoke_test.py --planner mbd   --with-plots --with-denoising-movie
```

Run a specific block — e.g. all seeds × all planners × block C only:

```bash
uv run python submit_experiments.py --block C --submit
```

Smoke test (seed 0, all artifacts) for one block — loop every planner over block C:

```bash
for planner in astar mbd eb-mbd astar-eb-mbd mppi rss ilqr; do
    uv run python run_smoke_test.py --planner "$planner" --block C \
        --with-movies --with-search-movie --with-denoising-movie
done
```

Submit the whole grid to SLURM:

```bash
# Default (no --block): enumerates A + B + C.
uv run python submit_experiments.py        # generate + submit sbatch files in slurm_scripts/
uv run python submit_experiments.py --block A   # restrict to one block (e.g. prior A-only behaviour)
```

See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) and [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the experiment design, and `scripts/README_SLURM.md` for SLURM specifics.

---

## Toy problems

Standalone scripts under `examples/` exercise individual planner behaviours without needing the MEDEA data. They are the fastest way to develop intuition for each planner.

### Bimodal floe — multimodal collapse demo

A single large floe centred on the straight-line path, leaving two equivalent corridors. Showcases how vanilla MBD averages the two modes and steers straight into the floe, vs how **EB-MBD** picks one corridor cleanly.

```bash
uv run python examples/toy_mbd_bimodal.py
uv run python examples/toy_mbd_bimodal.py --use-best-sample
uv run python examples/toy_mbd_bimodal.py --nsample 256 --ndiffuse 30
```

Outputs land in `outputs/toy_mbd_bimodal/`:

| File | Description |
|------|-------------|
| `bimodal_costmap.png` | Costmap + the two ideal bypass routes. |
| `bimodal_result.png` | Final mean trajectory vs global-best sample. |
| `eb_MBD.mp4` | EB-MBD denoising evolution. |
| `denoising/replan_0000_denoising.mp4` | Per-iteration sampled-trajectory fan. |
| `bimodal_*_histogram.mp4`, `cost_category_analysis.mp4`, … | Diagnostic per-iteration histograms and cost-category breakdowns. |

<video src="outputs/toy_mbd_bimodal/eb_MBD.mp4" controls width="380"></video>

### MBD scenario suite — five stress tests

`examples/toy_mbd_scenarios.py` packages five synthetic costmaps designed to break specific MBD failure modes (thin corridors, sequential gates, asymmetric corridors, clutter, hard rectangular obstacles). Detailed flag table is in [§9 below](#9-mbd-scenario-experiments-toy_mbd_scenariospy).

```bash
# One scenario × one method
uv run python examples/toy_mbd_scenarios.py --scenario thin_corridor --method eb_homotopy_mbd

# One scenario × every method (writes a comparison table)
uv run python examples/toy_mbd_scenarios.py --scenario sequential_gates --run-all-methods
```

Each (scenario, method) directory contains `costmap.png`, `result.png`, `trajectory_fan.mp4`, optional `eb_barrier_schedule.mp4` / `homotopy_class_evolution.png`, and a `denoising/` movie folder.

### Wall-with-opening planner toy

Densely packed ice with one gap; tests planner ability to detour to the opening rather than head straight.

```bash
uv run python examples/toy_planner_opening.py
uv run python examples/toy_planner_opening.py --planner diffusion-mpc
uv run python examples/toy_planner_opening.py --damage-cost     # damage-aware costmap
```

Outputs (`outputs/toy_planner/`): `toy_planner_lattice_search.mp4`, `toy_planner_opening.mp4`, `toy_planner_opening_costmap.mp4`, `toy_planner_hull_impact.png`.

### Thin-ice corridor toy

Dense 10 m ice everywhere except a 1 m horizontal strip; tests whether a damage-aware planner routes through the thin strip.

```bash
uv run python examples/toy_planner_thin_corridor.py
uv run python examples/toy_planner_thin_corridor.py --no-damage-cost   # baseline
```

Outputs (`outputs/toy_planner_thin_corridor/`): `toy_thin_corridor.mp4`, `toy_thin_corridor_3d.mp4`, `toy_thin_corridor_costmap.mp4`, `toy_thin_corridor_lattice_search.mp4`, `toy_thin_corridor_hull_impact.png`.

### Other useful examples

| Script | Purpose |
|--------|---------|
| `examples/visualize_costmap.py` | Render the costmap built from a simple ice field. |
| `examples/visualize_planned_path.py` | Plan a path and overlay it on the costmap. |
| `examples/visualize_diffusion.py` | Animate the MBD denoising trajectory distribution. |
| `examples/benchmark_planners.py` | Random-ice planner benchmark (see [§10](#10-planner-benchmark-random-ice-fields-all-planners)). |
| `examples/autonomous_navigation.py` | End-to-end navigator using `dpdem.navigation.AutonomousNavigator`. |
| `examples/ship_dynamics_demo.py`, `thick_ice_unbreakable_demo.py` | Focused physics demos. |

---

## Output gallery (what to look for in `--output-dir`)

| File pattern | Generated when | Description |
|--------------|----------------|-------------|
| `*_ship_fracture.mp4` | Always (unless `--quick`) | 2D top-down: floes, bonds, ship hull, fracture over time. |
| `*_ship_fracture_3d.mp4` | Always unless `--no-3d` | Same run with z-tilt, submergence, 3D hull. |
| `costmap_*.mp4` | `--plan` | Costmap background + red plan + blue actual path. |
| `lattice_search_*.mp4` | `--plan --planner lattice` (unless `--no-lattice-search-movie`) | A* expansion + chosen path. |
| `hull_impact_*.png` | When collisions occurred | Accumulated impact vectors on the hull. |
| `plan_only_*.png` | `--plan-only` | Costmap + planned path + ice thickness (no simulation). |
| `replan_NNNN_*` (density, top5, costs, terminal_scatter, denoising) | `--mbd-viz-dir DIR` (with diffusion-MPC) | Per-replan MBD diagnostics; add `--mbd-denoising-movie` for the per-replan denoising MP4. |
| `eb_barrier_schedule.mp4` | EB methods (toy scenarios / `--eb-mbd`) | Barrier offset c(t) tightening over reverse diffusion. |
| `homotopy_class_evolution.png` | homotopy methods | Sample-class assignments across diffusion iterations. |
| `trajectory_fan.mp4` | toy scenario suite | Sampled trajectory fan + crossing histogram. |
| `benchmark_results.csv` | `examples/benchmark_planners.py` (with sim) | Per-(concentration, planner) metrics: lengths, costs, impulse, fuel proxy. |

File format defaults to **MP4**. Use `--gif` to write GIFs (slower, larger, single-threaded). All file paths above are relative to whatever you pass as `--output-dir` (default `outputs/default/`).

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

## Full command reference

The sections above (full-MEDEA, synthetic maps, toy problems) cover the common workflows. This section catalogues every command-line invocation in detail. All commands use `run_medea_ship_fracture.py` unless noted. Use `uv run python ...` to run with the project environment.

### 1. Simple straight (no planning) — *currently mostly visual debugging only*

Minimal synthetic ice field (3 floes), no planning. **Caveat (current code state):** the bare `--simple` path uses `n_frames_simple = 300 × dt = 0.05 s` (15 s of sim time) and the AUTO-IceNav 3-DOF ship can't cover the ~260 m reference trajectory in that time — the ship looks stationary. `--kinematic-ship` is currently a silent no-op on the JAX backend, so it does not rescue this. Use this mode if you want to inspect the static ice field and bond network at `t=0`; otherwise jump to §3 (lattice with `--plan`).

```bash
uv run python run_medea_ship_fracture.py --simple
```

- **Output dir:** `outputs/default/` (or `--output-dir DIR`)
- **Generated:** 2D animation, 3D animation; hull impact plot if there are collisions.

### 2. Simple straight with thick ice

Override ice thickness (e.g. 10 m) for all cells. Useful for testing fracture vs unbreakable ice. Same dynamics caveat as §1.

```bash
uv run python run_medea_ship_fracture.py --simple --thick-ice 10
```

Optional: `--thrust-cap 25e6` (25 MN) to cap ship thrust; `--no-thrust-cap` for unlimited.

### 3. Lattice planner (A* on motion primitives)

Plan a path through the simple ice field with **lattice A*** (costmap + Dubins-like primitives), then simulate the ship following that path with receding-horizon replanning. `--polygon-replan` is required: during replans the runner otherwise builds a `JaxCostMapResult` (no `world_to_grid`) and the legacy lattice planner crashes. `--polygon-replan` falls back to the same `build_costmap_from_ice_state` used at initial plan time. Alternatively, `--no-replan` disables replanning entirely.

```bash
uv run python run_medea_ship_fracture.py --simple --plan --planner lattice --polygon-replan
```

- **Planner:** A* on motion primitives (AUTO-IceNav style).
- **Generated:** 2D animation, 3D animation, **lattice search movie** (A* expansion), **costmap movie** (costmap + planned path in red, actual path in blue), hull impact plot when applicable.
- Skip lattice search movie: add `--no-lattice-search-movie`.

### 4. Diffusion-MPC planner (default)

Plan and optimize control sequences using **model-based diffusion** (3-DOF dynamics and costmap). This is the default planner — `--plan` alone uses it.

```bash
# Cold start (default): diffusion runs from scratch, no A* warm-start
uv run python run_medea_ship_fracture.py --full-map --plan

# Warm start: run A* first, use result to seed diffusion
uv run python run_medea_ship_fracture.py --full-map --plan --warm-start
```

- **Planner:** Model-based diffusion over control sequences (3-DOF dynamics, costmap, path/waypoint rewards). Fully JAX/GPU-accelerated.
- **Generated:** 2D, 3D, costmap movie, hull impact plot.

> **Heads up:** the diffusion-MPC default horizon (`planning_jax_mbd_horizon=700`, `dt_substep=1.0 s` ≈ 2.8 km at the 4 m/s target speed) is tuned for the full MEDEA scene. On the small `--simple` field (~150 m wide) the planned trajectory will extend well past the costmap. Use `--planner lattice` for `--simple` (§3), or override the horizon explicitly if you need diffusion-MPC there.

### 4b. MPPI planner (Model Predictive Path Integral)

Same pipeline as diffusion-MPC but uses **MPPI** (importance-weighted sampling with fixed-variance noise). Same cost function (accumulated costmap + arc length + goal).

```bash
uv run python run_medea_ship_fracture.py --simple --plan --planner mppi
```

- **Planner:** MPPI (JAX, GPU-friendly). Cold start by default; add `--warm-start` for A* seed.

### 5. Kinematic ship (path verification) — *currently unsupported on JAX backend*

The historical purpose of this mode is to place the ship exactly on the planned path each frame (no dynamics) for a sanity check that red (plan) and blue (actual) coincide. **In the current code, `--kinematic-ship` is a silent no-op:** both `simulate_jax_chunked_replan` and `simulate_full_jax_scan` run AUTO-IceNav dynamics regardless. Treat the commands below as a placeholder until the kinematic path is restored.

```bash
# These currently behave the same as the corresponding --plan runs without --kinematic-ship.
uv run python run_medea_ship_fracture.py --simple --plan --planner lattice --polygon-replan --kinematic-ship
uv run python run_medea_ship_fracture.py --simple --plan --kinematic-ship
uv run python run_medea_ship_fracture.py --simple --plan --planner mppi --kinematic-ship
```

### 6. Full MEDEA map (diagonal ship)

Use the full satellite ice map; ship goes diagonal bottom-left → top-right. Requires MEDEA data: `data/ice_floes_MEDEA_fram_20100629.pkl`.

```bash
uv run python run_medea_ship_fracture.py --full-map
uv run python run_medea_ship_fracture.py --full-map --plan --planner lattice --fast-lattice --polygon-replan
uv run python run_medea_ship_fracture.py --full-map --plan
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

### 6c. Full MEDEA map with EB-MBD (Emerging-Barrier MBD)

Replicates the constraint + cost structure of `examples/toy_mbd_bimodal.py` on the full MEDEA map: the goal-line (`y = 600 m`) is enforced as a progressively tightening log-barrier instead of a big-M penalty, OOB is a linear soft cost on `x∈[x_min, x_max]` and `y ≥ y_min` (no top wall), and the costmap/OOB integrals are truncated at the top edge of the costmap (above the goal line). No big-M anywhere.

```bash
# Baseline diffusion-MPC (standard soft goal-distance cost, no EB barrier)
uv run python run_medea_ship_fracture.py --full-map --plan --planner diffusion-mpc \
    --mbd-viz-dir outputs/mbd_debug --mbd-denoising-movie --no-3d

# Same command with EB-MBD enabled (mirrors examples/toy_mbd_bimodal.py)
uv run python run_medea_ship_fracture.py --full-map --plan --planner diffusion-mpc \
    --mbd-viz-dir outputs/mbd_debug --mbd-denoising-movie --no-3d --eb-mbd
```

What `--eb-mbd` changes in this scenario (vs. the baseline call above):
- **Goal-line reaching** is enforced via the EB log-barrier on `g(u) = y_final − 600` rather than a 1e6 big-M penalty.
- **OOB** becomes a linear soft cost (`5 / m` of violation distance) over `x ∈ [x_origin, x_origin + width]` and `y ≥ y_origin`; the costmap query no longer adds an `oob_penalty=1e8` for off-grid hull corners.
- **Cost truncation** is set to the costmap's top edge: waypoints past the first crossing of `y_top` contribute zero costmap & zero OOB cost.

Tuning knobs (defaults shown):

| Flag | Meaning | Default |
|------|---------|---------|
| `--eb-c-max 1500` | Initial barrier offset [m] — should exceed the max expected constraint violation. | 1500 |
| `--eb-mu 50000` | Barrier hardness (cost units). | 50000 |
| `--eb-kappa 1.0` | Schedule shape: 1 = linear tightening, > 1 = slow-then-fast. | 1.0 |
| `--eb-oob-constraint` | Treat OOB as a hard EB barrier constraint instead of the default linear soft cost. | off |

The initial plan and every replan use the same EB-MBD setup. Per-replan debug plots (density, top-5, cost-breakdown) go to `--mbd-viz-dir`, and `--mbd-denoising-movie` adds a per-replan MP4 of the denoising evolution.

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

### 9. MBD scenario experiments (`toy_mbd_scenarios.py`)

Five synthetic scenarios designed to stress-test homotopy exploration. Each scenario has a fixed costmap and goal line; the planner must reach the goal near a specified x-position.

**Cost function (all scenarios):**
```
J(u) = ∫ costmap(traj) ds  +  1.0 × |x_crossing − goal_x|
```
Arc length, goal distance, and smoothness weights are all zero. The x-arrival penalty is always active.

**Scenarios:**

| Name | Description | Goal x (m) |
|------|-------------|-----------|
| `thin_corridor` | Wide floe leaving 80 m corridors — stresses ESS collapse | 400 |
| `sequential_gates` | Two offset gates requiring an S-curve path | 400 |
| `asymmetric` | Wide left / narrow right corridor — goal on narrow side | 760 |
| `cluttered` | 12 scattered ice disks — many-constraint performance | 400 |
| `sequential_gates_hard` | Gates as hard rectangular obstacles (EB/AL only) | 400 |

**Run one scenario with one method:**
```bash
uv run python examples/toy_mbd_scenarios.py --scenario thin_corridor --method eb_homotopy_mbd
uv run python examples/toy_mbd_scenarios.py --scenario sequential_gates --method eb_homotopy_mbd
uv run python examples/toy_mbd_scenarios.py --scenario asymmetric --method eb_homotopy_mbd
uv run python examples/toy_mbd_scenarios.py --scenario cluttered --method eb_homotopy_mbd
uv run python examples/toy_mbd_scenarios.py --scenario sequential_gates_hard --method eb_homotopy_mbd
```

**Run all methods on one scenario (comparison table):**
```bash
uv run python examples/toy_mbd_scenarios.py --scenario sequential_gates --run-all-methods
```

**Available methods:** `mbd`, `eb_mbd`, `reject_mbd`, `al_mbd`, `sa_mbd`, `homotopy_mbd`, `eb_homotopy_mbd`, `al_homotopy_mbd`, `lattice`

**Key tuning flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--nsample N` | 512 | Number of diffusion samples |
| `--ndiffuse N` | 50 | Diffusion iterations |
| `--temperature T` | 0.01 | Sampling temperature |
| `--horizon H` | 1200 | Planning horizon (steps) |
| `--eb-c-max C` | 1500.0 | EB barrier saturation level |
| `--eb-mu M` | 50000.0 | EB barrier sharpness |
| `--eb-oob-cost C` | 5.0 | OOB linear cost per metre |
| `--homotopy-n-obs-max N` | 3 | Max obstacles for H-signature (→ 2^N classes) |
| `--no-inflation` | off | Disable ship-beam dilation of obstacles |
| `--output-root DIR` | `outputs/scenarios` | Root output directory |

**Outputs per run** (under `outputs/scenarios/<scenario>/<method>/`):

| File | Description |
|------|-------------|
| `costmap.png` | Costmap with start/goal |
| `result.png` | Best trajectory on costmap |
| `trajectory_fan.mp4` | Fan of sampled trajectories over diffusion |
| `eb_barrier_schedule.mp4` | EB barrier tightening over iterations (EB methods) |
| `homotopy_class_evolution.png` | Class assignments over diffusion (homotopy methods) |
| `denoising/` | Per-iteration mean trajectory frames |

---

### 10. Planner benchmark (random ice fields, all planners)

The benchmark script generates **random ice fields** at several concentrations (AUTO-IceNav style), runs every selected planner on each field, and saves planned-path and result (planned + actual path) images with a **planner name suffix**, plus a CSV of metrics (path length, costmap cost, impulse, fuel proxy).

```bash
# Full benchmark: all planners, all concentrations, plan + simulate (will take long - instead try one planner at a time)
uv run python examples/benchmark_planners.py

# Plan-only (no physics simulation) — fastest, for comparing planned paths (good for vis)
uv run python examples/benchmark_planners.py --no-sim

# Specific planners and concentrations
uv run python examples/benchmark_planners.py --planners lattice diffusion-mpc --concentrations 0.2 0.4

# Warm start for sampling planners (A* seeds the optimization)
uv run python examples/benchmark_planners.py --warm-start --planners diffusion-mpc mppi gradcem

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
| `--warm-start` | off | For diffusion-mpc/mppi/sqp/gradcem: seed optimization from A* path (default is cold start). |
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

## Common flags (reference)

Shown by `run_medea_ship_fracture.py --help`; the ones most relevant for getting started are in the Quick-start and full-MEDEA sections above. Full reference:

| Flag | Effect |
|------|--------|
| `--simple` | Synthetic 3-floe ice field (no MEDEA data). |
| `--full-map` | Full MEDEA map; ship diagonal. |
| `--plan` | Plan path (costmap + planner); only with `--simple` or `--full-map`. |
| `--plan-only` | Plan and save costmap+path to PNG, then exit (no simulation). Implies `--plan`. |
| `--no-show` | With `--plan-only`: save to file only, do not open window (headless). |
| `--planner diffusion-mpc` | **Default.** Model-based diffusion MPC (JAX/GPU). |
| `--planner lattice` | Lattice A* on motion primitives. |
| `--planner mppi` | MPPI (importance-weighted sampling). |
| `--planner sqp` | Multiple-shooting SQP (jaxproxqp). |
| `--planner gradcem` | GradCEM (gradient + Cross-Entropy Method). |
| `--warm-start` | With diffusion-mpc/mppi/sqp/gradcem: seed from A* path before optimization (default is cold start). |
| `--kinematic-ship` | With `--plan`: ship follows path exactly (no dynamics). |
| `--thick-ice M` | Override ice thickness to M metres. |
| `--no-dpdem-bonds` | Use legacy bond model (force vs strength) instead of default DPDEM-style (hybrid criterion, damage, Mohr-Coulomb). |
| `--quick` | Simulation only, no animations. |
| `--no-3d` | Skip 3D animation. |
| `--no-lattice-search-movie` | With `--plan --planner lattice`: skip A* search movie. |
| `--output-dir DIR` | Write outputs to `DIR` (default `outputs/default`). |
| `--float32` | Use float32 instead of float64 for JAX computation (faster on GPU, less accurate). |
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
| `--ice-tank-psv` | Switch to the AUTO-IceNav ice-tank PSV-scale vessel (L≈83 m, B≈17 m, 8 kt) instead of the default Fossen supply vessel (L=76.2 m, B=17 m, 6 kt). |
| `--icebreaker` | Use the large icebreaker from config (L=158 m, B=28 m, 23 kt) instead of the default Fossen supply vessel. |
| `--eb-mbd` | With `--plan`: use Emerging-Barrier MBD — a log-barrier that tightens progressively over the reverse diffusion instead of big-M OOB/goal penalties. |
| `--eb-c-max M` | EB-MBD barrier offset at the start of reverse diffusion [m] (default 1500). |
| `--eb-mu F` | EB-MBD barrier hardness in cost units (default 50000). |
| `--eb-kappa K` | EB-MBD barrier schedule shape: 1 = linear tightening, >1 = slow-then-fast (default 1.0). |
| `--eb-oob-constraint` | EB-MBD: enforce OOB as a hard barrier constraint instead of the default linear soft cost. |
| `--eb-goal-x-weight W` | EB-MBD: soft penalty weight for x-deviation at goal-line crossing (default 0, disabled). |
| `--mbd-viz-dir DIR` | With `--planner diffusion-mpc`: save per-replan debug plots (density, top-5, cost breakdown) to `DIR`. |
| `--mbd-denoising-movie` | With `--mbd-viz-dir`: also save a denoising-evolution MP4 per replan step. |

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

- **Diffusion-MPC (`--planner diffusion-mpc`, default):** Model-based diffusion over control sequences with 3-DOF ship dynamics and a JAX-native GPU costmap. By default uses **cold start** (optimization runs from scratch each replan). Add `--warm-start` to seed from a lattice A* path first. Fully JAX/GPU-accelerated — the diffusion loop runs as a single JIT-compiled XLA program. See [docs/DIFFUSION_MPC.md](docs/DIFFUSION_MPC.md).
- **Lattice (`--planner lattice`):** A* on motion primitives (Dubins-like), costmap, ship footprint and turning radius respected.
- **MPPI (`--planner mppi`):** Model Predictive Path Integral (importance-weighted sampling) with the same cost function as diffusion-MPC. Cold start by default; `--warm-start` for A* seed.
- **SQP (`--planner sqp`):** Multiple-shooting Sequential Quadratic Programming (jaxproxqp) with 3-DOF dynamics and costmap. Cold start by default.
- **GradCEM (`--planner gradcem`):** GradCEM (gradient + Cross-Entropy Method) combining RMSprop with CEM elite selection and resampling. Cold start by default.

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

## JAX backend

All physics run through a **JAX-accelerated backend** — no NumPy simulation loop. The first call JIT-compiles the physics kernel (~30–60 s); subsequent calls are fast.

Two execution paths are selected automatically:

| Path | When | Description |
|------|------|-------------|
| **Chunked-replan** | `--plan` (with replanning) | Physics runs in `jax.lax.scan` segments between replan events. Python overhead is O(n_replans), not O(n_frames). Costmap rebuilt via GPU-native disk rasterization (`build_costmap_jax`). |
| **Full scan** | No replanning | All frames compiled into a single XLA program; zero per-frame Python overhead or GPU sync. |

Use `--float32` for faster GPU execution (default is float64 for numerical accuracy).

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

- **JAX-accelerated physics:** Full simulation JIT-compiled via `jax.lax.scan` for GPU/TPU execution. Physics overhead scales O(n_replans), not O(n_frames). First call triggers XLA compilation; subsequent calls are fast.
- **Paper-accurate physics:** Full equations of motion (Eqs. 7–11) from the DPDEM papers — translational and rotational dynamics, quadratic hydrodynamic drag, buoyancy moments, rotational drag, Rayleigh time step. Active by default; no flags needed.
- **DPDEM bond fracture (default):** Hybrid criterion (⟨σ⟩/σ̄)²+(τ/τ̄)², Mohr–Coulomb shear strength, damage variable D, Benzeggagh–Kenane mixed-mode fracture energy (Eqs. 14–19). Spatially-local crack propagation. Use `--no-dpdem-bonds` for the legacy force-vs-strength model.
- **Voronoi ice elements:** Irregular polygon shapes (paper Fig. 2); MEDEA floes tessellated for fracture.
- **Contact:** Hertz-like normal repulsion with Coulomb friction for ice-ice contact; penetration-based ship collision with thrust cap.
- **3D visualization:** Ice z-offset and tilt from ship-relative position; tapered 3D hull.
- **Ship dynamics:** Optional 3-DOF (Fossen-style M, D, Coriolis) with PD tracking; or kinematic mode for path verification.
- **Five planners:** Lattice A*, diffusion-MPC (default), MPPI, SQP, GradCEM — all with receding-horizon replanning and GPU-native costmap.

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
uv run python run_medea_ship_fracture.py --simple --plan --planner lattice --output-dir outputs/simple_lattice

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
