# MBD Planner Weight Calibration

This directory contains the calibration experiment for the diffusion-MPC
(Model-Based Diffusion) planner's cost function weights.

## Background

The MBD planner minimises a weighted sum of four terms:

```
J(u) = w_c · C_costmap
     + w_l · C_length
     + w_g · C_goal
     + w_s · C_smooth
```

| Symbol | Config key | Default | What it measures |
|--------|-----------|---------|-----------------|
| `w_c` | `planning_jax_mbd_costmap_weight` | 1.0 | Kinetic-energy penalty integrated along the hull-aware trajectory — captures ice collision risk |
| `w_l` | `planning_jax_mbd_length_weight` | 1.0 | Total path length [m] — penalises unnecessary detours |
| `w_g` | `planning_jax_mbd_goal_weight` | 0.1 | Euclidean distance from trajectory end to goal [m] — directional guidance |
| `w_s` | `planning_jax_mbd_smoothness_weight` | 0.005 | Mean squared control change — penalises jerky manoeuvres |

The default weights were set by intuition and may let one term dominate the
others.  This experiment collects the raw (unweighted) magnitudes of each term
during a realistic run of the full-map scenario and computes balanced weights
such that every term contributes comparably.

## Method

### Step 1 — Collect statistics (`collect_cost_stats.py`)

Probes the planner at **N equally-spaced positions** along the nominal full-map
route (ship start → goal) using the real MEDEA satellite ice field.  At each
position the diffusion-MPC planner is called with `return_cost_components=True`,
which returns the four raw component values for the optimised trajectory before
any weight is applied.  Results are written to a timestamped CSV under
`results/`.

The scenario mirrors:
```
uv run python run_medea_ship_fracture.py \
    --full-map --plan --no-3d --no-lattice-search-movie --replan-every 10
```

Key choices:
- **Cold start** (no A\* warm-up), matching the default CLI behaviour.
- **Static costmap** — built once from the initial ice state.  This is
  representative because the ice field changes slowly compared to the planner
  horizon.
- **Probe spacing** covers 5 % – 85 % of the route to ensure non-trivial
  remaining distances at every probe.

### Step 2 — Compute balanced weights (`analyze_cost_stats.py`)

Reads the CSV and applies a two-stage normalisation:

1. **Magnitude normalisation** — sets each weight so the mean contribution of
   every term equals the mean path length (the most interpretable reference
   scale):
   ```
   w_i^norm = mean(C_length) / mean(C_i)
   ```

2. **Domain-importance multipliers** — multiplies the normalised weights by
   fixed importance factors that encode the relative priority of each term:

   | Term | Importance | Rationale |
   |------|-----------|-----------|
   | costmap | 3× | Primary safety concern |
   | length | 1× | Baseline — kept at normalised value |
   | goal | 0.5× | Guidance heuristic, not a hard constraint |
   | smoothness | 0.2× | Soft comfort term |

   These multipliers are a starting point and can be adjusted after inspecting
   the generated plots.

The script prints a ready-to-paste config snippet and saves a diagnostic figure.

## Directory structure

```
calibration/
├── README.md                  # this file
├── collect_cost_stats.py      # Step 1: probe planner, write CSV
├── analyze_cost_stats.py      # Step 2: analyse CSV, compute weights
├── results/                   # output CSVs and plots (gitignored large files)
│   └── cost_stats_<ts>.csv
└── tests/
    └── test_cost_components.py   # unit tests for the cost-component feature
```

## Usage

### Run the full calibration

```bash
# Step 1 — probe 15 positions (≈ 15 min on CPU, <2 min on GPU)
uv run python calibration/collect_cost_stats.py --n-probes 15

# Step 2 — analyse and get recommended weights
uv run python calibration/analyze_cost_stats.py calibration/results/cost_stats_<timestamp>.csv
```

### Run tests

```bash
uv run pytest calibration/tests/ -v
```

### Apply the weights

Copy the config snippet printed by `analyze_cost_stats.py` into
`medea_ice_ship/config.py` under the `RunConfig` dataclass defaults, or pass
them as `replace_config(...)` overrides in your experiment script.

## Implementation notes

The `return_cost_components` feature was added to:
- `medea_ice_ship/jax_mbd_planner.py` — `jax_mbd_optimize()` evaluates
  `make_cost_components_fn()` on the final `Ybar_final` action sequence and
  returns a dict with raw component values.
- `medea_ice_ship/planning.py` — `optimize_with_diffusion()` threads the flag
  through to `jax_mbd_optimize()`.

Both changes are non-breaking: the flag defaults to `False` and is incompatible
with `return_diagnostics=True` (which uses a different code path).
