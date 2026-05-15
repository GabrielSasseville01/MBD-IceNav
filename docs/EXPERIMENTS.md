# Ice-Navigation Planner Benchmark

Detailed description of the synthetic benchmark used to compare ice-navigation planners.
Infrastructure lives in `run_experiment_trial.py`, `submit_experiments.py`,
`aggregate_results.py`, `medea_ice_ship/synthetic_map.py`, and `slurm_scripts/`.

---

## Map geometry (fixed for all trials)

| Parameter | Value |
|-----------|-------|
| Domain | 300 m wide (x) × 700 m long (y) |
| x bounds | [−150, 150] m |
| y bounds | [0, 700] m |
| Ship start | (0, 50) m, heading +y (π/2 rad) |
| Goal line | y = 650 m (any x within 15 m counts as success) |
| Floe radii | 3–25 m (uniform random) |

The ship navigates from near the southern edge to the goal line in the +y direction,
through randomly generated ice fields of varying concentration and thickness.

---

## Experiment blocks

### Block A — Full factorial (36 scenarios)

Crosses ice concentration, thickness distribution, and presence of ocean currents.

| Axis | Values |
|------|--------|
| Concentration | 0.05, 0.10, 0.20, 0.30, 0.40, 0.50 |
| Thickness mode | `uniform_thin`, `uniform_thick`, `bimodal` |
| With currents | False, True |

**Total: 6 × 3 × 2 = 36 scenarios.**

**Scenario ID format:** `A_conc{cc}_{thickness}[_curr]`
- e.g. `A_conc30_thin`, `A_conc20_bimodal_curr`

### Block B — Concentration trend (21 scenarios)

Higher concentration range, no currents — stress-tests planners at near-impassable ice density.

| Axis | Values |
|------|--------|
| Concentration | 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70 |
| Thickness mode | `uniform_thin`, `uniform_thick`, `bimodal` |
| With currents | — (always False) |

**Total: 7 × 3 = 21 scenarios.**

**Scenario ID format:** `B_conc{cc}_{thickness}`
- e.g. `B_conc50_bimodal`, `B_conc70_thick`

---

## Thickness modes

| Mode | Description |
|------|-------------|
| `uniform_thin` | All floes 0.3 m thick |
| `uniform_thick` | All floes 0.5 m thick |
| `bimodal` | 10% of floes at 1.5 m, rest at 0.3 m |

Ice layout (positions, shapes) is seeded by the trial seed. Thickness assignment uses a separate RNG (seed + 10,000) so that two trials with the same seed but different thickness modes have identical ice geometry.

---

## Ocean currents (Block A only)

When `with_currents=True`: uniform current of **0.3 m/s in +x**, 0 m/s in +y.

---

## Planners

Six planners are evaluated on every scenario and seed:

| ID | Description |
|----|-------------|
| `astar` | A\* lattice planner on a motion-primitive graph — deterministic, no trajectory optimizer |
| `mbd` | Model-Based Diffusion, cold start, goal-line constraint via goal_y_line |
| `eb-mbd` | MBD with Emerging-Barrier constraint: progressively tightening log-barrier on goal-line reaching + linear soft OOB cost |
| `mppi` | Model Predictive Path Integral |
| `rss` | Random Single Shooting (N = 2048 samples) |
| `ilqr` | Box-constrained iLQR trajectory optimizer |

All trajectory optimizers (mbd, eb-mbd, mppi, rss, ilqr) share:
- **Horizon:** 800 steps
- **Cold start:** no A\* warm-start; initial seed is a straight line to the goal
- **goal_weight = 0:** goal reaching is enforced via the `goal_y_line` argument, not a soft distance term

### EB-MBD barrier parameters

| Parameter | Value |
|-----------|-------|
| `eb_c_max` | 2000 m (initial barrier offset) |
| `eb_mu` | 50,000 (barrier hardness, cost units) |
| `eb_kappa` | 1.0 (linear schedule) |
| `goal_x_weight` | 1.0 (soft penalty on x-deviation at goal crossing) |
| `length_weight` | 1.0 (soft path-length cost) |

---

## Simulation settings

| Parameter | Value |
|-----------|-------|
| Physics timestep `dt` | 0.05 s |
| Max simulated time | 1800 s |
| Replan interval | 30 s |
| Goal tolerance | 15 m (final ship y ≥ goal_y − 15 counts as success) |
| Costmap margin | 30 m (vs 200 m default for the full MEDEA map) |
| Ship model | AUTO-IceNav (`use_auto_icenav_ship=True`) |

---

## Random seeds

Each (block, scenario, planner) triple is run with seeds **0, 1, 2, 3, 4** to average over ice-field randomness.

---

## Total job count

| Block | Scenarios | Planners | Seeds | Jobs |
|-------|-----------|----------|-------|------|
| A | 36 | 6 | 5 | 1,080 |
| B | 21 | 6 | 5 | 630 |
| **Total** | | | | **1,710** |

---

## Output format

Each trial writes a JSON file to:
```
outputs/experiments/{block}/{scenario_id}/{planner}/seed_{seed}.json
```

Fields:

| Field | Type | Description |
|-------|------|-------------|
| `block` | str | `"A"` or `"B"` |
| `scenario_id` | str | e.g. `"A_conc30_thin_curr"` |
| `concentration` | float | Target ice concentration |
| `thickness_mode` | str | `"uniform_thin"` / `"uniform_thick"` / `"bimodal"` |
| `with_currents` | bool | Whether uniform current was active |
| `planner` | str | Planner ID |
| `seed` | int | Random seed |
| `f_max` | float | Max ice–ship contact force magnitude [N] |
| `f_mean` | float | Mean ice–ship contact force [N] |
| `f_95` | float | 95th-percentile ice–ship contact force [N] |
| `path_length` | float | Total distance travelled by ship [m] |
| `final_y` | float | Ship y-position at end of simulation [m] |
| `success` | bool | `final_y >= goal_y_line − 15 m` |
| `simulated_time` | float | Simulated time elapsed [s] |
| `n_replans` | int | Number of replanning calls made |
| `planning_times` | list[float] | Wall time per replan call [s] |
| `wall_time_s` | float | Total wall time for the trial [s] |

---

## Running experiments

### Single trial (local or cluster)

```bash
uv run python run_experiment_trial.py \
    --block A \
    --scenario A_conc30_thin \
    --planner mbd \
    --seed 0 \
    --output-dir outputs/experiments
```

List all scenario IDs for a block:
```bash
uv run python run_experiment_trial.py --block A --list-scenarios
uv run python run_experiment_trial.py --block B --list-scenarios
```

### SLURM batch submission

```bash
# Dry-run: print sbatch commands without submitting
uv run python submit_experiments.py --dry-run

# Count total jobs
uv run python submit_experiments.py --count

# Submit everything
uv run python submit_experiments.py --submit

# Submit a single block
uv run python submit_experiments.py --block A --submit

# Submit a single planner across all scenarios/seeds
uv run python submit_experiments.py --planner mbd --submit

# Combine filters
uv run python submit_experiments.py --block A --planner eb-mbd --seed 0 --dry-run
```

Jobs that already have a result file at `outputs/experiments/…/seed_{seed}.json` are automatically skipped, so re-running `--submit` after partial completion is safe.

SLURM scripts are written to `slurm_scripts/` and logs to `slurm_logs/`.
Each job requests **8 CPUs, 32 GB RAM, 2 h wall time**.

### Aggregating results

```bash
uv run python aggregate_results.py
```

Collects all JSON result files and produces summary tables.

---

## Directory structure

```
run_experiment_trial.py          # single-trial runner
submit_experiments.py            # SLURM job generator + submitter
aggregate_results.py             # result aggregation
medea_ice_ship/synthetic_map.py  # map geometry + scenario definitions
slurm_scripts/                   # pre-generated per-job sbatch scripts
outputs/experiments/             # results (not tracked in git)
slurm_logs/                      # SLURM stdout/stderr (not tracked in git)
```
