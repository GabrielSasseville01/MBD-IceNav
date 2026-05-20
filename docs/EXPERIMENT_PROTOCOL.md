# Ice-Navigation Benchmark — Tuning & Evaluation Protocol

**Pre-registration date:** 2026-05-20
**Branch:** `feature/jax-vectorized`

This document records the methodology committed *before* tuning and test
runs begin. Once the test sweep is launched, the protocol is locked. Any
deviation must be justified in writing here, with the deviation date.

---

## 1. Paper framing

Benchmark paper. Six ice-navigation planners (`astar`, `mbd`, `eb-mbd`,
`mppi`, `rss`, `ilqr`) are evaluated on a fixed scenario suite (Blocks A
and B from `docs/EXPERIMENTS.md`). The final framing of the narrative may
shift with the data, but **the tuning protocol is fixed in advance** to
prevent the hyperparameter search from becoming a post-hoc fishing
expedition.

---

## 2. Train/test split

| Use | Seeds | Scenarios |
|---|---|---|
| Tuning | 0, 1 | 12 stratified scenarios per block (see §4) |
| Test (held-out)  | 2, 3, 4 | All scenarios in the block |

One tuned weight set per `(block, planner-group)` (see §5). Tuning sees
neither held-out seeds nor held-out scenarios — there is no data leakage
from tuning into test.

---

## 3. Tuning objective

For a candidate hyperparameter set `θ`, scored on the tuning eval set:

```
score(θ) = ( -mean( min(final_y, goal_y) / goal_y ),     # primary
              mean( f_mean over successful trials only )   # tie-break
            )
```

Minimised lexicographically (lower-better). Reasoning:
- Pure success-rate has no gradient when all trials fail or all succeed.
  Fractional `final_y / goal_y` is a smooth surrogate that is `1.0` on
  success and continuous otherwise.
- `f_mean` is computed only over successful trials. A planner that fails
  by stalling has near-zero `f_mean` and could win on the tiebreak
  otherwise.
- If no trials succeed in the tuning eval set, the tiebreak is `NaN` and
  the primary term decides.

Catastrophic forces (`f_max` above any ship-survival threshold) are
**not** filtered or penalised. They are reported transparently in the
test results; the reader sets their own threshold.

---

## 4. Tuning eval set (per block)

Twelve scenarios per block, stratified across difficulty:

**Block A (drop `uniform_thin` and `with_currents=True` for tuning):**
```
A_conc05_thick, A_conc05_bimodal,
A_conc10_thick, A_conc10_bimodal,
A_conc20_thick, A_conc20_bimodal,
A_conc30_thick, A_conc30_bimodal,
A_conc40_thick, A_conc40_bimodal,
A_conc50_thick, A_conc50_bimodal,
```

**Block B (drop `uniform_thin` for tuning; B has no currents):**
```
B_conc20_thick, B_conc20_bimodal,
B_conc30_thick, B_conc30_bimodal,
B_conc40_thick, B_conc40_bimodal,
B_conc50_thick, B_conc50_bimodal,
B_conc60_thick, B_conc60_bimodal,
B_conc70_thick, B_conc70_bimodal,
```

`uniform_thin` and `with_currents=True` are therefore **held out from
tuning** and become built-in generalisation tests on the test sweep.

---

## 5. Tuning surface

Three independent tuning groups per block.

### 5a. Group `astar` — A* cost weights

Tunes only A*, scored on A*-only trials.

| Knob | Grid (log-spaced unless noted) |
|---|---|
| `planning_ice_collision_cost` | `[0.5, 5, 50, 500]` |
| `planning_ke_cost_scale` | `[1e-7, 1e-6, 1e-5, 1e-4]` |
| `planning_heuristic_weight` | `[1, 2, 5, 10]` (linear) |

**64 grid points × 12 scenarios × 2 seeds = 1,536 trials per block.**

### 5b. Group `shared_trajopt` — cost weights shared across MBD/MPPI/RSS/iLQR

Scored as the mean of the four planners' lexicographic scores. EB-MBD is
excluded from this scoring to keep its tuning clean (§5c).

| Knob | Grid (log-spaced) |
|---|---|
| `planning_jax_mbd_costmap_weight` | `[0.1, 1, 10, 100]` |
| `planning_jax_mbd_length_weight` | `[0.01, 0.1, 1, 10]` |
| `planning_jax_mbd_goal_x_weight` | `[0.01, 0.1, 1, 10]` |

**64 grid points × 12 scenarios × 4 planners × 2 seeds = 6,144 trials
per block.**

### 5c. Group `eb_mbd_barrier` — EB-MBD barrier parameters

Conditional on the winning `shared_trajopt` weights. Scored on EB-MBD
trials only.

| Knob | Grid |
|---|---|
| `planning_jax_mbd_eb_c_max` | `[200, 600, 1200]` (log-ish) |
| `planning_jax_mbd_eb_mu` | `[1e4, 1e5, 1e6]` (log) |
| `planning_jax_mbd_eb_kappa` | `[0.5, 1, 2]` (linear) |

**27 grid points × 12 scenarios × 2 seeds = 648 trials per block.**

### 5d. Algorithm hyperparameters — frozen

Algorithm hyperparameters (e.g., `nsample`, `n_iterations`,
`max_iterations`) are **not** part of the tuning grid. They are held at
the values in `run_experiment_trial.py` and disclosed in the methods
section as "experiment-conditioned constants." Justifications:

- Map-conditioned: `planning_margin`, `planning_resolution`,
  `planning_num_headings`, `planning_turning_radius`. Fitted to the
  300 × 750 m synthetic domain.
- Design-choice: `planning_jax_mbd_goal_weight = 0.0` (goal enforced by
  `goal_y_line`, not soft penalty), except iLQR which uses 100.0 for
  differentiability.
- Solver-budget: `planning_rss_nsample = 2048`, `planning_max_iterations
  = 500_000`. Sensitivity studied via the ablations in §6.

### 5e. Horizon — per-planner ablation, then frozen

Planning horizon is treated separately from cost weights because
horizon-vs-weights interactions are typically weak and a joint grid
would 4× the tuning cost.

**Step 1** — per-planner horizon ablation at default cost weights:
4 trajopt planners × 4 horizons (`[100, 200, 400, 800]`) × 3 seeds × 2
scenarios (one easy, one hard per block) = **~200 trials**.

**Step 2** — bake the best horizon per planner into the constants in
`run_experiment_trial.py`. Disclosed as a table in the paper.

**Step 3** — cost-weight tuning (§5a-c) runs at the fixed per-planner
horizons.

**Step 4 (confirmation)** — after cost-weight tuning, re-run the winning
configuration at the chosen horizon ± 1 step to verify the decoupling
assumption (~50 trials).

---

## 6. Algorithm-knob sensitivity ablations (appendix only)

Per Q8 of the grilling session, two small ablations live in the paper
appendix to preempt reviewer comments about the frozen knob choices.

### 6a. Horizon ablation

The Step-1 horizon ablation from §5e is itself the appendix figure.
~200 trials.

### 6b. RSS / MBD nsample ablation

MBD `nsample ∈ {1024, 2048, 4096}` and RSS `nsample ∈ {1024, 2048,
4096}` at one representative scenario (e.g., `A_conc30_thick`), 3 seeds
each. ~18 trials.

---

## 7. Search method

Grid search, log-spaced ranges where appropriate. Reasons:
- Transparent and trivially reproducible (publish the ranges and grid
  points; anyone can re-derive the search).
- No surrogate-model assumptions to defend.
- For ≤3-knob search spaces, grid is competitive with BO at this budget.

Random seed for the sweep is the planner-trial seed; the grid order is
deterministic.

---

## 8. Sanity check before tuning

**Before launching the full grid**, run the centre-of-grid weight set
on the 12 tuning scenarios for all 6 planners and confirm:
- The infrastructure works end-to-end.
- Force magnitudes are within expected ranges.
- No planner has 0% success across all 12 scenarios at the centre weights
  (if so, the grid is mis-ranged and must be re-centred).

**72 trials (~30 min on the cluster at ~200-way concurrency).**

---

## 9. Test sweep

After tuning concludes, the winning `(block, group)` weights are baked
into `run_experiment_trial.py`. Then:

| Block | Scenarios | Planners | Seeds | Trials |
|---|---|---|---|---|
| A | 36 | 6 | {2, 3, 4} | 648 |
| B | 21 | 6 | {2, 3, 4} | 378 |
| **Total** | | | | **1,026** |

All test trials generate **full output artefacts**: JSON, map PNG, full
2D movie. (Tuning trials emit JSON only.)

---

## 10. Cross-block consistency check

Block A and Block B overlap on concentrations `{0.10, 0.20, 0.30, 0.40,
0.50}` (no currents). Apply Block A's tuned weights to Block B's overlap
scenarios (5 concs × 3 thickness × 3 seeds = **45 extra trials**) and
compare to Block B's own tuned-weight results.

- If results are similar → "per-block tuning was unnecessary; one set
  generalises."
- If results differ → "the cost landscape really does shift with the
  concentration range."

Either finding is publishable.

---

## 11. Output JSON schema (extended)

Every trial (tuning and test) writes a JSON containing **at minimum**:

### Existing fields
`block`, `scenario_id`, `concentration` (target), `thickness_mode`,
`with_currents`, `planner`, `seed`, `f_max`, `f_mean`, `f_95`,
`path_length`, `final_y`, `success`, `simulated_time`, `n_replans`,
`planning_times`, `wall_time_s`.

### New fields added before tuning starts
- `actual_concentration` — observed ice concentration after generation
- `n_floes` — number of generated floes
- `mean_thickness`, `max_thickness` — thickness statistics
- `n_thick_floes` — count of floes ≥ 1.0 m (for bimodal)
- `final_x`, `final_heading` — full final pose (not just y)
- `cost_components_per_replan` — list of `{costmap, length, goal_x,
  total}` per replan call (trajopt planners only; A* records the
  closed-form path cost)
- `overrides_applied` — dict of `(config_key → value)` actually used for
  this trial (so tuned-vs-default values are auditable from the JSON
  alone)

---

## 12. Reporting (paper structure)

### Main-text tables (per block)

Rows = concentration, columns = planner, cells = mean ± std across
`(thickness_mode × seeds)`. One table per metric:
- Success rate
- `f_mean`
- `path_length`

### Generalisation columns

For each planner, an additional pair of columns shows the
**generalisation gap**:
- `Δ success` on `uniform_thin` (vs the tuned-on `uniform_thick` +
  `bimodal` average)
- `Δ success` on `with_currents=True` (vs the tuned-on
  `with_currents=False`)

### Appendix

- Full per-scenario tables (all 6 metrics)
- Horizon ablation curves (§6a)
- `nsample` ablation curves (§6b)
- Cross-block consistency table (§10)

---

## 13. Compute budget summary

| Phase | Trials |
|---|---|
| Horizon ablation (§5e step 1) | ~200 |
| `nsample` ablation (§6b) | ~18 |
| Sanity check (§8) | 72 |
| Tuning — A* per block | 2 × 1,536 = 3,072 |
| Tuning — shared_trajopt per block | 2 × 6,144 = 12,288 |
| Tuning — eb_mbd_barrier per block | 2 × 648 = 1,296 |
| Confirmation (§5e step 4) | ~50 |
| Test sweep (§9) | 1,026 |
| Cross-block consistency (§10) | 45 |
| **Total** | **~18,000 trials** |

At ~200-way SLURM concurrency and ~15 min average per trial:
**~25 hours wall-clock**.

---

## 14. Execution order

1. Continue current smoke tests for infrastructure validation. **Discard
   the numbers — defaults are not the tuned weights.**
2. Code: extend `_compute_metrics` with the §11 fields; add CLI override
   support to `run_experiment_trial.py`.
3. Run horizon ablation (§5e step 1); bake per-planner horizons into
   the constants table.
4. Run centre-of-grid sanity check (§8); adjust ranges if obviously
   wrong.
5. Run tuning sweeps in order: A*, shared_trajopt, eb_mbd_barrier.
   Score and freeze the winning weights to
   `tuned_weights/{block}/{group}.json`.
6. Confirmation run at horizon ± 1 step.
7. Test sweep with tuned weights and held-out seeds 2-4.
8. Cross-block consistency analysis.
9. `aggregate_results.py` → tables and figures.

---

## Deviations log

*(Append a dated entry here if any decision above is changed.)*
