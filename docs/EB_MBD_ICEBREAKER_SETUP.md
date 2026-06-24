# EB-MBD setup for the icebreaker full-map command

Exact specification of the EB-MBD optimisation triggered by:

```bash
uv run python run_medea_ship_fracture.py \
    --plan --full-map --planner diffusion-mpc \
    --eb-mbd --eb-oob-constraint --eb-goal-x-weight=1.0 \
    --icebreaker --mbd-viz-dir outputs/eb_mbd_debug_icebreaker3 \
    --mbd-denoising-movie --no-3d
```

Use this as the contract to reproduce on toy scenarios via direct
`jax_mbd_optimize(...)` calls.

---

## 1. Scenario setup (full-map, icebreaker)

| Quantity                | Value                          | Source                                              |
|-------------------------|--------------------------------|-----------------------------------------------------|
| Domain                  | `[-700, 700] × [-700, 700] m`  | `run_medea_ship_fracture.py:662` (`DOMAIN_LIMIT=700`) |
| `start_pose`            | `(-545, -600, π/4)`            | `run_medea_ship_fracture.py:724,855-856`            |
| `goal` (point)          | `(-350, 600)`                  | `run_medea_ship_fracture.py:725,857`                |
| `goal_y_line`           | `600.0`                        | `run_medea_ship_fracture.py:726`                    |
| Initial velocity `nu0`  | `(2.0, 0.0, 0.0)` m/s          | `:995,1004,1268`                                    |
| Ship length `L`         | `158.0 m` (icebreaker)         | `config.py:240`, gated by `--icebreaker`            |
| Ship beam `B`           | `28.0 m`                       | `config.py:242`                                     |
| Ship mass               | `23 000 000 kg`                | `config.py:246`                                     |

The costmap covers exactly `[-700, 700]²` (full-map clamp) and is built by
`build_costmap_from_ice_state(...)` from the polygon ice field. EB-MBD's
boundary parameters are derived from this costmap (see §4).

---

## 2. CLI flags → planner kwargs

The diffusion-MPC branch eventually calls
`optimize_with_diffusion(...)` → `jax_mbd_optimize(...)` with the
following EB kwargs assembled in `run_medea_ship_fracture.py:966-984`:

| Flag                          | Value used | Resolved into `jax_mbd_optimize` kwarg          |
|-------------------------------|-----------:|--------------------------------------------------|
| `--eb-mbd`                    | `True`     | `use_eb_mbd=True`                                |
| `--eb-oob-constraint`         | `True`     | `eb_oob_as_constraint=True`                      |
| `--eb-goal-x-weight=1.0`      | `1.0`      | `planning_jax_mbd_goal_x_weight=1.0` via `replace_config` at `:1158-1163`, then read into `goal_x_weight` at `:3514` |
| `--eb-c-max`                  | `1500.0` (default) | `eb_c_max=1500.0`                         |
| `--eb-mu`                     | `50000.0` (default) | `eb_mu=50000.0`                          |
| `--eb-kappa`                  | `1.0` (default)    | `eb_kappa=1.0`                            |
| `--eb-length-weight`          | `0.0` (default)    | `planning_jax_mbd_length_weight=0.0`     |

EB boundary parameters (set in `:973-984`, defaulting to the costmap extent):

```python
eb_x_min = cost_map.x_origin                       # = -700.0
eb_x_max = cost_map.x_origin + cost_map.width      # =  700.0
eb_y_min = cost_map.y_origin                       # = -700.0
cost_truncate_y = cost_map.y_origin + cost_map.height  # = 700.0
```

`goal_y_line=600.0` is also forwarded.

---

## 3. Decision variable & dynamics

The decision is the normalised control sequence `Y ∈ [-1, 1]^{H × 3}`
(surge, sway, yaw). Each sample is unnormalised and rolled out through the
icebreaker's 3-DOF Fossen-style dynamics:

- `action_center = (surge_max/2, 0, 0)`, `action_scale = (surge_max/2, sway_max, yaw_max)`
- `surge_max = 2e6 N`, `sway_max = 5e5 N`, `yaw_max = 5e8 N·m`
  (`config.py:395-399`)
- Rollout horizon `H = 700` steps × `dt_action = dt_substep × n_substeps = 1.0 × 1 = 1.0 s` → 700 s lookahead
  (`config.py:365,401,404`)
- Mass `m = 23 000 000 kg`; `Izz = (1/12) m (L² + B²)`; added-mass and damping
  from `get_default_simulation_config()` via `make_dynamics_rollout(...)`
  (`jax_mbd_planner.py:3563-3580`)

The rollout returns waypoints `traj ∈ R^{(H+1) × 3}` (x, y, ψ) for every
sample, and an analogous body-frame velocity sequence for adaptive-velocity
costmaps.

---

## 4. Cost function `J(x)` (the “base cost”)

EB-MBD builds the cost from `make_cost_fn(...)` plus an additive
goal-x term (`jax_mbd_planner.py:3637-3657`):

```
J(u_norm) =  w_costmap · costmap_integral(traj)
           + w_length  · path_length(traj)
           + w_goal    · discounted_goal_distance(traj)
           + w_smooth  · jerk(u_norm)
           + w_goal_x  · goal_x_term(traj)        # added because goal_x_weight > 0
```

with weights from `config.py` (since `--eb-mbd` does NOT override them
unless `--eb-length-weight > 0`):

| Symbol      | Default value                                  | Where set                              |
|-------------|------------------------------------------------|----------------------------------------|
| `w_costmap` | `1.0 / 1.0 = 1.0` (`costmap_weight / costmap_normalizer`) | `config.py:377,393`           |
| `w_length`  | `0.0`                                          | `config.py:381` (overridden to `args.eb_length_weight = 0.0` at `:1162`) |
| `w_goal`    | `0.0`                                          | `config.py:379`                        |
| `w_smooth`  | `0.0`                                          | `config.py:383`                        |
| `w_goal_x`  | `1.0` (from `--eb-goal-x-weight=1.0`)          | `config.py:?` via `:1161`              |

### 4.1 `costmap_integral`

Trapezoidal integral of the per-waypoint costmap query along the trajectory
(`jax_mbd_planner.py:704-744`):

```
wp_costs[t] = max_{hull corner k} costmap_query(traj[t] + R(ψ_t) · corner_k)
seg_costmap[t] = 0.5 (wp_costs[t] + wp_costs[t+1]) · ||Δxy_t||
costmap_integral = Σ_t seg_costmap[t]   (truncated at cost_truncate_y)
```

`costmap_query(x, y)` returns the grid value at world (x, y); OOB cells
return `oob_penalty = 0.0` because EB-MBD forces it to zero
(`:3649`) so the hard barrier owns OOB handling.

Hull-corner sampling uses the 4 corners + centre of the icebreaker hull
(`half_l = L/2 = 79 m`, `half_w = B/2 = 14 m`) so the costmap penalty
respects the full footprint.

### 4.2 `path_length`

Sum of `||Δxy_t||` over active waypoints (those before the first
`y ≥ cost_truncate_y` crossing). Weight `w_length = 0`, so this term
is zeroed out for this command.

### 4.3 `goal_dist`

Exponentially-discounted mean Euclidean distance to `goal = (-350, 600)`
(`jax_mbd_planner.py:746-753`):

```
τ = (H+1) / 3
discounts ∝ exp(-t / τ)
goal_dist = Σ_t discounts[t] · ||traj[t,:2] - goal||
```

Weight `w_goal = 0`, so this term is also disabled.

### 4.4 `jerk`

`mean(ΔY²)` over the normalised control sequence (`:755-756`).
Weight `w_smooth = 0`, disabled.

### 4.5 `goal_x_term`  ← the only soft cost actually contributing

`make_terminal_goal_distance_cost_fn(...)` (`:1084-1143`),
weighted by `goal_x_weight = 1.0`:

- If the trajectory crosses `y = goal_y_line = 600`:
  `cost = w_goal_x · |x_crossing - goal_x|` where `x_crossing` is the
  linear interpolation between the two waypoints straddling the first
  `y ≥ 600` transition, and `goal_x = -350`.
- If it never reaches the goal line:
  `cost = w_goal_x · sqrt((x_final - goal_x)² + (goal_y_line - y_final)² + ε)`

This is what pulls samples toward the goal point even when the hard
barriers only enforce y-line crossing and in-bounds.

---

## 5. Constraints (the EB log barriers)

Two **independent additive** log-barriers are wired in
`jax_mbd_planner.py:3674-3768`:

### 5.1 Goal-reaching barrier — `constraint_fn`

`make_goal_constraint_fn(...)` (`:1384-1407`):

```
g_goal(u) = y_final - goal_y_line   (in metres)
```

Feasibility: `g_goal ≥ 0` means the trajectory's final waypoint reaches
or exceeds `y = 600`.

### 5.2 OOB barrier — `oob_constraint_fn` (active because `--eb-oob-constraint`)

`make_oob_constraint_fn(...)` (`:1278-1333`):

```
dist_left   = x_t - eb_x_min            (= x_t + 700)
dist_right  = eb_x_max - x_t            (= 700 - x_t)
dist_bottom = y_t - eb_y_min            (= y_t + 700)
g_oob(u) = min over active waypoints of min(dist_left, dist_right, dist_bottom)
```

Active waypoints are those before the first `y ≥ cost_truncate_y = 700`
crossing. **There is no top-wall constraint** — the trajectory is allowed
to exit through the top of the map.

Feasibility: `g_oob ≥ 0` ⇔ every active waypoint is inside
`[-700, 700] × [-700, ∞)`.

### 5.3 EB schedule

Defined in `run_eb_mbd_with_diagnostics(...)` (`:2380-2418`).
With `Ndiffuse = 50` reverse steps `i = 49, 48, …, 1`:

```
s_frac(i)  = i / (Ndiffuse - 1) = i / 49
c_s(i)     = eb_c_max     · s_frac(i)^κ   = 1500.0 · (i/49)^1.0     # for g_goal
c_s_oob(i) = eb_c_max_oob · s_frac(i)^κ   = 0.0    · (i/49)^1.0     # for g_oob
```

Note `eb_c_max_oob` defaults to `0.0` (`:3418`) so the OOB barrier is
**hard from the very first diffusion step**. Only the goal-line barrier
relaxes (it slowly tightens from a 1500 m budget down to 0 m).

### 5.4 Barrier reward applied per sample

For each sample `k`:

```
barrier_goal_k = eb_mu · log(max(g_goal_k + c_s(i),         1e-6))         # always added
barrier_oob_k  = 0                                  if g_oob_k + c_s_oob(i) ≥ 0
                = eb_mu · log(max(g_oob_k + c_s_oob(i), 1e-6))   otherwise   # one-sided
reward_k       = -J(x_k) + barrier_goal_k + barrier_oob_k
```

with `eb_mu = 50 000`. The goal barrier is two-sided (always pulls toward
larger `g_goal`); the OOB barrier is **one-sided** so in-bounds samples
do not contribute variance to the reward signal.

The next mean is obtained by importance-weighted aggregation:

```
weights_k ∝ softmax(reward_k / temperature)     # temperature = 0.01
Ybar_{i-1} = (1 - α_decay) · Σ_k weights_k · Y0_k + α_decay · Ybar_i
```

(reverse step is the standard MBD α-schedule from `beta0=1e-5` to
`betaT=0.05`, `:3507-3508`.)

---

## 6. Reverse-diffusion hyperparameters

From `config.py:365-406` (read in `jax_mbd_optimize`):

| Symbol            | Value      | Setting                                         |
|-------------------|-----------:|-------------------------------------------------|
| `H`               | 700        | `planning_jax_mbd_horizon`                      |
| `Nsample`         | 2048       | `planning_jax_mbd_nsample`                      |
| `Ndiffuse`        | 50         | `planning_jax_mbd_ndiffuse`                     |
| `temperature` (τ) | 0.01       | `planning_jax_mbd_temperature`                  |
| `beta0`           | 1e-5       | `planning_jax_mbd_beta0`                        |
| `betaT`           | 0.05       | `planning_jax_mbd_betaT`                        |
| `dt_substep`      | 1.0 s      | `planning_jax_mbd_dt_substep`                   |
| `n_substeps`      | 1          | `planning_jax_mbd_n_substeps`                   |
| `init_surge_bias` | 0.7        | `planning_jax_mbd_init_surge_bias`              |
| `warm_start`      | `False` for this command | `--warm-start` flag not passed   |

The receding-horizon controller calls `optimize_with_diffusion` once at
`replan_0000` (initial plan from the lattice path; with `warm_start=False`
the lattice is ignored) and again every `planning_replan_every_n_seconds`
seconds (currently `60.0`, see `config.py:296`).

---

## 7. Reproduction recipe for toy scenarios

Build a `RunConfig` whose `planning_jax_mbd_*` defaults match the table
in §6 (the shipped `config.py` defaults already do), then call
`jax_mbd_optimize(...)` directly:

```python
from medea_ice_ship.config import RunConfig
from medea_ice_ship.jax_mbd_planner import jax_mbd_optimize

run_cfg = RunConfig()           # picks up the §6 defaults
run_cfg.ship_length        = 158.0
run_cfg.ship_beam          = 28.0
run_cfg.ship_displacement  = 23_000e3
# Goal-x soft cost is read off the config (NOT a function kwarg):
run_cfg.planning_jax_mbd_goal_x_weight = 1.0
run_cfg.planning_jax_mbd_length_weight = 0.0   # explicit; matches the CLI

traj, diag = jax_mbd_optimize(
    cost_map=toy_cost_map,
    start_pose=(start_x, start_y, start_psi),
    goal=(goal_x, goal_y),
    run_cfg=run_cfg,
    start_nu=(2.0, 0.0, 0.0),
    lattice_path=None,
    warm_start=False,
    return_diagnostics=True,

    # ── goal / truncation ────────────────────────────
    goal_y_line=goal_y_line,                        # e.g. 600.0
    cost_truncate_y=float(toy_cost_map.y_origin
                          + toy_cost_map.height),   # top of the map

    # ── EB-MBD ───────────────────────────────────────
    use_eb_mbd=True,
    eb_x_min=float(toy_cost_map.x_origin),
    eb_x_max=float(toy_cost_map.x_origin + toy_cost_map.width),
    eb_y_min=float(toy_cost_map.y_origin),
    eb_c_max=1500.0,
    eb_mu=50_000.0,
    eb_kappa=1.0,
    eb_oob_as_constraint=True,         # ← --eb-oob-constraint
    # eb_c_max_oob defaults to 0.0 → OOB hard from step 0

    # leave everything else at the defaults:
    #   eb_ice_as_constraint=False, eb_c_max_ice=None,
    #   use_al_mbd=False, use_hybrid_eb_al_mbd=False,
    #   obstacle_rects=None, obstacle_disks=None, ...
)
```

What this gives you (and what the icebreaker run uses):

- Cost = `1·costmap_integral + 1·goal_x_term` (length / goal_dist / jerk all 0).
- Barriers = `μ·log(g_goal + c_s)` (goal-line, relaxes 1500→0 over 50 steps,
  κ=1 linear) plus one-sided `μ·log(g_oob + 0)` (left/right/bottom walls,
  hard from step 0, top wall not enforced).
- Sample selection via softmax of `(-J + barrier) / 0.01`, propagated through
  the standard MBD reverse-diffusion update.

To reproduce **per-replan** behaviour (what `replan_0001+` does), call
`jax_mbd_optimize` again with the same kwargs and the updated `start_pose`,
`start_nu`, and `cost_map`. The replan callback at
`run_medea_ship_fracture.py:1267-1278` does exactly this.
