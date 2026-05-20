#!/usr/bin/env python3
"""
SLURM job-array submitter for the planner-weight tuning grid sweep.

See docs/EXPERIMENT_PROTOCOL.md for the full methodology.

Three tuning groups per block (12 scenarios × N grid points × planners × seeds):

    Group              Grid    Planners    Array tasks    Trials/task    Total trials
    astar              4³ = 64       1          12×64 =  768            2          1 536
    shared_trajopt     4³ = 64       4          12×64 =  768            8          6 144
    eb_mbd_barrier     3³ = 27       1          12×27 =  324            2            648

The job array dimension is (scenarios × grid_points). Each array task
decodes its SLURM_ARRAY_TASK_ID, reads the grid overrides from
outputs/tuning/{group}/grid.json, and runs the (planners × seeds) loop
sequentially.

Outputs land at:
    {output_dir}/tuning/{group}/grid_{idx:04d}/{block}/{scenario}/{planner}/seed_{seed}.json

EB-MBD barrier tuning is conditional on the winning shared_trajopt
weights — pass them via --base-overrides-file pointing at
tuned_weights/{block}/shared_trajopt.json (produced by score_tuning.py).

Usage
-----
Count tasks:
    uv run python submit_tuning.py --group astar --count

Dry-run (write the array scripts, print sbatch commands):
    uv run python submit_tuning.py --block A --group astar --dry-run

Submit:
    uv run python submit_tuning.py --block A --group astar --submit

Submit EB-MBD (after shared_trajopt scoring is done):
    uv run python submit_tuning.py --block A --group eb_mbd_barrier \\
        --base-overrides-file tuned_weights/A/shared_trajopt.json --submit

Re-submit specific failed tasks of an existing array:
    uv run python submit_tuning.py --block A --group astar \\
        --array-spec '42,108-115' --submit
"""
from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

# ── Tuning grids (see docs/EXPERIMENT_PROTOCOL.md §5) ──────────────────────
GRID: dict[str, dict[str, list]] = {
    "astar": {
        "planning_ice_collision_cost": [0.5, 5.0, 50.0, 500.0],
        "planning_ke_cost_scale":      [1e-7, 1e-6, 1e-5, 1e-4],
        "planning_heuristic_weight":   [1.0, 2.0, 5.0, 10.0],
    },
    "shared_trajopt": {
        "planning_jax_mbd_costmap_weight": [0.1, 1.0, 10.0, 100.0],
        "planning_jax_mbd_length_weight":  [0.01, 0.1, 1.0, 10.0],
        "planning_jax_mbd_goal_x_weight":  [0.01, 0.1, 1.0, 10.0],
    },
    "eb_mbd_barrier": {
        "planning_jax_mbd_eb_c_max": [200.0, 600.0, 1200.0],
        "planning_jax_mbd_eb_mu":    [1e4, 1e5, 1e6],
        "planning_jax_mbd_eb_kappa": [0.5, 1.0, 2.0],
    },
}

PLANNERS_BY_GROUP: dict[str, list[str]] = {
    "astar":          ["astar"],
    "shared_trajopt": ["mbd", "mppi", "rss", "ilqr"],
    "eb_mbd_barrier": ["eb-mbd"],
}

TUNING_SCENARIOS: dict[str, list[str]] = {
    "A": [
        "A_conc05_thick", "A_conc05_bimodal",
        "A_conc10_thick", "A_conc10_bimodal",
        "A_conc20_thick", "A_conc20_bimodal",
        "A_conc30_thick", "A_conc30_bimodal",
        "A_conc40_thick", "A_conc40_bimodal",
        "A_conc50_thick", "A_conc50_bimodal",
    ],
    "B": [
        "B_conc20_thick", "B_conc20_bimodal",
        "B_conc30_thick", "B_conc30_bimodal",
        "B_conc40_thick", "B_conc40_bimodal",
        "B_conc50_thick", "B_conc50_bimodal",
        "B_conc60_thick", "B_conc60_bimodal",
        "B_conc70_thick", "B_conc70_bimodal",
    ],
}

TUNING_SEEDS = [0, 1]

GROUPS = list(GRID.keys())

DEFAULT_ACCOUNT = "rrg-bengioy-ad"
DEFAULT_CPUS    = 4
DEFAULT_MEM     = "16G"
DEFAULT_TIME    = "1:00:00"

# ── SLURM array template ──────────────────────────────────────────────────
SLURM_ARRAY_TEMPLATE = """\
#!/bin/bash
#SBATCH --account={account}
#SBATCH --job-name=tune-{group}-{block}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array={array_spec}
#SBATCH --output=slurm_logs/tune_{group}_{block}_%A_%a.out
#SBATCH --error=slurm_logs/tune_{group}_{block}_%A_%a.err

set -e
cd "${{SLURM_SUBMIT_DIR:-.}}"
export PATH="${{HOME}}/.local/bin:${{PATH}}"

# Add bundled nvidia CUDA libs so JAX can find them (no-op on CPU-only nodes)
NVIDIA_LIBS=$(find .venv/lib -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\\n' ':')
export LD_LIBRARY_PATH="${{NVIDIA_LIBS}}${{LD_LIBRARY_PATH}}"

# ── Array decode ──────────────────────────────────────────────────────────
SCENARIOS=({scenarios_bash})
N_SCENARIOS=${{#SCENARIOS[@]}}
N_GRID={n_grid}

TASK_ID=$SLURM_ARRAY_TASK_ID
SCENARIO_IDX=$((TASK_ID % N_SCENARIOS))
GRID_IDX=$((TASK_ID / N_SCENARIOS))
SCENARIO=${{SCENARIOS[$SCENARIO_IDX]}}
GRID_IDX_PADDED=$(printf '%04d' $GRID_IDX)

# ── Load grid overrides ───────────────────────────────────────────────────
GRID_JSON="{output_dir}/tuning/{group}/grid.json"
OVERRIDES_JSON=$(uv run python -c "
import json
g = json.load(open('$GRID_JSON'))
print(json.dumps(g[str($GRID_IDX)]))
")

OUTPUT_SUBPATH="tuning/{group}/grid_${{GRID_IDX_PADDED}}"

echo "─── Task $TASK_ID: block={block} scenario=$SCENARIO grid=$GRID_IDX ───"
echo "Overrides: $OVERRIDES_JSON"
{base_overrides_export}

# ── Run all (planner × seed) trials for this (scenario, grid) ──────────────
{trial_loop}
"""

_TRIAL_LOOP_TEMPLATE = """\
for PLANNER in {planners_bash}; do
    for SEED in {seeds_bash}; do
        OUT_JSON="{output_dir}/${{OUTPUT_SUBPATH}}/{block}/${{SCENARIO}}/${{PLANNER}}/seed_${{SEED}}.json"
        if [ -f "$OUT_JSON" ]; then
            echo "SKIP $OUT_JSON (exists)"
            continue
        fi
        echo "▸ planner=$PLANNER seed=$SEED"
        uv run python run_experiment_trial.py \\
            --block {block} \\
            --scenario "$SCENARIO" \\
            --planner "$PLANNER" \\
            --seed "$SEED" \\
            --output-dir {output_dir} \\
            --output-subpath "$OUTPUT_SUBPATH" \\
            --overrides-json "$OVERRIDES_JSON"{base_overrides_arg}
    done
done
"""


def _enumerate_grid(group: str) -> list[dict]:
    """Return a list of override dicts in deterministic grid order."""
    knobs = GRID[group]
    keys  = list(knobs.keys())
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(knobs[k] for k in keys))]


def _write_grid_index(output_root: Path, group: str, grid: list[dict]) -> Path:
    """Write the {grid_idx: overrides} map to outputs/tuning/{group}/grid.json."""
    grid_path = output_root / "tuning" / group / "grid.json"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(json.dumps(
        {idx: overrides for idx, overrides in enumerate(grid)},
        indent=2,
    ))
    return grid_path


def _compute_array_spec(
    n_scenarios: int, n_grid: int,
    scenario_filter: str | None, scenario_list: list[str],
    grid_filter: int | None,
    explicit_spec: str | None,
) -> str:
    """Return a SLURM --array spec string.

    If --array-spec is given, use it verbatim. Otherwise compute from
    --scenario / --grid-idx filters; defaults to the full range.
    """
    if explicit_spec is not None:
        return explicit_spec

    total = n_scenarios * n_grid
    if scenario_filter is None and grid_filter is None:
        return f"0-{total - 1}"

    scenario_indices = (range(n_scenarios) if scenario_filter is None
                        else [scenario_list.index(scenario_filter)])
    grid_indices = (range(n_grid) if grid_filter is None
                    else [grid_filter])

    task_ids = sorted({
        grid_idx * n_scenarios + scn_idx
        for grid_idx in grid_indices
        for scn_idx in scenario_indices
    })
    return ",".join(str(t) for t in task_ids)


def _build_array_script(
    *, group: str, block: str,
    grid: list[dict], output_dir: str,
    account: str, cpus: int, mem: str, time: str,
    array_spec: str,
    base_overrides_file: str | None,
) -> str:
    scenarios     = TUNING_SCENARIOS[block]
    scenarios_bash = " ".join(shlex.quote(s) for s in scenarios)
    planners      = PLANNERS_BY_GROUP[group]
    planners_bash = " ".join(shlex.quote(p) for p in planners)
    seeds_bash    = " ".join(str(s) for s in TUNING_SEEDS)

    base_overrides_arg = (
        f' \\\n            --overrides-file "{base_overrides_file}"'
        if base_overrides_file else ""
    )
    base_overrides_export = (
        f'echo "Base overrides: {base_overrides_file}"'
        if base_overrides_file else ""
    )

    trial_loop = _TRIAL_LOOP_TEMPLATE.format(
        planners_bash=planners_bash,
        seeds_bash=seeds_bash,
        output_dir=output_dir,
        block=block,
        base_overrides_arg=base_overrides_arg,
    )

    return SLURM_ARRAY_TEMPLATE.format(
        account=account,
        group=group,
        block=block,
        cpus=cpus,
        mem=mem,
        time=time,
        array_spec=array_spec,
        scenarios_bash=scenarios_bash,
        n_grid=len(grid),
        output_dir=output_dir,
        base_overrides_export=base_overrides_export,
        trial_loop=trial_loop,
    )


def main():
    p = argparse.ArgumentParser(
        description="Submit a SLURM job array for the planner-weight tuning grid."
    )
    p.add_argument("--block", choices=["A", "B"],
                   help="Submit only this block (default: both)")
    p.add_argument("--group", choices=GROUPS, required=True,
                   help="Tuning group: astar | shared_trajopt | eb_mbd_barrier")
    p.add_argument("--scenario",
                   help="Restrict array to one scenario ID (must be in the tuning eval set)")
    p.add_argument("--grid-idx", type=int,
                   help="Restrict array to one grid index (for re-runs / debugging)")
    p.add_argument("--array-spec", default=None,
                   help=("Verbatim SLURM --array spec, e.g. '0-100', '42,108', "
                         "'0-767%%50'. Overrides --scenario / --grid-idx filters. "
                         "Default: full range."))
    p.add_argument("--base-overrides-file", default=None,
                   help=("JSON of base overrides applied BEFORE the grid overrides. "
                         "Required for --group eb_mbd_barrier; typically "
                         "tuned_weights/{block}/shared_trajopt.json."))
    p.add_argument("--account", default=DEFAULT_ACCOUNT,
                   help=f"Alliance account (default: {DEFAULT_ACCOUNT})")
    p.add_argument("--cpus", type=int, default=DEFAULT_CPUS,
                   help=f"--cpus-per-task (default: {DEFAULT_CPUS})")
    p.add_argument("--mem", default=DEFAULT_MEM,
                   help=f"--mem (default: {DEFAULT_MEM})")
    p.add_argument("--time", default=DEFAULT_TIME,
                   help=f"--time (default: {DEFAULT_TIME})")
    p.add_argument("--output-dir", default="outputs",
                   help="Root output directory (default: outputs)")
    p.add_argument("--scripts-dir", default="slurm_scripts",
                   help="Directory to write per-array sbatch scripts")
    p.add_argument("--dry-run", action="store_true",
                   help="Write the array script(s) and print sbatch commands without submitting")
    p.add_argument("--submit", action="store_true",
                   help="Write the array script(s) and submit via sbatch")
    p.add_argument("--count", action="store_true",
                   help="Print the number of array tasks and total trials, then exit")
    args = p.parse_args()

    if not (args.dry_run or args.submit or args.count):
        p.error("Specify one of --dry-run, --submit, --count")

    if args.group == "eb_mbd_barrier" and not args.base_overrides_file and not args.count:
        p.error("--base-overrides-file is required for --group eb_mbd_barrier "
                "(should point at the tuned shared_trajopt weights for the block).")

    blocks       = [args.block] if args.block else ["A", "B"]
    output_dir   = Path(args.output_dir)
    scripts_dir  = Path(args.scripts_dir)

    grid = _enumerate_grid(args.group)
    n_grid = len(grid)
    n_scn  = len(TUNING_SCENARIOS["A"])  # 12 for both blocks
    n_trials_per_task = len(PLANNERS_BY_GROUP[args.group]) * len(TUNING_SEEDS)

    if args.count:
        total_tasks  = n_grid * n_scn * len(blocks)
        total_trials = total_tasks * n_trials_per_task
        print(f"{total_tasks} array tasks across {len(blocks)} block(s) "
              f"×  {n_trials_per_task} trials/task  =  {total_trials} trials")
        return

    scripts_dir.mkdir(parents=True, exist_ok=True)
    Path("slurm_logs").mkdir(exist_ok=True)
    _write_grid_index(output_dir, args.group, grid)

    for block in blocks:
        array_spec = _compute_array_spec(
            n_scenarios=n_scn, n_grid=n_grid,
            scenario_filter=args.scenario, scenario_list=TUNING_SCENARIOS[block],
            grid_filter=args.grid_idx,
            explicit_spec=args.array_spec,
        )

        script = _build_array_script(
            group=args.group, block=block,
            grid=grid, output_dir=str(output_dir),
            account=args.account, cpus=args.cpus, mem=args.mem, time=args.time,
            array_spec=array_spec,
            base_overrides_file=args.base_overrides_file,
        )

        script_path = scripts_dir / f"tune_array_{args.group}_{block}.sh"
        script_path.write_text(script)

        # Report sizing
        n_tasks = (array_spec.count(",") + array_spec.count("-")
                   + (1 if "," not in array_spec and "-" not in array_spec else 0))
        # Quick task-count for ranges: just parse it
        try:
            n_tasks = _count_array_spec(array_spec)
        except ValueError:
            n_tasks = "?"
        print(f"[{block}/{args.group}] array_spec={array_spec}  "
              f"({n_tasks} tasks × {n_trials_per_task} trials)  → {script_path}")

        if args.dry_run:
            print(f"  sbatch {script_path}")
        elif args.submit:
            ret = subprocess.run(["sbatch", str(script_path)],
                                 capture_output=True, text=True)
            if ret.returncode != 0:
                print(f"  ERROR submitting {script_path}: {ret.stderr.strip()}",
                      file=sys.stderr)
            else:
                print(f"  {ret.stdout.strip()}")

    print("\nDone.", file=sys.stderr)


def _count_array_spec(spec: str) -> int:
    """Count the number of tasks implied by a SLURM array spec."""
    # Strip any %N concurrency suffix
    spec = spec.split("%")[0]
    n = 0
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            # Handle stride a-b:s
            if ":" in hi:
                hi, stride = hi.split(":")
                n += (int(hi) - int(lo)) // int(stride) + 1
            else:
                n += int(hi) - int(lo) + 1
        else:
            n += 1
    return n


if __name__ == "__main__":
    main()
