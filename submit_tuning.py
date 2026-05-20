#!/usr/bin/env python3
"""
SLURM submitter for the planner-weight tuning grid sweep.

See docs/EXPERIMENT_PROTOCOL.md for the full methodology.

Three tuning groups per block, run sequentially:

    astar            64 grid pts × 12 scn × 1 planner × 2 seeds → 1 536 trials
    shared_trajopt   64 grid pts × 12 scn × 4 planners × 2 seeds → 6 144 trials
    eb_mbd_barrier   27 grid pts × 12 scn × 1 planner × 2 seeds →   648 trials

SLURM granularity: one job per (block, group, grid_idx, scenario). Inside
each job, planners × seeds run sequentially. With this bundling the per-
block job count is:

    astar          → 768
    shared_trajopt → 768  (each job: 8 trials)
    eb_mbd_barrier → 324

i.e. ~1 860 jobs per block, ~3 720 across both. Tunable via --bundle.

Outputs land at:
    {output_dir}/tuning/{group}/grid_{idx}/{block}/{scenario}/{planner}/seed_{seed}.json

The grid index → overrides mapping is written to
    {output_dir}/tuning/{group}/grid.json
once per group/block invocation, so the per-trial JSONs can be re-keyed
during scoring.

EB-MBD barrier tuning is conditional on the winning shared_trajopt
weights — pass them via --base-overrides-file pointing at
tuned_weights/{block}/shared_trajopt.json (produced by score_tuning.py).

Usage examples
--------------
Dry-run for A* group on block A:
    uv run python submit_tuning.py --block A --group astar --dry-run

Submit shared_trajopt for both blocks:
    uv run python submit_tuning.py --group shared_trajopt --submit

Submit EB-MBD barrier (after shared_trajopt scoring is done):
    uv run python submit_tuning.py --block A --group eb_mbd_barrier \\
        --base-overrides-file tuned_weights/A/shared_trajopt.json --submit

Count jobs without writing scripts:
    uv run python submit_tuning.py --group shared_trajopt --count
"""
from __future__ import annotations

import argparse
import itertools
import json
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

# ── Planners scored by each group ─────────────────────────────────────────
PLANNERS_BY_GROUP: dict[str, list[str]] = {
    "astar":          ["astar"],
    "shared_trajopt": ["mbd", "mppi", "rss", "ilqr"],
    "eb_mbd_barrier": ["eb-mbd"],
}

# ── Tuning eval set (12 stratified scenarios per block) ───────────────────
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

# ── SLURM template ─────────────────────────────────────────────────────────
SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=tune-{group}-{block}-{scenario}-g{grid_idx:04d}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/tune_{group}_{block}_{scenario}_g{grid_idx:04d}_%j.out
#SBATCH --error=slurm_logs/tune_{group}_{block}_{scenario}_g{grid_idx:04d}_%j.err

set -e
cd "${{SLURM_SUBMIT_DIR:-.}}"
export PATH="${{HOME}}/.local/bin:${{PATH}}"

# Add bundled nvidia CUDA libs so JAX can find them (no-op on CPU nodes)
NVIDIA_LIBS=$(find .venv/lib -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\\n' ':')
export LD_LIBRARY_PATH="${{NVIDIA_LIBS}}${{LD_LIBRARY_PATH}}"

# Per-grid-point overrides written by submit_tuning.py
OVERRIDES_JSON='{overrides_json}'
{base_overrides_export}

{trial_lines}
"""

_TRIAL_LINE_TEMPLATE = (
    'uv run python run_experiment_trial.py'
    ' --block {block}'
    ' --scenario {scenario}'
    ' --planner {planner}'
    ' --seed {seed}'
    ' --output-dir {output_dir}'
    ' --output-subpath {output_subpath}'
    ' --overrides-json "$OVERRIDES_JSON"'
    '{base_overrides_arg}'
)


def _enumerate_grid(group: str) -> list[dict]:
    """Return a list of override dicts in deterministic grid order."""
    knobs = GRID[group]
    keys  = list(knobs.keys())
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(knobs[k] for k in keys))]


def _write_grid_index(output_root: Path, group: str, grid: list[dict]) -> Path:
    """Write the {grid_idx: overrides} map to {tuning_dir}/{group}/grid.json."""
    grid_path = output_root / "tuning" / group / "grid.json"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(json.dumps(
        {idx: overrides for idx, overrides in enumerate(grid)},
        indent=2,
    ))
    return grid_path


def _write_slurm_script(
    *, scripts_dir: Path,
    group: str, block: str, scenario: str, grid_idx: int,
    overrides: dict, output_dir: str,
    base_overrides_file: str | None,
) -> Path:
    output_subpath = f"tuning/{group}/grid_{grid_idx:04d}"

    base_overrides_arg = (
        f' --overrides-file "{base_overrides_file}"' if base_overrides_file else ""
    )
    base_overrides_export = (
        f'# Conditional on tuned weights from previous group:\n'
        f'BASE_OVERRIDES_FILE="{base_overrides_file}"'
        if base_overrides_file else ""
    )

    trial_lines = []
    for planner in PLANNERS_BY_GROUP[group]:
        for seed in TUNING_SEEDS:
            trial_lines.append(_TRIAL_LINE_TEMPLATE.format(
                block=block,
                scenario=scenario,
                planner=planner,
                seed=seed,
                output_dir=output_dir,
                output_subpath=output_subpath,
                base_overrides_arg=base_overrides_arg,
            ))
    trial_lines_str = "\n".join(trial_lines)

    # JSON must survive shell single-quoting: escape any embedded single quotes.
    overrides_json = json.dumps(overrides).replace("'", "'\"'\"'")

    content = SLURM_TEMPLATE.format(
        group=group,
        block=block,
        scenario=scenario,
        grid_idx=grid_idx,
        overrides_json=overrides_json,
        base_overrides_export=base_overrides_export,
        trial_lines=trial_lines_str,
    )

    path = scripts_dir / f"tune_{group}_{block}_{scenario}_g{grid_idx:04d}.sh"
    path.write_text(content)
    return path


def _all_results_exist(output_dir: Path, group: str, grid_idx: int,
                       block: str, scenario: str) -> bool:
    """True iff every (planner × seed) JSON for this (group, grid, scn) exists."""
    base = (output_dir / "tuning" / group / f"grid_{grid_idx:04d}"
            / block / scenario)
    for planner in PLANNERS_BY_GROUP[group]:
        for seed in TUNING_SEEDS:
            if not (base / planner / f"seed_{seed}.json").exists():
                return False
    return True


def main():
    p = argparse.ArgumentParser(
        description="Submit SLURM jobs for the planner-weight tuning grid."
    )
    p.add_argument("--block", choices=["A", "B"],
                   help="Submit only this block (default: both)")
    p.add_argument("--group", choices=GROUPS, required=True,
                   help="Tuning group: astar | shared_trajopt | eb_mbd_barrier")
    p.add_argument("--scenario",
                   help="Submit only this scenario ID (must be in the tuning eval set)")
    p.add_argument("--grid-idx", type=int,
                   help="Submit only this grid index (for re-runs / debugging)")
    p.add_argument("--base-overrides-file", default=None,
                   help=("Path to a JSON of base overrides applied BEFORE the grid "
                         "overrides. Required for --group eb_mbd_barrier so the "
                         "previously tuned shared_trajopt weights are reused; "
                         "typically tuned_weights/{block}/shared_trajopt.json."))
    p.add_argument("--output-dir", default="outputs",
                   help="Root output directory (default: outputs)")
    p.add_argument("--scripts-dir", default="slurm_scripts",
                   help="Directory to write per-job sbatch scripts")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sbatch commands without executing")
    p.add_argument("--submit", action="store_true",
                   help="Actually submit all jobs via sbatch")
    p.add_argument("--count", action="store_true",
                   help="Print the number of jobs that would be submitted and exit")
    args = p.parse_args()

    if not (args.dry_run or args.submit or args.count):
        p.error("Specify one of --dry-run, --submit, --count")

    if args.group == "eb_mbd_barrier" and not args.base_overrides_file:
        p.error("--base-overrides-file is required for --group eb_mbd_barrier "
                "(should point at the tuned shared_trajopt weights for the block).")

    blocks       = [args.block] if args.block else ["A", "B"]
    output_dir   = Path(args.output_dir)
    scripts_dir  = Path(args.scripts_dir)

    if not args.count:
        scripts_dir.mkdir(parents=True, exist_ok=True)
        Path("slurm_logs").mkdir(exist_ok=True)

    grid = _enumerate_grid(args.group)
    if not args.count:
        grid_path = _write_grid_index(output_dir, args.group, grid)
        print(f"Grid index ({len(grid)} points) → {grid_path}", file=sys.stderr)

    # Enumerate jobs
    jobs = []
    for block in blocks:
        scenarios = TUNING_SCENARIOS[block]
        if args.scenario:
            if args.scenario not in scenarios:
                p.error(f"--scenario {args.scenario!r} is not in the tuning eval set "
                        f"for block {block}: {scenarios}")
            scenarios = [args.scenario]
        for scenario in scenarios:
            for grid_idx, overrides in enumerate(grid):
                if args.grid_idx is not None and grid_idx != args.grid_idx:
                    continue
                jobs.append((block, scenario, grid_idx, overrides))

    if args.count:
        n_trials_per_job = len(PLANNERS_BY_GROUP[args.group]) * len(TUNING_SEEDS)
        print(f"{len(jobs)} jobs  ×  {n_trials_per_job} trials/job  =  "
              f"{len(jobs) * n_trials_per_job} trials")
        return

    # Submit
    n_submitted = 0
    n_skipped   = 0
    for block, scenario, grid_idx, overrides in jobs:
        if _all_results_exist(output_dir, args.group, grid_idx, block, scenario):
            n_skipped += 1
            continue

        script_path = _write_slurm_script(
            scripts_dir=scripts_dir,
            group=args.group, block=block, scenario=scenario,
            grid_idx=grid_idx, overrides=overrides,
            output_dir=str(output_dir),
            base_overrides_file=args.base_overrides_file,
        )

        if args.dry_run:
            print(f"sbatch {script_path}")
        elif args.submit:
            ret = subprocess.run(["sbatch", str(script_path)],
                                 capture_output=True, text=True)
            if ret.returncode != 0:
                print(f"ERROR submitting {script_path}: {ret.stderr.strip()}",
                      file=sys.stderr)
            else:
                print(ret.stdout.strip())
        n_submitted += 1

    print(f"\nDone. {n_submitted} jobs {'submitted' if args.submit else 'listed'}, "
          f"{n_skipped} already complete (skipped).", file=sys.stderr)


if __name__ == "__main__":
    main()
