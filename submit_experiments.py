#!/usr/bin/env python3
"""
Generate SLURM sbatch submission commands for the full experiment suite.

Usage (generate and submit all jobs):
    uv run python submit_experiments.py --submit

Usage (dry-run: print commands only):
    uv run python submit_experiments.py --dry-run

Usage (single block):
    uv run python submit_experiments.py --block A --dry-run

Total jobs: 3 255
    Block A: 36 scenarios × 7 planners × 5 seeds = 1 260
    Block B: 21 scenarios × 7 planners × 5 seeds =   735
    Block C: 36 scenarios × 7 planners × 5 seeds = 1 260  (A grid, rigid floes)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ── Experiment grid ────────────────────────────────────────────────────────
PLANNERS = ["astar", "mbd", "eb-mbd", "astar-eb-mbd", "mppi", "rss", "ilqr"]
SEEDS    = [0, 1, 2, 3, 4]

# ── SLURM template ─────────────────────────────────────────────────────────
SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=icenav-{block}-{scenario}-{planner}-s{seed}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/icenav_{block}_{scenario}_{planner}_s{seed}_%j.out
#SBATCH --error=slurm_logs/icenav_{block}_{scenario}_{planner}_s{seed}_%j.err

set -e
cd "${{SLURM_SUBMIT_DIR:-.}}"
export PATH="${{HOME}}/.local/bin:${{PATH}}"

# Add bundled nvidia CUDA libs so JAX can find them (CPU-only for now, no-op on CPU nodes)
NVIDIA_LIBS=$(find .venv/lib -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\\n' ':')
export LD_LIBRARY_PATH="${{NVIDIA_LIBS}}${{LD_LIBRARY_PATH}}"

uv run python run_experiment_trial.py \\
    --block {block} \\
    --scenario {scenario} \\
    --planner {planner} \\
    --seed {seed} \\
    --output-dir {output_dir}
"""


def _iter_trials(blocks: list[str]):
    """Yield (block, scenario_id, planner, seed) for all trials in the given blocks."""
    # Import here so the script can be run without the full project installed
    # (e.g., just to inspect the job list).
    try:
        from medea_ice_ship.synthetic_map import list_scenarios
    except ImportError:
        print("ERROR: medea_ice_ship not importable. Run from the project root "
              "with `uv run python submit_experiments.py`.", file=sys.stderr)
        sys.exit(1)

    for block in blocks:
        for spec in list_scenarios(block):
            for planner in PLANNERS:
                for seed in SEEDS:
                    yield block, spec.scenario_id, planner, seed


def _write_script(block, scenario, planner, seed, scripts_dir: Path,
                  output_dir: str) -> Path:
    content = SLURM_TEMPLATE.format(
        block=block,
        scenario=scenario,
        planner=planner,
        seed=seed,
        output_dir=output_dir,
    )
    path = scripts_dir / f"{block}_{scenario}_{planner}_s{seed}.sh"
    path.write_text(content)
    return path


def main():
    p = argparse.ArgumentParser(
        description="Submit SLURM jobs for the ice-navigation experiment suite."
    )
    p.add_argument("--block", choices=["A", "B", "C"],
                   help="Submit only this block (default: A, B, and C)")
    p.add_argument("--planner", choices=PLANNERS,
                   help="Submit only this planner")
    p.add_argument("--scenario",
                   help="Submit only this scenario ID")
    p.add_argument("--seed", type=int, choices=SEEDS,
                   help="Submit only this seed")
    p.add_argument("--output-dir", default="outputs/experiments",
                   help="Root directory for JSON results (passed to trial runner)")
    p.add_argument("--scripts-dir", default="slurm_scripts",
                   help="Directory to write per-job sbatch scripts")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sbatch commands without executing them")
    p.add_argument("--submit", action="store_true",
                   help="Actually submit all jobs via sbatch")
    p.add_argument("--count", action="store_true",
                   help="Just print the number of jobs that would be submitted")
    args = p.parse_args()

    if not args.dry_run and not args.submit and not args.count:
        p.error("Specify --dry-run, --submit, or --count")

    blocks       = [args.block] if args.block else ["A", "B", "C"]
    scripts_dir  = Path(args.scripts_dir)
    output_dir   = args.output_dir

    if not args.count:
        scripts_dir.mkdir(parents=True, exist_ok=True)
        Path("slurm_logs").mkdir(exist_ok=True)

    trials = list(_iter_trials(blocks))

    # Apply optional filters
    if args.planner:
        trials = [(b, s, pl, sd) for b, s, pl, sd in trials if pl == args.planner]
    if args.scenario:
        trials = [(b, s, pl, sd) for b, s, pl, sd in trials if s == args.scenario]
    if args.seed is not None:
        trials = [(b, s, pl, sd) for b, s, pl, sd in trials if sd == args.seed]

    if args.count:
        print(f"{len(trials)} jobs")
        return

    n_submitted = 0
    n_skipped   = 0
    for block, scenario, planner, seed in trials:
        # Skip if result already exists
        result_path = (Path(output_dir) / block / scenario / planner
                       / f"seed_{seed}.json")
        if result_path.exists():
            n_skipped += 1
            continue

        script_path = _write_script(
            block, scenario, planner, seed, scripts_dir, output_dir
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

    print(f"\nDone.  {n_submitted} jobs {'submitted' if args.submit else 'listed'}, "
          f"{n_skipped} already complete (skipped).", file=sys.stderr)


if __name__ == "__main__":
    main()
