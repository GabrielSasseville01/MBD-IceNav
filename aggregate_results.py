#!/usr/bin/env python3
"""
Aggregate per-trial JSON results into summary tables and CSV files.

Usage:
    uv run python aggregate_results.py --output-dir outputs/experiments
    uv run python aggregate_results.py --output-dir outputs/experiments --block A
    uv run python aggregate_results.py --output-dir outputs/experiments --block B --csv

Output files (written next to --output-dir):
    results_blockA.csv   — Block A full factorial table (mean ± std across seeds)
    results_blockB.csv   — Block B concentration-trend table
    results_summary.txt  — human-readable tables for both blocks

Metrics per (scenario, planner):
    f_max_mean, f_max_std    — peak ice-hull force [N]
    f_mean_mean, f_mean_std  — mean ice-hull force [N]
    f_95_mean,  f_95_std     — 95th-pct ice-hull force [N]
    path_length_mean         — total travel distance [m]
    success_rate             — fraction of seeds that reached the goal
    plan_time_mean           — mean planning time per replan [s]
    n_seeds                  — number of completed seeds
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

PLANNERS   = ["astar", "mbd", "eb-mbd", "mppi", "rss", "ilqr"]
METRICS    = ["f_max", "f_mean", "f_95", "path_length", "success", "simulated_time"]


# ── JSON loader ────────────────────────────────────────────────────────────

def load_results(output_dir: Path, block: str | None = None) -> List[dict]:
    """Load all seed_*.json files under output_dir, optionally filtered by block."""
    results = []
    pattern = "*/*/seed_*.json" if block is None else f"{block}/*/*/seed_*.json"
    # Walk: output_dir / block / scenario / planner / seed_N.json
    search_root = output_dir
    for path in sorted(search_root.rglob("seed_*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            if block is not None and d.get("block") != block:
                continue
            # Compute per-replan mean planning time
            pt = d.get("planning_times", [])
            d["plan_time_mean_per_replan"] = float(np.mean(pt)) if pt else 0.0
            results.append(d)
        except Exception as e:
            print(f"WARNING: could not load {path}: {e}", file=sys.stderr)
    return results


# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate(results: List[dict]) -> Dict[tuple, dict]:
    """
    Group results by (block, scenario_id, planner) and compute
    mean ± std across seeds.

    Returns dict keyed by (block, scenario_id, planner) → stats dict.
    """
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in results:
        key = (r["block"], r["scenario_id"], r["planner"])
        groups[key].append(r)

    stats = {}
    for key, trials in groups.items():
        s: dict = {
            "block":          key[0],
            "scenario_id":    key[1],
            "planner":        key[2],
            "concentration":  trials[0]["concentration"],
            "thickness_mode": trials[0]["thickness_mode"],
            "with_currents":  trials[0]["with_currents"],
            "n_seeds":        len(trials),
        }

        for metric in METRICS:
            vals = [t[metric] for t in trials if metric in t]
            if not vals:
                s[f"{metric}_mean"] = float("nan")
                s[f"{metric}_std"]  = float("nan")
            elif metric == "success":
                s["success_rate"] = float(np.mean(vals))
            else:
                s[f"{metric}_mean"] = float(np.mean(vals))
                s[f"{metric}_std"]  = float(np.std(vals, ddof=0))

        pt_vals = [t["plan_time_mean_per_replan"] for t in trials]
        s["plan_time_mean"] = float(np.mean(pt_vals)) if pt_vals else float("nan")
        s["plan_time_std"]  = float(np.std(pt_vals,  ddof=0)) if pt_vals else float("nan")

        stats[key] = s

    return stats


# ── CSV writer ─────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "block", "scenario_id", "concentration", "thickness_mode", "with_currents",
    "planner", "n_seeds",
    "success_rate",
    "f_max_mean",  "f_max_std",
    "f_mean_mean", "f_mean_std",
    "f_95_mean",   "f_95_std",
    "path_length_mean", "path_length_std",
    "simulated_time_mean", "simulated_time_std",
    "plan_time_mean", "plan_time_std",
]


def write_csv(stats: Dict[tuple, dict], path: Path):
    rows = sorted(stats.values(),
                  key=lambda s: (s["scenario_id"], PLANNERS.index(s["planner"])
                                 if s["planner"] in PLANNERS else 99))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {path}  ({len(rows)} rows)")


# ── Human-readable table ───────────────────────────────────────────────────

def _fmt(val, std=None, digits=0):
    if val != val:   # nan
        return "   —"
    fmt = f"{{:.{digits}f}}"
    s = fmt.format(val)
    if std is not None and std == std:
        s += f"±{fmt.format(std)}"
    return s


def print_table(stats: Dict[tuple, dict], block: str, file=sys.stdout):
    rows = [s for s in stats.values() if s["block"] == block]
    if not rows:
        return

    scenarios = sorted({r["scenario_id"] for r in rows})
    planners  = [p for p in PLANNERS if any(r["planner"] == p for r in rows)]

    print(f"\n{'='*120}", file=file)
    print(f"Block {block} — F_max [N] (mean±std)  /  success rate", file=file)
    print(f"{'='*120}", file=file)

    hdr = f"{'Scenario':<30}"
    for pl in planners:
        hdr += f"  {pl:>18}"
    print(hdr, file=file)
    print("-" * len(hdr), file=file)

    for sc in scenarios:
        line = f"{sc:<30}"
        for pl in planners:
            key = (block, sc, pl)
            if key in stats:
                s = stats[key]
                fmax = _fmt(s.get("f_max_mean", float("nan")),
                            s.get("f_max_std",  float("nan")))
                succ = f"{s.get('success_rate', float('nan')):.0%}"
                cell = f"{fmax} ({succ})"
            else:
                cell = "—"
            line += f"  {cell:>18}"
        print(line, file=file)

    # Block B: also print concentration-vs-success summary
    if block == "B":
        print(f"\nBlock B — success rate vs concentration", file=file)
        print("-" * 80, file=file)
        concs = sorted({r["concentration"] for r in rows})
        hdr2 = f"{'Conc':>6}  {'Thickness':>12}"
        for pl in planners:
            hdr2 += f"  {pl:>8}"
        print(hdr2, file=file)
        print("-" * len(hdr2), file=file)
        for conc in concs:
            for tm in ["uniform_thin", "uniform_thick", "bimodal"]:
                sc_rows = [r for r in rows
                           if abs(r["concentration"] - conc) < 1e-6
                           and r["thickness_mode"] == tm]
                if not sc_rows:
                    continue
                line2 = f"{conc:>6.0%}  {tm:>12}"
                for pl in planners:
                    pr = next((r for r in sc_rows if r["planner"] == pl), None)
                    if pr:
                        line2 += f"  {pr.get('success_rate', float('nan')):>8.0%}"
                    else:
                        line2 += f"  {'—':>8}"
                print(line2, file=file)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Aggregate experiment JSON results into summary tables."
    )
    p.add_argument("--output-dir", default="outputs/experiments",
                   help="Root directory containing block/scenario/planner/seed_N.json files")
    p.add_argument("--block", choices=["A", "B"],
                   help="Aggregate only this block (default: both)")
    p.add_argument("--csv", action="store_true",
                   help="Write CSV files in addition to printing tables")
    p.add_argument("--save-txt", metavar="FILE",
                   help="Save human-readable summary to a text file")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: {output_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    blocks = [args.block] if args.block else ["A", "B"]

    print(f"Loading results from {output_dir} ...", flush=True)
    results = load_results(output_dir, block=args.block)
    print(f"  {len(results)} trial files loaded.")

    if not results:
        print("No results found. Have any trials completed?")
        return

    stats = aggregate(results)

    txt_lines = []
    for block in blocks:
        import io
        buf = io.StringIO()
        print_table(stats, block, file=buf)
        txt = buf.getvalue()
        print(txt)
        txt_lines.append(txt)

    if args.save_txt:
        Path(args.save_txt).write_text("\n".join(txt_lines))
        print(f"Summary saved → {args.save_txt}")

    if args.csv:
        for block in blocks:
            block_stats = {k: v for k, v in stats.items() if v["block"] == block}
            csv_path = output_dir.parent / f"results_block{block}.csv"
            write_csv(block_stats, csv_path)


if __name__ == "__main__":
    main()
