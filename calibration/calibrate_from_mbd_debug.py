#!/usr/bin/env python3
"""
Calibration from MBD debug output: compute balanced weights directly from a
``--mbd-viz-dir`` run.

The ``replan_*_costs.txt`` files produced by ``--mbd-viz-dir`` contain the raw
(unweighted) cost values for every sample trajectory at every replanning step.
This script reads those files and applies the same two-stage normalisation as
``analyze_cost_stats.py`` to produce weights calibrated on real run data rather
than the static-costmap probes from ``collect_cost_stats.py``.

Usage
-----
    uv run python calibration/calibrate_from_mbd_debug.py outputs/mbd_debug3
    uv run python calibration/calibrate_from_mbd_debug.py outputs/mbd_debug3 \\
        --rank all --no-plots
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Domain-importance multipliers — same as analyze_cost_stats.py
# ---------------------------------------------------------------------------
IMPORTANCE = {
    "costmap":    1.0,
    "length":     1.0,
    "goal":       1.0,
    "smoothness": 0.125,
}

CONFIG_KEYS = {
    "costmap":    "planning_jax_mbd_costmap_weight",
    "length":     "planning_jax_mbd_length_weight",
    "goal":       "planning_jax_mbd_goal_weight",
    "smoothness": "planning_jax_mbd_smoothness_weight",
}

# regex patterns for the costs.txt format
_RE_RANK   = re.compile(r"^Rank #(\d+)\s+total_cost\s*=\s*([\d.]+)")
_RE_TERM   = re.compile(r"^\s+(costmap|length|goal|jerk)\s+weighted=.*?raw=([\d.]+)")


def _parse_costs_file(path: Path) -> list[dict]:
    """
    Parse one replan_*_costs.txt file.

    Returns a list of dicts, one per rank, with keys:
        rank, total_cost, costmap, length, goal, smoothness
    """
    records = []
    current: dict | None = None
    for line in path.read_text().splitlines():
        m = _RE_RANK.match(line)
        if m:
            if current is not None:
                records.append(current)
            current = {"rank": int(m.group(1)), "total_cost": float(m.group(2))}
            continue
        if current is None:
            continue
        m = _RE_TERM.match(line)
        if m:
            term, raw = m.group(1), float(m.group(2))
            key = "smoothness" if term == "jerk" else term
            current[key] = raw
    if current is not None:
        records.append(current)
    return records


def _stats(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "n":      len(arr),
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min":    float(np.min(arr)),
        "median": float(np.median(arr)),
        "max":    float(np.max(arr)),
    }


def calibrate(debug_dir: Path, rank: str = "1", plot: bool = True) -> dict:
    """
    Read all costs.txt files in debug_dir, compute balanced weights.

    Parameters
    ----------
    rank : "1" to use only rank-1 trajectories, "all" to use all top-N ranks.
    """
    txt_files = sorted(debug_dir.glob("replan_*_costs.txt"))
    if not txt_files:
        print(f"ERROR: no replan_*_costs.txt files found in {debug_dir}", file=sys.stderr)
        sys.exit(1)

    use_rank = None if rank == "all" else int(rank)

    raw: dict[str, list[float]] = {k: [] for k in ("costmap", "length", "goal", "smoothness")}
    n_files = 0
    for fpath in txt_files:
        records = _parse_costs_file(fpath)
        for rec in records:
            if use_rank is not None and rec["rank"] != use_rank:
                continue
            for key in raw:
                if key in rec:
                    raw[key].append(rec[key])
        n_files += 1

    n_obs = len(raw["length"])
    print("=" * 70)
    print("MBD Weight Calibration from Debug Run")
    print(f"  Source:  {debug_dir}")
    print(f"  Files:   {n_files}  |  Observations (rank={rank}): {n_obs}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Summary statistics
    # ------------------------------------------------------------------
    stats = {k: _stats(raw[k]) for k in raw}

    TERM_LABELS = {
        "costmap":    "Costmap cost",
        "length":     "Path length [m]",
        "goal":       "Goal distance [m]",
        "smoothness": "Smoothness (jerk)",
    }
    print("\n--- Raw (unweighted) cost component statistics ---\n")
    hdr = f"  {'Term':<18} {'N':>5}  {'Mean':>12}  {'Std':>12}  {'Min':>12}  {'Median':>12}  {'Max':>12}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in raw:
        s = stats[k]
        print(f"  {TERM_LABELS[k]:<18} {s['n']:>5}  {s['mean']:>12.4f}  {s['std']:>12.4f}  "
              f"{s['min']:>12.4f}  {s['median']:>12.4f}  {s['max']:>12.4f}")

    # ------------------------------------------------------------------
    # 2. Normalisation weights
    # ------------------------------------------------------------------
    reference = stats["length"]["mean"]
    norm_weights = {k: reference / stats[k]["mean"] for k in raw}

    print(f"\n--- Normalisation weights (all terms → mean path length = {reference:.2f} m) ---\n")
    for k, w in norm_weights.items():
        print(f"  w_{k:<12} = {w:>12.5f}   (mean contribution: {w * stats[k]['mean']:.2f})")

    # ------------------------------------------------------------------
    # 3. Final weights = normalised × importance
    # ------------------------------------------------------------------
    final = {k: norm_weights[k] * IMPORTANCE[k] for k in raw}
    total_contrib = sum(final[k] * stats[k]["mean"] for k in raw)

    print(f"\n--- Recommended weights (normalised × importance {IMPORTANCE}) ---\n")
    for k, w in final.items():
        contrib = w * stats[k]["mean"]
        print(f"  w_{k:<12} = {w:>12.5f}   (mean contribution: {contrib:.2f},  {contrib/total_contrib*100:.1f}%)")
    print(f"\n  Total expected weighted cost (mean): {total_contrib:.2f}")

    # ------------------------------------------------------------------
    # 4. Recommended costmap_normalizer
    # ------------------------------------------------------------------
    # The normalizer is the ratio mean(costmap) / mean(length).
    # With costmap_normalizer set to this value, a costmap_weight of 1.0
    # gives the costmap term the same mean magnitude as the path-length term,
    # and the importance multiplier (IMPORTANCE["costmap"]) sets relative priority.
    normalizer = stats["costmap"]["mean"] / max(stats["length"]["mean"], 1e-12)
    print(f"\n--- Recommended costmap_normalizer ---\n")
    print(f"  mean(costmap_raw) / mean(path_length) = "
          f"{stats['costmap']['mean']:.2f} / {stats['length']['mean']:.2f} = {normalizer:.4g}")
    print(f"\n  With planning_jax_mbd_costmap_normalizer = {normalizer:.4g}:")
    print(f"    costmap_weight=1.0  → same mean magnitude as path-length")
    print(f"    costmap_weight=3.0  → 3× more emphasis on ice avoidance")
    print(f"  (Use importance multipliers above instead of very large/small weights.)")

    # ------------------------------------------------------------------
    # 5. Config snippet
    # ------------------------------------------------------------------
    print("\n--- Ready-to-use config snippet (medea_ice_ship/config.py) ---\n")
    mag_n = int(np.floor(np.log10(abs(normalizer)))) if normalizer > 0 else 0
    rounded_n = round(normalizer, max(0, 4 - 1 - mag_n))
    print(f"    planning_jax_mbd_costmap_normalizer = {rounded_n}")
    for k, cfg_key in CONFIG_KEYS.items():
        if k == "costmap":
            # costmap_weight is the semantic importance multiplier;
            # the normalizer above already handles magnitude scaling.
            # Outputting final["costmap"] here would double-divide by the normalizer.
            print(f"    {cfg_key} = {IMPORTANCE[k]}")
        else:
            w = final[k]
            mag = int(np.floor(np.log10(abs(w)))) if w != 0 else 0
            rounded = round(w, max(0, 4 - 1 - mag))
            print(f"    {cfg_key} = {rounded}")

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    if not plot:
        return final

    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("\nmatplotlib not available — skipping plots.", file=sys.stderr)
        return final

    colors = {"costmap": "#d62728", "length": "#1f77b4", "goal": "#2ca02c", "smoothness": "#9467bd"}
    replan_indices = list(range(n_files))

    # Collect per-file rank-1 raw values for time series
    ts: dict[str, list[float]] = {k: [] for k in raw}
    for fpath in txt_files:
        records = _parse_costs_file(fpath)
        r1 = next((r for r in records if r["rank"] == 1), None)
        if r1:
            for k in ts:
                ts[k].append(r1.get(k, float("nan")))

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"MBD Cost Component Statistics (from debug run)\n"
        f"{n_files} replans · rank={rank} · {debug_dir.name}",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    # Row 0: box plots
    for col_idx, k in enumerate(raw):
        ax = fig.add_subplot(gs[0, col_idx])
        bp = ax.boxplot(raw[k], patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor(colors[k])
        bp["boxes"][0].set_alpha(0.6)
        ax.set_title(TERM_LABELS[k], fontsize=8)
        ax.set_ylabel("Raw value", fontsize=7)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        s = stats[k]
        ax.annotate(f"μ={s['mean']:.3g}\nσ={s['std']:.3g}",
                    xy=(0.97, 0.96), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Row 1 left: rank-1 time series (min-max normalised for overlay)
    ax_ts = fig.add_subplot(gs[1, :2])
    for k in ts:
        vals = np.array(ts[k])
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        denom = (vmax - vmin) if vmax > vmin else 1.0
        ax_ts.plot(replan_indices, (vals - vmin) / denom,
                   marker="o", markersize=3, label=k, color=colors[k])
    ax_ts.set_xlabel("Replan index", fontsize=8)
    ax_ts.set_ylabel("Normalised raw value [0–1]", fontsize=8)
    ax_ts.set_title("Raw components along run (rank-1, min-max norm.)", fontsize=8)
    ax_ts.legend(fontsize=7)
    ax_ts.grid(True, alpha=0.3)

    # Row 1 right: weighted contribution bar + pie
    ax_bar = fig.add_subplot(gs[1, 2])
    weighted_means = {k: final[k] * stats[k]["mean"] for k in raw}
    vals_list = [weighted_means[k] for k in raw]
    bars = ax_bar.bar(list(raw.keys()), vals_list,
                      color=[colors[k] for k in raw], alpha=0.7, edgecolor="black", linewidth=0.5)
    ax_bar.set_title("Mean weighted contribution\n(recommended weights)", fontsize=8)
    ax_bar.set_ylabel("Weighted cost", fontsize=8)
    for bar, val in zip(bars, vals_list):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax_bar.tick_params(axis="x", labelsize=7)

    ax_pie = fig.add_subplot(gs[1, 3])
    fracs = [weighted_means[k] / total_contrib for k in raw]
    ax_pie.pie(fracs,
               labels=[f"{k}\n{f*100:.1f}%" for k, f in zip(raw.keys(), fracs)],
               colors=[colors[k] for k in raw],
               startangle=90,
               wedgeprops={"alpha": 0.7, "edgecolor": "black", "linewidth": 0.5})
    ax_pie.set_title("Fractional contribution", fontsize=8)

    plot_path = debug_dir / "weight_calibration.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")
    plt.show()

    return final


def main():
    parser = argparse.ArgumentParser(
        description="Compute MBD weights from --mbd-viz-dir debug output."
    )
    parser.add_argument("debug_dir", type=Path,
                        help="Directory containing replan_*_costs.txt files.")
    parser.add_argument("--rank", default="1",
                        help="Which rank to use: '1' (default) or 'all'.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not args.debug_dir.exists():
        print(f"ERROR: {args.debug_dir} not found.", file=sys.stderr)
        sys.exit(1)

    calibrate(args.debug_dir, rank=args.rank, plot=not args.no_plots)


if __name__ == "__main__":
    main()
