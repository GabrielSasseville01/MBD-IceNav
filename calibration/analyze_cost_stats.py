#!/usr/bin/env python3
"""
Calibration step 2: analyze raw cost statistics and compute balanced weights.

Reads the CSV produced by ``collect_cost_stats.py`` and:
  1. Prints a summary statistics table for each cost term.
  2. Computes normalized weights such that every term contributes equally
     on average (before domain-importance multipliers are applied).
  3. Applies suggested domain-importance multipliers and outputs a ready-to-use
     config snippet.
  4. Saves diagnostic plots (distributions, time series, contribution breakdown).

Usage
-----
    uv run python calibration/analyze_cost_stats.py calibration/results/cost_stats_<timestamp>.csv
    uv run python calibration/analyze_cost_stats.py calibration/results/cost_stats_<timestamp>.csv --no-plots
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Domain-importance multipliers
# ---------------------------------------------------------------------------
# After normalization each term contributes O(1).  These multipliers express
# the *relative* importance of each term in the planning objective:
#   - Ice avoidance (costmap) is the primary safety concern → highest weight
#   - Path length penalizes unnecessary detours → baseline (1×)
#   - Goal distance provides directional guidance → moderate
#   - Control smoothness is a soft comfort constraint → lowest
IMPORTANCE = {
    "costmap":    1.0,
    "length":     1.0,
    "goal":       1.0,
    "smoothness": 0.125,
}

TERM_KEYS = {
    "costmap":    "costmap_cost_raw",
    "length":     "path_length_raw",
    "goal":       "goal_dist_raw",
    "smoothness": "smoothness_raw",
}

TERM_LABELS = {
    "costmap":    "Costmap cost  (ice avoidance, KE model)",
    "length":     "Path length   [m]",
    "goal":       "Goal distance [m]",
    "smoothness": "Smoothness    (mean Δu²)",
}

CONFIG_KEYS = {
    "costmap":    "planning_jax_mbd_costmap_weight",
    "length":     "planning_jax_mbd_length_weight",
    "goal":       "planning_jax_mbd_goal_weight",
    "smoothness": "planning_jax_mbd_smoothness_weight",
}


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _stats(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "n":      len(arr),
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min":    float(np.min(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75":    float(np.percentile(arr, 75)),
        "max":    float(np.max(arr)),
    }


def analyze(csv_path: Path, plot: bool = True, output_dir: Path | None = None):
    records = _load_csv(csv_path)
    if not records:
        print("ERROR: CSV is empty.", file=sys.stderr)
        sys.exit(1)

    n = len(records)
    print("=" * 70)
    print(f"MBD Cost Weight Calibration Analysis")
    print(f"  Source:  {csv_path}")
    print(f"  Probes:  {n}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Summary statistics
    # ------------------------------------------------------------------
    raw_values = {}
    stats_table = {}
    for term, col in TERM_KEYS.items():
        vals = [float(r[col]) for r in records]
        raw_values[term] = vals
        stats_table[term] = _stats(vals)

    print("\n--- Raw (unweighted) cost component statistics ---\n")
    header = f"  {'Term':<14} {'N':>4}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Median':>10}  {'Max':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for term, s in stats_table.items():
        print(f"  {term:<14} {s['n']:>4}  {s['mean']:>10.4f}  {s['std']:>10.4f}  "
              f"{s['min']:>10.4f}  {s['median']:>10.4f}  {s['max']:>10.4f}")

    # ------------------------------------------------------------------
    # 2. Normalized weights
    # ------------------------------------------------------------------
    # Reference scale = mean path_length so that w_length = 1.0 exactly.
    # w_i = (mean_length / mean_i) → each term contributes mean_length on average.
    reference = stats_table["length"]["mean"]
    norm_weights = {
        term: reference / stats_table[term]["mean"]
        for term in TERM_KEYS
    }

    print("\n--- Normalization weights (contribution ≈ mean path_length for all terms) ---\n")
    print(f"  Reference scale (mean path length): {reference:.2f} m\n")
    for term, w in norm_weights.items():
        contrib = w * stats_table[term]["mean"]
        print(f"  w_{term:<12} = {w:>10.5f}   (mean contribution: {contrib:.2f})")

    # ------------------------------------------------------------------
    # 3. Final recommended weights = normalized × importance multiplier
    # ------------------------------------------------------------------
    final_weights = {
        term: norm_weights[term] * IMPORTANCE[term]
        for term in TERM_KEYS
    }

    print("\n--- Recommended weights (normalized × domain-importance multiplier) ---\n")
    print(f"  Importance multipliers: {IMPORTANCE}\n")
    for term, w in final_weights.items():
        contrib = w * stats_table[term]["mean"]
        print(f"  w_{term:<12} = {w:>10.5f}   (mean contribution: {contrib:.2f})")

    total_contrib = sum(final_weights[t] * stats_table[t]["mean"] for t in TERM_KEYS)
    print(f"\n  Total expected weighted cost (mean): {total_contrib:.2f}")
    print(f"  Fractional contributions:")
    for term in TERM_KEYS:
        frac = final_weights[term] * stats_table[term]["mean"] / total_contrib
        print(f"    {term:<12}: {frac*100:.1f}%")

    # ------------------------------------------------------------------
    # 4. Config snippet
    # ------------------------------------------------------------------
    print("\n--- Ready-to-use config snippet (medea_ice_ship/config.py) ---\n")
    for term, key in CONFIG_KEYS.items():
        w = final_weights[term]
        # Round to 4 significant figures for readability
        magnitude = int(np.floor(np.log10(abs(w)))) if w != 0 else 0
        rounded = round(w, max(0, 4 - 1 - magnitude))
        print(f"    {key} = {rounded}")

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    if not plot:
        return final_weights

    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("\nmatplotlib not available — skipping plots.", file=sys.stderr)
        return final_weights

    probe_idxs = [int(r["probe_idx"]) for r in records]
    alphas = [float(r["route_alpha"]) for r in records]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"MBD Cost Component Statistics\n"
        f"n={n} probes · {csv_path.name}",
        fontsize=12,
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

    colors = {"costmap": "#d62728", "length": "#1f77b4", "goal": "#2ca02c", "smoothness": "#9467bd"}

    # Row 0: box plots (raw distributions)
    for col_idx, term in enumerate(TERM_KEYS):
        ax = fig.add_subplot(gs[0, col_idx])
        vals = raw_values[term]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor(colors[term])
        bp["boxes"][0].set_alpha(0.6)
        ax.set_title(term, fontsize=9)
        ax.set_ylabel("Raw value", fontsize=8)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.annotate(f"μ={stats_table[term]['mean']:.3g}\nσ={stats_table[term]['std']:.3g}",
                    xy=(0.97, 0.96), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Row 1: time series along route
    ax_ts = fig.add_subplot(gs[1, :])
    for term in TERM_KEYS:
        vals_norm = np.array(raw_values[term])
        # Normalize to [0, 1] for overlay comparison
        vmin, vmax = vals_norm.min(), vals_norm.max()
        denom = (vmax - vmin) if vmax > vmin else 1.0
        ax_ts.plot(alphas, (vals_norm - vmin) / denom,
                   marker="o", markersize=4, label=term, color=colors[term])
    ax_ts.set_xlabel("Route fraction (0 = start, 1 = near goal)", fontsize=9)
    ax_ts.set_ylabel("Normalised raw value [0–1]", fontsize=9)
    ax_ts.set_title("Raw component evolution along route (min–max normalised for overlay)", fontsize=9)
    ax_ts.legend(fontsize=8, loc="upper right")
    ax_ts.grid(True, alpha=0.3)

    # Row 2: contribution breakdown with final weights (stacked bar)
    ax_bar = fig.add_subplot(gs[2, :2])
    weighted_means = {t: final_weights[t] * stats_table[t]["mean"] for t in TERM_KEYS}
    terms_list = list(TERM_KEYS.keys())
    values_list = [weighted_means[t] for t in terms_list]
    bar_colors = [colors[t] for t in terms_list]
    bars = ax_bar.bar(terms_list, values_list, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax_bar.set_title("Mean weighted contribution\n(recommended weights × mean raw value)", fontsize=9)
    ax_bar.set_ylabel("Weighted cost", fontsize=9)
    for bar, val in zip(bars, values_list):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values_list),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax_bar.tick_params(axis="x", labelsize=8)

    # Row 2 right: pie chart of fractional contributions
    ax_pie = fig.add_subplot(gs[2, 2:])
    fracs = [weighted_means[t] / total_contrib for t in terms_list]
    wedge_labels = [f"{t}\n{f*100:.1f}%" for t, f in zip(terms_list, fracs)]
    ax_pie.pie(fracs, labels=wedge_labels, colors=bar_colors, autopct=None,
               startangle=90, wedgeprops={"alpha": 0.7, "edgecolor": "black", "linewidth": 0.5})
    ax_pie.set_title("Fractional contribution\n(recommended weights)", fontsize=9)

    plt.tight_layout()

    if output_dir is None:
        output_dir = csv_path.parent
    plot_path = output_dir / (csv_path.stem + "_analysis.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")
    plt.show()

    return final_weights


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MBD cost statistics and compute balanced weights."
    )
    parser.add_argument("csv_path", type=Path, help="CSV file from collect_cost_stats.py")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for output plots (default: same as CSV).",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"ERROR: {args.csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    analyze(args.csv_path, plot=not args.no_plots, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
