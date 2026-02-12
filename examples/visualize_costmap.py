#!/usr/bin/env python3
"""
Visual test: costmap built from simple ice field (Section 1).

Builds the same ice state as --simple, creates the costmap, and saves a figure:
- Costmap as a 2D heatmap (open water = low cost, ice = high cost).
- Ice cell outlines overlaid so you can verify high cost aligns with ice.
- Domain and margin indicated.

Run: uv run python examples/visualize_costmap.py
Output: costmap_visualization.png (and optionally costmap_visualization.pdf).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors

from medea_ice_ship import (
    create_simple_ice_field,
    tessellate_all_floes,
    polygon_area,
    assign_thickness_by_area,
    build_costmap_from_ice_state,
)
from medea_ice_ship.config import get_default_run_config


def main():
    seed = 42
    np.random.seed(seed)
    run_cfg = get_default_run_config()

    # Same ice construction as run_medea_ship_fracture --simple
    sat_polygons, sat_centroids = create_simple_ice_field(seed=seed)
    all_cells, all_cell_centroids, all_parent_ids = tessellate_all_floes(
        sat_polygons, sat_centroids,
        cells_per_100m2=run_cfg.cells_per_100m2_simple,
        seed=seed,
    )
    polygons = all_cells
    centroids = all_cell_centroids
    sat_areas = np.array([polygon_area(p) for p in sat_polygons])
    sat_thicknesses = assign_thickness_by_area(
        sat_areas,
        min_thickness=run_cfg.thickness_min,
        max_thickness=run_cfg.thickness_max,
        power=run_cfg.thickness_power,
    )
    thicknesses = sat_thicknesses[all_parent_ids]

    all_verts = np.vstack(polygons)
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()

    resolution = 2.0
    margin = 20.0
    cost_map, bounds = build_costmap_from_ice_state(
        polygons, centroids, thicknesses,
        x_min, x_max, y_min, y_max,
        resolution=resolution,
        margin=margin,
        ship_mass=run_cfg.ship_displacement,
    )

    # Grid edges for pcolormesh (world coordinates)
    cols, rows = cost_map.cols, cost_map.rows
    x_origin = bounds["x_origin"]
    y_origin = bounds["y_origin"]
    X = x_origin + np.arange(cols + 1) * resolution
    Y = y_origin + np.arange(rows + 1) * resolution

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cost = cost_map.cost_map
    # Use a log scale so base_cost (1) and ice (e.g. 100+) both visible
    norm = colors.LogNorm(vmin=max(0.5, cost.min()), vmax=cost.max())
    pm = ax.pcolormesh(X, Y, cost, norm=norm, cmap="viridis", shading="flat")
    plt.colorbar(pm, ax=ax, label="Cost (log scale)")

    # Overlay ice polygon outlines (black)
    for poly in polygons:
        closed = np.vstack([poly, poly[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], "k-", linewidth=0.4, alpha=0.9)

    # Domain rectangle (ice extent, no margin)
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False, edgecolor="cyan", linewidth=1.5, linestyle="--", label="Ice domain"
    )
    ax.add_patch(rect)
    # Margin region: full costmap extent
    ax.axvline(x_origin, color="gray", linewidth=0.5, alpha=0.7)
    ax.axvline(x_origin + bounds["width"], color="gray", linewidth=0.5, alpha=0.7)
    ax.axhline(y_origin, color="gray", linewidth=0.5, alpha=0.7)
    ax.axhline(y_origin + bounds["height"], color="gray", linewidth=0.5, alpha=0.7)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title("Costmap from simple ice field (Section 1)\nYellow = high cost (ice), dark = low cost (open water)")
    ax.grid(True, alpha=0.3)

    out_dir = Path(__file__).resolve().parent.parent
    png_path = out_dir / "costmap_visualization.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")
    fig.savefig(out_dir / "costmap_visualization.pdf", bbox_inches="tight")
    print(f"Saved: {out_dir / 'costmap_visualization.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
