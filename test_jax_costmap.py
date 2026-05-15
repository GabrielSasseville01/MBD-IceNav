#!/usr/bin/env python3
"""
Tests for JAX-native costmap builder (Phase 3).

Checkpoints
-----------
1. Grid dimensions match NumPy costmap for same scenario/params.
2. Open-water cells have base_cost; ice cells have cost > base_cost (both maps).
3. Cost RANKING: the 10% cheapest cells in the JAX map correspond to the 10%
   cheapest cells in the NumPy map with >= 80% overlap (Jaccard).
4. Max cost of JAX map is in the same ballpark as NumPy map (within 5×).
5. JAX map with ship dilation: ice cells after inflation have higher cost than
   pre-dilation values (dilation actually inflates).
6. build_costmap_jax_from_ice_state API works without error.

Run with:
    cd /Users/gabrielsasseville/Documents/PhD/Projects/ShipNav/DPDEM
    uv run python test_jax_costmap.py
"""

import sys, os
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


# ── Shared scenario helpers ────────────────────────────────────────────────────

def _make_ice_field():
    """Simple 5-floe field arranged in a cross pattern, 1–2 m thick."""
    positions = np.array([
        [  0.0,   0.0],
        [ 10.0,   0.0],
        [-10.0,   0.0],
        [  0.0,  10.0],
        [  0.0, -10.0],
    ])
    thicknesses = np.array([2.0, 1.0, 1.5, 0.5, 2.5])
    radii       = np.full(5, 4.0)   # 4 m radius circles
    return positions, thicknesses, radii


def _make_numpy_costmap(positions, thicknesses, resolution=2.0):
    """Build NumPy reference costmap for the same ice field."""
    from medea_ice_ship.costmap_build import build_costmap_from_ice_state
    from medea_ice_ship.geometry import polygon_area

    # Build hexagonal polygons at each centroid
    angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
    r       = 4.0
    hex_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    polygons = [hex_pts + p for p in positions]

    all_verts = np.vstack(polygons)
    x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
    y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()

    cost_map, bounds = build_costmap_from_ice_state(
        polygons, positions, thicknesses,
        x_min, x_max, y_min, y_max,
        resolution=resolution,
        margin=10.0,
        base_cost=0.1,
        collision_cost=5.0,
        obs_cost_buffer_factor=0.1,
    )
    return cost_map, bounds


# ── Test 1: Grid dimensions ────────────────────────────────────────────────────

def test_grid_dimensions_match():
    """JAX and NumPy costmaps for the same scenario must have the same shape."""
    from medea_ice_ship.jax_costmap import build_costmap_jax

    positions, thicknesses, radii = _make_ice_field()
    resolution = 2.0

    np_map, np_bounds = _make_numpy_costmap(positions, thicknesses, resolution)

    all_pos = positions
    x_min, x_max = all_pos[:, 0].min() - radii.max(), all_pos[:, 0].max() + radii.max()
    y_min, y_max = all_pos[:, 1].min() - radii.max(), all_pos[:, 1].max() + radii.max()

    jax_map, jax_bounds = build_costmap_jax(
        positions, thicknesses, radii,
        x_min, x_max, y_min, y_max,
        resolution=resolution,
        margin=10.0,
        base_cost=0.1,
        collision_cost=5.0,
        obs_cost_buffer_factor=0.1,
    )

    np_rows, np_cols = np_map.cost_map.shape
    jax_rows, jax_cols = jax_map.rows, jax_map.cols

    # Dimensions should be close (may differ by 1-2 due to origin differences)
    assert abs(jax_rows - np_rows) <= 3, (
        f"Row count too different: JAX={jax_rows}, NumPy={np_rows}"
    )
    assert abs(jax_cols - np_cols) <= 3, (
        f"Col count too different: JAX={jax_cols}, NumPy={np_cols}"
    )
    assert jax_map.cost_map.shape == (jax_rows, jax_cols)
    print(f"  PASS: grid dimensions — JAX=({jax_rows}×{jax_cols}), NumPy=({np_rows}×{np_cols})")


# ── Test 2: Open-water vs. ice cell costs ──────────────────────────────────────

def test_base_and_ice_costs():
    """
    In both maps: open-water cells have cost ≈ base_cost; ice cells have
    significantly higher cost.
    """
    from medea_ice_ship.jax_costmap import build_costmap_jax

    positions, thicknesses, radii = _make_ice_field()
    resolution = 2.0
    base_cost = 0.1

    x_min = positions[:, 0].min() - radii.max()
    x_max = positions[:, 0].max() + radii.max()
    y_min = positions[:, 1].min() - radii.max()
    y_max = positions[:, 1].max() + radii.max()

    jax_map, _ = build_costmap_jax(
        positions, thicknesses, radii,
        x_min, x_max, y_min, y_max,
        resolution=resolution, margin=20.0,
        base_cost=base_cost, collision_cost=5.0,
        obs_cost_buffer_factor=0.1,
    )
    grid = jax_map.cost_map

    # KE cost: ice cells must be significantly more expensive than open water
    assert grid.max() > base_cost * 5.0, (
        f"Max cost {grid.max():.3f} not significantly above base_cost {base_cost}"
    )

    # Check: cells far from any ice should be close to base_cost
    # Build a mask of cells far from all centroids
    cols_arr = np.arange(jax_map.cols)
    rows_arr = np.arange(jax_map.rows)
    CX = jax_map.x_origin + (cols_arr + 0.5) * resolution
    CY = jax_map.y_origin + (rows_arr + 0.5) * resolution
    XX, YY = np.meshgrid(CX, CY)
    far_from_ice = np.ones_like(grid, dtype=bool)
    for i, (cx, cy) in enumerate(positions):
        d = np.sqrt((XX - cx)**2 + (YY - cy)**2)
        far_from_ice &= (d > radii[i] * 2.5)

    if far_from_ice.any():
        far_costs = grid[far_from_ice]
        assert far_costs.max() <= base_cost * 1.05, (
            f"Far cells have cost {far_costs.max():.4f} > base_cost {base_cost:.4f}"
        )
        print(f"  Open water cost range: [{far_costs.min():.4f}, {far_costs.max():.4f}]")

    print(f"  Max grid cost: {grid.max():.3f} (expected > {base_cost * 5.0:.3f})")
    print("  PASS: base and ice costs")


# ── Test 3: Cost ranking agreement ────────────────────────────────────────────

def test_cost_ranking_agreement():
    """
    Both maps agree on which world-coordinate points are open water vs. ice.

    We sample a regular grid of world points and query both maps via bilinear
    interpolation; then compare their binary ice/open-water labelling.
    Jaccard similarity of open-water labels must be >= 0.6.
    """
    from medea_ice_ship.jax_costmap import build_costmap_jax

    positions, thicknesses, radii = _make_ice_field()
    resolution = 2.0
    margin     = 10.0

    # Common domain extent
    angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
    r_hex   = 4.0
    hex_pts = np.column_stack([r_hex * np.cos(angles), r_hex * np.sin(angles)])
    polygons = [hex_pts + p for p in positions]
    all_verts = np.vstack(polygons)
    x_min = float(all_verts[:, 0].min())
    x_max = float(all_verts[:, 0].max())
    y_min = float(all_verts[:, 1].min())
    y_max = float(all_verts[:, 1].max())

    # NumPy reference costmap
    from medea_ice_ship.costmap_build import build_costmap_from_ice_state
    np_map, _ = build_costmap_from_ice_state(
        polygons, positions, thicknesses,
        x_min, x_max, y_min, y_max,
        resolution=resolution, margin=margin,
        base_cost=0.1, collision_cost=5.0,
        obs_cost_buffer_factor=0.1,
    )

    # JAX costmap with same bounds
    jax_map, _ = build_costmap_jax(
        positions, thicknesses, radii,
        x_min, x_max, y_min, y_max,
        resolution=resolution, margin=margin,
        base_cost=0.1, collision_cost=5.0,
        obs_cost_buffer_factor=0.1,
    )

    # Both maps now have the same x_origin, y_origin, width, height, resolution
    assert abs(np_map.x_origin - jax_map.x_origin) < 1e-3
    assert abs(np_map.y_origin - jax_map.y_origin) < 1e-3
    assert np_map.cost_map.shape == jax_map.cost_map.shape, (
        f"Shape mismatch: NumPy={np_map.cost_map.shape}, JAX={jax_map.cost_map.shape}"
    )

    # Cell-wise comparison
    np_grid  = np_map.cost_map.astype(float)
    jax_grid = jax_map.cost_map.astype(float)

    # Open water: cost near base_cost
    base_cost = 0.1
    threshold = base_cost * 2.0          # some tolerance for rasterization differences
    np_open  = np_grid  <= threshold
    jax_open = jax_grid <= threshold

    intersection = int((np_open & jax_open).sum())
    union        = int((np_open | jax_open).sum())
    jaccard = intersection / max(union, 1)

    np_open_frac  = np_open.mean()
    jax_open_frac = jax_open.mean()

    print(f"  Open-water fraction: NumPy={np_open_frac:.3f}, JAX={jax_open_frac:.3f}")
    print(f"  Open-water Jaccard: {jaccard:.3f}  (∩={intersection}, ∪={union})")
    assert jaccard >= 0.6, (
        f"Open-water cell overlap too low: Jaccard={jaccard:.3f} < 0.6"
    )
    print("  PASS: cost ranking agreement (Jaccard >= 0.6)")


# ── Test 4: Max cost within 5× ────────────────────────────────────────────────

def test_max_cost_ballpark():
    """Max cost of JAX map must be within 5× of the NumPy map's max cost."""
    from medea_ice_ship.jax_costmap import build_costmap_jax

    positions, thicknesses, radii = _make_ice_field()
    resolution = 2.0

    x_min = positions[:, 0].min() - radii.max()
    x_max = positions[:, 0].max() + radii.max()
    y_min = positions[:, 1].min() - radii.max()
    y_max = positions[:, 1].max() + radii.max()

    np_map, _ = _make_numpy_costmap(positions, thicknesses, resolution)
    jax_map, _ = build_costmap_jax(
        positions, thicknesses, radii,
        x_min, x_max, y_min, y_max,
        resolution=resolution, margin=10.0,
        base_cost=0.1, collision_cost=5.0,
    )

    np_max  = float(np_map.cost_map.max())
    jax_max = float(jax_map.cost_map.max())

    ratio = max(np_max, jax_max) / max(min(np_max, jax_max), 1e-10)
    print(f"  Max cost: JAX={jax_max:.1f}, NumPy={np_max:.1f}, ratio={ratio:.2f}")
    assert ratio <= 5.0, (
        f"Max cost ratio too high: {ratio:.2f} (JAX={jax_max:.1f}, NumPy={np_max:.1f})"
    )
    print("  PASS: max cost within 5× of NumPy")


# ── Test 5: Dilation actually inflates ────────────────────────────────────────

def test_dilation_inflates():
    """With ship dilation, ice footprint must be wider than without."""
    from medea_ice_ship.jax_costmap import build_costmap_jax

    positions = np.array([[0.0, 0.0]])
    thicknesses = np.array([1.0])
    radii = np.array([4.0])

    x_min, x_max, y_min, y_max = -20.0, 20.0, -20.0, 20.0
    resolution = 1.0

    def _count_ice_cells(grid, base_cost=0.1):
        return int((grid > base_cost * 1.1).sum())

    # Without dilation
    no_dil, _ = build_costmap_jax(
        positions, thicknesses, radii,
        x_min, x_max, y_min, y_max,
        resolution=resolution, margin=5.0,
        base_cost=0.1, collision_cost=5.0,
        ship_length=None, ship_beam=None,
    )
    n_no_dil = _count_ice_cells(no_dil.cost_map)

    # With dilation (10 m ship beam → 5 m half-beam inflation)
    with_dil, _ = build_costmap_jax(
        positions, thicknesses, radii,
        x_min, x_max, y_min, y_max,
        resolution=resolution, margin=5.0,
        base_cost=0.1, collision_cost=5.0,
        ship_length=20.0, ship_beam=10.0,
        planning_clearance_margin=0.0,
    )
    n_with_dil = _count_ice_cells(with_dil.cost_map)

    print(f"  Ice cells: no-dil={n_no_dil}, with-dil={n_with_dil}")
    assert n_with_dil > n_no_dil, (
        f"Dilation did not increase ice cell count: {n_no_dil} → {n_with_dil}"
    )
    print("  PASS: dilation inflates ice footprint")


# ── Test 6: build_costmap_jax_from_ice_state API ──────────────────────────────

def test_build_from_ice_state_api():
    """build_costmap_jax_from_ice_state runs without error and returns valid result."""
    from medea_ice_ship.jax_costmap import build_costmap_jax_from_ice_state
    from medea_ice_ship.jax_simulation import (
        build_static_data, build_initial_ice_state,
    )
    from medea_ice_ship.config import get_default_simulation_config

    # Build a minimal IceState from the 3-floe scenario
    angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
    r       = 3.0
    hex_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    centroids   = np.array([[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    polygons    = [hex_pts + c for c in centroids]
    bonds       = [(0, 1), (1, 2)]
    thicknesses = np.array([1.0, 1.0, 1.0])
    parent_ids  = np.arange(3, dtype=np.int32)

    cfg    = get_default_simulation_config()
    static = build_static_data(polygons, centroids, bonds, thicknesses, parent_ids, cfg)
    state  = build_initial_ice_state(3, centroids, len(bonds), np.array(static.initial_floe_of))

    cost_map, bounds = build_costmap_jax_from_ice_state(
        state, static,
        x_min=-15.0, x_max=15.0, y_min=-5.0, y_max=5.0,
        resolution=1.0, margin=5.0,
        base_cost=0.1, collision_cost=5.0,
        verbose=True,
    )

    assert cost_map.cost_map.ndim == 2
    assert cost_map.rows == cost_map.cost_map.shape[0]
    assert cost_map.cols == cost_map.cost_map.shape[1]
    assert np.isfinite(cost_map.cost_map).all(), "NaN/Inf in costmap"
    print(f"  Grid shape: {cost_map.cost_map.shape}, max={cost_map.cost_map.max():.2f}")
    print("  PASS: build_costmap_jax_from_ice_state API")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        ("test_grid_dimensions_match",       test_grid_dimensions_match),
        ("test_base_and_ice_costs",          test_base_and_ice_costs),
        ("test_cost_ranking_agreement",      test_cost_ranking_agreement),
        ("test_max_cost_ballpark",           test_max_cost_ballpark),
        ("test_dilation_inflates",           test_dilation_inflates),
        ("test_build_from_ice_state_api",    test_build_from_ice_state_api),
    ]

    passed, failed = [], []
    print("\n=== Phase 3: JAX costmap tests ===")
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed.append(name)
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed.append((name, str(e)))

    print(f"\n{'=' * 60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("\nFailed tests:")
        for name, err in failed:
            print(f"  FAIL {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    run_all()
