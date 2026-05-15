#!/usr/bin/env python3
"""
Validation tests for the GPU-native replan callback path.

Tests that build_costmap_jax is used correctly when --jax-sim is active,
producing costmaps compatible with the MBD planner and the existing
simulate_jax_chunked_replan interface.

Run with:
    cd /Users/gabrielsasseville/Documents/PhD/Projects/ShipNav/DPDEM
    uv run python test_jax_replan_callback.py
"""

import sys, os
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))


# ── Shared scenario ────────────────────────────────────────────────────────────

def _make_scenario(n_frames=20, dt=0.05):
    from medea_ice_ship.properties import ShipProperties

    angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
    r       = 3.0
    hex_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    centroids   = np.array([[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    polygons    = [hex_pts + c for c in centroids]
    bonds       = [(0, 1), (1, 2)]
    thicknesses = np.array([1.0, 1.0, 1.0])
    parent_ids  = np.arange(3, dtype=np.int32)
    radii       = np.full(3, r, dtype=np.float32)

    ship_length, ship_beam = 15.0, 5.0
    ship_props  = ShipProperties(length=ship_length, beam=ship_beam, draft=2.5)
    ship_pos    = np.array([-25.0, 0.0])

    ship_trajectory = [
        (np.array([-25.0 + i * 0.5, 0.0]), 0.0)
        for i in range(n_frames + 2)
    ]

    return dict(
        polygons=polygons, centroids=centroids, bonds=bonds,
        ship_pos=ship_pos, ship_length=ship_length, ship_beam=ship_beam,
        n_frames=n_frames, dt=dt, thicknesses=thicknesses, ship_props=ship_props,
        ship_trajectory=ship_trajectory, parent_ids=parent_ids, radii=radii,
    )


# ── Test 1: build_costmap_jax produces compatible output ──────────────────────

def test_jax_costmap_mbd_compatible():
    """
    build_costmap_jax result must be duck-type compatible with jax_mbd_optimize
    (both accept .cost_map, .x_origin, .y_origin, .resolution, .rows, .cols).
    """
    from medea_ice_ship.jax_costmap import build_costmap_jax

    sc = _make_scenario()
    centroids   = np.array(sc["centroids"])
    thicknesses = sc["thicknesses"]
    radii       = sc["radii"]

    x_min_f, x_max_f = float(centroids[:, 0].min()) - 5.0, float(centroids[:, 0].max()) + 5.0
    y_min_f, y_max_f = float(centroids[:, 1].min()) - 5.0, float(centroids[:, 1].max()) + 5.0

    cost_map_f, bounds = build_costmap_jax(
        centroids, thicknesses, radii,
        x_min_f, x_max_f, y_min_f, y_max_f,
        resolution=2.0, margin=10.0,
        collision_cost=5.0,
        obs_cost_buffer_factor=0.1,
        ship_length=15.0, ship_beam=5.0, planning_clearance_margin=5.0,
    )

    # Check required MBD-compatible attributes
    assert hasattr(cost_map_f, "cost_map"),   "Missing cost_map"
    assert hasattr(cost_map_f, "x_origin"),   "Missing x_origin"
    assert hasattr(cost_map_f, "y_origin"),   "Missing y_origin"
    assert hasattr(cost_map_f, "resolution"), "Missing resolution"
    assert hasattr(cost_map_f, "rows"),       "Missing rows"
    assert hasattr(cost_map_f, "cols"),       "Missing cols"
    assert hasattr(cost_map_f, "width"),      "Missing width"
    assert hasattr(cost_map_f, "height"),     "Missing height"

    grid = np.asarray(cost_map_f.cost_map)
    assert grid.ndim == 2, f"cost_map must be 2D, got shape {grid.shape}"
    assert np.isfinite(grid).all(), "cost_map has non-finite values"
    assert grid.max() > 0.1, "cost_map appears empty (max too low)"

    print(f"  Grid: {cost_map_f.rows}×{cost_map_f.cols}, "
          f"max cost={grid.max():.1f}, x_origin={cost_map_f.x_origin:.1f}")
    print("  PASS: build_costmap_jax → MBD-compatible JaxCostMapResult")


# ── Test 2: JAX callback produces valid trajectory ────────────────────────────

def test_jax_replan_callback_trajectory():
    """
    A replan callback using build_costmap_jax + MBD optimization must return
    a valid (n_remaining+1, ) trajectory list without error.
    """
    from medea_ice_ship.jax_costmap import build_costmap_jax
    from medea_ice_ship.jax_mbd_planner import jax_mbd_optimize
    from medea_ice_ship.config import get_default_run_config

    sc    = _make_scenario()
    plan_cfg = get_default_run_config()
    goal  = (30.0, 0.0)
    radii = sc["radii"]

    n_replan_calls = [0]

    def _jax_callback(frame, world_polygons, centroids, thicknesses,
                      ship_pos_f, ship_heading_f, n_remaining):
        n_replan_calls[0] += 1

        # JAX-native bounds (no world_polygons needed)
        _r_max = float(radii.max())
        x_min_f = float(centroids[:, 0].min()) - _r_max
        x_max_f = float(centroids[:, 0].max()) + _r_max
        y_min_f = float(centroids[:, 1].min()) - _r_max
        y_max_f = float(centroids[:, 1].max()) + _r_max

        cost_map_f, _ = build_costmap_jax(
            centroids, thicknesses, radii,
            x_min_f, x_max_f, y_min_f, y_max_f,
            resolution=plan_cfg.planning_resolution,
            margin=plan_cfg.planning_margin,
            collision_cost=plan_cfg.planning_ice_collision_cost,
        )

        start_f = (float(ship_pos_f[0]), float(ship_pos_f[1]), float(ship_heading_f))
        path_f  = np.array([[start_f[0], start_f[1], start_f[2]],
                             [goal[0],   goal[1],   start_f[2]]])

        optimized = jax_mbd_optimize(
            cost_map=cost_map_f,
            start_pose=start_f,
            goal=goal,
            run_cfg=plan_cfg,
            start_nu=(2.0, 0.0, 0.0),
            lattice_path=path_f,
            warm_start=False,
        )
        assert optimized.ndim == 2, f"MBD output must be 2D, got {optimized.shape}"
        assert optimized.shape[1] == 3, f"MBD output must be (T, 3), got {optimized.shape}"

        # Convert to ship_trajectory format
        traj = []
        for i in range(n_remaining + 1):
            t = i / max(n_remaining, 1)
            idx = min(int(t * (len(optimized) - 1)), len(optimized) - 1)
            traj.append((optimized[idx, :2].copy(), float(optimized[idx, 2])))
        return traj

    from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan

    print("  Running simulate_jax_chunked_replan with JAX-native replan callback...", flush=True)
    res = simulate_jax_chunked_replan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], np.array([0.0, 0.0]),
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=list(sc["ship_trajectory"]),
        replan_every_n_frames=10,
        replan_callback=_jax_callback,
        parent_ids=sc["parent_ids"],
        verbose=False,
    )

    assert len(res[0]) == sc["n_frames"] + 1, (
        f"Expected {sc['n_frames'] + 1} position frames, got {len(res[0])}"
    )
    # n_frames=20, chunk=10 → 2 replan calls
    assert n_replan_calls[0] == 2, f"Expected 2 replans, got {n_replan_calls[0]}"
    print(f"  {n_replan_calls[0]} replans, final ship pos: {res[3][-1][0]}")
    print("  PASS: JAX-native replan callback produces valid trajectory")


# ── Test 3: Bounds from centroids+radii match world_polygons bounds ────────────

def test_centroid_bounds_match_world_bounds():
    """
    Bounds from centroids + radii should enclose the same region as world_polygons
    bounds (with a small tolerance since disk bounds are approximate).
    """
    sc  = _make_scenario()
    centroids   = np.array(sc["centroids"])
    polygons    = sc["polygons"]
    radii       = sc["radii"]

    # CPU/world_polygons bounds
    all_verts = np.vstack(polygons)
    x_min_np = float(all_verts[:, 0].min())
    x_max_np = float(all_verts[:, 0].max())
    y_min_np = float(all_verts[:, 1].min())
    y_max_np = float(all_verts[:, 1].max())

    # Centroid+radii bounds
    _r_max = float(radii.max())
    x_min_jax = float(centroids[:, 0].min()) - _r_max
    x_max_jax = float(centroids[:, 0].max()) + _r_max
    y_min_jax = float(centroids[:, 1].min()) - _r_max
    y_max_jax = float(centroids[:, 1].max()) + _r_max

    # JAX bounds should enclose world_polygons bounds
    assert x_min_jax <= x_min_np + 0.1, f"JAX x_min {x_min_jax:.1f} > world x_min {x_min_np:.1f}"
    assert x_max_jax >= x_max_np - 0.1, f"JAX x_max {x_max_jax:.1f} < world x_max {x_max_np:.1f}"
    assert y_min_jax <= y_min_np + 0.1, f"JAX y_min {y_min_jax:.1f} > world y_min {y_min_np:.1f}"
    assert y_max_jax >= y_max_np - 0.1, f"JAX y_max {y_max_jax:.1f} < world y_max {y_max_np:.1f}"

    print(f"  World bounds: X=[{x_min_np:.1f}, {x_max_np:.1f}] Y=[{y_min_np:.1f}, {y_max_np:.1f}]")
    print(f"  JAX bounds:   X=[{x_min_jax:.1f}, {x_max_jax:.1f}] Y=[{y_min_jax:.1f}, {y_max_jax:.1f}]")
    print("  PASS: centroid+radii bounds enclose world polygon bounds")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        ("JAX costmap MBD compatible",          test_jax_costmap_mbd_compatible),
        ("JAX replan callback trajectory",      test_jax_replan_callback_trajectory),
        ("Centroid bounds match world bounds",  test_centroid_bounds_match_world_bounds),
    ]

    print("\n=== JAX-native replan callback tests ===\n")
    passed, failed = 0, 0
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn()
            passed += 1
        except Exception as exc:
            import traceback
            print(f"  FAIL: {exc}")
            traceback.print_exc()
            failed += 1
        print()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    run_all()
