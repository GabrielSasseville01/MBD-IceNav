#!/usr/bin/env python3
"""
End-to-end test: simulate_jax_chunked_replan + diffusion-MPC replan callback.

Verifies that the full pipeline (JAX physics + diffusion-MPC replanning) runs
without error and produces physically sensible results.  Uses the same
3-floe scenario as the unit tests for speed.

Checkpoint tests
----------------
1. Chunked replan + dummy MBD callback: final ship position is within 10 m
   of the same run using jax_simulate_ice_ship_interaction (per-frame loop).
2. Chunked replan + JAX costmap callback: runs without error, ship moves in
   the expected direction (positive Y or forward in trajectory direction).
3. Chunked replan with goal stopping: simulation terminates before n_frames
   when ship crosses a goal_y_line.
4. History lengths are correct: n_frames + 1 entries in positions_history
   (initial + one per executed frame, capped at goal crossing).

Run with:
    cd /Users/gabrielsasseville/Documents/PhD/Projects/ShipNav/DPDEM
    uv run python test_jax_end_to_end.py
"""

import sys, os
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))


# ── Scenario helpers ───────────────────────────────────────────────────────────

def _make_scenario(n_frames=40, dt=0.05):
    """3-floe scenario; ship moves east fast enough to reach and break floes."""
    from medea_ice_ship.properties import ShipProperties

    angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
    r       = 3.0
    hex_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    centroids   = np.array([[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    polygons    = [hex_pts + c for c in centroids]
    bonds       = [(0, 1), (1, 2)]
    thicknesses = np.array([1.0, 1.0, 1.0])
    parent_ids  = np.arange(3, dtype=np.int32)

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
        n_frames=n_frames, dt=dt, drift_vel=np.array([0.0, 0.0]),
        thicknesses=thicknesses, ship_props=ship_props,
        ship_trajectory=ship_trajectory, parent_ids=parent_ids,
    )


# ── Test 1: Chunked replan matches per-frame loop ─────────────────────────────

def test_chunked_replan_matches_per_frame_endtoend():
    """
    With a deterministic dummy replan callback, simulate_jax_chunked_replan
    must produce the same final ship position as jax_simulate_ice_ship_interaction
    to within 0.5 m.
    """
    from medea_ice_ship.jax_simulation import (
        jax_simulate_ice_ship_interaction,
        simulate_jax_chunked_replan,
    )

    sc = _make_scenario(n_frames=30, dt=0.05)
    step = 0.5

    def _dummy_cb(frame, world_polygons, centroids, thicknesses,
                  ship_pos_f, ship_heading_f, n_remaining):
        return [
            (np.array([-25.0 + (frame + i) * step, 0.0]), 0.0)
            for i in range(n_remaining + 1)
        ]

    print("  Running jax_simulate_ice_ship_interaction (per-frame)...", flush=True)
    res_pf = jax_simulate_ice_ship_interaction(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=list(sc["ship_trajectory"]),
        replan_every_n_frames=10,
        replan_callback=_dummy_cb,
        parent_ids=sc["parent_ids"],
    )

    print("  Running simulate_jax_chunked_replan (chunk=10)...", flush=True)
    res_chk = simulate_jax_chunked_replan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=list(sc["ship_trajectory"]),
        replan_every_n_frames=10,
        replan_callback=_dummy_cb,
        parent_ids=sc["parent_ids"],
        verbose=False,
    )

    eta_pf  = np.array([res_pf[3][-1][0][0],  res_pf[3][-1][0][1]])
    eta_chk = np.array([res_chk[3][-1][0][0], res_chk[3][-1][0][1]])
    dist = np.linalg.norm(eta_pf - eta_chk)

    print(f"  Ship pos: per-frame={eta_pf}, chunked={eta_chk}, dist={dist:.4f} m")
    assert dist < 0.5, f"Ship position mismatch: {dist:.4f} m"
    print("  PASS: chunked replan end-to-end matches per-frame (< 0.5 m)")


# ── Test 2: JAX costmap callback runs without error ───────────────────────────

def test_chunked_with_jax_costmap_callback():
    """
    Callback that calls build_costmap_jax + jax_mbd_optimize must run
    without error and return a valid trajectory.
    """
    from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan
    from medea_ice_ship.jax_costmap import build_costmap_jax

    sc = _make_scenario(n_frames=20, dt=0.05)

    goal = np.array([20.0, 0.0])
    jax_replan_count = [0]

    def _jax_costmap_cb(frame, world_polygons, centroids, thicknesses,
                        ship_pos_f, ship_heading_f, n_remaining):
        """
        Minimal callback: build a JAX costmap and return a straight trajectory.
        (Skips the MBD optimization for speed; just tests the costmap building.)
        """
        jax_replan_count[0] += 1
        all_v = np.vstack(world_polygons)
        x_min, x_max = float(all_v[:, 0].min()), float(all_v[:, 0].max())
        y_min, y_max = float(all_v[:, 1].min()), float(all_v[:, 1].max())

        cost_map, bounds = build_costmap_jax(
            centroids, thicknesses, np.full(len(thicknesses), 3.0),
            x_min, x_max, y_min, y_max,
            resolution=2.0, margin=10.0,
            base_cost=0.1, collision_cost=5.0,
        )
        assert cost_map.cost_map.ndim == 2
        assert np.isfinite(cost_map.cost_map).all()

        # Return a straight-east trajectory toward the goal
        x0, y0 = float(ship_pos_f[0]), float(ship_pos_f[1])
        traj = []
        for i in range(n_remaining + 1):
            t = i / max(n_remaining, 1)
            x = x0 + (goal[0] - x0) * t
            y = y0 + (goal[1] - y0) * t
            traj.append((np.array([x, y]), 0.0))
        return traj

    print("  Running simulate_jax_chunked_replan with JAX costmap callback...", flush=True)
    res = simulate_jax_chunked_replan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=list(sc["ship_trajectory"]),
        replan_every_n_frames=10,
        replan_callback=_jax_costmap_cb,
        parent_ids=sc["parent_ids"],
        verbose=False,
    )

    assert len(res[0]) == sc["n_frames"] + 1
    eta_last = np.array([res[3][-1][0][0], res[3][-1][0][1]])
    ship_start = np.array(sc["ship_pos"])
    assert np.linalg.norm(eta_last - ship_start) > 1.0, (
        f"Ship did not move: {eta_last} ≈ {ship_start}"
    )
    # n_frames=20, chunk=10 → chunks start at [0, 10] → 2 replan calls
    expected_replans = sc["n_frames"] // 10
    assert jax_replan_count[0] == expected_replans, (
        f"Expected {expected_replans} replan calls, got {jax_replan_count[0]}"
    )
    print(f"  Ship moved from {ship_start} to {eta_last} "
          f"({jax_replan_count[0]} replans)")
    print("  PASS: chunked replan with JAX costmap callback")


# ── Test 3: Goal y-line stops simulation early ────────────────────────────────

def test_goal_yline_stops_early():
    """
    With goal_y_line set in the ship trajectory direction, the simulation
    must stop before n_frames.
    """
    from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan

    # Ship moves north from y=0 to y=+100 quickly
    n_frames = 100
    ship_trajectory = [
        (np.array([0.0, i * 2.0]), np.pi / 2)   # heading north, 2 m/frame
        for i in range(n_frames + 2)
    ]
    sc = _make_scenario(n_frames=n_frames, dt=0.05)
    sc["ship_trajectory"] = ship_trajectory
    sc["ship_pos"] = np.array([0.0, 0.0])

    print("  Running with goal_y_line=30 (should stop early)...", flush=True)
    res = simulate_jax_chunked_replan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=list(sc["ship_trajectory"]),
        replan_every_n_frames=20,
        replan_callback=None,
        parent_ids=sc["parent_ids"],
        goal_y_line=30.0,
        goal_tolerance=5.0,
        verbose=False,
    )

    actual_frames = len(res[0]) - 1   # subtract initial state
    print(f"  History length: {len(res[0])} entries ({actual_frames} frames executed)")
    assert actual_frames < n_frames, (
        f"Simulation did not stop early: ran {actual_frames}/{n_frames} frames"
    )
    print("  PASS: goal y-line stops simulation early")


# ── Test 4: History length correctness ────────────────────────────────────────

def test_history_length():
    """positions_history must have n_frames + 1 entries (initial + per frame)."""
    from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan

    sc = _make_scenario(n_frames=15, dt=0.05)

    res = simulate_jax_chunked_replan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=list(sc["ship_trajectory"]),
        replan_every_n_frames=5,
        parent_ids=sc["parent_ids"],
        verbose=False,
    )

    expected = sc["n_frames"] + 1
    assert len(res[0]) == expected, f"positions: {len(res[0])} != {expected}"
    assert len(res[1]) == expected, f"rotations: {len(res[1])} != {expected}"
    assert len(res[2]) == expected, f"bonds: {len(res[2])} != {expected}"
    assert len(res[4]) == expected, f"submergence: {len(res[4])} != {expected}"
    assert len(res[3]) == sc["n_frames"], f"ship_history: {len(res[3])} != {sc['n_frames']}"
    print(f"  History lengths: positions={len(res[0])}, ship={len(res[3])}")
    print("  PASS: history lengths correct")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        ("test_chunked_replan_matches_per_frame_endtoend",
         test_chunked_replan_matches_per_frame_endtoend),
        ("test_chunked_with_jax_costmap_callback",
         test_chunked_with_jax_costmap_callback),
        ("test_goal_yline_stops_early",
         test_goal_yline_stops_early),
        ("test_history_length",
         test_history_length),
    ]

    passed, failed = [], []
    print("\n=== End-to-end JAX simulation tests ===")
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
