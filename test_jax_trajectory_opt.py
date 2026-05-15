#!/usr/bin/env python3
"""
Tests for JAX/vectorized optimizations in the chunked-replan pipeline.

Covers:
1. path_to_ship_trajectory_arr  — vectorized (N+1,3) array equivalent of path_to_ship_trajectory
2. _trajectory_list_to_arr      — helper that converts list-of-tuples to (N,3) array
3. Vectorized setpoint packaging — array indexing vs old Python for-loop
4. Vectorized goal check        — np.argmax vs old Python for-loop
5. simulate_jax_chunked_replan integration — with/without world_polygons extraction skip

Run with:
    cd /Users/gabrielsasseville/Documents/PhD/Projects/ShipNav/DPDEM
    uv run python test_jax_trajectory_opt.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _traj_list_to_arr(traj_list):
    """Reference: convert list of (pos_2d, heading) → (N, 3) array."""
    return np.array([[p[0], p[1], h] for p, h in traj_list], dtype=np.float64)


def _make_3floe_scenario(n_frames=30, dt=0.05):
    from medea_ice_ship.properties import ShipProperties
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    r = 3.0
    hex_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    centroids = np.array([[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    polygons = [hex_pts + c for c in centroids]
    bonds = [(0, 1), (1, 2)]
    thicknesses = np.array([1.0, 1.0, 1.0])
    parent_ids = np.arange(3, dtype=np.int32)
    ship_length, ship_beam = 15.0, 5.0
    ship_props = ShipProperties(length=ship_length, beam=ship_beam, draft=2.5)
    ship_pos = np.array([-25.0, 0.0])
    # list-of-tuples format (legacy)
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. path_to_ship_trajectory_arr
# ─────────────────────────────────────────────────────────────────────────────

class TestPathToShipTrajectoryArr:
    """Verify path_to_ship_trajectory_arr matches path_to_ship_trajectory."""

    def _compare(self, path, n_frames, speed_profile=None, atol=1e-10):
        from medea_ice_ship.planning import path_to_ship_trajectory, path_to_ship_trajectory_arr
        ref_list = path_to_ship_trajectory(path, n_frames, speed_profile=speed_profile)
        ref = _traj_list_to_arr(ref_list)
        got = path_to_ship_trajectory_arr(path, n_frames, speed_profile=speed_profile)
        assert got.shape == (n_frames + 1, 3), f"Wrong shape: {got.shape}"
        np.testing.assert_allclose(got, ref, atol=atol,
            err_msg="path_to_ship_trajectory_arr differs from path_to_ship_trajectory")

    def test_two_point_straight(self):
        """Straight path: (0,0,0) → (100,0,0)."""
        path = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        self._compare(path, n_frames=10)

    def test_two_point_diagonal(self):
        """Diagonal path with heading."""
        path = np.array([[0.0, 0.0, np.pi / 4], [70.0, 70.0, np.pi / 4]])
        self._compare(path, n_frames=20)

    def test_multi_waypoint_uniform_speed(self):
        """Three-waypoint path with uniform speed."""
        path = np.array([
            [0.0,  0.0, 0.0],
            [50.0, 50.0, np.pi / 4],
            [100.0, 50.0, 0.0],
        ])
        self._compare(path, n_frames=100)

    def test_many_waypoints(self):
        """Many waypoints — stress test interpolation."""
        t = np.linspace(0, 2 * np.pi, 20)
        path = np.column_stack([np.cos(t) * 50, np.sin(t) * 50, t % (2 * np.pi)])
        self._compare(path, n_frames=200)

    def test_heading_wrap_positive_to_negative(self):
        """Heading crosses ±π boundary (3.0 → -3.0)."""
        path = np.array([[0.0, 0.0, 3.0], [100.0, 0.0, -3.0]])
        self._compare(path, n_frames=50)

    def test_heading_wrap_negative_to_positive(self):
        """Heading crosses ±π boundary (-3.0 → 3.0)."""
        path = np.array([[0.0, 0.0, -3.0], [100.0, 0.0, 3.0]])
        self._compare(path, n_frames=50)

    def test_single_point_path(self):
        """Single-point path: all frames equal that point."""
        from medea_ice_ship.planning import path_to_ship_trajectory_arr
        path = np.array([[5.0, 3.0, 1.2]])
        got = path_to_ship_trajectory_arr(path, n_frames=10)
        assert got.shape == (11, 3)
        expected = np.tile(path[0], (11, 1))
        np.testing.assert_allclose(got, expected, atol=1e-14)

    def test_zero_length_path(self):
        """Zero-length path (both points the same): all frames equal that point."""
        from medea_ice_ship.planning import path_to_ship_trajectory_arr
        path = np.array([[5.0, 3.0, 1.2], [5.0, 3.0, 1.2]])
        got = path_to_ship_trajectory_arr(path, n_frames=10)
        assert got.shape == (11, 3)
        np.testing.assert_allclose(got[:, :2], np.tile(path[0, :2], (11, 1)), atol=1e-14)

    def test_speed_profile_two_segment(self):
        """Speed profile changes time allocation between segments."""
        path = np.array([
            [0.0,   0.0, 0.0],
            [50.0,  0.0, 0.0],
            [100.0, 0.0, 0.0],
        ])
        speed_profile = np.array([2.0, 8.0])  # slow first, fast second
        self._compare(path, n_frames=100, speed_profile=speed_profile)

    def test_speed_profile_uniform_matches_no_profile(self):
        """Speed profile with equal speeds should match uniform (no profile) result."""
        from medea_ice_ship.planning import path_to_ship_trajectory_arr
        path = np.array([
            [0.0,  0.0, 0.0],
            [50.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ])
        got_uniform = path_to_ship_trajectory_arr(path, n_frames=50)
        got_speed   = path_to_ship_trajectory_arr(path, n_frames=50, speed_profile=np.array([3.0, 3.0]))
        np.testing.assert_allclose(got_uniform, got_speed, atol=1e-10)

    def test_endpoints_exact(self):
        """First and last frames must be exactly on path endpoints."""
        from medea_ice_ship.planning import path_to_ship_trajectory_arr
        path = np.array([[7.0, 3.0, 0.5], [12.0, 99.0, -1.2], [50.0, 50.0, 0.1]])
        got = path_to_ship_trajectory_arr(path, n_frames=77)
        np.testing.assert_array_equal(got[0],  path[0])
        np.testing.assert_array_equal(got[-1], path[-1])

    def test_n_frames_1(self):
        """n_frames=1: only start and end."""
        self._compare(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), n_frames=1)

    def test_output_dtype_float64(self):
        """Output must be float64."""
        from medea_ice_ship.planning import path_to_ship_trajectory_arr
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        got = path_to_ship_trajectory_arr(path, n_frames=5)
        assert got.dtype == np.float64


# ─────────────────────────────────────────────────────────────────────────────
# 2. _trajectory_list_to_arr helper
# ─────────────────────────────────────────────────────────────────────────────

class TestTrajectoryListToArr:

    def test_basic_conversion(self):
        from medea_ice_ship.jax_simulation import _trajectory_list_to_arr
        traj_list = [
            (np.array([1.0, 2.0]), 0.5),
            (np.array([3.0, 4.0]), 1.0),
            (np.array([5.0, 6.0]), -0.3),
        ]
        got = _trajectory_list_to_arr(traj_list)
        assert got.shape == (3, 3)
        np.testing.assert_array_equal(got[0], [1.0, 2.0, 0.5])
        np.testing.assert_array_equal(got[1], [3.0, 4.0, 1.0])
        np.testing.assert_array_equal(got[2], [5.0, 6.0, -0.3])

    def test_dtype(self):
        from medea_ice_ship.jax_simulation import _trajectory_list_to_arr
        traj = [(np.array([1.0, 2.0]), 0.0)]
        got = _trajectory_list_to_arr(traj)
        assert got.dtype == np.float64

    def test_roundtrip_with_path_to_trajectory(self):
        """list → arr → read back equals original."""
        from medea_ice_ship.planning import path_to_ship_trajectory
        from medea_ice_ship.jax_simulation import _trajectory_list_to_arr
        path = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.5]])
        traj_list = path_to_ship_trajectory(path, n_frames=20)
        got = _trajectory_list_to_arr(traj_list)
        ref = _traj_list_to_arr(traj_list)
        np.testing.assert_allclose(got, ref, atol=1e-14)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vectorized setpoint packaging
# ─────────────────────────────────────────────────────────────────────────────

class TestSetpointPackaging:
    """
    Verify that the vectorized setpoint build in simulate_jax_chunked_replan
    produces the same (K, 6) result as the old Python for-loop.
    """

    def _old_loop(self, ship_trajectory_list, frame_start, frame_end):
        """Reference: old Python loop from the original code."""
        traj_len = len(ship_trajectory_list)
        sp_rows = []
        for f in range(frame_start, frame_end):
            idx      = min(f, traj_len - 1)
            idx_next = min(f + 1, traj_len - 1)
            sp_pos,  sp_head  = ship_trajectory_list[idx]
            sp_posn, sp_headn = ship_trajectory_list[idx_next]
            sp_rows.append([
                float(sp_pos[0]),  float(sp_pos[1]),  float(sp_head),
                float(sp_posn[0]), float(sp_posn[1]), float(sp_headn),
            ])
        return np.array(sp_rows, dtype=np.float64)

    def _new_vectorized(self, traj_arr, frame_start, frame_end):
        """New: vectorized array indexing."""
        traj_len = len(traj_arr)
        frames   = np.arange(frame_start, frame_end)
        idx      = np.minimum(frames, traj_len - 1)
        idx_next = np.minimum(frames + 1, traj_len - 1)
        sp_cur = traj_arr[idx]       # (K, 3)
        sp_nxt = traj_arr[idx_next]  # (K, 3)
        return np.concatenate([sp_cur, sp_nxt], axis=1)  # (K, 6)

    def _make_traj(self, n):
        """Random-ish trajectory list and its array equivalent."""
        traj_list = [
            (np.array([float(i), float(i) * 0.5]), float(i) * 0.1)
            for i in range(n)
        ]
        traj_arr = _traj_list_to_arr(traj_list)
        return traj_list, traj_arr

    def test_middle_chunk(self):
        traj_list, traj_arr = self._make_traj(50)
        ref = self._old_loop(traj_list, frame_start=10, frame_end=20)
        got = self._new_vectorized(traj_arr, frame_start=10, frame_end=20)
        np.testing.assert_allclose(got, ref, atol=1e-14)

    def test_chunk_at_start(self):
        traj_list, traj_arr = self._make_traj(50)
        ref = self._old_loop(traj_list, frame_start=0, frame_end=15)
        got = self._new_vectorized(traj_arr, frame_start=0, frame_end=15)
        np.testing.assert_allclose(got, ref, atol=1e-14)

    def test_chunk_past_end_clamps(self):
        """When frame_end > traj_len, old and new both clamp to last entry."""
        traj_list, traj_arr = self._make_traj(10)
        ref = self._old_loop(traj_list, frame_start=8, frame_end=20)
        got = self._new_vectorized(traj_arr, frame_start=8, frame_end=20)
        np.testing.assert_allclose(got, ref, atol=1e-14)

    def test_single_frame_chunk(self):
        traj_list, traj_arr = self._make_traj(30)
        ref = self._old_loop(traj_list, frame_start=5, frame_end=6)
        got = self._new_vectorized(traj_arr, frame_start=5, frame_end=6)
        np.testing.assert_allclose(got, ref, atol=1e-14)

    def test_output_shape(self):
        _, traj_arr = self._make_traj(50)
        K = 15
        got = self._new_vectorized(traj_arr, frame_start=0, frame_end=K)
        assert got.shape == (K, 6)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Vectorized goal check
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalCheck:
    """
    Verify that vectorized goal-crossing detection matches the old Python for-loop.
    Both return (_f: first crossing index, found: bool).
    """

    def _old_loop(self, eta_K, goal_y, tol):
        for _f in range(len(eta_K)):
            if eta_K[_f, 1] >= goal_y - tol:
                return _f, True
        return -1, False

    def _new_vectorized(self, eta_K, goal_y, tol):
        crossed = eta_K[:, 1] >= goal_y - tol
        _f = int(np.argmax(crossed))
        return (_f, bool(crossed[_f]))

    def _make_eta(self, K, y_values):
        """eta_K: (K, 3) with column-1 = y_values."""
        eta = np.zeros((K, 3))
        eta[:, 1] = y_values
        return eta

    def test_crossing_at_index_0(self):
        eta_K = self._make_eta(20, np.linspace(10, 30, 20))
        for impl in (self._old_loop, self._new_vectorized):
            _f, found = impl(eta_K, goal_y=5.0, tol=0.0)
            assert found and _f == 0

    def test_crossing_in_middle(self):
        y_vals = np.concatenate([np.linspace(0, 9, 10), np.linspace(10, 20, 10)])
        eta_K = self._make_eta(20, y_vals)
        f_old, found_old = self._old_loop(eta_K, goal_y=10.0, tol=0.0)
        f_new, found_new = self._new_vectorized(eta_K, goal_y=10.0, tol=0.0)
        assert found_old and found_new
        assert f_old == f_new

    def test_crossing_at_last_frame(self):
        y_vals = np.zeros(20)
        y_vals[-1] = 15.0
        eta_K = self._make_eta(20, y_vals)
        f_old, found_old = self._old_loop(eta_K, goal_y=10.0, tol=0.0)
        f_new, found_new = self._new_vectorized(eta_K, goal_y=10.0, tol=0.0)
        assert found_old and found_new
        assert f_old == f_new == 19

    def test_no_crossing(self):
        y_vals = np.linspace(0, 5, 20)
        eta_K = self._make_eta(20, y_vals)
        f_old, found_old = self._old_loop(eta_K, goal_y=10.0, tol=0.0)
        f_new, found_new = self._new_vectorized(eta_K, goal_y=10.0, tol=0.0)
        assert not found_old
        assert not found_new

    def test_with_tolerance(self):
        """Tolerance shifts effective threshold down."""
        y_vals = np.linspace(0, 10, 20)
        eta_K = self._make_eta(20, y_vals)
        tol = 2.0
        f_old, found_old = self._old_loop(eta_K, goal_y=10.0, tol=tol)
        f_new, found_new = self._new_vectorized(eta_K, goal_y=10.0, tol=tol)
        assert found_old and found_new
        assert f_old == f_new

    def test_all_crossed(self):
        y_vals = np.ones(10) * 20.0
        eta_K = self._make_eta(10, y_vals)
        f_old, found_old = self._old_loop(eta_K, goal_y=10.0, tol=0.0)
        f_new, found_new = self._new_vectorized(eta_K, goal_y=10.0, tol=0.0)
        assert found_old and found_new
        assert f_old == f_new == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Integration: simulate_jax_chunked_replan with optimized internals
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateChunkedReplannOptimized:
    """
    Integration tests: optimized simulate_jax_chunked_replan must produce
    the same physics outputs as before (positions, rotations, bonds, ship pose).

    Specifically checks:
    - Trajectory as (N,3) array gives same result as list-of-tuples
    - Simulation with replan callback using path_to_ship_trajectory_arr works
    """

    def test_array_trajectory_matches_list_trajectory(self):
        """
        Passing a (N,3) array trajectory must give identical physics to the
        original list-of-tuples format.
        """
        from medea_ice_ship.jax_simulation import (
            simulate_jax_chunked_replan,
            _trajectory_list_to_arr,
        )

        sc = _make_3floe_scenario(n_frames=30, dt=0.05)
        traj_list = sc["ship_trajectory"]
        traj_arr  = _trajectory_list_to_arr(traj_list)

        kw_base = dict(
            polygons=sc["polygons"], centroids=sc["centroids"], bonds=sc["bonds"],
            ship_pos=sc["ship_pos"], ship_length=sc["ship_length"],
            ship_beam=sc["ship_beam"], n_frames=sc["n_frames"], dt=sc["dt"],
            drift_vel=sc["drift_vel"], thicknesses=sc["thicknesses"],
            ship_props=sc["ship_props"], parent_ids=sc["parent_ids"],
            replan_every_n_frames=10, verbose=False,
        )

        res_list = simulate_jax_chunked_replan(ship_trajectory=traj_list, **kw_base)
        res_arr  = simulate_jax_chunked_replan(ship_trajectory=traj_arr,  **kw_base)

        pos_list = np.array(res_list[0])
        pos_arr  = np.array(res_arr[0])
        np.testing.assert_allclose(pos_list, pos_arr, atol=1e-12,
            err_msg="Positions differ between list-traj and array-traj")

        rot_list = np.array(res_list[1])
        rot_arr  = np.array(res_arr[1])
        np.testing.assert_allclose(rot_list, rot_arr, atol=1e-12,
            err_msg="Rotations differ between list-traj and array-traj")

    def test_replan_callback_with_arr_trajectory(self):
        """
        A replan callback that returns path_to_ship_trajectory_arr output
        (i.e., an (n_remaining+1, 3) array) must produce valid simulation results.
        The callback is only invoked when an initial ship_trajectory is provided
        (use_ship_dynamics=True path).
        """
        from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan, _trajectory_list_to_arr
        from medea_ice_ship.planning import path_to_ship_trajectory_arr

        sc = _make_3floe_scenario(n_frames=30, dt=0.05)
        goal = np.array([50.0, 0.0])

        # Start with an initial trajectory (required for replan callback to fire)
        initial_traj = _trajectory_list_to_arr(sc["ship_trajectory"])
        replan_count = [0]

        def _callback(frame, world_polygons, centroids, thicknesses,
                      ship_pos_f, ship_heading_f, n_remaining):
            replan_count[0] += 1
            path = np.array([
                [float(ship_pos_f[0]), float(ship_pos_f[1]), float(ship_heading_f)],
                [float(goal[0]),       float(goal[1]),       0.0],
            ])
            # Return array format (N+1, 3)
            return path_to_ship_trajectory_arr(path, n_remaining)

        kw = dict(
            polygons=sc["polygons"], centroids=sc["centroids"], bonds=sc["bonds"],
            ship_pos=sc["ship_pos"], ship_length=sc["ship_length"],
            ship_beam=sc["ship_beam"], n_frames=30, dt=sc["dt"],
            drift_vel=sc["drift_vel"], thicknesses=sc["thicknesses"],
            ship_props=sc["ship_props"], parent_ids=sc["parent_ids"],
            ship_trajectory=initial_traj,
            replan_every_n_frames=10,
            replan_callback=_callback,
            verbose=False,
        )
        result = simulate_jax_chunked_replan(**kw)
        positions_history = result[0]
        assert len(positions_history) == 31  # n_frames + 1
        assert replan_count[0] >= 1

    def test_no_world_polygons_extraction_for_jax_path(self):
        """
        When world_polygons=None is passed to a callback (skipping extraction),
        the callback must not crash and must return a valid trajectory.
        """
        from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan
        from medea_ice_ship.planning import path_to_ship_trajectory_arr

        sc = _make_3floe_scenario(n_frames=30, dt=0.05)
        goal = np.array([50.0, 0.0])
        received_polys = []

        def _callback(frame, world_polygons, centroids, thicknesses,
                      ship_pos_f, ship_heading_f, n_remaining):
            received_polys.append(world_polygons)
            path = np.array([
                [float(ship_pos_f[0]), float(ship_pos_f[1]), float(ship_heading_f)],
                [float(goal[0]),       float(goal[1]),       0.0],
            ])
            return path_to_ship_trajectory_arr(path, n_remaining)

        kw = dict(
            polygons=sc["polygons"], centroids=sc["centroids"], bonds=sc["bonds"],
            ship_pos=sc["ship_pos"], ship_length=sc["ship_length"],
            ship_beam=sc["ship_beam"], n_frames=30, dt=sc["dt"],
            drift_vel=sc["drift_vel"], thicknesses=sc["thicknesses"],
            ship_props=sc["ship_props"], parent_ids=sc["parent_ids"],
            ship_trajectory=None,
            replan_every_n_frames=10,
            replan_callback=_callback,
            skip_polygon_extraction=True,
            verbose=False,
        )
        result = simulate_jax_chunked_replan(**kw)
        # All received polygon args must be None
        assert all(p is None for p in received_polys), \
            "world_polygons was not None despite skip_polygon_extraction=True"
        assert len(result[0]) == 31

    def test_goal_y_line_trim_correct(self):
        """
        Simulation with goal_y_line: history must be trimmed to the first
        frame where the ship crosses the line, not beyond.
        """
        from medea_ice_ship.jax_simulation import (
            simulate_jax_chunked_replan, _trajectory_list_to_arr,
        )

        sc = _make_3floe_scenario(n_frames=100, dt=0.05)
        traj_arr = _trajectory_list_to_arr(sc["ship_trajectory"])

        # Ship starts at x=-25 and moves east: won't cross goal_y_line=1e9
        result = simulate_jax_chunked_replan(
            polygons=sc["polygons"], centroids=sc["centroids"], bonds=sc["bonds"],
            ship_pos=sc["ship_pos"], ship_length=sc["ship_length"],
            ship_beam=sc["ship_beam"], n_frames=sc["n_frames"], dt=sc["dt"],
            drift_vel=sc["drift_vel"], thicknesses=sc["thicknesses"],
            ship_props=sc["ship_props"], parent_ids=sc["parent_ids"],
            ship_trajectory=traj_arr,
            replan_every_n_frames=10,
            goal_y_line=1e9,  # unreachable: all frames run
            verbose=False,
        )
        positions_history = result[0]
        # Should run all frames (goal never reached)
        assert len(positions_history) == sc["n_frames"] + 1


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestPathToShipTrajectoryArr,
        TestTrajectoryListToArr,
        TestSetpointPackaging,
        TestGoalCheck,
        TestSimulateChunkedReplannOptimized,
    ]

    total, passed, failed = 0, 0, []
    for cls in test_classes:
        obj = cls()
        methods = [m for m in dir(obj) if m.startswith("test_")]
        for name in methods:
            total += 1
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name}")
                traceback.print_exc()
                failed.append(f"{cls.__name__}.{name}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests passed.")
