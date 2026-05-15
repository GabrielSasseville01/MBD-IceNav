#!/usr/bin/env python3
"""
Tests for JAX chunked replan simulation (Phases 1 and 2).

Checkpoints
-----------
1. _run_jax_segment (single chunk, K=n_frames) matches simulate_full_jax_scan
   to floating-point precision.
2. _extract_world_polygons_np matches the original Python loop exactly.
3. simulate_jax_chunked_replan (no replan) matches simulate_full_jax_scan.
4. simulate_jax_chunked_replan (with replan) matches jax_simulate_ice_ship_interaction
   with the same callback to float tolerance.
5. simulate_jax_chunked_replan short-chunk decomposition matches simulate_full_jax_scan
   (verifies the chunk-padding path is correct when n_frames % chunk != 0).

Run with:
    cd /Users/gabrielsasseville/Documents/PhD/Projects/ShipNav/DPDEM
    uv run python test_jax_chunked_replan.py
"""

import sys, os

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))


# ── Shared scenario helpers ────────────────────────────────────────────────────

def _make_3floe_scenario(n_frames=30, dt=0.05):
    """Three hexagonal floes in a line; ship moves east."""
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


# ── Test 1: _run_jax_segment single-chunk matches simulate_full_jax_scan ──────

def test_run_jax_segment_matches_full_scan():
    """
    A single call to _run_jax_segment covering all N frames must produce the
    same positions, rotations, bond states, and ship pose as simulate_full_jax_scan.
    """
    from medea_ice_ship.jax_simulation import (
        simulate_full_jax_scan,
        build_jax_sim_context,
        _run_jax_segment,
        build_initial_ice_state,
        build_initial_ship_state,
        build_static_data,
    )
    from medea_ice_ship.config import get_default_simulation_config

    sc = _make_3floe_scenario(n_frames=20, dt=0.05)

    # Reference: simulate_full_jax_scan
    print("  Running simulate_full_jax_scan reference...", flush=True)
    res_ref = simulate_full_jax_scan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=sc["ship_trajectory"],
        parent_ids=sc["parent_ids"],
    )

    # Build context and run via _run_jax_segment
    print("  Building JaxSimContext...", flush=True)
    ctx = build_jax_sim_context(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        parent_ids=sc["parent_ids"],
        chunk_size=sc["n_frames"],
        verbose=False,
    )

    cfg = get_default_simulation_config()
    static = build_static_data(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["thicknesses"], sc["parent_ids"], cfg,
    )
    state      = build_initial_ice_state(
        ctx.n, sc["centroids"], ctx.n_bonds, np.array(ctx.static.initial_floe_of),
    )
    ship_state = build_initial_ship_state(np.array(sc["ship_trajectory"][0][0]), 0.0)

    # Pre-compute setpoints
    K = sc["n_frames"]
    traj = sc["ship_trajectory"]
    sp_rows = []
    for f in range(K):
        idx      = min(f, len(traj) - 1)
        idx_next = min(f + 1, len(traj) - 1)
        sp_pos,  sh  = traj[idx]
        sp_posn, shn = traj[idx_next]
        sp_rows.append([float(sp_pos[0]),  float(sp_pos[1]),  float(sh),
                        float(sp_posn[0]), float(sp_posn[1]), float(shn)])
    setpoints_K6 = jnp.array(sp_rows, dtype=jnp.float64)
    v_water_Kn2  = jnp.zeros((K, ctx.n, 2), dtype=jnp.float64)

    print("  Running _run_jax_segment...", flush=True)
    final_state, final_ship, hist = _run_jax_segment(
        ctx, state, ship_state, setpoints_K6, v_water_Kn2,
    )

    # Compare
    pos_ref  = np.array(res_ref[0][-1])
    pos_seg  = np.array(hist[0][-1])
    np.testing.assert_allclose(
        pos_seg, pos_ref, atol=1e-8, rtol=1e-8,
        err_msg="Final ice positions differ between _run_jax_segment and simulate_full_jax_scan",
    )

    rot_ref  = np.array(res_ref[1][-1])
    rot_seg  = np.array(hist[1][-1])
    np.testing.assert_allclose(rot_seg, rot_ref, atol=1e-8,
                               err_msg="Final rotations differ")

    bond_ref = np.array(res_ref[2][-1])
    bond_seg = np.array(hist[2][-1])
    np.testing.assert_array_equal(bond_seg, bond_ref,
                                  err_msg="Bond states differ")

    eta_ref  = np.array([res_ref[3][-1][0][0], res_ref[3][-1][0][1], res_ref[3][-1][1]])
    eta_seg  = np.array(hist[3][-1])
    np.testing.assert_allclose(eta_seg, eta_ref, atol=1e-8,
                               err_msg="Ship final pose differs")

    print("  PASS: _run_jax_segment matches simulate_full_jax_scan")


# ── Test 2: _extract_world_polygons_np matches Python loop ────────────────────

def test_extract_world_polygons_matches_loop():
    """Vectorized extraction must match the original per-element Python loop exactly."""
    from medea_ice_ship.jax_simulation import (
        _extract_world_polygons_np,
        build_static_data,
        build_initial_ice_state,
    )
    from medea_ice_ship.config import get_default_simulation_config

    sc = _make_3floe_scenario(n_frames=1)
    cfg = get_default_simulation_config()
    static = build_static_data(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["thicknesses"], sc["parent_ids"], cfg,
    )

    n = len(sc["polygons"])
    state = build_initial_ice_state(n, sc["centroids"], len(sc["bonds"]),
                                    np.array(static.initial_floe_of))

    # Add some non-zero rotations/positions to make the test non-trivial
    fake_pos = np.array(state.positions).copy()
    fake_pos[0] += np.array([1.5, -0.7])
    fake_rot = np.array(state.rotations).copy()
    fake_rot[1] = 0.4

    from medea_ice_ship.jax_ice_state import IceState
    state = state._replace(
        positions=jnp.array(fake_pos),
        rotations=jnp.array(fake_rot),
    )

    # Reference: original Python loop
    pos_np        = np.array(state.positions)
    rot_np        = np.array(state.rotations)
    lp_np         = np.array(static.local_polys_padded)
    n_verts_np    = np.array(static.poly_n_verts)

    ref_polygons = []
    for i in range(n):
        nv  = int(n_verts_np[i])
        lp  = lp_np[i, :nv]
        c_r, s_r = np.cos(rot_np[i]), np.sin(rot_np[i])
        R   = np.array([[c_r, -s_r], [s_r, c_r]])
        ref_polygons.append(lp @ R.T + pos_np[i])

    # Vectorized version
    fast_polygons = _extract_world_polygons_np(state, static)

    assert len(fast_polygons) == len(ref_polygons), "Different number of polygons"
    for i, (ref, fast) in enumerate(zip(ref_polygons, fast_polygons)):
        np.testing.assert_allclose(
            fast, ref, atol=1e-12,
            err_msg=f"Polygon {i} differs between vectorized and loop extraction",
        )

    print("  PASS: _extract_world_polygons_np matches Python loop")


# ── Test 3: simulate_jax_chunked_replan (no replan) matches simulate_full_jax_scan ──

def test_chunked_no_replan_matches_full_scan():
    """
    With no replan_callback and chunk=n_frames, chunked replan must match
    simulate_full_jax_scan identically.
    """
    from medea_ice_ship.jax_simulation import (
        simulate_full_jax_scan,
        simulate_jax_chunked_replan,
    )

    sc = _make_3floe_scenario(n_frames=20, dt=0.05)

    print("  Running simulate_full_jax_scan reference...", flush=True)
    res_ref = simulate_full_jax_scan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=sc["ship_trajectory"],
        parent_ids=sc["parent_ids"],
    )

    print("  Running simulate_jax_chunked_replan (chunk=n_frames, no callback)...", flush=True)
    res_chk = simulate_jax_chunked_replan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=list(sc["ship_trajectory"]),   # mutable copy
        replan_every_n_frames=sc["n_frames"],           # single chunk
        replan_callback=None,
        parent_ids=sc["parent_ids"],
        verbose=False,
    )

    pos_ref = np.array(res_ref[0][-1])
    pos_chk = np.array(res_chk[0][-1])
    np.testing.assert_allclose(pos_ref, pos_chk, atol=1e-8, rtol=1e-8,
                               err_msg="Final positions differ")

    rot_ref = np.array(res_ref[1][-1])
    rot_chk = np.array(res_chk[1][-1])
    np.testing.assert_allclose(rot_ref, rot_chk, atol=1e-8,
                               err_msg="Final rotations differ")

    bond_ref = np.array(res_ref[2][-1])
    bond_chk = np.array(res_chk[2][-1])
    np.testing.assert_array_equal(bond_ref, bond_chk, err_msg="Bond states differ")

    eta_ref = np.array([res_ref[3][-1][0][0], res_ref[3][-1][0][1], res_ref[3][-1][1]])
    eta_chk = np.array([res_chk[3][-1][0][0], res_chk[3][-1][0][1], res_chk[3][-1][1]])
    np.testing.assert_allclose(eta_ref, eta_chk, atol=1e-8,
                               err_msg="Ship pose differs")

    assert len(res_ref[0]) == len(res_chk[0]), (
        f"History lengths differ: ref={len(res_ref[0])}, chk={len(res_chk[0])}"
    )

    print("  PASS: simulate_jax_chunked_replan (no replan) matches simulate_full_jax_scan")


# ── Test 4: Multiple short chunks match simulate_full_jax_scan ─────────────────

def test_chunked_multi_chunk_matches_full_scan():
    """
    With chunk < n_frames (and n_frames % chunk != 0), chunked replan must
    still match simulate_full_jax_scan to float precision.
    Tests the padding path for the final short chunk.
    """
    from medea_ice_ship.jax_simulation import (
        simulate_full_jax_scan,
        simulate_jax_chunked_replan,
    )

    sc = _make_3floe_scenario(n_frames=13, dt=0.05)  # 13 = 3*4 + 1 → tests padding

    print("  Running simulate_full_jax_scan reference (13 frames)...", flush=True)
    res_ref = simulate_full_jax_scan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=sc["ship_trajectory"],
        parent_ids=sc["parent_ids"],
    )

    print("  Running simulate_jax_chunked_replan (chunk=4, 4 chunks, 13 frames)...", flush=True)
    res_chk = simulate_jax_chunked_replan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=list(sc["ship_trajectory"]),
        replan_every_n_frames=4,
        replan_callback=None,
        parent_ids=sc["parent_ids"],
        verbose=False,
    )

    pos_ref = np.array(res_ref[0][-1])
    pos_chk = np.array(res_chk[0][-1])
    np.testing.assert_allclose(pos_ref, pos_chk, atol=1e-8, rtol=1e-8,
                               err_msg="Final positions differ (multi-chunk)")

    bond_ref = np.array(res_ref[2][-1])
    bond_chk = np.array(res_chk[2][-1])
    np.testing.assert_array_equal(bond_ref, bond_chk,
                                  err_msg="Bond states differ (multi-chunk)")

    eta_ref = np.array([res_ref[3][-1][0][0], res_ref[3][-1][0][1], res_ref[3][-1][1]])
    eta_chk = np.array([res_chk[3][-1][0][0], res_chk[3][-1][0][1], res_chk[3][-1][1]])
    np.testing.assert_allclose(eta_ref, eta_chk, atol=1e-8,
                               err_msg="Ship pose differs (multi-chunk)")

    assert len(res_ref[0]) == len(res_chk[0]), (
        f"History lengths differ: ref={len(res_ref[0])}, chk={len(res_chk[0])}"
    )

    print("  PASS: multi-chunk simulate_jax_chunked_replan matches simulate_full_jax_scan")


# ── Test 5: With replan callback, chunked matches per-frame JAX loop ──────────

def test_chunked_with_replan_matches_per_frame():
    """
    simulate_jax_chunked_replan with a dummy replan_callback (returns same
    straight-east trajectory) must match jax_simulate_ice_ship_interaction
    with the same callback to float tolerance.
    """
    from medea_ice_ship.jax_simulation import (
        jax_simulate_ice_ship_interaction,
        simulate_jax_chunked_replan,
    )

    sc = _make_3floe_scenario(n_frames=20, dt=0.05)

    replan_count = [0]

    def _dummy_callback(frame, world_polygons, centroids, thicknesses,
                        ship_pos_f, ship_heading_f, n_remaining):
        """Returns the same straight-east trajectory regardless of ice state."""
        replan_count[0] += 1
        step = 0.5  # same speed as initial trajectory
        return [
            (np.array([-25.0 + (frame + i) * step, 0.0]), 0.0)
            for i in range(n_remaining + 1)
        ]

    print("  Running jax_simulate_ice_ship_interaction (per-frame loop)...", flush=True)
    res_pf = jax_simulate_ice_ship_interaction(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=list(sc["ship_trajectory"]),
        replan_every_n_frames=5,
        replan_callback=_dummy_callback,
        parent_ids=sc["parent_ids"],
    )
    n_replans_pf = replan_count[0]
    replan_count[0] = 0

    print("  Running simulate_jax_chunked_replan (chunk=5)...", flush=True)
    res_chk = simulate_jax_chunked_replan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=list(sc["ship_trajectory"]),
        replan_every_n_frames=5,
        replan_callback=_dummy_callback,
        parent_ids=sc["parent_ids"],
        verbose=False,
    )
    n_replans_chk = replan_count[0]

    # Both should call callback the same number of times (once per chunk after frame 0)
    # per-frame: frames 5, 10, 15 → 3 calls; chunked: chunks starting at 5, 10, 15 → 3 calls
    assert n_replans_pf == n_replans_chk, (
        f"Replan count differs: per-frame={n_replans_pf}, chunked={n_replans_chk}"
    )

    pos_pf  = np.array(res_pf[0][-1])
    pos_chk = np.array(res_chk[0][-1])
    np.testing.assert_allclose(pos_pf, pos_chk, atol=1e-8, rtol=1e-8,
                               err_msg="Final positions differ (with replan callback)")

    eta_pf  = np.array([res_pf[3][-1][0][0], res_pf[3][-1][0][1], res_pf[3][-1][1]])
    eta_chk = np.array([res_chk[3][-1][0][0], res_chk[3][-1][0][1], res_chk[3][-1][1]])
    np.testing.assert_allclose(eta_pf, eta_chk, atol=1e-8,
                               err_msg="Ship pose differs (with replan callback)")

    print(f"  PASS: with replan callback, chunked matches per-frame (n_replans={n_replans_pf})")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        ("test_run_jax_segment_matches_full_scan",
         "1. _run_jax_segment single-chunk matches simulate_full_jax_scan",
         test_run_jax_segment_matches_full_scan),
        ("test_extract_world_polygons_matches_loop",
         "2. _extract_world_polygons_np matches Python loop",
         test_extract_world_polygons_matches_loop),
        ("test_chunked_no_replan_matches_full_scan",
         "3. simulate_jax_chunked_replan (no replan, single chunk) matches simulate_full_jax_scan",
         test_chunked_no_replan_matches_full_scan),
        ("test_chunked_multi_chunk_matches_full_scan",
         "4. Multi-chunk (with padding) matches simulate_full_jax_scan",
         test_chunked_multi_chunk_matches_full_scan),
        ("test_chunked_with_replan_matches_per_frame",
         "5. With replan callback, chunked matches jax_simulate_ice_ship_interaction",
         test_chunked_with_replan_matches_per_frame),
    ]

    passed = []
    failed = []

    for name, desc, fn in tests:
        print(f"\n[{name}]")
        print(f"  {desc}")
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
