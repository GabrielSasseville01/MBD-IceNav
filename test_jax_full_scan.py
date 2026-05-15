#!/usr/bin/env python3
"""
Tests for JAX-based connected components and full-scan simulation.

Run with:
    cd /Users/gabrielsasseville/Documents/PhD/Projects/ShipNav/DPDEM
    uv run python test_jax_full_scan.py
"""

import sys
import os

# Ensure float64 precision before any JAX import
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))

from collections import defaultdict


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bfs_partition(components):
    """Convert BFS result (list of lists) → {min_elem: sorted_members}."""
    return {min(comp): sorted(comp) for comp in components}


def _label_partition(labels_arr):
    """Convert (n,) label array → {min_elem: sorted_members}."""
    groups = defaultdict(list)
    for i, lbl in enumerate(int(x) for x in np.array(labels_arr)):
        groups[lbl].append(i)
    return {min(v): sorted(v) for v in groups.values()}


def assert_same_partition(bfs_result, jax_labels, msg=""):
    """Assert that BFS and JAX label propagation produce identical component sets."""
    bfs_part = _bfs_partition(bfs_result)
    jax_part = _label_partition(jax_labels)
    assert bfs_part == jax_part, (
        f"{msg}\nBFS:  {bfs_part}\nJAX:  {jax_part}"
    )


def _make_bonds(edges, all_intact=True):
    """Build (bond_i, bond_j, bond_intact) NumPy arrays from an edge list."""
    if not edges:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.ones(0, dtype=bool),
        )
    bi = np.array([e[0] for e in edges], dtype=np.int32)
    bj = np.array([e[1] for e in edges], dtype=np.int32)
    intact = np.full(len(edges), all_intact, dtype=bool)
    return bi, bj, intact


# ── Part 1: find_connected_components_jax ─────────────────────────────────────

def test_find_connected_components_chain():
    """Chain 0-1-2-3: all in one component."""
    from medea_ice_ship.bonds import find_connected_components
    from medea_ice_ship.jax_bonds import find_connected_components_jax

    n = 4
    bi, bj, intact = _make_bonds([(0, 1), (1, 2), (2, 3)])
    bfs  = find_connected_components(n, list(zip(bi, bj)), intact)
    jlbl = find_connected_components_jax(n, jnp.array(bi), jnp.array(bj), jnp.array(intact))
    assert_same_partition(bfs, jlbl, "chain 0-1-2-3")
    # All labels should be 0 (min index)
    assert np.all(np.array(jlbl) == 0), f"expected all-0 labels, got {np.array(jlbl)}"
    print("  PASS: chain graph")


def test_find_connected_components_star():
    """Star graph: hub 0 with spokes 1,2,3."""
    from medea_ice_ship.bonds import find_connected_components
    from medea_ice_ship.jax_bonds import find_connected_components_jax

    n = 4
    bi, bj, intact = _make_bonds([(0, 1), (0, 2), (0, 3)])
    bfs  = find_connected_components(n, list(zip(bi, bj)), intact)
    jlbl = find_connected_components_jax(n, jnp.array(bi), jnp.array(bj), jnp.array(intact))
    assert_same_partition(bfs, jlbl, "star 0-{1,2,3}")
    assert np.all(np.array(jlbl) == 0), f"expected all-0 labels, got {np.array(jlbl)}"
    print("  PASS: star graph")


def test_find_connected_components_disconnected():
    """Two disconnected pairs: 0-1 and 2-3."""
    from medea_ice_ship.bonds import find_connected_components
    from medea_ice_ship.jax_bonds import find_connected_components_jax

    n = 4
    bi, bj, intact = _make_bonds([(0, 1), (2, 3)])
    bfs  = find_connected_components(n, list(zip(bi, bj)), intact)
    jlbl = find_connected_components_jax(n, jnp.array(bi), jnp.array(bj), jnp.array(intact))
    assert_same_partition(bfs, jlbl, "disconnected pairs {0,1} {2,3}")
    labels_np = np.array(jlbl)
    assert labels_np[0] == labels_np[1] == 0, f"pair 0-1 labels: {labels_np[:2]}"
    assert labels_np[2] == labels_np[3] == 2, f"pair 2-3 labels: {labels_np[2:]}"
    print("  PASS: disconnected pairs")


def test_find_connected_components_single_node():
    """Single node, no bonds."""
    from medea_ice_ship.bonds import find_connected_components
    from medea_ice_ship.jax_bonds import find_connected_components_jax

    n = 1
    bi, bj, intact = _make_bonds([])
    bfs  = find_connected_components(n, [], intact)
    jlbl = find_connected_components_jax(n, jnp.array(bi), jnp.array(bj), jnp.array(intact))
    assert_same_partition(bfs, jlbl, "single node")
    assert int(jlbl[0]) == 0
    print("  PASS: single node")


def test_find_connected_components_broken_bonds():
    """All bonds broken: each node is its own component."""
    from medea_ice_ship.bonds import find_connected_components
    from medea_ice_ship.jax_bonds import find_connected_components_jax

    n = 3
    bi = np.array([0, 1], dtype=np.int32)
    bj = np.array([1, 2], dtype=np.int32)
    intact_false = np.zeros(2, dtype=bool)
    intact_true  = np.ones(2, dtype=bool)

    # BFS with intact=True bonds for reference (then compare both broken)
    bfs_broken = find_connected_components(n, [(0, 1), (1, 2)], intact_false)
    jlbl_broken = find_connected_components_jax(
        n, jnp.array(bi), jnp.array(bj), jnp.array(intact_false)
    )
    assert_same_partition(bfs_broken, jlbl_broken, "all bonds broken")
    labels_np = np.array(jlbl_broken)
    assert labels_np[0] == 0 and labels_np[1] == 1 and labels_np[2] == 2, (
        f"expected [0,1,2] for isolated nodes, got {labels_np}"
    )
    print("  PASS: broken bonds → isolated nodes")


def test_find_connected_components_larger_chain():
    """Long chain of 20 nodes: tests convergence with more than diameter iterations."""
    from medea_ice_ship.bonds import find_connected_components
    from medea_ice_ship.jax_bonds import find_connected_components_jax

    n = 20
    edges = [(i, i + 1) for i in range(n - 1)]
    bi, bj, intact = _make_bonds(edges)
    bfs  = find_connected_components(n, list(zip(bi, bj)), intact)
    jlbl = find_connected_components_jax(n, jnp.array(bi), jnp.array(bj), jnp.array(intact))
    assert_same_partition(bfs, jlbl, f"chain of {n} nodes")
    assert np.all(np.array(jlbl) == 0), f"expected all-0, got {np.array(jlbl)}"
    print(f"  PASS: large chain ({n} nodes)")


# ── Part 2: component_com_velocities_jax ─────────────────────────────────────

def test_component_com_velocities():
    """COM velocity: two components of equal mass should return their mean velocity."""
    from medea_ice_ship.jax_bonds import component_com_velocities_jax

    # Two components: {0,1} and {2,3}
    labels = jnp.array([0, 0, 2, 2], dtype=jnp.int32)
    masses = jnp.array([1.0, 1.0, 2.0, 2.0])
    vels   = jnp.array([[1.0, 0.0], [3.0, 0.0],   # comp 0: COM = (2, 0)
                         [0.0, 1.0], [0.0, 3.0]])   # comp 2: COM = (0, 2)

    v_com = component_com_velocities_jax(labels, masses, vels)
    v_com_np = np.array(v_com)

    np.testing.assert_allclose(v_com_np[0], [2.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(v_com_np[1], [2.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(v_com_np[2], [0.0, 2.0], atol=1e-10)
    np.testing.assert_allclose(v_com_np[3], [0.0, 2.0], atol=1e-10)
    print("  PASS: component COM velocities")


# ── Part 3: simulate_full_jax_scan matches jax_simulate_ice_ship_interaction ──

def _make_3floe_scenario(n_frames=30, dt=0.05):
    """
    Minimal 3-floe ice field for fast simulation tests.

    Three regular hexagons in a line; ship starts west and moves east.
    """
    from medea_ice_ship.properties import ShipProperties

    # Regular hexagon
    angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
    r       = 3.0
    hex_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    centroids   = np.array([[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    polygons    = [hex_pts + c for c in centroids]
    bonds       = [(0, 1), (1, 2)]
    thicknesses = np.array([1.0, 1.0, 1.0])
    parent_ids  = np.arange(3, dtype=np.int32)

    ship_length, ship_beam = 15.0, 5.0
    ship_props = ShipProperties(length=ship_length, beam=ship_beam, draft=2.5)
    ship_pos   = np.array([-25.0, 0.0])

    # Straight-east trajectory (one step per frame + one extra for lookahead)
    ship_trajectory = [
        (np.array([-25.0 + i * 0.5, 0.0]), 0.0)
        for i in range(n_frames + 2)
    ]

    return dict(
        polygons=polygons,
        centroids=centroids,
        bonds=bonds,
        ship_pos=ship_pos,
        ship_length=ship_length,
        ship_beam=ship_beam,
        n_frames=n_frames,
        dt=dt,
        drift_vel=np.array([0.0, 0.0]),
        thicknesses=thicknesses,
        ship_props=ship_props,
        ship_trajectory=ship_trajectory,
        parent_ids=parent_ids,
    )


def test_full_scan_matches_per_frame():
    """
    Full-scan and per-frame JAX simulations must produce identical results.

    Because both paths execute the same _make_frame_step factory and the same
    XLA operations, results should match to floating-point precision.
    """
    from medea_ice_ship.jax_simulation import (
        jax_simulate_ice_ship_interaction as sim_per_frame,
        simulate_full_jax_scan             as sim_scan,
    )

    sc = _make_3floe_scenario(n_frames=20, dt=0.05)

    print("  Running per-frame simulation...", flush=True)
    res_pf = sim_per_frame(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=sc["ship_trajectory"],
        parent_ids=sc["parent_ids"],
    )

    print("  Running full-scan simulation...", flush=True)
    res_fs = sim_scan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=sc["ship_trajectory"],
        parent_ids=sc["parent_ids"],
    )

    pos_pf = np.array(res_pf[0][-1])   # final positions
    pos_fs = np.array(res_fs[0][-1])

    np.testing.assert_allclose(
        pos_pf, pos_fs, atol=1e-8, rtol=1e-8,
        err_msg="Final ice positions differ between per-frame and full-scan paths",
    )

    rot_pf = np.array(res_pf[1][-1])
    rot_fs = np.array(res_fs[1][-1])
    np.testing.assert_allclose(rot_pf, rot_fs, atol=1e-8, rtol=1e-8,
                               err_msg="Final rotations differ")

    bond_pf = np.array(res_pf[2][-1])
    bond_fs = np.array(res_fs[2][-1])
    np.testing.assert_array_equal(bond_pf, bond_fs,
                                  err_msg="Bond intact states differ")

    # Ship final pose
    eta_pf = np.array([res_pf[3][-1][0][0], res_pf[3][-1][0][1], res_pf[3][-1][1]])
    eta_fs = np.array([res_fs[3][-1][0][0], res_fs[3][-1][0][1], res_fs[3][-1][1]])
    np.testing.assert_allclose(eta_pf, eta_fs, atol=1e-8,
                               err_msg="Ship final poses differ")

    # History length: n_frames + 1 entries (initial + n_frames)
    assert len(res_pf[0]) == sc["n_frames"] + 1
    assert len(res_fs[0]) == sc["n_frames"] + 1

    print("  PASS: full-scan matches per-frame (ice positions, rotations, bonds, ship)")


def test_full_scan_no_trajectory():
    """Full-scan with ship_trajectory=None (drift-only, ship stationary)."""
    from medea_ice_ship.jax_simulation import simulate_full_jax_scan as sim_scan

    sc = _make_3floe_scenario(n_frames=5, dt=0.05)

    res = sim_scan(
        sc["polygons"], sc["centroids"], sc["bonds"],
        sc["ship_pos"], sc["ship_length"], sc["ship_beam"],
        sc["n_frames"], sc["dt"], sc["drift_vel"],
        sc["thicknesses"], sc["ship_props"],
        ship_trajectory=None,
        parent_ids=sc["parent_ids"],
    )

    assert len(res[0]) == sc["n_frames"] + 1
    # Ship should not have moved (stationary setpoint)
    eta_last = res[3][-1]
    pos_last = eta_last[0]
    ship_start = np.array(sc["ship_pos"])
    # The ship starts at ship_pos; without a trajectory it receives its own
    # initial pose as setpoint, so it should stay very close.
    assert np.linalg.norm(pos_last - ship_start) < 5.0, (
        f"Ship moved unexpectedly without trajectory: {pos_last} vs {ship_start}"
    )
    print("  PASS: full-scan no-trajectory (drift-only)")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all():
    tests_cc = [
        test_find_connected_components_chain,
        test_find_connected_components_star,
        test_find_connected_components_disconnected,
        test_find_connected_components_single_node,
        test_find_connected_components_broken_bonds,
        test_find_connected_components_larger_chain,
        test_component_com_velocities,
    ]
    tests_sim = [
        test_full_scan_matches_per_frame,
        test_full_scan_no_trajectory,
    ]

    print("\n=== Part 1: JAX connected components ===")
    for t in tests_cc:
        print(f"\n[{t.__name__}]")
        t()

    print("\n=== Part 2: simulate_full_jax_scan ===")
    for t in tests_sim:
        print(f"\n[{t.__name__}]")
        t()

    print("\n\nAll tests passed.")


if __name__ == "__main__":
    run_all()
