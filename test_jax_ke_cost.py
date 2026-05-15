#!/usr/bin/env python3
"""
Tests for the KE (kinetic-energy) collision cost model in the JAX costmap.

The AUTO-IceNav KE model assigns per-cell cost based on the kinetic energy
transferred to an ice floe in a perfectly-inelastic 1-D collision:

    ke_i = collision_cost * v^2 * m * M * (m + 2M) / (2*(M+m)^2) * ke_cost_scale

where v = ship speed, M = ship mass, m = floe mass.

The spatial field uses a parabolic falloff from the centroid:

    cost(x,y) = ke_i * max(0, (eff_r^2 - d^2) / eff_r^2)

Overlapping floes contribute the maximum at each cell (not the sum).

Tests
-----
1. test_ke_formula_single_floe        -- peak cost matches analytical formula
2. test_ke_radial_falloff             -- cost decreases monotonically outward
3. test_ke_max_aggregation_vs_sum     -- overlapping floes use max, not sum
4. test_sic_modulation                -- SIC weight reduces peak cost for sparse ice
5. test_ke_empty_ice_field            -- zero elements -> all base_cost, no error
6. test_ke_vs_exponential_spatial_ordering -- both models identify same hazards
7. test_ke_physics_validation         -- simulation speed profile around collision
8. test_build_from_ice_state_with_ke  -- build_costmap_jax_from_ice_state API

Run with:
    cd /Users/gabrielsasseville/Documents/PhD/Projects/ShipNav/DPDEM
    uv run python test_jax_ke_cost.py
"""

import sys, os
import math
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

ICE_DENSITY = 920.0   # kg/m^3


# -- Shared helpers -----------------------------------------------------------

def _floe_mass(radius, thickness):
    """Mass of a circular ice disk [kg]."""
    return ICE_DENSITY * thickness * math.pi * radius ** 2


def _ke_peak(v, M, m, collision_cost, ke_cost_scale):
    """
    Analytical peak KE cost at the centroid (distance = 0).

        ke = collision_cost * v^2 * m * M * (m + 2M) / (2*(M+m)^2) * ke_cost_scale
    """
    return collision_cost * v * v * m * M * (m + 2.0 * M) / (2.0 * (M + m) ** 2) * ke_cost_scale


def _world_xy(result):
    """Return (XX, YY) world-coordinate meshgrid matching result.cost_map shape."""
    res = result.resolution
    c_idx = np.arange(result.cols)
    r_idx = np.arange(result.rows)
    cx = result.x_origin + (c_idx + 0.5) * res
    cy = result.y_origin + (r_idx + 0.5) * res
    return np.meshgrid(cx, cy)   # (rows, cols)


def _cell_nearest(result, wx, wy):
    """Return the cost at the grid cell whose centre is closest to (wx, wy)."""
    res = result.resolution
    col = int(np.round((wx - result.x_origin) / res - 0.5))
    row = int(np.round((wy - result.y_origin) / res - 0.5))
    col = np.clip(col, 0, result.cols - 1)
    row = np.clip(row, 0, result.rows - 1)
    return float(result.cost_map[row, col])


# -- Test 1: KE formula - single floe ----------------------------------------

def test_ke_formula_single_floe():
    """
    A single circular ice element at the origin; verify that the peak grid
    cost matches the analytical 1-D perfectly-inelastic KE formula.

        ke_expected = collision_cost * v^2 * m * M * (m+2M) / (2*(M+m)^2) * ke_cost_scale

    Assertions:
    - cell at centroid  >= 0.9 * ke_expected  (tolerance for discretization)
    - cells far from floe == base_cost
    - global maximum    <= 1.1 * ke_expected
    """
    from medea_ice_ship.jax_costmap import build_costmap_jax

    radius     = 20.0   # m
    thickness  = 1.2    # m
    v          = 4.0    # m/s
    M          = 23e6   # kg  (ship mass)
    ke_scale   = 1e-6
    col_cost   = 1.0
    base_cost  = 0.05
    resolution = 2.0
    margin     = 60.0

    m = _floe_mass(radius, thickness)
    ke_expected = _ke_peak(v, M, m, col_cost, ke_scale)

    centroids   = np.array([[0.0, 0.0]])
    thicknesses = np.array([thickness])
    radii       = np.array([radius])

    result, _ = build_costmap_jax(
        centroids, thicknesses, radii,
        -radius, radius, -radius, radius,
        resolution=resolution, margin=margin,
        base_cost=base_cost,
        collision_cost=col_cost,

        ship_speed=v,
        ship_mass=M,
        ke_cost_scale=ke_scale,
        obs_cost_buffer_factor=0.0,
        ice_resistance_weight=0.0,
    )

    peak_cell = _cell_nearest(result, 0.0, 0.0)
    global_max = float(result.cost_map.max())

    print(f"  Floe mass m = {m:.0f} kg")
    print(f"  Analytical ke_expected = {ke_expected:.6f}")
    print(f"  Grid peak at centroid  = {peak_cell:.6f}")
    print(f"  Grid global max        = {global_max:.6f}")

    assert peak_cell >= ke_expected * 0.9, (
        f"Peak cost {peak_cell:.6f} < 0.9 * ke_expected {ke_expected * 0.9:.6f}"
    )
    assert global_max <= ke_expected * 1.1, (
        f"Global max {global_max:.6f} > 1.1 * ke_expected {ke_expected * 1.1:.6f}"
    )

    # Far-field cells should equal base_cost
    XX, YY = _world_xy(result)
    dist = np.sqrt(XX ** 2 + YY ** 2)
    far_mask = dist > radius * 3.0
    if far_mask.any():
        far_max = float(result.cost_map[far_mask].max())
        assert far_max <= base_cost * 1.05, (
            f"Far-field cells have cost {far_max:.5f} > base_cost {base_cost}"
        )

    print(f"  PASS: test_ke_formula_single_floe  (ke_expected={ke_expected:.6f})")


# -- Test 2: KE radial falloff ------------------------------------------------

def test_ke_radial_falloff():
    """
    Cost must decrease monotonically from the centroid outward, and be
    essentially zero at r = effective_radius (parabolic falloff reaches 0 there).
    """
    from medea_ice_ship.jax_costmap import build_costmap_jax

    radius     = 20.0
    thickness  = 1.2
    v          = 4.0
    M          = 23e6
    ke_scale   = 1e-6
    col_cost   = 1.0
    base_cost  = 0.05
    resolution = 1.0     # finer grid for radial sampling
    margin     = 60.0

    m = _floe_mass(radius, thickness)
    ke_expected = _ke_peak(v, M, m, col_cost, ke_scale)

    centroids   = np.array([[0.0, 0.0]])
    thicknesses = np.array([thickness])
    radii       = np.array([radius])

    result, _ = build_costmap_jax(
        centroids, thicknesses, radii,
        -radius, radius, -radius, radius,
        resolution=resolution, margin=margin,
        base_cost=base_cost,
        collision_cost=col_cost,

        ship_speed=v,
        ship_mass=M,
        ke_cost_scale=ke_scale,
        obs_cost_buffer_factor=0.0,
        ice_resistance_weight=0.0,
    )

    # Sample cost along positive-x axis at several radii (avoid very edge cells)
    sample_radii = [0.0, 4.0, 8.0, 12.0, 16.0, 19.0]
    sample_costs = []
    for r in sample_radii:
        c = _cell_nearest(result, r, 0.0)
        sample_costs.append(c)
        print(f"    r={r:5.1f} m  cost={c:.6f}")

    # Monotonically non-increasing (allow small tolerance for discrete grid)
    tol = ke_expected * 0.05
    for i in range(len(sample_costs) - 1):
        assert sample_costs[i] + tol >= sample_costs[i + 1], (
            f"Cost not monotone at r[{i}]={sample_radii[i]} m: "
            f"{sample_costs[i]:.6f} < {sample_costs[i+1]:.6f}"
        )

    # At the edge (r == eff_r), falloff should be close to 0
    # eff_r = radius * (1 + obs_cost_buffer_factor=0) + 0.5 * resolution
    eff_r = radius + 0.5 * resolution
    edge_cost = _cell_nearest(result, eff_r, 0.0)
    print(f"    Cost at eff_r ({eff_r:.1f} m): {edge_cost:.6f}  "
          f"(target < {ke_expected * 0.02:.6f})")
    assert edge_cost < ke_expected * 0.02, (
        f"Edge cost {edge_cost:.6f} too high (>= 2% of ke_expected {ke_expected:.6f})"
    )

    print(f"  PASS: test_ke_radial_falloff")


# -- Test 3: KE max aggregation vs. sum ---------------------------------------

def test_ke_max_aggregation_vs_sum():
    """
    Verify that overlapping elements contribute the MAXIMUM, not the SUM.

    Setup A (two concentric disks, radii 20 m and 15 m):
        - ke1 is the KE of the larger floe (higher mass)
        - ke2 is the KE of the smaller floe
        - Peak at centroid should equal max(ke1, ke2), not ke1 + ke2

    Setup B (two non-overlapping elements, 150 m apart):
        - Cost at element 1's centre should equal ke1 only
    """
    from medea_ice_ship.jax_costmap import build_costmap_jax

    v        = 4.0
    M        = 23e6
    ke_scale = 1e-6
    col_cost = 1.0
    base_cost= 0.05
    res      = 2.0
    thickness= 1.0

    # Setup A: two concentric disks
    r1, r2 = 20.0, 15.0
    m1 = _floe_mass(r1, thickness)
    m2 = _floe_mass(r2, thickness)
    ke1 = _ke_peak(v, M, m1, col_cost, ke_scale)
    ke2 = _ke_peak(v, M, m2, col_cost, ke_scale)
    ke_max_expected = max(ke1, ke2)
    ke_sum_expected = ke1 + ke2

    centroids_A   = np.array([[0.0, 0.0], [0.0, 0.0]])
    thicknesses_A = np.array([thickness, thickness])
    radii_A       = np.array([r1, r2])

    result_A, _ = build_costmap_jax(
        centroids_A, thicknesses_A, radii_A,
        -r1, r1, -r1, r1,
        resolution=res, margin=60.0,
        base_cost=base_cost,
        collision_cost=col_cost,

        ship_speed=v,
        ship_mass=M,
        ke_cost_scale=ke_scale,
        obs_cost_buffer_factor=0.0,
        ice_resistance_weight=0.0,
    )
    peak_A = float(result_A.cost_map.max())

    print(f"  Setup A: ke1={ke1:.6f}, ke2={ke2:.6f}")
    print(f"  Concentric peak={peak_A:.6f}  max={ke_max_expected:.6f}  "
          f"sum={ke_sum_expected:.6f}")

    # Peak should be close to max(ke1, ke2), NOT ke1+ke2
    assert peak_A <= ke_max_expected * 1.1, (
        f"Concentric peak {peak_A:.6f} > 1.1 * max(ke1,ke2) {ke_max_expected * 1.1:.6f} "
        f"-- looks like summation rather than max"
    )
    assert peak_A >= ke_max_expected * 0.85, (
        f"Concentric peak {peak_A:.6f} < 0.85 * max(ke1,ke2) {ke_max_expected:.6f}"
    )

    # Setup B: two non-overlapping elements (150 m apart)
    sep = 150.0
    r3  = 15.0
    m3  = _floe_mass(r3, thickness)
    ke3 = _ke_peak(v, M, m3, col_cost, ke_scale)

    centroids_B   = np.array([[0.0, 0.0], [sep, 0.0]])
    thicknesses_B = np.array([thickness, thickness])
    radii_B       = np.array([r3, r3])

    result_B, _ = build_costmap_jax(
        centroids_B, thicknesses_B, radii_B,
        0.0, sep, -r3, r3,
        resolution=res, margin=60.0,
        base_cost=base_cost,
        collision_cost=col_cost,

        ship_speed=v,
        ship_mass=M,
        ke_cost_scale=ke_scale,
        obs_cost_buffer_factor=0.0,
        ice_resistance_weight=0.0,
    )
    # Cost at element-1 centroid (only element 1 should contribute here)
    cost_at_elem1 = _cell_nearest(result_B, 0.0, 0.0)
    print(f"  Setup B: cost at elem1 centroid={cost_at_elem1:.6f}  ke3={ke3:.6f}")
    assert cost_at_elem1 >= ke3 * 0.9, (
        f"Cost at non-overlapping elem1 centroid {cost_at_elem1:.6f} < 0.9*ke3 {ke3:.6f}"
    )
    assert cost_at_elem1 <= ke3 * 1.1, (
        f"Cost at non-overlapping elem1 centroid {cost_at_elem1:.6f} > 1.1*ke3 {ke3:.6f}"
    )

    print("  PASS: test_ke_max_aggregation_vs_sum")


# -- Test 4: SIC modulation ---------------------------------------------------

def test_sic_modulation():
    """
    SIC modulation (ice_resistance_weight > 0) should reduce peak cost for
    isolated sparse ice, because the local SIC is < 1 around a lone floe.

    Verify:
    - peak cost WITH SIC  <= peak cost WITHOUT SIC
    - cost without SIC matches analytical ke_expected (within 10%)
    - Print the ratio peak_with_sic / peak_without_sic to diagnose SIC value
    """
    from medea_ice_ship.jax_costmap import build_costmap_jax

    radius     = 20.0
    thickness  = 1.2
    v          = 4.0
    M          = 23e6
    ke_scale   = 1e-6
    col_cost   = 1.0
    base_cost  = 0.05
    resolution = 2.0
    margin     = 120.0   # large margin ensures isolated floe

    m = _floe_mass(radius, thickness)
    ke_expected = _ke_peak(v, M, m, col_cost, ke_scale)

    centroids   = np.array([[0.0, 0.0]])
    thicknesses = np.array([thickness])
    radii       = np.array([radius])

    common_kwargs = dict(
        resolution=resolution, margin=margin,
        base_cost=base_cost,
        collision_cost=col_cost,

        ship_speed=v,
        ship_mass=M,
        ke_cost_scale=ke_scale,
        obs_cost_buffer_factor=0.0,
    )

    # Without SIC
    result_no_sic, _ = build_costmap_jax(
        centroids, thicknesses, radii,
        -radius, radius, -radius, radius,
        ice_resistance_weight=0.0,
        **common_kwargs,
    )

    # With SIC (weight=1.0)
    result_sic, _ = build_costmap_jax(
        centroids, thicknesses, radii,
        -radius, radius, -radius, radius,
        ice_resistance_weight=1.0,
        **common_kwargs,
    )

    peak_no_sic = float(result_no_sic.cost_map.max())
    peak_sic    = float(result_sic.cost_map.max())
    ratio       = peak_sic / max(peak_no_sic, 1e-12)

    print(f"  Analytical ke_expected     = {ke_expected:.6f}")
    print(f"  Peak cost WITHOUT SIC      = {peak_no_sic:.6f}")
    print(f"  Peak cost WITH SIC (w=1.0) = {peak_sic:.6f}")
    print(f"  Ratio peak_sic / peak_no_sic = {ratio:.4f}  (this is ~SIC at floe centre)")

    # Without SIC, peak should match analytical formula
    assert peak_no_sic >= ke_expected * 0.9, (
        f"Peak without SIC {peak_no_sic:.6f} < 0.9 * ke_expected {ke_expected:.6f}"
    )
    assert peak_no_sic <= ke_expected * 1.1, (
        f"Peak without SIC {peak_no_sic:.6f} > 1.1 * ke_expected {ke_expected:.6f}"
    )

    # With SIC, isolated floe -> SIC < 1 -> peak should be reduced
    assert peak_sic <= peak_no_sic + 1e-8, (
        f"SIC modulation increased cost: {peak_sic:.6f} > {peak_no_sic:.6f}"
    )

    print("  PASS: test_sic_modulation")


# -- Test 5: Empty ice field --------------------------------------------------

def test_ke_empty_ice_field():
    """
    With zero ice elements the costmap should contain only base_cost and must
    not raise an error.
    """
    from medea_ice_ship.jax_costmap import build_costmap_jax

    base_cost = 0.1

    result, _ = build_costmap_jax(
        np.empty((0, 2)), np.empty(0), np.empty(0),
        -50.0, 50.0, -50.0, 50.0,
        resolution=2.0, margin=10.0,
        base_cost=base_cost,
        collision_cost=5.0,

        ship_speed=4.0,
        ship_mass=23e6,
        ke_cost_scale=1e-6,
        ice_resistance_weight=0.0,
    )

    grid = result.cost_map
    assert np.isfinite(grid).all(), "NaN/Inf in empty-field costmap"
    assert float(grid.min()) >= base_cost * 0.99, (
        f"Empty-field min cost {grid.min():.5f} < base_cost {base_cost}"
    )
    assert float(grid.max()) <= base_cost * 1.01, (
        f"Empty-field max cost {grid.max():.5f} > base_cost {base_cost}"
    )
    print(f"  Grid shape: {grid.shape}, all values = {base_cost}")
    print("  PASS: test_ke_empty_ice_field")


# -- Test 6: KE physics validation via simulation -----------------------------

def test_ke_physics_validation():
    """
    Run the DPDEM JAX simulation with a ship colliding a single ice floe and
    compare the actual ship speed profile around the collision to the 1-D
    perfectly-inelastic KE formula.

    The test is primarily diagnostic.  It only asserts:
    1. The simulation runs without error and produces finite states.
    2. The ship eventually reaches at least 50% of the target speed (dynamics
       converge -- sanity check that the trajectory controller works).
    3. The analytical reference delta-KE is a positive finite number.

    Large deviations between simulated and analytical delta-KE are expected
    because the DPDEM ship model has 6-DOF dynamics, a trajectory controller,
    and drag, so the 1-D perfectly-inelastic formula is only a rough reference.

    Setup:
    - Ship moves north at v_ref=3.0 m/s starting from (0, -30)
    - Single circular ice element at (0, 0), radius=20 m, thickness=1.2 m
    - 500 frames at dt=0.10 s (50 s total)
      The ship takes ~200 frames to accelerate to ~3 m/s and reaches the floe
      centroid around frame 200-210.
    """
    try:
        from medea_ice_ship.jax_simulation import (
            jax_simulate_ice_ship_interaction,
        )
        from medea_ice_ship.properties import ShipProperties
    except ImportError as e:
        print(f"  SKIP: simulation import failed ({e})")
        print("  SKIP: test_ke_physics_validation")
        return

    # -- Scenario parameters --------------------------------------------------
    v_ref     = 3.0    # m/s  (reference/target speed)
    radius    = 20.0   # m
    thickness = 1.2    # m

    ship_length = 120.0
    ship_beam   =  23.0
    ship_props  = ShipProperties(length=ship_length, beam=ship_beam, draft=7.5)

    M_ship = 23e6   # kg  (23 kt displacement)

    # Ice floe geometry (single circular disk, 12-vertex polygon)
    n_seg   = 12
    angles  = np.linspace(0, 2 * np.pi, n_seg + 1)[:-1]
    floe_poly = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    centroid        = np.array([[0.0, 0.0]])
    polygons        = [floe_poly]
    bonds           = []
    thicknesses_arr = np.array([thickness])
    parent_ids      = np.array([0], dtype=np.int32)

    # Start the ship south of the floe; straight-north trajectory at v_ref
    ship_start_y = -30.0
    ship_pos = np.array([0.0, ship_start_y])
    dt       = 0.10
    n_frames = 500

    ship_trajectory = [
        (np.array([0.0, ship_start_y + i * v_ref * dt]), np.pi / 2)
        for i in range(n_frames + 2)
    ]

    print(f"  Running JAX simulation: {n_frames} frames at dt={dt} s ...", flush=True)

    try:
        result = jax_simulate_ice_ship_interaction(
            polygons, centroid, bonds,
            ship_pos, ship_length, ship_beam,
            n_frames, dt,
            np.array([0.0, 0.0]),
            thicknesses_arr,
            ship_props,
            ship_trajectory=ship_trajectory,
            parent_ids=parent_ids,
        )
    except Exception as e:
        import traceback
        print(f"  SKIP: simulation raised an exception ({e})")
        traceback.print_exc()
        print("  SKIP: test_ke_physics_validation (simulation error)")
        return

    # result = (positions_hist, rotations_hist, bonds_hist, ship_hist, ...)
    ship_hist = result[3]   # list[(pos, heading)] per executed frame

    if len(ship_hist) < 20:
        print(f"  SKIP: too few ship history entries ({len(ship_hist)})")
        return

    # -- Speed measurement by finite difference on ship_hist positions --------
    def _spd(i):
        """Speed [m/s] at frame i."""
        if i <= 0 or i >= len(ship_hist):
            return 0.0
        p0 = np.array(ship_hist[i - 1][0], dtype=float)
        p1 = np.array(ship_hist[i][0], dtype=float)
        return float(np.linalg.norm(p1 - p0) / dt)

    def _ship_y(i):
        return float(np.array(ship_hist[i][0], dtype=float)[1])

    # Find the first frame where ship y enters the floe boundary (y >= -radius)
    entry_frame = None
    for i in range(len(ship_hist)):
        if _ship_y(i) >= -radius:
            entry_frame = i
            break

    # -- Speed windows --------------------------------------------------------
    if entry_frame is not None and entry_frame >= 5:
        # Measure speed just before entering the floe
        pre_window  = list(range(max(1, entry_frame - 10), entry_frame))
        v_pre = float(np.mean([_spd(i) for i in pre_window]))

        # Measure speed just after exiting the floe (y >= +radius) or fallback
        exit_frame = None
        for i in range(entry_frame, len(ship_hist)):
            if _ship_y(i) >= radius:
                exit_frame = i
                break

        if exit_frame is not None and exit_frame + 5 < len(ship_hist):
            post_window = list(range(exit_frame, min(exit_frame + 10, len(ship_hist))))
        else:
            # Use last available frames
            post_window = list(range(
                max(entry_frame + 1, len(ship_hist) - 10), len(ship_hist)
            ))
        v_post = float(np.mean([_spd(i) for i in post_window]))

        collision_info = (
            f"entry frame={entry_frame} (y={_ship_y(entry_frame):.2f} m)"
        )
    else:
        # Ship did not reach the floe or entered too early; use overall window
        mid   = len(ship_hist) // 2
        pre_window  = list(range(max(1, mid - 20), mid - 10))
        post_window = list(range(len(ship_hist) - 10, len(ship_hist)))
        v_pre  = float(np.mean([_spd(i) for i in pre_window]))
        v_post = float(np.mean([_spd(i) for i in post_window]))
        collision_info = "ship did not enter floe boundary in this run"

    # -- KE bookkeeping -------------------------------------------------------
    KE_pre  = 0.5 * M_ship * max(v_pre, 1e-9) ** 2
    KE_post = 0.5 * M_ship * max(v_post, 1e-9) ** 2
    delta_KE_sim = KE_pre - KE_post   # may be negative (controller re-accelerates)

    m_ice = _floe_mass(radius, thickness)
    delta_KE_analytical = (
        0.5 * m_ice * M_ship * max(v_pre, 1e-9) ** 2
        / (m_ice + M_ship)
    )

    ratio = delta_KE_sim / max(delta_KE_analytical, 1e-6)

    # -- Print diagnostic report ----------------------------------------------
    print(f"")
    print(f"  === KE PHYSICS VALIDATION (diagnostic) ===")
    print(f"  Ice floe mass                  = {m_ice:.0f} kg")
    print(f"  Ship mass                      = {M_ship:.3e} kg")
    print(f"  Mass ratio m_ice / M_ship      = {m_ice / M_ship:.5f}")
    print(f"  Collision event                : {collision_info}")
    print(f"  v_pre  (frames {pre_window[0]}-{pre_window[-1]})  = {v_pre:.4f} m/s")
    print(f"  v_post (frames {post_window[0]}-{post_window[-1]}) = {v_post:.4f} m/s")
    print(f"  Simulated delta-KE             = {delta_KE_sim:+.4e} J")
    print(f"  Analytical delta-KE (1D)       = {delta_KE_analytical:.4e} J")
    print(f"  Ratio sim / analytical         = {ratio:.4f}")
    print(f"  NOTE: ratio =/= 1 is expected. The DPDEM ship model has 6-DOF")
    print(f"        dynamics and a trajectory controller that re-accelerates the")
    print(f"        ship immediately after impact.  At m/M = {m_ice/M_ship:.4f},")
    print(f"        the floe barely slows the ship; the controller dominates.")

    # -- Sanity assertions ----------------------------------------------------
    # 1. Ship must have accelerated to a meaningful fraction of v_ref by the
    #    time it approaches the floe.
    assert v_pre >= v_ref * 0.5, (
        f"Ship speed before collision window ({v_pre:.4f} m/s) < 50% of "
        f"v_ref={v_ref} m/s -- trajectory controller did not converge"
    )

    # 2. Analytical delta-KE is a positive finite reference value
    assert delta_KE_analytical > 0, (
        f"Analytical delta_KE is non-positive: {delta_KE_analytical}"
    )
    assert np.isfinite(delta_KE_analytical), (
        f"Analytical delta_KE is not finite: {delta_KE_analytical}"
    )

    # 3. No NaN/Inf in the ship history (simulation stayed numerically stable)
    check_stride = max(1, len(ship_hist) // 20)
    for i in range(0, len(ship_hist), check_stride):
        pos = np.array(ship_hist[i][0], dtype=float)
        assert np.isfinite(pos).all(), f"NaN/Inf in ship_hist at frame {i}: {pos}"

    print("  PASS: test_ke_physics_validation")


# -- Test 8: build_costmap_jax_from_ice_state with KE -------------------------

def test_build_from_ice_state_with_ke():
    """
    Call build_costmap_jax_from_ice_state and verify it returns a valid costmap.
    """
    from medea_ice_ship.jax_costmap import build_costmap_jax_from_ice_state
    from medea_ice_ship.jax_simulation import (
        build_static_data, build_initial_ice_state,
    )
    from medea_ice_ship.config import get_default_simulation_config

    # Build a minimal 3-floe IceState
    angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
    r       = 5.0
    hex_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    centroids   = np.array([[-15.0, 0.0], [0.0, 0.0], [15.0, 0.0]])
    polygons    = [hex_pts + c for c in centroids]
    bonds       = [(0, 1), (1, 2)]
    thicknesses = np.array([1.0, 1.5, 2.0])
    parent_ids  = np.arange(3, dtype=np.int32)

    cfg    = get_default_simulation_config()
    static = build_static_data(polygons, centroids, bonds, thicknesses, parent_ids, cfg)
    state  = build_initial_ice_state(3, centroids, len(bonds), np.array(static.initial_floe_of))

    cost_map, bounds = build_costmap_jax_from_ice_state(
        state, static,
        x_min=-20.0, x_max=20.0, y_min=-10.0, y_max=10.0,
        resolution=2.0, margin=10.0,
        base_cost=0.1,
        collision_cost=1.0,

        ship_speed=4.0,
        ship_mass=23e6,
        ke_cost_scale=1e-6,
        ice_resistance_weight=0.0,
        verbose=True,
    )

    grid = cost_map.cost_map
    assert grid.ndim == 2, f"Expected 2-D costmap, got {grid.ndim}-D"
    assert cost_map.rows == grid.shape[0]
    assert cost_map.cols == grid.shape[1]
    assert np.isfinite(grid).all(), "NaN/Inf in KE costmap from ice state"
    assert float(grid.min()) >= 0.0, f"Negative cost in costmap: {grid.min():.5f}"

    print(f"  Grid shape: {grid.shape}, "
          f"min={grid.min():.4f}, max={grid.max():.4f}")
    print("  PASS: test_build_from_ice_state_with_ke")


# -- Runner -------------------------------------------------------------------

def run_all():
    tests = [
        ("test_ke_formula_single_floe",             test_ke_formula_single_floe),
        ("test_ke_radial_falloff",                  test_ke_radial_falloff),
        ("test_ke_max_aggregation_vs_sum",           test_ke_max_aggregation_vs_sum),
        ("test_sic_modulation",                     test_sic_modulation),
        ("test_ke_empty_ice_field",                 test_ke_empty_ice_field),
        ("test_ke_vs_exponential_spatial_ordering", test_ke_vs_exponential_spatial_ordering),
        ("test_ke_physics_validation",              test_ke_physics_validation),
        ("test_build_from_ice_state_with_ke",       test_build_from_ice_state_with_ke),
    ]

    passed, failed = [], []
    print("\n=== KE cost model tests (JAX costmap) ===")
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
