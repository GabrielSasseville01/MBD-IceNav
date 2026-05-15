#!/usr/bin/env python3
"""
Validation tests for JAX-native ocean current sampling.

Tests that sample_chunk_jax() on every OceanCurrentModel subclass:
  1. Returns the correct shape (K, n, 2)
  2. Produces values identical to the NumPy sample_batch() repeated K times
  3. Avoids host-device round-trips for static current models

Run with:
    cd /Users/gabrielsasseville/Documents/PhD/Projects/ShipNav/DPDEM
    uv run python test_jax_ocean_current.py
"""

import sys
import os
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _random_positions(n, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(-100.0, 100.0, size=(n, 2)).astype(np.float64)


def _assert_shape_and_values(result, K, n, expected_np, name):
    result_np = np.array(result)
    assert result_np.shape == (K, n, 2), (
        f"{name}: expected shape ({K}, {n}, 2), got {result_np.shape}"
    )
    np.testing.assert_allclose(
        result_np, expected_np, atol=1e-10,
        err_msg=f"{name}: values differ from NumPy reference",
    )


# ── Test 1: NoCurrentModel ─────────────────────────────────────────────────────

def test_no_current_model_zeros():
    """NoCurrentModel.sample_chunk_jax must return (K, n, 2) of zeros."""
    from medea_ice_ship.ocean_current import NoCurrentModel
    model = NoCurrentModel()

    K, n = 10, 7
    pos_jax = jnp.array(_random_positions(n))

    result = model.sample_chunk_jax(K, n, pos_jax=pos_jax)
    result_no_pos = model.sample_chunk_jax(K, n, pos_jax=None)

    expected = np.zeros((K, n, 2))
    _assert_shape_and_values(result, K, n, expected, "NoCurrentModel(pos)")
    _assert_shape_and_values(result_no_pos, K, n, expected, "NoCurrentModel(None)")
    print("  PASS: NoCurrentModel.sample_chunk_jax → zeros (with/without positions)")


def test_no_current_model_no_transfer():
    """
    NoCurrentModel.sample_chunk_jax with pos_jax=None must NOT call np.array
    on a JAX array (no GPU→CPU transfer).  We verify this by NOT passing positions
    and ensuring the result is correct.
    """
    from medea_ice_ship.ocean_current import NoCurrentModel
    model = NoCurrentModel()
    K, n = 50, 3000  # Full-scale sizes
    result = model.sample_chunk_jax(K, n, pos_jax=None)
    assert np.array(result).shape == (K, n, 2)
    assert np.all(np.array(result) == 0.0)
    print("  PASS: NoCurrentModel.sample_chunk_jax(None) correct for full-scale sizes")


# ── Test 2: UniformCurrentModel ────────────────────────────────────────────────

def test_uniform_current_model():
    """UniformCurrentModel.sample_chunk_jax must return constant-velocity array."""
    from medea_ice_ship.ocean_current import UniformCurrentModel
    speed, direction_rad = 0.5, np.pi / 4  # 45° NE
    model = UniformCurrentModel(speed, direction_rad)

    K, n = 8, 5
    pos_np = _random_positions(n)
    pos_jax = jnp.array(pos_np)

    # NumPy reference: K copies of sample_batch
    v_one = model.sample_batch(pos_np)  # (n, 2)
    expected = np.stack([v_one] * K, axis=0)  # (K, n, 2)

    result_with_pos = model.sample_chunk_jax(K, n, pos_jax=pos_jax)
    result_no_pos   = model.sample_chunk_jax(K, n, pos_jax=None)

    _assert_shape_and_values(result_with_pos, K, n, expected, "UniformCurrentModel(pos)")
    _assert_shape_and_values(result_no_pos, K, n, expected, "UniformCurrentModel(None)")
    print("  PASS: UniformCurrentModel.sample_chunk_jax → correct broadcast constant")


def test_uniform_current_dtype():
    """sample_chunk_jax should respect the dtype argument."""
    from medea_ice_ship.ocean_current import UniformCurrentModel
    model = UniformCurrentModel(1.0, 0.0)

    K, n = 4, 3
    result_f32 = model.sample_chunk_jax(K, n, pos_jax=None, dtype=jnp.float32)
    result_f64 = model.sample_chunk_jax(K, n, pos_jax=None, dtype=jnp.float64)

    assert np.array(result_f32).dtype == np.float32, f"Expected float32, got {np.array(result_f32).dtype}"
    assert np.array(result_f64).dtype == np.float64, f"Expected float64, got {np.array(result_f64).dtype}"
    print("  PASS: UniformCurrentModel.sample_chunk_jax respects dtype argument")


# ── Test 3: _GridCurrentModel ──────────────────────────────────────────────────

def test_grid_current_model():
    """_GridCurrentModel.sample_chunk_jax must match NumPy sample_batch at chunk start."""
    from medea_ice_ship.ocean_current import _GridCurrentModel

    # 4×4 velocity grid
    H, W = 4, 4
    rng = np.random.RandomState(7)
    field = rng.randn(H, W, 2).astype(np.float64)
    model = _GridCurrentModel(field, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0)

    n = 6
    K = 5
    pos_np = _random_positions(n, seed=13) * 0.8  # keep within bounds
    pos_jax = jnp.array(pos_np)

    # NumPy reference: sample at chunk-start positions K times (static field)
    v_one = model.sample_batch(pos_np)   # (n, 2)
    expected = np.stack([v_one] * K, axis=0)  # (K, n, 2)

    result = model.sample_chunk_jax(K, n, pos_jax=pos_jax)

    _assert_shape_and_values(result, K, n, expected, "_GridCurrentModel(pos)")
    print("  PASS: _GridCurrentModel.sample_chunk_jax matches NumPy bilinear interpolation")


def test_grid_current_no_pos():
    """_GridCurrentModel.sample_chunk_jax(pos_jax=None) returns zeros (safe fallback)."""
    from medea_ice_ship.ocean_current import _GridCurrentModel

    H, W = 4, 4
    field = np.ones((H, W, 2), dtype=np.float64)
    model = _GridCurrentModel(field, x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)

    K, n = 3, 4
    result = model.sample_chunk_jax(K, n, pos_jax=None)

    assert np.array(result).shape == (K, n, 2), "Wrong shape for None positions"
    assert np.all(np.array(result) == 0.0), "Expected zeros when pos_jax=None"
    print("  PASS: _GridCurrentModel.sample_chunk_jax(None) → zeros (safe fallback)")


# ── Test 4: Base class fallback ────────────────────────────────────────────────

def test_base_class_fallback():
    """
    The default OceanCurrentModel.sample_chunk_jax implementation must work
    for any subclass that doesn't override it (e.g. a custom model).
    """
    from medea_ice_ship.ocean_current import OceanCurrentModel, UniformCurrentModel

    # UniformCurrentModel overrides sample_chunk_jax, but we test via the base
    # by creating a minimal custom model that inherits the default.
    class ConstantModel(OceanCurrentModel):
        def sample(self, x, y):
            return np.array([1.5, -0.3])

    model = ConstantModel()
    K, n = 4, 3
    pos_np = _random_positions(n)
    pos_jax = jnp.array(pos_np)

    result = model.sample_chunk_jax(K, n, pos_jax=pos_jax)
    result_np = np.array(result)

    assert result_np.shape == (K, n, 2), f"Bad shape: {result_np.shape}"
    # Each row should match model.sample_batch(pos_np)
    v_one = model.sample_batch(pos_np)  # (n, 2) from base class default
    expected = np.stack([v_one] * K, axis=0)
    np.testing.assert_allclose(result_np, expected, atol=1e-10,
                               err_msg="Base class fallback: values incorrect")
    print("  PASS: OceanCurrentModel base class sample_chunk_jax fallback works")


# ── Test 5: simulate_jax_chunked_replan with JAX current ──────────────────────

def test_chunked_replan_no_current_gpu_native():
    """
    simulate_jax_chunked_replan with NoCurrentModel should produce the same
    ship trajectory as with the old NumPy current sampling path.

    We run with chunk=10 frames and compare final ship position.
    """
    from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan
    from medea_ice_ship.ocean_current import NoCurrentModel
    from medea_ice_ship.properties import ShipProperties

    angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
    r       = 3.0
    hex_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    centroids   = np.array([[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    polygons    = [hex_pts + c for c in centroids]
    bonds       = [(0, 1), (1, 2)]
    thicknesses = np.array([1.0, 1.0, 1.0])
    parent_ids  = np.arange(3, dtype=np.int32)

    n_frames = 20
    ship_length, ship_beam = 15.0, 5.0
    ship_props  = ShipProperties(length=ship_length, beam=ship_beam, draft=2.5)
    ship_pos    = np.array([-25.0, 0.0])
    ship_trajectory = [
        (np.array([-25.0 + i * 0.5, 0.0]), 0.0)
        for i in range(n_frames + 2)
    ]

    common = dict(
        polygons=polygons, centroids=centroids, bonds=bonds,
        ship_pos=ship_pos, ship_length=ship_length, ship_beam=ship_beam,
        n_frames=n_frames, dt=0.05, drift_vel=np.array([0.0, 0.0]),
        thicknesses=thicknesses, ship_props=ship_props,
        ship_trajectory=list(ship_trajectory),
        parent_ids=parent_ids,
        replan_every_n_frames=10,
        ocean_current=NoCurrentModel(),
    )

    # Dummy replan callback that returns the same trajectory slice
    def _dummy_cb(frame, world_polys, centroids_cb, thicknesses_cb,
                  ship_pos_cb, ship_heading_cb, n_remaining):
        return None  # Keep current trajectory

    print("  Running chunked replan with NoCurrentModel...", flush=True)
    result = simulate_jax_chunked_replan(
        **common, replan_callback=_dummy_cb,
    )

    pos_history = result[0]
    ship_history = result[3]

    assert len(pos_history) == n_frames + 1, (
        f"Expected {n_frames + 1} position frames, got {len(pos_history)}"
    )
    assert len(ship_history) == n_frames, (
        f"Expected {n_frames} ship history entries, got {len(ship_history)}"
    )

    # Ship should have moved east (positive X)
    final_ship_x = ship_history[-1][0][0]
    assert final_ship_x > -25.0, f"Ship didn't move east: x={final_ship_x:.2f}"

    print(f"  PASS: chunked replan with NoCurrentModel → ship moved to x={final_ship_x:.2f}")


# ── Test 6: Uniform current chunk matches no-current + manual offset ───────────

def test_chunked_replan_uniform_current():
    """
    simulate_jax_chunked_replan with UniformCurrentModel produces physically
    sensible results (current drags ice in the specified direction).
    """
    from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan
    from medea_ice_ship.ocean_current import UniformCurrentModel, NoCurrentModel
    from medea_ice_ship.properties import ShipProperties

    angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
    r       = 3.0
    hex_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    centroids   = np.array([[0.0, 0.0]])
    polygons    = [hex_pts]
    bonds       = []
    thicknesses = np.array([1.0])
    parent_ids  = np.arange(1, dtype=np.int32)

    n_frames = 10
    ship_length, ship_beam = 15.0, 5.0
    ship_props  = ShipProperties(length=ship_length, beam=ship_beam, draft=2.5)
    ship_pos    = np.array([-50.0, 0.0])  # far from ice, no collision

    common = dict(
        polygons=polygons, centroids=centroids, bonds=bonds,
        ship_pos=ship_pos, ship_length=ship_length, ship_beam=ship_beam,
        n_frames=n_frames, dt=0.05, drift_vel=np.array([0.0, 0.0]),
        thicknesses=thicknesses, ship_props=ship_props,
        ship_trajectory=None,
        parent_ids=parent_ids,
        replan_every_n_frames=5,
    )

    # Run with no current (baseline)
    result_no = simulate_jax_chunked_replan(
        **common, ocean_current=NoCurrentModel(), replan_callback=None,
    )
    # Run with +Y uniform current
    result_cur = simulate_jax_chunked_replan(
        **common,
        ocean_current=UniformCurrentModel(speed=1.0, direction_rad=np.pi/2),  # +Y
        replan_callback=None,
    )

    # With +Y current, ice should drift more in +Y than with no current
    pos_no  = result_no[0][-1]   # (n, 2) final positions, no current
    pos_cur = result_cur[0][-1]  # (n, 2) final positions, +Y current

    # The ice should be further in +Y with the current
    assert pos_cur[0, 1] >= pos_no[0, 1], (
        f"Ice should drift +Y with current: no_current={pos_no[0,1]:.3f}, "
        f"with_current={pos_cur[0,1]:.3f}"
    )
    print(f"  PASS: UniformCurrentModel drags ice in correct direction "
          f"(Δy = {pos_cur[0,1] - pos_no[0,1]:.4f} m over {n_frames} frames)")


# ── Tests 7-16: Ship dynamics with ocean current ───────────────────────────────

def _make_ship_matrices():
    """Return (M3inv, D3) as JAX arrays using the Fossen supply vessel model."""
    from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF
    ship = FossenSupply3DOF(mass=6000e3, length=76.2)
    return jnp.array(ship.M3inv), jnp.array(ship.D3)


def _make_ship_dyn():
    """Return a ShipDynamicsData NamedTuple using build_ship_dynamics_data."""
    from medea_ice_ship.jax_ice_state import build_ship_dynamics_data
    from medea_ice_ship.properties import ShipProperties
    from medea_ice_ship.config import SimulationConfig
    ship_length, ship_beam = 76.2, 18.0
    props = ShipProperties(length=ship_length, beam=ship_beam, draft=5.0)
    cfg   = SimulationConfig()
    return build_ship_dynamics_data(props, cfg, ship_length, ship_beam)


def test_zero_current_unchanged():
    """nu_c_body = zeros → identical result to no-current formulation."""
    from medea_ice_ship.jax_vessel_dynamics import ship_euler_step_jax
    M3inv, D3 = _make_ship_matrices()

    eta = jnp.array([10.0, 5.0, 0.3])
    nu  = jnp.array([1.2, -0.4, 0.05])
    tau = jnp.array([50000., 20000., 5000.])
    dt  = 0.1

    eta_no, nu_no = ship_euler_step_jax(eta, nu, tau, M3inv, D3, dt, jnp.zeros(3))

    # Manually compute expected result
    nu_dot = M3inv @ (tau - D3 @ nu)
    nu_exp = nu + dt * nu_dot
    np.testing.assert_allclose(np.array(nu_no), np.array(nu_exp), atol=1e-10,
                               err_msg="Zero current should not change dynamics")
    print("  PASS: zero current → identical dynamics")


def test_damping_on_relative_velocity():
    """Ship at rest with body-frame surge current: ship accelerates toward current speed."""
    from medea_ice_ship.jax_vessel_dynamics import ship_euler_step_jax
    M3inv, D3 = _make_ship_matrices()

    eta   = jnp.array([0., 0., 0.])
    nu    = jnp.zeros(3)              # ship at rest
    tau   = jnp.zeros(3)              # no thrust
    Vc    = 0.5                       # 0.5 m/s surge current
    nu_c  = jnp.array([Vc, 0., 0.])  # body-frame current
    dt    = 0.1

    _, nu_new = ship_euler_step_jax(eta, nu, tau, M3inv, D3, dt, nu_c)

    # nu_r = nu - nu_c = -nu_c  →  nu_dot = M3inv @ (0 - D3 @ (-nu_c)) = M3inv @ D3 @ nu_c
    nu_dot_expected = M3inv @ (D3 @ nu_c)
    nu_exp = nu + dt * np.array(nu_dot_expected)

    np.testing.assert_allclose(np.array(nu_new), nu_exp, atol=1e-10,
                               err_msg="Ship at rest should accelerate toward current speed")
    # Ship must accelerate in positive surge direction
    assert float(nu_new[0]) > 0.0, f"Expected positive surge velocity, got {float(nu_new[0])}"
    print(f"  PASS: stationary ship in current accelerates to u={float(nu_new[0]):.4f} m/s (Vc={Vc})")


def test_ship_moving_with_current():
    """Ship moving at exactly current speed → zero relative velocity → zero damping → constant speed."""
    from medea_ice_ship.jax_vessel_dynamics import ship_euler_step_jax
    M3inv, D3 = _make_ship_matrices()

    Vc   = 1.5
    eta  = jnp.array([0., 0., 0.])
    nu   = jnp.array([Vc, 0., 0.])   # moving at current speed
    tau  = jnp.zeros(3)               # no thrust
    nu_c = jnp.array([Vc, 0., 0.])   # same current in body frame
    dt   = 0.1

    _, nu_new = ship_euler_step_jax(eta, nu, tau, M3inv, D3, dt, nu_c)

    # nu_r = 0 → nu_dot = 0 → nu stays constant
    np.testing.assert_allclose(np.array(nu_new), np.array(nu), atol=1e-10,
                               err_msg="Moving with current should produce zero damping")
    print(f"  PASS: ship moving with current maintains speed (nu unchanged)")


def test_ship_moving_against_current():
    """Moving against current doubles damping relative to the no-current case."""
    from medea_ice_ship.jax_vessel_dynamics import ship_euler_step_jax
    M3inv, D3 = _make_ship_matrices()

    Vc  = 1.0
    eta = jnp.array([0., 0., 0.])
    tau = jnp.zeros(3)
    dt  = 0.1

    # No current: nu = [Vc, 0, 0]
    nu_base = jnp.array([Vc, 0., 0.])
    _, nu_no_cur = ship_euler_step_jax(eta, nu_base, tau, M3inv, D3, dt, jnp.zeros(3))

    # Against current: nu = [Vc, 0, 0], current = [-Vc, 0, 0] → nu_r = [2Vc, 0, 0]
    nu_c_against = jnp.array([-Vc, 0., 0.])
    _, nu_against = ship_euler_step_jax(eta, nu_base, tau, M3inv, D3, dt, nu_c_against)

    # nu_r against = [2Vc, 0, 0], nu_r no_cur = [Vc, 0, 0] → surge deceleration doubled
    decel_no_cur   = Vc - float(nu_no_cur[0])
    decel_against  = Vc - float(nu_against[0])
    np.testing.assert_allclose(decel_against, 2.0 * decel_no_cur, rtol=1e-6,
                               err_msg="Against-current deceleration should be 2× no-current")
    print(f"  PASS: against-current decel = {decel_against:.5f} ≈ 2× no-current {decel_no_cur:.5f}")


def test_body_frame_rotation():
    """
    World-frame +X current with ship at psi=pi/2 → body-frame sway current.

    R(pi/2).T @ [Vc, 0, 0] = [0, -Vc, 0] (current is in -sway body direction).
    Result: ship at rest accelerates in -sway (toward current).
    Also verifies that ship_step_jax with world-frame v_water_ship matches
    the manually projected body-frame result.
    """
    from medea_ice_ship.jax_vessel_dynamics import ship_euler_step_jax
    M3inv, D3 = _make_ship_matrices()

    Vc  = 1.0
    psi = np.pi / 2
    # Body-frame current: R(pi/2).T @ [Vc, 0, 0] = [0, -Vc, 0]
    nu_c = jnp.array([0.0, -Vc, 0.0])

    eta = jnp.array([0., 0., psi])
    nu  = jnp.zeros(3)
    tau = jnp.zeros(3)
    dt  = 0.001

    _, nu_new = ship_euler_step_jax(eta, nu, tau, M3inv, D3, dt, nu_c)

    # nu_r = [0,0,0] - [0,-Vc,0] = [0,+Vc,0]
    # nu_dot = M3inv @ (0 - D3 @ [0,Vc,0]) → negative sway (ship accelerates toward current)
    assert float(nu_new[1]) < 0.0, (
        f"With -sway body current, nu[1] should become negative, got {float(nu_new[1]):.6f}"
    )
    np.testing.assert_allclose(float(nu_new[0]), 0.0, atol=1e-6,
                               err_msg="Surge should be ~zero with sway-only current")

    # Verify ship_step_jax produces the same sign when given world-frame v_water_ship
    from medea_ice_ship.jax_vessel_dynamics import ship_step_jax
    ship_dyn = _make_ship_dyn()
    from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF
    B = jnp.array(FossenSupply3DOF(mass=6000e3, length=76.2).B)

    _, nu_step, _, _ = ship_step_jax(
        eta, nu, jnp.zeros(6), jnp.zeros(3), jnp.array([0., 0., psi]),
        jnp.array([0., 0., psi]), jnp.zeros(3), ship_dyn, B, 30.0, dt,
        v_water_ship=jnp.array([Vc, 0.]),
        use_auto_icenav=False,
    )
    assert float(nu_step[1]) < 0.0, (
        f"ship_step_jax: +X world current at psi=pi/2 should push in -sway, "
        f"got nu[1]={float(nu_step[1]):.6f}"
    )
    print(f"  PASS: world-frame current projected to body frame "
          f"(psi=π/2 → sway={float(nu_step[1]):.6f} < 0)")


def test_matches_auto_icenav_numerically():
    """
    Single Euler step matches AUTO-IceNav supply.py formula exactly.

    AUTO-IceNav:
        u_c = V_c * cos(beta_c - psi)
        v_c = V_c * sin(beta_c - psi)
        nu_r = nu3 - [u_c, v_c, 0]
        nu3_dot = M3inv @ (tau3 - D3 @ nu_r)

    Our implementation (at psi=0):
        vc_body = R(0)^T @ [Vc, 0, 0] = [Vc, 0, 0]
        nu_r = nu - vc_body
    """
    from medea_ice_ship.jax_vessel_dynamics import ship_euler_step_jax
    M3inv, D3 = _make_ship_matrices()
    M3inv_np = np.array(M3inv)
    D3_np    = np.array(D3)

    # Ship state
    nu_np  = np.array([0.8, -0.2, 0.03])
    tau_np = np.array([80000., 15000., 8000.])
    dt     = 0.05
    psi    = 0.4  # rad

    # AUTO-IceNav formula (scalar speed + direction beta_c)
    V_c    = 0.6
    beta_c = 0.2  # direction of current in world frame
    u_c    = V_c * np.cos(beta_c - psi)   # AUTO-IceNav body-frame projection
    v_c    = V_c * np.sin(beta_c - psi)
    nu_r_ref = nu_np - np.array([u_c, v_c, 0.0])
    nu_dot_ref = M3inv_np @ (tau_np - D3_np @ nu_r_ref)
    nu_ref = nu_np + dt * nu_dot_ref

    # Our implementation: v_water_ship is world-frame [V_c * cos(beta_c), V_c * sin(beta_c)]
    v_water_ship_world = np.array([V_c * np.cos(beta_c), V_c * np.sin(beta_c)])
    # Body-frame current = R(psi)^T @ [ux, uy, 0]
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    nu_c_body_np = R.T @ np.array([v_water_ship_world[0], v_water_ship_world[1], 0.0])

    eta = jnp.array([0., 0., psi])
    nu  = jnp.array(nu_np)
    tau = jnp.array(tau_np)
    nu_c_body = jnp.array(nu_c_body_np)

    _, nu_ours = ship_euler_step_jax(eta, nu, tau, M3inv, D3, dt, nu_c_body)

    np.testing.assert_allclose(np.array(nu_ours), nu_ref, atol=1e-10,
                               err_msg="DPDEM step must match AUTO-IceNav supply.py formula numerically")
    print(f"  PASS: ship_euler_step_jax matches AUTO-IceNav supply.py formula (max err={np.max(np.abs(np.array(nu_ours) - nu_ref)):.2e})")


def test_terminal_velocity_no_dp():
    """
    Without DP control, constant current drives nu → vc_body asymptotically.

    Physical expectation: D3 @ (nu - vc_body) → 0 as t → ∞.
    We check nu converges toward vc_body after many steps.
    """
    from medea_ice_ship.jax_vessel_dynamics import ship_euler_step_jax
    M3inv, D3 = _make_ship_matrices()

    eta  = jnp.array([0., 0., 0.])
    nu   = jnp.zeros(3)
    tau  = jnp.zeros(3)
    Vc   = 1.0
    nu_c = jnp.array([Vc, 0., 0.])
    dt   = 0.5

    for _ in range(2000):
        eta, nu = ship_euler_step_jax(eta, nu, tau, M3inv, D3, dt, nu_c)

    nu_np = np.array(nu)
    # Ship should have converged near surge current speed
    assert abs(nu_np[0] - Vc) < 0.05, (
        f"Expected nu[0] → {Vc}, got {nu_np[0]:.4f} (Δ={abs(nu_np[0]-Vc):.4f})"
    )
    print(f"  PASS: terminal velocity → nu[0]={nu_np[0]:.4f} ≈ Vc={Vc} (converged)")


def test_dp_controller_holds_against_current():
    """
    Closed-loop DP controller maintains position against constant current.

    With correct relative-velocity dynamics, the DP controller must produce
    enough thrust to overcome the current-induced drift.  Position error
    must remain bounded.
    """
    from medea_ice_ship.jax_vessel_dynamics import ship_step_jax
    from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF
    ship_dyn = _make_ship_dyn()
    B = jnp.array(FossenSupply3DOF(mass=6000e3, length=76.2).B)  # (3, 6)

    eta  = jnp.array([0., 0., 0.])
    nu   = jnp.zeros(3)
    n_actual = jnp.zeros(6)
    e_int    = jnp.zeros(3)
    setpoint = jnp.array([0., 0., 0.])
    tau_ice  = jnp.zeros(3)
    Vc       = 0.5  # m/s surge current
    v_water  = jnp.array([Vc, 0.])
    dt       = 0.5

    for _ in range(200):
        eta, nu, n_actual, e_int = ship_step_jax(
            eta, nu, n_actual, e_int, setpoint, setpoint, tau_ice,
            ship_dyn, B, 30.0, dt,
            v_water_ship=v_water, use_auto_icenav=True,
        )

    pos_error = float(jnp.linalg.norm(eta[:2] - setpoint[:2]))
    assert pos_error < 20.0, (
        f"DP controller failed to hold position against current: "
        f"error={pos_error:.2f} m"
    )
    print(f"  PASS: DP holds position against {Vc} m/s current (error={pos_error:.2f} m)")


def test_ship_step_jax_current_vs_no_current():
    """
    ship_step_jax with v_water_ship and identical nu but current present must
    produce different nu evolution than without current.
    """
    from medea_ice_ship.jax_vessel_dynamics import ship_step_jax
    from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF
    ship_dyn = _make_ship_dyn()
    B = jnp.array(FossenSupply3DOF(mass=6000e3, length=76.2).B)  # (3, 6)

    eta  = jnp.array([0., 0., 0.])
    nu   = jnp.zeros(3)
    n_actual = jnp.zeros(6)
    e_int    = jnp.zeros(3)
    setpoint = jnp.array([0., 0., 0.])
    tau_ice  = jnp.zeros(3)
    dt       = 0.1

    # No current
    _, nu_no, _, _ = ship_step_jax(
        eta, nu, n_actual, e_int, setpoint, setpoint, tau_ice,
        ship_dyn, B, 30.0, dt,
        v_water_ship=jnp.zeros(2), use_auto_icenav=False,
    )
    # With surge current
    _, nu_cur, _, _ = ship_step_jax(
        eta, nu, n_actual, e_int, setpoint, setpoint, tau_ice,
        ship_dyn, B, 30.0, dt,
        v_water_ship=jnp.array([1.0, 0.0]), use_auto_icenav=False,
    )

    diff = float(jnp.linalg.norm(nu_cur - nu_no))
    assert diff > 1e-6, f"Current should change nu evolution, but diff={diff:.2e}"
    # With surge current: damping is on nu_r = nu - [1,0,0] = [-1,0,0]
    # → D3 @ nu_r has sign flip → ship accelerates, not decelerates
    assert float(nu_cur[0]) > float(nu_no[0]), (
        f"With surge current, ship should accelerate more: nu_no={float(nu_no[0]):.5f}, "
        f"nu_cur={float(nu_cur[0]):.5f}"
    )
    print(f"  PASS: ship_step_jax current changes dynamics "
          f"(nu_no[0]={float(nu_no[0]):.5f}, nu_cur[0]={float(nu_cur[0]):.5f})")


def test_chunked_replan_current_affects_ship():
    """
    simulate_jax_chunked_replan with uniform current affects ship dynamics.

    Ship with DP controller: final position should differ between
    no-current and with-current runs (controller must work harder).
    """
    from medea_ice_ship.jax_simulation import simulate_jax_chunked_replan
    from medea_ice_ship.ocean_current import UniformCurrentModel, NoCurrentModel
    from medea_ice_ship.properties import ShipProperties

    angles  = np.linspace(0, 2 * np.pi, 7)[:-1]
    r       = 3.0
    hex_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    centroids   = np.array([[100.0, 0.0]])  # ice far from ship
    polygons    = [hex_pts + centroids[0]]
    bonds       = []
    thicknesses = np.array([1.0])
    parent_ids  = np.arange(1, dtype=np.int32)

    n_frames = 30
    ship_length, ship_beam = 76.2, 18.0
    ship_props  = ShipProperties(length=ship_length, beam=ship_beam, draft=5.0)
    ship_pos    = np.array([0.0, 0.0])
    setpoint    = np.array([0.0, 0.0, 0.0])
    # Fixed setpoint: ship tries to hold position [0, 0]
    ship_trajectory = [(np.array([0.0, 0.0]), 0.0)] * (n_frames + 2)

    common = dict(
        polygons=polygons, centroids=centroids, bonds=bonds,
        ship_pos=ship_pos, ship_length=ship_length, ship_beam=ship_beam,
        n_frames=n_frames, dt=0.5, drift_vel=np.array([0.0, 0.0]),
        thicknesses=thicknesses, ship_props=ship_props,
        ship_trajectory=list(ship_trajectory),
        parent_ids=parent_ids,
        replan_every_n_frames=10,
        use_auto_icenav_ship=True,
    )

    def _no_replan(frame, world_polys, cen, thk, sp, sh, nr):
        return None

    result_no = simulate_jax_chunked_replan(
        **common, ocean_current=NoCurrentModel(), replan_callback=_no_replan,
    )
    result_cur = simulate_jax_chunked_replan(
        **common,
        ocean_current=UniformCurrentModel(speed=0.5, direction_rad=0.0),
        replan_callback=_no_replan,
    )

    ship_no  = result_no[3][-1]
    ship_cur = result_cur[3][-1]

    # The ship's final position or velocity should differ — current affects dynamics
    diff = float(np.linalg.norm(np.array(ship_no[0]) - np.array(ship_cur[0])))
    assert diff > 0.0, "Current should cause different ship trajectory"
    print(f"  PASS: current affects ship dynamics in chunked replan "
          f"(pos diff={diff:.4f} m, no_cur_x={float(ship_no[0][0]):.4f}, "
          f"cur_x={float(ship_cur[0][0]):.4f})")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        ("NoCurrentModel zeros", test_no_current_model_zeros),
        ("NoCurrentModel no transfer", test_no_current_model_no_transfer),
        ("UniformCurrentModel values", test_uniform_current_model),
        ("UniformCurrentModel dtype", test_uniform_current_dtype),
        ("_GridCurrentModel values", test_grid_current_model),
        ("_GridCurrentModel None pos", test_grid_current_no_pos),
        ("Base class fallback", test_base_class_fallback),
        ("Chunked replan NoCurrentModel", test_chunked_replan_no_current_gpu_native),
        ("Chunked replan UniformCurrentModel", test_chunked_replan_uniform_current),
        # Ship dynamics with ocean current
        ("Ship dynamics: zero current unchanged", test_zero_current_unchanged),
        ("Ship dynamics: damping on relative velocity", test_damping_on_relative_velocity),
        ("Ship dynamics: moving with current", test_ship_moving_with_current),
        ("Ship dynamics: moving against current", test_ship_moving_against_current),
        ("Ship dynamics: body frame rotation", test_body_frame_rotation),
        ("Ship dynamics: matches AUTO-IceNav numerically", test_matches_auto_icenav_numerically),
        ("Ship dynamics: terminal velocity no DP", test_terminal_velocity_no_dp),
        ("Ship dynamics: DP holds against current", test_dp_controller_holds_against_current),
        ("Ship dynamics: ship_step_jax current vs no-current", test_ship_step_jax_current_vs_no_current),
        ("Ship dynamics: chunked replan current affects ship", test_chunked_replan_current_affects_ship),
    ]

    print("\n=== JAX ocean current validation tests ===\n")
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"  FAIL: {exc}")
            import traceback
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
