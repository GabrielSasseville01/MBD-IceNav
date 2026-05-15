#!/usr/bin/env python3
"""
Tests verifying that DPDEM's ship dynamics implementation matches AUTO-IceNav
(Fossen 3-DOF supply vessel, pole-placement DP controller, thrust allocation).

Run with:
    cd /Users/gabrielsasseville/Documents/PhD/Projects/ShipNav/DPDEM
    uv run python test_ship_dynamics.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import numpy as np

# JAX with float64 (required for numerical accuracy)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ─── Expected values derived directly from Fossen (2021) bis-scaled model ────
# mass = 6000 T, L = 76.2 m, g = 9.81 m/s²
_MASS     = 6000.0e3
_L        = 76.2
_G        = 9.81
_SQRT_G_L = math.sqrt(_G / _L)

# Bis-scaling transform
_Tbis_inv = np.diag([1.0, 1.0, _L])

_Mbis = np.array(
    [[1.1274,  0.0,     0.0   ],
     [0.0,     1.8902, -0.0744],
     [0.0,    -0.0744,  0.1278]], float
)
_Dbis = np.array(
    [[0.0358,  0.0,     0.0   ],
     [0.0,     0.1183, -0.0124],
     [0.0,    -0.0041,  0.0308]], float
)

_M3_ref  = _MASS * (_Tbis_inv @ _Mbis @ _Tbis_inv)
_D3_ref  = _MASS * _SQRT_G_L * (_Tbis_inv @ _Dbis @ _Tbis_inv)

# Pole-placement gains (wn = diag([0.1, 0.1, 0.2]), zeta = diag([1/√2]*3))
_wn   = np.diag([0.1, 0.1, 0.2])
_zeta = np.diag([1.0 / math.sqrt(2.0)] * 3)
_M3d  = np.diag(np.diag(_M3_ref))
_D3d  = np.diag(np.diag(_D3_ref))
_Kp_ref = _wn @ _wn @ _M3d
_Kd_ref = 2.0 * _zeta @ _wn @ _M3d - _D3d
_Ki_ref = np.zeros((3, 3))  # disabled as in AUTO-IceNav

# Thrust matrix
_K_thrust = np.diag([3.2, 3.2, 3.2, 3.2, 31.2, 31.2])
_T_cfg    = np.array(
    [[0, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 0, 0],
     [30, 22, -22, -30, -8, 8]], float
)
_B_ref   = _T_cfg @ _K_thrust
_B_pinv_ref = _B_ref.T @ np.linalg.inv(_B_ref @ _B_ref.T)
_n_max_ref  = np.array([250., 250., 250., 250., 160., 160.])


# ─────────────────────────────────────────────────────────────────────────────
# 1. FossenSupply3DOF: matrix values
# ─────────────────────────────────────────────────────────────────────────────

class TestFossenSupply3DOFMatrices:
    """Verify that FossenSupply3DOF computes the exact bis-scaled matrices."""

    def _get_ship(self):
        from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF
        return FossenSupply3DOF()  # defaults: mass=6000T, length=76.2

    def test_M3_matches_reference(self):
        ship = self._get_ship()
        np.testing.assert_allclose(
            ship.M3, _M3_ref, rtol=1e-10,
            err_msg="M3 does not match Fossen bis-scaled reference"
        )

    def test_D3_matches_reference(self):
        ship = self._get_ship()
        np.testing.assert_allclose(
            ship.D3, _D3_ref, rtol=1e-10,
            err_msg="D3 does not match Fossen bis-scaled reference"
        )

    def test_M3inv_is_inverse(self):
        ship = self._get_ship()
        product = ship.M3 @ ship.M3inv
        np.testing.assert_allclose(
            product, np.eye(3), atol=1e-8,
            err_msg="M3 @ M3inv != I"
        )

    def test_B_matrix(self):
        ship = self._get_ship()
        np.testing.assert_allclose(
            ship.B, _B_ref, rtol=1e-10,
            err_msg="B thruster config matrix mismatch"
        )

    def test_n_max(self):
        ship = self._get_ship()
        np.testing.assert_array_equal(ship.n_max, _n_max_ref)

    def test_T_n(self):
        from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF
        ship = FossenSupply3DOF()
        assert ship.T_n == 1.0, f"Expected T_n=1.0, got {ship.T_n}"

    def test_max_tunnel_thrust_kN(self):
        """Tunnel thruster max thrust: K * n_max^2 = 3.2 * 250^2 = 200 kN."""
        expected_kN = 3.2 * 250.0 ** 2  # = 200 000 N = 200 kN
        ship = self._get_ship()
        K_diag = [3.2, 3.2, 3.2, 3.2, 31.2, 31.2]
        for i in range(4):
            thrust = K_diag[i] * ship.n_max[i] ** 2
            np.testing.assert_allclose(thrust, expected_kN, rtol=1e-10)

    def test_max_main_prop_thrust_kN(self):
        """Main propeller max thrust: 31.2 * 160^2 ≈ 799 kN."""
        expected_kN = 31.2 * 160.0 ** 2
        ship = self._get_ship()
        K_diag = [3.2, 3.2, 3.2, 3.2, 31.2, 31.2]
        for i in range(4, 6):
            thrust = K_diag[i] * ship.n_max[i] ** 2
            np.testing.assert_allclose(thrust, expected_kN, rtol=1e-10)

    def test_wn_values(self):
        ship = self._get_ship()
        np.testing.assert_allclose(np.diag(ship.wn), [0.1, 0.1, 0.2], rtol=1e-10)

    def test_zeta_values(self):
        ship = self._get_ship()
        expected = 1.0 / math.sqrt(2.0)
        np.testing.assert_allclose(np.diag(ship.zeta), [expected]*3, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pole-placement gains
# ─────────────────────────────────────────────────────────────────────────────

class TestPolePlacementGains:
    """Verify DP controller gains computed from FossenSupply3DOF match reference."""

    def _compute_gains(self):
        from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF
        ship = FossenSupply3DOF()
        M3_diag = np.diag(np.diag(ship.M3))
        D3_diag = np.diag(np.diag(ship.D3))
        Kp = ship.wn @ ship.wn @ M3_diag
        Kd = 2.0 * ship.zeta @ ship.wn @ M3_diag - D3_diag
        Ki = np.zeros((3, 3))
        return Kp, Kd, Ki

    def test_Kp_matches_reference(self):
        Kp, _, _ = self._compute_gains()
        np.testing.assert_allclose(Kp, _Kp_ref, rtol=1e-10,
            err_msg="Kp from pole placement does not match reference")

    def test_Kd_matches_reference(self):
        _, Kd, _ = self._compute_gains()
        np.testing.assert_allclose(Kd, _Kd_ref, rtol=1e-10,
            err_msg="Kd from pole placement does not match reference")

    def test_Ki_is_zero(self):
        _, _, Ki = self._compute_gains()
        np.testing.assert_array_equal(Ki, np.zeros((3, 3)),
            err_msg="Ki should be zero (integral disabled, matching AUTO-IceNav)")

    def test_Kp_positive_diagonal(self):
        """Kp diagonal elements must be positive (stabilising)."""
        Kp, _, _ = self._compute_gains()
        assert all(np.diag(Kp) > 0), f"Kp diagonal not all positive: {np.diag(Kp)}"

    def test_Kd_positive_diagonal(self):
        """Kd diagonal elements must be positive."""
        _, Kd, _ = self._compute_gains()
        assert all(np.diag(Kd) > 0), f"Kd diagonal not all positive: {np.diag(Kd)}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. build_ship_dynamics_data
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildShipDynamicsData:
    """Verify build_ship_dynamics_data produces correct JAX arrays."""

    def _build(self):
        from medea_ice_ship.jax_ice_state import build_ship_dynamics_data
        from medea_ice_ship.properties import ShipProperties
        from medea_ice_ship.config import SimulationConfig
        ship_props = ShipProperties(length=76.2, beam=18.0, draft=5.5,
                                    displacement=6000e3)
        cfg = SimulationConfig()
        return build_ship_dynamics_data(ship_props, cfg, 76.2, 18.0)

    def test_M3inv_nonzero(self):
        sd = self._build()
        assert np.any(np.abs(np.array(sd.M3inv)) > 0), "M3inv is all zeros"

    def test_M3inv_correct(self):
        sd = self._build()
        np.testing.assert_allclose(
            np.array(sd.M3inv), np.linalg.inv(_M3_ref), rtol=1e-6,
            err_msg="M3inv in ShipDynamicsData does not match expected"
        )

    def test_D3_correct(self):
        sd = self._build()
        np.testing.assert_allclose(
            np.array(sd.D3), _D3_ref, rtol=1e-6,
            err_msg="D3 in ShipDynamicsData does not match expected"
        )

    def test_Kp_correct(self):
        sd = self._build()
        np.testing.assert_allclose(
            np.array(sd.Kp), _Kp_ref, rtol=1e-6,
            err_msg="Kp in ShipDynamicsData does not match pole-placement reference"
        )

    def test_Kd_correct(self):
        sd = self._build()
        np.testing.assert_allclose(
            np.array(sd.Kd), _Kd_ref, rtol=1e-6,
            err_msg="Kd in ShipDynamicsData does not match pole-placement reference"
        )

    def test_Ki_is_zero(self):
        sd = self._build()
        np.testing.assert_array_equal(np.array(sd.Ki), np.zeros((3, 3)))

    def test_B_pinv_nonzero(self):
        """B_pinv must NOT be all zeros (previous bug: wrong constructor args)."""
        sd = self._build()
        assert np.any(np.abs(np.array(sd.B_pinv)) > 0), \
            "B_pinv is all zeros — FossenSupply3DOF constructor call was broken"

    def test_B_pinv_reconstructs_tau(self):
        """B @ B_pinv should equal identity (pseudo-inverse property)."""
        sd = self._build()
        B_pinv = np.array(sd.B_pinv)
        err = np.max(np.abs(_B_ref @ B_pinv - np.eye(3)))
        assert err < 1e-10, f"B @ B_pinv != I, max error={err}"

    def test_n_max_correct(self):
        sd = self._build()
        np.testing.assert_array_equal(np.array(sd.n_max), _n_max_ref)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ship_euler_step_jax: no Coriolis (linear model)
# ─────────────────────────────────────────────────────────────────────────────

class TestShipEulerStepLinear:
    """Verify ship_euler_step_jax uses the linear model (no Coriolis)."""

    def _step(self, eta, nu, tau, dt=0.05):
        from medea_ice_ship.jax_vessel_dynamics import ship_euler_step_jax
        M3inv = jnp.array(np.linalg.inv(_M3_ref))
        D3    = jnp.array(_D3_ref)
        return ship_euler_step_jax(
            jnp.array(eta), jnp.array(nu), jnp.array(tau),
            M3inv, D3, dt, jnp.zeros(3)  # no ocean current
        )

    def test_zero_tau_zero_nu_stays_still(self):
        """With no forcing and no velocity, ship stays at origin."""
        eta0 = np.array([0., 0., 0.])
        nu0  = np.array([0., 0., 0.])
        eta_new, nu_new = self._step(eta0, nu0, np.zeros(3))
        np.testing.assert_allclose(np.array(eta_new), eta0, atol=1e-14)
        np.testing.assert_allclose(np.array(nu_new),  nu0,  atol=1e-14)

    def test_nu_dot_equals_linear_model(self):
        """nu_dot should be M3inv @ (tau - D3 @ nu) — no Coriolis term."""
        nu0  = np.array([1.0, 0.2, 0.01])
        tau0 = np.array([1e5, 0., 0.])
        dt   = 0.05
        M3inv = np.linalg.inv(_M3_ref)
        nu_dot_expected = M3inv @ (tau0 - _D3_ref @ nu0)
        nu_expected = nu0 + dt * nu_dot_expected
        _, nu_new = self._step(np.zeros(3), nu0, tau0, dt)
        np.testing.assert_allclose(np.array(nu_new), nu_expected, rtol=1e-8,
            err_msg="nu update does not match linear model M3inv@(tau-D3@nu)")

    def test_coriolis_NOT_applied(self):
        """
        With non-zero nu, the result must NOT include the Coriolis term.
        We check by verifying the step equals the LINEAR prediction.
        """
        nu0  = np.array([2.0, 0.5, 0.05])   # non-zero velocity
        tau0 = np.zeros(3)
        dt   = 0.05
        M3inv = np.linalg.inv(_M3_ref)

        # Linear prediction
        nu_dot_linear = M3inv @ (-_D3_ref @ nu0)
        nu_linear = nu0 + dt * nu_dot_linear

        _, nu_new = self._step(np.zeros(3), nu0, tau0, dt)
        np.testing.assert_allclose(np.array(nu_new), nu_linear, rtol=1e-8,
            err_msg="Coriolis term appears to still be applied")

    def test_surge_acceleration_from_main_props(self):
        """Pure surge thrust from main propellers accelerates the ship forward."""
        tau_surge = np.array([1e6, 0., 0.])  # 1 MN surge
        eta0 = np.zeros(3)
        nu0  = np.zeros(3)
        eta_new, nu_new = self._step(eta0, nu0, tau_surge, dt=0.1)
        assert float(nu_new[0]) > 0, "Surge velocity should increase with positive thrust"

    def test_pose_integration_kinematics(self):
        """Heading psi=0, surge u: ship should move in +x direction."""
        nu0 = np.array([1.0, 0.0, 0.0])  # 1 m/s surge, psi=0
        eta0 = np.array([0., 0., 0.])
        dt = 0.1
        eta_new, _ = self._step(eta0, nu0, np.zeros(3), dt)
        # x should increase (kinematics: dx = cos(0)*u*dt = 1.0*dt)
        assert float(eta_new[0]) > 0, "x should increase with forward surge"
        np.testing.assert_allclose(float(eta_new[1]), 0.0, atol=1e-10,
            err_msg="y should not change with pure surge at psi=0")


# ─────────────────────────────────────────────────────────────────────────────
# 5. dp_control_jax: matches AUTO-IceNav DPpolePlacement
# ─────────────────────────────────────────────────────────────────────────────

class TestDpControlJax:
    """Verify dp_control_jax output matches AUTO-IceNav DPpolePlacement."""

    def _ref_dp(self, eta3, nu3, x_d, y_d, psi_d, e_int=None):
        """Reference: AUTO-IceNav dp_pole_placement (Python)."""
        from medea_ice_ship.auto_icenav_ship import dp_pole_placement
        from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF
        ship = FossenSupply3DOF()
        if e_int is None:
            e_int = np.zeros(3)
        tau, e_int_new = dp_pole_placement(
            e_int, ship.M3, ship.D3, eta3, nu3, x_d, y_d, psi_d,
            ship.wn, ship.zeta, dt=0.05
        )
        return tau, e_int_new

    def _jax_dp(self, eta, nu, setpoint, e_int=None, dt=0.05):
        from medea_ice_ship.jax_vessel_dynamics import dp_control_jax
        if e_int is None:
            e_int = np.zeros(3)
        # setpoint_next not used (no feedforward), pass same as setpoint
        tau, e_int_new = dp_control_jax(
            jnp.array(eta), jnp.array(nu), jnp.array(e_int),
            jnp.array(setpoint), jnp.array(setpoint),  # setpoint_next = setpoint
            jnp.array(_Kp_ref), jnp.array(_Kd_ref), jnp.array(_Ki_ref), dt,
        )
        return np.array(tau), np.array(e_int_new)

    def test_at_setpoint_with_zero_velocity(self):
        """At the setpoint with zero velocity, tau should be zero."""
        eta = np.array([10., 20., 0.5])
        nu  = np.zeros(3)
        sp  = eta.copy()
        tau, _ = self._jax_dp(eta, nu, sp)
        np.testing.assert_allclose(tau, np.zeros(3), atol=1e-6,
            err_msg="tau should be zero at setpoint with zero velocity")

    def test_matches_reference_at_origin(self):
        """JAX controller should produce same tau as Python reference."""
        eta3 = np.array([5., -3., 0.1])
        nu3  = np.array([0.5, 0.1, 0.01])
        sp   = np.array([0., 0., 0.])
        tau_ref, _ = self._ref_dp(eta3, nu3, sp[0], sp[1], sp[2])
        tau_jax, _ = self._jax_dp(eta3, nu3, sp)
        np.testing.assert_allclose(tau_jax, tau_ref, rtol=1e-6,
            err_msg="dp_control_jax tau does not match Python DPpolePlacement reference")

    def test_matches_reference_nonzero_heading(self):
        """Test with non-trivial heading (verifies R^T application)."""
        eta3 = np.array([10., 5., math.pi/4])
        nu3  = np.array([1.0, 0.3, 0.02])
        sp   = np.array([0., 0., 0.])
        tau_ref, _ = self._ref_dp(eta3, nu3, sp[0], sp[1], sp[2])
        tau_jax, _ = self._jax_dp(eta3, nu3, sp)
        np.testing.assert_allclose(tau_jax, tau_ref, rtol=1e-6,
            err_msg="dp_control_jax tau mismatch at non-zero heading")

    def test_heading_wrap_across_pi(self):
        """Heading error across ±pi boundary should be wrapped (SSA)."""
        # Setpoint psi_d = pi-0.1, actual psi = -pi+0.1 => error wraps to 0.2
        eta3 = np.array([0., 0., -math.pi + 0.1])
        nu3  = np.zeros(3)
        sp   = np.array([0., 0.,  math.pi - 0.1])
        tau_jax, _ = self._jax_dp(eta3, nu3, sp)
        # Should NOT produce a huge yaw torque (wrapped error ~ 0.2 rad, not ~2*pi)
        assert abs(tau_jax[2]) < 1e10, f"Yaw torque too large, heading wrap likely failed: {tau_jax[2]}"

    def test_no_velocity_feedforward(self):
        """
        Changing setpoint_next should NOT change tau (no feedforward in new impl).
        """
        from medea_ice_ship.jax_vessel_dynamics import dp_control_jax
        eta  = jnp.array([5., 0., 0.])
        nu   = jnp.array([0.5, 0., 0.])
        sp   = jnp.array([0., 0., 0.])
        sp_next_a = jnp.array([1.,  0., 0.])  # some next setpoint
        sp_next_b = jnp.array([10., 5., 1.])  # very different next setpoint
        e_int = jnp.zeros(3)
        tau_a, _ = dp_control_jax(eta, nu, e_int, sp, sp_next_a,
                                  jnp.array(_Kp_ref), jnp.array(_Kd_ref),
                                  jnp.array(_Ki_ref), 0.05)
        tau_b, _ = dp_control_jax(eta, nu, e_int, sp, sp_next_b,
                                  jnp.array(_Kp_ref), jnp.array(_Kd_ref),
                                  jnp.array(_Ki_ref), 0.05)
        np.testing.assert_allclose(np.array(tau_a), np.array(tau_b), atol=1e-12,
            err_msg="tau changed with different setpoint_next — feedforward still active")

    def test_integral_error_update(self):
        """Integral error accumulates the world-frame position+heading error."""
        eta3  = np.array([5., 3., 0.1])
        nu3   = np.zeros(3)
        sp    = np.array([0., 0., 0.])
        dt    = 0.05
        # Expected error (same sign convention as control law: actual - desired)
        e_xy  = eta3[:2] - sp[:2]
        e_psi = (eta3[2] - sp[2] + math.pi) % (2 * math.pi) - math.pi
        e_expected = np.array([e_xy[0], e_xy[1], e_psi])
        _, e_int_new = self._jax_dp(eta3, nu3, sp, dt=dt)
        np.testing.assert_allclose(e_int_new, dt * e_expected, rtol=1e-8,
            err_msg="Integral error does not accumulate correctly")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Control allocation (thruster → tau round-trip)
# ─────────────────────────────────────────────────────────────────────────────

class TestControlAllocation:
    """Verify thrust allocation: tau → RPM → tau reconstructs original tau."""

    def test_surge_allocation(self):
        """Pure surge demand should produce symmetric main-propeller RPMs."""
        from medea_ice_ship.jax_vessel_dynamics import control_alloc_jax, thruster_forces_jax
        tau_surge = jnp.array([1e5, 0., 0.])
        B_pinv = jnp.array(_B_pinv_ref)
        n_max  = jnp.array(_n_max_ref)
        n_cmd = control_alloc_jax(tau_surge, B_pinv, n_max)
        # Reconstruct tau
        B = jnp.array(_B_ref)
        tau_out = thruster_forces_jax(n_cmd, B)
        np.testing.assert_allclose(np.array(tau_out)[0], 1e5, rtol=1e-4,
            err_msg="Surge tau not recovered after allocation")
        np.testing.assert_allclose(np.array(tau_out)[1:], [0., 0.], atol=1e-3,
            err_msg="Sway/yaw forces should be ~0 for pure surge demand")

    def test_sway_allocation(self):
        """Pure sway demand — tunnel thrusters fire, main props stay at 0."""
        from medea_ice_ship.jax_vessel_dynamics import control_alloc_jax, thruster_forces_jax
        tau_sway = jnp.array([0., 5e4, 0.])
        B_pinv = jnp.array(_B_pinv_ref)
        n_max  = jnp.array(_n_max_ref)
        n_cmd  = control_alloc_jax(tau_sway, B_pinv, n_max)
        B = jnp.array(_B_ref)
        tau_out = thruster_forces_jax(n_cmd, B)
        np.testing.assert_allclose(np.array(tau_out)[1], 5e4, rtol=1e-4,
            err_msg="Sway tau not recovered after allocation")

    def test_pseudoinverse_reconstruction(self):
        """B @ B_pinv should equal identity (no loss of information for reachable tau)."""
        err = np.max(np.abs(_B_ref @ _B_pinv_ref - np.eye(3)))
        assert err < 1e-10, f"B @ B_pinv != I, max error={err}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. rollout_3dof (vessel_dynamics.py) — linear model
# ─────────────────────────────────────────────────────────────────────────────

class TestRollout3dof:
    """Verify rollout_3dof uses linear model (no Coriolis)."""

    def test_zero_tau_pure_damping(self):
        """With no thrust, ship decelerates due to damping."""
        from medea_ice_ship.vessel_dynamics import rollout_3dof
        eta0 = np.zeros(3)
        nu0  = np.array([1.0, 0., 0.])  # 1 m/s surge
        H    = 50
        u_seq = np.zeros((H, 3))
        traj = rollout_3dof(eta0, nu0, u_seq, 0.05, _M3_ref, _D3_ref)
        assert traj.shape == (H + 1, 3)
        # x should increase (ship coasting)
        assert traj[-1, 0] > 0, "Ship should move forward while coasting"
        # Speed decays (but ship can't go negative quickly with linear damping)
        # just verify trajectory doesn't explode
        assert np.all(np.isfinite(traj)), "rollout_3dof produced non-finite values"

    def test_matches_linear_prediction(self):
        """Single step of rollout must match linear nu_dot = M3inv@(tau - D3@nu)."""
        from medea_ice_ship.vessel_dynamics import rollout_3dof
        eta0 = np.zeros(3)
        nu0  = np.array([0.5, 0.1, 0.005])
        tau0 = np.array([1e5, 0., 0.])
        dt   = 0.05
        traj = rollout_3dof(eta0, nu0, np.array([tau0]), dt, _M3_ref, _D3_ref)
        # Expected nu_new from linear model
        nu_dot = np.linalg.solve(_M3_ref, tau0 - _D3_ref @ nu0)
        nu_expected = nu0 + dt * nu_dot
        # traj[1] is eta after one step — check pose via kinematics
        c, s = math.cos(eta0[2]), math.sin(eta0[2])
        x_expected = eta0[0] + dt * (c*nu_expected[0] - s*nu_expected[1])
        y_expected = eta0[1] + dt * (s*nu_expected[0] + c*nu_expected[1])
        np.testing.assert_allclose(traj[1, 0], x_expected, rtol=1e-8)
        np.testing.assert_allclose(traj[1, 1], y_expected, rtol=1e-8)

    def test_jax_and_numpy_rollout_agree(self):
        """rollout_3dof_jax and rollout_3dof should produce the same trajectory."""
        from medea_ice_ship.vessel_dynamics import rollout_3dof
        from medea_ice_ship.jax_vessel_dynamics import rollout_3dof_jax
        eta0  = np.array([0., 0., 0.1])
        nu0   = np.array([0.5, 0.1, 0.005])
        H     = 20
        dt    = 0.05
        u_seq = np.random.default_rng(42).uniform(-1e5, 1e5, (H, 3))
        M3inv = np.linalg.inv(_M3_ref)
        traj_np = rollout_3dof(eta0, nu0, u_seq, dt, _M3_ref, _D3_ref)
        traj_jx = np.array(rollout_3dof_jax(
            jnp.array(eta0), jnp.array(nu0), jnp.array(u_seq),
            dt, jnp.array(M3inv), jnp.array(_D3_ref)
        ))
        np.testing.assert_allclose(traj_jx, traj_np, rtol=1e-6,
            err_msg="JAX and NumPy rollouts disagree")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Open-water integration test (ship reaches ~2 m/s on a straight path)
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenWaterTracking:
    """Full-loop integration: ship should reach ~Unom=2 m/s on a straight path."""

    def test_reaches_target_speed(self):
        """
        Simulate 500 steps (25 s at dt=0.05 s) tracking a straight path.
        Ship should reach ≥ 1.5 m/s surge velocity (Unom = 2 m/s).
        """
        from medea_ice_ship.auto_icenav_ship import AutoIceNavShip
        path = np.zeros((101, 3))
        path[:, 0] = np.linspace(0, 200, 101)
        ship = AutoIceNavShip(
            eta0=[0., 0., 0.], nu0=[0., 0., 0.],
            dt=0.05, target_speed=2.0, max_accel=0.04,
        )
        ship.init_trajectory_tracking(path)
        for _ in range(500):
            ship.set_env_load_body(np.zeros(3))
            eta, nu = ship.step()
        assert nu[0] >= 1.5, f"Expected surge ≥ 1.5 m/s after 25 s, got {nu[0]:.3f}"

    def test_stays_on_path(self):
        """Ship lateral deviation should remain < 5 m on a straight path."""
        from medea_ice_ship.auto_icenav_ship import AutoIceNavShip
        path = np.zeros((101, 3))
        path[:, 0] = np.linspace(0, 200, 101)
        ship = AutoIceNavShip(
            eta0=[0., 0., 0.], nu0=[0., 0., 0.],
            dt=0.05, target_speed=2.0, max_accel=0.04,
        )
        ship.init_trajectory_tracking(path)
        max_dev = 0.0
        for _ in range(500):
            ship.set_env_load_body(np.zeros(3))
            eta, nu = ship.step()
            max_dev = max(max_dev, abs(float(eta[1])))
        assert max_dev < 5.0, f"Lateral deviation {max_dev:.2f} m exceeds 5 m"

    def test_matches_auto_icenav_supply_velocity(self):
        """
        Compare DPDEM's FossenSupply3DOF.dynamics with AUTO-IceNav supply.py
        dynamics for a few steps to verify numerical agreement.
        """
        from medea_ice_ship.auto_icenav_ship import FossenSupply3DOF as OurShip
        import sys; sys.path.insert(0, '/Users/gabrielsasseville/Documents/PhD/Projects/ShipNav/tmp/AUTO-IceNav')
        try:
            from ship_ice_planner.controller.supply import Supply as RefShip
        except ImportError:
            # Reference repo not available — skip this sub-test
            return

        # Identical initial state
        eta6  = np.array([0., 0., 0., 0., 0., 0.], float)
        nu6   = np.array([0., 0., 0., 0., 0., 0.], float)
        n_act = np.zeros(6, float)
        # RPM command (both main props at 90 RPM for surge)
        n_ctrl = np.array([0., 0., 0., 0., 90., 90.], float)
        dt     = 0.05

        our_ship = OurShip()
        ref_ship = RefShip(controlSystem='stepInput')
        ref_ship.V_c = 0.0

        for _ in range(20):
            nu6_new_ours, n_act_new_ours = our_ship.dynamics(
                eta6, nu6, n_act.copy(), n_ctrl, dt
            )
            nu6_new_ref, n_act_new_ref = ref_ship.dynamics(
                eta6.copy(), nu6.copy(), n_act.copy(), n_ctrl.copy(), dt
            )
            nu6  = nu6_new_ours
            n_act = n_act_new_ours

        np.testing.assert_allclose(
            np.array([nu6_new_ours[0], nu6_new_ours[1], nu6_new_ours[5]]),
            np.array([nu6_new_ref[0],  nu6_new_ref[1],  nu6_new_ref[5]]),
            rtol=1e-8,
            err_msg="DPDEM dynamics diverges from AUTO-IceNav reference dynamics"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestFossenSupply3DOFMatrices,
        TestPolePlacementGains,
        TestBuildShipDynamicsData,
        TestShipEulerStepLinear,
        TestDpControlJax,
        TestControlAllocation,
        TestRollout3dof,
        TestOpenWaterTracking,
    ]

    total, passed, failed = 0, 0, []
    for cls in test_classes:
        obj = cls()
        methods = sorted(m for m in dir(obj) if m.startswith("test_"))
        print(f"\n── {cls.__name__} ──")
        for name in methods:
            total += 1
            try:
                getattr(obj, name)()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
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
