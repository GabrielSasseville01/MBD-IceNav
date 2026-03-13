"""
AUTO-IceNav-style ship dynamics and controller (ported, standalone).

This module implements a Fossen-style 3-DOF supply vessel model with:
- 3-DOF dynamics in surge/sway/yaw (M3, D3 as in AUTO-IceNav's Supply)
- DP pole-placement controller (DPpolePlacement)
- A simple trajectory setpoint generator along a path (similar role to
  TrajectorySetpoint in AUTO-IceNav)

It is designed to be used inside DPDEM without depending on the external
AUTO-IceNav repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import math
import numpy as np


# ── Basic helpers ─────────────────────────────────────────────────────────────


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    a = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return a


def R_z(yaw: float) -> np.ndarray:
    """Rotation matrix for yaw (Z-axis) from body to world: R(psi)."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


# ── DP pole-placement controller (MIMO nonlinear PID without integral) ───────


def dp_pole_placement(
    e_int: np.ndarray,
    M3: np.ndarray,
    D3: np.ndarray,
    eta3: np.ndarray,
    nu3: np.ndarray,
    x_d: float,
    y_d: float,
    psi_d: float,
    wn: np.ndarray,
    zeta: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DP pole-placement control law (ported from AUTO-IceNav / Fossen).

    Args:
        e_int: Integral error state (3,)
        M3: 3x3 mass matrix
        D3: 3x3 linear damping matrix
        eta3: [x, y, psi]
        nu3: [u, v, r]
        x_d, y_d, psi_d: desired position and heading
        wn: 3x3 diagonal matrix of natural frequencies
        zeta: 3x3 diagonal matrix of damping ratios
        dt: integration timestep

    Returns:
        tau: 3-vector of control forces/moment in body frame [X, Y, N]
        e_int_new: updated integral error (currently not used in tau)
    """
    # Diagonal approximations
    M3_diag = np.diag(np.diag(M3))
    D3_diag = np.diag(np.diag(D3))

    # PID gains (integral term currently unused in torque)
    Kp = wn @ wn @ M3_diag
    Kd = 2.0 * zeta @ wn @ M3_diag - D3_diag
    Ki = (1.0 / 10.0) * wn @ Kp  # kept for completeness; not applied to tau

    # Error in world frame
    e = np.array([eta3[0] - x_d, eta3[1] - y_d, wrap_angle(eta3[2] - psi_d)], dtype=float)

    # Body-frame control law: tau = -R^T Kp e - Kd nu3  (integral omitted)
    psi = float(eta3[2])
    R = np.eye(3, dtype=float)
    R[:2, :2] = R_z(psi)
    tau = - (R.T @ Kp @ e) - (Kd @ nu3)

    # Update integral error (Euler) – not used in tau currently, but we keep it
    e_int_new = e_int + dt * e

    return tau, e_int_new


# ── Fossen-style 3-DOF supply vessel with thrusters ──────────────────────────


@dataclass
class FossenSupply3DOF:
    """
    3-DOF model of an offshore supply vessel, following Fossen and AUTO-IceNav.

    This is a port of the relevant parts of AUTO-IceNav's `Supply` class:
    - 3-DOF dynamics in surge/sway/yaw (M3, D3)
    - Thruster configuration matrix B
    - Simple DP pole-placement controller to compute desired tau3
    - Control allocation via pseudo-inverse: tau3 -> u_alloc -> n (RPM)
    """

    mass: float = 6000.0e3  # kg
    length: float = 76.2    # m

    def __post_init__(self) -> None:
        g = 9.81

        # Thruster configuration and limits (from Fossen/AUTO-IceNav)
        self.T_n = 1.0  # propeller speed time constant [s]
        self.n_max = np.array([250, 250, 250, 250, 160, 160], dtype=float)

        # Thrust coefficients: K diag; tau3 = B @ (abs(n) * n)
        K = np.diag([3.2, 3.2, 3.2, 3.2, 31.2, 31.2])
        T_cfg = np.array(
            [
                [0, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 0, 0],
                [30, 22, -22, -30, -8, 8],
            ],
            dtype=float,
        )
        self.B = T_cfg @ K  # 3x6

        # 3-DOF bis-scaled inertia and damping matrices (from AUTO-IceNav)
        Tbis_inv = np.diag([1.0, 1.0, self.length])
        Mbis = np.array(
            [
                [1.1274, 0.0, 0.0],
                [0.0, 1.8902, -0.0744],
                [0.0, -0.0744, 0.1278],
            ],
            dtype=float,
        )
        Dbis = np.array(
            [
                [0.0358, 0.0, 0.0],
                [0.0, 0.1183, -0.0124],
                [0.0, -0.0041, 0.0308],
            ],
            dtype=float,
        )

        self.M3 = self.mass * (Tbis_inv @ Mbis @ Tbis_inv)
        self.M3inv = np.linalg.inv(self.M3)
        self.D3 = self.mass * math.sqrt(g / self.length) * (Tbis_inv @ Dbis @ Tbis_inv)

        # DP controller parameters (wn, zeta) – same structure as in AUTO-IceNav
        self.wn = np.diag([0.1, 0.1, 0.2])
        self.zeta = np.diag([1.0 / math.sqrt(2.0)] * 3)
        self.e_int = np.zeros(3, dtype=float)

        # Setpoints (x_d, y_d, psi_d)
        self.x_d = 0.0
        self.y_d = 0.0
        self.psi_d = 0.0

        # Ocean current (not used here; DPDEM will supply its own)
        self.V_c = 0.0
        self.beta_c = 0.0

        # Environmental load (ice + other) in body frame [X, Y, N]
        self.tau3_env = np.zeros(3, dtype=float)

    # -- Setpoints -------------------------------------------------------------

    def set_setpoint(self, x_d: float, y_d: float, psi_d: float) -> None:
        self.x_d = float(x_d)
        self.y_d = float(y_d)
        self.psi_d = float(psi_d)

    # -- DP controller and allocation -----------------------------------------

    def dp_control(self, eta3: np.ndarray, nu3: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute DP control input (propeller RPMs) given current state.

        Args:
            eta3: [x, y, psi]
            nu3: [u, v, r]
            dt: timestep

        Returns:
            n: propeller speeds (RPM) as 6-vector
        """
        tau3, self.e_int = dp_pole_placement(
            self.e_int,
            self.M3,
            self.D3,
            eta3,
            nu3,
            self.x_d,
            self.y_d,
            self.psi_d,
            self.wn,
            self.zeta,
            dt,
        )
        return self.control_allocation(tau3)

    def control_allocation(self, tau3: np.ndarray) -> np.ndarray:
        """
        Pseudo-inverse-based allocation: tau3 -> squared RPM -> RPM.
        """
        B_pinv = self.B.T @ np.linalg.inv(self.B @ self.B.T)
        u_alloc = B_pinv @ tau3  # squared RPM
        n = np.zeros_like(self.n_max)
        for i in range(len(n)):
            val = u_alloc[i]
            # n = sign(u_alloc) * sqrt(|u_alloc|)
            n[i] = math.copysign(math.sqrt(abs(val)), val) if val != 0.0 else 0.0
            # saturate
            n[i] = max(-self.n_max[i], min(self.n_max[i], n[i]))
        return n

    # -- 3-DOF vessel dynamics with propeller dynamics ------------------------

    def dynamics(
        self,
        eta6: np.ndarray,
        nu6: np.ndarray,
        n_actual: np.ndarray,
        n_control: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate 3-DOF vessel dynamics + propeller dynamics (Euler method).

        Args:
            eta6: [x, y, z, phi, theta, psi]
            nu6:  [u, v, w, p, q, r]
            n_actual: current propeller RPMs (6,)
            n_control: commanded propeller RPMs (6,)
            dt: timestep

        Returns:
            nu6_new: updated generalized velocity (6,)
            n_actual_new: updated RPMs (6,)
        """
        # Ocean current not used here; DPDEM has its own model
        nu_r = nu6.copy()

        # Control forces and moments with propeller saturation
        n = n_actual.copy()
        n_sq = np.zeros_like(n)
        for i in range(len(n)):
            n[i] = max(-self.n_max[i], min(self.n_max[i], n[i]))
            n_sq[i] = abs(n[i]) * n[i]

        tau3 = self.B @ n_sq
        tau3 = tau3 + self.tau3_env  # add environment loads (ice, etc.)

        nu3_r = np.array([nu_r[0], nu_r[1], nu_r[5]], dtype=float)
        nu3_dot = self.M3inv @ (tau3 - self.D3 @ nu3_r)

        nu_dot = np.array(
            [nu3_dot[0], nu3_dot[1], 0.0, 0.0, 0.0, nu3_dot[2]],
            dtype=float,
        )
        n_dot = (n_control - n_actual) / self.T_n

        nu6_new = nu6 + dt * nu_dot
        n_actual_new = n_actual + dt * n_dot
        return nu6_new, n_actual_new


# ── Trajectory setpoint generator ────────────────────────────────────────────


@dataclass
class TrajectorySetpoint:
    """
    Simple trajectory setpoint generator along a path [x, y, psi].

    This approximates the functionality of AUTO-IceNav's TrajectorySetpoint:
    - Tracks along the path at approximately constant target_speed.
    - Enforces a simple acceleration limit in along-track speed.
    """

    path: np.ndarray              # (N, 3): [x, y, psi]
    eta0: np.ndarray              # initial [x, y, psi]
    current_speed: float          # initial scalar vessel speed
    target_speed: float           # target scalar speed [m/s]
    max_accel: float              # max scalar acceleration [m/s^2]

    def __post_init__(self) -> None:
        self.path = np.asarray(self.path, dtype=float)
        self.eta0 = np.asarray(self.eta0, dtype=float)
        self.s = self._compute_arc_length(self.path[:, :2])
        self.s_total = float(self.s[-1]) if len(self.s) > 0 else 0.0
        self.v = float(self.current_speed)
        self.s_ref = 0.0  # current along-path distance

    @staticmethod
    def _compute_arc_length(xy: np.ndarray) -> np.ndarray:
        if xy.shape[0] < 2:
            return np.zeros(xy.shape[0], dtype=float)
        diffs = np.diff(xy, axis=0)
        seg = np.linalg.norm(diffs, axis=1)
        s = np.zeros(xy.shape[0], dtype=float)
        s[1:] = np.cumsum(seg)
        return s

    def _interp_path(self, s_query: float) -> Tuple[float, float, float]:
        if self.path.shape[0] == 0 or self.s_total <= 0.0:
            # fall back to initial pose
            return float(self.eta0[0]), float(self.eta0[1]), float(self.eta0[2])
        s_q = max(0.0, min(self.s_total, s_query))
        idx = int(np.searchsorted(self.s, s_q, side="right") - 1)
        idx = max(0, min(len(self.s) - 2, idx))
        s0, s1 = self.s[idx], self.s[idx + 1]
        t = 0.0 if s1 <= s0 else (s_q - s0) / (s1 - s0)
        p0 = self.path[idx]
        p1 = self.path[idx + 1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        psi0 = p0[2]
        psi1 = p1[2]
        dpsi = wrap_angle(psi1 - psi0)
        psi = wrap_angle(psi0 + t * dpsi)
        return float(x), float(y), float(psi)

    def update(self, dt: float) -> Tuple[float, float, float]:
        """
        Advance along the path and return new setpoint [x_d, y_d, psi_d].
        """
        # Speed ramp toward target_speed with max_accel
        if self.v < self.target_speed:
            self.v = min(self.target_speed, self.v + self.max_accel * dt)
        elif self.v > self.target_speed:
            self.v = max(self.target_speed, self.v - self.max_accel * dt)

        self.s_ref = min(self.s_total, self.s_ref + self.v * dt)
        x_d, y_d, psi_d = self._interp_path(self.s_ref)
        return x_d, y_d, psi_d


# ── High-level AutoIceNav-style ship wrapper for DPDEM ───────────────────────


class AutoIceNavShip:
    """
    High-level wrapper that combines:
    - FossenSupply3DOF vessel model
    - dp_pole_placement DP controller
    - TrajectorySetpoint generator

    This is what DPDEM will use when --auto-icenav-ship is enabled.
    """

    def __init__(
        self,
        eta0: Sequence[float],
        nu0: Sequence[float],
        dt: float,
        target_speed: float,
        max_accel: float,
    ) -> None:
        self.dt = float(dt)
        self.eta = np.array(eta0, dtype=float)  # [x, y, psi]
        self.nu = np.array(nu0, dtype=float)    # [u, v, r]
        self.n_actual = np.zeros(6, dtype=float)
        self.n_control = np.zeros(6, dtype=float)

        self.vessel = FossenSupply3DOF()
        self.target_speed = float(target_speed)
        self.max_accel = float(max_accel)
        self.setpoint_generator: Optional[TrajectorySetpoint] = None
        self.setpoint: Tuple[float, float, float] = (self.eta[0], self.eta[1], self.eta[2])

    def init_trajectory_tracking(self, path_xyh: np.ndarray) -> None:
        path = np.asarray(path_xyh, dtype=float)
        current_speed = float(np.sqrt(self.nu[0] ** 2 + self.nu[1] ** 2))
        if current_speed <= 0.0:
            current_speed = self.target_speed
        self.setpoint_generator = TrajectorySetpoint(
            path=path,
            eta0=self.eta.copy(),
            current_speed=current_speed,
            target_speed=self.target_speed,
            max_accel=self.max_accel,
        )
        self.setpoint = self.setpoint_generator.update(self.dt)

    def set_env_load_body(self, tau_env_body: np.ndarray) -> None:
        """
        Set environmental load (e.g. ice forces) in body frame [X, Y, N].
        """
        self.vessel.tau3_env = np.asarray(tau_env_body, dtype=float).copy()

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one control + dynamics integration step.

        Returns:
            eta: updated [x, y, psi]
            nu: updated [u, v, r]
        """
        # Update setpoint from trajectory
        if self.setpoint_generator is not None:
            self.setpoint = self.setpoint_generator.update(self.dt)
        x_d, y_d, psi_d = self.setpoint

        # Build eta3, nu3 for controller
        eta3 = np.array([self.eta[0], self.eta[1], self.eta[2]], dtype=float)
        nu3 = np.array([self.nu[0], self.nu[1], self.nu[2]], dtype=float)

        # DP controller: desired prop RPMs
        self.vessel.set_setpoint(x_d, y_d, psi_d)
        self.n_control = self.vessel.dp_control(eta3, nu3, self.dt)

        # 6-DOF generalized state for dynamics (only x,y,psi,u,v,r used)
        eta6 = np.array([self.eta[0], self.eta[1], 0.0, 0.0, 0.0, self.eta[2]], dtype=float)
        nu6 = np.array([self.nu[0], self.nu[1], 0.0, 0.0, 0.0, self.nu[2]], dtype=float)

        nu6_new, n_actual_new = self.vessel.dynamics(
            eta6, nu6, self.n_actual, self.n_control, self.dt
        )
        self.n_actual = n_actual_new

        # Integrate pose from new velocities
        u_g, v_g = (R_z(self.eta[2]) @ np.array([nu6_new[0], nu6_new[1]], dtype=float))
        self.eta[0] += self.dt * float(u_g)
        self.eta[1] += self.dt * float(v_g)
        self.eta[2] = wrap_angle(self.eta[2] + self.dt * float(nu6_new[5]))

        self.nu[0] = float(nu6_new[0])
        self.nu[1] = float(nu6_new[1])
        self.nu[2] = float(nu6_new[5])

        return self.eta.copy(), self.nu.copy()


if __name__ == "__main__":
    # Open-water test: straight path x=0 -> 50 m, y=0, psi=0
    path = np.zeros((101, 3))
    path[:, 0] = np.linspace(0, 50, 101)
    path[:, 1] = 0.0
    path[:, 2] = 0.0
    eta0 = [0.0, 0.0, 0.0]
    nu0 = [0.0, 0.0, 0.0]
    dt = 0.05
    target_speed = 2.0
    max_accel = 0.5
    ship = AutoIceNavShip(eta0=eta0, nu0=nu0, dt=dt, target_speed=target_speed, max_accel=max_accel)
    ship.init_trajectory_tracking(path)
    for _ in range(400):
        ship.set_env_load_body(np.zeros(3))
        eta, nu = ship.step()
    print(f"Open-water test: final x={eta[0]:.2f} m, y={eta[1]:.2f} m, psi={eta[2]:.3f} rad")
    assert eta[0] >= 10, f"Expected x >= 10 m, got {eta[0]}"
    print("OK")

