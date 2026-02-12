"""
MSS-style 3-DOF vessel dynamics (Fossen formulation).

Provides mass matrix M and damping D for surge–sway–yaw (3-DOF) ship motion,
matching the structure used in AUTO-IceNav / Marine Systems Simulator (MSS).
M = M_RB + M_A (rigid body + added mass), D = linear hydrodynamic damping.
Coriolis C(nu) is computed from M when advancing the state.

Also provides rollout_3dof for model-based diffusion: given initial (eta, nu)
and a control sequence u_seq (body-frame force/torque [X, Y, N]), returns
trajectory (x, y, psi) over horizon.
"""

from typing import Tuple
import numpy as np

from medea_ice_ship.properties import R_xy


def get_mss_style_3dof(
    mass: float,
    Izz: float,
    ship_damping_xy: float = 0.02,
    ship_damping_psi: float = 0.15,
    added_mass_surge_frac: float = 0.05,
    added_mass_sway_frac: float = 0.5,
    added_mass_yaw_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (M3, D3) for 3-DOF Fossen-style vessel dynamics (surge, sway, yaw).

    M3 = M_RB + M_A: rigid-body mass/inertia plus added mass (zero-frequency).
    D3 = linear damping diagonal (X_u, Y_v, N_r in SNAME notation).

    Args:
        mass: Ship mass [kg].
        Izz: Yaw moment of inertia [kg·m²].
        ship_damping_xy: Linear damping coefficient for surge/sway (force = -this × mass × u).
        ship_damping_psi: Linear damping coefficient for yaw (torque = -this × Izz × r).
        added_mass_surge_frac: Added mass in surge as fraction of mass (typical 0.05–0.2).
        added_mass_sway_frac: Added mass in sway as fraction of mass (typical 0.3–0.8).
        added_mass_yaw_frac: Added inertia in yaw as fraction of Izz (typical 0.1–0.3).

    Returns:
        M3: 3×3 mass matrix (symmetric positive definite).
        D3: 3×3 damping matrix (diagonal, non-negative).
    """
    # Rigid body M_RB + added mass M_A (Fossen: both positive definite).
    # M = diag(m + X_udot, m + Y_vdot, Izz + N_rdot); fractions are relative to m and Izz.
    X_udot = added_mass_surge_frac * mass
    Y_vdot = added_mass_sway_frac * mass
    N_rdot = added_mass_yaw_frac * Izz
    M3 = np.diag([mass + X_udot, mass + Y_vdot, Izz + N_rdot])

    D3 = np.diag([
        mass * ship_damping_xy,
        mass * ship_damping_xy,
        Izz * ship_damping_psi,
    ])
    return M3, D3


def coriolis_3dof(M3: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """
    Coriolis/centripetal matrix C(nu) for 3-DOF (surge, sway, yaw).

    For M symmetric and only surge–sway–yaw, C(nu) * nu gives the centripetal
    terms so that M * nu_dot + C(nu)*nu + D*nu = tau.

    Args:
        M3: 3×3 mass matrix.
        nu: Body-frame velocity [u, v, r].

    Returns:
        C: 3×3 matrix such that C @ nu is the Coriolis term.
    """
    m11, m22, m33 = M3[0, 0], M3[1, 1], M3[2, 2]
    u, v, r = nu[0], nu[1], nu[2]
    C = np.zeros((3, 3))
    C[0, 2] = -m22 * v
    C[1, 2] = m11 * u
    C[2, 0] = m22 * v
    C[2, 1] = -m11 * u
    return C


def rollout_3dof(
    eta_start: np.ndarray,
    nu_start: np.ndarray,
    u_seq: np.ndarray,
    dt: float,
    M3: np.ndarray,
    D3: np.ndarray,
) -> np.ndarray:
    """
    Roll out 3-DOF vessel dynamics for a sequence of body-frame controls.

    M * nu_dot + C(nu)*nu + D*nu = tau (tau = u_control in body frame).
    eta = [x, y, psi]; nu = [u, v, r]. Kinematics: eta_dot from R_xy(psi) @ nu.

    Args:
        eta_start: (3,) initial pose [x, y, psi] in world frame [m, m, rad].
        nu_start: (3,) initial body-frame velocity [u, v, r] [m/s, m/s, rad/s].
        u_seq: (horizon, 3) body-frame force/torque [X, Y, N] per step [N, N, N·m].
        dt: Time step [s].
        M3: 3×3 mass matrix.
        D3: 3×3 damping matrix.

    Returns:
        trajectory: (horizon+1, 3) array of [x, y, psi] in world frame.
    """
    eta = np.array(eta_start, dtype=float).copy()
    nu = np.array(nu_start, dtype=float).copy()
    horizon = len(u_seq)
    trajectory = np.zeros((horizon + 1, 3))
    trajectory[0] = eta

    for t in range(horizon):
        tau = np.array(u_seq[t], dtype=float)
        C = coriolis_3dof(M3, nu)
        nu_dot = np.linalg.solve(M3, tau - C @ nu - D3 @ nu)
        nu = nu + dt * nu_dot
        # Kinematics: eta_dot = R_xy(psi) @ [u, v] for position; psi_dot = r
        vel_global = R_xy(eta[2]) @ np.array([nu[0], nu[1]])
        eta[0] += dt * vel_global[0]
        eta[1] += dt * vel_global[1]
        eta[2] = (eta[2] + dt * nu[2]) % (2 * np.pi)
        eta[2] = (eta[2] + np.pi) % (2 * np.pi) - np.pi
        trajectory[t + 1] = eta

    return trajectory
