"""
JAX port of medea_ice_ship/vessel_dynamics.py + AUTO-IceNav DP controller.

Provides:
  - coriolis_3dof_jax   : Coriolis matrix (array-valued, @jax.jit)
  - ship_euler_step_jax : Single Euler sub-step for 3-DOF dynamics
  - rollout_3dof_jax    : Horizon rollout using jax.lax.scan (replaces Python for-loop)
  - dp_control_jax      : DP pole-placement controller (array-valued, @jax.jit)
  - control_alloc_jax   : Thruster allocation (vectorized, no loop over thrusters)
  - ship_step_jax       : Complete per-frame ship update for the simulation loop

All Python if/else branches have been replaced with jnp.where so that the
functions are JIT-compilable and free of data-dependent Python control flow.

References
----------
Fossen (2011) "Handbook of Marine Craft Hydrodynamics and Motion Control"
AUTO-IceNav ship model (FossenSupply3DOF)
"""

import jax
import jax.numpy as jnp
from functools import partial


# ── Low-level dynamics primitives ─────────────────────────────────────────────

@jax.jit
def coriolis_3dof_jax(M3: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
    """
    Coriolis/centripetal matrix C(nu) for 3-DOF (surge, sway, yaw).

    M * nu_dot + C(nu)*nu + D*nu = tau  (body frame)

    Parameters
    ----------
    M3 : (3, 3) mass matrix (M_RB + M_A)
    nu : (3,)   body-frame velocity [u, v, r]

    Returns
    -------
    C : (3, 3)
    """
    m11, m22 = M3[0, 0], M3[1, 1]
    u, v, r = nu[0], nu[1], nu[2]
    # Skew-symmetric Coriolis structure for diagonal M3
    return jnp.array([
        [0.0,       0.0,    -m22 * v],
        [0.0,       0.0,     m11 * u],
        [m22 * v,  -m11 * u, 0.0   ],
    ])


@jax.jit
def ship_euler_step_jax(
    eta: jnp.ndarray,        # (3,) [x, y, psi]
    nu: jnp.ndarray,         # (3,) [u, v, r]
    tau: jnp.ndarray,        # (3,) control + external forces [X, Y, N]
    M3inv: jnp.ndarray,      # (3, 3) precomputed inverse of mass matrix
    D3: jnp.ndarray,         # (3, 3) linear damping
    dt: float,
    nu_c_body: jnp.ndarray,  # (3,) ocean current in body frame [u_c, v_c, 0]
) -> tuple:  # (eta_new (3,), nu_new (3,))
    """Single forward-Euler step for 3-DOF Fossen vessel dynamics.

    Uses the linear state-space model from AUTO-IceNav / paper:
        nu_r   = nu - nu_c_body          (velocity relative to water)
        nu_dot = M3inv @ (tau - D3 @ nu_r)
    No Coriolis: the supply vessel model is intentionally linear (low-speed
    maneuvering), matching Fossen (2021) and AUTO-IceNav's supply.py.

    With nu_c_body = zeros this reduces to the no-current formulation.
    """
    nu_r = nu - nu_c_body   # relative velocity: what water "feels"
    nu_dot = M3inv @ (tau - D3 @ nu_r)
    nu_new = nu + dt * nu_dot

    psi = eta[2]
    c, s = jnp.cos(psi), jnp.sin(psi)
    vx = c * nu_new[0] - s * nu_new[1]
    vy = s * nu_new[0] + c * nu_new[1]
    psi_new = (psi + dt * nu_new[2] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    eta_new = jnp.array([eta[0] + dt * vx, eta[1] + dt * vy, psi_new])
    return eta_new, nu_new


def rollout_3dof_jax(
    eta0: jnp.ndarray,   # (3,)
    nu0: jnp.ndarray,    # (3,)
    u_seq: jnp.ndarray,  # (H, 3)  control per step
    dt: float,
    M3inv: jnp.ndarray,  # (3, 3)
    D3: jnp.ndarray,     # (3, 3)
) -> jnp.ndarray:         # (H+1, 3)  trajectory [x, y, psi]
    """
    Roll out 3-DOF vessel dynamics using jax.lax.scan.

    Replaces the Python for-loop in vessel_dynamics.rollout_3dof.

    Parameters
    ----------
    eta0  : (3,)   initial pose [x, y, psi]
    nu0   : (3,)   initial body-frame velocity [u, v, r]
    u_seq : (H, 3) control sequence (body-frame forces/torques)
    dt    : float  timestep [s]
    M3inv : (3, 3) inverse mass matrix (precomputed)
    D3    : (3, 3) damping matrix

    Returns
    -------
    trajectory : (H+1, 3) array of [x, y, psi] in world frame
    """
    _zeros3 = jnp.zeros(3)

    def step(carry, tau):
        eta, nu = carry
        eta_new, nu_new = ship_euler_step_jax(eta, nu, tau, M3inv, D3, dt, _zeros3)
        return (eta_new, nu_new), eta_new

    (_, _), traj = jax.lax.scan(step, (eta0, nu0), u_seq)
    # Prepend initial pose
    return jnp.concatenate([eta0[None], traj], axis=0)


# ── DP pole-placement controller ──────────────────────────────────────────────

@jax.jit
def dp_control_jax(
    eta: jnp.ndarray,          # (3,) [x, y, psi]
    nu: jnp.ndarray,           # (3,) [u, v, r]
    e_int: jnp.ndarray,        # (3,) integral error accumulator
    setpoint: jnp.ndarray,     # (3,) [x_d, y_d, psi_d]
    setpoint_next: jnp.ndarray,# (3,) reserved (unused; kept for API compatibility)
    Kp: jnp.ndarray,           # (3, 3) proportional gain
    Kd: jnp.ndarray,           # (3, 3) derivative gain
    Ki: jnp.ndarray,           # (3, 3) integral gain (zero = disabled, AUTO-IceNav default)
    dt: float,
) -> tuple:  # (tau (3,), e_int_new (3,))
    """
    DP pole-placement control law matching AUTO-IceNav's DPpolePlacement.

    Body-frame control law (Fossen 2021):
        e     = eta - [x_d, y_d, psi_d]        (world-frame error)
        e[2]  = ssa(e[2])                        (heading wrapped to [-pi,pi])
        tau   = -R(psi)^T @ Kp @ e - Kd @ nu   (no integral by default)
              - R(psi)^T @ Ki @ e_int           (if Ki != 0)

    This matches AUTO-IceNav supply.py / control.py DPpolePlacement exactly.
    No velocity feedforward (not in the reference implementation).

    Parameters
    ----------
    eta           : current pose [x, y, psi]
    nu            : current body-frame velocity [u, v, r]
    e_int         : accumulated integral error (3,)
    setpoint      : desired pose [x_d, y_d, psi_d]
    setpoint_next : unused (kept for API compatibility)
    Kp, Kd, Ki   : controller gain matrices (3, 3)
    dt            : integration step [s]

    Returns
    -------
    tau      : body-frame control forces/torques [X, Y, N]
    e_int_new: updated integral error
    """
    psi = eta[2]
    c, s = jnp.cos(psi), jnp.sin(psi)
    R = jnp.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0., 0., 1.]])

    # World-frame error: actual - desired (sign convention matching AUTO-IceNav)
    e_xy = eta[:2] - setpoint[:2]
    e_psi = (eta[2] - setpoint[2] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    e = jnp.array([e_xy[0], e_xy[1], e_psi])

    # Control law: tau = -R^T @ Kp @ e - Kd @ nu - R^T @ Ki @ e_int
    tau = -(R.T @ Kp @ e) - (Kd @ nu) - (R.T @ Ki @ e_int)

    # Update integral error (Euler)
    e_int_new = e_int + dt * e

    return tau, e_int_new


# ── Thruster allocation ────────────────────────────────────────────────────────

@jax.jit
def control_alloc_jax(
    tau3: jnp.ndarray,    # (3,) desired force/torque in body frame
    B_pinv: jnp.ndarray,  # (6, 3) pseudoinverse of thruster config matrix
    n_max: jnp.ndarray,   # (6,) RPM limits
) -> jnp.ndarray:          # (6,) commanded RPMs
    """
    Pseudo-inverse thruster allocation (vectorized over all thrusters).

    tau3 → u_alloc (squared RPM) → n (RPM), saturated.
    Replaces the Python for-loop over 6 thrusters in FossenSupply3DOF.
    """
    u_alloc = B_pinv @ tau3  # (6,) squared-RPM commands
    sign_u = jnp.sign(u_alloc)
    n_cmd = sign_u * jnp.sqrt(jnp.abs(u_alloc) + 1e-12)  # avoid sqrt(0) gradient issues
    return jnp.clip(n_cmd, -n_max, n_max)


# ── Propeller dynamics (first-order lag) ──────────────────────────────────────

@jax.jit
def propeller_dynamics_jax(
    n_actual: jnp.ndarray,  # (6,) current RPMs
    n_control: jnp.ndarray, # (6,) commanded RPMs
    T_n: float,             # time constant [s]
    dt: float,
) -> jnp.ndarray:            # (6,) updated RPMs
    """First-order propeller speed dynamics: dn/dt = (n_control - n_actual)/T_n."""
    return n_actual + dt * (n_control - n_actual) / T_n


# ── Thruster forces: RPM → body-frame tau ─────────────────────────────────────

@jax.jit
def thruster_forces_jax(
    n_actual: jnp.ndarray,  # (6,) actual propeller RPMs
    B: jnp.ndarray,         # (3, 6) thruster configuration matrix
) -> jnp.ndarray:            # (3,) resulting body-frame [X, Y, N]
    """tau = B @ (|n| * n) — quadratic thrust law."""
    return B @ (jnp.abs(n_actual) * n_actual)


# ── Full ship simulation step ──────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("vc_surge", "vc_sway", "vc_yaw"))
def _clip_velocity_jax(
    nu: jnp.ndarray,
    vc_surge: float, vc_sway: float, vc_yaw: float,
) -> jnp.ndarray:
    """Apply ship velocity clip matching simulation.py cfg.ship_velocity_clip."""
    return jnp.array([
        jnp.clip(nu[0], -vc_surge, vc_surge),
        jnp.clip(nu[1], -vc_sway,  vc_sway),
        jnp.clip(nu[2], -vc_yaw,   vc_yaw),
    ])


@partial(jax.jit, static_argnames=("use_auto_icenav", "dt",
                                    "vc_surge", "vc_sway", "vc_yaw"))
def ship_step_jax(
    eta: jnp.ndarray,          # (3,) current pose
    nu: jnp.ndarray,           # (3,) current velocity
    n_actual: jnp.ndarray,     # (6,) current propeller RPMs
    e_int: jnp.ndarray,        # (3,) integral error
    setpoint: jnp.ndarray,     # (3,) [x_d, y_d, psi_d]
    setpoint_next: jnp.ndarray,# (3,) next waypoint for velocity feedforward
    tau_ice: jnp.ndarray,      # (3,) ice reaction force in world frame (converted to body inside)
    ship_dyn,                  # ShipDynamicsData NamedTuple
    B: jnp.ndarray,            # (3, 6) thruster config (needed for thrust computation)
    T_n: float,
    dt: float,
    v_water_ship: jnp.ndarray = jnp.zeros(2),  # (2,) ocean current at ship pos, world frame [u_c, v_c]
    use_auto_icenav: bool = False,
    vc_surge: float = 15.0,    # velocity clip [m/s]
    vc_sway: float = 8.0,
    vc_yaw: float = 0.15,
) -> tuple:  # (eta_new, nu_new, n_actual_new, e_int_new)
    """
    One-step ship dynamics update (JIT-compiled).

    When use_auto_icenav=True uses the DP pole-placement controller
    (with propeller RPM dynamics).  Otherwise falls back to a simple PD
    torque computed directly from Kp/Kd applied to setpoint error.

    Parameters
    ----------
    eta, nu, n_actual, e_int : current state
    setpoint                 : (3,) desired pose [x_d, y_d, psi_d]
    tau_ice                  : (3,) ice forces on ship in WORLD frame [Fx, Fy, Mz]
    ship_dyn                 : ShipDynamicsData (M3inv, D3, Kp, Kd, Ki, B_pinv, n_max, …)
    B                        : (3, 6) thruster config matrix (for RPM → tau conversion)
    T_n                      : propeller time constant [s]
    dt                       : timestep [s]
    v_water_ship             : (2,) world-frame ocean current at ship position [u_c, v_c].
                               Pass jnp.zeros(2) (default) for no current.
                               Matches AUTO-IceNav supply.py: damping acts on nu_r = nu - R.T @ Vc.
    use_auto_icenav          : if True, use RPM-based DP; otherwise use direct tau PD

    Returns
    -------
    eta_new, nu_new, n_actual_new, e_int_new
    """
    psi = eta[2]
    c, s = jnp.cos(psi), jnp.sin(psi)
    R_mat = jnp.array([[c, -s, 0.],
                       [s,  c, 0.],
                       [0., 0., 1.]])

    # Convert ocean current from world → body frame (matching AUTO-IceNav supply.py)
    # nu_c_body = R(psi)^T @ [u_c, v_c, 0]
    nu_c_body = R_mat.T @ jnp.array([v_water_ship[0], v_water_ship[1], 0.0])

    # Convert ice forces from world → body frame
    tau_ice_body = R_mat.T @ tau_ice

    if use_auto_icenav:
        # DP controller → desired tau3
        tau_ctrl, e_int_new = dp_control_jax(
            eta, nu, e_int, setpoint, setpoint_next,
            ship_dyn.Kp, ship_dyn.Kd, ship_dyn.Ki, dt,
        )
        # Thrust allocation → RPM command
        n_cmd = control_alloc_jax(tau_ctrl, ship_dyn.B_pinv, ship_dyn.n_max)
        # Propeller dynamics
        n_actual_new = propeller_dynamics_jax(n_actual, n_cmd, T_n, dt)
        # Actual thrust produced
        tau_ctrl_actual = thruster_forces_jax(n_actual_new, B)
    else:
        # Simple PD with velocity feedforward (matches simulation.py control law)
        tau_ctrl_actual, e_int_new = dp_control_jax(
            eta, nu, e_int, setpoint, setpoint_next,
            ship_dyn.Kp, ship_dyn.Kd, ship_dyn.Ki, dt,
        )
        n_actual_new = n_actual  # unchanged

    # Total body-frame forcing
    tau_total = tau_ctrl_actual + tau_ice_body

    # Euler step (damping applied to velocity relative to water)
    eta_new, nu_new = ship_euler_step_jax(
        eta, nu, tau_total, ship_dyn.M3inv, ship_dyn.D3, dt, nu_c_body
    )
    nu_new = _clip_velocity_jax(nu_new, vc_surge, vc_sway, vc_yaw)

    return eta_new, nu_new, n_actual_new, e_int_new
