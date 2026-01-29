"""
Quaternion kinematics for rigid body dynamics.

Implements rotation matrix conversion, coordinate transforms, and
Euler's equations for rotational motion.

Reference: Equations 4-8 in Wang et al. (2024)

Quaternion convention: q = (q0, q1, q2, q3) where q0 is the scalar part
and (q1, q2, q3) is the vector part. This is the scalar-first convention.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple


@partial(jax.jit, static_argnums=())
def quaternion_to_rotation_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Implements Eq. 4 from the paper:
    
    R = [q0²+q1²-q2²-q3²   2(q1q2+q0q3)      2(q1q3-q0q2)    ]
        [2(q1q2-q0q3)      q0²-q1²+q2²-q3²   2(q2q3+q0q1)    ]
        [2(q1q3+q0q2)      2(q2q3-q0q1)      q0²-q1²-q2²+q3² ]
    
    Args:
        q: (4,) quaternion [q0, q1, q2, q3] (scalar first)
    
    Returns:
        (3, 3) rotation matrix R that transforms from local to global frame
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    
    # Diagonal terms
    q0_sq = q0 * q0
    q1_sq = q1 * q1
    q2_sq = q2 * q2
    q3_sq = q3 * q3
    
    # Products
    q0q1 = q0 * q1
    q0q2 = q0 * q2
    q0q3 = q0 * q3
    q1q2 = q1 * q2
    q1q3 = q1 * q3
    q2q3 = q2 * q3
    
    R = jnp.array([
        [q0_sq + q1_sq - q2_sq - q3_sq, 2.0*(q1q2 + q0q3),             2.0*(q1q3 - q0q2)],
        [2.0*(q1q2 - q0q3),             q0_sq - q1_sq + q2_sq - q3_sq, 2.0*(q2q3 + q0q1)],
        [2.0*(q1q3 + q0q2),             2.0*(q2q3 - q0q1),             q0_sq - q1_sq - q2_sq + q3_sq]
    ])
    
    return R


@partial(jax.jit, static_argnums=())
def rotation_matrix_to_quaternion(R: jnp.ndarray) -> jnp.ndarray:
    """
    Convert rotation matrix to quaternion.
    
    Uses Shepperd's method for numerical stability.
    
    Args:
        R: (3, 3) rotation matrix
    
    Returns:
        (4,) quaternion [q0, q1, q2, q3] (scalar first)
    """
    trace = jnp.trace(R)
    
    # Compute all four possibilities and select most stable
    q0_sq = (1.0 + trace) / 4.0
    q1_sq = (1.0 + 2.0*R[0,0] - trace) / 4.0
    q2_sq = (1.0 + 2.0*R[1,1] - trace) / 4.0
    q3_sq = (1.0 + 2.0*R[2,2] - trace) / 4.0
    
    # Find largest for numerical stability
    max_sq = jnp.maximum(jnp.maximum(q0_sq, q1_sq), jnp.maximum(q2_sq, q3_sq))
    
    def case_q0(R):
        q0 = jnp.sqrt(q0_sq)
        q1 = (R[2,1] - R[1,2]) / (4.0 * q0)
        q2 = (R[0,2] - R[2,0]) / (4.0 * q0)
        q3 = (R[1,0] - R[0,1]) / (4.0 * q0)
        return jnp.array([q0, q1, q2, q3])
    
    def case_q1(R):
        q1 = jnp.sqrt(q1_sq)
        q0 = (R[2,1] - R[1,2]) / (4.0 * q1)
        q2 = (R[0,1] + R[1,0]) / (4.0 * q1)
        q3 = (R[0,2] + R[2,0]) / (4.0 * q1)
        return jnp.array([q0, q1, q2, q3])
    
    def case_q2(R):
        q2 = jnp.sqrt(q2_sq)
        q0 = (R[0,2] - R[2,0]) / (4.0 * q2)
        q1 = (R[0,1] + R[1,0]) / (4.0 * q2)
        q3 = (R[1,2] + R[2,1]) / (4.0 * q2)
        return jnp.array([q0, q1, q2, q3])
    
    def case_q3(R):
        q3 = jnp.sqrt(q3_sq)
        q0 = (R[1,0] - R[0,1]) / (4.0 * q3)
        q1 = (R[0,2] + R[2,0]) / (4.0 * q3)
        q2 = (R[1,2] + R[2,1]) / (4.0 * q3)
        return jnp.array([q0, q1, q2, q3])
    
    q = jax.lax.cond(
        max_sq == q0_sq,
        case_q0,
        lambda R: jax.lax.cond(
            max_sq == q1_sq,
            case_q1,
            lambda R: jax.lax.cond(
                max_sq == q2_sq,
                case_q2,
                case_q3,
                R
            ),
            R
        ),
        R
    )
    
    return quaternion_normalize(q)


@partial(jax.jit, static_argnums=())
def quaternion_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    Multiply two quaternions.
    
    q1 * q2 represents first rotating by q2, then by q1.
    
    Args:
        q1: (4,) first quaternion [q0, q1, q2, q3]
        q2: (4,) second quaternion
    
    Returns:
        (4,) product quaternion
    """
    a0, a1, a2, a3 = q1[0], q1[1], q1[2], q1[3]
    b0, b1, b2, b3 = q2[0], q2[1], q2[2], q2[3]
    
    return jnp.array([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0
    ])


@partial(jax.jit, static_argnums=())
def quaternion_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    """
    Compute quaternion conjugate.
    
    q* = (q0, -q1, -q2, -q3)
    
    Args:
        q: (4,) quaternion
    
    Returns:
        (4,) conjugate quaternion
    """
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


@partial(jax.jit, static_argnums=())
def quaternion_normalize(q: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize quaternion to unit length.
    
    Args:
        q: (4,) quaternion
    
    Returns:
        (4,) unit quaternion
    """
    norm = jnp.linalg.norm(q)
    return q / (norm + 1e-10)


@partial(jax.jit, static_argnums=())
def quaternion_derivative(q: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    """
    Compute time derivative of quaternion given angular velocity.
    
    dQ/dt = 0.5 * Ω(ω) · Q
    
    where Ω(ω) is the quaternion derivative matrix. This can be written as:
    
    dq/dt = 0.5 * [0, ω]_q * q
    
    where [0, ω]_q is a pure quaternion.
    
    Args:
        q: (4,) quaternion [q0, q1, q2, q3]
        omega: (3,) angular velocity in global frame [rad/s]
    
    Returns:
        (4,) quaternion time derivative
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    wx, wy, wz = omega[0], omega[1], omega[2]
    
    # dq/dt = 0.5 * [0, wx, wy, wz] * [q0, q1, q2, q3]
    dq0 = 0.5 * (-q1*wx - q2*wy - q3*wz)
    dq1 = 0.5 * ( q0*wx + q2*wz - q3*wy)
    dq2 = 0.5 * ( q0*wy - q1*wz + q3*wx)
    dq3 = 0.5 * ( q0*wz + q1*wy - q2*wx)
    
    return jnp.array([dq0, dq1, dq2, dq3])


@partial(jax.jit, static_argnums=())
def integrate_quaternion(
    q: jnp.ndarray,
    omega: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Integrate quaternion forward in time.
    
    Uses first-order integration with renormalization.
    
    Args:
        q: (4,) current quaternion
        omega: (3,) angular velocity in global frame [rad/s]
        dt: Time step [s]
    
    Returns:
        (4,) updated quaternion (normalized)
    """
    dq = quaternion_derivative(q, omega)
    q_new = q + dt * dq
    return quaternion_normalize(q_new)


@partial(jax.jit, static_argnums=())
def rotate_vector(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate a vector by a quaternion.
    
    v' = q * [0, v] * q*
    
    This transforms v from the local frame to the global frame.
    
    Args:
        q: (4,) quaternion
        v: (3,) vector in local frame
    
    Returns:
        (3,) rotated vector in global frame
    """
    # Use rotation matrix for efficiency
    R = quaternion_to_rotation_matrix(q)
    return R @ v


@partial(jax.jit, static_argnums=())
def transform_to_local(
    v_global: jnp.ndarray,
    q: jnp.ndarray,
) -> jnp.ndarray:
    """
    Transform vector from global to local frame.
    
    Implements Eq. 5: v_l = R · v_g
    
    Note: In the paper, R transforms from global to local.
    Our R transforms local to global, so we use R^T.
    
    Args:
        v_global: (3,) vector in global frame
        q: (4,) quaternion representing orientation
    
    Returns:
        (3,) vector in local frame
    """
    R = quaternion_to_rotation_matrix(q)
    return R.T @ v_global


@partial(jax.jit, static_argnums=())
def transform_to_global(
    v_local: jnp.ndarray,
    q: jnp.ndarray,
) -> jnp.ndarray:
    """
    Transform vector from local to global frame.
    
    Args:
        v_local: (3,) vector in local frame
        q: (4,) quaternion representing orientation
    
    Returns:
        (3,) vector in global frame
    """
    R = quaternion_to_rotation_matrix(q)
    return R @ v_local


@partial(jax.jit, static_argnums=())
def euler_equations_local(
    omega_local: jnp.ndarray,
    moment_local: jnp.ndarray,
    inertia_local: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute angular acceleration from Euler's equations in the local frame.
    
    Implements Eqs. 6-8 from the paper:
    
    I_xx * dω_x/dt + (I_zz - I_yy) * ω_y * ω_z = M_x
    I_yy * dω_y/dt + (I_xx - I_zz) * ω_z * ω_x = M_y
    I_zz * dω_z/dt + (I_yy - I_xx) * ω_x * ω_y = M_z
    
    Solving for angular acceleration:
    
    dω_x/dt = (M_x - (I_zz - I_yy) * ω_y * ω_z) / I_xx
    dω_y/dt = (M_y - (I_xx - I_zz) * ω_z * ω_x) / I_yy
    dω_z/dt = (M_z - (I_yy - I_xx) * ω_x * ω_y) / I_zz
    
    Args:
        omega_local: (3,) angular velocity in local frame [rad/s]
        moment_local: (3,) total moment in local frame [N·m]
        inertia_local: (3,) diagonal inertia tensor [I_xx, I_yy, I_zz] [kg·m²]
    
    Returns:
        (3,) angular acceleration in local frame [rad/s²]
    """
    I_xx, I_yy, I_zz = inertia_local[0], inertia_local[1], inertia_local[2]
    omega_x, omega_y, omega_z = omega_local[0], omega_local[1], omega_local[2]
    M_x, M_y, M_z = moment_local[0], moment_local[1], moment_local[2]
    
    # Euler's equations
    domega_x = (M_x - (I_zz - I_yy) * omega_y * omega_z) / I_xx
    domega_y = (M_y - (I_xx - I_zz) * omega_z * omega_x) / I_yy
    domega_z = (M_z - (I_yy - I_xx) * omega_x * omega_y) / I_zz
    
    return jnp.array([domega_x, domega_y, domega_z])


@partial(jax.jit, static_argnums=())
def euler_equations_global(
    omega_global: jnp.ndarray,
    moment_global: jnp.ndarray,
    q: jnp.ndarray,
    inertia_local: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute angular acceleration in global frame.
    
    Transforms to local frame, applies Euler's equations, transforms back.
    
    Args:
        omega_global: (3,) angular velocity in global frame
        moment_global: (3,) moment in global frame
        q: (4,) orientation quaternion
        inertia_local: (3,) diagonal inertia in local frame
    
    Returns:
        (3,) angular acceleration in global frame
    """
    # Transform to local
    omega_local = transform_to_local(omega_global, q)
    moment_local = transform_to_local(moment_global, q)
    
    # Apply Euler's equations
    domega_local = euler_equations_local(omega_local, moment_local, inertia_local)
    
    # Transform back to global
    domega_global = transform_to_global(domega_local, q)
    
    return domega_global


@partial(jax.jit, static_argnums=())
def compute_angular_momentum(
    omega: jnp.ndarray,
    q: jnp.ndarray,
    inertia_local: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute angular momentum in global frame.
    
    L = R * I_local * R^T * omega_global = R * I_local * omega_local
    
    Args:
        omega: (3,) angular velocity in global frame
        q: (4,) orientation quaternion
        inertia_local: (3,) diagonal inertia in local frame
    
    Returns:
        (3,) angular momentum in global frame
    """
    omega_local = transform_to_local(omega, q)
    L_local = inertia_local * omega_local  # Element-wise for diagonal inertia
    L_global = transform_to_global(L_local, q)
    return L_global


# Identity quaternion (no rotation)
IDENTITY_QUATERNION = jnp.array([1.0, 0.0, 0.0, 0.0])


def quaternion_from_axis_angle(axis: jnp.ndarray, angle: float) -> jnp.ndarray:
    """
    Create quaternion from axis-angle representation.
    
    Args:
        axis: (3,) unit rotation axis
        angle: Rotation angle [rad]
    
    Returns:
        (4,) quaternion
    """
    axis = axis / (jnp.linalg.norm(axis) + 1e-10)
    half_angle = angle / 2.0
    
    q0 = jnp.cos(half_angle)
    q_vec = jnp.sin(half_angle) * axis
    
    return jnp.array([q0, q_vec[0], q_vec[1], q_vec[2]])
