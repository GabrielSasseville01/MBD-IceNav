"""
GNC (Guidance, Navigation, Control) utility functions.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley.

Adapted from AUTO-IceNav project.
"""

import numpy as np
import math


def ssa(angle: float) -> float:
    """
    Smallest-signed angle.
    
    Returns the angle in the range [-pi, pi).
    
    Args:
        angle: Angle in radians
        
    Returns:
        Angle wrapped to [-pi, pi)
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def sat(x: float, x_min: float, x_max: float) -> float:
    """
    Saturate a signal.
    
    Args:
        x: Input signal
        x_min: Minimum value
        x_max: Maximum value
        
    Returns:
        Saturated signal in [x_min, x_max]
    """
    return max(x_min, min(x, x_max))


def Rxy(psi: float) -> np.ndarray:
    """
    2D rotation matrix for heading angle.
    
    Args:
        psi: Heading angle in radians (positive counter-clockwise from x-axis)
        
    Returns:
        2x2 rotation matrix
    """
    c = math.cos(psi)
    s = math.sin(psi)
    return np.array([[c, -s], [s, c]])


def Rzyx(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    3D Euler angle rotation matrix using ZYX convention.
    
    Args:
        phi: Roll angle (rotation about x-axis)
        theta: Pitch angle (rotation about y-axis)
        psi: Yaw/heading angle (rotation about z-axis)
        
    Returns:
        3x3 rotation matrix R in SO(3)
    """
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)

    R = np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
        [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
        [-sth, cth * sphi, cth * cphi]
    ])

    return R


def Smtrx(a: np.ndarray) -> np.ndarray:
    """
    Skew-symmetric matrix.
    
    Computes the 3x3 skew-symmetric matrix S(a) such that:
    a × b = S(a) @ b
    
    Args:
        a: 3D vector
        
    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])


def Hmtrx(r: np.ndarray) -> np.ndarray:
    """
    6x6 system transformation matrix.
    
    H = [I    S(r)^T]
        [0    I     ]
        
    Used to transform between coordinate origins (e.g., CO to CG).
    Property: inv(H(r)) = H(-r)
    
    Args:
        r: Position vector from origin to new point
        
    Returns:
        6x6 transformation matrix
    """
    H = np.identity(6, float)
    H[0:3, 3:6] = Smtrx(r).T
    return H


def euclid_dist(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance between two points.
    
    Args:
        a: First point
        b: Second point
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(np.array(a) - np.array(b))


def crosstrack_error(p1: np.ndarray, p2: np.ndarray, p: np.ndarray) -> float:
    """
    Compute cross-track error (signed perpendicular distance from point to line).
    
    Args:
        p1: First point on line
        p2: Second point on line  
        p: Point to compute distance for
        
    Returns:
        Signed cross-track error (positive = right of line from p1 to p2)
    """
    # Direction vector of line
    d = np.array(p2) - np.array(p1)
    d_norm = np.linalg.norm(d)
    
    if d_norm < 1e-10:
        return euclid_dist(p, p1)
    
    # Vector from p1 to p
    v = np.array(p) - np.array(p1)
    
    # Cross product gives signed area of parallelogram
    # Divided by base length gives height (cross-track error)
    cross = d[0] * v[1] - d[1] * v[0]
    
    return cross / d_norm
