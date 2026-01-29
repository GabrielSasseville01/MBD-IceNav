"""
Kinematics module for DPDEM.

Handles quaternion operations, rotation matrices, and coordinate transforms.
"""

from .quaternion import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_normalize,
    quaternion_derivative,
    integrate_quaternion,
    rotate_vector,
    transform_to_local,
    transform_to_global,
    euler_equations_local,
)
