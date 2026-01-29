"""
Rigid body dynamics module for DPDEM.

Handles equations of motion for ice elements and the platform.
"""

from .ice_element import (
    compute_ice_element_acceleration,
    compute_ice_element_angular_acceleration,
    update_ice_element_state,
)
from .platform import (
    compute_platform_forces,
    compute_platform_moments,
    update_platform_state,
)
