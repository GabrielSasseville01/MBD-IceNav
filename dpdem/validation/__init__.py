"""
Validation module for DPDEM.

Contains validation tests and reference formulas.
"""

from .lindqvist import lindqvist_ice_resistance
from .tests import (
    test_hertz_contact,
    test_bond_fracture,
    test_single_particle_gravity,
)
