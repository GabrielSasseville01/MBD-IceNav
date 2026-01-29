"""
Time integration module for DPDEM.

Handles Velocity-Verlet integration and simulation stepping.
"""

from .timestepper import (
    velocity_verlet_step,
    simulation_step,
    run_simulation,
)
