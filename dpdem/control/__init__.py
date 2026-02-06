"""
Ship control module for DPDEM.

Provides controllers and trajectory tracking for autonomous ship navigation.
Based on Fossen (2021) - Handbook of Marine Craft Hydrodynamics and Motion Control.
"""

from .gnc import ssa, Rzyx, Rxy
from .state import ShipState
from .dp_control import DPController
from .trajectory import TrajectorySetpoint

__all__ = [
    'ssa',
    'Rzyx', 
    'Rxy',
    'ShipState',
    'DPController',
    'TrajectorySetpoint',
]
