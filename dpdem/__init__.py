"""
DPDEM - Dilated-Polyhedron-based Discrete Element Method

A JAX implementation of the discrete element method for simulating
ice-structure interactions, based on Wang et al. (2024).

Reference:
    Wang, X., Liu, L., Wang, S., & Ji, S. (2024). Interaction between ice cover
    and floating platform simulated by dilated-polyhedron-based DEM.
    Ocean Engineering, 301, 117413.
"""

__version__ = "0.1.0"

from . import config
from . import state
from . import geometry
from . import kinematics
from . import contact
from . import bonds
from . import hydrodynamics
from . import dynamics
from . import integration

# Autonomous navigation modules
from . import control
from . import planning
from .navigation import (
    AutonomousNavigator,
    NavigationConfig,
    NavigationState,
    apply_control_to_platform,
    get_control_moment,
)
