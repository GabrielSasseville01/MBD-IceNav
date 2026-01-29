"""
Visualization module for DPDEM.

Provides 2D and 3D visualization and animation of simulation results.
"""

from .animate import (
    animate_simulation_2d,
    animate_simulation_3d,
    create_animation_frames,
    save_animation,
)
from .plot import (
    plot_ice_field,
    plot_platform,
    plot_forces_time_history,
    plot_motion_time_history,
)
