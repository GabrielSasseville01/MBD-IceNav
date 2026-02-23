"""
MEDEA ice–ship fracture simulation: modular package for run_medea_ship_fracture.py.

Submodules: properties, data, geometry, bonds, tessellation, simulation, visualization, state_io.
"""

from medea_ice_ship.properties import IceProperties, ShipProperties, R_xy, ssa
from medea_ice_ship.data import load_medea_floes, scale_floes_to_world, create_simple_ice_field
from medea_ice_ship.geometry import (
    polygon_area,
    polygon_centroid,
    create_ship_hull_at_position_heading,
    point_in_polygon,
    point_inside_polygon_penetration,
)
from medea_ice_ship.bonds import create_proximity_bonds, assign_thickness_by_area, find_connected_components
from medea_ice_ship.tessellation import tessellate_all_floes
from medea_ice_ship.simulation import point_to_ship_collision, simulate_ice_ship_interaction
from medea_ice_ship.visualization import animate_ice_ship_2d, animate_ice_ship_3d, animate_costmap_movie, make_live_2d_callback, plot_hull_impact_vectors, animate_lattice_search
from medea_ice_ship.state_io import save_simulation_state, load_simulation_state
from medea_ice_ship.costmap_build import build_costmap_from_ice_state
from medea_ice_ship.planning import (
    create_medea_lattice_planner,
    create_medea_diffusion_mpc_planner,
    plan_path,
    optimize_with_diffusion,
    optimize_with_mppi,
    optimize_with_sqp,
    optimize_with_gradcem,
    path_to_ship_trajectory,
)

__all__ = [
    'IceProperties',
    'ShipProperties',
    'R_xy',
    'ssa',
    'load_medea_floes',
    'scale_floes_to_world',
    'create_simple_ice_field',
    'polygon_area',
    'polygon_centroid',
    'create_ship_hull_at_position_heading',
    'point_in_polygon',
    'point_inside_polygon_penetration',
    'create_proximity_bonds',
    'assign_thickness_by_area',
    'find_connected_components',
    'tessellate_all_floes',
    'point_to_ship_collision',
    'simulate_ice_ship_interaction',
    'animate_ice_ship_2d',
    'animate_ice_ship_3d',
    'animate_costmap_movie',
    'make_live_2d_callback',
    'plot_hull_impact_vectors',
    'animate_lattice_search',
    'save_simulation_state',
    'load_simulation_state',
    'build_costmap_from_ice_state',
    'create_medea_lattice_planner',
    'create_medea_diffusion_mpc_planner',
    'plan_path',
    'optimize_with_diffusion',
    'optimize_with_mppi',
    'optimize_with_sqp',
    'optimize_with_gradcem',
    'path_to_ship_trajectory',
]
