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
from medea_ice_ship.visualization import animate_ice_ship_2d, animate_ice_ship_3d
from medea_ice_ship.state_io import save_simulation_state, load_simulation_state

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
    'save_simulation_state',
    'load_simulation_state',
]
