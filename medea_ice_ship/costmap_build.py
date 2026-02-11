"""
Build a planning costmap from MEDEA ice state (Section 1 of PLANNER_INTEGRATION_CHECKLIST).

Takes polygons, centroids, thicknesses and domain bounds; computes cell masses from
area × thickness × ice_density; builds a DPDEM CostMap with configurable origin and margin.
"""

from typing import List, Tuple, Optional
import numpy as np

from dpdem.planning.cost_map import CostMap
from medea_ice_ship.geometry import polygon_area


def build_costmap_from_ice_state(
    polygons: List[np.ndarray],
    centroids: np.ndarray,
    thicknesses: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    resolution: float = 2.0,
    margin: float = 20.0,
    ship_mass: float = 23e6,
    base_cost: float = 1.0,
    collision_cost: float = 100.0,
    mass_cost_scale: float = 1e-5,
    ice_density: float = 920.0,
) -> Tuple[CostMap, dict]:
    """
    Build a CostMap from MEDEA ice field state (polygons, centroids, thicknesses).

    Cell mass = area(polygon) × thickness × ice_density. The costmap covers the domain
    plus a margin on all sides so the planner can place start/goal just outside the ice.

    Args:
        polygons: List of N polygon vertex arrays in world coordinates (each Nx2).
        centroids: Nx2 array of cell centroids in world coordinates [m].
        thicknesses: Length-N array of ice thickness per cell [m].
        x_min, x_max, y_min, y_max: Domain bounds in world coordinates [m].
        resolution: Grid resolution [m/cell].
        margin: Extra margin around domain [m]; map covers [x_min-margin, x_max+margin] etc.
        ship_mass: Ship mass [kg] for relative cost (default 23 kt).
        base_cost: Base traversal cost per cell (open water).
        collision_cost: Cost for cells containing ice.
        mass_cost_scale: Scaling for mass-based cost (heavier ice = higher cost).
        ice_density: Ice density [kg/m³] for mass computation.

    Returns:
        cost_map: DPDEM CostMap with origin (x_min - margin, y_min - margin) and
            size (domain_width + 2*margin) × (domain_height + 2*margin).
        bounds: Dict with keys x_min, x_max, y_min, y_max, x_origin, y_origin, width, height
            (all in world units) so planner and trajectory conversion use the same frame.
    """
    n = len(polygons)
    assert n == len(centroids) == len(thicknesses), "polygons, centroids, thicknesses length mismatch"

    areas = np.array([polygon_area(p) for p in polygons])
    masses = areas * thicknesses * ice_density

    # CostMap expects polygons in local coordinates (centroid at origin); it applies rotation and translation.
    local_polygons = [np.asarray(p, dtype=float) - np.asarray(c, dtype=float) for p, c in zip(polygons, centroids)]

    domain_width = x_max - x_min
    domain_height = y_max - y_min
    width = domain_width + 2.0 * margin
    height = domain_height + 2.0 * margin
    x_origin = x_min - margin
    y_origin = y_min - margin

    cost_map = CostMap(
        width=width,
        height=height,
        resolution=resolution,
        x_origin=x_origin,
        y_origin=y_origin,
        ship_mass=ship_mass,
        base_cost=base_cost,
        collision_cost=collision_cost,
        mass_cost_scale=mass_cost_scale,
    )
    cost_map.update_from_ice_field(
        positions=centroids,
        polygons=local_polygons,
        masses=masses,
        angles=None,
    )

    bounds = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "x_origin": x_origin,
        "y_origin": y_origin,
        "width": width,
        "height": height,
    }
    return cost_map, bounds
