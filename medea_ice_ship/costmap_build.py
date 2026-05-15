"""
Build a planning costmap from MEDEA ice state (Section 1 of PLANNER_INTEGRATION_CHECKLIST).

Takes polygons, centroids, thicknesses and domain bounds; computes cell masses from
area × thickness × ice_density; builds a DPDEM CostMap with configurable origin and margin.
Supports optional ridge density cost (AUTO-IceNav style) from precomputed ridge costmap.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import pickle

import numpy as np

from dpdem.planning.cost_map import CostMap
from medea_ice_ship.geometry import polygon_area


def _load_ridge_costmap(ridge_costmap_file: str) -> Optional[Tuple[np.ndarray, float]]:
    """Load ridge costmap from pickle. Returns (ridge_costmap, scale) or None."""
    path = Path(ridge_costmap_file)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        ridge = np.asarray(data["ridge_costmap"])
        scale = float(data["scale"])
        return (ridge, scale)
    except Exception:
        return None


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
    base_cost: float = 0.1,
    collision_cost: float = 5.0,
    ice_density: float = 920.0,
    wall_bounds: Optional[Tuple[float, float, float, float]] = None,
    ship_length: Optional[float] = None,
    ship_beam: Optional[float] = None,
    planning_clearance_margin: float = 0.0,
    ridge_costmap_file: Optional[str] = None,
    ridge_cost_weight: float = 0.0,
    ship_speed: float = 4.0,
    ke_cost_scale: float = 1e-6,
    obs_cost_buffer_factor: float = 0.1,
    ice_resistance_weight: float = 1.0,
    sic_kernel: Optional[Tuple[int, int]] = None,
) -> Tuple[CostMap, dict]:
    """
    Build a CostMap from MEDEA ice field state (polygons, centroids, thicknesses).

    Uses the AUTO-IceNav KE collision cost model. Cell mass = area × thickness × ice_density.
    The costmap covers the domain plus a margin on all sides.

    When ship_length and ship_beam are provided, ice obstacles are dilated by the ship's
    half-beam + planning_clearance_margin so planned paths keep the full ship footprint clear.

    Args:
        polygons: List of N polygon vertex arrays in world coordinates (each Nx2).
        centroids: Nx2 array of cell centroids in world coordinates [m].
        thicknesses: Length-N array of ice thickness per cell [m].
        x_min, x_max, y_min, y_max: Domain bounds in world coordinates [m].
        resolution: Grid resolution [m/cell].
        margin: Extra margin around domain [m].
        ship_mass: Ship mass [kg].
        base_cost: Base traversal cost per cell (open water).
        collision_cost: KE cost weight (collision_cost_weight in AUTO-IceNav model).
        ice_density: Ice density [kg/m³] for mass computation.
        wall_bounds: If set, (wx_min, wx_max, wy_min, wy_max) — cells outside these
            world-coordinate bounds are filled with impassable cost.
        ship_length: Ship length [m]. If provided with ship_beam, dilate ice.
        ship_beam: Ship beam (width) [m]. If provided with ship_length, dilate ice.
        ridge_costmap_file: Optional path to ridge costmap pickle.
        ridge_cost_weight: Weight for ridge density cost (additive; 0 = disabled).
        ship_speed: Ship speed [m/s] for KE cost.
        ke_cost_scale: Scale KE [J] to planning units (1e-6 gives ~0.1–100).
        obs_cost_buffer_factor: Polygon swell for collision cost (AUTO-IceNav: 0.1).
        ice_resistance_weight: SIC exponent; cost *= SIC^weight (AUTO-IceNav: 1).
        sic_kernel: (kh, kw) for SIC; odd. Default (31, 31) when ice_resistance_weight > 0.

    Returns:
        cost_map: DPDEM CostMap.
        bounds: Dict with domain bounds and costmap metadata.
    """
    n = len(polygons)
    assert n == len(centroids) == len(thicknesses), "polygons, centroids, thicknesses length mismatch"

    areas = np.array([polygon_area(p) for p in polygons])
    masses = areas * thicknesses * ice_density

    local_polygons = [np.asarray(p, dtype=float) - np.asarray(c, dtype=float) for p, c in zip(polygons, centroids)]

    domain_width = x_max - x_min
    domain_height = y_max - y_min
    width = domain_width + 2.0 * margin
    height = domain_height + 2.0 * margin
    x_origin = x_min - margin
    y_origin = y_min - margin

    # SIC kernel: odd, default (31, 31) when ice_resistance_weight > 0
    if sic_kernel is None and ice_resistance_weight > 0:
        kh = min(31, int(np.ceil(height / resolution)) | 1)
        kw = min(31, int(np.ceil(width / resolution)) | 1)
        sic_kernel = (kh, kw)

    cost_map = CostMap(
        width=width,
        height=height,
        resolution=resolution,
        x_origin=x_origin,
        y_origin=y_origin,
        ship_mass=ship_mass,
        base_cost=base_cost,
        collision_cost=collision_cost,
        ship_speed=ship_speed,
        ke_cost_scale=ke_cost_scale,
        obs_cost_buffer_factor=obs_cost_buffer_factor,
        ice_resistance_weight=ice_resistance_weight,
        sic_kernel=sic_kernel,
    )
    cost_map.update_from_ice_field(
        positions=centroids,
        polygons=local_polygons,
        masses=masses,
        angles=None,
        thicknesses=thicknesses,
    )

    # Ridge cost (AUTO-IceNav style): add ridge density cost to ice cells only
    if ridge_costmap_file and ridge_cost_weight > 0:
        ridge_data = _load_ridge_costmap(ridge_costmap_file)
        if ridge_data is not None:
            ridge_costmap, ridge_scale = ridge_data
            cost_map.apply_ridge_cost(
                ridge_costmap=ridge_costmap,
                ridge_scale=ridge_scale,
                polygon_vertices=polygons,
                ridge_weight=ridge_cost_weight,
            )

    if ship_length is not None and ship_beam is not None and ship_length > 0 and ship_beam > 0:
        # Inflate ice so the planner keeps the path away from ice. Half-beam gives baseline clearance;
        # planning_clearance_margin adds extra room (avoids collision when ship tracks at 4 m/s over many frames).
        inflation_radius = 0.5 * ship_beam + planning_clearance_margin
        cost_map.dilate(inflation_radius)

    if wall_bounds is not None:
        wx_min, wx_max, wy_min, wy_max = wall_bounds
        WALL_COST = 1e8
        nrows, ncols = cost_map.rows, cost_map.cols
        col_x = x_origin + (np.arange(ncols) + 0.5) * resolution
        row_y = y_origin + (np.arange(nrows) + 0.5) * resolution
        outside_x = (col_x < wx_min) | (col_x > wx_max)
        outside_y = (row_y < wy_min) | (row_y > wy_max)
        cost_map.cost_map[outside_y, :] = WALL_COST
        cost_map.cost_map[:, outside_x] = WALL_COST

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
