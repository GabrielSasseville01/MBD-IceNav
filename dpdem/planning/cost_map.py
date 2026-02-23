"""
Cost map for path planning.

Converts ice field state to a cost map for the lattice planner.
Higher costs represent areas with more/thicker ice.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Obstacle:
    """An obstacle (ice floe) in the cost map."""
    vertices: np.ndarray  # Polygon vertices
    centroid: np.ndarray  # Center position
    mass: float           # Mass in kg
    area: float           # Area in m^2
    

class CostMap:
    """
    Occupancy and cost map for ice navigation planning.
    
    Converts ice floe positions and properties to a discretized cost grid.
    """
    
    def __init__(
        self,
        width: float,
        height: float,
        resolution: float = 1.0,
        x_origin: float = 0.0,
        y_origin: float = 0.0,
        ship_mass: float = 1e6,
        base_cost: float = 1.0,
        collision_cost: float = 100.0,
        mass_cost_scale: float = 1e-5,
        damage_cost_scale: float = 0.0,
        damage_thickness_exponent: float = 0.5,
        thickness_ref: float = 1.0,
        ice_cost_exp_scale: float = 0.0,
    ):
        """
        Initialize cost map.

        World coordinates (x, y) map to grid (row, col) via:
          col = (x - x_origin) / resolution,  row = (y - y_origin) / resolution
        so the map covers [x_origin, x_origin+width) x [y_origin, y_origin+height) in world units.

        Args:
            width: Map width [m]
            height: Map height [m]
            resolution: Grid resolution [m/cell]
            x_origin: World x of grid column 0 [m]
            y_origin: World y of grid row 0 [m]
            ship_mass: Ship mass for relative cost calculation [kg]
            base_cost: Base traversal cost per cell (open water)
            collision_cost: Base cost multiplier for ice cells (used as the
                coefficient in front of the exponential thickness model).
            mass_cost_scale: Scaling factor for mass-based costs (legacy flat model)
            damage_cost_scale: Legacy power-law damage cost (kept for backward compat).
            damage_thickness_exponent: Legacy exponent for power-law damage cost.
            thickness_ref: Reference thickness [m] for cost scaling.
            ice_cost_exp_scale: Exponential growth rate for thickness-based cost.
                When > 0 and thickness is provided, cell cost is:
                  collision_cost * exp(ice_cost_exp_scale * thickness / thickness_ref)
                This gives thin ice a moderate cost and thick ice an extremely
                high cost while still keeping it technically traversable.
                When 0, falls back to the legacy flat collision_cost model.
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.x_origin = x_origin
        self.y_origin = y_origin
        self.ship_mass = ship_mass
        self.base_cost = base_cost
        self.collision_cost = collision_cost
        self.mass_cost_scale = mass_cost_scale
        self.damage_cost_scale = damage_cost_scale
        self.damage_thickness_exponent = damage_thickness_exponent
        self.thickness_ref = thickness_ref
        self.ice_cost_exp_scale = ice_cost_exp_scale

        # Grid dimensions
        self.cols = int(np.ceil(width / resolution))
        self.rows = int(np.ceil(height / resolution))

        # Cost grid
        self.cost_map = np.full((self.rows, self.cols), base_cost)

        # Obstacle list
        self.obstacles: List[Obstacle] = []
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return (rows, cols) shape."""
        return (self.rows, self.cols)
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        col = int((x - self.x_origin) / self.resolution)
        row = int((y - self.y_origin) / self.resolution)
        return (row, col)

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = self.x_origin + (col + 0.5) * self.resolution
        y = self.y_origin + (row + 0.5) * self.resolution
        return (x, y)
    
    def in_bounds(self, row: int, col: int) -> bool:
        """Check if grid cell is within bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def update_from_ice_field(
        self,
        positions: np.ndarray,
        polygons: List[np.ndarray],
        masses: np.ndarray,
        angles: Optional[np.ndarray] = None,
        thicknesses: Optional[np.ndarray] = None,
    ):
        """
        Update cost map from ice field state.

        Args:
            positions: Nx2 array of ice floe positions
            polygons: List of polygon vertex arrays (local coordinates)
            masses: N array of floe masses
            angles: N array of floe orientations (optional)
            thicknesses: N array of ice thicknesses [m] (optional; used for damage-potential cost when damage_cost_scale > 0)
        """
        # Reset cost map
        self.cost_map.fill(self.base_cost)
        self.obstacles.clear()

        n_floes = len(positions)
        if angles is None:
            angles = np.zeros(n_floes)

        for i in range(n_floes):
            pos = positions[i]
            poly_local = polygons[i]
            mass = masses[i]
            angle = angles[i]
            thick = float(thicknesses[i]) if thicknesses is not None and i < len(thicknesses) else None

            # Transform polygon to world coordinates
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            poly_world = poly_local @ R.T + pos

            # Compute area
            area = self._polygon_area(poly_world)

            # Store obstacle
            self.obstacles.append(Obstacle(
                vertices=poly_world,
                centroid=pos,
                mass=mass,
                area=area,
            ))

            # Rasterize polygon to grid (optionally with damage potential from thickness)
            self._rasterize_polygon(poly_world, mass, thickness=thick)
    
    def _polygon_area(self, vertices: np.ndarray) -> float:
        """Compute polygon area using shoelace formula."""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]
        return abs(area) / 2.0
    
    def _rasterize_polygon(self, vertices: np.ndarray, mass: float, thickness: Optional[float] = None):
        """Rasterize polygon to grid cells with thickness-dependent cost."""
        min_x, min_y = vertices.min(axis=0)
        max_x, max_y = vertices.max(axis=0)

        min_row, min_col = self.world_to_grid(min_x, min_y)
        max_row, max_col = self.world_to_grid(max_x, max_y)

        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(self.rows - 1, max_row)
        max_col = min(self.cols - 1, max_col)

        if self.ice_cost_exp_scale > 0 and thickness is not None and thickness > 0:
            t_ratio = thickness / max(self.thickness_ref, 1e-6)
            cell_cost = self.collision_cost * np.exp(self.ice_cost_exp_scale * t_ratio)
        else:
            relative_mass = mass / self.ship_mass
            cell_cost = self.collision_cost * (1.0 + relative_mass * self.mass_cost_scale)

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                x, y = self.grid_to_world(row, col)
                if self._point_in_polygon(x, y, vertices):
                    self.cost_map[row, col] = max(self.cost_map[row, col], cell_cost)
    
    def _point_in_polygon(self, x: float, y: float, vertices: np.ndarray) -> bool:
        """Check if point is inside polygon (ray casting)."""
        n = len(vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def get_cost(self, row: int, col: int) -> float:
        """Get cost at grid cell."""
        if not self.in_bounds(row, col):
            return np.inf
        return self.cost_map[row, col]
    
    def get_cost_world(self, x: float, y: float) -> float:
        """Get cost at world coordinates."""
        row, col = self.world_to_grid(x, y)
        return self.get_cost(row, col)
    
    def get_path_cost(self, path: np.ndarray) -> float:
        """
        Compute total cost along a path.
        
        Args:
            path: Nx2 or Nx3 array of waypoints
            
        Returns:
            Total accumulated cost
        """
        total_cost = 0.0
        
        for i in range(len(path) - 1):
            p1 = path[i, :2]
            p2 = path[i + 1, :2]
            
            # Sample points along segment
            dist = np.linalg.norm(p2 - p1)
            n_samples = max(2, int(dist / self.resolution))
            
            for t in np.linspace(0, 1, n_samples):
                p = p1 + t * (p2 - p1)
                total_cost += self.get_cost_world(p[0], p[1])
        
        return total_cost
    
    def dilate(self, radius: float):
        """
        Dilate obstacles by given radius (for collision margin).
        
        Args:
            radius: Dilation radius [m]
        """
        from scipy.ndimage import maximum_filter
        
        # Kernel size in cells
        kernel_size = int(np.ceil(2 * radius / self.resolution)) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply maximum filter (morphological dilation)
        self.cost_map = maximum_filter(self.cost_map, size=kernel_size)
