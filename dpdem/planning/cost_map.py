"""
Cost map for path planning.

Converts ice field state to a cost map for the lattice planner.
Supports thickness-based cost (DPDEM legacy) and KE-based cost (AUTO-IceNav).
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Max cost for impassable cells (AUTO-IceNav style)
MAX_COST = 1e10


@dataclass
class Obstacle:
    """An obstacle (ice floe) in the cost map."""
    vertices: np.ndarray  # Polygon vertices
    centroid: np.ndarray  # Center position
    mass: float           # Mass in kg
    area: float           # Area in m^2
    

def _compute_sic(bin_img: np.ndarray, kernel: Tuple[int, int]) -> np.ndarray:
    """Compute sea ice concentration via local mean of binary occupancy (AUTO-IceNav style)."""
    from scipy.ndimage import uniform_filter
    kh, kw = kernel
    if kh % 2 == 0:
        kh += 1
    if kw % 2 == 0:
        kw += 1
    sic = uniform_filter(bin_img.astype(np.float64), size=(kh, kw), mode="reflect")
    return np.clip(sic, 0.0, 1.0)


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
        use_ke_cost: bool = False,
        ship_speed: float = 4.0,
        ke_cost_scale: float = 1e-6,
        obs_cost_buffer_factor: float = 0.0,
        ice_resistance_weight: float = 0.0,
        sic_kernel: Optional[Tuple[int, int]] = None,
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
            collision_cost: Base cost multiplier for ice cells (thickness model) or
                collision_cost_weight (KE model).
            mass_cost_scale: Scaling factor for mass-based costs (legacy flat model)
            damage_cost_scale: Legacy power-law damage cost (kept for backward compat).
            damage_thickness_exponent: Legacy exponent for power-law damage cost.
            thickness_ref: Reference thickness [m] for cost scaling.
            ice_cost_exp_scale: Exponential growth rate for thickness-based cost.
            use_ke_cost: If True, use AUTO-IceNav KE-based cost (speed, mass, radial falloff).
            ship_speed: Ship speed [m/s] for KE cost (when use_ke_cost).
            ke_cost_scale: Scale KE [J] to planning units (default 1e-6 for cell-level mass).
            obs_cost_buffer_factor: Polygon swell factor for collision cost (0 = no swell).
            ice_resistance_weight: SIC exponent; cost *= SIC^weight (0 = no SIC).
            sic_kernel: (kh, kw) for SIC computation; must be odd. Required if ice_resistance_weight > 0.
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
        self.use_ke_cost = use_ke_cost
        self.ship_speed = ship_speed
        self.ke_cost_scale = ke_cost_scale
        self.obs_cost_buffer_factor = obs_cost_buffer_factor
        self.ice_resistance_weight = ice_resistance_weight
        self.sic_kernel = sic_kernel

        # Grid dimensions
        self.cols = int(np.ceil(width / resolution))
        self.rows = int(np.ceil(height / resolution))

        # Cost grid
        self.cost_map = np.full((self.rows, self.cols), base_cost)

        # Binary occupancy (ice=1, water=0) for SIC (AUTO-IceNav)
        self.bin_occ_map: Optional[np.ndarray] = None

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
            thicknesses: N array of ice thicknesses [m] (optional)
        """
        # Reset cost map (KE mode: start from 0; thickness mode: base_cost)
        if self.use_ke_cost:
            self.cost_map.fill(0.0)
            self.bin_occ_map = np.zeros((self.rows, self.cols), dtype=bool)
        else:
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

            if self.use_ke_cost:
                self._rasterize_polygon_ke(poly_world, mass)
            else:
                self._rasterize_polygon(poly_world, mass, thickness=thick)

        # KE mode: apply SIC and set base_cost for open water
        if self.use_ke_cost:
            if self.ice_resistance_weight > 0 and self.sic_kernel is not None:
                sic = _compute_sic(self.bin_occ_map, self.sic_kernel)
                self.cost_map *= np.power(np.maximum(sic, 1e-6), self.ice_resistance_weight)
            # Open water cells: set to base_cost (avoid 0 for path-cost accumulation)
            self.cost_map[self.cost_map < self.base_cost] = self.base_cost
    
    def _polygon_area(self, vertices: np.ndarray) -> float:
        """Compute polygon area using shoelace formula."""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]
        return abs(area) / 2.0
    
    def apply_ridge_cost(
        self,
        ridge_costmap: np.ndarray,
        ridge_scale: float,
        polygon_vertices: List[np.ndarray],
        ridge_weight: float,
    ) -> None:
        """
        Add ridge density cost to ice cells (AUTO-IceNav style).

        Ridge cost is added only to cells inside ice polygons. Ridge density is
        looked up at world coordinates; cells outside the ridge costmap get 0.

        Args:
            ridge_costmap: (H, W) array, ridge density 0–1, extent [0, W*scale] x [0, H*scale] m.
            ridge_scale: Meters per pixel of ridge costmap.
            polygon_vertices: List of polygon vertex arrays in world coordinates.
            ridge_weight: Cost = ridge_weight * ridge_density (additive).
        """
        rH, rW = ridge_costmap.shape
        for vertices in polygon_vertices:
            min_x, min_y = vertices.min(axis=0)
            max_x, max_y = vertices.max(axis=0)
            min_row, min_col = self.world_to_grid(min_x, min_y)
            max_row, max_col = self.world_to_grid(max_x, max_y)
            min_row = max(0, min_row)
            min_col = max(0, min_col)
            max_row = min(self.rows - 1, max_row)
            max_col = min(self.cols - 1, max_col)
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    x, y = self.grid_to_world(row, col)
                    if not self._point_in_polygon(x, y, vertices):
                        continue
                    # World (x,y) -> ridge pixel (col, row); ridge extent [0, rW*scale] x [0, rH*scale]
                    rx = int(x / ridge_scale)
                    ry = int(y / ridge_scale)
                    if 0 <= ry < rH and 0 <= rx < rW:
                        density = float(ridge_costmap[ry, rx])
                        self.cost_map[row, col] += ridge_weight * density

    def _rasterize_polygon_ke(self, vertices: np.ndarray, mass: float) -> None:
        """
        Rasterize polygon with KE-based cost and radial falloff (AUTO-IceNav style).

        constant_term = collision_cost * (v² * m_ice * m_ship * (m_ice + 2*m_ship)) / (2 * (m_ship + m_ice)²)
        cost at cell = constant_term * ((radius + 0.5*res)² - dist²) / (radius + 0.5*res)²
        """
        # Optionally swell polygon for collision cost
        poly = np.asarray(vertices, dtype=float)
        if self.obs_cost_buffer_factor > 0:
            centroid = poly.mean(axis=0)
            poly = centroid + (1.0 + self.obs_cost_buffer_factor) * (poly - centroid)

        centroid = poly.mean(axis=0)
        radius = float(np.max(np.linalg.norm(poly - centroid, axis=1)))
        radius_eff = radius + 0.5 * self.resolution

        # KE-based constant term (AUTO-IceNav Eq.)
        # Raw value is in J (energy); scale by 1e-12 so typical costs are ~1-1e4 (traversable).
        v = max(self.ship_speed, 1e-6)
        m_s, m_i = self.ship_mass, mass
        constant_term = self.collision_cost * (
            v * v * m_i * m_s * (m_i + 2 * m_s)
        ) / (2.0 * (m_s + m_i) ** 2)
        constant_term = constant_term * self.ke_cost_scale  # Normalize KE [J] to planning scale
        constant_term = min(constant_term, MAX_COST * 0.1)  # Keep traversable

        min_x, min_y = poly.min(axis=0)
        max_x, max_y = poly.max(axis=0)
        min_row, min_col = self.world_to_grid(min_x, min_y)
        max_row, max_col = self.world_to_grid(max_x, max_y)
        min_row = max(0, min_row)
        min_col = max(0, min_col)
        max_row = min(self.rows - 1, max_row)
        max_col = min(self.cols - 1, max_col)

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                x, y = self.grid_to_world(row, col)
                if not self._point_in_polygon(x, y, poly):
                    continue
                dist_sq = (x - centroid[0]) ** 2 + (y - centroid[1]) ** 2
                falloff = max(0.0, (radius_eff**2 - dist_sq) / (radius_eff**2))
                cell_cost = constant_term * falloff
                self.cost_map[row, col] = max(self.cost_map[row, col], cell_cost)

        # bin_occ_map: use original (non-swelled) polygon for SIC
        v_orig = np.asarray(vertices, dtype=float)
        min_x, min_y = v_orig.min(axis=0)
        max_x, max_y = v_orig.max(axis=0)
        r0, c0 = self.world_to_grid(min_x, min_y)
        r1, c1 = self.world_to_grid(max_x, max_y)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(self.rows - 1, r1), min(self.cols - 1, c1)
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                x, y = self.grid_to_world(row, col)
                if self._point_in_polygon(x, y, v_orig):
                    self.bin_occ_map[row, col] = True

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
