"""
Lattice-based path planner using A* search.

Plans paths through ice fields by searching a graph of motion primitives.
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field

from .primitives import Primitives, Primitive
from .cost_map import CostMap


@dataclass(order=True)
class Node:
    """A node in the search graph."""
    f_cost: float  # Total cost (g + h) - used for priority queue ordering
    g_cost: float = field(compare=False)  # Cost from start
    h_cost: float = field(compare=False)  # Heuristic cost to goal
    x: float = field(compare=False)
    y: float = field(compare=False)
    heading_idx: int = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)
    primitive: Optional[Primitive] = field(default=None, compare=False)


class LatticePlanner:
    """
    A* lattice planner for ice navigation.
    
    Searches a lattice graph using motion primitives to find
    a minimum-cost path from start to goal.
    """
    
    def __init__(
        self,
        primitives: Primitives,
        cost_map: CostMap,
        ship_length: float,
        ship_width: float,
        heuristic_weight: float = 1.0,
        goal_tolerance: float = 5.0,
    ):
        """
        Initialize planner.
        
        Args:
            primitives: Motion primitive set
            cost_map: Cost map for collision checking
            ship_length: Ship length for swath computation [m]
            ship_width: Ship width for swath computation [m]
            heuristic_weight: Weight for heuristic (>1 for weighted A*)
            goal_tolerance: Distance to goal for termination [m]
        """
        self.primitives = primitives
        self.cost_map = cost_map
        self.ship_length = ship_length
        self.ship_width = ship_width
        self.heuristic_weight = heuristic_weight
        self.goal_tolerance = goal_tolerance
        
        # Search statistics
        self.nodes_expanded = 0
        self.nodes_generated = 0
    
    def plan(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float],
        max_iterations: int = 10000,
    ) -> Optional[np.ndarray]:
        """
        Plan a path from start to goal.
        
        Args:
            start: Start pose (x, y, heading)
            goal: Goal position (x, y)
            max_iterations: Maximum search iterations
            
        Returns:
            Path as Nx3 array [x, y, heading], or None if no path found
        """
        self.nodes_expanded = 0
        self.nodes_generated = 0
        
        # Convert start heading to index
        start_heading_idx = self.primitives.world_heading_to_index(start[2])
        
        # Initialize start node
        h_start = self._heuristic(start[0], start[1], goal)
        start_node = Node(
            f_cost=h_start,
            g_cost=0.0,
            h_cost=h_start,
            x=start[0],
            y=start[1],
            heading_idx=start_heading_idx,
        )
        
        # Priority queue (min-heap on f_cost)
        open_set: List[Node] = [start_node]
        heapq.heapify(open_set)
        
        # Closed set - track visited (x, y, heading) states
        closed_set: Set[Tuple[int, int, int]] = set()
        
        # Best g-cost to reach each state
        g_costs: Dict[Tuple[int, int, int], float] = {}
        g_costs[self._state_key(start_node)] = 0.0
        
        while open_set and self.nodes_expanded < max_iterations:
            # Pop node with lowest f-cost
            current = heapq.heappop(open_set)
            
            # Check if already visited
            current_key = self._state_key(current)
            if current_key in closed_set:
                continue
            
            closed_set.add(current_key)
            self.nodes_expanded += 1
            
            # Check if goal reached
            dist_to_goal = np.sqrt((current.x - goal[0])**2 + (current.y - goal[1])**2)
            if dist_to_goal < self.goal_tolerance:
                return self._reconstruct_path(current)
            
            # Expand node
            for primitive in self.primitives.get_primitives(current.heading_idx):
                # Transform primitive to world coordinates
                world_heading = self.primitives.index_to_world_heading(current.heading_idx)
                path = self.primitives.transform_primitive(
                    primitive, current.x, current.y, world_heading
                )
                
                # End position
                end_x, end_y = path[-1, 0], path[-1, 1]
                end_heading_idx = primitive.end_heading
                
                # Check bounds
                if not self._in_bounds(end_x, end_y):
                    continue
                
                # Compute traversal cost
                traversal_cost = self._compute_path_cost(path, primitive)
                
                # Skip if path goes through high-cost area (collision)
                if traversal_cost > 1e6:
                    continue
                
                # New g-cost
                new_g = current.g_cost + traversal_cost
                
                # Check if this path is better
                neighbor_key = self._discretize_state(end_x, end_y, end_heading_idx)
                if neighbor_key in g_costs and g_costs[neighbor_key] <= new_g:
                    continue
                
                g_costs[neighbor_key] = new_g
                
                # Compute heuristic
                h = self._heuristic(end_x, end_y, goal)
                f = new_g + self.heuristic_weight * h
                
                # Create neighbor node
                neighbor = Node(
                    f_cost=f,
                    g_cost=new_g,
                    h_cost=h,
                    x=end_x,
                    y=end_y,
                    heading_idx=end_heading_idx,
                    parent=current,
                    primitive=primitive,
                )
                
                heapq.heappush(open_set, neighbor)
                self.nodes_generated += 1
        
        # No path found
        return None
    
    def _state_key(self, node: Node) -> Tuple[int, int, int]:
        """Get discretized state key for a node."""
        return self._discretize_state(node.x, node.y, node.heading_idx)
    
    def _discretize_state(
        self,
        x: float,
        y: float,
        heading_idx: int,
    ) -> Tuple[int, int, int]:
        """Discretize continuous state to grid cell (same frame as cost map)."""
        row, col = self.cost_map.world_to_grid(x, y)
        return (col, row, heading_idx)
    
    def _in_bounds(self, x: float, y: float) -> bool:
        """Check if position is within map bounds (world coords)."""
        ox, oy = self.cost_map.x_origin, self.cost_map.y_origin
        return (ox <= x < ox + self.cost_map.width and
                oy <= y < oy + self.cost_map.height)
    
    def _heuristic(self, x: float, y: float, goal: Tuple[float, float]) -> float:
        """
        Compute heuristic cost to goal.
        
        Uses Euclidean distance (admissible for Dubins paths).
        """
        return np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    
    def _compute_path_cost(self, path: np.ndarray, primitive: Primitive) -> float:
        """
        Compute cost of traversing a primitive path.
        
        Includes:
        - Distance cost (arc length)
        - Collision cost (from cost map)
        - Primitive cost multiplier (penalizes turns)
        """
        # Base cost is arc length
        cost = primitive.length
        
        # Sample cost map along path
        n_samples = max(5, int(primitive.length / self.cost_map.resolution))
        indices = np.linspace(0, len(path) - 1, n_samples, dtype=int)
        
        for idx in indices:
            x, y, heading = path[idx]
            
            # Get cost at ship center
            cell_cost = self.cost_map.get_cost_world(x, y)
            
            # Also check ship corners (simplified swath)
            corners = self._get_ship_corners(x, y, heading)
            for cx, cy in corners:
                corner_cost = self.cost_map.get_cost_world(cx, cy)
                cell_cost = max(cell_cost, corner_cost)
            
            cost += cell_cost * (primitive.length / n_samples)
        
        # Apply primitive cost multiplier
        cost *= primitive.cost_multiplier
        
        return cost
    
    def _get_ship_corners(
        self,
        x: float,
        y: float,
        heading: float,
    ) -> List[Tuple[float, float]]:
        """Get ship corner positions for collision checking."""
        c, s = np.cos(heading), np.sin(heading)
        
        # Half dimensions
        half_l = self.ship_length / 2
        half_w = self.ship_width / 2
        
        # Corner offsets in body frame
        corners_body = [
            (half_l, half_w),   # Front right
            (half_l, -half_w),  # Front left
            (-half_l, half_w),  # Rear right
            (-half_l, -half_w), # Rear left
        ]
        
        # Transform to world frame
        corners = []
        for dx, dy in corners_body:
            wx = x + c * dx - s * dy
            wy = y + s * dx + c * dy
            corners.append((wx, wy))
        
        return corners
    
    def _reconstruct_path(self, goal_node: Node) -> np.ndarray:
        """Reconstruct path from goal node back to start."""
        path_points = []
        
        # Collect nodes from goal to start
        nodes = []
        current = goal_node
        while current is not None:
            nodes.append(current)
            current = current.parent
        
        nodes.reverse()
        
        # Build path from primitives
        for i, node in enumerate(nodes):
            if node.primitive is not None and node.parent is not None:
                # Get transformed primitive path
                world_heading = self.primitives.index_to_world_heading(node.parent.heading_idx)
                prim_path = self.primitives.transform_primitive(
                    node.primitive,
                    node.parent.x,
                    node.parent.y,
                    world_heading,
                )
                
                # Add path points (skip first if not start to avoid duplicates)
                start_idx = 0 if i == 1 else 1
                for j in range(start_idx, len(prim_path)):
                    path_points.append(prim_path[j])
            elif i == 0:
                # Start node
                heading = self.primitives.index_to_world_heading(node.heading_idx)
                path_points.append([node.x, node.y, heading])
        
        if not path_points:
            return np.array([[goal_node.x, goal_node.y, 
                            self.primitives.index_to_world_heading(goal_node.heading_idx)]])
        
        return np.array(path_points)
    
    def update_cost_map(
        self,
        positions: np.ndarray,
        polygons: List[np.ndarray],
        masses: np.ndarray,
        angles: Optional[np.ndarray] = None,
    ):
        """
        Update cost map with new ice field state.
        
        Args:
            positions: Nx2 array of ice floe positions
            polygons: List of polygon vertex arrays
            masses: N array of floe masses
            angles: N array of floe orientations
        """
        self.cost_map.update_from_ice_field(positions, polygons, masses, angles)
        
        # Dilate by half ship width for collision margin
        self.cost_map.dilate(self.ship_width / 2)


def create_planner(
    map_width: float,
    map_height: float,
    ship_length: float,
    ship_width: float,
    ship_mass: float,
    turning_radius: float,
    resolution: float = 1.0,
    num_headings: int = 16,
    heuristic_weight: float = 1.5,
) -> LatticePlanner:
    """
    Create a configured lattice planner.
    
    Convenience function to set up all components.
    
    Args:
        map_width: Map width [m]
        map_height: Map height [m]
        ship_length: Ship length [m]
        ship_width: Ship width [m]
        ship_mass: Ship mass [kg]
        turning_radius: Minimum turning radius [m]
        resolution: Grid resolution [m/cell]
        num_headings: Number of discrete headings
        heuristic_weight: Weight for weighted A*
        
    Returns:
        Configured LatticePlanner
    """
    # Create primitives
    primitives = Primitives(
        turning_radius=turning_radius,
        num_headings=num_headings,
        scale=resolution,
    )
    
    # Create cost map
    cost_map = CostMap(
        width=map_width,
        height=map_height,
        resolution=resolution,
        ship_mass=ship_mass,
    )
    
    # Create planner
    planner = LatticePlanner(
        primitives=primitives,
        cost_map=cost_map,
        ship_length=ship_length,
        ship_width=ship_width,
        heuristic_weight=heuristic_weight,
    )
    
    return planner
