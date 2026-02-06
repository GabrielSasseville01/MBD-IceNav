"""
Motion primitives for lattice-based planning.

Generates Dubins-like motion primitives that respect ship kinematics
(minimum turning radius constraint).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Primitive:
    """A single motion primitive."""
    # Discretized path points: Nx3 array of [x, y, heading]
    path: np.ndarray
    # Start heading index
    start_heading: int
    # End heading index  
    end_heading: int
    # Arc length of primitive
    length: float
    # Cost multiplier (for preferring certain maneuvers)
    cost_multiplier: float = 1.0


class Primitives:
    """
    Motion primitive generator for lattice planning.
    
    Generates discrete motion primitives that connect lattice nodes
    while respecting the ship's kinematic constraints (minimum turning radius).
    """
    
    def __init__(
        self,
        turning_radius: float,
        num_headings: int = 16,
        scale: float = 1.0,
        num_arc_points: int = 20,
    ):
        """
        Initialize primitives.
        
        Args:
            turning_radius: Minimum turning radius [m]
            num_headings: Number of discrete heading angles
            scale: Grid scale factor (meters per cell)
            num_arc_points: Number of points to discretize each arc
        """
        self.turning_radius = turning_radius * scale
        self.num_headings = num_headings
        self.scale = scale
        self.num_arc_points = num_arc_points
        
        # Heading resolution
        self.heading_resolution = 2 * np.pi / num_headings
        
        # Generate primitive set
        self.primitives: Dict[int, List[Primitive]] = {}
        self._generate_primitives()
    
    def _generate_primitives(self):
        """Generate all primitives for each heading."""
        for start_h in range(self.num_headings):
            self.primitives[start_h] = []
            
            start_angle = start_h * self.heading_resolution
            
            # Generate primitives for different end headings
            # Allow turns up to 90 degrees (4 heading steps for 16 headings)
            max_heading_change = self.num_headings // 4
            
            for dh in range(-max_heading_change, max_heading_change + 1):
                end_h = (start_h + dh) % self.num_headings
                end_angle = end_h * self.heading_resolution
                
                # Generate path for this primitive
                if dh == 0:
                    # Straight line
                    path = self._generate_straight(start_angle)
                else:
                    # Arc
                    path = self._generate_arc(start_angle, end_angle, dh > 0)
                
                if path is not None:
                    # Compute arc length
                    length = self._compute_path_length(path)
                    
                    # Cost multiplier: slight penalty for turns
                    cost_mult = 1.0 + 0.1 * abs(dh)
                    
                    prim = Primitive(
                        path=path,
                        start_heading=start_h,
                        end_heading=end_h,
                        length=length,
                        cost_multiplier=cost_mult,
                    )
                    self.primitives[start_h].append(prim)
    
    def _generate_straight(self, heading: float) -> np.ndarray:
        """Generate straight line primitive."""
        # Move forward by one turning radius (typical primitive length)
        length = self.turning_radius
        
        dx = length * np.cos(heading)
        dy = length * np.sin(heading)
        
        # Discretize
        t = np.linspace(0, 1, self.num_arc_points)
        x = t * dx
        y = t * dy
        psi = np.full_like(t, heading)
        
        return np.column_stack([x, y, psi])
    
    def _generate_arc(
        self,
        start_angle: float,
        end_angle: float,
        turn_left: bool,
    ) -> Optional[np.ndarray]:
        """Generate arc primitive (Dubins path segment)."""
        # Angular change
        if turn_left:
            dtheta = (end_angle - start_angle) % (2 * np.pi)
            if dtheta > np.pi:
                dtheta -= 2 * np.pi
        else:
            dtheta = (end_angle - start_angle) % (2 * np.pi)
            if dtheta < -np.pi:
                dtheta += 2 * np.pi
            if dtheta > 0:
                dtheta -= 2 * np.pi
        
        # Arc center
        R = self.turning_radius
        if turn_left:
            # Center is to the left of start position
            cx = R * np.cos(start_angle + np.pi/2)
            cy = R * np.sin(start_angle + np.pi/2)
        else:
            # Center is to the right
            cx = R * np.cos(start_angle - np.pi/2)
            cy = R * np.sin(start_angle - np.pi/2)
        
        # Discretize arc
        t = np.linspace(0, 1, self.num_arc_points)
        theta = start_angle + t * dtheta
        
        # Start position is at origin, compute arc
        if turn_left:
            x = cx + R * np.cos(theta - np.pi/2) - cx + R * np.sin(start_angle)
            y = cy + R * np.sin(theta - np.pi/2) - cy - R * np.cos(start_angle)
        else:
            x = cx + R * np.cos(theta + np.pi/2) - cx - R * np.sin(start_angle)
            y = cy + R * np.sin(theta + np.pi/2) - cy + R * np.cos(start_angle)
        
        # Simpler parametric arc
        # Arc from (0,0) heading start_angle, turning through dtheta
        angles = start_angle + t * dtheta
        
        # Integrate to get positions
        x = np.zeros(len(t))
        y = np.zeros(len(t))
        ds = R * abs(dtheta) / (len(t) - 1)
        
        for i in range(1, len(t)):
            avg_angle = (angles[i-1] + angles[i]) / 2
            x[i] = x[i-1] + ds * np.cos(avg_angle)
            y[i] = y[i-1] + ds * np.sin(avg_angle)
        
        return np.column_stack([x, y, angles])
    
    def _compute_path_length(self, path: np.ndarray) -> float:
        """Compute total arc length of path."""
        diffs = np.diff(path[:, :2], axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    def get_primitives(self, heading_idx: int) -> List[Primitive]:
        """Get all primitives starting from given heading."""
        return self.primitives.get(heading_idx, [])
    
    def world_heading_to_index(self, heading: float) -> int:
        """Convert world heading [rad] to discrete heading index."""
        heading = heading % (2 * np.pi)
        return int(round(heading / self.heading_resolution)) % self.num_headings
    
    def index_to_world_heading(self, idx: int) -> float:
        """Convert discrete heading index to world heading [rad]."""
        return (idx % self.num_headings) * self.heading_resolution
    
    def transform_primitive(
        self,
        primitive: Primitive,
        x: float,
        y: float,
        heading: float,
    ) -> np.ndarray:
        """
        Transform primitive path to world coordinates.
        
        Args:
            primitive: Motion primitive
            x, y: Start position in world frame
            heading: Start heading in world frame
            
        Returns:
            Transformed path Nx3 [x, y, psi]
        """
        # Rotation angle (difference from primitive's reference heading)
        ref_heading = self.index_to_world_heading(primitive.start_heading)
        rotation = heading - ref_heading
        
        # Rotation matrix
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        
        # Transform positions
        positions = primitive.path[:, :2] @ R.T
        positions[:, 0] += x
        positions[:, 1] += y
        
        # Transform headings
        headings = primitive.path[:, 2] + rotation
        
        return np.column_stack([positions, headings])
