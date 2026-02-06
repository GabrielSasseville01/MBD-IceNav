"""
Trajectory setpoint generation for path following.

Generates smooth setpoints along a reference path, accounting for
ship dynamics and speed constraints.
"""

import numpy as np
from typing import Tuple, Optional, List
from .gnc import ssa, euclid_dist


class TrajectorySetpoint:
    """
    Generates setpoints for tracking a reference path.
    
    Moves a virtual target point along the path at controlled speed,
    providing smooth position and heading references for the controller.
    """
    
    def __init__(
        self,
        path: np.ndarray,
        eta: np.ndarray,
        target_speed: float = 1.0,
        max_acceleration: float = 0.5,
        lookahead_distance: float = 10.0,
        current_speed: float = 0.0,
    ):
        """
        Initialize trajectory setpoint generator.
        
        Args:
            path: Nx3 array of waypoints [x, y, psi]
            eta: Current ship pose [x, y, psi]
            target_speed: Desired cruise speed [m/s]
            max_acceleration: Maximum acceleration [m/s^2]
            lookahead_distance: Lookahead distance for path following [m]
            current_speed: Current ship speed [m/s]
        """
        self.path = np.atleast_2d(path)
        self.target_speed = target_speed
        self.max_acceleration = max_acceleration
        self.lookahead_distance = lookahead_distance
        
        # Current virtual target position along path
        self.path_index = 0
        self.path_param = 0.0  # Parametric position along current segment [0, 1]
        
        # Speed profile
        self.current_speed = current_speed
        
        # Find closest point on path to start
        self._find_closest_point(eta[:2])
        
        # Current setpoint
        self.setpoint = self._compute_setpoint()
    
    def _find_closest_point(self, position: np.ndarray):
        """Find the closest point on the path to the given position."""
        min_dist = np.inf
        
        for i in range(len(self.path) - 1):
            p1 = self.path[i, :2]
            p2 = self.path[i + 1, :2]
            
            # Project position onto segment
            segment = p2 - p1
            segment_len = np.linalg.norm(segment)
            
            if segment_len < 1e-6:
                continue
            
            # Parametric projection
            t = np.dot(position - p1, segment) / (segment_len ** 2)
            t = np.clip(t, 0.0, 1.0)
            
            # Closest point on segment
            closest = p1 + t * segment
            dist = np.linalg.norm(position - closest)
            
            if dist < min_dist:
                min_dist = dist
                self.path_index = i
                self.path_param = t
    
    def _compute_setpoint(self) -> np.ndarray:
        """Compute current setpoint from path."""
        if self.path_index >= len(self.path) - 1:
            # At end of path
            return self.path[-1].copy()
        
        # Interpolate position on current segment
        p1 = self.path[self.path_index]
        p2 = self.path[self.path_index + 1]
        
        setpoint = p1 + self.path_param * (p2 - p1)
        
        # Compute heading from path direction
        if len(setpoint) < 3:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            psi = np.arctan2(dy, dx)
            setpoint = np.array([setpoint[0], setpoint[1], psi])
        
        return setpoint
    
    def update(self, dt: float) -> np.ndarray:
        """
        Update setpoint for next time step.
        
        Args:
            dt: Time step [s]
            
        Returns:
            Current setpoint [x, y, psi]
        """
        if self.path_index >= len(self.path) - 1:
            return self.setpoint
        
        # Update speed (trapezoidal profile)
        if self.current_speed < self.target_speed:
            self.current_speed = min(
                self.current_speed + self.max_acceleration * dt,
                self.target_speed
            )
        
        # Distance to travel this step
        distance = self.current_speed * dt
        
        # Move along path
        while distance > 0 and self.path_index < len(self.path) - 1:
            p1 = self.path[self.path_index, :2]
            p2 = self.path[self.path_index + 1, :2]
            
            segment = p2 - p1
            segment_len = np.linalg.norm(segment)
            
            if segment_len < 1e-6:
                self.path_index += 1
                self.path_param = 0.0
                continue
            
            # Remaining distance on this segment
            remaining = (1.0 - self.path_param) * segment_len
            
            if distance < remaining:
                # Stay on this segment
                self.path_param += distance / segment_len
                distance = 0
            else:
                # Move to next segment
                distance -= remaining
                self.path_index += 1
                self.path_param = 0.0
        
        self.setpoint = self._compute_setpoint()
        return self.setpoint
    
    def replan_update(
        self,
        current_speed: float,
        position: Tuple[float, float],
        new_path: np.ndarray,
    ):
        """
        Update path after replanning.
        
        Args:
            current_speed: Current ship speed
            position: Current ship position (x, y)
            new_path: New path array Nx3
        """
        self.path = np.atleast_2d(new_path)
        self.current_speed = current_speed
        self.path_index = 0
        self.path_param = 0.0
        
        # Find closest point on new path
        self._find_closest_point(np.array(position))
        self.setpoint = self._compute_setpoint()
    
    def get_lookahead_point(self, position: np.ndarray) -> np.ndarray:
        """
        Get lookahead point for pure pursuit style following.
        
        Args:
            position: Current position [x, y]
            
        Returns:
            Lookahead point [x, y, psi]
        """
        # Start from current closest point
        idx = self.path_index
        param = self.path_param
        remaining_dist = self.lookahead_distance
        
        while remaining_dist > 0 and idx < len(self.path) - 1:
            p1 = self.path[idx, :2]
            p2 = self.path[idx + 1, :2]
            
            segment_len = np.linalg.norm(p2 - p1)
            if segment_len < 1e-6:
                idx += 1
                param = 0.0
                continue
            
            remaining_on_segment = (1.0 - param) * segment_len
            
            if remaining_dist < remaining_on_segment:
                param += remaining_dist / segment_len
                break
            else:
                remaining_dist -= remaining_on_segment
                idx += 1
                param = 0.0
        
        # Compute lookahead point
        if idx >= len(self.path) - 1:
            return self.path[-1].copy()
        
        p1 = self.path[idx]
        p2 = self.path[idx + 1]
        point = p1 + param * (p2 - p1)
        
        # Heading from path direction
        if len(point) < 3:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            psi = np.arctan2(dy, dx)
            point = np.array([point[0], point[1], psi])
        
        return point
    
    @property
    def is_complete(self) -> bool:
        """Check if we've reached the end of the path."""
        return self.path_index >= len(self.path) - 1
    
    @property
    def progress(self) -> float:
        """Return progress along path as fraction [0, 1]."""
        if len(self.path) <= 1:
            return 1.0
        
        # Count completed segments plus current segment progress
        total_segments = len(self.path) - 1
        completed = self.path_index + self.path_param
        return completed / total_segments
