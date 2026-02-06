"""
Autonomous navigation interface for DPDEM.

Bridges the lattice planner and DP controller with the DPDEM simulation,
enabling autonomous ship navigation through ice fields.
"""

import numpy as np
from typing import Tuple, Optional, List, NamedTuple
from dataclasses import dataclass, field

from .control import ShipState, DPController, TrajectorySetpoint
from .planning import LatticePlanner, CostMap, Primitives, create_planner


@dataclass
class NavigationConfig:
    """Configuration for autonomous navigation."""
    # Ship parameters
    ship_length: float = 50.0       # [m]
    ship_width: float = 15.0        # [m]
    ship_mass: float = 1e6          # [kg]
    turning_radius: float = 100.0   # [m]
    
    # Map parameters
    map_width: float = 1000.0       # [m]
    map_height: float = 1000.0      # [m]
    map_resolution: float = 2.0     # [m/cell]
    
    # Planner parameters
    num_headings: int = 16
    heuristic_weight: float = 1.5
    replan_interval: float = 5.0    # [s] - how often to replan
    goal_tolerance: float = 10.0    # [m]
    
    # Controller parameters
    wn: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.1]))
    zeta: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    target_speed: float = 2.0       # [m/s]
    lookahead_distance: float = 20.0  # [m]
    
    # Safety parameters
    max_force: float = 1e6          # [N] max control force
    max_moment: float = 5e6         # [Nm] max control moment


class NavigationState(NamedTuple):
    """Current navigation state."""
    ship_state: ShipState
    goal: np.ndarray
    path: Optional[np.ndarray]
    setpoint: np.ndarray
    control_force: np.ndarray
    distance_to_goal: float
    path_progress: float
    is_complete: bool


class AutonomousNavigator:
    """
    Autonomous navigation system for DPDEM simulation.
    
    Integrates:
    - Lattice-based path planner for global planning
    - DP controller for trajectory tracking
    - Trajectory setpoint generator for smooth motion
    """
    
    def __init__(self, config: NavigationConfig = None):
        """
        Initialize autonomous navigator.
        
        Args:
            config: Navigation configuration
        """
        self.config = config or NavigationConfig()
        
        # Create planner
        self.planner = create_planner(
            map_width=self.config.map_width,
            map_height=self.config.map_height,
            ship_length=self.config.ship_length,
            ship_width=self.config.ship_width,
            ship_mass=self.config.ship_mass,
            turning_radius=self.config.turning_radius,
            resolution=self.config.map_resolution,
            num_headings=self.config.num_headings,
            heuristic_weight=self.config.heuristic_weight,
        )
        
        # Create controller
        self.controller = DPController.from_ship_params(
            mass=self.config.ship_mass,
            length=self.config.ship_length,
            breadth=self.config.ship_width,
            wn=self.config.wn,
            zeta=self.config.zeta,
        )
        
        # Navigation state
        self.goal: Optional[np.ndarray] = None
        self.path: Optional[np.ndarray] = None
        self.trajectory: Optional[TrajectorySetpoint] = None
        
        # Timing
        self.time_since_replan = 0.0
        self.total_time = 0.0
        
        # Ship state (updated from simulation)
        self.ship_state = ShipState()
    
    def set_goal(self, goal: Tuple[float, float]):
        """
        Set navigation goal.
        
        Args:
            goal: Goal position (x, y)
        """
        self.goal = np.array(goal)
        self.path = None
        self.trajectory = None
        self.time_since_replan = self.config.replan_interval  # Force replan
        self.controller.reset()
    
    def update_ice_field(
        self,
        positions: np.ndarray,
        polygons: List[np.ndarray],
        masses: np.ndarray,
        angles: Optional[np.ndarray] = None,
    ):
        """
        Update the cost map with current ice field state.
        
        Call this each step to update the planner's knowledge of ice.
        
        Args:
            positions: Nx2 array of ice floe positions
            polygons: List of polygon vertex arrays
            masses: N array of floe masses
            angles: N array of floe orientations
        """
        self.planner.update_cost_map(positions, polygons, masses, angles)
    
    def update_from_platform(self, platform) -> ShipState:
        """
        Update ship state from DPDEM Platform.
        
        Args:
            platform: DPDEM Platform NamedTuple
            
        Returns:
            Updated ShipState
        """
        self.ship_state = ShipState.from_platform(platform)
        return self.ship_state
    
    def step(self, dt: float) -> Tuple[np.ndarray, NavigationState]:
        """
        Perform one navigation step.
        
        Args:
            dt: Time step [s]
            
        Returns:
            control_force: Control forces/moments [X, Y, N]
            nav_state: Current navigation state
        """
        if self.goal is None:
            # No goal set - return zero control
            return np.zeros(3), self._get_navigation_state(np.zeros(3))
        
        # Check if we've reached the goal
        dist_to_goal = np.linalg.norm(self.ship_state.position - self.goal)
        if dist_to_goal < self.config.goal_tolerance:
            # Goal reached - station keeping
            setpoint = np.array([self.goal[0], self.goal[1], self.ship_state.psi])
            control = self.controller.compute_control(
                self.ship_state.eta,
                self.ship_state.nu,
                setpoint,
                dt,
            )
            control = self._saturate_control(control)
            return control, self._get_navigation_state(control, is_complete=True)
        
        # Update timing
        self.time_since_replan += dt
        self.total_time += dt
        
        # Replan if needed
        if self.time_since_replan >= self.config.replan_interval or self.path is None:
            self._replan()
            self.time_since_replan = 0.0
        
        # Update trajectory setpoint
        if self.trajectory is not None:
            self.trajectory.update(dt)
            setpoint = self.trajectory.get_lookahead_point(self.ship_state.position)
        else:
            # No path - head directly to goal
            psi_desired = np.arctan2(
                self.goal[1] - self.ship_state.y,
                self.goal[0] - self.ship_state.x
            )
            setpoint = np.array([self.goal[0], self.goal[1], psi_desired])
        
        # Compute control
        control = self.controller.compute_control(
            self.ship_state.eta,
            self.ship_state.nu,
            setpoint,
            dt,
        )
        
        # Saturate control
        control = self._saturate_control(control)
        
        return control, self._get_navigation_state(control)
    
    def _replan(self):
        """Run the planner to compute a new path."""
        if self.goal is None:
            return
        
        # Plan from current position
        start = (self.ship_state.x, self.ship_state.y, self.ship_state.psi)
        goal = (self.goal[0], self.goal[1])
        
        path = self.planner.plan(start, goal)
        
        if path is not None and len(path) > 1:
            self.path = path
            
            # Create trajectory setpoint generator
            self.trajectory = TrajectorySetpoint(
                path=path,
                eta=self.ship_state.eta,
                target_speed=self.config.target_speed,
                lookahead_distance=self.config.lookahead_distance,
                current_speed=self.ship_state.speed,
            )
        else:
            # Planning failed - keep old path or go direct
            if self.path is None:
                # Create simple straight-line path
                psi = np.arctan2(
                    self.goal[1] - self.ship_state.y,
                    self.goal[0] - self.ship_state.x
                )
                self.path = np.array([
                    [self.ship_state.x, self.ship_state.y, self.ship_state.psi],
                    [self.goal[0], self.goal[1], psi]
                ])
                self.trajectory = TrajectorySetpoint(
                    path=self.path,
                    eta=self.ship_state.eta,
                    target_speed=self.config.target_speed,
                    current_speed=self.ship_state.speed,
                )
    
    def _saturate_control(self, control: np.ndarray) -> np.ndarray:
        """Apply saturation limits to control forces/moments."""
        control = control.copy()
        control[0] = np.clip(control[0], -self.config.max_force, self.config.max_force)
        control[1] = np.clip(control[1], -self.config.max_force, self.config.max_force)
        control[2] = np.clip(control[2], -self.config.max_moment, self.config.max_moment)
        return control
    
    def _get_navigation_state(
        self,
        control: np.ndarray,
        is_complete: bool = False,
    ) -> NavigationState:
        """Construct navigation state for output."""
        dist_to_goal = 0.0
        if self.goal is not None:
            dist_to_goal = np.linalg.norm(self.ship_state.position - self.goal)
        
        progress = 0.0
        if self.trajectory is not None:
            progress = self.trajectory.progress
        
        setpoint = np.zeros(3)
        if self.trajectory is not None:
            setpoint = self.trajectory.setpoint
        elif self.goal is not None:
            setpoint = np.array([self.goal[0], self.goal[1], self.ship_state.psi])
        
        return NavigationState(
            ship_state=self.ship_state.copy(),
            goal=self.goal.copy() if self.goal is not None else np.zeros(2),
            path=self.path.copy() if self.path is not None else None,
            setpoint=setpoint,
            control_force=control,
            distance_to_goal=dist_to_goal,
            path_progress=progress,
            is_complete=is_complete,
        )
    
    def get_path(self) -> Optional[np.ndarray]:
        """Get the current planned path."""
        return self.path.copy() if self.path is not None else None


def apply_control_to_platform(
    platform,
    control: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Convert control forces to platform accelerations.
    
    This is a helper to apply the navigator's control output
    to the DPDEM Platform state.
    
    Args:
        platform: DPDEM Platform NamedTuple
        control: Control forces [X, Y, N] in body frame
        dt: Time step
        
    Returns:
        force: 3D force vector in global frame [Fx, Fy, Fz]
    """
    # Get current heading from quaternion
    q = platform.quaternion
    psi = np.arctan2(
        2 * (q[0] * q[3] + q[1] * q[2]),
        1 - 2 * (q[2]**2 + q[3]**2)
    )
    
    # Rotation matrix (body to global, 2D)
    c, s = np.cos(psi), np.sin(psi)
    
    # Transform body forces to global frame
    Fx_global = c * control[0] - s * control[1]
    Fy_global = s * control[0] + c * control[1]
    
    # Return 3D force (no vertical component)
    return np.array([Fx_global, Fy_global, 0.0])


def get_control_moment(control: np.ndarray) -> np.ndarray:
    """
    Extract yaw moment from control for platform torque.
    
    Args:
        control: Control forces [X, Y, N]
        
    Returns:
        moment: 3D moment vector [Mx, My, Mz]
    """
    return np.array([0.0, 0.0, control[2]])
