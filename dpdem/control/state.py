"""
Ship state representation for control.

Follows notation from Fossen (2021):
- eta: Position and orientation [x, y, psi] in global/NED frame
- nu: Velocity [u, v, r] in body frame (surge, sway, yaw rate)
"""

import numpy as np
from typing import Tuple, List
from .gnc import Rxy


class ShipState:
    """
    Ship state for 3-DOF horizontal plane motion.
    
    Attributes:
        x: Global x position [m]
        y: Global y position [m]
        psi: Heading angle [rad] (positive counter-clockwise from x-axis)
        u: Surge velocity [m/s] (forward in body frame)
        v: Sway velocity [m/s] (lateral in body frame, positive = starboard)
        r: Yaw rate [rad/s] (positive counter-clockwise)
    """
    
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        psi: float = 0.0,
        u: float = 0.0,
        v: float = 0.0,
        r: float = 0.0,
        dt: float = 0.01,
    ):
        # Position (eta)
        self.x = x
        self.y = y
        self.psi = psi
        
        # Velocity (nu)
        self.u = u
        self.v = v
        self.r = r
        
        # Time step
        self.dt = dt
        
        # Control inputs (for logging)
        self.u_control = np.zeros(3)  # Control forces/moments
        self.u_actual = np.zeros(3)   # Actual actuator state
    
    @property
    def eta(self) -> np.ndarray:
        """Position vector [x, y, psi]."""
        return np.array([self.x, self.y, self.psi])
    
    @eta.setter
    def eta(self, value):
        self.x, self.y, self.psi = value
    
    @property
    def nu(self) -> np.ndarray:
        """Velocity vector [u, v, r] in body frame."""
        return np.array([self.u, self.v, self.r])
    
    @nu.setter
    def nu(self, value):
        self.u, self.v, self.r = value
    
    @property
    def position(self) -> np.ndarray:
        """2D position [x, y]."""
        return np.array([self.x, self.y])
    
    @property
    def speed(self) -> float:
        """Speed in horizontal plane: U = sqrt(u^2 + v^2)."""
        return np.sqrt(self.u**2 + self.v**2)
    
    def get_global_velocity(self) -> np.ndarray:
        """
        Get velocity in global frame.
        
        Returns:
            2D velocity [vx, vy] in global frame
        """
        return Rxy(self.psi) @ np.array([self.u, self.v])
    
    def integrate(self) -> np.ndarray:
        """
        Integrate position using current velocity (Euler method).
        
        Returns:
            Updated position [x, y, psi]
        """
        # Rotate body velocities to global frame
        vx, vy = self.get_global_velocity()
        
        # Update position
        self.x += self.dt * vx
        self.y += self.dt * vy
        self.psi = (self.psi + self.dt * self.r) % (2 * np.pi)
        
        return self.eta
    
    def copy(self) -> 'ShipState':
        """Create a copy of this state."""
        s = ShipState(
            x=self.x, y=self.y, psi=self.psi,
            u=self.u, v=self.v, r=self.r,
            dt=self.dt
        )
        s.u_control = self.u_control.copy()
        s.u_actual = self.u_actual.copy()
        return s
    
    @classmethod
    def from_platform(cls, platform, dt: float = 0.01) -> 'ShipState':
        """
        Create ShipState from DPDEM Platform object.
        
        Args:
            platform: DPDEM Platform NamedTuple
            dt: Time step
            
        Returns:
            ShipState instance
        """
        # Extract position (x, y from platform.position)
        x = float(platform.position[0])
        y = float(platform.position[1])
        
        # Extract heading from quaternion
        # For 2D motion, we just need yaw angle from quaternion
        q = platform.quaternion
        # Quaternion to yaw: psi = atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2^2 + q3^2))
        psi = np.arctan2(
            2 * (q[0] * q[3] + q[1] * q[2]),
            1 - 2 * (q[2]**2 + q[3]**2)
        )
        
        # Extract velocities
        # Platform velocity is in global frame, need to rotate to body frame
        v_global = np.array([float(platform.velocity[0]), float(platform.velocity[1])])
        R = Rxy(psi)
        v_body = R.T @ v_global  # Inverse rotation
        
        u = v_body[0]
        v = v_body[1]
        r = float(platform.angular_velocity[2])  # Yaw rate is about z-axis
        
        return cls(x=x, y=y, psi=psi, u=u, v=v, r=r, dt=dt)
    
    def __repr__(self) -> str:
        return (
            f"ShipState(x={self.x:.2f}, y={self.y:.2f}, psi={np.rad2deg(self.psi):.1f}°, "
            f"u={self.u:.2f} m/s, v={self.v:.2f} m/s, r={np.rad2deg(self.r):.1f}°/s)"
        )
