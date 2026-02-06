"""
Dynamic Positioning (DP) Controller.

Implements MIMO nonlinear PID control with pole placement for
trajectory tracking in the horizontal plane.

Reference: T. I. Fossen (2021). Handbook of Marine Craft Hydrodynamics and
Motion Control. 2nd. Edition, Wiley.
"""

import numpy as np
from typing import Tuple, Optional
from .gnc import Rzyx, ssa


class DPController:
    """
    Dynamic Positioning controller using PID pole placement.
    
    Controls surge (X), sway (Y), and yaw (N) forces/moments
    to track a desired position and heading setpoint.
    
    Attributes:
        wn: Natural frequency diagonal matrix (3x3)
        zeta: Damping ratio diagonal matrix (3x3)
        e_int: Integral error state
    """
    
    def __init__(
        self,
        M: np.ndarray,
        D: np.ndarray,
        wn: Optional[np.ndarray] = None,
        zeta: Optional[np.ndarray] = None,
        use_integral: bool = False,
    ):
        """
        Initialize DP controller.
        
        Args:
            M: 3x3 mass/inertia matrix (surge, sway, yaw)
            D: 3x3 damping matrix
            wn: Natural frequencies [rad/s] for x, y, psi. Default: [0.1, 0.1, 0.1]
            zeta: Damping ratios. Default: [1.0, 1.0, 1.0] (critically damped)
            use_integral: Whether to include integral action
        """
        self.M = np.diag(np.diag(M))  # Use diagonal elements
        self.D = np.diag(np.diag(D))
        
        # Default tuning parameters
        if wn is None:
            wn = np.array([0.1, 0.1, 0.1])
        if zeta is None:
            zeta = np.array([1.0, 1.0, 1.0])
        
        self.wn = np.diag(wn)
        self.zeta = np.diag(zeta)
        
        # Compute PID gains using pole placement
        # Kp = wn^2 * M
        # Kd = 2 * zeta * wn * M - D
        # Ki = (1/10) * wn * Kp
        self.Kp = self.wn @ self.wn @ self.M
        self.Kd = 2.0 * self.zeta @ self.wn @ self.M - self.D
        self.Ki = (1.0 / 10.0) * self.wn @ self.Kp
        
        self.use_integral = use_integral
        self.e_int = np.zeros(3)  # Integral error state
        
    def reset(self):
        """Reset integral error."""
        self.e_int = np.zeros(3)
    
    def compute_control(
        self,
        eta: np.ndarray,
        nu: np.ndarray,
        setpoint: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute control forces/moments.
        
        Args:
            eta: Current position [x, y, psi]
            nu: Current velocity [u, v, r] in body frame
            setpoint: Desired position [x_d, y_d, psi_d]
            dt: Time step [s]
            
        Returns:
            tau: Control forces/moments [X, Y, N] in body frame
        """
        x, y, psi = eta
        x_d, y_d, psi_d = setpoint
        
        # Position error in global frame
        e = np.array([x - x_d, y - y_d, ssa(psi - psi_d)])
        
        # Rotation matrix (global to body)
        R = Rzyx(0.0, 0.0, psi)
        
        # PD control law
        # tau = -R^T @ Kp @ e - Kd @ nu
        tau = -R.T @ self.Kp @ e - self.Kd @ nu
        
        # Add integral action if enabled
        if self.use_integral:
            tau -= R.T @ self.Ki @ self.e_int
            # Update integral (Euler integration)
            self.e_int += dt * e
            # Anti-windup: limit integral
            self.e_int = np.clip(self.e_int, -10.0, 10.0)
        
        return tau
    
    @classmethod
    def from_ship_params(
        cls,
        mass: float,
        length: float,
        breadth: float,
        wn: Optional[np.ndarray] = None,
        zeta: Optional[np.ndarray] = None,
    ) -> 'DPController':
        """
        Create controller from basic ship parameters.
        
        Estimates mass and damping matrices from ship dimensions.
        
        Args:
            mass: Ship mass [kg]
            length: Ship length [m]
            breadth: Ship breadth [m]
            wn: Natural frequencies
            zeta: Damping ratios
            
        Returns:
            DPController instance
        """
        # Estimate inertia about z-axis (yaw)
        # Approximation for box: I_zz = (1/12) * m * (L^2 + B^2)
        I_zz = (1.0 / 12.0) * mass * (length**2 + breadth**2)
        
        # Mass matrix (surge, sway, yaw)
        # Include added mass (rough approximation)
        m_surge = mass * 1.1  # ~10% added mass in surge
        m_sway = mass * 1.8   # ~80% added mass in sway
        I_yaw = I_zz * 1.5    # Added inertia
        
        M = np.diag([m_surge, m_sway, I_yaw])
        
        # Damping matrix (linear approximation)
        # These are rough estimates - should be tuned
        D_surge = 0.1 * mass  # Linear drag
        D_sway = 0.2 * mass
        D_yaw = 0.1 * I_zz
        
        D = np.diag([D_surge, D_sway, D_yaw])
        
        return cls(M=M, D=D, wn=wn, zeta=zeta)


def dp_pole_placement(
    e_int: np.ndarray,
    M3: np.ndarray,
    D3: np.ndarray,
    eta3: np.ndarray,
    nu3: np.ndarray,
    x_d: float,
    y_d: float,
    psi_d: float,
    wn: np.ndarray,
    zeta: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MIMO nonlinear PID pole placement (functional interface).
    
    Implements the control law from Fossen (2021).
    
    Args:
        e_int: Integral error state (3,)
        M3: 3x3 mass matrix
        D3: 3x3 damping matrix
        eta3: Current position [x, y, psi]
        nu3: Current velocity [u, v, r]
        x_d, y_d, psi_d: Desired position
        wn: 3x3 diagonal natural frequency matrix
        zeta: 3x3 diagonal damping ratio matrix
        dt: Time step
        
    Returns:
        tau: Control forces [X, Y, N]
        e_int: Updated integral error
    """
    # Use diagonal elements
    M3_diag = np.diag(np.diag(M3))
    D3_diag = np.diag(np.diag(D3))
    
    # PID gains from pole placement
    Kp = wn @ wn @ M3_diag
    Kd = 2.0 * zeta @ wn @ M3_diag - D3_diag
    Ki = (1.0 / 10.0) * wn @ Kp
    
    # Position error
    e = eta3 - np.array([x_d, y_d, psi_d])
    e[2] = ssa(e[2])  # Wrap heading error
    
    # Rotation matrix
    R = Rzyx(0.0, 0.0, eta3[2])
    
    # Control law
    tau = -R.T @ Kp @ e - Kd @ nu3 - R.T @ Ki @ e_int
    
    # Update integral
    e_int = e_int + dt * e
    
    return tau, e_int
