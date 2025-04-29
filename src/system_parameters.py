"""
System parameters for the powered descent guidance problem.
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class SystemParameters:
    """System parameters for the powered descent guidance problem."""
    # Initial state vector [r, ṙ]
    x0: np.ndarray = field(default_factory=lambda: np.array([2400, 450, -330, -10, -40, 10]))
    
    # Landing target coordinates (2D, alt=0)
    q: np.ndarray = field(default_factory=lambda: np.array([0, 0]))  # Only Y,Z coordinates since X (altitude) is constrained to 0
    
    # Mass parameters
    m0: float = 2000.0  # Initial mass [kg]
    mf: float = 300.0   # Final mass [kg]
    alpha: float = 5e-4  # Fuel consumption rate [s/m]
    
    # Thrust parameters
    Tmax: float = 24000.0  # Maximum thrust [N]
    rho1: float = 0.2 * 24000.0  # Lower bound thrust [N]
    rho2: float = 0.8 * 24000.0  # Upper bound thrust [N]
    
    # Simulation parameters
    tf: float = 50.0  # End time [s]
    dt: float = 1.0   # Time interval [s]
    
    # Constraint parameters
    glidelslope_angle: float = 30.0  # Minimum glide-slope angle [deg]
    theta_deg: float = 120.0  # Thrust angle [deg]
    velocity_max: float = 90.0  # Maximum velocity [m/s]
    
    # Planet parameters (Mars)
    omega: np.ndarray = field(default_factory=lambda: np.array([2.53e-5, 0, 6.62e-5]))  # Angular velocity [rad/s]
    gravity: np.ndarray = field(default_factory=lambda: np.array([-3.71, 0, 0]))  # Gravity vector [m/s²]
    
    def __post_init__(self):
        """Compute derived parameters after initialization."""
        self.N = int(self.tf / self.dt)
        self.zi = np.log(self.m0)
        self.zf = np.log(self.m0 - self.mf)
        self.gamma_tan = np.tan(np.deg2rad(self.glidelslope_angle))
        self.theta_cos = np.cos(np.deg2rad(self.theta_deg))
        
        # Unit vectors and matrices
        self.e1 = np.array([1, 0, 0])
        self.e2 = np.array([0, 1, 0])
        self.e3 = np.array([0, 0, 1])
        self.E = np.array([self.e2.T, self.e3.T])  # Projects 3D vector to YZ plane
        
        # System matrices
        S = np.array([
            [0, -self.omega[2], self.omega[1]],
            [self.omega[2], 0, -self.omega[0]],
            [-self.omega[1], self.omega[0], 0]
        ])
        self.A = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [-S**2, -2*S]
        ])
        self.B = np.block([
            [np.zeros((3, 3))],
            [np.eye(3)]
        ]) 