"""
Solver for the powered descent guidance optimization problem.

Author: Niko Natsoulas
"""

from typing import Tuple
import cvxpy as cp
import numpy as np
from .system_parameters import SystemParameters

class PoweredDescentGuidance:
    """Solver for the powered descent guidance optimization problem."""
    
    def __init__(self, params: SystemParameters):
        """Initialize the solver with system parameters."""
        self.params = params
        
    def _set_common_constraints(self, x: cp.Variable, z: cp.Variable, 
                              u: cp.Variable, gamma: cp.Variable) -> list:
        """Set up common optimization constraints for both problems."""
        constraints = []
        p = self.params
        
        # Boundary constraints
        constraints.extend([
            x[:,0] == p.x0,              # Initial state
            z[0,0] == p.zi,              # Initial mass
            z[0,p.N-1] >= p.zf,          # Final mass
            p.e1.T @ x[:3,p.N-1] == 0,   # Final altitude
            p.e1.T @ x[3:,p.N-1] == 0    # Final velocity
        ])
        
        # Velocity constraint
        constraints.append(cp.norm(x[3:6,p.N-1]) <= p.velocity_max)
        
        # Dynamic constraints
        for t in range(p.N-1):
            # Spacecraft dynamics
            constraints.append(
                x[:,t+1] == x[:,t] + 
                (p.A @ x[:,t] + p.B @ (p.gravity + u[:,t])) * p.dt
            )
            # Mass depletion dynamics
            constraints.append(
                z[:,t+1] == z[:,t] - p.alpha * gamma[:,t]
            )
            
            # Add thrust rate constraint
            if t < p.N-2:
                constraints.append(
                    cp.norm(u[:,t+1] - u[:,t], axis=0) <= p.max_thrust_change_rate  # Max thrust change rate
                )
        
        # Thrust constraints
        constraints.extend([
            cp.norm(u, axis=0) <= gamma[0,:],  # Upper bound
            p.e1 @ u >= gamma[0,:] * p.theta_cos  # Pointing constraint
        ])
        
        # Initial thrust should be small to ensure smooth start
        constraints.append(cp.norm(u[:,0]) <= p.initial_thrust_fraction * p.Tmax)
        
        # Slack variable bounds
        z0 = np.array([np.log(p.m0 - p.alpha * p.rho2 * p.dt * i) 
                      for i in range(p.N)])
        constraints.append(
            p.rho1 * np.exp(-p.zi) * 
            (1 - (z[0,:] - z0) + (z[0,:] - z0)**2/2) <= gamma[0,:]
        )
        
        # Glide slope constraint
        constraints.append(
            x[0,:] >= (cp.norm(x[1:3], axis=0) * p.gamma_tan)
        )
        
        return constraints

    def solve_minimum_error(self) -> Tuple[str, cp.Variable, cp.Variable, cp.Variable, cp.Variable]:
        """Solve the relaxed minimum-landing-error guidance problem."""
        p = self.params
        
        # Variables
        x = cp.Variable((6, p.N))     # State [pos(3), vel(3)]
        z = cp.Variable((1, p.N))     # ln(mass)
        u = cp.Variable((3, p.N))     # Thrust
        gamma = cp.Variable((1, p.N)) # Slack variable
        
        # Objective: minimize landing error, final velocity, and thrust variation
        objective = cp.Minimize(
            5.0 * cp.norm(p.E @ x[:3,p.N-1] - p.q) +  # Landing error
            0.5 * cp.sum(gamma) +                      # Fuel usage (reduced weight)
            1.0 * cp.norm(x[3:,p.N-1]) +              # Final velocity
            0.1 * cp.sum_squares(cp.diff(gamma[0,:]))  # Thrust smoothness
        )
        
        # Constraints
        constraints = self._set_common_constraints(x, z, u, gamma)
        constraints.extend([
            cp.norm(x[3:,p.N-1]) <= p.final_velocity_max,  # Final velocity limit
            cp.norm(p.E @ x[:3,p.N-1] - p.q) <= 10.0  # Landing accuracy
        ])
        
        # Final approach constraints
        for t in range(p.N-10, p.N):
            constraints.append(
                cp.norm(p.E @ x[:3,t] - p.q) <= 50.0
            )
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=True)
        
        return problem.status, x, u, gamma, z

    def solve_minimum_fuel(self, dP3: float) -> Tuple[str, cp.Variable, 
                                                     cp.Variable, cp.Variable, 
                                                     cp.Variable]:
        """Solve the relaxed minimum-fuel guidance problem."""
        p = self.params
        
        # Variables
        x = cp.Variable((6, p.N))     # State
        z = cp.Variable((1, p.N))     # ln(mass)
        u = cp.Variable((3, p.N))     # Thrust
        gamma = cp.Variable((1, p.N)) # Slack variable
        
        # Objective: minimize fuel consumption while penalizing final velocity
        objective = cp.Minimize(
            cp.sum(gamma) * p.dt +                     # Fuel consumption
            0.05 * cp.sum_squares(cp.diff(gamma[0,:])) + # Thrust smoothness
            1.0 * cp.norm(x[3:,p.N-1])                  # Penalize final speed
        )
        
        # Constraints
        constraints = self._set_common_constraints(x, z, u, gamma)
        constraints.extend([
            cp.norm(p.E @ x[:3,p.N-1] - p.q) <= dP3,
            cp.norm(x[3:,p.N-1]) <= p.final_velocity_max  # Limit final speed
        ])
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        
        return problem.status, x, u, gamma, z 