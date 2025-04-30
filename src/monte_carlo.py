"""
Monte Carlo simulation for the powered descent guidance problem.

This module provides functionality to run multiple simulations with varied
initial conditions and system parameters to analyze robustness and performance.

Author: Niko Natsoulas
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from .solver import PoweredDescentGuidance
from .system_parameters import SystemParameters

@dataclass
class MonteCarloResults:
    """Container for Monte Carlo simulation results."""
    landing_errors: np.ndarray  # Final position errors [m]
    final_velocities: np.ndarray  # Final velocity magnitudes [m/s]
    fuel_consumption: np.ndarray  # Fuel consumed [kg]
    success_rate: float  # Percentage of successful landings
    execution_times: np.ndarray  # Solver execution times [s]
    parameter_variations: Dict[str, np.ndarray]  # Record of varied parameters
    trajectories: np.ndarray  # Store all trajectories for visualization

class MonteCarloPDG:
    """Monte Carlo simulation for the Powered Descent Guidance problem."""
    
    def __init__(self, base_params: SystemParameters, num_simulations: int = 100):
        """Initialize Monte Carlo simulation.
        
        Args:
            base_params: Base system parameters to vary around
            num_simulations: Number of Monte Carlo runs
        """
        self.base_params = base_params
        self.num_simulations = num_simulations
        
    def _vary_parameters(self) -> SystemParameters:
        """Create a new parameter set with random variations.
        
        Returns:
            Modified SystemParameters object
        """
        params = SystemParameters()
        
        # Vary initial state with normal distribution
        pos_std = np.array([50.0, 20.0, 20.0])  # Position variation [m]
        vel_std = np.array([2.0, 2.0, 2.0])     # Velocity variation [m/s]
        
        pos_variation = np.random.normal(0, 1, 3) * pos_std
        vel_variation = np.random.normal(0, 1, 3) * vel_std
        
        # Ensure x0 is float64 before adding variations
        params.x0 = self.base_params.x0.astype(np.float64)
        params.x0[:3] += pos_variation
        params.x0[3:] += vel_variation
        
        # Vary mass parameters
        params.m0 = self.base_params.m0 * (1 + np.random.normal(0, 0.05))  # ±5% variation
        params.mf = self.base_params.mf * (1 + np.random.normal(0, 0.05))
        
        # Vary thrust parameters
        params.Tmax = self.base_params.Tmax * (1 + np.random.normal(0, 0.03))  # ±3% variation
        
        # Keep other parameters same as base
        params.tf = self.base_params.tf
        params.dt = self.base_params.dt
        params.alpha = self.base_params.alpha
        params.glidelslope_angle = self.base_params.glidelslope_angle
        params.theta_deg = self.base_params.theta_deg
        params.velocity_max = self.base_params.velocity_max
        
        # Recompute derived parameters
        params.__post_init__()
        
        return params
    
    def run_simulations(self) -> MonteCarloResults:
        """Run Monte Carlo simulations.
        
        Returns:
            MonteCarloResults object containing simulation statistics
        """
        # Initialize result arrays
        landing_errors = np.zeros(self.num_simulations)
        final_velocities = np.zeros(self.num_simulations)
        fuel_consumption = np.zeros(self.num_simulations)
        execution_times = np.zeros(self.num_simulations)
        successes = 0
        
        # Store all trajectories
        trajectories = np.zeros((self.num_simulations, 3, self.base_params.N))
        
        # Store parameter variations
        param_variations = {
            'initial_position': np.zeros((self.num_simulations, 3)),
            'initial_velocity': np.zeros((self.num_simulations, 3)),
            'initial_mass': np.zeros(self.num_simulations),
            'max_thrust': np.zeros(self.num_simulations)
        }
        
        for i in range(self.num_simulations):
            # Generate varied parameters
            params = self._vary_parameters()
            
            # Record parameter variations
            param_variations['initial_position'][i] = params.x0[:3] - self.base_params.x0[:3]
            param_variations['initial_velocity'][i] = params.x0[3:] - self.base_params.x0[3:]
            param_variations['initial_mass'][i] = params.m0
            param_variations['max_thrust'][i] = params.Tmax
            
            # Create and run solver
            solver = PoweredDescentGuidance(params)
            
            try:
                # Solve minimum fuel problem with reasonable landing error tolerance
                status, x, u, gamma, z = solver.solve_minimum_fuel(dP3=5.0)
                
                if status == "optimal":
                    successes += 1
                    
                    # Get numpy arrays from CVXPY variables
                    x_val = x.value
                    z_val = z.value
                    
                    # Store trajectory
                    trajectories[i] = x_val[:3]
                    
                    # Calculate metrics using numpy arrays
                    landing_errors[i] = float(np.linalg.norm(params.E @ x_val[:3,-1] - params.q))
                    final_velocities[i] = float(np.linalg.norm(x_val[3:,-1]))
                    fuel_consumption[i] = float(params.m0 - np.exp(z_val[0,-1]))
                
            except Exception as e:
                print(f"Simulation {i} failed: {str(e)}")
                continue
        
        return MonteCarloResults(
            landing_errors=landing_errors,
            final_velocities=final_velocities,
            fuel_consumption=fuel_consumption,
            success_rate=successes / self.num_simulations * 100,
            execution_times=execution_times,
            parameter_variations=param_variations,
            trajectories=trajectories
        )
    
    def plot_results(self, results: MonteCarloResults, save_dir: str):
        """Plot Monte Carlo simulation results and save to directory.
        
        Args:
            results: MonteCarloResults object to visualize
            save_dir: Directory to save plots
        """
        # Create distribution plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Landing error histogram
        ax1.hist(results.landing_errors, bins=20)
        ax1.set_title('Landing Error Distribution')
        ax1.set_xlabel('Landing Error [m]')
        ax1.set_ylabel('Count')
        
        # Final velocity histogram
        ax2.hist(results.final_velocities, bins=20)
        ax2.set_title('Final Velocity Distribution')
        ax2.set_xlabel('Final Velocity [m/s]')
        ax2.set_ylabel('Count')
        
        # Fuel consumption histogram
        ax3.hist(results.fuel_consumption, bins=20)
        ax3.set_title('Fuel Consumption Distribution')
        ax3.set_xlabel('Fuel Consumed [kg]')
        ax3.set_ylabel('Count')
        
        # Initial position variations
        scatter = ax4.scatter(
            results.parameter_variations['initial_position'][:, 1],
            results.parameter_variations['initial_position'][:, 2],
            c=results.landing_errors,
            cmap='viridis'
        )
        ax4.set_title('Landing Error vs Initial Position Variation')
        ax4.set_xlabel('Y Position Variation [m]')
        ax4.set_ylabel('Z Position Variation [m]')
        plt.colorbar(scatter, ax=ax4, label='Landing Error [m]')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'monte_carlo_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Create 3D trajectory plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all trajectories with alpha for transparency
        for i in range(len(results.trajectories)):
            traj = results.trajectories[i]
            # Plot with altitude on z-axis, x=cross-range, y=down-range
            ax.plot(traj[1], traj[2], traj[0], 'b-', alpha=0.1)
        
        # Plot target point at ground level
        ax.scatter([0], [0], [0], color='red', s=100, label='Target')
        
        # Set labels for proper orientation
        ax.set_xlabel('Cross-range [m]')
        ax.set_ylabel('Down-range [m]')
        ax.set_zlabel('Altitude [m]')
        ax.set_title(f'Monte Carlo Trajectories (n={self.num_simulations})')
        
        # Add grid for better depth perception
        ax.grid(True)
        
        # Adjust view angle for better visualization of descent
        ax.view_init(elev=20, azim=-45)
        
        # Add legend
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        plt.savefig(os.path.join(save_dir, 'monte_carlo_trajectories_3d.png'), dpi=300, bbox_inches='tight')
        plt.close() 