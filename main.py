"""
Main script for the powered descent guidance problem.
Sets up the problem parameters and solves both minimum landing error
and minimum fuel problems, followed by Monte Carlo analysis.

Author: Niko Natsoulas
"""

import numpy as np
import cvxpy as cp
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
from src.system_parameters import SystemParameters
from src.solver import PoweredDescentGuidance
from src.plotting import make_all_plots, save_animation_frames, create_gif
from src.monte_carlo import MonteCarloPDG

def run_monte_carlo_analysis(params: SystemParameters, save_dir: Path, n_sims: int = 100):
    """Run Monte Carlo analysis and save results."""
    print("\n=== Running Monte Carlo Analysis ===")
    
    # Create Monte Carlo simulator
    mc_sim = MonteCarloPDG(params, num_simulations=n_sims)
    
    # Run simulations
    print(f"Running {n_sims} Monte Carlo simulations...")
    start_time = time.time()
    results = mc_sim.run_simulations()
    end_time = time.time()
    
    # Print summary statistics
    print(f"\nMonte Carlo Results:")
    print(f"Success Rate: {results.success_rate:.1f}%")
    print(f"Average Landing Error: {np.mean(results.landing_errors):.2f} ± {np.std(results.landing_errors):.2f} m")
    print(f"Average Final Velocity: {np.mean(results.final_velocities):.2f} ± {np.std(results.final_velocities):.2f} m/s")
    print(f"Average Fuel Consumption: {np.mean(results.fuel_consumption):.2f} ± {np.std(results.fuel_consumption):.2f} kg")
    print(f"Total Simulation Time: {end_time - start_time:.1f} seconds")
    
    # Plot results
    mc_sim.plot_results(results, str(save_dir))

def main():
    # Create results directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"results_{timestamp}")
    save_dir.mkdir(exist_ok=True)

    # Define system parameters
    params = SystemParameters(
        x0=np.array([2400, 450, -330, -10, -40, 10]),  # Initial state [x,y,z, vx,vy,vz]
        q=np.array([0, 0]),  # Landing target (y,z coordinates)
        m0=2000.0,  # Initial mass [kg]
        mf=300.0,   # Final mass [kg]
        alpha=5e-4,  # Mass flow rate parameter [s/m]
        Tmax=24000.0,  # Maximum thrust [N]
        rho1=0.2 * 24000.0,  # Lower bound thrust [N]
        rho2=0.8 * 24000.0,  # Upper bound thrust [N]
        tf=50.0,  # Final time [s]
        dt=1.0,  # Time step [s]
        glidelslope_angle=30.0,  # Glide slope angle [deg]
        theta_deg=120.0,  # Thrust pointing angle [deg]
        velocity_max=90.0  # Maximum velocity [m/s]
    )

    # Create solver instance
    solver = PoweredDescentGuidance(params)

    print("\n=== Running Baseline Analysis ===")
    
    # Solve minimum landing error problem
    print("\nSolving minimum landing error problem...")
    status, x, u = solver.solve_minimum_error()
    
    if status == cp.OPTIMAL or status == "optimal_inaccurate":
        print("Minimum landing error problem solved successfully!")
        print(f"Status: {status}")
        # Generate plots
        figs = make_all_plots(x.value, u.value, None, None, params)
        # Save each figure
        for name, fig in figs.items():
            fig.savefig(save_dir / f'min_error_{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        # Create animation
        print("\nGenerating minimum error trajectory animation...")
        save_animation_frames(x.value, u.value, params, save_dir / 'min_error_animation')
        create_gif(save_dir / 'min_error_animation', save_dir / 'min_error_trajectory.gif')
    else:
        print(f"Minimum landing error problem failed with status: {status}")
        return

    # Calculate maximum allowable landing error for minimum fuel problem
    # This is the YZ distance from the target at the final time
    dP3 = np.linalg.norm(params.E @ x.value[:3,-1])
    print(f"\nMaximum allowable landing error: {dP3:.2f} m")

    # Solve minimum fuel problem
    print("\nSolving minimum fuel problem...")
    status, x, u, sigma, z = solver.solve_minimum_fuel(dP3)
    
    if status == cp.OPTIMAL or status == "optimal_inaccurate":
        print("Minimum fuel problem solved successfully!")
        print(f"Status: {status}")
        # Generate plots
        figs = make_all_plots(x.value, u.value, sigma.value, z.value, params)
        # Save each figure
        for name, fig in figs.items():
            fig.savefig(save_dir / f'min_fuel_{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        # Create animation
        print("\nGenerating minimum fuel trajectory animation...")
        save_animation_frames(x.value, u.value, params, save_dir / 'min_fuel_animation')
        create_gif(save_dir / 'min_fuel_animation', save_dir / 'min_fuel_trajectory.gif')
    else:
        print(f"Minimum fuel problem failed with status: {status}")
        return

    # Run Monte Carlo analysis
    run_monte_carlo_analysis(params, save_dir, n_sims=500)
    
    print(f"\nAll results saved to: {save_dir}")

if __name__ == "__main__":
    main() 