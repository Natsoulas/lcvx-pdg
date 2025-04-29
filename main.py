"""
Main script for the powered descent guidance problem.
Sets up the problem parameters and solves both minimum landing error
and minimum fuel problems.

Author: Niko Natsoulas
"""

import numpy as np
import cvxpy as cp
from src.system_parameters import SystemParameters
from src.solver import PoweredDescentGuidance
from src.plotting import make_all_plots

def main():
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

    # Solve minimum landing error problem
    print("\nSolving minimum landing error problem...")
    status, x, u = solver.solve_minimum_error()
    
    if status == cp.OPTIMAL or status == "optimal_inaccurate":
        print("Minimum landing error problem solved successfully!")
        print(f"Status: {status}")
        # Generate plots
        make_all_plots(x.value, u.value, None, None, params)
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
        make_all_plots(x.value, u.value, sigma.value, z.value, params)
    else:
        print(f"Minimum fuel problem failed with status: {status}")

if __name__ == "__main__":
    main() 