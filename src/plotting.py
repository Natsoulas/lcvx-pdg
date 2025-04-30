"""
Plotting functions for the powered descent guidance problem.

Author: Niko Natsoulas
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from .system_parameters import SystemParameters

def plot_trajectory3d(r, u):
    """3-D path coloured by thrust magnitude.
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    thr = np.linalg.norm(u.T, axis=1)
    norm = colors.Normalize(vmin=thr.min(), vmax=thr.max())
    cmap = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    
    # Reorient so altitude (x[0]) is the Z axis
    x, y, z = r[1], r[2], r[0]  # Rearrange coordinates
    
    for k in range(len(x)-1):
        ax.plot([x[k], x[k+1]], 
                [y[k], y[k+1]], 
                [z[k], z[k+1]], 
                color=cmap(norm(thr[k])), linewidth=2)
                
    ax.set(xlabel='Y [m]', ylabel='Z [m]', zlabel='Altitude [m]')
    ax.set_title('Powered-descent trajectory')
    fig.colorbar(cm.ScalarMappable(norm,cmap), ax=ax,
                 label='|Thrust| [m/s$^{2}$]')
    
    # Set a better viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig

def plot_time_histories(t, u, sigma, v, z, params: SystemParameters):
    """4-panel summary plot.
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    thrust = np.linalg.norm(u, axis=0)
    pointing = np.arccos(u[0]/thrust) * 180/np.pi   # angle wrt vertical

    fig, axs = plt.subplots(2,2, figsize=(6.3,5.6))
    
    # Plot thrust magnitude
    axs[0,0].plot(t, thrust, label='‖u‖')
    if sigma is not None:
        # Flatten sigma if it's 2D
        sigma_plot = sigma.flatten() if sigma.ndim > 1 else sigma
        axs[0,0].plot(t, sigma_plot, '--', label='σ (slack)')
    axs[0,0].set_ylabel('[m/s²]')
    axs[0,0].set_title('Thrust magnitude')
    axs[0,0].legend()

    # Plot pointing angle
    axs[0,1].plot(t, pointing)
    axs[0,1].axhline(params.theta_deg, color='k', ls='--')
    axs[0,1].set_title('Pointing angle')
    axs[0,1].set_ylabel('[deg]')

    # Plot speed
    axs[1,0].plot(t, np.linalg.norm(v, axis=0))
    axs[1,0].axhline(params.velocity_max, color='k', ls='--')
    axs[1,0].set_ylabel('[m/s]')
    axs[1,0].set_title('Speed')

    # Plot mass profile
    if z is not None:
        # Flatten z if it's 2D
        z_plot = z.flatten() if z.ndim > 1 else z
        axs[1,1].plot(t, np.exp(z_plot))
        axs[1,1].set_title('Mass profile')
        axs[1,1].set_ylabel('[kg]')
    else:
        axs[1,1].set_title('Mass profile (not available)')
        axs[1,1].set_ylabel('[kg]')

    for ax in axs.flat:
        ax.set_xlabel('Time [s]')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_groundtrack(r, params: SystemParameters):
    """XZ ground-track with glide-slope cone.
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    x, y, z = r
    fig, ax = plt.subplots(figsize=(5.2,3.5))
    ax.plot(y, z, lw=2)
    
    # Calculate cone boundaries based on data range
    cone = np.linspace(0, params.x0[0], 100)
    cone_width = cone/params.gamma_tan
    
    # Plot glide-slope cone
    ax.fill_between(cone*0, -cone_width, cone_width,
                    color='orange', alpha=0.15, label='Glide-slope cone')
    
    # Set axis limits based on data
    z_max = max(abs(z.max()), abs(z.min()))
    z_padding = z_max * 0.2  # Add 20% padding
    ax.set_ylim(-z_max - z_padding, z_max + z_padding)
    
    ax.set(xlabel='Downrange Y [m]', ylabel='Cross-range Z [m]',
           title='Ground-track (YZ-plane)')
    ax.legend()
    plt.tight_layout()
    return fig

def fancy_trajectory_plot(r, v, params: SystemParameters,
                         tick_dt=2.0,  # seconds between attitude arrows
                         cone_length=500,  # m
                         out='figs/traj_fancy.png'):
    """
    r : (3,N) position [m]; v : (3,N) velocity [m/s]
    gamma_tan : tan(gamma_gs) glide-slope
    theta_deg : pointing-cone half angle
    """
    # -------------------------------- colours -------------------------------
    speed = np.linalg.norm(v, axis=0)
    cmap = plt.get_cmap('turbo')
    norm = colors.Normalize(speed.min(), speed.max())
    rgb = cmap(norm(speed))

    # -------------------------------- figure --------------------------------
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 16))
    
    # Create a special gridspec to accommodate colorbar
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05])
    
    # Define the views: (elev, azim) pairs and their positions in the gridspec
    views = [
        (20, 45, '3D View', gs[0, 0]),      # Isometric
        (0, 0, 'Front View', gs[0, 1]),     # Front view (YZ plane)
        (0, 90, 'Side View', gs[1, 0]),     # Side view (XZ plane)
        (90, 0, 'Top View', gs[1, 1]),      # Top view (XY plane)
    ]
    
    axes = []
    for elev, azim, title, pos in views:
        ax = fig.add_subplot(pos, projection='3d', computed_zorder=False)
        axes.append(ax)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type('persp')
        
        # Set background color and edge color
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        
        # Make grid lines lighter
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Ground plane
        max_range = max(r[0].max(), r[1].max(), abs(r[1].min()), r[2].max(), abs(r[2].min())) * 1.2
        xx, yy = np.meshgrid(np.linspace(-max_range/2, max_range/2, 10),
                            np.linspace(-max_range/2, max_range/2, 10))
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray', zorder=0)

        # -------------------------------- glide-slope cone ----------------------
        a = np.linspace(0, 2*np.pi, 60)
        h = np.linspace(0, r[0].max(), 30)
        A,H = np.meshgrid(a,h)
        
        # Calculate cone coordinates
        R = H / params.gamma_tan
        Y = R*np.cos(A)  # Y coordinate
        Z = R*np.sin(A)  # Z coordinate
        X = H            # Altitude
        
        # Plot glide-slope cone with solid orange surface
        ax.plot_surface(Y, Z, X, color='orange', alpha=0.08, rcount=15, ccount=30, linewidth=0, zorder=3)
        
        # Add bold radial lines for glide-slope cone
        for i in range(8):
            angle = i * np.pi/4
            y_line = np.cos(angle) * R[:,0]
            z_line = np.sin(angle) * R[:,0]
            ax.plot(y_line, z_line, h, color='orange', alpha=0.8, linestyle='-', lw=2, zorder=4)
        
        # Add height rings for glide-slope cone
        ring_heights = np.linspace(0, h.max(), 4)
        for height in ring_heights:
            idx = np.abs(h - height).argmin()
            circle_y = R[idx] * np.cos(a)
            circle_z = R[idx] * np.sin(a)
            circle_x = np.full_like(a, h[idx])
            ax.plot(circle_y, circle_z, circle_x, color='orange', alpha=0.8, linestyle='-', lw=2, zorder=4)

        # -------------------------------- pointing cone -------------------------
        # Plot pointing cone at selected points along trajectory
        num_cones = 5  # Number of pointing cones to show
        cone_indices = np.linspace(0, r.shape[1]-1, num_cones).astype(int)
        
        for idx in cone_indices:
            # Create a small pointing cone at this trajectory point
            cone_height = 50.0  # Height of each pointing cone
            h_local = np.linspace(0, cone_height, 10)
            A_local, H_local = np.meshgrid(a, h_local)
            
            # Calculate cone radius based on height and angle
            cone_ang = np.deg2rad(params.theta_deg)
            R_local = np.tan(cone_ang) * H_local
            
            # Generate cone coordinates centered at the trajectory point
            Y_local = r[1,idx] + R_local * np.cos(A_local)
            Z_local = r[2,idx] + R_local * np.sin(A_local)
            X_local = r[0,idx] + H_local.reshape(-1,1)
            
            # Plot radial lines for pointing cone
            for i in range(8):
                angle = i * np.pi/4
                y_line = r[1,idx] + np.cos(angle) * (h_local * np.tan(cone_ang))
                z_line = r[2,idx] + np.sin(angle) * (h_local * np.tan(cone_ang))
                x_line = r[0,idx] + h_local
                ax.plot(y_line, z_line, x_line, color='blue', alpha=0.6, linestyle='--', lw=1.5, zorder=4)
            
            # Plot rings for pointing cone
            ring_heights_local = np.linspace(0, cone_height, 3)
            for h_ring in ring_heights_local:
                radius = h_ring * np.tan(cone_ang)
                circle_y = r[1,idx] + radius * np.cos(a)
                circle_z = r[2,idx] + radius * np.sin(a)
                circle_x = r[0,idx] + h_ring
                ax.plot(circle_y, circle_z, circle_x, color='blue', alpha=0.6, linestyle=':', lw=1.5, zorder=4)

        # Add target point marker
        ax.scatter([0], [0], [0], color='red', s=100, marker='x', linewidth=2, label='Target', zorder=12)

        # Main trajectory with enhanced visibility
        for k in range(r.shape[1]-1):
            # Main line
            ax.plot([r[1,k], r[1,k+1]], 
                   [r[2,k], r[2,k+1]], 
                   [r[0,k], r[0,k+1]], color=rgb[k], lw=4, zorder=10)
            # White halo effect
            ax.plot([r[1,k], r[1,k+1]], 
                   [r[2,k], r[2,k+1]], 
                   [r[0,k], r[0,k+1]], color='white', lw=6, alpha=0.3, zorder=9)

        # Velocity arrows (only in 3D and side views)
        if elev != 90 and title in ['3D View', 'Side View']:
            t = np.linspace(0, (r.shape[1]-1)*params.dt, r.shape[1])
            sel = np.abs((t/tick_dt)%1) < 1e-3
            ql = 100  # Increased arrow length
            ax.quiver(r[1,sel], r[2,sel], r[0,sel],
                     v[1,sel], v[2,sel], v[0,sel],
                     length=ql, normalize=True, color='k', arrow_length_ratio=0.2,
                     alpha=0.7, zorder=11)

        # Add text labels with arrows
        if title == '3D View' or title == 'Side View':
            # Glide-slope label
            label_height = r[0].max() * 0.7
            label_radius = (label_height / params.gamma_tan) * 0.8
            ax.text(label_radius*1.2, 0, label_height, 
                    'Glide-slope\nconstraint\n(30°)', 
                    color='orange', fontsize=10, ha='left', va='center', weight='bold')
            
            # Pointing cone label (now near one of the pointing cones)
            mid_idx = len(cone_indices) // 2
            label_pos = cone_indices[mid_idx]
            ax.text(r[1,label_pos] + 100, r[2,label_pos], r[0,label_pos], 
                    'Thrust pointing\nconstraint\n(120°)', 
                    color='blue', fontsize=10, ha='left', va='center', weight='bold')

        # Axes settings
        ax.set_xlabel('Y [m]', labelpad=10)
        ax.set_ylabel('Z [m]', labelpad=10)
        ax.set_zlabel('Altitude [m]', labelpad=10)
        ax.set_title(title, pad=20, fontsize=12, weight='bold')
        
        # Set axis limits
        ax.set_xlim(-max_range/2, max_range/2)
        ax.set_ylim(-max_range/2, max_range/2)
        ax.set_zlim(0, r[0].max()*1.1)
        
        ax.tick_params(axis='both', which='major', labelsize=9)
        if elev != 90:  # Keep normal aspect for all but top view
            ax.set_box_aspect((1.5, 1.5, 1))

    # Colorbar - now in its own subplot
    cax = fig.add_subplot(gs[:, -1])  # Span both rows
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Speed [m/s]', fontsize=11, labelpad=10, weight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Legend - only on first subplot
    axes[0].plot([], [], [], color='k', lw=3, label='Trajectory')
    axes[0].plot([], [], [], color='orange', lw=2, alpha=0.8, label='Glide-slope cone')
    axes[0].plot([], [], [], color='blue', lw=2, alpha=0.6, linestyle='--', label='Pointing cone')
    axes[0].legend(fontsize=10, loc='upper right')

    plt.tight_layout()
    Path('figs').mkdir(exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved {out}")
    return fig

def make_all_plots(x, u, sigma, z, params: SystemParameters):
    """Generate all plots for the solution.
    
    Returns:
        dict: Dictionary of figure names and their matplotlib Figure objects
    """
    r = x[:3,:]          # positions
    v = x[3:,:]          # velocities
    t = np.arange(params.N)*params.dt
    Path('figs').mkdir(exist_ok=True)

    # Create all figures
    fig_3d = plot_trajectory3d(r, u)
    fig_time = plot_time_histories(t, u, sigma, v, z, params)
    fig_ground = plot_groundtrack(r, params)
    fig_fancy = fancy_trajectory_plot(r, v, params)
    
    return {
        'trajectory3d': fig_3d,
        'time_histories': fig_time,
        'groundtrack': fig_ground,
        'fancy': fig_fancy
    } 