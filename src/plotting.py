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
from matplotlib.collections import LineCollection
from PIL import Image

def plot_trajectory3d(r, u):
    """3-D path coloured by thrust magnitude.
    
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    thr = np.linalg.norm(u.T, axis=1)
    norm = colors.Normalize(vmin=thr.min(), vmax=thr.max())
    cmap = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot with altitude on z-axis for better visualization
    x, y, z = r[1], r[2], r[0]  # Rearrange coordinates
    
    for k in range(len(x)-1):
        ax.plot([x[k], x[k+1]], 
                [y[k], y[k+1]], 
                [z[k], z[k+1]], 
                color=cmap(norm(thr[k])), linewidth=2)
                
    ax.set(xlabel='Cross-range [m]', ylabel='Down-range [m]', zlabel='Altitude [m]')
    ax.set_title('Powered-descent trajectory')
    
    # Add colorbar with proper mappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Thrust [N]')
    
    # Add projections onto walls
    ax.plot(x, y, np.zeros_like(z), 'k--', alpha=0.2)  # Ground projection
    ax.plot(x, np.full_like(y, y.max()), z, 'k--', alpha=0.2)  # Back wall
    ax.plot(np.full_like(x, x.min()), y, z, 'k--', alpha=0.2)  # Side wall
    
    # Add initial and final points
    ax.scatter([x[0]], [y[0]], [z[0]], color='g', s=100, label='Start')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color='r', s=100, label='End')
    
    ax.legend()
    
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

def draw_rocket(ax, pos_plot, thrust_plot, umax, scale=250):
    """Draw a rocket with proper orientation and thrust visualization.
    
    Args:
        ax: 3D axis to draw on
        pos_plot: [x,y,z] position in plotting coordinates
        thrust_plot: [ux,uy,uz] thrust vector at current position in plotting coordinates
        umax: Maximum acceleration magnitude for scaling
        scale: Size scaling factor
    """
    # Get thrust direction for rocket orientation
    thrust_dir = thrust_plot / (np.linalg.norm(thrust_plot) + 1e-6)  # Normalized thrust vector
    
    # Create rotation matrix to align rocket with thrust direction
    # First vector is UP (rocket points up, thrust points down)
    v1 = -thrust_dir  # Thrust direction (down)
    v2 = np.array([0, 1, 0])  # Initial guess for side direction
    v2 = v2 - np.dot(v2, v1) * v1  # Make perpendicular to v1
    v2 = v2 / (np.linalg.norm(v2) + 1e-6)  # Normalize
    v3 = np.cross(v1, v2)  # Third orthogonal vector
    R = np.vstack([-v1, v2, v3]).T  # Note: -v1 to make rocket point UP
    
    # Rocket dimensions (taller & slightly wider overall)
    h = scale * 2.0          # Body height doubled for tall Falcon-9 look
    r = scale / 12.0         # Slightly wider body radius
    nose_h = 0.5 * scale     # Taller nose cone for Starship look
    n = 20  # number of points for circles
    
    # Create rocket body points (cylinder)
    theta = np.linspace(0, 2*np.pi, n)
    circle = np.vstack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    
    # ------------------------------------------------ body ------------------------------------------------
    # Bottom circle (engine end) – this is pos_plot
    bottom = pos_plot.reshape(-1,1) + R @ (r * circle)
    ax.plot(bottom[0], bottom[1], bottom[2], 'k-', alpha=0.6)
    
    # Top circle
    top = pos_plot.reshape(-1,1) + R @ (r * circle + np.array([[h], [0], [0]]))
    ax.plot(top[0], top[1], top[2], 'k-', alpha=0.6)
    
    # Generate cylinder surface (body)
    n_theta = 48  # higher circumferential resolution
    n_h = 12      # more slices along height
    theta_c = np.linspace(0, 2*np.pi, n_theta)
    z_c = np.linspace(0, h, n_h)
    Theta_c, Z_c = np.meshgrid(theta_c, z_c)
    # Local coords
    Xl_c = Z_c                      # along body axis
    Yl_c = r * np.cos(Theta_c)
    Zl_c = r * np.sin(Theta_c)
    # Transform to world
    Xw_c = pos_plot[0] + R[0,0]*Xl_c + R[0,1]*Yl_c + R[0,2]*Zl_c
    Yw_c = pos_plot[1] + R[1,0]*Xl_c + R[1,1]*Yl_c + R[1,2]*Zl_c
    Zw_c = pos_plot[2] + R[2,0]*Xl_c + R[2,1]*Yl_c + R[2,2]*Zl_c
    ax.plot_surface(Xw_c, Yw_c, Zw_c, color='dimgray', shade=True, linewidth=0, antialiased=False, alpha=0.9)

    # Optional: thin black outline for body centre line
    ax.plot([pos_plot[0], pos_plot[0] + R[0,0]*h],
            [pos_plot[1], pos_plot[1] + R[1,0]*h],
            [pos_plot[2], pos_plot[2] + R[2,0]*h],
            color='k', linewidth=1, alpha=0.6)

    # Generate nose cone surface
    n_h_cone = 12  # more slices for smoother cone
    z_cone = np.linspace(0, nose_h, n_h_cone)
    Theta_k, Z_k = np.meshgrid(theta_c, z_cone)
    # Ogive/pointier cone: radius decreases with power 1.5 for slender shape
    R_k = r * (1 - (Z_k / nose_h))**1.5
    Xl_k = h + Z_k                 # offset after body
    Yl_k = R_k * np.cos(Theta_k)
    Zl_k = R_k * np.sin(Theta_k)
    Xw_k = pos_plot[0] + R[0,0]*Xl_k + R[0,1]*Yl_k + R[0,2]*Zl_k
    Yw_k = pos_plot[1] + R[1,0]*Xl_k + R[1,1]*Yl_k + R[1,2]*Zl_k
    Zw_k = pos_plot[2] + R[2,0]*Xl_k + R[2,1]*Yl_k + R[2,2]*Zl_k
    ax.plot_surface(Xw_k, Yw_k, Zw_k, color='silver', shade=True, linewidth=0, antialiased=False, alpha=0.95)

    # Outline ring at cylinder-cone join for better visibility
    ax.plot(top[0], top[1], top[2], color='k', linewidth=0.8, alpha=0.6)

    # Spine line up the centre of the cone
    cone_tip = pos_plot + R @ np.array([h + nose_h, 0, 0])
    ax.plot([top[0,0], cone_tip[0]],
            [top[1,0], cone_tip[1]],
            [top[2,0], cone_tip[2]],
            color='k', linewidth=0.8, alpha=0.6)

    # ------------------------------------------------ landing legs --------------------------------------
    leg_angles = [0, 120, 240]  # Three legs spaced evenly around
    leg_length = h * 0.3        # Legs shorter relative to taller body
    leg_angle = 30              # Bring legs closer to body (smaller splay)
    
    for theta in leg_angles:
        # Rotate leg direction around rocket axis
        c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
        leg_dir = R @ np.array([
            -np.cos(np.deg2rad(leg_angle)),  # Points down from rocket body
            np.sin(np.deg2rad(leg_angle)) * c,
            np.sin(np.deg2rad(leg_angle)) * s
        ])
        leg_end = pos_plot + leg_length * leg_dir
        ax.plot([pos_plot[0], leg_end[0]],
                [pos_plot[1], leg_end[1]],
                [pos_plot[2], leg_end[2]], 'k-', linewidth=1.5, alpha=0.7)
        
        # Add foot pad
        foot_width = leg_length/4
        foot_dir = np.cross(leg_dir, R[:,1])  # Perpendicular to leg
        foot_start = leg_end - foot_width/2 * foot_dir
        foot_end = leg_end + foot_width/2 * foot_dir
        ax.plot([foot_start[0], foot_end[0]],
                [foot_start[1], foot_end[1]],
                [foot_start[2], foot_end[2]], 'k-', linewidth=1.5, alpha=0.7)
    
    # ---------------------------------------------- thrust plume ----------------------------------------
    thrust_mag = np.linalg.norm(thrust_plot)
    thrust_scale = thrust_mag / (umax + 1e-9)
    
    # Quadratic scaling for smoother change and visibility across range
    plume_length = scale * (1.0 + 4.0 * thrust_scale**1.2)  # 1×scale idle → 5×scale full thrust
    plume_radius = r * (0.15 + 0.85 * thrust_scale)  # 0.15r idle → r at max
    if thrust_scale < 0.01:  # <1% of umax: suppress plume
        plume_length = 0.0  # Practically no plume
        plume_radius = 0.0
    
    # Use warm colormap (red-orange) along length
    cmap_flame = cm.get_cmap('autumn_r')
    n_segments = 12  # smoother gradient
    for j in range(n_segments):
        t0 = j / n_segments
        t1 = (j + 1) / n_segments
        seg_start = pos_plot + t0 * plume_length * (-thrust_dir)
        seg_end   = pos_plot + t1 * plume_length * (-thrust_dir)
        color_seg = cmap_flame(t0*0.8 + 0.2)  # more red near engine, yellow at end
        ax.plot([seg_start[0], seg_end[0]],
                [seg_start[1], seg_end[1]],
                [seg_start[2], seg_end[2]],
                color=color_seg, linewidth=4, alpha=0.85)
        # Add slight width by drawing circular offsets
        if plume_radius > 0:
            for theta_p in np.linspace(0, 2*np.pi, 6, endpoint=False):
                offset = plume_radius * np.array([0,
                    np.cos(theta_p),
                    np.sin(theta_p)])
                seg_start_o = seg_start + R @ offset
                seg_end_o   = seg_end   + R @ offset
                ax.plot([seg_start_o[0], seg_end_o[0]],
                        [seg_start_o[1], seg_end_o[1]],
                        [seg_start_o[2], seg_end_o[2]],
                        color=color_seg, linewidth=2, alpha=0.55)

def save_animation_frames(x, u, params, output_dir='animation', frames_per_sec: int = 5):
    """Save trajectory animation frames with temporal oversampling.
    
    Args:
        x (np.ndarray): State trajectory array (6×N)
        u (np.ndarray): Control trajectory array (3×N)
        params (SystemParameters): System parameters (contains dt)
        output_dir (str | Path): Directory where frame PNGs will be written
        frames_per_sec (int): Desired number of frames per simulation second
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    r = x[:3,:]  # positions
    v = x[3:,:]  # velocities
    
    # Precompute max thrust magnitude for scaling
    umax = np.linalg.norm(u, axis=0).max()
    if umax == 0:
        umax = 1.0
    # Get velocity range for consistent coloring
    vel_full = np.linalg.norm(v.T, axis=1)
    vel_norm = plt.Normalize(vel_full.min(), vel_full.max())
    cmap = plt.get_cmap('viridis')
    
    # Oversampling factor (frames between discrete simulation samples)
    k_sub = max(1, int(frames_per_sec * params.dt))

    frame_idx = 0
    # Iterate through each simulation step and create k_sub sub-frames via linear interpolation
    for i in range(len(r[0]) - 1):
        for sub in range(k_sub):
            frac = sub / k_sub
            # Linear interpolation of state & control
            pos_int = (1 - frac) * r[:, i] + frac * r[:, i + 1]
            thr_int = (1 - frac) * u[:, i] + frac * u[:, i + 1]

            # Alias for readability below
            xi, yi, zi = pos_int[1], pos_int[2], pos_int[0]

            fig = plt.figure(figsize=(12, 8))
            ax  = fig.add_subplot(111, projection='3d')

            # Plot trajectory up to this fractional time
            # positions already computed per integer index; for simplicity use original coarse history
            x_hist = r[1, :i+1]
            y_hist = r[2, :i+1]
            z_hist = r[0, :i+1]
            vel_hist = np.linalg.norm(v.T[:i+1], axis=1)

            for k in range(len(x_hist) - 1):
                ax.plot([x_hist[k], x_hist[k + 1]],
                        [y_hist[k], y_hist[k + 1]],
                        [z_hist[k], z_hist[k + 1]],
                        color=cmap(vel_norm(vel_hist[k])), linewidth=2)

            # Current rocket
            current_pos_plot = np.array([xi, yi, zi])
            current_thrust_plot = np.array([thr_int[1], thr_int[2], thr_int[0]])
            draw_rocket(ax, current_pos_plot, current_thrust_plot, umax)

            # ------------------------ ground plane & cones (unchanged) ------------------------
            # [We reuse the previous code unchanged …]
            # copied from earlier section but needs variables
            max_range = max(abs(r[1:].max()), abs(r[1:].min())) * 1.2
            xx, yy = np.meshgrid(np.linspace(-max_range, max_range, 10),
                                np.linspace(-max_range, max_range, 10))
            ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

            h0 = r[0, 0]
            theta_gs = np.deg2rad(params.glidelslope_angle)
            r_cone   = h0 / np.tan(theta_gs)
            phi = np.linspace(0, 2 * np.pi, 60)
            H  = np.linspace(0, h0, 30)
            PHI, HH = np.meshgrid(phi, H)
            RRR = HH / np.tan(theta_gs)
            X_cone = RRR * np.cos(PHI)
            Y_cone = RRR * np.sin(PHI)
            Z_cone = HH
            ax.plot_surface(X_cone, Y_cone, Z_cone, alpha=0.1, color='orange')

            # Ax aesthetics
            ax.set(xlabel='Cross-range [m]', ylabel='Down-range [m]', zlabel='Altitude [m]',
                   title=f'Powered Descent Trajectory (t = {(i + frac)*params.dt:.1f}s)')
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([0, r[0].max() * 1.2])
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=vel_norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Velocity [m/s]')
            ax.view_init(elev=20, azim=45)

            # Save frame
            plt.savefig(Path(output_dir) / f'frame_{frame_idx:04d}.png', dpi=100, bbox_inches='tight')
            plt.close()
            frame_idx += 1

    # Add final exact last state frame
    current_pos_plot = np.array([r[1, -1], r[2, -1], r[0, -1]])
    current_thrust_plot = np.array([u[1, -1], u[2, -1], u[0, -1]])
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # (optional: skip full plotting for brevity, but for consistency reuse previous small loop)
    ax.plot(r[1], r[2], r[0], color='k', alpha=0.2)
    draw_rocket(ax, current_pos_plot, current_thrust_plot, umax)
    ax.set_xlabel('Cross-range [m]'); ax.set_ylabel('Down-range [m]'); ax.set_zlabel('Altitude [m]')
    ax.set_xlim([-max_range, max_range]); ax.set_ylim([-max_range, max_range]); ax.set_zlim([0, r[0].max()*1.2])
    plt.savefig(Path(output_dir) / f'frame_{frame_idx:04d}.png', dpi=100, bbox_inches='tight')
    plt.close()

def create_gif(frame_dir='animation', output_file='trajectory.gif', duration=50):
    """Create GIF from animation frames.
    
    Args:
        frame_dir: Directory containing frames
        output_file: Output GIF filename
        duration: Duration for each frame in milliseconds
    """
    frames = []
    frame_files = sorted(Path(frame_dir).glob('frame_*.png'))
    
    for frame_file in frame_files:
        frames.append(Image.open(frame_file))
        
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    ) 