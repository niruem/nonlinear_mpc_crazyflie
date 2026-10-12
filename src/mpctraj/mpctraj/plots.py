#!/usr/bin/env python3
"""
Compare:
  1. Commanded RPY  (/cf2/rpyt)         vs actual RPY   (/poses)
  2. Desired position (/cf2/desired_pose)  vs actual position (/poses)
  3. Desired velocity (/cf2/desired_velocity) vs estimated velocity (/cf2/velocity_estimate)
  4. 2D trajectory (top-down XY + altitude over X)
  5. 3D trajectory

Usage:
    python3 plots.py <path_to_bag_directory>
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_3D = True
except ImportError:
    HAS_3D = False
    print("Warning: 3D plotting unavailable (matplotlib version conflict). "
          "Only 2D trajectory plot will be saved.")
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


# ── Helpers ─────────────────────────────────────────────────────────────────

def quat_to_rpy(w, x, y, z):
    rot = R.from_quat([x, y, z, w])
    return np.degrees(rot.as_euler('xyz', degrees=False))


def make_typestore():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    msg_dir = Path('/home/mumos_assignment4/cffly_ws/install')
    custom_types = {}
    for msg_file in msg_dir.rglob('*.msg'):
        parts = msg_file.parts
        try:
            msg_idx = parts.index('msg')
            pkg = parts[msg_idx - 1]
        except ValueError:
            continue
        typename = f'{pkg}/msg/{msg_file.stem}'
        try:
            custom_types.update(get_types_from_msg(msg_file.read_text(), typename))
        except Exception as e:
            print(f"  Skipping {typename}: {e}")
    print(f"  Registered {len(custom_types)} custom types")
    typestore.register(custom_types)
    return typestore


# ── Bag reader ───────────────────────────────────────────────────────────────

def read_bag(bag_path: str):
    typestore = make_typestore()

    poses_data    = []  # (t, x, y, z, roll, pitch, yaw)
    rpyt_data     = []  # (t, roll, pitch, yaw, thrust)
    des_pose_data = []  # (t, x, y, z)
    des_vel_data  = []  # (t, vx, vy, vz)
    est_vel_data  = []  # (t, vx, vy, vz)

    with Reader(bag_path) as reader:
        connections = {c.topic: c for c in reader.connections}
        print(f"Topics in bag: {list(connections.keys())}")

        for conn, timestamp, rawdata in reader.messages():
            t = timestamp / 1e9

            if conn.topic == '/poses':
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                for p in msg.poses:
                    if p.name == 'cf2':
                        pos = p.pose.position
                        ori = p.pose.orientation
                        rpy = quat_to_rpy(ori.w, ori.x, ori.y, ori.z)
                        poses_data.append((t, pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]))

            elif conn.topic == '/cf2/rpyt':
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                v = msg.values
                if len(v) >= 3:
                    rpyt_data.append((t, float(v[0]), float(v[1]), float(v[2]),
                                      float(v[3]) if len(v) > 3 else 0.0))

            elif conn.topic == '/cf2/desired_pose':
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                pos = msg.pose.position
                des_pose_data.append((t, pos.x, pos.y, pos.z))

            elif conn.topic == '/cf2/desired_velocity':
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                lv = msg.twist.linear
                des_vel_data.append((t, lv.x, lv.y, lv.z))

            elif conn.topic == '/cf2/velocity_estimate':
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                lv = msg.twist.linear
                est_vel_data.append((t, lv.x, lv.y, lv.z))

    def to_arr(lst, name):
        arr = np.array(lst) if lst else np.empty((0, 0))
        print(f"  {name:35s}: {len(arr)} samples")
        return arr

    return (
        to_arr(poses_data,    '/poses'),
        to_arr(rpyt_data,     '/cf2/rpyt'),
        to_arr(des_pose_data, '/cf2/desired_pose'),
        to_arr(des_vel_data,  '/cf2/desired_velocity'),
        to_arr(est_vel_data,  '/cf2/velocity_estimate'),
    )


# ── Signal alignment ─────────────────────────────────────────────────────────

def align(t_ref, data_ref, t_src, data_src):
    """Interpolate data_src onto t_ref timestamps, clipped to overlap."""
    t_start = max(t_ref[0], t_src[0])
    t_end   = min(t_ref[-1], t_src[-1])
    mask = (t_ref >= t_start) & (t_ref <= t_end)
    t_common = t_ref[mask]
    ref_clipped = data_ref[mask]
    src_interp = np.column_stack([
        np.interp(t_common, t_src, data_src[:, i])
        for i in range(data_src.shape[1])
    ])
    return t_common - t_common[0], ref_clipped, src_interp


# ── Generic comparison plot ───────────────────────────────────────────────────

def plot_panel(fig, gs, row, t, actual, desired, label, unit,
               color_actual, color_desired, is_last_row=False):
    error = actual - desired
    rmse  = np.sqrt(np.mean(error**2))
    mae   = np.mean(np.abs(error))

    ax = fig.add_subplot(gs[row, 0])
    ax.plot(t, desired, '--', color=color_desired, linewidth=1.2,
            label='Desired', alpha=0.85)
    ax.plot(t, actual,  '-',  color=color_actual,  linewidth=1.2,
            label='Actual',  alpha=0.95)
    ax.fill_between(t, desired, actual, alpha=0.15, color=color_actual)
    ax.set_ylabel(unit)
    ax.set_title(f'{label}   RMSE={rmse:.3f}   MAE={mae:.3f}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    if is_last_row:
        ax.set_xlabel('Time (s)')

    ax_e = fig.add_subplot(gs[row, 1])
    ax_e.plot(t, error, color=color_actual, linewidth=0.8)
    ax_e.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax_e.set_title(f'{label} error')
    ax_e.set_ylabel(unit)
    ax_e.grid(True, alpha=0.3)
    if is_last_row:
        ax_e.set_xlabel('Time (s)')

    return ax, ax_e


def make_figure(title, nrows, bag_path):
    fig = plt.figure(figsize=(14, 3.5 * nrows))
    fig.suptitle(f'{title}\n{bag_path}', fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(nrows, 2, figure=fig,
                           width_ratios=[3, 1], hspace=0.55, wspace=0.35)
    return fig, gs


def finalise(fig, filename):
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved → {filename}")


# ── Print stats ───────────────────────────────────────────────────────────────

def print_stats(labels, actual, desired, unit):
    print(f"\n{'─'*65}")
    print(f"  {'':12} {'RMSE':>10} {'MAE':>10} {'Max|err|':>12} {'Mean err':>12}  ({unit})")
    print(f"{'─'*65}")
    for i, lbl in enumerate(labels):
        err = actual[:, i] - desired[:, i]
        print(f"  {lbl:12} {np.sqrt(np.mean(err**2)):>10.4f} "
              f"{np.mean(np.abs(err)):>10.4f} "
              f"{np.max(np.abs(err)):>12.4f} "
              f"{np.mean(err):>12.4f}")
    print(f"{'─'*65}")


# ── Trajectory plots ──────────────────────────────────────────────────────────

def plot_trajectory(poses, des_pose, bag_path):
    actual_xyz  = poses[:, 1:4]
    desired_xyz = des_pose[:, 1:4]

    # ── 2D (top-down + altitude) ──────────────────────────────────────────────
    fig2d, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig2d.suptitle(f'Trajectory (2D)\n{bag_path}', fontsize=12, fontweight='bold')

    # XY top-down
    ax = axes[0]
    ax.plot(desired_xyz[:, 0], desired_xyz[:, 1], '--', color='#2980b9',
            linewidth=1.5, label='Desired', alpha=0.85)
    ax.plot(actual_xyz[:, 0],  actual_xyz[:, 1],  '-',  color='#e74c3c',
            linewidth=1.5, label='Actual',  alpha=0.95)
    ax.scatter(*desired_xyz[0,  :2], color='green',    s=80, zorder=5,
               marker='o', label='Start')
    ax.scatter(*desired_xyz[-1, :2], color='#2c3e50',  s=80, zorder=5,
               marker='x', label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Top-down (XY)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Altitude over X
    ax = axes[1]
    ax.plot(desired_xyz[:, 0], desired_xyz[:, 2], '--', color='#2980b9',
            linewidth=1.5, label='Desired', alpha=0.85)
    ax.plot(actual_xyz[:, 0],  actual_xyz[:, 2],  '-',  color='#e74c3c',
            linewidth=1.5, label='Actual',  alpha=0.95)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Altitude over X')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    finalise(fig2d, 'trajectory_2d.png')

    # ── 3D ────────────────────────────────────────────────────────────────────
    if not HAS_3D:
        print("Skipping 3D plot — install a consistent matplotlib version to enable it.\n"
              "  pip install --upgrade matplotlib")
        return

    fig3d = plt.figure(figsize=(10, 7))
    fig3d.suptitle(f'Trajectory (3D)\n{bag_path}', fontsize=12, fontweight='bold')
    ax3 = fig3d.add_subplot(111, projection='3d')

    ax3.plot(*desired_xyz.T, '--', color='#2980b9', linewidth=1.5,
             label='Desired', alpha=0.85)
    ax3.plot(*actual_xyz.T,  '-',  color='#e74c3c', linewidth=1.5,
             label='Actual',  alpha=0.95)

    ax3.scatter(*desired_xyz[0],  color='green',   s=60, zorder=5, label='Start')
    ax3.scatter(*desired_xyz[-1], color='#2c3e50', s=60, zorder=5,
                marker='^', label='End')

    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    finalise(fig3d, 'trajectory_3d.png')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    bag_path = sys.argv[1]
    print(f"\nReading bag: {bag_path}")

    poses, rpyt, des_pose, des_vel, est_vel = read_bag(bag_path)

    # ── 1. RPY comparison ────────────────────────────────────────────────────
    if len(poses) and len(rpyt):
        t, actual_rpy, cmd_rpy = align(
            poses[:, 0], poses[:, 4:7],
            rpyt[:, 0],  rpyt[:, 1:4]
        )
        print("\n── RPY (Actual − Commanded) ─────────────────────────────────")
        print_stats(['Roll', 'Pitch', 'Yaw'], actual_rpy, cmd_rpy, 'deg')

        fig, gs = make_figure('Commanded vs Actual RPY', 3, bag_path)
        colors_a = ['#e74c3c', '#2ecc71', '#3498db']
        colors_d = ['#c0392b', '#27ae60', '#2980b9']
        for i, lbl in enumerate(['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']):
            actual = actual_rpy[:, i]
            cmd    = -cmd_rpy[:, i] if i == 1 or i == 2 else cmd_rpy[:, i]
            plot_panel(fig, gs, i, t,
                       actual, cmd,
                       lbl, 'deg', colors_a[i], colors_d[i],
                       is_last_row=(i == 2))
        finalise(fig, 'rpy_comparison.png')

    # ── 2. Position comparison ───────────────────────────────────────────────
    if len(poses) and len(des_pose):
        t, actual_pos, desired_pos = align(
            poses[:, 0], poses[:, 1:4],
            des_pose[:, 0], des_pose[:, 1:4]
        )
        print("\n── Position (Actual − Desired) ──────────────────────────────")
        print_stats(['X', 'Y', 'Z'], actual_pos, desired_pos, 'm')

        fig, gs = make_figure('Desired vs Actual Position', 3, bag_path)
        colors_a = ['#e74c3c', '#2ecc71', '#3498db']
        colors_d = ['#c0392b', '#27ae60', '#2980b9']
        for i, lbl in enumerate(['X (m)', 'Y (m)', 'Z (m)']):
            plot_panel(fig, gs, i, t,
                       actual_pos[:, i], desired_pos[:, i],
                       lbl, 'm', colors_a[i], colors_d[i],
                       is_last_row=(i == 2))
        finalise(fig, 'position_comparison.png')

    # ── 3. Velocity comparison ───────────────────────────────────────────────
    if len(est_vel) and len(des_vel):
        t, actual_vel, desired_vel = align(
            est_vel[:, 0], est_vel[:, 1:4],
            des_vel[:, 0], des_vel[:, 1:4]
        )
        print("\n── Velocity (Estimated − Desired) ───────────────────────────")
        print_stats(['Vx', 'Vy', 'Vz'], actual_vel, desired_vel, 'm/s')

        fig, gs = make_figure('Desired vs Estimated Velocity', 3, bag_path)
        colors_a = ['#e74c3c', '#2ecc71', '#3498db']
        colors_d = ['#c0392b', '#27ae60', '#2980b9']
        for i, lbl in enumerate(['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']):
            plot_panel(fig, gs, i, t,
                       actual_vel[:, i], desired_vel[:, i],
                       lbl, 'm/s', colors_a[i], colors_d[i],
                       is_last_row=(i == 2))
        finalise(fig, 'velocity_comparison.png')

    # ── 4. Trajectory plots ──────────────────────────────────────────────────
    if len(poses) and len(des_pose):
        plot_trajectory(poses, des_pose, bag_path)

    plt.show()


if __name__ == '__main__':
    main()