#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
from pathlib import Path
import numpy as np
import time
from crazyflie_interfaces.msg import LogDataGeneric
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from geometry_msgs.msg import TwistStamped, PoseStamped
from collections import deque

TAKEOFF_DURATION = 2.0
HOVER_DURATION = 2.0

class FullStateController(Node):
    def __init__(self, cf, trajectory):
        super().__init__('fs_controller')

        # Parameters
        self.cf = cf
        self.trajectory = trajectory

        # Control loop at 100 Hz
        self.timer = self.create_timer(0.01, self.loop)

        # Trajectory tracking parameters
        self.trajectory_t0 = None
        self.trajectory_start_position = None

        # Velocity estimation
        self.prev_time = None
        self.prev_pos = None
        self.prev_vel = None

        self.err_vels = deque(maxlen=5) # Sliding window for velocity estimate anomaly compensation

        self.t = 0 # Trajectory time-stepping

        self.poses = {}

        # QoS for mocap
        qos = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT
        )

        # Subscription for MoCap pose 
        self.subscription = self.create_subscription(
            NamedPoseArray,
            '/poses',
            self.pose_callback,
            qos
        )

        # Subscription for Kalman Filter velocity estimate
        self.world_vel_subscription = self.create_subscription(
            LogDataGeneric,
            '/cf2/world_vel',
            self.wvel_callback,
            qos
        )

        # Publisher for velocity estimate 
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/cf2/velocity_estimate',
            10
        )

        self.desired_pub = self.create_publisher(PoseStamped, '/cf2/desired_pose', 10)
        self.desired_vel_pub = self.create_publisher(TwistStamped, '/cf2/desired_velocity', 10)

    def pose_callback(self, msg):
        for p in msg.poses:
            name = p.name
            pos = p.pose.position
            ori = p.pose.orientation

            self.poses[name] = {
                "pos": np.array([pos.x, pos.y, pos.z]),
                "quat": np.array([ori.w, ori.x, ori.y, ori.z])
            }

    def publish_velocity(self, vel: np.ndarray):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.twist.linear.x = float(vel[0])
        msg.twist.linear.y = float(vel[1])
        msg.twist.linear.z = float(vel[2])
        # Angular velocity not estimated — left as zero
        self.vel_pub.publish(msg)

    def wvel_callback(self, msg):
        self.world_vel = np.array(msg.values)  # [vx, vy, vz] in body frame

    def loop(self):

        if "cf2" not in self.poses:
            return 

        pos = self.poses["cf2"]["pos"]
        # quat = self.poses["cf2"]["quat"]
        
        # If there are no more desired trajectory states, land
        if self.t > self.trajectory.duration:

            # Reset commander to allow high-level commands
            self.cf.notifySetpointsStop()

            # Now land() will work
            self.cf.land(targetHeight=0.04, duration=2.5)
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=3))

            raise SystemExit 
        
        # Using real clock for robustness to varying loop time (latency, ocp solve time, etc)
        if self.trajectory_t0 is None:
            self.trajectory_t0 = self.get_clock().now()
            self.prev_time = self.trajectory_t0
            # Record initial trajectory position on first iteration
            self.trajectory_start_position = pos
            self.prev_pos = pos
            self.prev_vel = np.zeros(3)

            return # Skip this iteration to ensure valid dt and vel on next iteration

        now = self.get_clock().now()
        self.t = (now - self.trajectory_t0).nanoseconds / 1e9
        dt = (now - self.prev_time).nanoseconds / 1e9
        self.prev_time = now

        # Two point finite difference (apropriate due to sub cm accuracy of mocap)
        vel = (pos - self.prev_pos) / dt
        self.prev_pos = pos

        current_err_vel = np.abs(vel - self.prev_vel)
        
        means = np.mean(np.abs(self.err_vels), 0)

        if np.allclose(vel, 0) or (np.any(current_err_vel > means + 0.2)): 
            vel = self.world_vel
            
        # Desired states from trajectory generator
        ds = self.trajectory.eval(self.t) # Named-tuple object

        traj_pos = ds.pos + np.array([self.trajectory_start_position[0], self.trajectory_start_position[1], 0.3])

        self.cf.cmdFullState(traj_pos, ds.vel, ds.acc, ds.yaw, ds.omega)

        self.publish_velocity(vel)

        err_vel = np.abs(vel - self.prev_vel)
        self.err_vels.append(err_vel)
        self.prev_vel = vel

        # Publish desired pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = float(traj_pos[0])
        pose_msg.pose.position.y = float(traj_pos[1])
        pose_msg.pose.position.z = float(traj_pos[2])
        pose_msg.pose.orientation.w = 1.0
        self.desired_pub.publish(pose_msg)

        # Publish desired velocity
        vel_msg = TwistStamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = 'world'
        vel_msg.twist.linear.x = float(ds.vel[0])
        vel_msg.twist.linear.y = float(ds.vel[1])
        vel_msg.twist.linear.z = float(ds.vel[2])
        self.desired_vel_pub.publish(vel_msg)

def main(args=None):
    # Crazyswarm initializes rclpy internally
    
    swarm = Crazyswarm()
    
    cf = swarm.allcfs.crazyflies[0]

    traj = Trajectory()
    traj.loadcsv(Path(__file__).parent / "data/polynomials/figure8.csv")

    cf.takeoff(targetHeight=0.3, duration=TAKEOFF_DURATION)
    time.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    
    # Instantiate controller with trajectory
    node = FullStateController(cf, traj)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()




if __name__ == '__main__':
    main()