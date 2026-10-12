import rclpy
from rclpy.node import Node
from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from casadi import horzcat, diag
from scipy.linalg import block_diag
from .crazyflie_simplified_model import QuadrotorSimplified
from pathlib import Path
import importlib
import sys
import tf2_ros
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TwistStamped, PoseStamped

TAKEOFF_DURATION = 2.0
HOVER_DURATION = 2.0
LAND_DURATION = 3.0

class TrajectoryTrackingMPC(Node):
    def __init__(self, cf, trajectory, quadrotor, horizon, num_steps, code_export_directory):
        super().__init__('mpc_controller')

        # Parameters
        self.cf = cf
        self.model_name = "Icarus"
        self.trajectory = trajectory
        self.quad = quadrotor
        self.hover_control = np.array([0., 0., 0., self.quad.gravity * self.quad.mass]) # Hover keyframe

        # MPC parameters
        self.horizon = horizon
        self.num_steps = num_steps
        self.solver_locked = False
        self.mpc_N = num_steps
        self.mpc_tf = horizon
        self.last_u = self.hover_control # Initialize last_u to hover control to ensure safe commands if MPC fails on first iteration
        self.ocp_solver = None

        # Trajectory tracking parameters
        self.flight_mode = 'hover2trajectory'
        self.trajectory_t0 = None
        self.trajectory_start_position = None
        self.go_to_position = None
        self.current_height = None
        self.current_position = None

        # Velocity estimation
        self.prev_time = None
        self.prev_pos = None

        # Timing
        self.timer = self.create_timer(0.04, self.loop) # Control loop at 25Hz 
        self.t = 0

        # TF setup for pose extraction
        self.world_frame = 'world'
        self.body_frame = self.cf.prefix.lstrip('/')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Simulation telemetry uses the same structure as the hardware controller.
        # self.vel_pub = self.create_publisher(
        #     TwistStamped, f'/{self.body_frame}/velocity_estimate', 10
        # )
        # self.desired_pub = self.create_publisher(
        #     PoseStamped, f'/{self.body_frame}/desired_pose', 10
        # )
        # self.desired_vel_pub = self.create_publisher(
        #     TwistStamped, f'/{self.body_frame}/desired_velocity', 10
        # )

        # Ensure Path object
        self.acados_generated_files_path = Path(code_export_directory)

        try:
            if self.acados_generated_files_path.is_dir():
                sys.path.append(str(self.acados_generated_files_path))
            acados_ocp_solver_pyx = importlib.import_module('c_generated_code.acados_ocp_solver_pyx')
            self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(
                self.model_name, 'SQP', self.num_steps
            )
            print('Acados cython module imported successfully.')
        except ImportError:
            print('Generating acados solver...')
            self.generate_mpc()

    # def publish_velocity(self, vel: np.ndarray):
    #     msg = TwistStamped()
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.header.frame_id = 'world'
    #     msg.twist.linear.x = float(vel[0])
    #     msg.twist.linear.y = float(vel[1])
    #     msg.twist.linear.z = float(vel[2])

    #     # Angular velocity not estimated — left as zero
    #     self.vel_pub.publish(msg)

    def generate_mpc(self):
        '''As of now, there are only hard constraints. For better MPC feasibility,
           maybe implement soft constraints (but then include penalty terms in the
           cost function).'''

        f_expl, x, u = self.quad.dynamics()
        
        # Define model
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.name = self.model_name
        
        # Initialize the OCP
        ocp = AcadosOcp()
        ocp.model = model

        # Export pre-compiled solver
        ocp.code_export_directory = str(self.acados_generated_files_path / 'c_generated_code')

        # Dimensions
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        ocp.dims.N = self.num_steps
        ocp.solver_options.tf = self.horizon

        # Cost matrices
        Q = np.diag([12., 12., 12., 3., 3., 9., 0., 0., 1.])
        R = diag(horzcat(2, 2, 2, 5))
        W = block_diag(Q, R)

        # Formulate problem as Linear Least Squares (y - y_ref, i.e. x - x_ref) with weight matrix W
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.Vx = np.vstack([np.identity(nx), np.zeros((nu, nx))])
        ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.identity(nu)])
        ocp.cost.W = W
        ocp.cost.yref = np.zeros(ny)

        # Penalize end state
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W_e = Q
        ocp.cost.Vx_e = np.identity(nx)
        ocp.cost.yref_e = np.zeros(ny_e)

        # Constraints
        max_angle = np.radians(35)
        max_thrust = 0.56 # Maximum thrust in Newtons (from Crazyflie 2.1 datasheet, 0.56N per motor at full throttle)

        ocp.constraints.lbu = np.array([-max_angle, -max_angle, -np.radians(10), 0.])
        ocp.constraints.ubu = np.array([ max_angle,  max_angle,  np.radians(10), max_thrust])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3]) # Index for input bounds

        ocp.constraints.lbx = np.array([-10., -10., 0., -2., -2., -2., -max_angle, -max_angle, -np.radians(180)])
        ocp.constraints.ubx = np.array([ 10.,  10., 4.0,  2.,  2.,  2.,  max_angle,  max_angle,  np.radians(180)])
        ocp.constraints.idxbx = np.arange(nx)

        ocp.constraints.x0 = np.zeros(nx) # Initialization of initial state constraint (not important, will be overwritten at each MPC iteration)

        json_file = str(self.acados_generated_files_path / 'acados_ocp.json')

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_max_iter = 30
        AcadosOcpSolver.generate(ocp, json_file=json_file)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)

        sys.path.append(str(self.acados_generated_files_path))
        acados_ocp_solver_pyx = importlib.import_module('c_generated_code.acados_ocp_solver_pyx')

        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(
            self.model_name, 'SQP', self.num_steps
        )

    def solve_mpc(self, x0, yref, yref_e):
        if self.solver_locked:
            return
        self.solver_locked = True

        N = self.num_steps

        # Retrieve previous solution 
        prev_x = np.array([self.ocp_solver.get(i, 'x') for i in range(N+1)])
        prev_u = np.array([self.ocp_solver.get(i, 'u') for i in range(N)])

        for i in range(N):
            self.ocp_solver.set(i, 'yref', np.array([*yref[:, i], *self.hover_control]))

            # Warm start with previous solution (not necessary, but helps with convergence and reduces solve time)
            # Setting new solution to old solution shifted by one time step, i.e. u[0] = u_prev[1], x[0] = x_prev[1], etc.
            # Final time step is duplicated as a simple guess
            self.ocp_solver.set(i, 'x', prev_x[i+1])
            self.ocp_solver.set(i, 'u', prev_u[min(i+1, N-1)])

        self.ocp_solver.set(N, 'yref', yref_e)

        # Forcs equality constraint, i.e. lbx[0] = ubx[0] = x[0]
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)

        status = self.ocp_solver.solve()

        u0 = self.ocp_solver.get(0, "u")

        self.solver_locked = False
        return status, u0
    
    def navigator(self, t):

        if self.flight_mode == 'hover2trajectory':
            t_mpc_array = np.linspace(t, self.mpc_tf + t, self.mpc_N + 1)
            ramp_duration = TAKEOFF_DURATION  # seconds to reach target height

            yref = np.array([
                np.array([
                    *self.go_to_position[:2],
                    min(0.5, (t_mpc / ramp_duration) * 0.5),  # height ramps up
                    0., 0., 0., 0., 0., 0.
                ]) for t_mpc in t_mpc_array
            ]).T

            if self.t > (TAKEOFF_DURATION + HOVER_DURATION):
                self.flight_mode = 'trajectory'

        elif self.flight_mode == 'trajectory':
            t = t - (TAKEOFF_DURATION + HOVER_DURATION)
            t_mpc_array = np.linspace(t, self.mpc_tf + t, self.mpc_N + 1)
            yref_list = []
            traj_end_time = self.trajectory.duration

            for t_mpc in t_mpc_array:
                if t_mpc > traj_end_time:
                    ds = self.trajectory.eval(traj_end_time)
                else:
                    ds = self.trajectory.eval(t_mpc)

                yref_t = np.concatenate((
                    ds.pos[0:2] + self.trajectory_start_position[0:2],
                    np.array([0.5]),
                    # ds.pos + self.trajectory_start_position,
                    ds.vel,
                    np.array([0.0, 0.0, 0.0])
                ))
                yref_list.append(yref_t)

            yref = np.array(yref_list).T

            # If there are no more desired trajectory states, land
            if self.t > (TAKEOFF_DURATION + HOVER_DURATION) + self.trajectory.duration:
                self.flight_mode = 'land'
                self.current_height = 0.5
                

        elif self.flight_mode == 'land':
            t_mpc_array = np.linspace(t, self.mpc_tf + t, self.mpc_N + 1)
            land_start_t = (TAKEOFF_DURATION + HOVER_DURATION) + self.trajectory.duration     # when land mode begins
            ramp_duration = LAND_DURATION
            yref = np.array([
                np.array([
                    *self.go_to_position[:2],
                    max(0.0, self.current_height * (1.0 - (t_mpc - land_start_t) / ramp_duration)),
                    0., 0., 0., 0., 0., 0.
                ]) for t_mpc in t_mpc_array
            ]).T

            if self.current_position[2] < 0.05:
                self.cf.cmdVel(0.0,0.0,0.0,0.0)
                raise SystemExit
            
        return yref

    def loop(self):

        try:
            t = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.body_frame,
                rclpy.time.Time()
            )
        except tf2_ros.TransformException as exc:
            self.get_logger().warning(
                f"Waiting for TF {self.world_frame} <- {self.body_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return
        
        p = t.transform.translation
        q = t.transform.rotation

        pos = np.array([p.x, p.y, p.z])
        quat = np.array([q.w, q.x, q.y, q.z])
        self.current_position = pos
        
        # Using real clock for robustness to varying loop time (latency, ocp solve time, etc)
        if self.trajectory_t0 is None:
            self.trajectory_t0 = self.get_clock().now()
            self.prev_time = self.trajectory_t0
            # Record initial trajectory position on first iteration
            self.trajectory_start_position = pos
            self.prev_pos = pos

            if self.flight_mode == 'hover2trajectory':
                self.go_to_position = pos + np.array([0., 0., 0.5])

            if self.flight_mode == 'trajectory':
                self.go_to_position = pos

            return # Skip this iteration to ensure valid dt and vel on next iteration

        now = self.get_clock().now()
        self.t = (now - self.trajectory_t0).nanoseconds / 1e9
        dt = (now - self.prev_time).nanoseconds / 1e9
        self.prev_time = now

        if self.t > TAKEOFF_DURATION + HOVER_DURATION + self.trajectory.duration + LAND_DURATION:
            self.cf.cmdVel(0.0, 0.0, 0.0, 0.0)
            raise SystemExit

        trajectory = self.navigator(self.t)
        yref = trajectory[:, :-1]
        yref_e = trajectory[:, -1]

        # Estimate velocity from successive simulation poses
        vel = (pos - self.prev_pos) / dt
        self.prev_pos = pos
        
        r = R.from_quat(quat, scalar_first=True)
        roll, pitch, yaw = r.as_euler('xyz')

        x0 = np.array([*pos, *vel, roll, pitch, yaw])

        status, u_mpc = self.solve_mpc(x0, yref, yref_e)

        if status != 0:
            self.get_logger().warn(f"MPC failed: {status}")
            roll, pitch, yaw_rate, thrust = self.last_u # If MPC fails, use last control input as fallback
        else:
            roll, pitch, yaw_rate, thrust = u_mpc
            self.last_u = u_mpc

        # Thrust can be scaled linearly to PWM in simulation
        thrust_pwm = thrust / 0.4776 * 2**16

        # Preserve the simulation backend's angular command convention.
        self.cf.cmdVel(roll, pitch, yaw_rate, float(thrust_pwm))

        self.get_logger().info(f"umpc: {u_mpc}")

        # Publish velocity
        # self.publish_velocity(vel)

        # des_pos = yref[0:3, 0]
        # des_vel = yref[3:6, 0]

        # # Publish desired pose
        # pose_msg = PoseStamped()
        # pose_msg.header.stamp = self.get_clock().now().to_msg()
        # pose_msg.header.frame_id = 'world'
        # pose_msg.pose.position.x = float(des_pos[0])
        # pose_msg.pose.position.y = float(des_pos[1])
        # pose_msg.pose.position.z = float(des_pos[2])
        # pose_msg.pose.orientation.w = 1.0
        # self.desired_pub.publish(pose_msg)

        # # Publish desired velocity
        # vel_msg = TwistStamped()
        # vel_msg.header.stamp = self.get_clock().now().to_msg()
        # vel_msg.header.frame_id = 'world'
        # vel_msg.twist.linear.x = float(des_vel[0])
        # vel_msg.twist.linear.y = float(des_vel[1])
        # vel_msg.twist.linear.z = float(des_vel[2])
        # self.desired_vel_pub.publish(vel_msg)


def main():
    swarm = Crazyswarm()
    cf = swarm.allcfs.crazyflies[0]

    traj = Trajectory()
    traj.loadcsv(Path(__file__).parent / "data/polynomials/lemniscate.csv")
    
    model = QuadrotorSimplified()

    node = TrajectoryTrackingMPC(
        cf,
        traj,
        model,
        1.0,
        50,
        Path(__file__).parent / "acados_generated_files_sim"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
