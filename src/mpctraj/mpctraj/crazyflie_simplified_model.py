from casadi import SX, vertcat, horzcat, cos, sin
import numpy as np

class QuadrotorSimplified:
    def __init__(self):
        self.mass = 0.03558 
        self.gravity = 9.80665
        self.arm_length = 0.044
        self.Ixx = 2.3951e-5
        self.Iyy = 2.3951e-5
        self.Izz = 3.2347e-5
        self.cm = 2.4e-6
        self.tau = 0.08 # attitude time constant
    
    @staticmethod
    def euler_to_rotm(rpy):
        cpitch = cos(rpy[1])
        spitch = sin(rpy[1])
        croll = cos(rpy[0])
        sroll = sin(rpy[0])
        cyaw = cos(rpy[2])
        syaw = sin(rpy[2])
        Rotm = np.array([
            [cpitch*cyaw, sroll*spitch*cyaw - croll*syaw, croll*spitch*cyaw + sroll*syaw],
            [cpitch*syaw, sroll*spitch*syaw + croll*cyaw, croll*spitch*syaw - sroll*cyaw],
            [-spitch, sroll*cpitch, croll*cpitch]]
        )
        return Rotm
    
    def dynamics(self):
        px = SX.sym('px')
        py = SX.sym('py')
        pz = SX.sym('pz')
        vx = SX.sym('vx')
        vy = SX.sym('vy')
        vz = SX.sym('vz')
        roll = SX.sym('roll')
        pitch = SX.sym('pitch')
        yaw = SX.sym('yaw')
        roll_c = SX.sym('roll_c')
        pitch_c = SX.sym('pitch_c')
        yaw_c = SX.sym('yaw_c')
        thrust = SX.sym('thrust')

        # Setup state and control vectors
        x = vertcat(px,py,pz,vx,vy,vz,roll,pitch,yaw)
        u = vertcat(roll_c,pitch_c,yaw_c,thrust)

        cpitch = cos(pitch)
        spitch = sin(pitch)
        croll = cos(roll)
        sroll = sin(roll)
        cyaw = cos(yaw)
        syaw = sin(yaw)
        # Define rotation matrix from quadrotor body to inertial reference frames
        Rotm = vertcat(
            horzcat(cpitch*cyaw, sroll*spitch*cyaw - croll*syaw, croll*spitch*cyaw + sroll*syaw),
            horzcat(cpitch*syaw, sroll*spitch*syaw + croll*cyaw, croll*spitch*syaw - sroll*cyaw),
            horzcat(-spitch, sroll*cpitch, croll*cpitch)
        )
        
        f_vec = vertcat(0., 0., thrust)

        # velocity dynamics
        vdot = vertcat(0.,0.,-self.gravity) + Rotm @ f_vec / self.mass

        # Setup explicit ode equations
        pxdot = vx
        pydot = vy
        pzdot = vz
        vxdot = vdot[0]
        vydot = vdot[1]
        vzdot = vdot[2]
        rolldot = (roll_c - roll) / self.tau
        pitchdot = (pitch_c - pitch) / self.tau
        yawdot = yaw_c # Commanded yaw is already yaw rate

        # vector function of explicit dynamics
        f_expl = vertcat(pxdot,pydot,pzdot,vxdot,vydot,vzdot,rolldot,pitchdot,yawdot)

        return (f_expl, x, u)