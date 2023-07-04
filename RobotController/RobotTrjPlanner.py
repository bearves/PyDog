import numpy as np
import math
from scipy.spatial.transform import Rotation as rot
from RobotController import RobotCurves


class BodyTrjPlanner(object):
    """
        Body center reference trajectory planner.
    """

    # Planner parameter
    dt: float = 1.0/1000.

    # Reference body states
    ref_body_pos:    np.ndarray = np.zeros(3)
    ref_body_orn:    np.ndarray = np.array([0, 0, 0, 1])
    ref_body_vel:    np.ndarray = np.zeros(3)
    ref_body_angvel: np.ndarray = np.zeros(3)
    ref_body_height: float
    ref_pitch: float = 0
    ref_roll: float = 0
    
    filter_k: float  # the 1st order filter coefficient for terrain roll and pitch filtering

    # ground plane parameters
    # the ground plane is described as 
    #    ax + by + cz + d = 0
    # where n = (a,b,c) is the normal vector, |n| = 1
    ground_normal: np.ndarray = np.array([0, 0, 1])
    ground_d: float           = 0

    # Velocity command
    vel_cmd: np.ndarray = np.zeros(3)

    # useful constants
    vec_unit_z: np.ndarray = np.array([0, 0, 1])


    def __init__(self, dt: float):
        """
            Create and initialize a body trajectory planner.

            Parameters:
                dt (float): time step of the simulator.
        """
        self.dt = dt


    def set_ref_state(self,
                      body_pos: np.ndarray,
                      body_orn: np.ndarray,
                      body_vel: np.ndarray,
                      body_angvel: np.ndarray,
                      ):
        """
            Set reference states of the body.

            Parameters:
                body_pos    (array(3)): body position in WCS.
                body_orn    (array(4)): body orientation as quaternion in WCS.
                body_vel    (array(3)): body linear velocity in WCS.
                body_angvel (array(3)): body angular velocity in WCS.
        """
        self.ref_body_pos = body_pos.copy()
        self.ref_body_orn = body_orn.copy()
        self.ref_body_vel = body_vel.copy()
        self.ref_body_angvel = body_angvel.copy()
        self.ref_body_height = 0.3
        self.ref_pitch = 0
        self.ref_roll = 0
        self.filter_k = 0.7


    def update_ref_state(self, 
                         vel_cmd_wcs: np.ndarray, 
                         act_body_pos: np.ndarray, 
                         act_body_orn: np.ndarray,
                         ground_plane: tuple[np.ndarray, float]):
        """
            Integrate body reference states according to the received velocity command.

            Parameters:
                vel_cmd_wcs  (array(3)): velocity command in WCS, defined as [v_x, v_y, thetadot_z].
                act_body_pos (array(3)): current body position in WCS.
                act_body_orn (array(4)): current body orientation as quaternion in WCS. 
                ground_plane (tuple(array(3), float)): the normal and d parameter of current ground plane. 
        """
        # update velocity command
        self.vel_cmd = vel_cmd_wcs.copy()

        # update ground
        self.ground_normal = ground_plane[0].copy()
        self.ground_d = ground_plane[1]
        
        # Set horizontal ref vel
        self.ref_body_vel[0:2] = self.vel_cmd[0:2]
        self.ref_body_vel[2] = 0
        vel_mag = np.linalg.norm(self.ref_body_vel)
        # turn the ref vel parallel with the ground 
        if vel_mag > 1e-2:
            Rv = plane_rotation(self.ground_normal, self.ref_body_vel)
            self.ref_body_vel = vel_mag * Rv[:, 0]

        # correct drift in X and Y
        self.ref_body_pos[0:2] = act_body_pos[0:2]
        # correct drift in Z, hold ref body height according to the ground plane
        proj_pt, _ = plane_projection(self.ground_normal, self.ground_d, self.ref_body_pos, self.vec_unit_z)
        ideal_body_pos = proj_pt + self.ref_body_height * self.vec_unit_z
        # filter body pos due to possible jumps of the terrain height estimation
        self.ref_body_pos = self.filter_k * self.ref_body_pos + (1 - self.filter_k) * ideal_body_pos

        self.ref_body_angvel[0:2] = np.zeros(2) # currently, always let rx and ry be zero.
        self.ref_body_angvel[2] = self.vel_cmd[2]

        self.set_ref_body_orn(act_body_orn)
    

    def set_ref_body_orn(self, act_body_orn: np.ndarray):
        """
            Set body's reference orientation according to the estimated terrain.

            act_body_orn (array(4)): current body orientation as quaternion in WCS. 
        """
        # correct drift in Yaw
        yaw = rot.from_quat(act_body_orn).as_euler('ZYX')[0]
        x0 = np.array([math.cos(yaw), math.sin(yaw), 0])
        # get pitch and roll from terrain slope estimation 
        R = plane_rotation(self.ground_normal, x0)
        euler_ypr = rot.from_matrix(R).as_euler('ZYX')
        # filter roll and pitch due to possible jumps of terrain slope estimation
        self.ref_roll = self.filter_k * self.ref_roll + (1 - self.filter_k) * euler_ypr[2]
        self.ref_pitch = self.filter_k * self.ref_pitch + (1 - self.filter_k) * euler_ypr[1]
        self.ref_body_orn = rot.from_euler('ZYX', [yaw, self.ref_pitch, self.ref_roll]).as_quat()


    def predict_future_body_ref_traj(self,
                                     dt_mpc: float,
                                     horizon_length: int) -> np.ndarray:
        """
            Predict body reference states within the MPC's prediction horizon.

            Parameters:
                dt_mpc  (float):       The MPC solving time interval.
                horizon_length (int):  prediction horizon length of the MPC.
        """

        # get body euler angle from quaternion
        body_ypr = rot.from_quat(self.ref_body_orn).as_euler('ZYX')
        body_rpy = np.flip(body_ypr)
        yawdot = self.ref_body_angvel[2]

        # body state q = [theta p w v g]' \in R(13)
        future_body_traj = np.zeros((horizon_length, 13))
        for i in range(horizon_length):
            # theta ( roll, pitch, yaw )
            future_body_traj[i, 0:3] = [body_rpy[0], body_rpy[1], body_rpy[2] + yawdot * dt_mpc * i]
            future_body_traj[i, 3:6] = self.ref_body_pos + self.ref_body_vel * dt_mpc * i  # x, y, z
            future_body_traj[i, 6:9] = [0, 0, yawdot]  # wx, wy, wz
            future_body_traj[i, 9:12] = self.ref_body_vel  # vx, vy, vz
            future_body_traj[i, 12] = -9.81      # g constant

        return future_body_traj


class FootholdPlanner(object):
    """
        Foothold planner to predict the next foothold position for balance control and continuous locomotion.
    """

    # Planner parameters
    n_leg: int = 4 # number of legs

    # Place holders
    hip_pos_wcs_next_td:   np.ndarray = np.zeros(n_leg * 3) # predicted offset hip position at next touchdown moment in WCS

    current_footholds:     np.ndarray = np.zeros(n_leg * 3) # current footholds in WCS
    next_footholds:        np.ndarray = np.zeros(n_leg * 3) # next footholds in WCS

    current_leg_pos:       np.ndarray = np.zeros(n_leg * 3) # current leg tip position in WCS
    liftup_leg_pos:        np.ndarray = np.zeros(n_leg * 3) # leg tip position at lift up moment in WCS

    current_support_state: np.ndarray = np.ones(n_leg)      # current support states
    # support state at last time step, this is used for detecting liftup and touchdown events
    last_support_state:    np.ndarray = np.ones(n_leg)      

    # ground plane's normal and d parameter
    # the ground plane is described as 
    #    ax + by + cz + d = 0
    # where n = (a,b,c) is the normal vector, |n| = 1
    ground_normal: np.ndarray = np.array([0, 0, 1])
    ground_d: float           = 0


    def __init__(self):
        """
            Create a foothold planner
        """
        pass


    def init_footholds(self, r_leg: np.ndarray):
        """
            Initialize footholds.

            Parameters:
                r_leg (array(n_leg * 3)): leg tip position in WCS.
        """
        self.current_footholds = r_leg.copy()
        self.next_footholds = r_leg.copy()
        self.current_leg_pos = r_leg.copy()
        self.liftup_leg_pos = r_leg.copy()


    def set_current_state(self, 
                          r_leg: np.ndarray, 
                          support_state: np.ndarray,
                          ground_plane: tuple[np.ndarray, float]):
        """
            Update current states for foothold planner.

            Parameters:
                r_leg (array(n_leg * 3)): current leg tip position in WCS.
                support_state (array(n_leg)): current support state.
                ground_plane (tuple(array(3), float)): the normal and d parameter of current ground plane. 
        """
        self.current_leg_pos = r_leg.copy()
        self.last_support_state = self.current_support_state.copy()
        self.current_support_state = support_state.copy()

        # detect td event and lift up event
        for i in range(self.n_leg):
            # if td event, update current footholds use current leg pos at td moment
            if self.last_support_state[i] == 0 and self.current_support_state[i] == 1:
                self.current_footholds[0+i*3:3+i*3] = self.current_leg_pos[0+i*3:3+i*3]

            # if lft event, update liftup footpos use current leg pos at liftup moment
            if self.last_support_state[i] == 1 and self.current_support_state[i] == 0:
                self.liftup_leg_pos[0+i*3:3+i*3] = self.current_leg_pos[0+i*3:3+i*3]
        
        # update ground parameters
        self.ground_normal = ground_plane[0].copy()
        self.ground_d = ground_plane[1]


    def calculate_next_footholds(self,
                                 body_pos_now: np.ndarray,
                                 body_orn_now: np.ndarray,
                                 body_vel_now: np.ndarray,
                                 body_des_vel_cmd: np.ndarray,
                                 t_swing_left: np.ndarray,
                                 t_stance_all: np.ndarray,
                                 hip_pos_wrt_body: np.ndarray
                                 ):
        """
            Calculate next foothold according to current robot states.

            Parameters:
                body_pos_now (array(3)): current body position in WCS.
                body_orn_now (array(4)): current body orientation as quaternion in WCS.
                body_vel_now (array(3)): current body linear velocity in WCS.
                body_des_vel_cmd (array(3)): current desired velocity command, defined as [v_x v_y thetadot_z] in WCS.
                t_swing_left: (array(n_leg)): the time left for swing of all legs.
                t_stance_all: (array(n_leg)): total time of stance of all legs.
                hip_pos_wrt_body: (array(n_leg*3)): offset hip position in Body CS.
        """
        for leg in range(self.n_leg):
            k = 0.03
            # firstly, predict body position and orientation at the next touchdown moment
            # note: currently, assume footholds are in X-Y-yaw plane
            body_euler_now = rot.from_quat(body_orn_now).as_euler('ZYX')
            body_yaw_now = body_euler_now[0]
            body_des_vel = np.zeros(3)
            body_des_vel[0:2] = body_des_vel_cmd[0:2]
            body_des_yawdot = body_des_vel_cmd[2]
            body_pos_next_td = body_pos_now + t_swing_left[leg] * body_des_vel
            body_yaw_next_td = body_yaw_now + \
                t_swing_left[leg] * body_des_yawdot
            Rz_body_next = rot.from_rotvec(
                [0, 0, body_yaw_next_td]).as_matrix()
            # secondly, predict hip position in wcs at the next touchdown moment
            hip_pos_prj = hip_pos_wrt_body[0+leg*3:3+leg*3]
            hip_pos_prj[0] -= 0.029  # x compensation due to weight balance
            self.hip_pos_wcs_next_td[0+leg*3:3+leg*3] = body_pos_next_td + Rz_body_next @ hip_pos_prj

            # thirdly, calculate body relative height and v_now x omega_des_z
            vxw = np.array([body_vel_now[1], -body_vel_now[0]]) * body_des_yawdot # v_now x omega_des_z
            bh = body_pos_now[2] - np.mean(self.current_footholds[2::3])

            # fourthly, predict next foothold based on Raibert's law, with centrifugal compensation
            #
            #  p_next = p_h + Tst/2*v_now + k*(v_des-v_now) + 1/2*(bh/g)*(v_now x omega_des_z)
            #
            self.next_footholds[0+leg*3:2+leg*3] = \
                self.hip_pos_wcs_next_td[0+leg*3:2+leg*3] + \
                t_stance_all[leg] * 0.5 * body_vel_now[0:2] + \
                k * (body_vel_now[0:2] - body_des_vel_cmd[0:2]) + \
                0.5 * bh/9.81 * vxw

            # finally, calculate the projection point of the next foothold on the ground plane
            ground_point, _ = plane_projection(self.ground_normal,
                                               self.ground_d,
                                               self.next_footholds[0+leg*3:3+leg*3],
                                               [0, 0, -1])
            # slightly below ground, considering the foot size
            self.next_footholds[2+leg*3] = ground_point[2] - 0.004


    def predict_future_foot_pos(self,
                                dt_mpc: float,
                                horizon_length: int,
                                future_support_state: np.ndarray) -> np.ndarray:
        """
            Predict future foot position in WCS within the MPC's prediction horizon.
            In MPC, the predicted foot position is only functional for supporting legs.
            Thus here the swinging foot position is not predicted for simplicity and computational performance. 

            Parameters:
                dt_mpc  (float):      The MPC solving time interval.
                horizon_length (int): Prediction horizon length of the MPC.
                future_support_state (array(hz_len, n_leg)):  
                    Future support state within the MPC's prediction horizon. 
            
            Returns:
                future_r_leg_traj (array(hz_len, n_leg*3)): future foot position in WCS.
        """
        future_r_leg_traj = np.zeros((horizon_length, self.n_leg * 3))

        use_next_foothold_flag = np.zeros(self.n_leg)

        for i in range(horizon_length):
            for leg in range(self.n_leg):
                # once the leg enters the swing phase, use the estimated next foothold as the future footpos
                if future_support_state[i, leg] == 0:
                    use_next_foothold_flag[leg] = 1

                if use_next_foothold_flag[leg]:
                    # use next estimated footholds
                    future_r_leg_traj[i, 0+leg*3:3+leg*3] = self.next_footholds[0+leg*3:3+leg*3]
                else:
                    # use current leg pos
                    future_r_leg_traj[i, 0+leg*3:3+leg*3] = self.current_leg_pos[0+leg*3:3+leg*3]

        return future_r_leg_traj


    def get_current_footholds(self, leg: int):
        """
            Get the current foothold of the given leg.

            Parameters:
                leg (int): the index of the leg.

            Returns:
                foothold (array(3)): the current foothold of the given leg.
        """
        return self.current_footholds[0+leg*3:3+leg*3]


    def get_next_footholds(self, leg: int):
        """
            Get the next predicted foothold of the given leg.

            Parameters:
                leg (int): the index of the leg.

            Returns:
                foothold (array(3)): the next predicted of the given leg.
        """
        return self.next_footholds[0+leg*3:3+leg*3]


    def get_liftup_leg_pos(self, leg: int):
        """
            Get the lift up position of the given leg.

            Parameters:
                leg (int): the index of the leg.

            Returns:
                leg_pos (array(3)): the lift up position of the given leg.
        """
        return self.liftup_leg_pos[0+leg*3:3+leg*3]


class SwingTrjPlanner(object):
    """
        Swing trajectory planner for a single leg.
        It generates a smooth curve, interpolating between the lift up position
         and the next foothold position and to keep a clearance above the ground.
    """
    start_point: np.ndarray = np.zeros(3)  # start point of the swing trajectory
    end_point:   np.ndarray = np.zeros(3)  # end point of the swing trajectory
    step_height: float = 0.07              # step height of the swing trajectory

    kp_builder:  RobotCurves.BezierKeyPointBuilder  # key point builder for bezier curve generator
    bezier_crv:  RobotCurves.BezierCurve            # bezier curve generator
    sine_crv:    RobotCurves.SineCurve              # sine curve generator

    def __init__(self) -> None:
        """
            Create a Swing trajectory planner
        """
        self.kp_builder = RobotCurves.BezierKeyPointBuilder()
        self.bezier_crv = RobotCurves.BezierCurve()
        self.sine_crv   = RobotCurves.SineCurve()


    def set_start_point(self, sp: np.ndarray):
        """
            Set trajectory start point.

            Parameters:
                sp (array(3)): start point.
        """
        self.start_point = sp


    def set_end_point(self, ep: np.ndarray):
        """
            Set trajectory end point.

            Parameters:
                ep (array(3)): end point.
        """
        self.end_point = ep


    def get_tip_pos_vel(self, 
                        time_ratio: float, 
                        time_ratio_dot: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Get leg tip's position and velocity at given time ratio.

            Parameters:
                time_ratio (float): a scalar between 0.0 and 1.0, defined as 
                            time_ratio = (time_now - swing_start_time) / total_swing_time.
                time_ratio_dot (float): the time variation of time_ratio. defined as
                                time_ratio_dot = d(time_ratio)/dt

            Returns:
                tip_pos (array(3)): tip position at given time ratio.
                tip_vel (array(3)): tip velocity at given time ratio.
                tip_acc (array(3)): tip acceleration at given time ratio.
        """
        pivot = 0.5 * (1 - math.cos(math.pi * time_ratio))
        pivotdot = 0.5 * math.sin(math.pi * time_ratio) * math.pi * time_ratio_dot
        pivotddot = 0.5 * (math.pi**2) * math.cos(math.pi*time_ratio) * (time_ratio_dot**2)

        #self.sine_crv.set_key_points(self.start_point, self.end_point, self.step_height)
        #tip_pos, tip_vel, tip_acc = self.sine_crv.get_pva_at(pivot, pivotdot, pivotddot)
        key_pts = self.kp_builder.build_kp_normal(self.start_point, self.end_point, self.step_height, 1)
        self.bezier_crv.set_key_points(key_pts)
        tip_pos, tip_vel, tip_acc = self.bezier_crv.get_pva_at(pivot, pivotdot, pivotddot)

        return tip_pos, tip_vel, tip_acc


#####################################
#    Helper function
#####################################
def plane_projection(plane_n: np.ndarray,
                     plane_d: float,
                     point: np.ndarray, 
                     vector: np.ndarray) -> tuple[np.ndarray, float]:
    """
        Get the projection of a given point to a plane, alone a vector.
        The plane in the 3D space is described using the following equation:
            ax + by + cz + d = 0,
        where n = (a,b,c) is the plane's normal vector, |n| = 1.

        Parameters:
            plane_n (array(3)): normal of the plane.
            plane_d (float): d coefficient of the plane.
            point (array(3)): the given point.
            vector (array(3)): the vector of the projection direction.

        Returns:
            p_proj (array(3)): the point on the plane, projected by the given point.
            h (float): the distance from the given point to the projected point.
    """
    vector = vector / np.linalg.norm(vector)
    b = np.dot(plane_n, point) + plane_d
    h = -b / np.dot(plane_n, vector)
    p_proj = point + h * vector
    return (p_proj, h)


def plane_rotation(plane_normal: np.ndarray,
                   heading_vec: np.ndarray) -> np.ndarray:
    """
        Get the rotation matrix of the frame that is parallel to the ground plane, 
        given the frame's heading vector. The heading vector is the projected vector
        of the frame's x axis to the world's XY plane.

        Parameters:
            plane_normal (array(3)): normal of the plane.
            heading_vec  (array(3)): frame's heading vector.

        Returns:
            R    (array(3x3)): the frame's rotation matrix, w.r.t. WCS.
    """
    uz = plane_normal/np.linalg.norm(plane_normal)
    uy = np.cross(uz, heading_vec)
    uy = uy/np.linalg.norm(uy)
    ux = np.cross(uy, uz)
    R = np.vstack((ux, uy, uz)).T
    return R
