import numpy as np
import math
from scipy.spatial.transform import Rotation as rot


class BodyTrjPlanner(object):
    """
        Body center trajectory planner.
    """

    ref_body_pos = np.zeros(3)
    ref_body_orn = np.array([0, 0, 0, 1])
    ref_body_vel = np.zeros(3)
    ref_body_angvel = np.zeros(3)

    vel_cmd = np.zeros(3)

    dt = 1.0/1000.

    def __init__(self, dt: float):
        self.dt = dt

    def set_ref_state(self,
                      body_pos: np.ndarray,
                      body_orn: np.ndarray,
                      body_vel: np.ndarray,
                      body_angvel: np.ndarray):
        self.ref_body_pos = body_pos
        self.ref_body_orn = body_orn
        self.ref_body_vel = body_vel
        self.ref_body_angvel = body_angvel


    def update_ref_state(self, vel_cmd_wcs: np.ndarray):
        self.vel_cmd = vel_cmd_wcs.copy()
        
        self.ref_body_vel[0:2] = self.vel_cmd[0:2]
        self.ref_body_vel[2] = 0 # currently, robot height is invariant
        self.ref_body_pos[0:2] += self.ref_body_vel[0:2] * self.dt # update next position

        self.ref_body_angvel[0:2] = np.zeros(2)
        self.ref_body_angvel[2] = self.vel_cmd[2]
        wx, wy, wz = self.ref_body_angvel[0], self.ref_body_angvel[1], self.ref_body_angvel[2]
        mat_omega = np.array([[  0,  wz, -wy, wx],
                              [-wz,   0,  wx, wy],
                              [ wy, -wx,   0, wz],
                              [-wx, -wy, -wz,  0]])  # quaternion integration
        ref_body_orn_dot = 0.5 * mat_omega @ self.ref_body_orn 
        self.ref_body_orn += ref_body_orn_dot * self.dt 
        self.ref_body_orn /= np.linalg.norm(self.ref_body_orn) # normalize


    def predict_future_body_ref_traj(self,
                                     dt_mpc: float,
                                     horizon_length: int) -> np.ndarray:

        # get body euler angle from quaternion
        body_euler_ypr = rot.from_quat(self.ref_body_orn).as_euler('ZYX')
        body_euler_rpy = np.flip(body_euler_ypr)
        yawdot = self.ref_body_angvel[2]

        # body state q = [theta p w v g]' \in R(13)
        future_body_traj = np.zeros((horizon_length, 13))
        for i in range(horizon_length):
            # theta ( roll, pitch, yaw )
            future_body_traj[i, 0:3] = [0, 0, body_euler_rpy[2] + yawdot * dt_mpc * i]
            future_body_traj[i, 3:6] = self.ref_body_pos + self.ref_body_vel * dt_mpc * i  # x, y, z
            future_body_traj[i, 6:9] = [0, 0, yawdot]  # wx, wy, wz
            future_body_traj[i, 9:12] = self.ref_body_vel  # vx, vy, vz
            future_body_traj[i, 12] = -9.81      # g constant

        return future_body_traj


class FootholdPlanner(object):

    n_leg = 4
    hip_pos_wcs_next_td = np.zeros(n_leg * 3)

    current_footholds = np.zeros(n_leg * 3)
    next_footholds = np.zeros(n_leg * 3)

    current_leg_pos = np.zeros(n_leg * 3)
    liftup_leg_pos = np.zeros(n_leg * 3)

    current_support_state = np.ones(n_leg)
    last_support_state = np.ones(n_leg)

    def __init__(self):
        pass

    def init_footholds(self, r_leg: np.ndarray):
        self.current_footholds = r_leg.copy()
        self.next_footholds = r_leg.copy()
        self.current_leg_pos = r_leg.copy()
        self.liftup_leg_pos = r_leg.copy()

    def set_current_state(self, r_leg: np.ndarray, support_state: np.ndarray):
        self.current_leg_pos = r_leg.copy()
        self.last_support_state = self.current_support_state.copy()
        self.current_support_state = support_state.copy()

        # detect td event and lift up event
        for i in range(self.n_leg):
            # if td event, update current footholds use current leg pos at td moment
            if self.last_support_state[i] == 0 and self.current_support_state[i] == 1:
                self.current_footholds[0+i*3:3+i *
                                       3] = self.current_leg_pos[0+i*3:3+i*3]
            # if lft event, update liftup footpos use current leg pos at liftup moment
            if self.last_support_state[i] == 1 and self.current_support_state[i] == 0:
                self.liftup_leg_pos[0+i*3:3+i *
                                    3] = self.current_leg_pos[0+i*3:3+i*3]

    def calculate_next_footholds(self,
                                 body_pos_now: np.ndarray,
                                 body_orn_now: np.ndarray,
                                 body_vel_now: np.ndarray,
                                 body_des_vel_cmd: np.ndarray,
                                 t_swing_left: np.ndarray,
                                 t_stance_all: np.ndarray,
                                 hip_pos_wrt_body: np.ndarray):

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
            self.hip_pos_wcs_next_td[0+leg*3:3+leg *
                                     3] = body_pos_next_td + Rz_body_next @ hip_pos_prj
            # thirdly, predict next foothold based on Raibert's law
            #
            #  p_next = p_h + Tst/2 * v_now + k * (v_des - v_now)
            #
            self.next_footholds[0+leg*3:2+leg*3] = self.hip_pos_wcs_next_td[0+leg*3:2+leg*3] + \
                t_stance_all[leg] * 0.5 * body_vel_now[0:2] + \
                k * (body_vel_now[0:2] - body_des_vel_cmd[0:2])
            # slightly below ground, considering the foot size
            self.next_footholds[2+leg*3] = 0.0199

    def predict_future_foot_pos(self,
                                time_step: float,
                                horizon_length: int,
                                future_support_state: np.ndarray) -> np.ndarray:

        future_r_leg_traj = np.zeros((horizon_length, self.n_leg * 3))

        use_next_foothold_flag = np.zeros(self.n_leg)

        for i in range(horizon_length):
            for leg in range(self.n_leg):
                # once the leg enters the swing phase, use the estimated next foothold as the future footpos
                if future_support_state[i, leg] == 0:
                    use_next_foothold_flag[leg] = 1

                if use_next_foothold_flag[leg]:
                    # use next estimated footholds
                    future_r_leg_traj[i, 0+leg*3:3+leg *
                                      3] = self.next_footholds[0+leg*3:3+leg*3]
                else:
                    # use current leg pos
                    future_r_leg_traj[i, 0+leg*3:3+leg *
                                      3] = self.current_leg_pos[0+leg*3:3+leg*3]

        return future_r_leg_traj

    def get_current_footholds(self, leg):
        return self.current_footholds[0+leg*3:3+leg*3]

    def get_next_footholds(self, leg):
        return self.next_footholds[0+leg*3:3+leg*3]

    def get_liftup_leg_pos(self, leg):
        return self.liftup_leg_pos[0+leg*3:3+leg*3]


class SwingTrjPlanner(object):

    start_point: np.zeros(3)
    end_point: np.zeros(3)
    step_height: float = 0.08

    def __init__(self) -> None:
        pass

    def set_start_point(self, sp: np.ndarray):
        self.start_point = sp

    def set_end_point(self, ep: np.ndarray):
        self.end_point = ep

    def get_tip_pos_vel(self, time_ratio: float, time_ratio_dot: float) -> tuple[np.ndarray, np.ndarray]:
        pivot = 0.5 * (1 - math.cos(math.pi * time_ratio))
        tip_pos = (1 - pivot) * self.start_point + (pivot) * self.end_point
        # add height curve
        tip_pos[2] += self.step_height * math.sin(math.pi * pivot)

        pivotdot = 0.5 * math.sin(math.pi * time_ratio) * \
            math.pi * time_ratio_dot
        tip_vel = (self.end_point - self.start_point) * pivotdot
        tip_vel[2] += self.step_height * \
            math.cos(math.pi * pivot) * math.pi * pivotdot
        return tip_pos, tip_vel
