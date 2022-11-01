import numpy as np
from scipy.spatial.transform import Rotation as rot

class RobotSteer(object):

    # direction : forward, backward, move left, move right, turn left, turn right
    direction_flag = [0, 0, 0, 0, 0, 0]
    df_2_vc_index = [0, 0, 1, 1, 2, 2]
    df_2_vc_action = [1, -1, 1, -1, 1, -1]

    max_vel_cmd = np.array([1.5, 0.4, 3])
    vel_cmd_local = np.zeros(3)
    vel_cmd_local_filtered = np.zeros(3)
    vel_cmd_wcs = np.zeros(3)
    
    def __init__(self) -> None:
        pass

    def reset_vel_cmd(self):
        self.vel_cmd_local = np.zeros(3)
        self.vel_cmd_local_filtered = np.zeros(3)
        self.vel_cmd_wcs = np.zeros(3)

    def clear_direction_flag(self):
        self.direction_flag = [0, 0, 0, 0, 0, 0]

    def set_forward(self):
        self.direction_flag[0] = 1

    def set_backward(self):
        self.direction_flag[1] = 1

    def set_move_left(self):
        self.direction_flag[2] = 1

    def set_move_right(self):
        self.direction_flag[3] = 1

    def set_turn_left(self):
        self.direction_flag[4] = 1

    def set_turn_right(self):
        self.direction_flag[5] = 1
    
    def update_vel_cmd(self, body_orn):
        
        # update vel cmd
        for df in range(6):
            self.vel_cmd_local[self.df_2_vc_index[df]] += self.df_2_vc_action[df] * self.direction_flag[df] * 0.1
        
        self.vel_cmd_local = np.clip(self.vel_cmd_local, -self.max_vel_cmd, self.max_vel_cmd)
        self.vel_cmd_local *= 0.99 # if no key cmd, decay to zero

        # filter 
        phi = 0.02
        self.vel_cmd_local_filtered = (1. - phi) * self.vel_cmd_local_filtered + phi * self.vel_cmd_local

        # transform from body cs to wcs
        yaw = rot.from_quat(body_orn).as_euler('ZYX')[0]
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy],
                       [sy, cy]])
        
        self.vel_cmd_wcs[0:2] = Rz @ self.vel_cmd_local_filtered[0:2]
        self.vel_cmd_wcs[2] = self.vel_cmd_local_filtered[2]

    def get_vel_cmd_wcs(self):
        return self.vel_cmd_wcs

