import numpy as np
from enum import IntEnum
from scipy.spatial.transform import Rotation as rot

class DirectionFlag(IntEnum):
    """
        Definition of the direction flags: 
        [forward, backward, move left, move right, turn left, turn right]
    """
    FORWARD    = 0
    BACKWARD   = 1
    MOVE_LEFT  = 2
    MOVE_RIGHT = 3
    TURN_LEFT  = 4
    TURN_RIGHT = 5


class RobotSteer(object):
    """
        Robot steering controller.
        It converts direction flag commands to robot body velocity command in WCS.
    """
    # a flag indicating whether a direction is commanded
    direction_flag: list[int]   = [0, 0, 0, 0, 0, 0]    
    # the index of the value in the vel_cmd array on which a direction flag affects 
    df_2_vc_index:  list[int]   = [0, 0, 1, 1, 2, 2]     
    # the action of the value in the vel_cmd array on which a direction flag affects 
    # 1 means positive increasing, -1 means negative increasing
    df_2_vc_action: list[float] = [1, -1, 1, -1, 1, -1] 

    # velocity command is defined as [v_x, v_y, thetadot_z], representing a planar movement
    max_vel_cmd:            np.ndarray = np.array([1.5, 0.4, 2.5]) # maximum velocity
    vel_cmd_local:          np.ndarray = np.zeros(3)             # local velocity command
    vel_cmd_local_filtered: np.ndarray = np.zeros(3)             # filtered local velocity command
    vel_cmd_wcs:            np.ndarray = np.zeros(3)             # velocity command in WCS
    
    def __init__(self) -> None:
        """
            Create a robot steer instance.
        """
        pass


    def reset_vel_cmd(self):
        """
            Reset current velocity command to zero.
        """
        self.vel_cmd_local = np.zeros(3)
        self.vel_cmd_local_filtered = np.zeros(3)
        self.vel_cmd_wcs = np.zeros(3)

    def clear_direction_flag(self):
        """
            Clear all direction flags
        """
        self.direction_flag = [0, 0, 0, 0, 0, 0]

    
    def set_direction_flag(self, df:list[int]):
        """
            Set all direction flags
        """
        self.direction_flag = df.copy() 


    def set_forward(self):
        """
            Set move forward flag
        """
        self.direction_flag[DirectionFlag.FORWARD] = 1

    def set_backward(self):
        """
            Set move backward flag
        """
        self.direction_flag[DirectionFlag.BACKWARD] = 1

    def set_move_left(self):
        """
            Set move left flag
        """
        self.direction_flag[DirectionFlag.MOVE_LEFT] = 1

    def set_move_right(self):
        """
            Set move right flag
        """
        self.direction_flag[DirectionFlag.MOVE_RIGHT] = 1

    def set_turn_left(self):
        """
            Set turn left flag
        """
        self.direction_flag[DirectionFlag.TURN_LEFT] = 1

    def set_turn_right(self):
        """
            Set turn right flag
        """
        self.direction_flag[DirectionFlag.TURN_RIGHT] = 1
    
    def update_vel_cmd(self, body_orn: np.ndarray):
        """
            Update robot's velocity command according to current direction flag.

            Parameters:
                body_orn (array(4)): The current body orientation as quaternion in WCS.  
        """
        
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


    def get_vel_cmd_wcs(self) -> np.ndarray:
        """
            Get the current body velocity command in WCS.

            Returns:
                vel_cmd_wcs (array(3)): current body velocity command in WCS.
        """
        return self.vel_cmd_wcs

