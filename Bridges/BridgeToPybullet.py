import sys
import numpy as np
from scipy.spatial.transform import Rotation as rot

import pybullet as pb
import pybullet_data as pb_data


class BridgeToPybullet(object):

    n_leg: int
    n_jnt: int

    timestep: int
    dt: float
    leap: int
    max_trq: float

    robot: any
    ground: any
    keys: any

    act_jnt_id: list[int]
    
    body_init_pos : list
    body_init_axis_angle: list

    jnt_init_pos: np.ndarray
    jnt_init_vel: np.ndarray

    cnt: int # time count

    body_act_pos: np.ndarray
    body_act_orn: np.ndarray
    body_act_rm : np.ndarray

    body_act_vel: np.ndarray
    body_act_angvel: np.ndarray

    # measurement from ARHS sensor
    Rs_bcs: np.ndarray # rotation matrix of sensor frame w.r.t. Body CS
    snsr_acc: np.ndarray
    snsr_gyr: np.ndarray
    snsr_pose: np.ndarray

    jnt_act_pos: np.ndarray
    jnt_act_vel: np.ndarray
    jnt_ref_trq: np.ndarray

    log_file : any

    def __init__(self, timestep: int, urdf_path: str) -> None:
        # create the Robot instance.
        self.timestep = timestep
        self.dt = 0.001 * self.timestep
        self.leap = int(0.025 / self.dt)
        self.n_leg = 4
        self.n_jnt = self.n_leg * 3
        self.drv_jnt_id = [1,  3,  4,  # FR
                           6,  8,  9,  # FL
                          11, 13, 14,  # RR
                          16, 18, 19]  # RL

        self.body_init_pos = np.array([0, 0, 0.45])
        self.body_init_orn = np.array([0, 0, 0, 1])
        self.jnt_init_pos = np.array([-0.0, 0.75, -1.35,
                                      -0.0, 0.75, -1.35,
                                      -0.0, 0.75, -1.35,
                                      -0.0, 0.75, -1.35])
        self.jnt_init_vel = np.zeros(self.n_jnt)

        self.body_act_pos = self.body_init_pos
        self.body_act_orn = np.array([0, 0, 0, 1])
        self.body_act_rm = np.eye(3)

        self.body_act_vel = np.zeros(3)
        self.body_act_angvel = np.zeros(3)
        self.last_body_vel = np.zeros(3)

        self.Rs_bcs = np.eye(3)
        self.snsr_acc = np.zeros(3)
        self.snsr_gyr = np.zeros(3)
        self.snsr_pose = np.array([0, 0, 0, 1])

        self.jnt_act_pos = np.zeros(self.n_jnt)
        self.jnt_act_vel = np.zeros(self.n_jnt)
        self.jnt_ref_trq = np.zeros(self.n_jnt)

        self.log_file = open("log.csv", 'w')

        # start simulator
        pb.connect(pb.GUI)
        pb.resetDebugVisualizerCamera(1, 180, -12, [0.0, -0.0, 0.2])
        pb.setTimeStep(self.dt)
        pb.setAdditionalSearchPath(pb_data.getDataPath())

        # initialize the simulator parameters, GUI and others
        self.robot = pb.loadURDF(urdf_path, self.body_init_pos, \
                                 self.body_init_orn, useFixedBase=False)
        self.ground = pb.loadURDF("plane.urdf", useFixedBase=True)
        pb.setGravity(0, 0, -9.8)

        # set joint control mode for bullet model 
        jnt_num = pb.getNumJoints(self.robot)
        for i in range(jnt_num):
            info = pb.getJointInfo(self.robot, i)
            joint_type = info[2]
            # unlock joints
            if joint_type == pb.JOINT_REVOLUTE:
                pb.setJointMotorControl2(self.robot, i, pb.VELOCITY_CONTROL, force=0)

        # add a constraint to prevent the robot from flying away after loading
        # the constraint will be removed once the gait controller starts
        self.init_constraint = pb.createConstraint(self.ground, -1,  self.robot, -1, pb.JOINT_PRISMATIC, [0,0,1], [0, 0, 0], [0,0,0])


    def remove_init_constraint(self):
        pb.removeConstraint(self.init_constraint)

    
    def read_sensor_feedbacks(self, cnt: int):
        # set time count
        self.cnt = cnt
        # get robot body states
        base_pos = pb.getBasePositionAndOrientation(self.robot)
        base_vel = pb.getBaseVelocity(self.robot)
        self.body_act_pos = np.array(base_pos[0])
        self.body_act_rm = rot.from_quat(base_pos[1]).as_matrix()
        self.body_act_orn = np.array(base_pos[1])
        if self.body_act_orn[3] < 0:
            self.body_act_orn *= -1
        self.body_act_vel = np.array(base_vel[0])
        self.body_act_angvel = np.array(base_vel[1])

        # simulate accelerometer and gyro sensor
        body_acc_wcs = (self.body_act_vel - self.last_body_vel)/self.dt + [0, 0, 9.8]
        self.last_body_vel = self.body_act_vel
        body_gyr_wcs = self.body_act_angvel
        self.snsr_acc = self.Rs_bcs.T @ self.body_act_rm.T @ body_acc_wcs
        self.snsr_gyr = self.Rs_bcs.T @ self.body_act_rm.T @ body_gyr_wcs

        # read joint positions and calculate velocities
        for i in range(self.n_jnt):
            jnt_state = pb.getJointState(self.robot, self.drv_jnt_id[i])
            self.jnt_act_pos[i] = jnt_state[0]
            self.jnt_act_vel[i] = jnt_state[1]


    def read_user_inputs(self):
        self.keys = pb.getKeyboardEvents()
        return self.keys

    
    def set_motor_command(self, jnt_ref_trq: np.ndarray):
        self.max_trq = 50
        self.jnt_ref_trq = np.clip(jnt_ref_trq, -self.max_trq, self.max_trq)
        for i in range(self.n_jnt):
            pb.setJointMotorControl2(
                self.robot, self.drv_jnt_id[i], pb.TORQUE_CONTROL, force=jnt_ref_trq[i])

    
    def step(self):
        pb.stepSimulation()

    
    def track_robot(self):
        cam_pos = self.body_act_pos.copy()
        cam_pos[2] = 0.2
        pb.resetDebugVisualizerCamera(1, 180, -30, cam_pos)

    
    def save_data(self, support_state, support_phase):
        # concat all data
        data = np.zeros(1 + 3 + 3 + 4 + 12 + 12 + 12 + 4 + 4 + 3 + 3 + 4 + 3)
        data[0]     = self.cnt
        data[1:4]   = self.snsr_acc
        data[4:7]   = self.snsr_gyr
        data[7:11]  = self.snsr_pose
        data[11:23] = self.jnt_act_pos
        data[23:35] = self.jnt_act_vel
        data[35:47] = self.jnt_ref_trq
        data[47:51] = support_state
        data[51:55] = support_phase
        data[55:58] = self.body_act_pos
        data[58:61] = self.body_act_vel
        data[61:65] = self.body_act_orn
        data[65:68] = self.body_act_angvel

        data_str = np.array2string(data, precision=8, separator=',')
        data_str = data_str.strip('[]').replace('\n','') + '\n'
        self.log_file.write(data_str)


    def is_key_down(self, key: int):
        return (key in self.keys) and self.keys[key] & pb.KEY_IS_DOWN

    def is_key_trigger(self, key: int):
        return (key in self.keys) and self.keys[key] & pb.KEY_WAS_TRIGGERED

    def finalize(self):
        self.log_file.close()