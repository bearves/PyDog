import sys
import numpy as np
from scipy.spatial.transform import Rotation as rot

from controller import Supervisor
from controller import Motor, PositionSensor
from controller import Accelerometer, Gyro, InertialUnit
from controller import Node, Field
from controller import Keyboard, Joystick


class BridgeToWebots(object):

    n_leg: int
    n_jnt: int

    timestep: int
    dt: float
    leap: int
    max_trq: float

    robot: Supervisor

    motor_names: list[str]
    encoder_names: list[str]

    motors: list[Motor]
    encoders: list[PositionSensor]

    acc_sensor: Accelerometer
    gyro_sensor: Gyro
    imu_sensor: InertialUnit

    robot_node: Node
    translation_field: Field
    rotation_field: Field
    
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
    snsr_acc: np.ndarray
    snsr_gyro: np.ndarray
    snsr_pose: np.ndarray

    jnt_act_pos: np.ndarray
    jnt_act_vel: np.ndarray
    jnt_ref_trq: np.ndarray

    keyboard: Keyboard
    joystick: Joystick

    log_file : any

    def __init__(self) -> None:
        # create the Robot instance.
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt = 0.001 * self.timestep
        self.leap = int(0.025 / self.dt)
        self.n_leg = 4
        self.n_jnt = self.n_leg * 3
        self.motor_names = ['FR_hip_joint', 'FR_upper_joint', 'FR_lower_joint',
                            'FL_hip_joint', 'FL_upper_joint', 'FL_lower_joint',
                            'RR_hip_joint', 'RR_upper_joint', 'RR_lower_joint',
                            'RL_hip_joint', 'RL_upper_joint', 'RL_lower_joint']
        self.encoder_names = ['FR_hip_joint_sensor', 'FR_upper_joint_sensor', 'FR_lower_joint_sensor',
                              'FL_hip_joint_sensor', 'FL_upper_joint_sensor', 'FL_lower_joint_sensor',
                              'RR_hip_joint_sensor', 'RR_upper_joint_sensor', 'RR_lower_joint_sensor',
                              'RL_hip_joint_sensor', 'RL_upper_joint_sensor', 'RL_lower_joint_sensor']

        self.motors = []
        self.encoders = []

        for i in range(self.n_jnt):
            motor = self.robot.getDevice(self.motor_names[i])
            motor.setPosition(float('inf'))
            self.motors.append(motor)
            encoder = self.robot.getDevice(self.encoder_names[i])
            encoder.enable(self.timestep)
            self.encoders.append(encoder)
        
        self.acc_sensor = self.robot.getDevice('accelerometer')
        self.acc_sensor.enable(self.timestep)
        self.gyro_sensor = self.robot.getDevice('gyro')
        self.gyro_sensor.enable(self.timestep)
        self.imu_sensor = self.robot.getDevice('imu')
        self.imu_sensor.enable(self.timestep)
        
        self.robot_node = self.robot.getFromDef('DogA1')
        self.translation_field = self.robot_node.getField('translation')
        self.rotation_field = self.robot_node.getField('rotation')

        self.body_init_pos = np.array([0, 0, 0.335])
        self.body_init_axis_angle = np.array([0, 0, 1, 0])
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

        self.snsr_acc = np.zeros(3)
        self.snsr_gyro = np.zeros(3)
        self.snsr_pose = np.array([0, 0, 0, 1])

        self.jnt_act_pos = np.zeros(self.n_jnt)
        self.jnt_act_vel = np.zeros(self.n_jnt)
        self.jnt_ref_trq = np.zeros(self.n_jnt)

        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.timestep)

        self.joystick = self.robot.getJoystick()
        self.joystick.enable(self.timestep)

        self.log_file = open("log.csv", 'w')

    
    def reset_initial_position(self):
        # Set robot body to init position and orientation via supervisor mode
        self.rotation_field.setSFRotation(self.body_init_axis_angle.tolist()) # In webots, the rotation uses Axis-Angle form: [x, y, z, angle]
        self.translation_field.setSFVec3f(self.body_init_pos.tolist())
        # Set robot joints to init position via supervisor mode
        for i in range(self.n_jnt):
            jnt_node = self.robot.getFromDevice(self.motors[i]).getParentNode()
            jnt_node.setJointPosition(self.jnt_init_pos[i])
            self.jnt_act_pos = self.jnt_init_pos.copy()

    
    def read_sensor_feedbacks(self, cnt: int):
        # set time count
        self.cnt = cnt
        # get body pos and orientation in WCS
        self.body_act_pos = np.array(self.robot_node.getPosition())
        self.body_act_rm = np.array(self.robot_node.getOrientation()).reshape((3,3))
        self.body_act_orn = rot.from_matrix(self.body_act_rm).as_quat()
        if self.body_act_orn[3] < 0:
            self.body_act_orn *= -1
        #self.body_act_orn += np.random.normal(0, 0.02, (4))
        #self.body_act_orn /= np.linalg.norm(self.body_act_orn)
        # get body linear and angular velocity in WCS
        vel = self.robot_node.getVelocity()
        self.body_act_vel = np.array(vel[0:3]) # + np.random.normal(0, 0.1, (3))
        self.body_act_angvel = np.array(vel[3:6]) + np.random.normal(0, 0.014/57.3/np.sqrt(self.dt), (3))
        
        # get sensor acceleration in local measurement frame (LMF)
        self.snsr_acc = np.array(self.acc_sensor.getValues())
        # get sensor angular velocity in LMF
        self.snsr_gyro = np.array(self.gyro_sensor.getValues())
        # get sensor orientation in WCS (LMF -> WCS)
        self.snsr_pose = np.array(self.imu_sensor.getQuaternion())

        # read joint positions and calculate velocities
        last_pos = self.jnt_act_pos.copy()
        for i in range(self.n_jnt):
            self.jnt_act_pos[i] = self.encoders[i].getValue()
        self.jnt_act_vel = (self.jnt_act_pos - last_pos)/self.dt

    
    def read_user_inputs(self):
        return self.keyboard.getKey()

    
    def set_motor_command(self, jnt_ref_trq: np.ndarray):
        self.max_trq = 50
        self.jnt_ref_trq = np.clip(jnt_ref_trq, -self.max_trq, self.max_trq)
        for i in range(self.n_jnt):
            self.motors[i].setTorque(self.jnt_ref_trq[i])

    
    def step(self) -> int:
        return self.robot.step(self.timestep)

    
    def save_data(self, support_state, support_phase):
        # concat all data
        data = np.zeros(1 + 3 + 3 + 4 + 12 + 12 + 12 + 4 + 4 + 3 + 3 + 4 + 3)
        data[0]     = self.cnt
        data[1:4]   = self.snsr_acc
        data[4:7]   = self.snsr_gyro
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


    def finalize(self):
        self.log_file.close()