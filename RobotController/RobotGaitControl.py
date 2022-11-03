import os
import numpy as np

from RobotKinematics import RobotKineticModel
from RobotDynamics import RobotDynamicModel, JointOrderMapper
from GaitPatternGenerator import GaitPatternGenerator
from RobotTrjPlanner import BodyTrjPlanner, FootholdPlanner, SwingTrjPlanner
from RobotMPC import QuadConvexMPC
from RobotWBC import QuadSingleBodyWBC
from RobotWBIC import QuadWBIC
from RobotSteer import RobotSteer

class QuadControlInput(object):
    """
        Control inputs for quadruped robot
    """
    # constants
    n_leg: int = 4

    # current time
    time_now: float = 0

    # robot states
    # states can be obtained directly
    jnt_act_pos: np.ndarray = np.zeros(n_leg * 3)  # actual joint position
    jnt_act_vel: np.ndarray = np.zeros(n_leg * 3)  # actual joint velocity
    body_orn: np.ndarray = np.zeros(4)             # actual body orientation, as quaternion, in WCS
    # states need estimation 
    body_pos: np.ndarray = np.zeros(3)             # actual body position, in WCS
    body_vel: np.ndarray = np.zeros(3)             # actual body linear velocity, in WCS
    body_angvel: np.ndarray = np.zeros(3)          # actual body angular velocity, in WCS


class QuadRobotCommands(object):
    """
        Control commands from the operator
    """
    gait_switch : bool
    keys: any


class QuadControlOutput(object):
    """
        Control outputs for quadruped robot
    """
    # constants
    n_leg: int = 4
    # control output
    joint_tgt_trq: np.ndarray = np.zeros(n_leg *3) # joint target torque command


class QuadGaitController(object):
    """
        Quadruped robot gait controller.
    """
    # gait pattern generator
    stand_gpg:    GaitPatternGenerator
    trot_gpg:     GaitPatternGenerator
    current_gpg : GaitPatternGenerator
    gait_switch_cmd: bool

    # robot kinematic and dynamic models
    kin_model : RobotKineticModel
    dyn_model : RobotDynamicModel
    mapper    : JointOrderMapper
    
    # MPC related variables
    mpc: QuadConvexMPC
    mpc_solve_count: int = 0

    # Robot steer
    rst: RobotSteer

    def __init__(self) -> None:

        # setup gait generator
        self.stand_gait = GaitPatternGenerator(
            'stand', 0.5, np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]))

        self.trot_gait = GaitPatternGenerator('trot',  0.3, np.array(
            [0, 0.5, 0.5, 0]), np.array([0.5, 0.5, 0.5, 0.5]))
        
        # setup kinematics model
        self.kin_model = RobotKineticModel()

        # setup dynamic model
        urdf_file = r'./models/a1/a2_pino.urdf'
        mesh_dir  = r'./models/a1/'

        os.environ['mesh_dir'] = mesh_dir
        self.dyn_model = RobotDynamicModel()
        self.dyn_model.load_model(urdf_file, mesh_dir)

        self.mapper = JointOrderMapper()

        # setup robot steer
        self.rst = RobotSteer()


    def load(self):
        self.mpc_solve_count = 0
        self.current_gpg = self.stand_gait
        self.gait_switch_cmd = False


    def control_step(self, feedbacks: QuadControlInput, cmd: QuadRobotCommands) -> np.ndarray:
        # handle commands
        # update gait pattern
        # update robot states
        # body trajectory planning
        # foothold planning
        # leg trajectory planning
        # if use_mpc and mpc.need_solve
        #   solve mpc
        # elif use_wbc
        #   solve wbc
        # solve wbic
        # joint trq control
        pass
