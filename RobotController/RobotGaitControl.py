import os
import numpy as np
from scipy.spatial.transform import Rotation as rot

from RobotKinematics import RobotKineticModel
from RobotDynamics import RobotDynamicModel, JointOrderMapper
from GaitPatternGenerator import GaitPatternGenerator
from RobotTrjPlanner import BodyTrjPlanner, FootholdPlanner, SwingTrjPlanner
from RobotMPC import QuadConvexMPC
from RobotWBC import QuadSingleBodyWBC
from RobotWBIC import QuadWBIC
from RobotSteer import RobotSteer, DirectionFlag

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
    gait_switch   : bool
    direction_flag: list[int]


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
    # control constants
    n_leg: int = 4
    dt: float = 1./1000.
    use_mpc: bool = True

    # gait pattern generator
    stand_gait:    GaitPatternGenerator
    trot_gait:     GaitPatternGenerator
    current_gait:  GaitPatternGenerator

    # robot kinematic and dynamic models
    kin_model:  RobotKineticModel
    dyn_model:  RobotDynamicModel
    idx_mapper: JointOrderMapper

    # trj planners
    body_planner:       BodyTrjPlanner
    foothold_planner:   FootholdPlanner
    swing_traj_planner: list[SwingTrjPlanner]

    # controllers
    wbc:  QuadSingleBodyWBC
    mpc:  QuadConvexMPC
    wbic: QuadWBIC

    # Robot steer
    steer: RobotSteer

    # state variables
    count: int
    gait_switch_cmd: bool
    current_support_state: np.ndarray
    tip_act_pos: np.ndarray
    tip_act_vel: np.ndarray
    tip_ref_pos: np.ndarray
    tip_ref_vel: np.ndarray

    def __init__(self) -> None:
        """
            Create quadruped robot gait controller.
        """

        # setup gait generator
        self.stand_gait = GaitPatternGenerator(
            'stand', 0.5, np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]))

        self.trot_gait = GaitPatternGenerator('trot',  0.3, np.array(
            [0, 0.5, 0.5, 0]), np.array([0.5, 0.5, 0.5, 0.5]))
        
        self.current_gait = self.stand_gait

        # setup kinematics model
        self.kin_model = RobotKineticModel()

        # setup dynamic model
        urdf_file = r'./models/a1/a2_pino.urdf'
        mesh_dir  = r'./models/a1/'
        os.environ['mesh_dir'] = mesh_dir

        self.dyn_model = RobotDynamicModel()
        self.dyn_model.load_model(urdf_file, mesh_dir)

        self.idx_mapper = JointOrderMapper()

        # setup trajectory planners
        self.body_planner = BodyTrjPlanner(self.dt)
        self.foothold_planner = FootholdPlanner()

        self.swing_traj_planner = list()
        for i in range(self.n_leg):
            self.swing_traj_planner.append(SwingTrjPlanner())

        # setup controllers
        self.mpc = QuadConvexMPC(self.dt)
        self.mpc.cal_weight_matrices()
        self.wbc = QuadSingleBodyWBC(self.dt)
        self.wbic = QuadWBIC()

        # setup robot steer
        self.steer = RobotSteer()

        # init variables
        self.count = 0
        self.gait_switch_cmd = False
        self.current_support_state = np.ones(self.n_leg)
        self.tip_act_pos = np.zeros(self.n_leg * 3)
        self.tip_act_vel = np.zeros(self.n_leg * 3)
        self.tip_ref_pos = np.zeros(self.n_leg * 3)
        self.tip_ref_vel = np.zeros(self.n_leg * 3)


    def load(self, feedbacks: QuadControlInput):
        """
            Setup the gait controller's initial states.
            This method should be called once entering 
            the gait control state.
        """
        self.count = 0
        self.gait_switch_cmd = False
        self.current_gait = self.stand_gait
        self.current_gait.set_start_time(feedbacks.time_now)
        self.current_support_state = np.ones(self.n_leg)

        body_euler = rot.from_quat(feedbacks.body_orn).as_euler('ZYX')
        body_yaw = body_euler[0]
        self.body_planner.set_ref_state(
            feedbacks.body_pos,
            rot.from_rotvec([0, 0, body_yaw]).as_quat(),
            np.zeros(3),
            np.zeros(3))

        self.kin_model.update(
            feedbacks.body_pos, feedbacks.body_orn,
            feedbacks.body_vel, feedbacks.body_angvel,
            feedbacks.jnt_act_pos, feedbacks.jnt_act_vel
        )
        tip_init_pos, _ = self.kin_model.get_tip_state_world()
        self.foothold_planner.init_footholds(tip_init_pos)


    def control_step(self, feedbacks: QuadControlInput, cmd: QuadRobotCommands) -> np.ndarray:
        """
            Run the controller.
            This method should be called at every control step.
        """

        self.count += 1
        # handle commands
        self.handle_cmd(cmd)
        # update gait pattern and robot states
        self.update_gait_states(feedbacks)
        # trajectory planning
        self.trajectory_planning(feedbacks)

        # body control
        if self.use_mpc and self.mpc.need_solve(self.count - 1):
            self.solve_mpc()
        else:
            self.solve_wbc()
        # solve wbic
        self.solve_wbic()
        # joint trq control
        self.joint_trq_control()
        # log necessary data
        self.log_data()


    def handle_cmd(self, cmd: QuadRobotCommands):
        """
            Handle user commands.
        """
        # set gait switch flag. This flag will be cleared after
        # the gait has successfull switched
        if cmd.gait_switch:
            self.gait_switch_cmd = True
            cmd.gait_switch = False
        
        self.steer.set_direction_flag(cmd.direction_flag)

    
    def update_gait_states(self, feedbacks: QuadControlInput):
        """
            Update gait state variables
        """

        # switch gait generator if cmd received
        if self.gait_switch_cmd and \
           self.current_gait.gait_name != 'trot' and \
           self.current_gait.can_switch():

            self.current_gait = self.trot_gait
            self.current_gait.set_start_time(feedbacks.time_now)
            self.gait_switch_cmd = False
        
        # update gait pattern generator
        self.current_gait.set_current_time(feedbacks.time_now)
        self.current_support_state = \
            self.current_gait.get_current_support_state()

        # update vel cmd for planners
        self.steer.update_vel_cmd(feedbacks.body_orn)

        # update kinematic and dynamic model
        self.kin_model.update(
            feedbacks.body_pos, feedbacks.body_orn,
            feedbacks.body_vel, feedbacks.body_angvel,
            feedbacks.jnt_act_pos, feedbacks.jnt_act_vel
        )
        self.tip_act_pos, self.tip_act_vel = \
            self.kin_model.get_tip_state_world()

        self.dyn_model.update(
            feedbacks.body_pos, feedbacks.body_orn,
            feedbacks.body_vel, feedbacks.body_angvel,
            feedbacks.jnt_act_pos, feedbacks.jnt_act_vel
        )
        self.dyn_model.update_support_states(
            self.current_support_state
        )
    
    def trajectory_planning(self, feedbacks: QuadControlInput):
        """
            Plan reference trajectories for body and legs.
        """
        filtered_vel_cmd = self.steer.get_vel_cmd_wcs()
        # calculate trajectory for body in wcs
        self.body_planner.update_ref_state(filtered_vel_cmd)

        # calculate foothold for legs
        self.foothold_planner.set_current_state(
            self.tip_act_pos, 
            self.current_support_state)

        self.foothold_planner.calculate_next_footholds(
            feedbacks.body_pos,
            feedbacks.body_orn,
            feedbacks.body_vel,
            filtered_vel_cmd,
            self.current_gait.get_swing_time_left(),
            self.current_gait.get_stance_duration(),
            self.kin_model.get_hip_pos_body_with_offset()
        )

        # calculate trajectory for legs in wcs
        for leg in range(self.n_leg):
            idx = range(0+leg*3, 3+leg*3)
            if self.current_support_state[leg] == 1:  # stance
                self.tip_ref_pos[idx] = self.foothold_planner.get_current_footholds(leg)
                self.tip_ref_vel[idx] = np.zeros(3)
            else:  # swing
                self.swing_traj_planner[leg].set_start_point(
                    self.foothold_planner.get_liftup_leg_pos(leg))
                self.swing_traj_planner[leg].set_end_point(
                    self.foothold_planner.get_next_footholds(leg))
                # get swing time ratio
                swing_time_ratio, swing_time_ratio_dot = \
                    self.current_gait.get_current_swing_time_ratio(leg)
                # interpolate from last foothold to the next foothold
                self.tip_ref_pos[idx], self.tip_ref_vel[idx]\
                    = self.swing_traj_planner[leg].get_tip_pos_vel(
                        swing_time_ratio, swing_time_ratio_dot)

    
    def solve_mpc(self):
        # current body state
        body_euler_ypr = rot.from_quat(body_orn).as_euler('ZYX')

        # prevent euler angle range skip
        if body_euler_ypr[0] - last_body_euler_ypr[0] < -6.1:
            body_euler_ypr[0] = body_euler_ypr[0] + 2*np.pi
            print('Euler angle jump')
        elif body_euler_ypr[0] - last_body_euler_ypr[0] > 6.1:
            body_euler_ypr[0] = body_euler_ypr[0] - 2*np.pi
            print('Euler angle jump')

        current_state = np.ones(13)
        # theta ( roll, pitch, yaw )
        current_state[0:3] = np.flip(body_euler_ypr)
        current_state[3:6] = body_pos  # x, y, z
        current_state[6:9] = body_angvel  # wx, wy, wz
        current_state[9:12] = body_vel  # vx, vy, vz
        current_state[12] = -9.81      # g constant

        last_body_euler_ypr = body_euler_ypr

        # predict future support state
        support_state_future_traj = current_gpg.predict_mpc_support_state(
            mpc.horizon_length, mpc.dt_mpc)

        # generate future body trajectory
        body_ref_traj = body_planner.predict_future_body_ref_traj(
            time_step, mpc.horizon_length)
        body_future_euler = body_ref_traj[:, 0:3]

        # predict future foot position

        foothold_future_traj = foothold_planner.predict_future_foot_pos(time_step,
                                                                        mpc.horizon_length,
                                                                        support_state_future_traj)

        # solve mpc problem
        #print('MPC solving')
        mpc.update_mpc_matrices(
            body_future_euler, foothold_future_traj,
            support_state_future_traj, current_state,
            body_ref_traj)
        mpc.update_super_matrices()
        u_mpc = mpc.solve()
