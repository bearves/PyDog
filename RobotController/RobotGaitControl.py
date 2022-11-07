import os
import numpy as np
from scipy.spatial.transform import Rotation as rot

from RobotController.RobotKinematics import RobotKineticModel
from RobotController.RobotDynamics import RobotDynamicModel, JointOrderMapper
from RobotController.GaitPatternGenerator import GaitPatternGenerator
from RobotController.RobotTrjPlanner import BodyTrjPlanner, FootholdPlanner, SwingTrjPlanner
from RobotController.RobotMPC import QuadConvexMPC
from RobotController.RobotVMC import QuadSingleBodyVMC
from RobotController.RobotWBIC import QuadWBIC
from RobotController.RobotSteer import RobotSteer, DirectionFlag

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
    gait_switch   : bool = False                   # flag whether the user asks to switch gait
    direction_flag: list[int] = [0, 0, 0, 0, 0, 0] # flags whether a movement direction is required, see DirectionFlag in RobotSteer.py


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
    vmc:  QuadSingleBodyVMC
    mpc:  QuadConvexMPC
    wbic: QuadWBIC

    # Robot steer
    robot_steer: RobotSteer

    # state variables
    count: int
    gait_switch_cmd: bool
    current_support_state: np.ndarray
    tip_act_pos: np.ndarray
    tip_act_vel: np.ndarray
    tip_ref_pos: np.ndarray
    tip_ref_vel: np.ndarray

    last_body_euler_ypr: np.ndarray

    # control internal outputs
    u_mpc: np.ndarray
    u_vmc: np.ndarray
    ref_leg_force_wcs: np.ndarray
    jnt_ref_pos_wbic:  np.ndarray
    jnt_ref_vel_wbic:  np.ndarray
    jnt_ref_trq_wbic:  np.ndarray
    jnt_ref_trq_final: np.ndarray

    def __init__(self, use_mpc: bool) -> None:
        """
            Create quadruped robot gait controller.

            Parameters:
                use_mpc: Set the controller to use MPC as body controller, otherwise a VMC is utilized.
        """
        # setup controller options
        self.use_mpc = use_mpc

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
        self.vmc = QuadSingleBodyVMC(self.dt)
        self.wbic = QuadWBIC()

        # setup robot steer
        self.robot_steer = RobotSteer()

        # init variables
        self.count = 0
        self.gait_switch_cmd = False
        self.current_support_state = np.ones(self.n_leg)
        self.tip_act_pos = np.zeros(self.n_leg * 3)
        self.tip_act_vel = np.zeros(self.n_leg * 3)
        self.tip_ref_pos = np.zeros(self.n_leg * 3)
        self.tip_ref_vel = np.zeros(self.n_leg * 3)

        self.last_body_euler_ypr = np.zeros(3)

        self.u_mpc = np.zeros((3*self.n_leg, self.mpc.horizon_length))
        self.u_vmc = np.zeros(3*self.n_leg)
        self.ref_leg_force_wcs = np.zeros(3*self.n_leg)
        self.jnt_ref_pos_wbic = np.zeros(3*self.n_leg)
        self.jnt_ref_vel_wbic = np.zeros(3*self.n_leg)
        self.jnt_ref_trq_wbic = np.zeros(3*self.n_leg)
        self.jnt_ref_trq_final = np.zeros(3*self.n_leg)


    def load(self, feedbacks: QuadControlInput):
        """
            Setup the gait controller's initial states.
            This method should be called once entering the gait control state.

            Parameters:
                feedbacks (QuadControlInput): the robot feedbacks from simulator.
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

        self.last_body_euler_ypr = body_euler

        self.u_mpc = np.zeros((3*self.n_leg, self.mpc.horizon_length))
        self.u_vmc = np.zeros(3*self.n_leg)
        self.ref_leg_force_wcs = np.zeros(3*self.n_leg)
        self.jnt_ref_pos_wbic = np.zeros(3*self.n_leg)
        self.jnt_ref_vel_wbic = np.zeros(3*self.n_leg)
        self.jnt_ref_trq_wbic = np.zeros(3*self.n_leg)
        self.jnt_ref_trq_final = np.zeros(3*self.n_leg)


    def control_step(self, 
                     feedbacks: QuadControlInput, 
                     user_cmd: QuadRobotCommands) -> QuadControlOutput:
        """
            Run the controller.
            This method should be called at every control step.
            
            Parameters:
                feedbacks (QuadControlInput): the robot feedbacks from simulator.
                user_cmd  (QuadRobotCommands): the user command to the gait controller.

            Returns:
                output    (QuadControlOutput): the controller's outputs to the robot.
        """

        self.count += 1
        # handle commands
        self.handle_cmd(user_cmd)
        # update gait pattern and robot states
        self.update_gait_states(feedbacks)
        # trajectory planning
        self.trajectory_planning(feedbacks)

        # body control
        if self.use_mpc:
            if self.mpc.need_solve(self.count - 1):
                self.solve_mpc(feedbacks)
            self.ref_leg_force_wcs = self.u_mpc[:, 0]
        else:
            if self.vmc.need_solve(self.count - 1):
                self.solve_vmc(feedbacks)
            self.ref_leg_force_wcs = self.u_vmc
        # solve wbic
        self.solve_wbic(feedbacks)
        self.solve_static_dyn()
        # joint trq control
        self.joint_trq_control(feedbacks)

        output = QuadControlOutput()
        output.joint_tgt_trq = self.jnt_ref_trq_final

        return output


    def handle_cmd(self, user_cmd: QuadRobotCommands):
        """
            Handle user commands.

            Parameters:
                user_cmd  (QuadRobotCommands): the user command to the gait controller.
        """
        # set gait switch flag. This flag will be cleared after
        # the gait has successfully switched
        if user_cmd.gait_switch:
            self.gait_switch_cmd = True
            user_cmd.gait_switch = False
        
        self.robot_steer.set_direction_flag(user_cmd.direction_flag)

    
    def update_gait_states(self, feedbacks: QuadControlInput):
        """
            Update gait state variables.

            Parameters:
                feedbacks (QuadControlInput): the robot feedbacks from simulator.
        """

        # switch gait generator if cmd received
        if self.gait_switch_cmd and \
           self.current_gait.gait_name != 'trot' and \
           self.current_gait.can_switch():

            print("Switch to trot at %f" % feedbacks.time_now)
            self.current_gait = self.trot_gait
            self.current_gait.set_start_time(feedbacks.time_now)
            self.gait_switch_cmd = False
        
        # update gait pattern generator
        self.current_gait.set_current_time(feedbacks.time_now)
        self.current_support_state = \
            self.current_gait.get_current_support_state()

        # update vel cmd for planners
        self.robot_steer.update_vel_cmd(feedbacks.body_orn)

        # update kinematic model
        self.kin_model.update(
            feedbacks.body_pos, feedbacks.body_orn,
            feedbacks.body_vel, feedbacks.body_angvel,
            feedbacks.jnt_act_pos, feedbacks.jnt_act_vel
        )
        self.tip_act_pos, self.tip_act_vel = \
            self.kin_model.get_tip_state_world()

    
    def trajectory_planning(self, feedbacks: QuadControlInput):
        """
            Plan reference trajectories for body and legs.

            Parameters:
                feedbacks (QuadControlInput): the robot feedbacks from simulator.
        """
        filtered_vel_cmd = self.robot_steer.get_vel_cmd_wcs()
        # calculate trajectory for body in wcs
        self.body_planner.update_ref_state(
            filtered_vel_cmd, 
            feedbacks.body_pos, 
            feedbacks.body_orn)

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

    
    def solve_mpc(self, feedbacks: QuadControlInput):
        """
            Solve MPC problem for robot body balance control.

            Parameters:
                feedbacks (QuadControlInput): the robot feedbacks from simulator.
        """
        # Step 1. Build current body state
        current_state = self.build_current_state(feedbacks)

        # Step 2. Predict future support state
        support_state_future_traj = self.current_gait.predict_mpc_support_state(
            self.mpc.dt_mpc, self.mpc.horizon_length)

        # Step 3. Generate future body trajectory
        body_ref_traj = self.body_planner.predict_future_body_ref_traj(
            self.mpc.dt_mpc, self.mpc.horizon_length)
        body_future_euler = body_ref_traj[:, 0:3]

        # Step 4. Predict future foot position

        foothold_future_traj = \
            self.foothold_planner.predict_future_foot_pos(self.mpc.dt_mpc,
                                                          self.mpc.horizon_length,
                                                          support_state_future_traj)

        # Step 5. Solve MPC problem
        #print('MPC solving')
        self.mpc.update_mpc_matrices(
            body_future_euler, foothold_future_traj,
            support_state_future_traj, current_state,
            body_ref_traj)
        self.mpc.update_super_matrices()
        self.u_mpc = self.mpc.solve()


    def solve_vmc(self, feedbacks: QuadControlInput):
        """
            Solve VMC problem for robot body balance control.

            Parameters:
                feedbacks (QuadControlInput): the robot feedbacks from simulator.
        """
        # Step 1. Build current body state
        current_state = self.build_current_state(feedbacks)

        # Step 2. Get current body reference
        body_ref_traj = self.body_planner.predict_future_body_ref_traj(
            self.vmc.dt_vmc, 1)

        # Step 3. Solve VMC problem
        self.vmc.update_vmc_matrices(
            feedbacks.body_orn, self.tip_act_pos, 
            self.current_support_state, 
            current_state[0:12], 
            body_ref_traj[0, 0:12])
        
        self.u_vmc = self.vmc.solve()
    

    def solve_wbic(self, feedbacks: QuadControlInput):
        """
            Solve WBIC problem to obtain robot joint reference pos/vel and feed-forward trq.

            Parameters:
                feedbacks (QuadControlInput): the robot feedbacks from simulator.
        """
        # Step 1. Map joint order to Pinocchio's definition and 
        #         update dynamic model
        jnt_pos_act_pino = self.idx_mapper.convert_vec_to_pino(feedbacks.jnt_act_pos)
        jnt_vel_act_pino = self.idx_mapper.convert_vec_to_pino(feedbacks.jnt_act_vel)
        support_state_act_pino = self.idx_mapper.convert_sps_to_pino(self.current_support_state)

        self.dyn_model.update(
            feedbacks.body_pos, feedbacks.body_orn, 
            feedbacks.body_vel, feedbacks.body_angvel, 
            jnt_pos_act_pino, jnt_vel_act_pino)

        self.dyn_model.update_support_states(support_state_act_pino)

        # Step 2. Set ref position, velocity and acceleration for tasks
        task_ref = np.zeros(7+self.n_leg*3)
        task_ref[0:3] = self.body_planner.ref_body_pos # pos
        # FIXME: use ref body orn doesn't work.
        #task_ref[3:7] = self.body_planner.ref_body_orn # ori
        task_ref[3:7] = feedbacks.body_orn
        
        leg_tip_pos_wcs_ref_pino = self.idx_mapper.convert_vec_to_pino(self.tip_ref_pos)
        task_ref[7:19] = leg_tip_pos_wcs_ref_pino # leg tip pos

        task_dot_ref = np.zeros(6+self.n_leg*3)
        task_dot_ref[0:3] = self.body_planner.ref_body_vel # pos
        task_dot_ref[3:6] = self.body_planner.ref_body_angvel # ori
        leg_tip_vel_wcs_ref_pino = self.idx_mapper.convert_vec_to_pino(self.tip_ref_vel)
        task_dot_ref[6:18] = leg_tip_vel_wcs_ref_pino # leg tip pos

        task_ddot_ref = np.zeros(6+self.n_leg*3)
        # TODO: Plan accelerations for them
        #task_ddot_ref[0:3] = body_acc_ref # pos
        #task_ddot_ref[3:6] = body_angacc_ref # ori
        #leg_tip_acc_wcs_ref_pino = mapper.convert_vec_to_pino(leg_tip_acc_wcs_ref)
        #task_ddot_ref[6:18] = leg_tip_acc_wcs_ref_pino # leg tip pos

        ref_leg_force_wcs_pino = self.idx_mapper.convert_vec_to_pino(self.ref_leg_force_wcs)
        # now we can run wbic
        ret = self.wbic.update(self.dyn_model,
                          task_ref,
                          task_dot_ref,
                          task_ddot_ref,
                          ref_leg_force_wcs_pino
                          )
        dq = self.idx_mapper.convert_jvec_to_our(ret[0])
        self.jnt_ref_pos_wbic = feedbacks.jnt_act_pos + dq
        self.jnt_ref_vel_wbic = self.idx_mapper.convert_jvec_to_our(ret[1])
        self.jnt_ref_trq_wbic = self.idx_mapper.convert_jvec_to_our(ret[2])


    def solve_static_dyn(self):
        """
            Calculate joint trq using static dynamics.
                                
                                tau = J_leg * R_hip.T * R_body.T * f_wcs

            Parameters:
                feedbacks (QuadControlInput): the robot feedbacks from simulator.
        """
        
        self.jnt_trq_static_dyn = -self.kin_model.get_joint_trq(self.ref_leg_force_wcs)


    def joint_trq_control(self, feedbacks: QuadControlInput):
        """
            Joint PD controller with torque feed-forward.

                        trq_cmd = kp * jnt_pos_err + kd * jnt_vel_err + tau_ff
            
            Note that the output is saturated(clipped).

            Parameters:
                feedbacks (QuadControlInput): the robot feedbacks from simulator.
        """
        for leg in range(4):
            if self.current_support_state[leg] == 1:  # stance
                kp, kd = 10, 1
            else:  # swing
                kp, kd = 10, 1

        pos_err = self.jnt_ref_pos_wbic - feedbacks.jnt_act_pos
        vel_err = self.jnt_ref_vel_wbic - feedbacks.jnt_act_vel

        #self.jnt_ref_trq_wbic = self.jnt_trq_static_dyn.copy()
        trq_final = kp * pos_err + kd * vel_err + self.jnt_ref_trq_wbic

        max_trq = 50
        self.jnt_ref_trq_final = np.clip(trq_final, -max_trq, max_trq)


    def build_current_state(self, feedbacks: QuadControlInput) -> np.ndarray:
        """
            Build robot current state vector for MPC/VMC solver.

            Parameters:
                feedbacks (QuadControlInput): the robot feedbacks from simulator.

            Returns:
                current_state (array(13)): the current state vector for MPC/VMC usage.
        """
        # get body euler angles
        body_euler_ypr = rot.from_quat(feedbacks.body_orn).as_euler('ZYX')
        # prevent euler angle range skip
        # if body_euler_ypr[0] - self.last_body_euler_ypr[0] < -6.1:
        #    body_euler_ypr[0] = body_euler_ypr[0] + 2*np.pi
        #    print('Euler angle jump')
        #elif body_euler_ypr[0] - self.last_body_euler_ypr[0] > 6.1:
        #    body_euler_ypr[0] = body_euler_ypr[0] - 2*np.pi
        #    print('Euler angle jump')
        
        self.last_body_euler_ypr = body_euler_ypr

        current_state = np.ones(13) # current state for MPC
        current_state[0:3] = np.flip(body_euler_ypr) # theta ( roll, pitch, yaw )
        current_state[3:6] = feedbacks.body_pos  # x, y, z
        current_state[6:9] = feedbacks.body_angvel  # wx, wy, wz
        current_state[9:12] = feedbacks.body_vel  # vx, vy, vz
        current_state[12] = -9.81      # g constant

        return current_state