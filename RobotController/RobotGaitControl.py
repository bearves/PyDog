import os
import numpy as np
from scipy.spatial.transform import Rotation as rot

from RobotController.RobotKinematics import RobotKineticModel
from RobotController.RobotDynamics import RobotDynamicModel, JointOrderMapper
from RobotController.RobotFullStateEstimate import QuadFullStateEstimator
from RobotController.RobotPosVelEstimate import QuadPosVelEstimator
from RobotController.RobotTerrainEstimate import QuadTerrainEstimator
from RobotController.GaitPatternGenerator import GaitPatternGenerator
from RobotController.RobotTrjPlanner import BodyTrjPlanner, FootholdPlanner, SwingTrjPlanner
from RobotController.RobotMPC import QuadConvexMPC
from RobotController.RobotVMC import QuadSingleBodyVMC
from RobotController.RobotWBIC import QuadWBIC
from RobotController.RobotSteer import RobotSteer, DirectionFlag


class QuadControlInput(object):
    """
        Control inputs for quadruped robot.
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
    # raw measurement from sensor
    snsr_acc: np.ndarray = np.zeros(3)             # accelerometer readings, in sensor's local frame
    snsr_gyr: np.ndarray = np.zeros(3)             # gyro readings, in sensor's local frame


class QuadRobotCommands(object):
    """
        Control commands from the operator
    """
    # flag whether the user asks to switch gait
    gait_switch   : bool = False                   
    # flags whether a movement direction is required, see DirectionFlag in RobotSteer.py
    direction_flag: list[int] = [0, 0, 0, 0, 0, 0] 


class QuadControlOutput(object):
    """
        Control outputs for quadruped robot
    """
    # constants
    n_leg: int = 4
    # control output
    joint_tgt_trq: np.ndarray = np.zeros(n_leg *3) # joint target torque command
    support_state: np.ndarray = np.ones(n_leg)     # support state of all legs
    support_phase: np.ndarray = np.zeros(n_leg)    # support phase of all legs
    est_body_pos: np.ndarray = np.zeros(3)         # estimated body pos in WCS
    est_body_vel: np.ndarray = np.zeros(3)         # estimated body vel in WCS
    est_body_orn: np.ndarray = np.zeros(4)         # estimated body orn in WCS


class QuadRobotState(object):
    """
        Robot body state container
    """
    # constants
    n_leg: int = 4
    # body states
    body_pos: np.ndarray = np.zeros(3)
    body_vel: np.ndarray = np.zeros(3)
    body_orn: np.ndarray = np.array([0, 0, 0, 1])
    body_angvel: np.ndarray = np.zeros(3)
    # joint states
    jnt_pos: np.ndarray = np.zeros(n_leg*3)
    jnt_vel: np.ndarray = np.zeros(n_leg*3)

    def reset(self, 
              body_pos: np.ndarray,
              body_vel: np.ndarray,
              body_orn: np.ndarray,
              body_angvel: np.ndarray,
              jnt_pos: np.ndarray,
              jnt_vel: np.ndarray):
        """
            Reset body state.

            Parameters:
                body_pos (array(3)): body linear position in WCS.
                body_vel (array(3)): body linear velocity in WCS.
                body_orn (array(4)): body orientation in WCS.
                body_vel (array(3)): body angular velocity in WCS.
                jnt_pos  (array(n_leg*3)): joint position.
                jnt_vel  (array(n_leg*3)): joint velocity.
        """
        self.body_pos = body_pos.copy()
        self.body_vel = body_vel.copy()
        self.body_orn = body_orn.copy()
        self.body_angvel = body_angvel.copy()
        self.jnt_pos = jnt_pos.copy()
        self.jnt_vel = jnt_vel.copy()


class QuadGaitController(object):
    """
        Quadruped robot gait controller.
    """
    # control constants
    n_leg: int = 4
    dt: float = 1./1000.
    leap: int = 25
    use_mpc: bool = True
    use_se: bool = True

    # gait pattern generator
    stand_gait:    GaitPatternGenerator
    trot_gait:     GaitPatternGenerator
    current_gait:  GaitPatternGenerator

    # robot kinematic and dynamic models
    kin_model:  RobotKineticModel
    dyn_model:  RobotDynamicModel
    idx_mapper: JointOrderMapper

    # state estimator
    Rs_bcs: np.ndarray # Rotation matrix of the sensor measurement frame, w.r.t. body cs
    # state_estm: QuadFullStateEstimator
    state_estm: QuadPosVelEstimator
    terrn_estm: QuadTerrainEstimator

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
    current_support_phase: np.ndarray
    current_robot_state: QuadRobotState
    tip_act_pos: np.ndarray
    tip_act_vel: np.ndarray
    tip_ref_pos: np.ndarray
    tip_ref_vel: np.ndarray
    tip_ref_acc: np.ndarray

    last_body_euler_ypr: np.ndarray

    # control internal outputs
    u_mpc: np.ndarray
    u_vmc: np.ndarray
    ref_leg_force_wcs: np.ndarray
    jnt_ref_pos_wbic:  np.ndarray
    jnt_ref_vel_wbic:  np.ndarray
    jnt_ref_trq_wbic:  np.ndarray
    jnt_ref_trq_final: np.ndarray

    def __init__(self, 
                 dt: float, 
                 leap: int, 
                 urdf_file: str, 
                 mesh_dir: str, 
                 use_mpc: bool, 
                 use_se: bool, 
                 Rs_bcs: np.ndarray) -> None:
        """
            Create quadruped robot gait controller.

            Parameters:
                dt (float): base time step of the simulator.
                leap (int): the leap step of MPC/VMC, so that the controller solves MPC/VMC every (leap) dt.
                urdf_file (str): path of the robot's urdf file.
                mesh_dir (str): path of the parent directory to store the robot's mesh files.
                use_mpc(bool): Set the controller to use MPC as body controller, otherwise a VMC is utilized.
                use_se (bool): Set the controller to use state estimator to estimate robot states based on 
                               raw sensor's signal (Normal mode), otherwise the controller use robot state 
                               directly from the simulation environment (God mode). 
                Rs_bcs (array(3,3)): Rotation matrix of the sensor measurement frame, w.r.t. body cs
        """
        # setup controller options
        self.use_mpc = use_mpc
        self.use_se = use_se
        self.dt = dt
        self.leap = leap

        # setup gait generator
        self.stand_gait = GaitPatternGenerator(
            'stand', 0.5, np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]))

        # flying trot
        self.trot_gait = GaitPatternGenerator('trot',  0.3, np.array(
            [0, 0.5, 0.5, 0]), 0.35 * np.array([1, 1, 1, 1]))
        # walking trot
        #self.trot_gait = GaitPatternGenerator('trot', 0.5, np.array(
        #    [0, 0.5, 0.5, 0]), np.array([0.5, 0.5, 0.5, 0.5]))
            
        self.current_gait = self.stand_gait

        # setup kinematics model
        self.kin_model = RobotKineticModel()

        # setup dynamic model
        os.environ['mesh_dir'] = mesh_dir

        self.dyn_model = RobotDynamicModel()
        self.dyn_model.load_model(urdf_file, mesh_dir)

        self.idx_mapper = JointOrderMapper()

        # setup state estimator
        self.Rs_bcs = Rs_bcs
        # self.state_estm = QuadFullStateEstimator(self.dt)
        self.state_estm = QuadPosVelEstimator(self.dt)
        self.terrn_estm = QuadTerrainEstimator()

        # setup trajectory planners
        self.body_planner = BodyTrjPlanner(self.dt)
        self.foothold_planner = FootholdPlanner()

        self.swing_traj_planner = list()
        for i in range(self.n_leg):
            self.swing_traj_planner.append(SwingTrjPlanner())

        # setup controllers
        self.mpc = QuadConvexMPC(self.dt, self.leap)
        self.mpc.cal_weight_matrices()
        self.vmc = QuadSingleBodyVMC(self.dt, self.leap)
        self.wbic = QuadWBIC()

        # setup robot steer
        self.robot_steer = RobotSteer()

        # init variables
        self.count = 0
        self.gait_switch_cmd = False
        self.current_support_state = np.ones(self.n_leg)
        self.current_support_phase = np.zeros(self.n_leg)
        self.current_robot_state = QuadRobotState()
        self.tip_act_pos = np.zeros(self.n_leg * 3)
        self.tip_act_vel = np.zeros(self.n_leg * 3)
        self.tip_ref_pos = np.zeros(self.n_leg * 3)
        self.tip_ref_vel = np.zeros(self.n_leg * 3)
        self.tip_ref_acc = np.zeros(self.n_leg * 3)

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
        self.current_support_phase = np.zeros(self.n_leg)

        self.current_robot_state.reset(
            feedbacks.body_pos, feedbacks.body_vel,
            feedbacks.body_orn, feedbacks.body_angvel,
            feedbacks.jnt_act_pos, feedbacks.jnt_act_vel)

        body_euler = rot.from_quat(self.current_robot_state.body_orn).as_euler('ZYX')
        body_yaw = body_euler[0]
        self.body_planner.set_ref_state(    
            self.current_robot_state.body_pos,
            rot.from_rotvec([0, 0, body_yaw]).as_quat(),
            np.zeros(3),
            np.zeros(3))

        self.kin_model.update_leg(
            self.current_robot_state.jnt_pos,self.current_robot_state.jnt_vel
        )
        self.kin_model.update_body(
            self.current_robot_state.body_pos, self.current_robot_state.body_orn,
            self.current_robot_state.body_vel, self.current_robot_state.body_angvel,
        )

        self.state_estm.reset_state(
            self.current_robot_state.body_pos, 
            self.current_robot_state.body_vel,
            self.current_robot_state.body_orn,
            self.kin_model.get_tip_state_world()[0])

        tip_init_pos, _ = self.kin_model.get_tip_state_world()
        self.terrn_estm.reset(tip_init_pos)
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
        self.trajectory_planning()

        # body control
        if self.use_mpc:
            if self.mpc.need_solve(self.count - 1):
                self.solve_mpc()
            self.ref_leg_force_wcs = self.u_mpc[:, 0]
        else:
            if self.vmc.need_solve(self.count - 1):
                self.solve_vmc()
            self.ref_leg_force_wcs = self.u_vmc
        # solve wbic
        self.solve_wbic()
        self.solve_static_dyn()
        # joint trq control
        self.joint_trq_control()

        output = self.set_control_output()

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
        time_now = feedbacks.time_now
        if self.gait_switch_cmd and \
           self.current_gait.gait_name != 'trot' and \
           self.current_gait.can_switch():

            print("Switch to trot at %f" % time_now)
            self.current_gait = self.trot_gait
            self.current_gait.set_start_time(time_now)
            self.gait_switch_cmd = False
        
        # update gait pattern generator
        self.current_gait.set_current_time(time_now)
        self.current_support_state = \
            self.current_gait.get_current_support_state()
        self.current_support_phase = \
            self.current_gait.get_current_support_time_ratio_all()[0]

        self.kin_model.update_leg(
            self.current_robot_state.jnt_pos, self.current_robot_state.jnt_vel
        )

        # update state estimation
        if self.use_se:
            body_gyr = self.Rs_bcs @ feedbacks.snsr_gyr
            body_acc = self.Rs_bcs @ feedbacks.snsr_acc 
            # if use PosVel Estimator, the body orn from feedbacks should be passed to the estimator
            # if use FullState Estimator, don't pass the body orn to the estimator 
            self.state_estm.update(self.kin_model,
                                   feedbacks.body_orn,
                                   body_gyr, 
                                   body_acc,
                                   feedbacks.jnt_act_pos, 
                                   feedbacks.jnt_act_vel,
                                   self.jnt_ref_trq_final, 
                                   self.current_support_state,
                                   self.current_support_phase)
            
            # TODO: when using state estimator's result, 
            # the foothold planning should be adjusted for consistent body height.
            self.current_robot_state.reset(
                self.state_estm.get_est_body_pos_wcs(), self.state_estm.get_est_body_vel_wcs(),
                self.state_estm.get_est_body_orn_wcs(), feedbacks.body_angvel,
                feedbacks.jnt_act_pos, feedbacks.jnt_act_vel)
        else:
            self.current_robot_state.reset(
                feedbacks.body_pos, feedbacks.body_vel,
                feedbacks.body_orn, feedbacks.body_angvel,
                feedbacks.jnt_act_pos, feedbacks.jnt_act_vel)


        # update vel cmd for planners
        self.robot_steer.update_vel_cmd(self.current_robot_state.body_orn)

        # update kinematic model
        self.kin_model.update_body(
            self.current_robot_state.body_pos, self.current_robot_state.body_orn,
            self.current_robot_state.body_vel, self.current_robot_state.body_angvel,
        )
        self.tip_act_pos, self.tip_act_vel = \
            self.kin_model.get_tip_state_world()
        
        # terrain estimation
        self.terrn_estm.update(self.tip_act_pos, self.current_support_phase)

    
    def trajectory_planning(self):
        """
            Plan reference trajectories for body and legs.
        """
        filtered_vel_cmd = self.robot_steer.get_vel_cmd_wcs()
        # calculate trajectory for body in wcs
        self.body_planner.update_ref_state(
            filtered_vel_cmd, 
            self.current_robot_state.body_pos, 
            self.current_robot_state.body_orn,
            self.terrn_estm.get_plane())

        # calculate foothold for legs
        self.foothold_planner.set_current_state(
            self.tip_act_pos, 
            self.current_support_state,
            self.terrn_estm.get_plane())

        self.foothold_planner.calculate_next_footholds(
            self.current_robot_state.body_pos,
            self.current_robot_state.body_orn,
            self.current_robot_state.body_vel,
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
                self.tip_ref_acc[idx] = np.zeros(3)
            else:  # swing
                self.swing_traj_planner[leg].set_start_point(
                    self.foothold_planner.get_liftup_leg_pos(leg))
                self.swing_traj_planner[leg].set_end_point(
                    self.foothold_planner.get_next_footholds(leg))
                # get swing time ratio
                swing_time_ratio, swing_time_ratio_dot = \
                    self.current_gait.get_current_swing_time_ratio(leg)
                # interpolate from last foothold to the next foothold
                self.tip_ref_pos[idx], self.tip_ref_vel[idx], self.tip_ref_acc[idx]\
                    = self.swing_traj_planner[leg].get_tip_pos_vel(
                        swing_time_ratio, swing_time_ratio_dot)

    
    def solve_mpc(self):
        """
            Solve MPC problem for robot body balance control.
        """
        # Step 1. Build current body state
        current_state = self.build_current_state()

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
        #self.u_mpc = self.mpc.solve()
        self.u_mpc = self.mpc.reduce_solve()
        #print(np.linalg.norm(self.u_mpc - self.u_mpc_2))


    def solve_vmc(self):
        """
            Solve VMC problem for robot body balance control.
        """
        # Step 1. Build current body state
        current_state = self.build_current_state()

        # Step 2. Get current body reference
        body_ref_traj = self.body_planner.predict_future_body_ref_traj(
            self.vmc.dt_vmc, 1)

        # Step 3. Solve VMC problem
        self.vmc.update_vmc_matrices(
            self.current_robot_state.body_orn, self.tip_act_pos, 
            self.current_support_state, 
            current_state[0:12], 
            body_ref_traj[0, 0:12])
        
        self.u_vmc = self.vmc.solve()
    

    def solve_wbic(self):
        """
            Solve WBIC problem to obtain robot joint reference pos/vel and feed-forward trq.
        """
        # Step 1. Map joint order to Pinocchio's definition and 
        #         update dynamic model
        jnt_pos_act_pino = self.idx_mapper.convert_vec_to_pino(self.current_robot_state.jnt_pos)
        jnt_vel_act_pino = self.idx_mapper.convert_vec_to_pino(self.current_robot_state.jnt_vel)
        support_state_act_pino = self.idx_mapper.convert_sps_to_pino(self.current_support_state)

        self.dyn_model.update(
            self.current_robot_state.body_pos, self.current_robot_state.body_orn, 
            self.current_robot_state.body_vel, self.current_robot_state.body_angvel, 
            jnt_pos_act_pino, jnt_vel_act_pino)

        self.dyn_model.update_support_states(support_state_act_pino)

        # Step 2. Set ref position, velocity and acceleration for tasks
        task_ref = np.zeros(7+self.n_leg*3)
        task_ref[0:3] = self.body_planner.ref_body_pos # pos
        task_ref[3:7] = self.body_planner.ref_body_orn # ori
        
        leg_tip_pos_wcs_ref_pino = self.idx_mapper.convert_vec_to_pino(self.tip_ref_pos)
        task_ref[7:19] = leg_tip_pos_wcs_ref_pino # leg tip pos

        task_dot_ref = np.zeros(6+self.n_leg*3)
        task_dot_ref[0:3] = self.body_planner.ref_body_vel # vel
        task_dot_ref[3:6] = self.body_planner.ref_body_angvel # omega
        leg_tip_vel_wcs_ref_pino = self.idx_mapper.convert_vec_to_pino(self.tip_ref_vel)
        task_dot_ref[6:18] = leg_tip_vel_wcs_ref_pino # leg tip vel

        task_ddot_ref = np.zeros(6+self.n_leg*3)
        task_ddot_ref[0:3] = np.zeros(3) # acc
        task_ddot_ref[3:6] = np.zeros(3) # angacc
        leg_tip_acc_wcs_ref_pino = self.idx_mapper.convert_vec_to_pino(self.tip_ref_acc)
        task_ddot_ref[6:18] = leg_tip_acc_wcs_ref_pino # leg tip acc

        ref_leg_force_wcs_pino = self.idx_mapper.convert_vec_to_pino(self.ref_leg_force_wcs)
        # now we can run wbic
        ret = self.wbic.update(self.dyn_model,
                          task_ref,
                          task_dot_ref,
                          task_ddot_ref,
                          ref_leg_force_wcs_pino
                          )
        self.jnt_ref_pos_wbic = self.idx_mapper.convert_jvec_to_our(ret[0])
        self.jnt_ref_vel_wbic = self.idx_mapper.convert_jvec_to_our(ret[1])
        self.jnt_ref_trq_wbic = self.idx_mapper.convert_jvec_to_our(ret[2])

    def solve_static_dyn(self):
        """
            Calculate joint trq using static dynamics.
                                
                                tau = J_leg * R_hip.T * R_body.T * f_wcs
        """
        
        self.jnt_trq_static_dyn = -self.kin_model.get_joint_trq(self.ref_leg_force_wcs)


    def joint_trq_control(self):
        """
            Joint PD controller with torque feed-forward.

                        trq_cmd = kp * jnt_pos_err + kd * jnt_vel_err + tau_ff
            
            Note that the output is saturated(clipped).
        """
        for leg in range(4):
            if self.current_support_state[leg] == 1:  # stance
                kp, kd = 10, 1
            else:  # swing
                kp, kd = 20, 0.5

        pos_err = self.jnt_ref_pos_wbic - self.current_robot_state.jnt_pos
        vel_err = self.jnt_ref_vel_wbic - self.current_robot_state.jnt_vel

        #self.jnt_ref_trq_wbic = self.jnt_trq_static_dyn.copy()
        trq_final = kp * pos_err + kd * vel_err + self.jnt_ref_trq_wbic

        max_trq = 50
        self.jnt_ref_trq_final = np.clip(trq_final, -max_trq, max_trq)

    
    def set_control_output(self) -> QuadControlOutput:
        """
            Set controller's output for simulator to execuate.

            Returns:
                output (QuadControlOutput): output of controller.
        """
        # set controller outputs
        output = QuadControlOutput()
        output.joint_tgt_trq = self.jnt_ref_trq_final
        output.support_state = self.current_support_state
        output.support_phase = \
                self.current_gait.get_current_support_time_ratio_all()[0]
        output.est_body_pos = self.state_estm.get_est_body_pos_wcs()
        output.est_body_vel = self.state_estm.get_est_body_vel_wcs()
        output.est_body_orn = self.state_estm.get_est_body_orn_wcs()
        return output


    def build_current_state(self) -> np.ndarray:
        """
            Build robot current state vector for MPC/VMC solver.

            Returns:
                current_state (array(13)): the current state vector for MPC/VMC usage.
        """
        # get body euler angles
        body_euler_ypr = rot.from_quat(self.current_robot_state.body_orn).as_euler('ZYX')
        self.last_body_euler_ypr = body_euler_ypr

        current_state = np.ones(13) # current state for MPC
        current_state[0:3] = np.flip(body_euler_ypr) # theta ( roll, pitch, yaw )
        current_state[3:6] = self.current_robot_state.body_pos  # x, y, z
        current_state[6:9] = self.current_robot_state.body_angvel  # wx, wy, wz
        current_state[9:12] = self.current_robot_state.body_vel  # vx, vy, vz
        current_state[12] = -9.81      # g constant

        return current_state