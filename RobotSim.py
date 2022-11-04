import os
import numpy as np
import pybullet as pb
import pybullet_data as pb_data
from scipy.spatial.transform import Rotation as rot
import matplotlib.pyplot as plt

from RobotController import RobotKinematics as rkm
from RobotController import RobotMPC as rmpc
from RobotController import RobotWBC as rwbc
from RobotController import RobotTrjPlanner as tplan
from RobotController import GaitPatternGenerator as gpg
from RobotController import RobotSteer as rst
from RobotController import RobotDynamics as rdyn
from RobotController import RobotWBIC as rwbic

np.set_printoptions(precision=4, suppress=True)

# initialize the simulator parameters, GUI and others
time_step = 1./1000.
pb.connect(pb.GUI)
pb.resetDebugVisualizerCamera(1, 180, -12, [0.0, -0.0, 0.2])
pb.setTimeStep(time_step)
pb.setAdditionalSearchPath(pb_data.getDataPath())

# Load table and plane
init_body_euler_ypr = np.array([0, 0, 0])
last_body_euler_ypr = init_body_euler_ypr.copy()
start_orn = rot.from_euler('ZYX', init_body_euler_ypr).as_quat()
dog = pb.loadURDF("models/a1/a2_pino.urdf",
                  [0, 0, 0.5], start_orn, useFixedBase=False)
gnd = pb.loadURDF("plane.urdf", useFixedBase=True)
pb.setGravity(0, 0, -9.8)

# set joint control mode for bullet model 
jnt_num = pb.getNumJoints(dog)
for i in range(jnt_num):
    info = pb.getJointInfo(dog, i)
    joint_type = info[2]
    print(info[0], info[1], info[12])
    # unlock joints
    if joint_type == pb.JOINT_REVOLUTE:
        pb.setJointMotorControl2(dog, i, pb.VELOCITY_CONTROL, force=0)

# set active joints (remove fixed joints from the model)
act_jnt_num = 12
act_jnt_id = [1,  3,  4,  # FR
              6,  8,  9,  # FL
              11, 13, 14,  # RR
              16, 18, 19]  # RL
toe_link_id = [5, 10, 15, 20]

jnt_actpos = np.zeros(act_jnt_num)
jnt_actvel = np.zeros(act_jnt_num)
body_pos = np.zeros(3)
body_orn = np.array([0, 0, 0, 1])
body_vel = np.zeros(3)
body_angvel = np.zeros(3)

toe_pos_real = np.zeros(12)
toe_vel_real = np.zeros(12)
toe_pos_cal = np.zeros(12)
toe_vel_cal = np.zeros(12)

# setup gait pattern generator
stand_gait = gpg.GaitPatternGenerator(
    'stand', 0.5, np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]))
trot_gait = gpg.GaitPatternGenerator('trot',  0.3, np.array(
    [0, 0.5, 0.5, 0]), np.array([0.5, 0.5, 0.5, 0.5]))
current_gpg = stand_gait
gait_switch_cmd = False

# setup robot steer
robot_steer = rst.RobotSteer()

# setup robot kinematics
kinetic_model = rkm.RobotKineticModel()

# setup robot dynamics and mapper
urdf_file = r'./models/a1/a2_pino.urdf'
mesh_dir  = r'./models/a1/'
os.environ['mesh_dir'] = mesh_dir
dynamic_model = rdyn.RobotDynamicModel()
dynamic_model.load_model(urdf_file, mesh_dir)
mapper = rdyn.JointOrderMapper()

# setup trj planners
body_planner = tplan.BodyTrjPlanner(time_step)
foothold_planner = tplan.FootholdPlanner()
swing_traj_planner = [tplan.SwingTrjPlanner(),
                      tplan.SwingTrjPlanner(),
                      tplan.SwingTrjPlanner(),
                      tplan.SwingTrjPlanner()]

# setup wbc
wbc = rwbc.QuadSingleBodyWBC(time_step)

# setup mpc
mpc = rmpc.QuadConvexMPC(time_step)
mpc.cal_weight_matrices()
mpc_solve_count = 0

# setup wbic
wbic = rwbic.QuadWBIC()
tau_wbic = np.zeros(12)
qj_des = np.zeros(12)
qdotj_des = np.zeros(12)

tip_pos_wcs = np.zeros(12)
tip_vel_wcs = np.zeros(12)
jnt_tgtpos0 = np.array([-0.0, 0.75, -1.35,
                        -0.0, 0.75, -1.35,
                        -0.0, 0.75, -1.35,
                        -0.0, 0.75, -1.35])
jnt_tgtvel0 = np.array([0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0])
jnt_tgtpos = jnt_tgtpos0.copy()
jnt_tgtvel = jnt_tgtvel0.copy()

in_gait = False
in_gait_start_time = 2
u_mpc = np.zeros((12, mpc.horizon_length))
u_wbc = np.zeros(12)

count = 0

# log
log_size = 10000
time_line = np.zeros(log_size)
tip_pos_wcs_log = np.zeros((log_size, 12))
qj_log = np.zeros((log_size, 12))
jnt_tgtpos_log = np.zeros((log_size, 12))

while 1:
    count = count + 1
    time_now = count * time_step

    # get joint states
    for i in range(act_jnt_num):
        jnt_state = pb.getJointState(dog, act_jnt_id[i])
        jnt_actpos[i] = jnt_state[0]
        jnt_actvel[i] = jnt_state[1]

    for i in range(len(toe_link_id)):
        toe_state = pb.getLinkState(dog, toe_link_id[i],
                                    computeLinkVelocity=1,
                                    computeForwardKinematics=1)
        toe_pos_real[0 + i * 3:3 + i * 3] = toe_state[0]  # pos wrt world
        toe_vel_real[0 + i * 3:3 + i * 3] = toe_state[6]  # vel wrt world

    # get robot body states
    ret_pos = pb.getBasePositionAndOrientation(dog)
    body_pos = np.array(ret_pos[0])
    body_orn = np.array(ret_pos[1])
    ret_vel = pb.getBaseVelocity(dog)
    body_vel = np.array(ret_vel[0])
    body_angvel = np.array(ret_vel[1])

    kinetic_model.update(body_pos, body_orn, body_vel,
                         body_angvel, jnt_actpos, jnt_actvel)
    toe_pos_cal, toe_vel_cal = kinetic_model.get_tip_state_world()

    if not in_gait and time_now > in_gait_start_time:
        in_gait = True
        mpc_solve_count = 0
        # set initial state of gait generator
        current_gpg.set_start_time(time_now)
        # set initial body states as ref
        body_euler = rot.from_quat(body_orn).as_euler('ZYX')
        body_yaw = body_euler[0]
        body_planner.set_ref_state(
            body_pos,
            rot.from_rotvec([0, 0, body_yaw]).as_quat(),
            np.zeros(3),
            np.zeros(3))
        # set initial tip pos
        foothold_planner.init_footholds(toe_pos_cal)

        #pb.createConstraint(gnd, -1,  dog, -1, pb.JOINT_PRISMATIC, [0,0,1], body_pos, [0,0,0])

    if in_gait:
        mpc_solve_count += 1

        # switch gait generator if cmd received
        if gait_switch_cmd and current_gpg.gait_name != 'trot' and current_gpg.can_switch():
            current_gpg = trot_gait
            current_gpg.set_start_time(time_now)
            gait_switch_cmd = False

        # update gait pattern generator
        current_gpg.set_current_time(time_now)
        current_support_state = current_gpg.get_current_support_state()

        # update vel cmd for planners
        robot_steer.update_vel_cmd(body_orn)
        filtered_vel_cmd = robot_steer.get_vel_cmd_wcs()

        # calculate trajectory for body in wcs
        body_planner.update_ref_state(filtered_vel_cmd)

        # calculate foothold for legs
        foothold_planner.set_current_state(toe_pos_cal, current_support_state)
        foothold_planner.calculate_next_footholds(
            body_pos,
            body_orn,
            body_vel,
            filtered_vel_cmd,
            current_gpg.get_swing_time_left(),
            current_gpg.get_stance_duration(),
            kinetic_model.get_hip_pos_body_with_offset()
        )

        # calculate trajectory for legs in wcs
        for leg in range(4):
            idx = range(0+leg*3, 3+leg*3)
            if current_support_state[leg] == 1:  # stance
                tip_pos_wcs[idx] = foothold_planner.get_current_footholds(leg)
                tip_vel_wcs[idx] = np.zeros(3)
            else:  # swing
                swing_traj_planner[leg].set_start_point(
                    foothold_planner.get_liftup_leg_pos(leg))
                swing_traj_planner[leg].set_end_point(
                    foothold_planner.get_next_footholds(leg))
                # get swing time ratio
                swing_time_ratio, swing_time_ratio_dot = current_gpg.get_current_swing_time_ratio(
                    leg)
                # interpolate from last foothold to the next foothold
                tip_pos_wcs[idx], tip_vel_wcs[idx]\
                    = swing_traj_planner[leg].get_tip_pos_vel(swing_time_ratio, swing_time_ratio_dot)

        # calculate body control force for stance legs
        if mpc.need_solve(mpc_solve_count - 1):

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
                mpc.dt_mpc, mpc.horizon_length)

            # generate future body trajectory
            body_ref_traj = body_planner.predict_future_body_ref_traj(
                mpc.dt_mpc, mpc.horizon_length)
            body_future_euler = body_ref_traj[:, 0:3]

            # predict future foot position

            foothold_future_traj = foothold_planner.predict_future_foot_pos(mpc.dt_mpc,
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

            wbc.update_wbc_matrices(
                body_orn, toe_pos_cal, current_support_state, current_state[0:12], body_ref_traj[i, 0:12])
            u_wbc = wbc.solve()

        ###############################################
        # calculate wbic
        ###############################################
        # firstly, update dynamic model
        jnt_pos_act_pino = mapper.convert_vec_to_pino(jnt_actpos)
        jnt_vel_act_pino = mapper.convert_vec_to_pino(jnt_actvel)
        support_state_act_pino = mapper.convert_sps_to_pino(current_support_state)
        dynamic_model.update(body_pos, body_orn, body_vel, body_angvel, jnt_pos_act_pino, jnt_vel_act_pino)
        dynamic_model.update_support_states(support_state_act_pino)

        # then set ref position and velocity for tasks
        task_ref = np.zeros(7+12)
        #task_ref[0:3] = body_planner.ref_body_pos # pos
        task_ref[0:3] = body_pos # pos
        task_ref[3:7] = body_planner.ref_body_orn # ori
        leg_tip_pos_wcs_ref_pino = mapper.convert_vec_to_pino(tip_pos_wcs)
        task_ref[7:19] = leg_tip_pos_wcs_ref_pino # leg tip pos

        task_dot_ref = np.zeros(6+12)
        task_dot_ref[0:3] = body_planner.ref_body_vel # pos
        task_dot_ref[3:6] = body_planner.ref_body_angvel # ori
        leg_tip_vel_wcs_ref_pino = mapper.convert_vec_to_pino(tip_vel_wcs)
        task_dot_ref[6:18] = leg_tip_vel_wcs_ref_pino # leg tip pos

        task_ddot_ref = np.zeros(6+12)
        # TODO: Plan accelerations for them
        #task_ddot_ref[0:3] = body_acc_ref # pos
        #task_ddot_ref[3:6] = body_angacc_ref # ori
        #leg_tip_acc_wcs_ref_pino = mapper.convert_vec_to_pino(leg_tip_acc_wcs_ref)
        #task_ddot_ref[6:18] = leg_tip_acc_wcs_ref_pino # leg tip pos

        u_mpc_pino = mapper.convert_vec_to_pino(u_mpc[:, 0])
        # now we can run wbic
        ret = wbic.update(dynamic_model,
                          task_ref,
                          task_dot_ref,
                          task_ddot_ref,
                          u_mpc_pino
                          )
        dq = mapper.convert_jvec_to_our(ret[0])
        qj_des = jnt_actpos + dq
        qdotj_des = mapper.convert_jvec_to_our(ret[1])
        tau_wbic = mapper.convert_jvec_to_our(ret[2])

    # cal joint trq from MPC
    # tau = J*R*f
    #tau_mpc = -kinetic_model.get_joint_trq(u_mpc[:, 0])
    #tau_mpc = -kinetic_model.get_joint_trq(u_wbc)
    tau_mpc = tau_wbic


    # joint pd control
    if in_gait:
        jnt_tgtpos, jnt_tgtvel = kinetic_model.whole_body_ik(
            tip_pos_wcs, tip_vel_wcs)

    jnt_tgttrq = np.zeros(12)

    # TODO: check qj and qdot, why there is a large mismatch with jnt_tgtpos
    if in_gait:
        for leg in range(4):
            idx = range(0+leg*3, 3+leg*3)
            if current_support_state[leg] == 1:  # stance
                kp, kd = 10, 1
                pos_err = qj_des[idx]-jnt_actpos[idx]
                vel_err = qdotj_des[idx]-jnt_actvel[idx]

                jnt_tgttrq[idx] = kp * pos_err + kd * vel_err + tau_mpc[idx]  # just damp actual velocity
            else:  # swing
                kp, kd = 10, 1
                #print('Qj and jnt_tgtpos: swing leg', leg)
                #print('jnt_pos: ', jnt_tgtpos[idx])
                #print('Qj     : ', qj_des[idx])
                #print('jnt_vel: ', jnt_tgtvel[idx])
                #print('Qdotj  : ', qdotj_des[idx])
                #pos_err = jnt_tgtpos[idx]-jnt_actpos[idx]
                #vel_err = jnt_tgtvel[idx]-jnt_actvel[idx]
                pos_err = qj_des[idx]-jnt_actpos[idx]
                vel_err = qdotj_des[idx]-jnt_actvel[idx]

                jnt_tgttrq[idx] = kp * pos_err + kd * vel_err + tau_mpc[idx]
    else:
        kp, kd = 120, 2
        pos_err = jnt_tgtpos - jnt_actpos
        vel_err = jnt_tgtvel - jnt_actvel
        jnt_tgttrq = kp * pos_err + kd * vel_err

    max_trq = 50
    jnt_tgttrq = np.clip(jnt_tgttrq, -max_trq, max_trq)

    for i in range(act_jnt_num):
        pb.setJointMotorControl2(
            dog, act_jnt_id[i], pb.TORQUE_CONTROL, force=jnt_tgttrq[i])

    # keyboard event handler
    pb.stepSimulation()
    keys = pb.getKeyboardEvents()

    if ord('t') in keys and keys[ord('t')] & pb.KEY_WAS_TRIGGERED:
        print("switch gait to trot")
        gait_switch_cmd = True
    if ord('m') in keys and keys[ord('m')] & pb.KEY_WAS_TRIGGERED:
        print("plot data")
        plt.plot(time_line, qj_log[:,2], time_line, jnt_tgtpos_log[:,2])
        plt.grid(True)
        plt.legend(['qj','jnt_tgtpos'])
        plt.show()
    robot_steer.clear_direction_flag()
    if pb.B3G_UP_ARROW in keys and keys[pb.B3G_UP_ARROW] & pb.KEY_IS_DOWN:
        robot_steer.set_forward()
    if pb.B3G_DOWN_ARROW in keys and keys[pb.B3G_DOWN_ARROW] & pb.KEY_IS_DOWN:
        robot_steer.set_backward()
    if pb.B3G_LEFT_ARROW in keys and keys[pb.B3G_LEFT_ARROW] & pb.KEY_IS_DOWN:
        robot_steer.set_move_left()
    if pb.B3G_RIGHT_ARROW in keys and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_IS_DOWN:
        robot_steer.set_move_right()
    if ord('q') in keys and keys[ord('q')] & pb.KEY_IS_DOWN:
        robot_steer.set_turn_left()
    if ord('e') in keys and keys[ord('e')] & pb.KEY_IS_DOWN:
        robot_steer.set_turn_right()

    # log data
    time_line[count % log_size] = time_now
    tip_pos_wcs_log[count % log_size, :] = tip_pos_wcs
    qj_log[count % log_size, :] = qj_des
    jnt_tgtpos_log[count % log_size, :] = jnt_tgtpos

    #time.sleep(time_step)
    cam_pos = body_pos.copy()
    cam_pos[2] = 0.2
    pb.resetDebugVisualizerCamera(1, 180, -30, cam_pos)

    
