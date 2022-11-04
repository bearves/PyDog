import os
import numpy as np
import pybullet as pb
import pybullet_data as pb_data
from scipy.spatial.transform import Rotation as rot
import matplotlib.pyplot as plt

from RobotController import RobotGaitControl

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

gait_controller = RobotGaitControl.QuadGaitController()
feedbacks = RobotGaitControl.QuadControlInput()
user_cmd = RobotGaitControl.QuadRobotCommands()

jnt_tgtpos0 = np.array([-0.0, 0.75, -1.35,
                        -0.0, 0.75, -1.35,
                        -0.0, 0.75, -1.35,
                        -0.0, 0.75, -1.35])
jnt_tgtvel0 = np.array([0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0])
jnt_tgtpos = jnt_tgtpos0.copy()
jnt_tgtvel = jnt_tgtvel0.copy()

count = 0
in_gait_start_time = 6.0
in_gait = False

# log
log_size = 10000
time_line = np.zeros(log_size)
tip_pos_wcs_log = np.zeros((log_size, 12))
qj_log = np.zeros((log_size, 12))
jnt_tgtpos_log = np.zeros((log_size, 12))

while 1:
    count += 1
    time_now = (count-1) * time_step

    # get joint states
    for i in range(act_jnt_num):
        jnt_state = pb.getJointState(dog, act_jnt_id[i])
        jnt_actpos[i] = jnt_state[0]
        jnt_actvel[i] = jnt_state[1]

    # get robot body states
    ret_pos = pb.getBasePositionAndOrientation(dog)
    body_pos = np.array(ret_pos[0])
    body_orn = np.array(ret_pos[1])
    ret_vel = pb.getBaseVelocity(dog)
    body_vel = np.array(ret_vel[0])
    body_angvel = np.array(ret_vel[1])

    feedbacks.time_now = time_now
    feedbacks.body_pos = body_pos.copy()
    feedbacks.body_vel = body_vel.copy()
    feedbacks.body_orn = body_orn.copy()
    feedbacks.body_angvel = body_angvel.copy()
    feedbacks.jnt_act_pos = jnt_actpos.copy()
    feedbacks.jnt_act_vel = jnt_actvel.copy()

    if not in_gait and time_now > in_gait_start_time:
        in_gait = True
        gait_controller.load(feedbacks)
        # pb.createConstraint(gnd, -1,  dog, -1, pb.JOINT_PRISMATIC, [0,0,1], body_pos, [0,0,0])

    if in_gait:
        output = gait_controller.control_step(feedbacks, user_cmd)
        jnt_tgttrq = output.joint_tgt_trq.copy()
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
        user_cmd.gait_switch = True

    if ord('m') in keys and keys[ord('m')] & pb.KEY_WAS_TRIGGERED:
        print("plot data")
        plt.plot(time_line, qj_log[:,2], time_line, jnt_tgtpos_log[:,2])
        plt.grid(True)
        plt.legend(['qj','jnt_tgtpos'])
        plt.show()

    user_cmd.direction_flag = [0, 0, 0, 0, 0, 0]
    if pb.B3G_UP_ARROW in keys and keys[pb.B3G_UP_ARROW] & pb.KEY_IS_DOWN:
        user_cmd.direction_flag[0] = 1
    if pb.B3G_DOWN_ARROW in keys and keys[pb.B3G_DOWN_ARROW] & pb.KEY_IS_DOWN:
        user_cmd.direction_flag[1] = 1
    if pb.B3G_LEFT_ARROW in keys and keys[pb.B3G_LEFT_ARROW] & pb.KEY_IS_DOWN:
        user_cmd.direction_flag[2] = 1
    if pb.B3G_RIGHT_ARROW in keys and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_IS_DOWN:
        user_cmd.direction_flag[3] = 1
    if ord('q') in keys and keys[ord('q')] & pb.KEY_IS_DOWN:
        user_cmd.direction_flag[4] = 1
    if ord('e') in keys and keys[ord('e')] & pb.KEY_IS_DOWN:
        user_cmd.direction_flag[5] = 1

    # log data
    time_line[count % log_size] = time_now
    tip_pos_wcs_log[count % log_size, :] = gait_controller.tip_ref_pos
    qj_log[count % log_size, :] = gait_controller.jnt_ref_pos_wbic
    jnt_tgtpos_log[count % log_size, :] = jnt_tgtpos

    #time.sleep(time_step)
    cam_pos = body_pos.copy()
    cam_pos[2] = 0.2
    pb.resetDebugVisualizerCamera(1, 180, -10, cam_pos)

    
