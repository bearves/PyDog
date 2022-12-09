import numpy as np
from scipy.spatial.transform import Rotation as rot

import pybullet as pb
from Bridges.BridgeToPybullet import BridgeToPybullet

from RobotController import RobotGaitControl as rc
from RobotController.RobotSteer import DirectionFlag

np.set_printoptions(precision=4, suppress=True)

# initialize the simulator parameters, GUI and others
urdf_file = r'./models/a1/a2_pino.urdf'
mesh_dir  = r'./models/a1/'
bridge = BridgeToPybullet(4, urdf_file)

cnt = 0
in_gait_start_time = 3.0
in_gait = False

# load controller
Rs_bcs = np.eye(3)
gait_controller = rc.QuadGaitController(
    bridge.dt, bridge.leap, urdf_file, mesh_dir, 
    use_mpc=True, use_se=True, Rs_bcs=Rs_bcs)
        
feedbacks = rc.QuadControlInput()
output = rc.QuadControlOutput()
user_cmd = rc.QuadRobotCommands()

while 1:
    # Read the sensors
    bridge.read_sensor_feedbacks(cnt)

    # set inputs of controller
    feedbacks.time_now = cnt * bridge.dt
    feedbacks.body_pos = bridge.body_act_pos.copy()
    feedbacks.body_orn = bridge.body_act_orn.copy()
    feedbacks.body_vel = bridge.body_act_vel.copy()
    feedbacks.body_angvel = bridge.body_act_angvel.copy()
    feedbacks.jnt_act_pos = bridge.jnt_act_pos.copy()
    feedbacks.jnt_act_vel = bridge.jnt_act_vel.copy()
    feedbacks.snsr_acc = bridge.snsr_acc.copy()
    feedbacks.snsr_gyr = bridge.snsr_gyr.copy()
    if feedbacks.body_orn[3] < 0:
        feedbacks.body_orn *= -1.

    if not in_gait and feedbacks.time_now > in_gait_start_time:
        in_gait = True
        bridge.remove_init_constraint()
        gait_controller.load(feedbacks)

    if in_gait:
        output = gait_controller.control_step(feedbacks, user_cmd)
        jnt_ref_trq = output.joint_tgt_trq.copy()
    else:
        kp, kd = 20, 0.4
        pos_err = bridge.jnt_init_pos - feedbacks.jnt_act_pos
        vel_err = bridge.jnt_init_vel - feedbacks.jnt_act_vel
        jnt_ref_trq = kp * pos_err + kd * vel_err

    # set commands
    bridge.set_motor_command(jnt_ref_trq)

    # save data for analysis
    if in_gait:
        bridge.save_data(output.support_state, output.support_phase)
    
    # proceed simulation 
    bridge.step()

    # handle user inputs
    key = bridge.read_user_inputs()

    if bridge.is_key_trigger(ord('t')):
        print("switch gait to trot")
        user_cmd.gait_switch = True

    user_cmd.direction_flag = [0, 0, 0, 0, 0, 0]
    if bridge.is_key_down(pb.B3G_UP_ARROW):
        user_cmd.direction_flag[DirectionFlag.FORWARD] = 1
    if bridge.is_key_down(pb.B3G_DOWN_ARROW):
        user_cmd.direction_flag[DirectionFlag.BACKWARD] = 1
    if bridge.is_key_down(pb.B3G_LEFT_ARROW):
        user_cmd.direction_flag[DirectionFlag.MOVE_LEFT] = 1
    if bridge.is_key_down(pb.B3G_RIGHT_ARROW):
        user_cmd.direction_flag[DirectionFlag.MOVE_RIGHT] = 1
    if bridge.is_key_down(ord('q')):
        user_cmd.direction_flag[DirectionFlag.TURN_LEFT] = 1
    if bridge.is_key_down(ord('e')):
        user_cmd.direction_flag[DirectionFlag.TURN_RIGHT] = 1

    #track robot
    bridge.track_robot()

    cnt += 1

    
