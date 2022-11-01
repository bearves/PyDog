import sys,os
sys.path.append(os.getcwd()+'/..')

from scipy.spatial.transform import Rotation as Rot
import numpy as np

from RobotController import RobotMPC as ctrl
from RobotController import RobotWBIC as rwbic
from RobotController import RobotDynamics as rdyn
from RobotController import RobotKinematics as rkin

np.set_printoptions(precision=4, suppress=True)

mpc = ctrl.QuadConvexMPC(1.0/1000.)
kin = rkin.RobotKineticModel()

##################
# actual pos
##################
body_pos_act = np.array([0.0, -0.0, 0.01+0.30])
body_ori_act = Rot.from_rotvec([0, 0.02, 0.1]).as_quat()
body_rpy_act = np.flip(Rot.from_quat(body_ori_act).as_euler('ZYX'))
body_vel_act = np.zeros(3)
body_angvel_act = np.zeros(3)

support_state_act = np.array([1,0,0,1])

# just use for update kinematic model
jnt_pos_act = np.array([-0.0, 0.75, -1.3,
                        -0.0, 0.75, -1.3,
                        -0.0, 0.75, -1.3,
                        -0.0, 0.75, -1.3])
jnt_vel_act = np.array([0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0])

kin.update(body_pos_act, body_ori_act, body_vel_act, body_angvel_act, jnt_pos_act, jnt_vel_act)

leg_tip_pos_wcs_act, leg_tip_vel_wcs_act = kin.get_tip_state_world()

print('Init tip pos:')
print(leg_tip_pos_wcs_act)
print('Init tip vel:')
print(leg_tip_vel_wcs_act)

##################
# ref pos
##################
body_pos_ref = np.array([0.0, -0.0, 0.30])
body_ori_ref = Rot.from_rotvec([0, 0, 0.1]).as_quat()
body_rpy_ref = np.flip(Rot.from_quat(body_ori_ref).as_euler('ZYX'))
body_vel_ref = np.zeros(3)
body_angvel_ref = np.zeros(3)
body_acc_ref = np.zeros(3)
body_angacc_ref = np.zeros(3)

leg_tip_pos_wcs_ref = leg_tip_pos_wcs_act.copy()
leg_tip_vel_wcs_ref = leg_tip_vel_wcs_act.copy()
leg_tip_acc_wcs_ref = np.zeros(12)

##################
# MPC
##################

body_euler_list = np.zeros((mpc.horizon_length, 3))
r_leg_list = np.zeros((mpc.horizon_length, 12))
support_state_list = np.zeros((mpc.horizon_length, 4))
x_ref_list = np.zeros((mpc.horizon_length, mpc.dim_s))

x0 = np.zeros(13)
x0[0:3] = body_rpy_act
x0[3:6] = body_pos_act
x0[6:9] = body_angvel_act
x0[9:12] = body_vel_act
x0[12] = -9.81

x_ref = np.zeros(13)
x_ref[0:3] = body_rpy_ref
x_ref[3:6] = body_pos_ref
x_ref[6:9] = body_angvel_ref
x_ref[9:12] = body_vel_ref
x_ref[12] = -9.81

for i in range(mpc.horizon_length):
    body_euler_list[i, :] = body_rpy_act
    r_leg_list[i, :] = leg_tip_pos_wcs_act
    support_state_list[i, :] = support_state_act
    x_ref_list[i, :] = x_ref

mpc.cal_weight_matrices()
mpc.update_mpc_matrices(body_euler_list, r_leg_list, support_state_list, x0, x_ref_list)
mpc.update_super_matrices()

u_mpc = mpc.solve()
print('MPC Result:')
print(u_mpc[:, 0])

##################
# WBIC
##################
urdf_file = r'../models/a1/a2_pino.urdf'
mesh_dir  = r'../models/a1/'
os.environ['mesh_dir'] = mesh_dir

mapper = rdyn.JointOrderMapper()
model = rdyn.RobotDynamicModel()
model.load_model(urdf_file, mesh_dir)
wbic = rwbic.QuadWBIC()

# firstly, we need to update dynamic model
jnt_pos_act_pino = mapper.convert_vec_to_pino(jnt_pos_act)
jnt_vel_act_pino = mapper.convert_vec_to_pino(jnt_vel_act)
support_state_act_pino = mapper.convert_sps_to_pino(support_state_act)
model.update(body_pos_act, body_ori_act, body_vel_act, body_angvel_act, jnt_pos_act_pino, jnt_vel_act_pino)
model.update_support_states(support_state_act_pino)

task_ref = np.zeros(7+12)
task_ref[0:3] = body_pos_ref # pos
task_ref[3:7] = body_ori_ref # ori
leg_tip_pos_wcs_ref_pino = mapper.convert_vec_to_pino(leg_tip_pos_wcs_ref)
task_ref[7:19] = leg_tip_pos_wcs_ref_pino # leg tip pos

task_dot_ref = np.zeros(6+12)
task_dot_ref[0:3] = body_vel_ref # pos
task_dot_ref[3:6] = body_angvel_ref # ori
leg_tip_vel_wcs_ref_pino = mapper.convert_vec_to_pino(leg_tip_vel_wcs_ref)
task_dot_ref[6:18] = leg_tip_vel_wcs_ref_pino # leg tip pos

task_ddot_ref = np.zeros(6+12)
task_ddot_ref[0:3] = body_acc_ref # pos
task_ddot_ref[3:6] = body_angacc_ref # ori
leg_tip_acc_wcs_ref_pino = mapper.convert_vec_to_pino(leg_tip_acc_wcs_ref)
task_ddot_ref[6:18] = leg_tip_acc_wcs_ref_pino # leg tip pos

u_mpc_pino = mapper.convert_vec_to_pino(u_mpc[:, 0])
# now we can run wbic
ret = wbic.update(model,
                  task_ref,
                  task_dot_ref,
                  task_ddot_ref,
                  u_mpc_pino
                  )
dq = mapper.convert_jvec_to_our(ret[0])
qdot = mapper.convert_jvec_to_our(ret[1])
tau_wbic = mapper.convert_jvec_to_our(ret[2])
print('WBIC Result')
print('dq =', dq)
print('qdot = ', qdot)
print('tau_wbic = ', tau_wbic)

# verify
tau_mpc = -kin.get_joint_trq(u_mpc[:, 0])
print('tau_mpc  = ', tau_mpc)
