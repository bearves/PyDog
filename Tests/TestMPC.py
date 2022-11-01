import sys,os
sys.path.append(os.getcwd()+'/..')
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from RobotController import RobotMPC as ctrl

mpc = ctrl.QuadConvexMPC(1.0/240.)

body_rot = Rot.from_rotvec([0,0,0.1])
A, B = mpc.cal_state_equation(
        np.array([0, 0, 0.3]),
        np.flip(body_rot.as_euler('ZYX')),
        np.array([0.183, -0.047, 0,
         0.183,  0.047, 0,
        -0.183, -0.047, 0,
        -0.813,  0.047, 0]))

C, c, n_support = mpc.cal_friction_constraint(np.array([0,0,0,1]))
D, d, n_swing = mpc.cal_swing_force_constraints(np.array([0,1,0,1]))

#print(A)
#print(B)
#print(C)
#print(np.linalg.matrix_rank(C))
#print(n_support)
#print(D)
#print(np.linalg.matrix_rank(D))
#print(n_swing)

body_euler_list = np.zeros((mpc.horizon_length, 3))
r_leg_list = np.zeros((mpc.horizon_length, 12))
support_state_list = np.zeros((mpc.horizon_length, 4))
x_ref_list = np.zeros((mpc.horizon_length, mpc.dim_s))

x0 = np.array([0,0.02,0, # R, P, Y
               0,0,0, # x, y, z
               0,0,0, # wx, wy, wz
               0,0,0, # vx, vy, vz
               -9.81]) # gravity

for i in range(mpc.horizon_length):
    body_euler_list[i, :] = np.flip(body_rot.as_euler('ZYX'))
    r_leg_list[i, :] = np.array([0.183, -0.047, 0,
                                 0.183,  0.047, 0,
                                -0.183, -0.047, 0,
                                -0.183,  0.047, 0])
    support_state_list[i, :] = np.array([1,1,1,1])
    x_ref_list[i, :] = np.array([0,0,0,
                                 0,0,0,
                                 0,0,0,
                                 0,0,0,
                                 -9.81])

mpc.cal_weight_matrices()

mpc.update_mpc_matrices(body_euler_list, r_leg_list, support_state_list, x0, x_ref_list)
mpc.update_super_matrices()
#print(np.linalg.matrix_rank(mpc.Abar))
#print(np.linalg.matrix_rank(mpc.Bbar))
#print(mpc.Cbar.shape)

#print(mpc.H)
#print(mpc.G)


u_mpc = mpc.solve()
print('Result:')
print(u_mpc[:, 0])