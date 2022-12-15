import sys,os
sys.path.append(os.getcwd()+'/..')
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as rot
from RobotController import RobotKinematics as rkin
from RobotController import RobotFullStateEstimate as rse


dt = 1/100.
w = np.array([100,29,-3.34])

print(np.linalg.norm(expm(rse.skew(w * dt)) - rse.gamma0(w, dt)))
print(rse.gamma0(w,dt))
print(rse.gamma1(w,dt))
print(rse.gamma2(w,dt))
print(rse.gamma3(w,dt))


jnt_pos = np.array([-0.0, 0.75, -1.3,
                    -0.0, 0.75, -1.3,
                    -0.0, 0.75, -1.3,
                    -0.0, 0.75, -1.3])

kin = rkin.RobotKineticModel()
kin.update_leg(jnt_pos, np.zeros(12))

est = rse.QuadFullStateEstimator(0.001)
est.reset_state(np.array([0, 0, 1.]), np.zeros(3), np.array([0,0,0,1.]), kin.get_tip_state_world()[0])

est.update(kin,
           np.array([0, 0, 0.1]), 
           np.array([1, 2, -9.8]), 
           jnt_pos, 
           0 * jnt_pos, 
           0 * jnt_pos, 
           np.array([0,1,1,0]), 
           np.array([0, 0.5, 0.5, 0]))

print(est.get_results())

w = np.array([0, 0, 10.])
q0 = np.array([0, 0, 0, 1.])
dt = 0.1
dq = rse.so3_to_quat(w * dt)
print(dq)
q1 = rse.quat_prod(dq, q0)
print(q1)
rpy_q1 = rot.from_quat(q1).as_euler('ZYX')
print(rpy_q1)
print(rot.from_matrix(rse.gamma0(w, dt)).as_euler('ZYX'))
