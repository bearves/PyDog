import sys,os
sys.path.append(os.getcwd()+'/..')
import numpy as np
from scipy.linalg import expm
from RobotController import RobotStateEstimate as rse


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
est = rse.QuadStateEstimator(0.001)
est.reset_state(np.array([0, 0, 1]), np.zeros(3), np.array([0,0,0,1]), jnt_pos)

est.update(np.array([0, 0, 0.1]), np.array([1, 2, -9.8]), jnt_pos, 0 * jnt_pos, 0 * jnt_pos, np.array([0,1,1,0]))

print(est.get_results())