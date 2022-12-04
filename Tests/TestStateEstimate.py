import sys,os
sys.path.append(os.getcwd()+'/..')
import numpy as np
from matplotlib import pyplot as plt
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

# read test data from files
data = np.loadtxt('log1.csv', delimiter=',', dtype=np.float64)
cnt = data[:, 0]
snsr_acc =        data[:, 1:4]
snsr_gyro =       data[:, 4:7]
snsr_pose =       data[:, 7:11]
jnt_act_pos =     data[:, 11:23]
jnt_act_vel =     data[:, 23:35]
jnt_ref_trq =     data[:, 35:47]
support_state =   data[:, 47:51]
support_phase =   data[:, 51:55]
body_act_pos =    data[:, 55:58]
body_act_vel =    data[:, 58:61]
body_act_orn =    data[:, 61:65]
body_act_angvel = data[:, 65:68]

plt.plot(cnt, support_phase)
plt.grid(True)
plt.show()

# convert sensor data from IMU local frame to body frame/world frame
