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
