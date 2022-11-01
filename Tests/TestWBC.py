import sys,os
sys.path.append(os.getcwd()+'/..')
import numpy as np
from RobotController import RobotWBC as ctrl


wbc = ctrl.QuadSingleBodyWBC(1./1000.)

body_orn = np.array([0,0,0,1])
r_leg = np.array([0.183, -0.047, 0,
                  0.183,  0.047, 0,
                 -0.183, -0.047, 0,
                 -0.183,  0.047, 0])

support_state = np.array([1,0,1,0])
x_ref = np.array([0,0,0,
                  0,0,0,
                  0,0,0,
                  0,0,0])

x0 = np.array([0.2,0.02,0, # R, P, Y
               0,0,0, # x, y, z
               0,0,0, # wx, wy, wz
               0,0,0 # vx, vy, vz
               ]) 

wbc.update_wbc_matrices(
    body_orn,
    r_leg, 
    support_state,
    x0, 
    x_ref
)

print(wbc.H)
print(wbc.G)
print(wbc.C)
print(wbc.c)
print(wbc.D)
print(wbc.d)

ret = wbc.solve()
print(ret)