import sys,os
sys.path.append(os.getcwd()+'/..')
import numpy as np
from RobotController import RobotVMC as ctrl


vmc = ctrl.QuadSingleBodyVMC(1./1000.)

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

vmc.update_vmc_matrices(
    body_orn,
    r_leg, 
    support_state,
    x0, 
    x_ref
)

print(vmc.H)
print(vmc.G)
print(vmc.C)
print(vmc.c)
print(vmc.D)
print(vmc.d)

ret = vmc.solve()
print(ret)