import math
import numpy as np
import matplotlib.pyplot as plt

import sys,os
sys.path.append(os.getcwd()+'/..')

from RobotController import RobotCurves as rcv

np.set_printoptions(precision=4, suppress=True)

h = 0.18
edge_coe = 1.5

start_point = np.array([0, 0, 0])
end_point = np.array([0.4, 0, 0])

kp_builder = rcv.BezierKeyPointBuilder()

key_pts = kp_builder.build_kp_stair(start_point, end_point, h, edge_coe)
n_pts = key_pts.shape[1]

bezier_curve = rcv.BezierCurve()
bezier_curve.set_key_points(key_pts)
print(bezier_curve.coe)
print(bezier_curve.key_pts)

sine_curve = rcv.SineCurve()
sine_curve.set_key_points(key_pts[:,0], key_pts[:,n_pts-1], h)

tspan = 1
t = np.linspace(0, tspan, 1001)

pos_bzr = np.zeros((3, len(t)))
vel_bzr = np.zeros((3, len(t)))
acc_bzr = np.zeros((3, len(t)))

pos_sin = np.zeros((3, len(t)))
vel_sin = np.zeros((3, len(t)))
acc_sin = np.zeros((3, len(t)))

for i in range(len(t)):
    
    time_ratio = t[i] / tspan
    time_ratio_dot = 1 / tspan

    r     = 0.5 * (1 - math.cos(math.pi * time_ratio))
    rdot  = 0.5 * math.sin(math.pi * time_ratio) * math.pi * time_ratio_dot
    rddot = 0.5 * (math.pi**2) * math.cos(math.pi*time_ratio) * (time_ratio_dot**2)

    pos_bzr[:, i], vel_bzr[:, i], acc_bzr[:, i] = bezier_curve.get_pva_at(r, rdot, rddot)
    pos_sin[:, i], vel_sin[:, i], acc_sin[:, i] = sine_curve.get_pva_at(r, rdot, rddot)


plt.subplot(2,4,1)
plt.plot(t, pos_bzr[0,:], t, pos_bzr[2,:])
plt.grid(True)
plt.subplot(2,4,2)
plt.plot(t, vel_bzr[0,:], t, vel_bzr[2,:])
plt.grid(True)
plt.subplot(2,4,3)
plt.plot(t, acc_bzr[0,:], t, acc_bzr[2,:])
plt.grid(True)
plt.subplot(2,4,4)
plt.plot(pos_bzr[0,:], pos_bzr[2,:], '.')
plt.grid(True)
plt.axis("equal")

plt.subplot(2,4,5)
plt.plot(t, pos_sin[0,:], t, pos_sin[2,:])
plt.grid(True)
plt.subplot(2,4,6)
plt.plot(t, vel_sin[0,:], t, vel_sin[2,:])
plt.grid(True)
plt.subplot(2,4,7)
plt.plot(t, acc_sin[0,:], t, acc_sin[2,:])
plt.grid(True)
plt.subplot(2,4,8)
plt.plot(pos_sin[0,:], pos_sin[2,:], '.')
plt.grid(True)
plt.axis("equal")

plt.show()