import sys,os
sys.path.append(os.getcwd()+'/..')

import numpy as np
import RobotController.RobotKinematics as rkm

model = rkm.RobotKineticModel()

for i in range(10000):
    jnt_pos = np.random.uniform(0, 1, 3)
    jnt_pos = np.diag([np.pi/2, np.pi/2, np.pi/2]) @ jnt_pos + np.array([0, 0, -np.pi])
    if jnt_pos[2] > -0.1:
        jnt_pos[2] = -0.1
    if jnt_pos[2] < -2.7:
        jnt_pos[2] = -2.7

    #jnt_pos = np.array([0, 0, -0.1])
    jnt_vel = np.random.rand(3)
    tip_pos_hip, tip_vel_hip, Jac, tsag = model.leg_fk(model.LID['FR'], jnt_pos, jnt_vel)
    if tsag[2] > 0:
        continue

    jnt_pos_valid, jnt_vel_valid, Jac_valid, err = model.leg_ik(model.LID['FR'], tip_pos_hip, tip_vel_hip)


    #print(jnt_pos_valid)
    #print(jnt_vel_valid)
    #print(Jac)
    #print(tip_pos_hip)
    #print(err)
    print('Test %d' % i)
    print(np.linalg.norm(jnt_pos - jnt_pos_valid))
    print(np.linalg.norm(jnt_vel - jnt_vel_valid))

    if np.linalg.norm(jnt_pos - jnt_pos_valid) > 1e-5:
        print(jnt_pos)
        print(jnt_pos_valid)
        print(tip_pos_hip)
        print(tsag)
        break

