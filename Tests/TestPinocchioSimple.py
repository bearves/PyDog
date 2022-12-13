import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import mprint
from scipy.spatial.transform import Rotation as rot
#from pinocchio.visualize import MeshcatVisualizer

import sys,os
sys.path.append(os.getcwd()+'/..')

from RobotController import RobotKinematics as rkm

np.set_printoptions(precision=4, suppress=True)

urdf_file = r'../models/a1/fb.urdf'
mesh_dir  = r'../models/a1/'

# the order of legs and joints is DIFFERENT from the original urdf file!!!!! Thus different from the pybullet assignment
# the pybullet use COM pos as body pos, but pinocchio just use root pos as body pos
# so there is a small difference, but it dose not matter a lot

os.environ['mesh_dir'] = mesh_dir

robot = RobotWrapper.BuildFromURDF(urdf_file, mesh_dir, pin.JointModelFreeFlyer())

print('nq = %d, nv = %d' % (robot.nq, robot.nv))

root_pos = np.array([0,2,0])
root_ori = rot.from_euler('ZYX',[0.3, 0, 0.5]).as_quat()


q = pin.neutral(robot.model)
q[0:3] = root_pos
q[3:7] = root_ori
print(q) # x,y,z,qx,qy,qz,qw,qj1,qj2...,qj12

v = np.zeros(robot.nv)
v[0:3] = [1,0,0]

for i in range(robot.model.njoints):
    print(robot.model.names[i])

trunk_frame_id = 1
root_joint_id = 1

robot.forwardKinematics(q, v)
robot.computeJointJacobians(q)
robot.framesForwardKinematics(q)
#j = robot.getFrameJacobian(trunk_frame_id, rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
j = robot.getJointJacobian(root_joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

# Trick of cal Jdqd: Jdot*v = data.a[joint_id] after calling forwardKinematics(model,data,q,v,0*v)

print("\nFrame placements:")
for frame, oMf in zip(robot.model.frames, robot.data.oMf ):
    print(("{:<24} : {: .4f} {: .4f} {: .4f}"
          .format( frame.name, *oMf.translation.T.flat )))

print(j.shape)
print(j[0:3,0:6])

print('Joint velocity:')
print(robot.data.v[root_joint_id])
print((j @ v))
print('Frame velocity')
trunk_vel = pin.getFrameVelocity(robot.model, robot.data, trunk_frame_id, pin.ReferenceFrame.WORLD)
print(trunk_vel)

# Meshcat is required to run visualizer
#robot.setVisualizer(MeshcatVisualizer())
#robot.initViewer()
#robot.loadViewerModel("pinocchio")
#robot.display(q)

#while True:
#    c = input()
