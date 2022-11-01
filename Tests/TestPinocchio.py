from pyexpat import model
import pinocchio as pin
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import mprint
from scipy.spatial.transform import Rotation as rot
from pinocchio.visualize import MeshcatVisualizer

import sys,os
sys.path.append(os.getcwd()+'/..')

from RobotController import RobotKinematics as rkm
from RobotController import RobotDynamics as rdm

np.set_printoptions(precision=4, suppress=True)

#############################################################################
# IMPORTANT NOTES:
# (1)
# the order of legs and joints in pinocchio is DIFFERENT from the original urdf file!!!!! 
# thus is also different from the pybullet assignment.
# the pybullet use COM pos as body pos, but pinocchio just use root pos as body pos.
# so there is a small difference, but it dose not matter a lot.
# (2)
# q is defined as q = [x,y,z,qx,qy,qz,qw,qj1,qj2...,qj12]
# x,y,z,qx,qy,qz,qw are defined in WORLD cs
# however, v is not just time derivatives of q
# in pinocchio, v = [bvx bvy bvz bwx bwy bwz qdj1 qdj2 qdj3 ... qdj12]
# vx,vy,vz,wx,wy,wz are defined in BODY cs!


##############################################################################
# load model
urdf_file = r'../models/a1/a2_pino.urdf'
mesh_dir  = r'../models/a1/'
os.environ['mesh_dir'] = mesh_dir

robot = RobotWrapper.BuildFromURDF(urdf_file, mesh_dir, pin.JointModelFreeFlyer())

print(robot.nq)
print(robot.nv)

dyn = rdm.RobotDynamicModel()
dyn.load_model(urdf_file, mesh_dir)

mapper = rdm.JointOrderMapper()

##############################################################################
# set q, v, a
jnt_pos = np.array([-0.0, 0.75, -1.3,
                    -0.0, 0.75, -1.3,
                    -0.0, 0.75, -1.3,
                    -0.0, 0.75, -1.3])
jnt_vel = np.random.rand(12)
jnt_acc = np.random.rand(12)

root_pos = np.array([0,2,0])
root_ori = rot.from_euler('ZYX',[0.5, 0, 0]).as_quat()
root_vel = np.random.rand(3)
root_angvel = np.random.rand(3)
root_acc = np.random.rand(3)
root_angacc = np.random.rand(3)

q = pin.neutral(robot.model)
q[7:robot.nq] = mapper.convert_vec_to_pino(jnt_pos)
q[0:3] = root_pos
q[3:7] = root_ori
print(q) 

v = np.zeros(robot.nv)
v[6:robot.nv] = mapper.convert_vec_to_pino(jnt_vel)
v[0:3] = root_vel
v[3:6] = root_angvel
#v[0:3] = [1,0,0]
print(v) 

a = np.zeros(robot.nv)
a[6:robot.nv] = mapper.convert_vec_to_pino(jnt_acc)
a[0:3] = root_acc
a[3:6] = root_angacc
print(a)

dyn.update(root_pos, root_ori, v[0:3], v[3:6],
           mapper.convert_vec_to_pino(jnt_pos),
           mapper.convert_vec_to_pino(jnt_vel))


##############################################################################
# get joint and frame info
for i in range(robot.model.njoints):
    print(robot.model.names[i])

toe_frame_id_lists = []
for i in range(robot.model.nframes):
    frame_name = robot.model.frames[i].name
    if ('toe_fixed' in frame_name):
        print('Frame id = %d, name = %s' % (i, frame_name))
        toe_frame_id_lists.append(i)
trunk_frame_id = 1
root_joint_id = 1


##############################################################################
# calculate jacobians of a leg tip, aka. contact jacobian
robot.forwardKinematics(q, v)
robot.computeJointJacobians(q)
pin.updateFramePlacements(robot.model,robot.data)

leg_test = 1
j = robot.getFrameJacobian(toe_frame_id_lists[mapper.leg_id_pino(leg_test)], rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
#j = robot.getJointJacobian(root_joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

print("\nFrame placements:")
for frame, oMf in zip(robot.model.frames, robot.data.oMf):
    print(("{:<24} : {: .4f} {: .4f} {: .4f}"
          .format( frame.name, *oMf.translation.T.flat)))
    print(oMf.translation)

print(j.shape)
print(j[0:3,:])
#print(j[0:3,6:9])
#print(j[0:3,9:12])
#print(j[0:3,12:15])
#print(j[0:3,15:18])

jcm = dyn.get_leg_pos_jacobian(mapper.leg_id_pino(leg_test))
print(jcm[0:3,0:6])

##############################################################################
# verification of jacobians
my_model = rkm.RobotKineticModel()
body_rm = rot.from_quat(root_ori).as_matrix()
trunk_com_pos = root_pos + body_rm @ my_model.COM_OFFSET
#print(trunk_com_pos - root_pos)
my_model.update(trunk_com_pos, root_ori, root_vel, root_angvel, jnt_pos, jnt_vel)
jc = my_model.get_contact_jacobian_wcs(leg_test)

print(jc[0:3,0:6])
print("JCM - JC")
print(jc - mapper.convert_mat_to_our(jcm))
#print(jc[0:3,6:9])
#print(jc[0:3,9:12])
#print(jc[0:3,12:15])
#print(jc[0:3,15:18])


##############################################################################
# calculate jacobians of body ori and pos
j_body = robot.getFrameJacobian(trunk_frame_id, rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
print('J_body_pos = ')
print(j_body[0:3, 0:6])
print('J_body_ori = ')
print(j_body[3:6, 0:6])
print('Body RM = ')
print(body_rm)

##############################################################################
# get Mass matrix M(q), Coriolis term C(q, qd) and Gravity term G(q)
# M(q)qdd + C(q,qd)qd + G(q) = Jc(q).T Fr + [0 tau].T 

# Mass term
M = pin.crba(robot.model, robot.data, q)
print('Mass matrix = ')
print(robot.data.M)
print(M.shape)
#print(robot.data.M.T - robot.data.M)

# Gravity term
g = pin.computeGeneralizedGravity(robot.model, robot.data, q)
print('Gravity term = ')
print(robot.data.g)
print(g.shape)

# Coriolis term
C = pin.computeCoriolisMatrix(robot.model, robot.data, q, v)
print('Coriolis matrix = ')
print(robot.data.C)
print(C.shape)


##############################################################################
# calculate Jcdot*v of leg tips using dynamics
# Trick of cal Jdqd: Jdot*v = a_frame after calling forwardKinematics(model,data,q,v,0*v)
print('Tip Jdot @ v')
Jdot = pin.frameJacobianTimeVariation(
    robot.model, robot.data, 
    q, v, toe_frame_id_lists[0], 
    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
print(Jdot @ v)

robot.forwardKinematics(q, v, np.zeros(robot.nv))
af = pin.getFrameAcceleration(
    robot.model, robot.data,
    toe_frame_id_lists[0],
    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
print(af.vector)
print(af.vector - Jdot @ v)

##############################################################################
# calculate Jdot*v of body lin vel using dynamics.
# actually, for the trunk, Jdot * v is always zero. 
print('Trunk Jdot @ v')
JdotTrunk = pin.frameJacobianTimeVariation(
    robot.model, robot.data, 
    q, v, trunk_frame_id, 
    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
print(JdotTrunk[0:6, 0:6])
print(v)
print(JdotTrunk[0:3,0:3] @ v[0:3])

robot.forwardKinematics(q, v, np.zeros(robot.nv))
afTrunk = pin.getFrameAcceleration(
    robot.model, robot.data,
    trunk_frame_id,
    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
print(afTrunk.vector)
print(afTrunk.vector - JdotTrunk @ v)

##############################################################################
# calculate torque using dynamics
tau = pin.rnea(robot.model, robot.data, q, v, a)
print(tau)
print(M @ a + C @ v + g - tau) # these two are equivalent

# since we have contact force, the actual joint torque should be
#  tau_ = M @ a + C @ v + g - Jc(q).T Fr

# calculate all Jc(q), must be called after robot.computeJointJacobians(q)
Jc_all = np.zeros((12, 18))
for i in range(4):
    J = robot.getFrameJacobian(
        toe_frame_id_lists[i], 
        rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    Jc_all[0+i*3:3+i*3,:] = J[0:3,:] # we only need translational part

Fr = np.array([0, 0, 15,
               0, 0, 15,
               0, 0, 15,
               0, 0, 15])
tau_correct = tau - Jc_all.T @ Fr
print(tau_correct)

##############################################################################
# model visulization
#robot.setVisualizer(MeshcatVisualizer())
#robot.initViewer()
#robot.loadViewerModel("pinocchio")
#robot.display(q)

#while True:
#    c = input()
