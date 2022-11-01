import numpy as np
import numpy.matlib as mtl
from scipy.spatial.transform import Rotation as Rot
import math

class RobotKineticModel(object):
    # leg id def
    LID = {'FR' : 0, 'FL' : 1, 'RR' : 2, 'RL' : 3}

    # robot physical parameters
    NLEG = 4    # number of legs
    HBL = 0.183 # half body length
    HBW = 0.047 # half body width
    HO  = 0.08505 # hip offset
    LU  = 0.2   # upper leg length
    LL  = 0.2   # lower leg length
    #COM_OFFSET = np.array([0.012731, 0.002186, 0.000515]) # offset of the body's COM to its origin in the URDF
    COM_OFFSET = np.zeros(3)

    hip_pos_ = np.zeros(3 * NLEG)
    hip_R_ = np.zeros((NLEG, 3, 3))

    # robot state in wcs
    body_pos_ = np.zeros(3)
    body_vel_ = np.zeros(3)
    body_orn_ = np.array([0, 0, 0, 1])
    body_angvel_ = np.zeros(3)
    jnt_pos_ = np.zeros(3 * NLEG)
    jnt_vel_ = np.zeros(3 * NLEG)

    body_R_ = np.eye(3)
    body_euler_ = np.zeros(3)

    tip_pos_hip_ = np.zeros(3 * NLEG)
    tip_vel_hip_ = np.zeros(3 * NLEG)
    tip_pos_body_ = np.zeros(3 * NLEG)
    tip_vel_body_ = np.zeros(3 * NLEG)
    tip_pos_world_ = np.zeros(3 * NLEG)
    tip_vel_world_ = np.zeros(3 * NLEG)

    leg_jac = np.zeros((NLEG, 3, 3))

    def __init__(self):
        HBL = self.HBL
        HBW = self.HBW
        # assign hip pos of each leg
        self.hip_pos_ = np.array([ HBL, -HBW, 0, # FR
                                   HBL,  HBW, 0, # FL
                                  -HBL, -HBW, 0, # RR
                                  -HBL,  HBW, 0])# RL
        self.hip_pos_ = self.hip_pos_ - mtl.repmat(self.COM_OFFSET, 4, 1).flatten()

        # assign hip rotation matrix of each leg
        self.hip_R_[self.LID['FR'],:,:] = np.eye(3)
        self.hip_R_[self.LID['FL'],:,:] = np.eye(3)
        self.hip_R_[self.LID['RR'],:,:] = np.eye(3)
        self.hip_R_[self.LID['RL'],:,:] = np.eye(3)

        # assign leg symmetry flag of each leg, right leg = 1, left leg = -1
        self.leg_sym_flag_ = np.array([1, -1, 1, -1])

    def update(self, body_pos, body_orn, body_vel, body_angvel, jnt_actpos, jnt_actvel):
        self.body_pos_ = body_pos
        self.body_vel_ = body_vel
        self.body_orn_ = body_orn
        self.body_angvel_ = body_angvel
        self.jnt_pos_ = jnt_actpos
        self.jnt_vel_ = jnt_actvel

        self.body_R_ = Rot.from_quat(body_orn).as_matrix()
        self.body_euler_ = Rot.from_quat(body_orn).as_euler('ZYX')

        # update leg forward kinetics
        for i in range(self.NLEG):
            idx = range(0+i*3, 3+i*3)
            self.tip_pos_hip_[idx], self.tip_vel_hip_[idx], self.leg_jac[i, :, :], _ \
                = self.leg_fk(i, jnt_actpos[idx], jnt_actvel[idx])

            self.tip_pos_body_[idx] = self.hip_pos_[idx] + self.hip_R_[i,:,:] @ self.tip_pos_hip_[idx]
            self.tip_vel_body_[idx] = self.hip_R_[i,:,:] @ self.tip_vel_hip_[idx]

            self.tip_pos_world_[idx] = body_pos + self.body_R_ @ self.tip_pos_body_[idx]
            self.tip_vel_world_[idx] = body_vel + self.body_R_ @ self.tip_vel_body_[idx] + \
                                       np.cross(body_angvel, self.body_R_ @ self.tip_pos_body_[idx])
    
    def get_contact_jacobian_wcs(self, leg_id):
        idx = range(0+leg_id*3, 3+leg_id*3)
        ri = self.body_R_ @ self.tip_pos_body_[idx]
        rx, ry, rz = ri[0], ri[1], ri[2]
        jori = np.array([[  0, -rz,  ry],
                         [ rz,   0, -rx],
                         [-ry,  rx,   0]])
        jleg = self.body_R_ @ self.hip_R_[leg_id,:,:] @ self.leg_jac[leg_id,:,:]
        jc = np.zeros((3, 6 + self.NLEG * 3))

        # convert to body cs, as the same to the definitions used in Pinocchio and MIT codes
        jc[0:3,0:3] = np.eye(3) @ self.body_R_
        jc[0:3,3:6] = -jori @ self.body_R_
        jc[0:3,6+leg_id*3:9+leg_id*3] = jleg
        return jc
    
    
    def whole_body_ik(self, tip_pos_world, tip_vel_world):

        tip_pos_body = np.zeros(3 * self.NLEG)
        tip_pos_hip  = np.zeros(3 * self.NLEG)
        tip_vel_body = np.zeros(3 * self.NLEG)
        tip_vel_hip  = np.zeros(3 * self.NLEG)
        jnt_pos = np.zeros(3 * self.NLEG)
        jnt_vel = np.zeros(3 * self.NLEG)

        for i in range(self.NLEG):
            idx = range(0+i*3, 3+i*3)
            tip_pos_body[idx] = self.body_R_.T @ (tip_pos_world[idx] - self.body_pos_)
            tip_vel_body[idx] = self.body_R_.T @ (tip_vel_world[idx] +
                                                  -np.cross(self.body_angvel_, self.body_R_ @tip_pos_body[idx]) +
                                                  -self.body_vel_)

            tip_pos_hip[idx] = self.hip_R_[i, :, :].T @ (tip_pos_body[idx] - self.hip_pos_[idx])
            tip_vel_hip[idx] = self.hip_R_[i, :, :].T @ tip_vel_body[idx]

            jnt_pos[idx], jnt_vel[idx], _, _ = self.leg_ik(i, tip_pos_hip[idx], tip_vel_hip[idx])
        return jnt_pos, jnt_vel


    def leg_fk(self, leg_id, jnt_pos, jnt_vel):
        sym_flag = self.leg_sym_flag_[leg_id]
        th1, th2, th3 = jnt_pos[0], jnt_pos[1], jnt_pos[2]
        LU = self.LU
        LL = self.LL
        HO = self.HO * sym_flag

        # position fk
        tsag = np.array([
            -LU*np.sin(th2)-LL*np.sin(th2+th3),
            -HO,
            -LU*np.cos(th2)-LL*np.cos(th2+th3)
        ])
        Rx = Rot.from_rotvec([th1, 0, 0]).as_matrix()
        tip_pos_hip = Rx @ tsag

        # jacobian mat
        dRxdth3 = np.array([[0,            0,            0],
                            [0, -np.sin(th1), -np.cos(th1)],
                            [0,  np.cos(th1), -np.sin(th1)]])
        dtsagdth2 = np.array([
            -LU*np.cos(th2)-LL*np.cos(th2+th3),
            0,
             LU*np.sin(th2)+LL*np.sin(th2+th3)
        ])
        dtsagdth3 = np.array([
            -LL*np.cos(th2+th3),
            0,
             LL*np.sin(th2+th3)
        ])
        Jac = np.eye(3)
        Jac[:, 0] = dRxdth3 @ tsag
        Jac[:, 1] = Rx @ dtsagdth2
        Jac[:, 2] = Rx @ dtsagdth3

        # velocity fk
        tip_vel_hip = Jac @ jnt_vel

        return tip_pos_hip, tip_vel_hip, Jac, tsag


    def leg_ik(self, leg_id, tip_pos_hip, tip_vel_hip):

        sym_flag = self.leg_sym_flag_[leg_id]
        LU = self.LU
        LL = self.LL
        HO = self.HO * sym_flag

        px = tip_pos_hip[0]
        py = tip_pos_hip[1]
        pz = tip_pos_hip[2]

        zsagsq = py**2 + pz**2 - HO**2
        if zsagsq < 0.0001:
            zsagsq = 0.0001
            print('Warning: zsag**2 < 0')
        zsag = -math.sqrt(zsagsq)
        thr = math.atan2(zsag, HO)
        th1 = thr - math.atan2(pz, -py)

        costh3 = (px**2 + zsag**2 - LU**2 - LL**2)/2./LL/LU
        costh3 = np.clip(costh3, -0.99, 0.99)
        th3 = -math.acos(costh3)

        thn = math.atan2(-LU-LL*math.cos(th3), LL*math.sin(th3))
        th2 = math.atan2(zsag, -px) - thn

        jnt_pos = np.array([th1, th2, th3])
        tip_pos_hip_valid, _, Jac, _ = self.leg_fk(leg_id, jnt_pos, np.zeros(3))
        jnt_vel = np.linalg.inv(Jac) @ tip_vel_hip

        err = np.linalg.norm(tip_pos_hip_valid - tip_pos_hip)

        return jnt_pos, jnt_vel, Jac, err


    def get_tip_state_world(self):
        return self.tip_pos_world_, self.tip_vel_world_

    def get_tip_state_body(self):
        return self.tip_pos_body_, self.tip_vel_body_

    def get_tip_state_hip(self):
        return self.tip_pos_hip_, self.tip_vel_hip_

    def get_hip_pos_body_with_offset(self):
        hip_pos_offset = self.hip_pos_.copy()
        for i in range(self.NLEG):
            hip_pos_offset[1+i*3] += -self.HO * self.leg_sym_flag_[i]
        return hip_pos_offset

    def get_joint_trq(self, tip_force_wcs):
        tau = np.zeros(12)
        for i in range(self.NLEG):
            tip_force_body = self.body_R_.T @ tip_force_wcs[0+i*3:3+i*3]
            tip_force_hip = self.hip_R_[i, :, :] @ tip_force_body
            tau[0+i*3:3+i*3] = self.leg_jac[i, :, :].T @ tip_force_hip
        return tau
