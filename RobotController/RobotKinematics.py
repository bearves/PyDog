import math
import numpy as np
import numpy.matlib as mtl
from scipy.spatial.transform import Rotation as Rot


class RobotKineticModel(object):
    """
        Kinematic model of the quadruped robot.

        IMPORTANT NOTES:
        (1)
        The joint order follows the pybullet model, named as Our Order, defined as
                [abad upper lower]_FR
                [abad upper lower]_FL
                [abad upper lower]_RR
                [abad upper lower]_RL
    """
    # leg id definition
    LID: dict[str:int] = {'FR' : 0, 'FL' : 1, 'RR' : 2, 'RL' : 3}

    # robot physical parameters
    n_leg: int = 4    # number of legs
    HBL: float = 0.183 # half body length
    HBW: float = 0.047 # half body width
    HO:  float = 0.08505 # hip offset
    LU:  float = 0.2   # upper leg length
    LL:  float = 0.2   # lower leg length
    COM_OFFSET: np.ndarray = np.zeros(3)# offset of the body's COM to its origin in the URDF
    #COM_OFFSET: np.ndarray = np.array([0.012731, 0.002186, 0.000515]) 

    hip_pos_: np.ndarray = np.zeros(3 * n_leg)
    hip_R_  : np.ndarray = np.zeros((n_leg, 3, 3))

    # robot state in wcs
    body_pos_     : np.ndarray = np.zeros(3)
    body_vel_     : np.ndarray = np.zeros(3)
    body_orn_     : np.ndarray = np.array([0, 0, 0, 1])
    body_angvel_  : np.ndarray = np.zeros(3)
    jnt_pos_      : np.ndarray= np.zeros(3 * n_leg)
    jnt_vel_      : np.ndarray = np.zeros(3 * n_leg)

    body_R_       : np.ndarray = np.eye(3)
    body_euler_   : np.ndarray = np.zeros(3)

    tip_pos_hip_  : np.ndarray = np.zeros(3 * n_leg)
    tip_vel_hip_  : np.ndarray = np.zeros(3 * n_leg)
    tip_pos_body_ : np.ndarray = np.zeros(3 * n_leg)
    tip_vel_body_ : np.ndarray = np.zeros(3 * n_leg)
    tip_pos_world_: np.ndarray = np.zeros(3 * n_leg)
    tip_vel_world_: np.ndarray = np.zeros(3 * n_leg)

    leg_jac_: np.ndarray = np.zeros((n_leg, 3, 3))

    def __init__(self):
        """
            Create a robot kinematic model.
        """
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

    def update_leg(self, 
               jnt_actpos: np.ndarray, 
               jnt_actvel: np.ndarray):
        """
            Update joint states and calculate robot leg kinematics.

            Parameters:
                joint_actpos   (array(n_leg*3)): current joint position, in Our/Pybullet's order.
                joint_actvel   (array(n_leg*3)): current joint velocity, in Our/Pybullet's order.
        """

        self.jnt_pos_ = jnt_actpos.copy()
        self.jnt_vel_ = jnt_actvel.copy()

        # update leg forward kinetics
        for i in range(self.n_leg):
            idx = range(0+i*3, 3+i*3)
            self.tip_pos_hip_[idx], self.tip_vel_hip_[idx], self.leg_jac_[i, :, :], _ \
                = self.leg_fk(i, jnt_actpos[idx], jnt_actvel[idx])

            self.tip_pos_body_[idx] = self.hip_pos_[idx] + self.hip_R_[i,:,:] @ self.tip_pos_hip_[idx]
            self.tip_vel_body_[idx] = self.hip_R_[i,:,:] @ self.tip_vel_hip_[idx]
    

    def update_body(self, 
               body_pos: np.ndarray, 
               body_orn: np.ndarray, 
               body_vel: np.ndarray, 
               body_angvel: np.ndarray):
        """
            Update body states and calculate robot kinematics.
            This method must be called after update_leg().

            Parameters:
                body_pos       (array(3)): current position of the body center, in WCS.
                body_orn       (array(4)): current orientation in quaternion of the body, in WCS.
                body_vel       (array(3)): current linear velocity of the body center, in WCS.
                body_angvel    (array(3)): current angular velocity of the body, in WCS.
        """

        self.body_pos_ = body_pos.copy()
        self.body_vel_ = body_vel.copy()
        self.body_orn_ = body_orn.copy()
        self.body_angvel_ = body_angvel.copy()

        self.body_R_ = Rot.from_quat(body_orn).as_matrix()
        self.body_euler_ = Rot.from_quat(body_orn).as_euler('ZYX')

        # update leg forward kinetics
        for i in range(self.n_leg):
            idx = range(0+i*3, 3+i*3)
            self.tip_pos_world_[idx] = body_pos + self.body_R_ @ self.tip_pos_body_[idx]
            self.tip_vel_world_[idx] = body_vel + self.body_R_ @ self.tip_vel_body_[idx] + \
                                       np.cross(body_angvel, self.body_R_ @ self.tip_pos_body_[idx])

    
    def get_contact_jacobian_wcs(self, leg_id: int) -> np.ndarray:
        """
            Calculate the contact jacobian of the given leg.
                    v_tip_leg_wcs = Jc * v
            Note that v is the generalized velocity, defined as
                    v = [vx vy vz wx wy wz vj1 vj2 ... vj12]
            [vx vy vz] and [wx wy wz] are the body linear and angular velocity, in Body local CS.
            [vj1, vj2, ... vj12] is the joint velocity, in Our/Pybullet order.

            Parameters:
                leg_id (int): index of the given leg.

            Returns:
                jc (array(3, 6+n_leg*3)): contact jacobian of the given leg
        """
        idx = range(0+leg_id*3, 3+leg_id*3)
        ri = self.body_R_ @ self.tip_pos_body_[idx]
        rx, ry, rz = ri[0], ri[1], ri[2]
        jori = np.array([[  0, -rz,  ry],
                         [ rz,   0, -rx],
                         [-ry,  rx,   0]])
        jleg = self.body_R_ @ self.hip_R_[leg_id,:,:] @ self.leg_jac_[leg_id,:,:]
        jc = np.zeros((3, 6 + self.n_leg * 3))

        # convert body generalized velocity from world cs to body cs, 
        # as the same to the definitions used in Pinocchio and MIT codes
        jc[0:3,0:3] = np.eye(3) @ self.body_R_
        jc[0:3,3:6] = -jori @ self.body_R_
        jc[0:3,6+leg_id*3:9+leg_id*3] = jleg
        return jc
    
    
    def whole_body_ik(self, 
                      tip_pos_wcs: np.ndarray, 
                      tip_vel_wcs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
            Calculate the inverse kinematics for the whole robot.

            Parameters:
                tip_pos_wcs (array(n_leg*3)): tip position of legs in WCS.
                tip_vel_wcs (array(n_leg*3)): tip velocity of legs in WCS.

            Returns:
                jnt_pos (array(n_leg*3)): joint position of legs, in Our order.
                jnt_vel (array(n_leg*3)): joint velocity of legs, in Our order.
        """

        tip_pos_body = np.zeros(3 * self.n_leg)
        tip_pos_hip  = np.zeros(3 * self.n_leg)
        tip_vel_body = np.zeros(3 * self.n_leg)
        tip_vel_hip  = np.zeros(3 * self.n_leg)
        jnt_pos = np.zeros(3 * self.n_leg)
        jnt_vel = np.zeros(3 * self.n_leg)

        for i in range(self.n_leg):
            idx = range(0+i*3, 3+i*3)
            tip_pos_body[idx] = self.body_R_.T @ (tip_pos_wcs[idx] - self.body_pos_)
            tip_vel_body[idx] = self.body_R_.T @ (tip_vel_wcs[idx] +
                                                  -np.cross(self.body_angvel_, self.body_R_ @tip_pos_body[idx]) +
                                                  -self.body_vel_)

            tip_pos_hip[idx] = self.hip_R_[i, :, :].T @ (tip_pos_body[idx] - self.hip_pos_[idx])
            tip_vel_hip[idx] = self.hip_R_[i, :, :].T @ tip_vel_body[idx]

            jnt_pos[idx], jnt_vel[idx], _, _ = self.leg_ik(i, tip_pos_hip[idx], tip_vel_hip[idx])
        return jnt_pos, jnt_vel


    def leg_fk(self, 
               leg_id: int, 
               jnt_pos: np.ndarray, 
               jnt_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
            Forward kinematics for a single leg.

            Parameters:
                leg_id (int): index of the given leg.
                jnt_pos (array(3)): joint position of the given leg, in Our order.
                jnt_vel (array(3)): joint velocity of the given leg, in Our order.

            Returns:
                tip_pos_hip (array(3)): tip position of the leg in hip CS, i.e. leg local CS.
                tip_vel_hip (array(3)): tip velocity of the leg in hip CS, i.e. leg local CS.
        """
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


    def leg_ik(self, 
               leg_id: int, 
               tip_pos_hip: np.ndarray, 
               tip_vel_hip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
            Inverse kinematics for a single leg.

            Parameters:
                leg_id (int): index of the given leg.
                tip_pos_hip (array(3)): tip position of the leg in hip CS, i.e. leg local CS.
                tip_vel_hip (array(3)): tip velocity of the leg in hip CS, i.e. leg local CS.

            Returns:
                jnt_pos (array(3)): joint position of the given leg, in Our order.
                jnt_vel (array(3)): joint velocity of the given leg, in Our order.
        """

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


    def get_tip_state_world(self) -> tuple[np.ndarray, np.ndarray]:
        """
            Get the tip positions and velocities of all legs, in WCS.

            Returns:
                tip_pos_world (array(3*n_leg)): tip position of the leg in WCS.
                tip_vel_world (array(3*n_leg)): tip velocity of the leg in WCS.
        """
        return self.tip_pos_world_, self.tip_vel_world_


    def get_tip_state_body(self) -> tuple[np.ndarray, np.ndarray]:
        """
            Get the tip positions and velocities of all legs, in Body CS.

            Returns:
                tip_pos_body (array(3*n_leg)): tip position of the leg in Body CS.
                tip_vel_body (array(3*n_leg)): tip velocity of the leg in Body CS.
        """
        return self.tip_pos_body_, self.tip_vel_body_


    def get_tip_state_hip(self) -> tuple[np.ndarray, np.ndarray]:
        """
            Get the tip positions and velocities of all legs, in Hip CS.

            Returns:
                tip_pos_hip (array(3*n_leg)): tip position of the leg in Hip CS.
                tip_vel_hip (array(3*n_leg)): tip velocity of the leg in Hip CS.
        """
        return self.tip_pos_hip_, self.tip_vel_hip_


    def get_hip_pos_body_with_offset(self) -> np.ndarray:
        """
            Get the hip position with the HIP OFFSET of all legs, in Body CS.
            The offset hip position is just above the tip position when the leg is standing upright. 
            This value is useful for predicting the next foothold.

            Returns:
                hip_pos_offset (array(3*n_leg)): hip positions with the HIP OFFSET.

        """
        hip_pos_offset = self.hip_pos_.copy()
        for i in range(self.n_leg):
            hip_pos_offset[1+i*3] += -self.HO * self.leg_sym_flag_[i]
        return hip_pos_offset


    def get_joint_trq(self, 
                      tip_force_wcs: np.ndarray) -> np.ndarray:
        """
            Get the joint torque based on static dynamics.
            For each leg, we have

                    tau = J_leg.T * R_hip.T * R_body.T * f_r

            where 

                tau is the joint torque of each leg,
                J_leg is the leg jacobian of each leg, mapping joint velocity to tip velocity in Hip CS,
                R_hip is the rotational matrix from Hip CS to Body CS,
                R_body is the rotational matrix from Body CS to WCS,
                f_r is the tip force in WCS. 

            Parameters:
                tip_force_wcs (array(n_leg*3)): tip force of all legs in WCS.

            Returns:
                tau (array(n_leg*3)): joint torque of all joints.

        """
        tau = np.zeros(self.n_leg * 3)
        for i in range(self.n_leg):
            tip_force_body = self.body_R_.T @ tip_force_wcs[0+i*3:3+i*3]
            tip_force_hip = self.hip_R_[i, :, :].T @ tip_force_body
            tau[0+i*3:3+i*3] = self.leg_jac_[i, :, :].T @ tip_force_hip
        return tau
