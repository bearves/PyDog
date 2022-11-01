import os
import numpy as np
from scipy.spatial.transform import Rotation as rot
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper


class RobotDynamicModel(object):
    """
        Dynamic model of the quadruped robot based on pinocchio library
    
        IMPORTANT NOTES:
        (1)
        The order of legs and joints in pinocchio is DIFFERENT from the original urdf file,
        thus is also different from the pybullet and our own kinematic model.
        (2)
        All inputs and outputs in this class follow the definition of pinocchio model. 
        Before calling the methods in this model, you SHOULD re-order the inputs, including
        q, v, a, fr, support_states by yourself. The returns, including Jc, M, C, g are
        also sujected to the pinocchio's order. You SHOULD re-order them if necessary.
        (3)
        The pybullet use COM pos as body pos, but pinocchio just use root pos as body pos.
        so there is a small difference, but it dose not matter a lot.
        (4)
        q is defined as q = [x,y,z,qx,qy,qz,qw,qj1,qj2...,qj12]
        x,y,z,qx,qy,qz,qw are defined in WORLD cs.
        however, v is not just time derivatives of q,
        in pinocchio, v = [bvx bvy bvz bwx bwy bwz qdj1 qdj2 qdj3 ... qdj12],
        vx,vy,vz,wx,wy,wz are defined in BODY cs.
    """

    n_legs: int = 4
    nq: int = 7 + 3*n_legs
    nv: int = 6 + 3*n_legs
    support_thres: float = 0.8

    q: np.ndarray = np.zeros(6 + n_legs * 3)
    body_R: np.ndarray = np.eye(3)
    robot: pin.RobotWrapper | None = None
    support_state: np.ndarray = np.zeros(n_legs)

    # data holders
    J_leg: np.ndarray = np.zeros((n_legs, 3, nv))
    M : np.ndarray = np.zeros((nv, nv))
    g : np.ndarray = np.zeros(nv)
    C : np.ndarray = np.zeros((nv, nv))

    # frame and joint id params, these params should be adjusted according to the URDF file
    body_frame_id: int = 1
    root_joint_id: int = 1
    # FL, FR, RL, RR, used inside pinocchio
    toe_frame_id_lists: list[int] = [11, 21, 31, 41]

    def __init__(self) -> None:
        self.q[3:7] = np.array([0, 0, 0, 1])  # unit quaternion


    def load_model(self, urdf_file: str, mesh_dir: str):
        """
            Load the quadruped robot model from URDF file.
        """
        os.environ['mesh_dir'] = mesh_dir
        self.robot = RobotWrapper.BuildFromURDF(
            urdf_file, mesh_dir,
            pin.JointModelFreeFlyer())

    def update(self,
               body_pos: np.ndarray,
               body_ori: np.ndarray,
               body_vel: np.ndarray,
               body_angvel: np.ndarray,
               joint_pos: np.ndarray,
               joint_vel: np.ndarray):
        """
            Update model states and calculate robot dynamics.
            q is defined as q = [x,y,z,qx,qy,qz,qw,qj1,qj2...,qj12]
                where x,y,z,qx,qy,qz,qw are defined in WORLD cs.
            However, v is not just time derivatives of q,
                in pinocchio, v = [bvx bvy bvz bwx bwy bwz qdj1 qdj2 qdj3 ... qdj12]
                where, vx,vy,vz,wx,wy,wz are defined in BODY cs.
        """

        # set configuration vector q and qdot of the body
        self.q = pin.neutral(self.robot.model)
        self.v = np.zeros(self.robot.nv)
        self.q[0:3] = body_pos
        self.q[3:7] = body_ori
        self.body_R = rot.from_quat(body_ori).as_matrix()

        self.v[0:3] = self.body_R.T @ body_vel
        self.v[3:6] = self.body_R.T @ body_angvel

        # set configuration vector q and qdot of the joints
        for i in range(3 * self.n_legs):
            self.q[7+i] = joint_pos[i]
            self.v[6+i] = joint_vel[i]

        # update forward kinematics
        self.robot.forwardKinematics(self.q, self.v)
        self.robot.computeJointJacobians(self.q)
        pin.updateFramePlacements(self.robot.model,self.robot.data)

        for leg in range(self.n_legs):
            self.J_leg[leg, :, :] = self.robot.getFrameJacobian(
                self.toe_frame_id_lists[leg],
                rf_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, :]

        # get Mass matrix M(q), Coriolis term C(q, qd) and Gravity term G(q)
        # M(q)qdd + C(q,qd)qd + G(q) = Jc(q).T Fr + [0 tau].T 

        # Mass term
        self.M = pin.crba(self.robot.model, self.robot.data, self.q)

        # Gravity term
        self.g = pin.computeGeneralizedGravity(self.robot.model, self.robot.data, self.q)

        # Coriolis term
        self.C = pin.computeCoriolisMatrix(self.robot.model, self.robot.data, self.q, self.v)


    def update_support_states(self, support_state: list | np.ndarray):
        """
            Update robot support states 
        """
        self.support_state = support_state

    
    def get_body_pos_wcs(self) -> np.ndarray:
        """
            Obtain body position in WCS
        """
        return self.q[0:3]

    
    def get_body_ori_wcs(self) -> np.ndarray:
        """
            Obtain body orientation quaternion in WCS
        """
        return self.q[3:7]


    def get_body_vel_wcs(self) -> np.ndarray:
        """
            Obtain body velocity in WCS
        """
        return self.body_R @ self.v[0:3]


    def get_body_angvel_wcs(self) -> np.ndarray:
        """
            Obtain body angular velocity, aka. omega in WCS
        """
        return self.body_R @ self.v[3:6]


    def get_tip_pos_wcs(self, leg_id: int) -> np.ndarray:
        """
            Obtain tip position in WCS
        """
        return self.robot.data.oMf[self.toe_frame_id_lists[leg_id]].translation

    def get_tip_vel_wcs(self, leg_id: int) -> np.ndarray:
        """
            Obtain tip velocity in WCS
        """
        return self.J_leg[leg_id, :, :] @ self.v
    

    def get_n_support_legs(self) -> int:
        """
            Obtain number of supporting legs
        """
        return np.sum(self.support_state > self.support_thres)

    
    def is_leg_supporting(self, leg_id) -> bool:
        """
            return whether a leg is supporting
        """
        return self.support_state[leg_id] > self.support_thres


    def get_contact_jacobian_or_none(self) -> np.ndarray | None:
        """
            Obtain contact jacobian Jc. 
            If no leg is in supporting phase, return None
        """
        
        n_support_leg = self.get_n_support_legs()

        if n_support_leg <= 0:
            return None

        Jc = np.zeros((3*n_support_leg, self.robot.nv))
        cnt = 0
        for leg in range(self.n_legs):
            if self.support_state[leg] > self.support_thres:
                # add support leg tip's jacobian to the contact jacobian matrix
                Jc[3*cnt:3+3*cnt, :] = self.J_leg[leg, :, :]
                cnt += 1
        return Jc

    def get_contact_jcdqd_or_none(self) -> np.ndarray | None:
        """
            Obtain contact jacobian time variation Jcdot*qdot. 
            If no leg is in supporting phase, return None.
        """
        n_support_leg = self.get_n_support_legs()

        if n_support_leg <= 0:
            return None

        Jcdqd = np.zeros(3*n_support_leg)
        cnt = 0
        for leg in range(self.n_legs):
            if self.support_state[leg] > self.support_thres:
                # add support leg tip's jacobian to the contact jacobian matrix
                Jcdqd[3*cnt:3+3*cnt] = self.get_leg_pos_jdqd(leg)
                cnt += 1
        return Jcdqd

    def get_body_ori_jacobian(self) -> np.ndarray:
        """
            Obtain jacobian of the body orientation.
        """
        J_body_ori = np.zeros((3, self.nv))
        J_body_ori[0:3, 3:6] = self.body_R
        return J_body_ori


    def get_body_ori_jdqd(self) -> np.ndarray:
        """
            Obtain Jdot * qd of the body orientation
        """
        Jdqd_body_ori = np.zeros(3)
        return Jdqd_body_ori


    def get_body_pos_jacobian(self) -> np.ndarray:
        """
            Obtain jacobian of the body position.
        """
        J_body_pos = np.zeros((3, self.nv))
        J_body_pos[0:3, 0:3] = self.body_R
        return J_body_pos


    def get_body_pos_jdqd(self) -> np.ndarray:
        """
            Obtain Jdot * qd of the body position.
        """
        Jdqd_body_pos = np.zeros(3)
        return Jdqd_body_pos


    def get_leg_pos_jacobian(self, leg_id: int) -> np.ndarray:
        """
            Obtain jacobian of the leg tip pos.
            The leg id follows the definition of pinocchio model.
        """
        J_leg_pos = self.J_leg[leg_id, :, :]
        return J_leg_pos


    def get_leg_pos_jdqd(self, leg_id: int) -> np.ndarray:
        """
            Obtain Jdot * qd of the leg tip pos.
            The leg id follows the definition of pinocchio model.
        """
        Jd_leg = pin.frameJacobianTimeVariation(
            self.robot.model, self.robot.data, 
            self.q, self.v, self.toe_frame_id_lists[leg_id], 
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jdqd = Jd_leg @ self.v
        return Jdqd[0:3]


    def get_mass_mat(self) -> np.ndarray:
        """
            Obtain Mass matrix
        """
        return self.M


    def get_coriolis_mat(self) -> np.ndarray:
        """
            Obtain Coriolis matrix
        """
        return self.C


    def get_gravity_term(self) -> np.ndarray:
        """
            Obtain gravity term
        """
        return self.g


    def get_tau(self, a: np.ndarray, fr: np.ndarray) -> np.ndarray:
        """
            Calculate joint trq through inverse dynamics.
        """
        # calculate net torque using dynamics
        net_tau = self.M @ a + self.C @ self.v + self.g

        # since we have contact force, the actual joint torque should be
        #  tau_ = M @ a + C @ v + g - Jc(q).T Fr

        # calculate all Jc(q), must be called after robot.computeJointJacobians(q)
        Jc_all = np.zeros((12, 18))
        for i in range(4):
            Jc_all[0+i*3:3+i*3,:] = self.J_leg[i, 0:3, :] # we only need translational part

        tau_with_fr = net_tau - Jc_all.T @ fr

        return tau_with_fr


class JointOrderMapper(object):

    LID_OUR  = {'FR' : 0, 'FL' : 1, 'RR' : 2, 'RL' : 3}
    LID_PINO = {'FR' : 1, 'FL' : 0, 'RR' : 3, 'RL' : 2}

    # leg mapping from our model to pinocchio
    leg_map : list[int] = [1, 0, 3, 2]  
    # joint mapping from our model to pinocchio
    jnt_map : list[int] = [3,  4,  5, 0, 1, 2,
                           9, 10, 11, 6, 7, 8]

    # whole joint mapping from our model to pinocchio, including the float-base joint
    q_map : list[int] = [ 0, 1, 2, 3, 4, 5, # linear, angular 
                          9,10,11, 6, 7, 8,
                         15,16,17,12,13,14]

    def __init__(self) -> None:
        pass


    def convert_vec_to_pino(self, vec: np.ndarray) -> np.ndarray:
        """
            convert 12x1 vector to pino order, 
            the vector can be q_j, v_j, a_j, fr_tip 
        """
        vec_pino = 0 * vec
        vec_pino[self.jnt_map] = vec
        return vec_pino


    def convert_sps_to_pino(self, support_states: np.ndarray) -> np.ndarray:
        """
            convert 4x1 leg support states to pino order
        """
        ss_pino = 0 * support_states
        ss_pino[self.leg_map] = support_states
        return ss_pino


    def leg_id_pino(self, leg_id: int) -> int:
        """
            get leg id in pino order
        """
        return self.leg_map[leg_id]

    
    def convert_mat_to_our(self, mat: np.ndarray) -> np.ndarray:
        """
            convert nx(6+12) mat and vec to our order,
            the mat can be Jc, Jori, Jpos, Jleg, M, C, g, tau
        """
        mat_our = 0 * mat
        mat_our = mat[:, self.q_map]
        return mat_our

    def convert_jvec_to_our(self, vec: np.ndarray) -> np.ndarray:
        """
            convert nj(12) vec to our order,
            the mat can be q_j, qdot_j, qddot_j, tau_j
        """
        vec_our = 0 * vec
        vec_our = vec[self.jnt_map]
        return vec_our