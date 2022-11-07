import numpy as np
from scipy.spatial.transform import Rotation as rot
from RobotController import RobotDynamics as rdyn


class WBCTask(object):
    """
        Base class of WBC tasks.
    """

    # Task constants
    n_leg: int = 4              # number of legs
    nt   : int = 1              # dimension of task configuration
    nv   : int = 6 + n_leg * 3  # dimension of model's qdot

    # place holders
    Jt:      np.ndarray = np.zeros((nt, nv))       # Jacobian of the task:  v_task = Jt * qdot
    Jtdqd:   np.ndarray = np.zeros(nt)             # Task jacobian time variation times qdot term
    pos_err: np.ndarray = np.zeros(nt)             # position error of the task
    vel_err: np.ndarray = np.zeros(nt)             # velocity error of the task
    vel_des: np.ndarray = np.zeros(nt)             # desired velocity of the task
    acc_des: np.ndarray = np.zeros(nt)             # desired acceleration of the task

    def __init__(self, nt: int) -> None:
        """
            Create and initialize a whole body control task instance.

            Parameters:
                nt (int): dimension of the task
        """
        self.nt = nt
        self.Jt = np.zeros((nt, self.nv))
        self.Jtdqd = np.zeros(nt)
        self.pos_err = np.zeros(nt)
        self.vel_err = np.zeros(nt)
        self.vel_des = np.zeros(nt)
        self.acc_des = np.zeros(nt)

    def update(self,
               model: rdyn.RobotDynamicModel,
               ref_x_wcs: np.ndarray,
               ref_xdot_wcs: np.ndarray,
               ref_xddot_wcs: np.ndarray):
        """
            Update task commands and kinematics.
            This method should be override by child classes that inherent this class.

            Parameters:
                model (reference of RobotDynamicModel): Robot dynamic model instance.
                ref_x_wcs (array(7+n_leg*3)):  reference robot position in WCS. Defined as
                    [x y z qx qy qz qw leg1_pos_x leg1_pos_y leg1_pos_z ... leg4_pos_z]
                ref_xdot_wcs (array(6+n_leg*3)): reference robot velocity in WCS. Defined as
                    [vx vy vz wx wy wz leg1_vel_x leg1_vel_y leg1_vel_z ... leg4_vel_z]
                ref_xddot_wcs (array(6+n_leg*3)): reference robot acceleration in WCS. Defined as
                    [ax ay az alfax alfay alfaz leg1_acc_x leg1_acc_y leg1_acc_z ... leg4_acc_z]
        """
        pass

    def get_jacobian(self) -> np.ndarray:
        """
            Get task jacobian.

            Returns:
                Jt (array(nt, nv)): task jacobian.
        """
        return self.Jt

    def get_jdqd(self) -> np.ndarray:
        """
            Get task jacobian time variation times qdot term:
                        dJt/dt * dq/dt

            Returns:
                Jtdqd (array(nt, nv)): task jacobian time variation times qdot term.
        """
        return self.Jtdqd

    def get_pos_err(self) -> np.ndarray:
        """
            Get position error of the task.

            Returns:
                pos_err (array(nt)): position error of the task.
        """
        return self.pos_err

    def get_vel_des(self) -> np.ndarray:
        """
            Get desired velocity of the task.

            Returns:
                vel_des (array(nt)): desired velocity of the task.
        """
        return self.vel_des

    def get_acc_des(self) -> np.ndarray:
        """
            Get desired acceleration of the task.

            Returns:
                acc_des (array(nt)): desired acceleration of the task.
        """
        return self.acc_des


class BodyOriTask(WBCTask):
    """
        Task to control the body orientation.
    """

    def __init__(self) -> None:
        """
            Create a Body orientation task and set default KP/KD parameters.
        """
        super().__init__(3)
        self.kp = np.diag([50, 50, 50])
        self.kd = np.diag([1, 1, 1])

    def update(self,
               model: rdyn.RobotDynamicModel,
               ref_x_wcs: np.ndarray,
               ref_xdot_wcs: np.ndarray,
               ref_xddot_wcs: np.ndarray):

        
        self.Jt = model.get_body_ori_jacobian()
        self.Jtdqd = model.get_body_ori_jdqd()
        
        act_ori_wcs = model.get_body_ori_wcs()
        act_omega_wcs = model.get_body_angvel_wcs()

        ref_ori_wcs = ref_x_wcs[3:7]
        ref_omega_wcs = ref_xdot_wcs[3:6]
        ref_angacc_wcs = ref_xddot_wcs[3:6]

        # orientation diff
        # FIXME: the ori_diff_SO3 might be wrong
        body_R = rot.from_quat(act_ori_wcs).as_matrix()
        body_R_ref = rot.from_quat(ref_ori_wcs).as_matrix()
        R_diff = body_R.T @ body_R_ref
        ori_diff_SO3 = rot.from_matrix(R_diff).as_rotvec()
        print(ori_diff_SO3)
        
        self.pos_err = ori_diff_SO3
        self.vel_err = ref_omega_wcs - act_omega_wcs
        self.vel_des = ref_omega_wcs
        self.acc_des = ref_angacc_wcs + self.kp @ self.pos_err + self.kd @ self.vel_err


class BodyPosTask(WBCTask):
    """
        Task to control the body position.
    """

    def __init__(self) -> None:
        """
            Create a Body position task and set default KP/KD parameters.
        """
        super().__init__(3)
        self.kp = np.diag([50, 50, 50])
        self.kd = np.diag([1, 1, 1])
        

    def update(self,
               model: rdyn.RobotDynamicModel,
               ref_x_wcs: np.ndarray,
               ref_xdot_wcs: np.ndarray,
               ref_xddot_wcs: np.ndarray):
        
        self.Jt = model.get_body_pos_jacobian()
        self.Jtdqd = model.get_body_pos_jdqd()
        
        act_pos_wcs = model.get_body_pos_wcs()
        act_vel_wcs = model.get_body_vel_wcs()

        ref_pos_wcs = ref_x_wcs[0:3]
        ref_vel_wcs = ref_xdot_wcs[0:3]
        ref_acc_wcs = ref_xddot_wcs[0:3]

        self.pos_err = ref_pos_wcs - act_pos_wcs
        self.vel_err = ref_vel_wcs - act_vel_wcs
        self.vel_des = ref_vel_wcs
        self.acc_des = ref_acc_wcs + self.kp @ self.pos_err + self.kd @ self.vel_err


class TipPosTask(WBCTask):
    """
        Task to control the position of the leg tip.
    """

    def __init__(self, leg_id: int) -> None:
        """
            Create a Leg tip position task and set default KP/KD parameters.

            Parameters:
                leg_id (int): index of the leg. In Pinocchio's order
        """
        super().__init__(3)
        self.kp = np.diag([100, 100, 100])
        self.kd = np.diag([5, 5, 5])
        self.leg_id = leg_id


    def update(self,
               model: rdyn.RobotDynamicModel,
               ref_x_wcs: np.ndarray,
               ref_xdot_wcs: np.ndarray,
               ref_xddot_wcs: np.ndarray):
        
        leg_id = self.leg_id
        self.Jt = model.get_leg_pos_jacobian(leg_id)
        self.Jtdqd = model.get_leg_pos_jdqd(leg_id)
        
        act_pos_wcs = model.get_tip_pos_wcs(leg_id)
        act_vel_wcs = model.get_tip_vel_wcs(leg_id)

        ref_pos_wcs = ref_x_wcs[7+leg_id*3:10+leg_id*3]
        ref_vel_wcs = ref_xdot_wcs[6+leg_id*3:9+leg_id*3]
        ref_acc_wcs = ref_xddot_wcs[6+leg_id*3:9+leg_id*3]

        self.pos_err = ref_pos_wcs - act_pos_wcs
        self.vel_err = ref_vel_wcs - act_vel_wcs
        self.vel_des = ref_vel_wcs
        self.acc_des = ref_acc_wcs + self.kp @ self.pos_err + self.kd @ self.vel_err
