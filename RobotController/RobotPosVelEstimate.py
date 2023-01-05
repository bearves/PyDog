import math
import numpy as np
from scipy.spatial.transform import Rotation as rot
import scipy.linalg
from RobotController import RobotKinematics as rkin

class QuadPosVelEstimator(object):

    """
        Implementation of a EKF-based state estimator for quadruped robot.
        This implementation is derived from ETH's paper (Micheal Bloesch, et. al, 2013).
        However, only the robot's position and velocity is estimated, and the orientation
        is directly obtained from IMU. The gyro and acc bias are also not estimated. 
    """

    # basic params
    dt:     float       # time step of the simulator
    n_leg:  int = 4     # number of robot legs
    ns:     int         # dimension of the estimating state
    nis:    int         # independent dimension of the estimating state
    nm:     int         # dimension of the measurement vector
    nl:     int         # dimension of tip pos vector, nl = n_leg * 3

    # state and measurement covariances
    Qf:     np.ndarray  # acceleration covariance, 3x3
    Qw:     np.ndarray  # gyro covariance, 3x3

    Qpst:   np.ndarray  # tip position covariance at stance phase, 3x3
    Qpsw:   np.ndarray  # tip velocity covariance at swing phase, 3x3

    Rs:     np.ndarray  # leg measurement covariance from encoder, 3x3


    # estimator states
    xk : np.ndarray     # estimating state at k moment, dim(xk) = ns x 1 
    Pk : np.ndarray     # estimating state covariance at k moment, dim(Pk) = nis x nis
    xkp1: np.ndarray    # predicted next state at k+1 moment,  dim(xkp1) = ns x 1 
    Pkp1: np.ndarray    # predicted next state covariance at k+1 moment,  dim(Pkp1) = nis x nis

    # states not to estimate
    qk: np.ndarray       # body orientation at k moment, in WCS

    # placeholders
    Fk: np.ndarray      # state matrix that xk+1 = Fk*xik + wk, dim(Fk)= nis x nis
    Qk: np.ndarray      # covariance of the state equation noise, aka. COV(wk), dim(Qk) = nis x nis

    sk: np.ndarray      # actual measurements from the leg FK and encoders, dim(sk) = nm x 1
    ssk: np.ndarray     # predicted measurements from states, that ssk = Hk * xik + vk, dim(ssk) = nm x 1
    yk: np.ndarray      # measurement residue that yk = sk - ssk, dim(yk) = nm x 1
    Hk: np.ndarray      # measurement matrix that ssk = Hk * xik + vk, dim(Hk) = nm x nis
    Rk: np.ndarray      # covariance of the measurement noise, aka. COV(vk), dim(Rk) = nm x nm

    Kk: np.ndarray      # Kalman gain that dx = Kk * yk, dim(Kk) = nis x nm
    dx: np.ndarray      # state correction, that xik_est = xik_predict + dx, dim(dx) = nis x 1

    # internal state place holders
    Ck: np.ndarray      # rotation matrix 3x3, mapping vector from WCS to BCS (for the compliance to the ETH's paper)
    ak: np.ndarray      # absolute acceleration in WCS, i.e. ak = Ck.T (fk_measure - acc_bias) + g
    fk: np.ndarray      # net acceleration in BCS, i.e. fk = fk_measure - acc_bias
    wk: np.ndarray      # measured angular velocity in BCS

    # useful constants
    g: np.ndarray = np.array([0, 0, -9.81]) # gravitational acceleration
    I3: np.ndarray = np.eye(3)              # 3x3 Identity mat
    O3: np.ndarray = np.zeros(3)            # 3x3 Zero mat

    
    def __init__(self, dt: float) -> None:
        """
            Initialize the state estimator and setup covariances of signals and states.

            Parameters:
                dt (float): the time step of each estimation run.
        """
        
        self.dt = dt
        # state definition:
        # xk = [body_pos, body_vel, tip_pos1, tip_pos2..., tip_posN]
        # NOTE: body_pos, body_vel, tip_pos_i in WCS,
        self.ns = 3+3+3*self.n_leg
        self.nis = self.ns 
        self.xk = np.zeros(self.ns)

        # measurement definition
        # sk = [rel_tip_pos1, rel_tip_pos2, ..., rel_tip_posN]
        self.nm = 3 * self.n_leg
        self.nl = 3 * self.n_leg

        # setup state and measurement covariance parameters
        self.Qf = 1e-2 * np.eye(3)
        self.Qw = 1e-2 * np.eye(3)

        self.Qpst = 1e-1 * np.eye(3)
        self.Qpsw = 1e30 * np.eye(3)

        self.Rs = 1e-6 * np.eye(3)


    def reset_state(self, 
                    body_pos: np.ndarray, 
                    body_vel: np.ndarray,
                    body_orn: np.ndarray,
                    tip_pos_wcs: np.ndarray):
        """
            Reset robot state as the initial estimation.
            
            Parameters:
                body_pos (array(3)): initial body position in WCS.
                body_vel (array(3)): initial body velocity in WCS.
                body_orn (array(4)): initial body orientation in WCS, expressed in quaternion.
                tip_pos_wcs  (array(n_leg*3)): initial tip position in WCS.
        """
        self.qk = body_orn        
        self.xk[0:3] = body_pos             # p in WCS
        self.xk[3:6] = body_vel             # v in WCS
        self.xk[6:6+self.nl] = tip_pos_wcs  # p_tip in WCS
        self.Pk = 1e-16 * np.eye(self.nis)  # set initial covariance matrix 


    def update(self, 
               kin_model : rkin.RobotKineticModel,
               body_orn_wcs: np.ndarray,
               body_gyr_bcs: np.ndarray, 
               body_acc_bcs: np.ndarray, 
               jnt_act_pos: np.ndarray, 
               jnt_act_vel: np.ndarray, 
               jnt_act_trq: np.ndarray, 
               support_state: np.ndarray,
               support_phase: np.ndarray):
        """
            Update and estimate robot states.

            Parameters:
                kin_model (rkin.RobotKineticModel): reference to the robot kinetic model.
                body_orn_wcs (array(4)): body orientation, from BCS to WCS.
                body_gyr_bcs (array(3)): body angular velocity, in BCS
                body_acc_bcs (array(3)): body linear acceleration, in BCS.
                jnt_act_pos (array(n_leg*3)): actual position of joints.
                jnt_act_vel (array(n_leg*3)): actual velocity of joints.
                jnt_act_trq (array(n_leg*3)): actual torque of joints.
                support_state (array(n_leg)): the supporting state of all legs.
                support_phase (array(n_leg)): the phase of supporting state of all legs. 
                                            for each leg, phase is a 0-1 scalar, that 
                                            phase=0 at touchdown, phase=1 at lifting up.
        """
        
        # calculate leg kinematics for measurement error
        self.update_kinetic_measurement(kin_model, jnt_act_pos, jnt_act_vel)
        # predict next state: xk+1_pred = pred(xk)
        self.state_predict(body_orn_wcs, body_gyr_bcs, body_acc_bcs)
        # update matrices of Kalman filter Fk, Qk, Hk, Rk
        self.update_matrices(jnt_act_trq, support_state, support_phase)
        # calculate Kalman gains and adjustment dx
        self.estimate()
        # correct next state: xk+1_est = xk+1_pred + dx
        self.state_correct()


    def update_kinetic_measurement(self, 
                             kin_model : rkin.RobotKineticModel,
                             jnt_act_pos: np.ndarray,
                             jnt_act_vel: np.ndarray):
        """
            Update kinetic model and get leg kinetic measurements, i.e. sk.

            Parameters:
                kin_model (rkin.RobotKineticModel): reference to the robot kinetic model.
                jnt_act_pos (array(n_leg*3)): actual position of joints.
                jnt_act_vel (array(n_leg*3)): actual velocity of joints.
        """

        # kinetic model has updated leg states, note that we just use leg forward kinetics in BODY cs,
        # thus the body states have not been updated yet
        tip_pos_wrt_body, tip_vel_wrt_body = kin_model.get_tip_state_body()
        self.sk = tip_pos_wrt_body


    def state_predict(self, 
                      body_orn_wcs: np.ndarray,
                      body_gyr_bcs: np.ndarray, 
                      body_acc_bcs: np.ndarray):
        """
            Predict x_k+1 using system model.

            Parameters:
                body_orn_wcs (array(4)): body orientation, from BCS to WCS.
                body_gyr_bcs (array(3)): body angular velocity, in BCS
                body_acc_bcs (array(3)): body linear acceleration, in BCS.
        """
        dt = self.dt
        d2t = self.dt**2
        rk = self.xk[0:3]
        vk = self.xk[3:6]
    
        self.qk = body_orn_wcs

        self.fk = body_acc_bcs
        self.wk = body_gyr_bcs

        self.Ck = rot.from_quat(self.qk).as_matrix().T # complaint to ETH's definition
        self.ak = self.Ck.T @ self.fk + self.g
        rkp1 = rk + dt * vk + 0.5 * d2t * self.ak
        vkp1 = vk + dt * self.ak

        self.xkp1 = self.xk.copy()
        self.xkp1[0:3] = rkp1
        self.xkp1[3:6] = vkp1 


    def update_matrices(self, 
                        jnt_act_trq: np.ndarray,
                        support_state: np.ndarray,
                        support_phase: np.ndarray):
        """
            Update Fk, Qk, Hk, and Rk for EKF estimation. See ETH's paper for details.

            Parameters:
                jnt_act_trq (array(n_leg*3)): the joint actual torque of all legs.
                support_state (array(n_leg)): the supporting state of all legs.
                support_phase (array(n_leg)): the phase of supporting state of all legs. 
                                            for each leg, phase is a 0-1 scalar, that 
                                            phase=0 at touchdown, phase=1 at lifting up.
        """
        # Update Fk
        dt = self.dt
        d2t = dt**2
        d3t = dt**3

        Fk = np.eye(self.nis) # assign diagnal elements to I3
        Fk[0:3, 3:6] = dt * self.I3
        self.Fk = Fk.copy()

        # Update Qk
        Qk = np.zeros((self.nis, self.nis))
        Qk[0:3, 0:3] = d3t / 3. * self.Qf
        Qk[0:3, 3:6] = d2t / 2. * self.Qf

        Qk[3:6, 0:3] = d2t / 2. * self.Qf 
        Qk[3:6, 3:6] = dt       * self.Qf 

        for leg in range(self.n_leg):
            Qp = self.get_tip_pos_cov(leg, jnt_act_trq, support_state, support_phase)
            Qk[6+leg*3:9+leg*3, 6+leg*3:9+leg*3] = dt * self.Ck.T @ Qp @ self.Ck

        self.Qk = Qk.copy()

        # Update yk
        self.ssk = np.zeros(self.nm) # measurement vector
        for leg in range(self.n_leg):
            self.ssk[0+leg*3:3+leg*3] = self.Ck @ (self.xkp1[6+leg*3:9+leg*3] - self.xkp1[0:3])
        self.yk = self.sk - self.ssk # innovation vector

        # Update Hk
        Hk = np.zeros((self.nm, self.nis))
        for leg in range(self.n_leg):
            idx = range(0+leg*3, 3+leg*3)
            Hk[idx, 0:3] = -self.Ck
            Hk[idx, 6+leg*3:9+leg*3] = self.Ck
        self.Hk = Hk.copy()

        # Update Rk
        Rk = np.zeros((self.nm, self.nm))
        for leg in range(self.n_leg):
            Rk[0+leg*3:3+leg*3, 0+leg*3:3+leg*3] = self.Rs
        self.Rk = Rk.copy()


    def estimate(self):
        """
            Do estimation using the Extended-Kalman filter.
            The Kalman gain Kk, correcting vector dx and state covariance Pk is updated. 
        """
        Pkp1 = self.Fk @ self.Pk @ self.Fk.T + self.Qk          # dim(Pkp1) = nis x nis
        Sk = self.Hk @ Pkp1 @ self.Hk.T + self.Rk               # dim(Sk) = nm x nm
        Sk = 0.5 * (Sk + Sk.T)
        # Augment for solve once
        RHS = np.hstack((self.yk.reshape((self.nm, 1)), self.Hk))
        invSyH = np.linalg.solve(Sk, RHS)
        self.Kk = Pkp1 @ self.Hk.T
        self.dx = self.Kk @ invSyH[:, 0]                        # dim(dx) = nis x 1
        self.Pk = (np.eye(self.nis) - self.Kk @ invSyH[:, 1:]) @ Pkp1        # dim(Pk) = nis x nis
        self.Pk = 0.5 * (self.Pk + self.Pk.T)


    def state_correct(self):
        """
            Correct the robot state x using the correcting vector dx.
            dx is calculated by the estimate() function.
        """
        self.xk = self.xkp1.copy()
        self.xk += self.dx


    def get_results(self) -> np.ndarray:
        """
            Get the estimated results.

            Returns:
                xk (array(ns)): the estimated robot state.
        """
        return self.xk.copy()
    
    
    def get_est_body_pos_wcs(self) -> np.ndarray:
        """
            Get the estimated body position in WCS.

            Returns:
                pos (array(3)): the estimated body position.
        """
        return self.xk[0:3]


    def get_est_body_vel_wcs(self) -> np.ndarray:
        """
            Get the estimated body velocity in WCS.

            Returns:
                vel (array(3)): the estimated body velocity.
        """
        return self.xk[3:6]


    def get_est_body_orn_wcs(self) -> np.ndarray:
        """
            Get the estimated body orientation in WCS.

            Returns:
                orn (array(3)): the estimated body orientation.
        """
        return self.qk


    def get_tip_pos_cov(self, 
                        leg_id: int, 
                        jnt_act_trq: np.ndarray,
                        support_state: np.ndarray,
                        support_phase: np.ndarray) -> np.ndarray:
        """
            Get the covariance matrix Qp of the tip position state.
            Qp is varying according to the contact status.
            If the leg is in contact with the ground, its tip position covariance is low,
            the estimator believes the leg tip should be stay at its original position,
            otherwise its position covariance is very high and the estimator believes the 
            leg tip position should be corrected by the leg kinetic measurements.

            Parameters:
                leg_id (int): the index of leg.
                jnt_act_trq (array(n_leg*3)): the joint actual torque of all legs.
                support_state (array(n_leg)): the supporting state of all legs.
                support_phase (array(n_leg)): the phase of supporting state of all legs. 
                                            for each leg, phase is a 0-1 scalar, that 
                                            phase=0 at touchdown, phase=1 at lifting up.
            
            Returns:
                Qp (array(3,3)): the position covariance of the given leg's tip.

        """
        trust = 0.0
        stage1 = 0.05
        stage2 = 0.25
        gap = stage2 - stage1
        
        if (support_state[leg_id] < 0.7):
            trust = 0 # swing
        else: # stance phase
            if (support_phase[leg_id] < stage1 or support_phase[leg_id] > 1-stage1):
                trust = 0.0
            elif (support_phase[leg_id] > stage2 and support_phase[leg_id] < 1-stage2):
                trust = 1.0
            elif (support_phase[leg_id] >= stage1 and support_phase[leg_id] <= stage2):
                trust = (support_phase[leg_id] - stage1)/gap
            elif (support_phase[leg_id] <= 1-stage1 and support_phase[leg_id] >= 1-stage2):
                trust = (1 - stage1 - support_phase[leg_id])/gap
            else:
                trust = 0.0
        # TODO: take consideration of the leg force
        return self.Qpst * trust + self.Qpsw * (1-trust)

