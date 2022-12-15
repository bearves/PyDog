import math
import numpy as np
from scipy.spatial.transform import Rotation as rot
import scipy.linalg
from RobotController import RobotKinematics as rkin

class QuadFullStateEstimator(object):

    """
        Implementation of a EKF-based state estimator for quadruped robot.
        This implementation is derived from ETH's paper (Micheal Bloesch, et. al, 2013)
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
    Qbf:    np.ndarray  # acceleration bias covariance, 3x3
    Qw:     np.ndarray  # gyro covariance, 3x3
    Qbw:    np.ndarray  # gyro bias covariance, 3x3 

    Qpst:   np.ndarray  # tip position covariance at stance phase, 3x3
    Qpsw:   np.ndarray  # tip velocity covariance at swing phase, 3x3

    Rs:     np.ndarray  # leg measurement covariance from encoder, 3x3


    # estimator states
    xk : np.ndarray     # estimating state at k moment, dim(xk) = ns x 1 
    Pk : np.ndarray     # estimating state covariance at k moment, dim(Pk) = nis x nis
    xkp1: np.ndarray    # predicted next state at k+1 moment,  dim(xkp1) = ns x 1 
    Pkp1: np.ndarray    # predicted next state covariance at k+1 moment,  dim(Pkp1) = nis x nis

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
    Ckp1: np.ndarray    # predicted rotation matrix 3x3
    ak: np.ndarray      # absolute acceleration in WCS, i.e. ak = Ck.T (fk_measure - acc_bias) + g
    fk: np.ndarray      # net acceleration in BCS, i.e. fk = fk_measure - acc_bias

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
        # xk = [body_pos, body_vel, body_orn, tip_pos1, tip_pos2..., tip_posN, acc_bias, gyro_bias]
        # NOTE: body_pos, body_vel, tip_pos_i in WCS,
        #       body_orn in Body CS, mapping vector from WCS to BCS (for the compliance to the ETH's paper)
        #       acc_bias and gyro_bias in Body CS
        self.ns = 3+3+4+3*self.n_leg+3+3
        self.nis = self.ns - 1 # dim of body orientation = 4, independent dim of body orientation = 3 
        self.xk = np.zeros(self.ns)

        # measurement definition
        # sk = [rel_tip_pos1, rel_tip_pos2, ..., rel_tip_posN]
        self.nm = 3 * self.n_leg
        self.nl = 3 * self.n_leg

        # setup state and measurement covariance parameters
        self.Qf = 1e-2 * np.eye(3)
        self.Qbf = 1e-2 * np.eye(3)
        self.Qw = 1e-2 * np.eye(3)
        self.Qbw = 1e-2 * np.eye(3)

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
        self.xk[0:3] = body_pos             # p in WCS
        self.xk[3:6] = body_vel             # v in WCS
        self.xk[6:10] = quat_inv(body_orn)  # in ETH paper, q from WCS to BCS is used
        self.xk[10:10+self.nl] = tip_pos_wcs # p_tip in WCS
        self.xk[10+self.nl:13+self.nl] = np.zeros(3)  # acc_bias in BCS
        self.xk[13+self.nl:16+self.nl] = np.zeros(3)  # gyro_bias in BCS

        self.Pk = 1e-16 * np.eye(self.nis)  # set initial covariance matrix 


    def update(self, 
               kin_model : rkin.RobotKineticModel,
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
        self.state_predict(body_gyr_bcs, body_acc_bcs)
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
                      body_gyr_bcs: np.ndarray, 
                      body_acc_bcs: np.ndarray):
        """
            Predict x_k+1 using system model.

            Parameters:
                body_gyr_bcs (array(3)): body angular velocity, in BCS
                body_acc_bcs (array(3)): body linear acceleration, in BCS.
        """
        dt = self.dt
        d2t = self.dt**2
        rk = self.xk[0:3]
        vk = self.xk[3:6]
        qk = self.xk[6:10]
        bfk = self.xk[10+self.nl:13+self.nl]
        bwk = self.xk[13+self.nl:16+self.nl]

        self.fk = body_acc_bcs - bfk
        self.wk = body_gyr_bcs - bwk

        self.Ck = rot.from_quat(self.xk[6:10]).as_matrix()
        self.ak = self.Ck.T @ self.fk + self.g
        rkp1 = rk + dt * vk + 0.5 * d2t * self.ak
        vkp1 = vk + dt * self.ak
        # special updating for rotations. 
        # (NOTE: THe ETH's paper may have a bug here, after test, wk should be -wk )
        qkp1 = quat_prod(so3_to_quat(dt * -self.wk), qk)

        self.xkp1 = self.xk.copy()
        self.xkp1[0:3] = rkp1
        self.xkp1[3:6] = vkp1 
        self.xkp1[6:10] = qkp1


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
        d4t = dt**4
        d5t = dt**5
        g0 = gamma0(self.wk, self.dt)
        g1 = gamma1(self.wk, self.dt)
        g2 = gamma2(self.wk, self.dt)
        g3 = gamma3(self.wk, self.dt)

        Fk = np.eye(self.nis) # assign diagnal elements to I3
        Fk[0:3, 3:6] = dt * self.I3
        Fk[0:3, 6:9] = -0.5 * d2t * self.Ck.T @ skew(self.fk)
        Fk[0:3, 9+self.nl:12+self.nl] = -0.5 * d2t * self.Ck.T
        Fk[3:6, 6:9] = - dt * self.Ck.T @ skew(self.fk)
        Fk[3:6, 9+self.nl:12+self.nl] = - dt * self.Ck.T
        Fk[6:9, 6:9] = g0.T
        Fk[6:9, 12+self.nl:15+self.nl] = -g1.T
        self.Fk = Fk.copy()

        # Update Qk
        Qk = np.zeros((self.nis, self.nis))
        Qk[0:3, 0:3] = d3t / 3. * self.Qf + d5t / 20. * self.Qbf
        Qk[0:3, 3:6] = d2t / 2. * self.Qf + d4t / 8.  * self.Qbf
        Qk[0:3, 9+self.nl:12+self.nl] = - d3t / 6. * self.Ck.T @ self.Qbf

        Qk[3:6, 0:3] = d2t / 2. * self.Qf + d4t / 8. * self.Qbf
        Qk[3:6, 3:6] = dt       * self.Qf + d3t / 3. * self.Qbf
        Qk[3:6, 9+self.nl:12+self.nl] = - d2t / 2. * self.Ck.T @ self.Qbf

        Qk[6:9, 6:9] = dt * self.Qw + (g3 + g3.T) @ self.Qbw
        Qk[6:9, 12+self.nl:15+self.nl] = -g2.T @ self.Qbw

        Qk[9+self.nl:12+self.nl, 0:3] = -d3t/6. * self.Qbf @ self.Ck 
        Qk[9+self.nl:12+self.nl, 3:6] = -d2t/2. * self.Qbf @ self.Ck
        Qk[9+self.nl:12+self.nl, 9+self.nl:12+self.nl] = dt * self.Qbf

        Qk[12+self.nl:15+self.nl, 6:9] = - self.Qbw @ g2
        Qk[12+self.nl:15+self.nl, 12+self.nl:15+self.nl] = dt * self.Qbw

        for leg in range(self.n_leg):
            Qp = self.get_tip_pos_cov(leg, jnt_act_trq, support_state, support_phase)
            Qk[9+leg*3:12+leg*3, 9+leg*3:12+leg*3] = dt * self.Ck.T @ Qp @ self.Ck

        self.Qk = Qk.copy()

        # Update yk
        self.ssk = np.zeros(self.nm) # measurement vector
        self.Ckp1 = rot.from_quat(self.xkp1[6:10]).as_matrix()
        for leg in range(self.n_leg):
            self.ssk[0+leg*3:3+leg*3] = self.Ckp1 @ (self.xkp1[10+leg*3:13+leg*3] - self.xkp1[0:3])
        self.yk = self.sk - self.ssk # innovation vector

        # Update Hk
        Hk = np.zeros((self.nm, self.nis))
        for leg in range(self.n_leg):
            idx = range(0+leg*3, 3+leg*3)
            Hk[idx, 0:3] = -self.Ckp1
            Hk[idx, 6:9] = skew(self.ssk[idx])
            Hk[idx, 9+leg*3:12+leg*3] = self.Ckp1
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
        self.xk[0:6] += self.dx[0:6]
        # special updating for rotations. 
        # (NOTE: THe ETH's paper may have a bug here, after test, dx[6:9] should be -dx[6:9] )
        self.xk[6:10] = quat_prod(so3_to_quat(-self.dx[6:9]), self.xkp1[6:10])
        self.xk[10:10+self.nl] += self.dx[9:9+self.nl]
        self.xk[10+self.nl:] += self.dx[9+self.nl:]


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
        return quat_inv(self.xk[6:10])


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


###########################################
#     Necessary math functions            #
###########################################
def skew(vec: np.ndarray) -> np.ndarray:
    """
        Get the skew-symmetric matrix of a 3-dimensional vector
        to do cross product in matrix form, i.e.  v x n = skew(v) @ n

        Parameters:
            vec (array(3)): the 3-dimensional vector.
        
        Returns:
            vec_skew (array(3,3)): the skew-symmetric matrix of vec.
    """
    rx, ry, rz = vec[0], vec[1], vec[2]
    vec_skew = np.array([[  0, -rz,  ry],
                         [ rz,   0, -rx],
                         [-ry,  rx,   0]])
    return vec_skew


def quat_prod(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
        Get the quaternion product of two quaternions, i.e. q = q1 * q2

        Parameters:
            q1 (array(4)): the first quaternion.
            q2 (array(4)): the second quaternion.
        
        Returns:
            q  (array(4)): the product of q1 and q2.
    """
    s1 = q1[3]
    s2 = q2[3]
    v1 = q1[0:3]
    v2 = q2[0:3]
    q = np.zeros(4)
    q[0:3] = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    q[3] = s1 * s2 - np.dot(v1, v2)
    return q


def quat_inv(q: np.ndarray) -> np.ndarray:
    """
        Get the inverse quaternion of a quaternion, that qinv * q = [0,0,0,1]

        Parameters:
            q (array(4)): the quaternion.
        
        Returns:
            qinv  (array(4)): the inverse of q.
    """
    qinv = q.copy()
    qinv[0:3] *= -1.
    return qinv


def so3_to_quat(so3: np.ndarray) -> np.ndarray:
    """
        Get the quaternion form of the exponential of a so3 vector, i.e. q = exp(so3).  

        Parameters:
            so3 (array(3)): the so3 vector.
        
        Returns:
            q (array(4)): the quaternion.
    """
    n = np.linalg.norm(so3)
    q = np.zeros(4)
    q[0:3] = math.sin(0.5*n) / n * so3 
    q[3] = math.cos(0.5*n)
    return q


def gamma0(w: np.ndarray, dt: float) -> np.ndarray:
    """
        Get the gamma0 matrix from a small rotation. (see ETH's paper for definition).
        Here the Rodrigues' formula is adopted to get a closed-form solution.

        Parameters:
            w (array(3)): angular velocity.
            dt (float): time interval
        
        Returns:
            gamma (array(3,3)): the gamma0 matrix.
    """
    th = np.linalg.norm(w)
    wn = 1/th * w
    wnx = skew(wn)
    # apply Rodrigues' rotation formula
    gamma = np.eye(3) + \
            math.sin(th*dt) * wnx + \
            (1 - math.cos(th*dt)) * (wnx @ wnx)
    return gamma


def gamma1(w: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """
        Get the gamma1 matrix from a small rotation. (see ETH's paper for definition).
        Here the Rodrigues' formula is adopted to get a closed-form solution. 

        Parameters:
            w (array(3)): angular velocity.
            dt (float): time interval
        
        Returns:
            gamma (array(3,3)): the gamma1 matrix.
    """
    th = np.linalg.norm(w)
    wn = 1/th * w
    wnx = skew(wn)
    gamma = dt * np.eye(3) + \
            (1 - math.cos(th*dt)) / th * wnx + \
            (th*dt - math.sin(th*dt)) / th * (wnx @ wnx)
    return gamma


def gamma2(w: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """
        Get the gamma2 matrix from a small rotation. (see ETH's paper for definition).
        Here the Rodrigues' formula is adopted to get a closed-form solution.

        Parameters:
            w (array(3)): angular velocity.
            dt (float): time interval
        
        Returns:
            gamma (array(3,3)): the gamma2 matrix.
    """
    th = np.linalg.norm(w)
    wn = 1/th * w
    wnx = skew(wn)
    gamma = 1./2. * dt**2 * np.eye(3) + \
            (th*dt - math.sin(th*dt)) / (th**2) * wnx +  \
            (math.cos(th*dt) - 1. + 1./2. * dt**2 * th**2) / (th**2) * (wnx @ wnx)
    return gamma


def gamma3(w: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """
        Get the gamma3 matrix from a small rotation. (see ETH's paper for definition).
        Here the Rodrigues' formula is adopted to get a closed-form solution.

        Parameters:
            w (array(3)): angular velocity.
            dt (float): time interval
        
        Returns:
            gamma (array(3,3)): the gamma3 matrix.
    """
    th = np.linalg.norm(w)
    wn = 1/th * w
    wnx = skew(wn)
    gamma = 1./6. * dt**3 * np.eye(3) + \
            (math.cos(th*dt) - 1. + 1./2. * dt**2 * th**2) / (th**3) * wnx +  \
            (math.sin(th*dt) - th*dt + 1./6. * dt**3 * th**3) / (th**3) * (wnx @ wnx)
    return gamma
