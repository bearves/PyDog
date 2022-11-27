import math
import numpy as np
from scipy.spatial.transform import Rotation as rot


class QuadStateEstimator(object):

    """
        Implementation of a EKF-based state estimator for quadruped robot.
    """

    # basic params
    dt:     float       # time step of the simulator
    n_legs: int = 4     # number of robot legs
    ns:     int         # dimension of the estimating state
    nis:    int         # independent dimension of the estimating state
    nm:     int         # dimension of the measurement vector
    nl:     int         # dimension of tip pos vector, nl = n_legs * 3

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
    Rk: np.ndarray      # covariance of the measurment noise, aka. COV(vk), dim(Rk) = nm x nm

    Kk: np.ndarray      # Kalman gain that dx = Kk * yk, dim(Kk) = nis x nm
    dx: np.ndarray      # state corretion, that xik_est = xik_predict + dx, dim(dx) = nis x 1

    # internal state place holders
    Ck: np.ndarray      # rotation matrix 3x3, mapping vector from WCS to BCS (for the compliance to the ETH's paper)
    ak: np.ndarray      # absolute acceleration in WCS, i.e. ak = Ck.T (fk_measure - acc_bias) + g
    fk: np.ndarray      # net acceleration in BCS, i.e. fk = fk_measure - acc_bias

    # useful constants
    g: np.ndarray = np.array([0, 0, -9.81]) # gravititional acceleration
    I3: np.ndarray = np.eye(3)              # 3x3 Identity mat
    O3: np.ndarray = np.zeros(3)            # 3x3 Zero mat

    
    def __init__(self, dt: float) -> None:
        
        self.dt = dt
        # state definition:
        # xk = [body_pos, body_vel, body_orn, tip_pos1, tip_pos2..., tip_posN, acc_bias, gyro_bias]
        # NOTE: body_pos, body_vel, tip_pos_i in WCS,
        #       body_orn in Body CS, mapping vector from WCS to BCS (for the compliance to the ETH's paper)
        #       acc_bias and gyro_bias in Body CS
        self.ns = 3+3+4+3*self.n_legs+3+3
        self.nis = self.ns - 1 # dim of body orientation = 4, independent dim of body orientation = 3 
        self.xk = np.zeros(self.ns)

        # measurement definition
        # sk = [rel_tip_pos1, rel_tip_pos2, ..., rel_tip_posN]
        self.nm = 3 * self.n_legs
        self.nl = 3 * self.n_legs


    def reset_state(self, 
                    body_pos: np.ndarray, 
                    body_vel: np.ndarray,
                    body_orn: np.ndarray,
                    tip_pos_wcs: np.ndarray):
        """
            Reset robot state.
        """
        self.xk[0:3] = body_pos
        self.xk[3:6] = body_vel
        self.xk[6:10] = body_orn
        self.xk[10:10+self.nl] = tip_pos_wcs
        self.xk[10+self.nl:13+self.nl] = np.zeros(3)
        self.xk[13+self.nl:16+self.nl] = np.zeros(3)

        self.Pk = np.zeros((self.nis, self.nis))


    def update(self, 
               body_ang_vel: np.ndarray, 
               body_lin_acc: np.ndarray, 
               jnt_act_pos: np.ndarray, 
               jnt_act_vel: np.ndarray, 
               jnt_act_trq: np.ndarray, 
               support_state: np.ndarray):
        
        # predict next state: xk+1_pred = pred(xk)
        self.state_predict(body_ang_vel, body_lin_acc)
        # update matrices of Kalman filter Fk, Qk, Hk, Rk
        self.update_matrices()
        # calculate Kalman gains and adjustment dx
        self.estimate()
        # correct next state: xk+1_est = xk+1_pred + dx
        self.state_correct()


    def state_predict(self, body_ang_vel, body_lin_acc):

        dt = self.dt
        d2t = self.dt**2
        rk = self.xk[0:3]
        vk = self.xk[3:6]
        qk = self.xk[6:10]
        bfk = self.xk[10+self.nl:13+self.nl]
        bwk = self.xk[13+self.nl:16+self.nl]

        self.fk = body_lin_acc - bfk
        self.wk = body_ang_vel - bwk

        self.ak = self.Ck.T @ self.fk + self.g
        self.Ck = rot.from_quat(qk).as_matrix()
        rkp1 = rk + dt * vk + 0.5 * d2t * self.ak
        vkp1 = vk + dt * self.ak
        qkp1 = quat_prod(so3_to_quat(dt * self.wk), qk)

        self.xkp1 = self.xk.copy()
        self.xkp1[0:3] = rkp1
        self.xkp1[3:6] = vkp1 
        self.xkp1[6:10] = qkp1


    def update_matrices(self):
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
        Fk[6:9, 12+self.nl:15+self.nl] = g1.T
        self.Fk = Fk

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

        for leg in range(self.n_legs):
            Qp = np.eye(3) # TODO: adjusting according to the actual support state of each leg
            Qk[9+leg*3:12+leg*3, 9+leg*3:12+leg*3] = dt * self.Ck.T @ Qp @ self.Ck

        self.Qk = Qk

        # Update yk
        

        # Update Hk
        Hk = np.zeros((self.nm, self.nis))

        # Update Rk


    def estimate(self):
        Pkp1 = self.Fk @ self.Pk @ self.Fk.T + self.Qk          # dim(Pkp1) = nis x nis
        Sk = self.Hk @ Pkp1 @ self.Hk.T + self.Rk               # dim(Sk) = nm x nm
        self.Kk = Pkp1 @ self.Hk.T @ np.linalg.inv(Sk)          # dim(Kk) = nis x nm
        self.dx = self.Kk @ self.yk                             # dim(dx) = nis x 1
        self.Pk = (np.eye(self.nis) - self.Kk @ self.Hk) @ Pkp1 # dim(Pk) = nis x nis


    def state_correct(self):
        pass


    def get_results(self) -> np.ndarray:
        pass 

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


def quat_prod(q1, q2):
    s1 = q1[3], s2 = q2[3]
    v1 = q1[0:3], v2 = q2[0:3]
    q = np.zeros(4)
    q[0:3] = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    q[3] = s1 * s2 - np.dot(v1, v2)
    return q


def so3_to_quat(so3):
    n = np.linalg.norm(so3)
    q = np.zeros(4)
    q[0:3] = math.sin(0.5*n) / n * so3
    q[3] = math.cos(0.5*n)
    return q


def gamma0(w, dt):
    th = np.linalg.norm(w)
    wn = 1/th * w
    wnx = skew(wn)
    # apply Rodrigues' rotation formula
    gamma = np.eye(3) + \
            math.sin(th*dt) * wnx + \
            (1 - math.cos(th*dt)) * (wnx @ wnx)
    return gamma


def gamma1(w, dt):
    th = np.linalg.norm(w)
    wn = 1/th * w
    wnx = skew(wn)
    gamma = dt * np.eye(3) + \
            (1 - math.cos(th*dt)) / th * wnx + \
            (th*dt - math.sin(th*dt)) / th * (wnx @ wnx)
    return gamma


def gamma2(w, dt):
    th = np.linalg.norm(w)
    wn = 1/th * w
    wnx = skew(wn)
    gamma = 1./2. * dt**2 * np.eye(3) + \
            (th*dt - math.sin(th*dt)) / (th**2) * wnx +  \
            (math.cos(th*dt) - 1. + 1./2. * dt**2 * th**2) / (th**2) * (wnx @ wnx)
    return gamma

def gamma3(w, dt):
    th = np.linalg.norm(w)
    wn = 1/th * w
    wnx = skew(wn)
    gamma = 1./6. * dt**3 * np.eye(3) + \
            (math.cos(th*dt) - 1. + 1./2. * dt**2 * th**2) / (th**3) * wnx +  \
            (math.sin(th*dt) - th*dt + 1./6. * dt**3 * th**3) / (th**3) * (wnx @ wnx)
    return gamma
