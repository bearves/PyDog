import math
import numpy as np
from scipy.special import comb


class BezierKeyPointBuilder(object):
    """
        Key point builder of the Bezier curve generator for swing trajectory.
    """
    def __init__(self) -> None:
        """
            create a key point builder
        """
        pass 


    def build_kp_normal(self, 
                        start: np.ndarray, 
                        end: np.ndarray, 
                        height: float, 
                        steep: float) -> np.ndarray:
        """
            build swing curve's key points for normal terrain condition.

            Parameters:
                start (array(3)):  start point of the swing curve in 3D Cartesian space.
                end   (array(3)):  end point of the swing curve in 3D Cartesian space.
                height (float)  :  nominal height of the step.
                steep  (float)  :  a coefficient to indicate the steepness of the swing curve.
                                   the more steep, the more the curve is like a rectangle. 
        """
        key_pts = np.zeros((3, 5))
        key_pts[:, 0] = start
        key_pts[:, 1] = start
        key_pts[:, 2] = 0.5 * (start + end)
        key_pts[:, 3] = end
        key_pts[:, 4] = end
        
        key_pts[2, 1] += height * steep
        key_pts[2, 2] += height
        key_pts[2, 3] += height * steep
        return key_pts
    
    def build_kp_stair(self, 
                       start: np.ndarray, 
                       end: np.ndarray, 
                       height: float, 
                       steep: float) -> np.ndarray:
        """
            build swing curve's key points for stair terrain condition.

            Parameters:
                start (array(3)):  start point of the swing curve in 3D Cartesian space.
                end   (array(3)):  end point of the swing curve in 3D Cartesian space.
                height (float)  :  nominal height of the step.
                steep  (float)  :  a coefficient to indicate the steepness of the swing curve.
                                   the more steep, the more the curve is like a rectangle. 
        """
        key_pts = np.zeros((3, 6))
        key_pts[:, 0] = start
        key_pts[:, 1] = -0.2 * end +  1.2 * start
        key_pts[:, 2] =  0.3 * end +  0.7 * start
        key_pts[:, 3] =  0.7 * end +  0.3 * start
        key_pts[:, 4] =  1.2 * end + -0.2 * start
        key_pts[:, 5] = end
        
        key_pts[2, 1] += height * steep
        key_pts[2, 2] += height
        key_pts[2, 3] += height
        key_pts[2, 4] += height * steep
        return key_pts


class BezierCurve(object):
    """
        Beizer curve generator
    """
    n_pts   : int         # number of the control points of the curve
    order   : int         # polynomial order of the curve
    key_pts : np.ndarray  # control points, i.e. key points
    coe     : np.ndarray  # binomial coefficient of the bezier curve term

    # useful number sequence
    seq_0_to_N : np.ndarray   # sequence of [0, 1, 2, ..., order]
    seq_N_to_0 : np.ndarray   # sequence of [order, order-1, ..., 0] 

    def __init__(self) -> None:
        """
            create a Bezier curve generator
        """
        pass


    def set_key_points(self, key_pts: np.ndarray):
        """
            set the control point of the Bezier curve.

            Parameters:
                key_pts (array(3,n)):  control points in 3D cartesian space.
        """
        n_pts = key_pts.shape[1]
        
        assert key_pts.shape[0] == 3, "shape of key points must be (3, n)"
        assert n_pts >= 3, "number of key points must be more than 3"

        self.n_pts = n_pts
        self.order = n_pts - 1
        self.coe = comb(self.order, np.linspace(0, self.order, n_pts))
        self.key_pts = key_pts.copy()
        self.seq_0_to_N = np.arange(0, self.n_pts)
        self.seq_N_to_0 = np.flip(self.seq_0_to_N) # reverse


    def get_pva_at(self, r: float, rdot: float, rddot: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            get the position, velocity, acceleration of the curve at the pivot point (r, rdot, rddot).

            Parameters:
                r (float): the pivot point between 0 and 1.
                rdot (float): the time derivative of r.
                rddot(float): the second order time derivative of r.

            Returns:
                p (array(3)): the position of the curve at the pivot point.
                v (array(3)): the velocity of the curve at the pivot point.
                a (array(3)): the acceleration of the curve at the pivot point.
        """
        s = 1 - r
        rlog = np.logspace(0, self.order, self.n_pts, base = r)
        slog = np.logspace(self.order, 0, self.n_pts, base = s)

        # p(t) = p0 * C(N,0) * (1-r)^N     * r^0 + 
        #        p1 * C(N,1) * (1-r)^(N-1) * r^1 +
        #        p2 * C(N,2) * (1-r)^(N-2) * r^2 +
        #        ...
        #        pN * C(N,N) * (1-r)^0     * r^N 
        p = self.key_pts @ (slog * rlog * self.coe).T

        # v(t) = p0 * C(N,0) * (1-r)^(N-1) * (N-0) * -1 * r^0 + p0 * C(N,0) * (1-r)^(N-0) * r^-1 * 0 + 
        #        p1 * C(N,1) * (1-r)^(N-2) * (N-1) * -1 * r^1 + p1 * C(N,1) * (1-r)^(N-1) * r^0  * 1 +
        #        p2 * C(N,2) * (1-r)^(N-3) * (N-2) * -1 * r^2 + p2 * C(N,2) * (1-r)^(N-2) * r^1  * 2 +
        #        ...
        #        pN * C(N,N) * (1-r)^-1    * 0     * -1 * r^N + pN * C(N,N) * (1-r)^0     * r^(N-1) * N
        if math.fabs(r) > 1e-6:
            rvlog = rlog / r * self.seq_0_to_N
        else:
            rvlog = 0 * rlog
            rvlog[1] = 1
        
        if math.fabs(s) > 1e-6:
            svlog = slog / s * self.seq_N_to_0
        else:
            svlog = 0 * slog
            svlog[self.n_pts - 2] = 1

        dpdr = self.key_pts @ ((-1 * svlog * rlog + slog * rvlog) * self.coe).T
        v = dpdr * rdot

        # a(t) = p0 * C(N,0) * (1-r)^(N-2) * (N-0) * (N-1) * r^0 + p0 * C(N,0) * (1-r)^(N-1) * (N-0) * -2 * r^-1    * 0 + p0 * C(N,0) * (1-r)^(N-0) * r^-2    * 0 * -1 +
        #        p1 * C(N,1) * (1-r)^(N-3) * (N-1) * (N-2) * r^1 + p1 * C(N,1) * (1-r)^(N-2) * (N-1) * -2 * r^0     * 1 + p1 * C(N,1) * (1-r)^(N-1) * r^-1    * 1 * 0
        #        p2 * C(N,2) * (1-r)^(N-4) * (N-2) * (N-3) * r^2 + p2 * C(N,2) * (1-r)^(N-3) * (N-2) * -2 * r^1     * 2 + p2 * C(N,2) * (1-r)^(N-2) * r^0     * 2 * 1
        #        ...
        #        pN * C(N,N) * (1-r)^-2    * 0     * -1    * r^N + pN * C(N,N) * (1-r)^-1    * 0     * -2 * r^(N-1) * N + pN * C(N,N) * (1-r)^0     * r^(N-2) * N * N-1
        if math.fabs(r) > 1e-6:
            ralog = rvlog / r * (self.seq_0_to_N - 1)
        else:
            ralog = 0 * rvlog
            ralog[2] = 2
        
        if math.fabs(s) > 1e-6:
            salog = svlog / s * (self.seq_N_to_0 - 1)
        else:
            salog = 0 * svlog
            salog[self.n_pts - 3] = 2
        
        d2pdr2 = self.key_pts @ ((salog * rlog - 2 * svlog * rvlog + slog * ralog) * self.coe).T
        a = d2pdr2 * rdot * rdot + dpdr * rddot
        return p, v, a


class SineCurve(object):
    """
        Sine curve generator
    """
    start_point : np.ndarray
    end_point : np.ndarray
    height : float

    def __init__(self) -> None:
        """
            create a sin curve generator
        """
        pass


    def set_key_points(self, start: np.ndarray, end: np.ndarray, height: float):
        """
            set the sine curve's key points.

            Parameters:
                start (array(3)):  start point of the sine curve in 3D Cartesian space.
                end   (array(3)):  end point of the sine curve in 3D Cartesian space.
                height (float)  :  nominal height of the sine curve.
        """
        self.start_point = start
        self.end_point = end
        self.height = height


    def get_pva_at(self, r: float, rdot: float, rddot: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            get the position, velocity, acceleration of the curve at the pivot point (r, rdot, rddot).

            Parameters:
                r (float): the pivot point between 0 and 1.
                rdot (float): the time derivative of r.
                rddot(float): the second order time derivative of r.

            Returns:
                p (array(3)): the position of the curve at the pivot point.
                v (array(3)): the velocity of the curve at the pivot point.
                a (array(3)): the acceleration of the curve at the pivot point.
        """
        p = (1 - r) * self.start_point + r * self.end_point
        # add height curve
        p[2] += self.height * math.sin(math.pi * r)

        v = (self.end_point - self.start_point) * rdot
        v[2] += self.height * math.cos(math.pi * r) * math.pi * rdot

        a = (self.end_point - self.start_point) * rddot
        a[2] += self.height * math.cos(math.pi * r) * math.pi * rddot - \
                self.height * math.sin(math.pi * r) * (math.pi * rdot)**2
        return p, v, a

