import numpy as np
np.set_printoptions(precision=4)

class QuadTerrainEstimator(object):
    """
        Terrain plane estimator of quadruped robot.
    """

    # constants
    n_leg: int = 4   # number of robot legs
    n_step: int = 2  # number of steps for ground estimation, the bigger n_step, the slower ground plane changes. 
    
    # states
    gnd_pts: np.ndarray     # tip position when the leg tip is on the ground 
    search_flag: np.ndarray # a flag indicating where to search a grounded point during a step

    # ground plane fitting helper matrices
    A: np.ndarray    # matrix holding all grounded points for fitting

    # ground plane fitting results
    # the ground plane is expressed as 
    # ax + by + cz + d = 0
    # where n = (a, b, c) is the normal vector, |n| = 1
    # d is the fourth parameter.
    plane_normal: np.ndarray   # normal vector of the fitted plane
    plane_d: float             # d parameter of the fitted plane


    def __init__(self) -> None:
        """
            Initialize the ground plane estimator
        """
        self.gnd_pts = np.zeros((self.n_leg*3, self.n_step))
        self.search_flag = np.zeros(self.n_leg)
        self.plane_normal = np.array([0, 0, 1])
        self.plane_d = 0


    def reset(self, tip_pos_wcs: np.ndarray):
        """
            Set initial values for ground.

            Parameters:
                tip_pos_wcs (array(n_leg*3)): the initial leg tip positions in WCS. 
        """
        self.search_flag = np.zeros(self.n_leg)
        for i in range(self.n_step):
            self.gnd_pts[:, i] = tip_pos_wcs
        self.fit_plane()


    def update(self, tip_pos_wcs: np.ndarray, support_phase: np.ndarray):
        """
            Update ground points and estimate ground planes.

            Parameters:
                tip_pos_wcs (array(n_leg*3)): the current leg tip positions in WCS.
                support_phase (array(n_leg)): the phase of supporting state of all legs. 
                                              for each leg, phase is a 0-1 scalar, that 
                                              phase=0 at touchdown, phase=1 at lifting up.
        """
        # Record tip pos of the grounded point, the grounded point is searched according
        # to the support phase.
        # NOTE: Here, we assume the leg tip at the moment of 70% supporting phase is largely
        # possible to solidly stand on the ground.
        for leg in range(self.n_leg):
            if self.search_flag[leg] == 0 and support_phase[leg] >= 0.3 and support_phase[leg] < 0.49:
                # set search flag during the first half of stance
                self.search_flag[leg] = 1
            if self.search_flag[leg] == 1 and support_phase[leg] >= 0.7:
                # found the 70% stance moment and reset search flag
                self.search_flag[leg] = 0
                # push back new guessed stance point to the grounded point set
                if (self.n_step > 1):
                    self.gnd_pts[leg*3:3+leg*3, 1:self.n_step] = self.gnd_pts[leg*3:3+leg*3, 0:self.n_step-1]
                self.gnd_pts[leg*3:3+leg*3, 0] = tip_pos_wcs[leg*3:3+leg*3]
                # fit new ground plane according to the new grounded points
                self.fit_plane()

    
    def fit_plane(self):
        """
            Fit ground plane using grounded points by SVD decomposition.
            See https://www.ltu.se/cms_fs/1.51590!/svd-fitting.pdf
            See also https://blog.csdn.net/iamqianrenzhan/article/details/103463130 (in Chinese)
        """
        self.A = self.gnd_pts.T.reshape((self.n_leg * self.n_step, 3), order='C')
        center = np.mean(self.A, axis=0)
        result = np.linalg.svd(self.A - center)
        normal = result[2][:, 2]
        normal = normal / np.linalg.norm(normal)
        if (normal[2] < 0):
            normal *= -1.
        
        d = -np.dot(center, normal)
        self.plane_normal = normal.copy()
        self.plane_d = d
        
        #print('New terrain estimated:', self.get_plane_normal())
        #if (self.get_plane_normal()[2] < 0.6):
        #    print('Strange result:')
        #    print(self.gnd_pts)


    def get_plane_normal(self) -> np.ndarray:
        """
            Get the normal of the fitted plane.

            Returns:
                plane_normal (array(3)): normal vector of the fitted plane.
        """
        return self.plane_normal
    

    def get_plane_d(self) -> float:
        """
            Get the d coefficient of the fitted plane.

            Returns:
                plane_d (float): the d coefficient of the fitted plane.
        """
        return self.plane_d

    
    def get_plane(self) -> tuple[np.ndarray, float]:
        """
            Get the parameters, including normal and d of the fitted plane.
            the ground plane is expressed as 
                            ax + by + cz + d = 0
            where n = (a, b, c) is the normal vector, |n| = 1
                  d is the fourth parameter.

            Returns:
                plane_normal (array(3)): normal vector of the fitted plane.
                plane_d (float): the d coefficient of the fitted plane.
        """
        return (self.plane_normal, self.plane_d)

