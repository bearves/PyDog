import numpy as np
np.set_printoptions(precision=4)

class QuadTerrainEstimator(object):

    n_leg: int = 4
    n_step: int = 2
    
    gnd_pts: np.ndarray
    search_flag: np.ndarray

    A: np.ndarray 
    b: np.ndarray

    plane_normal: np.ndarray
    plane_d: float

    def __init__(self) -> None:
        self.gnd_pts = np.zeros((self.n_leg*3, self.n_step))
        self.search_flag = np.zeros(self.n_leg)
        self.b = -1. * np.ones(self.n_leg * self.n_step)
        self.plane_normal = np.array([0, 0, 1])
        self.plane_d = 0


    def reset(self, tip_pos_wcs: np.ndarray):
        # set initial values for ground
        self.search_flag = np.zeros(self.n_leg)
        for i in range(self.n_step):
            self.gnd_pts[:, i] = tip_pos_wcs
        self.solve_plane()


    def update(self, tip_pos_wcs: np.ndarray, support_phase: np.ndarray):
        # record tip pos of the neutral point
        for leg in range(self.n_leg):
            if self.search_flag[leg] == 0 and support_phase[leg] >= 0.3 and support_phase[leg] < 0.49:
                # set search flag
                self.search_flag[leg] = 1
            if self.search_flag[leg] == 1 and support_phase[leg] >= 0.7:
                # found the neutral stance moment and reset search flag
                self.search_flag[leg] = 0
                # push back new neutral stance point
                if (self.n_step > 1):
                    self.gnd_pts[leg*3:3+leg*3, 1:self.n_step] = self.gnd_pts[leg*3:3+leg*3, 0:self.n_step-1]
                self.gnd_pts[leg*3:3+leg*3, 0] = tip_pos_wcs[leg*3:3+leg*3]
                # calculate new ground plane
                self.solve_plane()

    
    def solve_plane(self):
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
        
        print('New terrain estimated:', self.get_plane_normal())
        if (self.get_plane_normal()[2] < 0.6):
            print('Strange result:')
            print(self.A)
            print(self.gnd_pts)


    def get_plane_normal(self) -> np.ndarray:
        return self.plane_normal
    

    def get_point_projection(self, 
                             point: np.ndarray, 
                             vector: np.ndarray) -> tuple[np.ndarray, float]:
        b = np.dot(self.plane_normal, point) + self.plane_d
        h = -b / np.dot(self.plane_normal, vector)
        p_proj = point + h * vector
        return (p_proj, h)