import numpy as np

class QuadTerrainEstimator(object):

    n_leg: int = 4
    n_step: int = 2
    
    gnd_pts: np.ndarray

    def __init__(self) -> None:
        self.gnd_pts = np.zeros((self.n_leg*3, self.n_step))

    def reset(self, tip_pos_wcs):
        # set initial values for ground
        for i in range(self.n_step):
            self.gnd_pts[:, i] = tip_pos_wcs


    def update(self, tip_pos_wcs, support_phase):
        # record tip pos of the neutral point
        for leg in range(self.n_leg):
            if support_phase >= 0.5:
                # found a neutral point
                pass
        # calculate ground plane
        # update ground normal 