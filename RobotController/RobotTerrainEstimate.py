import numpy as np

class QuadTerrainEstimator(object):

    n_leg: int = 4
    n_step: int = 2
    
    gnd_pts: np.ndarray

    def __init__(self) -> None:
        self.gnd_pts = np.zeros((self.n_leg*3, self.n_step))

    def reset(self):
        # set initial value for ground
        pass

    def update(self, tip_pos, support_phase):
        # take tip pos and support phase as input
        # record tip pos of the neutral point
        # calculate ground plane
        # update ground normal 
        pass 