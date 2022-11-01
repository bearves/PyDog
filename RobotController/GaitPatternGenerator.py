import numpy as np
import math


class GaitPatternGenerator(object):
    
    n_leg: int = 4
    gait_name: str = "Stand"
    total_period: float = 0.5
    offset: np.ndarray = np.array([0, 0, 0, 0])
    duty: np.ndarray   = np.array([1, 1, 1, 1])

    time: float = 0
    start_time: float = 0
    net_time: float = 0
    cycles: int = 0
    total_phase: float = 0
    
    phase: np.ndarray = np.zeros(4)

    def __init__(self, name: str, total_peroid: float, offset: np.ndarray, duty: np.ndarray) -> None:
        self.gait_name = name
        self.total_period = total_peroid
        self.offset = offset
        self.duty = duty
        self.time = 0
        self.start_time = 0
        self.net_time = 0
        self.cycles = 0


    def set_start_time(self, time_now: float):
        self.start_time = time_now


    def set_current_time(self, time_now: float):
        self.time = time_now
        self.net_time = self.time - self.start_time
        self.total_phase, self.cycles = math.modf(self.net_time/self.total_period)
        for i in range(self.n_leg):
            self.phase[i] = self.total_phase - self.offset[i]
            if self.phase[i] < 0:
                self.phase[i] += 1.

    
    def get_current_support_state(self) -> np.ndarray:
        support_state = np.zeros(4)
        for i in range(self.n_leg):
            if self.phase[i] < self.duty[i]:
                support_state[i] = 1 # stance
            else:
                support_state[i] = 0 # swing
        return support_state
        

    def predict_mpc_support_state(self, horizon_length: int, dt_mpc: float) -> np.ndarray:
        mpc_support_state = np.zeros((horizon_length, self.n_leg))
        for t in range(horizon_length):
            predict_time = t * dt_mpc + self.time
            total_phase, cycle = math.modf((predict_time - self.start_time)/self.total_period)
            for i in range(self.n_leg):
                phase_leg = total_phase - self.offset[i]
                if phase_leg < 0:
                    phase_leg += 1.
                if phase_leg < self.duty[i]:
                    mpc_support_state[t, i] = 1 # stance
                else:
                    mpc_support_state[t, i] = 0 # swing

        return mpc_support_state
    
    def can_switch(self) -> bool:
        if math.fabs(self.total_phase) < 1e-6:
            return True
        return False

    def get_current_swing_time_ratio(self, leg: int) -> tuple[float, float]:
        if self.phase[leg] < self.duty[leg]:
            return 0, 0
        else:
            swing_time_ratio = (self.phase[leg] - self.duty[leg])/(1. - self.duty[leg])
            swing_time_ratio_dot = 1/self.total_period/(1. - self.duty[leg])
            return swing_time_ratio, swing_time_ratio_dot
    
    def get_swing_time_left(self):
        swing_time_left = np.zeros(self.n_leg)
        for leg in range(self.n_leg):
            if self.phase[leg] < self.duty[leg]: # stance
                swing_time_left[leg] = 0
            else:
                swing_time_ratio = (self.phase[leg] - self.duty[leg])/(1. - self.duty[leg])
                swing_time = (1. - self.duty[leg]) * self.total_period
                swing_time_left[leg] = (1. - swing_time_ratio) * swing_time
        return swing_time_left

    
    def get_stance_duration(self):
        return self.total_period * self.duty