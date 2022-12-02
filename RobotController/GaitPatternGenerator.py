import numpy as np
import math


class GaitPatternGenerator(object):
    """
        Gait pattern generator to generate swing and stance sequences, aka. support state, for each legs.
    """
    
    n_leg: int = 4
    gait_name: str = "Stand"                     # name
    total_period: float = 0.5                    # gait period of a whole step
    offset: np.ndarray = np.array([0, 0, 0, 0])  # offset ratio of the touch down time in a step
    duty: np.ndarray   = np.array([1, 1, 1, 1])  # ratio of the supporting time in a step

    time: float = 0        # current absolute time
    start_time: float = 0  # time point to start planning
    net_time: float = 0    # abs_time - start_time
    cycles: int = 0        # number of steps for the whole robot
    total_phase: float = 0 # phase in a step of the whole robot
    
    phase: np.ndarray = np.zeros(4) # phase for each leg

    def __init__(self, name: str, total_period: float, offset: np.ndarray, duty: np.ndarray) -> None:
        """
            Create a gait pattern generator.

            Parameters:
                name (str): gait name
                total_period (float): gait period of a whole step
                offset (array(n_leg)): offset ratio of the touch down time for each leg in a step
                duty (array(n_leg)): ratio of the supporting time for each leg in a step
        """
        self.gait_name = name
        self.total_period = total_period
        self.offset = offset
        self.duty = duty
        self.time = 0
        self.start_time = 0
        self.net_time = 0
        self.cycles = 0


    def set_start_time(self, time_now: float):
        """
            Set the start time point for the generator to plan the swing/stance sequence.

            Parameters:
                time_now (float) : current absolute time.
        """
        self.start_time = time_now


    def set_current_time(self, time_now: float):
        """
            Set the current time of the gait pattern generator, and update gait phases for each leg.

            Parameters:
                time_now (float) : current absolute time.
        """
        self.time = time_now
        self.net_time = self.time - self.start_time
        self.total_phase, self.cycles = math.modf(self.net_time/self.total_period)
        for i in range(self.n_leg):
            self.phase[i] = self.total_phase - self.offset[i]
            if self.phase[i] < 0:
                self.phase[i] += 1.

    
    def get_current_support_state(self) -> np.ndarray:
        """
            Obtain the support state at current time.

            Returns:
                support_state (array(n_leg)) : current support state of each leg, stance = 1, swing = 0.
        """
        support_state = np.zeros(4)
        for i in range(self.n_leg):
            if self.phase[i] < self.duty[i]:
                support_state[i] = 1 # stance
            else:
                support_state[i] = 0 # swing
        return support_state
        

    def predict_mpc_support_state(self, dt_mpc: float, horizon_length: int) -> np.ndarray:
        """
            Obtain the support state sequence within the prediction horizon.

            Parameters:
                dt_mpc        (float): the time interval of a prediction step in MPC.
                horizon_length (int) : length of the MPC horizon, i.e. the prediction steps ahead current time.
            
            Returns:
                mpc_support_state (array(horizon_length, n_leg)): 
                    the support state sequence within the prediction horizon.
        """
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
        """
            Indicate whether a step has finished, and the robot can switch to another gait pattern.

            Returns:
                indicator (bool): can switch = True, cannot switch = False.
        """
        if math.fabs(self.total_phase) < 1e-2:
            return True
        return False


    def get_current_swing_time_ratio(self, leg: int) -> tuple[float, float]:
        """
            Get the current time ratio and its time derivative in the swing phase for a leg.
            This is useful for swing leg trajectory planning and the next foothold prediction.

            Parameters:
                leg (int): the index of the leg.

            Returns:
                swing_time_ratio (float): time ratio in the swing phase.
                swing_time_ratio_dot (float): time derivative of the time ratio in the swing phase. 
        """
        if self.phase[leg] < self.duty[leg]:
            return 0, 0
        else:
            swing_time_ratio = (self.phase[leg] - self.duty[leg])/(1. - self.duty[leg])
            swing_time_ratio_dot = 1/self.total_period/(1. - self.duty[leg])
            return swing_time_ratio, swing_time_ratio_dot

    def get_current_support_time_ratio(self, leg: int) -> tuple[float, float]:
        """
            Get the current time ratio and its time derivative in the support phase for a leg.
            This is useful for state estimator to adjust the covariance of leg tip position in the 
            estimation model.

            Parameters:
                leg (int): the index of the leg.

            Returns:
                support_time_ratio (float): time ratio in the support phase.
                support_time_ratio_dot (float): time derivative of the time ratio in the support phase. 
        """
        if self.phase[leg] > self.duty[leg]:
            return 0, 0
        else:
            support_time_ratio = self.phase[leg]/self.duty[leg]
            support_time_ratio_dot = 1/self.total_period/self.duty[leg]
            return support_time_ratio, support_time_ratio_dot
    

    def get_swing_time_left(self) -> np.ndarray:
        """
            Get the time left in the swing phase for all legs.
            This is useful for swing leg trajectory planning and the next foothold prediction.

            Returns:
                swing_time_left (array(n_leg)): time left in the swing phase for all legs.
        """
        swing_time_left = np.zeros(self.n_leg)
        for leg in range(self.n_leg):
            if self.phase[leg] < self.duty[leg]: # stance
                swing_time_left[leg] = 0
            else:
                swing_time_ratio = (self.phase[leg] - self.duty[leg])/(1. - self.duty[leg])
                swing_time = (1. - self.duty[leg]) * self.total_period
                swing_time_left[leg] = (1. - swing_time_ratio) * swing_time
        return swing_time_left

    
    def get_stance_duration(self) -> float:
        """
            Get the time duration of the stance phase.

            Returns:
                stance_duration (float): time duration of the stance phase.
        """
        return self.total_period * self.duty