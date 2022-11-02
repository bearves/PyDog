import sys,os
sys.path.append(os.getcwd()+'/..')

import numpy as np
import RobotController.GaitPatternGenerator as gg

trot_generator = gg.GaitPatternGenerator('trot', 0.5, np.array([0,0.5,0.5,0]), np.array([0.8, 0.8, 0.8, 0.8]))

trot_generator.set_start_time(0.5)
trot_generator.set_current_time(1.0)

ss = trot_generator.get_current_support_state()

print(ss)

ss_pred = trot_generator.predict_mpc_support_state(32, 32 * 0.001)
print(ss_pred)