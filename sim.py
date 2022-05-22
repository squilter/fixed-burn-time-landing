#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

TOTAL_BURN_TIME=3.45

def thrust(burn_time_remaining):
    # https://www.thrustcurve.org/motors/Estes/F15/
    # in seconds
    time = [0.148, 0.228, 0.294, 0.353, 0.382, 0.419, 0.477, 0.52, 0.593, 0.688, 0.855, 1.037, 1.205, 
            1.423, 1.452, 1.503, 1.736, 1.955, 2.21, 2.494, 2.763, 3.12, 3.382, 3.404, 3.418, 3.45]
    
    # in Newtons
    thrust = [7.638, 12.253, 16.391, 20.21, 22.756, 25.26, 23.074, 20.845, 19.093, 17.5, 16.225, 15.427, 14.948,
              14.627, 15.741, 14.785, 14.623, 14.303, 14.141, 13.819, 13.338, 13.334, 13.013, 9.352, 4.895, 0]

    time_lookup = TOTAL_BURN_TIME - burn_time_remaining
    if time_lookup < 0 or time_lookup > TOTAL_BURN_TIME:
        return 0
    return np.interp(time_lookup, time, thrust)

def mass(burn_time_remaining):
    # From a screengrab, seemed like mass was 1.1kg before burn started and 1.04kg after it finished
    slope = (1.10-1.04) / TOTAL_BURN_TIME
    return np.clip(burn_time_remaining * slope + 1.04, 1.04, 1.10)

# plot thrust curve
plt.gca().invert_xaxis()
times = np.linspace(4, -1, 100)
plt.plot(times, [thrust(t) for t in times])
plt.show()
