import numpy as np
import matplotlib.pyplot as plt

# The dataset says 3.45 but it's convenient if this is a multiple of DT
TOTAL_BURN_TIME=3.334

DT = 1/3
TIME_BUCKETS = int(TOTAL_BURN_TIME * 1/DT)
VEL_BUCKETS = int(20)
HEIGHT_BUCKETS = int(40 * 1/DT)

times = np.linspace(0, TOTAL_BURN_TIME, TIME_BUCKETS)
vels = np.linspace(0, 19, VEL_BUCKETS)
heights = np.linspace(0, 40, HEIGHT_BUCKETS)
actions = np.linspace(20,100,9)

def thrust(burn_time_remaining):
    # https://www.thrustcurve.org/motors/Estes/F15/
    # in seconds
    time = [0.148, 0.228, 0.294, 0.353, 0.382, 0.419, 0.477, 0.52, 0.593, 0.688, 0.855, 1.037, 1.205, 
            1.423, 1.452, 1.503, 1.736, 1.955, 2.21, 2.494, 2.763, 3.12, 3.382, 3.404, 3.418, 3.45]
    
    # in Newtons
    thrust = [7.638, 12.253, 16.391, 20.21, 22.756, 25.26, 23.074, 20.845, 19.093, 17.5, 16.225, 15.427, 14.948,
              14.627, 15.741, 14.785, 14.623, 14.303, 14.141, 13.819, 13.338, 13.334, 13.013, 9.352, 4.895, 0]

    # Shifting the dataset 50ms makes it fit much better at dt=1/3
    time_lookup = TOTAL_BURN_TIME - burn_time_remaining + 0.05
    if time_lookup < 0 or time_lookup > 3.45:
        return 0
    return np.interp(time_lookup, time, thrust)

def mass(burn_time_remaining):
    # From a screengrab, seemed like mass was 1.1kg before burn started and 1.04kg after it finished
    slope = (1.10-1.04) / TOTAL_BURN_TIME
    return np.clip(burn_time_remaining * slope + 1.04, 1.04, 1.10)

def nearest(a, a0):
    # find nearest value in a to a0
    return a[np.abs(a - a0).argmin()]

def nearest_state(t, vel, height):
    new_state = (nearest(times, t), nearest(vels, vel), nearest(heights, height))
    return new_state

def dynamics(state, action, dt):
    t, vel, height = state
    # Assume actuation is instantaneous
    new_vel = vel + (-(action/100 * thrust(t))/mass(t) + 9.8)*dt
    new_height = height - new_vel*dt
    new_t = t - dt
    return nearest_state(new_t, new_vel, new_height)

# Plot thrust curve
if(__name__ == '__main__'):
    plt.gca().invert_xaxis()
    # Since we are discretizing with only a few time buckets, it's important that the result is not distorted by discretization.
    # Try plotting with TIME_BUCKETS=10 and with 1000 and make sure it looks similar
    TIME_BUCKETS = 10
    times = np.linspace(TOTAL_BURN_TIME, 0, TIME_BUCKETS)
    plt.plot(times, [thrust(t) for t in times])
    plt.show()