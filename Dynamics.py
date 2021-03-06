import numpy as np
import matplotlib.pyplot as plt
import bisect
from scipy import integrate

# The dataset says 3.45 but it's convenient if this is a multiple of DT
TOTAL_BURN_TIME = 3.333333333334

DT = 1 / 3 # must be something like 1/3, 1/6, 1/12
TIME_BUCKETS = int(TOTAL_BURN_TIME * 1 / DT) # don't touch
VEL_BUCKETS = 30
HEIGHT_BUCKETS = 150
ACTION_BUCKETS = 21
HEIGHT_MAX = 30
VEL_MAX = 18
VEL_MIN=-2
HEIGHT_MIN=-2

times = np.linspace(-DT, TOTAL_BURN_TIME, TIME_BUCKETS+2)
assert np.all((np.diff(times)-DT)<0.0000001)
# Gotta allow it to go negative so that it can be punished for doing that
vels = np.linspace(VEL_MIN, VEL_MAX, VEL_BUCKETS)
heights = np.linspace(HEIGHT_MIN, HEIGHT_MAX, HEIGHT_BUCKETS)
# best if these are ints because we write them to memory as uint8's:
actions = np.linspace(40, 100, ACTION_BUCKETS)
assert(np.all([a.is_integer() for a in actions]))

valid_states = set()  # (time, vel, height)
for t in times:
    for vel in vels:
        for height in heights:
            state = (t, vel, height)
            valid_states.add(state)


def thrust(burn_time_remaining):
    # https://www.thrustcurve.org/motors/Estes/F15/
    # in seconds
    time = [
        0.148,
        0.228,
        0.294,
        0.353,
        0.382,
        0.419,
        0.477,
        0.52,
        0.593,
        0.688,
        0.855,
        1.037,
        1.205,
        1.423,
        1.452,
        1.503,
        1.736,
        1.955,
        2.21,
        2.494,
        2.763,
        3.12,
        3.382,
        3.404,
        3.418,
        3.45,
    ]

    # in Newtons
    thrust = [
        7.638,
        12.253,
        16.391,
        20.21,
        22.756,
        25.26,
        23.074,
        20.845,
        19.093,
        17.5,
        16.225,
        15.427,
        14.948,
        14.627,
        15.741,
        14.785,
        14.623,
        14.303,
        14.141,
        13.819,
        13.338,
        13.334,
        13.013,
        9.352,
        4.895,
        0,
    ]

    time_lookup = TOTAL_BURN_TIME - burn_time_remaining
    if time_lookup < 0 or time_lookup > 3.45:
        return 0
    return np.interp(time_lookup, time, thrust)

# integration is expensive so just save the result for subsequent calls
# TODO check this again
dv_precomputed = {}
def delta_v(t0, t1, action):
    if (t0, t1, action) not in dv_precomputed:
        f = lambda t : thrust(t)/mass(t)
        dv_precomputed[(t0, t1, action)] = (action / 100) * integrate.quad(f, t1, t0)[0]

    return dv_precomputed[(t0, t1, action)]

def mass(burn_time_remaining):
    # From a screengrab, seemed like mass was 1.1kg before burn started and 1.04kg after it finished
    slope = (1.10 - 1.04) / TOTAL_BURN_TIME
    return np.clip(burn_time_remaining * slope + 1.04, 1.04, 1.10)


def nearest(a, a0):
    # find nearest value in a to a0
    return a[np.abs(a - a0).argmin()]


def nearest_state(t, vel, height):
    new_state = (nearest(times, t), nearest(vels, vel), nearest(heights, height))
    return new_state


def dynamics(state, action):
    t,v,h = dynamics_dt(state, action, DT)
    if t < times[0]:
        t = times[0]
    old_t = t
    t = nearest(times,t)
    assert abs(old_t-t) < 0.000001, "Something about DT is broken!"
    return (t,v,h)

def dynamics_dt(state, action, dt):
    t, vel, height = state
    # Assume actuation is instantaneous
    new_t = t - dt
    new_vel = vel - delta_v(t, new_t, action) + 9.8 * dt
    new_height = height - (vel+new_vel)/2 * dt
    return (new_t, new_vel, new_height)

def dynamics_dt_no_motor(state, dt):
    t, vel, height = state
    new_vel = vel + 9.8*dt
    new_height = height - (vel+new_vel)/2 * dt
    new_t = t # t represents motor burn time remaining, so no change
    return (new_t, new_vel, new_height)

# Plot thrust curve
if __name__ == "__main__":
    plt.gca().invert_xaxis()
    ts = np.linspace(TOTAL_BURN_TIME, 0, 1000)
    plt.plot(ts, [thrust(t) for t in ts])
    plt.show()
