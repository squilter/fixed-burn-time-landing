#!/usr/bin/env python3

import time

import numpy as np
import matplotlib.pyplot as plt

# The dataset says 3.45 but it's convenient if this is a multiple of DT
TOTAL_BURN_TIME=3.334

DT = 1/3
TIME_BUCKETS = int(TOTAL_BURN_TIME * 1/DT)
VEL_BUCKETS = int(20)
HEIGHT_BUCKETS = int(40 * 1/DT)

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

print(f"Time {TIME_BUCKETS}; Vel {VEL_BUCKETS}; Pos {HEIGHT_BUCKETS}; Table size {TIME_BUCKETS*VEL_BUCKETS*HEIGHT_BUCKETS} bytes")

# plot thrust curve
# plt.gca().invert_xaxis()
# times = np.linspace(TOTAL_BURN_TIME, 0, TIME_BUCKETS)
# plt.plot(times, [thrust(t) for t in times])
# plt.show()

costs = {} # maps from (time, vel, height) -> cost-to-go
times = np.linspace(0, TOTAL_BURN_TIME, TIME_BUCKETS)
vels = np.linspace(0, 19, VEL_BUCKETS)
heights = np.linspace(0, 40, HEIGHT_BUCKETS)
actions = np.linspace(20,100,9)
for t in times:
    for vel in vels:
        for height in heights:
            state = (t, vel, height)
            costs[state] = 0

def nearest(a, a0):
    # find nearest value in a to a0
    return a[np.abs(a - a0).argmin()]

def nearest_state(t, vel, height):
    new_state = (nearest(times, t), nearest(vels, vel), nearest(heights, height))
    assert new_state in costs
    return new_state

def dynamics(state, action):
    assert(state in costs)
    assert(action in actions)
    t, vel, height = state
    # Assume actuation is instantaneous
    new_vel = vel + (-(action/100 * thrust(t))/mass(t) + 9.8)*DT
    # print(f"with throttle {action} at time {t}, vel {vel} goes to {new_vel} which rounds to {nearest(vels, new_vel)}")
    new_height = height - new_vel*DT
    new_state = nearest_state(t - DT, new_vel, new_height)
    # assert new_state != state, "Self transition detected!"
    return new_state

def is_landed(state):
    assert state in costs
    t, vel, height = state
    if t > 0.00001:
        return False
    if vel > 1.00001:
         return False
    if height > 0.00001:
        return False
    return True

def is_crashed(state):
    t, vel, height = state
    return height <= 0.00001 and vel >= 2

def cost(state, action):
    assert state in costs
    assert action in actions
    if is_landed(state):
        return 0
    if is_crashed(state):
        return float("inf")
    
    # Let's have a slight preference for keeping throttle around 80%
    cost = 1
    if action > 91 or action < 69:
        cost = 1.1
    
    return costs[dynamics(state, action)] + cost

def value_iteration_batch_update():
    for state in costs.keys():
        new_costs = []
        for action in actions:
            new_costs.append(cost(state, action))
        new_cost = min(new_costs)
        costs[state] = new_cost

def extract_policy():
    policy = {}
    for state in costs.keys():
        min_cost = float("inf")
        best_action = None
        for action in actions:
            c = cost(state, action)
            if c < min_cost:
                min_cost = c
                best_action = action
        policy[state] = best_action
    return policy

def count_non_crashing_states():
    good = 0
    for c in costs.values():
        if c != float("inf"):
            good += 1
    return good

def calc_policy(batches = 10):
    start_time = time.time()
    for i in range(batches):
        value_iteration_batch_update()
        print(f"Batch {i} complete after {time.time() - start_time:.2f}s. ", end="")
        print(f"Landing from {100*count_non_crashing_states()/(TIME_BUCKETS*VEL_BUCKETS*HEIGHT_BUCKETS):.2f}% of starting configurations")
    return extract_policy()

def demo(policy):
    state = nearest_state(3, 12, 13)
    print(f"Starting at {state}")
    while not (is_landed(state) or is_crashed(state)):
        action = policy[state]
        state = dynamics(state, action)
        print(f"Applying {action}% for {DT:.2f}s. Resulting state is {state}")

policy = calc_policy()

demo(policy)