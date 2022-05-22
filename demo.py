#!/usr/bin/env python3

import time

import numpy as np
import matplotlib.pyplot as plt
from Dynamics import *

print(f"Time {TIME_BUCKETS}; Vel {VEL_BUCKETS}; Pos {HEIGHT_BUCKETS}; Table size {TIME_BUCKETS*VEL_BUCKETS*HEIGHT_BUCKETS} bytes")

costs = {} # maps from (time, vel, height) -> cost-to-go
for t in times:
    for vel in vels:
        for height in heights:
            state = (t, vel, height)
            costs[state] = 0

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
    
    return costs[dynamics(state, action, DT)] + cost

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

def calc_policy(batches = 5):
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
        state = dynamics(state, action, DT)
        print(f"Applying {action}% for {DT:.2f}s. Resulting state is {state}")

policy = calc_policy()

demo(policy)