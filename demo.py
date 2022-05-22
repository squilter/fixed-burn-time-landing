#!/usr/bin/env python3

import time

import numpy as np
import matplotlib.pyplot as plt
from Dynamics import *
from ValueIteration import ValueIterator

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

def loss(state, action):
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
    
    return cost

def demo(policy):
    state = nearest_state(3, 12, 13)
    print(f"Starting at {state}")
    while not (is_landed(state) or is_crashed(state)):
        action = policy[state]
        state = dynamics(state, action)
        print(f"Applying {action}% for {DT:.2f}s. Resulting state is {state}")

valueIterator = ValueIterator(costs.keys(), actions, dynamics, loss)
policy = valueIterator.calc_policy()

demo(policy)