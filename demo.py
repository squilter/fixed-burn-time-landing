#!/usr/bin/env python3

import time
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt

from Dynamics import *
from ValueIteration import ValueIterator
from Visualizer import plot_policy, sim

EPS = 0.01

# Returns a small number when something good happens and a big number when something bad happens
def loss(state, action):
    t, vel, height = state
    assert state in valid_states
    assert action in actions

    # Penalize crashing
    if t < -EPS:
        return 100*abs(vel) + 200*abs(height)

    loss = 1
    # don't hit the ground before it's time
    if t > 1.0 and height <= 0.0:
        return 999999
    
    # penalize relying on 100% throttle
    loss += 0.01*(action-80)**2

    return loss

if __name__ == "__main__":
    policy = None
    costs = None
    if not os.path.exists("policy.p") or "y" == input("Recompute policy? [y/n]: ".lower()):
        print(f"Generating table with size {TIME_BUCKETS*VEL_BUCKETS*HEIGHT_BUCKETS}:")
        print(f"Time Options:\n{times};")
        print(f"Speed Options:\n{vels}")
        print(f"Height Options:\n{heights}")
        valueIterator = ValueIterator(valid_states, actions, dynamics, loss)
        policy, costs = valueIterator.calc_policy()
        with open("policy.p", "wb") as f:
            pickle.dump((policy, costs), f)
    else:
        with open("policy.p", "rb") as f:
            policy, costs = pickle.load(f)

    plot_policy(policy)
    plot_policy(costs, threshold=500)

    print("### DEMO: Start burn at 13m with 12m/s speed ###")
    starting_state = nearest_state(3, 12, 13)
    print(f"Starting cost: {costs[starting_state]}")
    sim(policy, starting_state)

    print("### DEMO: Start burn at 11m with 9/s speed ###")
    starting_state = nearest_state(3, 9, 11)
    print(f"Starting cost: {costs[starting_state]}")
    sim(policy, starting_state)

    print("### DEMO: Start burn at 2m with 15m/s speed ###")
    starting_state = nearest_state(3.0, 15, 2)
    print(f"Starting cost: {costs[starting_state]}")
    sim(policy, starting_state)
