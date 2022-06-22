#!/usr/bin/env python3

import time
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt

from Dynamics import *
from ValueIteration import ValueIterator
from Visualizer import plot_policy, sim
from autocoder import write_policy

EPS = 0.01

# Returns a small number when something good happens and a big number when something bad happens
def loss(state, action):
    t, vel, height = state
    assert state in valid_states
    assert action in actions

    if t < -EPS:
        return 0

    # Penalize crashing
    if t < EPS:
        return (height*10)**2 + (vel*10)**2 + abs(action-80)/1000

    loss = 1
    # don't hit the ground before it's time
    if t > 1.0 and height <= 0.5:
        loss += t**2 * (10*height-0.5)**2
    
    # penalize relying on 100% throttle. Prefer 80%.
    loss += 0.1*(action-80)**2

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
        # sorting on time=descending requires only 1 batch. This is called backward induction.
        policy, costs = valueIterator.calc_policy(sort_index=0, batches=1)
        with open("policy.p", "wb") as f:
            pickle.dump((policy, costs), f)
    else:
        with open("policy.p", "rb") as f:
            policy, costs = pickle.load(f)

    plot_policy(policy, costs)
    # plot_policy(costs, threshold=5, result_label='Feasibility')
    sim(policy, 25)
    sim(policy, 18)
    sim(policy, 14)
    sim(policy, 35)
    write_policy(policy)