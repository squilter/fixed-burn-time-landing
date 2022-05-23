#!/usr/bin/env python3

import time
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt

from Dynamics import *
from ValueIteration import ValueIterator

EPS = 0.01


def is_crashed(state):
    t, vel, height = state
    # hitting the ground hard at any point is a crash
    if height <= EPS and vel > 2 + EPS:
        return True
    # Not being landed when the motor burns out is a crash
    if t < EPS and not is_landed(state):
        return True
    return False


def is_landed(state):
    t, vel, height = state
    return t < EPS and vel < 0.6 + EPS and height < 0.6 + EPS


# Returns a small number when something good happens and a big number when something bad happens
def loss(state, action):
    assert state in valid_states
    assert action in actions
    if is_landed(state):
        return 0
    if is_crashed(state):
        return float("inf")

    # Let's have a slight preference for keeping throttle around 80%
    loss = 1
    loss += 0.01 * abs(action - 80)

    return loss


def demo(policy, state):
    while not (is_landed(state) or is_crashed(state)):
        action = policy[state]
        if action is None:
            print("Infeasible!")
            return
        print(f"Now at height {state[2]:.2f} with speed {state[1]:.2f} with {state[0]:.2f} seconds burn remaining. Applying {action}% for {DT:.2f}s.")
        state = dynamics(state, action)
    print(f"Now at height {state[2]:.2f} with speed {state[1]:.2f} with {state[0]:.2f} seconds burn remaining. {'Landed!' if is_landed(state) else 'Crashed!'}")


if __name__ == "__main__":
    policy = None
    if not os.path.exists("policy.p") or "y" == input("Recompute policy? [y/n]: ".lower()):
        print(f"Generating table with size {TIME_BUCKETS*VEL_BUCKETS*HEIGHT_BUCKETS}:")
        print(f"Time Options:\n{times};")
        print(f"Speed Options:\n{vels}")
        print(f"Height Options:\n{heights}")
        valueIterator = ValueIterator(valid_states, actions, dynamics, loss)
        policy = valueIterator.calc_policy()
        with open("policy.p", "wb") as f:
            pickle.dump(policy, f)
    else:
        with open("policy.p", "rb") as f:
            policy = pickle.load(f)

    print("### DEMO: Start burn at 13m with 12m/s speed ###")
    starting_state = state = nearest_state(3, 12, 13)
    demo(policy, starting_state)

    print("### DEMO: Start burn at 11m with 9/s speed ###")
    starting_state = state = nearest_state(3, 9, 11)
    demo(policy, starting_state)

    print("### DEMO: Start burn at 2m with 15m/s speed ###")
    starting_state = state = nearest_state(3.0, 15, 2)
    demo(policy, starting_state)
