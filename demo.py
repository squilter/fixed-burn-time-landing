#!/usr/bin/env python3

import time

import numpy as np
import matplotlib.pyplot as plt
from Dynamics import *
from ValueIteration import ValueIterator

EPS = 0.00001

valid_states = set()  # (time, vel, height)
for t in times:
    for vel in vels:
        for height in heights:
            state = (t, vel, height)
            valid_states.add(state)

def is_crashed(state):
    t, vel, height = state
    return height <= 0.00001 and vel >= 2

def is_landed(state):
    t, vel, height = state
    return t < EPS and vel < 1+EPS and height < EPS

def loss(state, action):
    assert state in valid_states
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


def demo(policy, state):
    print("### DEMO ###")
    while not (is_landed(state) or is_crashed(state)):
        action = policy[state]
        state = dynamics(state, action)
        print(f"Currently at height {state[2]} with speed {state[1]} with {state[0]} seconds burn remaining. Applying {action}% for {DT:.2f}s.")
    print(f"Ended with height {state[2]}, speed {state[1]} and {state[0]} seconds burn remaining.")

if __name__ == "__main__":
    print(f"Generating table with size {TIME_BUCKETS*VEL_BUCKETS*HEIGHT_BUCKETS}:")
    print(f"Time {TIME_BUCKETS}; Vel {VEL_BUCKETS}; Pos {HEIGHT_BUCKETS}")

    valueIterator = ValueIterator(valid_states, actions, dynamics, loss)
    policy = valueIterator.calc_policy(batches = 10)

    starting_state = state = nearest_state(3, 12, 13)
    demo(policy, starting_state)
