#!/usr/bin/env python3

import time
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt

from Dynamics import *
from ValueIteration import ValueIterator
from Visualizer import plot_policy

EPS = 0.01

# Returns a small number when something good happens and a big number when something bad happens
def loss(state, action):
    t, vel, height = state
    assert state in valid_states
    assert action in actions

    # No penalty at time 0
    if t < EPS:
        return 0
    
    loss = 1

    # On the last step before 0, penalize crashing
    if nearest(times, t) == times[1]:
        (last_t, last_vel, last_height) = dynamics(state,action)
        loss += 100*last_height**2
        loss += 100*last_vel**2
    
    # penalize relying on 100% throttle
    loss += 0.01*(action-80)**2

    return loss


def demo(policy, state):
    t, vel, height = state
    while t > EPS and height > EPS:
        action = policy[state]
        if action is None:
            print("Infeasible!")
            return
        print(f"Now at height {state[2]:.2f} with speed {state[1]:.2f} with {state[0]:.2f} seconds burn remaining. Applying {action}% for {DT:.2f}s.")
        state = dynamics(state, action)
        t, vel, height = state
    print(f"Now at height {state[2]:.2f} with speed {state[1]:.2f} with {state[0]:.2f} seconds burn remaining.")


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

    print("### DEMO: Start burn at 13m with 12m/s speed ###")
    starting_state = state = nearest_state(3, 12, 13)
    demo(policy, starting_state)

    print("### DEMO: Start burn at 11m with 9/s speed ###")
    starting_state = state = nearest_state(3, 9, 11)
    demo(policy, starting_state)

    print("### DEMO: Start burn at 2m with 15m/s speed ###")
    starting_state = state = nearest_state(3.0, 15, 2)
    demo(policy, starting_state)

    plot_policy(policy)

    plot_policy(costs, threshold=250)

    # consider plotting the policy. at least a timeslice of it.

    # I could try to turn the policy table into a polynomial
    # arrange states (with valid control laws) into a matrix A with 3 columns
    # Arrange the corresponding controls into a vector with the same length as A
    # Now we could easily solve for a linear policy by solving Ax=b with least squares
    # This would result in a linear control policy

    # To get higher order fit, I can put them into a matrix A where rows are 1,x,y,z,x^2,y^2,z^2,x^3,y^3,... (the 1 at the beginning allows a +c term)
    # least regressions on this gives me more coefficients show the linear example from above. Note that, using this function, we don’t need to turn y into a column vector.
    # If I think an exponential function or somehting might fit better, I could use scipy.optimize.curve_fit to fit to arbitrary functions (returns coefficients)
