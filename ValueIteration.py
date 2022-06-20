import time
from tqdm import tqdm
import bisect
from scipy import interpolate
import numpy as np
from Dynamics import nearest

# Evaluates a cost function or a control policy using bilinear interpolation
def weighted_evaluate(policy, time_keys, vel_keys, height_keys, state):
    t,v,h=state
    # Find nearest 4 neighbors to the resulting state
    # Calculate weights for them that sum to 1
    # Generate the weighted average of the resulting cost

    # indices of nearest mesh neighbors
    v1 = bisect.bisect_left(vel_keys, v) - 1
    v2 = v1+1
    h1 = bisect.bisect_left(height_keys, h) - 1
    h2 = h1+1

    if v1 < 0 :
        v1 = 0
    if v2 > len(vel_keys)-1:
        v2 = len(vel_keys)-1
    if h1 < 0 :
        h1 = 0
    if h2 > len(height_keys)-1:
        h2 = len(height_keys)-1

    neighbors = []
    neighbors.append( (t, vel_keys[v1], height_keys[h1]) )
    neighbors.append( (t, vel_keys[v1], height_keys[h2]) )
    neighbors.append( (t, vel_keys[v2], height_keys[h1]) )
    neighbors.append( (t, vel_keys[v2], height_keys[h2]) )

    right_vel_weight = 1
    right_height_weight = 1
    if abs(vel_keys[v2] - vel_keys[v1]) > 0.00001:
        right_vel_weight = (v - vel_keys[v1]) / (vel_keys[v2] - vel_keys[v1])
    if abs(height_keys[h2] - height_keys[h1]) > 0.00001:
        right_height_weight = (h - height_keys[h1]) / (height_keys[h2] - height_keys[h1])

    weights = [1,1,1,1]
    weights[0] = (1-right_vel_weight) * (1-right_height_weight)
    weights[1] = (1-right_vel_weight) * right_height_weight
    weights[2] = right_vel_weight * (1-right_height_weight)
    weights[3] = right_vel_weight * right_height_weight
    assert abs(np.sum(weights) - 1) < 0.000001

    weighted_evaluate = 0
    for s,w in zip(neighbors, weights):
        t,v,h = s
        assert t in time_keys, f"{t}"
        assert v in vel_keys
        assert h in height_keys
        assert not np.isnan(w)
        assert not np.isnan(policy[s])
        weighted_evaluate += w * policy[s]
    
    return weighted_evaluate

class ValueIterator:
    def __init__(self, states, actions, transition_func, loss_func):
        self._costs = {}  # maps from (time, vel, height) -> cost-to-go
        for s in states:
            self._costs[s] = 0
        self._actions = actions
        self._transition_func = transition_func
        self._loss_func = loss_func
        self._time_keys, self._vel_keys, self._height_keys = zip(*states)
        self._time_keys = sorted(set(self._time_keys))
        self._vel_keys = sorted(set(self._vel_keys))
        self._height_keys = sorted(set(self._height_keys))

    def _cost_func(self, state, action):
        new_state = self._transition_func(state, action)
        weighted_cost_at_next_state = weighted_evaluate(self._costs, self._time_keys, self._vel_keys, self._height_keys, new_state)

        return weighted_cost_at_next_state + self._loss_func(
            state, action
        )

    def _value_iteration_batch_update(self, sort_index):
        states_to_visit = list(self._costs.keys())

        if sort_index is not None:
            states_to_visit.sort(key=lambda x: x[0])

        policy = {}
        for state in tqdm(states_to_visit, desc="Batch Update"):
            costs_and_actions = []
            for action in self._actions:
                costs_and_actions.append((self._cost_func(state, action), action))
            costs_and_actions.sort(key=lambda x: x[0])
            new_cost = costs_and_actions[0][0]
            policy[state] = costs_and_actions[0][1]
            self._costs[state] = new_cost
        return policy, self._costs

    def calc_policy(self, sort_index = None, batches=10):
        start_time = time.time()
        policy = None
        costs = None
        for i in range(batches):
            policy, costs = self._value_iteration_batch_update(sort_index)
            print(f"Batch {i+1}/{batches} complete after {time.time() - start_time:.2f}s. ", end="")
        return policy, costs

