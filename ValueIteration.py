import time
from tqdm import tqdm
import bisect
from scipy import interpolate
import numpy as np
from Dynamics import nearest


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
        t, v, h = self._transition_func(state, action)

        # Find nearest 4 neighbors to the resulting state
        # Calculate weights for them that sum to 1
        # Generate the weighted average of the resulting cost

        # indices of nearest mesh neighbors
        v1 = bisect.bisect_left(self._vel_keys, v) - 1
        v2 = v1+1
        h1 = bisect.bisect_left(self._height_keys, h) - 1
        h2 = h1+1

        if v1 < 0 :
            v1 = 0
        if v2 > len(self._vel_keys)-1:
            v2 = len(self._vel_keys)-1
        if h1 < 0 :
            h1 = 0
        if h2 > len(self._height_keys)-1:
            h2 = len(self._height_keys)-1

        neighbors = []
        neighbors.append( (t, self._vel_keys[v1], self._height_keys[h1]) )
        neighbors.append( (t, self._vel_keys[v1], self._height_keys[h2]) )
        neighbors.append( (t, self._vel_keys[v2], self._height_keys[h1]) )
        neighbors.append( (t, self._vel_keys[v2], self._height_keys[h2]) )

        # TODO use a distance metric that sums to 1
        weights = [0.25, 0.25, 0.25, 0.25]

        weighted_cost_at_next_state = 0

        for s,w in zip(neighbors, weights):
            t,v,h = s
            assert t in self._time_keys, f"{t}"
            assert v in self._vel_keys
            assert h in self._height_keys
            weighted_cost_at_next_state += w * self._costs[s]

        return weighted_cost_at_next_state + self._loss_func(
            state, action
        )

    def _value_iteration_batch_update(self):
        states_to_visit = list(self._costs.keys())

        # Big brain alert: since every possible action causes the same decrease in t,
        # we can run value iteration on t increasing and it converges in a single batch
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

    def _count_non_crashing_states(self):
        good = 0
        for c in self._costs.values():
            if c != float("inf"):
                good += 1
        return good

    def calc_policy(self, batches=1):
        start_time = time.time()
        policy = None
        costs = None
        for i in range(batches):
            policy, costs = self._value_iteration_batch_update()
            print(f"Batch {i+1}/{batches} complete after {time.time() - start_time:.2f}s. ", end="")
            print(f"Landing from {100*self._count_non_crashing_states()/(len(self._costs.keys())):.2f}% of starting configurations.")
        return policy, costs
