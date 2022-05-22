import time


class ValueIterator:
    def __init__(self, states, actions, transition_func, loss_func):
        self._costs = {}  # maps from (time, vel, height) -> cost-to-go
        for s in states:
            self._costs[s] = 0
        self._actions = actions
        self._transition_func = transition_func
        self._loss_func = loss_func

    def _cost_func(self, state, action):
        return self._costs[self._transition_func(state, action)] + self._loss_func(
            state, action
        )

    def _value_iteration_batch_update(self):
        for state in self._costs.keys():
            new_costs = []
            for action in self._actions:
                new_costs.append(self._cost_func(state, action))
            new_cost = min(new_costs)
            self._costs[state] = new_cost

    def _extract_policy(self):
        policy = {}
        for state in self._costs.keys():
            min_cost = float("inf")
            best_action = None
            for action in self._actions:
                c = self._cost_func(state, action)
                if c < min_cost:
                    min_cost = c
                    best_action = action
            policy[state] = best_action
        return policy

    def _count_non_crashing_states(self):
        good = 0
        for c in self._costs.values():
            if c != float("inf"):
                good += 1
        return good

    def calc_policy(self, batches=5):
        start_time = time.time()
        for i in range(batches):
            self._value_iteration_batch_update()
            print(f"Batch {i} complete after {time.time() - start_time:.2f}s. ", end="")
            print(
                f"Landing from {100*self._count_non_crashing_states()/(len(self._costs.keys())):.2f}% of starting configurations"
            )
        return self._extract_policy()
