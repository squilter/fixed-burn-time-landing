import time
from tqdm import tqdm


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
