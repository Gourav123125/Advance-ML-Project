# agent/q_agent.py

import random
from config import ALPHA, GAMMA, EPSILON, EPSILON_DECAY, MIN_EPSILON


class QAgent:
    def __init__(self):
        self.q_table = {}
        self.epsilon = EPSILON

    def _bucket(self, x):
        if x <= 5:
            return 0
        elif x <= 10:
            return 1
        else:
            return 2

    def _state_to_tuple(self, state):
        return (
            self._bucket(state["N"]),
            self._bucket(state["S"]),
            self._bucket(state["E"]),
            self._bucket(state["W"])
        )

    def _get_q(self, state_t, action):
        return self.q_table.get((state_t, action), 0.0)

    def choose_action(self, state):
        state_t = self._state_to_tuple(state)

        if random.random() < self.epsilon:
            return random.choice([0, 1])

        q_values = [self._get_q(state_t, a) for a in [0, 1]]
        return q_values.index(max(q_values))

    def update(self, state, action, reward, next_state):
        state_t = self._state_to_tuple(state)
        next_state_t = self._state_to_tuple(next_state)

        old_q = self._get_q(state_t, action)
        next_max = max([self._get_q(next_state_t, a) for a in [0, 1]])

        new_q = old_q + ALPHA * (reward + GAMMA * next_max - old_q)

        self.q_table[(state_t, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * EPSILON_DECAY, MIN_EPSILON)