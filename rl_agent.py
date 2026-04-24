# rl_agent.py
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, n_actions=3, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def _state_to_key(self, state):
        return tuple(state.round(2))  # discretize a bit

    def act(self, state, greedy=False):
        key = self._state_to_key(state)
        if (not greedy) and (np.random.rand() < self.epsilon):
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[key]))

    def update(self, state, action, reward, next_state, done):
        key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)
        best_next = 0.0 if done else np.max(self.Q[next_key])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[key][action]
        self.Q[key][action] += self.alpha * td_error
