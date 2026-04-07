import random

import numpy as np
from env.models import Action

class QLearningAgent:

    def __init__(self):
        self.Q = {}  # Q-table
        self.alpha = 0.1

    def featurize(self, state):
        user = state.user
        return tuple(np.round(user.interest_vector + [user.fatigue], 2))

    #  Get best single action
    def act(self, state):
        values = []

        for item in state.items:
            key = (self.featurize(state), item.id)
            values.append(self.Q.get(key, 0))

        best_idx = np.argmax(values)
        return Action(item_id=state.items[best_idx].id)

    #  Get top K recommendations
    def get_top_k_actions(self, state, k=10):
        scores = {}

        for item in state.items:
            key = (self.featurize(state), item.id)
            scores[item.id] = self.Q.get(key, 0)

        # EXPLORATION (30% random)
        if random.random() < 0.3:
            return random.sample(list(scores.keys()), k)

        # EXPLOITATION
        sorted_items = sorted(scores, key=scores.get, reverse=True)

        return sorted_items[:k]

    # Update Q-values
    def update(self, state, action, reward):
        state_key = self.featurize(state)
        key = (state_key, action)

        if key not in self.Q:
            self.Q[key] = 0

        self.Q[key] += self.alpha * reward

q_learning_agent = QLearningAgent()