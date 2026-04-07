import numpy as np
from env.models import Action

Q = {}

def featurize(state):
    user = state.user
    return tuple(np.round(user.interest_vector + [user.fatigue], 2))

def q_learning_agent(state):
    values = []

    for item in state.items:
        key = (featurize(state), item.id)
        values.append(Q.get(key, 0))

    best_idx = np.argmax(values)
    return Action(item_id=state.items[best_idx].id)

def get_top_k_actions(self, state, k=10):
    q_values = self.get_q_values(state)

    # sort by Q value
    sorted_actions = sorted(q_values, key=q_values.get, reverse=True)

    return sorted_actions[:k]