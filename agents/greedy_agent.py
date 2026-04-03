import numpy as np
from env.models import Action

def greedy_agent(state):
    user = state.user

    best_score = -1
    best_id = None

    for item in state.items:
        score = (
            np.dot(user.interest_vector, item.topic_vector)
            + 0.3 * item.quality
        )

        if score > best_score:
            best_score = score
            best_id = item.id

    return Action(item_id=best_id)