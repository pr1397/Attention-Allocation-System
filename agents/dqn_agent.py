import torch
import torch.nn as nn
import numpy as np
from env.models import Action

# -------- Neural Network --------
class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# -------- Feature Function --------
def featurize(state, item):
    user = state.user

    return np.array(
        list(user.interest_vector)
        + [user.fatigue, user.session_time]
        + list(item.topic_vector)
        + [item.quality, item.length]
    )


# -------- Inference Agent --------
def dqn_agent(state, model):
    values = []

    for item in state.items:
        x = torch.FloatTensor(featurize(state, item))
        values.append(model(x).item())

    best_idx = np.argmax(values)
    return Action(item_id=state.items[best_idx].id)