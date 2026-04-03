import random
import torch
import torch.nn as nn
import numpy as np
from collections import deque

from env.environment import AttentionEnv
from agents.dqn_agent import QNetwork, featurize

# -------- Hyperparameters --------
EPISODES = 2000
BATCH_SIZE = 64
GAMMA = 0.95
LR = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE = 50
BUFFER_SIZE = 5000

# -------- Replay Buffer --------
buffer = deque(maxlen=BUFFER_SIZE)

def sample_batch():
    batch = random.sample(buffer, BATCH_SIZE)

    states, targets = [], []

    for state, item, reward, next_state, done in batch:
        x = torch.FloatTensor(featurize(state, item))

        if done:
            target = reward
        else:
            next_qs = []
            for next_item in next_state.items:
                x_next = torch.FloatTensor(featurize(next_state, next_item))
                next_qs.append(target_model(x_next).item())

            target = reward + GAMMA * max(next_qs)

        states.append(x)
        targets.append([target])

    return torch.stack(states), torch.FloatTensor(targets)


# -------- Init --------
env = AttentionEnv()

sample_state = env.reset()
sample_item = sample_state.items[0]
input_dim = len(featurize(sample_state, sample_item))

model = QNetwork(input_dim)
target_model = QNetwork(input_dim)
target_model.load_state_dict(model.state_dict())

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

epsilon = EPSILON_START

rewards_log = []

# -------- Training Loop --------
for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy
        if random.random() < epsilon:
            item = random.choice(state.items)
        else:
            qs = []
            for i in state.items:
                x = torch.FloatTensor(featurize(state, i))
                qs.append(model(x).item())
            item = state.items[np.argmax(qs)]

        next_state, reward, done, _ = env.step(type("A", (), {"item_id": item.id})())

        buffer.append((state, item, reward.value, next_state, done))

        state = next_state
        total_reward += reward.value

        # Train
        if len(buffer) > BATCH_SIZE:
            states, targets = sample_batch()

            preds = model(states)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    rewards_log.append(total_reward)

    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_model.load_state_dict(model.state_dict())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

# Save model
torch.save(model.state_dict(), "dqn_model.pth")

# Save rewards
np.save("rewards.npy", rewards_log)