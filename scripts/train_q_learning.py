from env.environment import AttentionEnv
from agents.q_learning_agent import Q, featurize
import random
import numpy as np

def choose_action(state, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(state.items).id

    values = []
    for item in state.items:
        key = (featurize(state), item.id)
        values.append(Q.get(key, 0))

    return state.items[np.argmax(values)].id

def train():
    env = AttentionEnv()

    for episode in range(2000):
        state = env.reset()
        done = False

        while not done:
            action_id = choose_action(state)
            next_state, reward, done, _ = env.step(type("A", (), {"item_id": action_id})())

            key = (featurize(state), action_id)

            next_values = [
                Q.get((featurize(next_state), item.id), 0)
                for item in next_state.items
            ]

            max_next = max(next_values) if next_values else 0

            Q[key] = Q.get(key, 0) + 0.1 * (
                reward.value + 0.9 * max_next - Q.get(key, 0)
            )

            state = next_state

if __name__ == "__main__":
    train()