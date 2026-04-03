from env.environment import AttentionEnv
from agents.greedy_agent import greedy_agent
from agents.q_learning_agent import q_learning_agent
from agents.dqn_agent import dqn_agent, QNetwork

# Initialize environment
env = AttentionEnv()

# For DQN agent
input_dim = 3 + 1 + 1 + 3 + 1 + 1  # Based on featurize: interest_vector(3) + fatigue + session_time + topic_vector(3) + quality + length
dqn_model = QNetwork(input_dim)

# Run episode with Q-Learning Agent
print("Running episode with Q-Learning Agent...")
state = env.reset()
done = False
total_q = 0
while not done:
    action = q_learning_agent(state)
    state, reward, done, _ = env.step(action)
    total_q += reward.value
print(f"Total Reward Q-Learning: {total_q}")

# Run episode with DQN Agent
print("Running episode with DQN Agent...")
state = env.reset()
done = False
total_dqn = 0
while not done:
    action = dqn_agent(state, dqn_model)
    state, reward, done, _ = env.step(action)
    total_dqn += reward.value
print(f"Total Reward DQN: {total_dqn}")