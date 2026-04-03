import matplotlib.pyplot as plt
import numpy as np
from agents.dqn_agent import dqn_agent, QNetwork
from env.tasks import task_easy, task_medium, task_hard
from agents.greedy_agent import greedy_agent
from agents.q_learning_agent import q_learning_agent

def run_episode(env, agent):
    state = env.reset()
    total_reward = 0

    done = False
    while not done:
        action = agent(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward.value

    return total_reward

def plot_results():
    # Initialize DQN model
    input_dim = 3 + 1 + 1 + 3 + 1 + 1  # interest_vector(3) + fatigue + session_time + topic_vector(3) + quality + length
    dqn_model = QNetwork(input_dim)

    tasks = [
        ("Easy", task_easy()),
        ("Medium", task_medium()),
        ("Hard", task_hard()),
    ]

    agents = [
        ("Greedy", greedy_agent),
        ("Q-Learning", q_learning_agent),
        ("DQN", lambda state: dqn_agent(state, dqn_model)),
    ]

    results = {}
    for task_name, env in tasks:
        results[task_name] = {}
        for agent_name, agent in agents:
            reward = run_episode(env, agent)
            results[task_name][agent_name] = reward

    # Plotting
    task_names = list(results.keys())
    agent_names = list(results[task_names[0]].keys())
    num_tasks = len(task_names)
    num_agents = len(agent_names)

    x = np.arange(num_tasks)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, agent_name in enumerate(agent_names):
        rewards = [results[task][agent_name] for task in task_names]
        ax.bar(x + i * width, rewards, width, label=agent_name)

    ax.set_xlabel('Task Difficulty')
    ax.set_ylabel('Total Reward')
    ax.set_title('Agent Performance Across Tasks')
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_names)
    ax.legend()

    plt.tight_layout()
    plt.savefig('agent_performance.png')
    plt.show()

if __name__ == "__main__":
    plot_results()