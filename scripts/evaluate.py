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

def evaluate():
    # Initialize DQN model
    input_dim = 3 + 1 + 1 + 3 + 1 + 1  # interest_vector(3) + fatigue + session_time + topic_vector(3) + quality + length
    dqn_model = QNetwork(input_dim)

    tasks = [
        ("Easy", task_easy(), 4.0),
        ("Medium", task_medium(), 7.0),
        ("Hard", task_hard(), 11.0),
    ]

    for name, env, norm in tasks:
        reward = run_episode(env, greedy_agent)
        reward_q_learning = run_episode(env, q_learning_agent)
        reward_dqn = run_episode(env, lambda state: dqn_agent(state, dqn_model))
        score = min(1.0, reward / norm)
        score_q_learning = min(1.0, reward_q_learning / norm)
        score_dqn = min(1.0, reward_dqn / norm)

        print(f"{name}: reward={reward:.2f}, score={score:.2f}")
        print(f"{name}: reward_q_learning={reward_q_learning:.2f}, score_q_learning={score_q_learning:.2f}")
        print(f"{name}: reward_dqn={reward_dqn:.2f}, score_dqn={score_dqn:.2f}")
        print(f"____________________________________________________________")



if __name__ == "__main__":
    evaluate()