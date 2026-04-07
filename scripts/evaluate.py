import torch
from agents.dqn_agent import dqn_agent, QNetwork
from agents.greedy_agent import greedy_agent
from agents.q_learning_agent import q_learning_agent
from agents.baseline_agent import llm_agent         
from env.tasks import task_easy, task_medium, task_hard
from agents.hybrid_agent import hybrid_agent, load_q_table
from agents.q_learning_agent import Q
load_q_table(Q)  



def run_episode(env, agent) -> float:
    state = env.reset()
    total_reward = 0.0
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

    # Load trained weights if available
    try:
        dqn_model.load_state_dict(torch.load("dqn_model.pth", map_location="cpu"))
        dqn_model.eval()
        print("[INFO] Loaded dqn_model.pth")
    except FileNotFoundError:
        print("[WARN] dqn_model.pth not found — DQN using random weights")

    tasks = [
        ("Easy",   task_easy,   4.0),
        ("Medium", task_medium, 7.0),
        ("Hard",   task_hard,  11.0),
    ]

    agents = [
        ("Greedy",      lambda state: greedy_agent(state)),
        ("LLM",         lambda state: llm_agent(state)),
        ("Q-Learning",  lambda state: q_learning_agent(state)),
        ("DQN",         lambda state: dqn_agent(state, dqn_model)),
        ("Hybrid",      lambda state: hybrid_agent(state, dqn_model, Q)),
    ]

    for task_name, task_fn, norm in tasks:
        print(f"\n{'='*56}")
        print(f"Task: {task_name}")
        print(f"{'='*56}")

        for agent_name, agent_fn in agents:
            # IMPORTANT: fresh env for every agent so state isn't shared
            env = task_fn()
            reward = run_episode(env, agent_fn)
            score = min(1.0, reward / norm)
            print(f"  {agent_name:<14} reward={reward:6.2f}  score={score:.3f}")


if __name__ == "__main__":
    evaluate()