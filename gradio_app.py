"""
Gradio UI — Attention Allocation System
========================================
Run locally:   python gradio_app.py
On HF Spaces:  imported automatically if you add `demo.launch()` at the bottom
               and set CMD to `python gradio_app.py` in Dockerfile.
"""

import os
import json
import time
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO
from PIL import Image

from env.tasks import task_easy, task_medium, task_hard
from env.models import Action
from agents.greedy_agent import greedy_agent
from agents.q_learning_agent import q_learning_agent, Q as ql_Q
from agents.baseline_agent import llm_agent

try:
    from agents.dqn_agent import dqn_agent, QNetwork
    import torch
    _dqn_model = None
    _DQN_PATH  = os.path.join(os.path.dirname(__file__), "dqn_model.pth")
    if os.path.exists(_DQN_PATH):
        # Infer input dim from a dummy env
        _dummy = task_easy()
        _dummy.reset()
        from agents.dqn_agent import featurize as dqn_feat
        _DQN_AVAILABLE = True
    else:
        _DQN_AVAILABLE = False
except Exception:
    _DQN_AVAILABLE = False

try:
    from agents.hybrid_agent import hybrid_agent, load_q_table
    load_q_table(ql_Q)
    _HYBRID_AVAILABLE = True
except Exception:
    _HYBRID_AVAILABLE = False

# ── Task factory ──────────────────────────────────────────────────────────────
TASK_MAP = {
    "Easy   (5 items)":   (task_easy,   4.0),
    "Medium (10 items)":  (task_medium, 7.0),
    "Hard   (15 items)":  (task_hard,  11.0),
}

AGENT_COLORS = {
    "Greedy":     "#4C72B0",
    "Q-Learning": "#DD8452",
    "DQN":        "#55A868",
    "LLM":        "#C44E52",
    "Hybrid":     "#8172B2",
}


# ── Agent runner ──────────────────────────────────────────────────────────────
def _get_dqn_model(state):
    global _dqn_model
    if _dqn_model is None and _DQN_AVAILABLE:
        try:
            sample_feat = __import__(
                "agents.dqn_agent", fromlist=["featurize"]
            ).featurize(state, state.items[0])
            input_dim   = len(sample_feat)
            _dqn_model  = QNetwork(input_dim)
            _dqn_model.load_state_dict(torch.load(_DQN_PATH, map_location="cpu"))
            _dqn_model.eval()
        except Exception as e:
            print(f"[UI] DQN load failed: {e}")
            return None
    return _dqn_model


def _pick_action(agent_name: str, state) -> Action:
    if agent_name == "Greedy":
        return greedy_agent(state)
    elif agent_name == "Q-Learning":
        return q_learning_agent(state)
    elif agent_name == "DQN":
        model = _get_dqn_model(state)
        if model:
            return dqn_agent(state, model)
        return greedy_agent(state)  # fallback
    elif agent_name == "LLM":
        return llm_agent(state)
    elif agent_name == "Hybrid":
        if _HYBRID_AVAILABLE:
            return hybrid_agent(state)
        return greedy_agent(state)
    return greedy_agent(state)


def run_episode(agent_name: str, task_label: str):
    task_fn, norm = TASK_MAP[task_label]
    env   = task_fn()
    state = env.reset()

    rewards     = []
    item_ids    = []
    fatigues    = []
    done        = False
    log_lines   = []

    while not done:
        action              = _pick_action(agent_name, state)
        state, reward, done, _ = env.step(action)
        r = float(reward.value)
        rewards.append(r)
        item_ids.append(action.item_id)
        fatigues.append(state.user.fatigue)
        log_lines.append(
            f"Step {len(rewards):>2} │ item={action.item_id:<3} │ reward={r:+.3f} │ fatigue={state.user.fatigue:.2f}"
        )

    total  = sum(rewards)
    score  = min(1.0, max(0.0, total / norm))
    return rewards, item_ids, fatigues, score, total, log_lines


# ── Chart builders ────────────────────────────────────────────────────────────
def _make_reward_chart(all_results: dict, task_label: str) -> Image.Image:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Agent Comparison — {task_label}", fontsize=13, fontweight="bold")

    # Left: per-step rewards
    ax = axes[0]
    for agent, (rewards, _, _, _, _, _) in all_results.items():
        ax.plot(range(1, len(rewards)+1), rewards,
                marker="o", label=agent, color=AGENT_COLORS.get(agent, "#888"),
                linewidth=2, markersize=5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Per-step rewards")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Right: total reward + score bar chart
    ax2 = axes[1]
    agents = list(all_results.keys())
    scores = [all_results[a][3] for a in agents]  # score is index 3
    colors = [AGENT_COLORS.get(a, "#888") for a in agents]
    bars   = ax2.bar(agents, scores, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Normalised score (0–1)")
    ax2.set_title("Final scores")
    for bar, s in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{s:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _make_fatigue_chart(all_results: dict, task_label: str) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_title(f"Fatigue over time — {task_label}", fontsize=12)
    for agent, (_, _, fatigues, _, _, _) in all_results.items():
        ax.plot(range(1, len(fatigues)+1), fatigues,
                marker="s", label=agent, color=AGENT_COLORS.get(agent, "#888"),
                linewidth=1.8, markersize=4)
    ax.axhline(1.2, color="red", linestyle="--", linewidth=1.2, label="Max fatigue (1.2)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fatigue")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── Main UI function ──────────────────────────────────────────────────────────
def compare_agents(agents_selected: list, task_label: str, progress=gr.Progress()):
    if not agents_selected:
        return None, None, "Select at least one agent.", ""

    all_results = {}
    progress(0, desc="Running episodes...")

    for i, agent in enumerate(agents_selected):
        progress((i+1) / len(agents_selected), desc=f"Running {agent}...")
        try:
            result = run_episode(agent, task_label)
            all_results[agent] = result
        except Exception as e:
            return None, None, f"Error running {agent}: {e}", ""

    reward_chart  = _make_reward_chart(all_results, task_label)
    fatigue_chart = _make_fatigue_chart(all_results, task_label)

    # Summary table (markdown)
    rows = ["| Agent | Steps | Total Reward | Score |", "|---|---|---|---|"]
    for agent, (rewards, _, _, score, total, _) in all_results.items():
        rows.append(f"| {agent} | {len(rewards)} | {total:.3f} | **{score:.3f}** |")
    summary_md = "\n".join(rows)

    # Detailed step log
    log_parts = []
    for agent, (_, _, _, _, _, lines) in all_results.items():
        log_parts.append(f"=== {agent} ===")
        log_parts.extend(lines)
        log_parts.append("")
    step_log = "\n".join(log_parts)

    return reward_chart, fatigue_chart, summary_md, step_log


def run_single(agent_name: str, task_label: str, progress=gr.Progress()):
    progress(0.2, desc=f"Running {agent_name}...")
    try:
        rewards, item_ids, fatigues, score, total, lines = run_episode(agent_name, task_label)
    except Exception as e:
        return None, f"Error: {e}", ""

    # Single episode chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    color = AGENT_COLORS.get(agent_name, "#888")

    axes[0].bar(range(1, len(rewards)+1), rewards, color=color, alpha=0.8, edgecolor="white")
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_title(f"{agent_name} — per-step rewards")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Reward")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(range(1, len(fatigues)+1), fatigues, color=color,
                 marker="o", linewidth=2)
    axes[1].axhline(1.2, color="red", linestyle="--", linewidth=1.2, label="Max (1.2)")
    axes[1].set_title("Fatigue progression")
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("Fatigue")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"{agent_name} on {task_label}  │  score={score:.3f}  │  total reward={total:.3f}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    chart = Image.open(buf).copy()

    summary = f"**Score:** {score:.3f}   |   **Total reward:** {total:.3f}   |   **Steps:** {len(rewards)}"
    log     = "\n".join(lines)
    progress(1.0)
    return chart, summary, log


# ── Gradio layout ──────────────────────────────────────────────────────────────
AVAILABLE_AGENTS = ["Greedy", "Q-Learning", "LLM"]
if _DQN_AVAILABLE:
    AVAILABLE_AGENTS.append("DQN")
if _HYBRID_AVAILABLE:
    AVAILABLE_AGENTS.append("Hybrid")

with gr.Blocks(title="Attention Allocation System", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Attention Allocation System
    **Content recommendation RL environment** — compare how Greedy, Q-Learning, DQN, LLM, and Hybrid agents
    manage user engagement, diversity, and fatigue across easy / medium / hard tasks.
    """)

    with gr.Tabs():

        # ── Tab 1: Compare all agents ──────────────────────────────────────
        with gr.TabItem("Compare agents"):
            with gr.Row():
                agents_cb = gr.CheckboxGroup(
                    choices=AVAILABLE_AGENTS,
                    value=["Greedy", "Q-Learning", "LLM"],
                    label="Agents to compare"
                )
                task_dd = gr.Dropdown(
                    choices=list(TASK_MAP.keys()),
                    value="Easy   (5 items)",
                    label="Task difficulty"
                )
            run_btn = gr.Button("Run comparison", variant="primary")

            with gr.Row():
                reward_img  = gr.Image(label="Reward chart", type="pil")
                fatigue_img = gr.Image(label="Fatigue chart", type="pil")

            summary_out = gr.Markdown(label="Summary")
            log_out     = gr.Textbox(label="Step-by-step log", lines=12, max_lines=20)

            run_btn.click(
                fn=compare_agents,
                inputs=[agents_cb, task_dd],
                outputs=[reward_img, fatigue_img, summary_out, log_out]
            )

        # ── Tab 2: Single episode deep-dive ───────────────────────────────
        with gr.TabItem("Single episode"):
            with gr.Row():
                single_agent = gr.Dropdown(
                    choices=AVAILABLE_AGENTS,
                    value="LLM",
                    label="Agent"
                )
                single_task = gr.Dropdown(
                    choices=list(TASK_MAP.keys()),
                    value="Hard   (15 items)",
                    label="Task difficulty"
                )
            single_btn = gr.Button("Run episode", variant="primary")

            single_chart   = gr.Image(label="Episode chart", type="pil")
            single_summary = gr.Markdown()
            single_log     = gr.Textbox(label="Step log", lines=10, max_lines=15)

            single_btn.click(
                fn=run_single,
                inputs=[single_agent, single_task],
                outputs=[single_chart, single_summary, single_log]
            )

        # ── Tab 3: About ───────────────────────────────────────────────────
        with gr.TabItem("About"):
            gr.Markdown("""
            ## How the environment works

            Each episode simulates a content feed session:
            - The **user** has an `interest_vector` (3D topic preferences), a `fatigue` level, and a session timer
            - **Items** each have a `topic_vector`, `quality`, `novelty`, and `length`
            - The agent picks one item per step; session ends when all items are consumed or fatigue > 1.2

            ### Reward formula
            ```
            reward = engagement + 0.6 × diversity + 0.3 × quality − 0.5 × fatigue
            ```

            ### Agents
            | Agent | Strategy |
            |---|---|
            | Greedy | Picks highest engagement + quality, ignores diversity |
            | Q-Learning | Tabular RL — learns state→action values from reward signals |
            | DQN | Neural Q-network — generalises across continuous state space |
            | LLM | Uses an LLM to reason about each item (requires HF_TOKEN) |
            | Hybrid | LLM shortlists top-3 candidates, Q-Learning picks the best one |

            ### Tasks
            | Task | Items | Norm score |
            |---|---|---|
            | Easy | 5 | 4.0 |
            | Medium | 10 | 7.0 |
            | Hard | 15 | 11.0 |
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)