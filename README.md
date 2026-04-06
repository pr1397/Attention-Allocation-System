# Attention Allocation System

## Overview

This environment simulates a real-world content recommendation system where an LLM agent selects items to maximise user engagement while managing fatigue and diversity. It models real-world platforms like YouTube, Instagram, or news feed personalisation.

Built with the [OpenEnv](https://github.com/raun/openenv-course) framework for the Meta × PyTorch Hackathon (Round 1).

## Environment Description

An agent observes a user's interest profile and a pool of available content items, then recommends one item per step. The session ends when all items are consumed or user fatigue exceeds the threshold.

### Observation Space

| Field                    | Type               | Description                              |
|--------------------------|--------------------|------------------------------------------|
| `user.interest_vector`   | List[float] (len 3)| User's topic interests (sums to 1)       |
| `user.fatigue`           | float              | Accumulated fatigue (0.0 → 1.2 max)      |
| `user.session_time`      | int                | Number of steps taken so far             |
| `items[].topic_vector`   | List[float] (len 3)| What the item is about                   |
| `items[].quality`        | float              | Content quality score (0–1)              |
| `items[].novelty`        | float              | How new/fresh the item is (0–1)          |
| `items[].length`         | int                | Item length (1–5); affects fatigue       |

### Action Space

Pick one `item_id` from the currently available items.

### Reward Function

```
reward = engagement + 0.6 * diversity + 0.3 * quality - 0.5 * fatigue

engagement = dot(user.interest_vector, item.topic_vector)
diversity  = 1 - dot(last_item.topic_vector, item.topic_vector)
```

Fatigue increases by `0.1 * item.length` each step. Session terminates if `fatigue > 1.2`.

## Tasks

| Task   | Items | Norm Score | Difficulty |
|--------|-------|------------|------------|
| Easy   | 5     | 4.0        | Low        |
| Medium | 10    | 7.0        | Medium     |
| Hard   | 15    | 11.0       | High       |

## Agents

Three baseline agents are included for comparison:

- **Greedy** — picks the item with highest interest alignment + quality score
- **Q-Learning** — tabular RL agent trained over 2000 episodes
- **DQN** — neural network Q-function trained with experience replay

The inference script uses an **LLM** (via OpenAI-compatible API) as the agent.

## Setup

```bash
# Clone the repo
git clone https://github.com/pr1397/Attention-Allocation-System
cd Attention-Allocation-System

# Install dependencies
pip install -r requirements.txt
```

## Running Inference

```bash
# Set required environment variables
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Run the inference script (from repo root)
python inference.py
```

Expected output format:
```
[START] task=easy env=attention_allocation model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=item_id=3 reward=1.24 done=false error=null
[STEP] step=2 action=item_id=0 reward=0.98 done=false error=null
[END] success=true steps=5 score=0.821 rewards=1.24,0.98,...
```

## Running Baseline Agents

```bash
# Evaluate all agents across all tasks
python -m scripts.evaluate

# Plot performance comparison chart
python -m scripts.plot_results

# Run a single episode
python -m scripts.run_single_episode

# Train Q-Learning agent
python -m scripts.train_q_learning

# Train DQN agent
python -m scripts.train_dqn
```

## Baseline scores

Scores from running `python inference.py` with Qwen/Qwen2.5-72B-Instruct:

| Task   | Steps | Score |
|--------|-------|-------|
| Easy   | 5     | 0.854 |
| Medium | 7     | 0.554 |
| Hard   | 8     | 0.429 |

## Running with Docker

```bash
docker build -t attention-allocation .

docker run \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  attention-allocation
```

## HF Spaces Deployment

The environment is deployed on Hugging Face Spaces at:
**[ADD YOUR HF SPACE URL HERE]**

The Space exposes a REST API compatible with the OpenEnv spec:
- `POST /reset` — start a new episode
- `POST /step` — take one action
- `GET /state` — get current observation

## Environment Variables

| Variable      | Required | Default                           | Description               |
|---------------|----------|-----------------------------------|---------------------------|
| `HF_TOKEN`    | Yes      | —                                 | Hugging Face API key      |
| `API_BASE_URL`| No       | `https://router.huggingface.co/v1`| LLM endpoint              |
| `MODEL_NAME`  | No       | `Qwen/Qwen2.5-72B-Instruct`       | Model to use for inference|

## Project Structure

```
.
├── inference.py          # LLM inference script (hackathon entry point)
├── openenv.yaml          # OpenEnv spec declaration
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── app.py                # HF Spaces FastAPI server
├── env/
│   ├── environment.py    # AttentionEnv (step/reset/state)
│   ├── models.py         # Pydantic models
│   └── tasks.py          # easy / medium / hard task factories
├── agents/
│   ├── greedy_agent.py
│   ├── q_learning_agent.py
│   └── dqn_agent.py
└── scripts/
    ├── evaluate.py
    ├── plot_results.py
    ├── train_q_learning.py
    └── train_dqn.py
```

## HF Spaces Deployment

The environment is deployed on Hugging Face Spaces at:
**https://huggingface.co/spaces/prashasti12/attention-env**