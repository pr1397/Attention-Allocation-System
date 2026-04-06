"""
Inference Script — Attention Allocation System
===============================================
The LLM acts as the recommendation agent. Each step it receives
the current user state and available items, then picks an item_id.

Environment variables required at runtime (injected by the hackathon infra):
    API_BASE_URL  – LLM endpoint  (default: HuggingFace router)
    MODEL_NAME    – model string  (default: Qwen2.5-72B-Instruct)
    HF_TOKEN      – API key

Stdout format (strictly required):
    [START] task=<n> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<null|msg>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI
from env.tasks import task_easy, task_medium, task_hard

# Safe local dev helper — silently ignored if dotenv isn't installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
BENCHMARK    = "attention_allocation"
TEMPERATURE  = 0.2
MAX_TOKENS   = 64

# ── Logging helpers ────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int,
            score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are a content recommendation agent optimising for long-term user satisfaction.

    The reward formula is:
        reward = engagement + 0.6 * diversity + 0.3 * quality - 0.5 * fatigue

    Where:
        engagement = dot_product(user.interest_vector, item.topic_vector)
        diversity  = 1 - dot_product(last_item.topic_vector, item.topic_vector)
        quality    = item.quality
        fatigue    = current user fatigue level

    Important rules:
    - Fatigue increases by (0.1 * item.length) after each pick. Session ends if fatigue > 1.2.
    - Prefer shorter items when user fatigue is above 0.8.
    - Avoid items with topic_vector similar to recently recommended items.
    - High quality items are always a bonus.

    Reply with ONLY a JSON object — no explanation, no markdown:
    {"item_id": <integer>}
""").strip()


def build_user_prompt(state, history_ids: List[int]) -> str:
    user = state.user
    items_desc = "\n".join(
        f"  id={item.id}  topic={[round(v, 2) for v in item.topic_vector]}"
        f"  quality={item.quality:.2f}  length={item.length}  novelty={item.novelty:.2f}"
        for item in state.items
    )
    return textwrap.dedent(f"""
        User state:
          interest_vector : {[round(v, 2) for v in user.interest_vector]}
          fatigue         : {user.fatigue:.2f}  (session ends if > 1.2)
          session_time    : {user.session_time}

        Last 3 recommended item ids: {history_ids[-3:] if history_ids else 'none yet'}

        Available items:
        {items_desc}

        Pick the item_id that maximises: engagement + 0.6*diversity + 0.3*quality - 0.5*fatigue
        Reply ONLY with: {{"item_id": <integer>}}
    """).strip()


# ── LLM call ──────────────────────────────────────────────────────────────────
def get_llm_action(client: OpenAI, state, history_ids: List[int]) -> int:
    prompt = build_user_prompt(state, history_ids)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw)
        item_id = int(parsed["item_id"])

        available_ids = [item.id for item in state.items]
        if item_id not in available_ids:
            print(f"[WARN] LLM chose invalid id={item_id}, "
                  f"available={available_ids}. Falling back.", flush=True)
            item_id = available_ids[0]

        return item_id

    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}. Falling back.", flush=True)
        return state.items[0].id


# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(client: OpenAI, env, task_name: str, norm: float) -> float:
    from env.models import Action

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    state = env.reset()
    rewards: List[float] = []
    history_ids: List[int] = []
    done = False
    step = 0

    while not done:
        step += 1
        try:
            item_id = get_llm_action(client, state, history_ids)
            action = Action(item_id=item_id)
            state, reward, done, _ = env.step(action)

            r = float(reward.value)
            rewards.append(r)
            history_ids.append(item_id)
            log_step(step=step, action=f"item_id={item_id}",
                     reward=r, done=done, error=None)

        except Exception as exc:
            log_step(step=step, action="null", reward=0.0,
                     done=True, error=str(exc))
            done = True

    total_reward = sum(rewards)
    score = min(1.0, max(0.0, total_reward / norm)) if norm > 0 else 0.0
    log_end(success=score > 0.0, steps=step, score=score, rewards=rewards)
    return score


# ── Task registry ──────────────────────────────────────────────────────────────
TASKS = [
    ("easy",   task_easy,   4.0),
    ("medium", task_medium, 7.0),
    ("hard",   task_hard,  11.0),
]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = {}
    for task_name, task_fn, norm in TASKS:
        env = task_fn()
        score = run_episode(client, env, task_name, norm)
        all_scores[task_name] = score
        print(flush=True)

    print("=" * 40, flush=True)
    print(f"{'Task':<10} {'Score':>6}", flush=True)
    print("-" * 40, flush=True)
    for task_name, score in all_scores.items():
        print(f"{task_name:<10} {score:>6.3f}", flush=True)
    print("=" * 40, flush=True)


if __name__ == "__main__":
    main()