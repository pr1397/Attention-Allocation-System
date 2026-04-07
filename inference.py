"""
Inference Script - Attention Allocation System
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
import numpy as np
from typing import List, Optional

from openai import OpenAI
from env.tasks import task_easy, task_medium, task_hard

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
TEMPERATURE  = 0.0
MAX_TOKENS   = 64

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def greedy_fallback(state):
    user = state.user
    best_score = float("-inf")
    best_id    = state.items[0].id
    for item in state.items:
        engagement     = float(np.dot(user.interest_vector, item.topic_vector))
        length_penalty = item.length * 0.3 if user.fatigue > 0.8 else 0.0
        score          = engagement + 0.3 * item.quality - 0.5 * user.fatigue - length_penalty
        if score > best_score:
            best_score = score
            best_id    = item.id
    return best_id

SYSTEM_PROMPT = textwrap.dedent("""
    You are a content recommendation agent optimising for long-term user satisfaction.

    Reward formula each step:
        reward = engagement + 0.6 * diversity + 0.3 * quality - 0.5 * fatigue

    Where:
        engagement = dot_product(user.interest_vector, item.topic_vector)  [pre-computed]
        diversity  = 1 - dot_product(last_item.topic_vector, item.topic_vector)  [pre-computed]
        quality    = item.quality
        fatigue    = current user fatigue

    Critical rules:
    - Session ENDS if fatigue exceeds 1.2. Fatigue += 0.1 * item.length per step.
    - When fatigue > 0.8, ONLY pick items where fatigue_after shows (safe).
    - Avoid topics similar to ANY item in full history.
    - est_reward is pre-computed - just pick the highest value item marked (safe).

    Reply with ONLY: {"item_id": <integer>}
""").strip()

def build_user_prompt(state, history):
    user       = state.user
    last_topic = history[-1]["topic"] if history else None

    lines = []
    for item in state.items:
        engagement = float(np.dot(user.interest_vector, item.topic_vector))
        diversity  = (1.0 - float(np.dot(last_topic, item.topic_vector))) if last_topic else 1.0
        est_reward = engagement + 0.6 * diversity + 0.3 * item.quality - 0.5 * user.fatigue
        fat_after  = user.fatigue + 0.1 * item.length
        note       = "ENDS SESSION" if fat_after > 1.2 else "safe"
        lines.append(
            f"  id={item.id}  est_reward={est_reward:.3f}  engagement={engagement:.3f}"
            f"  diversity={diversity:.3f}  quality={item.quality:.2f}"
            f"  length={item.length}  fatigue_after={fat_after:.2f}({note})"
        )

    history_str    = " → ".join(f"id={h['id']}" for h in history) if history else "none yet"
    fatigue_warn   = ""
    if user.fatigue > 0.8:
        fatigue_warn = f"\n  *** URGENT: {user.fatigue:.2f}/1.2 - avoid ENDS SESSION items ***"

    return textwrap.dedent(f"""
        User state:
          interest_vector : {[round(v,2) for v in user.interest_vector]}
          fatigue         : {user.fatigue:.2f}/1.2{fatigue_warn}
          session_time    : {user.session_time}

        Full pick history: {history_str}

        Available items:
        {chr(10).join(lines)}

        Pick the highest est_reward item that shows (safe).
        Reply ONLY with: {{"item_id": <integer>}}
    """).strip()

def get_llm_action(client, state, history):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(state, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw     = (response.choices[0].message.content or "").strip()
        raw     = raw.replace("```json","").replace("```","").strip()
        item_id = int(json.loads(raw)["item_id"])

        available_ids = [i.id for i in state.items]
        if item_id not in available_ids:
            print(f"[WARN] LLM chose invalid id={item_id}. Using greedy fallback.", flush=True)
            return greedy_fallback(state)
        return item_id

    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}. Using greedy fallback.", flush=True)
        return greedy_fallback(state)

def run_episode(client, env, task_name, norm):
    from env.models import Action
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    state   = env.reset()
    rewards = []
    history = []
    done    = False
    step    = 0

    while not done:
        step += 1
        try:
            item_id     = get_llm_action(client, state, history)
            chosen_item = next(i for i in state.items if i.id == item_id)
            topic_saved = list(chosen_item.topic_vector)  # save BEFORE step removes it
            eng_saved   = float(np.dot(state.user.interest_vector, topic_saved))

            action              = Action(item_id=item_id)
            state, reward, done, _ = env.step(action)

            r = float(reward.value)
            rewards.append(r)
            history.append({"id": item_id, "topic": topic_saved, "engagement": eng_saved})
            log_step(step=step, action=f"item_id={item_id}", reward=r, done=done, error=None)

        except Exception as exc:
            log_step(step=step, action="null", reward=0.0, done=True, error=str(exc))
            done = True

    total_reward = sum(rewards)
    score = min(1.0, max(0.0, total_reward / norm)) if norm > 0 else 0.0
    log_end(success=score > 0.0, steps=step, score=score, rewards=rewards)
    return score

TASKS = [
    ("easy",   task_easy,   4.0),
    ("medium", task_medium, 7.0),
    ("hard",   task_hard,  11.0),
]

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores = {}
    for task_name, task_fn, norm in TASKS:
        env   = task_fn()
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