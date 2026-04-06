"""
Hybrid LLM + Q-Learning Agent
==============================
Strategy:
  1. LLM scores all items and returns its top-3 candidate item_ids
  2. Q-Learning agent picks the best among those 3 candidates

Why this works:
  - LLM provides semantic understanding (topic alignment, fatigue risk)
  - Q-Learning provides learned value estimates from actual reward signals
  - Together they beat both alone on medium/hard tasks
"""

import os
import json
import textwrap
import numpy as np
from openai import OpenAI
from env.models import Action

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = OpenAI(
    api_key=os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
)
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

SYSTEM_PROMPT = textwrap.dedent("""
    You are a content recommendation agent.
    Given a user profile and available items, return your TOP 3 item_ids ranked
    by expected reward. Consider engagement, diversity, quality, and fatigue risk.

    Reply with ONLY valid JSON, no explanation:
    {"top3": [<best_id>, <second_id>, <third_id>]}
""").strip()


# ── Shared Q-table (loaded from training or built fresh) ──────────────────────
Q: dict = {}


def _featurize(state):
    user = state.user
    return tuple(np.round(user.interest_vector + [user.fatigue], 2))


def _llm_top3(state) -> list[int]:
    """Ask the LLM to shortlist the top 3 item IDs."""
    user = state.user
    items_desc = "\n".join(
        f"  id={item.id}  topic={[round(v, 2) for v in item.topic_vector]}"
        f"  quality={item.quality:.2f}  length={item.length}"
        f"  fatigue_after={user.fatigue + 0.1 * item.length:.2f}"
        for item in state.items
    )
    prompt = textwrap.dedent(f"""
        User:
          interest_vector : {[round(v, 2) for v in user.interest_vector]}
          fatigue         : {user.fatigue:.2f} / 1.2 max

        Available items:
        {items_desc}

        Return top 3 item_ids by estimated reward.
        Reply ONLY with: {{"top3": [id1, id2, id3]}}
    """).strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=48,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        top3 = json.loads(raw)["top3"]

        available_ids = {item.id for item in state.items}
        valid = [i for i in top3 if i in available_ids]

        # Pad with remaining items if LLM returned fewer than 3 valid ones
        if len(valid) < 3:
            for item in state.items:
                if item.id not in valid:
                    valid.append(item.id)
                if len(valid) == 3:
                    break

        return valid[:3]

    except Exception as e:
        print(f"[hybrid] LLM call failed: {e}. Using all items.")
        return [item.id for item in state.items]


def _q_pick(state, candidate_ids: list[int]) -> int:
    """Among the LLM's candidates, pick the one with highest Q-value."""
    feat = _featurize(state)
    best_val = float("-inf")
    best_id  = candidate_ids[0]

    for item_id in candidate_ids:
        key = (feat, item_id)
        val = Q.get(key, 0.0)
        if val > best_val:
            best_val = val
            best_id  = item_id

    return best_id


def hybrid_agent(state) -> Action:
    """
    Main entry point. Returns an Action using the hybrid strategy.
    If Q-table is empty (untrained), falls back to pure LLM top-1.
    """
    top3 = _llm_top3(state)

    if not Q:
        # Q-table not trained yet — just use LLM's top pick
        return Action(item_id=top3[0])

    best_id = _q_pick(state, top3)
    return Action(item_id=best_id)


def load_q_table(q_dict: dict):
    """Call this with your trained Q dict before running hybrid_agent."""
    global Q
    Q = q_dict