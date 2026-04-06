import os
import json
import textwrap

from openai import OpenAI
from env.models import Action

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Uses the same free HuggingFace endpoint as inference.py — no payment needed
client = OpenAI(
    api_key=os.getenv("HF_TOKEN") or os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
)
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

SYSTEM_PROMPT = textwrap.dedent("""
    You are a content recommendation agent.
    Given a user profile and available items, pick the ONE item_id that maximises:
        reward = dot(user.interest_vector, item.topic_vector)
                 + 0.6 * diversity
                 + 0.3 * item.quality
                 - 0.5 * user.fatigue

    Diversity = 1 - similarity to the last recommended item.
    Prefer shorter items (lower length) when fatigue is above 0.8.

    Reply with ONLY valid JSON, no explanation:
    {"item_id": <integer>}
""").strip()


def llm_agent(state) -> Action:
    user = state.user
    items_desc = "\n".join(
        f"  id={item.id}  topic={[round(v,2) for v in item.topic_vector]}"
        f"  quality={item.quality:.2f}  length={item.length}"
        for item in state.items
    )
    prompt = textwrap.dedent(f"""
        User:
          interest_vector : {[round(v,2) for v in user.interest_vector]}
          fatigue         : {user.fatigue:.2f}
          session_time    : {user.session_time}

        Available items:
        {items_desc}

        Reply ONLY with: {{"item_id": <integer>}}
    """).strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=32,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        item_id = int(json.loads(raw)["item_id"])

        available_ids = [item.id for item in state.items]
        if item_id not in available_ids:
            print(f"[WARN] LLM chose invalid id={item_id}, falling back.")
            item_id = available_ids[0]

        return Action(item_id=item_id)

    except Exception as e:
        print(f"[WARN] LLM call failed: {e}. Falling back to first item.")
        return Action(item_id=state.items[0].id)