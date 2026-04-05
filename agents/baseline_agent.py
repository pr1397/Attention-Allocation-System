# requirements specify the need for a baseline agent using OpenAI API, but it's currently commented out in the evaluation script as payment needed for API calls. The baseline agent is implemented in baseline_agent.py and can be integrated into the evaluation once API access is set up.
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
from env.models import Action

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_agent(state):
    prompt = f"""
    You are optimizing a content feed.

    User:
    interest: {state.user.interest_vector}
    fatigue: {state.user.fatigue}

    Items:
    {[(i.id, i.topic_vector, i.quality) for i in state.items]}

    Pick the best item id.
    Return ONLY the number.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    item_id = int(response.choices[0].message.content.strip())
    return Action(item_id=item_id)