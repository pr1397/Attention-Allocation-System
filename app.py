"""
HF Spaces server — exposes the AttentionEnv via a REST API
that satisfies the OpenEnv spec (reset / step / state endpoints).

The hackathon automated checker pings this Space URL and calls reset()
to verify the environment is live and returns a 200.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn

from env.environment import AttentionEnv
from env.models import Action
from env.tasks import task_easy, task_medium, task_hard, grade_easy, grade_medium, grade_hard

app = FastAPI(title="Attention Allocation System", version="1.0.0")

# One global env instance per worker (fine for evaluation purposes)
_env: Optional[AttentionEnv] = None


def get_env() -> AttentionEnv:
    global _env
    if _env is None:
        _env = AttentionEnv()
    return _env


# ── Request/Response models ───────────────────────────────────────────────────
class StepRequest(BaseModel):
    item_id: int


class ResetResponse(BaseModel):
    observation: dict
    done: bool = False


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict = {}


class BaselineResponse(BaseModel):
    scores: Dict[str, float]


class GraderResponse(BaseModel):
    task: str
    score: float
    total_reward: float


class TasksResponse(BaseModel):
    tasks: List[str]
    action_schema: Dict[str, str]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Attention Allocation System",
        "description": "Content recommendation RL environment (OpenEnv compatible)",
        "endpoints": ["/reset", "/step", "/state", "/health", "/baseline", "/grader", "/tasks"],
    }


@app.get("/health")
def health():
    """Hackathon automated ping endpoint — must return 200."""
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset():
    """Start a new episode. Returns the initial observation."""
    env = get_env()
    obs = env.reset()
    return ResetResponse(observation=obs.model_dump(), done=False)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Take one action. Returns next observation, reward, and done flag."""
    env = get_env()
    try:
        action = Action(item_id=request.item_id)
        obs, reward, done, info = env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=reward.value,
            done=done,
            info=info,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    """Get the current observation without taking an action."""
    env = get_env()
    if env.user is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state().model_dump()


@app.get("/baseline", response_model=BaselineResponse)
def baseline():
    """
    Run baseline inference on all 3 tasks (easy, medium, hard).
    Returns the baseline score for each task.
    """
    from openai import OpenAI
    import os

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="No API key configured for baseline inference")
    
    api_base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    model_name = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    
    client = OpenAI(base_url=api_base_url, api_key=api_key)
    
    from inference import run_episode
    
    scores = {}
    tasks = [
        ("easy", task_easy, 4.0),
        ("medium", task_medium, 7.0),
        ("hard", task_hard, 11.0),
    ]
    
    try:
        for task_name, task_fn, norm in tasks:
            env = task_fn()
            score = run_episode(client, env, task_name, norm)
            scores[task_name] = score
        
        return BaselineResponse(scores=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline inference failed: {str(e)}")


@app.get("/grader")
def grader():
    """
    Calculate the grader score for the current episode.
    Returns the normalized score based on the number of items in the current environment.
    """
    env = get_env()
    if env.user is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    if not hasattr(env, 'history') or len(env.history) == 0:
        raise HTTPException(status_code=400, detail="No actions taken yet in the episode")
    
    total_reward = getattr(env, 'total_reward', 0.0)
    
    if env.num_items == 5:
        task_name = "easy"
        score = grade_easy(total_reward)
    elif env.num_items == 10:
        task_name = "medium"
        score = grade_medium(total_reward)
    elif env.num_items == 15:
        task_name = "hard"
        score = grade_hard(total_reward)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task with {env.num_items} items")
    
    return GraderResponse(task=task_name, score=score, total_reward=total_reward)


@app.get("/tasks", response_model=TasksResponse)
def tasks():
    """
    Returns the available tasks and the action schema required for each step.
    """
    return TasksResponse(
        tasks=["easy", "medium", "hard"],
        action_schema={
            "item_id": "integer (ID of the item to recommend)"
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
