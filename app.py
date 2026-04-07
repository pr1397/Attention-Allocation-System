"""
HF Spaces server - exposes the AttentionEnv via a REST API
that satisfies the OpenEnv spec (reset / step / state endpoints).

The hackathon automated checker pings this Space URL and calls reset()
to verify the environment is live and returns a 200.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env.environment import AttentionEnv
from env.models import Action

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


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Attention Allocation System",
        "description": "Content recommendation RL environment (OpenEnv compatible)",
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
def health():
    """Hackathon automated ping endpoint - must return 200."""
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
