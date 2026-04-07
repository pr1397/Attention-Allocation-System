"""
HF Spaces server - exposes the AttentionEnv via a REST API
compatible with the OpenEnv spec (reset / step / state endpoints),
PLUS additional endpoints for a YouTube-style recommendation UI.
"""

from env.environment import AttentionEnv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import random
from fastapi.middleware.cors import CORSMiddleware

from env.models import Action
from agents.q_learning_agent import q_learning_agent,get_top_k_actions

app = FastAPI(title="Attention Allocation System", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[AttentionEnv] = None

# ── Mock Video Pool (replace later with real dataset) ──────────────────────────
VIDEO_POOL = [
    {
        "id": i,
        "title": f"Video {i}",
        "thumbnail": f"https://picsum.photos/300/200?random={i}",
        "category": i % 5,
    }
    for i in range(50)
]


def get_env() -> AttentionEnv:
    global _env
    if _env is None:
        _env = AttentionEnv()
    return _env


# ── Request models ─────────────────────────────────────────────────────────────
class StepRequest(BaseModel):
    item_id: int


class FeedbackRequest(BaseModel):
    video_id: int
    action: str  # "click" | "watch" | "skip"
    watch_time: Optional[float] = 0


# ── Base Endpoints (UNCHANGED) ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Attention Allocation System",
        "description": "Content recommendation RL environment (OpenEnv compatible)",
        "endpoints": [
            "/reset",
            "/step",
            "/state",
            "/health",
            "/recommend-feed",
            "/feedback",
        ],
    }


@app.get("/health")
def health():
    """Hackathon automated ping - must return 200."""
    return {"status": "ok"}


@app.post("/reset")
def reset():
    """Start a new episode. Returns the initial observation."""
    env = get_env()
    obs = env.reset()
    return {"observation": obs.model_dump(), "done": False}


@app.post("/step")
def step(request: StepRequest):
    """Take one action. Returns next observation, reward, and done flag."""
    env = get_env()
    try:
        action = Action(item_id=request.item_id)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.value,
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    """Get current observation without taking an action."""
    env = get_env()
    if env.user is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state().model_dump()


@app.get("/recommend-feed")
def recommend_feed():
    env = get_env()

    # Initialize if needed
    if env.user is None:
        env.reset()

    state = env.state()

    try:
        # Exploration vs Exploitation
        if random.random() < 0.2:
            selected_ids = random.sample(range(len(VIDEO_POOL)), 12)
        else:
            # Use RL agent
            selected_ids = get_top_k_actions(state, k=12)

    except Exception:
        # fallback (cold start)
        selected_ids = random.sample(range(len(VIDEO_POOL)), 12)

    # Map IDs → video objects
    videos = [VIDEO_POOL[i] for i in selected_ids if i < len(VIDEO_POOL)]

    return videos


# ──  Feedback Endpoint (connects UI → RL loop) ─────────────────────────────
@app.post("/feedback")
def feedback(request: FeedbackRequest):
    env = get_env()

    if env.user is None:
        env.reset()

    state = env.state()

    try:
        action = Action(item_id=request.video_id)

        obs, reward, done, info = env.step(action)

        # Reward shaping (UI signals → RL reward)
        if request.action == "click":
            reward = 1
        elif request.action == "watch":
            reward = min(2, request.watch_time / 10)
        elif request.action == "skip":
            reward = -2

        # Update RL agent
        q_learning_agent.update(state, request.video_id, reward)

        return {
            "reward": reward,
            "done": done,
            "new_state": obs.model_dump(),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Run Server ─────────────────────────────────────────────────────────────────
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()