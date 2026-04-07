"""
HF Spaces server — REST API for OpenEnv spec + Angular UI backend.
"""
import os, random
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env.environment import AttentionEnv
from env.models import Action
from env.tasks import task_easy, task_medium, task_hard
from agents.greedy_agent import greedy_agent
from agents.q_learning_agent import q_learning_agent

app = FastAPI(title="Attention Allocation System", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


_env: Optional[AttentionEnv] = None

def get_env() -> AttentionEnv:
    global _env
    if _env is None:
        _env = AttentionEnv()
        _env.reset()
    return _env

CATEGORIES = ["Technology", "Gaming", "Education", "Music", "Fitness"]
VIDEO_POOL = []
for i in range(60):
    cat_idx = i % 5
    NUM_TOPICS = len(CATEGORIES)
    topic = [0.0] * NUM_TOPICS
    topic[cat_idx] = 0.7
    topic[(cat_idx + 1) % NUM_TOPICS] = 0.2
    topic[(cat_idx + 2) % NUM_TOPICS] = 0.1
    VIDEO_POOL.append({
        "id":           i,
        "title":        f"{CATEGORIES[cat_idx]} Video {i}",
        "thumbnail":    f"https://picsum.photos/seed/{i+100}/400/700",
        "channel":      f"Channel {i%10}",
        "category":     CATEGORIES[cat_idx],
        "duration":     f"{random.randint(1,15)}:{random.randint(10,59)}",
        "views":        f"{random.randint(10,999)}K",
        "topic_vector": topic,
        "quality":      round(random.uniform(0.5, 1.0), 2),
        "length":       random.randint(1, 5),
    })

class StepRequest(BaseModel):
    item_id: int

class FeedbackRequest(BaseModel):
    video_id: int
    action: str
    watch_time: Optional[float] = 0.0

@app.get("/")
def root():
    return {"name": "Attention Allocation System", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    env = get_env()
    obs = env.reset()
    return {"observation": obs.model_dump(), "done": False}

@app.post("/step")
def step(request: StepRequest):
    env = get_env()
    try:
        action = Action(item_id=request.item_id)
        obs, reward, done, info = env.step(action)
        return {"observation": obs.model_dump(), "reward": reward.value, "done": done, "info": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    env = get_env()
    if env.user is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state().model_dump()

@app.get("/recommend-feed")
def recommend_feed():
    env     = get_env()
    interest = list(env.user.interest_vector) if env.user else [0.33, 0.33, 0.34]
    fatigue  = env.user.fatigue if env.user else 0.0

    def score(v):
        return float(np.dot(interest, v["topic_vector"])) + 0.3*v["quality"] - 0.5*fatigue

    scored  = sorted(VIDEO_POOL, key=score, reverse=True)
    feed    = scored[:10] + random.sample(VIDEO_POOL, 5)
    random.shuffle(feed)
    return feed[:15]

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    env = get_env()
    env_item_id = request.video_id % max(len(env.items), 1)
    try:
        obs, reward, done, _ = env.step(Action(item_id=env_item_id))
        shaped = {"click": 0.5, "watch": min(2.0, request.watch_time/10), "skip": -0.5}.get(request.action, float(reward.value))
        if done:
            env.reset()
        return {"reward": shaped, "done": done, "new_state": obs.model_dump()}
    except Exception:
        env.reset()
        return {"reward": 0.0, "done": True, "new_state": {}}

@app.get("/analytics")
def analytics():
    def run(env_fn, agent_fn, norm):
        env, state = env_fn(), None
        state = env.reset()
        steps, total, done = [], 0.0, False
        while not done:
            action = agent_fn(state)
            state, reward, done, _ = env.step(action)
            r = float(reward.value)
            total += r
            steps.append({"step": state.user.session_time, "item_id": action.item_id,
                          "reward": round(r,3), "fatigue": round(state.user.fatigue,2),
                          "cumulative": round(total,3)})
        return {"steps": steps, "total_reward": round(total,3),
                "score": round(min(1.0, max(0.0, total/norm)),3), "steps_taken": len(steps)}

    tasks   = [("easy", task_easy, 4.0), ("medium", task_medium, 7.0), ("hard", task_hard, 11.0)]
    agents  = {"greedy": greedy_agent, "q_learning": q_learning_agent}
    results = {}
    for tname, tfn, norm in tasks:
        results[tname] = {aname: run(tfn, afn, norm) for aname, afn in agents.items()}
    return results


# ── Run Server ─────────────────────────────────────────────────────────────────
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()