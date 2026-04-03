from pydantic import BaseModel
from typing import List

# pydantic models used for OpenEnv compliance. Base class adds data validation, conversion

class UserState(BaseModel):
    interest_vector: List[float]  # what the user likes (e.g., [sports, tech, music])
    fatigue: float # measure of the boredom
    session_time: int # how long the user has been interacting

class Item(BaseModel):
    id: int
    topic_vector: List[float] # what the item is about
    quality: float # is the quality of the item good
    novelty: float # is the item new to the user
    length: int # how long the item is (e.g., in seconds for videos, or word count for articles)

# Observation is what the agent sees
class Observation(BaseModel):
    user: UserState
    items: List[Item]

# Which item id the agent chooses
class Action(BaseModel):
    item_id: int

# Feedback from the environment after taking an action
class Reward(BaseModel):
    value: float