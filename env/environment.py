import numpy as np
import random
from .models import Observation, UserState, Item, Action, Reward

class AttentionEnv:

    def __init__(self, num_items=50, num_topics=5, seed=42):
        self.user = None
        self.num_items = num_items
        self.num_topics = num_topics
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

    # Initialise a new episode
    def reset(self) -> Observation:
        self.user = UserState(
            interest_vector=np.random.dirichlet(np.ones(self.num_topics)).tolist(),
            fatigue=0.0,
            session_time=0
        )

        self.base_items = self._generate_items()
        self.items = self.base_items.copy()
        self.history = []
        self.last_action = None

        return self.state()

    def state(self) -> Observation:
        return Observation(user=self.user, items=self.items)

    def get_rl_state(self):
        return (
            tuple(np.round(self.user.interest_vector, 2)),
            round(self.user.fatigue, 2),
            self.last_action if self.last_action is not None else -1
        )

    def _generate_items(self):
        return [
            Item(
                id=i,
                topic_vector=np.random.dirichlet(np.ones(self.num_topics)).tolist(),
                quality=float(np.random.rand()),
                novelty=float(np.random.rand()),
                length=random.randint(1, 5)
            )
            for i in range(self.num_items)
        ]

    def step(self, action: Action):
        item = next(x for x in self.items if x.id == action.item_id)
        user = self.user

        # Engagement
        engagement = np.dot(user.interest_vector, item.topic_vector)

        # Diversity
        diversity = 1.0
        if self.history:
            last = self.history[-1]
            similarity = np.dot(last.topic_vector, item.topic_vector)
            diversity -= similarity

        # Reward
        reward_value = (
            engagement
            + 0.6 * diversity
            + 0.3 * item.quality
            - 0.5 * user.fatigue
        )

        # Update state
        user.fatigue += 0.1 * item.length
        user.session_time += 1

        self.history.append(item)
        self.last_action = item.id
        done = user.fatigue > 2.0

        return self.state(), Reward(value=float(reward_value)), done, {}