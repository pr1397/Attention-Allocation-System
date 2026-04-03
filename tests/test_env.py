from env.environment import AttentionEnv
from env.models import Action

def test_env():
    env = AttentionEnv()
    state = env.reset()

    action = Action(item_id=state.items[0].id)
    next_state, reward, done, _ = env.step(action)

    assert next_state is not None
    assert reward.value is not None