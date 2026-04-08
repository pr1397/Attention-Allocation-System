from .environment import AttentionEnv

# EASY
def task_easy():
    return AttentionEnv(num_items=5)

def grade_easy(total_reward):
    return min(1.0, total_reward / 4.5)

# MEDIUM
def task_medium():
    return AttentionEnv(num_items=8)

def grade_medium(total_reward):
    return min(1.0, total_reward / 8)

# HARD
def task_hard():
    return AttentionEnv(num_items=15)

def grade_hard(total_reward):
    return min(1.0, total_reward / 12.5)