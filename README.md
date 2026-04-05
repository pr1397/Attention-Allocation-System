# Attention allocation system

# Overview
This environment simulates a real-world content recommendation system where an agent selects items to maximize user engagement while managing fatigue and diversity.
Applying RL to logic real world recommendation systems such as youtube, instagram or news apps personalisation
The goal of the model is to find a recommendation that maximisizes the engagement, maintaining diversity and reducing fatigue.

# Observation Space
- User:
  - interest_vector
  - fatigue
  - session_time
- Items:
  - topic_vector
  - quality
  - novelty
  - length

# Action Space
- Select an item_id from available items

# Reward Function
reward =
+ engagement (interest alignment)
+ diversity (avoid repetition)
+ quality
- fatigue penalty

## Tasks
| Task | Items | Difficulty |
|------|------|-----------|
| Easy | 5 | Low |
| Medium | 10 | Medium |
| Hard | 15 | High |

# Setup

```bash
# To install requirements
pip install -r requirements.txt
# To run evaluation scripts
python -m scripts.evaluate 
# To plot graphs
python -m scripts.plot_results
# To run a single episode
python -m scripts.run_single_episode
# To train q learning
python -m scripts.train_q_learning
# To train Deep QN
python -m scripts.train_dqn