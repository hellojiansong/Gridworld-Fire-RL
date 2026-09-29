# GridWorld Fire Game

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MARL](https://img.shields.io/badge/Environment-Multi--Agent%20GridWorld-blue?style=flat)](#)
[![Algorithms](https://img.shields.io/badge/Algorithms-PPO%20%7C%20A2C-brightgreen?style=flat)](#)
[![Explainability](https://img.shields.io/badge/XAI-Grad--CAM-purple?style=flat)](#)
[![Self-Play](https://img.shields.io/badge/Training-Self--Play-orange?style=flat)](#)

A two-player simultaneous grid-world environment where players move across procedurally generated terrain, leave fire trails, collect boosters, and try to outlive their opponent. The environment is built for reinforcement learning experiments and supports human play, agent vs. agent, and evaluation tooling.

---

## Game Mechanics

- **Simultaneous Turns**: Two players choose actions simultaneously on a square grid.
- **Procedural Elevation (0–5)**: Generated terrain with hills and one-way cliff ridges.
- **Climbing & Cliff Rules**: Stepping up $>1$ elevation level is blocked; dropping down $>1$ level inflicts cliff damage (-1 life).
- **Dynamic Fire Trails**: Moving leaves a fire trail on the vacated tile.
  - Safe for the owner, lethal to the opponent (-1 life).
  - Stepping onto your own fire converts it into enemy fire once vacated.
  - Repeatedly attempting blocked moves ignites the current tile and inflicts damage.
- **Boosters**: Collecting a booster resets active fire timers and permanently extends the player's fire trail duration by +1 step.
- **Collision Handling**: Moving into the same cell or swapping positions simultaneously resolves safely by blocking both moves.
- **Termination**: An episode ends when a player loses all lives (each starts with 3), or when the maximum step limit is reached (the player with more remaining lives wins; equal lives result in a draw).

---

## Installation

```bash
pip install numpy torch matplotlib trueskill tqdm pygame
```

---

## Project Structure

```text
src/
  env.py                          # GridWorldFireGame environment
  heuristic_agent.py              # Rule-based agent
  random_agent.py                 # Random-action agent
  play_vs_model.py                # Interactive play (human vs model, model vs model, etc.)
  evaluate.py                     # Head-to-head evaluation between agents
  tournament.py                   # Round-robin tournament with Elo and TrueSkill ratings
  actor_gradcam.py                # Grad-CAM visualisation for trained checkpoints
  visualization.py                # Pygame-based 2.5D isometric rendering
  tested_networks/
    network.py                    # Baseline: shared conv trunk with BatchNorm
    network_groupnorm.py          # GroupNorm variant (fixes train/eval mode mismatch in PPO)
    grid_full_separate.py         # Separate actor and critic conv trunks with BatchNorm
    better_actor.py               # Deeper actor trunk (6 conv layers)
    territorial_reward.py         # Architecture used with the territorial reward variant
  training/
    train_full_pipeline.py        # Main training pipeline (BatchNorm baseline)
    train_full_pipeline_groupnorm.py  # Training pipeline for the GroupNorm network
    utils.py                      # State normalization, checkpoint helpers
checkpoints/                      # Saved .pt model weights
training_logs/                    # CSV logs written during training
results/                          # Tournament CSV output
```

---

## Network Architectures

| Name | File | Normalization | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline** | `network.py` | BatchNorm | Shared conv trunk, split FC heads. |
| **GroupNorm** | `network_groupnorm.py` | GroupNorm | Fixes train/eval mismatch in PPO importance ratios. |
| **Grid-Sep** | `grid_full_separate.py` | BatchNorm | Fully separate actor and critic conv trunks. |
| **BetterActor** | `better_actor.py` | LayerNorm (Actor FC) | Deeper actor (6 conv layers + residual blocks), separate trunks. |
| **Territorial** | `territorial_reward.py` | BatchNorm | Trained with territorial spatial fire-coverage reward shaping. |

> **Adaptive Spatial Pooling**: All architectures use an adaptive average-pooling layer ($8 \times 8$), allowing a single network to operate seamlessly across different grid sizes ($10 \times 10 \rightarrow 15 \times 15$).
>
> **GroupNorm Benefit**: The GroupNorm network replaces BatchNorm with `nn.GroupNorm` so that rollout collection (`eval` mode) and the PPO update (`train` mode) produce identical normalization statistics, avoiding corruption of the importance ratio.

---

## Training Pipeline

The pipeline runs in three phases:
1. **Imitation Pretraining** — Supervised learning on heuristic agent trajectories.
2. **Critic Warmup** — Value-only PPO updates to stabilise the critic before policy updates.
3. **Self-Play PPO** — Alternating updates over a $10 \times 10 \rightarrow 15 \times 15$ curriculum.

To train the GroupNorm network:
```bash
python src/training/train_full_pipeline_groupnorm.py
```

To train the baseline:
```bash
python src/training/train_full_pipeline.py
```

*Edit the capital-letter constants at the top of each file to configure paths, hyperparameters, and reward methods before running.*

---

## Play

Launch interactive play from the terminal (rendered via Pygame):
```bash
python src/play_vs_model.py
```
The script prompts for mode (human vs model, model vs model, etc.), grid size, architecture, and checkpoint.

---

## Evaluation

Head-to-head evaluation between any combination of agents (random, heuristic, or a `.pt` checkpoint):
```bash
python src/evaluate.py --games 1000 --agents \
  ppo_agent_a_groupnorm.pt \
  ppo_agent_b_defensive.pt \
  heuristic \
  random
```

---

## Tournament

Round-robin tournament with Elo and TrueSkill ratings, a ranked leaderboard, and a report figure:
```bash
python src/tournament.py --games 200
python src/tournament.py --games 100 --agents heuristic random ppo_agent_a_groupnorm.pt
python src/tournament.py --games 200 --sort-by elo --no-show
```
Results are saved to `results/`.

---

## Grad-CAM

Visualise what parts of the game state (terrain, fire, players, boosters) each network attends to:
```bash
python src/actor_gradcam.py --checkpoint checkpoints/ppo_agent_a_groupnorm.pt --architecture GroupNorm
```
Available `--architecture` options: `grid full separate`, `better actor`, `territorial reward`, `GroupNorm`.

---

## Environment API

```python
from env import GridWorldFireGame

env = GridWorldFireGame(
    grid_size=10,
    player_lives=3,
    max_episode_length=50,
    render=True,
    seed=42,
)

state = env.reset()
done = False

while not done:
    state, reward_1, reward_2, done, info = env.step(
        action_player1=3,  # right
        action_player2=2,  # left
    )
```

### Actions

| Value | Direction |
| :---: | :---: |
| `0` | Up |
| `1` | Down |
| `2` | Left |
| `3` | Right |
