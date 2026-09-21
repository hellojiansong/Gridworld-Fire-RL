# GridWorld Fire Game

A two-player simultaneous grid-world environment where players move across procedurally generated terrain, leave fire trails, collect boosters, and try to outlive their opponent. The environment is built for reinforcement learning experiments and supports human play, agent vs. agent, and evaluation tooling.

## Game mechanics

- Two players move simultaneously on a square grid
- Procedurally generated terrain with elevation levels 0–5
- Moving leaves a fire trail; trail duration scales with terrain elevation
- Stepping down too far causes cliff damage; stepping up too far is blocked
- Booster pickups available on the map
- Collision handling when both players enter the same cell
- Episode ends when a player loses all lives, both lose lives simultaneously, or max steps is reached

## Installation

```bash
pip install numpy torch matplotlib trueskill tqdm
```

## Project structure

```
src/
  env.py                          # GridWorldFireGame environment
  heuristic_agent.py              # Rule-based agent
  random_agent.py                 # Random-action agent
  play_vs_model.py                # Interactive play (human vs model, model vs model, etc.)
  evaluate.py                     # Head-to-head evaluation between agents
  tournament.py                   # Round-robin tournament with Elo and TrueSkill ratings
  actor_gradcam.py                # Grad-CAM visualisation for trained checkpoints
  visualization.py                # Pygame-based rendering
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

## Network architectures

| Name | File | Normalisation | Notes |
|---|---|---|---|
| Baseline | `network.py` | BatchNorm | Shared conv trunk, split FC heads |
| **GroupNorm** | `network_groupnorm.py` | GroupNorm | Fixes train/eval mismatch in PPO importance ratios |
| Grid-Sep | `grid_full_separate.py` | BatchNorm | Fully separate actor and critic conv trunks |
| BetterActor | `better_actor.py` | — | Deeper actor (6 conv layers), separate trunks |
| Territorial | `territorial_reward.py` | BatchNorm | Trained with territorial reward shaping |

The GroupNorm network replaces BatchNorm with `nn.GroupNorm` so that rollout collection (eval mode) and the PPO update (train mode) produce identical normalisation statistics, avoiding corruption of the importance ratio.

## Training pipeline

The pipeline runs in three phases:

1. **Imitation pretraining** — supervised learning on heuristic agent trajectories
2. **Critic warmup** — value-only PPO updates to stabilise the critic before policy updates
3. **Self-play PPO** — alternating updates over a 10×10 → 15×15 curriculum

To train the GroupNorm network:

```bash
python src/training/train_full_pipeline_groupnorm.py
```

To train the baseline:

```bash
python src/training/train_full_pipeline.py
```

Edit the capital-letter constants at the top of each file to configure paths, hyperparameters, and reward methods before running.

## Play

Launch interactive play from the terminal:

```bash
python src/play_vs_model.py
```

The script prompts for mode (human vs model, model vs model, etc.), grid size, architecture, and checkpoint.

## Evaluation

Head-to-head evaluation between any combination of agents (`random`, `heuristic`, or a `.pt` checkpoint):

```bash
python src/evaluate.py --games 1000 --agents \
  ppo_agent_a_groupnorm.pt \
  ppo_agent_b_defensive.pt \
  heuristic \
  random
```

## Tournament

Round-robin tournament with Elo and TrueSkill ratings, a ranked leaderboard, and a report figure:

```bash
python src/tournament.py --games 200
python src/tournament.py --games 100 --agents heuristic random ppo_agent_a_groupnorm.pt
python src/tournament.py --games 200 --sort-by elo --no-show
```

Results are saved to `results/`.

## Grad-CAM

Visualise what parts of the game state each network attends to:

```bash
python src/actor_gradcam.py --checkpoint checkpoints/ppo_agent_a_groupnorm.pt --architecture GroupNorm
```

Available `--architecture` options: `grid full separate`, `better actor`, `territorial reward`, `GroupNorm`.

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
|---|---|
| `0` | Up |
| `1` | Down |
| `2` | Left |
| `3` | Right |
