# Training Guide

This document covers the full training pipeline, network architecture, environment mechanics, and evaluation for the PPO agents in GridWorldFireGame.

---

## Project Structure

```
src/
├── env.py                          # GridWorldFireGame environment
├── heuristic_agent.py              # HeuristicAgent (rule-based baseline)
├── random_agent.py                 # RandomAgent (random-action baseline)
├── actor_gradcam.py                # Grad-CAM visualization for actor networks
├── play_vs_model.py                # Interactive game launcher (human/model/heuristic)
├── evaluate.py                     # Head-to-head evaluation between agents
├── tournament.py                   # Round-robin tournament with Elo and TrueSkill
├── visualization.py                # Pygame isometric UI
├── generate_hill_map.py            # Procedural terrain generator
├── visualize_hill_map.py           # Terrain visualization utility
├── map_seeds/
│   └── valid_seeds_10.npy          # Pre-computed valid terrain seeds (10×10)
├── tested_networks/
│   ├── network.py                  # Baseline PPONetwork (shared trunk, BatchNorm)
│   ├── network_groupnorm.py        # GroupNorm variant (fixes PPO train/eval mismatch)
│   ├── grid_full_separate.py       # Separate actor and critic conv trunks
│   ├── better_actor.py             # Deeper actor (6 conv layers + residual block)
│   └── territorial_reward.py       # GridFullSeparate trained with territorial reward
└── training/
    ├── train_full_pipeline.py      # Pipeline for the baseline network
    ├── train_full_pipeline_groupnorm.py  # Pipeline for the GroupNorm network
    └── utils.py                    # State normalisation, checkpointing, seeding
```

---

## Pipeline Overview

Two pipeline scripts exist, one for each primary network. Both follow the same three-phase structure:

```
heuristic agent trajectories
          ↓
train_imitation()           — supervised pretraining on heuristic actions
          ↓
checkpoints/pretrained_pipeline.pt
          ↓
run_ppo_stage()             — alternating PPO self-play / vs heuristic curriculum
          ↓
checkpoints/ppo_agent_a_<name>.pt
checkpoints/ppo_agent_b_<name>.pt
training_logs/imitation_training.csv
training_logs/ppo_episode_metrics.csv
```

| Script | Network | Notes |
|--------|---------|-------|
| `train_full_pipeline.py` | `PPONetwork` (BatchNorm) | Generates imitation data from scratch; alternating self-play |
| `train_full_pipeline_groupnorm.py` | `PPONetwork` (GroupNorm) | Loads existing imitation checkpoint; curriculum vs heuristic |

Configuration is done by editing the capital-case constants at the top of each file — there are no CLI arguments.

---

## Phase 1 — Imitation Pretraining

**Function:** `train_imitation()`

Trains the network actor path with supervised cross-entropy on heuristic agent trajectories. Player 2's state is perspective-flipped so both players' data looks identical to the network.

Only the actor path is trained — the conv backbone, `fc1_actor`, `fc2_actor`, and the `actor` head. At the start of PPO, the critic FC layers are overwritten with a copy of the actor FC layers so the critic starts with meaningful features rather than random weights.

An 80/20 train/validation split is applied per epoch. Training stops early when validation loss stops improving for `IMITATION_PATIENCE` consecutive epochs. The best checkpoint (lowest val loss) is saved to `PRETRAINED_SAVE_PATH`.

To skip imitation pretraining when you already have a checkpoint:

```python
RUN_IMITATION_PRETRAINING = False
PRETRAINED_SAVE_PATH = "checkpoints/your_existing_checkpoint.pt"
```

### Imitation constants

| Constant | Default | Description |
|----------|---------|-------------|
| `RUN_IMITATION_PRETRAINING` | `True` | Set to `False` to skip and use an existing checkpoint |
| `IMITATION_EPOCHS` | 30 | Max training epochs |
| `IMITATION_BATCH_SIZE` | 64 | Minibatch size |
| `IMITATION_LR` | 1e-3 | Adam learning rate |
| `IMITATION_PATIENCE` | 5 | Early stopping patience on validation loss |

**What to expect:** validation accuracy around 65–75% before early stopping.

---

## Phase 2 — PPO Training

**Function:** `run_ppo_stage()`

Both pipeline scripts use vectorised parallel environments (`DualRewardVecEnv`) and a reward RMS normaliser (`RunningMeanStd`) before GAE. The first `WARMUP_UPDATES` PPO updates train the critic only (no policy or entropy loss), letting the value head calibrate before the actor gradient starts.

The two pipelines differ in their curriculum strategy:

**`train_full_pipeline.py`** — alternating self-play:
- Agent A (offensive reward) and Agent B (defensive reward) train in the same rollout, switching which agent gets updated every `SWITCH_EVERY_ROLLOUTS` rollouts.
- Grid curriculum: 2000 episodes on 10×10 → 3000 episodes on 15×15.

**`train_full_pipeline_groupnorm.py`** — heuristic-anchored curriculum:
- Stages alternate between `train_a_vs_heuristic`, `train_b_vs_heuristic`, and `self_play`.
- Training ends on a `train_a_vs_heuristic` stage so the final checkpoint is anchored to heuristic performance rather than eroded by self-play.
- `SAVE_EVERY_STAGE = True` checkpoints after each stage.

### PPO hyperparameters

| Constant | Value | Description |
|----------|-------|-------------|
| `NUM_ENVS` | 12 | Parallel environment workers |
| `ROLLOUT_STEPS` | 1024–4096 | Steps collected before each PPO update |
| `WARMUP_UPDATES` | 0–10 | Critic-only updates at the start of training |
| `PPO_LR` | 3e-4 | Adam learning rate |
| `GAMMA` | 0.99–0.995 | Discount factor |
| `GAE_LAMBDA` | 0.95 | GAE smoothing (0 = TD, 1 = full Monte Carlo) |
| `CLIP_EPSILON` | 0.2 | PPO clipping range |
| `PPO_EPOCHS` | 4 | Gradient passes over each rollout |
| `MINIBATCH_SIZE` | 128 | Samples per gradient step |
| `VALUE_COEFF` | 0.25 | Weight of critic loss |
| `ENTROPY_COEFF` | 0.01 | Entropy bonus weight |
| `MAX_GRAD_NORM` | 0.5 | Gradient clipping threshold |

---

## Running the Pipelines

```bash
python src/training/train_full_pipeline.py
python src/training/train_full_pipeline_groupnorm.py
```

Edit the constants at the top of the relevant file before running.

---

## Network Architectures

All networks take a 4-channel board state `(batch, 4, H, W)` and output action logits and a state value.

### `PPONetwork` — `tested_networks/network.py`

Shared convolutional trunk with BatchNorm feeding into separate actor and critic linear heads. The critic receives a detached copy of the backbone features so critic gradients do not flow back into the CNN.

```
Input: (batch, 4, H, W)
  └─ Conv2d(4 → 32) + BatchNorm2d + ReLU
  └─ Conv2d(32 → 64) + BatchNorm2d + ReLU
  └─ Conv2d(64 → 64) + BatchNorm2d + ReLU
  └─ AdaptiveAvgPool2d → (8, 8)
  └─ Flatten → 4096

  Actor tower:                    Critic tower (detached):
  Linear(4096→256) + ReLU         Linear(4096→256) + ReLU
  Linear(256→128)  + ReLU         Linear(256→128)  + ReLU
  Linear(128→4) → action logits   Linear(128→1) → value V(s)
```

### `PPONetwork` — `tested_networks/network_groupnorm.py`

Same architecture as above with BatchNorm replaced by GroupNorm. GroupNorm behaves identically in `train()` and `eval()`, avoiding the importance-ratio corruption that BatchNorm causes when PPO rolls out in `eval()` but updates in `train()`.

### `GridFullSeparatePPONetwork` — `tested_networks/grid_full_separate.py`

Fully separate convolutional trunks for actor and critic. Each trunk runs 3 Conv2d layers + BatchNorm + pooling independently.

```
Actor trunk:  Conv(4→32→64→64) + BN + Pool → Linear(4096→256→128→4 logits)
Critic trunk: Conv(4→32→64→64) + BN + Pool → Linear(4096→256→128→1 value)
```

### `BetterActorPPONetwork` — `tested_networks/better_actor.py`

Deeper actor trunk (6 conv layers, residual block, LayerNorm) paired with a shallower critic.

```
Actor trunk:  6 conv layers (4→32→64→96→128→128→128)
              + residual skip on last two 128-channel convs
              + Pool → LayerNorm → Linear(4096→1024) → LayerNorm → Linear(1024→256→4 logits)

Critic trunk: 4 conv layers (4→32→64→96→128)
              + Pool → Linear(4096→512→128→1 value)
```

### `TerritorialRewardPPONetwork` — `tested_networks/territorial_reward.py`

Same architecture as `GridFullSeparatePPONetwork`, trained with the territorial reward method.

---

## State Channels

| Channel | Content | Normalisation |
|---------|---------|---------------|
| 0 | Terrain elevation (integer 0–5) | ÷ 5 → [0, 1] |
| 1 | Player positions: +lives/max\_lives (self), −lives/max\_lives (opponent) | already in [−1, 1] |
| 2 | Fire trails: positive timer = self fire, negative = opponent fire | ÷ 10 → [−1, 1] |
| 3 | Booster locations: 1.0 where a booster exists, else 0 | no change |

For player 2's perspective, channels 1 and 2 are negated by `state_for_player()` in `utils.py` so the network always sees itself as the positive player.

---

## Reward System

Rewards are computed in `src/env.py`. Per-step rewards accumulate each turn; terminal rewards replace the step reward on the final turn.

### Per-step rewards

| Event | Reward | Notes |
|-------|--------|-------|
| Active burning tile (each, per step) | +1 | Passive — rewarded for tiles on fire |
| Wall or grid-border hit | −1 | Moving into an impassable or out-of-bounds tile |
| Player collision | −1 | Both players attempt the same tile |
| Cliff fall (elevation drop > 1) | −20, −1 life | Moving to a tile more than 1 lower than current |
| Stepping on enemy fire | −20, −1 life | Fire owner receives +20 |
| Stepping on own fire | −5 | No life lost |
| Booster pickup | +5 | Also refreshes all of the player's fire timers |

### Terminal rewards (replace step reward on final turn)

| Outcome | Reward |
|---------|--------|
| Win (opponent loses last life) | +100 |
| Loss (you lose last life) | −100 |
| Draw (both die simultaneously, or time limit with equal lives) | −100 to both |

Time limit is `MAX_EPISODE_LENGTH` turns. Draws are penalised as heavily as losses to encourage decisive play over stalling.

### Reward methods

| Method | Description |
|--------|-------------|
| `"default"` | Standard rewards as above |
| `"offensive"` | Additional bonus for enemy stepping on your fire |
| `"defensive"` | Additional bonus for avoiding enemy fire and staying alive |

---

## Evaluation

### Head-to-head — `src/evaluate.py`

Runs round-robin games between any combination of agents:

```bash
python src/evaluate.py --games 1000 --agents \
    ppo_agent_a_groupnorm.pt \
    ppo_agent_b_defensive.pt \
    heuristic \
    random
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--games` | 100 | Games per matchup |
| `--grid-size` | 10 | Board size |
| `--agents` | `heuristic random` | Space-separated list: `random`, `heuristic`, or `.pt` filename from `checkpoints/` |
| `--all-checkpoints` | off | Add all `.pt` files in `checkpoints/` to the agent list |

Architecture is inferred from the checkpoint filename (e.g. `groupnorm` → GroupNorm, `better_actor` → BetterActor).

### Tournament — `src/tournament.py`

Round-robin tournament with Elo and TrueSkill ratings, a ranked leaderboard, and a report figure:

```bash
python src/tournament.py --games 200
python src/tournament.py --games 200 --sort-by elo --no-show
```

Results are saved to `results/`.

---

## Other Tools

### Interactive Play — `src/play_vs_model.py`

```bash
python src/play_vs_model.py
```

Supports human vs model, human vs heuristic, model vs heuristic, and model vs model via interactive prompts or CLI flags.

### Grad-CAM — `src/actor_gradcam.py`

Applies Gradient-weighted Class Activation Maps to a trained actor to show which board cells influenced a given action decision.

```bash
python src/actor_gradcam.py --checkpoint checkpoints/ppo_agent_a_groupnorm.pt --architecture GroupNorm
```

Available `--architecture` options: `grid full separate`, `better actor`, `territorial reward`, `GroupNorm`.

### Terrain Seeds — `src/map_seeds/valid_seeds_10.npy`

Pre-computed numpy array of RNG seeds that produce valid, fully-connected terrain maps for a 10×10 grid. Used to skip the connectivity validation step during training when speed matters.

---

## Reading the Training Log

### Imitation phase

```
imitation epoch 005 | train_loss=0.8312 train_acc=0.712 | val_loss=0.8901 val_acc=0.698
```

Target: val accuracy 65–75% before early stopping.

### PPO phase

```
episode 0240 | grid=10 | updated=agent_a | policy_loss=0.021 value_loss=1.843 entropy=0.312
```

| Field | Healthy range |
|-------|--------------|
| `policy_loss` | ±0.001 – ±0.1 |
| `value_loss` | 0.1 – 10 (drops over time) |
| `entropy` | 0.1 – 0.8 (max = ln(4) ≈ 1.386) |

Episode metrics are written to `training_logs/ppo_episode_metrics.csv` for plotting.
