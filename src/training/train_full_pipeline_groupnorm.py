"""
Full training pipeline:

1. Imitation pretraining from heuristic state/action data.
2. Initialize two PPO agents from the imitation checkpoint.
3. Copy actor hidden FC layers into critic hidden FC layers.
4. Alternating PPO self-play over a 10x10 -> 15x15 curriculum.

The PPO update follows the original train_ppo.py structure:
- multiple vectorized environments
- reward RMS normalization before GAE
- bootstrapped rollout returns
- value-only updates for the first WARMUP_UPDATES PPO updates

No argparse is used. Edit the capital variables below before running.
"""

import csv
import multiprocessing as mp
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TRAINING_DIR = os.path.join(SRC_DIR, "training")

for path in [SRC_DIR, TRAINING_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from env import GridWorldFireGame
from heuristic_agent import HeuristicAgent
from tested_networks.network_groupnorm import PPONetwork
from utils import normalize_state, state_for_player


# ---- Files ----

# Imitation checkpoint used to initialize both PPO agents.
PRETRAINED_SAVE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "pretrained_pipeline.pt")

# Final PPO checkpoints for both agents.
PPO_AGENT_A_SAVE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_agent_a_groupnorm.pt")
PPO_AGENT_B_SAVE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "ppo_agent_b_groupnorm.pt")

# CSV logs for plotting. These are the WORKING files written this run.
LOG_DIR = os.path.join(PROJECT_ROOT, "training_logs")
IMITATION_CSV_PATH = os.path.join(LOG_DIR, "imitation_training.csv")
PPO_EPISODE_CSV_PATH = os.path.join(LOG_DIR, "ppo_episode_metrics.csv")

# Archived history kept in groupnorm_logs/. On resume these are copied into the working
# files above so new rows append onto the full history; the archive itself is left
# untouched as a snapshot. On a fresh run the working files are truncated instead.
GROUPNORM_LOG_DIR = os.path.join(LOG_DIR, "groupnorm_logs")
IMITATION_ARCHIVE_PATH = os.path.join(GROUPNORM_LOG_DIR, "imitation_training.csv")
PPO_EPISODE_ARCHIVE_PATH = os.path.join(GROUPNORM_LOG_DIR, "ppo_episode_metrics_groupnorm.csv")

IMITATION_FIELDS = ["epoch", "train_loss"]

PPO_EPISODE_FIELDS = [
    "episode",
    "grid_size",
    "agent_name",
    "episodic_return",
    "win",
    "loss",
    "draw",
    "boosters_picked",
    "avg_fire_trail_length",
    "own_fire_steps",
    "enemy_fire_steps",
]


# ---- Imitation learning settings ----

# Set False if you already have PRETRAINED_SAVE_PATH and want to skip imitation.
RUN_IMITATION_PRETRAINING = False

# Two-phase curriculum imitation: learn fast on 10x10 first, then fine-tune on the
# 15x15 eval grid. Both phases train the SAME network/optimizer, so the 10x10
# knowledge carries into the 15x15 phase ("learn both"). Early stopping is
# per-phase; the best 15x15 model is what gets saved as the PPO prior.
# (grid_size, max_epochs, episodes_per_epoch)
IMITATION_SCHEDULE = [
    (10, 25, 500),
    (15, 10, 300),
]
IMITATION_BATCH_SIZE = 64
IMITATION_LR = 1e-3
IMITATION_PATIENCE = 5


# ---- PPO settings ----

RESUME_TRAINING = True       # False = restart PPO from the imitation checkpoint
SAVE_EVERY_STAGE = True

PLAYER_LIVES = 3
MAX_EPISODE_LENGTH = 500
POOL_SIZE = 8
SEED = 42

# Skip the O(grid^4) map traversability check on every reset. 15x15 seeds are not
# baked, so each reset otherwise pays for a generate+is_valid rejection loop. With
# this True, the first generated map is used as-is (slightly lower map quality,
# much faster resets). Set False to restore validated maps.
BYPASS_MAP_VALIDATION = True

# Agent A controls real player 1. Agent B controls real player 2.
AGENT_A_REWARD_METHOD = "default"
AGENT_B_REWARD_METHOD = "default"

GRID_SCHEDULE = [
    (15, 1500, "train_a_vs_heuristic"),  # quick first checkpoint to verify learning
]
for _ in range(10):
    GRID_SCHEDULE += [
        (15, 7500, "train_a_vs_heuristic"),
        (15, 500, "train_b_vs_heuristic"),
        (15, 500, "self_play"),
    ]
GRID_SCHEDULE += [
    (15, 7400, "train_a_vs_heuristic"),  # end on agent_a so the final save is heuristic-anchored
]

NUM_ENVS = 12
ROLLOUT_STEPS = 1024
WARMUP_UPDATES = 0

PPO_LR = 3e-4
GAMMA = 0.995        # longer horizon so the +500 terminal win propagates over 150-300 turns
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
PPO_EPOCHS = 4
MINIBATCH_SIZE = 128
VALUE_COEFF = 0.25
ENTROPY_COEFF = 0.01  # low: prefer exploitation over exploration
MAX_GRAD_NORM = 0.5


class RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, values: List[float]) -> None:
        batch = np.array(values, dtype=np.float64)
        if len(batch) == 0:
            return

        batch_count = len(batch)
        delta = batch.mean() - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        self.var = (
            self.var * self.count
            + batch.var() * batch_count
            + delta ** 2 * self.count * batch_count / total_count
        ) / total_count
        self.count = total_count

    @property
    def std(self) -> float:
        return float(max(np.sqrt(self.var), 1e-8))


@dataclass
class Agent:
    name: str
    real_player_id: int
    reward_method: str
    net: torch.nn.Module
    optimizer: torch.optim.Optimizer
    reward_rms: RunningMeanStd = field(default_factory=RunningMeanStd)
    update_count: int = 0


class DualRewardGridWorldFireGame(GridWorldFireGame):
    """Same environment, but each agent can use a different reward method."""

    def __init__(self, reward_method_p1: str, reward_method_p2: str, **kwargs):
        super().__init__(reward_method="default", **kwargs)
        valid = {"default", "offensive", "defensive"}
        if reward_method_p1 not in valid or reward_method_p2 not in valid:
            raise ValueError(f"reward methods must be one of {sorted(valid)}")
        self.reward_methods = {
            1: reward_method_p1,
            2: reward_method_p2,
        }

    def _calculate_rewards(self, reward_signals):
        calculators = {
            "default": self._calculate_default_rewards,
            "offensive": self._calculate_offensive_rewards,
            "defensive": self._calculate_defensive_rewards,
        }

        rewards_for_p1_method = calculators[self.reward_methods[1]](reward_signals)
        rewards_for_p2_method = calculators[self.reward_methods[2]](reward_signals)

        return {
            1: rewards_for_p1_method[1],
            2: rewards_for_p2_method[2],
        }


def dual_reward_worker(
    remote,
    parent_remote,
    grid_size,
    player_lives,
    max_episode_length,
    seed,
    reward_method_p1,
    reward_method_p2,
):
    parent_remote.close()
    set_seed(seed)
    env = DualRewardGridWorldFireGame(
        grid_size=grid_size,
        player_lives=player_lives,
        max_episode_length=max_episode_length,
        reward_method_p1=reward_method_p1,
        reward_method_p2=reward_method_p2,
        seed=seed,
        validate_maps=not BYPASS_MAP_VALIDATION,
    )
    env.reset()

    while True:
        try:
            command, data = remote.recv()

            if command == "step":
                action_1, action_2 = data
                raw_state, reward_1, reward_2, done, info = env.step(action_1, action_2)

                if done:
                    env.reset()
                    raw_state = env.get_state()

                remote.send((raw_state, reward_1, reward_2, done, info))

            elif command == "reset":
                env.reset()
                remote.send((env.get_state(), env.info()))

            elif command == "close":
                remote.close()
                break

            else:
                raise NotImplementedError(command)

        except EOFError:
            break


class DualRewardVecEnv:
    def __init__(self, num_envs: int, grid_size: int, base_seed: int):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [
            mp.Process(
                target=dual_reward_worker,
                args=(
                    work_remote,
                    remote,
                    grid_size,
                    PLAYER_LIVES,
                    MAX_EPISODE_LENGTH,
                    base_seed + i,
                    AGENT_A_REWARD_METHOD,
                    AGENT_B_REWARD_METHOD,
                ),
            )
            for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes))
        ]

        for process in self.processes:
            process.daemon = True
            process.start()

        for remote in self.work_remotes:
            remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        raw_states, infos = zip(*results)
        return list(raw_states), list(infos)

    def step(self, actions_1, actions_2):
        for remote, action_1, action_2 in zip(self.remotes, actions_1, actions_2):
            remote.send(("step", (int(action_1), int(action_2))))

        results = [remote.recv() for remote in self.remotes]
        raw_states, rewards_1, rewards_2, dones, infos = zip(*results)
        return list(raw_states), list(rewards_1), list(rewards_2), list(dones), list(infos)

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_network() -> torch.nn.Module:
    return PPONetwork(pool_size=POOL_SIZE)


def save_model(net: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    tqdm.write(f"  saved checkpoint -> {path}  ({size_mb:.1f} MB)")


def load_model(net: torch.nn.Module, path: str, optimizer=None) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    size_mb = os.path.getsize(path) / 1024 / 1024
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    opt_info = ""
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        opt_info = " + optimizer state"
    tqdm.write(f"  loaded checkpoint <- {path}  ({size_mb:.1f} MB){opt_info}")


def actor_parameters(net: torch.nn.Module):
    modules = [
        net.conv1,
        net.norm1,
        net.conv2,
        net.norm2,
        net.conv3,
        net.norm3,
        net.fc1_actor,
        net.fc2_actor,
        net.actor,
    ]
    for module in modules:
        yield from module.parameters()


def copy_actor_hidden_layers_to_critic(net: torch.nn.Module) -> None:
    net.fc1_critic.load_state_dict(net.fc1_actor.state_dict())
    net.fc2_critic.load_state_dict(net.fc2_actor.state_dict())


def generate_imitation_data(num_episodes: int, grid_size: int, epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run heuristic agents for num_episodes and return (states, actions) tensors in memory."""
    dataset_states = []
    dataset_actions = []

    agent1 = HeuristicAgent(player_id=1)
    agent2 = HeuristicAgent(player_id=2)

    ep_bar = tqdm(
        range(num_episodes),
        desc=f"  [Imitation e{epoch:02d}] collecting {grid_size}x{grid_size}",
        unit="ep",
        leave=False,
    )
    for ep_idx in ep_bar:
        ep_seed = SEED + epoch * num_episodes + ep_idx
        set_seed(ep_seed)

        env = GridWorldFireGame(
            grid_size=grid_size,
            player_lives=PLAYER_LIVES,
            max_episode_length=MAX_EPISODE_LENGTH,
            validate_maps=not BYPASS_MAP_VALIDATION,
        )
        env.reset()
        done = False

        while not done:
            raw_state = env.get_state()
            info = env.info()

            a1 = agent1.select_action(raw_state, info)
            a2 = agent2.select_action(raw_state, info)

            norm_state = normalize_state(raw_state)
            dataset_states.append(state_for_player(norm_state, player_id=1))
            dataset_actions.append(a1)
            dataset_states.append(state_for_player(norm_state, player_id=2))
            dataset_actions.append(a2)

            _, _, _, done, _ = env.step(a1, a2)

        ep_bar.set_postfix(samples=len(dataset_states))

    states = torch.tensor(np.array(dataset_states), dtype=torch.float32)
    actions = torch.tensor(dataset_actions, dtype=torch.long)
    return states, actions


def write_csv_header(path: str, fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv(path: str, fieldnames: List[str], row: Dict) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def prepare_csv(working_path: str, archive_path: str, fieldnames: List[str], resuming: bool) -> None:
    """
    Set up a working CSV for this run.

    Fresh run (resuming=False): write a new header, overwriting any existing file.
    Resume (resuming=True): copy the groupnorm_logs archive into the working path so new
    rows append onto the full history. The archive is never modified. If no archive
    exists yet, keep an existing working file or start a fresh one so appends work.
    """
    os.makedirs(os.path.dirname(os.path.abspath(working_path)), exist_ok=True)

    if not resuming:
        write_csv_header(working_path, fieldnames)
        return

    if os.path.exists(archive_path):
        shutil.copyfile(archive_path, working_path)
        tqdm.write(f"  resume: continuing CSV from archive {archive_path} -> {working_path}")
    elif not os.path.exists(working_path):
        write_csv_header(working_path, fieldnames)
        tqdm.write(f"  resume: no archive at {archive_path}; started fresh {working_path}")
    else:
        tqdm.write(f"  resume: appending to existing {working_path} (no archive found)")


def last_episode_in_csv(path: str, field: str = "episode") -> int:
    """Largest value in `field` across the CSV (0 if missing/empty). Used to keep
    episode numbering monotonic when continuing an archived log."""
    if not os.path.exists(path):
        return 0
    last = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                last = max(last, int(row[field]))
            except (ValueError, KeyError, TypeError):
                continue
    return last


def _run_imitation_epoch(net, optimizer, device, grid_size, episodes, global_epoch):
    """Generate fresh heuristic data and run one supervised epoch. Returns metrics."""
    states, actions = generate_imitation_data(episodes, grid_size, epoch=global_epoch)

    # Shuffle then split 80/20
    indices = torch.randperm(len(states))
    states = states[indices]
    actions = actions[indices]
    split = int(0.8 * len(states))

    train_states, train_actions = states[:split], actions[:split]
    val_states, val_actions = states[split:], actions[split:]

    net.train()
    train_loss_total = 0.0
    train_correct = 0
    train_total = 0
    train_batches = 0

    num_train_batches = (len(train_states) + IMITATION_BATCH_SIZE - 1) // IMITATION_BATCH_SIZE
    train_bar = tqdm(
        range(0, len(train_states), IMITATION_BATCH_SIZE),
        desc=f"  [Imitation e{global_epoch:02d} {grid_size}x{grid_size}] train",
        total=num_train_batches,
        unit="batch",
        leave=False,
    )
    for start in train_bar:
        batch_states = train_states[start:start + IMITATION_BATCH_SIZE].to(device)
        batch_actions = train_actions[start:start + IMITATION_BATCH_SIZE].to(device)

        logits, _ = net(batch_states)
        loss = F.cross_entropy(logits, batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_total += float(loss.item())
        train_correct += int((logits.argmax(dim=-1) == batch_actions).sum().item())
        train_total += int(batch_actions.numel())
        train_batches += 1

        train_bar.set_postfix(
            loss=f"{train_loss_total / train_batches:.4f}",
            acc=f"{train_correct / max(1, train_total):.3f}",
        )

    net.eval()
    val_loss_total = 0.0
    val_correct = 0
    val_total = 0
    val_batches = 0

    with torch.no_grad():
        for start in range(0, len(val_states), IMITATION_BATCH_SIZE):
            batch_states = val_states[start:start + IMITATION_BATCH_SIZE].to(device)
            batch_actions = val_actions[start:start + IMITATION_BATCH_SIZE].to(device)

            logits, _ = net(batch_states)
            loss = F.cross_entropy(logits, batch_actions)

            val_loss_total += float(loss.item())
            val_correct += int((logits.argmax(dim=-1) == batch_actions).sum().item())
            val_total += int(batch_actions.numel())
            val_batches += 1

    train_loss = train_loss_total / max(1, train_batches)
    val_loss = val_loss_total / max(1, val_batches)
    train_acc = train_correct / max(1, train_total)
    val_acc = val_correct / max(1, val_total)

    append_csv(IMITATION_CSV_PATH, IMITATION_FIELDS, {"epoch": global_epoch, "train_loss": train_loss})
    return train_loss, val_loss, train_acc, val_acc


def train_imitation() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = make_network().to(device)
    optimizer = torch.optim.Adam(actor_parameters(net), lr=IMITATION_LR)

    global_epoch = 0
    for grid_size, max_epochs, episodes in IMITATION_SCHEDULE:
        # best_val_loss resets each phase: 10x10 and 15x15 validation losses are on
        # different data distributions and are not comparable. The same net keeps
        # training across phases, and the best model of each phase is saved (so the
        # final checkpoint is the best 15x15 model, built on top of 10x10 learning).
        best_val_loss = float("inf")
        patience_counter = 0

        phase_bar = tqdm(
            range(1, max_epochs + 1),
            desc=f"[Imitation {grid_size}x{grid_size}] epochs",
            unit="epoch",
        )
        for _ in phase_bar:
            global_epoch += 1
            train_loss, val_loss, train_acc, val_acc = _run_imitation_epoch(
                net, optimizer, device, grid_size, episodes, global_epoch
            )

            improved = val_loss < best_val_loss
            phase_bar.set_postfix(
                tr_loss=f"{train_loss:.4f}",
                tr_acc=f"{train_acc:.3f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.3f}",
                best=f"{best_val_loss:.4f}",
                patience=f"{patience_counter}/{IMITATION_PATIENCE}",
            )

            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                save_model(net, optimizer, PRETRAINED_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= IMITATION_PATIENCE:
                    tqdm.write(f"  {grid_size}x{grid_size} imitation early stopping")
                    break


def create_agent(name: str, real_player_id: int, reward_method: str, device, load_path: str, is_pretrained: bool) -> Agent:
    mode = "imitation pretrained" if is_pretrained else "PPO resume"
    tqdm.write(f"\ninitializing {name} (player {real_player_id}, {mode}, reward={reward_method})")
    net = make_network().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=PPO_LR)

    load_model(net, load_path, optimizer if not is_pretrained else None)

    if is_pretrained:
        copy_actor_hidden_layers_to_critic(net)
        tqdm.write(f"  copied actor hidden layers -> critic for {name}")

    return Agent(
        name=name,
        real_player_id=real_player_id,
        reward_method=reward_method,
        net=net,
        optimizer=optimizer,
    )


def policy_batch(agent: Agent, normalized_states: List[np.ndarray], device):
    was_training = agent.net.training
    agent.net.eval()

    observations = [
        state_for_player(state, agent.real_player_id)
        for state in normalized_states
    ]
    obs_tensor = torch.from_numpy(np.stack(observations)).float().to(device)

    with torch.no_grad():
        logits, values = agent.net(obs_tensor)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

    result = (
        actions.cpu().numpy(),
        log_probs.cpu().numpy(),
        values.squeeze(-1).cpu().numpy(),
        observations,
    )
    if was_training:
        agent.net.train()
    return result


def compute_gae(rewards, values, dones, next_value):
    advantages = torch.zeros(len(rewards), dtype=torch.float32)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - float(dones[t])
        next_val = next_value if t == len(rewards) - 1 else values[t + 1].item()
        delta = float(rewards[t] + GAMMA * next_val * next_nonterminal - values[t].item())
        gae = float(delta + GAMMA * GAE_LAMBDA * next_nonterminal * gae)
        advantages[t] = gae

    return advantages


def new_episode_stats():
    return {
        1: {
            "return": 0.0,
            "boosters": 0,
            "own_fire_steps": 0,
            "enemy_fire_steps": 0,
            "fire_tiles_sum": 0,
        },
        2: {
            "return": 0.0,
            "boosters": 0,
            "own_fire_steps": 0,
            "enemy_fire_steps": 0,
            "fire_tiles_sum": 0,
        },
    }


def log_episode_metrics(
    episode_stats,
    agent_a,
    agent_b,
    global_episode: int,
    grid_size: int,
    episode_length: int,
    winner,
) -> None:
    agents_by_player = {
        getattr(agent_a, "player_id", getattr(agent_a, "real_player_id", 1)): agent_a,
        getattr(agent_b, "player_id", getattr(agent_b, "real_player_id", 2)): agent_b,
    }

    for player in [1, 2]:
        agent = agents_by_player[player]
        draw = winner is None
        win = winner == player
        loss = winner is not None and winner != player
        
        agent_name = getattr(agent, "name", "heuristic_agent")

        append_csv(
            PPO_EPISODE_CSV_PATH,
            PPO_EPISODE_FIELDS,
            {
                "episode": global_episode,
                "grid_size": grid_size,
                "agent_name": agent_name,
                "episodic_return": episode_stats[player]["return"],
                "win": int(win),
                "loss": int(loss),
                "draw": int(draw),
                "boosters_picked": episode_stats[player]["boosters"],
                "avg_fire_trail_length": episode_stats[player]["fire_tiles_sum"] / max(1, episode_length),
                "own_fire_steps": episode_stats[player]["own_fire_steps"],
                "enemy_fire_steps": episode_stats[player]["enemy_fire_steps"],
            },
        )


def update_episode_stats(episode_stats, rewards, info) -> None:
    for player in [1, 2]:
        details = info["last_move_details"][player]
        episode_stats[player]["return"] += rewards[player]
        episode_stats[player]["boosters"] += int(details["picked_booster"])
        episode_stats[player]["own_fire_steps"] += int(details["stepped_on_own_fire"])
        episode_stats[player]["enemy_fire_steps"] += int(details["stepped_on_enemy_fire"])
        episode_stats[player]["fire_tiles_sum"] += info["players"][player]["num_burning_tiles"]


def build_ppo_batch(rollout, active_agent: Agent, raw_states, device):
    flat_rewards = [
        reward
        for env_rewards in rollout["rewards"]
        for reward in env_rewards
    ]
    active_agent.reward_rms.update(flat_rewards)
    reward_std = max(active_agent.reward_rms.std, 1e-8)

    normalized_states = [normalize_state(state) for state in raw_states]
    next_observations = [
        state_for_player(state, active_agent.real_player_id)
        for state in normalized_states
    ]
    next_tensor = torch.from_numpy(np.stack(next_observations)).float().to(device)

    was_training = active_agent.net.training
    active_agent.net.eval()
    with torch.no_grad():
        _, next_values = active_agent.net(next_tensor)
    if was_training:
        active_agent.net.train()
    next_values = next_values.squeeze(-1).cpu().numpy()

    states = []
    actions = []
    old_log_probs = []
    advantages = []
    returns = []

    for env_idx in range(NUM_ENVS):
        if not rollout["rewards"][env_idx]:
            continue

        bootstrap_value = (
            0.0
            if rollout["dones"][env_idx][-1]
            else float(next_values[env_idx])
        )
        normalized_rewards = [
            reward / reward_std
            for reward in rollout["rewards"][env_idx]
        ]
        value_tensors = [
            torch.tensor(value, dtype=torch.float32)
            for value in rollout["values"][env_idx]
        ]
        env_advantages = compute_gae(
            rewards=normalized_rewards,
            values=value_tensors,
            dones=rollout["dones"][env_idx],
            next_value=bootstrap_value,
        )
        env_returns = env_advantages + torch.stack(value_tensors)

        states.extend(rollout["states"][env_idx])
        actions.extend(rollout["actions"][env_idx])
        old_log_probs.extend(rollout["log_probs"][env_idx])
        advantages.append(env_advantages)
        returns.append(env_returns)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
    advantages = torch.cat(advantages).to(device)
    returns = torch.cat(returns).to(device)

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return states, actions, old_log_probs, advantages, returns


def ppo_update(agent: Agent, batch, device) -> Dict[str, float]:
    states, actions, old_log_probs, advantages, returns = batch
    agent.net.train()

    policy_losses = []
    value_losses = []
    entropies = []
    n = states.shape[0]
    value_only = agent.update_count <= WARMUP_UPDATES

    for _ in range(PPO_EPOCHS):
        indices = torch.randperm(n, device=device)

        for start in range(0, n, MINIBATCH_SIZE):
            batch_idx = indices[start:start + MINIBATCH_SIZE]

            logits, values = agent.net(states[batch_idx])
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions[batch_idx])
            entropy = dist.entropy().mean()
            values = values.squeeze(-1)

            ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
            unclipped = ratio * advantages[batch_idx]
            clipped = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages[batch_idx]

            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[batch_idx])

            if value_only:
                loss = VALUE_COEFF * value_loss
            else:
                loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy

            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.net.parameters(), MAX_GRAD_NORM)
            agent.optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

    agent.update_count += 1

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
    }


def _new_rollout_buffer():
    return {
        "states": [[] for _ in range(NUM_ENVS)],
        "actions": [[] for _ in range(NUM_ENVS)],
        "log_probs": [[] for _ in range(NUM_ENVS)],
        "values": [[] for _ in range(NUM_ENVS)],
        "rewards": [[] for _ in range(NUM_ENVS)],
        "dones": [[] for _ in range(NUM_ENVS)],
    }


def collect_rollout(
    env: DualRewardVecEnv,
    raw_states,
    infos,
    episode_stats,
    agent_a,
    agent_b,
    learners: List[Agent],
    grid_size: int,
    global_episode: int,
    target_episode: int,
    device,
):
    # One independent rollout buffer per learning agent, keyed by player id.
    # In self-play this collects trajectories for BOTH agents in a single pass
    # instead of discarding the non-active agent's experience.
    rollouts = {learner.real_player_id: _new_rollout_buffer() for learner in learners}
    completed_episodes = 0
    rollout_iters = max(1, ROLLOUT_STEPS // NUM_ENVS)

    for _ in range(rollout_iters):
        if global_episode >= target_episode:
            break

        normalized_states = [normalize_state(state) for state in raw_states]

        if isinstance(agent_a, Agent):
            actions_1, log_probs_1, values_1, obs_1 = policy_batch(agent_a, normalized_states, device)
        else:
            actions_1 = [agent_a.select_action(raw_states[i], infos[i]) for i in range(NUM_ENVS)]
            log_probs_1 = [0.0] * NUM_ENVS
            values_1 = [0.0] * NUM_ENVS
            obs_1 = [None] * NUM_ENVS

        if isinstance(agent_b, Agent):
            actions_2, log_probs_2, values_2, obs_2 = policy_batch(agent_b, normalized_states, device)
        else:
            # Heuristic agent fallback
            actions_2 = [agent_b.select_action(raw_states[i], infos[i]) for i in range(NUM_ENVS)]
            log_probs_2 = [0.0] * NUM_ENVS
            values_2 = [0.0] * NUM_ENVS
            obs_2 = [None] * NUM_ENVS

        next_raw_states, rewards_1, rewards_2, dones, next_infos = env.step(actions_1, actions_2)

        per_player = {
            1: (obs_1, actions_1, log_probs_1, values_1, rewards_1),
            2: (obs_2, actions_2, log_probs_2, values_2, rewards_2),
        }

        for env_idx in range(NUM_ENVS):
            rewards = {
                1: float(rewards_1[env_idx]),
                2: float(rewards_2[env_idx]),
            }
            update_episode_stats(episode_stats[env_idx], rewards, next_infos[env_idx])

            for player_id, buffer in rollouts.items():
                obs_p, actions_p, log_probs_p, values_p, rewards_p = per_player[player_id]
                buffer["states"][env_idx].append(obs_p[env_idx])
                buffer["actions"][env_idx].append(int(actions_p[env_idx]))
                buffer["log_probs"][env_idx].append(float(log_probs_p[env_idx]))
                buffer["values"][env_idx].append(float(values_p[env_idx]))
                buffer["rewards"][env_idx].append(float(rewards_p[env_idx]))
                buffer["dones"][env_idx].append(float(dones[env_idx]))

            if dones[env_idx]:
                if global_episode < target_episode:
                    global_episode += 1
                    completed_episodes += 1
                    log_episode_metrics(
                        episode_stats=episode_stats[env_idx],
                        agent_a=agent_a,
                        agent_b=agent_b,
                        global_episode=global_episode,
                        grid_size=grid_size,
                        episode_length=next_infos[env_idx]["turn_count"],
                        winner=next_infos[env_idx]["winner"],
                    )
                episode_stats[env_idx] = new_episode_stats()

        raw_states = next_raw_states
        infos = next_infos

    # Build one PPO batch per learner that actually collected steps.
    batches = []
    for learner in learners:
        buffer = rollouts[learner.real_player_id]
        steps = sum(len(env_rewards) for env_rewards in buffer["rewards"])
        if steps == 0:
            continue
        batch = build_ppo_batch(buffer, learner, raw_states, device)
        batches.append((learner, batch))

    return batches, raw_states, infos, episode_stats, global_episode, completed_episodes


def run_ppo_stage(
    agent_a,
    agent_b,
    learners: List[Agent],
    grid_size: int,
    stage_episodes: int,
    first_global_episode: int,
    device,
    stage_desc: str = "",
) -> int:
    env = DualRewardVecEnv(
        num_envs=NUM_ENVS,
        grid_size=grid_size,
        base_seed=SEED + first_global_episode + grid_size,
    )
    raw_states, infos = env.reset()
    episode_stats = [new_episode_stats() for _ in range(NUM_ENVS)]
    global_episode = first_global_episode
    target_episode = first_global_episode + stage_episodes

    ppo_bar = tqdm(
        total=stage_episodes,
        desc=f"[PPO {grid_size}x{grid_size}] {stage_desc}",
        unit="ep",
    )
    try:
        while global_episode < target_episode:
            batches, raw_states, infos, episode_stats, global_episode, completed = collect_rollout(
                env=env,
                raw_states=raw_states,
                infos=infos,
                episode_stats=episode_stats,
                agent_a=agent_a,
                agent_b=agent_b,
                learners=learners,
                grid_size=grid_size,
                global_episode=global_episode,
                target_episode=target_episode,
                device=device,
            )

            if not batches:
                break

            # Update every learning agent on its own trajectories. In self-play
            # this trains agent_a and agent_b in the same rollout.
            postfix = {}
            for learner, batch in batches:
                losses = ppo_update(learner, batch, device)
                postfix[f"{learner.name}_p"] = f"{losses['policy_loss']:.3f}"
                postfix[f"{learner.name}_v"] = f"{losses['value_loss']:.3f}"
                postfix[f"{learner.name}_ent"] = f"{losses['entropy']:.3f}"

            ppo_bar.update(completed)
            ppo_bar.set_postfix(**postfix)

    finally:
        ppo_bar.close()
        env.close()

    return global_episode


def main() -> None:
    set_seed(SEED)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    resume_a = RESUME_TRAINING and os.path.exists(PPO_AGENT_A_SAVE_PATH)
    resume_b = RESUME_TRAINING and os.path.exists(PPO_AGENT_B_SAVE_PATH)
    resuming = resume_a or resume_b
    print(f"resume: agent_a={resume_a}  agent_b={resume_b}")

    # Set up working CSVs: on resume continue from the groupnorm_logs archive (no
    # overwrite); on a fresh run start with a clean header. Done before imitation
    # so train_imitation() can append straight away.
    prepare_csv(IMITATION_CSV_PATH, IMITATION_ARCHIVE_PATH, IMITATION_FIELDS, resuming)
    prepare_csv(PPO_EPISODE_CSV_PATH, PPO_EPISODE_ARCHIVE_PATH, PPO_EPISODE_FIELDS, resuming)

    if RUN_IMITATION_PRETRAINING and not (resume_a and resume_b):
        print("\n=== imitation pretraining ===")
        train_imitation()
    elif not os.path.exists(PRETRAINED_SAVE_PATH) and not (resume_a and resume_b):
        raise FileNotFoundError(f"Missing pretrained checkpoint: {PRETRAINED_SAVE_PATH}")
    else:
        print(f"skipping imitation (resume_a={resume_a}, resume_b={resume_b})")

    print("\n=== initializing PPO agents ===")

    agent_a_ppo = create_agent(
        name="agent_a",
        real_player_id=1,
        reward_method=AGENT_A_REWARD_METHOD,
        device=device,
        load_path=PPO_AGENT_A_SAVE_PATH if resume_a else PRETRAINED_SAVE_PATH,
        is_pretrained=not resume_a,
    )
    agent_b_ppo = create_agent(
        name="agent_b",
        real_player_id=2,
        reward_method=AGENT_B_REWARD_METHOD,
        device=device,
        load_path=PPO_AGENT_B_SAVE_PATH if resume_b else PRETRAINED_SAVE_PATH,
        is_pretrained=not resume_b,
    )
    
    agent_a_heuristic = HeuristicAgent(player_id=1, seed=SEED)
    agent_b_heuristic = HeuristicAgent(player_id=2, seed=SEED)

    # Continue episode numbering past the archived history so appended rows stay
    # monotonic for plotting (0 on a fresh run).
    global_episode = last_episode_in_csv(PPO_EPISODE_CSV_PATH) if resuming else 0
    if resuming:
        print(f"continuing episode numbering from {global_episode}")
    print(f"\n=== PPO curriculum ({len(GRID_SCHEDULE)} stages) ===")

    for grid_size, stage_episodes, mode in GRID_SCHEDULE:
        if mode == "self_play":
            current_a = agent_a_ppo
            current_b = agent_b_ppo
            learners = [agent_a_ppo, agent_b_ppo]
            stage_desc = "self-play"
        elif mode == "train_a_vs_heuristic":
            current_a = agent_a_ppo
            current_b = agent_b_heuristic
            learners = [agent_a_ppo]
            stage_desc = "agent_a(PPO) vs agent_b(heuristic)"
        elif mode == "train_b_vs_heuristic":
            current_a = agent_a_heuristic
            current_b = agent_b_ppo
            learners = [agent_b_ppo]
            stage_desc = "agent_a(heuristic) vs agent_b(PPO)"
        else:
            raise ValueError(f"Unknown curriculum mode: {mode}")

        global_episode = run_ppo_stage(
            agent_a=current_a,
            agent_b=current_b,
            learners=learners,
            grid_size=grid_size,
            stage_episodes=stage_episodes,
            first_global_episode=global_episode,
            device=device,
            stage_desc=stage_desc,
        )

        if SAVE_EVERY_STAGE:
            save_model(agent_a_ppo.net, agent_a_ppo.optimizer, PPO_AGENT_A_SAVE_PATH)
            save_model(agent_b_ppo.net, agent_b_ppo.optimizer, PPO_AGENT_B_SAVE_PATH)
            tqdm.write(f"checkpoints saved — {grid_size}x{grid_size} {mode} complete")

    # Final guaranteed save
    save_model(agent_a_ppo.net, agent_a_ppo.optimizer, PPO_AGENT_A_SAVE_PATH)
    save_model(agent_b_ppo.net, agent_b_ppo.optimizer, PPO_AGENT_B_SAVE_PATH)

    tqdm.write(f"saved {agent_a_ppo.name} to {PPO_AGENT_A_SAVE_PATH}")
    tqdm.write(f"saved {agent_b_ppo.name} to {PPO_AGENT_B_SAVE_PATH}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
