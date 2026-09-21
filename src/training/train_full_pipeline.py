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
import glob
import multiprocessing as mp
import os
import random
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TRAINING_DIR = os.path.join(SRC_DIR, "training")

for path in [SRC_DIR, TRAINING_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from env import GridWorldFireGame
from heuristic_agent import HeuristicAgent
from tested_networks.grid_full_separate import GridFullSeparatePPONetwork as PPONetwork
from utils import normalize_state, state_for_player


# ---- Files ----

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Base imitation data path. Numbered chunks such as imitation_data_1.pt are also loaded.
IMITATION_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "imitation_data.pt")

# Imitation checkpoint used to initialize both PPO agents.
PRETRAINED_SAVE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", f"pretrained_pipeline_{RUN_TIMESTAMP}.pt")

# Final PPO checkpoints for both agents.
PPO_AGENT_A_SAVE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", f"ppo_agent_a_{RUN_TIMESTAMP}.pt")
PPO_AGENT_B_SAVE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", f"ppo_agent_b_{RUN_TIMESTAMP}.pt")

# CSV logs for plotting.
LOG_DIR = os.path.join(PROJECT_ROOT, "training_logs")
IMITATION_CSV_PATH = os.path.join(LOG_DIR, f"imitation_training_{RUN_TIMESTAMP}.csv")
PPO_EPISODE_CSV_PATH = os.path.join(LOG_DIR, f"ppo_episode_metrics_{RUN_TIMESTAMP}.csv")
PPO_UPDATE_CSV_PATH = os.path.join(LOG_DIR, f"ppo_update_metrics_{RUN_TIMESTAMP}.csv")

IMITATION_FIELDS = ["epoch", "train_loss"]

PPO_UPDATE_FIELDS = [
    "episode",
    "grid_size",
    "updated_agent",
    "policy_loss",
    "value_loss",
    "entropy",
    "explained_variance",
    "avg_return_agent_a",
    "avg_return_agent_b",
    "completed_episodes",
]

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


# ---- Data collection settings ----

# Set False if imitation data already exists at IMITATION_DATA_PATH.
RUN_DATA_COLLECTION = True

DATA_COLLECTION_EPISODES = 1000
DATA_COLLECTION_GRID_SIZE = 10
DATA_COLLECTION_PLAYER_LIVES = 3
DATA_COLLECTION_WORKERS = 0       # 0 = auto (cpu_count - 1)
DATA_COLLECTION_FLUSH_SAMPLES = 250_000


# ---- Imitation learning settings ----

# Set False if you already have PRETRAINED_SAVE_PATH and want to skip imitation.
RUN_IMITATION_PRETRAINING = True

IMITATION_EPOCHS = 30
IMITATION_BATCH_SIZE = 64
IMITATION_LR = 1e-3
IMITATION_PATIENCE = 5


# ---- PPO settings ----

PLAYER_LIVES = 3
MAX_EPISODE_LENGTH = 300
POOL_SIZE = 8
SEED = 42

# Agent A controls real player 1. Agent B controls real player 2.
AGENT_A_REWARD_METHOD = "offensive"
AGENT_B_REWARD_METHOD = "defensive"

# Total PPO curriculum: 2000 episodes on 10x10, then 3000 on 15x15.
GRID_SCHEDULE = [
    (10, 2000),
    (15, 3000),
]

# Alternate which agent is updated every N PPO rollouts/updates.
SWITCH_EVERY_ROLLOUTS = 8

# Original PPO-style rollout/update settings.
NUM_ENVS = 12
ROLLOUT_STEPS = 4096
WARMUP_UPDATES = 10

PPO_LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
PPO_EPOCHS = 4
MINIBATCH_SIZE = 128
VALUE_COEFF = 0.25
ENTROPY_COEFF = 0.01
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
                remote.send(env.get_state())

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
        return [remote.recv() for remote in self.remotes]

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


def load_model(net: torch.nn.Module, path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    net.load_state_dict(state_dict)


def actor_parameters(net: torch.nn.Module):
    modules = [
        net.conv1,
        net.batch_norm1,
        net.conv2,
        net.batch_norm2,
        net.conv3,
        net.batch_norm3,
        net.fc1_actor,
        net.fc2_actor,
        net.actor,
    ]
    for module in modules:
        yield from module.parameters()


def copy_actor_hidden_layers_to_critic(net: torch.nn.Module) -> None:
    net.fc1_critic.load_state_dict(net.fc1_actor.state_dict())
    net.fc2_critic.load_state_dict(net.fc2_actor.state_dict())


def find_imitation_chunks(data_path: str) -> List[str]:
    base_dir = os.path.dirname(data_path)
    base_name = os.path.basename(data_path).replace(".pt", "")
    chunk_paths = sorted(glob.glob(os.path.join(base_dir, f"{base_name}_*.pt")))

    if chunk_paths:
        return chunk_paths

    if os.path.exists(data_path):
        return [data_path]

    raise FileNotFoundError(f"No imitation data found for {data_path}")


def load_imitation_chunk(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, weights_only=False)

    if isinstance(data, dict) and "states" in data and "actions" in data:
        states = data["states"].float()
        actions = data["actions"].long()
    else:
        states = torch.stack([
            torch.from_numpy(s) if not isinstance(s, torch.Tensor) else s
            for s, _ in data
        ]).float()
        actions = torch.tensor([a for _, a in data], dtype=torch.long)

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


def _dc_initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _collect_one_episode(args):
    ep_idx, grid_size, player_lives, base_seed = args
    set_seed(base_seed + ep_idx)

    dataset = []
    env = GridWorldFireGame(grid_size=grid_size, player_lives=player_lives)
    agent1 = HeuristicAgent(player_id=1)
    agent2 = HeuristicAgent(player_id=2)

    env.reset()
    done = False

    while not done:
        raw_state = env.get_state()
        info = env.info()

        a1 = agent1.select_action(raw_state, info)
        a2 = agent2.select_action(raw_state, info)

        norm_state = normalize_state(raw_state)
        dataset.append((state_for_player(norm_state, player_id=1), a1))
        dataset.append((state_for_player(norm_state, player_id=2), a2))

        _, _, _, done, _ = env.step(a1, a2)

    return dataset


def _save_imitation_chunk(samples, save_path, chunk_idx):
    chunk_path = save_path.replace(".pt", f"_{chunk_idx}.pt")
    os.makedirs(os.path.dirname(os.path.abspath(chunk_path)), exist_ok=True)

    states = np.array([s for s, _ in samples], dtype=np.float32)
    actions = np.array([a for _, a in samples], dtype=np.int64)

    torch.save(
        {"states": torch.from_numpy(states), "actions": torch.from_numpy(actions)},
        chunk_path,
    )
    print(f"saved chunk {chunk_idx} ({len(samples)} samples) to {chunk_path}")
    return chunk_path


def collect_imitation_data() -> None:
    num_cores = DATA_COLLECTION_WORKERS
    if num_cores <= 0:
        num_cores = max(1, mp.cpu_count() - 1)

    print(
        f"collecting {DATA_COLLECTION_EPISODES} episodes "
        f"({DATA_COLLECTION_GRID_SIZE}x{DATA_COLLECTION_GRID_SIZE}, "
        f"{DATA_COLLECTION_PLAYER_LIVES} lives) with {num_cores} workers..."
    )

    os.makedirs(os.path.dirname(os.path.abspath(IMITATION_DATA_PATH)), exist_ok=True)

    args_list = [
        (i, DATA_COLLECTION_GRID_SIZE, DATA_COLLECTION_PLAYER_LIVES, SEED)
        for i in range(DATA_COLLECTION_EPISODES)
    ]

    dataset = []
    chunk_counter = 0
    saved_paths = []

    pool = mp.Pool(processes=num_cores, initializer=_dc_initializer)
    try:
        for i, ep_data in enumerate(pool.imap_unordered(_collect_one_episode, args_list)):
            dataset.extend(ep_data)

            if (i + 1) % 10 == 0:
                print(f"  collected {i + 1}/{DATA_COLLECTION_EPISODES} episodes | buffered: {len(dataset)}")

            if len(dataset) >= DATA_COLLECTION_FLUSH_SAMPLES:
                chunk_counter += 1
                saved_paths.append(_save_imitation_chunk(dataset, IMITATION_DATA_PATH, chunk_counter))
                dataset.clear()

        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("interrupted — saving buffered data...")
        pool.terminate()
        pool.join()
    finally:
        if dataset:
            chunk_counter += 1
            saved_paths.append(_save_imitation_chunk(dataset, IMITATION_DATA_PATH, chunk_counter))
            dataset.clear()

    if chunk_counter == 0:
        print("no data collected")
    elif chunk_counter == 1 and saved_paths[0] != IMITATION_DATA_PATH:
        os.replace(saved_paths[0], IMITATION_DATA_PATH)
        print(f"imitation data saved to {IMITATION_DATA_PATH}")
    else:
        print(f"imitation data saved as {chunk_counter} chunks (base: {IMITATION_DATA_PATH})")


def train_imitation() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_paths = find_imitation_chunks(IMITATION_DATA_PATH)

    net = make_network().to(device)
    optimizer = torch.optim.Adam(actor_parameters(net), lr=IMITATION_LR)

    write_csv_header(IMITATION_CSV_PATH, IMITATION_FIELDS)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, IMITATION_EPOCHS + 1):
        net.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0
        train_batches = 0

        random.shuffle(chunk_paths)

        for chunk_path in chunk_paths:
            states, actions = load_imitation_chunk(chunk_path)
            split = int(0.8 * len(states))

            states = states[:split]
            actions = actions[:split]

            indices = torch.randperm(len(states))
            states = states[indices]
            actions = actions[indices]

            for start in range(0, len(states), IMITATION_BATCH_SIZE):
                batch_states = states[start:start + IMITATION_BATCH_SIZE].to(device)
                batch_actions = actions[start:start + IMITATION_BATCH_SIZE].to(device)

                logits, _ = net(batch_states)
                loss = F.cross_entropy(logits, batch_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_total += float(loss.item())
                train_correct += int((logits.argmax(dim=-1) == batch_actions).sum().item())
                train_total += int(batch_actions.numel())
                train_batches += 1

        net.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0

        with torch.no_grad():
            for chunk_path in chunk_paths:
                states, actions = load_imitation_chunk(chunk_path)
                split = int(0.8 * len(states))

                states = states[split:]
                actions = actions[split:]

                for start in range(0, len(states), IMITATION_BATCH_SIZE):
                    batch_states = states[start:start + IMITATION_BATCH_SIZE].to(device)
                    batch_actions = actions[start:start + IMITATION_BATCH_SIZE].to(device)

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

        append_csv(
            IMITATION_CSV_PATH,
            IMITATION_FIELDS,
            {
                "epoch": epoch,
                "train_loss": train_loss,
            },
        )

        print(
            f"imitation epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(net, optimizer, PRETRAINED_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= IMITATION_PATIENCE:
                print("imitation early stopping")
                break


def create_agent(name: str, real_player_id: int, reward_method: str, device) -> Agent:
    net = make_network().to(device)
    load_model(net, PRETRAINED_SAVE_PATH)
    copy_actor_hidden_layers_to_critic(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=PPO_LR)

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


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    target_var = torch.var(targets, unbiased=False)
    if target_var.item() < 1e-8:
        return float("nan")
    error_var = torch.var(targets - predictions, unbiased=False)
    return float((1.0 - error_var / target_var).item())


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
    agent_a: Agent,
    agent_b: Agent,
    global_episode: int,
    grid_size: int,
    episode_length: int,
    winner,
) -> None:
    agents_by_player = {
        agent_a.real_player_id: agent_a,
        agent_b.real_player_id: agent_b,
    }

    for player in [1, 2]:
        agent = agents_by_player[player]
        draw = winner is None
        win = winner == player
        loss = winner is not None and winner != player

        append_csv(
            PPO_EPISODE_CSV_PATH,
            PPO_EPISODE_FIELDS,
            {
                "episode": global_episode,
                "grid_size": grid_size,
                "agent_name": agent.name,
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
    reward_std = active_agent.reward_rms.std

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
    value_only = agent.update_count < WARMUP_UPDATES

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

    agent.net.eval()
    with torch.no_grad():
        _, final_values = agent.net(states)
        final_values = final_values.squeeze(-1)
        value_explained_variance = explained_variance(final_values, returns)
    agent.net.train()

    agent.update_count += 1

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "explained_variance": value_explained_variance,
    }


def collect_rollout(
    env: DualRewardVecEnv,
    raw_states,
    episode_stats,
    agent_a: Agent,
    agent_b: Agent,
    active_agent: Agent,
    grid_size: int,
    global_episode: int,
    target_episode: int,
    device,
):
    rollout = {
        "states": [[] for _ in range(NUM_ENVS)],
        "actions": [[] for _ in range(NUM_ENVS)],
        "log_probs": [[] for _ in range(NUM_ENVS)],
        "values": [[] for _ in range(NUM_ENVS)],
        "rewards": [[] for _ in range(NUM_ENVS)],
        "dones": [[] for _ in range(NUM_ENVS)],
    }
    completed_episodes = 0
    completed_returns = {
        agent_a.name: [],
        agent_b.name: [],
    }
    rollout_iters = max(1, ROLLOUT_STEPS // NUM_ENVS)

    for _ in range(rollout_iters):
        if global_episode >= target_episode:
            break

        normalized_states = [normalize_state(state) for state in raw_states]
        actions_1, log_probs_1, values_1, obs_1 = policy_batch(agent_a, normalized_states, device)
        actions_2, log_probs_2, values_2, obs_2 = policy_batch(agent_b, normalized_states, device)

        next_raw_states, rewards_1, rewards_2, dones, infos = env.step(actions_1, actions_2)

        if active_agent.real_player_id == 1:
            active_obs = obs_1
            active_actions = actions_1
            active_log_probs = log_probs_1
            active_values = values_1
            active_rewards = rewards_1
        else:
            active_obs = obs_2
            active_actions = actions_2
            active_log_probs = log_probs_2
            active_values = values_2
            active_rewards = rewards_2

        for env_idx in range(NUM_ENVS):
            rewards = {
                1: float(rewards_1[env_idx]),
                2: float(rewards_2[env_idx]),
            }
            update_episode_stats(episode_stats[env_idx], rewards, infos[env_idx])

            rollout["states"][env_idx].append(active_obs[env_idx])
            rollout["actions"][env_idx].append(int(active_actions[env_idx]))
            rollout["log_probs"][env_idx].append(float(active_log_probs[env_idx]))
            rollout["values"][env_idx].append(float(active_values[env_idx]))
            rollout["rewards"][env_idx].append(float(active_rewards[env_idx]))
            rollout["dones"][env_idx].append(float(dones[env_idx]))

            if dones[env_idx]:
                if global_episode < target_episode:
                    global_episode += 1
                    completed_episodes += 1
                    completed_returns[agent_a.name].append(
                        episode_stats[env_idx][agent_a.real_player_id]["return"]
                    )
                    completed_returns[agent_b.name].append(
                        episode_stats[env_idx][agent_b.real_player_id]["return"]
                    )
                    log_episode_metrics(
                        episode_stats=episode_stats[env_idx],
                        agent_a=agent_a,
                        agent_b=agent_b,
                        global_episode=global_episode,
                        grid_size=grid_size,
                        episode_length=infos[env_idx]["turn_count"],
                        winner=infos[env_idx]["winner"],
                    )
                episode_stats[env_idx] = new_episode_stats()

        raw_states = next_raw_states

    avg_completed_returns = {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in completed_returns.items()
    }

    collected_steps = sum(len(env_rewards) for env_rewards in rollout["rewards"])
    if collected_steps == 0:
        return None, raw_states, episode_stats, global_episode, completed_episodes, avg_completed_returns

    batch = build_ppo_batch(rollout, active_agent, raw_states, device)
    return batch, raw_states, episode_stats, global_episode, completed_episodes, avg_completed_returns


def run_ppo_stage(
    agent_a: Agent,
    agent_b: Agent,
    grid_size: int,
    stage_episodes: int,
    first_global_episode: int,
    device,
) -> int:
    print(f"starting PPO stage on {grid_size}x{grid_size} with {NUM_ENVS} envs")
    env = DualRewardVecEnv(
        num_envs=NUM_ENVS,
        grid_size=grid_size,
        base_seed=SEED + first_global_episode + grid_size,
    )
    raw_states = env.reset()
    episode_stats = [new_episode_stats() for _ in range(NUM_ENVS)]
    active_agent = agent_a
    global_episode = first_global_episode
    target_episode = first_global_episode + stage_episodes
    rollouts_since_switch = 0

    try:
        while global_episode < target_episode:
            batch, raw_states, episode_stats, global_episode, completed, avg_returns = collect_rollout(
                env=env,
                raw_states=raw_states,
                episode_stats=episode_stats,
                agent_a=agent_a,
                agent_b=agent_b,
                active_agent=active_agent,
                grid_size=grid_size,
                global_episode=global_episode,
                target_episode=target_episode,
                device=device,
            )

            if batch is None:
                break

            losses = ppo_update(active_agent, batch, device)
            rollouts_since_switch += 1

            append_csv(
                PPO_UPDATE_CSV_PATH,
                PPO_UPDATE_FIELDS,
                {
                    "episode": global_episode,
                    "grid_size": grid_size,
                    "updated_agent": active_agent.name,
                    "policy_loss": losses["policy_loss"],
                    "value_loss": losses["value_loss"],
                    "entropy": losses["entropy"],
                    "explained_variance": losses["explained_variance"],
                    "avg_return_agent_a": avg_returns[agent_a.name],
                    "avg_return_agent_b": avg_returns[agent_b.name],
                    "completed_episodes": completed,
                },
            )

            print(
                f"episode {global_episode:04d} | grid={grid_size} | "
                f"updated={active_agent.name} | "
                f"policy_loss={losses['policy_loss']:.3f} "
                f"value_loss={losses['value_loss']:.3f} "
                f"entropy={losses['entropy']:.3f} "
                f"avg_return_{agent_a.name}={avg_returns[agent_a.name]:.2f} "
                f"avg_return_{agent_b.name}={avg_returns[agent_b.name]:.2f}"
            )

            if rollouts_since_switch >= SWITCH_EVERY_ROLLOUTS:
                active_agent = agent_b if active_agent is agent_a else agent_a
                rollouts_since_switch = 0

    finally:
        env.close()

    return global_episode


def main() -> None:
    set_seed(SEED)
    os.makedirs(LOG_DIR, exist_ok=True)

    if RUN_DATA_COLLECTION:
        collect_imitation_data()
    elif not any([
        os.path.exists(IMITATION_DATA_PATH),
        glob.glob(IMITATION_DATA_PATH.replace(".pt", "_*.pt")),
    ]):
        raise FileNotFoundError(f"No imitation data found at {IMITATION_DATA_PATH}. Set RUN_DATA_COLLECTION=True.")

    if RUN_IMITATION_PRETRAINING:
        train_imitation()
    elif not os.path.exists(PRETRAINED_SAVE_PATH):
        raise FileNotFoundError(f"Missing pretrained checkpoint: {PRETRAINED_SAVE_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_a = create_agent(
        name="agent_a",
        real_player_id=1,
        reward_method=AGENT_A_REWARD_METHOD,
        device=device,
    )
    agent_b = create_agent(
        name="agent_b",
        real_player_id=2,
        reward_method=AGENT_B_REWARD_METHOD,
        device=device,
    )

    write_csv_header(PPO_EPISODE_CSV_PATH, PPO_EPISODE_FIELDS)
    write_csv_header(PPO_UPDATE_CSV_PATH, PPO_UPDATE_FIELDS)

    global_episode = 0

    for grid_size, stage_episodes in GRID_SCHEDULE:
        global_episode = run_ppo_stage(
            agent_a=agent_a,
            agent_b=agent_b,
            grid_size=grid_size,
            stage_episodes=stage_episodes,
            first_global_episode=global_episode,
            device=device,
        )

    save_model(agent_a.net, agent_a.optimizer, PPO_AGENT_A_SAVE_PATH)
    save_model(agent_b.net, agent_b.optimizer, PPO_AGENT_B_SAVE_PATH)

    print(f"saved {agent_a.name} to {PPO_AGENT_A_SAVE_PATH}")
    print(f"saved {agent_b.name} to {PPO_AGENT_B_SAVE_PATH}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
