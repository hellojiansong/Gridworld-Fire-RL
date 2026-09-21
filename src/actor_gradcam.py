"""
Actor Grad-CAM viewer for Fire Grid Game PPO checkpoints.

This script creates a combined Grad-CAM report from the 4-channel game state,
showing terrain, entities/fire, per-player actor heatmaps, and action probabilities.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from env import GridWorldFireGame
from tested_networks.better_actor import BetterActorPPONetwork
from tested_networks.grid_full_separate import GridFullSeparatePPONetwork
from tested_networks.territorial_reward import TerritorialRewardPPONetwork
from tested_networks.network_groupnorm import PPONetwork as GroupNormPPONetwork
from training.utils import normalize_state, state_for_player


ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

ARCH_OPTIONS = [
    "grid full separate",
    "better actor",
    "territorial reward",
    "GroupNorm",
]
DEFAULT_GRID_SIZE = 15
DEFAULT_PLAYER_LIVES = 3
DEFAULT_POOL_SIZE = 8
DEFAULT_REWARD_METHOD = "default"
DEFAULT_SEED = 1234
GRADCAM_STEPS = (0, 10, 30, 90)


def build_network(architecture: str, grid_size: int, pool_size: int, device: torch.device):
    if architecture == "grid full separate":
        net = GridFullSeparatePPONetwork(pool_size=pool_size)
        target_layer = net.actor_conv3
    elif architecture == "better actor":
        net = BetterActorPPONetwork(pool_size=pool_size)
        target_layer = net.actor_conv6
    elif architecture == "territorial reward":
        net = TerritorialRewardPPONetwork(pool_size=pool_size)
        target_layer = net.actor_conv3
    elif architecture == "GroupNorm":
        net = GroupNormPPONetwork(pool_size=pool_size)
        # Shared conv trunk (conv1/2/3) then split FC heads; conv3 is the last
        # conv feeding the actor, so it is the Grad-CAM target.
        target_layer = net.conv3
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return net.to(device), target_layer


def load_model(path: str, net: nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.eval()


def choose_option(prompt, options, default=None):
    print(f"\n{prompt}")
    for index, option in enumerate(options, start=1):
        marker = " (default)" if option == default else ""
        print(f"[{index}] {option}{marker}")

    while True:
        choice = input("Choose a number: ").strip()

        if choice == "" and default is not None:
            return default

        try:
            choice_index = int(choice) - 1
        except ValueError:
            print("Please enter a number.")
            continue

        if 0 <= choice_index < len(options):
            return options[choice_index]

        print("Number out of range. Try again.")


def list_model_paths():
    if not os.path.exists(CHECKPOINTS_DIR):
        return []

    model_files = sorted(
        file_name
        for file_name in os.listdir(CHECKPOINTS_DIR)
        if file_name.endswith(".pt")
    )
    return [os.path.join(CHECKPOINTS_DIR, file_name) for file_name in model_files]


def require_model(path):
    if path is None:
        raise ValueError("No model path selected.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find model at {path}")

    return path


def choose_model_path(label):
    model_paths = list_model_paths()

    if not model_paths:
        path = input(f"No .pt files found. Enter model path for {label}: ").strip()
        return require_model(path)

    labels = [os.path.basename(path) for path in model_paths]
    labels.append("Enter custom path")

    selected = choose_option(f"Select model checkpoint for {label}", labels)
    if selected == "Enter custom path":
        path = input(f"Enter model path for {label}: ").strip()
        return require_model(path)

    selected_index = labels.index(selected)
    return require_model(model_paths[selected_index])


def model_action(net, raw_state, player_id: int, device: torch.device, greedy: bool = True):
    norm_state = normalize_state(raw_state)
    model_state = state_for_player(norm_state, player_id)
    tensor = torch.from_numpy(model_state).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits, _ = net(tensor)
        probs = torch.softmax(logits, dim=-1)
        if greedy:
            action = int(logits.argmax(dim=-1).item())
        else:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())
    return action, probs.squeeze(0).detach().cpu().numpy()


def collect_gradcam_states(net_p1, net_p2, device):
    env = GridWorldFireGame(
        grid_size=DEFAULT_GRID_SIZE,
        player_lives=DEFAULT_PLAYER_LIVES,
        reward_method=DEFAULT_REWARD_METHOD,
        seed=DEFAULT_SEED,
    )
    raw_state = env.reset()
    snapshots = [(0, raw_state.copy(), False)]
    target_steps = set(GRADCAM_STEPS)
    final_snapshot = None

    step = 0
    while True:
        step += 1
        action_1, _ = model_action(net_p1, raw_state, 1, device, greedy=True)
        action_2, _ = model_action(net_p2, raw_state, 2, device, greedy=True)
        raw_state, _, _, done, _info = env.step(action_1, action_2)

        if step in target_steps:
            snapshots.append((step, raw_state.copy(), False))

        if done:
            final_snapshot = (step, raw_state.copy(), True)
            break

    snapshots.append(final_snapshot)
    return snapshots


def actor_gradcam(net, target_layer, raw_state, player_id: int, action: Optional[int], device):
    norm_state = normalize_state(raw_state)
    model_state = state_for_player(norm_state, player_id)
    tensor = torch.from_numpy(model_state).unsqueeze(0).float().to(device)

    saved = {}

    def forward_hook(_module, _inputs, output):
        saved["activations"] = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(forward_hook)
    net.zero_grad(set_to_none=True)

    logits, _ = net(tensor)
    if action is None:
        action = int(logits.argmax(dim=-1).item())
    score = logits[0, action]
    score.backward()
    handle.remove()

    activations = saved["activations"]
    gradients = activations.grad
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam,
        size=raw_state.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    cam = cam[0, 0].detach().cpu().numpy()
    cam = cam - cam.min()
    max_value = cam.max()
    if max_value > 1e-8:
        cam = cam / max_value

    action_probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    return cam, action, action_probs


def state_to_board_rgb(raw_state):
    terrain, players, fire, boosters = raw_state
    terrain_norm = np.clip(terrain / 5.0, 0.0, 1.0)

    board = np.zeros((*terrain.shape, 3), dtype=np.float32)
    board[..., 0] = 0.18 + 0.55 * terrain_norm
    board[..., 1] = 0.20 + 0.55 * terrain_norm
    board[..., 2] = 0.18 + 0.45 * terrain_norm

    p1_fire = np.clip(fire, 0, 10) / 10.0
    p2_fire = np.clip(-fire, 0, 10) / 10.0
    board[..., 0] = np.maximum(board[..., 0], 0.95 * p1_fire)
    board[..., 1] *= 1.0 - 0.35 * p1_fire
    board[..., 2] *= 1.0 - 0.55 * p1_fire
    board[..., 2] = np.maximum(board[..., 2], 0.95 * p2_fire)
    board[..., 0] *= 1.0 - 0.45 * p2_fire
    board[..., 1] *= 1.0 - 0.30 * p2_fire

    booster_mask = boosters > 0.5
    board[booster_mask] = np.array([0.1, 0.95, 0.25])

    p1_mask = players > 0
    p2_mask = players < 0
    board[p1_mask] = np.array([1.0, 0.05, 0.05])
    board[p2_mask] = np.array([0.05, 0.2, 1.0])

    return np.clip(board, 0.0, 1.0)


def player_lives_booster_rgb(raw_state):
    players = raw_state[1]
    boosters = raw_state[3]

    panel = np.full((*players.shape, 3), 0.92, dtype=np.float32)
    panel[boosters > 0.5] = np.array([1.0, 0.85, 0.05])
    panel[players > 0] = np.array([1.0, 0.05, 0.05])
    panel[players < 0] = np.array([0.05, 0.2, 1.0])
    return panel


def player_booster_fire_rgb(raw_state):
    players = raw_state[1]
    fire = raw_state[2]
    boosters = raw_state[3]

    panel = np.full((*players.shape, 3), 0.92, dtype=np.float32)

    p1_fire = np.clip(fire, 0, 10) / 10.0
    p2_fire = np.clip(-fire, 0, 10) / 10.0
    panel[..., 0] = np.maximum(panel[..., 0] * (1.0 - 0.65 * p2_fire), 0.95 * p1_fire)
    panel[..., 1] *= 1.0 - 0.45 * np.maximum(p1_fire, p2_fire)
    panel[..., 2] = np.maximum(panel[..., 2] * (1.0 - 0.65 * p1_fire), 0.95 * p2_fire)

    panel[boosters > 0.5] = np.array([1.0, 0.85, 0.05])
    panel[players > 0] = np.array([1.0, 0.05, 0.05])
    panel[players < 0] = np.array([0.05, 0.2, 1.0])
    return np.clip(panel, 0.0, 1.0)


def draw_grid(ax, grid_size):
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(np.arange(grid_size), fontsize=7)
    ax.set_yticklabels(np.arange(grid_size), fontsize=7)
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.35, alpha=0.35)
    ax.tick_params(which="major", length=0)
    ax.tick_params(which="minor", left=False, bottom=False)


def draw_action_probs(ax, probs, action, player_id):
    action_labels = [ACTION_NAMES[i] for i in range(4)]
    colors = ["#888888"] * 4
    colors[action] = "#d62728" if player_id == 1 else "#1f77b4"
    ax.bar(action_labels, probs, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title(f"Player {player_id} action probabilities, selected={ACTION_NAMES[action]}")
    ax.set_ylabel("probability")


def model_display_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def save_visualization(
    raw_state,
    p1_result,
    p2_result,
    step,
    is_final_step,
    output_path,
    p1_model_name,
    p2_model_name,
):
    grid_size = raw_state.shape[-1]
    step_label = f"step={step}"
    if is_final_step:
        step_label += " FINAL"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    terrain_im = axes[0, 0].imshow(raw_state[0], cmap="viridis", vmin=0, vmax=5, interpolation="nearest")
    axes[0, 0].set_title("Terrain elevation")
    draw_grid(axes[0, 0], grid_size)
    fig.colorbar(terrain_im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(player_booster_fire_rgb(raw_state), interpolation="nearest")
    axes[1, 0].set_title("Players / boosters / fire trails")
    draw_grid(axes[1, 0], grid_size)
    axes[1, 0].legend(
        handles=[
            Patch(facecolor=(1.0, 0.05, 0.05), label="Player 1 model"),
            Patch(facecolor=(0.05, 0.2, 1.0), label="Player 2 model"),
            Patch(facecolor=(1.0, 0.85, 0.05), label="Booster"),
            Patch(facecolor=(0.95, 0.0, 0.0), label="Player 1 fire"),
            Patch(facecolor=(0.0, 0.0, 0.95), label="Player 2 fire"),
        ],
        loc="upper right",
        framealpha=0.9,
        fontsize=8,
    )

    p1_heatmap, p1_action, p1_probs = p1_result
    p2_heatmap, p2_action, p2_probs = p2_result

    p1_im = axes[0, 1].imshow(p1_heatmap, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 1].set_title(f"Player 1 Grad-CAM, action={ACTION_NAMES[p1_action]}")
    draw_grid(axes[0, 1], grid_size)
    fig.colorbar(p1_im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    p2_im = axes[1, 1].imshow(p2_heatmap, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 1].set_title(f"Player 2 Grad-CAM, action={ACTION_NAMES[p2_action]}")
    draw_grid(axes[1, 1], grid_size)
    fig.colorbar(p2_im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    draw_action_probs(axes[0, 2], p1_probs, p1_action, 1)
    draw_action_probs(axes[1, 2], p2_probs, p2_action, 2)

    fig.suptitle(
        f"Fire Grid Game actor focus at {step_label}. "
        f"Player 1: {p1_model_name} | Player 2: {p2_model_name}",
        fontsize=14,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Actor Grad-CAM viewer for two Fire Grid Game PPO models.")
    parser.add_argument("--model", type=str, default=None, help="Shared checkpoint path for both players.")
    parser.add_argument("--model-p1", type=str, default=None, help="Checkpoint path for Player 1.")
    parser.add_argument("--model-p2", type=str, default=None, help="Checkpoint path for Player 2.")
    parser.add_argument("--architecture", choices=ARCH_OPTIONS, default=None, help="Shared architecture for both players.")
    parser.add_argument("--architecture-p1", choices=ARCH_OPTIONS, default=None, help="Architecture for Player 1.")
    parser.add_argument("--architecture-p2", choices=ARCH_OPTIONS, default=None, help="Architecture for Player 2.")
    parser.add_argument("--output", type=str, default=None, help="Base output folder. Defaults to ./gradcam.")
    return parser.parse_args()


def fill_missing_args(args):
    args.model_p1 = args.model_p1 or args.model or choose_model_path("Player 1")
    args.model_p2 = args.model_p2 or args.model or choose_model_path("Player 2")

    if args.architecture_p1 is None:
        args.architecture_p1 = args.architecture or choose_option(
            "Select architecture for Player 1 model",
            ARCH_OPTIONS,
            default="grid full separate",
        )

    if args.architecture_p2 is None:
        args.architecture_p2 = args.architecture or choose_option(
            "Select architecture for Player 2 model",
            ARCH_OPTIONS,
            default="grid full separate",
        )

    args.model_p1 = require_model(args.model_p1)
    args.model_p2 = require_model(args.model_p2)
    if args.output is None:
        args.output = os.path.join(ROOT_DIR, "gradcam")
    return args


def safe_name(value):
    safe_chars = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "model"


def create_run_output_dir(base_output_dir, model_p1_path, model_p2_path):
    model_p1_name = os.path.splitext(os.path.basename(model_p1_path))[0]
    model_p2_name = os.path.splitext(os.path.basename(model_p2_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        base_output_dir,
        f"{safe_name(model_p1_name)}_vs_{safe_name(model_p2_name)}_{timestamp}",
    )
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def output_path_for_step(run_dir, step, is_final_step=False):
    suffix = f"final_step{step}" if is_final_step else f"step{step}"
    return os.path.join(run_dir, f"gradcam_actor_{suffix}.png")


def main():
    args = fill_missing_args(parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_p1, target_layer_p1 = build_network(args.architecture_p1, DEFAULT_GRID_SIZE, DEFAULT_POOL_SIZE, device)
    net_p2, target_layer_p2 = build_network(args.architecture_p2, DEFAULT_GRID_SIZE, DEFAULT_POOL_SIZE, device)
    load_model(args.model_p1, net_p1, device)
    load_model(args.model_p2, net_p2, device)
    run_dir = create_run_output_dir(args.output, args.model_p1, args.model_p2)
    p1_model_name = model_display_name(args.model_p1)
    p2_model_name = model_display_name(args.model_p2)
    print(f"Grad-CAM run folder: {run_dir}")
    print(f"Player 1 model: {args.model_p1}")
    print(f"Player 1 architecture: {args.architecture_p1}")
    print(f"Player 2 model: {args.model_p2}")
    print(f"Player 2 architecture: {args.architecture_p2}")

    snapshots = collect_gradcam_states(net_p1, net_p2, device)

    produced_steps = []
    final_step = None
    for step, raw_state, is_final_step in snapshots:
        p1_result = actor_gradcam(net_p1, target_layer_p1, raw_state, 1, None, device)
        p2_result = actor_gradcam(net_p2, target_layer_p2, raw_state, 2, None, device)
        step_output = output_path_for_step(run_dir, step, is_final_step)
        save_visualization(
            raw_state,
            p1_result,
            p2_result,
            step,
            is_final_step,
            step_output,
            p1_model_name,
            p2_model_name,
        )
        label = f"final step {step}" if is_final_step else f"step {step}"
        print(f"Saved combined actor Grad-CAM {label} to {step_output}")
        for player_id, result in ((1, p1_result), (2, p2_result)):
            _heatmap, action, probs = result
            print(f"  Player {player_id} explained action: {action} ({ACTION_NAMES[action]})")
            print(f"  Player {player_id} action probabilities: { {ACTION_NAMES[i]: float(probs[i]) for i in range(4)} }")

        if is_final_step:
            final_step = step
        else:
            produced_steps.append(step)

    missing_steps = [step for step in GRADCAM_STEPS if step not in produced_steps]
    if missing_steps:
        print(f"Skipped steps not reached before episode ended: {missing_steps}")
    print(f"Final episode step: {final_step}")


if __name__ == "__main__":
    main()
