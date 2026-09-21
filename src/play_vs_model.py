import argparse
import os

import torch
from torch.distributions import Categorical

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")

MODE_OPTIONS = ["human-vs-model", "human-vs-heuristic", "model-vs-heuristic", "model-vs-model"]
MIN_GRID_SIZE = 10
MAX_GRID_SIZE = 50
PLAYER_LIVES_OPTIONS = [1, 2, 3]
YES_NO_OPTIONS = ["yes", "no"]
REWARD_METHOD = "default"
ARCH_OPTIONS = ["grid full separate", "better actor", "territorial reward", "GroupNorm"]
POOL_SIZE = 8

from env import GridWorldFireGame
from heuristic_agent import HeuristicAgent
from tested_networks.better_actor import BetterActorPPONetwork
from tested_networks.grid_full_separate import GridFullSeparatePPONetwork
from tested_networks.territorial_reward import TerritorialRewardPPONetwork
from tested_networks.network_groupnorm import PPONetwork as GroupNormPPONetwork
import visualization
from visualization import FireGameUI
from training.utils import normalize_state, state_for_player, state_to_tensor, load_checkpoint


class NetworkAgent:
    """A wrapper for a trained PPO policy model."""

    def __init__(
        self,
        model_path: str,
        player_id: int,
        greedy: bool = True,
        grid_size: int = 10,
        architecture: str = "grid full separate",
    ):
        self.player_id = player_id
        self.greedy = greedy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if architecture == "grid full separate":
            self.net = GridFullSeparatePPONetwork(pool_size=POOL_SIZE).to(self.device)
        elif architecture == "better actor":
            self.net = BetterActorPPONetwork(pool_size=POOL_SIZE).to(self.device)
        elif architecture == "territorial reward":
            self.net = TerritorialRewardPPONetwork(pool_size=POOL_SIZE).to(self.device)
        elif architecture == "GroupNorm":
            self.net = GroupNormPPONetwork(pool_size=POOL_SIZE).to(self.device)
        else:
            raise ValueError(f"architecture must be one of {ARCH_OPTIONS}.")

        load_checkpoint(model_path, self.net)
        self.net.eval()

    def select_action(self, state, info):
        norm_state = normalize_state(state)
        model_state = state_for_player(norm_state, self.player_id)
        tensor_state = state_to_tensor(model_state, self.device)

        with torch.no_grad():
            logits, _ = self.net(tensor_state)
            if self.greedy:
                return int(logits.argmax(dim=-1).item())
            return int(Categorical(logits=logits).sample().item())


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


def choose_int(prompt, minimum, maximum, default=None):
    default_text = f" (default {default})" if default is not None else ""

    while True:
        choice = input(f"\n{prompt} [{minimum}-{maximum}]{default_text}: ").strip()

        if choice == "" and default is not None:
            return default

        try:
            value = int(choice)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if minimum <= value <= maximum:
            return value

        print(f"Value must be between {minimum} and {maximum}.")


def list_model_paths():
    if not os.path.exists(CHECKPOINTS_DIR):
        return []

    model_files = sorted(
        file_name
        for file_name in os.listdir(CHECKPOINTS_DIR)
        if file_name.endswith(".pt")
    )
    return [os.path.join(CHECKPOINTS_DIR, file_name) for file_name in model_files]


def choose_model_path(label):
    model_paths = list_model_paths()

    if not model_paths:
        path = input(f"No .pt files found. Enter path for {label}: ").strip()
        return require_model(path, label)

    labels = [os.path.basename(path) for path in model_paths]
    labels.append("Enter custom path")

    selected = choose_option(f"Select model for {label}", labels)
    if selected == "Enter custom path":
        path = input(f"Enter path for {label}: ").strip()
        return require_model(path, label)

    selected_index = labels.index(selected)
    return require_model(model_paths[selected_index], label)


def require_model(path, label):
    if path is None:
        raise ValueError(f"No model path selected for {label}.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {label} model at {path}")

    return path


def yes_no_to_bool(value):
    return value == "yes"


def parse_args():
    parser = argparse.ArgumentParser(description="Play or watch trained PPO models.")
    parser.add_argument("--mode", choices=MODE_OPTIONS, default=None)
    parser.add_argument("--model", type=str, default=None, help="Shared model path.")
    parser.add_argument("--model-p1", type=str, default=None, help="Model path for player 1.")
    parser.add_argument("--model-p2", type=str, default=None, help="Model path for player 2.")
    parser.add_argument("--grid-size", type=int, default=None, help="Grid size from 10 to 50.")
    parser.add_argument("--player-lives", type=int, choices=PLAYER_LIVES_OPTIONS, default=None)
    parser.add_argument(
        "--architecture",
        choices=ARCH_OPTIONS,
        default=None,
        help=(
            "Model architecture: grid full separate=separate actor/critic trunk checkpoints, "
            "better actor=better_actor.ipynb checkpoints."
        ),
    )
    parser.add_argument("--architecture-p1", choices=ARCH_OPTIONS, default=None)
    parser.add_argument("--architecture-p2", choices=ARCH_OPTIONS, default=None)
    parser.add_argument("--debug", choices=YES_NO_OPTIONS, default=None)
    parser.add_argument("--greedy", choices=YES_NO_OPTIONS, default=None)
    parser.add_argument("--turn-mode", choices=["automatic", "manual"], default=None)
    return parser.parse_args()


def fill_missing_args(args):
    if args.mode is None:
        args.mode = choose_option("Select play mode", MODE_OPTIONS, default="human-vs-model")

    if args.grid_size is not None and not (MIN_GRID_SIZE <= args.grid_size <= MAX_GRID_SIZE):
        raise ValueError(f"--grid-size must be between {MIN_GRID_SIZE} and {MAX_GRID_SIZE}.")

    if args.grid_size is None:
        args.grid_size = choose_int("Enter grid size", MIN_GRID_SIZE, MAX_GRID_SIZE, default=10)

    if args.player_lives is None:
        args.player_lives = choose_option("Select player lives", PLAYER_LIVES_OPTIONS, default=3)

    if args.debug is None:
        args.debug = choose_option("Enable env debug output?", YES_NO_OPTIONS, default="no")

    uses_model = args.mode in {"human-vs-model", "model-vs-heuristic", "model-vs-model"}

    if uses_model and args.mode != "model-vs-model" and args.architecture is None:
        args.architecture = choose_option("Select model architecture", ARCH_OPTIONS, default="grid full separate")
    elif args.mode == "model-vs-model":
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
    elif not uses_model:
        args.architecture = "grid full separate"

    if uses_model and args.greedy is None:
        args.greedy = choose_option("Use greedy model actions?", YES_NO_OPTIONS, default="yes")
    elif not uses_model:
        args.greedy = "yes"

    if args.mode == "human-vs-model":
        args.model_p2 = args.model_p2 or args.model or choose_model_path("Player 2")

    elif args.mode == "human-vs-heuristic":
        pass

    elif args.mode == "model-vs-heuristic":
        args.model_p1 = args.model_p1 or args.model or choose_model_path("Player 1")

    elif args.mode == "model-vs-model":
        args.model_p1 = args.model_p1 or args.model or choose_model_path("Player 1")
        args.model_p2 = args.model_p2 or args.model or choose_model_path("Player 2")

    if args.turn_mode is None:
        args.turn_mode = choose_option("Turn mode", ["automatic", "manual"], default="automatic")

    args.debug = yes_no_to_bool(args.debug)
    args.greedy = yes_no_to_bool(args.greedy)
    return args


def patch_visualization(grid_size):
    visualization.GRID_SIZE = grid_size
    visualization.SCREEN_WIDTH = 1000
    visualization.SCREEN_HEIGHT = 650
    visualization.ARENA_WIDTH = visualization.SCREEN_WIDTH - visualization.HUD_WIDTH
    visualization.TILE_SIZE = max(
        4,
        int(
            (visualization.ARENA_WIDTH - (visualization.SAFE_MARGIN * 2))
            / (grid_size - 1)
            / 2
        ),
    )
    visualization.ISO_OFFSET_X = visualization.ARENA_WIDTH // 2
    visualization.ISO_OFFSET_Y = 150 + (grid_size * 2)


def build_agents(args):
    if args.mode == "human-vs-model":
        model_path = require_model(args.model_p2, "Player 2")
        print(f"Loading Player 2 PPO model from {model_path}...")
        return None, NetworkAgent(
            model_path=model_path,
            player_id=2,
            greedy=args.greedy,
            grid_size=args.grid_size,
            architecture=args.architecture,
        )

    if args.mode == "human-vs-heuristic":
        return None, HeuristicAgent(player_id=2)

    if args.mode == "model-vs-heuristic":
        model_path = require_model(args.model_p1, "Player 1")
        print(f"Loading Player 1 PPO model from {model_path}...")
        return (
            NetworkAgent(
                model_path=model_path,
                player_id=1,
                greedy=args.greedy,
                grid_size=args.grid_size,
                architecture=args.architecture,
            ),
            HeuristicAgent(player_id=2),
        )

    if args.mode == "model-vs-model":
        model_p1_path = require_model(args.model_p1, "Player 1")
        model_p2_path = require_model(args.model_p2, "Player 2")
        print(f"Loading Player 1 PPO model from {model_p1_path}...")
        print(f"Loading Player 2 PPO model from {model_p2_path}...")
        return (
            NetworkAgent(
                model_path=model_p1_path,
                player_id=1,
                greedy=args.greedy,
                grid_size=args.grid_size,
                architecture=args.architecture_p1,
            ),
            NetworkAgent(
                model_path=model_p2_path,
                player_id=2,
                greedy=args.greedy,
                grid_size=args.grid_size,
                architecture=args.architecture_p2,
            ),
        )

    raise ValueError(f"Unknown mode: {args.mode}")


def main():
    args = fill_missing_args(parse_args())
    patch_visualization(args.grid_size)

    game_env = GridWorldFireGame(
        grid_size=args.grid_size,
        player_lives=args.player_lives,
        reward_method=REWARD_METHOD,
        debug=args.debug,
    )

    player_1_agent, player_2_agent = build_agents(args)

    print("\n--- Starting Match ---")
    print(f"Mode: {args.mode}")
    print(f"Grid size: {args.grid_size}")
    if args.mode == "model-vs-model":
        print(f"Player 1 model architecture: {args.architecture_p1}")
        print(f"Player 2 model architecture: {args.architecture_p2}")
    elif args.mode != "human-vs-heuristic":
        print(f"Model architecture: {args.architecture}")
    if player_1_agent is None:
        print("You are Player 1 (Red). Use WASD.")
    if player_2_agent is None:
        print("You are Player 2 (Blue). Use arrow keys.")
    print("----------------------\n")

    ui = FireGameUI(game_env, player_1_agent=player_1_agent, player_2_agent=player_2_agent,
                    step_mode=args.turn_mode)
    ui.run()


if __name__ == "__main__":
    main()
