"""
Runs tournaments between agents (heuristics and model networks).
Returns win rates and other statistics.

Example usage:
    python src/evaluate.py --games 1000 --agents \
    ppo_agent_a_offensive_better_actor.pt \
    ppo_agent_a_groupnorm.pt \
    ppo_agent_b_territorial.pt \
    ppo_agent_b_defensive_Grid_full_sep.pt \
    random \
    heuristic
"""

import os
import csv
import argparse
import itertools
from datetime import datetime
import numpy as np
from env import GridWorldFireGame
from heuristic_agent import HeuristicAgent, HeuristicWeights
from random_agent import RandomAgent
from play_vs_model import NetworkAgent

def _infer_architecture(filename):
    name = filename.lower()
    if "better_actor" in name or "betteractor" in name:
        return "better actor"
    if "territorial" in name or "territoryial" in name:
        return "territorial reward"
    if "groupnorm" in name:
        return "GroupNorm"
    return "grid full separate"

def get_agent(agent_name, player_id, grid_size=10):
    if agent_name == "random":
        return RandomAgent(seed=player_id)
    elif agent_name == "heuristic":
        return HeuristicAgent(player_id=player_id, weights=HeuristicWeights(), seed=player_id)
    elif agent_name.endswith(".pt"):
        model_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", agent_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_path} not found.")
        arch = _infer_architecture(agent_name)
        return NetworkAgent(model_path=model_path, player_id=player_id, grid_size=grid_size, greedy=True, architecture=arch)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

def run_tournament(agent_1, agent_2, n_games=500, grid_size=10, seed=0, verbose=True):
    """
    Runs n_games between agent_1 (P1) and agent_2 (P2).
    Each game uses a different map seed for fairness.

    Returns a dict with win/loss/draw counts and average stats.
    """
    results = {"agent1_wins": 0, "agent2_wins": 0, "draws": 0}
    total_turns = []
    total_reward_1 = []
    total_reward_2 = []
    turns_on_win = []
    turns_on_loss = []
    turns_on_draw = []
    win_margins = []          # agent1 lives remaining on agent1 wins
    hits_landed_per_ep = []   # times opponent stepped on agent1's fire
    hits_taken_per_ep = []    # times agent1 stepped on opponent's fire

    for game_idx in range(n_games):
        env = GridWorldFireGame(
            grid_size=grid_size,
            player_lives=3,
            max_episode_length=500,
            seed=seed + game_idx,  # different map each game
        )

        state = env.reset()
        info = env.info()
        done = False

        cumulative_reward_1 = 0
        cumulative_reward_2 = 0

        while not done:
            a1 = agent_1.select_action(state, info)
            a2 = agent_2.select_action(state, info)
            state, r1, r2, done, info = env.step(a1, a2)
            cumulative_reward_1 += r1
            cumulative_reward_2 += r2

        winner = info["winner"]
        turns = info["turn_count"]
        p1_info = info["players"][1]

        if winner == 1:
            results["agent1_wins"] += 1
            turns_on_win.append(turns)
            win_margins.append(p1_info["lives_left"])
        elif winner == 2:
            results["agent2_wins"] += 1
            turns_on_loss.append(turns)
        else:
            results["draws"] += 1
            turns_on_draw.append(turns)

        total_turns.append(turns)
        total_reward_1.append(cumulative_reward_1)
        total_reward_2.append(cumulative_reward_2)
        hits_landed_per_ep.append(p1_info["num_enemy_fire_conversions"])
        hits_taken_per_ep.append(p1_info["num_enemy_fire_hits"])

        if verbose and (game_idx + 1) % 50 == 0:
            print(f"  Game {game_idx + 1}/{n_games} done...")

    results["avg_turns"] = round(np.mean(total_turns), 1)
    results["avg_reward_agent1"] = round(np.mean(total_reward_1), 1)
    results["avg_reward_agent2"] = round(np.mean(total_reward_2), 1)
    results["agent1_winrate"] = round(results["agent1_wins"] / n_games * 100, 1)
    results["agent2_winrate"] = round(results["agent2_wins"] / n_games * 100, 1)
    results["draw_rate"] = round(results["draws"] / n_games * 100, 1)
    results["avg_turns_on_win"] = round(np.mean(turns_on_win), 1) if turns_on_win else None
    results["avg_turns_on_loss"] = round(np.mean(turns_on_loss), 1) if turns_on_loss else None
    results["avg_turns_on_draw"] = round(np.mean(turns_on_draw), 1) if turns_on_draw else None
    results["avg_win_margin"] = round(np.mean(win_margins), 2) if win_margins else None
    results["avg_hits_landed"] = round(np.mean(hits_landed_per_ep), 2)
    results["avg_hits_taken"] = round(np.mean(hits_taken_per_ep), 2)

    return results


def _fmt(val, decimals=1):
    return f"{val:.{decimals}f}" if val is not None else "n/a"


def print_results(results, name_1="Agent 1", name_2="Agent 2"):
    print("\n" + "=" * 60)
    print(f"  {name_1:20s}  vs  {name_2:20s}")
    print("=" * 60)
    print(f"  Wins:              {results['agent1_wins']:>5}  ({results['agent1_winrate']}%)"
          f"      {results['agent2_wins']:>5}  ({results['agent2_winrate']}%)")
    print(f"  Draws:             {results['draws']:>5}  ({results['draw_rate']}%)")
    print(f"  Avg turns (all):   {results['avg_turns']}")
    print(f"  Avg turns on win:  {_fmt(results['avg_turns_on_win'])}")
    print(f"  Avg turns on loss: {_fmt(results['avg_turns_on_loss'])}")
    print(f"  Avg turns on draw: {_fmt(results['avg_turns_on_draw'])}")
    print(f"  Avg win margin:    {_fmt(results['avg_win_margin'], 2)} lives  (agent 1 wins only)")
    print(f"  Hits landed/ep:    {_fmt(results['avg_hits_landed'], 2)}  (opp stepped on agent1 fire)")
    print(f"  Hits taken/ep:     {_fmt(results['avg_hits_taken'], 2)}  (agent1 stepped on opp fire)")
    print("=" * 60)


def save_results_to_csv(rows, output_dir=None):
    """
    Saves a list of matchup result dicts to a timestamped CSV file.
    Each row must include 'agent1' and 'agent2' keys in addition to the
    fields returned by run_tournament().
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"tournament_{timestamp}.csv")

    fieldnames = [
        "agent1", "agent2", "games",
        "agent1_wins", "agent2_wins", "draws",
        "agent1_winrate", "agent2_winrate", "draw_rate",
        "avg_turns", "avg_turns_on_win", "avg_turns_on_loss", "avg_turns_on_draw",
        "avg_win_margin", "avg_hits_landed", "avg_hits_taken",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {filename}")
    return filename


def print_summary_matrix(all_rows, agents):
    """Prints a win-rate matrix: row agent win rate against column agent."""
    col_w = max(12, *(len(a) for a in agents))
    header = f"{'':>{col_w}}" + "".join(f"  {a:>{col_w}}" for a in agents)
    print("\n" + "=" * len(header))
    print("  Win-rate matrix (row vs column, from row's perspective)")
    print("=" * len(header))
    print(header)

    lookup = {}
    for row in all_rows:
        lookup[(row["agent1"], row["agent2"])] = row["agent1_winrate"]
        lookup[(row["agent2"], row["agent1"])] = row["agent2_winrate"]

    for a1 in agents:
        cells = []
        for a2 in agents:
            if a1 == a2:
                cells.append(f"{'--':>{col_w}}")
            else:
                wr = lookup.get((a1, a2))
                cells.append(f"{f'{wr}%':>{col_w}}" if wr is not None else f"{'n/a':>{col_w}}")
        print(f"{a1:>{col_w}}" + "".join(f"  {c}" for c in cells))

    print("=" * len(header))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agents in a round-robin tournament.")
    parser.add_argument("--games", type=int, default=100, help="Number of games per matchup")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size of the game")
    parser.add_argument("--agents", nargs="+", default=["heuristic", "random"],
                        help="Agents to evaluate: 'random', 'heuristic', or a .pt filename from checkpoints/")
    parser.add_argument("--all-checkpoints", action="store_true", help="Add all .pt files in checkpoints/ to the agent list")
    args = parser.parse_args()

    agents_to_run = list(dict.fromkeys(args.agents))  # deduplicate, preserve order

    if args.all_checkpoints:
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        if os.path.exists(checkpoints_dir):
            for f in sorted(os.listdir(checkpoints_dir)):
                if f.endswith(".pt") and f not in agents_to_run:
                    agents_to_run.append(f)

    if len(agents_to_run) < 2:
        print("Need at least 2 agents to run a tournament.")
        exit(1)

    matchups = list(itertools.combinations(agents_to_run, 2))
    print(f"Agents:   {agents_to_run}")
    print(f"Matchups: {len(matchups)}  ({args.games} games each)\n")

    all_rows = []

    for a1_name, a2_name in matchups:
        try:
            agent1 = get_agent(a1_name, player_id=1, grid_size=args.grid_size)
            agent2 = get_agent(a2_name, player_id=2, grid_size=args.grid_size)
        except Exception as e:
            print(f"Skipping {a1_name} vs {a2_name}: {e}")
            continue

        print(f"Running: {a1_name}  vs  {a2_name} ...")
        results = run_tournament(agent1, agent2, n_games=args.games, grid_size=args.grid_size, verbose=False)
        print_results(results, name_1=a1_name, name_2=a2_name)
        all_rows.append({"agent1": a1_name, "agent2": a2_name, "games": args.games, **results})

    if all_rows:
        print_summary_matrix(all_rows, agents_to_run)
        save_results_to_csv(all_rows)