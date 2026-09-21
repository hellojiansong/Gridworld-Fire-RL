"""
Round-robin tournament with Elo and TrueSkill ratings.

Every agent plays every other agent for --games games per matchup. To remove
first-player (side) advantage, each pairing is split evenly: half the games with
agent A as Player 1 and half with agent A as Player 2. Ratings are derived from
the resulting per-game outcomes:

  * Win%      - wins / games over all matchups.
  * Elo       - sequential Elo updates; to make the rating independent of the
                (arbitrary) order games are replayed in, we average the final
                Elo over many random shuffles of the game list.
  * TrueSkill - Microsoft's Bayesian skill rating (mu, sigma), likewise averaged
                over shuffles. We report the conservative skill mu - 3*sigma.

Outputs:
  * A ranked leaderboard table in the console (rankable by win%, Elo, TrueSkill).
  * A report-ready figure (Elo bars, TrueSkill bars, head-to-head heatmap).
  * A CSV of head-to-head results and a CSV of the final ratings.

Example:
    python src/tournament.py --games 200
    python src/tournament.py --games 100 --agents heuristic random ppo_agent_a_groupnorm.pt
    python src/tournament.py --games 200 --sort-by elo --no-show
"""

import os
import csv
import argparse
import itertools
from datetime import datetime

import numpy as np

import matplotlib.pyplot as plt

import trueskill

from evaluate import get_agent, run_tournament


# Default roster (the agents requested for the comparison). Missing checkpoints
# are skipped with a warning, so this works even before every .pt is present.
DEFAULT_AGENTS = [
    "heuristic",
    "ppo_agent_b_defensive_Grid_full_sep.pt",
    "ppo_agent_a_offensive_better_actor.pt",
    "ppo_agent_a_groupnorm.pt",
    "ppo_agent_b_territorial.pt",
    "random",
]

# Short, human-friendly labels for plots/tables (falls back to the raw name).
DISPLAY_NAMES = {
    "heuristic": "Heuristic",
    "random": "Random",
    "ppo_agent_b_defensive_Grid_full_sep.pt": "PPO Grid-Sep (def)",
    "ppo_agent_a_offensive_better_actor.pt": "PPO BetterActor (off)",
    "ppo_agent_a_groupnorm.pt": "PPO GroupNorm (default)",
    "ppo_agent_b_territorial.pt": "PPO Territorial (def)",
}

ELO_INITIAL = 1000.0
ELO_K = 32.0


def display_name(agent):
    return DISPLAY_NAMES.get(agent, agent.replace(".pt", ""))


# ---- Tournament play ----
def play_matchup(name_1, name_2, n_games, grid_size, seed):
    """
    Play name_1 vs name_2 over n_games, split between both side assignments.

    Returns a dict of head-to-head counts from name_1's perspective:
        {agent1_wins, agent2_wins, draws, games}
    plus the averaged per-perspective extra stats for the report CSV.
    """
    n_first = n_games // 2          # name_1 plays as Player 1
    n_second = n_games - n_first    # name_1 plays as Player 2

    wins_1 = wins_2 = draws = 0
    rows = []

    # Leg 1: name_1 = P1, name_2 = P2
    if n_first > 0:
        a1 = get_agent(name_1, player_id=1, grid_size=grid_size)
        a2 = get_agent(name_2, player_id=2, grid_size=grid_size)
        r = run_tournament(a1, a2, n_games=n_first, grid_size=grid_size,
                           seed=seed, verbose=False)
        wins_1 += r["agent1_wins"]
        wins_2 += r["agent2_wins"]
        draws += r["draws"]
        rows.append(r)

    # Leg 2: name_2 = P1, name_1 = P2  (use a different seed band so maps differ)
    if n_second > 0:
        a1 = get_agent(name_2, player_id=1, grid_size=grid_size)
        a2 = get_agent(name_1, player_id=2, grid_size=grid_size)
        r = run_tournament(a1, a2, n_games=n_second, grid_size=grid_size,
                           seed=seed + 100000, verbose=False)
        wins_2 += r["agent1_wins"]  # name_2 was P1 here
        wins_1 += r["agent2_wins"]  # name_1 was P2 here
        draws += r["draws"]

    return {
        "agent1": name_1,
        "agent2": name_2,
        "games": n_games,
        "agent1_wins": wins_1,
        "agent2_wins": wins_2,
        "draws": draws,
        "agent1_winrate": round(wins_1 / n_games * 100, 1) if n_games else 0.0,
        "agent2_winrate": round(wins_2 / n_games * 100, 1) if n_games else 0.0,
        "draw_rate": round(draws / n_games * 100, 1) if n_games else 0.0,
    }


# ---- Ratings ----
def build_game_list(matchups):
    """
    Expand head-to-head counts into a flat list of single-game outcomes:
        (winner_name, loser_name, is_draw)
    For draws the (a, b) order does not matter.
    """
    games = []
    for m in matchups:
        a, b = m["agent1"], m["agent2"]
        games += [(a, b, False)] * m["agent1_wins"]
        games += [(b, a, False)] * m["agent2_wins"]
        games += [(a, b, True)] * m["draws"]
    return games


def _elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def compute_elo(agents, games, n_shuffles=200, seed=0):
    """Average final Elo over many random replay orders (order-independent)."""
    rng = np.random.default_rng(seed)
    totals = {a: 0.0 for a in agents}

    for _ in range(n_shuffles):
        order = rng.permutation(len(games))
        ratings = {a: ELO_INITIAL for a in agents}
        for idx in order:
            winner, loser, is_draw = games[idx]
            exp_w = _elo_expected(ratings[winner], ratings[loser])
            exp_l = 1.0 - exp_w
            score_w, score_l = (0.5, 0.5) if is_draw else (1.0, 0.0)
            ratings[winner] += ELO_K * (score_w - exp_w)
            ratings[loser] += ELO_K * (score_l - exp_l)
        for a in agents:
            totals[a] += ratings[a]

    return {a: totals[a] / n_shuffles for a in agents}


def compute_trueskill(agents, games, n_shuffles=200, seed=0):
    """Average TrueSkill mu/sigma over many random replay orders."""
    env = trueskill.TrueSkill(draw_probability=_draw_probability(games))
    rng = np.random.default_rng(seed)
    mu_tot = {a: 0.0 for a in agents}
    sigma_tot = {a: 0.0 for a in agents}

    for _ in range(n_shuffles):
        order = rng.permutation(len(games))
        ratings = {a: env.create_rating() for a in agents}
        for idx in order:
            winner, loser, is_draw = games[idx]
            r_w, r_l = ratings[winner], ratings[loser]
            if is_draw:
                new_w, new_l = env.rate_1vs1(r_w, r_l, drawn=True)
            else:
                new_w, new_l = env.rate_1vs1(r_w, r_l)
            ratings[winner], ratings[loser] = new_w, new_l
        for a in agents:
            mu_tot[a] += ratings[a].mu
            sigma_tot[a] += ratings[a].sigma

    mu = {a: mu_tot[a] / n_shuffles for a in agents}
    sigma = {a: sigma_tot[a] / n_shuffles for a in agents}
    return mu, sigma


def _draw_probability(games):
    if not games:
        return 0.1
    draws = sum(1 for _, _, d in games if d)
    p = draws / len(games)
    return min(max(p, 1e-4), 0.8)  # keep TrueSkill numerically happy


def compute_winrates(agents, matchups):
    """Aggregate W/L/D and win% per agent across all matchups."""
    stats = {a: {"wins": 0, "losses": 0, "draws": 0, "games": 0} for a in agents}
    for m in matchups:
        a, b = m["agent1"], m["agent2"]
        stats[a]["wins"] += m["agent1_wins"]
        stats[a]["losses"] += m["agent2_wins"]
        stats[a]["draws"] += m["draws"]
        stats[a]["games"] += m["games"]
        stats[b]["wins"] += m["agent2_wins"]
        stats[b]["losses"] += m["agent1_wins"]
        stats[b]["draws"] += m["draws"]
        stats[b]["games"] += m["games"]
    for a in agents:
        g = stats[a]["games"]
        stats[a]["winrate"] = (stats[a]["wins"] / g * 100) if g else 0.0
    return stats


# ---- Reporting ----
def _ranks(values, agents, higher_is_better=True):
    """Return {agent: rank} where rank 1 is best."""
    ordered = sorted(agents, key=lambda a: values[a], reverse=higher_is_better)
    return {a: i + 1 for i, a in enumerate(ordered)}


def print_leaderboard(agents, stats, elo, mu, sigma, sort_by="winrate"):
    ts_cons = {a: mu[a] - 3 * sigma[a] for a in agents}

    rank_wr = _ranks({a: stats[a]["winrate"] for a in agents}, agents)
    rank_elo = _ranks(elo, agents)
    rank_ts = _ranks(ts_cons, agents)

    key = {
        "winrate": lambda a: stats[a]["winrate"],
        "elo": lambda a: elo[a],
        "trueskill": lambda a: ts_cons[a],
    }[sort_by]
    ordered = sorted(agents, key=key, reverse=True)

    header = (
        f"{'#':>2}  {'Agent':<24} {'Games':>6} {'W':>5} {'L':>5} {'D':>5} "
        f"{'Win%':>6} {'Elo':>7} {'TS mu':>7} {'TS sig':>7} {'TS mu-3sig':>11} "
        f"{'rWR':>4} {'rElo':>5} {'rTS':>4}"
    )
    line = "=" * len(header)
    print("\n" + line)
    print(f"  TOURNAMENT LEADERBOARD  (sorted by {sort_by})")
    print(line)
    print(header)
    print("-" * len(header))
    for i, a in enumerate(ordered, start=1):
        s = stats[a]
        print(
            f"{i:>2}  {display_name(a):<24} {s['games']:>6} {s['wins']:>5} "
            f"{s['losses']:>5} {s['draws']:>5} {s['winrate']:>5.1f}% "
            f"{elo[a]:>7.1f} {mu[a]:>7.2f} {sigma[a]:>7.2f} {ts_cons[a]:>11.2f} "
            f"{rank_wr[a]:>4} {rank_elo[a]:>5} {rank_ts[a]:>4}"
        )
    print(line)
    print("  rWR/rElo/rTS = rank by Win% / Elo / TrueSkill (mu-3sig). "
          "TS mu-3sig is the conservative skill estimate.")
    print(line)


def head_to_head_matrix(agents, matchups):
    """Win% of row agent vs column agent (row's perspective)."""
    n = len(agents)
    idx = {a: i for i, a in enumerate(agents)}
    mat = np.full((n, n), np.nan)
    for m in matchups:
        i, j = idx[m["agent1"]], idx[m["agent2"]]
        mat[i, j] = m["agent1_winrate"]
        mat[j, i] = m["agent2_winrate"]
    return mat


def make_figure(agents, stats, elo, mu, sigma, matchups, games, out_path):
    ts_cons = {a: mu[a] - 3 * sigma[a] for a in agents}
    labels = [display_name(a) for a in agents]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.45)

    # --- Panel 1: Elo (sorted) -------------------------------------------- #
    ax1 = fig.add_subplot(gs[0, 0])
    order = sorted(range(len(agents)), key=lambda k: elo[agents[k]])
    y = np.arange(len(agents))
    vals = [elo[agents[k]] for k in order]
    bars = ax1.barh(y, vals, color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels([labels[k] for k in order], fontsize=9)
    ax1.set_xlabel("Elo rating")
    ax1.set_title("(a) Elo", fontweight="bold")
    ax1.axvline(ELO_INITIAL, color="grey", ls="--", lw=1, label=f"start ({int(ELO_INITIAL)})")
    ax1.legend(fontsize=8, loc="lower right")
    for b, v in zip(bars, vals):
        ax1.text(v + (max(vals) - min(vals)) * 0.01, b.get_y() + b.get_height() / 2,
                 f"{v:.0f}", va="center", fontsize=8)

    # --- Panel 2: TrueSkill (mu with +/-3 sigma) -------------------------- #
    ax2 = fig.add_subplot(gs[0, 1])
    order2 = sorted(range(len(agents)), key=lambda k: ts_cons[agents[k]])
    mus = [mu[agents[k]] for k in order2]
    errs = [3 * sigma[agents[k]] for k in order2]
    ax2.barh(y, mus, xerr=errs, color="#55A868", edgecolor="black",
             linewidth=0.6, error_kw={"ecolor": "#333333", "capsize": 3, "lw": 1})
    ax2.set_yticks(y)
    ax2.set_yticklabels([labels[k] for k in order2], fontsize=9)
    ax2.set_xlabel(r"TrueSkill $\mu$  (bars), $\pm 3\sigma$ (whiskers)")
    ax2.set_title("(b) TrueSkill", fontweight="bold")

    # --- Panel 3: head-to-head heatmap ------------------------------------ #
    ax3 = fig.add_subplot(gs[0, 2])
    mat = head_to_head_matrix(agents, matchups)
    im = ax3.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax3.set_xticks(np.arange(len(agents)))
    ax3.set_yticks(np.arange(len(agents)))
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_title("(c) Head-to-head win% (row vs col)", fontweight="bold")
    for i in range(len(agents)):
        for j in range(len(agents)):
            if not np.isnan(mat[i, j]):
                ax3.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                         fontsize=8, color="black")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("win %", fontsize=8)

    fig.suptitle(
        f"Agent Tournament - round-robin, {games} games/matchup "
        f"(both sides), {len(agents)} agents",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")
    return fig


def save_ratings_csv(agents, stats, elo, mu, sigma, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "games", "wins", "losses", "draws",
                         "winrate", "elo", "trueskill_mu", "trueskill_sigma",
                         "trueskill_mu_minus_3sigma"])
        ordered = sorted(agents, key=lambda a: elo[a], reverse=True)
        for a in ordered:
            s = stats[a]
            writer.writerow([a, s["games"], s["wins"], s["losses"], s["draws"],
                             round(s["winrate"], 2), round(elo[a], 2),
                             round(mu[a], 3), round(sigma[a], 3),
                             round(mu[a] - 3 * sigma[a], 3)])
    print(f"Ratings CSV saved to {out_path}")


def save_matchups_csv(matchups, out_path):
    fields = ["agent1", "agent2", "games", "agent1_wins", "agent2_wins",
              "draws", "agent1_winrate", "agent2_winrate", "draw_rate"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(matchups)
    print(f"Matchups CSV saved to {out_path}")


# ---- Main ----
def resolve_agents(requested):
    """Keep order, drop duplicates, and skip .pt files that don't exist."""
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    resolved = []
    for a in dict.fromkeys(requested):
        if a.endswith(".pt"):
            path = os.path.join(checkpoints_dir, a)
            if not os.path.exists(path):
                print(f"  [skip] checkpoint not found: {a}")
                continue
        resolved.append(a)
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Round-robin tournament with Elo and TrueSkill ratings.")
    parser.add_argument("--games", type=int, default=200,
                        help="Games per matchup (split evenly across both sides).")
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS,
                        help="Agents: 'random', 'heuristic', or a .pt filename from checkpoints/.")
    parser.add_argument("--seed", type=int, default=0, help="Base map seed.")
    parser.add_argument("--shuffles", type=int, default=200,
                        help="Random replay orders averaged for Elo/TrueSkill.")
    parser.add_argument("--sort-by", choices=["winrate", "elo", "trueskill"],
                        default="winrate", help="Leaderboard sort key.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: ../results).")
    parser.add_argument("--show", action="store_true",
                        help="Open the figure in a window in addition to saving it.")
    args = parser.parse_args()

    agents = resolve_agents(args.agents)
    if len(agents) < 2:
        print("Need at least 2 available agents to run a tournament.")
        return

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    pairs = list(itertools.combinations(agents, 2))
    print(f"Agents ({len(agents)}): {[display_name(a) for a in agents]}")
    print(f"Matchups: {len(pairs)}  x  {args.games} games "
          f"(= {len(pairs) * args.games} games total)\n")

    matchups = []
    for k, (a, b) in enumerate(pairs):
        print(f"[{k + 1}/{len(pairs)}] {display_name(a)}  vs  {display_name(b)} ...")
        m = play_matchup(a, b, n_games=args.games, grid_size=args.grid_size,
                         seed=args.seed + k * 1000)
        matchups.append(m)
        print(f"      {display_name(a)} {m['agent1_wins']}  -  "
              f"{m['agent2_wins']} {display_name(b)}   (draws: {m['draws']})")

    # Ratings
    games = build_game_list(matchups)
    stats = compute_winrates(agents, matchups)
    elo = compute_elo(agents, games, n_shuffles=args.shuffles, seed=args.seed)
    mu, sigma = compute_trueskill(agents, games, n_shuffles=args.shuffles, seed=args.seed)

    # Report
    print_leaderboard(agents, stats, elo, mu, sigma, sort_by=args.sort_by)
    save_matchups_csv(matchups, os.path.join(out_dir, f"tournament_matchups_{ts}.csv"))
    save_ratings_csv(agents, stats, elo, mu, sigma,
                     os.path.join(out_dir, f"tournament_ratings_{ts}.csv"))
    fig_path = os.path.join(out_dir, f"tournament_report_{ts}.png")

    make_figure(agents, stats, elo, mu, sigma, matchups, args.games, fig_path)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
