#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data_agents"

# Adjust these settings before running the script.
SMOOTHING_WINDOW = 250
SAVE_DIR = PROJECT_ROOT / "plots" / "combined_agent_metrics"
SHOW_PLOTS = False
PLOT_IMITATION = True

AGENT_SOURCES = (
    {
        "label": "Offensive - agent_a",
        "log_dir": DATA_ROOT / "defensive_offensive_training_logs",
        "agent_name": "agent_a",
    },
    {
        "label": "Defensive - agent_b",
        "log_dir": DATA_ROOT / "defensive_offensive_training_logs",
        "agent_name": "agent_b",
    },
    {
        "label": "Territorial - agent_b",
        "log_dir": DATA_ROOT / "territorial_training_logs",
        "agent_name": "agent_b",
    },
)

OUTCOME_NAMES = {
    "Offensive": ("Offensive", "Defensive"),
    "Defensive": ("Defensive", "Offensive"),
    "Territorial": ("Territorial", "agent_a"),
}


def smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window, min_periods=1).mean()


def save_or_show(filename: str, save_dir: Path | None, show: bool) -> None:
    plt.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_lines(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    ylabel: str | None = None,
    group_col: str | None = None,
    rolling: bool = False,
    rolling_window: int = SMOOTHING_WINDOW,
    save_dir: Path | None = None,
    filename: str = "plot.png",
    show: bool = True,
    label_fn=None,
) -> None:
    plt.figure(figsize=(12, 5))

    if group_col is None:
        for y_col in y_cols:
            y = df[y_col]
            if rolling:
                y = smooth(y, rolling_window)
            plt.plot(df[x_col], y, label=y_col)
    else:
        for group_name, group in df.groupby(group_col, sort=False):
            group = group.sort_values(x_col)
            for y_col in y_cols:
                y = group[y_col]
                if rolling:
                    y = smooth(y, rolling_window)
                if label_fn is None:
                    label = group_name if len(y_cols) == 1 else f"{group_name} - {y_col}"
                else:
                    label = label_fn(group_name, y_col)
                plt.plot(group[x_col], y, label=label)

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel or ", ".join(y_cols))
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    save_or_show(filename, save_dir, show)


def outcome_label(agent_label: str, outcome: str) -> str:
    run_label, agent_name = agent_label.rsplit(" - ", 1)
    player_name, opponent = OUTCOME_NAMES.get(
        run_label,
        (run_label, "agent_b" if agent_name == "agent_a" else "agent_a"),
    )

    if outcome == "win":
        return f"{player_name} win / {opponent} loss"
    if outcome == "loss":
        return f"{opponent} win / {player_name} loss"
    if outcome == "draw":
        return f"Draw ({player_name} vs {opponent})"
    return f"{player_name} - {outcome}"


def read_agent_episode_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for source in AGENT_SOURCES:
        csv_path = source["log_dir"] / "ppo_episode_metrics.csv"
        if not csv_path.exists():
            print(f"Missing PPO episode log: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df = df[df["agent_name"] == source["agent_name"]].copy()
        df["agent_label"] = source["label"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No PPO episode logs found.")
    return pd.concat(frames, ignore_index=True)


def read_agent_update_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for source in AGENT_SOURCES:
        csv_path = source["log_dir"] / "ppo_update_metrics.csv"
        if not csv_path.exists():
            print(f"Missing PPO update log: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df = df[df["updated_agent"] == source["agent_name"]].copy()
        df["agent_label"] = source["label"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No PPO update logs found.")
    return pd.concat(frames, ignore_index=True)


def read_imitation_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen_dirs: set[Path] = set()

    for source in AGENT_SOURCES:
        log_dir = source["log_dir"]
        if log_dir in seen_dirs:
            continue
        seen_dirs.add(log_dir)

        csv_path = log_dir / "imitation_training.csv"
        if not csv_path.exists():
            print(f"Missing imitation log: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df["run_label"] = log_dir.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No imitation logs found.")
    return pd.concat(frames, ignore_index=True)


def plot_episode_metrics(ppo_df: pd.DataFrame, rolling_window: int, save_dir: Path | None, show: bool) -> None:
    suffix = f"({rolling_window}-Episode Rolling Mean)" if rolling_window > 1 else "(Unsmoothed)"

    plot_lines(
        ppo_df,
        x_col="episode",
        y_cols=["episodic_return"],
        title=f"PPO Episodic Return {suffix}",
        ylabel="episodic return",
        group_col="agent_label",
        rolling=True,
        rolling_window=rolling_window,
        save_dir=save_dir,
        filename="ppo_episodic_return.png",
        show=show,
    )

    plot_lines(
        ppo_df,
        x_col="episode",
        y_cols=["win", "loss", "draw"],
        title=f"PPO Win/Loss/Draw Rate {suffix}",
        ylabel="rate",
        group_col="agent_label",
        rolling=True,
        rolling_window=rolling_window,
        save_dir=save_dir,
        filename="ppo_win_loss_draw_rate.png",
        show=show,
        label_fn=outcome_label,
    )

    plot_lines(
        ppo_df,
        x_col="episode",
        y_cols=["boosters_picked"],
        title=f"Boosters Picked {suffix}",
        ylabel="boosters picked",
        group_col="agent_label",
        rolling=True,
        rolling_window=rolling_window,
        save_dir=save_dir,
        filename="boosters_picked.png",
        show=show,
    )

    plot_lines(
        ppo_df,
        x_col="episode",
        y_cols=["avg_fire_trail_length"],
        title=f"Average Fire Trail Length {suffix}",
        ylabel="average active fire tiles",
        group_col="agent_label",
        rolling=True,
        rolling_window=rolling_window,
        save_dir=save_dir,
        filename="avg_fire_trail_length.png",
        show=show,
    )

    plot_lines(
        ppo_df,
        x_col="episode",
        y_cols=["own_fire_steps", "enemy_fire_steps"],
        title=f"Own/Enemy Fire Steps {suffix}",
        ylabel="steps per episode",
        group_col="agent_label",
        rolling=True,
        rolling_window=rolling_window,
        save_dir=save_dir,
        filename="own_enemy_fire_steps.png",
        show=show,
    )


def main() -> None:
    if SMOOTHING_WINDOW < 1:
        raise ValueError("SMOOTHING_WINDOW must be at least 1")

    save_dir = SAVE_DIR
    suffix = f"({SMOOTHING_WINDOW}-Point Rolling Mean)" if SMOOTHING_WINDOW > 1 else "(Unsmoothed)"

    print(f"Using smoothing window: {SMOOTHING_WINDOW}")

    if PLOT_IMITATION:
        imitation_df = read_imitation_metrics()
        plot_lines(
            imitation_df,
            x_col="epoch",
            y_cols=["train_loss"],
            title=f"Imitation Learning Loss {suffix}",
            ylabel="loss",
            group_col="run_label",
            rolling=True,
            rolling_window=SMOOTHING_WINDOW,
            save_dir=save_dir,
            filename="imitation_learning_loss.png",
            show=SHOW_PLOTS,
        )

    update_df = read_agent_update_metrics()
    plot_lines(
        update_df,
        x_col="episode",
        y_cols=["explained_variance"],
        title=f"PPO Critic Explained Variance by Update {suffix}",
        ylabel="explained variance",
        group_col="agent_label",
        rolling=True,
        rolling_window=SMOOTHING_WINDOW,
        save_dir=save_dir,
        filename="ppo_critic_explained_variance.png",
        show=SHOW_PLOTS,
    )

    ppo_df = read_agent_episode_metrics()
    plot_episode_metrics(ppo_df, SMOOTHING_WINDOW, save_dir, SHOW_PLOTS)

    if save_dir is not None:
        print(f"Saved plots to {save_dir}")


if __name__ == "__main__":
    main()
