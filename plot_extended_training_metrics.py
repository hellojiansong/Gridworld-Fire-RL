#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "data_agents" / "training_logs_extended_training"
PPO_CSV = LOG_DIR / "ppo_episode_metrics.csv"
IMITATION_CSV = LOG_DIR / "imitation_training.csv"

# Adjust these settings before running the script.
SMOOTHING_WINDOW = 250
SAVE_DIR = PROJECT_ROOT / "plots" / "extended_training_metrics"
SHOW_PLOTS = False
PLOT_IMITATION = True
INCLUDED_AGENTS = {"agent_a", "heuristic_agent"}


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


def outcome_label(agent_name: str, outcome: str) -> str:
    if outcome == "win":
        return f"{agent_name} win / opponent loss"
    if outcome == "loss":
        return f"opponent win / {agent_name} loss"
    if outcome == "draw":
        return f"Draw involving {agent_name}"
    return f"{agent_name} - {outcome}"


def read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")

    df = pd.read_csv(path)
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    return df


def plot_episode_metrics(ppo_df: pd.DataFrame, rolling_window: int, save_dir: Path | None, show: bool) -> None:
    suffix = f"({rolling_window}-Episode Rolling Mean)" if rolling_window > 1 else "(Unsmoothed)"

    plot_lines(
        ppo_df,
        x_col="episode",
        y_cols=["episodic_return"],
        title=f"Extended Training Episodic Return {suffix}",
        ylabel="episodic return",
        group_col="agent_name",
        rolling=True,
        rolling_window=rolling_window,
        save_dir=save_dir,
        filename="extended_episodic_return.png",
        show=show,
    )

    plot_lines(
        ppo_df,
        x_col="episode",
        y_cols=["win", "loss", "draw"],
        title=f"Extended Training Win/Loss/Draw Rate {suffix}",
        ylabel="rate",
        group_col="agent_name",
        rolling=True,
        rolling_window=rolling_window,
        save_dir=save_dir,
        filename="extended_win_loss_draw_rate.png",
        show=show,
        label_fn=outcome_label,
    )

    plot_lines(
        ppo_df,
        x_col="episode",
        y_cols=["boosters_picked"],
        title=f"Extended Training Boosters Picked {suffix}",
        ylabel="boosters picked",
        group_col="agent_name",
        rolling=True,
        rolling_window=rolling_window,
        save_dir=save_dir,
        filename="extended_boosters_picked.png",
        show=show,
    )

    plot_lines(
        ppo_df,
        x_col="episode",
        y_cols=["avg_fire_trail_length"],
        title=f"Extended Training Average Fire Trail Length {suffix}",
        ylabel="average active fire tiles",
        group_col="agent_name",
        rolling=True,
        rolling_window=rolling_window,
        save_dir=save_dir,
        filename="extended_avg_fire_trail_length.png",
        show=show,
    )

    plot_lines(
        ppo_df,
        x_col="episode",
        y_cols=["own_fire_steps", "enemy_fire_steps"],
        title=f"Extended Training Own/Enemy Fire Steps {suffix}",
        ylabel="steps per episode",
        group_col="agent_name",
        rolling=True,
        rolling_window=rolling_window,
        save_dir=save_dir,
        filename="extended_own_enemy_fire_steps.png",
        show=show,
    )


def main() -> None:
    if SMOOTHING_WINDOW < 1:
        raise ValueError("SMOOTHING_WINDOW must be at least 1")

    print(f"Using extended-training logs: {LOG_DIR}")
    print(f"Using smoothing window: {SMOOTHING_WINDOW}")

    if PLOT_IMITATION:
        imitation_df = read_csv(IMITATION_CSV, {"epoch", "train_loss"})
        suffix = f"({SMOOTHING_WINDOW}-Point Rolling Mean)" if SMOOTHING_WINDOW > 1 else "(Unsmoothed)"
        plot_lines(
            imitation_df,
            x_col="epoch",
            y_cols=["train_loss"],
            title=f"Extended Training Imitation Learning Loss {suffix}",
            ylabel="loss",
            rolling=True,
            rolling_window=SMOOTHING_WINDOW,
            save_dir=SAVE_DIR,
            filename="extended_imitation_learning_loss.png",
            show=SHOW_PLOTS,
        )

    ppo_df = read_csv(
        PPO_CSV,
        {
            "episode",
            "agent_name",
            "episodic_return",
            "win",
            "loss",
            "draw",
            "boosters_picked",
            "avg_fire_trail_length",
            "own_fire_steps",
            "enemy_fire_steps",
        },
    )
    ppo_df = ppo_df[ppo_df["agent_name"].isin(INCLUDED_AGENTS)].copy()
    plot_episode_metrics(ppo_df, SMOOTHING_WINDOW, SAVE_DIR, SHOW_PLOTS)

    if SAVE_DIR is not None:
        print(f"Saved extended-training plots to {SAVE_DIR}")


if __name__ == "__main__":
    main()
