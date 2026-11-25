"""Summarize the requested DeepRL batch and evaluate its saved policies on CPU."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path
import io
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis"
sys.path.insert(0, str(ROOT))

GAMES = ("snake", "flap", "lander")
ALGORITHMS = ("dqn", "ppo", "a2c", "sac")
BUDGETS = (100_000, 300_000, 500_000, 1_000_000, 3_000_000, 5_000_000)
TRAIN_SEEDS = (12, 24, 36)
FIRST_TEST_SEED = 30_000
TEST_EPISODES = 50
GROUP = ["game", "algorithm", "budget"]
COLORS = dict(zip(ALGORITHMS, ("#3169a8", "#d05d31", "#268071", "#794da1")))


def evaluate_model(job):
    import torch
    from experiments.evaluate import evaluate

    game, algorithm, budget, train_seed, path = job
    torch.set_num_threads(1)
    with redirect_stdout(io.StringIO()):
        episodes = evaluate(path, TEST_EPISODES, FIRST_TEST_SEED, game=game)
    return [
        dict(game=game, algorithm=algorithm, budget=budget, train_seed=train_seed, **row)
        for row in episodes
    ]


def main():
    import pandas as pd

    jobs = []
    training = []
    training_bins = []
    for game, algorithm, budget, train_seed in product(GAMES, ALGORITHMS, BUDGETS, TRAIN_SEEDS):
        path = ROOT / "runs" / game / algorithm / f"step_{budget}" / f"seed_{train_seed}"
        config = json.loads((path / "config.json").read_text())
        assert (config["game"], config["algorithm"], config["steps"], config["seed"]) == (
            game, algorithm, budget, train_seed
        ), path
        log = pd.read_csv(path / "train.csv")
        assert len(log) and not log.isna().any().any(), path
        assert log["step"].diff().dropna().gt(0).all() and log["step"].iloc[-1] <= budget, path
        assert log["length"].sum() == log["step"].iloc[-1], path
        tail = log.tail(100)
        training.append(dict(
            game=game, algorithm=algorithm, budget=budget, train_seed=train_seed,
            train_episodes=len(log), last_logged_step=log["step"].iloc[-1],
            tail_episodes=len(tail), tail_score=tail["score"].mean(),
            tail_return=tail["return"].mean(), a2c_profile=config.get("a2c_profile", ""),
        ))
        if budget == max(BUDGETS):
            bins = ((log["step"] - 1) // 100_000 + 1) * 100_000
            by_window = log.groupby(bins.rename("window_end")).agg(
                episodes=("score", "size"), score=("score", "mean"),
                episode_return=("return", "mean"),
            ).reset_index()
            training_bins.append(by_window.assign(game=game, algorithm=algorithm, train_seed=train_seed))
        jobs.append((game, algorithm, budget, train_seed, path))
    jobs.extend((game, "random", 0, -1, None) for game in GAMES)
    print(f"Validated {len(training)} training logs. Evaluating {len(jobs)} policies.", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as executor:
        for index, rows in enumerate(executor.map(evaluate_model, jobs), start=1):
            results.extend(rows)
            if index % 18 == 0 or index == len(jobs):
                print(f"Evaluated {index}/{len(jobs)}", flush=True)

    episodes = pd.DataFrame(results)
    by_seed = GROUP + ["train_seed"]
    assert len(episodes) == len(jobs) * TEST_EPISODES
    assert episodes.groupby(by_seed)["seed"].nunique().eq(TEST_EPISODES).all()
    models = episodes.groupby(by_seed, as_index=False).agg(
        test_episodes=("score", "size"), score_mean=("score", "mean"),
        score_max=("score", "max"), return_mean=("return", "mean"), steps_mean=("steps", "mean"),
    )
    for reason in ("landed", "crash", "out_of_bounds", "time_limit"):
        rates = episodes["reason"].eq(reason).groupby([episodes[key] for key in by_seed]).mean()
        models = models.merge(rates.rename(f"{reason}_rate").reset_index(), on=by_seed)
    models = models.merge(pd.DataFrame(training), how="left", on=by_seed, validate="one_to_one")
    summary = models.groupby(GROUP, as_index=False).agg(
        policies=("train_seed", "size"), test_episodes=("test_episodes", "sum"),
        score_mean=("score_mean", "mean"), score_sd=("score_mean", "std"),
        score_min_seed=("score_mean", "min"), score_max_seed=("score_mean", "max"),
        score_max_episode=("score_max", "max"), return_mean=("return_mean", "mean"),
        return_sd=("return_mean", "std"), landed_rate=("landed_rate", "mean"),
        landed_sd=("landed_rate", "std"), crash_rate=("crash_rate", "mean"),
        out_of_bounds_rate=("out_of_bounds_rate", "mean"), time_limit_rate=("time_limit_rate", "mean"),
        tail_score=("tail_score", "mean"), tail_return=("tail_return", "mean"),
    )
    assert len(summary) == 75 and summary.loc[summary.algorithm != "random", "policies"].eq(3).all()
    assert models.loc[models.game == "lander", "score_mean"].sub(
        models.loc[models.game == "lander", "landed_rate"]
    ).abs().lt(1e-9).all()
    episodes.to_csv(OUTPUT / "evaluation_episodes.csv", index=False)
    models.to_csv(OUTPUT / "per_seed.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    plot(summary, models)
    bins = pd.concat(training_bins, ignore_index=True)
    bins.to_csv(OUTPUT / "training_windows_5m.csv", index=False)
    plot_training(bins)


def plot(summary, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = (
        ("snake", "score_mean", "score_sd", "Snake · apples per episode"),
        ("flap", "score_mean", "score_sd", "Flappy Bird · pipes per episode"),
        ("lander", "return_mean", "return_sd", "LunarLander · return per episode"),
        ("lander", "landed_rate", "landed_sd", "LunarLander · landing rate"),
    )
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.4), layout="constrained")
    for ax, (game, metric, sd, title) in zip(axs.flat, panels):
        sub = summary[summary.game == game]
        baseline = sub.loc[sub.algorithm == "random", metric].iloc[0]
        ax.axhline(baseline, color="#767e8a", linestyle="--", linewidth=1.2)
        for algorithm, color in COLORS.items():
            rows = sub[sub.algorithm == algorithm]
            points = models[(models.game == game) & (models.algorithm == algorithm)]
            ax.scatter(points["budget"], points[metric], s=13, color=color, alpha=.35)
            ax.errorbar(rows["budget"], rows[metric], yerr=rows[sd].fillna(0), marker="o",
                        color=color, capsize=3, linewidth=1.5, markersize=4)
        ax.set(xscale="log", xticks=BUDGETS, xlabel="Training environment steps")
        ax.set_xticklabels(["100k", "300k", "500k", "1M", "3M", "5M"])
        ax.set_title(title, loc="left", fontweight="bold")
        if metric == "landed_rate":
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", color="#e1e5ea")
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=color, marker="o", label=algorithm.upper())
               for algorithm, color in COLORS.items()]
    handles.append(Line2D([], [], color="#767e8a", linestyle="--", label="Random"))
    fig.legend(handles=handles, loc="outside upper center", ncols=5, frameon=False)
    fig.savefig(OUTPUT / "evaluation_by_budget.png", dpi=190)
    plt.close(fig)


def plot_training(bins):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    for ax, (game, metric, title) in zip(axes, (
        ("snake", "score", "Snake · apples"),
        ("flap", "score", "Flappy Bird · pipes"),
        ("lander", "episode_return", "LunarLander · return"),
    )):
        for algorithm, color in COLORS.items():
            selected = bins[(bins.game == game) & (bins.algorithm == algorithm)]
            stats = selected.groupby("window_end")[metric].agg(["mean", "std", "size"])
            assert len(stats) == 50 and stats["size"].eq(3).all()
            x = stats.index / 1_000_000
            ax.plot(x, stats["mean"], color=color, label=algorithm.upper(), linewidth=1.6)
            ax.fill_between(x, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                            color=color, alpha=.12, linewidth=0)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e1e5ea")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Training environment steps (millions)")
    axes[0].legend(loc="best", ncols=4, frameon=False)
    fig.savefig(OUTPUT / "training_curves_5m.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
