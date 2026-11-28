"""Read the existing training and evaluation CSV files for the dashboard."""

import csv
from dataclasses import dataclass
from functools import cached_property
import json
from math import isfinite
from pathlib import Path
from statistics import fmean

from games.lander import LANDER_GAMES

EVALUATIONS = (
    ("heldout.csv", "独立测试"),
    ("evaluation.csv", "保存的评估"),
    ("validation.csv", "验证集"),
)


def csv_rows(path, fields):
    try:
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                try:
                    values = tuple(converter(row[name]) for name, converter in fields)
                except (KeyError, TypeError, ValueError):
                    continue
                if all(isfinite(value) for value in values):
                    yield values
    except (OSError, UnicodeError, csv.Error):
        return


@dataclass(frozen=True)
class Evaluation:
    filename: str
    label: str
    rows: tuple

    @cached_property
    def seeds(self):
        return tuple(sorted(seed for seed, _, _ in self.rows))

    @cached_property
    def score(self):
        return fmean(score for _, score, _ in self.rows)

    @cached_property
    def reward(self):
        return fmean(reward for _, _, reward in self.rows)

    @cached_property
    def best_score(self):
        return max(score for _, score, _ in self.rows)

    @cached_property
    def best_return(self):
        return max(reward for _, _, reward in self.rows)


@dataclass(frozen=True)
class Run:
    path: Path
    game: str
    algorithm: str
    seed: int
    steps: int
    shaped: bool
    completed_at: str
    max_episode_steps: int | None = None
    snake_spatial: bool = False

    @cached_property
    def training(self):
        metric = "return" if self.game in LANDER_GAMES else "score"
        fields = (("step", int), (metric, float))
        return tuple((step, value) for step, value in csv_rows(self.path / "train.csv", fields)
                     if 0 < step <= self.steps)

    @cached_property
    def curve(self):
        bins = [[] for _ in range(32)]
        for step, value in self.training:
            bins[(step - 1) * len(bins) // self.steps].append((step, value))
        return tuple((fmean(step for step, _ in bucket), fmean(value for _, value in bucket))
                     for bucket in bins if bucket)

    @cached_property
    def evaluations(self):
        fields = (("seed", int), ("score", float), ("return", float))
        datasets = []
        for filename, label in EVALUATIONS:
            rows = tuple(csv_rows(self.path / filename, fields))
            if rows:
                datasets.append(Evaluation(filename, label, rows))
        return tuple(datasets)


def load_runs(root, environments, algorithms):
    runs = []
    for path in root.glob("*/*/step_*/seed_*/config.json"):
        if not path.with_name("model.pt").is_file():
            continue
        try:
            config = json.loads(path.read_text())
            game, algorithm = config["game"], config["algorithm"]
            seed, steps = int(config["seed"]), int(config["steps"])
            completed_at = config["completed_at"]
            if (steps < 1 or game not in environments or algorithm not in algorithms
                    or not isinstance(completed_at, str)):
                continue
            relative = path.parent.relative_to(root).parts
            if relative != (game, algorithm, f"step_{steps}", f"seed_{seed}"):
                continue
            shaped = game == "snake" and algorithm == "a2c" and config.get("a2c_profile") == "improved"
            limit = config.get("max_episode_steps")
            if type(limit) is not int or limit <= 0:
                limit = None
            runs.append(Run(path.parent, game, algorithm, seed, steps, shaped, completed_at, limit,
                            game == "snake" and config.get("snake_spatial") is True))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(runs, key=lambda run: run.completed_at, reverse=True)


def comparable_runs(runs, selected, evaluation):
    """Group compatible saved runs by algorithm while retaining one run per training seed."""
    groups = {}
    for run in runs:
        if (run.game, run.steps, run.shaped, run.snake_spatial) != (
                selected.game, selected.steps, selected.shaped, selected.snake_spatial):
            continue
        for candidate in run.evaluations:
            if (candidate.filename, candidate.seeds) == (evaluation.filename, evaluation.seeds):
                by_seed = groups.setdefault(run.algorithm, {})
                if run is selected or run.seed not in by_seed:
                    by_seed[run.seed] = candidate
                break
    return groups
