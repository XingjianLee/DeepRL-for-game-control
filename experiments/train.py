"""Save an isolated run directory and its per-episode training data."""

import csv
from datetime import datetime, timezone
from fcntl import flock, LOCK_EX
from importlib.metadata import version
import json
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory

import gymnasium
import numpy as np
import torch

from experiments.evaluate import make_agent, make_env
from games.lander import LANDER_GAMES
from games.snake import FoodDistanceReward

A2C_PROFILES = ("baseline", "improved")


def train(game, algorithm, seed, steps, output_root, *, a2c_profile=None, overwrite=False):
    if a2c_profile is not None and (algorithm != "a2c" or a2c_profile not in A2C_PROFILES):
        raise ValueError("An A2C profile must be baseline or improved and requires A2C.")
    if game == "lander-continuous" and algorithm != "sac":
        raise ValueError("Continuous Lunar Lander requires SAC.")
    run = Path(output_root) / game / algorithm / f"step_{steps}" / f"seed_{seed}"
    run.parent.mkdir(parents=True, exist_ok=True)
    backup = run.with_name(f".{run.name}.backup")
    with run.with_name(f".{run.name}.lock").open("a") as lock:
        flock(lock, LOCK_EX)
        if backup.exists():
            if run.exists():
                rmtree(backup)
            else:
                backup.rename(run)
        if not overwrite and all((run / name).is_file()
                                 for name in ("config.json", "train.csv", "model.pt")):
            print(f"Skipping existing result: {run}", flush=True)
            return run

        with TemporaryDirectory(prefix=f".{run.name}.training-", dir=run.parent) as staging:
            result = Path(staging)
            train_into(result, game, algorithm, seed, steps, a2c_profile)
            if run.exists():
                run.rename(backup)
            try:
                result.rename(run)
            except BaseException:
                if backup.exists() and not run.exists():
                    backup.rename(run)
                raise
            if backup.exists():
                rmtree(backup)
    print(f"Saved {steps:,} steps to {run}")
    return run


def train_into(run, game, algorithm, seed, steps, a2c_profile):
    improved = a2c_profile == "improved"
    config = {
        "game": game, "algorithm": algorithm, "seed": seed, "steps": steps,
        "device": "cpu", "torch": str(torch.__version__),
        "numpy": np.__version__, "gymnasium": gymnasium.__version__,
    }
    if game == "flap":
        config["flappy_bird_gymnasium"] = version("flappy-bird-gymnasium")
    if game in LANDER_GAMES:
        config["box2d"] = version("Box2D")
    if algorithm == "a2c":
        config["a2c_profile"] = a2c_profile or "baseline"
    torch.manual_seed(seed)
    env = make_env(game)
    if improved and game == "snake":
        env = FoodDistanceReward(env)
    try:
        agent = make_agent(algorithm, env)
        with (run / "train.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("step", "length", "return", "score"))

            def log_episode(step, length, episode_return, score):
                writer.writerow((step, length, episode_return, score))

            options = {"tuned": True} if improved and game == "flap" else {}
            agent.train(env, steps, seed, log_episode, **options)
        torch.save(agent.network.state_dict(), run / "model.pt")
    finally:
        env.close()
    config["completed_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
