"""Save an isolated run directory and its per-episode training data."""

import csv
from datetime import datetime, timezone
from fcntl import flock, LOCK_EX
from importlib.metadata import version
import json
from pathlib import Path
from shutil import copyfile, rmtree
from tempfile import TemporaryDirectory

import gymnasium
import numpy as np
import torch

from experiments.checkpoint import Checkpoints, TrainingEnv, level_seeds
from experiments.evaluate import make_agent, make_env
from games.lander import LANDER_GAMES
from games.snake import FoodDistanceReward

A2C_PROFILES = ("baseline", "improved")
DEFAULT_CHECKPOINTS = (500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000)


def train(game, algorithm, seed, steps, output_root, *, a2c_profile=None, overwrite=False,
          max_episode_steps=None, level_pool_size=None, checkpoint_steps=None, resume=None):
    resume = Path(resume) if resume else None
    previous = json.loads((resume / "config.json").read_text()) if resume else None
    if previous:
        if (game, algorithm, seed) != tuple(previous[key] for key in ("game", "algorithm", "seed")):
            raise ValueError("Resume must use the same game, algorithm, and training seed.")
        for key, value in (("max_episode_steps", max_episode_steps), ("level_pool_size", level_pool_size),
                           ("a2c_profile", a2c_profile)):
            if value is not None and value != previous.get(key):
                raise ValueError(f"Resume must keep {key} from the checkpoint.")
        max_episode_steps = previous.get("max_episode_steps")
        level_pool_size = previous.get("level_pool_size")
        a2c_profile = previous.get("a2c_profile")
        if steps <= previous["steps"]:
            raise ValueError("The total step budget must exceed the checkpoint step.")
    if a2c_profile is not None and (algorithm != "a2c" or a2c_profile not in A2C_PROFILES):
        raise ValueError("An A2C profile must be baseline or improved and requires A2C.")
    if game == "lander-continuous" and algorithm != "sac":
        raise ValueError("Continuous Lunar Lander requires SAC.")
    if max_episode_steps is not None and max_episode_steps <= 0:
        raise ValueError("Episode limit must be positive.")
    if level_pool_size is not None and level_pool_size <= 0:
        raise ValueError("Level pool size must be positive.")
    if checkpoint_steps is None:
        checkpoint_steps = DEFAULT_CHECKPOINTS if (max_episode_steps or level_pool_size or resume) else ()
    start = previous["steps"] if resume else 0
    milestones = {step for step in checkpoint_steps if start < step <= steps}
    if resume:
        milestones.add(steps)
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
            existing = json.loads((run / "config.json").read_text())
            if any(existing.get(key) != value for key, value in (
                    ("max_episode_steps", max_episode_steps), ("level_pool_size", level_pool_size),
                    ("resume_from_step", start if resume else None))):
                raise ValueError(f"Different training settings exist at {run}. Use a separate --output-root.")
            print(f"Skipping existing result: {run}", flush=True)
            return run

        with TemporaryDirectory(prefix=f".{run.name}.training-", dir=run.parent) as staging:
            result = Path(staging)
            checkpoint_root = run.with_name(f"{run.name}_checkpoints")
            conflicts = [step for step in sorted(milestones) if (checkpoint_root / f"step_{step}").exists()]
            if conflicts:
                raise FileExistsError(f"Checkpoints already exist at {checkpoint_root}: {conflicts}")
            train_into(result, game, algorithm, seed, steps, a2c_profile,
                       max_episode_steps, level_pool_size, milestones, checkpoint_root, resume, previous)
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


def train_into(run, game, algorithm, seed, steps, a2c_profile, max_episode_steps,
               level_pool_size, milestones, checkpoint_root, resume, previous):
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
    env_options = {} if max_episode_steps is None else {"max_episode_steps": max_episode_steps}
    env = make_env(game, **env_options)
    if improved and game == "snake":
        env = FoodDistanceReward(env)
    try:
        levels = previous["level_seeds"] if resume else level_seeds(env, level_pool_size, game)
        if max_episode_steps or level_pool_size or milestones or resume:
            config.update(
                max_episode_steps=max_episode_steps, level_pool_size=level_pool_size, level_seeds=levels,
                schedule_steps=previous["schedule_steps"] if resume else steps,
                resume_from_step=previous["steps"] if resume else None,
            )
        if levels or milestones or resume:
            env = TrainingEnv(env, levels)
        agent = make_agent(algorithm, env)
        state = None
        log_path = run / "train.csv"
        if resume:
            state = torch.load(resume / "state.pt", map_location="cpu", weights_only=True)
            if state["step"] != previous["steps"]:
                raise ValueError("Checkpoint metadata disagrees with its saved step.")
            agent.network.load_state_dict(torch.load(resume / "model.pt", map_location="cpu", weights_only=True))
            copyfile(resume / "train.csv", log_path)
        with log_path.open("a" if resume else "w", newline="") as file:
            writer = csv.writer(file)
            if not resume:
                writer.writerow(("step", "length", "return", "score"))

            def log_episode(step, length, episode_return, score):
                writer.writerow((step, length, episode_return, score))

            options = {"tuned": True} if improved and game == "flap" else {}
            if milestones or resume:
                options["checkpoints"] = Checkpoints(checkpoint_root, milestones, config, file, state)
            agent.train(env, steps, seed, log_episode, **options)
        torch.save(agent.network.state_dict(), run / "model.pt")
    finally:
        env.close()
    config["completed_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
