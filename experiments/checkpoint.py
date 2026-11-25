"""Training environment scheduling and resumable snapshots."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch


class TrainingEnv(gym.Wrapper):
    def __init__(self, env, levels):
        super().__init__(env)
        self.levels = levels
        self.pool_rng = None
        self.queue = []
        self.actions = []

    def reset(self, *, seed=None, options=None):
        if self.levels:
            if self.pool_rng is None:
                self.pool_rng = np.random.default_rng(seed)
            if not self.queue:
                self.queue = self.pool_rng.permutation(self.levels).tolist()
            seed = self.queue.pop()
        self.reset_rng = self.unwrapped.np_random.bit_generator.state if seed is None else None
        self.reset_seed = seed
        self.actions = []
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        self.actions.append(action.tolist() if isinstance(action, np.ndarray) else int(action))
        return self.env.step(action)

    def snapshot(self):
        return {
            "reset_seed": self.reset_seed, "reset_rng": self.reset_rng, "actions": self.actions,
            "queue": self.queue, "pool_rng": self.pool_rng.bit_generator.state if self.pool_rng else None,
            "action_rng": self.action_space.np_random.bit_generator.state,
        }

    def restore(self, state):
        self.reset_seed, self.reset_rng = state["reset_seed"], state["reset_rng"]
        if self.reset_rng is not None:
            self.unwrapped.np_random.bit_generator.state = self.reset_rng
        obs, _ = self.env.reset(seed=self.reset_seed)
        self.actions, self.queue = state["actions"], state["queue"]
        for action in self.actions:
            if isinstance(action, list):
                action = np.asarray(action, dtype=self.action_space.dtype)
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                raise ValueError("Checkpoint episode ended before its saved position.")
        if state["pool_rng"] is not None:
            self.pool_rng = np.random.default_rng()
            self.pool_rng.bit_generator.state = state["pool_rng"]
        self.action_space.np_random.bit_generator.state = state["action_rng"]
        return obs


def level_seeds(env, count, game):
    if not count:
        return []
    if game != "snake":
        return list(range(1_000_000, 1_000_000 + count))
    # Give each initial food square a representative before adding more layouts.
    pool, extras, starts = [], [], set()
    seed = 1_000_000
    while len(pool) < min(count, env.unwrapped.size ** 2 - 3):
        env.reset(seed=seed)
        food = env.unwrapped.food
        if food in starts:
            extras.append(seed)
        else:
            starts.add(food)
            pool.append(seed)
        seed += 1
    pool.extend(extras[:count - len(pool)])
    pool.extend(range(seed, seed + count - len(pool)))
    return pool


class Checkpoints:
    def __init__(self, root, milestones, config, log, previous):
        self.root = root
        self.milestones = milestones
        self.config = config
        self.log = log
        self.previous = previous
        self.step = previous["step"] if previous else 0

    def start(self, env, seed, **objects):
        if self.previous is None:
            obs, _ = env.reset(seed=seed)
            return obs, 0.0, 0
        for name, obj in objects.items():
            state = self.previous["objects"][name]
            if isinstance(obj, np.random.Generator):
                obj.bit_generator.state = state
            elif isinstance(obj, torch.Tensor):
                with torch.no_grad():
                    obj.copy_(state)
            else:
                obj.load_state_dict(state)
        obs = env.restore(self.previous["environment"])
        torch.set_rng_state(self.previous["torch_rng"])
        return obs, self.previous["episode_return"], self.previous["length"]

    def save(self, step, env, network, episode_return, length, **objects):
        if step not in self.milestones:
            return
        state = {}
        for name, obj in objects.items():
            if isinstance(obj, np.random.Generator):
                state[name] = obj.bit_generator.state
            elif isinstance(obj, torch.Tensor):
                state[name] = obj.detach()
            else:
                state[name] = obj.state_dict()
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"step_{step}"
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination}")
        self.log.flush()
        config = dict(self.config, steps=step)
        with TemporaryDirectory(dir=self.root, prefix=".checkpoint-") as temporary:
            path = Path(temporary)
            torch.save(network.state_dict(), path / "model.pt")
            torch.save({
                "step": step, "episode_return": float(episode_return), "length": length,
                "torch_rng": torch.get_rng_state(), "environment": env.snapshot(), "objects": state,
            }, path / "state.pt")
            copyfile(self.log.name, path / "train.csv")
            (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            path.rename(destination)
        print(f"Checkpoint at {step:,} steps: {destination}", flush=True)
