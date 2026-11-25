"""Independent seeded evaluation of trained and reference policies."""

import csv
from collections import Counter
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import games  # Registers Snake-v0.
from algorithm.a2c import A2C
from algorithm.continuous_sac import ContinuousSAC
from algorithm.dqn import DQN
from algorithm.ppo import PPO
from algorithm.sac import SAC
from games.flap import ENV_ID as FLAP_ENV_ID, make_flap
from games.lander import CONTINUOUS_ENV_ID, ENV_ID as LANDER_ENV_ID, LANDER_GAMES, make_lander

ENVIRONMENTS = {
    "snake": "Snake-v0", "flap": FLAP_ENV_ID,
    "lander": LANDER_ENV_ID, "lander-continuous": CONTINUOUS_ENV_ID,
}
ALGORITHMS = {"dqn": DQN, "ppo": PPO, "a2c": A2C, "sac": SAC}


def make_env(game, **kwargs):
    if game == "flap":
        return make_flap(**kwargs)
    if game in LANDER_GAMES:
        return make_lander(continuous=game == "lander-continuous", **kwargs)
    return gym.make(ENVIRONMENTS[game], **kwargs)


def make_agent(algorithm, env):
    if isinstance(env.action_space, gym.spaces.Box):
        if algorithm != "sac":
            raise ValueError("Continuous actions require SAC.")
        return ContinuousSAC(env.observation_space.shape[0], env.action_space)
    return ALGORITHMS[algorithm](env.observation_space.shape[0], env.action_space.n)


def load_agent(run, env):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    if env.spec.id != ENVIRONMENTS[config["game"]]:
        raise ValueError("Model and environment refer to different games.")
    agent = make_agent(config["algorithm"], env)
    agent.network.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
    agent.network.eval()
    return agent


def evaluate(run, episodes, seed, *, game="snake", output=None, max_episode_steps=None):
    options = {} if max_episode_steps is None else {"max_episode_steps": max_episode_steps}
    env = make_env(game, **options)
    rows = []
    try:
        agent = load_agent(run, env) if run is not None else None
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        for episode_seed in range(seed, seed + episodes):
            obs, info = env.reset(seed=episode_seed)
            rng = np.random.default_rng(episode_seed)
            if agent is None and not discrete:
                env.action_space.seed(episode_seed)
            reward_sum, steps = 0.0, 0
            while True:
                if agent is not None:
                    action = agent.predict(obs)
                elif discrete:
                    action = int(rng.integers(env.action_space.n))
                else:
                    action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
                steps += 1
                if terminated or truncated:
                    rows.append({
                        "seed": episode_seed, "score": info["score"], "return": reward_sum,
                        "steps": steps,
                        "reason": info.get("reason") or ("collision" if terminated else "time_limit"),
                    })
                    break
    finally:
        env.close()

    if output is not None:
        with Path(output).open("x", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    scores = np.array([row["score"] for row in rows])
    label = str(run) if run is not None else "random"
    if game in LANDER_GAMES:
        returns = np.array([row["return"] for row in rows])
        outcomes = Counter(row["reason"] for row in rows)
        print(f"{label} | {episodes} episodes | return {returns.mean():.2f} ± {returns.std():.2f}"
              f" | landed {scores.mean():.1%} | crashed {outcomes['crash'] / episodes:.1%}"
              f" | out of bounds {outcomes['out_of_bounds'] / episodes:.1%}"
              f" | time limit {outcomes['time_limit'] / episodes:.1%}")
    else:
        print(f"{label} | {episodes} episodes | score {scores.mean():.2f} ± {scores.std():.2f}"
              f" | max {scores.max()}")
    return rows
