"""Adapt the third-party Flappy Bird environment to the project policies."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import HumanRendering, TransformObservation

ENV_ID = "FlappyBird-v0"
SCREEN_SIZE = (288, 512)


def make_flap(*, render_mode=None, max_episode_steps=3_600):
    # Upstream human rendering consumes the Pygame events needed by our controls.
    upstream_mode = "rgb_array" if render_mode == "human" else render_mode
    env = gym.make(
        f"flappy_bird_gymnasium:{ENV_ID}", use_lidar=False, screen_size=SCREEN_SIZE,
        render_mode=upstream_mode, max_episode_steps=max_episode_steps,
    )
    space = env.observation_space
    float32_space = Box(space.low.astype(np.float32), space.high.astype(np.float32))
    env = TransformObservation(env, lambda obs: obs.astype(np.float32), float32_space)
    if render_mode == "human":
        env = HumanRendering(env)
    return env
