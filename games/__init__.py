"""Gymnasium environments used by the course project."""

from gymnasium.envs.registration import register

register(id="Snake-v0", entry_point="games.snake:SnakeEnv", max_episode_steps=300)
