"""Replay memory and one-step targets for off-policy training."""

import numpy as np
import torch


def td_targets(reward, next_q, terminated, gamma):
    # Time-limit truncation still bootstraps from the final observation.
    return reward + gamma * next_q * (~terminated)


class ReplayBuffer:
    def __init__(self, observation_shape, capacity, *, action_shape=()):
        self.capacity = capacity
        self.count = 0
        self.observations = np.empty((capacity, *observation_shape), np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty((capacity, *action_shape), np.float32 if action_shape else np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.terminals = np.empty(capacity, np.bool_)

    def add(self, obs, action, reward, next_obs, terminated):
        slot = self.count % self.capacity
        self.observations[slot], self.next_observations[slot] = obs, next_obs
        self.actions[slot], self.rewards[slot], self.terminals[slot] = action, reward, terminated
        self.count += 1

    def sample(self, rng, batch_size):
        indices = rng.integers(min(self.count, self.capacity), size=batch_size)
        return (torch.from_numpy(self.observations[indices]),
                torch.from_numpy(self.actions[indices]),
                torch.from_numpy(self.rewards[indices]),
                torch.from_numpy(self.next_observations[indices]),
                torch.from_numpy(self.terminals[indices]))
