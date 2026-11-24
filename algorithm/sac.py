"""Discrete Soft Actor-Critic with a categorical policy and twin Q critics."""

from copy import deepcopy
from math import log

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from algorithm.replay import ReplayBuffer, td_targets


def action_network(observation_size, action_count):
    return nn.Sequential(
        nn.Linear(observation_size, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_count),
    )


def soft_value(log_probs, q_values, temperature):
    return (log_probs.exp() * (q_values - temperature * log_probs)).sum(dim=-1)


class SAC:
    def __init__(self, observation_size, action_count):
        self.network = action_network(observation_size, action_count)

    @torch.no_grad()
    def predict(self, observation):
        return int(self.network(torch.from_numpy(observation)).argmax())

    def train(self, env, steps, seed, on_episode):
        rng = np.random.default_rng(seed)
        size, action_count = env.observation_space.shape[0], env.action_space.n
        critics = nn.ModuleList(action_network(size, action_count) for _ in range(2))
        targets = deepcopy(critics).requires_grad_(False)
        actor_optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(critics.parameters(), lr=3e-4)
        log_temperature = nn.Parameter(torch.tensor(log(0.2), dtype=torch.float32))
        temperature_optimizer = torch.optim.Adam([log_temperature], lr=3e-4)
        target_entropy = 0.3 * log(action_count)

        replay = ReplayBuffer(env.observation_space.shape, 50_000)
        obs, _ = env.reset(seed=seed)
        episode_return, length = 0.0, 0
        for step in range(1, steps + 1):
            if step <= 1_000:
                action = int(rng.integers(action_count))
            else:
                with torch.no_grad():
                    logits = self.network(torch.from_numpy(obs))
                    action = int(Categorical(logits=logits).sample())

            next_obs, reward, terminated, truncated, info = env.step(action)
            replay.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_return += reward
            length += 1

            if step >= 1_000 and step % 4 == 0:
                states, actions, rewards, next_states, terminals = replay.sample(rng, 64)
                with torch.no_grad():
                    temperature = log_temperature.exp()
                    next_log_probs = self.network(next_states).log_softmax(dim=-1)
                    next_q = torch.minimum(targets[0](next_states), targets[1](next_states))
                    expected_q = td_targets(
                        rewards, soft_value(next_log_probs, next_q, temperature), terminals, 0.99,
                    )
                critic_loss = sum(
                    F.smooth_l1_loss(critic(states).gather(1, actions[:, None]).squeeze(1), expected_q)
                    for critic in critics
                )
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                with torch.no_grad():
                    min_q = torch.minimum(critics[0](states), critics[1](states))
                log_probs = self.network(states).log_softmax(dim=-1)
                actor_loss = -soft_value(log_probs, min_q, temperature).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                with torch.no_grad():
                    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
                temperature_loss = log_temperature * (entropy - target_entropy)
                temperature_optimizer.zero_grad()
                temperature_loss.backward()
                temperature_optimizer.step()

                with torch.no_grad():
                    for target_param, critic_param in zip(targets.parameters(), critics.parameters()):
                        target_param.lerp_(critic_param, 0.005)

            if terminated or truncated:
                on_episode(step, length, episode_return, info["score"])
                obs, _ = env.reset()
                episode_return, length = 0.0, 0
