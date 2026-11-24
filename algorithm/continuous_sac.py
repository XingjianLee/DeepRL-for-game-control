"""Soft Actor-Critic for bounded, continuous action vectors."""

from copy import deepcopy
from math import log

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from algorithm.replay import ReplayBuffer, td_targets


def mlp(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, output_size),
    )


class SquashedGaussian(nn.Module):
    def __init__(self, observation_size, action_low, action_high):
        super().__init__()
        self.layers = mlp(observation_size, 2 * len(action_low))
        self.register_buffer("scale", torch.as_tensor((action_high - action_low) / 2))
        self.register_buffer("bias", torch.as_tensor((action_high + action_low) / 2))

    def forward(self, observations):
        mean, _ = self.layers(observations).chunk(2, dim=-1)
        return mean.tanh() * self.scale + self.bias

    def sample(self, observations):
        mean, log_std = self.layers(observations).chunk(2, dim=-1)
        distribution = Normal(mean, log_std.clamp(-20.0, 2.0).exp())
        raw = distribution.rsample()
        action = raw.tanh() * self.scale + self.bias
        # Stable log of the tanh derivative, including the affine action scaling.
        log_jacobian = 2 * (log(2.0) - raw - F.softplus(-2 * raw)) + self.scale.log()
        log_prob = (distribution.log_prob(raw) - log_jacobian).sum(dim=-1)
        return action, log_prob


class ContinuousSAC:
    def __init__(self, observation_size, action_space):
        self.network = SquashedGaussian(observation_size, action_space.low, action_space.high)

    @torch.no_grad()
    def predict(self, observation):
        return self.network(torch.from_numpy(observation)).numpy()

    def train(self, env, steps, seed, on_episode):
        rng = np.random.default_rng(seed)
        env.action_space.seed(seed)
        action_size = env.action_space.shape[0]
        input_size = env.observation_space.shape[0] + action_size
        critics = nn.ModuleList(mlp(input_size, 1) for _ in range(2))
        targets = deepcopy(critics).requires_grad_(False)
        actor_optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(critics.parameters(), lr=3e-4)
        log_temperature = nn.Parameter(torch.tensor(log(0.2), dtype=torch.float32))
        temperature_optimizer = torch.optim.Adam([log_temperature], lr=3e-4)
        target_entropy = -float(action_size)
        replay = ReplayBuffer(env.observation_space.shape, 100_000,
                              action_shape=env.action_space.shape)

        obs, _ = env.reset(seed=seed)
        episode_return, length = 0.0, 0
        for step in range(1, steps + 1):
            if step <= 1_000:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _ = self.network.sample(torch.from_numpy(obs))
                    action = action.numpy()

            next_obs, reward, terminated, truncated, info = env.step(action)
            replay.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_return += reward
            length += 1

            if step >= 1_000 and step % 2 == 0:
                states, actions, rewards, next_states, terminals = replay.sample(rng, 128)
                with torch.no_grad():
                    temperature = log_temperature.exp()
                    next_actions, next_log_probs = self.network.sample(next_states)
                    next_inputs = torch.cat((next_states, next_actions), dim=-1)
                    next_q = torch.minimum(targets[0](next_inputs), targets[1](next_inputs)).squeeze(-1)
                    expected_q = td_targets(rewards, next_q - temperature * next_log_probs,
                                            terminals, 0.99)
                inputs = torch.cat((states, actions), dim=-1)
                critic_loss = sum(F.mse_loss(critic(inputs).squeeze(-1), expected_q)
                                  for critic in critics)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                critics.requires_grad_(False)
                sampled_actions, log_probs = self.network.sample(states)
                sampled_inputs = torch.cat((states, sampled_actions), dim=-1)
                min_q = torch.minimum(critics[0](sampled_inputs), critics[1](sampled_inputs)).squeeze(-1)
                actor_loss = (temperature * log_probs - min_q).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                critics.requires_grad_(True)

                temperature_loss = -log_temperature * (log_probs.detach().mean() + target_entropy)
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
