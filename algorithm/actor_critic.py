"""Categorical actor-critic network, trajectory sampling and GAE targets."""

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, observation_size, action_count):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(observation_size, 64), nn.Tanh(),
                                      nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, action_count)
        self.critic = nn.Linear(64, 1)

    def forward(self, observation):
        features = self.features(observation)
        return self.actor(features), self.critic(features).squeeze(-1)


def generalized_advantage(rewards, values, next_values, terminated, episode_ended,
                          gamma=0.99, gae_lambda=0.95):
    deltas = rewards + gamma * next_values * (~terminated) - values
    advantages = torch.empty_like(deltas)
    carry = 0.0
    for index in range(len(deltas) - 1, -1, -1):
        # Truncation permits bootstrap but the next episode's advantage cannot leak in.
        carry = deltas[index] + gamma * gae_lambda * carry * (~episode_ended[index])
        advantages[index] = carry
    return advantages


class ActorCriticAgent:
    def __init__(self, observation_size, action_count):
        self.network = ActorCritic(observation_size, action_count)

    @torch.no_grad()
    def predict(self, observation):
        logits, _ = self.network(torch.from_numpy(observation))
        return int(logits.argmax())


def collect_rollouts(env, network, steps, seed, rollout_steps, on_episode):
    """Yield sampled actions, saved policy log-probabilities and detached targets."""
    obs, _ = env.reset(seed=seed)
    episode_return, length = 0.0, 0
    for start in range(0, steps, rollout_steps):
        rollout = []
        for step in range(start + 1, min(start + rollout_steps, steps) + 1):
            with torch.no_grad():
                logits, value = network(torch.from_numpy(obs))
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            ended = terminated or truncated
            rollout.append((torch.from_numpy(obs), torch.from_numpy(next_obs),
                            action, log_prob, value, reward, terminated, ended))
            obs = next_obs
            episode_return += reward
            length += 1
            if ended:
                on_episode(step, length, episode_return, info["score"])
                obs, _ = env.reset()
                episode_return, length = 0.0, 0

        (obs_batch, next_obs_batch, action_batch, log_prob_batch, value_batch,
         reward_batch, terminal_batch, end_batch) = zip(*rollout)
        observations = torch.stack(obs_batch)
        next_observations = torch.stack(next_obs_batch)
        actions = torch.stack(action_batch)
        old_log_probs = torch.stack(log_prob_batch)
        values = torch.stack(value_batch)
        rewards = torch.tensor(reward_batch, dtype=torch.float32)
        terminals = torch.tensor(terminal_batch, dtype=torch.bool)
        episode_ends = torch.tensor(end_batch, dtype=torch.bool)
        with torch.no_grad():
            _, next_values = network(next_observations)
            advantages = generalized_advantage(rewards, values, next_values,
                                               terminals, episode_ends)
            returns = advantages + values
        yield observations, actions, old_log_probs, advantages, returns
