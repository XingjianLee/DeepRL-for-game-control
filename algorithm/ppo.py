"""Clipped PPO with categorical actions and generalized advantage estimation."""

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from algorithm.actor_critic import ActorCriticAgent, collect_rollouts


class PPO(ActorCriticAgent):
    def train(self, env, steps, seed, on_episode):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        rollouts = collect_rollouts(env, self.network, steps, seed, 512, on_episode)
        for observations, actions, old_log_probs, advantages, returns in rollouts:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            for _ in range(4):
                for batch in torch.randperm(len(observations)).split(64):
                    logits, predicted_values = self.network(observations[batch])
                    distribution = Categorical(logits=logits)
                    ratio = (distribution.log_prob(actions[batch]) - old_log_probs[batch]).exp()
                    advantage = advantages[batch]
                    policy_loss = -torch.minimum(ratio * advantage,
                                                 ratio.clamp(0.8, 1.2) * advantage).mean()
                    value_loss = F.mse_loss(predicted_values, returns[batch])
                    loss = policy_loss + 0.5 * value_loss - 0.01 * distribution.entropy().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    optimizer.step()
