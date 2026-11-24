"""Advantage Actor-Critic with one on-policy update per rollout."""

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from algorithm.actor_critic import ActorCriticAgent, collect_rollouts


class A2C(ActorCriticAgent):
    def train(self, env, steps, seed, on_episode, *, tuned=False):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=5e-4 if tuned else 3e-4)
        rollouts = collect_rollouts(env, self.network, steps, seed, 32, on_episode)
        processed = 0
        entropy_weight = 0.01
        for observations, actions, _, advantages, returns in rollouts:
            if tuned:
                progress = processed / steps
                optimizer.param_groups[0]["lr"] = 5e-4 - 4e-4 * progress
                entropy_weight = 0.03 - 0.025 * progress
                if len(actions) > 1:
                    scale = advantages.std(unbiased=False) + 1e-8
                    advantages = (advantages - advantages.mean()) / scale
                processed += len(actions)
            logits, values = self.network(observations)
            distribution = Categorical(logits=logits)
            policy_loss = -(distribution.log_prob(actions) * advantages).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + 0.5 * value_loss - entropy_weight * distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            optimizer.step()
