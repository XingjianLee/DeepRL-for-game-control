"""DQN with a replay buffer and a periodically synchronized target network."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from algorithm.replay import ReplayBuffer, td_targets


class DQN:
    def __init__(self, observation_size, action_count):
        self.network = nn.Sequential(
            nn.Linear(observation_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_count),
        )

    @torch.no_grad()
    def predict(self, observation):
        return int(self.network(torch.from_numpy(observation)).argmax())

    def next_q_values(self, next_states, target):
        return target(next_states).amax(dim=1)

    def train(self, env, steps, seed, on_episode, *, checkpoints=None):
        rng = np.random.default_rng(seed)
        target = deepcopy(self.network)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        capacity, batch_size, warmup = 50_000, 64, 1_000
        replay = ReplayBuffer(env.observation_space.shape, capacity)

        objects = dict(rng=rng, replay=replay, target=target, optimizer=optimizer)
        if checkpoints:
            obs, episode_return, length = checkpoints.start(env, seed, **objects)
            start = checkpoints.step
        else:
            obs, _ = env.reset(seed=seed)
            start, episode_return, length = 0, 0.0, 0
        schedule_steps = checkpoints.config["schedule_steps"] if checkpoints else steps
        for step in range(start + 1, steps + 1):
            epsilon = max(0.05, 1.0 - 0.95 * step / max(warmup, schedule_steps * 0.5))
            if step <= warmup or rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                action = self.predict(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            replay.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_return += reward
            length += 1

            if step >= warmup:
                states, actions, rewards, next_states, terminals = replay.sample(rng, batch_size)
                chosen_q = self.network(states).gather(1, actions[:, None]).squeeze(1)
                with torch.no_grad():
                    next_q = self.next_q_values(next_states, target)
                    expected_q = td_targets(rewards, next_q, terminals, 0.99)
                loss = F.smooth_l1_loss(chosen_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
                optimizer.step()
                if step % 500 == 0:
                    target.load_state_dict(self.network.state_dict())

            if terminated or truncated:
                on_episode(step, length, episode_return, info["score"])
                obs, _ = env.reset()
                episode_return, length = 0.0, 0
            if checkpoints:
                checkpoints.save(step, env, self.network, episode_return, length, **objects)
