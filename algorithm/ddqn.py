"""Double DQN with separate action selection and value estimation."""

from algorithm.dqn import DQN


class DoubleDQN(DQN):
    def next_q_values(self, next_states, target):
        next_actions = self.network(next_states).argmax(dim=1, keepdim=True)
        return target(next_states).gather(1, next_actions).squeeze(1)
