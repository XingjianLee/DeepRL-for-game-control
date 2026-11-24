"""Expose Gymnasium's Lunar Lander outcomes through the project's score field."""

import gymnasium as gym

ENV_ID = "LunarLander-v3"
CONTINUOUS_ENV_ID = "LunarLanderContinuous-v3"
LANDER_GAMES = ("lander", "lander-continuous")


class LandingOutcome(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, {**info, "score": 0}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        landed = terminated and reward == 100
        reason = None
        if landed:
            reason = "landed"
        elif terminated:
            reason = "out_of_bounds" if abs(obs[0]) >= 1.0 else "crash"
        elif truncated:
            reason = "time_limit"
        return obs, reward, terminated, truncated, {**info, "score": int(landed), "reason": reason}


def make_lander(*, continuous=False, **kwargs):
    env_id = CONTINUOUS_ENV_ID if continuous else ENV_ID
    return LandingOutcome(gym.make(env_id, continuous=continuous, **kwargs))
