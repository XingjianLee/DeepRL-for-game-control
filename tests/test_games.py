from importlib.util import find_spec
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import games
from experiments.evaluate import make_env
from games.lander import LandingOutcome
from games.snake import FoodDistanceReward, SnakeEnv


class LanderTests(unittest.TestCase):
    def test_landing_outcomes_preserve_reward_and_episode_flags(self):
        env = LandingOutcome(gym.Env())
        for x, reward, terminated, truncated, score, reason in (
            (0.0, 100.0, True, False, 1, "landed"),
            (0.9, -100.0, True, False, 0, "crash"),
            (1.1, -100.0, True, False, 0, "out_of_bounds"),
            (0.0, 2.0, False, True, 0, "time_limit"),
        ):
            step = (np.array([x], dtype=np.float32), reward, terminated, truncated, {})
            with self.subTest(reason=reason), patch.object(env.env, "step", return_value=step):
                _, actual_reward, ended, limited, info = env.step(0)
                self.assertEqual((actual_reward, ended, limited), (reward, terminated, truncated))
                self.assertEqual(info, {"score": score, "reason": reason})

    @unittest.skipUnless(find_spec("Box2D"), "Box2D package is optional")
    def test_official_lander_contract(self):
        env = make_env("lander", render_mode="rgb_array")
        try:
            obs, info = env.reset(seed=42)
            np.testing.assert_array_equal(obs, env.reset(seed=42)[0])
            self.assertEqual(info["score"], 0)
            self.assertEqual(obs.shape, (8,))
            self.assertEqual(env.action_space.n, 4)
            self.assertEqual(env.render().shape, (400, 600, 3))
            for _ in range(1_000):
                _, reward, terminated, truncated, info = env.step(0)
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertEqual(reward, -100.0)
            self.assertIn(info["reason"], ("crash", "out_of_bounds"))
        finally:
            env.close()


class SnakeTests(unittest.TestCase):
    def test_spatial_observation_respects_walls_and_growth(self):
        env = SnakeEnv(spatial_features=True)
        obs, _ = env.reset(seed=0)
        self.assertTrue(env.observation_space.contains(obs))
        env.heading = 0
        env.body = [(1, y) for y in range(10)] + [(2, 9)]
        env.food = (2, 0)
        # Left and right are isolated by the body. Only right eats and keeps the old tail.
        np.testing.assert_allclose(env._observation()[107:113],
                                   [0, 0, 9 / 99, 1, 78 / 99, 1])

    def test_food_distance_shaping_keeps_potential_on_time_limit(self):
        base = gym.make("Snake-v0", max_episode_steps=1)
        env = FoodDistanceReward(base)
        try:
            with patch.object(base.unwrapped, "_place_food", return_value=(8, 5)):
                env.reset(seed=42)
                _, reward, terminated, truncated, info = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["score"], 0)
            self.assertAlmostEqual(reward, 0.5 * (3 - 0.99 * 2) / 9)
        finally:
            env.close()

    def test_gymnasium_and_render_contract(self):
        env = SnakeEnv(render_mode="rgb_array")
        try:
            check_env(env, skip_render_check=True)
            self.assertEqual(env.render().shape, (360, 320, 3))
        finally:
            env.close()

    def test_seed_and_external_time_limit(self):
        env = gym.make("Snake-v0", max_episode_steps=1)
        try:
            obs, _ = env.reset(seed=17)
            other, _ = env.reset(seed=17)
            np.testing.assert_array_equal(obs, other)
            _, _, terminated, truncated, _ = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
        finally:
            env.close()

    def test_growth_and_wall_collision(self):
        env = SnakeEnv()
        env.reset(seed=0)
        env.food = (6, 5)
        _, reward, terminated, _, info = env.step(0)
        self.assertEqual((reward, terminated, info["score"], len(env.body)), (1.0, False, 1, 4))
        self.assertNotIn(env.food, env.body)
        env.body = [(9, 5), (8, 5), (7, 5)]
        _, reward, terminated, _, info = env.step(0)
        self.assertEqual((reward, terminated, info["reason"]), (-1.0, True, "wall"))

    def test_body_collision_and_moving_tail(self):
        env = SnakeEnv()
        env.reset()
        env.heading = 0
        env.food = (0, 0)
        env.body = [(5, 5), (4, 5), (4, 4), (5, 4), (6, 4)]
        _, _, terminated, _, info = env.step(0)
        self.assertTrue(terminated)
        self.assertEqual(info["reason"], "body")

        env.body.pop()
        _, _, terminated, _, _ = env.step(0)
        self.assertFalse(terminated)
        self.assertEqual(env.body[0], (5, 4))


@unittest.skipUnless(find_spec("flappy_bird_gymnasium"), "Flappy Bird package is optional")
class FlapTests(unittest.TestCase):
    def test_adapter_preserves_seeded_observations_and_game_score(self):
        env = make_env("flap", render_mode="rgb_array")
        try:
            obs, info = env.reset(seed=42)
            np.testing.assert_array_equal(obs, env.reset(seed=42)[0])
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(info["score"], 0)
            self.assertEqual(env.render().shape, (512, 288, 3))
            for _ in range(100):
                obs, _, terminated, truncated, info = env.step(0)
                self.assertTrue(env.observation_space.contains(obs))
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertEqual(info["score"], 0)
        finally:
            env.close()
