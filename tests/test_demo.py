import unittest
from collections import defaultdict
from unittest.mock import Mock, patch

import pygame
import numpy as np

from games.snake import SnakeEnv
from ui.demo import play


class DemoTests(unittest.TestCase):
    def test_continuous_lander_commands_both_thrusters(self):
        env = Mock()
        env.reset.return_value = (None, {"score": 0})
        env.step.return_value = (None, 0.0, False, False, {"score": 0})
        frames = [[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)], [],
                  [pygame.event.Event(pygame.QUIT)]]
        key_states = [defaultdict(bool, {pygame.K_UP: True, pygame.K_LEFT: True}),
                      defaultdict(bool)]
        with (
            patch("ui.demo.make_env", return_value=env),
            patch("pygame.display.set_caption"),
            patch("pygame.event.get", side_effect=frames),
            patch("pygame.key.get_pressed", side_effect=key_states),
        ):
            play("lander-continuous", None, 42)
        np.testing.assert_array_equal([call.args[0] for call in env.step.call_args_list],
                                      [[1, -1], [-1, 0]])

    def test_lander_uses_held_thrusters_and_waits_to_start(self):
        env = Mock()
        env.reset.return_value = (None, {"score": 0})
        env.step.return_value = (None, 0.0, False, False, {"score": 0})
        up = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)
        quit_event = pygame.event.Event(pygame.QUIT)
        frames = [[], [up], [], [], [quit_event]]
        key_states = [defaultdict(bool, {pygame.K_UP: True}),
                      defaultdict(bool, {pygame.K_UP: True}), defaultdict(bool)]
        with (
            patch("ui.demo.make_env", return_value=env),
            patch("pygame.display.set_caption"),
            patch("pygame.event.get", side_effect=frames),
            patch("pygame.key.get_pressed", side_effect=key_states),
        ):
            play("lander", None, 42)
        self.assertEqual([call.args[0] for call in env.step.call_args_list], [2, 2, 0])

    def test_manual_waits_for_input_and_steers_by_absolute_direction(self):
        env = SnakeEnv()
        key = lambda value: pygame.event.Event(pygame.KEYDOWN, key=value)
        frames = [[], [], [key(pygame.K_UP)], [key(pygame.K_RIGHT)], [],
                  [key(pygame.K_r)], [], [pygame.event.Event(pygame.QUIT)]]
        with (
            patch("ui.demo.make_env", return_value=env),
            patch("pygame.display.set_caption"),
            patch("pygame.event.get", side_effect=frames),
            patch("pygame.time.get_ticks", side_effect=[0, 250, 500, 540, 700, 900, 1100, 1300]),
            patch.object(env, "step", wraps=env.step) as step,
        ):
            play("snake", None, 42)
        self.assertEqual([call.args[0] for call in step.call_args_list], [1, 2])
        self.assertEqual(env.body[0], (5, 5))
