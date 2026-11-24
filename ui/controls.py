"""Keyboard mappings shared by the solo player and experiment dashboard."""

import numpy as np
import pygame

HEADINGS = {pygame.K_UP: 0, pygame.K_RIGHT: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3}
THRUSTERS = {pygame.K_UP: 2, pygame.K_SPACE: 2, pygame.K_LEFT: 1, pygame.K_RIGHT: 3}


def snake_turn(key, heading):
    turn = (HEADINGS[key] - heading) % 4
    return None if turn == 2 else (0, 2, 0, 1)[turn]


def lander_thrust(pressed, continuous):
    if continuous:
        main = 1.0 if pressed[pygame.K_UP] or pressed[pygame.K_SPACE] else -1.0
        side = int(pressed[pygame.K_RIGHT]) - int(pressed[pygame.K_LEFT])
        return np.array([main, side], dtype=np.float32)
    return next((thrust for key, thrust in THRUSTERS.items() if pressed[key]), 0)
