"""Pygame rendering for Snake."""

import numpy as np
import pygame


def draw(env):
    cell = 32
    width = env.size * cell
    height = width + 40
    canvas = pygame.Surface((width, height))
    canvas.fill((16, 23, 28))
    for y in range(env.size):
        for x in range(env.size):
            color = (29, 40, 46) if (x + y) % 2 else (34, 46, 53)
            pygame.draw.rect(canvas, color, (x * cell, y * cell, cell, cell))
    for index, (x, y) in enumerate(env.body):
        color = (169, 230, 137) if index == 0 else (66, 158, 105)
        pygame.draw.rect(canvas, color, (x * cell + 2, y * cell + 2, cell - 4, cell - 4),
                         border_radius=5)
    if env.food is not None:
        x, y = env.food
        pygame.draw.circle(canvas, (250, 129, 106),
                           (x * cell + cell // 2, y * cell + cell // 2), cell // 3)
    if not pygame.font.get_init():
        pygame.font.init()
    label = pygame.font.Font(None, 26).render(f"Score  {env.score}", True, (230, 237, 235))
    canvas.blit(label, (12, width + 10))
    if env.render_mode == "rgb_array":
        return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))

    if env.window is None:
        pygame.display.init()
        env.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake")
        env.clock = pygame.time.Clock()
    env.window.blit(canvas, (0, 0))
    pygame.display.flip()
    pygame.event.pump()
    env.clock.tick(env.metadata["render_fps"])
    return None
