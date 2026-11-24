"""Interactive playback with either keyboard input or a trained model."""

from experiments.evaluate import load_agent, make_env
from games.lander import LANDER_GAMES
from ui.controls import HEADINGS, THRUSTERS, lander_thrust, snake_turn


def play(game, run, seed):
    import pygame

    snake = game == "snake"
    lander = game in LANDER_GAMES
    continuous = game == "lander-continuous"
    title, controls, start_key = {
        "snake": ("Snake", "Arrow keys", "Arrow keys or Space"),
        "flap": ("Flappy Bird", "Space to flap", "Space"),
        "lander": ("Lunar Lander", "Up/Space: main engine | Left/Right: side engines",
                   "Up/Left/Right or Space"),
        "lander-continuous": ("Lunar Lander Continuous",
                              "Up/Space: main engine | Left/Right: side engines",
                              "Up/Left/Right or Space"),
    }[game]
    env = make_env(game, render_mode="human")
    try:
        agent = load_agent(run, env) if run is not None else None
        ready_title = f"{title} | {start_key} to start | Esc to quit"
        obs, _ = env.reset(seed=seed)
        score = 0
        total_reward = 0.0
        playing = True
        finished = False
        started = agent is not None
        action = 0
        next_move = 0
        clock = pygame.time.Clock()
        pygame.display.set_caption(title if agent is not None else ready_title)
        if agent is None:
            print(f"Click the window, then press {start_key} to start.")
        print("R: restart  |  Esc: quit")
        while playing:
            now = pygame.time.get_ticks()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    playing = False
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    playing = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset(seed=seed)
                    score = 0
                    total_reward = 0.0
                    pygame.display.set_caption(title if agent is not None else ready_title)
                    finished = False
                    started = agent is not None
                    action = 0
                    next_move = now
                elif agent is None and not finished:
                    if lander:
                        if event.key not in THRUSTERS:
                            continue
                    elif not snake and event.key == pygame.K_SPACE:
                        action = 1
                    elif snake and event.key in HEADINGS:
                        turn = snake_turn(event.key, env.unwrapped.heading)
                        if turn is None:
                            continue
                        action = turn
                    elif not snake or event.key != pygame.K_SPACE:
                        continue
                    if not started:
                        started = True
                        next_move = now
                        pygame.display.set_caption(f"{title} | {controls} | R to restart")
            if not playing:
                break
            if started and not finished and now >= next_move:
                if agent is not None:
                    action = agent.predict(obs)
                elif lander:
                    action = lander_thrust(pygame.key.get_pressed(), continuous)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                action = 0
                next_move = now + (180 if snake and agent is None else 0)
                finished = terminated or truncated
                if finished:
                    reason = info.get("reason") or ("collision" if terminated else "time_limit")
                    result = f"Return {total_reward:.1f}" if lander else f"Score {info['score']}"
                    pygame.display.set_caption(f"{title} | {result} | {reason} | R to restart")
                    print(f"{result} ({reason})")
                elif game == "flap" and info["score"] != score:
                    score = info["score"]
                    pygame.display.set_caption(f"{title} | Score {score} | R to restart")
            clock.tick(60)
    finally:
        env.close()
