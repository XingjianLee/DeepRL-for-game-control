"""Course project commands for training, evaluation, and playback."""

import argparse
from importlib.util import find_spec
from itertools import product
from pathlib import Path

import torch

from experiments.evaluate import ALGORITHMS, ENVIRONMENTS, evaluate
from experiments.train import A2C_PROFILES, train
from games.lander import LANDER_GAMES
from ui.history import load_runs

RUNS = Path(__file__).resolve().parent / "runs"


def run_directory(value):
    if not value.strip():
        raise argparse.ArgumentTypeError(
            "empty model path; set the shell variable or pass a run directory"
        )
    path = Path(value).expanduser()
    if not (path / "config.json").is_file() or not (path / "model.pt").is_file():
        raise argparse.ArgumentTypeError(f"{path} must contain config.json and model.pt")
    return path


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train one or more models")
    training.add_argument("--game", choices=ENVIRONMENTS, nargs="+", default=["snake"])
    training.add_argument("--algorithm", choices=ALGORITHMS, nargs="+", required=True)
    training.add_argument("--a2c-profile", choices=A2C_PROFILES,
                          help="A2C training settings (default: baseline)")
    training.add_argument("--steps", type=positive, nargs="+", default=[50_000])
    training.add_argument("--seed", type=nonnegative, nargs="+", default=[42])
    training.add_argument("--max-episode-steps", type=positive, help="Override the per-episode time limit")
    training.add_argument("--level-pool-size", type=positive, help="Rotate through this many fixed environment seeds")
    training.add_argument("--snake-spatial", action="store_true", default=None,
                          help="Add reachability and tail connectivity features for Snake")
    training.add_argument("--checkpoint-steps", type=positive, nargs="+", help="Save at these total interaction counts")
    training.add_argument("--resume", type=run_directory, help="Continue from a checkpoint directory")
    training.add_argument("--output-root", type=Path, default=RUNS)
    training.add_argument("--overwrite", action="store_true", help="Replace an existing result after training succeeds")

    testing = commands.add_parser("evaluate", help="Score models or a random policy")
    testing.add_argument("--run", type=run_directory, nargs="+", help="Model directories to evaluate")
    testing.add_argument("--episodes", type=positive, default=100)
    testing.add_argument("--seed", type=nonnegative, default=10_000)
    testing.add_argument("--output", type=Path, help="Write per-episode CSV for a single policy")
    testing.add_argument("--max-episode-steps", type=positive, help="Override the evaluation time limit")

    playback = commands.add_parser("play", help="Play manually or replay a model")
    model = playback.add_mutually_exclusive_group()
    model.add_argument("--run", type=run_directory, help="Play a specific saved run")
    model.add_argument("--algorithm", choices=ALGORITHMS, help="Play the latest saved run")
    playback.add_argument("--seed", type=nonnegative, default=20_000)

    dashboard = commands.add_parser("dashboard", help="Open the gameplay and experiment dashboard")
    dashboard.add_argument("--runs", type=Path, default=RUNS, help="Folder containing saved runs")
    dashboard.add_argument("--seed", type=nonnegative, default=20_000)

    web = commands.add_parser("web", help="Serve the browser dashboard locally")
    web.add_argument("--runs", type=Path, default=RUNS, help="Folder containing saved runs")
    web.add_argument("--seed", type=nonnegative, default=20_000)
    web.add_argument("--port", type=positive, default=8000)

    for command in (testing, playback, dashboard, web):
        command.add_argument("--game", choices=ENVIRONMENTS, default="snake")
    args = parser.parse_args()
    selected_games = ([game for game in args.game if game != "lander-continuous" or "sac" in args.algorithm]
                      if args.command == "train" else [args.game])
    if (args.command not in ("dashboard", "web")
            and any(game in LANDER_GAMES for game in selected_games) and find_spec("Box2D") is None):
        parser.error("Lunar Lander requires Box2D==2.3.10. See README.md for installation.")
    if args.command == "play" and args.game == "lander-continuous" and args.algorithm not in (None, "sac"):
        parser.error("--game lander-continuous supports --algorithm sac")
    torch.set_num_threads(1)

    if args.command == "web":
        if args.port > 65535:
            web.error("--port must be at most 65535")
        from ui.web import serve

        serve(args.runs, args.game, args.seed, args.port)
    elif args.command == "dashboard":
        from ui.dashboard import launch

        launch(args.runs, args.game, args.seed)
    elif args.command == "train":
        if args.snake_spatial and any(game != "snake" for game in args.game):
            training.error("--snake-spatial only supports --game snake")
        pairs = list(product(args.game, args.algorithm))
        jobs = [(game, algorithm, steps, seed) for game, algorithm in pairs
                if game != "lander-continuous" or algorithm == "sac"
                for steps in args.steps for seed in args.seed]
        if not jobs:
            training.error("--game lander-continuous supports --algorithm sac")
        if args.a2c_profile and not any(algorithm == "a2c" for _, algorithm, _, _ in jobs):
            training.error("--a2c-profile requires a compatible A2C training job")
        if args.resume and (len(jobs) != 1 or not (args.resume / "state.pt").is_file()):
            training.error("--resume requires one training job and a checkpoint containing state.pt")
        for game, algorithm in pairs:
            if game == "lander-continuous" and algorithm != "sac":
                print(f"Skipping {game} / {algorithm}: continuous actions require SAC.", flush=True)
        for index, (game, algorithm, steps, seed) in enumerate(jobs, start=1):
            if len(jobs) > 1:
                print(f"[{index}/{len(jobs)}] {game} / {algorithm} / {steps:,} steps / seed {seed}",
                      flush=True)
            profile = args.a2c_profile if algorithm == "a2c" else None
            try:
                train(game, algorithm, seed, steps, args.output_root,
                      a2c_profile=profile, overwrite=args.overwrite,
                      max_episode_steps=args.max_episode_steps, level_pool_size=args.level_pool_size,
                      checkpoint_steps=args.checkpoint_steps, resume=args.resume, snake_spatial=args.snake_spatial)
            except (ValueError, FileExistsError) as error:
                training.error(str(error))
    elif args.command == "evaluate":
        runs = args.run or [None]
        if args.output and len(runs) != 1:
            testing.error("--output requires evaluating a single policy")
        for run in runs:
            evaluate(run, args.episodes, args.seed, game=args.game, output=args.output,
                     max_episode_steps=args.max_episode_steps)
    else:
        from ui.demo import play

        run = args.run
        if args.algorithm:
            models = load_runs(RUNS, ENVIRONMENTS, ALGORITHMS)
            run = next((record.path for record in models
                        if record.game == args.game and record.algorithm == args.algorithm), None)
            if run is None:
                playback.error(f"No saved {args.algorithm} model in {root}. Train one or pass --run.")
            print(f"Model: {run}", flush=True)
        play(args.game, run, args.seed)


if __name__ == "__main__":
    main()
