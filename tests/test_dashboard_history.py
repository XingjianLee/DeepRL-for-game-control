import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ui.history import comparable_runs, load_runs


class DashboardHistoryTests(unittest.TestCase):
    def test_loads_finished_runs_and_compares_only_matching_experiments(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)

            def save(algorithm, seed, completed_at, steps, eval_seed, *, profile="baseline", saved=True):
                path = root / "snake" / algorithm / f"step_{steps}" / f"seed_{seed}"
                path.mkdir(parents=True)
                (path / "config.json").write_text(json.dumps({
                    "game": "snake", "algorithm": algorithm, "seed": seed,
                    "steps": steps, "a2c_profile": profile, "completed_at": completed_at,
                }))
                if saved:
                    (path / "model.pt").touch()
                (path / "evaluation.csv").write_text(f"seed,score,return\n{eval_seed},2,1\n")
                (path / "train.csv").write_text("step,score,return\n10,3,2\n", encoding="utf-8")

            save("ppo", 42, "1", 50, 100)
            save("ppo", 23, "2", 50, 100)
            save("dqn", 7, "3", 50, 100)
            save("dqn", 23, "4", 51, 100)
            save("dqn", 42, "5", 50, 101)
            save("a2c", 42, "6", 50, 100, profile="improved")
            save("sac", 42, "7", 50, 100, saved=False)
            runs = load_runs(root, {"snake"}, {"dqn", "ppo", "a2c", "sac"})
            self.assertEqual(len(runs), 6)
            selected = runs[-1]
            self.assertEqual(selected.curve, ((10.0, 3.0),))
            evaluation = selected.evaluations[0]
            groups = comparable_runs(runs, selected, evaluation)
            self.assertEqual(set(groups), {"dqn", "ppo"})
            self.assertEqual(set(groups["dqn"]), {7})
            self.assertIs(groups["ppo"][42], evaluation)
            self.assertEqual(evaluation.score, 2)
