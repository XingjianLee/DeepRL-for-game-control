from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from ui.web import Lab, Replay


class WebLeaderboardTests(unittest.TestCase):
    def test_model_replay_uses_saved_episode_limit(self):
        with patch("ui.web.load_agent", return_value=SimpleNamespace(predict=lambda _: 0)):
            for model, limit in (
                (SimpleNamespace(path=Path("."), max_episode_steps=2_000, snake_spatial=False), 2_000),
                (SimpleNamespace(path=Path("."), max_episode_steps=None, snake_spatial=False), 300),
                (None, 300),
            ):
                with self.subTest(model=model):
                    replay = Replay("snake", model, 20_000)
                    try:
                        self.assertEqual(replay.env.spec.max_episode_steps, limit)
                    finally:
                        replay.env.close()

    def test_finished_human_and_model_runs_share_seed_rankings(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            lab = Lab(root, "snake", 42)
            lab.runs["model"] = SimpleNamespace(
                game="snake", path=root, algorithm="ppo", steps=100, seed=7,
                max_episode_steps=None, snake_spatial=False,
            )
            try:
                for mode in ("manual", "ai"):
                    session = str(uuid4())
                    with patch("ui.web.load_agent", return_value=SimpleNamespace(predict=lambda _: 0)):
                        lab.reset(dict(session=session, game="snake", seed=42, mode=mode, run="model"))
                    lab.advance(dict(session=session, count=1, player="小李"))
                    ranking = lab.leaderboard(dict(game="snake", seed=42))
                    self.assertEqual(len(ranking), int(mode == "ai"))
                    lab.advance(dict(session=session, count=8))
                    lab.advance(dict(session=session, count=8))
                rows = Lab(root, "snake", 42).leaderboard(dict(game="snake", seed=42))
                self.assertEqual([row["model"] for row in rows], [0, 1])
                self.assertEqual(rows[0]["player"], "小李")
                self.assertIn("种子 7", rows[1]["player"])
                self.assertEqual(lab.leaderboard(dict(game="snake", seed=43)), [])
                self.assertEqual(lab.leaderboard(dict(game="flap", seed=42)), [])
                self.assertEqual(lab.delete_score({"id": rows[1]["id"]}), {"deleted": True})
                lab.advance(dict(session=session, count=1))
                self.assertEqual(Lab(root, "snake", 42).leaderboard(dict(game="snake", seed=42)), rows[:1])
                self.assertEqual(lab.delete_score({"id": rows[1]["id"]}), {"deleted": False})
            finally:
                lab.close()
