from pathlib import Path
from importlib.util import find_spec
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import numpy as np

from algorithm.actor_critic import generalized_advantage
from algorithm.continuous_sac import SquashedGaussian
from algorithm.ddqn import DoubleDQN
from algorithm.dqn import td_targets
from algorithm.sac import soft_value
from experiments.evaluate import evaluate, load_agent, make_agent, make_env
from experiments.train import train


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dqn_bootstraps_only_when_not_terminal(self):
        target = td_targets(torch.tensor([1.0, 1.0]), torch.tensor([4.0, 4.0]),
                            torch.tensor([True, False]), 0.5)
        torch.testing.assert_close(target, torch.tensor([1.0, 3.0]))

    def test_double_dqn_selects_online_and_values_with_target(self):
        agent = DoubleDQN(2, 2)
        agent.network = torch.nn.Identity()
        next_states = torch.tensor([[4.0, 1.0], [2.0, 3.0]])
        target_values = torch.tensor([[2.0, 9.0], [10.0, 3.0]])
        values = agent.next_q_values(next_states, lambda _: target_values)
        torch.testing.assert_close(values, torch.tensor([2.0, 3.0]))

    def test_sac_soft_value_averages_over_all_actions_with_entropy(self):
        probs = torch.tensor([[0.25, 0.75]])
        log_probs = probs.log()
        value = soft_value(log_probs, torch.tensor([[2.0, 6.0]]), 0.2)
        entropy = -(probs * log_probs).sum()
        torch.testing.assert_close(value, (5.0 + 0.2 * entropy).reshape(1))

    def test_continuous_policy_scales_actions_and_has_finite_log_prob(self):
        low = np.array([-2.0, 0.0], dtype=np.float32)
        high = np.array([2.0, 1.0], dtype=np.float32)
        policy = SquashedGaussian(8, low, high)
        with torch.no_grad():
            policy.layers[-1].weight.zero_()
            policy.layers[-1].bias.copy_(torch.tensor([-20.0, 20.0, 0.0, 0.0]))
        actions, log_prob = policy.sample(torch.zeros(5, 8))
        self.assertEqual(actions.shape, (5, 2))
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue(((actions >= torch.from_numpy(low)) &
                         (actions <= torch.from_numpy(high))).all())
        log_prob.mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters()))

    @unittest.skipUnless(find_spec("Box2D"), "Box2D package is optional")
    def test_continuous_sac_trains_and_reloads_with_vector_actions(self):
        with TemporaryDirectory() as root, make_env("lander-continuous") as env:
            obs, _ = env.reset(seed=42)
            torch.manual_seed(42)
            initial = make_agent("sac", env)
            run = train("lander-continuous", "sac", 42, 1_020, root)
            completed = load_agent(run, env)
            self.assertTrue(env.action_space.contains(completed.predict(obs)))
            self.assertTrue(any(not torch.equal(before, after) for before, after in
                                zip(initial.network.parameters(), completed.network.parameters())))
            self.assertEqual(len(evaluate(run, 2, 45_000, game="lander-continuous")), 2)

    def test_gae_bootstraps_truncation_without_leaking_next_episode(self):
        advantages = generalized_advantage(
            torch.tensor([1.0, 10.0, 100.0]), torch.zeros(3),
            torch.tensor([4.0, 8.0, 50.0]), torch.tensor([False, True, True]),
            torch.tensor([True, True, True]), gamma=0.5,
        )
        torch.testing.assert_close(advantages, torch.tensor([3.0, 10.0, 100.0]))

    def test_a2c_updates_once_per_rollout_including_partial_rollout(self):
        env = make_env("snake")
        try:
            agent = make_agent("a2c", env)
            optimizer = torch.optim.Adam(agent.network.parameters())
            with (
                patch("algorithm.a2c.torch.optim.Adam", return_value=optimizer),
                patch.object(optimizer, "step", wraps=optimizer.step) as updates,
                patch.object(env, "step", wraps=env.step) as interactions,
            ):
                agent.train(env, 65, 42, lambda *_: None, tuned=True)
            self.assertEqual(interactions.call_count, 65)
            self.assertEqual(updates.call_count, 3)
            self.assertTrue(all(torch.isfinite(p).all() for p in agent.network.parameters()))
        finally:
            env.close()

    def test_short_training_updates_weights_and_reloads(self):
        with TemporaryDirectory() as root:
            for algorithm, steps in (("dqn", 1_020), ("ppo", 128), ("a2c", 128), ("sac", 1_020)):
                with self.subTest(algorithm=algorithm):
                    run = train("snake", algorithm, 42, steps, root)
                    env = make_env("snake")
                    try:
                        torch.manual_seed(42)
                        initial = make_agent(algorithm, env)
                        trained = load_agent(run, env)
                        differences = (not torch.equal(before, after) for before, after in
                                       zip(initial.network.parameters(), trained.network.parameters()))
                        self.assertTrue(any(differences))
                        self.assertTrue(all(torch.isfinite(p).all() for p in trained.network.parameters()))
                    finally:
                        env.close()
                    output = Path(root) / f"{algorithm}.csv"
                    rows = evaluate(run, 2, 10_000, output=output)
                    self.assertEqual([row["seed"] for row in rows], [10_000, 10_001])
                    self.assertTrue(output.is_file())

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            for algorithm in ("ppo", "a2c", "dqn", "sac", "ddqn"):
                with self.subTest(algorithm=algorithm):
                    uninterrupted = train(
                        "snake", algorithm, 12, 1_113, root / "full",
                        max_episode_steps=35, level_pool_size=5, checkpoint_steps=[1_057],
                    )
                    checkpoint = uninterrupted.with_name("seed_12_checkpoints") / "step_1057"
                    resumed = train("snake", algorithm, 12, 1_113, root / "resumed", resume=checkpoint)
                    self.assertEqual((uninterrupted / "train.csv").read_bytes(),
                                     (resumed / "train.csv").read_bytes())
                    expected = torch.load(uninterrupted / "model.pt", weights_only=True)
                    actual = torch.load(resumed / "model.pt", weights_only=True)
                    for key in expected:
                        self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_one_result_per_budget_and_seed_skip_and_overwrite(self):
        with TemporaryDirectory() as root:
            run = train("snake", "ppo", 42, 16, root)
            self.assertEqual(run.relative_to(root).parts, ("snake", "ppo", "step_16", "seed_42"))
            evaluation = run / "evaluation.csv"
            evaluation.write_text("original")
            with patch("experiments.train.train_into", side_effect=RuntimeError("training failed")):
                self.assertEqual(train("snake", "ppo", 42, 16, root), run)
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    train("snake", "ppo", 42, 16, root, overwrite=True)
            self.assertEqual(evaluation.read_text(), "original")
            self.assertEqual(train("snake", "ppo", 42, 16, root, overwrite=True), run)
            self.assertFalse(evaluation.exists())
            self.assertTrue((run / "model.pt").is_file())
