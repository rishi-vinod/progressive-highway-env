"""Headless correctness checks for Stage 1; uses only standard-library unittest."""

import importlib
import unittest
from types import SimpleNamespace

from driving_common import ENV_CONFIG, EVALUATION_SEEDS, EpisodeMetrics, make_env, run_episode
import gymnasium as gym
import numpy as np

baseline = importlib.import_module("01_rule_based_drive")


class Stage1Tests(unittest.TestCase):
    def test_controller_uses_only_ego_velocity_and_never_changes_lane(self):
        observation = np.zeros((51, 5), dtype=np.float32)
        observation[0, 0] = 1
        for speed, expected in [(25, "FASTER"), (30, "IDLE"), (31, "SLOWER")]:
            observation[0, 3] = speed
            self.assertEqual(baseline.cruise_action(observation), baseline.ACTION_CODES[expected])
            observation[1:] = 1000  # Traffic cannot change this baseline's decision.
            self.assertEqual(baseline.cruise_action(observation), baseline.ACTION_CODES[expected])

    def test_metrics_include_initial_and_intermediate_minimum(self):
        ego = SimpleNamespace(position=np.array([0.0, 0.0]), speed=10.0, crashed=False)
        other = SimpleNamespace(position=np.array([3.0, 4.0]))
        metrics = EpisodeMetrics()
        metrics.observe(ego, [ego, other])
        self.assertEqual(metrics.closest_distance_m, 5.0)
        other.position = np.array([0.0, 1.0])
        ego.speed = 20.0
        ego.crashed = True
        metrics.observe(ego, [ego, other], dt=0.5)
        other.position = np.array([0.0, 9.0])
        ego.speed = 30.0
        ego.crashed = False
        metrics.observe(ego, [ego, other], dt=0.5)
        self.assertEqual(metrics.closest_distance_m, 1.0)
        self.assertEqual(metrics.average_speed_mps, 20.0)
        self.assertEqual(metrics.physics_ticks, 2)
        self.assertTrue(metrics.collision)  # Collision flag is sticky.

    def test_empty_traffic_does_not_log_infinity(self):
        ego = SimpleNamespace(position=np.zeros(2), speed=25.0, crashed=False)
        metrics = EpisodeMetrics()
        metrics.observe(ego, [ego])
        self.assertIsNone(metrics.closest_distance_m)

    def test_measurement_preserves_stock_dynamics_and_real_observations(self):
        measured = make_env(render_mode=None)
        stock = gym.make("highway-v0", config=ENV_CONFIG, render_mode=None)
        try:
            measured_obs, _ = measured.reset(seed=0)
            stock_obs, _ = stock.reset(seed=0)
            np.testing.assert_array_equal(measured_obs, stock_obs)
            self.assertEqual(measured_obs.shape, (51, 5))
            self.assertEqual(measured_obs[0, 3], 25.0)  # Not a normalized speed.
            traffic = measured_obs[1:][measured_obs[1:, 0] == 1]
            distances = np.abs(traffic[:, 1] - measured_obs[0, 1])
            self.assertTrue(np.all(np.diff(distances) >= 0))
            initial_lane = measured.unwrapped.vehicle.lane_index
            for action in [3, 3, 1]:
                observed, reward, terminated, truncated, _ = measured.step(action)
                stock_observed, stock_reward, stock_terminated, stock_truncated, _ = stock.step(action)
                np.testing.assert_array_equal(observed, stock_observed)
                self.assertEqual((reward, terminated, truncated), (stock_reward, stock_terminated, stock_truncated))
                self.assertEqual(measured.unwrapped.vehicle.lane_index, initial_lane)
            self.assertEqual(measured.unwrapped.episode_metrics.physics_ticks, 45)
            self.assertAlmostEqual(measured.unwrapped.episode_metrics.elapsed_s, 3.0)
            measured.reset(seed=0)
            self.assertEqual(measured.unwrapped.episode_metrics.physics_ticks, 0)
        finally:
            measured.close()
            stock.close()

    def test_seeded_runner_repeats_and_uses_fixed_benchmark_seeds(self):
        first, _ = run_episode(baseline.create_controller, 0, max_steps=3)
        second, _ = run_episode(baseline.create_controller, 0, max_steps=3)
        self.assertEqual(first, second)
        self.assertEqual(first.end_reason, "step_limit")
        self.assertEqual(EVALUATION_SEEDS, tuple(range(50)))

    def test_runner_closes_environment_on_controller_error(self):
        created = []

        def factory(render_mode):
            env = make_env(render_mode=render_mode)
            created.append(env)
            return env

        with self.assertRaises(ValueError):
            run_episode(lambda seed: lambda obs: 999, 0, env_factory=factory)
        self.assertTrue(created[0].unwrapped.done)
        self.assertIsNone(created[0].unwrapped.viewer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
