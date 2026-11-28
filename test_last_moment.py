"""Headless threshold, actuator, hysteresis, and replay regressions."""

import copy
from dataclasses import replace
import json
import unittest

import numpy as np
from assist import AssistController
from last_moment import INTERVENTION_CONFIG, object_risks
from reactive import IDLE, FASTER, SLOWER, LANE_LEFT, LANE_RIGHT
from driving_common import ENV_CONFIG
from scenarios import Session, replay_document


def observed(gap=100, closing=10, blocked=False):
    obs = np.zeros((51, 5), dtype=np.float32)
    obs[0] = [1, 100, 4, 30, 0]
    obs[1] = [1, 105 + gap, 4, 30 - closing, 0]
    if blocked:
        obs[2] = [1, 100, 0, 30, 0]
        obs[3] = [1, 100, 8, 30, 0]
    return obs


class LastMomentTests(unittest.TestCase):
    def test_brake_boundary_and_no_early_headway_takeover(self):
        # Isolate the brake threshold from the independent steering deadline.
        config = replace(INTERVENTION_CONFIG, a_lat_max=100, t_margin=0.01)
        boundary = 10 ** 2 / (2 * config.alpha * config.a_max)
        helper = AssistController(config=config, policy_dt=1/30)
        self.assertEqual(helper.choose(observed(boundary + 0.02), FASTER), FASTER)
        self.assertEqual(helper.mode, 'manual')
        helper.choose(observed(boundary - 0.02, blocked=True), IDLE)
        self.assertTrue(helper.decision_info['brake_trigger'])
        self.assertEqual(helper.decision_info['rescue_control']['acceleration'], -8)
        helper = AssistController()
        self.assertEqual(helper.choose(observed(25), IDLE), IDLE)
        self.assertEqual(helper.mode, 'manual')

    def test_steer_deadline_and_blocked_fallback(self):
        c = INTERVENTION_CONFIG
        deadline = (2 * (2 + c.lateral_clearance_m) / c.a_lat_max) ** 0.5 + c.t_margin
        helper = AssistController()
        helper.choose(observed((deadline + 0.01) * 10), IDLE)
        self.assertEqual(helper.mode, 'manual')
        action = helper.choose(observed((deadline - 0.01) * 10), IDLE)
        self.assertIn(action, (LANE_LEFT, LANE_RIGHT))
        self.assertTrue(helper.decision_info['steer_trigger'])
        helper = AssistController()
        self.assertEqual(helper.choose(observed(4, blocked=True), IDLE), SLOWER)
        self.assertTrue(helper.decision_info['braking_impossible'])
        self.assertIn(AssistController().choose(observed(4), IDLE), (LANE_LEFT, LANE_RIGHT))

    def test_all_objects_nonclosing_parallel_and_zero_gap(self):
        helper = AssistController()
        self.assertEqual(helper.choose(observed(1, closing=0), FASTER), FASTER)
        self.assertEqual(helper.mode, 'manual')
        self.assertEqual(helper.choose(observed(1, closing=-2), IDLE), IDLE)
        obs = observed(1)
        obs[1, 2] = 8
        helper.choose(obs, IDLE)
        self.assertEqual(helper.mode, 'manual')
        obs[2] = [1, 108, 4, 10, 0]
        helper.choose(obs, IDLE)
        self.assertEqual(helper.mode, 'override')
        self.assertEqual(len(helper.decision_info['objects']), 2)
        helper = AssistController()
        helper.choose(observed(0), IDLE)
        self.assertTrue(helper.decision_info['braking_impossible'])
        json.dumps(helper.decision_info, allow_nan=False)

    def test_cut_in_and_requested_lane_conflict(self):
        obs = observed(5)
        obs[1, 2:5] = [8, 20, -4]
        self.assertTrue(object_risks(obs, IDLE, INTERVENTION_CONFIG)[0]['relevant'])
        obs[1, 4] = 0
        self.assertFalse(object_risks(obs, IDLE, INTERVENTION_CONFIG)[0]['relevant'])
        self.assertTrue(object_risks(obs, LANE_RIGHT, INTERVENTION_CONFIG)[0]['relevant'])

    def test_safe_hold_ramp_retrigger_and_disable(self):
        helper = AssistController(policy_dt=0.1)
        helper.choose(observed(4, blocked=True), IDLE)
        clear = observed(100)
        for _ in range(5):
            helper.choose(clear, IDLE)
            self.assertEqual(helper.mode, 'override')
        helper.choose(clear, IDLE)
        self.assertEqual(helper.mode, 'blending')
        clear[0, 2] = 4.4
        clear[0, 4] = 0.5  # Safe driver steering must not cancel the ramp.
        weights = []
        for _ in range(4):
            helper.choose(clear, FASTER)
            weights.append(helper.decision_info['rescue_control']['weight'])
        self.assertTrue(all(a > b for a, b in zip(weights, weights[1:])))
        helper.choose(observed(4, blocked=True), IDLE)
        self.assertEqual(helper.mode, 'override')
        self.assertEqual(helper.decision_info['event'], 'handback_cancelled')
        clear[0, 2:5] = [4, 30, 0]
        for _ in range(12):
            helper.choose(clear, IDLE)
        self.assertEqual(helper.mode, 'manual')
        self.assertIsNotNone(helper.interventions[0]['handback_time_s'])
        disabled = AssistController(enabled=False)
        self.assertEqual(disabled.choose(observed(1), FASTER), FASTER)
        self.assertEqual(disabled.mode, 'manual')

    def test_fast_rear_car_blocks_escape_lane(self):
        obs = observed(4, blocked=True)
        obs[2] = [1, 85, 0, 40, 0]
        helper = AssistController()
        self.assertEqual(helper.choose(obs, IDLE), SLOWER)
        self.assertEqual(helper.decision_info['maneuver'], 'brake')

    def test_actuator_blends_real_acceleration_and_steering(self):
        config = copy.deepcopy(ENV_CONFIG)
        config.update(vehicles_count=0, initial_lane_id=1,
                      simulation_frequency=30, policy_frequency=30)
        session = Session(0, config=config)
        try:
            ego = session.env.unwrapped.vehicle
            ego.speed = ego.target_speed = 20
            control = dict(weight=0.5, end_weight=0.5, duration_s=1/30,
                           acceleration=-8, target_lane=2, a_max=8, a_lat_max=4,
                           human_action=FASTER, blending=True)
            session.step(IDLE, 'assist', FASTER, {'rescue_control': control})
            # On first blend frame the driver target starts at current speed;
            # FASTER requests 25 m/s, giving (25-20)/0.6 m/s² before mixing.
            expected = 0.5 * -8 + 0.5 * (5 / 0.6)
            self.assertAlmostEqual(ego.speed, 20 + expected / 30, places=5)
            self.assertGreater(ego.action['steering'], 0)
            self.assertLess(ego.action['steering'], ego.MAX_STEERING_ANGLE)
        finally:
            session.close()

    def test_simulated_max_braking_blend_and_replay(self):
        from highway_env.vehicle.kinematics import Vehicle
        config = copy.deepcopy(ENV_CONFIG)
        config.update(vehicles_count=0, initial_lane_id=1,
                      simulation_frequency=30, policy_frequency=30)
        session = Session(0, config=config)
        try:
            base = session.env.unwrapped
            ego = base.vehicle
            ego.speed = ego.target_speed = 30
            # Bumper gap 12: the steering deadline is reached, but both lanes
            # are blocked. Eight m/s² braking has enough space to match 20 m/s.
            for i, (dx, y, speed) in enumerate([(17, 4, 20), (0, 0, 30), (0, 8, 30)]):
                car = Vehicle(base.road, [ego.position[0] + dx, y], speed=speed)
                car.project_id = f'test_{i}'
                base.road.vehicles.append(car)
            session.observation = base.observation_type.observe()
            helper = AssistController(policy_dt=1/30)
            action = helper.choose(session.observation, IDLE)
            session.step(action, helper.label, IDLE, helper.decision_info)
            self.assertAlmostEqual(ego.speed, 30 - 8/30, places=5)
            for _ in range(100):
                action = helper.choose(session.observation, IDLE)
                session.step(action, helper.label, IDLE, helper.decision_info)
                self.assertFalse(session.result().collision)
            self.assertEqual(helper.mode, 'manual')
            self.assertIsNotNone(session.document()['interventions'][0]['handback_time_s'])
        finally:
            session.close()
        # A seeded, unmodified episode can be reconstructed exactly, including
        # physical overrides and per-step logs, without rerunning the controller.
        session = Session(10)
        helper = AssistController(policy_dt=1)
        try:
            for _ in range(30):
                action = helper.choose(session.observation, FASTER)
                session.step(action, helper.label, FASTER, helper.decision_info)
                if session.ended:
                    break
            saved = session.document()
            self.assertTrue(saved['interventions'])
            json.dumps(saved, allow_nan=False)
            self.assertEqual(replay_document(saved), session.result())
        finally:
            session.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
