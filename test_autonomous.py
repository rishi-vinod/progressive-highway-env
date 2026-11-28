"""Headless tests for style differences and shared emergency physics."""

import copy
import unittest

from assist import AssistController, ASSIST_TARGET_SPEEDS
from autonomous import STYLE_CONFIG
from reactive import IDLE, FASTER, SLOWER, LANE_LEFT, LANE_RIGHT
from test_project import observation
from driving_common import ENV_CONFIG
from scenarios import Session, replay_document


def autonomous(style='Calm', dt=0.2):
    helper = AssistController(style=style, policy_dt=dt)
    helper.toggle()
    return helper


class AutonomousTests(unittest.TestCase):
    def test_flow_speeds_and_equal_speed_following(self):
        obs = observation([(180, 0, 20, 0)], speed=20)
        for style, expected in [('Calm', 20), ('Quick', 22.5), ('Rage', 30)]:
            helper = autonomous(style)
            action = helper.choose(obs, IDLE)
            self.assertEqual(helper.decision_info['planner']['cruise_speed_mps'], expected)
            self.assertEqual(action, IDLE if style == 'Calm' else FASTER)
        obs = observation([(145, 4, 20, 0)], speed=20)
        helper = autonomous()
        self.assertEqual(helper.choose(obs, IDLE), IDLE)
        self.assertEqual(helper.takeovers, 0)

    def test_rage_passes_immediately_calm_and_quick_wait(self):
        # Slow leader and faster nearby flow; unobstructed passing corridors.
        obs = observation([(160, 4, 15, 0), (190, 0, 25, 0), (190, 8, 25, 0)], speed=20)
        for style in STYLE_CONFIG:
            helper = autonomous(style)
            first_pass = None
            for tick in range(45):
                action = helper.choose(obs, IDLE)
                if action in (LANE_LEFT, LANE_RIGHT):
                    first_pass = (tick + 1) * 0.2
                    break
            self.assertIsNotNone(first_pass, style)
            self.assertGreaterEqual(first_pass + 1e-9, STYLE_CONFIG[style].pass_delay_s)
            if style == 'Rage':
                self.assertEqual(first_pass, 0.2)
            # A command already pending is not repeated, even before motion.
            self.assertNotIn(helper.choose(obs, IDLE), (LANE_LEFT, LANE_RIGHT))

    def test_all_styles_use_same_physics_and_resume_auto(self):
        hazard = observation([(109, 4, 10, 0), (100, 0, 30, 0), (100, 8, 30, 0)])
        manual = AssistController(policy_dt=0.1)
        manual.choose(hazard, IDLE)
        for style in STYLE_CONFIG:
            helper = autonomous(style, dt=0.1)
            helper.enabled = False  # Disabling manual rescue cannot disable autonomous safety.
            self.assertEqual(helper.choose(hazard, IDLE), SLOWER)
            for field in ('TTC_s', 'a_req_mps2', 'brake_trigger', 'steer_trigger', 'config'):
                self.assertEqual(helper.decision_info[field], manual.decision_info[field])
            self.assertEqual(helper.decision_info['rescue_control']['acceleration'], -8)
            for _ in range(12):
                helper.choose(observation(), IDLE)
            self.assertEqual(helper.mode, 'auto')
            self.assertEqual(helper.decision_info['return_control_to'], 'autonomous')
            helper.toggle()
            self.assertEqual(helper.mode, 'manual')
            self.assertFalse(helper.autonomous)

    def test_rage_does_not_pass_through_side_or_fast_rear_traffic(self):
        obs = observation([(160, 4, 15, 0), (85, 0, 40, 0), (100, 8, 20, 0)], speed=20)
        helper = autonomous('Rage')
        for _ in range(20):
            self.assertNotIn(helper.choose(obs, IDLE), (LANE_LEFT, LANE_RIGHT))

    def test_autonomous_decisions_and_physical_controls_replay(self):
        config = copy.deepcopy(ENV_CONFIG)
        config.update(simulation_frequency=15, policy_frequency=5, duration=8)
        config['action']['target_speeds'] = ASSIST_TARGET_SPEEDS
        session = Session(10, config=config)
        helper = autonomous('Rage')
        try:
            while not session.ended:
                action = helper.choose(session.observation, IDLE)
                session.step(action, helper.label, IDLE, helper.decision_info)
            saved = session.document()
            self.assertTrue(all(r['decision_info']['autonomous'] for r in saved['steps']))
            self.assertEqual(replay_document(saved), session.result())
        finally:
            session.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
