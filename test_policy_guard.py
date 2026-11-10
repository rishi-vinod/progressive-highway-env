"""Headless regression tests for the optional learned-policy action guard."""

import copy
import unittest

from driving_common import ENV_CONFIG
from policy_guard import GuardedPolicy, checked_action
from reactive import IDLE, FASTER, SLOWER, LANE_LEFT, LANE_RIGHT
from scenarios import Session, replay_document
from test_project import observation


class PolicyGuardTests(unittest.TestCase):
    def test_front_rear_and_side_traffic_block_lane_changes(self):
        cases = [(110, 8, 15, 0), (98, 8, 25, 0), (70, 8, 40, 0)]
        for car in cases:
            with self.subTest(car=car):
                action, reason = checked_action(observation([car], speed=20), LANE_RIGHT)
                self.assertEqual(action, SLOWER)
                self.assertEqual(reason, "target_lane_not_clear")
        self.assertEqual(checked_action(observation(), LANE_RIGHT), (LANE_RIGHT, None))
        self.assertEqual(checked_action(observation(), FASTER), (FASTER, None))

    def test_boundaries_in_progress_and_front_danger(self):
        obs = observation()
        obs[0, 2] = 0
        self.assertEqual(checked_action(obs, LANE_LEFT)[0], SLOWER)
        obs[0, 2] = 5
        obs[0, 4] = 1
        self.assertEqual(checked_action(obs, LANE_RIGHT), (IDLE, "lane_change_in_progress"))
        obs = observation([(110, 4, 15, 0)])
        self.assertEqual(checked_action(obs, FASTER), (SLOWER, "current_lane_danger"))
        self.assertEqual(checked_action(obs, SLOWER), (SLOWER, None))

    def test_model_cannot_modify_guard_input(self):
        def model(obs):
            obs[1:] = 0
            return LANE_RIGHT
        obs = observation([(98, 8, 25, 0)], speed=20)
        original = obs.copy()
        policy = GuardedPolicy(model)
        self.assertEqual(policy(obs), SLOWER)
        self.assertTrue((original == obs).all())
        self.assertEqual(policy.decision_info["requested_action"], LANE_RIGHT)
        self.assertTrue(policy.decision_info["guard_intervened"])

    def test_requested_and_executed_actions_survive_replay(self):
        config = copy.deepcopy(ENV_CONFIG)
        config["initial_lane_id"] = 3
        session = Session(10, config=config)
        policy = GuardedPolicy(lambda obs: LANE_RIGHT)
        try:
            for _ in range(2):
                action = policy(session.observation)
                session.step(action, "learned_guarded", decision_info=policy.decision_info)
            document = session.document()
            self.assertTrue(document["steps"][0]["decision_info"]["guard_intervened"])
            self.assertEqual(document["steps"][0]["ego_action"], SLOWER)
            self.assertEqual(replay_document(document), session.result())
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
