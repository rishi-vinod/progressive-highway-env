"""Headless integration checks for all completed stages; no display is opened."""

import copy
import csv
import io
import unittest

from driving_common import ENV_CONFIG, new_run_directory
from reactive import (reactive_action, assess_lane, danger, LANE_LEFT, LANE_RIGHT,
                      SLOWER, IDLE, FASTER)
from assist import AssistController
from hazards import seeded_schedule
from scenarios import Session, replay_document
from local_ui import parse_command, execute_command
import numpy as np


def observation(vehicles=(), speed=30.0):
    obs = np.zeros((51, 5), dtype=np.float32)
    obs[0] = [1, 100, 4, speed, 0]
    for index, row in enumerate(vehicles, 1):
        obs[index] = [1, *row]
    return obs


class ControllerTests(unittest.TestCase):
    def test_clear_road_and_blocked_lanes(self):
        self.assertEqual(reactive_action(observation(speed=25)), FASTER)
        blocked = observation([(112, 4, 15, 0), (100, 0, 30, 0), (100, 8, 30, 0)])
        self.assertEqual(reactive_action(blocked), SLOWER)

    def test_fast_rear_car_blocks_lane_change(self):
        obs = observation([(112, 4, 15, 0), (70, 0, 40, 0), (100, 8, 30, 0)])
        self.assertFalse(assess_lane(obs, 0).safe)
        self.assertEqual(reactive_action(obs), SLOWER)

    def test_clear_adjacent_lane_is_used(self):
        obs = observation([(112, 4, 15, 0), (100, 8, 30, 0)])
        self.assertEqual(reactive_action(obs), LANE_LEFT)

    def test_cut_in_detector_uses_observed_lateral_speed(self):
        self.assertTrue(danger(observation([(108, 8, 20, -2)])))
        self.assertFalse(danger(observation([(108, 8, 20, 2)])))

    def test_takeover_handback_and_disable(self):
        controller = AssistController(policy_dt=1)
        threatened = observation([(112, 4, 15, 0), (100, 0, 30, 0), (100, 8, 30, 0)])
        self.assertEqual(controller.choose(threatened, FASTER), SLOWER)
        self.assertTrue(controller.active)
        self.assertEqual(controller.takeovers, 1)
        controller.choose(observation(), IDLE)
        self.assertEqual(controller.mode, "override")
        controller.choose(observation(), IDLE)
        self.assertEqual(controller.mode, "blending")
        controller.choose(observation(), FASTER, acknowledge=True)
        self.assertFalse(controller.active)
        controller.enabled = False
        self.assertEqual(controller.choose(threatened, FASTER), FASTER)

    def test_unsafe_lane_request_and_automatic_handback(self):
        helper = AssistController(policy_dt=1)
        obs = observation([(108, 8, 15, 0)], speed=25)
        self.assertEqual(helper.mode, "manual")
        self.assertNotEqual(helper.choose(obs, LANE_RIGHT), LANE_RIGHT)
        self.assertEqual(helper.mode, "override")
        helper.toggle()
        self.assertEqual(helper.mode, "override")
        # Finish the escape in the selected left lane before returning control.
        clear = observation()
        clear[0, 2] = 0
        for _ in range(5):
            helper.choose(clear, IDLE)
        self.assertEqual(helper.mode, "manual")
        helper.choose(obs, LANE_RIGHT, acknowledge=True)
        self.assertEqual(helper.mode, "override")
        for _ in range(3):
            helper.choose(clear, IDLE)
        self.assertEqual(helper.choose(clear, LANE_LEFT, acknowledge=True), LANE_LEFT)
        self.assertEqual(helper.mode, "manual")

    def test_voluntary_auto_follows_and_manual_driver_is_trusted(self):
        helper = AssistController()
        helper.toggle()
        obs = observation([(125, 4, 20, 0), (100, 0, 20, 0), (100, 8, 20, 0)], speed=25)
        self.assertEqual(helper.choose(obs, IDLE), SLOWER)
        obs[0, 3] = 20
        self.assertEqual(helper.choose(obs, IDLE), SLOWER)
        obs[0, 3] = 15
        self.assertEqual(helper.choose(obs, IDLE), IDLE)
        helper.toggle()
        self.assertEqual(helper.choose(observation([(93, 4, 30, 0)], speed=25), SLOWER), SLOWER)
        self.assertEqual(helper.mode, "manual")

    def test_one_second_contact_boundary_and_crossing(self):
        for row, expected in [
            ((116, 4, 20, 0), "override"),  # Brake before one-second contact.
            ((115, 4, 20, 0), "override"),  # exactly one second
            ((110, 4, 20, 0), "override"),  # exactly 0.5 s
            ((106, 4, 30, 0), "manual"),    # No closing speed: trust the driver.
            ((110, 8, 20, 0), "manual"),    # parallel lane
            ((110, 8, 20, -4), "override"), # crossing at 0.5 s
        ]:
            with self.subTest(row=row):
                helper = AssistController()
                helper.choose(observation([row]), IDLE)
                self.assertEqual(helper.mode, expected)

    def test_driving_styles_and_far_leader(self):
        flow = observation([(250, 4, 20, 0)], speed=25)
        def drive(style, obs):
            helper = AssistController(style=style)
            helper.toggle()
            return helper.choose(obs, IDLE)
        self.assertEqual(drive("Calm", flow), SLOWER)
        self.assertEqual(drive("Quick", flow), IDLE)
        self.assertEqual(drive("Rage", flow), FASTER)
        flow[1, 1] = 150
        self.assertEqual(drive("Rage", flow), SLOWER)
        self.assertNotIn(drive("Quick", flow), (LANE_LEFT, LANE_RIGHT))

    def test_front_braking_precedes_escape_and_merge_acceleration(self):
        obs = observation([(121, 4, 9, 0), (100, 0, 25, 0), (85, 8, 29, 0)], speed=25)
        helper = AssistController(policy_dt=1/30)
        self.assertEqual(helper.choose(obs, IDLE), SLOWER)
        helper.merge_lane, helper.merge_speed, helper.merge_age = 2, 30, 0
        self.assertEqual(helper.choose(obs, IDLE), SLOWER)
        obs[3, 1] = 96  # Adjacent car is already alongside.
        self.assertEqual(AssistController(policy_dt=1/30).choose(obs, IDLE), SLOWER)

    def test_distant_leader_does_not_trigger_at_high_speed(self):
        from assist import ASSIST_TARGET_SPEEDS
        from highway_env.vehicle.kinematics import Vehicle
        config = copy.deepcopy(ENV_CONFIG)
        config.update(initial_lane_id=1, vehicles_count=0, simulation_frequency=30, policy_frequency=30)
        config["action"]["target_speeds"] = ASSIST_TARGET_SPEEDS
        session = Session(0, config=config)
        try:
            base = session.env.unwrapped
            ego = base.vehicle
            ego.speed = 30  # 108 km/h: above the reported 80 km/h threshold.
            ego.target_speed = 30
            for index, (dx, y, speed) in enumerate([(60, 4, 20), (0, 0, 25), (-15, 8, 29)]):
                car = Vehicle(base.road, [ego.position[0] + dx, y], speed=speed)
                car.project_id = f"traffic_{index}"
                base.road.vehicles.append(car)
            helper = AssistController(policy_dt=1/30)
            session.observation = base.observation_type.observe()
            actions = []
            speeds = []
            for _ in range(180):  # Allow the discrete actuator to slow down.
                action = helper.choose(session.observation, IDLE)
                actions.append(action)
                session.step(action, helper.label, IDLE, helper.decision_info)
                speeds.append(ego.speed)
                self.assertFalse(session.result().collision, (actions, ego.position, ego.speed, base.road.contact_ids))
            self.assertEqual(actions[0], IDLE)
            self.assertTrue(helper.interventions)
            self.assertLess(min(speeds), 30)
        finally:
            session.close()

    def test_manual_speed_commands_reach_the_simulator(self):
        from assist import ASSIST_TARGET_SPEEDS
        config = copy.deepcopy(ENV_CONFIG)
        config["vehicles_count"] = 0
        config["simulation_frequency"] = config["policy_frequency"] = 30
        config["action"]["target_speeds"] = ASSIST_TARGET_SPEEDS
        session = Session(10, config=config)
        try:
            helper = AssistController(policy_dt=1/30)
            ego = session.env.unwrapped.vehicle
            for request in (FASTER, SLOWER):
                before = ego.speed
                for _ in range(30):
                    action = helper.choose(session.observation, request)
                    session.step(action, helper.label, request)
                self.assertEqual(helper.mode, "manual")
                self.assertGreater((ego.speed - before) * (1 if request == FASTER else -1), 1)
        finally:
            session.close()


class ScenarioTests(unittest.TestCase):
    def test_saved_headless_settings_are_overridden_for_window_request(self):
        # Inspect configuration only: constructing a human-mode environment
        # would open a display during reset and is intentionally not done here.
        from hazards import scenario_settings
        saved = copy.deepcopy(ENV_CONFIG)
        saved.update({"offscreen_rendering": True, "real_time_rendering": False})
        original = copy.deepcopy(saved)
        windowed = scenario_settings("human", config=saved)
        self.assertFalse(windowed["offscreen_rendering"])
        self.assertTrue(windowed["real_time_rendering"])
        self.assertEqual(saved, original)
        for key in ENV_CONFIG:
            self.assertEqual(windowed[key], saved[key])
        for mode in (None, "rgb_array"):
            offscreen = scenario_settings(mode, config=windowed)
            self.assertTrue(offscreen["offscreen_rendering"])
            self.assertFalse(offscreen["real_time_rendering"])

    def test_headless_session_never_creates_viewer_from_saved_window_settings(self):
        config = copy.deepcopy(ENV_CONFIG)
        config.update({"offscreen_rendering": False, "real_time_rendering": True})
        session = Session(0, config=config)
        try:
            session.step(IDLE)
            self.assertIsNone(session.env.unwrapped.render_mode)
            self.assertIsNone(session.env.unwrapped.viewer)
            self.assertTrue(session.env.unwrapped.config["offscreen_rendering"])
            self.assertFalse(session.env.unwrapped.config["real_time_rendering"])
            self.assertEqual(replay_document(session.document()), session.result())
        finally:
            session.close()

    def test_custom_controller_label_is_preserved(self):
        import json
        from project_runner import run_recorded
        directory = new_run_directory("test_labels")
        run_recorded("2", 0, max_steps=1, controller=reactive_action,
                     controller_name="reactive", save_directory=directory)
        with (directory / "seed_00.json").open(encoding="utf-8") as stream:
            document = json.load(stream)
        self.assertEqual(document["steps"][0]["controller"], "reactive")

    def test_each_forced_behavior_triggers_and_cut_ins_move_laterally(self):
        config = copy.deepcopy(ENV_CONFIG)
        config["initial_lane_id"] = 1
        for kind in ("hard_brake", "cut_in_left", "cut_in_right"):
            with self.subTest(kind=kind):
                session = Session(10, [{"kind": kind, "trigger_time_s": 0.2,
                                        "gap_m": 30, "spawn": True}], config=config)
                try:
                    actor = session.env.unwrapped.road.vehicles[-1]
                    original_y = float(actor.position[1])
                    session.step(IDLE)
                    self.assertTrue(session.env.unwrapped.events[0]["triggered"])
                    if kind != "hard_brake":
                        self.assertGreater(abs(float(actor.position[1]) - original_y), 0.1)
                finally:
                    session.close()

    def test_seeded_schedule_and_replay(self):
        self.assertEqual(seeded_schedule(10), seeded_schedule(10))
        session = Session(10, seeded_schedule(10, "hard_brake"))
        try:
            for _ in range(5):
                session.step(reactive_action(session.observation.copy()))
                if session.ended:
                    break
            saved = session.document()
            self.assertTrue(any(e["triggered"] for e in saved["hazard_schedule"]))
            self.assertEqual(replay_document(saved), session.result())
            saved["steps"][0]["cars"][0]["speed"] += 1
            with self.assertRaises(AssertionError):
                replay_document(saved)
        finally:
            session.close()

    def test_distance_trigger_and_manual_replay(self):
        session = Session(7, [{"kind": "hard_brake", "trigger_distance_m": 10.0,
                               "gap_m": 30.0, "spawn": True}])
        try:
            session.command("key:up")
            session.step(FASTER, "human", FASTER)
            self.assertTrue(session.env.unwrapped.events[0]["triggered"])
            self.assertEqual(replay_document(session.document()), session.result())
        finally:
            session.close()

    def test_command_parser_and_runtime_event_replay(self):
        self.assertEqual(parse_command("make the car in front brake hard")["kind"], "hard_brake")
        self.assertEqual(parse_command("force a cut-in from the left lane")["kind"], "cut_in_left")
        with self.assertRaises(ValueError):
            parse_command("import os")
        session = Session(0)
        try:
            session.step(IDLE)
            execute_command(session, "brake hard")
            self.assertIn("actor_id", session.commands[0]["event"])
            session.step(IDLE)
            execute_command(session, "save scenario")
            self.assertEqual(session.commands[0]["effective_step"], 2)
            self.assertIn("wall_clock_timestamp", session.commands[0])
            self.assertEqual(replay_document(session.document()), session.result())
        finally:
            session.close()

    def test_collision_attribution_and_secondary_flag(self):
        session = Session(0)
        try:
            for _ in range(40):
                session.step(FASTER)
                if session.ended:
                    break
            self.assertTrue(session.result().collision)
            self.assertTrue(session.hazard_metrics()["secondary_collision"])
            self.assertTrue(session.hazard_metrics()["collision_partners"])
        finally:
            session.close()

    def test_lower_targets_allow_braking_below_twenty(self):
        session = Session(0)
        try:
            for _ in range(3):
                session.step(SLOWER)
                if session.ended:
                    break
            self.assertLess(session.observation[0, 3], 20)
        finally:
            session.close()


class LearningTests(unittest.TestCase):
    def test_baseline_reuse_rejects_changed_conditions(self):
        from learning import read_reactive_baseline
        from driving_common import write_json
        config = copy.deepcopy(ENV_CONFIG)
        config["duration"] = 1
        session = Session(0, config=config)
        try:
            session.step(IDLE, "reactive")
            directory = new_run_directory("test_baseline_reuse")
            write_json(directory / "seed_00.json", session.document())
            actual_config = session.log["env_config"]
            _, result = read_reactive_baseline(directory, 0, actual_config)
            self.assertEqual(result, session.result())
            changed = copy.deepcopy(actual_config)
            changed["simulation_frequency"] = 5
            with self.assertRaises(ValueError):
                read_reactive_baseline(directory, 0, changed)
        finally:
            session.close()

    def test_features_ignore_world_x_and_padding_and_reward_unsafe_traffic(self):
        from learning import encode_observation
        from learning_features import safety_reward
        clear = observation()
        shifted = clear.copy()
        shifted[:, 1] += 500
        np.testing.assert_array_equal(encode_observation(clear), encode_observation(shifted))
        risky = observation([(115, 4, 15, 0)])
        self.assertLess(safety_reward(risky, IDLE, risky, False),
                        safety_reward(clear, IDLE, clear, False))
        self.assertEqual(safety_reward(clear, FASTER, clear, True), -10)
        self.assertEqual(encode_observation(clear, version=1).shape, (51, 5))

    def test_scenario_training_recreates_events_without_using_logged_actions(self):
        from learning import TrainingEnv
        session = Session(1000, seeded_schedule(1000, "hard_brake"))
        try:
            saved = session.document()
        finally:
            session.close()
        # The training path must ignore any per-step records, including malformed
        # action data, because it reconstructs an environment for new actions.
        saved["steps"] = [{"ego_action": "not a training input"}]
        writer = csv.DictWriter(io.StringIO(), fieldnames=["seed"])
        env = TrainingEnv(writer, scenario=saved)
        try:
            env.reset()
            self.assertEqual(env.current_seed, 1000)
            env.step(IDLE)
            env.step(IDLE)
            env.step(IDLE)
            self.assertTrue(env.unwrapped.events[0]["triggered"])
            env.reset()
            self.assertFalse(env.unwrapped.events[0]["triggered"])
        finally:
            env.close()

    def test_training_uses_disjoint_seeds_and_negative_collision_reward(self):
        from learning import TrainingEnv, TRAIN_SEEDS, HELD_OUT_SEEDS, VALIDATION_SEEDS
        self.assertFalse(set(TRAIN_SEEDS) & set(HELD_OUT_SEEDS))
        self.assertFalse(set(VALIDATION_SEEDS) & set(TRAIN_SEEDS + HELD_OUT_SEEDS))
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=["seed", "collision", "closest_distance_m", "average_speed_mps", "steps", "episode_reward"])
        env = TrainingEnv(writer)
        try:
            obs, _ = env.reset()
            self.assertEqual(env.current_seed, 1000)
            self.assertEqual(obs.shape, (24,))
            self.assertTrue(np.isfinite(obs).all())
            self.assertFalse(env.unwrapped.config["normalize_reward"])
            self.assertLess(env.unwrapped.config["collision_reward"], 0)
            # Stock reward calculation must actually produce a negative value.
            env.unwrapped.vehicle.crashed = True
            self.assertLess(env.unwrapped._reward(IDLE), 0)
            env.reset()
            self.assertEqual(env.current_seed, 1001)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
