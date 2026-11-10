"""Versioned local scenario logs and deterministic replay. No arbitrary code loading."""

import copy
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from driving_common import EpisodeResult, METRIC_DEFINITIONS, new_run_directory, write_json
from hazards import make_scenario_env
import numpy as np


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def snapshot(env):
    return [{"id": v.project_id, "position": [float(x) for x in v.position],
             "speed": float(v.speed), "heading": float(v.heading), "crashed": bool(v.crashed)}
            for v in env.unwrapped.road.vehicles]


class Session:
    """The controller gets observations only; this session owns logs and events."""

    def __init__(self, seed, schedule=None, render_mode=None, config=None):
        self.env = make_scenario_env(seed, render_mode, schedule, config)
        self.step_number = 0
        self.ended = False
        self.end_reason = "step_limit"
        self.commands = []
        self.records = []
        try:
            self.observation, _ = self.env.reset(seed=seed)
            self.env.action_space.seed(seed)
            self.log = {"schema_version": 1, "seed": seed,
                        "env_config": copy.deepcopy(self.env.unwrapped.config),
                        "initial_schedule": copy.deepcopy(schedule or []),
                        "metric_definitions": METRIC_DEFINITIONS,
                        "initial_cars": snapshot(self.env), "commands": self.commands,
                        "steps": self.records}
        except BaseException:
            self.env.close()
            raise

    def command(self, text, event=None, *, wall_clock=None):
        """Record a command and the NEXT policy step where it takes effect."""
        entry = {"text": text, "wall_clock_timestamp": wall_clock or timestamp(),
                 "effective_step": self.step_number + 1, "event": copy.deepcopy(event)}
        if event is not None:
            # Runtime commands operate on existing cars; they cannot teleport.
            event = copy.deepcopy(event)
            event["spawn"] = False
            event["trigger_time_s"] = self.env.unwrapped.episode_metrics.elapsed_s
            resolved_event = self.env.unwrapped.add_event(event)
            # Freeze the chosen actor and target lane so scenario training does
            # not silently switch to a different car when the ego drives differently.
            entry["event"] = copy.deepcopy(resolved_event)
        self.commands.append(entry)
        return entry

    def step(self, action, controller="rule", human_action=None, decision_info=None):
        if self.ended:
            raise RuntimeError("Episode already ended")
        if not self.env.action_space.contains(action):
            raise ValueError("Action must be one of the five discrete actions")
        # Use highway-env's manual-control support for human actions. The UI
        # translates keypresses into its discrete action interface; assist uses
        # normal env.step control so manual input cannot override the rescue.
        self.env.unwrapped.config["manual_control"] = controller == "human"
        if controller == "human":
            self.env.unwrapped.action_type.act(int(action))
        self.observation, reward, terminated, truncated, info = self.env.step(int(action))
        self.step_number += 1
        self.ended = bool(terminated or truncated)
        if self.ended:
            self.end_reason = "collision" if info["crashed"] else "terminated" if terminated else "time_limit"
        self.records.append({"step": self.step_number,
                             "simulation_time_s": self.env.unwrapped.episode_metrics.elapsed_s,
                             "ego_action": int(action), "human_action": human_action,
                             "controller": controller, "reward": float(reward),
                             "terminated": bool(terminated), "truncated": bool(truncated),
                             "cars": snapshot(self.env)})
        if decision_info is not None:
            self.records[-1]["decision_info"] = copy.deepcopy(decision_info)
        return self.observation

    def result(self):
        m = self.env.unwrapped.episode_metrics
        return EpisodeResult(self.log["seed"], m.collision, m.closest_distance_m,
                             m.average_speed_mps, self.step_number, m.elapsed_s,
                             m.physics_ticks, self.end_reason)

    def hazard_metrics(self):
        base = self.env.unwrapped
        triggered = [e for e in base.events if e["triggered"]]
        contacts = base.road.contact_ids if self.result().collision else set()
        hazard_ids = {e["actor_id"] for e in triggered}
        # Do not call a hazard avoided if the run ended early or another crash
        # prevented exposure. A full horizon and no ego collision are required.
        avoided = len(triggered) if self.end_reason == "time_limit" and not self.result().collision else 0
        return {"hazards_triggered": len(triggered), "hazards_avoided": avoided,
                "hazard_collisions": len(hazard_ids & contacts),
                "secondary_collision": bool(contacts - hazard_ids),
                "collision_partners": sorted(contacts)}

    def document(self, extra=None):
        return {**self.log, "hazard_schedule": copy.deepcopy(self.env.unwrapped.events),
                "episode": asdict(self.result()), "hazard_metrics": self.hazard_metrics(),
                "extra": extra or {}}

    def save(self, directory=None, extra=None):
        directory = directory or new_run_directory("scenario")
        path = directory / "scenario.json"
        write_json(path, self.document(extra))
        return path

    def close(self):
        self.env.close()


def assert_same(actual, expected, path="root"):
    """Check the whole recorded trajectory, not only the final collision flag."""
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"Different keys at {path}")
        for key in expected:
            assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"Different length at {path}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert_same(a, b, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=1e-9, atol=1e-8):
            raise AssertionError(f"Numeric mismatch at {path}")
    elif actual != expected:
        raise AssertionError(f"Mismatch at {path}")


def replay_document(saved):
    if saved.get("schema_version") != 1:
        raise ValueError("Unsupported scenario schema")
    session = Session(saved["seed"], saved["initial_schedule"], config=saved["env_config"])
    try:
        assert_same(snapshot(session.env), saved["initial_cars"], "initial_cars")
        for record in saved["steps"]:
            for command in saved["commands"]:
                if command["effective_step"] == record["step"]:
                    session.command(command["text"], command["event"], wall_clock=command["wall_clock_timestamp"])
            session.step(record["ego_action"], record["controller"], record.get("human_action"),
                         record.get("decision_info"))
            assert_same(session.records[-1], record, f"step_{record['step']}")
        # Commands entered just before save can be scheduled for the next step.
        for command in saved["commands"]:
            if command["effective_step"] == session.step_number + 1:
                session.command(command["text"], command["event"], wall_clock=command["wall_clock_timestamp"])
        session.end_reason = saved["episode"]["end_reason"]
        assert_same(asdict(session.result()), saved["episode"], "episode")
        assert_same(session.env.unwrapped.events, saved["hazard_schedule"], "hazard_schedule")
        assert_same(session.hazard_metrics(), saved["hazard_metrics"], "hazard_metrics")
        return session.result()
    finally:
        session.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Replay a local JSON scenario headlessly and verify every recorded state.")
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()
    with args.scenario.open(encoding="utf-8") as stream:
        result = replay_document(json.load(stream))
    print(f"Replay verified: seed={result.seed}, steps={result.steps}, collision={result.collision}")


if __name__ == "__main__":
    main()
