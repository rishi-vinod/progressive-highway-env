"""Headless, observation-only decision trace for the saved Stage 5 policy.

Use --debug to save one CSV row per decision under logs/. No vehicle state is
fed back to the policy; observation rows are identified by their row number.
"""

import argparse
import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from driving_common import EVALUATION_SEEDS
from learning import load_policy
from reactive import CAR_LENGTH_M, LANE_WIDTH_M, SLOWER, lane_number
from scenarios import Session

BRAKING_RANGE_M = 45.0


def leader(observation, lane):
    ego = observation[0]
    candidates = []
    for index, row in enumerate(observation[1:], start=1):
        if row[0] < 0.5 or abs(float(row[2]) - lane * LANE_WIDTH_M) >= 2.5:
            continue
        dx = float(row[1] - ego[1])
        if dx > 0:
            candidates.append((dx, index, row))
    if not candidates:
        return "none", None, None, None
    dx, index, row = min(candidates)
    closing = float(ego[3] - row[3])
    gap = dx - CAR_LENGTH_M
    return f"obs_row_{index}", gap, closing, gap / closing if closing > 0 else None


def trace(model_path, seeds, *, debug=False, output=None):
    with (model_path.parent / "training.json").open(encoding="utf-8") as stream:
        config = copy.deepcopy(json.load(stream)["training_config"])
    config.update({"manual_control": False, "hazard_schedule": []})
    controller = load_policy(model_path, guard=True)
    horizon = int(config["duration"] * config["policy_frequency"])
    rows, episodes, signatures = [], [], []
    for seed in seeds:
        session = Session(seed, config=config)
        controller.reset()
        changes = brakes_without_leader = 0
        try:
            for step in range(1, horizon + 1):
                obs = session.observation.copy()
                lane = lane_number(obs)
                leader_id, gap, closing, ttc = leader(obs, lane)
                action = controller(obs)
                decision = controller.decision_info
                requested = decision["requested_action"]
                target_lane = lane + (-1 if action == 0 else 1 if action == 2 else 0)
                # Only log the simulator target for diagnosis; the controller
                # never receives it or any ground-truth vehicle state.
                target_speed = float(session.env.unwrapped.vehicle.target_speed)
                reason = decision["guard_reason"] or ("model_brake" if action == 4 else
                         "model_lane_change" if action in (0, 2) else "")
                if action == SLOWER and (gap is None or gap > BRAKING_RANGE_M):
                    brakes_without_leader += 1
                changes += action in (0, 2)
                signatures.append((seed, step, action, requested, float(obs[0, 1]), float(obs[0, 3])))
                if debug:
                    rows.append({"seed": seed, "step": step, "ego_speed_mps": float(np.linalg.norm(obs[0, 3:5])),
                                 "ego_lane": lane, "target_lane": target_lane, "target_speed_mps": target_speed,
                                 "requested_action": requested, "chosen_action": action, "leader_id": leader_id,
                                 "leader_bumper_gap_m": gap, "leader_closing_speed_mps": closing,
                                 "ttc_s": ttc, "reason": reason})
                session.step(action, "learned_guarded", decision_info=decision)
                if session.ended:
                    break
            result = session.result()
            episodes.append({"seed": seed, "collision": result.collision,
                             "closest_distance_m": result.closest_distance_m,
                             "average_speed_mps": result.average_speed_mps,
                             "lane_changes": changes, "brakes_without_leader": brakes_without_leader})
        finally:
            session.close()
    if debug:
        output = output or Path("logs") / (
            f"stage5_decisions_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}.csv"
        )
        output.parent.mkdir(exist_ok=True)
        with output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Decision log: {output}")
    n = len(episodes)
    summary = {"episodes": n, "collisions": sum(e["collision"] for e in episodes),
               "closest_distance_m": min(e["closest_distance_m"] for e in episodes),
               "average_speed_mps": sum(e["average_speed_mps"] for e in episodes) / n,
               "brakes_without_leader": sum(e["brakes_without_leader"] for e in episodes),
               "lane_changes_per_episode": sum(e["lane_changes"] for e in episodes) / n}
    print(json.dumps(summary, indent=2))
    return episodes, summary, signatures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--debug", action="store_true", help="Save per-step CSV under logs/.")
    parser.add_argument("--seeds", type=int, default=50, choices=range(1, 51))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeat-check", action="store_true", help="Run seed 0 twice and compare metrics.")
    args = parser.parse_args()
    trace(args.model, EVALUATION_SEEDS[:args.seeds], debug=args.debug, output=args.output)
    if args.repeat_check:
        first = trace(args.model, (0,))
        second = trace(args.model, (0,))
        if first != second:
            raise AssertionError("Seed 0 did not reproduce the same result")
        print("Determinism check: seed 0 produced identical actions, sampled states, and metrics twice.")
