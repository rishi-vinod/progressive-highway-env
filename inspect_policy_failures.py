"""Replay saved failures headlessly and inspect the observations before impact."""

import argparse
import json
from pathlib import Path

from scenarios import Session
from reactive import assess_lane, lane_number
from learning_features import lane_traffic, lane_risk


def inspect(path):
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    session = Session(document["seed"], document["initial_schedule"], config=document["env_config"])
    try:
        for record in document["steps"]:
            for command in document["commands"]:
                if command["effective_step"] == record["step"]:
                    session.command(command["text"], command["event"],
                                    wall_clock=command["wall_clock_timestamp"])
            if record["step"] > len(document["steps"]) - 3:
                obs = session.observation
                lane = lane_number(obs)
                print(f"seed={document['seed']} step={record['step']} action={record['ego_action']} "
                      f"ego_y={obs[0, 2]:.3f} vx={obs[0, 3]:.3f} vy={obs[0, 4]:.3f}")
                for candidate in (lane - 1, lane, lane + 1):
                    if 0 <= candidate < 4:
                        traffic = lane_traffic(obs, candidate)
                        print(f"  lane={candidate} safe={assess_lane(obs, candidate).safe} "
                              f"front_gap/speed/rear_gap/speed={tuple(round(x, 3) for x in traffic)} "
                              f"risk={lane_risk(obs, candidate, include_rear=True):.3f}")
            session.step(record["ego_action"], record["controller"], record.get("human_action"))
        print(f"collision={session.result().collision}; partners={session.hazard_metrics()['collision_partners']}")
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", type=Path, nargs="+")
    args = parser.parse_args()
    for scenario in args.scenarios:
        inspect(scenario)
