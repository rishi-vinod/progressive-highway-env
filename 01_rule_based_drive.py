"""Stage 1: cruise in the starting lane, ignoring traffic. No learning."""

import argparse
from dataclasses import asdict

from driving_common import (
    METRIC_DEFINITIONS,
    new_run_directory,
    nonnegative_int,
    positive_int,
    run_episode,
    write_json,
)
import numpy as np
from highway_env.envs.common.action import DiscreteMetaAction

TARGET_SPEED_MPS = 30.0
SPEED_TOLERANCE_MPS = 0.5
# Read action codes from the installed library, not from guessed numeric values.
ACTION_CODES = {name: code for code, name in DiscreteMetaAction.ACTIONS_ALL.items()}


def cruise_action(observation: np.ndarray) -> int:
    """Use only observed ego velocity; never inspect other vehicles or plans.

    IDLE retains the stock low-level lane/speed targets; it does not stop the car.
    Choosing no lateral actions keeps the starting lane. The target is the top
    configured speed (30 m/s), so repeated FASTER commands saturate at that target.
    """
    speed = float(np.linalg.norm(observation[0, 3:5]))
    if speed < TARGET_SPEED_MPS - SPEED_TOLERANCE_MPS:
        return ACTION_CODES["FASTER"]
    if speed > TARGET_SPEED_MPS + SPEED_TOLERANCE_MPS:
        return ACTION_CODES["SLOWER"]
    return ACTION_CODES["IDLE"]


def create_controller(seed: int):
    """Shared plugin interface. This deterministic baseline needs no random state."""
    return cruise_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--max-steps", type=positive_int, default=40)
    parser.add_argument("--headless", action="store_true", help="Use render_mode=None.")
    args = parser.parse_args()
    print(f"Stage 1: stay in lane, cruise at {TARGET_SPEED_MPS:.1f} m/s. No learning; crashes expected.")
    try:
        result, config = run_episode(
            create_controller, args.seed, max_steps=args.max_steps,
            headless=args.headless, verbose=True,
        )
        directory = new_run_directory("stage1")
        write_json(directory / "episode.json", {
            "controller": "01_rule_based_drive:create_controller",
            "target_speed_mps": TARGET_SPEED_MPS,
            "env_config": config,
            "metric_definitions": METRIC_DEFINITIONS,
            "episode": asdict(result),
        })
        distance = "n/a" if result.closest_distance_m is None else f"{result.closest_distance_m:.3f} m"
        print(
            f"Episode: collision={'yes' if result.collision else 'no'} "
            f"closest_distance={distance} average_speed={result.average_speed_mps:.3f} m/s "
            f"end={result.end_reason}"
        )
        print(f"Saved {directory / 'episode.json'}")
    except KeyboardInterrupt:
        print("\nStopped early; no completed episode recorded.")
    finally:
        print("Environment closed.")


if __name__ == "__main__":
    main()
