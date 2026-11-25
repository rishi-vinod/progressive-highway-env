"""A controlled passing reference: stock IDM traffic, observation-only ego.

Three independently controlled cars start ahead in the ego lane. Their IDM
and MOBIL decisions run inside highway-env; the ego controller receives only
the current kinematics observation, never their objects or future actions.
"""

import argparse
import copy
from dataclasses import asdict

from driving_common import ENV_CONFIG, EVALUATION_SEEDS, run_episode, positive_int, nonnegative_int
from hazards import ScenarioEnv, scenario_settings
from reactive import OvertakingController
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle

TRAFFIC_OFFSETS_M = (70.0, 150.0, 230.0)
TRAFFIC_SPEED_MPS = 20.0


class OvertakeDemoEnv(ScenarioEnv):
    """Place stock autonomous vehicles once at reset; never steer them."""

    def _create_vehicles(self):
        starting = Vehicle.create_random(self.road, speed=25.0, lane_id=1, spacing=2)
        ego = self.action_type.vehicle_class(
            self.road, starting.position, starting.heading, starting.speed
        )
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)
        for offset in TRAFFIC_OFFSETS_M:
            car = IDMVehicle(
                self.road,
                np.array([float(ego.position[0]) + offset, 4.0]),
                heading=0.0, speed=TRAFFIC_SPEED_MPS,
                target_speed=TRAFFIC_SPEED_MPS,
            )
            car.randomize_behavior()  # Uses the environment's seeded RNG.
            self.road.vehicles.append(car)


def make_demo_env(render_mode=None):
    config = copy.deepcopy(ENV_CONFIG)
    config.update({"vehicles_count": len(TRAFFIC_OFFSETS_M), "initial_lane_id": 1,
                   "simulation_frequency": 30})
    settings = scenario_settings(render_mode, config=config)
    spec = EnvSpec(id="progressive-overtake-demo-v0", entry_point=OvertakeDemoEnv)
    return gym.make(spec, config=settings, render_mode=render_mode)


def run_demo(seed, *, headless=True, max_steps=40, verbose=False):
    opened = []

    def factory(render_mode=None):
        env = make_demo_env(render_mode)
        opened.append(env)
        return env

    result, _ = run_episode(lambda _: OvertakingController(), seed, max_steps=max_steps,
                            headless=headless, verbose=verbose, env_factory=factory)
    base = opened[0].unwrapped
    ego_x = float(base.vehicle.position[0])
    passed = sum(ego_x > float(car.position[0]) + 5.0 for car in base.road.vehicles[1:])
    return result, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=positive_int, default=40)
    parser.add_argument("--evaluate", action="store_true", help="Run fixed 50 seeds headlessly.")
    parser.add_argument("--debug", action="store_true", help="Print each observed decision and its reason.")
    args = parser.parse_args()
    if args.evaluate:
        results = [run_demo(seed, headless=True, max_steps=args.max_steps)
                   for seed in EVALUATION_SEEDS]
        episodes = [result for result, _ in results]
        print({"episodes": len(episodes), "collisions": sum(r.collision for r in episodes),
               "closest_distance_m": min(r.closest_distance_m for r in episodes),
               "average_speed_mps": sum(r.average_speed_mps for r in episodes) / len(episodes),
               "cars_passed_per_episode": sum(passed for _, passed in results) / len(results),
               "all_three_passed": sum(passed == 3 for _, passed in results)})
    else:
        result, passed = run_demo(args.seed, headless=args.headless,
                                  max_steps=args.max_steps, verbose=args.debug)
        print({**asdict(result), "cars_passed": passed})


if __name__ == "__main__":
    main()
