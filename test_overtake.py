"""Headless checks for the observation-only passing controller."""

import unittest

from overtake_demo import make_demo_env, run_demo
from reactive import FASTER, IDLE, LANE_LEFT, LANE_RIGHT, OvertakingController
from test_project import observation
from highway_env.vehicle.behavior import IDMVehicle


class OvertakeTests(unittest.TestCase):
    def test_changes_lane_only_for_near_slow_leader(self):
        controller = OvertakingController()
        self.assertEqual(controller(observation(speed=25)), FASTER)
        self.assertEqual(controller(observation([(200, 4, 20, 0)], speed=25)), FASTER)
        controller.reset()
        self.assertEqual(controller(observation([(145, 4, 20, 0)], speed=25)), LANE_RIGHT)
        self.assertEqual(controller.reason, "pass_observed_slow_leader")

    def test_fast_rear_car_blocks_target_lane(self):
        controller = OvertakingController()
        obs = observation([(145, 4, 20, 0), (90, 8, 40, 0)], speed=25)
        self.assertEqual(controller(obs), LANE_LEFT)
        self.assertEqual(OvertakingController()(observation(speed=30)), IDLE)

    def test_traffic_uses_its_own_stock_driver(self):
        env = make_demo_env(render_mode=None)
        try:
            env.reset(seed=0)
            cars = env.unwrapped.road.vehicles[1:]
            self.assertEqual(len(cars), 3)
            self.assertTrue(all(isinstance(car, IDMVehicle) for car in cars))
            before = [float(car.position[0]) for car in cars]
            env.step(IDLE)
            self.assertTrue(all(float(car.position[0]) > x for car, x in zip(cars, before)))
        finally:
            env.close()

    def test_reference_passes_every_car_and_repeats(self):
        first = run_demo(0)
        second = run_demo(0)
        self.assertEqual(first, second)
        self.assertFalse(first[0].collision)
        self.assertEqual(first[1], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
