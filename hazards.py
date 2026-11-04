"""Developer-forced vehicle behavior. Only the simulator reads these schedules."""

import copy
from functools import partial

from driving_common import ENV_CONFIG, MeasuredHighwayEnv
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec
from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

KINDS = ("hard_brake", "cut_in_left", "cut_in_right")


class TrackedEgo(MDPVehicle):
    """Attribute collisions using the library's collision solver, not proximity."""

    def _is_colliding(self, other, dt):
        result = super()._is_colliding(other, dt)
        if (result[0] or result[1]) and self.solid and other.solid:
            # The solver may assign an impact which is applied on the next tick.
            # Count these contacts as collisions only when the ego actually crashes.
            if hasattr(self.road, "contact_ids"):
                self.road.contact_ids.add(getattr(other, "project_id", "object"))
        return result


class TrackedActions(DiscreteMetaAction):
    @property
    def vehicle_class(self):
        return partial(TrackedEgo, target_speeds=self.target_speeds)


class ForcedVehicle(IDMVehicle):
    """Follow stock IDM until a seeded time/distance trigger, then force an event."""

    event = None

    def act(self, action=None):
        if self.event is None or self.crashed:
            return super().act(action)
        event = self.event
        elapsed = self.road.metrics.elapsed_s
        distance = float(self.road.ego.position[0] - self.road.ego_start_x)
        ready = (distance >= event["trigger_distance_m"] if "trigger_distance_m" in event
                 else elapsed + 1e-9 >= event["trigger_time_s"])
        if not event["triggered"] and ready:
            event["triggered"] = True
            event["triggered_time_s"] = elapsed
            event["triggered_physics_tick"] = self.road.metrics.physics_ticks
            event["triggered_distance_m"] = distance
            self.target_lane_index = ("0", "1", event["target_lane"])
        if not event["triggered"]:
            return super().act(action)
        if elapsed - event["triggered_time_s"] > event.get("duration_s", 3.0):
            return super().act(action)
        steering = self.steering_control(self.target_lane_index)
        acceleration = (-6.0 if self.speed > 0.5 else 0.0) if event["kind"] == "hard_brake" else self.speed_control(self.target_speed)
        Vehicle.act(self, {"steering": float(np.clip(steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)),
                           "acceleration": float(np.clip(acceleration, -6.0, 3.0))})


class HardBrakeVehicle(ForcedVehicle):
    """Forced braking at up to 6 m/s² after the scheduled trigger."""


class CutInVehicle(ForcedVehicle):
    """Forced lane change using the stock low-level steering controller."""


def seeded_schedule(seed, kind=None):
    rng = np.random.default_rng(seed)
    return [{"kind": kind or str(rng.choice(KINDS)),
             "trigger_time_s": float(rng.uniform(1.0, 2.5)),
             "gap_m": float(rng.uniform(24.0, 34.0)), "duration_s": 3.0, "spawn": True}]


class ScenarioEnv(MeasuredHighwayEnv):
    def define_spaces(self):
        super().define_spaces()
        options = {k: v for k, v in self.config["action"].items() if k != "type"}
        self.action_type = TrackedActions(self, **options)
        self.action_space = self.action_type.space()

    def _reset(self):
        super()._reset()
        self.road.contact_ids = set()
        self.road.ego_start_x = float(self.vehicle.position[0])
        self.events = []
        for i, vehicle in enumerate(self.road.vehicles):
            vehicle.project_id = f"car_{i}"
        for event in self.config.get("hazard_schedule", []):
            self.add_event(event)
        # Re-sample the reset state after the developer arranged the scenario.
        from driving_common import EpisodeMetrics
        self.episode_metrics = EpisodeMetrics()
        self.road.metrics = self.episode_metrics
        self.episode_metrics.observe(self.vehicle, self.road.vehicles)

    def add_event(self, specification):
        """At reset, stage an actor; at runtime, use an existing traffic car."""
        event = copy.deepcopy(specification)
        if event.get("kind") not in KINDS:
            raise ValueError("Unknown hazard kind")
        lane = int(event.get("target_lane", self.vehicle.lane_index[2]))
        source_lane = lane + (-1 if event["kind"] == "cut_in_left" else 1 if event["kind"] == "cut_in_right" else 0)
        if not 0 <= source_lane < self.config["lanes_count"]:
            if not event.get("spawn", False):
                raise ValueError("There is no lane on that side")
            source_lane = lane + (1 if lane == 0 else -1)
            event["kind"] = "cut_in_right" if source_lane > lane else "cut_in_left"
        used = {e["actor_id"] for e in self.events}
        candidates = [v for v in self.road.vehicles[1:] if v.project_id not in used and not v.crashed]
        if event.get("spawn", False):
            if not candidates:
                raise ValueError("No traffic vehicle available for a hazard")
            original = candidates[-1]
            position = np.array([self.vehicle.position[0] + event.get("gap_m", 28.0), source_lane * 4.0])
            # Never start with overlapping vehicles. Shift only the staged actor,
            # before the episode begins; running commands never teleport cars.
            while any(np.linalg.norm(v.position - position) < 10 for v in self.road.vehicles if v is not original):
                position[0] += 12.0
            speed, heading = 25.0, 0.0
        else:
            if "actor_id" in event:
                candidates = [v for v in candidates if v.project_id == event["actor_id"]]
            else:
                candidates = [v for v in candidates if v.lane_index[2] == source_lane and v.position[0] > self.vehicle.position[0]]
            if not candidates:
                raise ValueError("No eligible car ahead in that lane (or the scheduled actor is unavailable)")
            original = min(candidates, key=lambda v: float(v.position[0] - self.vehicle.position[0]))
            position, speed, heading = original.position.copy(), original.speed, original.heading
        cls = HardBrakeVehicle if event["kind"] == "hard_brake" else CutInVehicle
        actor = cls(self.road, position, heading, speed, target_speed=speed, enable_lane_change=False)
        actor.project_id = original.project_id
        actor.DELTA = original.DELTA
        actor.event = event
        event.update({"actor_id": actor.project_id, "target_lane": lane, "triggered": False})
        self.road.vehicles[self.road.vehicles.index(original)] = actor
        self.events.append(event)
        return event


def make_scenario_env(seed, render_mode=None, schedule=None, config=None):
    settings = copy.deepcopy(ENV_CONFIG if config is None else config)
    settings["hazard_schedule"] = copy.deepcopy(schedule or [])
    spec = EnvSpec(id="progressive-scenarios-v0", entry_point=ScenarioEnv)
    return gym.make(spec, config=settings, render_mode=render_mode)
