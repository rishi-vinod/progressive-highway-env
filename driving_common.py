"""Shared environment, observation-only controller interface, and local metrics.

Controllers receive only an observation. Ground truth stays in the metrics code.
No learning or runtime network access is used here.
"""

import argparse
import copy
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

os.environ["MPLCONFIGDIR"] = str(Path("runs") / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str((Path("runs") / "cache").resolve())

import gymnasium as gym
import highway_env
import numpy as np
import pygame
from gymnasium.envs.registration import EnvSpec
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road

gym.register_envs(highway_env)

# This fixed seed set must stay the same across all stage comparisons.
EVALUATION_SEEDS = tuple(range(50))
FEATURES = ["presence", "x", "y", "vx", "vy"]
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "features": FEATURES,
        "vehicles_count": 51,  # Ego + capacity for every stock traffic vehicle.
        "absolute": True,  # All rows: world positions (m) and velocities (m/s).
        "normalize": False,
        "clip": False,
        "order": "sorted",
        "see_behind": True,
        "observe_intentions": False,
        "include_obstacles": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": True,
        "target_speeds": [20, 25, 30],
    },
    "vehicles_count": 50,
    "lanes_count": 4,
    "duration": 40,
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "manual_control": False,
}
METRIC_DEFINITIONS = {
    "collision": "Ego crashed at any sampled physics tick, including reset.",
    "closest_distance_m": (
        "Minimum Euclidean center-to-center distance from ego to any other car, "
        "at reset and after every physics tick; null if no other cars exist."
    ),
    "average_speed_mps": (
        "Trapezoidal integral of ego speed over physics time / elapsed time. "
        "Includes all ticks through the end of the final policy step."
    ),
    "summary_speed": "Unweighted mean of per-episode average speeds.",
}


class Controller(Protocol):
    def __call__(self, observation: np.ndarray) -> int:
        """Choose a discrete action using only the current observation."""
        ...


ControllerFactory = Callable[[int], Controller]


@dataclass
class EpisodeMetrics:
    """Accumulate physics samples without exposing them to a controller."""

    collision: bool = False
    closest_distance_m: float | None = None
    elapsed_s: float = 0.0
    speed_integral: float = 0.0
    previous_speed: float | None = None
    physics_ticks: int = 0

    def observe(self, ego, vehicles, dt: float = 0.0) -> None:
        speed = float(ego.speed)
        self.collision = self.collision or bool(ego.crashed)
        positions = [vehicle.position for vehicle in vehicles if vehicle is not ego]
        if positions:
            distance = float(np.linalg.norm(np.asarray(positions) - ego.position, axis=1).min())
            if self.closest_distance_m is None or distance < self.closest_distance_m:
                self.closest_distance_m = distance
        if dt > 0:
            if self.previous_speed is None:
                raise RuntimeError("Metrics must sample the initial state before stepping.")
            self.speed_integral += (self.previous_speed + speed) * 0.5 * dt
            self.elapsed_s += dt
            self.physics_ticks += 1
        self.previous_speed = speed

    @property
    def average_speed_mps(self) -> float:
        return self.speed_integral / self.elapsed_s if self.elapsed_s else 0.0


class MeasuredRoad(Road):
    """Preserve stock physics and sample ground truth after each physics tick."""

    metrics: EpisodeMetrics | None = None
    ego = None

    def step(self, dt: float) -> None:
        super().step(dt)
        if self.metrics is not None and self.ego is not None:
            self.metrics.observe(self.ego, self.vehicles, dt)


class MeasuredHighwayEnv(HighwayEnv):
    """Stock highway-v0 behavior plus measurement hooks; no library edits."""

    def _create_road(self) -> None:
        super()._create_road()
        # Reuse the stock road network and RNG. Only the post-step hook differs.
        original = self.road
        self.road = MeasuredRoad(
            network=original.network,
            np_random=original.np_random,
            record_history=original.record_history,
            neighbour_vehicles_connected_lanes=original.neighbour_vehicles_connected_lanes,
        )

    def _reset(self) -> None:
        super()._reset()
        self.episode_metrics = EpisodeMetrics()
        self.road.metrics = self.episode_metrics
        self.road.ego = self.vehicle
        self.episode_metrics.observe(self.vehicle, self.road.vehicles)

    def render(self):
        # Stock step() renders again after intermediate frames. Avoid reopening
        # a window whose close event has already set self.done during a frame.
        if self.done:
            return None
        return super().render()


def make_env(render_mode=None):
    """Create the same measured stock-traffic environment for every controller."""
    spec = EnvSpec(id="progressive-highway-v0", entry_point=MeasuredHighwayEnv)
    return gym.make(spec, config=copy.deepcopy(ENV_CONFIG), render_mode=render_mode)


def print_observation_config(config: dict) -> None:
    print("Observation config: " + json.dumps(config["observation"], sort_keys=True))
    print(
        "Rows: ego first; traffic sorted by absolute longitudinal lane distance. "
        "x/y in metres, vx/vy in m/s; presence=0 marks padding. "
        "The stock perception radius still limits visible traffic."
    )


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    collision: bool
    closest_distance_m: float | None
    average_speed_mps: float
    steps: int
    elapsed_s: float
    physics_ticks: int
    end_reason: str


def run_episode(
    controller_factory: ControllerFactory,
    seed: int,
    *,
    max_steps: int = 40,
    headless: bool = True,
    verbose: bool = False,
    env_factory=make_env,
) -> tuple[EpisodeResult, dict]:
    """Run a fresh controller and environment; always close the environment.

    Future controllers plug in as factory(seed) -> callable(observation) -> int.
    An environment factory may be supplied for later authorized scenario stages;
    it must provide episode_metrics with the same sampling definitions.
    """
    if seed < 0 or max_steps < 1:
        raise ValueError("seed must be nonnegative and max_steps must be positive")
    env = None
    steps = 0
    end_reason = "step_limit"
    try:
        controller = controller_factory(seed)
        env = env_factory(render_mode=None if headless else "human")
        observation, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        base = env.unwrapped
        config = copy.deepcopy(base.config)
        if verbose:
            print_observation_config(config)
            print(f"seed={seed} headless={headless} other_cars={len(base.road.vehicles) - 1}")
        for _ in range(max_steps):
            if base.done:
                end_reason = "window_closed"
                break
            # A defensive copy prevents controllers from modifying the runner's
            # observation. No env, info, vehicle objects, or metrics are passed.
            action = controller(observation.copy())
            if not env.action_space.contains(action):
                raise ValueError(f"Controller returned an invalid discrete action: {action!r}")
            observation, reward, terminated, truncated, info = env.step(int(action))
            steps += 1
            if verbose:
                print(
                    f"step={steps:03d} action={int(action)} ({base.action_type.actions[int(action)]}) "
                    f"reward={reward:.6f} collision={bool(info['crashed'])} "
                    f"ego_speed={float(np.linalg.norm(observation[0, 3:5])):.3f} m/s",
                    flush=True,
                )
            if base.done:
                end_reason = "window_closed"
                break
            if terminated or truncated:
                end_reason = "collision" if info["crashed"] else "terminated" if terminated else "time_limit"
                break
        metrics = base.episode_metrics
        return EpisodeResult(
            seed=seed,
            collision=metrics.collision,
            closest_distance_m=metrics.closest_distance_m,
            average_speed_mps=metrics.average_speed_mps,
            steps=steps,
            elapsed_s=metrics.elapsed_s,
            physics_ticks=metrics.physics_ticks,
            end_reason=end_reason,
        ), config
    except pygame.error:
        if env is None or not env.unwrapped.done:
            raise
        # Treat an interrupted render like Ctrl+C: never log it as a full episode.
        raise KeyboardInterrupt("Window closed during rendering") from None
    finally:
        if env is not None:
            env.close()
        pygame.quit()


def new_run_directory(label: str) -> Path:
    """Write only beneath runs/, using a unique name so nothing is overwritten."""
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in label):
        raise ValueError("Run labels must contain lowercase letters, digits, _ or -")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path("runs") / f"{label}_{timestamp}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path: Path, content) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_episode_csv(path: Path, results: list[EpisodeResult]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(EpisodeResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["collision"] = "yes" if result.collision else "no"
            writer.writerow(row)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number
