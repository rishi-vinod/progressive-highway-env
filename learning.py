"""Stage 5 only: short local CPU learning and held-out evaluation.

The training code uses observations, seeds and event schedules. It never trains
from saved per-step vehicle records or downloads a pretrained model.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

from driving_common import (ENV_CONFIG, EVALUATION_SEEDS, positive_int,
                            nonnegative_int, new_run_directory, write_json, write_episode_csv,
                            EpisodeResult)
from hazards import make_scenario_env, seeded_schedule
from project_runner import run_recorded
from evaluate import summarize
from reactive import reactive_action
from learning_features import FEATURE_VERSION, FEATURE_COUNT, encode_lanes, safety_reward
import gymnasium as gym
import numpy as np

TRAIN_SEEDS = tuple(range(1000, 1050))
HELD_OUT_SEEDS = EVALUATION_SEEDS
VALIDATION_SEEDS = tuple(range(2000, 2005))


def encode_observation(observation, version=FEATURE_VERSION):
    """Numerical conditioning only; all inputs still come from the observation."""
    if version == FEATURE_VERSION:
        return encode_lanes(observation)
    if version != 1:
        raise ValueError(f"Unsupported feature version: {version}")
    encoded = observation.astype(np.float32, copy=True)
    present = encoded[:, 0] > 0.5
    # Remove world-x drift while retaining ego lane and absolute ego speed.
    encoded[1:, 1:3] -= encoded[0, 1:3]
    encoded[0, 1] = 0
    encoded[:, 1:3] /= np.array([200.0, 12.0], dtype=np.float32)
    encoded[:, 3:5] /= 30.0
    encoded[~present] = 0
    return np.clip(encoded, -10, 10)


class TrainingEnv(gym.Wrapper):
    """Cycle training seeds, resetting the scene/event schedule each episode."""

    def __init__(self, writer, *, hazards=False, scenario=None):
        self.training_seeds = (int(scenario["seed"]),) if scenario else TRAIN_SEEDS
        if set(self.training_seeds) & set(HELD_OUT_SEEDS + VALIDATION_SEEDS):
            raise ValueError("Training scenario overlaps reserved evaluation/validation seeds; use seed 1000-1049.")
        self.cursor = 0
        self.writer = writer
        self.hazards = hazards
        self.scenario = scenario
        self.policy_steps = 0
        self.total_reward = 0.0
        self.unavailable_events = 0
        config = copy.deepcopy(scenario["env_config"] if scenario else ENV_CONFIG)
        config.update({"collision_reward": -5.0, "normalize_reward": False, "manual_control": False})
        super().__init__(make_scenario_env(self.training_seeds[0], config=config))
        self.observation_space = gym.spaces.Box(-10, 10, (FEATURE_COUNT,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.current_seed = self.training_seeds[self.cursor % len(self.training_seeds)]
        self.cursor += 1
        self.policy_steps = 0
        self.total_reward = 0.0
        self.unavailable_events = 0
        schedule = (self.scenario["initial_schedule"] if self.scenario else
                    seeded_schedule(self.current_seed) if self.hazards else [])
        self.env.unwrapped.config["hazard_schedule"] = copy.deepcopy(schedule)
        observation, info = self.env.reset(seed=self.current_seed, options=options)
        self.previous_observation = observation.copy()
        return encode_observation(observation), info

    def step(self, action):
        if self.scenario:
            for command in self.scenario["commands"]:
                if command["effective_step"] == self.policy_steps + 1 and command["event"] is not None:
                    try:
                        self.env.unwrapped.add_event(command["event"])
                    except ValueError:
                        # A different learned trajectory can crash the scheduled
                        # actor first. Do not resurrect or teleport it.
                        self.unavailable_events += 1
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = safety_reward(self.previous_observation, action, observation, info["crashed"])
        self.previous_observation = observation.copy()
        self.policy_steps += 1
        self.total_reward += float(reward)
        info["unavailable_scheduled_events"] = self.unavailable_events
        if terminated or truncated:
            m = self.env.unwrapped.episode_metrics
            self.writer.writerow({"seed": self.current_seed, "collision": "yes" if m.collision else "no",
                                  "closest_distance_m": m.closest_distance_m,
                                  "average_speed_mps": m.average_speed_mps,
                                  "steps": self.policy_steps, "episode_reward": self.total_reward})
        return encode_observation(observation), reward, terminated, truncated, info


def load_policy(path):
    from stable_baselines3 import DQN
    import torch
    torch.set_num_threads(1)
    model = DQN.load(path, device="cpu")
    # Older local checkpoints used the full 51-by-5 observation array.
    if model.observation_space.shape == (51, 5):
        version = 1
    elif model.observation_space.shape == (FEATURE_COUNT,):
        version = FEATURE_VERSION
    else:
        raise ValueError(f"Unsupported policy observation shape: {model.observation_space.shape}")
    if str(model.device) != "cpu":
        raise RuntimeError("Only CPU inference is allowed")
    print(f"Loaded CPU policy: {path}; feature_version={version}; "
          f"gradient_updates={model._n_updates}", flush=True)

    def controller(observation):
        action, _ = model.predict(encode_observation(observation, version), deterministic=True)
        return int(action)

    return controller


def train(timesteps=10000, *, hazards=False, scenario_path=None):
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    import torch
    torch.set_num_threads(1)
    scenario = None
    if scenario_path:
        with Path(scenario_path).open(encoding="utf-8") as stream:
            scenario = json.load(stream)
        if scenario.get("schema_version") != 1:
            raise ValueError("Unsupported scenario schema")
        if scenario["seed"] in HELD_OUT_SEEDS:
            raise ValueError("Saved training scenarios must use seeds outside held-out 0-49.")
    directory = new_run_directory("stage5")

    class Progress(BaseCallback):
        best_score = None
        selected_step = None
        validation_runs = []

        def validate(self):
            def policy(observation):
                action, _ = self.model.predict(encode_observation(observation), deterministic=True)
                return int(action)

            results = []
            config = copy.deepcopy(env.unwrapped.config)
            config["hazard_schedule"] = []
            horizon = int(config["duration"] * config["policy_frequency"])
            for seed in VALIDATION_SEEDS:
                result, _ = run_recorded("3" if hazards else "2", seed,
                                         controller=policy, config=config, max_steps=horizon)
                results.append(result)
            summary = summarize(results)
            score = (-summary["collisions"], summary["mean_episode_average_speed_mps"])
            checkpoint = f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(directory / checkpoint)
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.selected_step = self.num_timesteps
            self.validation_runs.append({"timesteps": self.num_timesteps, "checkpoint": checkpoint,
                                         "summary": summary})
            print(f"Validation at {self.num_timesteps}: collisions={summary['collisions']}/5, "
                  f"speed={summary['mean_episode_average_speed_mps']:.3f} m/s", flush=True)

        def _on_step(self):
            if self.num_timesteps % 250 == 0:
                print(f"CPU training: {self.num_timesteps}/{timesteps} transitions", flush=True)
            if self.num_timesteps % 2000 == 0:
                self.validate()
            return True

    fields = ["seed", "collision", "closest_distance_m", "average_speed_mps", "steps", "episode_reward"]
    with (directory / "training_episodes.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        env = TrainingEnv(writer, hazards=hazards, scenario=scenario)
        try:
            model = DQN("MlpPolicy", env, device="cpu", seed=123,
                        learning_rate=5e-4, buffer_size=20000, learning_starts=256,
                        batch_size=64, train_freq=1, gradient_steps=1, gamma=0.95,
                        target_update_interval=250, exploration_fraction=0.6,
                        exploration_final_eps=0.05, policy_kwargs={"net_arch": [64, 64]},
                        tensorboard_log=None, verbose=0)
            model.set_logger(configure(str(directory), ["csv"]))
            initial_weights = [p.detach().clone() for p in model.policy.parameters()]
            print(f"Training {timesteps} transitions on CPU; train seeds={list(env.training_seeds)}", flush=True)
            progress = Progress()
            model.learn(total_timesteps=timesteps, callback=progress)
            if not progress.validation_runs or progress.validation_runs[-1]["timesteps"] != model.num_timesteps:
                progress.validate()
            weights_changed = any(not torch.equal(before, after.detach())
                                  for before, after in zip(initial_weights, model.policy.parameters()))
            # Select using validation only, then freeze before held-out testing.
            selected = DQN.load(directory / f"checkpoint_{progress.selected_step}.zip", device="cpu")
            selected.save(directory / "policy.zip")
            metadata = {"device": str(model.device), "requested_timesteps": timesteps,
                        "actual_timesteps": model.num_timesteps, "weights_changed": weights_changed,
                        "gradient_updates": model._n_updates, "training_seeds": list(env.training_seeds),
                        "held_out_seeds": list(HELD_OUT_SEEDS), "hazards": hazards,
                        "validation_seeds": list(VALIDATION_SEEDS),
                        "validation": progress.validation_runs,
                        "selected_timesteps": progress.selected_step,
                        "selected_gradient_updates": selected._n_updates,
                        "feature_version": FEATURE_VERSION,
                        "reward": "safe progress: speed/60 - lane risk - lane-change risk; collision=-10",
                        "scenario_training": scenario is not None,
                        "training_config": env.unwrapped.config,
                        "claim": "Short local training; evaluation must establish performance."}
            write_json(directory / "training.json", metadata)
            print(f"Saved {directory / 'policy.zip'}; weights_changed={weights_changed}; updates={model._n_updates}")
        finally:
            env.close()
    return directory


def read_reactive_baseline(directory, seed, config):
    """Reuse a completed local baseline only under exactly matching conditions."""
    with (Path(directory) / f"seed_{seed:02d}.json").open(encoding="utf-8") as stream:
        document = json.load(stream)
    if (document["seed"] != seed or document["episode"]["seed"] != seed
            or document["env_config"] != config or document["commands"] or document["initial_schedule"]
            or document["episode"]["end_reason"] not in ("collision", "time_limit")
            or document["episode"]["steps"] != len(document["steps"])
            or not all(s["controller"] == "reactive" for s in document["steps"])):
        raise ValueError("Baseline seed, full-horizon status, controller or config does not match.")
    return document, EpisodeResult(**document["episode"])


def evaluate_policy(model_path, *, smoke=False, hazards=False, reuse_reactive=None):
    # Check the training provenance before using the fixed held-out benchmark.
    with (Path(model_path).parent / "training.json").open(encoding="utf-8") as stream:
        provenance = json.load(stream)
    if set(provenance["training_seeds"]) & set(HELD_OUT_SEEDS):
        raise ValueError("Training and evaluation seeds overlap")
    learned = load_policy(model_path)
    evaluation_config = copy.deepcopy(provenance["training_config"])
    evaluation_config["manual_control"] = False
    evaluation_config["hazard_schedule"] = []
    horizon = int(evaluation_config["duration"] * evaluation_config["policy_frequency"])
    seeds = HELD_OUT_SEEDS[:2] if smoke else HELD_OUT_SEEDS
    if reuse_reactive is not None and hazards:
        raise ValueError("Baseline reuse supports stock traffic only; rerun the paired hazard comparison.")
    baseline_source = (Path(reuse_reactive).resolve().relative_to(Path.cwd()).as_posix()
                       if reuse_reactive is not None else None)
    if baseline_source is not None:
        print(f"Checking and reusing stored reactive baseline: {baseline_source}", flush=True)
    directory = new_run_directory("stage5_evaluation")
    summaries = {}
    for name, controller in (("reactive", reactive_action), ("learned", learned)):
        child = directory / name
        child.mkdir()
        results = []
        for seed in seeds:
            if name == "reactive" and reuse_reactive is not None:
                document, result = read_reactive_baseline(reuse_reactive, seed, evaluation_config)
                write_json(child / f"seed_{seed:02d}.json", document)
            else:
                result, _ = run_recorded("3" if hazards else "2", seed, controller=controller,
                                         save_directory=child, config=evaluation_config, max_steps=horizon,
                                         controller_name=name)
            results.append(result)
            print(f"{name} held-out seed={seed:02d} collision={result.collision} "
                  f"speed={result.average_speed_mps:.3f}", flush=True)
        write_episode_csv(child / "episodes.csv", results)
        summaries[name] = summarize(results)
    write_json(directory / "summary.json", {"summaries": summaries, "evaluation_seeds": list(seeds),
                                             "training_seeds": provenance["training_seeds"],
                                             "validation_seeds": provenance.get("validation_seeds", []),
                                             "reused_reactive_baseline": reuse_reactive is not None,
                                             "reactive_baseline_source": baseline_source,
                                             "smoke_test": smoke, "hazards": hazards})
    print("controller episodes collisions min_distance_m mean_speed_mps")
    for name, s in summaries.items():
        print(f"{name:10s} {s['episodes']:8d} {s['collisions']:10d} {s['closest_distance_m']:14.3f} "
              f"{s['mean_episode_average_speed_mps']:14.3f}")
    print(f"Saved held-out comparison: {directory}")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--timesteps", type=positive_int, default=10000)
    train_parser.add_argument("--hazards", action="store_true")
    train_parser.add_argument("--scenario", type=Path, help="Recreate seed/events, never train on per-step records.")
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("model", type=Path)
    eval_parser.add_argument("--smoke-test", action="store_true")
    eval_parser.add_argument("--hazards", action="store_true")
    eval_parser.add_argument("--reuse-reactive", type=Path,
                             help="Reuse an existing reactive result folder only if seeds/config/horizon match.")
    drive_parser = subparsers.add_parser("drive", help="Watch a saved learned policy; no further training.")
    drive_parser.add_argument("model", type=Path)
    drive_parser.add_argument("--seed", type=nonnegative_int, default=0)
    drive_parser.add_argument("--headless", action="store_true")
    drive_parser.add_argument("--hazards", action="store_true")
    drive_parser.add_argument("--max-steps", type=positive_int, default=None)
    args = parser.parse_args()
    if args.operation == "train":
        train(args.timesteps, hazards=args.hazards, scenario_path=args.scenario)
    elif args.operation == "evaluate":
        evaluate_policy(args.model, smoke=args.smoke_test, hazards=args.hazards,
                        reuse_reactive=args.reuse_reactive)
    else:
        with (args.model.parent / "training.json").open(encoding="utf-8") as stream:
            provenance = json.load(stream)
        config = copy.deepcopy(provenance["training_config"])
        config.update({"manual_control": False, "hazard_schedule": []})
        horizon = args.max_steps or int(config["duration"] * config["policy_frequency"])
        try:
            result, _ = run_recorded("3" if args.hazards else "2", args.seed,
                                     max_steps=horizon, headless=args.headless,
                                     controller=load_policy(args.model), controller_name="learned",
                                     config=config, verbose=True)
            print(result)
        except KeyboardInterrupt:
            print("Interrupted; environment closed.")


if __name__ == "__main__":
    main()
