"""Common local run/evaluation entry points for stages 0 through 4."""

import argparse
import copy
import importlib
from dataclasses import asdict

from driving_common import (ENV_CONFIG, EVALUATION_SEEDS, positive_int, nonnegative_int, new_run_directory,
                            print_observation_config, write_json, write_episode_csv)
from hazards import KINDS, seeded_schedule
from scenarios import Session
from reactive import reactive_action
from assist import AssistController
from evaluate import summarize
import numpy as np

baseline = importlib.import_module("01_rule_based_drive").cruise_action


def run_recorded(stage, seed, *, max_steps=40, headless=True, assist_on=True, kind=None,
                 controller=None, save_directory=None, verbose=False, config=None,
                 controller_name="custom"):
    schedule = seeded_schedule(seed, kind) if stage in ("3", "3.5") else []
    if config is None and stage in ("3.5", "4"):
        config = copy.deepcopy(ENV_CONFIG)
        config["policy_frequency"] = 5
    session = Session(seed, schedule, render_mode=None if headless else "human", config=config)
    helper = AssistController(assist_on, 1 / session.env.unwrapped.config["policy_frequency"])
    rng = np.random.default_rng(seed)
    try:
        if verbose:
            print_observation_config(session.env.unwrapped.config)
        for _ in range(max_steps):
            observation = session.observation.copy()
            human_action = None
            if stage == "3.5":
                human_action = int(baseline(observation))
                action = helper.choose(observation, human_action)
                label = "assist" if helper.active else "stand_in"
            else:
                if controller is not None:
                    action, label = int(controller(observation)), controller_name
                elif stage == "0":
                    action, label = int(rng.integers(5)), "random"
                elif stage == "1":
                    action, label = baseline(observation), "rule"
                else:
                    action, label = reactive_action(observation), "rule"
            decision_info = getattr(controller, "decision_info", None) if controller is not None else None
            session.step(int(action), label, human_action, decision_info)
            if verbose:
                if decision_info and decision_info["guard_intervened"]:
                    print(f"GUARD ACTIVE: requested={decision_info['requested_action']} "
                          f"executed={action} reason={decision_info['guard_reason']}", flush=True)
                print(f"step={session.step_number:03d} action={action} controller={label} "
                      f"collision={session.result().collision}", flush=True)
            if session.ended or session.env.unwrapped.done:
                if session.env.unwrapped.done and not session.ended:
                    session.end_reason = "window_closed"
                break
        extra = {"stage": stage, "assist_enabled": assist_on if stage == "3.5" else None,
                 "takeovers": helper.takeovers, **session.hazard_metrics()}
        if controller is not None and hasattr(controller, "decision_info"):
            extra["guard_interventions"] = sum(
                step.get("decision_info", {}).get("guard_intervened", False) for step in session.records)
        result = session.result()
        if save_directory is not None:
            write_json(save_directory / f"seed_{seed:02d}.json", session.document(extra))
        elif verbose:
            print(f"Scenario saved: {session.save(extra=extra)}")
        return result, extra
    finally:
        session.close()


def stage_main(stage):
    parser = argparse.ArgumentParser(description=f"Stage {stage}: local rules, no learning.")
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--max-steps", type=positive_int, default=200 if stage == "3.5" else 40)
    parser.add_argument("--headless", action="store_true")
    if stage == "3":
        parser.add_argument("--hazard", choices=KINDS, default=None)
    if stage == "3.5":
        parser.add_argument("--assist", choices=("on", "off"), default="on")
    args = parser.parse_args()
    if stage == "3.5" and not args.headless:
        from local_ui import run_window
        run_window(stage, args.seed, args.max_steps, args.assist == "on")
        return
    try:
        result, extra = run_recorded(stage, args.seed, max_steps=args.max_steps,
                                     headless=args.headless, kind=getattr(args, "hazard", None),
                                     assist_on=getattr(args, "assist", "on") == "on", verbose=True)
        print(asdict(result))
        if stage in ("3", "3.5"):
            print(extra)
    except KeyboardInterrupt:
        print("Interrupted; environment closed.")


def evaluate_stages(stages, *, smoke=False, max_steps=None):
    seeds = EVALUATION_SEEDS[:2] if smoke else EVALUATION_SEEDS
    root = new_run_directory("benchmark")
    summaries = {}
    print(f"{'SMOKE TEST' if smoke else 'FULL BENCHMARK'} seeds={list(seeds)}", flush=True)
    for stage in stages:
        modes = (False, True) if stage == "3.5" else (True,)
        for assist_on in modes:
            name = f"stage_{stage}"
            if stage == "3.5":
                name += "_assist_on" if assist_on else "_assist_off"
            directory = root / name
            directory.mkdir()
            results, details = [], []
            complete = False
            try:
                for seed in seeds:
                    budget = max_steps if max_steps is not None else 200 if stage in ("3.5", "4") else 40
                    result, extra = run_recorded(stage, seed, max_steps=budget,
                                                 assist_on=assist_on, save_directory=directory)
                    results.append(result)
                    details.append(extra)
                    print(f"{name} seed={seed:02d} collision={result.collision} "
                          f"closest={result.closest_distance_m:.3f}m speed={result.average_speed_mps:.3f}m/s",
                          flush=True)
                complete = True
            finally:
                summary = summarize(results) if results else {}
                triggered = sum(d["hazards_triggered"] for d in details)
                avoided = sum(d["hazards_avoided"] for d in details)
                summary.update({"hazards_triggered": triggered, "hazards_avoided": avoided,
                                "avoided_percent": 100 * avoided / triggered if triggered else None,
                                "secondary_collisions": sum(d["secondary_collision"] for d in details),
                                "takeovers": sum(d["takeovers"] for d in details),
                                "complete": complete, "requested_seeds": list(seeds),
                                "completed_seeds": [r.seed for r in results], "smoke_test": smoke})
                write_episode_csv(directory / "episodes.csv", results)
                write_json(directory / "summary.json", summary)
                summaries[name] = summary
    print("\nstage                         n crashes hazards avoided% secondary closest_m avg_m/s takeovers")
    for name, s in summaries.items():
        avoided = "n/a" if s["avoided_percent"] is None else f"{s['avoided_percent']:.1f}"
        print(f"{name:28s} {s['episodes']:2d} {s['collisions']:7d} {s['hazards_triggered']:7d} "
              f"{avoided:>8s} {s['secondary_collisions']:9d} {s['closest_distance_m']:9.3f} "
              f"{s['mean_episode_average_speed_mps']:7.3f} {s['takeovers']:9d}")
    write_json(root / "summary.json", summaries)
    print(f"Saved benchmark: {root}")
    return root


def benchmark_main():
    parser = argparse.ArgumentParser(description="Same fixed 50 seeds and metrics for every implemented nonlearning stage.")
    parser.add_argument("--stage", choices=("all", "0", "1", "2", "3", "3.5", "4"), default="all")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-steps", type=positive_int, default=None,
                        help="Optional policy-step cap; default is 40 simulated seconds for each profile.")
    args = parser.parse_args()
    stages = ("0", "1", "2", "3", "3.5", "4") if args.stage == "all" else (args.stage,)
    evaluate_stages(stages, smoke=args.smoke_test, max_steps=args.max_steps)
