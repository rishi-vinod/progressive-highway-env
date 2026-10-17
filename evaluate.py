"""Evaluate observation-only controllers on the same 50 seeds, always headless."""

import argparse
import importlib
from dataclasses import asdict

from driving_common import (
    EVALUATION_SEEDS,
    METRIC_DEFINITIONS,
    new_run_directory,
    positive_int,
    print_observation_config,
    run_episode,
    write_episode_csv,
    write_json,
)


def load_controller_factory(specification: str):
    """Load local project Python code using module:function syntax."""
    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Use a controller specification such as 01_rule_based_drive:create_controller")
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise ValueError("Controller factory must be callable")
    return factory


def summarize(results):
    distances = [r.closest_distance_m for r in results if r.closest_distance_m is not None]
    return {
        "episodes": len(results),
        "collisions": sum(r.collision for r in results),
        "collision_rate_percent": 100 * sum(r.collision for r in results) / len(results),
        "closest_distance_m": min(distances) if distances else None,
        "mean_episode_closest_distance_m": sum(distances) / len(distances) if distances else None,
        "mean_episode_average_speed_mps": sum(r.average_speed_mps for r in results) / len(results),
        "step_limited_episodes": sum(r.end_reason == "step_limit" for r in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="01_rule_based_drive:create_controller")
    parser.add_argument("--max-steps", type=positive_int, default=40)
    parser.add_argument("--smoke-test", action="store_true", help="Only seeds 0 and 1; not the 50-seed benchmark.")
    args = parser.parse_args()
    factory = load_controller_factory(args.controller)
    seeds = EVALUATION_SEEDS[:2] if args.smoke_test else EVALUATION_SEEDS
    label = "SMOKE TEST" if args.smoke_test else "50-SEED EVALUATION"
    print(f"{label}: controller={args.controller} seeds={list(seeds)} render_mode=None", flush=True)
    directory = new_run_directory("evaluation")
    results = []
    config = None
    interrupted = False
    try:
        for seed in seeds:
            result, config = run_episode(factory, seed, max_steps=args.max_steps, headless=True)
            if not results:
                print_observation_config(config)
                print("seed  collision  closest_m  average_m/s  steps  end")
            results.append(result)
            # Save each finished episode immediately; retain progress on Ctrl+C.
            write_json(directory / f"seed_{seed:02d}.json", asdict(result))
            distance = "n/a" if result.closest_distance_m is None else f"{result.closest_distance_m:.3f}"
            print(
                f"{seed:4d}  {'yes' if result.collision else 'no':9s}  {distance:>9s}  "
                f"{result.average_speed_mps:11.3f}  {result.steps:5d}  {result.end_reason}",
                flush=True,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; saving completed episodes only.")
    finally:
        summary = summarize(results) if results else None
        write_episode_csv(directory / "episodes.csv", results)
        write_json(directory / "summary.json", {
            "controller": args.controller,
            "requested_seeds": list(seeds),
            "completed_seeds": [r.seed for r in results],
            "max_steps": args.max_steps,
            "smoke_test": args.smoke_test,
            "interrupted": interrupted,
            "complete": len(results) == len(seeds),
            "env_config": config,
            "metric_definitions": METRIC_DEFINITIONS,
            "summary": summary,
        })
        if summary:
            print("\nSummary (speed mean gives each episode equal weight)")
            print(f"{'metric':38s} value")
            for name, value in summary.items():
                displayed = f"{value:.3f}" if isinstance(value, float) else str(value)
                print(f"{name:38s} {displayed}")
        print(f"Saved per-episode JSON, episodes.csv and summary.json in {directory}")


if __name__ == "__main__":
    main()
