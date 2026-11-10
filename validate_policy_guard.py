"""Check a guard on five separate seeds, then two known crash regressions.

This is a validation/regression run, not the fixed 50-seed benchmark. No training
or checkpoint selection occurs here. Every episode has the full 40-second budget.
"""

import argparse
import copy
import json
from pathlib import Path

from driving_common import new_run_directory, write_json, write_episode_csv
from evaluate import summarize
from learning import load_policy
from project_runner import run_recorded

VALIDATION_SEEDS = (2005, 2006, 2007, 2008, 2009)
REGRESSION_SEEDS = (10, 46)


def validate(model):
    with (model.parent / "training.json").open(encoding="utf-8") as stream:
        provenance = json.load(stream)
    reserved = set(provenance["training_seeds"] + provenance.get("validation_seeds", []))
    if reserved & set(VALIDATION_SEEDS):
        raise ValueError("Guard validation seeds overlap model training/selection seeds")
    config = copy.deepcopy(provenance["training_config"])
    config.update({"manual_control": False, "hazard_schedule": []})
    horizon = int(config["duration"] * config["policy_frequency"])
    policy = load_policy(model, guard=True)
    directory = new_run_directory("guard_validation")
    summaries = {}
    for name, seeds in (("validation", VALIDATION_SEEDS), ("regressions", REGRESSION_SEEDS)):
        child = directory / name
        child.mkdir()
        results = []
        interventions = 0
        for seed in seeds:
            result, extra = run_recorded("2", seed, max_steps=horizon, controller=policy,
                                         controller_name="learned_guarded", config=config,
                                         save_directory=child)
            results.append(result)
            interventions += extra["guard_interventions"]
            print(f"{name} seed={seed} collision={result.collision} speed={result.average_speed_mps:.3f} "
                  f"guard_interventions={extra['guard_interventions']}", flush=True)
        write_episode_csv(child / "episodes.csv", results)
        summaries[name] = {**summarize(results), "seeds": list(seeds),
                           "guard_interventions": interventions}
    write_json(directory / "summary.json", summaries)
    print(f"Saved validation/regressions: {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    validate(parser.parse_args().model)
