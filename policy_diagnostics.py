"""Inspect a local training/evaluation run without modifying it or loading weights."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def diagnose(training, evaluation):
    counts = Counter()
    episodes = []
    failures = []
    for path in sorted(evaluation.glob("seed_[0-9][0-9].json")):
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        counts.update(step["ego_action"] for step in document["steps"])
        episodes.append(document["episode"])
        if document["episode"]["collision"]:
            failures.append((document["seed"], [s["ego_action"] for s in document["steps"][-5:]]))
    labels = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
    print(f"Evaluation episodes={len(episodes)}, collisions={sum(e['collision'] for e in episodes)}")
    total = sum(counts.values())
    for action, label in labels.items():
        percent = 100 * counts[action] / total if total else 0
        print(f"{label:12s} {counts[action]:5d} actions ({percent:.1f}%)")
    print(f"Collision seeds: {[seed for seed, _ in failures]}")
    for seed, actions in failures:
        print(f"  seed={seed}: last actions={[labels[action] for action in actions]}")
    with (training / "training_episodes.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    recent = rows[-20:]
    print(f"Completed training episodes={len(rows)}; last {len(recent)} collisions="
          f"{sum(r['collision'] == 'yes' for r in recent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_directory", type=Path)
    parser.add_argument("evaluation_directory", type=Path)
    args = parser.parse_args()
    diagnose(args.training_directory, args.evaluation_directory)
