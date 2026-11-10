"""Audit complete local benchmark artifacts without executing or training a model."""

import argparse
import json
from pathlib import Path


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def audit(directory, repair_labels=False):
    root_summary = read_json(directory / "summary.json")
    stage5 = "summaries" in root_summary
    summaries = root_summary["summaries"] if stage5 else root_summary
    total, repaired = 0, 0
    for name, summary in summaries.items():
        documents = []
        child = directory / name
        for path in sorted(child.glob("seed_[0-9][0-9].json")):
            document = read_json(path)
            # Early development runs used 'learned' for all custom controllers.
            # Explicit opt-in repairs ONLY those names, preserving an original
            # backup. Numerical measurements and vehicle states are untouched.
            if stage5 and repair_labels and any(step["controller"] != name for step in document["steps"]):
                with path.with_suffix(".before-label-fix.json").open("x", encoding="utf-8") as backup:
                    json.dump(document, backup, indent=2)
                    backup.write("\n")
                for step in document["steps"]:
                    step["controller"] = name
                with path.open("w", encoding="utf-8") as stream:
                    json.dump(document, stream, indent=2, allow_nan=False)
                    stream.write("\n")
                repaired += 1
            if stage5:
                assert all(step["controller"] == name for step in document["steps"]), "Wrong controller label"
            if name == "learned_guarded":
                for step in document["steps"]:
                    decision = step["decision_info"]
                    assert decision["requested_action"] in range(5)
                    assert decision["executed_action"] == step["ego_action"]
                    changed = decision["requested_action"] != decision["executed_action"]
                    assert decision["guard_intervened"] == changed
                    assert bool(decision["guard_reason"]) == changed
                assert document["extra"]["guard_interventions"] == sum(
                    step["decision_info"]["guard_intervened"] for step in document["steps"])
            assert len(document["steps"]) == document["episode"]["steps"]
            assert document["env_config"]["observation"]["normalize"] is False
            assert len(document["initial_cars"]) == 51
            assert all(len(step["cars"]) == 51 for step in document["steps"])
            assert all("wall_clock_timestamp" in c and "effective_step" in c for c in document["commands"])
            assert document["episode"]["end_reason"] in ("collision", "time_limit"), "Shortened benchmark"
            documents.append(document)
        assert sorted(d["seed"] for d in documents) == list(range(50)), "Missing or duplicate evaluation seeds"
        assert summary["episodes"] == 50
        assert summary["collisions"] == sum(d["episode"]["collision"] for d in documents)
        if name == "learned_guarded":
            assert root_summary["guard_enabled"]
            assert summary["guard_interventions"] == sum(d["extra"]["guard_interventions"] for d in documents)
        if not stage5:
            assert summary["complete"]
            assert summary["hazards_triggered"] == sum(d["hazard_metrics"]["hazards_triggered"] for d in documents)
            assert summary["secondary_collisions"] == sum(d["hazard_metrics"]["secondary_collision"] for d in documents)
        print(f"Verified {name}: 50 complete episodes, collisions={summary['collisions']}")
        total += len(documents)
    if stage5:
        assert not set(root_summary["training_seeds"]) & set(root_summary["evaluation_seeds"])
        validation = set(root_summary.get("validation_seeds", []))
        assert not validation & set(root_summary["training_seeds"])
        assert not validation & set(root_summary["evaluation_seeds"])
    print(f"Verified {total} episodes. Controller-label repairs: {repaired}; originals retained when repaired.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repair-controller-labels", action="store_true",
                        help="Back up and correct controller names from early Stage 5 development runs only.")
    args = parser.parse_args()
    audit(args.directory, args.repair_controller_labels)
