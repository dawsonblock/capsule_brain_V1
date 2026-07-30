"""Build the real counterfactual dataset from measured real outcomes (Section 7).

v0.4.0 fixes:
  * Only EXECUTED outcomes with non-None utility are eligible.
  * Counterfactual completeness: a task is only included if ALL its allowed
    actions have EXECUTED outcomes (Section 4.3).
  * No -1000 placeholder is ever used. Missing outcomes are excluded, not
    scored.
  * Experience quality weights are applied per Section 5.3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.learner import TrainingExample, compute_dataset_digest
from capsule_brain.autolearn.schema import Action, ExecutiveState
from capsule_brain.autolearn.utility import compute_experience_quality

from .config import QualificationConfig
from .run_counterfactuals import _state_from_task, load_typed_outcomes
from .schemas import (
    CounterfactualOutcome,
    OutcomeAvailability,
    VerificationOutcome,
    is_counterfactually_complete_dict,
    read_json,
    sha256_json,
    write_json,
)


def _best_and_second(utilities: list[tuple[Action, float]]) -> tuple[Action, float, float]:
    if not utilities:
        raise ValueError("no actions for task")
    sorted_u = sorted(utilities, key=lambda x: x[1], reverse=True)
    best, u_best = sorted_u[0]
    u_second = sorted_u[1][1] if len(sorted_u) > 1 else u_best
    return best, u_best, u_second


def _example_to_dict(ex: TrainingExample) -> dict[str, Any]:
    return {
        "features": ex.features,
        "best_action": ex.best_action.value,
        "utility_margin": ex.utility_margin,
        "weight": ex.weight,
        "task_id": ex.task_id,
        "task_family": ex.task_family,
        "split": ex.split,
    }


def build_dataset(config: QualificationConfig) -> dict[str, Any]:
    """Build the counterfactual dataset from typed outcomes.

    Only counterfactually-complete tasks (all allowed actions EXECUTED) are
    included. Incomplete tasks are recorded but excluded from the dataset.
    """
    artifacts = Path(config.artifacts_dir)
    outcomes = load_typed_outcomes(config)
    if not outcomes:
        raise RuntimeError("no counterfactual outcomes found")

    bm = read_json(artifacts / "benchmark_manifest.json")
    task_by_id = {t["task_id"]: t for t in bm["tasks"]}

    # Group outcomes by task.
    outcomes_by_task: dict[str, list[CounterfactualOutcome]] = {}
    for o in outcomes:
        outcomes_by_task.setdefault(o.task_id, []).append(o)

    # Build the (task, action) -> outcome map for completeness checks.
    outcomes_by_task_action: dict[tuple[str, str], CounterfactualOutcome] = {}
    for o in outcomes:
        outcomes_by_task_action[(o.task_id, o.action_id)] = o

    # Only consider tasks that actually have outcomes (not all tasks in the
    # benchmark manifest -- in smoke mode only a subset was executed).
    tasks_with_outcomes = set(outcomes_by_task.keys())

    # Filter to counterfactually-complete tasks only.
    complete_task_ids: list[str] = []
    incomplete_task_ids: list[str] = []
    for task_id in tasks_with_outcomes:
        task = task_by_id.get(task_id)
        if task is None:
            continue
        if is_counterfactually_complete_dict(task, outcomes_by_task_action):
            complete_task_ids.append(task_id)
        else:
            incomplete_task_ids.append(task_id)

    examples: list[TrainingExample] = []
    family_stats: dict[str, Any] = {}
    q_scores: list[float] = []

    for task_id in complete_task_ids:
        task = task_by_id[task_id]
        task_outcomes = outcomes_by_task[task_id]
        # Only EXECUTED outcomes contribute utilities.
        utilities: list[tuple[Action, float]] = []
        for o in task_outcomes:
            if o.availability is OutcomeAvailability.EXECUTED and o.utility is not None:
                utilities.append((Action(o.action_id), float(o.utility)))
        if not utilities:
            continue
        best, u_best, u_second = _best_and_second(utilities)
        margin = u_best - u_second
        margin_weight = 1.0 + min(4.0, max(0.0, margin))

        state = _state_from_task(task)
        fv = FeatureExtractor().extract(state)
        split = task.get("split", "train")

        # Evidence-quality weight (Section 5.3).
        q = compute_experience_quality(
            runtime_type="real",
            verifier_type=task_outcomes[0].execution_metadata.get("verification_evidence", {}).get("verifier_type", "deterministic"),
            action_set_complete=len(task_outcomes) == len(task["allowed_actions"]),
            isolation_enforced=True,
        )
        q_scores.append(q.quality_score)
        weight = q.quality_score * margin_weight

        ex = TrainingExample(
            features=fv.as_list(),
            best_action=best,
            utility_margin=margin,
            weight=weight,
            task_id=task_id,
            task_family=task.get("family", "unknown"),
            split=split,
        )
        examples.append(ex)

        family = task.get("family", "unknown")
        family_stats.setdefault(family, {"n": 0, "best_action": {}})
        family_stats[family]["n"] += 1
        family_stats[family]["best_action"][best.value] = family_stats[family]["best_action"].get(best.value, 0) + 1

    train = [ex for ex in examples if ex.split == "experience"]
    val = [ex for ex in examples if ex.split == "validation"]
    test = [ex for ex in examples if ex.split == "test"]
    ood = [ex for ex in examples if ex.split == "ood"]
    safety = [ex for ex in examples if ex.split == "safety"]

    dataset_digest = compute_dataset_digest(examples)

    manifest = {
        "schema_version": "dataset-manifest/2",
        "protocol_version": "0.4.0",
        "dataset_digest": dataset_digest,
        "total_examples": len(examples),
        "train_count": len(train),
        "validation_count": len(val),
        "test_count": len(test),
        "ood_count": len(ood),
        "safety_count": len(safety),
        "complete_task_count": len(complete_task_ids),
        "incomplete_task_count": len(incomplete_task_ids),
        "incomplete_task_ids": incomplete_task_ids,
        "family_stats": family_stats,
        "mean_q_score": sum(q_scores) / len(q_scores) if q_scores else 0.0,
        "splits": {
            "train": [_example_to_dict(ex) for ex in train],
            "validation": [_example_to_dict(ex) for ex in val],
            "test": [_example_to_dict(ex) for ex in test],
            "ood": [_example_to_dict(ex) for ex in ood],
            "safety": [_example_to_dict(ex) for ex in safety],
        },
        "actions": [a.value for a in Action.learned()],
    }

    for split_name, split_ex in [
        ("train", train), ("validation", val), ("test", test),
        ("ood", ood), ("safety", safety),
    ]:
        path = artifacts / f"dataset_{split_name}.json"
        write_json(path, [_example_to_dict(ex) for ex in split_ex])

    manifest_path = artifacts / "dataset_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the v0.4.0 counterfactual dataset.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "qualification"], default="smoke")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    args = parser.parse_args()
    config = QualificationConfig(mode=args.mode, runtime=args.runtime, artifacts_dir=args.artifacts_dir)
    manifest = build_dataset(config)
    print(f"dataset: {manifest['total_examples']} examples")
    print(f"  complete_tasks={manifest['complete_task_count']} incomplete_tasks={manifest['incomplete_task_count']}")
    print(f"  train={manifest['train_count']} val={manifest['validation_count']} "
          f"test={manifest['test_count']} ood={manifest['ood_count']} safety={manifest['safety_count']}")
    print(f"  digest={manifest['dataset_digest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
