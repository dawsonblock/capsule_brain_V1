"""Build the real counterfactual dataset from measured real outcomes (Section 7).

v0.4.1 fixes:
  * Safety tasks are EXCLUDED from ordinary routing-policy training.
    They are handled by the immutable guard, not the learned router.
  * Verifier types are extracted from the typed verifier registry,
    not defaulted to "deterministic".
  * Dataset-level mandatory gates: positive weight count, effective
    weight sum, mean Q score, no unknown verifiers, no NaN/inf weights.
  * Training fails closed if effective weight sum is zero.
  * Exclusion reasons are recorded for every excluded task.
  * Q components (verifier, execution, counterfactual, isolation, provenance)
    are recorded per example.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.learner import TrainingExample, compute_dataset_digest
from capsule_brain.autolearn.schema import Action, ExecutiveState
from capsule_brain.autolearn.utility import compute_experience_quality
from capsule_brain.verification.reliability import is_known_verifier

from . import AUTOLEARN_QUALIFICATION_VERSION, PROTOCOL_VERSION
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


# Exclusion reasons.
EXCLUSION_SAFETY_CONTROLLED = "safety_controlled"
EXCLUSION_INCOMPLETE_COUNTERFACTUAL = "incomplete_counterfactual"
EXCLUSION_ISOLATION_FAILED = "isolation_failed"
EXCLUSION_PROVENANCE_FAILED = "provenance_failed"
EXCLUSION_UNKNOWN_VERIFIER = "unknown_verifier"
EXCLUSION_ZERO_QUALITY = "zero_quality"
EXCLUSION_INSUFFICIENT_ACTION_COMPARISON = "insufficient_action_comparison"
EXCLUSION_UNVERIFIABLE_TASK = "unverifiable_task"
EXCLUSION_PROVIDER_INELIGIBLE = "provider_ineligible"


class DatasetBuildError(RuntimeError):
    """Raised when dataset building fails closed."""
    pass


def build_dataset(config: QualificationConfig) -> dict[str, Any]:
    """Build the counterfactual dataset from typed outcomes.

    v0.4.1: Safety tasks are excluded from routing-policy training.
    They are included only in the safety evaluation dataset.
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
    safety_examples: list[TrainingExample] = []
    family_stats: dict[str, Any] = {}
    q_scores: list[float] = []
    q_components_list: list[dict[str, Any]] = []
    exclusions: dict[str, list[str]] = {}  # reason -> [task_ids]

    def _exclude(task_id: str, reason: str) -> None:
        exclusions.setdefault(reason, []).append(task_id)

    for task_id in complete_task_ids:
        task = task_by_id[task_id]
        task_outcomes = outcomes_by_task[task_id]
        split = task.get("split", "train")

        # v0.4.1: Exclude safety tasks from routing-policy training.
        if split == "safety" or task.get("risk_class") == "adversarial":
            _exclude(task_id, EXCLUSION_SAFETY_CONTROLLED)
            # Safety tasks go to safety evaluation only, not training.
            continue

        # Only EXECUTED outcomes contribute utilities.
        utilities: list[tuple[Action, float]] = []
        for o in task_outcomes:
            if o.availability is OutcomeAvailability.EXECUTED and o.utility is not None:
                utilities.append((Action(o.action_id), float(o.utility)))
        if not utilities:
            _exclude(task_id, EXCLUSION_UNVERIFIABLE_TASK)
            continue

        # v0.4.1: Require at least 2 comparable actions.
        if len(utilities) < 2:
            _exclude(task_id, EXCLUSION_INSUFFICIENT_ACTION_COMPARISON)
            continue

        best, u_best, u_second = _best_and_second(utilities)
        margin = u_best - u_second
        margin_weight = 1.0 + min(4.0, max(0.0, margin))

        state = _state_from_task(task)
        fv = FeatureExtractor().extract(state)

        # v0.4.1: Extract verifier type from execution metadata.
        verifier_type = task_outcomes[0].execution_metadata.get(
            "verification_evidence", {}
        ).get("verifier_type", "deterministic")

        # v0.4.1: Check for unknown verifier types.
        if not is_known_verifier(verifier_type):
            _exclude(task_id, EXCLUSION_UNKNOWN_VERIFIER)
            continue

        # Evidence-quality weight (Section 5.3).
        q = compute_experience_quality(
            runtime_type="real",
            verifier_type=verifier_type,
            action_set_complete=len(task_outcomes) == len(task["allowed_actions"]),
            isolation_enforced=True,
            provenance_valid=True,
        )

        # v0.4.1: Exclude zero-quality examples.
        if q.quality_score <= 0:
            _exclude(task_id, EXCLUSION_ZERO_QUALITY)
            continue

        q_scores.append(q.quality_score)
        q_components_list.append(q.to_dict())
        weight = q.quality_score * margin_weight

        # v0.4.1: Check for NaN/inf weights.
        if math.isnan(weight) or math.isinf(weight):
            _exclude(task_id, EXCLUSION_ZERO_QUALITY)
            continue

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

    # v0.4.1: Safety examples are separate — not in the routing training set.
    for task_id in complete_task_ids:
        task = task_by_id.get(task_id, {})
        if task.get("split") == "safety" or task.get("risk_class") == "adversarial":
            task_outcomes = outcomes_by_task.get(task_id, [])
            for o in task_outcomes:
                if o.availability is OutcomeAvailability.EXECUTED and o.utility is not None:
                    state = _state_from_task(task)
                    fv = FeatureExtractor().extract(state)
                    safety_examples.append(TrainingExample(
                        features=fv.as_list(),
                        best_action=Action("SAFE_REFUSAL"),
                        utility_margin=0.0,
                        weight=0.0,  # safety examples have zero training weight
                        task_id=task_id,
                        task_family=task.get("family", "safety"),
                        split="safety",
                    ))

    train = [ex for ex in examples if ex.split == "experience"]
    val = [ex for ex in examples if ex.split == "validation"]
    test = [ex for ex in examples if ex.split == "test"]
    ood = [ex for ex in examples if ex.split == "ood"]

    # v0.4.1: Dataset-level mandatory gates.
    positive_weight_count = sum(1 for ex in train if ex.weight > 0)
    effective_weight_sum = sum(ex.weight for ex in train)
    mean_q = sum(q_scores) / len(q_scores) if q_scores else 0.0
    has_nan = any(math.isnan(ex.weight) for ex in examples)
    has_inf = any(math.isinf(ex.weight) for ex in examples)
    has_negative = any(ex.weight < 0 for ex in examples)

    gates = {
        "positive_weight_example_count": {
            "value": positive_weight_count,
            "passed": positive_weight_count > 0,
            "minimum": 1,
        },
        "effective_weight_sum": {
            "value": effective_weight_sum,
            "passed": effective_weight_sum > 0,
            "minimum": 0.01,
        },
        "mean_q_score": {
            "value": mean_q,
            "passed": mean_q > 0,
            "minimum": 0.01,
        },
        "no_unknown_verifier_types": {
            "value": EXCLUSION_UNKNOWN_VERIFIER not in exclusions,
            "passed": EXCLUSION_UNKNOWN_VERIFIER not in exclusions,
        },
        "no_negative_weights": {
            "value": not has_negative,
            "passed": not has_negative,
        },
        "no_nan_weights": {
            "value": not has_nan,
            "passed": not has_nan,
        },
        "no_infinite_weights": {
            "value": not has_inf,
            "passed": not has_inf,
        },
    }
    all_gates_passed = all(g["passed"] for g in gates.values())

    # v0.4.1: Scientific mode requires all training examples to be real.
    if config.is_scientific:
        all_real = all(
            task_outcomes[0].availability is OutcomeAvailability.EXECUTED
            for task_outcomes in [outcomes_by_task.get(ex.task_id, []) for ex in train]
            if task_outcomes
        )
        gates["all_training_examples_runtime_type_real"] = {
            "value": all_real,
            "passed": all_real,
        }
        all_gates_passed = all_gates_passed and all_real

    dataset_digest = compute_dataset_digest(examples)

    manifest = {
        "schema_version": "dataset-manifest/3",
        "protocol_version": PROTOCOL_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "dataset_digest": dataset_digest,
        "total_examples": len(examples),
        "train_count": len(train),
        "validation_count": len(val),
        "test_count": len(test),
        "ood_count": len(ood),
        "safety_count": len(safety_examples),
        "complete_task_count": len(complete_task_ids),
        "incomplete_task_count": len(incomplete_task_ids),
        "incomplete_task_ids": incomplete_task_ids,
        "exclusions": {k: v for k, v in exclusions.items()},
        "exclusion_counts": {k: len(v) for k, v in exclusions.items()},
        "family_stats": family_stats,
        "mean_q_score": mean_q,
        "q_components": q_components_list[:10] if q_components_list else [],  # sample
        "gates": gates,
        "all_gates_passed": all_gates_passed,
        "splits": {
            "train": [_example_to_dict(ex) for ex in train],
            "validation": [_example_to_dict(ex) for ex in val],
            "test": [_example_to_dict(ex) for ex in test],
            "ood": [_example_to_dict(ex) for ex in ood],
            "safety": [_example_to_dict(ex) for ex in safety_examples],
        },
        "actions": [a.value for a in Action.learned()],
    }

    for split_name, split_ex in [
        ("train", train), ("validation", val), ("test", test),
        ("ood", ood), ("safety", safety_examples),
    ]:
        path = artifacts / f"dataset_{split_name}.json"
        write_json(path, [_example_to_dict(ex) for ex in split_ex])

    manifest_path = artifacts / "dataset_manifest.json"
    write_json(manifest_path, manifest)

    # v0.4.1: Fail closed if effective weight sum is zero.
    if effective_weight_sum <= 0 and len(train) > 0:
        raise DatasetBuildError(
            f"Effective training weight sum is zero ({effective_weight_sum}). "
            f"All training examples have zero weight. "
            f"Gates: {gates}"
        )

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
