"""Train the candidate learned policy on real D_experience (Section 9).

Uses SupervisedRouterLearner on (features, best_verified_action,
utility_margin_weight) triples. Trains on D_experience only; uses D_validation
for model selection. No test/ood/safety data leaks into training.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.learner import LearnerConfig, SupervisedRouterLearner, TrainingExample
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.source_hash import compute_source_tree_hash

from .build_dataset import build_dataset
from .config import QualificationConfig
from .schemas import read_json, write_json
from . import PACKAGE_VERSION


def _load_split(artifacts: Path, split: str) -> list[TrainingExample]:
    path = artifacts / f"dataset_{split}.json"
    data = read_json(path)
    out: list[TrainingExample] = []
    for row in data:
        out.append(TrainingExample(
            features=list(row["features"]),
            best_action=Action(row["best_action"]),
            utility_margin=float(row.get("utility_margin", 0.0)),
            weight=float(row.get("weight", 1.0)),
            task_id=str(row.get("task_id", "")),
            task_family=str(row.get("task_family", "unknown")),
            split=str(row.get("split", split)),
        ))
    return out


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


def train_candidate(config: QualificationConfig) -> tuple[LearnedPolicy, dict[str, Any], dict[str, Any]]:
    # Build the dataset first if missing.
    artifacts = Path(config.artifacts_dir)
    if not (artifacts / "dataset_manifest.json").exists():
        build_dataset(config)

    train = _load_split(artifacts, "train")
    val = _load_split(artifacts, "validation")

    if not train:
        raise RuntimeError("D_experience is empty; cannot train candidate policy")

    actions = tuple(Action.learned())
    learner_cfg = LearnerConfig(
        learning_rate=0.1,
        n_epochs=400,
        l2=1e-3,
        weight_cap=4.0,
        min_weight=1.0,
        confidence_threshold=0.5,
        temperature=1.0,
        actions=actions,
    )
    learner = SupervisedRouterLearner(learner_cfg)

    from .build_dataset import build_dataset
    # Recompute digests from the on-disk manifests.
    dataset_manifest = read_json(artifacts / "dataset_manifest.json")
    split_manifest = read_json(artifacts / "split_manifest.json")
    source_hash = compute_source_tree_hash(Path(__file__).parent)

    policy, metrics = learner.train(
        train,
        val,
        policy_id="candidate_v032",
        policy_version="candidate_v032_real",
        training_data_digest=dataset_manifest["dataset_digest"],
        split_manifest_digest=split_manifest.get("prompt_digest", ""),
        parent_policy="baseline_v2",
    )

    # Convert metrics to a serializable dict (it already has to_dict).
    training_record = {
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": "0.3.2",
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "n_train": metrics.n_train,
        "n_validation": metrics.n_validation,
        "n_epochs": metrics.n_epochs,
        "train_loss": metrics.train_loss,
        "train_accuracy": metrics.train_accuracy,
        "validation_accuracy": metrics.validation_accuracy,
        "source_hash": source_hash,
        "training_data_digest": policy.training_data_digest,
        "split_manifest_digest": policy.split_manifest_digest,
        "parent_policy": policy.parent_policy,
        "hyperparameters": policy.hyperparameters,
    }

    candidate_path = artifacts / "candidate_policy.json"
    write_json(candidate_path, policy.to_dict())
    training_path = artifacts / "candidate_training.json"
    write_json(training_path, training_record)

    return policy, training_record, metrics.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train candidate policy on real D_experience.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    args = parser.parse_args()
    config = QualificationConfig(runtime=args.runtime, artifacts_dir=args.artifacts_dir)
    if config.is_simulated:
        print(config.simulated_banner, file=sys.stderr)
    policy, record, metrics = train_candidate(config)
    print(f"candidate policy: {policy.policy_id}")
    print(f"  train_accuracy={record['train_accuracy']:.4f} "
          f"validation_accuracy={record['validation_accuracy']:.4f}")
    print(f"  wrote {config.artifacts_dir}/candidate_policy.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
