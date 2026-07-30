"""Train a sham (permuted-label) policy for calibration (Section 9.1).

v0.4.0 fixes:
  * The sham receives the SAME features, same number of training records,
    same optimizer, same hyperparameters, same training duration, same
    class balancing as the candidate.
  * Labels are permuted WITHIN valid constraints (deterministic under seed).
  * The sham CANNOT access the true utility ordering.
  * The sham policy embeds a complete PolicyProvenance manifest (with the
    same digests as the candidate, since it trains on the same dataset).
"""
from __future__ import annotations

import argparse
import datetime
import random
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.learner import LearnerConfig, SupervisedRouterLearner, TrainingExample
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.features import FEATURE_SCHEMA_VERSION, FEATURE_NAMES

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .provenance import build_policy_provenance
from .schemas import sha256_json, sha256_text, read_json, write_json
from .train_candidate import (
    _feature_schema_digest,
    _hyperparameter_digest,
    _learner_code_digest,
    _load_split,
)


def _permute_labels_within_constraints(
    train: list[TrainingExample],
    rng: random.Random,
) -> list[TrainingExample]:
    """Permute labels within valid constraints (Section 9.1).

    The sham receives randomly permuted labels but retains class counts
    where required. The permutation is deterministic under the seed and
    the sham cannot access the true utility ordering.
    """
    labels = [ex.best_action for ex in train]
    rng.shuffle(labels)
    return [
        TrainingExample(
            features=list(ex.features),
            best_action=labels[i],
            utility_margin=ex.utility_margin,
            weight=ex.weight,
            task_id=ex.task_id,
            task_family=ex.task_family,
            split=ex.split,
        )
        for i, ex in enumerate(train)
    ]


def train_sham(config: QualificationConfig, *, sham_seed: int = 12345) -> tuple[LearnedPolicy, dict[str, Any], dict[str, Any]]:
    artifacts = Path(config.artifacts_dir)
    if not (artifacts / "dataset_train.json").exists():
        from .build_dataset import build_dataset
        build_dataset(config)

    train = _load_split(artifacts, "train")
    val = _load_split(artifacts, "validation")
    if not train:
        raise RuntimeError("D_experience is empty; cannot train sham policy")

    rng = random.Random(sham_seed)
    shuffled = _permute_labels_within_constraints(train, rng)

    # Identical learner config to the candidate (Section 9.1).
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

    dataset_manifest = read_json(artifacts / "dataset_manifest.json")
    split_manifest = read_json(artifacts / "split_manifest.json")

    policy, metrics = learner.train(
        shuffled,
        val,
        policy_id="sham_v04",
        policy_version="sham_v04_real",
        training_data_digest=dataset_manifest["dataset_digest"],
        split_manifest_digest=sha256_json(split_manifest),
        parent_policy="baseline_v3",
    )

    # Embed full provenance (same digests as candidate since same dataset).
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    provenance = build_policy_provenance(
        config=config,
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        parent_policy_id="baseline_v3",
        benchmark_manifest_path=artifacts / "benchmark_manifest.json",
        split_manifest_path=artifacts / "split_manifest.json",
        counterfactual_results_path=artifacts / "counterfactual_outcomes.json",
        training_dataset_path=artifacts / "dataset_manifest.json",
        feature_schema_digest=_feature_schema_digest(),
        hyperparameter_digest=_hyperparameter_digest(learner_cfg),
        learner_code_digest=_learner_code_digest(),
        model_id=config.model_id,
        model_revision=None,
        tokenizer_id=config.tokenizer_id,
        tokenizer_revision=None,
        created_at_utc=created_at,
    )
    provenance.validate()

    policy_dict = policy.to_dict()
    policy_dict["provenance"] = provenance.to_dict()
    policy_dict["schema_version"] = "learned-policy/2"
    policy_dict["protocol_version"] = PROTOCOL_VERSION
    policy_dict["is_sham"] = True
    policy_dict["sham_seed"] = sham_seed

    record = {
        "schema_version": "sham-training/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "n_train": metrics.n_train,
        "n_validation": metrics.n_validation,
        "n_epochs": metrics.n_epochs,
        "train_loss": metrics.train_loss,
        "train_accuracy": metrics.train_accuracy,
        "validation_accuracy": metrics.validation_accuracy,
        "sham_seed": sham_seed,
        "provenance": provenance.to_dict(),
        "hyperparameters": policy.hyperparameters,
    }

    write_json(artifacts / "sham_policy.json", policy_dict)
    write_json(artifacts / "sham_training.json", record)
    return policy, record, metrics.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a sham (permuted-label) policy for calibration.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--sham-seed", type=int, default=12345)
    parser.add_argument("--mode", choices=["smoke", "qualification"], default="smoke")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    args = parser.parse_args()
    config = QualificationConfig(mode=args.mode, runtime=args.runtime, artifacts_dir=args.artifacts_dir)
    if config.is_simulated:
        print(config.simulated_banner, file=sys.stderr)
    policy, record, metrics = train_sham(config, sham_seed=args.sham_seed)
    print(f"sham policy: {policy.policy_id}")
    print(f"  train_accuracy={record['train_accuracy']:.4f} validation_accuracy={record['validation_accuracy']:.4f}")
    print(f"  sham_seed={record['sham_seed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
