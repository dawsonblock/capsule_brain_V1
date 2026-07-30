"""Train a sham (permuted-label) policy for calibration (Section 9).

Takes the same D_experience, shuffles the best_action labels, and trains the
same SupervisedRouterLearner. This gives a calibration signal: if the
candidate outperforms the sham by a significant margin, the improvement is not
just overfitting.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.learner import LearnerConfig, SupervisedRouterLearner, TrainingExample
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.source_hash import compute_source_tree_hash

from .config import QualificationConfig
from .schemas import read_json, write_json
from .train_candidate import _load_split


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
    labels = [ex.best_action for ex in train]
    rng.shuffle(labels)
    shuffled = [TrainingExample(
        features=list(ex.features),
        best_action=labels[i],
        utility_margin=ex.utility_margin,
        weight=ex.weight,
        task_id=ex.task_id,
        task_family=ex.task_family,
        split=ex.split,
    ) for i, ex in enumerate(train)]

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
    source_hash = compute_source_tree_hash(Path(__file__).parent)

    policy, metrics = learner.train(
        shuffled,
        val,
        policy_id="sham_v032",
        policy_version="sham_v032_real",
        training_data_digest=dataset_manifest["dataset_digest"],
        split_manifest_digest="",
        parent_policy="baseline_v2",
    )

    record = {
        "package_version": "2.15.3",
        "autolearn_qualification_version": "0.3.2",
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "n_train": metrics.n_train,
        "n_validation": metrics.n_validation,
        "n_epochs": metrics.n_epochs,
        "train_loss": metrics.train_loss,
        "train_accuracy": metrics.train_accuracy,
        "validation_accuracy": metrics.validation_accuracy,
        "sham_seed": sham_seed,
        "source_hash": source_hash,
        "training_data_digest": policy.training_data_digest,
    }

    sham_path = artifacts / "sham_policy.json"
    write_json(sham_path, policy.to_dict())
    training_path = artifacts / "sham_training.json"
    write_json(training_path, record)

    return policy, record, metrics.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a sham (permuted-label) policy for calibration.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--sham-seed", type=int, default=12345)
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    args = parser.parse_args()
    config = QualificationConfig(runtime=args.runtime, artifacts_dir=args.artifacts_dir)
    if config.is_simulated:
        print(config.simulated_banner, file=sys.stderr)
    policy, record, metrics = train_sham(config, sham_seed=args.sham_seed)
    print(f"sham policy: {policy.policy_id}")
    print(f"  train_accuracy={record['train_accuracy']:.4f} "
          f"validation_accuracy={record['validation_accuracy']:.4f}")
    print(f"  wrote {config.artifacts_dir}/sham_policy.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
