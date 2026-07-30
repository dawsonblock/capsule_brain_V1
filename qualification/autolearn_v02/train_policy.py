"""Train the supervised router from the counterfactual dataset (v0.2).

Reads dataset.json + split_manifest.json, trains a multinomial logistic
regression with weighted cross-entropy on the train split, validates on a
slice of train (since v0.2 has no explicit validation split — archetype
splits produce train/test/ood only), and registers the resulting policy as
an immutable candidate artifact.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from capsule_brain.autolearn.dataset import load_split_manifest, split_examples
from capsule_brain.autolearn.learner import (
    LearnerConfig,
    SupervisedRouterLearner,
    TrainingExample,
    compute_dataset_digest,
)
from capsule_brain.autolearn.policy import LEARNED_POLICY_VERSION_DEFAULT
from capsule_brain.autolearn.registry import PolicyRegistry
from capsule_brain.autolearn.schema import Action


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _load_dataset(path: Path) -> list[TrainingExample]:
    data = json.loads(path.read_text())
    examples: list[TrainingExample] = []
    for e in data["examples"]:
        examples.append(TrainingExample(
            features=list(e["features"]),
            best_action=Action(e["best_action"]),
            utility_margin=float(e["utility_margin"]),
            weight=float(e["weight"]),
            task_id=e["task_id"],
            task_family=e["task_family"],
            split=e["split"],
        ))
    return examples


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    examples = _load_dataset(out_dir / "dataset.json")
    split_manifest = load_split_manifest(out_dir / "split_manifest.json")
    splits = split_examples(examples)
    train = splits["train"]
    # v0.2 has no explicit validation split; use a slice of train.
    validation = train[: max(1, len(train) // 10)]
    if not validation:
        validation = train[:1]

    print(f"train={len(train)} validation={len(validation)}")

    learner_config = LearnerConfig(
        actions=tuple(Action.all()),
        n_epochs=800,
        learning_rate=0.01,
        l2=0.1,
        confidence_threshold=0.4,
    )
    learner = SupervisedRouterLearner(learner_config)

    dataset_digest = compute_dataset_digest(examples)
    policy_id = f"candidate_v2_{int(time.time())}"
    policy, metrics = learner.train(
        train, validation,
        policy_id=policy_id,
        policy_version=LEARNED_POLICY_VERSION_DEFAULT,
        training_data_digest=dataset_digest,
        split_manifest_digest=split_manifest.digest,
        parent_policy="baseline_v2",
    )
    print("training metrics:")
    print(json.dumps(metrics.to_dict(), indent=2))

    registry = PolicyRegistry(out_dir / "policies")
    manifest = registry.register(
        policy,
        training_data_digest=dataset_digest,
        split_manifest_digest=split_manifest.digest,
        hyperparameters=policy.hyperparameters,
        metrics=metrics.to_dict(),
        parent_policy="baseline_v2",
    )
    print(f"registered candidate policy: {policy.policy_id}")
    print(f"artifact: {manifest.artifact_path}")

    (out_dir / "candidate_policy_id.txt").write_text(policy.policy_id)
    manifest_text = json.dumps(manifest.to_dict(), sort_keys=True, indent=2)
    (out_dir / "policy_manifest.json").write_text(manifest_text)
    print(f"policy manifest SHA-256: {_compute_sha256(manifest_text)}")


if __name__ == "__main__":
    main()
