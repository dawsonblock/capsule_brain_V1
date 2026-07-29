"""Train the supervised router from the counterfactual dataset.

Reads dataset.json + split_manifest.json, trains a multinomial logistic
regression with weighted cross-entropy on the train split, validates on the
validation split, and registers the resulting policy as an immutable
candidate artifact in the PolicyRegistry.
"""
from __future__ import annotations

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
    validation = splits["validation"] or splits["test"]  # reflection_req is val
    if not validation:
        # Fall back to a slice of train if no validation examples exist.
        validation = train[: max(1, len(train) // 10)]

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
    policy_id = f"candidate_{int(time.time())}"
    policy, metrics = learner.train(
        train, validation,
        policy_id=policy_id,
        policy_version=LEARNED_POLICY_VERSION_DEFAULT,
        training_data_digest=dataset_digest,
        split_manifest_digest=split_manifest.digest,
        parent_policy="baseline_v1",
    )
    print("training metrics:")
    print(json.dumps(metrics.to_dict(), indent=2))

    registry = PolicyRegistry(out_dir / "policies")
    manifest = registry.register(
        policy,
        training_data_digest=dataset_digest,
        split_manifest_digest=split_manifest.digest,
        hyperparameters=learner_config.to_dict(),
        metrics=metrics.to_dict(),
        parent_policy="baseline_v1",
    )
    print(f"registered candidate policy: {policy.policy_id}")
    print(f"artifact: {manifest.artifact_path}")

    # Persist the policy id so evaluate/shadow/promote can find it.
    (out_dir / "candidate_policy_id.txt").write_text(policy.policy_id)
    (out_dir / "policy_manifest.json").write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2)
    )


if __name__ == "__main__":
    main()
