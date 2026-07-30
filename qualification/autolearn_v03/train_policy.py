"""Train the learned executive policy from the dataset (v0.3).

Section 19: Same small logistic router as v0.2, trained with weighted
cross-entropy. The policy only trains on the 4 learned actions
(ANSWER_DIRECT, RETRIEVE_MEMORY, CALL_TOOL, START_WORKFLOW). REFLECT and
ASK_OPERATOR remain runtime-controlled.

Writes:
- policy_manifest.json (with training metrics)
- policies/policies/<policy_id>.json (the policy artifact)
- policies/policies/<policy_id>.manifest.json (the registry manifest)
- candidate_policy_id.txt (the policy ID for downstream scripts)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capsule_brain.autolearn.features import FEATURE_SCHEMA_VERSION, FeatureExtractor
from capsule_brain.autolearn.learner import SupervisedRouterLearner, LearnerConfig
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.registry import PolicyRegistry
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.utility import UtilityConfig

from tasks import build_all_tasks, build_split_assignments


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    data = json.loads((out_dir / "dataset.json").read_text())
    examples_data = data["examples"]
    print(f"loaded {len(examples_data)} examples")

    # Build split manifest digest.
    tasks = build_all_tasks()
    assignments = build_split_assignments(tasks)
    split_digest = _compute_sha256(json.dumps(assignments, sort_keys=True))

    # Build dataset digest.
    dataset_text = (out_dir / "dataset.json").read_text()
    dataset_digest = _compute_sha256(dataset_text)

    # Build training examples.
    from capsule_brain.autolearn.learner import TrainingExample

    examples = []
    for e in examples_data:
        examples.append(TrainingExample(
            task_id=e["task_id"],
            task_family=e["task_family"],
            split=e["split"],
            best_action=Action(e["best_action"]),
            utility_margin=e["utility_margin"],
            weight=e["weight"],
            features=e["features"],
        ))

    # Train only on the 4 learned actions.
    learner_config = LearnerConfig(
        actions=tuple(Action.learned()),
        n_epochs=800,
        learning_rate=0.01,
        l2=0.1,
        confidence_threshold=0.4,
    )

    # Split examples into train and validation.
    train_examples = [e for e in examples if e.split == "train"]
    val_examples = [e for e in examples if e.split == "test"]
    # If no validation examples, use a small slice of train.
    if not val_examples:
        val_examples = train_examples[:max(1, len(train_examples) // 10)]

    learner = SupervisedRouterLearner(learner_config)
    policy, training_metrics = learner.train(
        train_examples,
        val_examples,
        policy_id=f"learned_v03_{_compute_sha256(dataset_digest)[:8]}",
        policy_version="learned_router_v2",
        training_data_digest=dataset_digest,
        split_manifest_digest=split_digest,
        parent_policy="baseline_v2",
    )

    # Attach provenance.
    policy.hyperparameters = {
        "n_epochs": learner_config.n_epochs,
        "learning_rate": learner_config.learning_rate,
        "l2": learner_config.l2,
        "confidence_threshold": learner_config.confidence_threshold,
        "actions": [a.value for a in learner_config.actions],
    }
    policy.metrics = training_metrics.to_dict() if hasattr(training_metrics, "to_dict") else {}

    # Register the policy.
    registry = PolicyRegistry(out_dir / "policies")
    manifest = registry.register(
        policy,
        training_data_digest=dataset_digest,
        split_manifest_digest=split_digest,
        hyperparameters=policy.hyperparameters,
        metrics=policy.metrics,
        parent_policy="baseline_v2",
    )

    # Write policy manifest (summary).
    policy_manifest = {
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "policy_type": policy.policy_type,
        "feature_schema_version": policy.feature_schema_version,
        "parent_policy": policy.parent_policy,
        "training_data_digest": policy.training_data_digest,
        "split_manifest_digest": policy.split_manifest_digest,
        "hyperparameters": policy.hyperparameters,
        "metrics": policy.metrics,
        "trained_at": policy.trained_at,
        "artifact_path": str(manifest.artifact_path),
        "artifact_sha256": manifest.artifact_sha256,
        "manifest_path": str(manifest.manifest_path),
        "manifest_sha256": manifest.manifest_sha256,
    }
    text = json.dumps(policy_manifest, sort_keys=True, indent=2)
    (out_dir / "policy_manifest.json").write_text(text)
    (out_dir / "candidate_policy_id.txt").write_text(policy.policy_id)

    print(f"trained policy: {policy.policy_id}")
    print(f"policy SHA-256: {manifest.artifact_sha256}")
    print(f"wrote {out_dir / 'policy_manifest.json'}")


if __name__ == "__main__":
    main()
