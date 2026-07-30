"""Train a sham-learning control policy (v2.16.0 Priority 4).

The sham learner uses identical optimizer, architecture, epochs, feature
extractor, and capacity as the real learner — but destroys the experience
signal by shuffling action labels within task-family strata.

Qualification requires: AutoLearn > Baseline AND AutoLearn > Sham.

Writes:
- sham_policy.json (the sham policy artifact)
- sham_policy_manifest.json (training metrics + sham metadata)
- sham_policy_id.txt (the sham policy ID for downstream evaluation)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.learner import LearnerConfig, SupervisedRouterLearner, TrainingExample
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.registry import PolicyRegistry
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.sham_learner import train_sham_policy

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

    # Use the SAME config as the real policy.
    learner_config = LearnerConfig(
        actions=tuple(Action.learned()),
        n_epochs=800,
        learning_rate=0.01,
        l2=0.1,
        confidence_threshold=0.4,
    )

    # Train the sham policy with shuffled action labels.
    sham_policy, sham_metrics = train_sham_policy(
        examples,
        learner_config,
        sham_method="shuffled_actions",
        seed=42,
        policy_id=f"sham_v03_{_compute_sha256(dataset_digest)[:8]}",
        training_data_digest=dataset_digest,
        split_manifest_digest=split_digest,
        parent_policy="baseline_v2",
    )

    # Register the sham policy.
    registry = PolicyRegistry(out_dir / "policies")
    try:
        manifest = registry.register(
            sham_policy,
            training_data_digest=dataset_digest,
            split_manifest_digest=split_digest,
            hyperparameters=sham_policy.hyperparameters,
            metrics=sham_metrics.to_dict() if hasattr(sham_metrics, "to_dict") else {},
            parent_policy="baseline_v2",
        )
    except FileExistsError:
        # Already registered — load existing.
        manifest = registry.load_manifest(sham_policy.policy_id)

    # Write sham policy manifest.
    sham_manifest = {
        "policy_id": sham_policy.policy_id,
        "policy_version": sham_policy.policy_version,
        "policy_type": sham_policy.policy_type,
        "sham_method": "shuffled_actions",
        "sham_seed": 42,
        "feature_schema_version": sham_policy.feature_schema_version,
        "parent_policy": sham_policy.parent_policy,
        "training_data_digest": sham_policy.training_data_digest,
        "split_manifest_digest": sham_policy.split_manifest_digest,
        "hyperparameters": sham_policy.hyperparameters,
        "metrics": sham_policy.metrics,
        "trained_at": sham_policy.trained_at,
        "artifact_path": str(manifest.artifact_path),
        "artifact_sha256": manifest.artifact_sha256,
    }
    text = json.dumps(sham_manifest, sort_keys=True, indent=2)
    (out_dir / "sham_policy_manifest.json").write_text(text)
    (out_dir / "sham_policy_id.txt").write_text(sham_policy.policy_id)

    print(f"trained sham policy: {sham_policy.policy_id}")
    print(f"sham method: shuffled_actions")
    print(f"wrote {out_dir / 'sham_policy_manifest.json'}")


if __name__ == "__main__":
    main()
