"""Build the supervised dataset from counterfactual outcomes (v0.3).

Reads counterfactuals.json, builds archetype-based split assignments, then
builds the weighted training dataset (a* = argmax_a U(task, a)) and writes:
- split_manifest.json (with SHA-256)
- dataset_manifest.json (with SHA-256)
- dataset.json (the training examples)

Section 15: not_executed actions are excluded from argmax (utility=-1000).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capsule_brain.autolearn.dataset import (
    build_dataset,
    save_dataset_manifest,
    save_split_manifest,
    split_examples,
)
from capsule_brain.autolearn.learner import LearnerConfig
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.utility import UtilityConfig

from tasks import build_all_tasks, build_split_assignments


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _load_counterfactuals(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["rows"]


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows_data = _load_counterfactuals(out_dir / "counterfactuals.json")
    print(f"loaded {len(rows_data)} counterfactual rows")

    tasks = build_all_tasks()
    assignments = build_split_assignments(tasks)

    # Build split manifest from archetype assignments.
    from capsule_brain.autolearn.dataset import SplitManifest

    assign_str = json.dumps(assignments, sort_keys=True)
    split_digest = _compute_sha256(assign_str)

    split = SplitManifest(
        digest=split_digest,
        assignments=assignments,
        family_to_split={},
    )

    print(f"split digest: {split.digest}")
    from collections import Counter
    split_counts = Counter(assignments.values())
    print(f"split counts: {dict(split_counts)}")

    # Build training examples: for each task, pick a* = argmax_a U(task, a).
    # Section 15: exclude not_executed actions from argmax.
    from capsule_brain.autolearn.features import FeatureExtractor
    from capsule_brain.autolearn.learner import TrainingExample

    extractor = FeatureExtractor()
    utility_config = UtilityConfig()
    learner_config = LearnerConfig(
        actions=tuple(Action.learned()),
        n_epochs=800,
        learning_rate=0.01,
        l2=0.1,
        confidence_threshold=0.4,
    )

    # Group rows by task.
    by_task: dict[str, list[dict]] = {}
    for r in rows_data:
        by_task.setdefault(r["task_id"], []).append(r)

    examples: list[TrainingExample] = []
    task_by_id = {t.task_id: t for t in tasks}

    for task_id, task_rows in by_task.items():
        # Filter out not_executed actions for argmax.
        eligible = [r for r in task_rows if not r.get("not_executed", False)]
        if not eligible:
            continue

        # Find best action by utility.
        best = max(eligible, key=lambda r: r["utility"])
        best_action = Action(best["action"])
        best_utility = best["utility"]

        # Compute utility margin against second-best.
        sorted_rows = sorted(eligible, key=lambda r: r["utility"], reverse=True)
        if len(sorted_rows) > 1:
            margin = sorted_rows[0]["utility"] - sorted_rows[1]["utility"]
        else:
            margin = sorted_rows[0]["utility"]

        # Weight: 1 + min(weight_cap, max(0, margin))
        weight = 1.0 + min(5.0, max(0.0, margin))

        task = task_by_id[task_id]
        fv = extractor.extract(task.state)
        split_name = assignments.get(task_id, "train")

        examples.append(TrainingExample(
            task_id=task_id,
            task_family=task.family,
            split=split_name,
            best_action=best_action,
            utility_margin=margin,
            weight=weight,
            features=fv.as_list(),
        ))

    print(f"built {len(examples)} examples")

    # Build dataset manifest.
    family_counts: dict[str, int] = {}
    ds_split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0, "ood": 0}
    action_counts: dict[str, int] = {}
    best_action_counts: dict[str, int] = {}

    for e in examples:
        family_counts[e.task_family] = family_counts.get(e.task_family, 0) + 1
        ds_split_counts[e.split] = ds_split_counts.get(e.split, 0) + 1
        best_action_counts[e.best_action.value] = best_action_counts.get(e.best_action.value, 0) + 1

    from capsule_brain.autolearn.dataset import DatasetManifest

    dataset_text = json.dumps({
        "examples": [
            {
                "task_id": e.task_id,
                "task_family": e.task_family,
                "split": e.split,
                "best_action": e.best_action.value,
                "utility_margin": e.utility_margin,
                "weight": e.weight,
                "features": e.features,
            }
            for e in examples
        ],
    }, sort_keys=True, indent=2)
    (out_dir / "dataset.json").write_text(dataset_text)
    dataset_digest = _compute_sha256(dataset_text)
    print(f"dataset SHA-256: {dataset_digest}")

    dataset_manifest = DatasetManifest(
        dataset_version="dataset_v3",
        n_tasks=len(by_task),
        n_rows=len(examples),
        family_counts=family_counts,
        split_counts=ds_split_counts,
        action_counts=action_counts,
        best_action_counts=best_action_counts,
        utility_config=utility_config.to_dict(),
        feature_schema_version="exec_features_v2",
        dataset_digest=dataset_digest,
        split_manifest_digest=split_digest,
    )

    save_split_manifest(split, out_dir / "split_manifest.json")
    save_dataset_manifest(dataset_manifest, out_dir / "dataset_manifest.json")

    splits = split_examples(examples)
    print(f"train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])} ood={len(splits['ood'])}")
    print(f"best action counts: {best_action_counts}")


if __name__ == "__main__":
    main()
