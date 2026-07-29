"""Build the supervised dataset from counterfactual outcomes.

Reads counterfactuals.json (produced by run_counterfactuals.py), builds a
deterministic split manifest by task family, then builds the weighted
training dataset (a* = argmax_a U(task, a)) and writes:
- split_manifest.json
- dataset_manifest.json
- dataset.json (the training examples)
"""
from __future__ import annotations

import json
from pathlib import Path

from capsule_brain.autolearn.dataset import (
    build_dataset,
    build_split_manifest,
    save_dataset_manifest,
    save_split_manifest,
    split_examples,
)
from capsule_brain.autolearn.learner import LearnerConfig
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.utility import UtilityConfig

from run_counterfactuals import CounterfactualRow  # noqa: E402
from tasks import build_all_tasks


def _load_counterfactuals(path: Path) -> list[CounterfactualRow]:
    from capsule_brain.autolearn.schema import (
        ActionMetadata,
        ExecutiveExperience,
        ExecutiveState,
        Outcome,
        Provenance,
    )

    data = json.loads(path.read_text())
    rows: list[CounterfactualRow] = []
    # Reconstruct ExecutiveExperience objects from the saved outcome.
    tasks_by_id = {t.task_id: t for t in build_all_tasks()}
    for r in data["rows"]:
        task = tasks_by_id[r["task_id"]]
        oc = r["outcome"]
        outcome = Outcome(
            verified_success=bool(oc.get("verified_success", False)),
            verification_status=str(oc.get("verification_status", "skip")),
            latency_ms=float(oc.get("latency_ms", 0.0) or 0.0),
            token_count=int(oc.get("token_count", 0) or 0),
            tool_failures=int(oc.get("tool_failures", 0) or 0),
            workflow_iterations=int(oc.get("workflow_iterations", 0) or 0),
            operator_intervention=bool(oc.get("operator_intervention", False)),
            safety_violation=bool(oc.get("safety_violation", False)),
        )
        exp = ExecutiveExperience(
            task_id=r["task_id"],
            state=task.state,
            chosen_action=Action(r["action"]),
            action_metadata=ActionMetadata(),
            outcome=outcome,
            utility=float(r["utility"]),
            provenance=Provenance(
                source="counterfactual",
                task_family=r["task_family"],
                task_id=r["task_id"],
                extra={"expected_action": r.get("expected_action", "")},
            ),
        )
        rows.append(CounterfactualRow(
            task_id=r["task_id"],
            task_family=r["task_family"],
            action=Action(r["action"]),
            experience=exp,
            utility=float(r["utility"]),
        ))
    return rows


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = _load_counterfactuals(out_dir / "counterfactuals.json")
    print(f"loaded {len(rows)} counterfactual rows")

    # Build deterministic split. All families are included in training so
    # the model learns all 6 actions (including ASK_OPERATOR for safety
    # tasks and REFLECT for reflection tasks). Tasks are hash-split by
    # task_id into train (60%) / validation (15%) / test (25%). The OOD
    # evaluation uses the test split's hardest tasks (highest difficulty)
    # as a proxy for distribution shift — these are held-out tasks the
    # model has never seen, with the most unusual feature profiles.
    split = build_split_manifest(
        rows,
        train_families=[],
        validation_families=[],
        test_families=[],
        ood_families=[],
        train_fraction=0.6,
        validation_fraction=0.15,
        test_fraction=0.25,
        seed=0,
    )
    print(f"split digest: {split.digest}")
    print(f"family_to_split: {split.family_to_split}")

    utility_config = UtilityConfig()
    learner_config = LearnerConfig(
        actions=tuple(Action.all()),
        n_epochs=800,
        learning_rate=0.01,
        l2=0.1,
        confidence_threshold=0.4,
    )
    built = build_dataset(
        rows, split,
        utility_config=utility_config,
        learner_config=learner_config,
    )
    print(f"built {len(built.examples)} examples")
    print(f"split counts: {built.dataset_manifest.split_counts}")
    print(f"best action counts: {built.dataset_manifest.best_action_counts}")

    save_split_manifest(built.split_manifest, out_dir / "split_manifest.json")
    save_dataset_manifest(built.dataset_manifest, out_dir / "dataset_manifest.json")
    (out_dir / "dataset.json").write_text(json.dumps({
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
            for e in built.examples
        ],
    }, sort_keys=True, indent=2))

    splits = split_examples(built.examples)
    print(f"train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])} ood={len(splits['ood'])}")


if __name__ == "__main__":
    main()
