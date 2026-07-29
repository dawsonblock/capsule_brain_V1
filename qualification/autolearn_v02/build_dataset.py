"""Build the supervised dataset from counterfactual outcomes (v0.2).

Reads counterfactuals.json, builds archetype-based split assignments, then
builds the weighted training dataset (a* = argmax_a U(task, a)) and writes:
- split_manifest.json (with SHA-256)
- dataset_manifest.json (with SHA-256)
- dataset.json (the training examples)
"""
from __future__ import annotations

import hashlib
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
from tasks import build_all_tasks, build_split_assignments


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _load_counterfactuals(path: Path) -> list[CounterfactualRow]:
    from capsule_brain.autolearn.schema import (
        ActionMetadata,
        ExecutiveExperience,
        Outcome,
        Provenance,
    )

    data = json.loads(path.read_text())
    rows: list[CounterfactualRow] = []
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
            runtime_error=bool(oc.get("runtime_error", False)),
            action_completed=bool(oc.get("action_completed", False)),
            tool_name=oc.get("tool_name"),
            tool_calls_executed=int(oc.get("tool_calls_executed", 0) or 0),
            tool_result_valid=bool(oc.get("tool_result_valid", False)),
            final_answer_grounded=bool(oc.get("final_answer_grounded", False)),
            workflow_name=oc.get("workflow_name"),
            workflow_run_id=oc.get("workflow_run_id"),
            acceptance_present=bool(oc.get("acceptance_present", False)),
            acceptance_passed=bool(oc.get("acceptance_passed", False)),
            exit_code=oc.get("exit_code"),
            unnecessary_tool_use=bool(oc.get("unnecessary_tool_use", False)),
            unnecessary_workflow_use=bool(oc.get("unnecessary_workflow_use", False)),
        )
        exp = ExecutiveExperience(
            task_id=r["task_id"],
            state=task.state,
            chosen_action=Action(r["action"]),
            action_metadata=ActionMetadata(),
            outcome=outcome,
            utility=float(r["utility"]),
            provenance=Provenance(
                source="counterfactual_v2",
                task_family=r["task_family"],
                task_id=r["task_id"],
                extra={
                    "expected_action": r.get("expected_action", ""),
                    "archetype": r.get("archetype", ""),
                },
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

    # Build archetype-based split assignments.
    tasks = build_all_tasks()
    assignments = build_split_assignments(tasks)

    # Build a split manifest from the archetype assignments.
    # We use the existing build_split_manifest but pass the assignments
    # through the family_to_split mechanism. Since v0.2 splits by archetype
    # (not family), we build a custom split manifest.
    from capsule_brain.autolearn.dataset import SplitManifest
    import hashlib as _hl

    # Compute the split digest from the assignments.
    assign_str = json.dumps(assignments, sort_keys=True)
    split_digest = _compute_sha256(assign_str)

    split = SplitManifest(
        digest=split_digest,
        assignments=assignments,
        family_to_split={},  # v0.2 uses archetype splits, not family splits
    )

    print(f"split digest: {split.digest}")
    from collections import Counter
    split_counts = Counter(assignments.values())
    print(f"split counts: {dict(split_counts)}")

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
            for e in built.examples
        ],
    }, sort_keys=True, indent=2)
    (out_dir / "dataset.json").write_text(dataset_text)
    dataset_digest = _compute_sha256(dataset_text)
    print(f"dataset SHA-256: {dataset_digest}")

    splits = split_examples(built.examples)
    print(f"train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])} ood={len(splits['ood'])}")


if __name__ == "__main__":
    main()
