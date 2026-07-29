"""Run the candidate policy in shadow mode on the held-out tasks.

Production action: baseline policy. Shadow action: candidate policy. Records
both decisions and reports disagreement statistics. The candidate action is
not executed (these are held-out tasks; the outcome lookup is only used to
score disagreements after the fact).
"""
from __future__ import annotations

import json
from pathlib import Path

from capsule_brain.autolearn.baseline import BaselinePolicy
from capsule_brain.autolearn.dataset import load_split_manifest
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.registry import PolicyRegistry
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.shadow import ShadowEvaluator

from tasks import build_all_tasks


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    candidate_id = (out_dir / "candidate_policy_id.txt").read_text().strip()
    registry = PolicyRegistry(out_dir / "policies")
    candidate = registry.load_policy(candidate_id)

    split_manifest = load_split_manifest(out_dir / "split_manifest.json")
    tasks = build_all_tasks()
    tasks_by_id = {t.task_id: t for t in tasks}

    # Run shadow over all held-out (test + ood) tasks.
    held_out = [
        tid for tid, sp in split_manifest.assignments.items()
        if sp in ("test", "ood", "validation")
    ]
    if not held_out:
        held_out = list(tasks_by_id.keys())

    shadow = ShadowEvaluator(
        baseline=BaselinePolicy(),
        candidate=candidate,
        confidence_threshold=candidate.confidence_threshold,
    )
    for tid in held_out:
        task = tasks_by_id[tid]
        shadow.evaluate(
            task.state,
            task_id=tid,
            allowed_actions=task.allowed_actions,
        )

    disagreements = shadow.disagreement_records()
    print(f"shadow records: {shadow.stats.n}")
    print(f"disagreements: {shadow.stats.disagreements}")
    print(f"abstentions: {shadow.stats.abstentions}")
    print(f"distribution_shifts: {shadow.stats.distribution_shifts}")

    (out_dir / "shadow_eval.json").write_text(json.dumps({
        "candidate_policy_id": candidate.policy_id,
        "stats": shadow.stats.to_dict(),
        "disagreements": [
            {
                "task_id": r.task_id,
                "production_action": r.production_action.value,
                "shadow_action": r.shadow_action.value,
                "shadow_confidence": r.shadow_confidence,
                "shadow_abstained": r.shadow_abstained,
                "shadow_reason": r.shadow_reason,
            }
            for r in disagreements
        ],
    }, sort_keys=True, indent=2))
    print(f"wrote {out_dir / 'shadow_eval.json'}")


if __name__ == "__main__":
    main()
