"""Run shadow mode evaluation for the candidate policy (v0.3)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capsule_brain.autolearn.baseline import BaselinePolicyV2
from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.registry import PolicyRegistry
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.shadow import (
    DistributionShiftConfig,
    DistributionShiftGuard,
)

from tasks import build_all_tasks, build_split_assignments


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    candidate_id = (out_dir / "candidate_policy_id.txt").read_text().strip()
    registry = PolicyRegistry(out_dir / "policies")
    candidate = registry.load_policy(candidate_id)

    tasks = build_all_tasks()
    assignments = build_split_assignments(tasks)
    shadow_tasks = [t for t in tasks if assignments.get(t.task_id) in ("train", "test")]

    extractor = FeatureExtractor()
    baseline = BaselinePolicyV2()

    # Build distribution-shift guard with training means.
    train_tasks = [t for t in tasks if assignments.get(t.task_id) == "train"]
    train_features = [extractor.extract(t.state).as_list() for t in train_tasks]
    n_features = len(train_features[0]) if train_features else 0
    training_means = [
        sum(f[i] for f in train_features) / len(train_features)
        for i in range(n_features)
    ] if train_features else []

    shift_guard = DistributionShiftGuard(
        training_means=training_means,
        config=DistributionShiftConfig(max_mean_distance=2.0, min_window=5),
    )

    records: list[dict] = []
    disagreements = 0
    abstentions = 0
    distribution_shifts = 0

    for task in shadow_tasks:
        fv = extractor.extract(task.state)
        fv_list = fv.as_list()
        shift_guard.observe(fv_list)
        shifted = shift_guard.shifted()
        if shifted:
            distribution_shifts += 1

        try:
            c_dec = candidate.select_action(task.state, extractor=extractor)
        except Exception as exc:
            c_dec = type("D", (), {
                "action": Action.ANSWER_DIRECT,
                "confidence": 0.0,
                "abstained": True,
                "reason": f"error: {exc}",
            })()

        b_dec = baseline.select_action(task.state)

        disagreement = c_dec.action != b_dec.action
        if disagreement:
            disagreements += 1
        if getattr(c_dec, "abstained", False):
            abstentions += 1

        records.append({
            "task_id": task.task_id,
            "archetype": task.archetype,
            "split": assignments.get(task.task_id, "train"),
            "baseline_action": b_dec.action.value,
            "candidate_action": c_dec.action.value,
            "candidate_confidence": c_dec.confidence,
            "disagreement": disagreement,
            "abstained": getattr(c_dec, "abstained", False),
            "distribution_shift": shifted,
            "expected_action": task.expected_action.value,
        })

    shadow_text = json.dumps({
        "candidate_policy_id": candidate.policy_id,
        "records": len(records),
        "disagreements": disagreements,
        "abstentions": abstentions,
        "distribution_shifts": distribution_shifts,
        "details": records,
    }, sort_keys=True, indent=2)
    (out_dir / "shadow_eval.json").write_text(shadow_text)
    print(f"shadow records: {len(records)}")
    print(f"disagreements: {disagreements}")
    print(f"abstentions: {abstentions}")
    print(f"distribution_shifts: {distribution_shifts}")
    print(f"shadow SHA-256: {_compute_sha256(shadow_text)}")


if __name__ == "__main__":
    main()
