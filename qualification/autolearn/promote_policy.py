"""Apply the promotion gate to the candidate policy.

Reads evaluation.json, runs the promotion gate, and either promotes the
candidate (flips the active pointer in the PolicyRegistry) or rejects it.
The artifact is never modified — promotion flips a pointer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from capsule_brain.autolearn.evaluator import PairedEvaluation
from capsule_brain.autolearn.promotion import GateConfig, evaluate_promotion_gate
from capsule_brain.autolearn.registry import PolicyRegistry


def _paired_from_dict(data: dict) -> PairedEvaluation:
    # Reconstruct just enough of PairedEvaluation for the gate. The gate
    # only reads: baseline_metrics, candidate_metrics, delta_ci_low,
    # delta_ci_high, mean_delta, task_evaluations (family + delta).
    from capsule_brain.autolearn.evaluator import ActionMetrics, TaskEvaluation
    from capsule_brain.autolearn.schema import Action

    bm = ActionMetrics(
        verified_success_rate=data["baseline_metrics"]["verified_success_rate"],
        mean_utility=data["baseline_metrics"]["mean_utility"],
        safety_violations=data["baseline_metrics"]["safety_violations"],
        tool_precision=data["baseline_metrics"]["tool_precision"],
        tool_recall=data["baseline_metrics"]["tool_recall"],
    )
    cm = ActionMetrics(
        verified_success_rate=data["candidate_metrics"]["verified_success_rate"],
        mean_utility=data["candidate_metrics"]["mean_utility"],
        safety_violations=data["candidate_metrics"]["safety_violations"],
        tool_precision=data["candidate_metrics"]["tool_precision"],
        tool_recall=data["candidate_metrics"]["tool_recall"],
    )
    te = [
        TaskEvaluation(
            task_id=t["task_id"],
            task_family=t["task_family"],
            baseline_action=Action(t["baseline_action"]),
            candidate_action=Action(t["candidate_action"]),
            baseline_utility=t["baseline_utility"],
            candidate_utility=t["candidate_utility"],
            delta=t["delta"],
            disagreement=t["disagreement"],
            baseline_outcome={},
            candidate_outcome={},
        )
        for t in data["task_evaluations"]
    ]
    return PairedEvaluation(
        task_evaluations=te,
        baseline_metrics=bm,
        candidate_metrics=cm,
        mean_delta=data["mean_delta"],
        median_delta=data["median_delta"],
        delta_ci_low=data["delta_ci_low"],
        delta_ci_high=data["delta_ci_high"],
        wins=data["wins"],
        ties=data["ties"],
        losses=data["losses"],
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    candidate_id = (out_dir / "candidate_policy_id.txt").read_text().strip()
    eval_data = json.loads((out_dir / "evaluation.json").read_text())
    test_eval = _paired_from_dict(eval_data["test"])
    ood_eval = _paired_from_dict(eval_data["ood"]) if eval_data.get("ood") else None

    gate_config = GateConfig(
        tool_precision_min=0.5,
        tool_recall_min=0.5,
        ood_score_floor=-1.0,  # OOD must not catastrophically regress
        family_regression_max_delta=-5.0,
    )
    result = evaluate_promotion_gate(test_eval, config=gate_config, ood_eval=ood_eval)
    print(f"gate passed: {result.passed}")
    print(f"reason: {result.reason}")
    for g in result.gates:
        print(f"  {g['name']}: passed={g['passed']}")

    registry = PolicyRegistry(out_dir / "policies")
    if result.passed:
        registry.set_active(candidate_id, reason=result.reason)
        print(f"PROMOTED candidate {candidate_id} to active")
    else:
        registry.reject(candidate_id, reason=result.reason)
        print(f"REJECTED candidate {candidate_id}")

    (out_dir / "promotion_result.json").write_text(json.dumps({
        "candidate_policy_id": candidate_id,
        "gate_result": result.to_dict(),
        "promoted": result.passed,
        "timestamp": time.time(),
    }, sort_keys=True, indent=2))
    print(f"wrote {out_dir / 'promotion_result.json'}")


if __name__ == "__main__":
    main()
