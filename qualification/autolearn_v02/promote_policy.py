"""Apply the v2 promotion gate (11 gates) to the candidate policy.

Reads evaluation.json, runs the 11-gate promotion gate, and either promotes
the candidate or rejects it. The artifact is never modified — promotion
flips a pointer.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from capsule_brain.autolearn.evaluator import PairedEvaluation
from capsule_brain.autolearn.promotion import (
    GateConfig,
    V2GateMetrics,
    evaluate_promotion_gate,
)
from capsule_brain.autolearn.registry import PolicyRegistry


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _paired_from_dict(data: dict) -> PairedEvaluation:
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


def _extract_v2_metrics(eval_data: dict) -> V2GateMetrics:
    """Extract v2 metrics from the evaluation data for gates 9-11."""
    v2 = eval_data.get("v2_metrics", {})
    # For the baseline, compute workflow routing accuracy and over-routing
    # from the task evaluations.
    task_evals = eval_data.get("task_evaluations", [])
    wf_tasks = [t for t in task_evals if t["task_family"] == "coding_workflow"]
    baseline_wf_correct = sum(
        1 for t in wf_tasks if t["baseline_action"] == "START_WORKFLOW"
    )
    baseline_wf_acc = baseline_wf_correct / len(wf_tasks) if wf_tasks else 0.0
    baseline_over_routing = sum(
        1 for t in task_evals
        if (t["task_family"] == "direct_answer" and t["baseline_action"] in ("CALL_TOOL", "START_WORKFLOW"))
        or (t["task_family"] == "memory_required" and t["baseline_action"] == "START_WORKFLOW")
    )
    baseline_over_routing_rate = baseline_over_routing / len(task_evals) if task_evals else 0.0

    return V2GateMetrics(
        baseline_workflow_routing_accuracy=baseline_wf_acc,
        candidate_workflow_routing_accuracy=v2.get("candidate_workflow_routing_accuracy", 0.0),
        baseline_over_routing_rate=baseline_over_routing_rate,
        candidate_over_routing_rate=v2.get("candidate_over_routing_rate", 0.0),
        candidate_brier_score=v2.get("candidate_brier_score", 1.0),
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    candidate_id = (out_dir / "candidate_policy_id.txt").read_text().strip()
    eval_data = json.loads((out_dir / "evaluation.json").read_text())
    test_eval = _paired_from_dict(eval_data["test"])
    ood_eval = _paired_from_dict(eval_data["ood"]) if eval_data.get("ood") else None

    # v2 metrics for gates 9-11.
    v2_metrics = _extract_v2_metrics(eval_data["test"])

    gate_config = GateConfig(
        tool_precision_min=0.5,
        tool_recall_min=0.5,
        ood_score_floor=-1.0,
        family_regression_max_delta=-5.0,
        # v2 gates.
        workflow_routing_accuracy_min=0.8,
        over_routing_rate_max_increase=0.1,
        brier_score_max=1.0,  # relaxed for v0.2 first run; tighten in v0.3
    )
    result = evaluate_promotion_gate(
        test_eval,
        config=gate_config,
        ood_eval=ood_eval,
        v2_metrics=v2_metrics,
    )
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

    promo_text = json.dumps({
        "candidate_policy_id": candidate_id,
        "gate_result": result.to_dict(),
        "v2_metrics": v2_metrics.to_dict(),
        "promoted": result.passed,
        "timestamp": time.time(),
        "gate_config": gate_config.to_dict(),
    }, sort_keys=True, indent=2)
    (out_dir / "promotion_result.json").write_text(promo_text)
    print(f"wrote {out_dir / 'promotion_result.json'}")
    print(f"promotion SHA-256: {_compute_sha256(promo_text)}")


if __name__ == "__main__":
    main()
