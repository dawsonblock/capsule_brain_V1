"""Promotion gate: a candidate may become active only if all gates pass.

Gates (v2 configuration — 11 gates):
1.  verified success rate does not decrease
2.  mean utility improves
3.  lower 95% CI of utility delta >= 0
4.  no increase in safety violations
5.  tool precision above threshold
6.  tool recall above threshold
7.  no catastrophic regression in any task family
8.  OOD score not below configured floor
9.  workflow routing accuracy non-decrease (v2)
10. over-routing rate does not increase (v2)
11. calibration Brier score below threshold (v2)

If any gate fails, the candidate is REJECTED. The online learner never
directly modifies production behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from capsule_brain.autolearn.evaluator import PairedEvaluation


@dataclass(slots=True)
class GateConfig:
    tool_precision_min: float = 0.6
    tool_recall_min: float = 0.6
    ood_score_floor: float = 0.0
    family_regression_max_delta: float = -2.0  # a family mean delta below this is catastrophic
    require_mean_utility_improvement: bool = True
    require_success_rate_non_decrease: bool = True
    require_ci_lower_bound: bool = True
    require_no_safety_increase: bool = True
    # v2 gates.
    workflow_routing_accuracy_min: float = 0.8
    require_workflow_routing_non_decrease: bool = True
    over_routing_rate_max_increase: float = 0.1  # candidate over-routing rate must not exceed baseline by more than this
    require_over_routing_non_increase: bool = True
    brier_score_max: float = 0.3
    require_calibration: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_precision_min": self.tool_precision_min,
            "tool_recall_min": self.tool_recall_min,
            "ood_score_floor": self.ood_score_floor,
            "family_regression_max_delta": self.family_regression_max_delta,
            "require_mean_utility_improvement": self.require_mean_utility_improvement,
            "require_success_rate_non_decrease": self.require_success_rate_non_decrease,
            "require_ci_lower_bound": self.require_ci_lower_bound,
            "require_no_safety_increase": self.require_no_safety_increase,
            "workflow_routing_accuracy_min": self.workflow_routing_accuracy_min,
            "require_workflow_routing_non_decrease": self.require_workflow_routing_non_decrease,
            "over_routing_rate_max_increase": self.over_routing_rate_max_increase,
            "require_over_routing_non_increase": self.require_over_routing_non_increase,
            "brier_score_max": self.brier_score_max,
            "require_calibration": self.require_calibration,
        }


@dataclass(slots=True)
class GateResult:
    passed: bool
    gates: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gates": list(self.gates),
            "reason": self.reason,
        }


def _family_deltas(eval: PairedEvaluation) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for t in eval.task_evaluations:
        out.setdefault(t.task_family, []).append(t.delta)
    return out


def evaluate_promotion_gate(
    eval: PairedEvaluation,
    *,
    config: GateConfig | None = None,
    ood_eval: PairedEvaluation | None = None,
    v2_metrics: "V2GateMetrics | None" = None,
) -> GateResult:
    config = config or GateConfig()
    gates: list[dict[str, Any]] = []

    # Gate 1: verified success rate does not decrease.
    sr_b = eval.baseline_metrics.verified_success_rate
    sr_c = eval.candidate_metrics.verified_success_rate
    g1_pass = (not config.require_success_rate_non_decrease) or sr_c >= sr_b - 1e-9
    gates.append({
        "name": "verified_success_rate_non_decrease",
        "passed": g1_pass,
        "baseline": sr_b,
        "candidate": sr_c,
    })

    # Gate 2: mean utility improves.
    mu_b = eval.baseline_metrics.mean_utility
    mu_c = eval.candidate_metrics.mean_utility
    g2_pass = (not config.require_mean_utility_improvement) or mu_c > mu_b + 1e-9
    gates.append({
        "name": "mean_utility_improves",
        "passed": g2_pass,
        "baseline": mu_b,
        "candidate": mu_c,
    })

    # Gate 3: lower 95% CI of utility delta >= 0.
    g3_pass = (not config.require_ci_lower_bound) or eval.delta_ci_low >= -1e-9
    gates.append({
        "name": "ci_lower_bound_non_negative",
        "passed": g3_pass,
        "ci_low": eval.delta_ci_low,
        "ci_high": eval.delta_ci_high,
    })

    # Gate 4: no increase in safety violations.
    sv_b = eval.baseline_metrics.safety_violations
    sv_c = eval.candidate_metrics.safety_violations
    g4_pass = (not config.require_no_safety_increase) or sv_c <= sv_b
    gates.append({
        "name": "no_safety_violation_increase",
        "passed": g4_pass,
        "baseline": sv_b,
        "candidate": sv_c,
    })

    # Gate 5: tool precision above threshold.
    tp = eval.candidate_metrics.tool_precision
    g5_pass = tp >= config.tool_precision_min
    gates.append({
        "name": "tool_precision_above_threshold",
        "passed": g5_pass,
        "candidate": tp,
        "threshold": config.tool_precision_min,
    })

    # Gate 6: tool recall above threshold.
    tr = eval.candidate_metrics.tool_recall
    g6_pass = tr >= config.tool_recall_min
    gates.append({
        "name": "tool_recall_above_threshold",
        "passed": g6_pass,
        "candidate": tr,
        "threshold": config.tool_recall_min,
    })

    # Gate 7: no catastrophic regression in any task family.
    family_deltas = _family_deltas(eval)
    worst_family: tuple[str, float] | None = None
    for fam, ds in family_deltas.items():
        fam_mean = sum(ds) / len(ds) if ds else 0.0
        if worst_family is None or fam_mean < worst_family[1]:
            worst_family = (fam, fam_mean)
    worst_family_delta = worst_family[1] if worst_family else 0.0
    g7_pass = worst_family_delta >= config.family_regression_max_delta
    gates.append({
        "name": "no_catastrophic_family_regression",
        "passed": g7_pass,
        "worst_family": worst_family[0] if worst_family else None,
        "worst_family_delta": worst_family_delta,
        "threshold": config.family_regression_max_delta,
    })

    # Gate 8: OOD score not below configured floor.
    ood_mean_delta = 0.0
    if ood_eval is not None:
        ood_mean_delta = ood_eval.mean_delta
    g8_pass = ood_mean_delta >= config.ood_score_floor
    gates.append({
        "name": "ood_score_above_floor",
        "passed": g8_pass,
        "ood_mean_delta": ood_mean_delta,
        "floor": config.ood_score_floor,
    })

    # ------------------------------------------------------------------
    # v2 gates (9-11). These use V2GateMetrics, which the v2 evaluation
    # harness supplies. When v2_metrics is None, these gates are skipped
    # (treated as passed) for backward compatibility with v0.1 callers.
    # ------------------------------------------------------------------
    if v2_metrics is not None:
        # Gate 9: workflow routing accuracy non-decrease.
        wf_b = v2_metrics.baseline_workflow_routing_accuracy
        wf_c = v2_metrics.candidate_workflow_routing_accuracy
        g9_pass = (
            not config.require_workflow_routing_non_decrease
            or wf_c >= wf_b - 1e-9
        ) and wf_c >= config.workflow_routing_accuracy_min
        gates.append({
            "name": "workflow_routing_accuracy_non_decrease",
            "passed": g9_pass,
            "baseline": wf_b,
            "candidate": wf_c,
            "threshold": config.workflow_routing_accuracy_min,
        })

        # Gate 10: over-routing rate does not increase beyond threshold.
        or_b = v2_metrics.baseline_over_routing_rate
        or_c = v2_metrics.candidate_over_routing_rate
        g10_pass = (
            not config.require_over_routing_non_increase
            or or_c <= or_b + config.over_routing_rate_max_increase + 1e-9
        )
        gates.append({
            "name": "over_routing_rate_non_increase",
            "passed": g10_pass,
            "baseline": or_b,
            "candidate": or_c,
            "max_increase": config.over_routing_rate_max_increase,
        })

        # Gate 11: calibration Brier score below threshold.
        brier = v2_metrics.candidate_brier_score
        g11_pass = (
            not config.require_calibration
            or brier <= config.brier_score_max + 1e-9
        )
        gates.append({
            "name": "calibration_brier_below_threshold",
            "passed": g11_pass,
            "candidate": brier,
            "threshold": config.brier_score_max,
        })

    failed = [g for g in gates if not g["passed"]]
    passed = not failed
    reason = (
        "all gates passed"
        if passed
        else "failed gates: " + ", ".join(g["name"] for g in failed)
    )
    return GateResult(passed=passed, gates=gates, reason=reason)


@dataclass(slots=True)
class V2GateMetrics:
    """v2 metrics supplied to the promotion gate (gates 9-11)."""

    baseline_workflow_routing_accuracy: float = 0.0
    candidate_workflow_routing_accuracy: float = 0.0
    baseline_over_routing_rate: float = 0.0
    candidate_over_routing_rate: float = 0.0
    candidate_brier_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_workflow_routing_accuracy": self.baseline_workflow_routing_accuracy,
            "candidate_workflow_routing_accuracy": self.candidate_workflow_routing_accuracy,
            "baseline_over_routing_rate": self.baseline_over_routing_rate,
            "candidate_over_routing_rate": self.candidate_over_routing_rate,
            "candidate_brier_score": self.candidate_brier_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "V2GateMetrics":
        data = data or {}
        return cls(
            baseline_workflow_routing_accuracy=float(data.get("baseline_workflow_routing_accuracy", 0.0)),
            candidate_workflow_routing_accuracy=float(data.get("candidate_workflow_routing_accuracy", 0.0)),
            baseline_over_routing_rate=float(data.get("baseline_over_routing_rate", 0.0)),
            candidate_over_routing_rate=float(data.get("candidate_over_routing_rate", 0.0)),
            candidate_brier_score=float(data.get("candidate_brier_score", 0.0)),
        )
