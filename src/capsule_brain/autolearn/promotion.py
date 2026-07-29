"""Promotion gate: a candidate may become active only if all gates pass.

Gates (initial configuration):
1. verified success rate does not decrease
2. mean utility improves
3. lower 95% CI of utility delta >= 0
4. no increase in safety violations
5. tool precision above threshold
6. tool recall above threshold
7. no catastrophic regression in any task family
8. OOD score not below configured floor

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

    failed = [g for g in gates if not g["passed"]]
    passed = not failed
    reason = (
        "all gates passed"
        if passed
        else "failed gates: " + ", ".join(g["name"] for g in failed)
    )
    return GateResult(passed=passed, gates=gates, reason=reason)
