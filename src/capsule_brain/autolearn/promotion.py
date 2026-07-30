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


# ---------------------------------------------------------------------------
# v0.3.1 Section 22: Promotion gate v3.1 (24 mandatory gates)
# ---------------------------------------------------------------------------

from enum import Enum


class GateStatus(Enum):
    """Status of a single promotion gate."""
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True, slots=True)
class GateV31:
    """A single gate in the v3.1 promotion gate (24 gates).

    Every gate must report: name, status, observed, threshold, reason,
    evidence_artifact.
    """
    name: str
    status: GateStatus
    observed: Any
    threshold: Any
    reason: str
    evidence_artifact: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "observed": self.observed,
            "threshold": self.threshold,
            "reason": self.reason,
            "evidence_artifact": self.evidence_artifact,
        }


@dataclass(slots=True)
class GateConfigV31:
    """v3.1 gate configuration: 24 mandatory gates."""
    # Runtime type gates (1-5)
    require_real_training_runtime: bool = True
    require_real_validation_runtime: bool = True
    require_real_test_runtime: bool = True
    require_real_ood_runtime: bool = True
    require_real_post_promotion_runtime: bool = True
    # Parity gates (6-8)
    require_policy_serialization_parity: bool = True
    require_feature_schema_match: bool = True
    # Performance gates (9-11)
    require_verified_success_non_decrease: bool = True
    require_mean_utility_improvement: bool = True
    epsilon_utility: float = 0.1  # minimum practical effect
    require_ci_lower_bound_above_epsilon: bool = True
    # Control gates (11-12)
    require_candidate_beats_sham: bool = True
    # Safety gates (12)
    require_no_safety_regression: bool = True
    # Family gates (13)
    require_no_catastrophic_family_regression: bool = True
    family_regression_max_delta: float = -2.0
    # Tool/workflow gates (14-17)
    tool_precision_min: float = 0.6
    tool_recall_min: float = 0.6
    workflow_precision_min: float = 0.6
    workflow_recall_min: float = 0.6
    # OOD gate (18)
    ood_utility_floor: float = 0.0
    # Calibration gate (19)
    calibration_max: float = 0.3
    # Coverage gate (20)
    effective_coverage_min: float = 0.60
    # Runtime error gate (21)
    runtime_error_max_increase: float = 0.1
    # Provenance gate (22)
    require_provenance_complete: bool = True
    # Artifacts gate (23)
    require_all_artifacts_exist: bool = True
    # Tests gate (24)
    require_all_tests_pass: bool = True


@dataclass(slots=True)
class GateResultV31:
    """Result of the v3.1 promotion gate (24 gates)."""
    passed: bool
    gates: list[GateV31] = field(default_factory=list)

    @property
    def reason(self) -> str:
        failed = [g for g in self.gates if g.status == GateStatus.FAIL]
        if not failed:
            return "all gates passed"
        return "failed gates: " + ", ".join(g.name for g in failed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gates": [g.to_dict() for g in self.gates],
            "reason": self.reason,
        }


@dataclass(slots=True)
class PromotionEvidence:
    """Evidence supplied to the v3.1 promotion gate."""
    # Runtime types
    training_runtime_type: str = ""
    validation_runtime_type: str = ""
    test_runtime_type: str = ""
    ood_runtime_type: str = ""
    post_promotion_runtime_type: str = ""
    # Parity
    policy_serialization_parity: bool = False
    feature_schema_match: bool = False
    # Performance
    baseline_verified_success_rate: float = 0.0
    candidate_verified_success_rate: float = 0.0
    baseline_mean_utility: float = 0.0
    candidate_mean_utility: float = 0.0
    delta_ci_low: float = 0.0
    delta_ci_high: float = 0.0
    mean_delta: float = 0.0
    # Sham comparison
    sham_mean_utility: float = 0.0
    candidate_beats_sham: bool = False
    # Safety
    baseline_safety_violations: int = 0
    candidate_safety_violations: int = 0
    # Family
    worst_family: str = ""
    worst_family_delta: float = 0.0
    # Tool/workflow
    tool_precision: float = 0.0
    tool_recall: float = 0.0
    workflow_precision: float = 0.0
    workflow_recall: float = 0.0
    # OOD
    ood_mean_delta: float = 0.0
    # Calibration
    brier_score: float = 0.0
    # Coverage
    effective_candidate_coverage: float = 0.0
    # Runtime error
    baseline_runtime_error_rate: float = 0.0
    candidate_runtime_error_rate: float = 0.0
    # Provenance
    provenance_complete: bool = False
    # Artifacts
    all_artifacts_exist: bool = False
    # Tests
    all_tests_pass: bool = False
    # Evidence artifact paths
    artifact_paths: dict[str, str] = field(default_factory=dict)


def evaluate_promotion_gate_v31(
    evidence: PromotionEvidence,
    config: GateConfigV31 | None = None,
) -> GateResultV31:
    """Evaluate the 24 mandatory v3.1 promotion gates.

    Promotion must fail unless every mandatory condition passes.
    """
    config = config or GateConfigV31()
    gates: list[GateV31] = []

    def _gate(name: str, status: GateStatus, observed: Any, threshold: Any,
              reason: str, artifact: str = "") -> None:
        gates.append(GateV31(
            name=name, status=status, observed=observed,
            threshold=threshold, reason=reason, evidence_artifact=artifact,
        ))

    # Gate 1: training runtime type == real
    s = GateStatus.PASS if evidence.training_runtime_type == "real" else GateStatus.FAIL
    if config.require_real_training_runtime and s == GateStatus.FAIL:
        s = GateStatus.FAIL
    _gate("training_runtime_real", s, evidence.training_runtime_type, "real",
          "training must use real runtime", "training_results.json")

    # Gate 2: validation runtime type == real
    s = GateStatus.PASS if evidence.validation_runtime_type == "real" else GateStatus.FAIL
    _gate("validation_runtime_real", s, evidence.validation_runtime_type, "real",
          "validation must use real runtime", "validation_results.json")

    # Gate 3: test runtime type == real
    s = GateStatus.PASS if evidence.test_runtime_type == "real" else GateStatus.FAIL
    _gate("test_runtime_real", s, evidence.test_runtime_type, "real",
          "test must use real runtime", "test_results.json")

    # Gate 4: OOD runtime type == real
    s = GateStatus.PASS if evidence.ood_runtime_type == "real" else GateStatus.FAIL
    _gate("ood_runtime_real", s, evidence.ood_runtime_type, "real",
          "OOD must use real runtime", "ood_results.json")

    # Gate 5: post-promotion runtime type == real
    s = GateStatus.PASS if evidence.post_promotion_runtime_type == "real" else GateStatus.NOT_RUN
    if evidence.post_promotion_runtime_type == "":
        s = GateStatus.NOT_RUN
    _gate("post_promotion_runtime_real", s, evidence.post_promotion_runtime_type, "real",
          "post-promotion must use real runtime", "post_promotion_results.json")

    # Gate 6: policy serialization parity passes
    s = GateStatus.PASS if evidence.policy_serialization_parity else GateStatus.FAIL
    _gate("policy_serialization_parity", s, evidence.policy_serialization_parity, True,
          "serialized and reloaded policies must produce identical predictions",
          "candidate_policy.json")

    # Gate 7: feature schema matches
    s = GateStatus.PASS if evidence.feature_schema_match else GateStatus.FAIL
    _gate("feature_schema_match", s, evidence.feature_schema_match, True,
          "feature schema must match between training and inference",
          "candidate_policy.json")

    # Gate 8: verified success does not decrease
    s = GateStatus.PASS if evidence.candidate_verified_success_rate >= evidence.baseline_verified_success_rate - 1e-9 else GateStatus.FAIL
    _gate("verified_success_non_decrease", s,
          evidence.candidate_verified_success_rate, evidence.baseline_verified_success_rate,
          "candidate verified success rate must not decrease",
          "test_results.json")

    # Gate 9: mean utility improves over baseline
    s = GateStatus.PASS if evidence.candidate_mean_utility > evidence.baseline_mean_utility + 1e-9 else GateStatus.FAIL
    _gate("mean_utility_improves", s,
          evidence.candidate_mean_utility, evidence.baseline_mean_utility,
          "candidate mean utility must improve over baseline",
          "test_results.json")

    # Gate 10: paired bootstrap lower bound > epsilon
    s = GateStatus.PASS if evidence.delta_ci_low > config.epsilon_utility else GateStatus.FAIL
    _gate("ci_lower_bound_above_epsilon", s,
          evidence.delta_ci_low, config.epsilon_utility,
          f"lower CI must exceed epsilon={config.epsilon_utility}",
          "test_results.json")

    # Gate 11: candidate beats sham
    s = GateStatus.PASS if evidence.candidate_beats_sham else GateStatus.FAIL
    _gate("candidate_beats_sham", s,
          evidence.candidate_mean_utility, evidence.sham_mean_utility,
          "candidate must beat sham control",
          "sham_results.json")

    # Gate 12: no safety regression
    s = GateStatus.PASS if evidence.candidate_safety_violations <= evidence.baseline_safety_violations else GateStatus.FAIL
    _gate("no_safety_regression", s,
          evidence.candidate_safety_violations, evidence.baseline_safety_violations,
          "candidate must not increase safety violations",
          "safety_results.json")

    # Gate 13: no catastrophic family regression
    s = GateStatus.PASS if evidence.worst_family_delta >= config.family_regression_max_delta else GateStatus.FAIL
    _gate("no_catastrophic_family_regression", s,
          evidence.worst_family_delta, config.family_regression_max_delta,
          f"worst family delta must not be below {config.family_regression_max_delta}",
          "test_results.json")

    # Gate 14: tool precision >= threshold
    s = GateStatus.PASS if evidence.tool_precision >= config.tool_precision_min else GateStatus.FAIL
    _gate("tool_precision", s, evidence.tool_precision, config.tool_precision_min,
          f"tool precision must be >= {config.tool_precision_min}",
          "test_results.json")

    # Gate 15: tool recall >= threshold
    s = GateStatus.PASS if evidence.tool_recall >= config.tool_recall_min else GateStatus.FAIL
    _gate("tool_recall", s, evidence.tool_recall, config.tool_recall_min,
          f"tool recall must be >= {config.tool_recall_min}",
          "test_results.json")

    # Gate 16: workflow precision >= threshold
    s = GateStatus.PASS if evidence.workflow_precision >= config.workflow_precision_min else GateStatus.FAIL
    _gate("workflow_precision", s, evidence.workflow_precision, config.workflow_precision_min,
          f"workflow precision must be >= {config.workflow_precision_min}",
          "test_results.json")

    # Gate 17: workflow recall >= threshold
    s = GateStatus.PASS if evidence.workflow_recall >= config.workflow_recall_min else GateStatus.FAIL
    _gate("workflow_recall", s, evidence.workflow_recall, config.workflow_recall_min,
          f"workflow recall must be >= {config.workflow_recall_min}",
          "test_results.json")

    # Gate 18: OOD utility >= floor
    s = GateStatus.PASS if evidence.ood_mean_delta >= config.ood_utility_floor else GateStatus.FAIL
    _gate("ood_utility_above_floor", s, evidence.ood_mean_delta, config.ood_utility_floor,
          f"OOD mean delta must be >= {config.ood_utility_floor}",
          "ood_results.json")

    # Gate 19: calibration <= threshold
    s = GateStatus.PASS if evidence.brier_score <= config.calibration_max + 1e-9 else GateStatus.FAIL
    _gate("calibration_below_threshold", s, evidence.brier_score, config.calibration_max,
          f"Brier score must be <= {config.calibration_max}",
          "calibration_results.json")

    # Gate 20: effective candidate coverage >= minimum
    s = GateStatus.PASS if evidence.effective_candidate_coverage >= config.effective_coverage_min else GateStatus.FAIL
    _gate("effective_coverage", s, evidence.effective_candidate_coverage, config.effective_coverage_min,
          f"effective candidate coverage must be >= {config.effective_coverage_min}",
          "coverage_results.json")

    # Gate 21: runtime-error rate does not materially increase
    delta_re = evidence.candidate_runtime_error_rate - evidence.baseline_runtime_error_rate
    s = GateStatus.PASS if delta_re <= config.runtime_error_max_increase + 1e-9 else GateStatus.FAIL
    _gate("runtime_error_no_increase", s, delta_re, config.runtime_error_max_increase,
          f"runtime error rate increase must be <= {config.runtime_error_max_increase}",
          "test_results.json")

    # Gate 22: provenance chain is complete
    s = GateStatus.PASS if evidence.provenance_complete else GateStatus.FAIL
    _gate("provenance_complete", s, evidence.provenance_complete, True,
          "provenance chain must be complete",
          "provenance_manifest.json")

    # Gate 23: all mandatory artifacts exist
    s = GateStatus.PASS if evidence.all_artifacts_exist else GateStatus.FAIL
    _gate("all_artifacts_exist", s, evidence.all_artifacts_exist, True,
          "all mandatory artifacts must exist",
          "artifacts/")

    # Gate 24: all existing tests pass
    s = GateStatus.PASS if evidence.all_tests_pass else GateStatus.FAIL
    _gate("all_tests_pass", s, evidence.all_tests_pass, True,
          "all existing tests must pass",
          "test_results/")

    failed = [g for g in gates if g.status in (GateStatus.FAIL, GateStatus.BLOCKED, GateStatus.NOT_RUN)]
    passed = not failed
    return GateResultV31(passed=passed, gates=gates)
