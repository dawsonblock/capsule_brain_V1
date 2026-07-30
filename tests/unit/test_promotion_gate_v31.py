"""Tests for promotion gate v3.1 (v0.3.1 / 2.15.2 Section 22).

Verifies that:
- Simulated evidence cannot promote.
- Candidate must beat baseline.
- Candidate must beat sham.
- Lower CI must exceed epsilon.
- Safety regression blocks promotion.
- Low effective coverage blocks promotion.
- Missing provenance blocks promotion.
- Failed parity blocks promotion.
- All 24 gates are evaluated.
- Gate status uses PASS/FAIL/BLOCKED/NOT_RUN.
"""
from __future__ import annotations

import pytest

from capsule_brain.autolearn.promotion import (
    GateConfigV31,
    GateStatus,
    PromotionEvidence,
    evaluate_promotion_gate_v31,
)


def _passing_evidence() -> PromotionEvidence:
    """Evidence that passes all gates."""
    return PromotionEvidence(
        training_runtime_type="real",
        validation_runtime_type="real",
        test_runtime_type="real",
        ood_runtime_type="real",
        post_promotion_runtime_type="real",
        policy_serialization_parity=True,
        feature_schema_match=True,
        baseline_verified_success_rate=0.7,
        candidate_verified_success_rate=0.8,
        baseline_mean_utility=5.0,
        candidate_mean_utility=6.0,
        delta_ci_low=0.5,
        delta_ci_high=1.5,
        mean_delta=1.0,
        sham_mean_utility=4.0,
        candidate_beats_sham=True,
        baseline_safety_violations=0,
        candidate_safety_violations=0,
        worst_family="direct",
        worst_family_delta=0.1,
        tool_precision=0.8,
        tool_recall=0.7,
        workflow_precision=0.75,
        workflow_recall=0.65,
        ood_mean_delta=0.2,
        brier_score=0.2,
        effective_candidate_coverage=0.7,
        baseline_runtime_error_rate=0.05,
        candidate_runtime_error_rate=0.06,
        provenance_complete=True,
        all_artifacts_exist=True,
        all_tests_pass=True,
    )


class TestPromotionGateV31:
    """Test the 24-gate promotion gate."""

    def test_all_gates_pass(self):
        evidence = _passing_evidence()
        result = evaluate_promotion_gate_v31(evidence)
        assert result.passed is True
        assert len(result.gates) == 24

    def test_simulated_training_blocks_promotion(self):
        evidence = _passing_evidence()
        evidence.training_runtime_type = "simulated"
        result = evaluate_promotion_gate_v31(evidence)
        assert result.passed is False
        assert any(g.name == "training_runtime_real" and g.status == GateStatus.FAIL
                   for g in result.gates)

    def test_simulated_test_blocks_promotion(self):
        evidence = _passing_evidence()
        evidence.test_runtime_type = "simulated"
        result = evaluate_promotion_gate_v31(evidence)
        assert result.passed is False

    def test_candidate_must_beat_baseline(self):
        evidence = _passing_evidence()
        evidence.candidate_mean_utility = 4.0  # less than baseline 5.0
        result = evaluate_promotion_gate_v31(evidence)
        assert result.passed is False
        assert any(g.name == "mean_utility_improves" and g.status == GateStatus.FAIL
                   for g in result.gates)

    def test_candidate_must_beat_sham(self):
        evidence = _passing_evidence()
        evidence.candidate_beats_sham = False
        result = evaluate_promotion_gate_v31(evidence)
        assert result.passed is False
        assert any(g.name == "candidate_beats_sham" and g.status == GateStatus.FAIL
                   for g in result.gates)

    def test_lower_ci_must_exceed_epsilon(self):
        evidence = _passing_evidence()
        evidence.delta_ci_low = 0.05  # below default epsilon 0.1
        result = evaluate_promotion_gate_v31(evidence)
        assert result.passed is False
        assert any(g.name == "ci_lower_bound_above_epsilon" and g.status == GateStatus.FAIL
                   for g in result.gates)

    def test_safety_regression_blocks(self):
        evidence = _passing_evidence()
        evidence.candidate_safety_violations = 5
        evidence.baseline_safety_violations = 0
        result = evaluate_promotion_gate_v31(evidence)
        assert result.passed is False
        assert any(g.name == "no_safety_regression" and g.status == GateStatus.FAIL
                   for g in result.gates)

    def test_low_effective_coverage_blocks(self):
        evidence = _passing_evidence()
        evidence.effective_candidate_coverage = 0.3  # below 0.60
        result = evaluate_promotion_gate_v31(evidence)
        assert result.passed is False
        assert any(g.name == "effective_coverage" and g.status == GateStatus.FAIL
                   for g in result.gates)

    def test_missing_provenance_blocks(self):
        evidence = _passing_evidence()
        evidence.provenance_complete = False
        result = evaluate_promotion_gate_v31(evidence)
        assert result.passed is False
        assert any(g.name == "provenance_complete" and g.status == GateStatus.FAIL
                   for g in result.gates)

    def test_failed_parity_blocks(self):
        evidence = _passing_evidence()
        evidence.policy_serialization_parity = False
        result = evaluate_promotion_gate_v31(evidence)
        assert result.passed is False
        assert any(g.name == "policy_serialization_parity" and g.status == GateStatus.FAIL
                   for g in result.gates)

    def test_missing_post_promotion_is_not_run(self):
        evidence = _passing_evidence()
        evidence.post_promotion_runtime_type = ""
        result = evaluate_promotion_gate_v31(evidence)
        # NOT_RUN counts as not passed
        assert result.passed is False
        gate = next(g for g in result.gates if g.name == "post_promotion_runtime_real")
        assert gate.status == GateStatus.NOT_RUN

    def test_all_24_gates_present(self):
        evidence = _passing_evidence()
        result = evaluate_promotion_gate_v31(evidence)
        names = [g.name for g in result.gates]
        expected = [
            "training_runtime_real",
            "validation_runtime_real",
            "test_runtime_real",
            "ood_runtime_real",
            "post_promotion_runtime_real",
            "policy_serialization_parity",
            "feature_schema_match",
            "verified_success_non_decrease",
            "mean_utility_improves",
            "ci_lower_bound_above_epsilon",
            "candidate_beats_sham",
            "no_safety_regression",
            "no_catastrophic_family_regression",
            "tool_precision",
            "tool_recall",
            "workflow_precision",
            "workflow_recall",
            "ood_utility_above_floor",
            "calibration_below_threshold",
            "effective_coverage",
            "runtime_error_no_increase",
            "provenance_complete",
            "all_artifacts_exist",
            "all_tests_pass",
        ]
        assert names == expected

    def test_gate_has_required_fields(self):
        evidence = _passing_evidence()
        result = evaluate_promotion_gate_v31(evidence)
        for g in result.gates:
            d = g.to_dict()
            assert "name" in d
            assert "status" in d
            assert "observed" in d
            assert "threshold" in d
            assert "reason" in d
            assert "evidence_artifact" in d

    def test_gate_status_enum_values(self):
        assert GateStatus.PASS.value == "PASS"
        assert GateStatus.FAIL.value == "FAIL"
        assert GateStatus.BLOCKED.value == "BLOCKED"
        assert GateStatus.NOT_RUN.value == "NOT_RUN"

    def test_custom_epsilon(self):
        evidence = _passing_evidence()
        evidence.delta_ci_low = 0.3
        config = GateConfigV31(epsilon_utility=0.5)
        result = evaluate_promotion_gate_v31(evidence, config)
        assert result.passed is False
        gate = next(g for g in result.gates if g.name == "ci_lower_bound_above_epsilon")
        assert gate.status == GateStatus.FAIL
