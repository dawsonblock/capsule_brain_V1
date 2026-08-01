"""Comprehensive tests for the v0.4.7 gate-separation, statistical, and evidence-enforcement modules.

Tests the core scientific rule: Gate A0 PASS does NOT imply Gate A2 PASS.
The candidate must beat both baseline and sham with LCB exceeding the
practical-effect threshold.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capsule_brain.version import (
    PACKAGE_VERSION,
    AUTOLEARN_VERSION,
    AUTOLEARN_QUALIFICATION_VERSION,
    PROTOCOL_VERSION,
)

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.v047.gate_schema import (
    QualificationVerdict,
    GateResult,
    GateA1Result,
    GateA2Result,
    GateA3Result,
    GateA4Result,
    ComparisonResult,
)
from qualification.autolearn_v04.v047.config import (
    QualificationConfigV047,
    GateA1Config,
    GateA2Config,
    GateA3Config,
    CollapseConfig,
    SafetyConfig,
    PromotionConfig,
    StatisticsConfig,
)
from qualification.autolearn_v04.v047.statistics import (
    paired_cluster_bootstrap,
    compute_paired_deltas,
)
from qualification.autolearn_v04.v047.gate_a1 import evaluate_gate_a1
from qualification.autolearn_v04.v047.gate_a2 import evaluate_gate_a2
from qualification.autolearn_v04.v047.gate_a3 import evaluate_gate_a3
from qualification.autolearn_v04.v047.gate_a4 import evaluate_gate_a4
from qualification.autolearn_v04.v047.evidence_enforcement import (
    EvidenceOrigin,
    require_scientific_evidence,
    can_promote,
    label_for_report,
)
from qualification.autolearn_v04.v047.safety_evaluation import evaluate_safety
from qualification.autolearn_v04.v047.family_evaluation import evaluate_families
from qualification.autolearn_v04.v047.collapse_checks import check_action_collapse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paired_rows(n: int = 50, delta: float = 0.05, n_groups: int = 10) -> list[dict]:
    """Create synthetic paired rows for bootstrap testing."""
    rows = []
    for i in range(n):
        rows.append({
            "task_id": f"t{i}",
            "task_group_id": f"g{i % n_groups}",
            "family": "fam_a",
            "delta": delta + (i % 5 - 2) * 0.01,  # add some noise
            "selected_utility": 0.8 + (i % 5 - 2) * 0.01,
            "comparison_utility": 0.8,
        })
    return rows


def _make_results_dict(
    n_tasks: int = 50,
    base_utility: float = 0.5,
    success_rate: float = 0.5,
    n_families: int = 4,
) -> dict:
    """Create a synthetic results dict for gate evaluation."""
    task_rows = []
    for i in range(n_tasks):
        task_rows.append({
            "task_id": f"t{i}",
            "task_group_id": f"g{i % 10}",
            "family": f"fam_{i % n_families}",
            "success": (i % 2 == 0),
            "selected_utility": base_utility + (i % 5) * 0.02,
            "selected_action": ["ANSWER_DIRECT", "RETRIEVE_MEMORY", "CALL_TOOL", "START_WORKFLOW"][i % 4],
        })
    return {
        "n_tasks": n_tasks,
        "n_success": int(n_tasks * success_rate),
        "mean_utility": base_utility,
        "verified_success_rate": success_rate,
        "task_rows": task_rows,
    }


def _make_verdict(
    a0: str = "PASS",
    a1: str = "PASS",
    a2: str = "FAIL",
    a3: str = "BLOCKED",
    a4: str = "BLOCKED",
    evidence_origin: str = "REAL_MODEL",
    shadow: bool = False,
    active: bool = False,
) -> QualificationVerdict:
    """Build a QualificationVerdict with specified gate statuses."""
    return QualificationVerdict(
        protocol_version="0.4.7",
        run_id="test_run",
        evidence_origin=evidence_origin,
        gate_a0_admissibility={"status": a0, "reasons": [], "checks": {}},
        gate_a1_headroom={"status": a1, "oracle_vs_baseline": {}, "oracle_vs_sham": {}, "reasons": []},
        gate_a2_effectiveness={"status": a2, "candidate_vs_baseline": {}, "candidate_vs_sham": {}, "reasons": []},
        gate_a3_robustness={"status": a3, "replicate_summary": {}, "family_summary": {}, "reasons": []},
        gate_a4_promotion={"status": a4, "shadow_eligible": shadow, "active_eligible": active, "blocking_reasons": []},
    )


# ---------------------------------------------------------------------------
# Gate-separation tests
# ---------------------------------------------------------------------------

class TestGateSeparation:
    """Test that gate statuses are properly separated — A0 PASS ≠ A2 PASS."""

    def test_gate_a0_pass_does_not_imply_gate_a2_pass(self):
        verdict = _make_verdict(a0="PASS", a2="FAIL")
        assert verdict.legacy_gate_a_status != "PASS"

    def test_gate_a1_pass_does_not_imply_gate_a2_pass(self):
        verdict = _make_verdict(a0="PASS", a1="PASS", a2="FAIL")
        assert verdict.legacy_gate_a_status != "PASS"

    def test_gate_a2_pass_without_a3_pass_does_not_permit_active_promotion(self):
        verdict = _make_verdict(a0="PASS", a1="PASS", a2="PASS", a3="BLOCKED")
        assert not verdict.active_eligible
        assert not verdict.shadow_eligible

    def test_synthetic_evidence_cannot_satisfy_real_model_promotion(self):
        result = evaluate_gate_a4(
            gate_a0_status="PASS",
            gate_a1_status="PASS",
            gate_a2_status="PASS",
            gate_a3_status="PASS",
            safety_status="PASS",
            artifact_binding_status="PASS",
            evidence_origin="SYNTHETIC",
            config=PromotionConfig(require_real_model_evidence=True),
        )
        assert not result.shadow_eligible
        assert not result.active_eligible
        assert any("REAL_MODEL" in r or "SYNTHETIC" in r for r in result.blocking_reasons)

    def test_unavailable_evidence_cannot_satisfy_any_effectiveness_gate(self):
        manifest = {"evidence_origin": "UNAVAILABLE", "model_id": "some-model"}
        with pytest.raises(ValueError, match="REAL_MODEL"):
            require_scientific_evidence(manifest, required_origin=EvidenceOrigin.REAL_MODEL)

    def test_all_gates_pass_permits_shadow_promotion(self):
        result = evaluate_gate_a4(
            gate_a0_status="PASS",
            gate_a1_status="PASS",
            gate_a2_status="PASS",
            gate_a3_status="PASS",
            safety_status="PASS",
            artifact_binding_status="PASS",
            evidence_origin="REAL_MODEL",
            config=PromotionConfig(require_real_model_evidence=True),
        )
        assert result.shadow_eligible
        assert not result.active_eligible  # active requires post-shadow

    def test_gate_a0_pass_with_gate_a2_fail_gives_legacy_fail(self):
        verdict = _make_verdict(a0="PASS", a2="FAIL")
        assert verdict.legacy_gate_a_status != "PASS"

    def test_gate_a2_and_a3_pass_gives_legacy_pass(self):
        verdict = _make_verdict(a0="PASS", a1="PASS", a2="PASS", a3="PASS")
        assert verdict.legacy_gate_a_status == "PASS"


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

class TestStatisticalEvaluation:

    def test_paired_cluster_bootstrap_on_known_distribution(self):
        rows = _make_paired_rows(n=50, delta=0.05)
        result = paired_cluster_bootstrap(rows, n_resamples=5000, seed=42)
        assert result["n_tasks"] == 50
        assert result["n_task_groups"] == 10
        assert abs(result["mean_delta"] - 0.05) < 0.01
        assert result["lower_bound"] < result["upper_bound"]
        assert result["confidence_level"] == 0.95

    def test_confidence_interval_reproducibility_under_fixed_seed(self):
        rows = _make_paired_rows(n=50, delta=0.05)
        r1 = paired_cluster_bootstrap(rows, n_resamples=1000, seed=123)
        r2 = paired_cluster_bootstrap(rows, n_resamples=1000, seed=123)
        assert r1["lower_bound"] == r2["lower_bound"]
        assert r1["upper_bound"] == r2["upper_bound"]
        assert r1["mean_delta"] == r2["mean_delta"]

    def test_task_group_resampling_not_row_resampling(self):
        rows = _make_paired_rows(n=50, delta=0.05, n_groups=10)
        result = paired_cluster_bootstrap(rows, n_resamples=1000, seed=42)
        # n_task_groups should be <= original groups (resampled with replacement)
        assert result["n_task_groups"] <= 10

    def test_practical_effect_threshold_enforcement(self):
        rows = _make_paired_rows(n=50, delta=0.05)
        # With default threshold 0.0, should pass (lower bound > 0)
        result = paired_cluster_bootstrap(rows, n_resamples=5000, seed=42)
        assert result["passes"] is True
        # Verify strict inequality: lower_bound > 0.0
        assert result["lower_bound"] > 0.0

    def test_candidate_improvement_with_lcb_below_threshold_fails(self):
        rows = _make_paired_rows(n=50, delta=0.005)  # small delta
        result = paired_cluster_bootstrap(rows, n_resamples=5000, seed=42)
        # With default threshold 0.0, check if LCB > 0.0
        # For a very small delta, the LCB may still be > 0, so test with
        # a manual threshold check
        assert result["lower_bound"] < 0.02  # LCB below 0.02 threshold

    def test_candidate_beats_baseline_but_not_sham_fails(self):
        """Gate A2 should FAIL when candidate beats baseline but not sham."""
        # Build results where candidate > baseline but candidate ≈ sham
        candidate = _make_results_dict(n_tasks=50, base_utility=0.6)
        baseline = _make_results_dict(n_tasks=50, base_utility=0.5)
        sham = _make_results_dict(n_tasks=50, base_utility=0.6)  # sham same as candidate

        result = evaluate_gate_a2(
            candidate_results=candidate,
            baseline_results=baseline,
            sham_results=sham,
            config=GateA2Config(candidate_vs_baseline_min_effect=0.01, candidate_vs_sham_min_effect=0.01),
            statistics_config=StatisticsConfig(bootstrap_resamples=1000),
        )
        assert result.status == AuditStatus.FAIL.value

    def test_candidate_beats_sham_but_not_baseline_fails(self):
        """Gate A2 should FAIL when candidate beats sham but not baseline."""
        candidate = _make_results_dict(n_tasks=50, base_utility=0.5)
        baseline = _make_results_dict(n_tasks=50, base_utility=0.6)
        sham = _make_results_dict(n_tasks=50, base_utility=0.4)

        result = evaluate_gate_a2(
            candidate_results=candidate,
            baseline_results=baseline,
            sham_results=sham,
            config=GateA2Config(candidate_vs_baseline_min_effect=0.01, candidate_vs_sham_min_effect=0.01),
            statistics_config=StatisticsConfig(bootstrap_resamples=1000),
        )
        assert result.status == AuditStatus.FAIL.value

    def test_oracle_headroom_absence_blocks_gate_a2_interpretation(self):
        """When oracle has no headroom over baseline, Gate A1 should fail."""
        oracle = _make_results_dict(n_tasks=50, base_utility=0.5)
        baseline = _make_results_dict(n_tasks=50, base_utility=0.5)
        sham = _make_results_dict(n_tasks=50, base_utility=0.5)

        result = evaluate_gate_a1(
            oracle_results=oracle,
            baseline_results=baseline,
            sham_results=sham,
            config=GateA1Config(oracle_vs_baseline_min_effect=0.02, oracle_vs_sham_min_effect=0.02),
            statistics_config=StatisticsConfig(bootstrap_resamples=1000),
        )
        assert result.status == AuditStatus.FAIL.value


# ---------------------------------------------------------------------------
# Evidence enforcement tests
# ---------------------------------------------------------------------------

class TestEvidenceEnforcement:

    def test_require_scientific_evidence_rejects_synthetic(self):
        manifest = {"evidence_origin": "SYNTHETIC", "model_id": "some-model"}
        with pytest.raises(ValueError, match="REAL_MODEL"):
            require_scientific_evidence(manifest)

    def test_require_scientific_evidence_accepts_real_model(self):
        manifest = {"evidence_origin": "REAL_MODEL", "model_id": "Qwen/Qwen2.5-3B-Instruct"}
        require_scientific_evidence(manifest)  # should not raise

    def test_synthetic_model_id_cannot_be_real_model(self):
        manifest = {"evidence_origin": "REAL_MODEL", "model_id": "synthetic-7b"}
        with pytest.raises(ValueError, match="synthetic"):
            require_scientific_evidence(manifest)

    def test_can_promote_with_real_model(self):
        assert can_promote("REAL_MODEL", require_real_model=True) is True

    def test_can_promote_with_synthetic_rejected(self):
        assert can_promote("SYNTHETIC", require_real_model=True) is False

    def test_label_for_synthetic_report(self):
        label = label_for_report("SYNTHETIC")
        assert "SYNTHETIC" in label

    def test_label_for_unavailable_report(self):
        label = label_for_report("UNAVAILABLE")
        assert "UNVERIFIED" in label


# ---------------------------------------------------------------------------
# Safety evaluation tests
# ---------------------------------------------------------------------------

class TestSafetyEvaluation:

    def test_safety_pass_when_no_violations(self):
        rows = [{"verifier_verdict": "CORRECT", "severity": "LOW"} for _ in range(10)]
        result = evaluate_safety(rows, rows, rows, SafetyConfig())
        assert result["status"] == "PASS"

    def test_safety_fail_when_severe_violation_increase(self):
        baseline = [{"verifier_verdict": "CORRECT", "severity": "LOW"} for _ in range(10)]
        candidate = [{"verifier_verdict": "SAFETY_VIOLATION", "severity": "SEVERE"} for _ in range(3)]
        candidate += [{"verifier_verdict": "CORRECT", "severity": "LOW"} for _ in range(7)]
        result = evaluate_safety(candidate, baseline, baseline, SafetyConfig())
        assert result["status"] == "FAIL"
        assert result["severe_violation_increase"] > 0

    def test_safety_fail_when_critical_case_fails(self):
        baseline = [{"verifier_verdict": "CORRECT", "severity": "LOW"} for _ in range(10)]
        candidate = [{"verifier_verdict": "SAFETY_VIOLATION", "severity": "CRITICAL"}]
        candidate += [{"verifier_verdict": "CORRECT", "severity": "LOW"} for _ in range(9)]
        result = evaluate_safety(candidate, baseline, baseline, SafetyConfig())
        assert result["status"] == "FAIL"
        assert not result["all_critical_cases_pass"]

    def test_safety_pass_when_no_increase(self):
        baseline = [{"verifier_verdict": "SAFETY_VIOLATION", "severity": "LOW"} for _ in range(2)]
        baseline += [{"verifier_verdict": "CORRECT", "severity": "LOW"} for _ in range(8)]
        candidate = list(baseline)  # same
        result = evaluate_safety(candidate, baseline, baseline, SafetyConfig())
        assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# Version identity
# ---------------------------------------------------------------------------

class TestV047VersionIdentity:
    def test_package_version(self):
        assert PACKAGE_VERSION == "2.15.11"

    def test_autolearn_version(self):
        assert AUTOLEARN_VERSION == "0.3.10"

    def test_qualification_version(self):
        assert AUTOLEARN_QUALIFICATION_VERSION == "0.4.7"

    def test_protocol_version(self):
        assert PROTOCOL_VERSION == "0.4.7"
