"""Comprehensive tests for the v0.4.7 artifact integrity, promotion binding,
run directory, and reporting modules.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.v047.gate_schema import (
    QualificationVerdict,
    GateResult,
    GateA4Result,
)
from qualification.autolearn_v04.v047.config import (
    PromotionConfig,
    SafetyConfig,
    GateA3Config,
    CollapseConfig,
)
from qualification.autolearn_v04.v047.run_directory import (
    REQUIRED_FILES,
    REQUIRED_DIRS,
    create_run_directory,
    validate_run_directory,
    compute_checksums,
    write_checksums_file,
    verify_checksums,
    build_run_manifest,
    build_source_manifest,
)
from qualification.autolearn_v04.v047.artifact_dag import (
    ArtifactDAG,
    ArtifactNode,
    ARTIFACT_TYPES,
)
from qualification.autolearn_v04.v047.promotion_manifest import (
    PromotionManifest,
    verify_promotion_manifest,
)
from qualification.autolearn_v04.v047.gate_a4 import evaluate_gate_a4
from qualification.autolearn_v04.v047.report import generate_v047_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verdict_dict(
    a0: str = "PASS",
    a1: str = "PASS",
    a2: str = "FAIL",
    a3: str = "BLOCKED",
    a4: str = "BLOCKED",
    evidence_origin: str = "REAL_MODEL",
    shadow: bool = False,
    active: bool = False,
) -> dict:
    return QualificationVerdict(
        protocol_version="0.4.7",
        run_id="test_run_001",
        evidence_origin=evidence_origin,
        gate_a0_admissibility={"status": a0, "reasons": [], "checks": {}},
        gate_a1_headroom={"status": a1, "oracle_vs_baseline": {}, "oracle_vs_sham": {}, "reasons": []},
        gate_a2_effectiveness={"status": a2, "candidate_vs_baseline": {}, "candidate_vs_sham": {}, "reasons": []},
        gate_a3_robustness={"status": a3, "replicate_summary": {}, "family_summary": {}, "reasons": []},
        gate_a4_promotion={"status": a4, "shadow_eligible": shadow, "active_eligible": active, "blocking_reasons": []},
    ).to_dict()


def _make_evidence_summary() -> dict:
    return {
        "n_tasks": 80,
        "n_counterfactual_outcomes": 320,
        "n_experience_rows": 120,
        "n_safety_rows": 10,
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "model_revision": "abc123",
        "provider_class": "real_model",
        "evidence_origin": "REAL_MODEL",
        "verified_success_rate": 0.27,
        "mean_utility": 0.55,
    }


def _make_family_eval() -> dict:
    return {
        "families": {
            "fam_a": {"task_count": 30, "status": "SUFFICIENT"},
            "fam_b": {"task_count": 5, "status": "INSUFFICIENT_SUPPORT"},
        },
        "critical_regressions": [],
        "n_sufficient": 1,
        "n_insufficient": 1,
    }


def _make_collapse_check() -> dict:
    return {
        "status": "PASS",
        "entropy": 1.2,
        "max_action_share": 0.5,
        "action_coverage": 4,
        "abstention_share": 0.0,
        "invalid_action_rate": 0.0,
    }


def _make_safety_check() -> dict:
    return {
        "status": "PASS",
        "n_safety_tasks": 10,
        "severe_violation_increase": 0,
        "all_critical_cases_pass": True,
    }


# ---------------------------------------------------------------------------
# Artifact integrity tests
# ---------------------------------------------------------------------------

class TestArtifactIntegrity:

    def test_missing_checksum_fails(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Create some files but no CHECKSUMS.sha256
        (run_dir / "RUN_MANIFEST.json").write_text("{}")
        result = validate_run_directory(run_dir)
        assert not result["valid"]
        assert "CHECKSUMS.sha256" in result.get("missing", [])

    def test_changed_artifact_fails(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "RUN_MANIFEST.json").write_text('{"v": 1}')
        write_checksums_file(run_dir)
        # Modify the file
        (run_dir / "RUN_MANIFEST.json").write_text('{"v": 2}')
        result = verify_checksums(run_dir)
        assert not result["valid"]
        assert len(result["mismatches"]) > 0

    def test_wrong_run_id_fails(self):
        h64 = "a" * 64
        dag = ArtifactDAG()
        dag.add_node(ArtifactNode("a1", "SOURCE", "p1", h64, run_id="run_A"))
        dag.add_node(ArtifactNode("a2", "MODEL", "p2", h64, parents=["a1"], run_id="run_B"))
        result = dag.validate()
        assert not result["valid"]
        assert any("run" in issue.lower() or "cross" in issue.lower() for issue in result["issues"])

    def test_cross_run_policy_reuse_fails(self):
        h64 = "a" * 64
        dag = ArtifactDAG()
        dag.add_node(ArtifactNode("s1", "SOURCE", "p1", h64, run_id="run_A"))
        dag.add_node(ArtifactNode("m1", "MODEL", "p2", h64, parents=["s1"], run_id="run_A"))
        dag.add_node(ArtifactNode("c1", "CANDIDATE_POLICY", "p3", h64, parents=["m1"], run_id="run_B"))
        result = dag.validate()
        assert not result["valid"]

    def test_missing_source_manifest_fails(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        result = validate_run_directory(run_dir)
        assert not result["valid"]
        assert "SOURCE_MANIFEST.json" in result.get("missing", [])

    def test_dag_cycle_detection(self):
        h64 = "a" * 64
        dag = ArtifactDAG()
        dag.add_node(ArtifactNode("a", "SOURCE", "p", h64, run_id="r"))
        dag.add_node(ArtifactNode("b", "MODEL", "p", h64, parents=["a"], run_id="r"))
        # Create cycle: a depends on b
        dag.nodes["a"].parents = ["b"]
        result = dag.validate()
        assert not result["valid"]
        assert any("cycle" in issue.lower() for issue in result["issues"])

    def test_dag_missing_parent(self):
        h64 = "a" * 64
        dag = ArtifactDAG()
        dag.add_node(ArtifactNode("a", "MODEL", "p", h64, parents=["nonexistent"], run_id="r"))
        result = dag.validate()
        assert not result["valid"]
        assert any("missing" in issue.lower() and "parent" in issue.lower() for issue in result["issues"])

    def test_dag_valid(self):
        h64 = "a" * 64  # valid 64-char sha256
        dag = ArtifactDAG()
        dag.add_node(ArtifactNode("s", "SOURCE", "p1", h64, run_id="r"))
        dag.add_node(ArtifactNode("m", "MODEL", "p2", h64, parents=["s"], run_id="r"))
        dag.add_node(ArtifactNode("c", "CANDIDATE_POLICY", "p3", h64, parents=["m"], run_id="r"))
        result = dag.validate()
        assert result["valid"], f"Expected valid DAG but got issues: {result['issues']}"


# ---------------------------------------------------------------------------
# Promotion tests
# ---------------------------------------------------------------------------

class TestPromotionBinding:

    def test_promotion_manifest_binds_all_required_artifacts(self):
        manifest = PromotionManifest(
            promotion_id="",
            policy_id="pol_1",
            policy_digest="d1",
            run_id="run_1",
            source_digest="d2",
            benchmark_digest="d3",
            training_data_digest="d4",
            evaluation_digest="d5",
            gate_a0_digest="d6",
            gate_a1_digest="d7",
            gate_a2_digest="d8",
            gate_a3_digest="d9",
            safety_digest="d10",
        )
        d = manifest.to_dict()
        assert all(k in d for k in [
            "policy_id", "policy_digest", "run_id", "source_digest",
            "benchmark_digest", "training_data_digest", "evaluation_digest",
            "gate_a0_digest", "gate_a1_digest", "gate_a2_digest", "gate_a3_digest",
            "safety_digest",
        ])

    def test_altered_policy_digest_invalidates_promotion(self):
        manifest = PromotionManifest(
            promotion_id="",
            policy_id="pol_1",
            policy_digest="correct_digest",
            run_id="run_1",
            source_digest="d2",
            benchmark_digest="d3",
            training_data_digest="d4",
            evaluation_digest="d5",
            gate_a0_digest="d6",
            gate_a1_digest="d7",
            gate_a2_digest="d8",
            gate_a3_digest="d9",
            safety_digest="d10",
        )
        manifest.promotion_id = manifest.compute_promotion_id()
        # Wrong policy digest
        policy_artifact = {"sha256": "wrong_digest"}
        gate_results = {}
        result = verify_promotion_manifest(manifest.to_dict(), policy_artifact, gate_results)
        assert not result["valid"]

    def test_gate_a0_only_policy_cannot_promote(self):
        result = evaluate_gate_a4(
            gate_a0_status="PASS",
            gate_a1_status="NOT_RUN",
            gate_a2_status="NOT_RUN",
            gate_a3_status="NOT_RUN",
            safety_status="NOT_RUN",
            artifact_binding_status="PASS",
            evidence_origin="REAL_MODEL",
            config=PromotionConfig(),
        )
        assert not result.shadow_eligible
        assert not result.active_eligible

    def test_gate_a2_only_policy_cannot_become_active(self):
        result = evaluate_gate_a4(
            gate_a0_status="PASS",
            gate_a1_status="PASS",
            gate_a2_status="PASS",
            gate_a3_status="BLOCKED",
            safety_status="PASS",
            artifact_binding_status="PASS",
            evidence_origin="REAL_MODEL",
            config=PromotionConfig(),
        )
        assert not result.active_eligible
        assert not result.shadow_eligible

    def test_shadow_and_active_eligibility_distinct(self):
        result = evaluate_gate_a4(
            gate_a0_status="PASS",
            gate_a1_status="PASS",
            gate_a2_status="PASS",
            gate_a3_status="PASS",
            safety_status="PASS",
            artifact_binding_status="PASS",
            evidence_origin="REAL_MODEL",
            config=PromotionConfig(),
        )
        assert result.shadow_eligible
        assert not result.active_eligible  # active requires post-shadow

    def test_severe_safety_regression_blocks_promotion(self):
        result = evaluate_gate_a4(
            gate_a0_status="PASS",
            gate_a1_status="PASS",
            gate_a2_status="PASS",
            gate_a3_status="PASS",
            safety_status="FAIL",
            artifact_binding_status="PASS",
            evidence_origin="REAL_MODEL",
            config=PromotionConfig(),
        )
        assert not result.shadow_eligible

    def test_promotion_id_computation(self):
        m1 = PromotionManifest(
            promotion_id="", policy_id="p", policy_digest="d1", run_id="r",
            source_digest="d2", benchmark_digest="d3", training_data_digest="d4",
            evaluation_digest="d5", gate_a0_digest="d6", gate_a1_digest="d7",
            gate_a2_digest="d8", gate_a3_digest="d9", safety_digest="d10",
        )
        m2 = PromotionManifest(
            promotion_id="", policy_id="p", policy_digest="d1", run_id="r",
            source_digest="d2", benchmark_digest="d3", training_data_digest="d4",
            evaluation_digest="d5", gate_a0_digest="d6", gate_a1_digest="d7",
            gate_a2_digest="d8", gate_a3_digest="d9", safety_digest="d10",
        )
        assert m1.compute_promotion_id() == m2.compute_promotion_id()
        # Different digest → different ID
        m2.policy_digest = "different"
        assert m1.compute_promotion_id() != m2.compute_promotion_id()


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerializationParity:

    def test_qualification_verdict_serialization(self):
        verdict = QualificationVerdict(
            protocol_version="0.4.7",
            run_id="test",
            evidence_origin="REAL_MODEL",
            gate_a0_admissibility={"status": "PASS"},
            gate_a1_headroom={"status": "PASS"},
            gate_a2_effectiveness={"status": "FAIL"},
            gate_a3_robustness={"status": "BLOCKED"},
            gate_a4_promotion={"status": "BLOCKED", "shadow_eligible": False},
        )
        d = verdict.to_dict()
        j = verdict.to_json()
        assert d["protocol_version"] == "0.4.7"
        assert json.loads(j)["protocol_version"] == "0.4.7"
        assert d["gate_a0_admissibility"]["status"] == "PASS"

    def test_machine_verdict_format(self):
        verdict = QualificationVerdict(
            run_id="test",
            evidence_origin="REAL_MODEL",
            gate_a0_admissibility={"status": "PASS"},
            gate_a1_headroom={"status": "PASS"},
            gate_a2_effectiveness={"status": "FAIL"},
            gate_a3_robustness={"status": "BLOCKED"},
            gate_a4_promotion={"status": "BLOCKED", "shadow_eligible": False, "active_eligible": False},
        )
        mv = verdict.to_machine_verdict()
        required_keys = [
            "release", "autolearn", "qualification", "protocol",
            "gate_a0_admissibility", "gate_a1_headroom", "gate_a2_effectiveness",
            "gate_a3_robustness", "gate_a4_promotion",
            "shadow_eligible", "active_eligible", "evidence_origin",
        ]
        for key in required_keys:
            assert key in mv, f"Missing key: {key}"

    def test_legacy_gate_a_status_conservative(self):
        verdict = QualificationVerdict(
            gate_a0_admissibility={"status": "PASS"},
            gate_a2_effectiveness={"status": "FAIL"},
            gate_a3_robustness={"status": "BLOCKED"},
        )
        # A0 PASS must never map to legacy PASS
        assert verdict.legacy_gate_a_status != "PASS"


# ---------------------------------------------------------------------------
# Reporting tests
# ---------------------------------------------------------------------------

class TestReporting:

    def test_report_never_says_gate_a_pass_when_only_a0_passed(self):
        verdict = _make_verdict_dict(a0="PASS", a2="FAIL", a3="BLOCKED")
        report = generate_v047_report(
            verdict=verdict,
            config={},
            evidence_summary=_make_evidence_summary(),
            family_evaluation=_make_family_eval(),
            collapse_check=_make_collapse_check(),
            safety_check=_make_safety_check(),
            checksums={},
        )
        # Must NOT contain "Gate A passed" or "Gate A: PASS"
        assert "Gate A passed" not in report
        assert "Gate A: PASS" not in report

    def test_evidence_origin_appears_in_report(self):
        verdict = _make_verdict_dict()
        report = generate_v047_report(
            verdict=verdict,
            config={},
            evidence_summary=_make_evidence_summary(),
            family_evaluation=_make_family_eval(),
            collapse_check=_make_collapse_check(),
            safety_check=_make_safety_check(),
            checksums={},
        )
        assert "REAL_MODEL" in report or "REAL-MODEL" in report

    def test_synthetic_labeled_in_report(self):
        verdict = _make_verdict_dict(evidence_origin="SYNTHETIC")
        summary = _make_evidence_summary()
        summary["evidence_origin"] = "SYNTHETIC"
        report = generate_v047_report(
            verdict=verdict,
            config={},
            evidence_summary=summary,
            family_evaluation=_make_family_eval(),
            collapse_check=_make_collapse_check(),
            safety_check=_make_safety_check(),
            checksums={},
        )
        assert "SYNTHETIC" in report

    def test_run_id_appears_in_report(self):
        verdict = _make_verdict_dict()
        report = generate_v047_report(
            verdict=verdict,
            config={},
            evidence_summary=_make_evidence_summary(),
            family_evaluation=_make_family_eval(),
            collapse_check=_make_collapse_check(),
            safety_check=_make_safety_check(),
            checksums={},
        )
        assert "test_run_001" in report

    def test_report_has_all_sections(self):
        verdict = _make_verdict_dict()
        report = generate_v047_report(
            verdict=verdict,
            config={},
            evidence_summary=_make_evidence_summary(),
            family_evaluation=_make_family_eval(),
            collapse_check=_make_collapse_check(),
            safety_check=_make_safety_check(),
            checksums={},
        )
        # Check for key section markers
        assert "Run identity" in report or "Run Identity" in report
        assert "Promotion" in report
        assert "Checksums" in report or "checksums" in report.lower()
        assert "Limitations" in report or "limitations" in report.lower()
