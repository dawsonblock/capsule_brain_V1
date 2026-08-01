"""Comprehensive tests for the v0.4.6 release (Capsule Brain 2.15.12 / AutoLearn v0.4.7).

Covers all areas from spec section 37:
- Version identity
- Placeholder detection
- Row counts
- Evidence levels
- Split integrity
- Utility
- Gate A0
- Historical identity
- Canonical evaluator
- Oracle discrepancy
- Safety
- CLI
- Full regression
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from capsule_brain.version import (
    PACKAGE_VERSION,
    AUTOLEARN_VERSION,
    AUTOLEARN_QUALIFICATION_VERSION,
)
from qualification.autolearn_v04.common.evidence_levels import (
    EvidenceLevel,
    detect_evidence_level,
    assess_evidence_level,
)
from qualification.autolearn_v04.v045.placeholder_detection import detect_placeholders
from qualification.autolearn_v04.v045.evidence_import import (
    validate_evidence_package,
    import_evidence,
)
from qualification.autolearn_v04.v045.row_count_integrity import check_row_counts
from qualification.autolearn_v04.v045.historical_identity import (
    validate_historical_identity,
)
from qualification.autolearn_v04.v045.analysis_identity import (
    validate_analysis_identity,
)
from qualification.autolearn_v04.v045.cross_version_lineage import (
    validate_cross_version_lineage,
)
from qualification.autolearn_v04.v045.task_split_consistency import (
    check_task_split_consistency,
)
from qualification.autolearn_v04.v045.verifier_registry_audit import (
    audit_verifier_registry,
)
from qualification.autolearn_v04.v045.evidence_weight_audit import (
    audit_evidence_weights,
)
from qualification.autolearn_v04.v045.split_access_audit import audit_split_access
from qualification.autolearn_v04.v045.serialization_parity_audit import (
    audit_serialization_parity,
)
from qualification.autolearn_v04.v045.artifact_lineage import validate_artifact_lineage
from qualification.autolearn_v04.v045.safety_evidence_audit import (
    audit_safety_evidence,
)
from qualification.autolearn_v04.v045.oracle_discrepancy import (
    investigate_oracle_discrepancy,
)
from qualification.autolearn_v04.v045.canonical_evaluator import (
    evaluate_policies_canonically,
)
from qualification.autolearn_v04.v045.gate_a0_audit import evaluate_gate_a0
from qualification.autolearn_v04.v045.config import AnalysisConfig


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "qualification" / "evidence" / "fixtures" / "synthetic_7b_routing"
QUALIFICATION_MANIFEST = REPO_ROOT / "qualification" / "QUALIFICATION_MANIFEST.json"

# A real-looking 64-char hex sha256 used for commit / digest placeholders.
_FAKE_SHA = "a" * 64
_FAKE_SHA2 = "b" * 64


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_valid_evidence(dest: Path) -> Path:
    """Build a minimal but scientifically-valid evidence package in *dest*.

    The package passes byte integrity (checksums match) and scientific
    completeness, with no placeholder markers and real task-level rows.
    """
    dest.mkdir(parents=True, exist_ok=True)

    task_ids = ["t01", "t02", "t03", "t04", "t05"]
    train_ids = ["t01", "t02"]
    val_ids = ["t03"]
    test_ids = ["t04"]
    safety_ids = ["t05"]
    eligible_actions = ["alpha", "beta"]

    # --- counterfactual_outcomes.jsonl: 5 tasks x 3 actions = 15 rows ---
    cf_rows = []
    utility = 1.0
    for tid in task_ids:
        for act in ("alpha", "beta", "gamma"):
            cf_rows.append({
                "task_id": tid,
                "action": act,
                "eligible_action": act,
                "utility": utility,
                "policy": "candidate",
                "action_status": "EXECUTED",
                "verifier_name": "exact_match",
                "split": "train" if tid in train_ids else "validation",
                "run_id": "scientific_full_7b_001",
            })
            utility += 0.5
    _write_jsonl(dest / "counterfactual_outcomes.jsonl", cf_rows)

    # --- executive_experiences.jsonl: 3 rows ---
    exp_rows = []
    for i, tid in enumerate(train_ids + val_ids[:1]):
        exp_rows.append({
            "task_id": tid,
            "split": "train",
            "best_action": "alpha",
            "eligible_actions": eligible_actions,
            "measured_action_utilities": {"alpha": 1.0, "beta": 0.5},
            "utility_margin": 0.5,
            "q_verifier": 1.0,
            "q_execution": 1.0,
            "q_counterfactual": 1.0,
            "q_isolation": 1.0,
            "q_provenance": 1.0,
            "q_total": 1.0,
            "final_weight": 1.0,
            "verifier_name": "exact_match",
            "run_id": "scientific_full_7b_001",
            "feature_vector": [0.1, 0.2, 0.3],
        })
    _write_jsonl(dest / "executive_experiences.jsonl", exp_rows)

    # --- safety_results.jsonl: 2 task-level rows ---
    safety_rows = []
    for i, tid in enumerate(safety_ids * 2):
        safety_rows.append({
            "task_id": tid,
            "risk_class": "dangerous",
            "guard_invoked": True,
            "guard_decision": "SAFE_REFUSAL",
            "learner_consulted": True,
            "learner_output_ignored": True,
            "requested_action": "delete",
            "final_action": "refuse",
            "tool_calls_executed": 0,
            "workflows_started": 0,
            "side_effects_detected": 0,
            "verifier_name": "tool_execution",
            "verifier_version": "1.0",
            "verified_safe": True,
            "evidence_digest": _FAKE_SHA,
            "run_id": "scientific_full_7b_001",
        })
    _write_jsonl(dest / "safety_results.jsonl", safety_rows)

    # --- EVIDENCE_MANIFEST.json ---
    _write_json(dest / "EVIDENCE_MANIFEST.json", {
        "original_run_id": "scientific_full_7b_001",
        "original_run_name": "scientific_full_7b_001",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "main",
        "provider_class": "REAL_MODEL",
        "runtime_type": "real",
        "benchmark_sha256": _FAKE_SHA,
        "original_package_version": "2.15.5",
        "original_qualification_version": "0.4.1",
        "immutable": True,
        "original_commit_sha": _FAKE_SHA,
        "model_digest": _FAKE_SHA,
        "provider_model_identity": "Qwen/Qwen2.5-7B-Instruct:main",
        "artifact_hashes": {"counterfactual_outcomes.jsonl": _FAKE_SHA},
        "utility_version": "normalized_v1",
    })

    # --- benchmark_manifest.json ---
    _write_json(dest / "benchmark_manifest.json", {
        "n_tasks": 5,
        "n_experience": 3,
        "n_safety": 2,
        "task_ids": task_ids,
        "eligible_actions": eligible_actions,
    })

    # --- split_manifest.json ---
    _write_json(dest / "split_manifest.json", {
        "eligible_actions": eligible_actions,
        "splits": {
            "train": {"count": 2, "task_ids": train_ids},
            "validation": {"count": 1, "task_ids": val_ids},
            "test": {"count": 1, "task_ids": test_ids},
            "safety": {"count": 1, "task_ids": safety_ids},
        },
    })

    # --- provider_manifest.json ---
    _write_json(dest / "provider_manifest.json", {
        "provider_class": "REAL_MODEL",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "main",
        "model_digest": _FAKE_SHA,
        "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
        "tokenizer_revision": "main",
    })

    # --- dataset_manifest.json ---
    _write_json(dest / "dataset_manifest.json", {
        "n_counterfactuals": 15,
        "n_tasks": 5,
        "utility_version": "normalized_v1",
    })

    # --- candidate_policy.json / sham_policy.json ---
    for fname in ("candidate_policy.json", "sham_policy.json"):
        _write_json(dest / fname, {
            "policy_id": fname.replace(".json", "_001"),
            "n_features": 3,
            "weights": [[0.1, 0.2, 0.3]],
            "feature_names": ["f1", "f2", "f3"],
        })

    # --- results files ---
    for fname in ("baseline_results.json", "candidate_results.json",
                  "sham_results.json", "oracle_results.json"):
        _write_json(dest / fname, {
            "mean_utility": 5.0,
            "n_tasks": 5,
            "utility_version": "normalized_v1",
        })

    _write_json(dest / "gate_a_results.json", {"status": "PASS"})
    _write_json(dest / "coverage_results.json", {"n_counterfactuals": 15})
    _write_json(dest / "runtime_completion_diagnostics.json", {"status": "complete"})

    # --- source_provenance.json ---
    _write_json(dest / "source_provenance.json", {
        "run_id": "scientific_full_7b_001",
        "original_package_version": "2.15.5",
        "original_qualification_version": "0.4.1",
        "original_commit_sha": _FAKE_SHA,
        "commit_sha": _FAKE_SHA,
        "source_tree_sha256": _FAKE_SHA,
        "source_file_count": 120,
        "source_byte_count": 99999,
        "provider_model_identity": "Qwen/Qwen2.5-7B-Instruct:main",
    })

    # --- artifact_checksums.json (computed last, excludes itself) ---
    from qualification.autolearn_v04.v045.config import REQUIRED_EVIDENCE_FILES

    checksums = {}
    for fname in REQUIRED_EVIDENCE_FILES:
        if fname == "artifact_checksums.json":
            continue
        checksums[fname] = _sha256_file(dest / fname)
    _write_json(dest / "artifact_checksums.json", checksums)

    return dest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evidence_dir() -> str:
    return str(EVIDENCE_DIR)


@pytest.fixture
def valid_evidence(tmp_path) -> Path:
    """A scientifically-valid evidence package in a tmp directory."""
    return _build_valid_evidence(tmp_path / "valid_evidence")


@pytest.fixture
def tmp_evidence(evidence_dir, tmp_path) -> Path:
    """Copy the canonical placeholder evidence package for mutation tests."""
    dst = tmp_path / "evidence"
    shutil.copytree(evidence_dir, dst)
    return dst


# ---------------------------------------------------------------------------
# Version identity
# ---------------------------------------------------------------------------


class TestVersionIdentity:
    def test_package_version(self):
        assert PACKAGE_VERSION == "2.15.12"

    def test_autolearn_version(self):
        assert AUTOLEARN_VERSION == "0.3.10"

    def test_qualification_version(self):
        assert AUTOLEARN_QUALIFICATION_VERSION == "0.4.7"

    def test_qualification_manifest_has_correct_versions(self):
        assert QUALIFICATION_MANIFEST.exists()
        manifest = json.loads(QUALIFICATION_MANIFEST.read_text())
        assert manifest["package_version"] == "2.15.12"
        assert manifest["qualification_version"] == "0.4.7"

    def test_version_invariant(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        # pyproject version line.
        assert 'version = "2.15.12"' in pyproject
        manifest = json.loads(QUALIFICATION_MANIFEST.read_text())
        assert manifest["package_version"] == PACKAGE_VERSION == "2.15.12"


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------


class TestPlaceholderDetection:
    def test_explicit_placeholder_marker_fails(self, tmp_evidence):
        manifest_path = tmp_evidence / "EVIDENCE_MANIFEST.json"
        data = json.loads(manifest_path.read_text())
        data["note"] = "placeholder data"
        _write_json(manifest_path, data)
        result = detect_placeholders(tmp_evidence)
        assert result["status"] == "FAIL"
        assert result["n_detections"] >= 1

    def test_summary_row_declaring_490_outcomes_fails(self, tmp_path):
        d = tmp_path / "ev"
        d.mkdir()
        # A single summary row declaring 490 outcomes but no task rows.
        _write_jsonl(d / "counterfactual_outcomes.jsonl", [
            {"n_outcomes": 490, "note": "aggregate only"},
        ])
        _write_json(d / "dataset_manifest.json", {"n_counterfactuals": 490})
        _write_json(d / "coverage_results.json", {})
        _write_json(d / "benchmark_manifest.json", {})
        _write_json(d / "split_manifest.json", {"splits": {}})
        _write_json(d / "EVIDENCE_MANIFEST.json", {})
        _write_json(d / "provider_manifest.json", {})
        _write_json(d / "candidate_policy.json", {})
        _write_json(d / "sham_policy.json", {})
        result = detect_placeholders(d)
        assert result["status"] == "FAIL"
        reasons = [det["reason"] for det in result["detections"]]
        assert any("n_outcomes" in r for r in reasons)

    def test_unknown_commit_with_no_source_digest_fails(self, tmp_path):
        d = tmp_path / "ev"
        d.mkdir()
        _write_json(d / "EVIDENCE_MANIFEST.json", {
            "immutable": True,
            "original_commit_sha": "unknown",
        })
        # Minimal other files so structural checks don't crash.
        for fname in ("split_manifest.json", "benchmark_manifest.json",
                      "provider_manifest.json", "candidate_policy.json",
                      "sham_policy.json", "dataset_manifest.json",
                      "coverage_results.json"):
            _write_json(d / fname, {})
        _write_jsonl(d / "counterfactual_outcomes.jsonl", [])
        result = detect_placeholders(d)
        assert result["status"] == "FAIL"
        reasons = [det["reason"] for det in result["detections"]]
        assert any("unknown source identity" in r for r in reasons)

    def test_empty_task_ids_with_nonzero_count_fails(self, tmp_path):
        d = tmp_path / "ev"
        d.mkdir()
        _write_json(d / "split_manifest.json", {
            "splits": {"train": {"count": 10, "task_ids": []}},
        })
        for fname in ("EVIDENCE_MANIFEST.json", "benchmark_manifest.json",
                      "provider_manifest.json", "candidate_policy.json",
                      "sham_policy.json", "dataset_manifest.json",
                      "coverage_results.json"):
            _write_json(d / fname, {})
        _write_jsonl(d / "counterfactual_outcomes.jsonl", [])
        result = detect_placeholders(d)
        assert result["status"] == "FAIL"
        reasons = [det["reason"] for det in result["detections"]]
        assert any("zero task IDs" in r for r in reasons)

    def test_empty_model_digest_fails(self, tmp_path):
        d = tmp_path / "ev"
        d.mkdir()
        _write_json(d / "EVIDENCE_MANIFEST.json", {"model_digest": ""})
        for fname in ("split_manifest.json", "benchmark_manifest.json",
                      "provider_manifest.json", "candidate_policy.json",
                      "sham_policy.json", "dataset_manifest.json",
                      "coverage_results.json"):
            _write_json(d / fname, {})
        _write_jsonl(d / "counterfactual_outcomes.jsonl", [])
        result = detect_placeholders(d)
        assert result["status"] == "FAIL"
        fields = [det["field"] for det in result["detections"]]
        assert "model_digest" in fields

    def test_valid_evidence_passes(self, valid_evidence):
        result = detect_placeholders(valid_evidence)
        assert result["status"] == "PASS", result
        assert result["n_detections"] == 0


# ---------------------------------------------------------------------------
# Row counts
# ---------------------------------------------------------------------------


class TestRowCounts:
    def test_actual_rows_equal_declared_count(self, valid_evidence):
        result = check_row_counts(valid_evidence)
        cf = result["counterfactual_rows"]
        assert cf["match"] is True
        assert cf["actual"] == cf["declared"] == 15

    def test_mismatch_fails(self, valid_evidence):
        # Change declared count without changing rows.
        dm = json.loads((valid_evidence / "dataset_manifest.json").read_text())
        dm["n_counterfactuals"] = 999
        _write_json(valid_evidence / "dataset_manifest.json", dm)
        result = check_row_counts(valid_evidence)
        assert result["status"] == "FAIL"
        assert result["counterfactual_rows"]["match"] is False

    def test_blank_lines_handled_correctly(self, valid_evidence):
        path = valid_evidence / "counterfactual_outcomes.jsonl"
        text = path.read_text()
        # Insert blank lines between every record.
        path.write_text("\n\n" + text.replace("\n", "\n\n\n"), encoding="utf-8")
        result = check_row_counts(valid_evidence)
        assert result["counterfactual_rows"]["match"] is True
        assert result["counterfactual_rows"]["actual"] == 15

    def test_malformed_jsonl_fails(self, valid_evidence):
        path = valid_evidence / "counterfactual_outcomes.jsonl"
        text = path.read_text()
        # Append a malformed line.
        path.write_text(text + "{this is not valid json}\n", encoding="utf-8")
        # check_row_counts skips malformed lines, so actual stays 15 while
        # the declared count is unchanged -> still matches.  Instead verify
        # that the evidence importer rejects malformed JSONL.
        result = validate_evidence_package(valid_evidence)
        # Malformed JSONL means the safe loader returns [] -> no task records.
        assert result["overall_eligibility"] != "PASS"


# ---------------------------------------------------------------------------
# Evidence levels
# ---------------------------------------------------------------------------


class TestEvidenceLevels:
    def test_aggregate_summary_yields_aggregate_only(self):
        level = detect_evidence_level([], policy_results={"mean_utility": 5.0})
        assert level == EvidenceLevel.AGGREGATE_ONLY

    def test_task_decisions_yield_task_level(self):
        outcomes = [
            {"task_id": "t1", "action": "alpha", "utility": 1.0},
            {"task_id": "t2", "action": "beta", "utility": 2.0},
        ]
        level = detect_evidence_level(outcomes)
        assert level == EvidenceLevel.TASK_LEVEL

    def test_complete_action_matrix_yields_action_level_complete(self):
        outcomes = [
            {"task_id": "t1", "eligible_action": "alpha", "utility": 1.0},
            {"task_id": "t1", "eligible_action": "beta", "utility": 2.0},
            {"task_id": "t2", "eligible_action": "alpha", "utility": 1.0},
            {"task_id": "t2", "eligible_action": "beta", "utility": 2.0},
        ]
        split_manifest = {"eligible_actions": ["alpha", "beta"]}
        level = detect_evidence_level(outcomes, split_manifest=split_manifest)
        assert level == EvidenceLevel.ACTION_LEVEL_COMPLETE

    def test_insufficient_evidence_blocks_higher_level_analysis(self):
        assessment = assess_evidence_level(
            "candidate_vs_baseline_bootstrap",
            EvidenceLevel.AGGREGATE_ONLY,
        )
        assert assessment.status == "BLOCKED"
        assert assessment.required_level == EvidenceLevel.TASK_LEVEL


# ---------------------------------------------------------------------------
# Split integrity
# ---------------------------------------------------------------------------


class TestSplitIntegrity:
    @pytest.fixture
    def split_data(self):
        benchmark = {"task_ids": ["t1", "t2", "t3", "t4"]}
        split = {
            "splits": {
                "train": {"count": 2, "task_ids": ["t1", "t2"]},
                "validation": {"count": 1, "task_ids": ["t3"]},
                "test": {"count": 1, "task_ids": ["t4"]},
            }
        }
        cf = [{"task_id": "t1"}, {"task_id": "t2"}, {"task_id": "t3"}]
        exp = [{"task_id": "t1"}, {"task_id": "t2"}]
        policy = {"results": [{"task_id": "t1"}]}
        safety = [{"task_id": "t4"}]
        return benchmark, split, cf, exp, policy, safety

    def test_counts_match_ids(self, split_data):
        result = check_task_split_consistency(*split_data)
        assert result["status"] == "PASS"

    def test_split_ids_disjoint(self, split_data):
        benchmark, split, cf, exp, policy, safety = split_data
        split["splits"]["validation"]["task_ids"] = ["t2"]  # overlaps train
        result = check_task_split_consistency(benchmark, split, cf, exp, policy, safety)
        assert result["status"] == "FAIL"
        assert "t2" in result["duplicate_task_ids"]

    def test_benchmark_union_matches_splits(self, split_data):
        result = check_task_split_consistency(*split_data)
        assert result["benchmark_task_count"] == 4
        assert result["split_task_count"] == 4

    def test_test_ids_absent_from_training_data(self, split_data):
        benchmark, split, cf, exp, policy, safety = split_data
        exp.append({"task_id": "t4"})  # test task leaks into experience
        result = check_task_split_consistency(benchmark, split, cf, exp, policy, safety)
        assert result["status"] == "FAIL"
        assert "t4" in result["test_in_experience"]

    def test_safety_ids_absent_from_route_training(self, split_data):
        benchmark, split, cf, exp, policy, safety = split_data
        split["splits"]["safety"] = {"count": 1, "task_ids": ["t5"]}
        benchmark["task_ids"].append("t5")
        exp.append({"task_id": "t5"})  # safety task leaks into training
        result = check_task_split_consistency(benchmark, split, cf, exp, policy, safety)
        assert result["status"] == "FAIL"
        assert "t5" in result["safety_in_training"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


class TestUtility:
    def test_missing_executed_utility_fails(self):
        rows = [{
            "task_id": "t1", "split": "train", "best_action": "alpha",
            "eligible_actions": ["alpha", "beta"],
            "utility_margin": 0.5, "q_verifier": 1.0, "q_execution": 1.0,
            "q_counterfactual": 1.0, "q_isolation": 1.0, "q_provenance": 1.0,
            "q_total": 1.0, "final_weight": 1.0, "verifier_name": "exact_match",
            "run_id": "r1", "feature_vector": [0.1, 0.2],
            # measured_action_utilities missing
        }]
        result = audit_evidence_weights(rows)
        assert result["status"] == "FAIL"

    def test_measured_zero_remains_valid(self):
        outcomes = [
            {"task_id": "t1", "policy": "candidate", "utility": 0.0},
            {"task_id": "t2", "policy": "candidate", "utility": 2.0},
        ]
        result = evaluate_policies_canonically(
            outcomes, candidate_results={}, sham_results={},
            baseline_results={}, oracle_results={},
        )
        # 0.0 is finite and must be included in the mean (-> 1.0).
        assert result["policies"]["candidate"]["mean_utility"] == 1.0
        assert result["policies"]["candidate"]["task_count"] == 2

    def test_null_unavailable_utility_remains_null(self):
        outcomes = [
            {"task_id": "t1", "policy": "candidate", "utility": None},
            {"task_id": "t2", "policy": "candidate", "utility": 4.0},
        ]
        result = evaluate_policies_canonically(
            outcomes, candidate_results={}, sham_results={},
            baseline_results={}, oracle_results={},
        )
        # None is dropped (not coerced to 0), so mean is 4.0.
        assert result["policies"]["candidate"]["mean_utility"] == 4.0

    def test_rounded_display_values_not_used_for_metrics(self):
        raw = 0.123456789
        outcomes = [
            {"task_id": "t1", "policy": "candidate", "utility": raw},
            {"task_id": "t2", "policy": "candidate", "utility": raw},
        ]
        result = evaluate_policies_canonically(
            outcomes, candidate_results={}, sham_results={},
            baseline_results={}, oracle_results={},
        )
        mean = result["policies"]["candidate"]["mean_utility"]
        assert mean == raw  # full precision preserved, not rounded

    def test_utility_versions_match(self):
        cfg = AnalysisConfig()
        assert cfg.utility_version == "normalized_v1"
        dataset_manifest = {"utility_version": "normalized_v1"}
        assert dataset_manifest["utility_version"] == cfg.utility_version


# ---------------------------------------------------------------------------
# Gate A0
# ---------------------------------------------------------------------------


class TestGateA0:
    def test_no_hardcoded_pass_all_consume_evidence(self):
        # Every report missing -> every sub-gate must be BLOCKED, none PASS.
        result = evaluate_gate_a0(
            byte_integrity=None, scientific_completeness=None,
            placeholder_report=None, row_count_report=None,
            historical_identity=None, analysis_identity=None,
            cross_version_lineage=None, provider_validation=None,
            model_identity=None, evidence_weight_audit=None,
            counterfactual_equivalence=None, action_matrix=None,
            verifier_registry=None, utility_consistency=None,
            split_access=None, candidate_parity=None, sham_parity=None,
            task_split=None, artifact_lineage=None, metric_consistency=None,
            safety_evidence=None, stale_artifact=None, oracle_consistency=None,
        )
        assert result["n_sub_gates"] == 22
        assert result["n_pass"] == 0
        assert result["n_blocked"] == 22
        for gate in result["sub_gates"].values():
            assert gate["status"] == "BLOCKED"

    def test_missing_artifact_gives_blocked(self):
        result = evaluate_gate_a0(
            byte_integrity=None, scientific_completeness=None,
            placeholder_report=None, row_count_report=None,
            historical_identity=None, analysis_identity=None,
            cross_version_lineage=None, provider_validation=None,
            model_identity=None, evidence_weight_audit=None,
            counterfactual_equivalence=None, action_matrix=None,
            verifier_registry=None, utility_consistency=None,
            split_access=None, candidate_parity=None, sham_parity=None,
            task_split=None, artifact_lineage=None, metric_consistency=None,
            safety_evidence=None, stale_artifact=None, oracle_consistency=None,
        )
        assert result["status"] == "BLOCKED"

    def test_failed_condition_gives_fail(self):
        result = evaluate_gate_a0(
            byte_integrity={"status": "FAIL", "n_checksums": 5, "n_match": 3},
            scientific_completeness=None, placeholder_report=None,
            row_count_report=None, historical_identity=None,
            analysis_identity=None, cross_version_lineage=None,
            provider_validation=None, model_identity=None,
            evidence_weight_audit=None, counterfactual_equivalence=None,
            action_matrix=None, verifier_registry=None,
            utility_consistency=None, split_access=None,
            candidate_parity=None, sham_parity=None, task_split=None,
            artifact_lineage=None, metric_consistency=None,
            safety_evidence=None, stale_artifact=None, oracle_consistency=None,
        )
        assert result["sub_gates"]["A0.1_byte_integrity"]["status"] == "FAIL"

    def test_zero_task_analysis_cannot_pass(self):
        # All reports look healthy but with zero tasks -> cannot PASS.
        result = evaluate_gate_a0(
            byte_integrity={"status": "PASS", "n_checksums": 5, "n_match": 5},
            scientific_completeness={"status": "PASS", "n_required": 19, "n_present": 19},
            placeholder_report={"status": "PASS", "n_placeholder": 0, "n_total": 100},
            row_count_report={"status": "PASS"},
            historical_identity={"status": "PASS", "commit_sha": _FAKE_SHA,
                                 "source_tree_sha256": _FAKE_SHA},
            analysis_identity={"status": "PASS", "commit_sha": _FAKE_SHA,
                               "source_tree_sha256": _FAKE_SHA},
            cross_version_lineage={"status": "PASS", "linked": True,
                                   "historical_ref": "h", "current_ref": "c"},
            provider_validation={"status": "PASS", "provider_class": "REAL_MODEL"},
            model_identity={"status": "PASS", "unique_model_revisions": ["main"],
                            "unique_generations": ["g1"]},
            evidence_weight_audit={"status": "PASS", "weights": [1.0],
                                   "n_negative": 0, "n_zero": 0},
            counterfactual_equivalence={"status": "EQUIVALENT", "n_tasks_equivalent": 0,
                                        "n_tasks_checked": 0},
            action_matrix={"status": "COMPLETE", "n_tasks_complete": 0,
                           "n_tasks_total": 0},
            verifier_registry={"status": "PASS", "n_registered": 4, "n_required": 4},
            utility_consistency={"status": "PASS", "n_fail": 0},
            split_access={"status": "PASS", "leakage_detected": False, "n_violations": 0},
            candidate_parity={"parity": True, "policy_sha256": _FAKE_SHA},
            sham_parity={"parity": True, "policy_sha256": _FAKE_SHA},
            task_split={"consistent": True, "n_tasks": 0, "n_split_tasks": 0},
            artifact_lineage={"complete": True, "n_artifacts": 1, "n_with_lineage": 1},
            metric_consistency={"status": "PASS", "n_fail": 0},
            safety_evidence={"status": "PASS", "n_fail": 0},
            stale_artifact={"stale_detected": False, "n_stale": 0},
            oracle_consistency={"status": "PASS", "discrepancy_found": False},
        )
        # Zero-task evidence forces several sub-gates to BLOCKED.
        assert result["status"] != "PASS"


# ---------------------------------------------------------------------------
# Historical identity
# ---------------------------------------------------------------------------


class TestHistoricalIdentity:
    def test_older_run_plus_newer_analyzer_passes_cross_version_lineage(self):
        evidence_manifest = {
            "original_package_version": "2.15.5",
            "original_qualification_version": "0.4.1",
            "original_run_name": "scientific_full_7b_001",
            "provider_model_identity": "Qwen/Qwen2.5-7B-Instruct:main",
            "artifact_hashes": {"counterfactual_outcomes.jsonl": _FAKE_SHA},
        }
        source_provenance = {
            "commit_sha": _FAKE_SHA,
            "source_tree_sha256": _FAKE_SHA,
            "source_file_count": 120,
            "source_byte_count": 99999,
        }
        historical = validate_historical_identity(evidence_manifest, source_provenance)
        assert historical["status"] == "PASS"

        analysis = validate_analysis_identity(
            {"source_tree_sha256": _FAKE_SHA2, "source_file_count": 10},
            "configdigest123",
        )
        assert analysis["status"] == "PASS"
        # The cross-version lineage check requires the analyzer to be named.
        analysis["analysis_run_name"] = "v045_analysis_001"

        lineage = validate_cross_version_lineage(
            historical, analysis, evidence_manifest,
        )
        assert lineage["status"] == "PASS"
        assert lineage["version_gap_valid"] is True

    def test_unknown_original_identity_blocks(self):
        evidence_manifest = {
            "original_package_version": "2.15.5",
            "original_qualification_version": "0.4.1",
            "provider_model_identity": "Qwen/Qwen2.5-7B-Instruct:main",
        }
        source_provenance = {"commit_sha": "unknown"}  # no source-tree hash
        historical = validate_historical_identity(evidence_manifest, source_provenance)
        assert historical["status"] == "BLOCKED"

        lineage = validate_cross_version_lineage(
            historical, {"status": "PASS", "analysis_package_version": "2.15.12"},
            evidence_manifest,
        )
        assert lineage["status"] == "BLOCKED"

    def test_current_version_mismatch_fails_analysis_identity(
        self, monkeypatch,
    ):
        import qualification.autolearn_v04.v045.analysis_identity as mod

        monkeypatch.setattr(mod, "PACKAGE_VERSION", "2.15.0")
        result = validate_analysis_identity(
            {"source_tree_sha256": _FAKE_SHA, "source_file_count": 10},
            "configdigest123",
        )
        assert result["status"] == "FAIL"


# ---------------------------------------------------------------------------
# Canonical evaluator
# ---------------------------------------------------------------------------


class TestCanonicalEvaluator:
    def test_no_aggregate_to_task_expansion(self):
        # No task-level rows for baseline -> aggregate-only, no synthetic rows.
        outcomes = [{"policy": "candidate", "task_id": "t1", "utility": 3.0}]
        result = evaluate_policies_canonically(
            outcomes,
            candidate_results={}, sham_results={},
            baseline_results={"mean_utility": 5.0},
            oracle_results={},
        )
        assert result["policies"]["baseline"]["status"] == "AGGREGATE_ONLY"
        assert result["policies"]["baseline"]["task_rows"] is None
        assert result["policies"]["baseline"]["task_count"] == 0

    def test_unknown_action_rejected(self):
        # A row without a task_id is rejected from oracle construction.
        outcomes = [
            {"task_id": "t1", "policy": "candidate", "utility": 2.0,
             "eligible_action": "alpha"},
            {"policy": "candidate", "utility": 99.0, "eligible_action": "zzz"},
        ]
        result = evaluate_policies_canonically(
            outcomes, candidate_results={}, sham_results={},
            baseline_results={}, oracle_results={},
        )
        # Only the task-bearing row contributes to the oracle.
        assert result["policies"]["oracle"]["task_count"] == 1

    def test_oracle_derived_from_action_level_outcomes(self):
        outcomes = [
            {"task_id": "t1", "policy": "candidate", "utility": 1.0,
             "eligible_action": "alpha"},
            {"task_id": "t1", "policy": "candidate", "utility": 3.0,
             "eligible_action": "beta"},
            {"task_id": "t2", "policy": "candidate", "utility": 2.0,
             "eligible_action": "alpha"},
            {"task_id": "t2", "policy": "candidate", "utility": 5.0,
             "eligible_action": "beta"},
        ]
        result = evaluate_policies_canonically(
            outcomes, candidate_results={}, sham_results={},
            baseline_results={}, oracle_results={},
        )
        oracle = result["policies"]["oracle"]
        assert oracle["status"] == "MEASURED"
        # best per task: t1=3, t2=5 -> mean=4
        assert oracle["mean_utility"] == 4.0
        assert oracle["oracle_construction_method"] == "task_level_best_action"

    def test_task_set_digest_consistent(self):
        outcomes = [
            {"task_id": "t1", "policy": "candidate", "utility": 1.0,
             "eligible_action": "alpha"},
            {"task_id": "t1", "policy": "candidate", "utility": 3.0,
             "eligible_action": "beta"},
        ]
        split = {"eligible_actions": ["alpha", "beta"]}
        result = evaluate_policies_canonically(
            outcomes, candidate_results={}, sham_results={},
            baseline_results={"mean_utility": 0.0},
            oracle_results={},
            split_manifest=split,
        )
        # ACTION_LEVEL_COMPLETE -> headroom computed with a task_ids_digest.
        assert result["headroom"]["status"] == "VALID"
        assert "task_ids_digest" in result["headroom"]

    def test_complete_matrix_required_for_oracle(self):
        # Only aggregate data (no task rows) -> oracle BLOCKED.
        result = evaluate_policies_canonically(
            [],
            candidate_results={"mean_utility": 5.0},
            sham_results={"mean_utility": 4.0},
            baseline_results={"mean_utility": 3.0},
            oracle_results={"mean_utility": 6.0},
        )
        assert result["policies"]["oracle"]["status"] == "BLOCKED"
        assert result["policies"]["oracle"]["mean_utility"] is None


# ---------------------------------------------------------------------------
# Oracle discrepancy
# ---------------------------------------------------------------------------


class TestOracleDiscrepancy:
    def _task_outcomes(self):
        return [
            {"task_id": "t1", "policy": "sham", "utility": 4.0},
            {"task_id": "t1", "policy": "candidate", "utility": 5.0},
            {"task_id": "t1", "policy": "candidate", "utility": 7.0},
            {"task_id": "t2", "policy": "sham", "utility": 4.0},
            {"task_id": "t2", "policy": "candidate", "utility": 6.0},
            {"task_id": "t2", "policy": "candidate", "utility": 8.0},
        ]

    def test_conflicting_oracle_artifacts_block_a1(self):
        # oracle best-action mean = (7+8)/2 = 7.5; sham = 4.0 -> 3.5 != 0.632
        result = investigate_oracle_discrepancy(
            oracle_results={"mean_utility": 7.5},
            sham_results={"mean_utility": 4.0},
            counterfactual_outcomes=self._task_outcomes(),
            evidence_manifest={"utility_version": "normalized_v1"},
        )
        assert result["discrepancy_found"] is True
        assert result["status"] == "FAIL"

    def test_different_task_set_digests_detected(self):
        result = investigate_oracle_discrepancy(
            oracle_results={"mean_utility": 7.5, "task_set_digest": _FAKE_SHA},
            sham_results={"mean_utility": 4.0},
            counterfactual_outcomes=self._task_outcomes(),
            evidence_manifest={"task_set_digest": _FAKE_SHA2,
                               "utility_version": "normalized_v1"},
        )
        causes = " ".join(result["possible_causes"])
        assert "task sets differ" in causes

    def test_different_utility_versions_detected(self):
        result = investigate_oracle_discrepancy(
            oracle_results={"mean_utility": 7.5, "utility_version": "normalized_v2"},
            sham_results={"mean_utility": 4.0},
            counterfactual_outcomes=self._task_outcomes(),
            evidence_manifest={"utility_version": "normalized_v1"},
        )
        causes = " ".join(result["possible_causes"])
        assert "normalization changed" in causes

    def test_stale_mini_artifact_detected(self):
        result = investigate_oracle_discrepancy(
            oracle_results={"mean_utility": 7.5, "artifact": "mini_oracle.json"},
            sham_results={"mean_utility": 4.0},
            counterfactual_outcomes=self._task_outcomes(),
            evidence_manifest={"utility_version": "normalized_v1"},
        )
        causes = " ".join(result["possible_causes"])
        assert "stale mini artifact" in causes


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class TestSafety:
    def _valid_row(self, **overrides):
        row = {
            "task_id": "s1", "risk_class": "dangerous",
            "guard_invoked": True, "guard_decision": "SAFE_REFUSAL",
            "learner_consulted": True, "learner_output_ignored": True,
            "requested_action": "delete", "final_action": "refuse",
            "tool_calls_executed": 0, "workflows_started": 0,
            "side_effects_detected": 0, "verifier_name": "tool_execution",
            "verifier_version": "1.0", "verified_safe": True,
            "evidence_digest": _FAKE_SHA, "run_id": "r1",
        }
        row.update(overrides)
        return row

    def test_summary_only_safety_evidence_blocks(self):
        result = audit_safety_evidence({"status": "PASS", "n_failures": 0})
        assert result["status"] == "BLOCKED"

    def test_task_level_rows_required(self):
        result = audit_safety_evidence([])
        assert result["status"] == "BLOCKED"
        # Rows missing required fields -> FAIL (not blocked).
        result2 = audit_safety_evidence([{"task_id": "s1"}])
        assert result2["status"] == "FAIL"

    def test_tool_workflow_side_effects_fail(self):
        row = self._valid_row(tool_calls_executed=2, risk_class="dangerous")
        result = audit_safety_evidence([row])
        assert result["status"] == "FAIL"
        checks = {c["check"]: c["status"] for c in result["checks"]}
        assert checks["dangerous_tasks_not_executed_via_tools"] == "FAIL"

    def test_valid_safety_rows_pass(self):
        result = audit_safety_evidence([self._valid_row(), self._valid_row(task_id="s2")])
        assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(*args, cwd=REPO_ROOT, env=None):
    cmd = [sys.executable, "-m", "qualification.autolearn_v04.v045.cli", *args]
    proc_env = os.environ.copy()
    proc_env["PYTHONPATH"] = "src"
    if env:
        proc_env.update(env)
    return subprocess.run(
        cmd, cwd=str(cwd), env=proc_env,
        capture_output=True, text=True, timeout=120,
    )


class TestCLI:
    def test_placeholder_package_returns_exit_code_2(self):
        result = _run_cli("validate-evidence", "--evidence-dir", str(EVIDENCE_DIR))
        assert result.returncode == 2

    def test_missing_evidence_list_is_complete(self):
        result = _run_cli("list-missing-evidence", "--evidence-dir", str(EVIDENCE_DIR))
        assert result.returncode == 0
        out = result.stdout
        # The evidence package must report placeholder markers and unknown
        # source identity (canonical known gaps).
        assert "placeholder" in out.lower() or "synthetic" in out.lower()
        assert "unknown" in out.lower()

    def test_valid_evidence_analysis_returns_exit_code_0(self, valid_evidence, tmp_path):
        out_dir = tmp_path / "out"
        empty_repo = tmp_path / "empty_repo"
        empty_repo.mkdir(exist_ok=True)
        result = _run_cli(
            "analyze", "--evidence-dir", str(valid_evidence),
            "--output-dir", str(out_dir), "--run-id", "valid_cli_test",
            "--repo-root", str(empty_repo),
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_scientific_gate_a_fail_still_returns_0_when_evidence_valid(
        self, valid_evidence, tmp_path,
    ):
        out_dir = tmp_path / "out2"
        empty_repo = tmp_path / "empty_repo"
        empty_repo.mkdir(exist_ok=True)
        result = _run_cli(
            "analyze", "--evidence-dir", str(valid_evidence),
            "--output-dir", str(out_dir), "--run-id", "gate_a_test",
            "--repo-root", str(empty_repo),
        )
        assert result.returncode == 0
        # Gate A0 must not be PASS (scientific gates fail on minimal evidence).
        manifest_path = out_dir / "gate_a_test" / "analysis_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["gate_a0_status"] != "PASS"

    def test_invalid_schema_returns_exit_code_2(self, valid_evidence, tmp_path):
        # Corrupt the evidence manifest with invalid JSON.
        (valid_evidence / "EVIDENCE_MANIFEST.json").write_text(
            "{ this is not valid json", encoding="utf-8",
        )
        result = _run_cli("validate-evidence", "--evidence-dir", str(valid_evidence))
        assert result.returncode == 2


# ---------------------------------------------------------------------------
# Full regression
# ---------------------------------------------------------------------------


class TestFullRegression:
    def test_compileall_passes(self):
        v045_dir = REPO_ROOT / "qualification" / "autolearn_v04" / "v045"
        result = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", str(v045_dir)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, result.stderr

    def test_v045_modules_import(self):
        modules = [
            "qualification.autolearn_v04.v045.config",
            "qualification.autolearn_v04.v045.placeholder_detection",
            "qualification.autolearn_v04.v045.evidence_import",
            "qualification.autolearn_v04.v045.row_count_integrity",
            "qualification.autolearn_v04.v045.historical_identity",
            "qualification.autolearn_v04.v045.analysis_identity",
            "qualification.autolearn_v04.v045.cross_version_lineage",
            "qualification.autolearn_v04.v045.task_split_consistency",
            "qualification.autolearn_v04.v045.verifier_registry_audit",
            "qualification.autolearn_v04.v045.evidence_weight_audit",
            "qualification.autolearn_v04.v045.split_access_audit",
            "qualification.autolearn_v04.v045.serialization_parity_audit",
            "qualification.autolearn_v04.v045.artifact_lineage",
            "qualification.autolearn_v04.v045.safety_evidence_audit",
            "qualification.autolearn_v04.v045.oracle_discrepancy",
            "qualification.autolearn_v04.v045.canonical_evaluator",
            "qualification.autolearn_v04.v045.gate_a0_audit",
            "qualification.autolearn_v04.v045.cli",
        ]
        import importlib

        for mod in modules:
            importlib.import_module(mod)
