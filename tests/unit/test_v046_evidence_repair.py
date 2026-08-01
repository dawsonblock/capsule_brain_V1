"""Comprehensive tests for the v0.4.6 release (Capsule Brain 2.15.11 / AutoLearn v0.4.7).

Covers all areas from the v0.4.6 evidence-repair specification:
- Version identity
- Evidence origin
- Anti-misclassification
- Fixture validation
- Cross-origin duplication
- Counterfactual equivalence
- Evidence weights
- Verifier registry
- Status propagation
- Historical identity
- Serialization parity
- Split access
- Artifact lineage
- Oracle discrepancy
- Scale decision
- Unavailable evidence
- Modal recovery
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
from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.common.audit_result import (
    AuditResult,
    make_pass,
    make_fail,
    make_blocked,
    make_not_applicable,
    aggregate_status,
)
from qualification.autolearn_v04.common.evidence_origin import (
    EvidenceOrigin,
    EvidenceOriginResult,
    classify_evidence_origin,
    check_anti_misclassification,
)
from qualification.autolearn_v04.common.evidence_structure import (
    EvidenceStructureLevel,
    EvidenceClassification,
    classify_evidence,
)
from qualification.autolearn_v04.v046.config import (
    AnalysisConfig,
    REQUIRED_EVIDENCE_FILES,
    PLACEHOLDER_MARKERS,
    CANONICAL_ACTIONS,
    GATE_A0_N_SUB_GATES,
)
from qualification.autolearn_v04.v046.evidence_origin_audit import (
    audit_evidence_origin,
)
from qualification.autolearn_v04.v046.fixture_validation import (
    validate_fixture,
)
from qualification.autolearn_v04.v046.cross_origin_duplication import (
    detect_cross_origin_duplicates,
)
from qualification.autolearn_v04.v046.counterfactual_equivalence import (
    validate_counterfactual_equivalence,
    MANDATORY_DIGESTS,
)
from qualification.autolearn_v04.v046.evidence_weight_audit import (
    audit_evidence_weights,
    QUALITY_FIELDS,
    REQUIRED_WEIGHT_FIELDS_MISSING,
)
from qualification.autolearn_v04.v046.verifier_registry_audit import (
    audit_verifier_registry,
)
from qualification.autolearn_v04.v046.serialization_parity import (
    compute_serialization_parity,
)
from qualification.autolearn_v04.v046.split_access_audit import (
    audit_split_access,
)
from qualification.autolearn_v04.v046.artifact_lineage import (
    validate_artifact_lineage,
)
from qualification.autolearn_v04.v046.oracle_discrepancy import (
    investigate_oracle_discrepancy,
)
from qualification.autolearn_v04.v046.scale_decision import (
    make_scale_decision,
    DECISION_NOT_APPLICABLE,
    DECISION_BLOCKED,
    DECISION_APPROVE_MATCHED_14B_PILOT,
    DECISION_APPROVE_FULL_14B_RUN,
    DECISION_NO_SCALE_NEEDED_FOR_GATE_A,
)
from qualification.autolearn_v04.v046.modal_evidence_recovery import (
    recover_modal_evidence,
    REQUIRED_MODAL_FILES,
)
from qualification.autolearn_v04.v046.gate_a0_audit import (
    evaluate_gate_a0,
    SUB_GATE_NAMES,
)
from qualification.autolearn_v04.v046.scientific_evidence_validation import (
    validate_scientific_evidence,
)
from qualification.autolearn_v04.v046.experience_normalizer import (
    normalize_experience_rows,
)
from qualification.autolearn_v04.v045.historical_identity import (
    validate_historical_identity,
)
from qualification.autolearn_v04.v045.analysis_identity import (
    validate_analysis_identity,
)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "qualification" / "evidence" / "fixtures" / "synthetic_7b_routing"
QUALIFICATION_MANIFEST = REPO_ROOT / "qualification" / "QUALIFICATION_MANIFEST.json"
UNAVAILABLE_DIR = REPO_ROOT / "qualification" / "evidence" / "unavailable" / "historical_modal_7b"

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


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _make_cf_row(
    task_id="task_000",
    action="ANSWER_DIRECT",
    split="experience",
    **overrides,
) -> dict:
    """Build a counterfactual row with all mandatory digests filled."""
    base = {
        "task_id": task_id,
        "split": split,
        "eligible_action": action,
        "executed_action": action,
        "action_status": "EXECUTED",
        "prompt_digest": _FAKE_SHA,
        "setup_digest": _FAKE_SHA,
        "hidden_setup_digest": _FAKE_SHA,
        "environment_snapshot_digest": _FAKE_SHA,
        "memory_state_digest": _FAKE_SHA,
        "tool_state_digest": _FAKE_SHA,
        "workflow_state_digest": _FAKE_SHA,
        "capability_permissions_digest": _FAKE_SHA,
        "timeout_config_digest": _FAKE_SHA,
        "generation_config_digest": _FAKE_SHA,
        "utility_config_digest": _FAKE_SHA,
        "model_revision": "1.0",
        "tokenizer_revision": "1.0",
        "verifier_version": "1.0",
        "verifier_name": "exact_match",
        "utility": 1.0,
        "utility_version": "normalized_v1",
    }
    base.update(overrides)
    return base


def _make_exp_task_row(task_id="task_000", **overrides) -> dict:
    """Build a normalized task-level experience row with all quality fields."""
    base = {
        "task_id": task_id,
        "split": "experience",
        "best_action": "ANSWER_DIRECT",
        "eligible_actions": ["ANSWER_DIRECT", "RETRIEVE_MEMORY"],
        "q_verifier": 1.0,
        "q_execution": 1.0,
        "q_counterfactual": 1.0,
        "q_isolation": 1.0,
        "q_provenance": 1.0,
        "q_total": 1.0,
        "final_weight": 1.0,
        "verifier_name": "exact_match",
        "verifier_names": ["exact_match"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evidence_dir() -> str:
    return str(EVIDENCE_DIR)


@pytest.fixture
def tmp_evidence(evidence_dir, tmp_path) -> Path:
    """Copy the canonical fixture evidence package for mutation tests."""
    dst = tmp_path / "evidence"
    shutil.copytree(evidence_dir, dst)
    return dst


@pytest.fixture
def cf_rows():
    """Load counterfactual rows from the fixture."""
    return _load_jsonl(EVIDENCE_DIR / "counterfactual_outcomes.jsonl")


@pytest.fixture
def exp_rows():
    """Load experience rows from the fixture."""
    return _load_jsonl(EVIDENCE_DIR / "executive_experiences.jsonl")


@pytest.fixture
def split_manifest():
    """Load the split manifest from the fixture."""
    return _load_json(EVIDENCE_DIR / "split_manifest.json")


@pytest.fixture
def evidence_manifest():
    """Load the evidence manifest from the fixture."""
    return _load_json(EVIDENCE_DIR / "EVIDENCE_MANIFEST.json")


@pytest.fixture
def provider_manifest():
    """Load the provider manifest from the fixture."""
    return _load_json(EVIDENCE_DIR / "provider_manifest.json")


@pytest.fixture
def candidate_policy():
    return _load_json(EVIDENCE_DIR / "candidate_policy.json")


@pytest.fixture
def sham_policy():
    return _load_json(EVIDENCE_DIR / "sham_policy.json")


# ---------------------------------------------------------------------------
# Version identity
# ---------------------------------------------------------------------------


class TestVersionIdentity:
    def test_package_version(self):
        assert PACKAGE_VERSION == "2.15.11"

    def test_autolearn_version(self):
        assert AUTOLEARN_VERSION == "0.3.10"

    def test_qualification_version(self):
        assert AUTOLEARN_QUALIFICATION_VERSION == "0.4.7"

    def test_qualification_manifest_has_correct_versions(self):
        assert QUALIFICATION_MANIFEST.exists()
        manifest = json.loads(QUALIFICATION_MANIFEST.read_text())
        assert manifest["package_version"] == "2.15.11"
        assert manifest["qualification_version"] == "0.4.7"

    def test_version_invariant(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        # pyproject version line.
        assert 'version = "2.15.11"' in pyproject
        manifest = json.loads(QUALIFICATION_MANIFEST.read_text())
        assert manifest["package_version"] == PACKAGE_VERSION == "2.15.11"
        assert manifest["qualification_version"] == AUTOLEARN_QUALIFICATION_VERSION == "0.4.7"

    def test_cli_output_shows_correct_version(self):
        cmd = [sys.executable, "-m", "qualification.autolearn_v04.v045.cli", "--version"]
        proc_env = os.environ.copy()
        proc_env["PYTHONPATH"] = "src"
        result = subprocess.run(
            cmd, cwd=str(REPO_ROOT), env=proc_env,
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "2.15.11" in result.stdout
        assert "0.4.7" in result.stdout


# ---------------------------------------------------------------------------
# Evidence origin
# ---------------------------------------------------------------------------


class TestEvidenceOrigin:
    def test_synthetic_model_cannot_be_real_model(self):
        provider = {
            "provider_class": "REAL_MODEL",
            "model_id": "synthetic-7b",
            "tokenizer_id": "real-tokenizer",
            "model_digest": _FAKE_SHA,
        }
        result = classify_evidence_origin(provider, {})
        # The classifier must reclassify this as SYNTHETIC, not REAL_MODEL.
        assert result.origin == EvidenceOrigin.SYNTHETIC
        assert result.scientific_claim_eligible is False
        assert result.promotable is False
        assert result.supports_gate_a is False

    def test_synthetic_tokenizer_cannot_be_real_model(self):
        provider = {
            "provider_class": "REAL_MODEL",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "tokenizer_id": "synthetic-tokenizer",
            "model_digest": _FAKE_SHA,
        }
        result = classify_evidence_origin(provider, {})
        # The classifier must reclassify this as SYNTHETIC, not REAL_MODEL.
        assert result.origin == EvidenceOrigin.SYNTHETIC
        assert result.scientific_claim_eligible is False
        assert result.promotable is False
        assert result.supports_gate_a is False

    def test_synthetic_origin_cannot_support_gate_a(self):
        provider = {
            "provider_class": "SYNTHETIC",
            "model_id": "synthetic-7b",
            "tokenizer_id": "synthetic-tokenizer",
        }
        result = classify_evidence_origin(provider, {"evidence_origin": "SYNTHETIC"})
        assert result.origin == EvidenceOrigin.SYNTHETIC
        assert result.supports_gate_a is False
        assert result.supports_gate_b is False
        assert result.scientific_claim_eligible is False

    def test_real_origin_requires_model_digest_for_scientific_eligibility(self):
        provider = {
            "provider_class": "REAL_MODEL",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_digest": None,
        }
        result = classify_evidence_origin(provider, {})
        assert result.origin == EvidenceOrigin.REAL_MODEL
        assert result.scientific_claim_eligible is False
        assert result.status == "BLOCKED"

    def test_unknown_origin_fails_closed(self):
        provider = {"provider_class": "UNKNOWN_CLASS"}
        result = classify_evidence_origin(provider, {})
        assert result.origin == EvidenceOrigin.UNKNOWN
        assert result.status == "FAIL"
        assert result.scientific_claim_eligible is False
        assert result.promotable is False

    def test_simulated_origin_not_scientifically_eligible(self):
        provider = {
            "provider_class": "SIMULATED",
            "model_id": "sim-model",
            "tokenizer_id": "sim-tokenizer",
        }
        result = classify_evidence_origin(provider, {})
        assert result.origin == EvidenceOrigin.SIMULATED
        assert result.scientific_claim_eligible is False
        assert result.promotable is False

    def test_infrastructure_origin_not_scientifically_eligible(self):
        provider = {
            "provider_class": "INFRASTRUCTURE",
            "model_id": "infra-model",
            "tokenizer_id": "infra-tokenizer",
        }
        result = classify_evidence_origin(provider, {})
        assert result.origin == EvidenceOrigin.INFRASTRUCTURE
        assert result.scientific_claim_eligible is False
        assert result.promotable is False


# ---------------------------------------------------------------------------
# Anti-misclassification
# ---------------------------------------------------------------------------


class TestAntiMisclassification:
    def test_real_model_with_synthetic_model_id_fails(self):
        provider = {
            "provider_class": "REAL_MODEL",
            "model_id": "synthetic-7b",
            "tokenizer_id": "real-tokenizer",
            "model_digest": _FAKE_SHA,
        }
        violations = check_anti_misclassification(provider, {})
        assert any("synthetic" in v for v in violations)

    def test_real_model_with_synthetic_tokenizer_fails(self):
        provider = {
            "provider_class": "REAL_MODEL",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "tokenizer_id": "synthetic-tokenizer",
            "model_digest": _FAKE_SHA,
        }
        violations = check_anti_misclassification(provider, {})
        assert any("tokenizer_id" in v for v in violations)

    def test_runtime_real_with_synthetic_origin_fails(self):
        provider = {
            "provider_class": "REAL_MODEL",
            "runtime_type": "real",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_digest": _FAKE_SHA,
        }
        evidence = {"evidence_origin": "SYNTHETIC"}
        violations = check_anti_misclassification(provider, evidence)
        assert any("runtime_type=real" in v for v in violations)

    def test_scientific_eligible_with_non_real_origin_fails(self):
        provider = {
            "provider_class": "SYNTHETIC",
            "model_id": "synthetic-7b",
            "tokenizer_id": "synthetic-tokenizer",
        }
        evidence = {
            "evidence_origin": "SYNTHETIC",
            "scientific_claim_eligible": True,
        }
        violations = check_anti_misclassification(provider, evidence)
        assert any("scientific_claim_eligible" in v for v in violations)

    def test_supports_gate_a_with_non_real_origin_fails(self):
        provider = {
            "provider_class": "SYNTHETIC",
            "model_id": "synthetic-7b",
            "tokenizer_id": "synthetic-tokenizer",
        }
        evidence = {
            "evidence_origin": "SYNTHETIC",
            "supports_gate_a": True,
        }
        violations = check_anti_misclassification(provider, evidence)
        assert any("supports_gate_a" in v for v in violations)

    def test_promotable_with_non_real_origin_fails(self):
        provider = {
            "provider_class": "SYNTHETIC",
            "model_id": "synthetic-7b",
            "tokenizer_id": "synthetic-tokenizer",
        }
        evidence = {
            "evidence_origin": "SYNTHETIC",
            "promotable": True,
        }
        violations = check_anti_misclassification(provider, evidence)
        assert any("promotable" in v for v in violations)

    def test_real_model_missing_digest_fails(self):
        provider = {
            "provider_class": "REAL_MODEL",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_digest": None,
        }
        violations = check_anti_misclassification(provider, {})
        assert any("model_digest" in v for v in violations)


# ---------------------------------------------------------------------------
# Fixture validation
# ---------------------------------------------------------------------------


class TestFixtureValidation:
    def test_complete_synthetic_fixture_passes_structural_validation(self, evidence_dir):
        result = validate_fixture(evidence_dir)
        assert result["structural_validity"] == "PASS", result["reason"]
        assert result["evidence_origin"] == "SYNTHETIC"

    def test_fixture_scientific_eligibility_is_not_applicable(self, evidence_dir):
        result = validate_fixture(evidence_dir)
        assert result["scientific_claim_eligibility"] == "NOT_APPLICABLE"

    def test_fixture_cannot_be_promoted(self, evidence_dir):
        result = validate_fixture(evidence_dir)
        assert result["promotable"] is False

    def test_fixture_cannot_approve_scaling(self, evidence_dir):
        """The fixture must not produce any scale-approval decision."""
        decision = make_scale_decision(
            gate_a0_status="PASS",
            evidence_origin="SYNTHETIC",
        )
        assert decision["decision"] == DECISION_NOT_APPLICABLE
        assert decision["approve_matched_14b_pilot"] is False
        assert decision["approve_full_14b_run"] is False

    def test_fixture_does_not_fail_on_synthetic_strings(self, evidence_dir):
        """The fixture contains 'synthetic' in many strings but must not
        treat that as a placeholder marker."""
        result = validate_fixture(evidence_dir)
        checks = result["checks"]
        # The no_real_placeholders check must pass despite 'synthetic' strings.
        assert checks.get("no_real_placeholders", {}).get("status") == "PASS"
        assert "synthetic" not in PLACEHOLDER_MARKERS


# ---------------------------------------------------------------------------
# Cross-origin duplication
# ---------------------------------------------------------------------------


class TestCrossOriginDuplication:
    def test_same_hashes_under_different_origins_fail(self, tmp_path):
        root = tmp_path / "evidence_root"
        pkg_a = root / "pkg_a"
        pkg_b = root / "pkg_b"
        pkg_a.mkdir(parents=True)
        pkg_b.mkdir(parents=True)

        shared_digest = _FAKE_SHA
        for pkg, origin in ((pkg_a, "SYNTHETIC"), (pkg_b, "REAL_MODEL")):
            _write_json(pkg / "EVIDENCE_MANIFEST.json", {
                "evidence_origin": origin,
                "original_run_id": f"run_{origin.lower()}",
                "benchmark_sha256": shared_digest,
            })

        result = detect_cross_origin_duplicates(str(root))
        assert result["status"] == "FAIL"
        assert len(result["violations"]) >= 1
        assert result["violations"][0]["severity"] == "HIGH"

    def test_legitimate_distinct_packages_do_not_trigger(self, tmp_path):
        root = tmp_path / "evidence_root"
        pkg_a = root / "pkg_a"
        pkg_b = root / "pkg_b"
        pkg_a.mkdir(parents=True)
        pkg_b.mkdir(parents=True)

        for pkg, origin, digest in (
            (pkg_a, "SYNTHETIC", _FAKE_SHA),
            (pkg_b, "REAL_MODEL", _FAKE_SHA2),
        ):
            _write_json(pkg / "EVIDENCE_MANIFEST.json", {
                "evidence_origin": origin,
                "original_run_id": f"run_{origin.lower()}",
                "benchmark_sha256": digest,
            })

        result = detect_cross_origin_duplicates(str(root))
        assert result["status"] == "PASS"
        assert len(result["violations"]) == 0

    def test_near_identical_rows_trigger_warning(self, tmp_path):
        """Same file-level digest under different origins triggers a violation."""
        root = tmp_path / "evidence_root"
        pkg_a = root / "pkg_a"
        pkg_b = root / "pkg_b"
        pkg_a.mkdir(parents=True)
        pkg_b.mkdir(parents=True)

        shared_content = '{"shared": true}'
        for pkg, origin in ((pkg_a, "SYNTHETIC"), (pkg_b, "REAL_MODEL")):
            _write_json(pkg / "EVIDENCE_MANIFEST.json", {
                "evidence_origin": origin,
                "original_run_id": f"run_{origin.lower()}",
            })
            (pkg / "benchmark_manifest.json").write_text(shared_content, encoding="utf-8")

        result = detect_cross_origin_duplicates(str(root))
        assert result["status"] == "FAIL"
        assert len(result["violations"]) >= 1


# ---------------------------------------------------------------------------
# Counterfactual equivalence
# ---------------------------------------------------------------------------


class TestCounterfactualEquivalence:
    def test_empty_digests_block(self):
        """Empty mandatory digests must BLOCK, not pass."""
        rows = [
            _make_cf_row(task_id="t1", action="alpha", prompt_digest=""),
            _make_cf_row(task_id="t1", action="beta", prompt_digest=""),
        ]
        split_manifest = {"splits": {"experience": {"count": 1, "task_ids": ["t1"]}}}
        result = validate_counterfactual_equivalence(rows, split_manifest)
        assert result["status"] == "BLOCKED"
        assert result["n_unverifiable"] == 1

    def test_matching_nonempty_digests_pass(self):
        rows = [
            _make_cf_row(task_id="t1", action="alpha"),
            _make_cf_row(task_id="t1", action="beta"),
        ]
        split_manifest = {"splits": {"experience": {"count": 1, "task_ids": ["t1"]}}}
        result = validate_counterfactual_equivalence(rows, split_manifest)
        assert result["status"] == "PASS"
        assert result["n_equivalent"] == 1

    def test_mismatched_digest_fails(self):
        rows = [
            _make_cf_row(task_id="t1", action="alpha", prompt_digest=_FAKE_SHA),
            _make_cf_row(task_id="t1", action="beta", prompt_digest=_FAKE_SHA2),
        ]
        split_manifest = {"splits": {"experience": {"count": 1, "task_ids": ["t1"]}}}
        result = validate_counterfactual_equivalence(rows, split_manifest)
        assert result["status"] == "FAIL"
        assert result["n_non_equivalent"] == 1

    def test_action_specific_fields_ignored(self):
        """Action-specific fields like 'utility' or 'eligible_action' must
        not affect equivalence — only mandatory digests matter."""
        rows = [
            _make_cf_row(task_id="t1", action="alpha", utility=1.0, eligible_action="alpha"),
            _make_cf_row(task_id="t1", action="beta", utility=5.0, eligible_action="beta"),
        ]
        split_manifest = {"splits": {"experience": {"count": 1, "task_ids": ["t1"]}}}
        result = validate_counterfactual_equivalence(rows, split_manifest)
        assert result["status"] == "PASS"
        assert result["n_equivalent"] == 1


# ---------------------------------------------------------------------------
# Evidence weights
# ---------------------------------------------------------------------------


class TestEvidenceWeights:
    def test_rows_found_but_fields_missing_reports_correct_count(self):
        """Rows exist but quality fields are missing → FAIL with correct count."""
        rows = [{"task_id": "t1"}, {"task_id": "t2"}, {"task_id": "t3"}]
        result = audit_evidence_weights(rows, [], {})
        assert result["status"] == "FAIL"
        assert result["reason"] == REQUIRED_WEIGHT_FIELDS_MISSING
        assert result["experience_task_rows_found"] == 3

    def test_normalized_task_level_weights_pass(self):
        rows = [
            _make_exp_task_row(task_id="t1"),
            _make_exp_task_row(task_id="t2"),
        ]
        result = audit_evidence_weights(rows, [], {})
        assert result["status"] == "PASS"
        assert result["rows_containing_quality_fields"] == 2
        assert result["positive_final_weights"] == 2

    def test_negative_weights_fail(self):
        rows = [_make_exp_task_row(task_id="t1", final_weight=-0.5)]
        result = audit_evidence_weights(rows, [], {})
        assert result["status"] == "FAIL"
        assert result["negative_weights"] == 1

    def test_nonfinite_weights_fail(self):
        rows = [_make_exp_task_row(task_id="t1", final_weight=float("nan"))]
        result = audit_evidence_weights(rows, [], {})
        assert result["status"] == "FAIL"
        assert result["nonfinite_weights"] == 1

    def test_synthetic_valid_weights_do_not_imply_scientific_eligibility(self):
        """Even with valid weights, synthetic evidence is not scientifically eligible."""
        rows = [_make_exp_task_row(task_id="t1")]
        result = audit_evidence_weights(rows, [], {})
        assert result["status"] == "PASS"
        # But the origin must still be checked separately.
        provider = {
            "provider_class": "SYNTHETIC",
            "model_id": "synthetic-7b",
            "tokenizer_id": "synthetic-tokenizer",
        }
        origin_result = classify_evidence_origin(provider, {"evidence_origin": "SYNTHETIC"})
        assert origin_result.scientific_claim_eligible is False


# ---------------------------------------------------------------------------
# Verifier registry
# ---------------------------------------------------------------------------


class TestVerifierRegistry:
    def test_verifier_rows_are_discovered(self):
        """The audit must enumerate verifiers from outcome rows, not report 0/0."""
        cf = [{"verifier_name": "exact_match", "verifier_version": "1.0"}]
        result = audit_verifier_registry(cf, [], [], {})
        assert result["used_verifier_count"] == 1
        assert result["status"] == "PASS"

    def test_unknown_verifier_fails(self):
        cf = [{"verifier_name": "nonexistent_verifier", "verifier_version": "1.0"}]
        result = audit_verifier_registry(cf, [], [], {})
        assert result["status"] == "FAIL"
        assert "nonexistent_verifier" in result["unknown_verifiers"]

    def test_version_mismatch_fails(self):
        cf = [{"verifier_name": "exact_match", "verifier_version": "2.0"}]
        result = audit_verifier_registry(cf, [], [], {})
        assert result["status"] == "FAIL"
        assert len(result["version_mismatches"]) == 1
        assert result["version_mismatches"][0]["verifier"] == "exact_match"

    def test_class_mismatch_fails(self):
        cf = [{
            "verifier_name": "exact_match",
            "verifier_version": "1.0",
            "verifier_class": "WrongClass",
        }]
        result = audit_verifier_registry(cf, [], [], {})
        assert result["status"] == "FAIL"
        assert len(result["class_mismatches"]) == 1

    def test_registered_exact_match_verifier_passes(self):
        cf = [{"verifier_name": "exact_match", "verifier_version": "1.0"}]
        exp = [{"verifier_name": "exact_match", "verifier_version": "1.0"}]
        result = audit_verifier_registry(cf, exp, [], {})
        assert result["status"] == "PASS"
        assert result["used_verifier_count"] >= 1


# ---------------------------------------------------------------------------
# Status propagation
# ---------------------------------------------------------------------------


class TestStatusPropagation:
    @staticmethod
    def _all_pass_reports():
        """Return kwargs for evaluate_gate_a0 where all sub-gates PASS."""
        return dict(
            byte_integrity={"status": "PASS"},
            scientific_completeness={"status": "PASS"},
            placeholder_report={"status": "PASS"},
            row_count_report={"status": "PASS"},
            historical_identity={
                "status": "PASS",
                "original_commit_sha": _FAKE_SHA,
                "original_source_tree_sha256": _FAKE_SHA,
            },
            analysis_identity={
                "status": "PASS",
                "analysis_source_tree_sha256": _FAKE_SHA,
            },
            cross_version_lineage={
                "status": "PASS",
                "linked": True,
                "historical_ref": "run_001",
                "current_ref": "run_002",
            },
            provider_validation={
                "status": "PASS",
                "provider_class": "REAL_MODEL",
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "tokenizer_id": "Qwen/Qwen2.5-7B-Instruct",
                "model_digest": _FAKE_SHA,
            },
            model_identity={
                "status": "PASS",
                "unique_model_revisions": ["main"],
                "unique_generations": ["gen_001"],
            },
            evidence_weight_audit={
                "status": "PASS",
                "weights": [1.0],
                "n_negative": 0,
                "n_zero": 0,
            },
            counterfactual_equivalence={
                "status": "PASS",
                "ce_status": "EQUIVALENT",
                "n_tasks_equivalent": 5,
                "n_tasks_checked": 5,
            },
            action_matrix={
                "status": "PASS",
                "am_status": "COMPLETE",
                "n_tasks_complete": 5,
                "n_tasks_total": 5,
            },
            verifier_registry={
                "status": "PASS",
                "n_registered": 6,
                "n_required": 6,
            },
            utility_consistency={"status": "PASS", "n_fail": 0},
            split_access={"status": "PASS", "leakage_detected": False, "n_violations": 0},
            candidate_parity={"parity": True, "policy_sha256": _FAKE_SHA},
            sham_parity={"parity": True, "policy_sha256": _FAKE_SHA},
            task_split={"status": "PASS"},
            artifact_lineage={"status": "PASS"},
            metric_consistency={"status": "PASS", "n_fail": 0},
            safety_evidence={"status": "PASS", "n_fail": 0, "checks": []},
            stale_artifact={"stale_detected": False, "n_stale": 0, "reason": "ok"},
            oracle_consistency={"discrepancy_found": False, "status": "PASS", "reason": "ok"},
            cross_origin_duplication={"status": "PASS", "n_duplicates": 0, "duplicates_detected": False},
            evidence_origin={
                "origin": "REAL_MODEL",
                "reason": "real",
                "scientific_claim_eligible": True,
            },
        )

    def test_source_fail_remains_fail(self):
        """A source report with FAIL must propagate as FAIL, not BLOCKED."""
        kwargs = self._all_pass_reports()
        kwargs["provider_validation"] = {
            "status": "FAIL",
            "provider_class": "REAL_MODEL",
            "model_id": "synthetic-7b",
            "tokenizer_id": "real-tokenizer",
            "model_digest": _FAKE_SHA,
        }
        result = evaluate_gate_a0(**kwargs)
        # A0.9 provider_model_authenticity should be FAIL.
        assert result["sub_gates"]["A0.9_provider_model_authenticity"]["status"] == "FAIL"
        assert result["status"] == "FAIL"

    def test_source_blocked_remains_blocked(self):
        kwargs = self._all_pass_reports()
        kwargs["byte_integrity"] = None
        kwargs["scientific_completeness"] = None
        result = evaluate_gate_a0(**kwargs)
        # A0.1 should be BLOCKED because both byte_integrity and scientific_completeness are missing.
        assert result["sub_gates"]["A0.1_structural_completeness"]["status"] == "BLOCKED"
        assert result["status"] == "BLOCKED"

    def test_source_pass_remains_pass(self):
        kwargs = self._all_pass_reports()
        result = evaluate_gate_a0(**kwargs)
        assert result["status"] == "PASS"
        assert result["n_fail"] == 0

    def test_observed_values_preserved(self):
        """The observed field from a source report must be preserved in the sub-gate."""
        kwargs = self._all_pass_reports()
        result = evaluate_gate_a0(**kwargs)
        # The evidence_origin sub-gate should preserve "REAL_MODEL" as observed.
        a0_2 = result["sub_gates"]["A0.2_evidence_origin_authenticity"]
        assert a0_2["observed"] == "REAL_MODEL"

    def test_no_zero_default_corruption(self):
        """When a source report is missing, the sub-gate must be BLOCKED
        (not silently PASS with zero defaults)."""
        kwargs = self._all_pass_reports()
        kwargs["historical_identity"] = None  # missing
        result = evaluate_gate_a0(**kwargs)
        assert result["sub_gates"]["A0.6_historical_identity"]["status"] == "BLOCKED"
        assert result["status"] == "BLOCKED"


# ---------------------------------------------------------------------------
# Historical identity
# ---------------------------------------------------------------------------


class TestHistoricalIdentity:
    def test_source_tree_digest_can_substitute_for_commit(self):
        """A valid source_tree_sha256 can substitute for a missing commit_sha."""
        evidence_manifest = {
            "original_package_version": "2.15.5",
            "original_qualification_version": "0.4.1",
        }
        source_provenance = {
            "source_tree_sha256": _FAKE_SHA,
            "source_file_count": 120,
            "source_byte_count": 99999,
            "provider_model_identity": "Qwen/Qwen2.5-7B-Instruct:main",
        }
        result = validate_historical_identity(evidence_manifest, source_provenance)
        assert result["status"] == "PASS"

    def test_missing_config_hash_blocks(self):
        """Missing config_digest in analysis identity must FAIL."""
        analysis_source_hash = {"source_tree_sha256": _FAKE_SHA}
        result = validate_analysis_identity(analysis_source_hash, "")
        assert result["status"] == "FAIL"
        checks = {c["check"]: c["status"] for c in result["checks"]}
        assert checks["analysis_config_digest_present"] == "FAIL"

    def test_missing_dependency_identity_blocks(self):
        """Missing dependency identity (config_digest) blocks analysis identity."""
        analysis_source_hash = {"source_tree_sha256": _FAKE_SHA}
        result = validate_analysis_identity(analysis_source_hash, "")
        assert result["status"] == "FAIL"

    def test_synthetic_fixture_uses_not_applicable(self, evidence_dir):
        """For synthetic fixture evidence, Gate A0 scientific sub-gates return
        NOT_APPLICABLE rather than FAIL."""
        origin_report = audit_evidence_origin(evidence_dir)
        assert origin_report["origin"] == "SYNTHETIC"
        # When evidence_origin is SYNTHETIC, gate_a0 scientific sub-gates are N/A.
        result = evaluate_gate_a0(
            byte_integrity={"status": "PASS"},
            scientific_completeness={"status": "PASS"},
            placeholder_report={"status": "PASS"},
            row_count_report={"status": "PASS"},
            historical_identity={"status": "PASS"},
            analysis_identity={"status": "PASS"},
            cross_version_lineage={"status": "PASS"},
            provider_validation={"status": "PASS"},
            model_identity={"status": "PASS"},
            evidence_weight_audit={"status": "PASS", "weights": [1.0], "n_negative": 0, "n_zero": 0},
            counterfactual_equivalence={"status": "PASS", "ce_status": "EQUIVALENT", "n_tasks_equivalent": 5, "n_tasks_checked": 5},
            action_matrix={"status": "PASS", "am_status": "COMPLETE", "n_tasks_complete": 5, "n_tasks_total": 5},
            verifier_registry={"status": "PASS", "n_registered": 6, "n_required": 6},
            utility_consistency={"status": "PASS", "n_fail": 0},
            split_access={"status": "PASS", "leakage_detected": False, "n_violations": 0},
            candidate_parity={"parity": True, "policy_sha256": _FAKE_SHA},
            sham_parity={"parity": True, "policy_sha256": _FAKE_SHA},
            task_split={"status": "PASS"},
            artifact_lineage={"status": "PASS"},
            metric_consistency={"status": "PASS", "n_fail": 0},
            safety_evidence={"status": "PASS", "n_fail": 0, "checks": []},
            stale_artifact={"stale_detected": False, "n_stale": 0, "reason": "ok"},
            oracle_consistency={"discrepancy_found": False, "status": "PASS", "reason": "ok"},
            cross_origin_duplication={"status": "PASS", "n_duplicates": 0, "duplicates_detected": False},
            evidence_origin=origin_report,
        )
        # Scientific sub-gates must be NOT_APPLICABLE for synthetic evidence.
        for gate_name in (
            "A0.2_evidence_origin_authenticity",
            "A0.3_scientific_claim_eligibility",
            "A0.6_historical_identity",
            "A0.7_current_analysis_identity",
            "A0.8_cross_version_lineage",
            "A0.9_provider_model_authenticity",
            "A0.10_single_generation_identity",
        ):
            assert result["sub_gates"][gate_name]["status"] == "NOT_APPLICABLE", (
                f"{gate_name} should be NOT_APPLICABLE for synthetic fixture"
            )
        assert result["n_not_applicable"] >= 7


# ---------------------------------------------------------------------------
# Serialization parity
# ---------------------------------------------------------------------------


class TestSerializationParity:
    def test_exact_reload_parity_passes(self):
        policy = {
            "policy_id": "candidate",
            "policy_type": "candidate",
            "model_id": "synthetic-7b",
            "model_revision": "1.0",
            "feature_schema_digest": _FAKE_SHA,
            "feature_transform_digest": "exec_features_v2",
            "weights": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "bias": [0.0, 0.0],
            "action_ordering": ["ANSWER_DIRECT", "RETRIEVE_MEMORY"],
        }
        result = compute_serialization_parity(policy, "candidate")
        assert result["status"] == "PASS"
        assert result["max_abs_diff"] <= 1e-9
        assert result["selected_action_parity"] is True

    def test_changed_action_ordering_fails(self):
        """Different action_ordering produces a different policy hash."""
        policy = {
            "policy_id": "candidate",
            "policy_type": "candidate",
            "model_id": "synthetic-7b",
            "weights": [[1.0, 0.0], [0.0, 1.0]],
            "bias": [0.0, 0.0],
            "action_ordering": ["alpha", "beta"],
        }
        result = compute_serialization_parity(policy, "candidate")
        hash1 = result["policy_hash"]
        # Change the ordering — the policy hash should differ.
        changed_policy = dict(policy)
        changed_policy["action_ordering"] = ["beta", "alpha"]
        result2 = compute_serialization_parity(changed_policy, "candidate")
        hash2 = result2["policy_hash"]
        assert hash1 != hash2

    def test_changed_feature_transform_fails(self):
        """Changing feature_transform_digest changes the feature_transform_hash."""
        policy = {
            "policy_id": "candidate",
            "policy_type": "candidate",
            "model_id": "synthetic-7b",
            "feature_schema_digest": _FAKE_SHA,
            "feature_transform_digest": "exec_features_v2",
            "weights": [[0.1, 0.2]],
            "bias": [0.0],
            "action_ordering": ["alpha"],
        }
        result = compute_serialization_parity(policy, "candidate")
        hash1 = result["feature_transform_hash"]
        changed = dict(policy)
        changed["feature_transform_digest"] = "exec_features_v3"
        result2 = compute_serialization_parity(changed, "candidate")
        hash2 = result2["feature_transform_hash"]
        assert hash1 != hash2

    def test_changed_weights_fail(self):
        """Changing weights after reload changes logits."""
        validation_fixture = {"feature_vector": [1.0, 1.0]}
        policy = {
            "policy_id": "candidate",
            "policy_type": "candidate",
            "model_id": "synthetic-7b",
            "weights": [[1.0, 0.0], [0.0, 1.0]],
            "bias": [0.0, 0.0],
            "action_ordering": ["alpha", "beta"],
        }
        result = compute_serialization_parity(policy, "candidate", validation_fixture)
        original_logits = result["original_logits"]
        # Change weights — reloaded logits would differ.
        changed = dict(policy)
        changed["weights"] = [[2.0, 0.0], [0.0, 1.0]]
        result2 = compute_serialization_parity(changed, "candidate", validation_fixture)
        assert result2["original_logits"] != original_logits


# ---------------------------------------------------------------------------
# Split access
# ---------------------------------------------------------------------------


class TestSplitAccess:
    def test_no_test_access_passes(self):
        """Training stages only accessing experience split → PASS."""
        access_log = [
            {"stage": "train_candidate", "split": "experience", "operation": "read"},
            {"stage": "train_sham", "split": "experience", "operation": "read"},
            {"stage": "calibrate_threshold", "split": "validation", "operation": "read"},
            {"stage": "gate_a_evaluate", "split": "test", "operation": "read"},
        ]
        split_manifest = {"splits": {"experience": {}, "validation": {}, "test": {}}}
        result = audit_split_access(access_log, split_manifest)
        assert result["status"] == "PASS"
        assert len(result["forbidden_accesses"]) == 0

    def test_premature_test_access_fails(self):
        """A training stage accessing the test split → FAIL."""
        access_log = [
            {"stage": "train_candidate", "split": "test", "operation": "read"},
            {"stage": "gate_a_evaluate", "split": "test", "operation": "read"},
        ]
        split_manifest = {"splits": {"experience": {}, "validation": {}, "test": {}}}
        result = audit_split_access(access_log, split_manifest)
        assert result["status"] == "FAIL"
        assert len(result["forbidden_accesses"]) >= 1

    def test_absent_log_blocks(self):
        """An absent access log → BLOCKED."""
        split_manifest = {"splits": {"experience": {}, "validation": {}, "test": {}}}
        result = audit_split_access(None, split_manifest)
        assert result["status"] == "BLOCKED"

    def test_retrospective_diagnostics_labeled(self):
        """Retrospective diagnostic stages are not flagged as forbidden."""
        access_log = [
            {"stage": "train_candidate", "split": "experience", "operation": "read"},
            {"stage": "gate_a_evaluate", "split": "test", "operation": "read"},
            {"stage": "retrospective_diagnostic", "split": "test", "operation": "read"},
        ]
        split_manifest = {"splits": {"experience": {}, "validation": {}, "test": {}}}
        result = audit_split_access(access_log, split_manifest)
        # The retrospective diagnostic accesses test but after the test stage,
        # so it should not be a violation (it appears after gate_a_evaluate).
        assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# Artifact lineage
# ---------------------------------------------------------------------------


class TestArtifactLineage:
    def test_synthetic_parent_in_real_run_fails(self):
        """A synthetic artifact as parent in a real run → FAIL."""
        artifacts = {
            "art_a": {
                "artifact_type": "model",
                "run_id": "run_001",
                "parent_artifact_hashes": ["art_b"],
                "created_at": "2026-01-01",
                "producer_module": "trainer",
                "evidence_origin": "REAL_MODEL",
            },
            "art_b": {
                "artifact_type": "data",
                "run_id": "run_001",
                "parent_artifact_hashes": [],
                "created_at": "2026-01-01",
                "producer_module": "loader",
                "evidence_origin": "SYNTHETIC",
            },
        }
        result = validate_artifact_lineage(artifacts, "REAL_MODEL")
        assert result["status"] == "FAIL"

    def test_missing_parent_fails(self):
        artifacts = {
            "art_a": {
                "artifact_type": "model",
                "run_id": "run_001",
                "parent_artifact_hashes": ["nonexistent"],
                "created_at": "2026-01-01",
                "producer_module": "trainer",
            },
        }
        result = validate_artifact_lineage(artifacts, "SYNTHETIC")
        assert result["status"] == "FAIL"
        assert result["n_missing_parents"] == 1

    def test_cycles_fail(self):
        artifacts = {
            "art_a": {
                "artifact_type": "model",
                "run_id": "run_001",
                "parent_artifact_hashes": ["art_b"],
                "created_at": "2026-01-01",
                "producer_module": "trainer",
            },
            "art_b": {
                "artifact_type": "data",
                "run_id": "run_001",
                "parent_artifact_hashes": ["art_a"],
                "created_at": "2026-01-01",
                "producer_module": "loader",
            },
        }
        result = validate_artifact_lineage(artifacts, "SYNTHETIC")
        assert result["status"] == "FAIL"
        assert result["n_cycles"] >= 1

    def test_full_fixture_dag_passes(self):
        """A well-formed DAG with no cycles, no missing parents, no cross-origin."""
        artifacts = {
            "art_root": {
                "artifact_type": "data",
                "run_id": "run_001",
                "parent_artifact_hashes": [],
                "created_at": "2026-01-01",
                "producer_module": "loader",
                "evidence_origin": "SYNTHETIC",
            },
            "art_model": {
                "artifact_type": "model",
                "run_id": "run_001",
                "parent_artifact_hashes": ["art_root"],
                "created_at": "2026-01-01",
                "producer_module": "trainer",
                "evidence_origin": "SYNTHETIC",
            },
            "art_results": {
                "artifact_type": "results",
                "run_id": "run_001",
                "parent_artifact_hashes": ["art_model"],
                "created_at": "2026-01-01",
                "producer_module": "evaluator",
                "evidence_origin": "SYNTHETIC",
            },
        }
        result = validate_artifact_lineage(artifacts, "SYNTHETIC")
        assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# Oracle discrepancy
# ---------------------------------------------------------------------------


class TestOracleDiscrepancy:
    def test_synthetic_vs_historical_reports_synthetic_vs_real(self):
        """SYNTHETIC origin with historical_headroom → SYNTHETIC_VS_REAL."""
        oracle_results = {"mean_utility": 5.0}
        sham_results = {"mean_utility": 3.0}
        evidence_manifest = {"evidence_origin": "SYNTHETIC"}
        benchmark_manifest = {}
        result = investigate_oracle_discrepancy(
            oracle_results, sham_results, [], evidence_manifest, benchmark_manifest,
            historical_headroom=2.0,
        )
        assert result["classification"] == "SYNTHETIC_VS_REAL"
        assert result["discrepancy_found"] is True

    def test_insufficient_evidence_reports_unresolved(self):
        """Missing oracle/sham means → UNRESOLVED."""
        oracle_results = {}
        sham_results = {}
        evidence_manifest = {"evidence_origin": "REAL_MODEL"}
        benchmark_manifest = {}
        result = investigate_oracle_discrepancy(
            oracle_results, sham_results, [], evidence_manifest, benchmark_manifest,
        )
        assert result["classification"] == "UNRESOLVED"
        assert result["status"] == "BLOCKED"

    def test_no_unsupported_cause_selected(self):
        """When multiple possible causes exist, classification is UNRESOLVED
        (not a single unsupported cause)."""
        oracle_results = {
            "mean_utility": 5.0,
            "task_rows": [{"task_id": "t1"}],
            "policy_id": "same_policy",
        }
        sham_results = {
            "mean_utility": 3.0,
            "task_rows": [{"task_id": "t2"}],
            "policy_id": "same_policy",
        }
        evidence_manifest = {"evidence_origin": "SYNTHETIC"}
        benchmark_manifest = {}
        result = investigate_oracle_discrepancy(
            oracle_results, sham_results, [], evidence_manifest, benchmark_manifest,
        )
        # With SYNTHETIC origin (no historical_headroom), multiple causes
        # (DIFFERENT_TASK_SET + DIFFERENT_SHAM_POLICY + SYNTHETIC_VS_REAL)
        # should result in UNRESOLVED, not a single unsupported cause.
        assert result["classification"] == "UNRESOLVED"
        assert "possible_causes" in result
        assert len(result["possible_causes"]) > 1


# ---------------------------------------------------------------------------
# Scale decision
# ---------------------------------------------------------------------------


class TestScaleDecision:
    def test_synthetic_fixture_returns_not_applicable(self):
        result = make_scale_decision(
            gate_a0_status="PASS",
            evidence_origin="SYNTHETIC",
        )
        assert result["decision"] == DECISION_NOT_APPLICABLE
        assert result["approve_matched_14b_pilot"] is False
        assert result["approve_full_14b_run"] is False

    def test_missing_evidence_returns_blocked(self):
        result = make_scale_decision(
            gate_a0_status="PASS",
            evidence_origin="UNKNOWN",
        )
        assert result["decision"] == DECISION_BLOCKED

    def test_failed_gate_a0_returns_blocked(self):
        result = make_scale_decision(
            gate_a0_status="FAIL",
            evidence_origin="REAL_MODEL",
        )
        assert result["decision"] == DECISION_BLOCKED

    def test_synthetic_never_approves_scaling(self):
        """A synthetic fixture must never produce any approval decision."""
        for a0_status in ("PASS", "FAIL", "BLOCKED"):
            result = make_scale_decision(
                gate_a0_status=a0_status,
                evidence_origin="SYNTHETIC",
            )
            assert result["decision"] == DECISION_NOT_APPLICABLE
            assert result["decision"] not in (
                DECISION_APPROVE_MATCHED_14B_PILOT,
                DECISION_APPROVE_FULL_14B_RUN,
                DECISION_NO_SCALE_NEEDED_FOR_GATE_A,
            )


# ---------------------------------------------------------------------------
# Unavailable evidence
# ---------------------------------------------------------------------------


class TestUnavailableEvidence:
    def test_historical_modal_unavailable_manifest_exists(self):
        assert UNAVAILABLE_DIR.exists()
        manifest_path = UNAVAILABLE_DIR / "EVIDENCE_UNAVAILABLE.json"
        assert manifest_path.exists()

    def test_unavailable_manifest_has_correct_fields(self):
        manifest_path = UNAVAILABLE_DIR / "EVIDENCE_UNAVAILABLE.json"
        manifest = _load_json(manifest_path)
        assert manifest["status"] == "UNAVAILABLE"
        assert manifest["scientific_claim_eligible"] is False
        assert manifest["gate_a_status"] == "BLOCKED"
        assert manifest["approve_matched_14b_pilot"] is False
        assert manifest["approve_full_14b_run"] is False
        assert "reason" in manifest

    def test_unavailable_manifest_blocks_scaling(self):
        """The unavailable manifest must block any scale decision."""
        manifest_path = UNAVAILABLE_DIR / "EVIDENCE_UNAVAILABLE.json"
        manifest = _load_json(manifest_path)
        assert manifest["gate_a_status"] == "BLOCKED"
        # The scale decision for a BLOCKED gate_a0 with non-REAL_MODEL origin is BLOCKED.
        decision = make_scale_decision(
            gate_a0_status=manifest["gate_a_status"],
            evidence_origin="UNKNOWN",
        )
        assert decision["decision"] == DECISION_BLOCKED


# ---------------------------------------------------------------------------
# Modal recovery
# ---------------------------------------------------------------------------


class TestModalRecovery:
    def test_missing_modal_evidence_writes_missing_file(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        output_dir = tmp_path / "output"

        result = recover_modal_evidence(
            str(source_dir), str(output_dir),
            expected_run_id="test_run_001",
        )
        assert result["status"] == "MISSING"
        assert len(result["missing_files"]) > 0
        # The MISSING_MODAL_EVIDENCE.json report must be written.
        report_path = output_dir / "MISSING_MODAL_EVIDENCE.json"
        assert report_path.exists()
        report = _load_json(report_path)
        assert report["status"] == "MISSING"
        assert report["expected_run_id"] == "test_run_001"
        assert report["n_missing"] > 0

    def test_recovery_does_not_synthesize(self, tmp_path):
        """Recovery must never synthesize missing evidence — it stops and reports."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        output_dir = tmp_path / "output"

        result = recover_modal_evidence(
            str(source_dir), str(output_dir),
        )
        assert result["status"] == "MISSING"
        # No evidence files should be created in the output directory.
        assert not any(f.suffix in (".jsonl", ".json") for f in output_dir.iterdir()
                        if f.name != "MISSING_MODAL_EVIDENCE.json")
        # The reason must explicitly state no synthesis occurred.
        assert "synthesized" in result["reason"].lower() or "no evidence synthesized" in result["reason"].lower()


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


def _make_exp_action_row(task_id="task_000", action="ANSWER_DIRECT", **overrides):
    """Build an action-level experience row for the normalizer."""
    base = {
        "task_id": task_id,
        "action": action,
        "utility": 5.0,
        "verified_success": action == "ANSWER_DIRECT",
        "split": "experience",
        "family": "direct_answer",
        "archetype": "direct_answer_archetype",
        "policy_version": "candidate_training",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "experience_id": _FAKE_SHA,
        "state": {
            "available_tools": list(CANONICAL_ACTIONS),
            "context_length": 10,
            "conversation_features": [0.9, 0.1, 0.1],
            "memory_features": [0.5, 0.2, 0.0, 0.0, 0.0],
            "prompt_features": [10.0, 1.0, 0.0, 0.0, 0.0],
            "model_id": "synthetic-7b",
        },
        "outcome": {
            "verified_success": action == "ANSWER_DIRECT",
            "latency_ms": 100.0,
            "token_count": 50,
            "tool_calls_executed": 0,
            "tool_failures": 0,
            "runtime_error": False,
            "safety_violation": False,
            "operator_intervention": False,
            "action_completed": True,
            "verification_status": "verified" if action == "ANSWER_DIRECT" else "failed",
        },
        "provenance": {
            "source": "exact_match",
            "task_family": "direct_answer",
            "task_id": task_id,
            "policy_version": "candidate_training",
        },
    }
    base.update(overrides)
    return base


def _make_safety_row(task_id="task_007", **overrides):
    """Build a safety results row with all required fields."""
    base = {
        "task_id": task_id,
        "risk_class": "elevated",
        "guard_invoked": True,
        "guard_decision": "SAFE_REFUSAL",
        "learner_consulted": False,
        "learner_output_ignored": True,
        "requested_action": "CALL_TOOL",
        "final_action": "SAFE_REFUSAL",
        "tool_calls_executed": 0,
        "workflows_started": 0,
        "side_effects_detected": False,
        "verifier_name": "exact_match",
        "verifier_version": "1.0",
        "verified_safe": True,
        "evidence_digest": _FAKE_SHA,
        "run_id": "synthetic_v046_test_001",
    }
    base.update(overrides)
    return base


def _make_results(mean_utility, task_rows):
    """Build a results dict with consistent mean_utility."""
    return {
        "policy_id": "test",
        "run_id": "synthetic_v046_test_001",
        "split": "test",
        "mean_utility": mean_utility,
        "mean_utility_full_precision": mean_utility,
        "task_count": len(task_rows),
        "task_rows": task_rows,
    }


def _generate_synthetic_evidence(ev_dir: Path) -> None:
    """Generate a minimal synthetic evidence package in *ev_dir*.

    Creates all required files for the v0.4.6 orchestrator to run
    end-to-end.  Uses 8 tasks (4 experience, 2 test, 1 ood, 1 safety)
    with 4 counterfactual outcomes per task.
    """
    tasks = [
        ("task_000", "experience", "direct_answer", "direct_answer_archetype"),
        ("task_001", "experience", "memory_required", "memory_required_archetype"),
        ("task_002", "experience", "tool_required", "tool_required_archetype"),
        ("task_003", "experience", "workflow_required", "workflow_required_archetype"),
        ("task_004", "test", "direct_answer", "direct_answer_archetype"),
        ("task_005", "test", "memory_required", "memory_required_archetype"),
        ("task_006", "ood", "tool_required", "tool_required_archetype"),
        ("task_007", "safety", "workflow_required", "workflow_required_archetype"),
    ]

    # --- EVIDENCE_MANIFEST.json ---
    evidence_manifest = {
        "evidence_package_version": "1.0.0",
        "original_run_id": "synthetic_v046_test_001",
        "original_package_version": "2.15.5",
        "original_qualification_version": "0.4.1",
        "original_commit_sha": None,
        "model_id": "synthetic-7b",
        "model_revision": "1.0",
        "model_digest": None,
        "tokenizer_id": "synthetic-tokenizer",
        "tokenizer_revision": "1.0",
        "provider_class": "SYNTHETIC",
        "runtime_type": "synthetic",
        "evidence_origin": "SYNTHETIC",
        "scientific_claim_eligible": False,
        "promotable": False,
        "provider_model_identity": "synthetic-7b",
        "n_tasks": 8,
        "n_counterfactual_outcomes": 32,
        "n_experiences": 16,
        "n_validation_tasks": 0,
        "n_test_tasks": 2,
        "n_ood_tasks": 1,
        "n_safety_tasks": 1,
        "creation_timestamp": "2026-01-01T00:00:00+00:00",
        "immutable": True,
        "scientific_result": "synthetic_fixture",
        "promoted": False,
    }
    _write_json(ev_dir / "EVIDENCE_MANIFEST.json", evidence_manifest)

    # --- benchmark_manifest.json ---
    benchmark_manifest = {
        "manifest_version": "1.0",
        "total_tasks": 8,
        "actions": list(CANONICAL_ACTIONS),
        "families": ["direct_answer", "memory_required", "tool_required", "workflow_required"],
        "tasks": [
            {
                "task_id": tid,
                "split": split,
                "family": family,
                "archetype": archetype,
                "allowed_actions": list(CANONICAL_ACTIONS),
                "verifier_spec": {"type": "exact_match", "tolerance": 0.0},
                "prompt": f"Test prompt for {tid}",
                "generator_family": family,
                "template_family": f"{family}_template",
                "capability_family": "reasoning",
                "entity_family": "entity_0",
                "risk_class": "benign" if split != "safety" else "elevated",
                "crossover_type": "none",
                "group_id": "group_0",
                "generator_seed": 42,
                "setup_spec": {"features": [1.0, 0.0], "n_features": 2},
                "expected_output_digest": _FAKE_SHA,
            }
            for tid, split, family, archetype in tasks
        ],
    }
    _write_json(ev_dir / "benchmark_manifest.json", benchmark_manifest)

    # --- split_manifest.json ---
    split_manifest = {
        "run_id": "synthetic_v046_test_001",
        "benchmark_digest": _FAKE_SHA,
        "splits": {
            "experience": {"count": 4, "task_ids": ["task_000", "task_001", "task_002", "task_003"]},
            "test": {"count": 2, "task_ids": ["task_004", "task_005"]},
            "ood": {"count": 1, "task_ids": ["task_006"]},
            "safety": {"count": 1, "task_ids": ["task_007"]},
        },
    }
    _write_json(ev_dir / "split_manifest.json", split_manifest)

    # --- provider_manifest.json ---
    provider_manifest = {
        "provider_class": "SYNTHETIC",
        "runtime_type": "synthetic",
        "model_id": "synthetic-7b",
        "model_revision": "1.0",
        "model_digest": None,
        "tokenizer_id": "synthetic-tokenizer",
        "tokenizer_revision": "1.0",
        "generation_config_digest": _FAKE_SHA,
        "supports_gate_a": False,
        "supports_gate_b": False,
        "scientific_claim_eligible": False,
        "promotable": False,
    }
    _write_json(ev_dir / "provider_manifest.json", provider_manifest)

    # --- source_provenance.json ---
    source_provenance = {
        "original_commit_sha": None,
        "original_source_tree_sha256": _FAKE_SHA,
        "original_source_file_count": 10,
        "original_source_byte_count": 5000,
        "original_qualification_source_sha256": _FAKE_SHA,
        "original_config_sha256": _FAKE_SHA,
        "original_dependency_lock_sha256": _FAKE_SHA,
        "original_benchmark_sha256": _FAKE_SHA,
        "original_provider_manifest_sha256": _FAKE_SHA,
        "source_tree_sha256": _FAKE_SHA,
        "source_file_count": 10,
        "source_byte_count": 5000,
        "hash_algorithm_version": "SHA-256",
    }
    _write_json(ev_dir / "source_provenance.json", source_provenance)

    # --- counterfactual_outcomes.jsonl ---
    cf_rows = []
    for tid, split, family, archetype in tasks:
        for action in CANONICAL_ACTIONS:
            cf_rows.append(_make_cf_row(
                task_id=tid,
                action=action,
                split=split,
                family=family,
                archetype=archetype,
                verifier_name="exact_match",
            ))
    _write_jsonl(ev_dir / "counterfactual_outcomes.jsonl", cf_rows)

    # --- executive_experiences.jsonl ---
    exp_rows = []
    for tid, split, family, archetype in tasks[:4]:  # experience tasks only
        for action in CANONICAL_ACTIONS:
            exp_rows.append(_make_exp_action_row(
                task_id=tid,
                action=action,
                split=split,
                family=family,
                archetype=archetype,
            ))
    _write_jsonl(ev_dir / "executive_experiences.jsonl", exp_rows)

    # --- safety_results.jsonl ---
    safety_rows = [_make_safety_row(task_id="task_007")]
    _write_jsonl(ev_dir / "safety_results.jsonl", safety_rows)

    # --- results files (candidate, sham, baseline, oracle) ---
    test_task_rows = [
        {"task_id": "task_004", "selected_action": "ANSWER_DIRECT", "selected_utility": 5.0, "success": True, "family": "direct_answer"},
        {"task_id": "task_005", "selected_action": "RETRIEVE_MEMORY", "selected_utility": 3.0, "success": False, "family": "memory_required"},
    ]
    candidate_mean = sum(r["selected_utility"] for r in test_task_rows) / len(test_task_rows)
    _write_json(ev_dir / "candidate_results.json", _make_results(candidate_mean, test_task_rows))

    sham_task_rows = [
        {"task_id": "task_004", "selected_action": "RETRIEVE_MEMORY", "selected_utility": 3.0, "success": False, "family": "direct_answer"},
        {"task_id": "task_005", "selected_action": "ANSWER_DIRECT", "selected_utility": 3.0, "success": False, "family": "memory_required"},
    ]
    sham_mean = sum(r["selected_utility"] for r in sham_task_rows) / len(sham_task_rows)
    _write_json(ev_dir / "sham_results.json", _make_results(sham_mean, sham_task_rows))

    baseline_task_rows = [
        {"task_id": "task_004", "selected_action": "ANSWER_DIRECT", "selected_utility": 2.0, "success": False, "family": "direct_answer"},
        {"task_id": "task_005", "selected_action": "ANSWER_DIRECT", "selected_utility": 2.0, "success": False, "family": "memory_required"},
    ]
    baseline_mean = sum(r["selected_utility"] for r in baseline_task_rows) / len(baseline_task_rows)
    _write_json(ev_dir / "baseline_results.json", _make_results(baseline_mean, baseline_task_rows))

    oracle_task_rows = [
        {"task_id": "task_004", "selected_action": "ANSWER_DIRECT", "selected_utility": 3.0, "success": True, "family": "direct_answer"},
        {"task_id": "task_005", "selected_action": "RETRIEVE_MEMORY", "selected_utility": 3.0, "success": False, "family": "memory_required"},
    ]
    oracle_mean = sum(r["selected_utility"] for r in oracle_task_rows) / len(oracle_task_rows)
    _write_json(ev_dir / "oracle_results.json", _make_results(oracle_mean, oracle_task_rows))

    # --- gate_a_results.json ---
    _write_json(ev_dir / "gate_a_results.json", {
        "status": "NOT_APPLICABLE",
        "reason": "synthetic fixture",
    })

    # --- dataset_manifest.json ---
    _write_json(ev_dir / "dataset_manifest.json", {
        "manifest_version": "1.0",
        "n_features": 2,
        "feature_names": ["prompt_length", "log_prompt_length"],
        "actions": list(CANONICAL_ACTIONS),
        "families": ["direct_answer", "memory_required", "tool_required", "workflow_required"],
        "splits": {
            "train": {"n_examples": 4, "source_split": "experience"},
            "test": {"n_examples": 2, "source_split": "test"},
            "ood": {"n_examples": 1, "source_split": "ood"},
            "safety": {"n_examples": 1, "source_split": "safety"},
        },
        "metadata": {
            "created_at": "2026-01-01T00:00:00Z",
            "generator": "synthetic_7b_generator",
            "utility_version": "normalized_v1",
            "n_actions": 4,
            "n_families": 4,
        },
        "feature_schema_version": "exec_features_v2",
    })

    # --- candidate_policy.json ---
    _write_json(ev_dir / "candidate_policy.json", {
        "policy_id": "candidate",
        "policy_type": "candidate",
        "model_id": "synthetic-7b",
        "model_revision": "1.0",
        "feature_schema_digest": _FAKE_SHA,
        "feature_transform_digest": "exec_features_v2",
        "weights": [[0.1, 0.0], [0.0, 0.1], [0.0, 0.0], [0.0, 0.0]],
        "bias": [0.1, 0.0, -0.05, 0.0],
        "action_ordering": list(CANONICAL_ACTIONS),
        "normalization_state": {},
        "training_seed": 42,
        "training_split_digest": "experience",
        "learner_config_digest": "logistic_regression",
        "policy_sha256": _FAKE_SHA,
    })

    # --- sham_policy.json ---
    _write_json(ev_dir / "sham_policy.json", {
        "policy_id": "sham",
        "policy_type": "sham",
        "model_id": "synthetic-7b",
        "model_revision": "1.0",
        "feature_schema_digest": _FAKE_SHA,
        "feature_transform_digest": "exec_features_v2",
        "weights": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "bias": [0.0, 0.0, 0.0, 0.0],
        "action_ordering": list(CANONICAL_ACTIONS),
        "normalization_state": {},
        "training_seed": 42,
        "training_split_digest": "experience",
        "learner_config_digest": "logistic_regression",
        "policy_sha256": _FAKE_SHA2,
    })

    # --- split_access_log.json ---
    _write_json(ev_dir / "split_access_log.json", [
        {"stage": "candidate_training", "split": "experience", "operation": "read"},
        {"stage": "sham_training", "split": "experience", "operation": "read"},
        {"stage": "gate_a_evaluate", "split": "test", "operation": "evaluate"},
    ])

    # --- artifact_lineage.json ---
    _write_json(ev_dir / "artifact_lineage.json", {
        "art_root": {
            "artifact_type": "data",
            "run_id": "synthetic_v046_test_001",
            "parent_artifact_hashes": [],
            "created_at": "2026-01-01",
            "producer_module": "loader",
            "evidence_origin": "SYNTHETIC",
        },
    })

    # --- coverage_results.json ---
    _write_json(ev_dir / "coverage_results.json", {
        "status": "PASS",
        "coverage": 1.0,
    })

    # --- runtime_completion_diagnostics.json ---
    _write_json(ev_dir / "runtime_completion_diagnostics.json", {
        "status": "PASS",
        "completion_rate": 1.0,
    })

    # --- artifact_checksums.json ---
    # Compute SHA-256 of all required files (except artifact_checksums.json itself).
    from qualification.autolearn_v04.v045.config import REQUIRED_EVIDENCE_FILES as V045_REQUIRED
    checksums = {}
    for fname in V045_REQUIRED:
        fpath = ev_dir / fname
        if fpath.exists() and fname != "artifact_checksums.json":
            checksums[fname] = _sha256_file(fpath)
    _write_json(ev_dir / "artifact_checksums.json", checksums)


class TestV046EndToEndPipeline:
    """End-to-end integration test for the v0.4.6 orchestrator."""

    def test_v046_end_to_end_pipeline(self, tmp_path):
        """Run the full v0.4.6 pipeline on a minimal synthetic evidence package.

        Asserts that:
        - Gate A0 is not BLOCKED (all required evidence is present)
        - Scientific sub-gates are NOT_APPLICABLE for synthetic evidence
        - All 30 required artifacts are produced
        - The analysis manifest exists and has correct fields
        - The scale decision exists and is NOT_APPLICABLE for synthetic
        """
        from qualification.autolearn_v04.v046.orchestrator import run_all_v046_diagnostics

        # 1. Create a temporary evidence directory
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        _generate_synthetic_evidence(ev_dir)

        # 2. Run the v0.4.6 orchestrator on this evidence
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        run_id = "v046_e2e_test_001"
        manifest = run_all_v046_diagnostics(
            evidence_dir=str(ev_dir),
            output_dir=str(output_dir),
            run_id=run_id,
            repo_root=str(REPO_ROOT),
            force=True,
        )

        # 3. Assert the analysis manifest exists and has correct fields
        out_path = output_dir / run_id
        manifest_path = out_path / "analysis_manifest.json"
        assert manifest_path.exists(), "analysis_manifest.json must exist"
        assert isinstance(manifest, dict)
        assert manifest["analysis_run_id"] == run_id
        assert manifest["evidence_origin"] == "SYNTHETIC"
        assert manifest["overall_evidence_eligibility"] == "PASS"

        # 4. Assert the scale decision exists
        scale_decision_path = out_path / "model_scale_decision.json"
        assert scale_decision_path.exists(), "model_scale_decision.json must exist"
        scale_decision = _load_json(scale_decision_path)
        assert scale_decision["decision"] == DECISION_NOT_APPLICABLE
        assert scale_decision["approve_full_14b_run"] is False
        assert scale_decision["approve_matched_14b_pilot"] is False

        # 5. Assert Gate A0 is not BLOCKED (all required evidence is present)
        gate_a0_path = out_path / "gate_a0_v046.json"
        assert gate_a0_path.exists(), "gate_a0_v046.json must exist"
        gate_a0 = _load_json(gate_a0_path)
        assert gate_a0["status"] != "BLOCKED", (
            f"Gate A0 should not be BLOCKED, but is: {gate_a0['overall_reason']}"
        )
        assert gate_a0["n_blocked"] == 0, (
            f"No sub-gates should be BLOCKED, but {gate_a0['n_blocked']} are"
        )

        # 6. Assert scientific sub-gates are NOT_APPLICABLE
        sub_gates = gate_a0["sub_gates"]
        scientific_gate_names = [
            "A0.2_evidence_origin_authenticity",
            "A0.3_scientific_claim_eligibility",
            "A0.6_historical_identity",
            "A0.7_current_analysis_identity",
            "A0.8_cross_version_lineage",
            "A0.9_provider_model_authenticity",
            "A0.10_single_generation_identity",
        ]
        for sg_name in scientific_gate_names:
            assert sub_gates[sg_name]["status"] == "NOT_APPLICABLE", (
                f"{sg_name} should be NOT_APPLICABLE for synthetic, "
                f"got {sub_gates[sg_name]['status']}"
            )

        # 7. Assert structural sub-gates are PASS
        # (A0.24 oracle consistency is a known limitation for synthetic
        # evidence — SYNTHETIC_VS_REAL is always a possible cause.)
        structural_gate_names = [
            "A0.1_structural_completeness",
            "A0.4_cross_origin_duplication",
            "A0.5_source_provenance",
            "A0.11_positive_evidence_weights",
            "A0.12_counterfactual_equivalence",
            "A0.13_complete_action_matrix",
            "A0.14_verifier_registry_completeness",
            "A0.15_utility_consistency",
            "A0.16_split_access_integrity",
            "A0.17_candidate_serialization_parity",
            "A0.18_sham_serialization_parity",
            "A0.19_task_split_consistency",
            "A0.20_artifact_lineage",
            "A0.21_metric_consistency",
            "A0.22_safety_evidence_integrity",
            "A0.23_no_stale_artifact_reuse",
        ]
        for sg_name in structural_gate_names:
            assert sub_gates[sg_name]["status"] == "PASS", (
                f"{sg_name} should be PASS for synthetic, "
                f"got {sub_gates[sg_name]['status']}: "
                f"{sub_gates[sg_name]['reason']}"
            )

        # 8. Assert all 30 required artifacts are produced
        required_artifacts = [
            "analysis_manifest.json",
            "analysis_source_provenance.json",
            "analysis_config.json",
            "evidence_origin_audit.json",
            "cross_origin_duplication_report.json",
            "experience_normalization_report.json",
            "normalized_experience_rows.json",
            "evidence_weight_audit.json",
            "verifier_registry_audit.json",
            "counterfactual_equivalence_report.json",
            "action_matrix_completeness.json",
            "split_access_audit.json",
            "candidate_serialization_parity.json",
            "sham_serialization_parity.json",
            "artifact_lineage_report.json",
            "oracle_discrepancy_report.json",
            "historical_identity_report.json",
            "analysis_identity_report.json",
            "cross_version_lineage_report.json",
            "model_identity_consistency.json",
            "task_split_consistency.json",
            "safety_evidence_integrity.json",
            "utility_consistency_report.json",
            "metric_consistency_report.json",
            "gate_a0_v046.json",
            "model_scale_decision.json",
            "final_provenance_manifest.json",
            "artifact_checksums.json",
            "GATE_A_V046_EVIDENCE_REPAIR_REPORT.md",
            "fixture_validation_report.json",
        ]
        missing = [
            name for name in required_artifacts
            if not (out_path / name).exists()
        ]
        assert not missing, f"Missing {len(missing)} artifacts: {missing}"
        assert len(required_artifacts) == 30
