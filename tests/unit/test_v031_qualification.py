"""Tests for v0.3.1 qualification modules.

Tests:
- Split disjointness validation
- Exact verifiers reject substring matching
- Provenance source tree hash is not a git hash
- Report integrity: rejected candidate is never marked promoted
- Report integrity: missing stage is NOT RUN
- Promotion gate v3.1 integration
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Add the project root to the path so we can import qualification modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qualification.autolearn_v031.schemas import SplitEntry, VerificationEvidence
from qualification.autolearn_v031.splits import (
    validate_split_disjointness,
    validate_split_sizes,
    validate_no_test_leakage,
)
from qualification.autolearn_v031.verifiers import (
    verify_direct_answer,
    verify_memory_result,
    verify_tool_result,
    verify_workflow_result,
    verify_safety,
)
from qualification.autolearn_v031.provenance import (
    compute_source_tree_sha256,
    compute_json_sha256,
)
from qualification.autolearn_v031.report import generate_report, StageStatus


class TestSplitDisjointness:
    """Test that split validation catches overlaps."""

    def test_disjoint_splits_pass(self):
        entries = [
            SplitEntry("t1", "direct", "arch_a", "gen1", 42, "experience",
                       ("ANSWER_DIRECT",), "deterministic", "d1", "p1"),
            SplitEntry("t2", "memory", "arch_b", "gen1", 43, "validation",
                       ("RETRIEVE_MEMORY",), "deterministic", "d2", "p2"),
            SplitEntry("t3", "tool", "arch_c", "gen2", 44, "test",
                       ("CALL_TOOL",), "deterministic", "d3", "p3"),
            SplitEntry("t4", "workflow", "arch_d", "gen2", 45, "ood",
                       ("START_WORKFLOW",), "deterministic", "d4", "p4"),
        ]
        errors = validate_split_disjointness(entries)
        assert errors == []

    def test_overlapping_archetypes_fail(self):
        entries = [
            SplitEntry("t1", "direct", "arch_a", "gen1", 42, "experience",
                       ("ANSWER_DIRECT",), "deterministic", "d1", "p1"),
            SplitEntry("t2", "direct", "arch_a", "gen1", 43, "test",
                       ("ANSWER_DIRECT",), "deterministic", "d2", "p2"),
        ]
        errors = validate_split_disjointness(entries)
        assert len(errors) > 0
        assert "arch_a" in errors[0]

    def test_duplicate_task_id_fails(self):
        entries = [
            SplitEntry("t1", "direct", "arch_a", "gen1", 42, "experience",
                       ("ANSWER_DIRECT",), "deterministic", "d1", "p1"),
            SplitEntry("t1", "direct", "arch_b", "gen1", 43, "test",
                       ("ANSWER_DIRECT",), "deterministic", "d2", "p2"),
        ]
        errors = validate_split_disjointness(entries)
        assert len(errors) > 0
        assert "t1" in errors[0]

    def test_minimum_sizes_enforced(self):
        entries = [
            SplitEntry("t1", "direct", "arch_a", "gen1", 42, "experience",
                       ("ANSWER_DIRECT",), "deterministic", "d1", "p1"),
        ]
        errors = validate_split_sizes(entries)
        assert len(errors) > 0
        assert any("experience" in e for e in errors)

    def test_test_leakage_detected(self):
        errors = validate_no_test_leakage({"t1", "t2"}, {"t2", "t3"})
        assert len(errors) > 0
        assert "leakage" in errors[0]


class TestExactVerifiers:
    """Test that exact verifiers reject substring matching."""

    def test_exact_json_answer_passes(self):
        result = verify_direct_answer('{"answer": 42}', {"answer": 42})
        assert result.passed is True
        assert result.verifier_type == "deterministic"
        assert result.confidence == 1.0

    def test_substring_match_rejected(self):
        """A response that contains the answer as a substring but is not
        valid JSON must be rejected."""
        result = verify_direct_answer("The answer is 42", {"answer": 42})
        assert result.passed is False
        assert "json_parse_error" in result.error_code

    def test_wrong_value_rejected(self):
        result = verify_direct_answer('{"answer": 41}', {"answer": 42})
        assert result.passed is False
        assert result.error_code == "value_mismatch"

    def test_extra_keys_rejected(self):
        result = verify_direct_answer('{"answer": 42, "extra": "foo"}', {"answer": 42})
        assert result.passed is False
        assert result.error_code == "key_set_mismatch"

    def test_answer_in_prompt_rejected(self):
        result = verify_direct_answer(
            '{"answer": "hello world"}', {"answer": "hello world"},
            prompt="What is hello world?",
        )
        assert result.passed is False
        assert result.error_code == "answer_leaked_in_prompt"

    def test_memory_secret_must_match_exactly(self):
        result = verify_memory_result(
            '{"result": "MEM-WRONG"}',
            {"result": "MEM-CORRECT123"},
        )
        assert result.passed is False
        assert result.error_code == "secret_mismatch"

    def test_memory_service_not_used_rejected(self):
        result = verify_memory_result(
            '{"result": "MEM-CORRECT"}',
            {"result": "MEM-CORRECT"},
            memory_service_used=False,
        )
        assert result.passed is False
        assert result.error_code == "memory_service_not_used"

    def test_tool_count_mismatch_rejected(self):
        result = verify_tool_result(
            '{"result": "TOOL-ABC"}',
            {"result": "TOOL-ABC"},
            tool_invocation_count=2,
            expected_tool_count=1,
        )
        assert result.passed is False
        assert result.error_code == "tool_count_mismatch"

    def test_workflow_acceptance_not_passed_rejected(self):
        result = verify_workflow_result(
            "done", {},
            acceptance_passed=False,
            exit_code=0,
            artifact_exists=True,
        )
        assert result.passed is False
        assert result.error_code == "acceptance_not_passed"

    def test_workflow_nonzero_exit_rejected(self):
        result = verify_workflow_result(
            "done", {},
            acceptance_passed=True,
            exit_code=1,
            artifact_exists=True,
        )
        assert result.passed is False
        assert result.error_code == "nonzero_exit_code"

    def test_safety_not_preempted_rejected(self):
        result = verify_safety(
            action_taken="ANSWER_DIRECT",
            safety_preempted=False,
        )
        assert result.passed is False
        assert result.error_code == "safety_not_preempted"

    def test_safety_preempted_passes(self):
        result = verify_safety(
            action_taken="ASK_OPERATOR",
            safety_preempted=True,
        )
        assert result.passed is True


class TestProvenance:
    """Test provenance computation."""

    def test_source_tree_hash_is_not_empty(self, tmp_path):
        # Create a minimal source tree.
        (tmp_path / "src.py").write_text("print('hello')")
        h = compute_source_tree_sha256(tmp_path)
        assert len(h) == 64  # SHA-256 hex
        assert h != ""

    def test_source_tree_hash_is_deterministic(self, tmp_path):
        (tmp_path / "a.py").write_text("a = 1")
        h1 = compute_source_tree_sha256(tmp_path)
        h2 = compute_source_tree_sha256(tmp_path)
        assert h1 == h2

    def test_source_tree_hash_excludes_pycache(self, tmp_path):
        (tmp_path / "a.py").write_text("a = 1")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "a.pyc").write_text("compiled")
        h1 = compute_source_tree_sha256(tmp_path)
        # Changing pyc should not change hash.
        (tmp_path / "__pycache__" / "a.pyc").write_text("different")
        h2 = compute_source_tree_sha256(tmp_path)
        assert h1 == h2

    def test_source_tree_hash_excludes_sqlite(self, tmp_path):
        (tmp_path / "a.py").write_text("a = 1")
        (tmp_path / "data.sqlite").write_text("database")
        h1 = compute_source_tree_sha256(tmp_path)
        (tmp_path / "data.sqlite").write_text("changed database")
        h2 = compute_source_tree_sha256(tmp_path)
        assert h1 == h2

    def test_json_sha256_is_deterministic(self):
        h1 = compute_json_sha256({"a": 1, "b": 2})
        h2 = compute_json_sha256({"b": 2, "a": 1})
        assert h1 == h2  # sort_keys ensures determinism


class TestReportIntegrity:
    """Test that report integrity is maintained."""

    def test_rejected_candidate_not_marked_promoted(self, tmp_path):
        """A rejected candidate must never be marked as promoted in the report."""
        # Write a promotion_result that shows rejection.
        (tmp_path / "promotion_result.json").write_text(json.dumps({
            "passed": False,
            "gates": [{"name": "mean_utility_improves", "status": "FAIL",
                       "observed": 3.0, "threshold": 5.0, "reason": "test"}],
        }))
        report = generate_report(tmp_path, active_policy_id="")
        assert "BLOCKED" in report
        assert "PASS" not in report.split("Promotion")[1].split("Post-promotion")[0]

    def test_missing_stage_is_not_run(self, tmp_path):
        """A missing stage should be reported as NOT_RUN."""
        report = generate_report(tmp_path)
        assert "NOT_RUN" in report

    def test_active_policy_from_registry(self, tmp_path):
        """The report should show the active policy ID, not assume candidate."""
        (tmp_path / "promotion_result.json").write_text(json.dumps({
            "passed": True,
            "gates": [],
        }))
        report = generate_report(tmp_path, active_policy_id="baseline_v3")
        assert "baseline_v3" in report

    def test_losing_candidate_not_marked_superior(self, tmp_path):
        """A losing candidate must not be marked as superior."""
        (tmp_path / "test_results.json").write_text(json.dumps({
            "mean_delta": -0.5,
            "delta_ci_low": -0.87,
            "delta_ci_high": 0.08,
        }))
        report = generate_report(tmp_path)
        assert "FAIL" in report
        # The final claim should be the narrower statement.
        assert "has not yet been qualified" in report
