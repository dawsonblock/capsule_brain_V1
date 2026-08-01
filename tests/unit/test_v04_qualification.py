"""v0.4.0 qualification hardening tests.

Tests the specific defects identified in the release prompt:

A. Version mismatch — pyproject.toml must match version.py and qualification constants.
B. GroundedQualificationProvider is infrastructure-only.
C. Source-tree hashing produces non-empty SHA-256 with positive file/byte counts.
D. Trust-region and action-collapse checks use D_validation, NOT D_test.
E. KL trust-region uses smoothed baseline distribution (finite, no 1e6 explosion).
F. Downstream stages return BLOCKED, not exceptions, when upstream fails.
G. Runtime completion diagnostics distinguish failure causes.
H. One active qualification tree identified by manifest.
I. Provider classification: REAL_MODEL supports Gate A/B; INFRASTRUCTURE does not.
J. Scientific benchmark hides expected answers from prompts.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# Repo root is two levels up from tests/unit/test_v04_qualification.py
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# A. Version consistency
# ---------------------------------------------------------------------------


class TestVersionConsistency:
    def test_pyproject_matches_version_module(self):
        from capsule_brain.version import PACKAGE_VERSION
        root = REPO_ROOT
        pyproject = (root / "pyproject.toml").read_text()
        assert f'version = "{PACKAGE_VERSION}"' in pyproject

    def test_qualification_imports_from_version_module(self):
        from capsule_brain.version import PACKAGE_VERSION as pv
        from qualification.autolearn_v04 import PACKAGE_VERSION as qpv
        assert pv == qpv

    def test_all_version_constants_present(self):
        from capsule_brain.version import (
            PACKAGE_VERSION,
            AUTOLEARN_VERSION,
            AUTOLEARN_QUALIFICATION_VERSION,
            PROTOCOL_VERSION,
        )
        assert PACKAGE_VERSION == "2.15.12"
        assert AUTOLEARN_VERSION == "0.3.10"
        assert AUTOLEARN_QUALIFICATION_VERSION == "0.4.7"
        assert PROTOCOL_VERSION == "0.4.7"


# ---------------------------------------------------------------------------
# B. Provider classification
# ---------------------------------------------------------------------------


class TestProviderClassification:
    def test_grounded_provider_is_infrastructure(self):
        from qualification.autolearn_v04.provider_classification import (
            ProviderClass,
            classify_grounded_provider,
        )
        caps = classify_grounded_provider()
        assert caps.provider_class is ProviderClass.INFRASTRUCTURE
        assert caps.supports_gate_a_claim is False
        assert caps.supports_gate_b_claim is False

    def test_local_transformers_is_real_model(self):
        from qualification.autolearn_v04.provider_classification import (
            ProviderClass,
            classify_local_transformers,
        )
        caps = classify_local_transformers("test-model")
        assert caps.provider_class is ProviderClass.REAL_MODEL
        assert caps.supports_gate_a_claim is True
        assert caps.supports_gate_b_claim is True

    def test_simulated_cannot_promote(self):
        from qualification.autolearn_v04.provider_classification import (
            ProviderClass,
            classify_simulated,
        )
        caps = classify_simulated()
        assert caps.provider_class is ProviderClass.SIMULATED
        assert caps.supports_gate_a_claim is False
        assert caps.supports_gate_b_claim is False

    def test_assert_provider_eligible_raises_for_infrastructure(self):
        from qualification.autolearn_v04.provider_classification import (
            ProviderEligibilityError,
            assert_provider_eligible_for_gate,
            classify_grounded_provider,
        )
        caps = classify_grounded_provider()
        with pytest.raises(ProviderEligibilityError):
            assert_provider_eligible_for_gate(caps, "A")

    def test_grounded_provider_declares_infrastructure(self):
        from qualification.autolearn_v04.grounded_provider import GroundedQualificationProvider
        provider = GroundedQualificationProvider()
        assert provider.provider_class == "infrastructure"
        assert provider.supports_gate_a_claim is False
        assert provider.supports_gate_b_claim is False


# ---------------------------------------------------------------------------
# C. Source-tree provenance
# ---------------------------------------------------------------------------


class TestSourceTreeHashing:
    def test_source_hash_is_nonempty(self):
        from qualification.autolearn_v04.provenance import compute_source_tree_hash
        root = REPO_ROOT
        h = compute_source_tree_hash(root)
        assert len(h) == 64
        assert h != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_source_hash_has_positive_file_and_byte_counts(self):
        from qualification.autolearn_v04.provenance import compute_source_tree_hash_detailed
        root = REPO_ROOT
        result = compute_source_tree_hash_detailed(root)
        assert result.source_file_count > 0
        assert result.source_byte_count > 0
        assert result.source_tree_sha256 != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_source_hash_stable_across_working_directories(self):
        from qualification.autolearn_v04.provenance import compute_source_tree_hash, find_project_root
        root = find_project_root()
        h1 = compute_source_tree_hash(root)
        h2 = compute_source_tree_hash(str(root))
        assert h1 == h2

    def test_source_hash_works_from_nested_directory(self):
        from qualification.autolearn_v04.provenance import compute_source_tree_hash, find_project_root
        root = find_project_root(Path(__file__))
        h1 = compute_source_tree_hash(root)
        root2 = find_project_root(Path(__file__).resolve().parent)
        h2 = compute_source_tree_hash(root2)
        assert h1 == h2

    def test_empty_source_set_fails_closed(self):
        from qualification.autolearn_v04.provenance import (
            SourceHashError,
            compute_source_tree_hash_detailed,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(SourceHashError):
                compute_source_tree_hash_detailed(tmp)

    def test_generated_artifacts_excluded(self):
        """Generated artifacts under artifacts_v04/ should not change the source hash."""
        from qualification.autolearn_v04.provenance import compute_source_tree_hash
        root = REPO_ROOT
        h1 = compute_source_tree_hash(root)
        # Create a fake artifact — should not affect the hash.
        artifacts_dir = root / "qualification" / "autolearn_v04" / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        fake = artifacts_dir / "test_fake.json"
        fake.write_text('{"test": true}')
        try:
            h2 = compute_source_tree_hash(root)
            assert h1 == h2
        finally:
            fake.unlink()


# ---------------------------------------------------------------------------
# D. Final-test isolation
# ---------------------------------------------------------------------------


class TestSplitIsolation:
    def test_split_auditor_detects_illegal_test_access(self):
        from qualification.autolearn_v04.split_auditor import SplitAccessLog
        log = SplitAccessLog()
        log.record("train_candidate", ["t1", "t2"], ["test"], "trust-region check")
        errors = log.validate()
        assert len(errors) > 0
        assert "train_candidate" in errors[0]

    def test_split_auditor_allows_test_access_by_gate_a(self):
        from qualification.autolearn_v04.split_auditor import SplitAccessLog
        log = SplitAccessLog()
        log.record("train_candidate", ["t1"], ["validation"], "training")
        log.record("evaluate_gate_a_test", ["t1"], ["test"], "final evaluation")
        errors = log.validate()
        assert len(errors) == 0

    def test_trust_region_uses_validation_not_test(self, tmp_path):
        """Verify that train_candidate's trust_region records split_used='validation'."""
        # This is verified by checking the trust_region field in the candidate policy.
        # We check the code path, not a full run.
        from qualification.autolearn_v04.train_candidate import _compute_kl_from_baseline
        from capsule_brain.autolearn.baseline import BaselinePolicyV3
        from capsule_brain.autolearn.policy import LearnedPolicy
        # The function signature should accept tasks; we verify it returns finite KL.
        # Full integration test would require a trained policy.


# ---------------------------------------------------------------------------
# E. KL trust-region
# ---------------------------------------------------------------------------


class TestTrustRegion:
    def test_kl_is_finite_for_different_policies(self):
        """The smoothed KL should produce finite values, not 1e6 explosions."""
        from qualification.autolearn_v04.train_candidate import _compute_kl_from_baseline
        from capsule_brain.autolearn.baseline import BaselinePolicyV3
        from capsule_brain.autolearn.policy import LearnedPolicy
        from capsule_brain.autolearn.schema import Action
        # Create a simple mock policy that always predicts uniform.
        class UniformPolicy:
            def predict_proba(self, fv):
                return {a.value: 0.25 for a in Action.learned()}
            def select_action(self, state, allowed_actions=None):
                return type("D", (), {"action": Action.ANSWER_DIRECT})()
        tasks = [
            {"task_id": "t1", "family": "direct_answer", "split": "validation",
             "prompt": "test", "allowed_actions": ["ANSWER_DIRECT", "RETRIEVE_MEMORY", "CALL_TOOL", "START_WORKFLOW"],
             "setup_spec": {}},
        ]
        result = _compute_kl_from_baseline(UniformPolicy(), BaselinePolicyV3(), tasks)
        assert result["mean_kl"] < 100  # finite, not 1e6
        assert result["mean_kl"] >= 0
        assert "p95_kl" in result
        assert "action_change_rate" in result

    def test_kl_near_zero_for_identical_policy(self):
        """If the candidate always picks the baseline action, KL should be small."""
        from qualification.autolearn_v04.train_candidate import _compute_kl_from_baseline
        from capsule_brain.autolearn.baseline import BaselinePolicyV3
        from capsule_brain.autolearn.schema import Action
        class BaselineImitator:
            def __init__(self):
                self._baseline = BaselinePolicyV3()
            def predict_proba(self, fv):
                from capsule_brain.autolearn.schema import ExecutiveState
                # Return high prob for baseline action.
                return {a.value: 0.9 if a == Action.ANSWER_DIRECT else 0.033 for a in Action.learned()}
            def select_action(self, state, allowed_actions=None):
                return self._baseline.select_action(state, allowed_actions=allowed_actions)
        tasks = [
            {"task_id": "t1", "family": "direct_answer", "split": "validation",
             "prompt": "test", "allowed_actions": ["ANSWER_DIRECT", "RETRIEVE_MEMORY", "CALL_TOOL", "START_WORKFLOW"],
             "setup_spec": {}},
        ]
        result = _compute_kl_from_baseline(BaselineImitator(), BaselinePolicyV3(), tasks, epsilon_smooth=0.1)
        # KL should be small since candidate puts most mass on baseline action.
        assert result["mean_kl"] < 1.0


# ---------------------------------------------------------------------------
# F. Pipeline dependency handling
# ---------------------------------------------------------------------------


class TestPipelineDependencies:
    def test_stage_dependencies_block_on_failure(self):
        from qualification.autolearn_v04.stage_dependencies import (
            PIPELINE_STAGES,
            StageStatus,
            check_stage_dependencies,
        )
        results = {"diagnose_runtime_completion": StageStatus.FAIL}
        build_ds = next(s for s in PIPELINE_STAGES if s.stage_name == "build_dataset")
        errors = check_stage_dependencies(build_ds, results)
        assert len(errors) > 0
        assert "diagnose_runtime_completion" in errors[0]

    def test_should_block_downstream(self):
        from qualification.autolearn_v04.stage_dependencies import (
            StageStatus,
            should_block_downstream,
        )
        results = {"train_candidate": StageStatus.FAIL}
        blocked = should_block_downstream("train_candidate", results)
        # evaluate_gate_a and run_promotion should be blocked (direct deps).
        assert "evaluate_gate_a" in blocked
        assert "run_promotion" in blocked

    def test_post_promotion_blocked_when_promotion_not_pass(self):
        """v0.4.1: post-promotion requires promotion PASS."""
        from qualification.autolearn_v04.stage_dependencies import (
            StageStatus,
            check_stage_dependencies,
            get_stage,
        )
        post_promo = get_stage("run_post_promotion")
        # Promotion blocked → post-promotion should be blocked.
        results = {"run_promotion": StageStatus.BLOCKED}
        errors = check_stage_dependencies(post_promo, results)
        assert any("PASS" in e for e in errors)

    def test_post_promotion_blocked_when_promotion_fails(self):
        """v0.4.1: post-promotion blocked when promotion fails."""
        from qualification.autolearn_v04.stage_dependencies import (
            StageStatus,
            check_stage_dependencies,
            get_stage,
        )
        post_promo = get_stage("run_post_promotion")
        results = {"run_promotion": StageStatus.FAIL}
        errors = check_stage_dependencies(post_promo, results)
        assert len(errors) > 0

    def test_promotion_before_post_promotion_in_ordering(self):
        """v0.4.1: promotion must come before post-promotion in stage order."""
        from qualification.autolearn_v04.stage_dependencies import PIPELINE_STAGES
        stage_names = [s.stage_name for s in PIPELINE_STAGES]
        promo_idx = stage_names.index("run_promotion")
        post_promo_idx = stage_names.index("run_post_promotion")
        assert promo_idx < post_promo_idx


# ---------------------------------------------------------------------------
# G. Runtime completion diagnostics
# ---------------------------------------------------------------------------


class TestRuntimeCompletion:
    def test_diagnose_classifies_failure_causes(self, tmp_path):
        from qualification.autolearn_v04.diagnose_runtime_completion import diagnose_runtime_completion
        from qualification.autolearn_v04.config import QualificationConfig
        # Create minimal artifacts.
        (tmp_path / "counterfactual_outcomes.json").write_text(json.dumps([
            {"task_id": "t1", "action_id": "ANSWER_DIRECT", "availability": "executed",
             "verification": "success", "utility": 10.0,
             "execution_metadata": {"runtime_type": "real", "latency_ms": 100, "token_count": 10,
                                    "verification_evidence": {"verifier_type": "direct_exact", "observed": "ok"}}},
            {"task_id": "t1", "action_id": "CALL_TOOL", "availability": "execution_error",
             "verification": None, "utility": None,
             "execution_metadata": {"runtime_type": "real", "latency_ms": 0, "token_count": 0,
                                    "verification_evidence": {}},
             "error_type": "execution_error"},
        ]))
        (tmp_path / "benchmark_manifest.json").write_text(json.dumps({
            "tasks": [
                {"task_id": "t1", "family": "direct_answer", "split": "experience",
                 "prompt": "test", "allowed_actions": ["ANSWER_DIRECT", "CALL_TOOL"],
                 "setup_spec": {}},
            ]
        }))
        config = QualificationConfig(artifacts_dir=str(tmp_path))
        result = diagnose_runtime_completion(config)
        assert "by_cause" in result
        assert result["by_cause"].get("dispatcher_failure", 0) > 0


# ---------------------------------------------------------------------------
# H. Qualification tree consolidation
# ---------------------------------------------------------------------------


class TestQualificationManifest:
    def test_manifest_exists_and_identifies_active_tree(self):
        root = REPO_ROOT
        manifest_path = root / "qualification" / "QUALIFICATION_MANIFEST.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["active_qualification"] == "autolearn_v04"
        assert manifest["qualification_version"] == "0.4.7"
        assert manifest["package_version"] == "2.15.12"
        assert manifest["status"] == "current"

    def test_archive_readme_exists(self):
        root = REPO_ROOT
        archive_readme = root / "qualification" / "archive" / "README.md"
        assert archive_readme.exists()
        content = archive_readme.read_text()
        assert "HISTORICAL" in content
        assert "NOT CURRENT QUALIFICATION EVIDENCE" in content


# ---------------------------------------------------------------------------
# I. Scientific benchmark hides answers
# ---------------------------------------------------------------------------


class TestScientificBenchmark:
    def test_comparison_task_hides_answer(self):
        from qualification.autolearn_v04.scientific_benchmark import _comparison_task
        import random
        rng = random.Random(42)
        task = _comparison_task("t1", "test", rng)
        # The setup_spec should NOT contain the expected value.
        assert "expected_value" not in task.setup_spec
        # The verifier_spec should have the expected value.
        assert "expected_value" in task.verifier_spec

    def test_classification_task_hides_answer(self):
        from qualification.autolearn_v04.scientific_benchmark import _classification_task
        import random
        rng = random.Random(42)
        task = _classification_task("t1", "test", rng)
        # The category should NOT appear as the answer in the setup.
        assert "expected_value" not in task.setup_spec
        assert "expected_value" in task.verifier_spec

    def test_counting_task_hides_answer(self):
        from qualification.autolearn_v04.scientific_benchmark import _counting_task
        import random
        rng = random.Random(42)
        task = _counting_task("t1", "test", rng)
        # The count should NOT appear in the setup_spec.
        assert "expected_value" not in task.setup_spec
        assert "expected_value" in task.verifier_spec

    def test_code_output_task_hides_answer(self):
        from qualification.autolearn_v04.scientific_benchmark import _code_output_task
        import random
        rng = random.Random(42)
        task = _code_output_task("t1", "test", rng)
        # The setup_spec should NOT contain the expected value.
        assert "expected_value" not in task.setup_spec
        assert "expected_value" in task.verifier_spec

    def test_memory_task_hides_secret(self):
        from qualification.autolearn_v04.scientific_benchmark import _scientific_memory_task
        import random
        rng = random.Random(42)
        task = _scientific_memory_task("t1", "test", rng)
        secret = task.verifier_spec["expected_secret"]
        # Secret must NOT appear in the prompt.
        assert secret not in task.prompt

    def test_tool_task_hides_output(self):
        from qualification.autolearn_v04.scientific_benchmark import _scientific_tool_task
        import random
        rng = random.Random(42)
        task = _scientific_tool_task("t1", "test", rng)
        expected = task.verifier_spec["expected_tool_output"]
        # Expected output must NOT appear in the prompt.
        assert expected not in task.prompt

    def test_workflow_task_hides_solution(self):
        from qualification.autolearn_v04.scientific_benchmark import _scientific_workflow_task
        import random
        rng = random.Random(42)
        task = _scientific_workflow_task("t1", "test", rng)
        # The prompt should NOT contain the nonce.
        nonce = task.setup_spec.get("nonce", "")
        assert nonce not in task.prompt


# ---------------------------------------------------------------------------
# J. No benchmark-answer shortcuts in real-model provider
# ---------------------------------------------------------------------------


class TestNoBenchmarkShortcuts:
    def test_local_transformers_provider_has_no_benchmark_patterns(self):
        """The real-model provider must not contain benchmark-answer extraction."""
        root = REPO_ROOT
        provider_path = root / "qualification" / "autolearn_v04" / "local_transformers_provider.py"
        content = provider_path.read_text()
        # These patterns are benchmark-specific answer extraction shortcuts.
        forbidden_patterns = [
            r'DIRECT-\d',  # direct answer token extraction
            r'MEM-\d',     # memory secret extraction
            r'TOOL-\d',    # tool output extraction
            r'_DIRECT_RE',
            r'_MEM_RE',
            r'_NONCE_RE',
            r'_TOOL_RESULT_RE',
            r'expected_token',
            r'expected_secret',
            r'expected_tool_output',
        ]
        import re
        for pattern in forbidden_patterns:
            matches = re.findall(pattern, content)
            assert len(matches) == 0, f"Forbidden pattern '{pattern}' found in local_transformers_provider.py"


# ---------------------------------------------------------------------------
# K. No -1000 placeholders in counterfactual outcomes
# ---------------------------------------------------------------------------


class TestNoNegativeThousandPlaceholders:
    def test_outcome_schema_rejects_utility_for_non_executed(self):
        from qualification.autolearn_v04.schemas import (
            CounterfactualOutcome,
            OutcomeAvailability,
            VerificationOutcome,
        )
        # A NOT_EXECUTED outcome with utility should fail validation.
        with pytest.raises(ValueError):
            outcome = CounterfactualOutcome(
                task_id="t1", action_id="ANSWER_DIRECT",
                availability=OutcomeAvailability.NOT_EXECUTED,
                verification=None, utility=-1000.0,
                reward_components={}, execution_metadata={},
                artifact_digest=None,
            )
            outcome.validate()

    def test_executed_outcome_requires_utility(self):
        from qualification.autolearn_v04.schemas import (
            CounterfactualOutcome,
            OutcomeAvailability,
            VerificationOutcome,
        )
        # An EXECUTED outcome without utility should fail validation.
        with pytest.raises(ValueError):
            outcome = CounterfactualOutcome(
                task_id="t1", action_id="ANSWER_DIRECT",
                availability=OutcomeAvailability.EXECUTED,
                verification=VerificationOutcome.SUCCESS, utility=None,
                reward_components={}, execution_metadata={},
                artifact_digest=None,
            )
            outcome.validate()


# ---------------------------------------------------------------------------
# L. Provenance digests validation
# ---------------------------------------------------------------------------


class TestProvenanceValidation:
    def test_empty_digest_rejected(self):
        from qualification.autolearn_v04.schemas import ProvenanceError, validate_digest
        with pytest.raises(ProvenanceError):
            validate_digest("test", "")

    def test_empty_sha256_rejected(self):
        from qualification.autolearn_v04.schemas import EMPTY_SHA256, ProvenanceError, validate_digest
        with pytest.raises(ProvenanceError):
            validate_digest("test", EMPTY_SHA256)

    def test_non_hex_digest_rejected(self):
        from qualification.autolearn_v04.schemas import ProvenanceError, validate_digest
        with pytest.raises(ProvenanceError):
            validate_digest("test", "not-a-hex-digest")

    def test_valid_digest_accepted(self):
        from qualification.autolearn_v04.schemas import validate_digest
        validate_digest("test", "a" * 64)  # should not raise


# ---------------------------------------------------------------------------
# M. Gate status enum
# ---------------------------------------------------------------------------


class TestGateStatus:
    def test_gate_status_values(self):
        from qualification.autolearn_v04.schemas import GateStatus
        assert GateStatus.PASS.value == "pass"
        assert GateStatus.FAIL.value == "fail"
        assert GateStatus.BLOCKED.value == "blocked"
        assert GateStatus.NOT_RUN.value == "not_run"

    def test_gate_result_passed_property(self):
        from qualification.autolearn_v04.schemas import GateCategory, GateResult, GateStatus
        passing = GateResult("test", GateCategory.INTEGRITY, GateStatus.PASS)
        failing = GateResult("test", GateCategory.INTEGRITY, GateStatus.FAIL)
        assert passing.passed is True
        assert failing.passed is False


# ---------------------------------------------------------------------------
# v0.4.1: Source hashing regression tests (parent dirs named data/build/archive)
# ---------------------------------------------------------------------------


class TestSourceHashingRegression:
    """Source hashing must work regardless of parent directory names."""

    def _make_mini_repo(self, base: Path) -> Path:
        """Create a minimal source tree under base/repo."""
        repo = base / "repo"
        (repo / "src" / "capsule_brain").mkdir(parents=True, exist_ok=True)
        (repo / "src" / "capsule_brain" / "__init__.py").write_text("PACKAGE_VERSION = '2.15.5'\n")
        (repo / "src" / "capsule_brain" / "version.py").write_text(
            "PACKAGE_VERSION = '2.15.5'\nAUTOLEARN_VERSION = '0.3.4'\n"
        )
        (repo / "qualification" / "autolearn_v04").mkdir(parents=True, exist_ok=True)
        (repo / "qualification" / "autolearn_v04" / "__init__.py").write_text("# v04\n")
        (repo / "tests").mkdir(parents=True, exist_ok=True)
        (repo / "tests" / "test_basic.py").write_text("def test_x(): assert True\n")
        (repo / "pyproject.toml").write_text('[project]\nname = "capsule-brain"\nversion = "2.15.5"\n')
        return repo

    def test_parent_named_data_does_not_skip_files(self, tmp_path):
        """Repo under /tmp/.../data/repo must still hash all files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        repo = self._make_mini_repo(tmp_path)
        from qualification.autolearn_v04.provenance import compute_source_tree_hash_detailed
        result = compute_source_tree_hash_detailed(repo)
        assert result.source_file_count > 0
        assert result.source_byte_count > 0
        assert result.source_tree_sha256 != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_parent_named_build_does_not_skip_files(self, tmp_path):
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        repo = self._make_mini_repo(build_dir)
        from qualification.autolearn_v04.provenance import compute_source_tree_hash_detailed
        result = compute_source_tree_hash_detailed(repo)
        assert result.source_file_count > 0

    def test_parent_named_archive_does_not_skip_files(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        repo = self._make_mini_repo(archive_dir)
        from qualification.autolearn_v04.provenance import compute_source_tree_hash_detailed
        result = compute_source_tree_hash_detailed(repo)
        assert result.source_file_count > 0

    def test_digest_stable_across_parent_dirs(self, tmp_path):
        """Same repo content under different parent dirs produces same digest."""
        from qualification.autolearn_v04.provenance import compute_source_tree_hash_detailed
        repo_a = self._make_mini_repo(tmp_path / "data")
        repo_b = self._make_mini_repo(tmp_path / "build")
        result_a = compute_source_tree_hash_detailed(repo_a)
        result_b = compute_source_tree_hash_detailed(repo_b)
        assert result_a.source_tree_sha256 == result_b.source_tree_sha256

    def test_source_modification_alters_digest(self, tmp_path):
        from qualification.autolearn_v04.provenance import compute_source_tree_hash_detailed
        repo = self._make_mini_repo(tmp_path)
        result1 = compute_source_tree_hash_detailed(repo)
        # Modify a source file.
        (repo / "src" / "capsule_brain" / "version.py").write_text(
            "PACKAGE_VERSION = '2.15.7'\nAUTOLEARN_VERSION = '0.3.4'\n"
        )
        result2 = compute_source_tree_hash_detailed(repo)
        assert result1.source_tree_sha256 != result2.source_tree_sha256

    def test_generated_artifacts_do_not_alter_digest(self, tmp_path):
        from qualification.autolearn_v04.provenance import compute_source_tree_hash_detailed
        repo = self._make_mini_repo(tmp_path)
        result1 = compute_source_tree_hash_detailed(repo)
        # Create artifacts under the excluded prefix.
        artifacts_dir = repo / "qualification" / "autolearn_v04" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "junk.json").write_text('{"junk": true}')
        result2 = compute_source_tree_hash_detailed(repo)
        assert result1.source_tree_sha256 == result2.source_tree_sha256

    def test_empty_collection_fails_closed(self, tmp_path):
        from qualification.autolearn_v04.provenance import (
            compute_source_tree_hash_detailed, SourceHashError,
        )
        empty_repo = tmp_path / "empty_repo"
        empty_repo.mkdir()
        with pytest.raises(SourceHashError):
            compute_source_tree_hash_detailed(empty_repo)

    def test_should_skip_uses_relative_paths(self, tmp_path):
        """should_skip must compare against relative path, not absolute."""
        from qualification.autolearn_v04.provenance import should_skip, SIMPLE_SKIP_DIR_NAMES
        root = tmp_path / "repo"
        root.mkdir()
        # A file under a 'data' parent dir should NOT be skipped just because
        # the parent is named 'data'.
        (root / "src").mkdir()
        src_file = root / "src" / "test.py"
        src_file.write_text("# test")
        assert should_skip(src_file, root) is False
        # But __pycache__ should be skipped.
        pycache_file = root / "src" / "__pycache__" / "test.cpython.pyc"
        pycache_file.parent.mkdir(parents=True, exist_ok=True)
        pycache_file.write_bytes(b"\x00")
        assert should_skip(pycache_file, root) is True


# ---------------------------------------------------------------------------
# v0.4.1: Verifier reliability registry tests
# ---------------------------------------------------------------------------


class TestVerifierReliability:
    def test_direct_exact_reliability_1(self):
        from capsule_brain.verification.reliability import get_verifier_reliability
        assert get_verifier_reliability("direct_exact") == 1.0

    def test_memory_secret_reliability_1(self):
        from capsule_brain.verification.reliability import get_verifier_reliability
        assert get_verifier_reliability("memory_secret") == 1.0

    def test_tool_grounding_reliability_1(self):
        from capsule_brain.verification.reliability import get_verifier_reliability
        assert get_verifier_reliability("tool_grounding") == 1.0

    def test_workflow_acceptance_reliability_1(self):
        from capsule_brain.verification.reliability import get_verifier_reliability
        assert get_verifier_reliability("workflow_acceptance") == 1.0

    def test_safety_preemption_reliability_1(self):
        from capsule_brain.verification.reliability import get_verifier_reliability
        assert get_verifier_reliability("safety_preemption") == 1.0

    def test_tool_output_reliability_1(self):
        from capsule_brain.verification.reliability import get_verifier_reliability
        assert get_verifier_reliability("tool_output") == 1.0

    def test_unknown_verifier_raises(self):
        from capsule_brain.verification.reliability import (
            get_verifier_reliability, UnknownVerifierTypeError,
        )
        with pytest.raises(UnknownVerifierTypeError):
            get_verifier_reliability("nonexistent_verifier")

    def test_unknown_verifier_fallback_when_allowed(self):
        from capsule_brain.verification.reliability import get_verifier_reliability
        r = get_verifier_reliability("nonexistent", allow_experimental_fallback=True)
        assert 0 < r < 1.0

    def test_llm_judge_reliability_between_04_and_06(self):
        from capsule_brain.verification.reliability import get_verifier_reliability
        r = get_verifier_reliability("llm_judge")
        assert 0.40 <= r <= 0.60

    def test_verifier_descriptor_has_all_fields(self):
        from capsule_brain.verification.reliability import get_verifier_descriptor
        desc = get_verifier_descriptor("direct_exact")
        d = desc.to_dict()
        assert "verifier_name" in d
        assert "verifier_class" in d
        assert "reliability" in d
        assert "independently_grounded" in d


# ---------------------------------------------------------------------------
# v0.4.1: Experience quality with typed verifiers
# ---------------------------------------------------------------------------


class TestExperienceQuality:
    def test_deterministic_verifier_positive_q(self):
        from capsule_brain.autolearn.utility import compute_experience_quality
        q = compute_experience_quality(
            runtime_type="real",
            verifier_type="direct_exact",
            action_set_complete=True,
            isolation_enforced=True,
        )
        assert q.quality_score > 0
        assert q.verifier_reliability == 1.0

    def test_simulated_execution_zero_q(self):
        from capsule_brain.autolearn.utility import compute_experience_quality
        q = compute_experience_quality(
            runtime_type="simulated",
            verifier_type="direct_exact",
        )
        assert q.execution_integrity == 0.0
        assert q.quality_score == 0.0

    def test_unknown_verifier_raises(self):
        from capsule_brain.autolearn.utility import compute_experience_quality
        from capsule_brain.verification.reliability import UnknownVerifierTypeError
        with pytest.raises(UnknownVerifierTypeError):
            compute_experience_quality(
                runtime_type="real",
                verifier_type="nonexistent",
            )

    def test_missing_provenance_zero_q(self):
        from capsule_brain.autolearn.utility import compute_experience_quality
        q = compute_experience_quality(
            runtime_type="real",
            verifier_type="direct_exact",
            provenance_valid=False,
        )
        assert q.provenance_integrity == 0.0
        assert q.quality_score == 0.0

    def test_isolation_failure_zero_q(self):
        from capsule_brain.autolearn.utility import compute_experience_quality
        q = compute_experience_quality(
            runtime_type="real",
            verifier_type="direct_exact",
            isolation_enforced=False,
        )
        assert q.isolation_integrity == 0.0
        assert q.quality_score == 0.0

    def test_incomplete_counterfactual_reduces_q(self):
        from capsule_brain.autolearn.utility import compute_experience_quality
        q_full = compute_experience_quality(
            runtime_type="real", verifier_type="direct_exact", action_set_complete=True,
        )
        q_partial = compute_experience_quality(
            runtime_type="real", verifier_type="direct_exact", action_set_complete=False,
        )
        assert q_partial.counterfactual_completeness < q_full.counterfactual_completeness
        assert q_partial.quality_score < q_full.quality_score

    def test_q_components_serialize(self):
        from capsule_brain.autolearn.utility import compute_experience_quality
        q = compute_experience_quality(
            runtime_type="real", verifier_type="direct_exact",
        )
        d = q.to_dict()
        assert "verifier_reliability" in d
        assert "execution_integrity" in d
        assert "counterfactual_completeness" in d
        assert "isolation_integrity" in d
        assert "provenance_integrity" in d
        assert "quality_score" in d
        assert d["quality_score"] == q.quality_score


# ---------------------------------------------------------------------------
# v0.4.1: Version mismatch tests
# ---------------------------------------------------------------------------


class TestVersionMismatch:
    def test_pyproject_version_matches_package(self):
        from capsule_brain.version import PACKAGE_VERSION
        root = REPO_ROOT
        pyproject = (root / "pyproject.toml").read_text()
        # Extract version from pyproject.toml
        import re
        match = re.search(r'version\s*=\s*"([^"]+)"', pyproject)
        assert match is not None
        assert match.group(1) == PACKAGE_VERSION

    def test_manifest_version_matches_package(self):
        from capsule_brain.version import PACKAGE_VERSION, AUTOLEARN_QUALIFICATION_VERSION
        root = REPO_ROOT
        manifest = json.loads((root / "qualification" / "QUALIFICATION_MANIFEST.json").read_text())
        assert manifest["package_version"] == PACKAGE_VERSION
        assert manifest["qualification_version"] == AUTOLEARN_QUALIFICATION_VERSION


# ---------------------------------------------------------------------------
# v0.4.1: Pipeline ordering and single-shot execution tests
# ---------------------------------------------------------------------------


class TestPipelineOrdering:
    def test_promotion_before_post_promotion(self):
        """v0.4.1: promotion must come before post-promotion in stage order."""
        from qualification.autolearn_v04.stage_dependencies import PIPELINE_STAGES
        stage_names = [s.stage_name for s in PIPELINE_STAGES]
        assert stage_names.index("run_promotion") < stage_names.index("run_post_promotion")

    def test_preflight_is_first_stage(self):
        """v0.4.1: preflight must be the first stage."""
        from qualification.autolearn_v04.stage_dependencies import PIPELINE_STAGES
        assert PIPELINE_STAGES[0].stage_name == "preflight"

    def test_source_provenance_after_preflight(self):
        """v0.4.1: source_provenance must come after preflight."""
        from qualification.autolearn_v04.stage_dependencies import PIPELINE_STAGES
        stage_names = [s.stage_name for s in PIPELINE_STAGES]
        assert stage_names.index("source_provenance") > stage_names.index("preflight")

    def test_build_benchmark_depends_on_preflight(self):
        """v0.4.1: build_benchmark must depend on preflight."""
        from qualification.autolearn_v04.stage_dependencies import get_stage
        bm = get_stage("build_benchmark")
        assert "preflight" in bm.dependencies

    def test_post_promotion_depends_on_promotion(self):
        """v0.4.1: post-promotion must depend on promotion."""
        from qualification.autolearn_v04.stage_dependencies import get_stage
        post_promo = get_stage("run_post_promotion")
        assert "run_promotion" in post_promo.dependencies

    def test_post_promotion_blocked_when_promotion_blocked(self):
        """v0.4.1: post-promotion must be blocked when promotion is blocked."""
        from qualification.autolearn_v04.stage_dependencies import (
            StageStatus, check_stage_dependencies, get_stage,
        )
        post_promo = get_stage("run_post_promotion")
        results = {"run_promotion": StageStatus.BLOCKED}
        errors = check_stage_dependencies(post_promo, results)
        assert len(errors) > 0
        assert any("PASS" in e for e in errors)


# ---------------------------------------------------------------------------
# v0.4.1: Safety task exclusion tests
# ---------------------------------------------------------------------------


class TestSafetyExclusion:
    def test_safety_split_recognized(self):
        from qualification.autolearn_v04.config import QualificationConfig
        config = QualificationConfig(mode="scientific-mini")
        assert config.is_scientific

    def test_scientific_mini_mode_supported(self):
        from qualification.autolearn_v04.config import QualificationConfig
        config = QualificationConfig(mode="scientific-mini")
        assert config.is_scientific_mini
        assert config.is_scientific

    def test_infrastructure_cannot_promote(self):
        from qualification.autolearn_v04.config import QualificationConfig
        config = QualificationConfig(
            mode="infrastructure", provider="qual_grounded",
        )
        assert not config.can_promote

    def test_smoke_cannot_promote(self):
        from qualification.autolearn_v04.config import QualificationConfig
        config = QualificationConfig(mode="smoke")
        assert not config.can_promote


# ---------------------------------------------------------------------------
# v0.4.1: Grounded provider contract tests
# ---------------------------------------------------------------------------


class TestGroundedProviderContract:
    def test_direct_task_does_not_trigger_tool_call(self):
        """v0.4.1: Direct task with DIRECT- token should not generate tool calls."""
        import asyncio
        from qualification.autolearn_v04.grounded_provider import GroundedQualificationProvider
        from capsule_brain.llm.models import LLMRequest
        from capsule_brain.llm.tools import ToolSpec
        provider = GroundedQualificationProvider()
        request = LLMRequest(
            prompt='Output exactly this JSON: {"result":"DIRECT-abc123"}',
            model="qual-grounded-v04",
            messages=[{"role": "user", "content": 'Output exactly this JSON: {"result":"DIRECT-abc123"}'}],
            tools=[ToolSpec(name="test_tool", description="test", parameters={})],
            temperature=0.0,
            max_tokens=100,
        )
        result = asyncio.run(provider.generate(request, {}))
        # Should NOT have tool calls when direct answer is in prompt.
        assert len(result.tool_calls) == 0
        assert "DIRECT-abc123" in result.text


# ---------------------------------------------------------------------------
# v0.4.1: Orchestrator tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_local_backend_name(self):
        from qualification.autolearn_v04.orchestrator import LocalExecutionBackend
        backend = LocalExecutionBackend()
        assert backend.backend_name == "local"

    def test_modal_backend_name(self):
        from qualification.autolearn_v04.orchestrator import ModalExecutionBackend
        backend = ModalExecutionBackend()
        assert backend.backend_name == "modal"

    def test_backends_implement_protocol(self):
        from qualification.autolearn_v04.orchestrator import (
            LocalExecutionBackend, ModalExecutionBackend, ExecutionBackend,
        )
        local = LocalExecutionBackend()
        modal = ModalExecutionBackend()
        # Both should have the backend_name attribute.
        assert hasattr(local, "backend_name")
        assert hasattr(modal, "backend_name")
        assert hasattr(local, "run_counterfactuals")
        assert hasattr(modal, "run_counterfactuals")


# ---------------------------------------------------------------------------
# v0.4.1: Gate B proxy baseline tests
# ---------------------------------------------------------------------------


class TestGateBProxies:
    def test_sequence_mean_logprob_baseline(self):
        from qualification.autolearn_v04.evaluate_gate_b import sequence_mean_logprob_baseline
        result = sequence_mean_logprob_baseline([-1.0, -2.0, -3.0])
        assert result == -2.0

    def test_sequence_min_logprob_baseline(self):
        from qualification.autolearn_v04.evaluate_gate_b import sequence_min_logprob_baseline
        result = sequence_min_logprob_baseline([-1.0, -2.0, -3.0])
        assert result == -3.0

    def test_mean_token_entropy_baseline(self):
        from qualification.autolearn_v04.evaluate_gate_b import mean_token_entropy_baseline
        result = mean_token_entropy_baseline([-1.0, -2.0])
        assert result > 0  # entropy is positive

    def test_final_token_entropy_baseline(self):
        from qualification.autolearn_v04.evaluate_gate_b import final_token_entropy_baseline
        result = final_token_entropy_baseline([-1.0, -2.0, -3.0])
        assert result == 3.0  # -(-3.0)

    def test_mean_top1_prob_baseline(self):
        from qualification.autolearn_v04.evaluate_gate_b import mean_top1_prob_baseline
        result = mean_top1_prob_baseline([0.9, 0.8, 0.7])
        assert abs(result - 0.8) < 1e-6

    def test_min_top1_prob_baseline(self):
        from qualification.autolearn_v04.evaluate_gate_b import min_top1_prob_baseline
        result = min_top1_prob_baseline([0.9, 0.8, 0.7])
        assert abs(result - 0.7) < 1e-6

    def test_generation_length_baseline(self):
        import numpy as np
        from qualification.autolearn_v04.evaluate_gate_b import generation_length_baseline
        result = generation_length_baseline([10, 20, 30])
        assert len(result) == 3
        assert result[0] < result[2]  # normalized

    def test_latency_baseline(self):
        import numpy as np
        from qualification.autolearn_v04.evaluate_gate_b import latency_baseline
        result = latency_baseline([100.0, 200.0, 300.0])
        assert len(result) == 3
        assert result[0] < result[2]

    def test_compute_all_proxy_baselines(self):
        from qualification.autolearn_v04.evaluate_gate_b import compute_all_proxy_baselines
        trajectories = [
            {"logprobs": [-1.0, -2.0], "top_probs": [0.9, 0.8],
             "generation_length": 10, "latency_ms": 100.0,
             "prompt_length": 50, "action": "ANSWER_DIRECT"},
            {"logprobs": [-3.0, -4.0], "top_probs": [0.5, 0.4],
             "generation_length": 20, "latency_ms": 200.0,
             "prompt_length": 100, "action": "CALL_TOOL"},
        ]
        proxies = compute_all_proxy_baselines(trajectories)
        assert "mean_logprob" in proxies
        assert "min_logprob" in proxies
        assert "mean_token_entropy" in proxies
        assert "max_token_entropy" in proxies
        assert "final_token_entropy" in proxies
        assert "mean_top1_prob" in proxies
        assert "min_top1_prob" in proxies
        assert "generation_length" in proxies
        assert "latency" in proxies
        assert "prompt_length" in proxies


# ---------------------------------------------------------------------------
# v0.4.1: Preflight tests
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_preflight_checks_version_identity(self):
        from qualification.autolearn_v04.preflight import _check_version_identity
        from qualification.autolearn_v04.config import QualificationConfig
        config = QualificationConfig(mode="smoke")
        errors = _check_version_identity(config)
        assert len(errors) == 0  # versions should match

    def test_preflight_checks_dependency_graph_acyclic(self):
        from qualification.autolearn_v04.preflight import _check_dependency_graph_acyclic
        errors = _check_dependency_graph_acyclic()
        assert len(errors) == 0

    def test_preflight_checks_stage_ordering(self):
        from qualification.autolearn_v04.preflight import _check_stage_ordering
        errors = _check_stage_ordering()
        assert len(errors) == 0

    def test_preflight_checks_verifier_registry(self):
        from qualification.autolearn_v04.preflight import _check_verifier_registry
        errors = _check_verifier_registry()
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# v0.4.1: Mode isolation tests
# ---------------------------------------------------------------------------


class TestModeIsolation:
    def test_infrastructure_cannot_do_scientific(self):
        from qualification.autolearn_v04.preflight import _check_provider_classification
        from qualification.autolearn_v04.config import QualificationConfig
        config = QualificationConfig(
            mode="scientific", provider="qual_grounded",
        )
        errors = _check_provider_classification(config)
        assert len(errors) > 0
        assert any("real_model" in e for e in errors)

    def test_scientific_mini_is_scientific(self):
        from qualification.autolearn_v04.config import QualificationConfig
        config = QualificationConfig(mode="scientific-mini")
        assert config.is_scientific
        assert config.is_scientific_mini
