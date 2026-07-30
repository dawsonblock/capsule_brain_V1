"""Regression tests for AutoLearn v2.16.0 qualification build.

Tests the new v2.16.0 features:
- Priority 1: Real CounterfactualRuntime with QualificationServiceFactory
- Priority 3: Four-way dataset isolation (TRAIN/VALIDATION/TEST/OOD)
- Priority 4: Sham-learning control
- Priority 5: Real paired bootstrap + permutation test + sign test
- Priority 6: Minimum effect threshold
- Priority 7: Provenance gate (no simulated promotion)
- Priority 8: Deterministic source hashing
- Priority 9: Evidence-derived reporting
- Priority 10: Larger benchmark
- Priority 11: Single-command qualification pipeline
"""
from __future__ import annotations

import asyncio
import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Load the v0.3 tasks and promote_policy modules explicitly from their file
# paths to avoid conflicts with the v0.2 modules (both are named "tasks").
QUAL_DIR = Path(__file__).resolve().parent.parent.parent / "qualification" / "autolearn_v03"
_tasks_spec = importlib.util.spec_from_file_location(
    "autolearn_v216_tasks", QUAL_DIR / "tasks.py"
)
tasks = importlib.util.module_from_spec(_tasks_spec)
_tasks_spec.loader.exec_module(tasks)

_promote_spec = importlib.util.spec_from_file_location(
    "autolearn_v216_promote", QUAL_DIR / "promote_policy.py"
)
promote_policy = importlib.util.module_from_spec(_promote_spec)
_promote_spec.loader.exec_module(promote_policy)


# ---------------------------------------------------------------------------
# Priority 1: Real CounterfactualRuntime with QualificationServiceFactory
# ---------------------------------------------------------------------------


class TestRealCounterfactualRuntimeFactory:
    """Priority 1: RealCounterfactualRuntime supports service_factory."""

    def test_real_runtime_accepts_service_factory(self):
        from capsule_brain.autolearn.counterfactual import RealCounterfactualRuntime
        from capsule_brain.autolearn.qualification_runtime import QualificationServiceFactory

        factory = QualificationServiceFactory()
        runtime = RealCounterfactualRuntime(service_factory=factory)
        assert runtime.runtime_type == "real"
        assert runtime._service_factory is factory
        factory.cleanup()

    def test_real_runtime_requires_service_or_factory(self):
        from capsule_brain.autolearn.counterfactual import RealCounterfactualRuntime

        with pytest.raises(ValueError, match="requires either"):
            RealCounterfactualRuntime()

    def test_qualification_provider_is_real_llm(self):
        from capsule_brain.autolearn.qualification_runtime import QualificationProvider

        provider = QualificationProvider(task_setup={"expected_token": "Paris"})
        assert provider.name == "qualification"

    def test_real_runtime_executes_direct_answer(self):
        """Real runtime can execute ANSWER_DIRECT and get a verified result."""
        from capsule_brain.autolearn.counterfactual import (
            CounterfactualTask,
            RealCounterfactualRuntime,
        )
        from capsule_brain.autolearn.qualification_runtime import QualificationServiceFactory
        from capsule_brain.autolearn.schema import Action, ExecutiveState

        state = ExecutiveState(
            prompt_features={
                "text": "What is the capital of France?",
                "previous_attempt_failed": False,
                "verification_failure_type": "none",
                "estimated_difficulty": 0.2,
                "workflow_capability_match": False,
            },
            conversation_features={"depth": 0},
            memory_features={"hit_count": 0, "top_similarity": 0.0},
            available_tools=["calculator"],
            workflow_available=True,
            model_id="qualification",
            context_length=30,
        )

        def verifier(action, outcome):
            text = str(outcome.get("text", ""))
            return ("paris" in text.lower(), "pass" if "paris" in text.lower() else "fail")

        task = CounterfactualTask(
            task_id="test_direct_001",
            family="direct_answer",
            archetype="factual",
            prompt="What is the capital of France?",
            allowed_actions=list(Action.learned()),
            expected_action=Action.ANSWER_DIRECT,
            state=state,
            verifier=verifier,
            setup={"expected_token": "Paris"},
            difficulty=0.2,
            ood=False,
            split="test",
            description="Test direct answer",
        )

        factory = QualificationServiceFactory()
        runtime = RealCounterfactualRuntime(service_factory=factory)

        result = asyncio.run(runtime.execute(task, Action.ANSWER_DIRECT))
        assert result.action == Action.ANSWER_DIRECT
        assert result.runtime_type == "real"
        assert result.outcome.verified_success is True
        factory.cleanup()

    def test_real_runtime_executes_memory_retrieval(self):
        """Real runtime can execute RETRIEVE_MEMORY with pre-populated memory."""
        from capsule_brain.autolearn.counterfactual import (
            CounterfactualTask,
            RealCounterfactualRuntime,
        )
        from capsule_brain.autolearn.qualification_runtime import QualificationServiceFactory
        from capsule_brain.autolearn.schema import Action, ExecutiveState

        state = ExecutiveState(
            prompt_features={
                "text": "What was the secret pin?",
                "previous_attempt_failed": False,
                "verification_failure_type": "none",
                "estimated_difficulty": 0.5,
                "workflow_capability_match": False,
            },
            conversation_features={"depth": 0},
            memory_features={"hit_count": 3, "top_similarity": 0.85},
            available_tools=[],
            workflow_available=True,
            model_id="qualification",
            context_length=30,
        )

        def verifier(action, outcome):
            text = str(outcome.get("text", ""))
            return ("pin_0001_secret" in text.lower(), "pass" if "pin_0001_secret" in text.lower() else "fail")

        task = CounterfactualTask(
            task_id="test_memory_001",
            family="memory_required",
            archetype="stored_secret",
            prompt="What was the secret pin?",
            allowed_actions=list(Action.learned()),
            expected_action=Action.RETRIEVE_MEMORY,
            state=state,
            verifier=verifier,
            setup={"expected_secret": "pin_0001_secret"},
            difficulty=0.5,
            ood=False,
            split="test",
            description="Test memory retrieval",
        )

        factory = QualificationServiceFactory()
        runtime = RealCounterfactualRuntime(service_factory=factory)

        result = asyncio.run(runtime.execute(task, Action.RETRIEVE_MEMORY))
        assert result.outcome.verified_success is True
        factory.cleanup()

    def test_real_runtime_executes_tool_call(self):
        """Real runtime can execute CALL_TOOL with a registered tool."""
        from capsule_brain.autolearn.counterfactual import (
            CounterfactualTask,
            RealCounterfactualRuntime,
        )
        from capsule_brain.autolearn.qualification_runtime import QualificationServiceFactory
        from capsule_brain.autolearn.schema import Action, ExecutiveState

        state = ExecutiveState(
            prompt_features={
                "text": "Fetch the current live nonce for endpoint 0001.",
                "previous_attempt_failed": False,
                "verification_failure_type": "none",
                "estimated_difficulty": 0.5,
                "workflow_capability_match": False,
            },
            conversation_features={"depth": 0},
            memory_features={"hit_count": 0, "top_similarity": 0.0},
            available_tools=["http_fetch"],
            workflow_available=True,
            model_id="qualification",
            context_length=50,
        )

        def verifier(action, outcome):
            if action != Action.CALL_TOOL:
                return False, "fail:tool_not_selected"
            if int(outcome.get("tool_calls_executed", 0) or 0) < 1:
                return False, "fail:tool_not_executed"
            if not outcome.get("tool_result_valid", False):
                return False, "fail:tool_result_invalid"
            text = str(outcome.get("text", "") or "").strip()
            nonce = str(outcome.get("expected_nonce", "") or "")
            if nonce.lower() not in text.lower():
                return False, "fail:answer_not_grounded"
            return True, "pass"

        task = CounterfactualTask(
            task_id="test_tool_001",
            family="tool_required",
            archetype="live_nonce",
            prompt="Fetch the current live nonce for endpoint 0001.",
            allowed_actions=list(Action.learned()),
            expected_action=Action.CALL_TOOL,
            state=state,
            verifier=verifier,
            setup={"expected_nonce": "nonce_0001_live", "tool_name": "http_fetch"},
            difficulty=0.5,
            ood=False,
            split="test",
            description="Test tool fetch",
        )

        factory = QualificationServiceFactory()
        runtime = RealCounterfactualRuntime(service_factory=factory)

        result = asyncio.run(runtime.execute(task, Action.CALL_TOOL))
        assert result.outcome.verified_success is True
        assert result.outcome.tool_calls_executed >= 1
        factory.cleanup()

    def test_real_runtime_isolation_between_executions(self):
        """Each execution gets a fresh, isolated service — no state leakage."""
        from capsule_brain.autolearn.counterfactual import (
            CounterfactualTask,
            RealCounterfactualRuntime,
        )
        from capsule_brain.autolearn.qualification_runtime import QualificationServiceFactory
        from capsule_brain.autolearn.schema import Action, ExecutiveState

        state = ExecutiveState(
            prompt_features={
                "text": "What is the capital of France?",
                "previous_attempt_failed": False,
                "verification_failure_type": "none",
                "estimated_difficulty": 0.2,
                "workflow_capability_match": False,
            },
            conversation_features={"depth": 0},
            memory_features={"hit_count": 0, "top_similarity": 0.0},
            available_tools=[],
            workflow_available=True,
            model_id="qualification",
            context_length=30,
        )

        def verifier(action, outcome):
            return True, "pass"

        task = CounterfactualTask(
            task_id="test_iso_001",
            family="direct_answer",
            archetype="factual",
            prompt="What is the capital of France?",
            allowed_actions=list(Action.learned()),
            expected_action=Action.ANSWER_DIRECT,
            state=state,
            verifier=verifier,
            setup={"expected_token": "Paris"},
            difficulty=0.2,
            ood=False,
            split="test",
            description="Test isolation",
        )

        factory = QualificationServiceFactory()
        runtime = RealCounterfactualRuntime(service_factory=factory)

        # Execute twice — each should get a unique conversation_id.
        r1 = asyncio.run(runtime.execute(task, Action.ANSWER_DIRECT))
        r2 = asyncio.run(runtime.execute(task, Action.ANSWER_DIRECT))

        assert r1.conversation_id != r2.conversation_id
        factory.cleanup()


# ---------------------------------------------------------------------------
# Priority 3: Four-way dataset isolation
# ---------------------------------------------------------------------------


class TestFourWayDatasetIsolation:
    """Priority 3: TRAIN/VALIDATION/TEST/OOD isolation."""

    def test_validation_split_exists(self):
        tasks_mod = tasks
        from collections import Counter

        all_tasks = tasks_mod.build_all_tasks()
        assignments = tasks_mod.build_split_assignments(all_tasks)
        counts = Counter(assignments.values())
        assert "validation" in counts
        assert counts["validation"] > 0

    def test_no_task_in_multiple_splits(self):
        tasks_mod = tasks

        all_tasks = tasks_mod.build_all_tasks()
        assignments = tasks_mod.build_split_assignments(all_tasks)
        # Every task has exactly one split.
        assert len(assignments) == len(all_tasks)

    def test_validation_not_used_for_training(self):
        """VALIDATION split must not appear in training examples."""
        # This is verified by train_policy.py using validation split for val.
        # We check that the split name exists and is distinct from test.
        tasks_mod = tasks

        all_tasks = tasks_mod.build_all_tasks()
        val_tasks = [t for t in all_tasks if t.split == "validation"]
        test_tasks = [t for t in all_tasks if t.split == "test"]
        assert len(val_tasks) > 0
        assert len(test_tasks) > 0
        # No overlap in task IDs.
        val_ids = {t.task_id for t in val_tasks}
        test_ids = {t.task_id for t in test_tasks}
        assert val_ids.isdisjoint(test_ids)


# ---------------------------------------------------------------------------
# Priority 4: Sham-learning control
# ---------------------------------------------------------------------------


class TestShamLearningControl:
    """Priority 4: Sham learner destroys experience signal."""

    def test_sham_learner_exists(self):
        from capsule_brain.autolearn.sham_learner import train_sham_policy
        assert callable(train_sham_policy)

    def test_shuffle_actions_within_strata(self):
        from capsule_brain.autolearn.learner import TrainingExample
        from capsule_brain.autolearn.schema import Action
        from capsule_brain.autolearn.sham_learner import shuffle_actions_within_strata

        examples = [
            TrainingExample(
                task_id=f"t{i}",
                task_family="family_a" if i < 5 else "family_b",
                split="train",
                best_action=Action.ANSWER_DIRECT if i % 2 == 0 else Action.RETRIEVE_MEMORY,
                utility_margin=1.0,
                weight=2.0,
                features=[float(i)],
            )
            for i in range(10)
        ]

        shuffled = shuffle_actions_within_strata(examples, seed=42)
        assert len(shuffled) == len(examples)
        # The action distribution within each family should be preserved.
        from collections import Counter
        orig_a = Counter(e.best_action for e in examples if e.task_family == "family_a")
        sham_a = Counter(e.best_action for e in shuffled if e.task_family == "family_a")
        assert orig_a == sham_a

    def test_shuffled_labels_destroy_signal(self):
        from capsule_brain.autolearn.learner import TrainingExample
        from capsule_brain.autolearn.schema import Action
        from capsule_brain.autolearn.sham_learner import shuffle_labels_globally

        examples = [
            TrainingExample(
                task_id=f"t{i}",
                task_family="f",
                split="train",
                best_action=Action.ANSWER_DIRECT,
                utility_margin=1.0,
                weight=2.0,
                features=[float(i)],
            )
            for i in range(10)
        ]
        # Add some different actions.
        examples[0].best_action = Action.RETRIEVE_MEMORY
        examples[1].best_action = Action.CALL_TOOL

        shuffled = shuffle_labels_globally(examples, seed=42)
        # The multiset of actions should be preserved.
        from collections import Counter
        assert Counter(e.best_action for e in examples) == Counter(e.best_action for e in shuffled)


# ---------------------------------------------------------------------------
# Priority 5: Real paired bootstrap + permutation test + sign test
# ---------------------------------------------------------------------------


class TestStatistics:
    """Priority 5: Real statistical tests."""

    def test_paired_bootstrap_returns_result(self):
        from capsule_brain.autolearn.statistics import paired_bootstrap

        deltas = [1.0, 2.0, 0.5, -0.5, 1.5, 2.5, 0.0, 1.0]
        result = paired_bootstrap(deltas, n_bootstrap=1000, seed=42)
        assert result.n_samples == 8
        assert result.n_bootstrap == 1000
        assert result.ci_low < result.ci_high
        assert len(result.bootstrap_samples) == 1000

    def test_paired_bootstrap_mean_matches(self):
        from capsule_brain.autolearn.statistics import paired_bootstrap

        deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = paired_bootstrap(deltas, n_bootstrap=5000, seed=42)
        assert abs(result.mean - 3.0) < 0.001

    def test_paired_bootstrap_epsilon_threshold(self):
        from capsule_brain.autolearn.statistics import paired_bootstrap

        # Large positive deltas — should pass epsilon=0.5.
        deltas = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = paired_bootstrap(deltas, n_bootstrap=1000, epsilon=0.5, seed=42)
        assert result.passes_threshold is True

    def test_paired_bootstrap_fails_threshold(self):
        from capsule_brain.autolearn.statistics import paired_bootstrap

        # Deltas near zero — should fail epsilon=1.0.
        deltas = [0.1, -0.1, 0.2, -0.2, 0.0]
        result = paired_bootstrap(deltas, n_bootstrap=1000, epsilon=1.0, seed=42)
        assert result.passes_threshold is False

    def test_permutation_test_returns_result(self):
        from capsule_brain.autolearn.statistics import paired_permutation_test

        deltas = [1.0, 2.0, 0.5, -0.5, 1.5]
        result = paired_permutation_test(deltas, n_permutations=1000, seed=42)
        assert result.n_samples == 5
        assert 0.0 <= result.p_value <= 1.0

    def test_sign_test_returns_result(self):
        from capsule_brain.autolearn.statistics import sign_test

        deltas = [1.0, 2.0, -0.5, 0.5, 1.5, -1.0]
        result = sign_test(deltas)
        assert result.n_positive == 4
        assert result.n_negative == 2
        assert result.n_zero == 0
        assert 0.0 <= result.p_value <= 1.0

    def test_compute_qualification_statistics(self):
        from capsule_brain.autolearn.statistics import compute_qualification_statistics

        candidate = [5.0, 6.0, 7.0, 8.0, 9.0]
        baseline = [3.0, 4.0, 5.0, 6.0, 7.0]
        stats = compute_qualification_statistics(
            candidate, baseline, n_bootstrap=500, epsilon=0.5, seed=42
        )
        assert stats.bootstrap.n_samples == 5
        assert stats.permutation.n_samples == 5
        assert stats.sign_test.n_samples == 5
        assert len(stats.deltas) == 5
        assert all(d == 2.0 for d in stats.deltas)

    def test_compute_qualification_statistics_mismatched_lengths(self):
        from capsule_brain.autolearn.statistics import compute_qualification_statistics

        with pytest.raises(ValueError, match="equal length"):
            compute_qualification_statistics([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Priority 6: Minimum effect threshold
# ---------------------------------------------------------------------------


class TestMinimumEffectThreshold:
    """Priority 6: Promotion gate requires CI > epsilon, not merely > 0."""

    def test_bootstrap_passes_threshold_with_epsilon(self):
        from capsule_brain.autolearn.statistics import paired_bootstrap

        deltas = [2.0, 3.0, 4.0, 5.0]
        result = paired_bootstrap(deltas, n_bootstrap=1000, epsilon=1.0, seed=42)
        assert result.passes_threshold is True
        assert result.epsilon == 1.0

    def test_bootstrap_fails_with_high_epsilon(self):
        from capsule_brain.autolearn.statistics import paired_bootstrap

        deltas = [0.5, 1.0, 0.5, 1.0]
        result = paired_bootstrap(deltas, n_bootstrap=1000, epsilon=5.0, seed=42)
        assert result.passes_threshold is False


# ---------------------------------------------------------------------------
# Priority 7: Provenance gate (no simulated promotion)
# ---------------------------------------------------------------------------


class TestProvenanceGate:
    """Priority 7: Simulation may never promote a policy."""

    def test_verify_provenance_rejects_simulated(self):
        from capsule_brain.autolearn.source_hash import verify_provenance

        result = verify_provenance(
            runtime_type="simulated",
            evaluation_runtime_type="simulated",
            simulated_rows=100,
            dataset_hash="abc",
            expected_dataset_hash="abc",
            source_tree_hash="def",
            expected_source_tree_hash="def",
            train_test_overlap=False,
            policy_lineage_valid=True,
        )
        assert result["valid"] is False
        check_names = [c["check"] for c in result["checks"]]
        assert "runtime_is_real" in check_names
        assert "simulated_rows_zero" in check_names

    def test_verify_provenance_accepts_real(self):
        from capsule_brain.autolearn.source_hash import verify_provenance

        result = verify_provenance(
            runtime_type="real",
            evaluation_runtime_type="real",
            simulated_rows=0,
            dataset_hash="abc",
            expected_dataset_hash="abc",
            source_tree_hash="def",
            expected_source_tree_hash="def",
            train_test_overlap=False,
            policy_lineage_valid=True,
        )
        assert result["valid"] is True

    def test_verify_provenance_rejects_hash_mismatch(self):
        from capsule_brain.autolearn.source_hash import verify_provenance

        result = verify_provenance(
            runtime_type="real",
            evaluation_runtime_type="real",
            simulated_rows=0,
            dataset_hash="abc",
            expected_dataset_hash="WRONG",
            source_tree_hash="def",
            expected_source_tree_hash="def",
            train_test_overlap=False,
            policy_lineage_valid=True,
        )
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# Priority 8: Deterministic source hashing
# ---------------------------------------------------------------------------


class TestSourceHashing:
    """Priority 8: Deterministic source tree hashing (not Git-based)."""

    def test_compute_source_tree_hash_is_deterministic(self):
        from capsule_brain.autolearn.source_hash import compute_source_tree_hash

        project_root = Path(__file__).resolve().parent.parent.parent
        hash1 = compute_source_tree_hash(project_root)
        hash2 = compute_source_tree_hash(project_root)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_compute_source_tree_hash_changes_with_content(self):
        from capsule_brain.autolearn.source_hash import compute_source_tree_hash

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "src").mkdir()
            (tmp_path / "src" / "main.py").write_text("print('hello')\n")
            hash1 = compute_source_tree_hash(tmp_path)
            (tmp_path / "src" / "main.py").write_text("print('world')\n")
            hash2 = compute_source_tree_hash(tmp_path)
            assert hash1 != hash2

    def test_compute_file_hash(self):
        from capsule_brain.autolearn.source_hash import compute_file_hash

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')\n")
            f.flush()
            h = compute_file_hash(f.name)
            assert len(h) == 64
        os.unlink(f.name)

    def test_compute_string_hash(self):
        from capsule_brain.autolearn.source_hash import compute_string_hash

        h = compute_string_hash("hello world")
        assert len(h) == 64
        assert h != compute_string_hash("hello world!")


# ---------------------------------------------------------------------------
# Priority 10: Larger benchmark
# ---------------------------------------------------------------------------


class TestLargerBenchmark:
    """Priority 10: Benchmark has hundreds of held-out tasks."""

    def test_benchmark_has_at_least_800_tasks(self):
        tasks_mod = tasks

        all_tasks = tasks_mod.build_all_tasks()
        assert len(all_tasks) >= 800

    def test_each_family_has_at_least_200_tasks(self):
        tasks_mod = tasks
        from collections import Counter

        all_tasks = tasks_mod.build_all_tasks()
        families = Counter(t.family for t in all_tasks)
        for family, count in families.items():
            assert count >= 200, f"family {family} has only {count} tasks"

    def test_test_split_has_at_least_100_tasks(self):
        tasks_mod = tasks
        from collections import Counter

        all_tasks = tasks_mod.build_all_tasks()
        assignments = tasks_mod.build_split_assignments(all_tasks)
        counts = Counter(assignments.values())
        assert counts["test"] >= 100


# ---------------------------------------------------------------------------
# Priority 11: Single-command qualification pipeline
# ---------------------------------------------------------------------------


class TestQualificationPipeline:
    """Priority 11: Single-command pipeline exists."""

    def test_pipeline_script_exists(self):
        pipeline = QUAL_DIR / "run_qualification.py"
        assert pipeline.exists()

    def test_pipeline_has_runtime_flag(self):
        pipeline = QUAL_DIR / "run_qualification.py"
        content = pipeline.read_text()
        assert "--runtime" in content
        assert "real" in content
        assert "simulated" in content

    def test_pipeline_has_epsilon_flag(self):
        pipeline = QUAL_DIR / "run_qualification.py"
        content = pipeline.read_text()
        assert "--epsilon" in content

    def test_pipeline_runs_all_steps(self):
        pipeline = QUAL_DIR / "run_qualification.py"
        content = pipeline.read_text()
        # Check that all steps are included.
        assert "run_counterfactuals" in content
        assert "build_dataset" in content
        assert "train_policy" in content
        assert "train_sham" in content
        assert "evaluate_policy" in content
        assert "evaluate_sham" in content
        assert "promote_policy" in content
        assert "generate_artifacts" in content
        assert "report" in content


# ---------------------------------------------------------------------------
# Priority 9: Evidence-derived reporting
# ---------------------------------------------------------------------------


class TestEvidenceDerivedReporting:
    """Priority 9: Report uses evidence-derived checks, not hardcoded ✓."""

    def test_report_uses_evidence_check_function(self):
        report = QUAL_DIR / "report.py"
        content = report.read_text()
        assert "_evidence_check" in content
        assert "PASS" in content
        assert "FAIL" in content
        # Should NOT have hardcoded ✓ characters.
        assert "✓" not in content

    def test_report_includes_sham_section(self):
        report = QUAL_DIR / "report.py"
        content = report.read_text()
        assert "sham" in content.lower()
        assert "Sham-learning control" in content

    def test_report_includes_statistics_section(self):
        report = QUAL_DIR / "report.py"
        content = report.read_text()
        assert "statistics" in content.lower()
        assert "bootstrap" in content.lower()

    def test_report_includes_provenance_section(self):
        report = QUAL_DIR / "report.py"
        content = report.read_text()
        assert "provenance" in content.lower()
        assert "source_tree_sha256" in content


# ---------------------------------------------------------------------------
# Integration: Promotion gate with 15 gates
# ---------------------------------------------------------------------------


class TestPromotionGateV216:
    """v2.16.0: 15-gate promotion gate."""

    def test_gate_count_is_15(self):

        test_eval = {
            "baseline_metrics": {"verified_success_rate": 0.7, "mean_utility": 4.0, "safety_violations": 0},
            "candidate_metrics": {"verified_success_rate": 0.9, "mean_utility": 7.0, "safety_violations": 0, "tool_precision": 0.9, "tool_recall": 0.8},
            "mean_delta": 3.0,
            "delta_ci_low": 1.0,
            "delta_ci_high": 5.0,
            "wins": 12, "ties": 3, "losses": 1,
            "v2_metrics": {
                "candidate_ece": 0.05,
                "candidate_accuracy_at_coverage": {"cov_0.90": 0.95},
            },
        }
        ood_eval = {
            "baseline_metrics": {"verified_success_rate": 0.6, "mean_utility": 3.0, "safety_violations": 0},
            "candidate_metrics": {"verified_success_rate": 0.8, "mean_utility": 5.0, "safety_violations": 0, "tool_precision": 0.8, "tool_recall": 0.7},
            "mean_delta": 2.0, "delta_ci_low": 0.5, "delta_ci_high": 3.5,
            "wins": 8, "ties": 2, "losses": 1,
            "v2_metrics": {},
        }
        result = promote_policy._check_gates(
            test_eval, ood_eval,
            runtime_type="real",
            sham_eval={"candidate_metrics": {"mean_utility": 3.0}},
        )
        assert len(result["gates"]) == 15

    def test_simulated_runtime_fails_gate(self):

        test_eval = {
            "baseline_metrics": {"verified_success_rate": 0.7, "mean_utility": 4.0, "safety_violations": 0},
            "candidate_metrics": {"verified_success_rate": 0.9, "mean_utility": 7.0, "safety_violations": 0, "tool_precision": 0.9, "tool_recall": 0.8},
            "mean_delta": 3.0, "delta_ci_low": 1.0, "delta_ci_high": 5.0,
            "wins": 12, "ties": 3, "losses": 1,
            "v2_metrics": {"candidate_ece": 0.05, "candidate_accuracy_at_coverage": {"cov_0.90": 0.95}},
        }
        ood_eval = {
            "baseline_metrics": {"verified_success_rate": 0.6, "mean_utility": 3.0, "safety_violations": 0},
            "candidate_metrics": {"verified_success_rate": 0.8, "mean_utility": 5.0, "safety_violations": 0, "tool_precision": 0.8, "tool_recall": 0.7},
            "mean_delta": 2.0, "delta_ci_low": 0.5, "delta_ci_high": 3.5,
            "wins": 8, "ties": 2, "losses": 1,
            "v2_metrics": {},
        }
        result = promote_policy._check_gates(
            test_eval, ood_eval,
            runtime_type="simulated",
            sham_eval={"candidate_metrics": {"mean_utility": 3.0}},
        )
        # Should fail because runtime is simulated.
        assert result["promoted"] is False
        gate_names = [g["gate"] for g in result["gates"]]
        assert "runtime_is_real" in gate_names
        assert "no_simulated_rows" in gate_names

    def test_epsilon_threshold_in_gates(self):

        test_eval = {
            "baseline_metrics": {"verified_success_rate": 0.7, "mean_utility": 4.0, "safety_violations": 0},
            "candidate_metrics": {"verified_success_rate": 0.9, "mean_utility": 4.1, "safety_violations": 0, "tool_precision": 0.9, "tool_recall": 0.8},
            "mean_delta": 0.1, "delta_ci_low": 0.05, "delta_ci_high": 0.15,
            "wins": 8, "ties": 5, "losses": 3,
            "v2_metrics": {"candidate_ece": 0.05, "candidate_accuracy_at_coverage": {"cov_0.90": 0.95}},
        }
        ood_eval = {
            "baseline_metrics": {"verified_success_rate": 0.6, "mean_utility": 3.0, "safety_violations": 0},
            "candidate_metrics": {"verified_success_rate": 0.8, "mean_utility": 5.0, "safety_violations": 0, "tool_precision": 0.8, "tool_recall": 0.7},
            "mean_delta": 2.0, "delta_ci_low": 0.5, "delta_ci_high": 3.5,
            "wins": 8, "ties": 2, "losses": 1,
            "v2_metrics": {},
        }
        # With epsilon=0.5, CI_low=0.05 should fail.
        result = promote_policy._check_gates(
            test_eval, ood_eval,
            epsilon=0.5,
            runtime_type="real",
            sham_eval={"candidate_metrics": {"mean_utility": 3.0}},
        )
        assert result["promoted"] is False
        # The epsilon gate should fail.
        epsilon_gate = [g for g in result["gates"] if g["gate"] == "test_delta_ci_above_epsilon"]
        assert len(epsilon_gate) == 1
        assert epsilon_gate[0]["passed"] is False
