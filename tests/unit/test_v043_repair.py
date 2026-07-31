"""Tests for AutoLearn v0.4.3 repair modules.

Covers all v0.4.3 repair analysis modules:
- Canonical policy evaluation (evaluate_all_policies, result_to_dict)
- Metric consistency (check_metric_consistency)
- Sham identity (validate_sham_identity)
- Gate A0 audit (audit_gate_a0)
- Decision trace (reconstruct_decision_trace)
- Raw vs executed (compare_raw_vs_executed)
- Calibration fallback audit (audit_calibration_fallback)
- Headroom concentration (analyze_headroom_concentration)
- High margin recovery (evaluate_high_margin_recovery)
- Feature signal (analyze_feature_signal)
- Missing state analysis (analyze_missing_state)
- Feature registry (build_feature_registry)
- Learner validation (validate_learners)
- Soft utility targets (compare_soft_targets)
- Class balance (analyze_class_balance)
- Sham ensemble (run_sham_ensemble)
- Candidate stability (analyze_candidate_stability)
- Parameter diagnostics (diagnose_parameters)
- Utility consistency (audit_utility_consistency)
- Counterfactual completeness (audit_counterfactual_completeness)
- Gate A layers (evaluate_gate_a_v043)
- Scale decision (make_scale_decision_v043)
- Matched pilot manifest (check_pilot_eligibility, generate_pilot_manifest)
- Report generator (generate_v043_report)
"""
from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path
from collections import Counter

from qualification.autolearn_v04.v043.data import (
    LoadedRun,
    TaskInfo,
    Outcome,
    PolicyEvaluation,
)
from qualification.autolearn_v04.v043.canonical_evaluator import (
    evaluate_all_policies,
    evaluate_policy_from_counterfactuals,
    result_to_dict,
    make_oracle_fn,
    make_baseline_fn,
    make_majority_fn,
    make_raw_candidate_fn,
    make_executed_candidate_fn,
    make_executed_sham_fn,
    make_frequency_fn,
    make_random_fn,
    compute_headroom_metrics,
    PolicyEvalResult,
)
from qualification.autolearn_v04.v043.config import (
    PACKAGE_VERSION,
    QUALIFICATION_VERSION,
    MINIMUM_DENOMINATOR,
    HEADROOM_PERCENTAGE_SCALE,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEED,
    A1_HEADROOM_THRESHOLD,
    A3_PRACTICAL_THRESHOLD,
    A4_PRACTICAL_THRESHOLD,
    A5_MIN_RECOVERY_FRACTION,
    METRIC_TOLERANCE_FLOAT,
    METRIC_TOLERANCE_STRICT,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Production feature schema — needed for LearnedPolicy.from_dict.
_FEATURE_NAMES = [
    "prompt_length",
    "log_prompt_length",
    "code_indicators",
    "arithmetic_indicators",
    "structured_output_request",
    "tool_required_keywords",
    "retrieval_indicators",
    "prior_conversation_depth",
    "semantic_memory_hit_count",
    "semantic_memory_similarity",
    "num_available_tools",
    "workflow_available",
    "previous_attempt_failed",
    "verification_failure_type_id",
    "model_identity_id",
    "estimated_difficulty",
    "workflow_capability_match",
]
_FEATURE_SCHEMA_VERSION = "exec_features_v2"
_ACTIONS = ["ANSWER_DIRECT", "RETRIEVE_MEMORY", "CALL_TOOL", "START_WORKFLOW"]


def _make_policy_dict(actions: list[str], n_features: int = 17) -> dict:
    """Build a policy dict compatible with LearnedPolicy.from_dict."""
    rng = np.random.default_rng(42)
    W = rng.standard_normal((len(actions), n_features)) * 0.01
    b = np.zeros(len(actions))
    return {
        "policy_id": "test_policy",
        "policy_version": "learned_router_v2",
        "policy_type": "multinomial_logistic",
        "feature_schema_version": _FEATURE_SCHEMA_VERSION,
        "feature_names": list(_FEATURE_NAMES),
        "actions": actions,
        "weights": W.tolist(),
        "bias": b.tolist(),
        "confidence_threshold": 0.5,
        "temperature": 1.0,
        "training_data_digest": "abc123",
        "split_manifest_digest": "def456",
        "parent_policy": "",
        "hyperparameters": {},
        "metrics": {},
        "trained_at": "2024-01-01T00:00:00Z",
        "protocol_version": "0.4.3",
        "provenance": {
            "model_id": "qual-grounded-v04",
            "benchmark_manifest_digest": "bench_digest_abc",
            "protocol_version": "0.4.3",
            "policy_id": "test_policy",
        },
    }


def _make_mini_run() -> LoadedRun:
    """Create a small synthetic LoadedRun for testing v0.4.3 modules.

    The run has:
    - 30 tasks across experience/validation/test splits (10 each)
    - 4 actions with deterministic utility structure
    - Counterfactual outcomes for every (task, action) pair
    - Candidate and sham policies compatible with LearnedPolicy.from_dict
    - Evaluation artifacts for all four standard policies
    - SHA-256 digests for all lineage artifacts
    - Training, validation, and test examples with 17-dim feature vectors
    """
    run = LoadedRun(
        artifacts_dir=Path("/tmp/test_v043"),
        benchmark_manifest={"tasks": [], "protocol_version": "0.4.3"},
        split_manifest={"splits": {}, "protocol_version": "0.4.3"},
    )
    run.all_actions = list(_ACTIONS)
    run.all_families = ["direct_answer", "memory_required", "tool_required", "workflow_required"]
    run.all_archetypes = [
        "direct_answer:experience:comparison",
        "memory_required:experience:recall",
        "tool_required:test:execution",
        "workflow_required:validation:orchestration",
    ]
    run.all_splits = ["experience", "validation", "test"]

    # --- Tasks ---
    task_counter = 0
    for split in ("experience", "validation", "test"):
        for i in range(10):
            fam_idx = task_counter % 4
            family = run.all_families[fam_idx]
            archetype = f"{family}:{split}:variant_{i}"
            tid = f"{split}_{family}_{i:04d}"
            ti = TaskInfo(
                task_id=tid,
                split=split,
                family=family,
                archetype=archetype,
                allowed_actions=list(run.all_actions),
                verifier_spec={"type": "direct_exact"},
                prompt=f"Test prompt {task_counter} for {family}",
                generator_family="gen",
                template_family="tmpl",
                capability_family="cap",
                entity_family="ent",
                risk_class="benign",
                crossover_type="none",
                group_id=f"group_{task_counter}",
                generator_seed=task_counter,
                setup_spec={"secret": f"secret_{task_counter}"},
                expected_output_digest=f"digest_{task_counter}",
            )
            run.tasks[tid] = ti
            task_counter += 1

    # --- Benchmark manifest with tasks ---
    bm_tasks = []
    for tid, task in run.tasks.items():
        bm_tasks.append({
            "task_id": tid,
            "split": task.split,
            "family": task.family,
            "archetype": task.archetype,
            "allowed_actions": task.allowed_actions,
            "verifier_spec": task.verifier_spec,
            "prompt": task.prompt,
            "generator_family": task.generator_family,
            "template_family": task.template_family,
            "capability_family": task.capability_family,
            "entity_family": task.entity_family,
            "risk_class": task.risk_class,
            "crossover_type": task.crossover_type,
            "group_id": task.group_id,
            "generator_seed": task.generator_seed,
            "setup_spec": task.setup_spec,
            "expected_output_digest": task.expected_output_digest,
        })
    run.benchmark_manifest = {
        "tasks": bm_tasks,
        "protocol_version": "0.4.3",
        "provider_class": "modal_gpu",
    }

    # --- Split manifest ---
    test_ids = sorted(tid for tid, t in run.tasks.items() if t.split == "test")
    val_ids = sorted(tid for tid, t in run.tasks.items() if t.split == "validation")
    train_ids = sorted(tid for tid, t in run.tasks.items() if t.split == "experience")
    run.split_manifest = {
        "test_task_ids": test_ids,
        "validation_task_ids": val_ids,
        "train_task_ids": train_ids,
        "protocol_version": "0.4.3",
    }

    # --- Outcomes ---
    # Each family has a "best" action that gets utility 10, others get 1.
    best_action_map = {
        "direct_answer": "ANSWER_DIRECT",
        "memory_required": "RETRIEVE_MEMORY",
        "tool_required": "CALL_TOOL",
        "workflow_required": "START_WORKFLOW",
    }
    for tid, task in run.tasks.items():
        best_action = best_action_map.get(task.family, "ANSWER_DIRECT")
        for action in run.all_actions:
            if action == best_action:
                utility = 10.0
                verification = "success"
            else:
                utility = 1.0
                verification = "failure"
            o = Outcome(
                task_id=tid,
                action_id=action,
                availability="executed",
                verification=verification,
                utility=utility,
                execution_metadata={
                    "model_id": "qual-grounded-v04",
                    "model_revision": "abc",
                    "latency_ms": 100,
                    "tokens_used": 50,
                },
                reward_components={
                    "correctness": 1.0 if action == best_action else 0.0,
                    "latency_penalty": 0.01,
                    "token_penalty": 0.01,
                },
            )
            run.outcomes.append(o)
            run.outcomes_by_task.setdefault(tid, []).append(o)
            run.outcomes_by_task_action[(tid, action)] = o

    # --- Policies ---
    run.candidate_policy = _make_policy_dict(run.all_actions)
    # Slightly different weights for sham.
    sham_dict = _make_policy_dict(run.all_actions)
    sham_dict["policy_id"] = "test_sham"
    sham_dict["is_sham"] = True
    sham_dict["sham_seed"] = 12345
    sham_dict["provenance"]["policy_id"] = "test_sham"
    run.sham_policy = sham_dict

    # --- Training metadata ---
    run.candidate_training = {
        "n_train": 10,
        "train_accuracy": 0.8,
        "train_loss": 0.5,
        "protocol_version": "0.4.3",
        "model_id": "qual-grounded-v04",
        "benchmark_manifest_digest": "bench_digest_abc",
    }
    run.sham_training = {
        "n_train": 10,
        "train_accuracy": 0.3,
        "train_loss": 1.5,
        "shuffle_mode": "global",
        "seed": 12345,
        "protocol_version": "0.4.3",
        "model_id": "qual-grounded-v04",
        "benchmark_manifest_digest": "bench_digest_abc",
    }

    # --- Gate A result ---
    run.gate_a_result = {
        "protocol_version": "0.4.3",
        "overall_status": "FAIL",
        "mean_utility": {"candidate": 5.5, "sham": 5.5, "baseline": 5.5, "oracle": 10.0},
        "delta_candidate_baseline": {"mean": 0.0, "ci_low": -0.5, "ci_high": 0.5},
        "delta_candidate_sham": {"mean": 0.0, "ci_low": -0.5, "ci_high": 0.5},
        "delta_oracle_sham": {"mean": 4.5},
        "recovered_headroom": {"fraction": 0.0, "percent": 0.0},
    }

    # --- Qualification report ---
    run.qualification_report = {
        "autolearn_version": "0.3.6",
        "provider": "modal_gpu",
        "provenance": {
            "model_id": "qual-grounded-v04",
            "benchmark_manifest_digest": "bench_digest_abc",
        },
        "protocol_version": "0.4.3",
        "recovered_headroom": {"fraction": 0.0, "percent": 0.0},
    }

    # --- Dataset examples ---
    def _make_example(tid: str, task: TaskInfo, split: str) -> dict:
        best = best_action_map.get(task.family, "ANSWER_DIRECT")
        # 17-dim feature vector.
        feats = [
            float(len(task.prompt)),   # prompt_length
            float(np.log(len(task.prompt) + 1)),  # log_prompt_length
            1.0 if "code" in task.prompt.lower() else 0.0,  # code_indicators
            0.0,  # arithmetic_indicators
            0.0,  # structured_output_request
            0.0,  # tool_required_keywords
            0.0,  # retrieval_indicators
            0.0,  # prior_conversation_depth
            0.0,  # semantic_memory_hit_count
            0.0,  # semantic_memory_similarity
            1.0,  # num_available_tools
            1.0,  # workflow_available
            0.0,  # previous_attempt_failed
            0.0,  # verification_failure_type_id
            0.0,  # model_identity_id
            0.5,  # estimated_difficulty
            1.0 if task.family == "workflow_required" else 0.0,  # workflow_capability_match
        ]
        return {
            "task_id": tid,
            "best_action": best,
            "weight": 2.0,
            "split": split,
            "features": feats,
            "task_family": task.family,
            "utility_margin": 9.0,
        }

    for tid, task in run.tasks.items():
        ex = _make_example(tid, task, task.split)
        if task.split == "experience":
            run.train_examples.append(ex)
        elif task.split == "validation":
            run.validation_examples.append(ex)
        elif task.split == "test":
            run.test_examples.append(ex)

    # --- Evaluation artifacts ---
    def _make_eval(policy_id: str, mean_u: float, task_ids: list[str]) -> PolicyEvaluation:
        per_task = {}
        for tid in task_ids:
            task = run.tasks[tid]
            best = best_action_map.get(task.family, "ANSWER_DIRECT")
            per_task[tid] = {
                "task_id": tid,
                "action": best,
                "utility": run.get_utility_for_action(tid, best) or 0.0,
                "success": True,
            }
        return PolicyEvaluation(
            policy_id=policy_id,
            split="test",
            mean_utility=mean_u,
            success_rate=1.0,
            n_complete_tasks=len(task_ids),
            n_manifest_tasks=len(task_ids),
            n_missing_tasks=0,
            per_task=per_task,
            metrics={"task_results": [{"task_id": t, "action": per_task[t]["action"], "utility": per_task[t]["utility"]} for t in task_ids]},
            raw_artifact={"split_name": "test", "mean_utility": mean_u},
        )

    run.eval_candidate = _make_eval("candidate", 5.5, test_ids)
    run.eval_sham = _make_eval("sham", 5.5, test_ids)
    run.eval_baseline = _make_eval("baseline", 5.5, test_ids)
    run.eval_oracle = _make_eval("oracle", 10.0, test_ids)

    # --- SHA-256 digests ---
    run.benchmark_manifest_sha256 = "a" * 64
    run.split_manifest_sha256 = "b" * 64
    run.counterfactual_results_sha256 = "c" * 64
    run.candidate_policy_sha256 = "d" * 64
    run.sham_policy_sha256 = "e" * 64
    run.gate_a_result_sha256 = "f" * 64
    run.qualification_report_sha256 = "0" * 64

    return run


@pytest.fixture
def mini_run():
    return _make_mini_run()


@pytest.fixture
def canonical_results(mini_run):
    """Pre-compute canonical results for modules that depend on them."""
    return evaluate_all_policies(mini_run, seed=DEFAULT_SEED)


# ---------------------------------------------------------------------------
# A. Canonical policy evaluation
# ---------------------------------------------------------------------------


class TestCanonicalEvaluator:
    """Test canonical policy evaluation."""

    def test_evaluate_all_policies_returns_all_keys(self, mini_run):
        results = evaluate_all_policies(mini_run, seed=DEFAULT_SEED)
        expected_keys = {
            "oracle", "baseline", "majority", "frequency",
            "random", "candidate_raw", "candidate_executed", "sham",
        }
        assert set(results.keys()) == expected_keys

    def test_oracle_mean_is_highest(self, mini_run):
        results = evaluate_all_policies(mini_run, seed=DEFAULT_SEED)
        oracle_u = results["oracle"].mean_utility
        for name, res in results.items():
            if name == "oracle":
                continue
            assert oracle_u >= res.mean_utility - 1e-9

    def test_result_to_dict_is_serializable(self, mini_run):
        results = evaluate_all_policies(mini_run, seed=DEFAULT_SEED)
        for name, res in results.items():
            d = result_to_dict(res)
            # Must be JSON-serializable.
            json.dumps(d)
            assert d["policy_id"] == res.policy_id
            assert "mean_utility" in d
            assert "headroom_vs_reference" in d

    def test_headroom_metrics_zero_denominator(self):
        """compute_headroom_metrics must handle zero denominator gracefully."""
        result = compute_headroom_metrics(
            candidate_mean=5.0,
            reference_mean=5.0,
            oracle_mean=5.0,
            n_tasks=10,
            reference_policy_id="sham",
            candidate_policy_id="candidate",
        )
        # When denominator is too small, fraction is None (not NaN).
        assert result["recovered_headroom_fraction"] is None
        assert result["recovered_headroom_percent"] is None
        assert result.get("error") == "denominator_too_small"

    def test_headroom_metrics_normal(self):
        result = compute_headroom_metrics(
            candidate_mean=7.5,
            reference_mean=5.0,
            oracle_mean=10.0,
            n_tasks=10,
            reference_policy_id="sham",
            candidate_policy_id="candidate",
        )
        assert abs(result["recovered_headroom_fraction"] - 0.5) < 1e-9

    def test_candidate_executed_has_headroom_vs_reference(self, mini_run):
        results = evaluate_all_policies(mini_run, seed=DEFAULT_SEED)
        ce = results["candidate_executed"]
        assert ce.headroom_vs_reference is not None
        assert "recovered_headroom_fraction" in ce.headroom_vs_reference

    def test_evaluate_policy_from_counterfactuals_returns_result(self, mini_run):
        fn = make_oracle_fn(mini_run)
        result = evaluate_policy_from_counterfactuals(
            mini_run, mini_run.test_task_ids, fn, "oracle",
        )
        assert isinstance(result, PolicyEvalResult)
        assert result.n_tasks == len(mini_run.test_task_ids)
        assert result.mean_utility > 0


# ---------------------------------------------------------------------------
# B. Metric consistency
# ---------------------------------------------------------------------------


class TestMetricConsistency:
    """Test metric consistency cross-check."""

    def test_check_metric_consistency_returns_dict(self, mini_run, canonical_results):
        from qualification.autolearn_v04.v043.metric_consistency import check_metric_consistency
        result = check_metric_consistency(mini_run, canonical_results)
        assert "overall_status" in result
        assert "checks" in result
        assert isinstance(result["checks"], list)
        assert "n_checks" in result

    def test_check_metric_consistency_has_expected_checks(self, mini_run, canonical_results):
        from qualification.autolearn_v04.v043.metric_consistency import check_metric_consistency
        result = check_metric_consistency(mini_run, canonical_results)
        check_names = {c["name"] for c in result["checks"]}
        # Must include the core cross-checks.
        assert "mean_candidate_utility" in check_names
        assert "candidate_minus_sham" in check_names
        assert "recovered_headroom_fraction" in check_names


# ---------------------------------------------------------------------------
# C. Sham identity
# ---------------------------------------------------------------------------


class TestShamIdentity:
    """Test sham identity validation."""

    def test_validate_sham_identity_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.sham_identity import validate_sham_identity
        result = validate_sham_identity(mini_run)
        assert "primary_sham_id" in result
        assert "overall_status" in result
        assert "consistency_checks" in result
        assert "n_checks" in result

    def test_sham_identity_has_audit_notes(self, mini_run):
        from qualification.autolearn_v04.v043.sham_identity import validate_sham_identity
        result = validate_sham_identity(mini_run)
        assert "audit_notes" in result
        notes = result["audit_notes"]
        assert "alternate_sources_audited" in notes


# ---------------------------------------------------------------------------
# D. Gate A0 audit
# ---------------------------------------------------------------------------


class TestGateA0Audit:
    """Test Gate A0 infrastructure audit."""

    def test_audit_gate_a0_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.gate_a0_audit import audit_gate_a0
        result = audit_gate_a0(mini_run)
        assert "sub_gates" in result
        assert "overall_status" in result
        assert "n_pass" in result
        assert "n_fail" in result
        assert "n_blocked" in result
        assert "n_total" in result

    def test_gate_a0_has_15_sub_gates(self, mini_run):
        from qualification.autolearn_v04.v043.gate_a0_audit import audit_gate_a0
        result = audit_gate_a0(mini_run)
        assert result["n_total"] == 15
        assert len(result["sub_gates"]) == 15

    def test_gate_a0_sub_gate_ids(self, mini_run):
        from qualification.autolearn_v04.v043.gate_a0_audit import audit_gate_a0
        result = audit_gate_a0(mini_run)
        for i in range(1, 16):
            assert f"A0.{i}" in result["sub_gates"]


# ---------------------------------------------------------------------------
# E. Decision trace
# ---------------------------------------------------------------------------


class TestDecisionTrace:
    """Test candidate decision trace reconstruction."""

    def test_reconstruct_decision_trace_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.decision_trace import reconstruct_decision_trace
        result = reconstruct_decision_trace(mini_run)
        assert "per_task" in result
        assert isinstance(result["per_task"], list)
        # Should have entries for test tasks.
        assert len(result["per_task"]) > 0

    def test_decision_trace_has_action_distributions(self, mini_run):
        from qualification.autolearn_v04.v043.decision_trace import reconstruct_decision_trace
        result = reconstruct_decision_trace(mini_run)
        assert "action_distributions" in result


# ---------------------------------------------------------------------------
# F. Raw vs executed
# ---------------------------------------------------------------------------


class TestRawVsExecuted:
    """Test raw vs executed candidate comparison."""

    def test_compare_raw_vs_executed_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.raw_vs_executed import compare_raw_vs_executed
        result = compare_raw_vs_executed(mini_run, seed=DEFAULT_SEED)
        assert "raw_result" in result
        assert "executed_result" in result
        assert "bottleneck_classification" in result
        assert "interpretation" in result

    def test_raw_vs_executed_has_comparison(self, mini_run):
        from qualification.autolearn_v04.v043.raw_vs_executed import compare_raw_vs_executed
        result = compare_raw_vs_executed(mini_run, seed=DEFAULT_SEED)
        assert "comparison" in result
        assert isinstance(result["comparison"], dict)


# ---------------------------------------------------------------------------
# G. Calibration fallback audit
# ---------------------------------------------------------------------------


class TestCalibrationFallback:
    """Test calibration and fallback audit."""

    def test_audit_calibration_fallback_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.calibration_fallback_audit import audit_calibration_fallback
        result = audit_calibration_fallback(mini_run, seed=DEFAULT_SEED)
        assert "validation_curves" in result
        assert "production_threshold" in result
        assert "final_test_exploratory" in result
        assert "interpretation" in result

    def test_calibration_validation_curves_structure(self, mini_run):
        from qualification.autolearn_v04.v043.calibration_fallback_audit import audit_calibration_fallback
        result = audit_calibration_fallback(mini_run, seed=DEFAULT_SEED)
        vc = result["validation_curves"]
        assert "threshold_sweeps" in vc
        assert "risk_coverage" in vc
        assert "utility_coverage" in vc

    def test_calibration_final_test_is_exploratory(self, mini_run):
        from qualification.autolearn_v04.v043.calibration_fallback_audit import audit_calibration_fallback
        result = audit_calibration_fallback(mini_run, seed=DEFAULT_SEED)
        ft = result["final_test_exploratory"]
        assert ft.get("qualifying") is False
        assert "EXPLORATORY" in ft.get("label", "")


# ---------------------------------------------------------------------------
# H. Headroom concentration
# ---------------------------------------------------------------------------


class TestHeadroomConcentration:
    """Test oracle headroom concentration analysis."""

    def test_analyze_headroom_concentration_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.headroom_concentration import analyze_headroom_concentration
        result = analyze_headroom_concentration(mini_run)
        assert "aggregate_headroom" in result
        assert "concentration_metrics" in result
        assert "concentration_status" in result
        assert "interpretation" in result

    def test_headroom_concentration_metrics(self, mini_run):
        from qualification.autolearn_v04.v043.headroom_concentration import analyze_headroom_concentration
        result = analyze_headroom_concentration(mini_run)
        cm = result["concentration_metrics"]
        assert "hhi" in cm
        assert "gini" in cm
        assert "top_10_share" in cm
        assert "effective_headroom_task_count" in cm

    def test_headroom_concentration_has_by_group(self, mini_run):
        from qualification.autolearn_v04.v043.headroom_concentration import analyze_headroom_concentration
        result = analyze_headroom_concentration(mini_run)
        assert "by_group" in result
        bg = result["by_group"]
        assert "family" in bg
        assert "archetype" in bg


# ---------------------------------------------------------------------------
# I. High margin recovery
# ---------------------------------------------------------------------------


class TestHighMarginRecovery:
    """Test high-margin task recovery evaluation."""

    def test_evaluate_high_margin_recovery_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.high_margin_recovery import evaluate_high_margin_recovery
        result = evaluate_high_margin_recovery(mini_run, seed=DEFAULT_SEED)
        assert "margin_buckets" in result
        assert "gate_a5" in result
        assert "interpretation" in result

    def test_gate_a5_has_status(self, mini_run):
        from qualification.autolearn_v04.v043.high_margin_recovery import evaluate_high_margin_recovery
        result = evaluate_high_margin_recovery(mini_run, seed=DEFAULT_SEED)
        ga5 = result["gate_a5"]
        assert "status" in ga5
        assert ga5["status"] in ("PASS", "FAIL", "NOT_RUN")
        assert "recovery_fraction" in ga5 or ga5["status"] == "NOT_RUN"


# ---------------------------------------------------------------------------
# J. Feature signal
# ---------------------------------------------------------------------------


class TestFeatureSignal:
    """Test feature signal / learnability analysis."""

    def test_analyze_feature_signal_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.feature_signal import analyze_feature_signal
        result = analyze_feature_signal(mini_run, seed=DEFAULT_SEED)
        assert "models" in result
        assert "learnability_conclusion" in result
        assert "interpretation" in result

    def test_feature_signal_has_all_models(self, mini_run):
        from qualification.autolearn_v04.v043.feature_signal import analyze_feature_signal
        result = analyze_feature_signal(mini_run, seed=DEFAULT_SEED)
        expected_models = {
            "intercept_only", "marginal_action_frequency", "family_only",
            "archetype_only", "family_plus_archetype",
            "features_excluding_family", "features_excluding_archetype",
            "full_feature_model", "shuffled_feature_model",
        }
        assert expected_models.issubset(set(result["models"].keys()))

    def test_feature_signal_conclusion_valid(self, mini_run):
        from qualification.autolearn_v04.v043.feature_signal import analyze_feature_signal
        result = analyze_feature_signal(mini_run, seed=DEFAULT_SEED)
        assert result["learnability_conclusion"] in (
            "FEATURE_SIGNAL_PRESENT",
            "FEATURE_SIGNAL_WEAK",
            "FEATURE_SIGNAL_ABSENT",
            "TEMPLATE_LEAKAGE_DOMINANT",
        )


# ---------------------------------------------------------------------------
# K. Missing state analysis
# ---------------------------------------------------------------------------


class TestMissingState:
    """Test missing pre-action state analysis."""

    def test_analyze_missing_state_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.missing_state_analysis import analyze_missing_state
        result = analyze_missing_state(mini_run)
        assert "state_variables" in result
        assert "missing_feature_candidates" in result
        assert "interpretation" in result

    def test_missing_state_candidates_are_leakage_safe(self, mini_run):
        from qualification.autolearn_v04.v043.missing_state_analysis import analyze_missing_state
        result = analyze_missing_state(mini_run)
        for cand in result["missing_feature_candidates"]:
            assert cand["available_before_action"] is True
            assert cand["not_derived_from_verifier"] is True
            assert cand["not_derived_from_outcome"] is True


# ---------------------------------------------------------------------------
# L. Feature registry
# ---------------------------------------------------------------------------


class TestFeatureRegistry:
    """Test feature registry construction."""

    def test_build_feature_registry_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.feature_registry import build_feature_registry
        result = build_feature_registry(mini_run)
        assert "features" in result
        assert "rejected_features" in result
        assert "n_accepted" in result
        assert "n_rejected" in result
        assert "registry_sha256" in result

    def test_feature_registry_has_assertions(self, mini_run):
        from qualification.autolearn_v04.v043.feature_registry import build_feature_registry
        result = build_feature_registry(mini_run)
        assert "assertion_results" in result
        assert isinstance(result["assertion_results"], list)
        assert len(result["assertion_results"]) >= 4

    def test_feature_registry_no_leakage_in_accepted(self, mini_run):
        from qualification.autolearn_v04.v043.feature_registry import build_feature_registry
        result = build_feature_registry(mini_run)
        for f in result["features"]:
            assert f["available_before_action"] is True
            assert f["post_outcome"] is False
            assert f["verifier_derived"] is False


# ---------------------------------------------------------------------------
# M. Learner validation
# ---------------------------------------------------------------------------


class TestLearnerValidation:
    """Test learner validation comparison."""

    def test_validate_learners_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.learner_validation import validate_learners
        result = validate_learners(mini_run, seed=DEFAULT_SEED)
        assert "learners" in result
        assert "best_validation_learner" in result
        assert "interpretation" in result

    def test_validate_learners_has_multiple_learners(self, mini_run):
        from qualification.autolearn_v04.v043.learner_validation import validate_learners
        result = validate_learners(mini_run, seed=DEFAULT_SEED)
        assert "multinomial_logistic" in result["learners"]

    def test_validate_learners_test_results_exploratory(self, mini_run):
        from qualification.autolearn_v04.v043.learner_validation import validate_learners
        result = validate_learners(mini_run, seed=DEFAULT_SEED)
        test_res = result.get("test_results_exploratory", {})
        if test_res:
            assert test_res.get("EXPLORATORY") is True


# ---------------------------------------------------------------------------
# N. Soft utility targets
# ---------------------------------------------------------------------------


class TestSoftUtilityTargets:
    """Test soft-utility training target comparison."""

    def test_compare_soft_targets_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.soft_utility_targets import compare_soft_targets
        result = compare_soft_targets(mini_run, seed=DEFAULT_SEED)
        assert "models" in result or "error" in result
        if "models" in result:
            assert "hard_label" in result["models"]
            assert "soft_utility" in result["models"]
            assert "tie_aware_soft" in result["models"]

    def test_compare_soft_targets_has_interpretation(self, mini_run):
        from qualification.autolearn_v04.v043.soft_utility_targets import compare_soft_targets
        result = compare_soft_targets(mini_run, seed=DEFAULT_SEED)
        if "error" not in result:
            assert "interpretation" in result


# ---------------------------------------------------------------------------
# O. Class balance
# ---------------------------------------------------------------------------


class TestClassBalance:
    """Test class balance analysis."""

    def test_analyze_class_balance_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.class_balance import analyze_class_balance
        result = analyze_class_balance(mini_run, seed=DEFAULT_SEED)
        assert "options" in result or "error" in result
        if "options" in result:
            assert "no_class_weighting" in result["options"]

    def test_class_balance_has_interpretation(self, mini_run):
        from qualification.autolearn_v04.v043.class_balance import analyze_class_balance
        result = analyze_class_balance(mini_run, seed=DEFAULT_SEED)
        if "error" not in result:
            assert "interpretation" in result


# ---------------------------------------------------------------------------
# P. Sham ensemble
# ---------------------------------------------------------------------------


class TestShamEnsemble:
    """Test sham seed ensemble analysis."""

    def test_run_sham_ensemble_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.sham_ensemble import run_sham_ensemble
        result = run_sham_ensemble(mini_run, n_seeds=3, base_seed=10000)
        assert "by_type" in result
        assert "candidate_percentile" in result
        assert "candidate_beats_sham" in result
        assert "interpretation" in result

    def test_sham_ensemble_has_global_label_shuffle(self, mini_run):
        from qualification.autolearn_v04.v043.sham_ensemble import run_sham_ensemble
        result = run_sham_ensemble(mini_run, n_seeds=3, base_seed=10000)
        assert "global_label_shuffle" in result["by_type"]
        primary = result["by_type"]["global_label_shuffle"]
        assert "mean_utility" in primary
        assert "std_utility" in primary
        assert "seeds" in primary

    def test_sham_ensemble_candidate_utility(self, mini_run):
        from qualification.autolearn_v04.v043.sham_ensemble import run_sham_ensemble
        result = run_sham_ensemble(mini_run, n_seeds=3, base_seed=10000)
        assert "candidate_utility" in result
        assert isinstance(result["candidate_utility"], float)


# ---------------------------------------------------------------------------
# Q. Candidate stability
# ---------------------------------------------------------------------------


class TestCandidateStability:
    """Test candidate training stability analysis."""

    def test_analyze_candidate_stability_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.candidate_stability import analyze_candidate_stability
        result = analyze_candidate_stability(mini_run, n_seeds=3, base_seed=20000)
        assert "per_seed" in result
        assert "aggregate" in result
        assert "stability_classification" in result
        assert "interpretation" in result

    def test_candidate_stability_classification_valid(self, mini_run):
        from qualification.autolearn_v04.v043.candidate_stability import analyze_candidate_stability
        result = analyze_candidate_stability(mini_run, n_seeds=3, base_seed=20000)
        assert result["stability_classification"] in ("STABLE", "ROUTER_UNSTABLE")

    def test_candidate_stability_has_percentile(self, mini_run):
        from qualification.autolearn_v04.v043.candidate_stability import analyze_candidate_stability
        result = analyze_candidate_stability(mini_run, n_seeds=3, base_seed=20000)
        assert "original_candidate_percentile" in result


# ---------------------------------------------------------------------------
# R. Parameter diagnostics
# ---------------------------------------------------------------------------


class TestParameterDiagnostics:
    """Test router parameter diagnostics."""

    def test_diagnose_parameters_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.parameter_diagnostics import diagnose_parameters
        result = diagnose_parameters(mini_run, n_bootstrap=5, seed=DEFAULT_SEED)
        assert "coefficient_matrix" in result
        assert "standardized_coefficients" in result
        assert "diagnostic_checks" in result
        assert "interpretation" in result

    def test_parameter_diagnostics_has_coefficient_matrix(self, mini_run):
        from qualification.autolearn_v04.v043.parameter_diagnostics import diagnose_parameters
        result = diagnose_parameters(mini_run, n_bootstrap=5, seed=DEFAULT_SEED)
        cm = result["coefficient_matrix"]
        assert "W" in cm
        assert "b" in cm
        assert "actions" in cm
        assert "feature_names" in cm

    def test_parameter_diagnostics_has_checks(self, mini_run):
        from qualification.autolearn_v04.v043.parameter_diagnostics import diagnose_parameters
        result = diagnose_parameters(mini_run, n_bootstrap=5, seed=DEFAULT_SEED)
        checks = result["diagnostic_checks"]
        expected = {
            "coefficients_near_zero",
            "one_feature_dominates",
            "action_logits_nearly_constant",
            "family_features_dominate",
            "normalization_inconsistent",
            "regularization_too_strong",
            "class_separation_weak",
        }
        assert expected.issubset(set(checks.keys()))


# ---------------------------------------------------------------------------
# S. Utility consistency
# ---------------------------------------------------------------------------


class TestUtilityConsistency:
    """Test utility consistency audit."""

    def test_audit_utility_consistency_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.utility_consistency import audit_utility_consistency
        result = audit_utility_consistency(mini_run)
        assert "per_outcome_audit" in result
        assert "consistency_checks" in result
        assert "overall_status" in result
        assert "interpretation" in result

    def test_utility_consistency_has_all_checks(self, mini_run):
        from qualification.autolearn_v04.v043.utility_consistency import audit_utility_consistency
        result = audit_utility_consistency(mini_run)
        expected = {
            "reward_components_keys_consistent",
            "utility_values_consistent",
            "none_not_converted_to_zero",
            "non_executed_consistency",
            "no_double_penalties",
            "latency_normalization",
            "token_normalization",
            "utility_version_and_normalization",
        }
        assert expected.issubset(set(result["consistency_checks"].keys()))


# ---------------------------------------------------------------------------
# T. Counterfactual completeness
# ---------------------------------------------------------------------------


class TestCounterfactualCompleteness:
    """Test counterfactual completeness audit."""

    def test_audit_counterfactual_completeness_returns_dict(self, mini_run):
        from qualification.autolearn_v04.v043.counterfactual_completeness import audit_counterfactual_completeness
        result = audit_counterfactual_completeness(mini_run)
        assert "per_task" in result
        assert "summary" in result
        assert "interpretation" in result

    def test_counterfactual_completeness_summary(self, mini_run):
        from qualification.autolearn_v04.v043.counterfactual_completeness import audit_counterfactual_completeness
        result = audit_counterfactual_completeness(mini_run)
        summary = result["summary"]
        assert "COMPLETE" in summary
        assert "INCOMPLETE" in summary
        assert "NONCOMPARABLE" in summary

    def test_counterfactual_completeness_has_headroom(self, mini_run):
        from qualification.autolearn_v04.v043.counterfactual_completeness import audit_counterfactual_completeness
        result = audit_counterfactual_completeness(mini_run)
        assert "headroom_all_tasks" in result
        assert "headroom_complete_only" in result


# ---------------------------------------------------------------------------
# U. Gate A layers
# ---------------------------------------------------------------------------


class TestGateALayers:
    """Test corrected Gate A layered sub-gate evaluation."""

    def _build_gate_a(self, mini_run, canonical_results):
        from qualification.autolearn_v04.v043.gate_a_layers import evaluate_gate_a_v043
        from qualification.autolearn_v04.v043.gate_a0_audit import audit_gate_a0
        from qualification.autolearn_v04.v043.headroom_concentration import analyze_headroom_concentration
        from qualification.autolearn_v04.v043.feature_signal import analyze_feature_signal
        from qualification.autolearn_v04.v043.high_margin_recovery import evaluate_high_margin_recovery
        from qualification.autolearn_v04.v043.sham_ensemble import run_sham_ensemble
        from qualification.autolearn_v04.v043.metric_consistency import check_metric_consistency

        a0 = audit_gate_a0(mini_run)
        hc = analyze_headroom_concentration(mini_run)
        fs = analyze_feature_signal(mini_run, seed=DEFAULT_SEED)
        hm = evaluate_high_margin_recovery(mini_run, seed=DEFAULT_SEED)
        se = run_sham_ensemble(mini_run, n_seeds=3, base_seed=10000)
        mc = check_metric_consistency(mini_run, canonical_results)

        return evaluate_gate_a_v043(
            run=mini_run,
            gate_a0_result=a0,
            headroom_concentration_result=hc,
            feature_signal_result=fs,
            canonical_results=canonical_results,
            high_margin_result=hm,
            sham_ensemble_result=se,
            metric_consistency_result=mc,
            n_bootstrap=100,
            seed=DEFAULT_SEED,
        )

    def test_evaluate_gate_a_v043_returns_dict(self, mini_run, canonical_results):
        result = self._build_gate_a(mini_run, canonical_results)
        assert "sub_gates" in result
        assert "overall_status" in result
        assert "n_pass" in result
        assert "n_fail" in result
        assert "n_blocked" in result
        assert "interpretation" in result

    def test_gate_a_has_a0_through_a5(self, mini_run, canonical_results):
        result = self._build_gate_a(mini_run, canonical_results)
        for name in ("A0", "A1", "A2", "A3", "A4", "A5"):
            assert name in result["sub_gates"]
            assert "status" in result["sub_gates"][name]


# ---------------------------------------------------------------------------
# V. Scale decision
# ---------------------------------------------------------------------------


class TestScaleDecision:
    """Test model-scale decision engine."""

    def _build_gate_a_result(self, mini_run, canonical_results):
        from qualification.autolearn_v04.v043.gate_a_layers import evaluate_gate_a_v043
        from qualification.autolearn_v04.v043.gate_a0_audit import audit_gate_a0
        from qualification.autolearn_v04.v043.headroom_concentration import analyze_headroom_concentration
        from qualification.autolearn_v04.v043.feature_signal import analyze_feature_signal
        from qualification.autolearn_v04.v043.high_margin_recovery import evaluate_high_margin_recovery
        from qualification.autolearn_v04.v043.sham_ensemble import run_sham_ensemble
        from qualification.autolearn_v04.v043.metric_consistency import check_metric_consistency

        a0 = audit_gate_a0(mini_run)
        hc = analyze_headroom_concentration(mini_run)
        fs = analyze_feature_signal(mini_run, seed=DEFAULT_SEED)
        hm = evaluate_high_margin_recovery(mini_run, seed=DEFAULT_SEED)
        se = run_sham_ensemble(mini_run, n_seeds=3, base_seed=10000)
        mc = check_metric_consistency(mini_run, canonical_results)

        return evaluate_gate_a_v043(
            run=mini_run,
            gate_a0_result=a0,
            headroom_concentration_result=hc,
            feature_signal_result=fs,
            canonical_results=canonical_results,
            high_margin_result=hm,
            sham_ensemble_result=se,
            metric_consistency_result=mc,
            n_bootstrap=100,
            seed=DEFAULT_SEED,
        )

    def _build_scale_decision(self, mini_run, canonical_results):
        from qualification.autolearn_v04.v043.scale_decision import make_scale_decision_v043
        from qualification.autolearn_v04.v043.headroom_concentration import analyze_headroom_concentration
        from qualification.autolearn_v04.v043.feature_signal import analyze_feature_signal
        from qualification.autolearn_v04.v043.raw_vs_executed import compare_raw_vs_executed
        from qualification.autolearn_v04.v043.candidate_stability import analyze_candidate_stability
        from qualification.autolearn_v04.v043.sham_ensemble import run_sham_ensemble
        from qualification.autolearn_v04.v043.counterfactual_completeness import audit_counterfactual_completeness
        from qualification.autolearn_v04.v043.utility_consistency import audit_utility_consistency

        gate_a = self._build_gate_a_result(mini_run, canonical_results)
        hc = analyze_headroom_concentration(mini_run)
        fs = analyze_feature_signal(mini_run, seed=DEFAULT_SEED)
        rve = compare_raw_vs_executed(mini_run, seed=DEFAULT_SEED)
        cs = analyze_candidate_stability(mini_run, n_seeds=3, base_seed=20000)
        se = run_sham_ensemble(mini_run, n_seeds=3, base_seed=10000)
        cc = audit_counterfactual_completeness(mini_run)
        uc = audit_utility_consistency(mini_run)

        return make_scale_decision_v043(
            run=mini_run,
            gate_a_result=gate_a,
            headroom_concentration_result=hc,
            feature_signal_result=fs,
            raw_vs_executed_result=rve,
            candidate_stability_result=cs,
            sham_ensemble_result=se,
            counterfactual_completeness_result=cc,
            utility_consistency_result=uc,
        )

    def test_make_scale_decision_returns_dict(self, mini_run, canonical_results):
        result = self._build_scale_decision(mini_run, canonical_results)
        assert "decision" in result
        assert "reason" in result
        assert "approve_full_14b_run" in result
        assert "approve_matched_14b_pilot" in result
        assert "required_next_action" in result
        assert "scale_decision_statement" in result

    def test_scale_decision_approve_full_14b_always_false(self, mini_run, canonical_results):
        result = self._build_scale_decision(mini_run, canonical_results)
        assert result["approve_full_14b_run"] is False

    def test_scale_decision_has_blocking_conditions(self, mini_run, canonical_results):
        result = self._build_scale_decision(mini_run, canonical_results)
        assert "blocking_conditions_checked" in result
        assert isinstance(result["blocking_conditions_checked"], dict)


# ---------------------------------------------------------------------------
# W. Matched pilot manifest
# ---------------------------------------------------------------------------


class TestMatchedPilotManifest:
    """Test matched 14B pilot eligibility and manifest generation."""

    def test_check_pilot_eligibility_all_pass(self):
        from qualification.autolearn_v04.v043.matched_pilot_manifest import check_pilot_eligibility
        result = check_pilot_eligibility(
            gate_a0_status="PASS",
            counterfactual_completeness_status="PASS",
            utility_consistency_status="PASS",
            sham_validity=True,
            oracle_headroom=0.5,
            headroom_concentration_status="BROAD",
            feature_signal_conclusion="FEATURE_SIGNAL_PRESENT",
            candidate_stability_classification="STABLE",
            scale_decision="APPROVE_MATCHED_14B_PILOT",
        )
        assert result["eligible"] is True
        assert result["n_failing"] == 0

    def test_check_pilot_eligibility_some_fail(self):
        from qualification.autolearn_v04.v043.matched_pilot_manifest import check_pilot_eligibility
        result = check_pilot_eligibility(
            gate_a0_status="FAIL",
            counterfactual_completeness_status="PASS",
            utility_consistency_status="PASS",
            sham_validity=True,
            oracle_headroom=0.5,
            headroom_concentration_status="BROAD",
            feature_signal_conclusion="FEATURE_SIGNAL_PRESENT",
            candidate_stability_classification="STABLE",
            scale_decision="APPROVE_MATCHED_14B_PILOT",
        )
        assert result["eligible"] is False
        assert "gate_a0_passes" in result["failed_conditions"]

    def test_generate_pilot_manifest_not_eligible(self, mini_run):
        from qualification.autolearn_v04.v043.matched_pilot_manifest import generate_pilot_manifest
        eligibility = {
            "eligible": False,
            "failed_conditions": ["gate_a0_passes"],
            "n_conditions": 9,
            "n_passing": 8,
            "n_failing": 1,
        }
        result = generate_pilot_manifest(mini_run, eligibility)
        assert result["pilot_approved"] is False
        assert result["pilot_manifest"] is None

    def test_generate_pilot_manifest_eligible(self, mini_run):
        from qualification.autolearn_v04.v043.matched_pilot_manifest import generate_pilot_manifest
        eligibility = {
            "eligible": True,
            "failed_conditions": [],
            "n_conditions": 9,
            "n_passing": 9,
            "n_failing": 0,
        }
        result = generate_pilot_manifest(mini_run, eligibility)
        assert result["pilot_approved"] is True
        manifest = result["pilot_manifest"]
        assert manifest is not None
        assert "task_ids" in manifest
        assert "entries" in manifest
        assert manifest["is_full_qualification_run"] is False


# ---------------------------------------------------------------------------
# X. Report generator
# ---------------------------------------------------------------------------


class TestReportGenerator:
    """Test v0.4.3 report generation."""

    def test_generate_v043_report_returns_string(self, mini_run, canonical_results):
        from qualification.autolearn_v04.v043.report_generator import generate_v043_report
        from qualification.autolearn_v04.v043.metric_consistency import check_metric_consistency
        from qualification.autolearn_v04.v043.sham_identity import validate_sham_identity
        from qualification.autolearn_v04.v043.gate_a0_audit import audit_gate_a0
        from qualification.autolearn_v04.v043.decision_trace import reconstruct_decision_trace
        from qualification.autolearn_v04.v043.raw_vs_executed import compare_raw_vs_executed
        from qualification.autolearn_v04.v043.calibration_fallback_audit import audit_calibration_fallback
        from qualification.autolearn_v04.v043.headroom_concentration import analyze_headroom_concentration
        from qualification.autolearn_v04.v043.high_margin_recovery import evaluate_high_margin_recovery
        from qualification.autolearn_v04.v043.feature_signal import analyze_feature_signal
        from qualification.autolearn_v04.v043.missing_state_analysis import analyze_missing_state
        from qualification.autolearn_v04.v043.feature_registry import build_feature_registry
        from qualification.autolearn_v04.v043.learner_validation import validate_learners
        from qualification.autolearn_v04.v043.soft_utility_targets import compare_soft_targets
        from qualification.autolearn_v04.v043.class_balance import analyze_class_balance
        from qualification.autolearn_v04.v043.sham_ensemble import run_sham_ensemble
        from qualification.autolearn_v04.v043.candidate_stability import analyze_candidate_stability
        from qualification.autolearn_v04.v043.parameter_diagnostics import diagnose_parameters
        from qualification.autolearn_v04.v043.utility_consistency import audit_utility_consistency
        from qualification.autolearn_v04.v043.counterfactual_completeness import audit_counterfactual_completeness
        from qualification.autolearn_v04.v043.gate_a_layers import evaluate_gate_a_v043
        from qualification.autolearn_v04.v043.scale_decision import make_scale_decision_v043
        from qualification.autolearn_v04.v043.matched_pilot_manifest import check_pilot_eligibility, generate_pilot_manifest

        # Run all modules.
        mc = check_metric_consistency(mini_run, canonical_results)
        sham_id = validate_sham_identity(mini_run)
        a0 = audit_gate_a0(mini_run)
        dt = reconstruct_decision_trace(mini_run)
        rve = compare_raw_vs_executed(mini_run, seed=DEFAULT_SEED)
        cal = audit_calibration_fallback(mini_run, seed=DEFAULT_SEED)
        hc = analyze_headroom_concentration(mini_run)
        hm = evaluate_high_margin_recovery(mini_run, seed=DEFAULT_SEED)
        fs = analyze_feature_signal(mini_run, seed=DEFAULT_SEED)
        ms = analyze_missing_state(mini_run)
        fr = build_feature_registry(mini_run)
        lv = validate_learners(mini_run, seed=DEFAULT_SEED)
        su = compare_soft_targets(mini_run, seed=DEFAULT_SEED)
        cb = analyze_class_balance(mini_run, seed=DEFAULT_SEED)
        se = run_sham_ensemble(mini_run, n_seeds=3, base_seed=10000)
        cs = analyze_candidate_stability(mini_run, n_seeds=3, base_seed=20000)
        uc = audit_utility_consistency(mini_run)
        pd = diagnose_parameters(mini_run, n_bootstrap=5, seed=DEFAULT_SEED)
        cc = audit_counterfactual_completeness(mini_run)

        gate_a = evaluate_gate_a_v043(
            run=mini_run,
            gate_a0_result=a0,
            headroom_concentration_result=hc,
            feature_signal_result=fs,
            canonical_results=canonical_results,
            high_margin_result=hm,
            sham_ensemble_result=se,
            metric_consistency_result=mc,
            n_bootstrap=100,
            seed=DEFAULT_SEED,
        )

        sd = make_scale_decision_v043(
            run=mini_run,
            gate_a_result=gate_a,
            headroom_concentration_result=hc,
            feature_signal_result=fs,
            raw_vs_executed_result=rve,
            candidate_stability_result=cs,
            sham_ensemble_result=se,
            counterfactual_completeness_result=cc,
            utility_consistency_result=uc,
            learner_validation_result=lv,
        )

        eligibility = check_pilot_eligibility(
            gate_a0_status=a0["overall_status"],
            counterfactual_completeness_status="PASS" if cc.get("summary", {}).get("COMPLETE", 0) > 0 else "FAIL",
            utility_consistency_status=uc["overall_status"],
            sham_validity=sham_id["overall_status"] == "PASS",
            oracle_headroom=hc["aggregate_headroom"],
            headroom_concentration_status=hc["concentration_status"],
            feature_signal_conclusion=fs["learnability_conclusion"],
            candidate_stability_classification=cs["stability_classification"],
            scale_decision=sd["decision"],
        )
        pilot = generate_pilot_manifest(mini_run, eligibility)

        canonical_output = {name: result_to_dict(res) for name, res in canonical_results.items()}

        report = generate_v043_report(
            run=mini_run,
            canonical_results=canonical_output,
            metric_consistency=mc,
            sham_identity=sham_id,
            gate_a0=a0,
            decision_trace=dt,
            raw_vs_executed=rve,
            calibration=cal,
            headroom_concentration=hc,
            high_margin=hm,
            feature_signal=fs,
            missing_state=ms,
            feature_registry=fr,
            learner_validation=lv,
            soft_utility=su,
            class_balance=cb,
            sham_ensemble=se,
            candidate_stability=cs,
            parameter_diagnostics=pd,
            utility_consistency=uc,
            counterfactual_completeness=cc,
            gate_a=gate_a,
            scale_decision=sd,
            pilot_manifest=pilot,
        )
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Gate A v0.4.3 Repair Report" in report
        assert QUALIFICATION_VERSION in report
