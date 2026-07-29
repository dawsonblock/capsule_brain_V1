"""Unit tests for Capsule Brain AutoLearn v0.1.

Covers all required unit test cases from the spec:
- deterministic feature extraction
- feature schema versioning
- utility calculation
- counterfactual best-action selection
- weighted dataset generation
- policy serialization
- incompatible policy rejection
- baseline fallback
- invalid-action rejection
- promotion gate pass/fail
- confidence abstention
- shadow-policy disagreement logging
- distribution-shift fallback
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from capsule_brain.autolearn.baseline import BaselinePolicy
from capsule_brain.autolearn.dataset import (
    build_dataset,
    build_split_manifest,
    split_examples,
)
from capsule_brain.autolearn.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FeatureExtractor,
)
from capsule_brain.autolearn.learner import (
    LearnerConfig,
    SupervisedRouterLearner,
    TrainingExample,
    utility_margin_weight,
)
from capsule_brain.autolearn.policy import (
    LEARNED_POLICY_VERSION_DEFAULT,
    LearnedPolicy,
    PolicyLoadError,
)
from capsule_brain.autolearn.promotion import GateConfig, evaluate_promotion_gate
from capsule_brain.autolearn.registry import (
    PROMOTION_STATUS_ACTIVE,
    PROMOTION_STATUS_REJECTED,
    PolicyRegistry,
)
from capsule_brain.autolearn.schema import (
    Action,
    ActionMetadata,
    ExecutiveExperience,
    ExecutiveState,
    Outcome,
    SCHEMA_VERSION,
)
from capsule_brain.autolearn.service import AutoLearnService, ExecutiveController
from capsule_brain.autolearn.shadow import (
    DistributionShiftConfig,
    DistributionShiftGuard,
    ShadowEvaluator,
)
from capsule_brain.autolearn.utility import UtilityConfig, UtilityFunction
from capsule_brain.autolearn.evaluator import (
    ActionMetrics,
    PairedEvaluation,
    TaskEvaluation,
    bootstrap_ci,
    evaluate_paired,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(
    prompt: str = "hello",
    *,
    available_tools: list[str] | None = None,
    workflow_available: bool = False,
    memory_hit_count: int = 0,
    top_similarity: float = 0.0,
    previous_attempt_failed: bool = False,
    model_id: str = "qwen2.5-coder:3b",
    workflow_capability_match: bool = False,
) -> ExecutiveState:
    return ExecutiveState(
        prompt_features={
            "text": prompt,
            "previous_attempt_failed": previous_attempt_failed,
            "verification_failure_type": "none",
            "estimated_difficulty": 0.5,
            "workflow_capability_match": workflow_capability_match,
        },
        conversation_features={"depth": 0},
        memory_features={"hit_count": memory_hit_count, "top_similarity": top_similarity},
        available_tools=list(available_tools or []),
        workflow_available=workflow_available,
        model_id=model_id,
        context_length=len(prompt),
    )


# ---------------------------------------------------------------------------
# Deterministic feature extraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    def test_deterministic_same_state_same_vector(self):
        ext = FeatureExtractor()
        state = _state("What is the weather in Tokyo?")
        v1 = ext.extract(state)
        v2 = ext.extract(state)
        assert v1.values == v2.values
        assert v1.schema_version == FEATURE_SCHEMA_VERSION

    def test_different_prompts_different_vectors(self):
        ext = FeatureExtractor()
        v1 = ext.extract(_state("What is 2+2?"))
        v2 = ext.extract(_state("Remember the pin I stored earlier."))
        assert v1.values != v2.values

    def test_feature_count_matches_names(self):
        ext = FeatureExtractor()
        fv = ext.extract(_state("test"))
        assert len(fv.values) == len(fv.names)
        assert fv.names == FEATURE_NAMES

    def test_code_indicators_detected(self):
        ext = FeatureExtractor()
        fv = ext.extract(_state("def foo():\n    return 42"))
        d = fv.as_dict()
        assert d["code_indicators"] > 0

    def test_tool_keywords_detected(self):
        ext = FeatureExtractor()
        fv = ext.extract(_state("Fetch the live nonce using the tool."))
        d = fv.as_dict()
        assert d["tool_required_keywords"] > 0

    def test_retrieval_indicators_detected(self):
        ext = FeatureExtractor()
        fv = ext.extract(_state("Recall what we discussed last time."))
        d = fv.as_dict()
        assert d["retrieval_indicators"] > 0

    def test_workflow_available_flag(self):
        ext = FeatureExtractor()
        fv = ext.extract(_state("test", workflow_available=True))
        assert fv.as_dict()["workflow_available"] == 1.0

    def test_num_available_tools(self):
        ext = FeatureExtractor()
        fv = ext.extract(_state("test", available_tools=["a", "b", "c"]))
        assert fv.as_dict()["num_available_tools"] == 3.0


# ---------------------------------------------------------------------------
# Feature schema versioning
# ---------------------------------------------------------------------------


class TestFeatureSchemaVersioning:
    def test_schema_version_is_non_empty_string(self):
        assert isinstance(FEATURE_SCHEMA_VERSION, str)
        assert FEATURE_SCHEMA_VERSION

    def test_feature_names_are_stable(self):
        assert "prompt_length" in FEATURE_NAMES
        assert "workflow_available" in FEATURE_NAMES
        assert len(FEATURE_NAMES) == 17

    def test_policy_rejects_incompatible_schema(self):
        policy = LearnedPolicy(
            policy_id="test",
            feature_schema_version="exec_features_v999",
        )
        # from_dict should reject a mismatched schema.
        data = policy.to_dict()
        data["feature_schema_version"] = "exec_features_v999"
        with pytest.raises(PolicyLoadError, match="incompatible feature schema"):
            LearnedPolicy.from_dict(data)


# ---------------------------------------------------------------------------
# Utility calculation
# ---------------------------------------------------------------------------


class TestUtilityCalculation:
    def test_verified_success_dominates_cost(self):
        ufn = UtilityFunction(UtilityConfig())
        # A correct expensive answer.
        good = Outcome(verified_success=True, latency_ms=5000, token_count=4000)
        # A cheap wrong answer.
        bad = Outcome(verified_success=False, latency_ms=100, token_count=50)
        u_good, _ = ufn.compute(good)
        u_bad, _ = ufn.compute(bad)
        assert u_good > u_bad

    def test_safety_violation_is_large_negative(self):
        ufn = UtilityFunction(UtilityConfig())
        safe = Outcome(verified_success=True)
        unsafe = Outcome(verified_success=True, safety_violation=True)
        u_safe, _ = ufn.compute(safe)
        u_unsafe, _ = ufn.compute(unsafe)
        assert u_unsafe < u_safe
        assert u_unsafe < 0  # safety penalty dominates

    def test_unverified_does_not_get_success_reward(self):
        ufn = UtilityFunction(UtilityConfig())
        verified = Outcome(verified_success=True, verification_status="pass")
        unverified = Outcome(verified_success=False, verification_status="skip")
        u_v, c_v = ufn.compute(verified)
        u_u, c_u = ufn.compute(unverified)
        assert c_v["success_term"] > 0
        assert c_u["success_term"] == 0.0
        assert u_v > u_u

    def test_configurable_weights(self):
        cfg = UtilityConfig(w_success=100.0)
        ufn = UtilityFunction(cfg)
        u, c = ufn.compute(Outcome(verified_success=True))
        assert c["success_term"] == 100.0

    def test_utility_components_stored(self):
        ufn = UtilityFunction(UtilityConfig())
        _, components = ufn.compute(Outcome(
            verified_success=True, latency_ms=1000, token_count=500,
            tool_failures=1, operator_intervention=True,
        ))
        assert "success_term" in components
        assert "latency_penalty" in components
        assert "safety_penalty" in components


# ---------------------------------------------------------------------------
# Counterfactual best-action selection
# ---------------------------------------------------------------------------


class TestCounterfactualBestAction:
    def test_best_action_is_argmax_utility(self):
        from capsule_brain.autolearn.dataset import CounterfactualRow
        rows = []
        for action in [Action.ANSWER_DIRECT, Action.CALL_TOOL]:
            outcome = Outcome(
                verified_success=(action == Action.CALL_TOOL),
                verification_status="pass" if action == Action.CALL_TOOL else "fail",
                latency_ms=400 if action == Action.CALL_TOOL else 200,
            )
            exp = ExecutiveExperience(
                task_id="t1", state=_state(), chosen_action=action, outcome=outcome,
            )
            rows.append(CounterfactualRow(task_id="t1", task_family="f", action=action, experience=exp, utility=0))
        split = build_split_manifest(rows, train_families=["f"])
        built = build_dataset(rows, split)
        assert len(built.examples) == 1
        assert built.examples[0].best_action == Action.CALL_TOOL

    def test_utility_margin_computed(self):
        from capsule_brain.autolearn.dataset import CounterfactualRow
        rows = []
        for action in [Action.ANSWER_DIRECT, Action.CALL_TOOL]:
            outcome = Outcome(
                verified_success=True,
                verification_status="pass",
                latency_ms=200,
            )
            exp = ExecutiveExperience(
                task_id="t1", state=_state(), chosen_action=action, outcome=outcome,
            )
            rows.append(CounterfactualRow(task_id="t1", task_family="f", action=action, experience=exp, utility=0))
        split = build_split_manifest(rows, train_families=["f"])
        built = build_dataset(rows, split)
        # Both actions have the same utility -> margin is 0.
        assert built.examples[0].utility_margin == 0.0


# ---------------------------------------------------------------------------
# Weighted dataset generation
# ---------------------------------------------------------------------------


class TestWeightedDataset:
    def test_weight_increases_with_margin(self):
        cfg = LearnerConfig(weight_cap=4.0, min_weight=1.0)
        assert utility_margin_weight(0.0, cfg) == 1.0
        assert utility_margin_weight(2.0, cfg) == 3.0
        assert utility_margin_weight(10.0, cfg) == 5.0  # capped

    def test_split_examples_by_split(self):
        examples = [
            TrainingExample(features=[0.0], best_action=Action.ANSWER_DIRECT,
                            utility_margin=1.0, weight=1.0, task_id="t1", split="train"),
            TrainingExample(features=[0.0], best_action=Action.CALL_TOOL,
                            utility_margin=1.0, weight=1.0, task_id="t2", split="test"),
        ]
        splits = split_examples(examples)
        assert len(splits["train"]) == 1
        assert len(splits["test"]) == 1

    def test_split_manifest_is_deterministic(self):
        from capsule_brain.autolearn.dataset import CounterfactualRow
        rows = []
        for i in range(10):
            for action in [Action.ANSWER_DIRECT]:
                exp = ExecutiveExperience(task_id=f"t{i}", state=_state(), chosen_action=action, outcome=Outcome())
                rows.append(CounterfactualRow(task_id=f"t{i}", task_family=f"f{i}", action=action, experience=exp, utility=0))
        s1 = build_split_manifest(rows, seed=42)
        s2 = build_split_manifest(rows, seed=42)
        assert s1.digest == s2.digest
        assert s1.assignments == s2.assignments

    def test_split_manifest_different_seeds_different_digest(self):
        from capsule_brain.autolearn.dataset import CounterfactualRow
        rows = []
        for i in range(20):
            exp = ExecutiveExperience(task_id=f"t{i}", state=_state(), chosen_action=Action.ANSWER_DIRECT, outcome=Outcome())
            rows.append(CounterfactualRow(task_id=f"t{i}", task_family=f"f{i}", action=Action.ANSWER_DIRECT, experience=exp, utility=0))
        s1 = build_split_manifest(rows, seed=1)
        s2 = build_split_manifest(rows, seed=2)
        assert s1.digest != s2.digest


# ---------------------------------------------------------------------------
# Policy serialization
# ---------------------------------------------------------------------------


class TestPolicySerialization:
    def test_round_trip(self):
        policy = LearnedPolicy(
            policy_id="test_policy",
            policy_version="v1",
            weights=[[1.0] * len(FEATURE_NAMES) for _ in Action.all()],
            bias=[0.5] * len(Action.all()),
            confidence_threshold=0.6,
            temperature=0.8,
        )
        text = policy.to_json()
        restored = LearnedPolicy.from_json(text)
        assert restored.policy_id == "test_policy"
        assert restored.confidence_threshold == 0.6
        assert restored.temperature == 0.8
        assert restored.weights == policy.weights
        assert restored.bias == policy.bias

    def test_from_dict_preserves_hyperparameters(self):
        policy = LearnedPolicy(
            policy_id="test",
            hyperparameters={"lr": 0.01, "feature_means": [0.0, 1.0]},
        )
        restored = LearnedPolicy.from_dict(policy.to_dict())
        assert restored.hyperparameters["lr"] == 0.01
        assert restored.hyperparameters["feature_means"] == [0.0, 1.0]


# ---------------------------------------------------------------------------
# Incompatible policy rejection
# ---------------------------------------------------------------------------


class TestIncompatiblePolicyRejection:
    def test_wrong_feature_schema_rejected(self):
        data = LearnedPolicy(policy_id="test").to_dict()
        data["feature_schema_version"] = "wrong"
        with pytest.raises(PolicyLoadError):
            LearnedPolicy.from_dict(data)

    def test_wrong_feature_names_rejected(self):
        data = LearnedPolicy(policy_id="test").to_dict()
        data["feature_names"] = ["wrong", "names"]
        with pytest.raises(PolicyLoadError):
            LearnedPolicy.from_dict(data)

    def test_controller_falls_back_on_incompatible_policy(self):
        # A policy with wrong schema can't even be loaded, so the controller
        # starts with no learned policy and uses baseline.
        controller = ExecutiveController(baseline=BaselinePolicy(), learned=None)
        dec = controller.select_action(_state("What is 2+2?"))
        assert dec.policy_type == "baseline"
        assert dec.policy_fallback is False  # no learned policy is not a fallback


# ---------------------------------------------------------------------------
# Baseline fallback
# ---------------------------------------------------------------------------


class TestBaselineFallback:
    def test_no_learned_policy_uses_baseline(self):
        controller = ExecutiveController(baseline=BaselinePolicy(), learned=None)
        dec = controller.select_action(_state("hello"))
        assert dec.policy_type == "baseline"
        assert dec.action in Action.all()

    def test_learned_policy_error_falls_back(self):
        class BrokenPolicy(LearnedPolicy):
            def select_action(self, *a, **kw):
                raise RuntimeError("boom")

        broken = BrokenPolicy(policy_id="broken")
        controller = ExecutiveController(baseline=BaselinePolicy(), learned=broken)
        dec = controller.select_action(_state("hello"))
        assert dec.policy_type == "baseline"
        assert dec.policy_fallback is True
        assert "boom" in dec.policy_error

    def test_fallback_recorded(self):
        controller = ExecutiveController(baseline=BaselinePolicy(), learned=None)
        controller.select_action(_state("hello"))
        assert controller._decisions == 1
        assert controller._fallbacks == 0  # no learned policy is not a fallback


# ---------------------------------------------------------------------------
# Invalid-action rejection
# ---------------------------------------------------------------------------


class TestInvalidActionRejection:
    def test_disallowed_action_falls_back(self):
        # The policy respects allowed_actions: when only ANSWER_DIRECT is
        # allowed but the model is biased toward CALL_TOOL, the only allowed
        # action has very low confidence, so the policy abstains and the
        # controller falls back to baseline.
        policy = LearnedPolicy(
            policy_id="test",
            weights=[[0.0] * len(FEATURE_NAMES) for _ in Action.all()],
            bias=[0.0] * len(Action.all()),
        )
        # Bias CALL_TOOL heavily so it dominates the softmax.
        call_idx = list(Action.all()).index(Action.CALL_TOOL)
        policy.bias[call_idx] = 100.0
        controller = ExecutiveController(baseline=BaselinePolicy(), learned=policy)
        dec = controller.select_action(
            _state("hello"),
            allowed_actions=[Action.ANSWER_DIRECT],
        )
        # The learned policy can only pick ANSWER_DIRECT (the only allowed
        # action), but its confidence is near zero -> abstention -> fallback.
        assert dec.policy_fallback is True
        assert dec.action == Action.ANSWER_DIRECT  # baseline picked it

    def test_invalid_action_rejected_by_controller(self):
        # Directly test the _is_valid_action defense-in-depth check.
        from capsule_brain.autolearn.service import _is_valid_action
        assert _is_valid_action(Action.ANSWER_DIRECT, [Action.ANSWER_DIRECT]) is True
        assert _is_valid_action(Action.CALL_TOOL, [Action.ANSWER_DIRECT]) is False
        assert _is_valid_action(Action.ANSWER_DIRECT, None) is True
        assert _is_valid_action("not_an_action", None) is False


# ---------------------------------------------------------------------------
# Promotion gate pass/fail
# ---------------------------------------------------------------------------


class TestPromotionGate:
    def _make_eval(self, mean_delta=1.0, ci_low=0.5, ci_high=2.0,
                   baseline_safety=0, candidate_safety=0,
                   baseline_success=0.8, candidate_success=0.9,
                   baseline_utility=5.0, candidate_utility=6.0,
                   tool_precision=0.8, tool_recall=0.8,
                   family_delta=0.5):
        bm = ActionMetrics(
            verified_success_rate=baseline_success,
            mean_utility=baseline_utility,
            safety_violations=baseline_safety,
        )
        cm = ActionMetrics(
            verified_success_rate=candidate_success,
            mean_utility=candidate_utility,
            safety_violations=candidate_safety,
            tool_precision=tool_precision,
            tool_recall=tool_recall,
        )
        te = [TaskEvaluation(
            task_id="t1", task_family="f",
            baseline_action=Action.ANSWER_DIRECT, candidate_action=Action.ANSWER_DIRECT,
            baseline_utility=5.0, candidate_utility=5.0 + family_delta,
            delta=family_delta, disagreement=False,
            baseline_outcome={}, candidate_outcome={},
        )]
        return PairedEvaluation(
            task_evaluations=te, baseline_metrics=bm, candidate_metrics=cm,
            mean_delta=mean_delta, median_delta=mean_delta,
            delta_ci_low=ci_low, delta_ci_high=ci_high,
            wins=1, ties=0, losses=0,
        )

    def test_all_gates_pass(self):
        ev = self._make_eval()
        result = evaluate_promotion_gate(ev)
        assert result.passed is True

    def test_safety_violation_increase_fails(self):
        ev = self._make_eval(baseline_safety=0, candidate_safety=1)
        result = evaluate_promotion_gate(ev)
        assert result.passed is False
        assert any(g["name"] == "no_safety_violation_increase" and not g["passed"] for g in result.gates)

    def test_utility_regression_fails(self):
        ev = self._make_eval(candidate_utility=4.0, mean_delta=-1.0, ci_low=-2.0, ci_high=0.0)
        result = evaluate_promotion_gate(ev)
        assert result.passed is False

    def test_ci_lower_bound_negative_fails(self):
        ev = self._make_eval(ci_low=-0.5)
        result = evaluate_promotion_gate(ev)
        assert result.passed is False
        assert any(g["name"] == "ci_lower_bound_non_negative" and not g["passed"] for g in result.gates)

    def test_tool_precision_below_threshold_fails(self):
        ev = self._make_eval(tool_precision=0.3)
        result = evaluate_promotion_gate(ev, config=GateConfig(tool_precision_min=0.5))
        assert result.passed is False

    def test_ood_below_floor_fails(self):
        ev = self._make_eval()
        ood_ev = self._make_eval(mean_delta=-5.0, ci_low=-6.0, ci_high=-4.0)
        result = evaluate_promotion_gate(ev, ood_eval=ood_ev, config=GateConfig(ood_score_floor=-1.0))
        assert result.passed is False
        assert any(g["name"] == "ood_score_above_floor" and not g["passed"] for g in result.gates)


# ---------------------------------------------------------------------------
# Confidence abstention
# ---------------------------------------------------------------------------


class TestConfidenceAbstention:
    def test_low_confidence_abstains(self):
        # A policy with all-zero weights produces uniform probabilities.
        policy = LearnedPolicy(
            policy_id="test",
            confidence_threshold=0.9,  # higher than uniform 1/6
        )
        dec = policy.select_action(_state("hello"))
        assert dec.abstained is True
        assert dec.confidence < 0.9

    def test_high_confidence_does_not_abstain(self):
        policy = LearnedPolicy(
            policy_id="test",
            confidence_threshold=0.1,
        )
        # Bias one action heavily.
        policy.bias[0] = 100.0
        dec = policy.select_action(_state("hello"))
        assert dec.abstained is False
        assert dec.confidence > 0.1

    def test_controller_abstention_falls_back_to_baseline(self):
        policy = LearnedPolicy(policy_id="test", confidence_threshold=0.99)
        controller = ExecutiveController(
            baseline=BaselinePolicy(), learned=policy, confidence_threshold=0.99,
        )
        dec = controller.select_action(_state("hello"))
        assert dec.policy_fallback is True
        assert dec.abstained is True
        assert dec.policy_type == "baseline"


# ---------------------------------------------------------------------------
# Shadow-policy disagreement logging
# ---------------------------------------------------------------------------


class TestShadowDisagreement:
    def test_disagreement_logged(self):
        # Baseline picks ANSWER_DIRECT; candidate biased to pick CALL_TOOL.
        candidate = LearnedPolicy(policy_id="test", confidence_threshold=0.0)
        call_idx = list(Action.all()).index(Action.CALL_TOOL)
        candidate.bias[call_idx] = 100.0

        shadow = ShadowEvaluator(
            baseline=BaselinePolicy(),
            candidate=candidate,
            confidence_threshold=0.0,
        )
        rec = shadow.evaluate(_state("hello"), task_id="t1")
        # Baseline picks ANSWER_DIRECT for a simple prompt; candidate picks CALL_TOOL.
        assert rec.disagreement is True
        assert rec.production_action != rec.shadow_action

    def test_no_disagreement_when_actions_match(self):
        candidate = LearnedPolicy(policy_id="test", confidence_threshold=0.0)
        # Bias ANSWER_DIRECT (same as baseline for a simple prompt).
        answer_idx = list(Action.all()).index(Action.ANSWER_DIRECT)
        candidate.bias[answer_idx] = 100.0

        shadow = ShadowEvaluator(
            baseline=BaselinePolicy(),
            candidate=candidate,
            confidence_threshold=0.0,
        )
        rec = shadow.evaluate(_state("hello"), task_id="t1")
        assert rec.disagreement is False

    def test_shadow_stats_tracked(self):
        candidate = LearnedPolicy(policy_id="test", confidence_threshold=0.0)
        shadow = ShadowEvaluator(
            baseline=BaselinePolicy(),
            candidate=candidate,
            confidence_threshold=0.0,
        )
        shadow.evaluate(_state("hello"), task_id="t1")
        shadow.evaluate(_state("world"), task_id="t2")
        assert shadow.stats.n == 2


# ---------------------------------------------------------------------------
# Distribution-shift fallback
# ---------------------------------------------------------------------------


class TestDistributionShift:
    def test_no_shift_when_within_threshold(self):
        guard = DistributionShiftGuard(
            training_means=[0.0, 0.0, 0.0],
            config=DistributionShiftConfig(max_mean_distance=1.5, min_window=2),
        )
        for _ in range(5):
            guard.observe([0.1, 0.1, 0.1])
        assert not guard.shifted()

    def test_shift_detected_when_far_from_training(self):
        guard = DistributionShiftGuard(
            training_means=[0.0, 0.0, 0.0],
            config=DistributionShiftConfig(max_mean_distance=0.5, min_window=3),
        )
        for _ in range(5):
            guard.observe([10.0, 10.0, 10.0])
        assert guard.shifted()

    def test_controller_falls_back_on_shift(self):
        guard = DistributionShiftGuard(
            training_means=[0.0] * len(FEATURE_NAMES),
            config=DistributionShiftConfig(max_mean_distance=0.001, min_window=2),
        )
        candidate = LearnedPolicy(policy_id="test", confidence_threshold=0.0)
        answer_idx = list(Action.all()).index(Action.ANSWER_DIRECT)
        candidate.bias[answer_idx] = 100.0
        controller = ExecutiveController(
            baseline=BaselinePolicy(),
            learned=candidate,
            shift_guard=guard,
        )
        # Observe enough points to trigger shift detection.
        for _ in range(5):
            dec = controller.select_action(_state("hello"))
        # The last decision should have detected distribution shift.
        assert dec.distribution_shift_detected is True
        assert dec.policy_fallback is True


# ---------------------------------------------------------------------------
# PolicyRegistry
# ---------------------------------------------------------------------------


class TestPolicyRegistry:
    def test_register_and_load(self, tmp_path):
        registry = PolicyRegistry(tmp_path / "policies")
        policy = LearnedPolicy(policy_id="p1", policy_version="v1")
        manifest = registry.register(
            policy,
            training_data_digest="abc",
            split_manifest_digest="def",
        )
        assert manifest.policy_id == "p1"
        loaded = registry.load_policy("p1")
        assert loaded.policy_id == "p1"

    def test_register_is_write_once(self, tmp_path):
        registry = PolicyRegistry(tmp_path / "policies")
        policy = LearnedPolicy(policy_id="p1")
        registry.register(policy, training_data_digest="a", split_manifest_digest="b")
        with pytest.raises(FileExistsError):
            registry.register(policy, training_data_digest="a", split_manifest_digest="b")

    def test_set_active_flips_pointer(self, tmp_path):
        registry = PolicyRegistry(tmp_path / "policies")
        policy = LearnedPolicy(policy_id="p1")
        registry.register(policy, training_data_digest="a", split_manifest_digest="b")
        registry.set_active("p1", reason="test")
        assert registry.get_active_id() == "p1"
        active = registry.load_active()
        assert active is not None
        assert active[0].policy_id == "p1"
        assert active[1].promotion_status == PROMOTION_STATUS_ACTIVE

    def test_reject_marks_rejected(self, tmp_path):
        registry = PolicyRegistry(tmp_path / "policies")
        policy = LearnedPolicy(policy_id="p1")
        registry.register(policy, training_data_digest="a", split_manifest_digest="b")
        registry.reject("p1", reason="failed gate")
        manifest = registry.load_manifest("p1")
        assert manifest.promotion_status == PROMOTION_STATUS_REJECTED

    def test_cannot_activate_rejected(self, tmp_path):
        registry = PolicyRegistry(tmp_path / "policies")
        policy = LearnedPolicy(policy_id="p1")
        registry.register(policy, training_data_digest="a", split_manifest_digest="b")
        registry.reject("p1", reason="failed")
        with pytest.raises(RuntimeError):
            registry.set_active("p1", reason="attempt")

    def test_previous_active_archived_on_promotion(self, tmp_path):
        registry = PolicyRegistry(tmp_path / "policies")
        p1 = LearnedPolicy(policy_id="p1")
        p2 = LearnedPolicy(policy_id="p2")
        registry.register(p1, training_data_digest="a", split_manifest_digest="b")
        registry.register(p2, training_data_digest="c", split_manifest_digest="d")
        registry.set_active("p1", reason="first")
        registry.set_active("p2", reason="second")
        assert registry.get_active_id() == "p2"
        assert registry.load_manifest("p1").promotion_status == "archived"


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_ci_contains_mean(self):
        deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = bootstrap_ci(deltas, iterations=500, seed=0)
        mean = sum(deltas) / len(deltas)
        assert lo <= mean <= hi

    def test_ci_is_deterministic(self):
        deltas = [1.0, -2.0, 3.0, 0.5, -0.5]
        lo1, hi1 = bootstrap_ci(deltas, iterations=500, seed=42)
        lo2, hi2 = bootstrap_ci(deltas, iterations=500, seed=42)
        assert lo1 == lo2
        assert hi1 == hi2


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------


class TestSchemaRoundTrip:
    def test_executive_experience_round_trip(self):
        exp = ExecutiveExperience(
            task_id="t1",
            state=_state("hello"),
            chosen_action=Action.CALL_TOOL,
            action_metadata=ActionMetadata(tool_name="my_tool"),
            outcome=Outcome(verified_success=True, verification_status="pass", latency_ms=500),
            utility=8.5,
            policy_version="learned_v1",
        )
        d = exp.to_dict()
        restored = ExecutiveExperience.from_dict(d)
        assert restored.task_id == "t1"
        assert restored.chosen_action == Action.CALL_TOOL
        assert restored.outcome.verified_success is True
        assert restored.action_metadata.tool_name == "my_tool"
        assert restored.utility == 8.5

    def test_schema_version_present(self):
        exp = ExecutiveExperience()
        assert exp.schema_version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# AutoLearnService
# ---------------------------------------------------------------------------


class TestAutoLearnService:
    def test_service_starts_with_baseline(self, tmp_path):
        import asyncio
        svc = AutoLearnService(cfg={"policy_root": str(tmp_path / "policies")})
        asyncio.run(svc.start())
        assert svc.active_policy_id is None
        assert svc.controller.has_learned_policy is False
        asyncio.run(svc.stop())

    def test_service_loads_active_policy(self, tmp_path):
        import asyncio
        registry = PolicyRegistry(tmp_path / "policies")
        policy = LearnedPolicy(policy_id="p1", policy_version="v1")
        registry.register(policy, training_data_digest="a", split_manifest_digest="b")
        registry.set_active("p1", reason="test")

        svc = AutoLearnService(cfg={"policy_root": str(tmp_path / "policies")})
        asyncio.run(svc.start())
        assert svc.active_policy_id == "p1"
        assert svc.controller.has_learned_policy is True
        asyncio.run(svc.stop())

    def test_service_falls_back_on_incompatible_policy(self, tmp_path):
        import asyncio
        # Manually write an incompatible policy artifact.
        registry_root = tmp_path / "policies"
        registry_root.mkdir(parents=True, exist_ok=True)
        policies_dir = registry_root / "policies"
        policies_dir.mkdir(parents=True, exist_ok=True)
        incompatible = {
            "policy_id": "bad",
            "policy_version": "v1",
            "policy_type": "multinomial_logistic",
            "feature_schema_version": "exec_features_v999",
            "feature_names": list(FEATURE_NAMES),
            "actions": [a.value for a in Action.all()],
            "weights": [[0.0] * len(FEATURE_NAMES) for _ in Action.all()],
            "bias": [0.0] * len(Action.all()),
        }
        (policies_dir / "bad.json").write_text(json.dumps(incompatible))
        (policies_dir / "bad.manifest.json").write_text(json.dumps({
            "policy_id": "bad",
            "policy_version": "v1",
            "policy_type": "multinomial_logistic",
            "feature_schema_version": "exec_features_v999",
            "training_data_digest": "",
            "split_manifest_digest": "",
            "promotion_status": "active",
        }))
        (registry_root / "active.json").write_text(json.dumps({"policy_id": "bad"}))

        svc = AutoLearnService(cfg={"policy_root": str(registry_root)})
        asyncio.run(svc.start())
        # The incompatible policy should not be loaded; falls back to baseline.
        assert svc.controller.has_learned_policy is False
        asyncio.run(svc.stop())
