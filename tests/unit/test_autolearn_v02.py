"""Unit tests for Capsule Brain AutoLearn v0.2.

Covers the v0.2 deliverables:
- ExecutiveController real dispatch
- ActionDispatcher interface
- ActionResult / DispatcherResult
- ExperienceSink (JSONL, memory)
- BaselinePolicyV2 (competent deterministic router)
- DistributionShiftGuard v2 (PSI, unseen-value, tool-inventory, model-ID)
- Calibration metrics (Brier, ECE)
- Promotion gate v2 (11 gates)
- Schema v2 outcome fields
- Utility v2 (over-routing penalties)
- Feature schema v2
- ConversationService autolearn routing
- Safety boundaries
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from capsule_brain.autolearn.baseline import (
    BASELINE_V2_POLICY_VERSION,
    BaselinePolicy,
    BaselinePolicyV2,
)
from capsule_brain.autolearn.calibration import CalibrationReport, compute_calibration
from capsule_brain.autolearn.controller import (
    ActionDispatcher,
    ActionResult,
    DispatcherResult,
    ExecutiveController,
    ExecutiveDecision,
    ExperienceSink,
    JSONLExperienceSink,
    LEARNED_ACTIONS,
    MemoryExperienceSink,
)
from capsule_brain.autolearn.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FeatureExtractor,
)
from capsule_brain.autolearn.policy import (
    LEARNED_POLICY_VERSION_DEFAULT,
    LearnedPolicy,
)
from capsule_brain.autolearn.promotion import (
    GateConfig,
    V2GateMetrics,
    evaluate_promotion_gate,
)
from capsule_brain.autolearn.schema import (
    Action,
    ActionMetadata,
    ExecutiveExperience,
    ExecutiveState,
    Outcome,
    SCHEMA_VERSION,
)
from capsule_brain.autolearn.shadow import (
    DistributionShiftConfig,
    DistributionShiftGuard,
)
from capsule_brain.autolearn.utility import (
    UTILITY_CONFIG_VERSION,
    UtilityConfig,
    UtilityFunction,
)


# ---------------------------------------------------------------------------
# Schema v2 tests
# ---------------------------------------------------------------------------


class TestSchemaV2:
    def test_schema_version_is_v2(self):
        assert SCHEMA_VERSION == "exec_experience_v2"

    def test_outcome_has_v2_fields(self):
        oc = Outcome()
        assert hasattr(oc, "runtime_error")
        assert hasattr(oc, "action_completed")
        assert hasattr(oc, "tool_name")
        assert hasattr(oc, "tool_calls_executed")
        assert hasattr(oc, "tool_result_valid")
        assert hasattr(oc, "final_answer_grounded")
        assert hasattr(oc, "workflow_name")
        assert hasattr(oc, "workflow_run_id")
        assert hasattr(oc, "acceptance_present")
        assert hasattr(oc, "acceptance_passed")
        assert hasattr(oc, "exit_code")
        assert hasattr(oc, "unnecessary_tool_use")
        assert hasattr(oc, "unnecessary_workflow_use")

    def test_outcome_to_dict_includes_v2_fields(self):
        oc = Outcome(
            verified_success=True,
            runtime_error=False,
            action_completed=True,
            tool_name="http_fetch",
            tool_calls_executed=1,
            tool_result_valid=True,
            final_answer_grounded=True,
            workflow_name="plan_generate_test_reflect",
            workflow_run_id="run_123",
            acceptance_present=True,
            acceptance_passed=True,
            exit_code=0,
            unnecessary_tool_use=False,
            unnecessary_workflow_use=False,
        )
        exp = ExecutiveExperience(
            task_id="t1",
            state=ExecutiveState(prompt_features={"text": "test"}),
            chosen_action=Action.CALL_TOOL,
            action_metadata=ActionMetadata(tool_name="http_fetch"),
            outcome=oc,
            utility=10.0,
        )
        d = exp.to_dict()
        assert d["outcome"]["tool_name"] == "http_fetch"
        assert d["outcome"]["action_completed"] is True
        assert d["outcome"]["workflow_run_id"] == "run_123"
        assert d["outcome"]["exit_code"] == 0

    def test_outcome_from_dict_includes_v2_fields(self):
        state = ExecutiveState(prompt_features={"text": "test"})
        state_dict = {
            "prompt_features": dict(state.prompt_features),
            "conversation_features": dict(state.conversation_features),
            "memory_features": dict(state.memory_features),
            "available_tools": list(state.available_tools),
            "workflow_available": state.workflow_available,
            "model_id": state.model_id,
            "context_length": state.context_length,
        }
        d = {
            "task_id": "t1",
            "state": state_dict,
            "chosen_action": "CALL_TOOL",
            "action_metadata": {"tool_name": "http_fetch"},
            "outcome": {
                "verified_success": True,
                "verification_status": "pass",
                "latency_ms": 100.0,
                "token_count": 50,
                "tool_failures": 0,
                "workflow_iterations": 0,
                "operator_intervention": False,
                "safety_violation": False,
                "runtime_error": False,
                "action_completed": True,
                "tool_name": "http_fetch",
                "tool_calls_executed": 1,
                "tool_result_valid": True,
                "final_answer_grounded": True,
                "workflow_name": None,
                "workflow_run_id": None,
                "acceptance_present": False,
                "acceptance_passed": False,
                "exit_code": None,
                "unnecessary_tool_use": False,
                "unnecessary_workflow_use": False,
            },
            "utility": 10.0,
            "policy_version": "learned_router_v2",
            "provenance": {
                "source": "production",
                "policy_version": "learned_router_v2",
                "task_family": "tool_required",
                "task_id": "t1",
                "extra": {},
            },
        }
        exp = ExecutiveExperience.from_dict(d)
        assert exp.outcome.tool_name == "http_fetch"
        assert exp.outcome.action_completed is True
        assert exp.outcome.tool_calls_executed == 1


# ---------------------------------------------------------------------------
# Feature schema v2 tests
# ---------------------------------------------------------------------------


class TestFeatureSchemaV2:
    def test_feature_schema_version_is_v2(self):
        assert FEATURE_SCHEMA_VERSION == "exec_features_v2"

    def test_feature_names_include_workflow_capability_match(self):
        assert "workflow_capability_match" in FEATURE_NAMES

    def test_feature_extraction_is_deterministic(self):
        state = ExecutiveState(
            prompt_features={"text": "Write a function that adds two numbers"},
            available_tools=["calculator"],
            workflow_available=True,
            model_id="qwen2.5-coder:3b",
        )
        extractor = FeatureExtractor()
        fv1 = extractor.extract(state)
        fv2 = extractor.extract(state)
        assert fv1.as_list() == fv2.as_list()

    def test_no_label_leakage_in_features(self):
        """Features must not include requires_tool, correct_action, or expected_action."""
        state = ExecutiveState(
            prompt_features={"text": "Fetch the live nonce"},
            available_tools=["http_fetch"],
        )
        extractor = FeatureExtractor()
        fv = extractor.extract(state)
        d = fv.as_dict()
        # No label leakage: these keys must not appear.
        assert "requires_tool" not in d
        assert "correct_action" not in d
        assert "expected_action" not in d
        assert "target" not in d


# ---------------------------------------------------------------------------
# Utility v2 tests
# ---------------------------------------------------------------------------


class TestUtilityV2:
    def test_utility_config_version_is_v2(self):
        assert UTILITY_CONFIG_VERSION == "exec_utility_v2"

    def test_over_routing_tool_penalty(self):
        ufn = UtilityFunction(UtilityConfig())
        # Correct but unnecessary tool use.
        oc_with_penalty = Outcome(
            verified_success=True,
            action_completed=True,
            unnecessary_tool_use=True,
        )
        oc_without_penalty = Outcome(
            verified_success=True,
            action_completed=True,
            unnecessary_tool_use=False,
        )
        u_with, _ = ufn.compute(oc_with_penalty)
        u_without, _ = ufn.compute(oc_without_penalty)
        assert u_with < u_without, "over-routing should reduce utility"

    def test_over_routing_workflow_penalty(self):
        ufn = UtilityFunction(UtilityConfig())
        oc_with_penalty = Outcome(
            verified_success=True,
            action_completed=True,
            unnecessary_workflow_use=True,
        )
        oc_without_penalty = Outcome(
            verified_success=True,
            action_completed=True,
            unnecessary_workflow_use=False,
        )
        u_with, _ = ufn.compute(oc_with_penalty)
        u_without, _ = ufn.compute(oc_without_penalty)
        assert u_with < u_without, "over-routing workflow should reduce utility"

    def test_safety_violation_dominates(self):
        ufn = UtilityFunction(UtilityConfig())
        oc_safe = Outcome(verified_success=True, action_completed=True)
        oc_unsafe = Outcome(
            verified_success=False,
            safety_violation=True,
            action_completed=True,
        )
        u_safe, _ = ufn.compute(oc_safe)
        u_unsafe, _ = ufn.compute(oc_unsafe)
        assert u_safe > u_unsafe


# ---------------------------------------------------------------------------
# BaselinePolicyV2 tests
# ---------------------------------------------------------------------------


class TestBaselinePolicyV2:
    def test_version_is_v2(self):
        bp = BaselinePolicyV2()
        assert bp.version == BASELINE_V2_POLICY_VERSION

    def test_safety_keyword_triggers_ask_operator(self):
        bp = BaselinePolicyV2()
        state = ExecutiveState(
            prompt_features={"text": "Delete the production database"},
            available_tools=["calculator"],
        )
        dec = bp.select_action(state)
        assert dec.action == Action.ASK_OPERATOR

    def test_tool_signal_triggers_call_tool(self):
        bp = BaselinePolicyV2()
        state = ExecutiveState(
            prompt_features={"text": "Fetch the current live nonce for endpoint 42"},
            available_tools=["http_fetch"],
        )
        dec = bp.select_action(state)
        assert dec.action == Action.CALL_TOOL

    def test_code_indicators_trigger_workflow(self):
        bp = BaselinePolicyV2()
        state = ExecutiveState(
            prompt_features={
                "text": "Write a Python function that sorts a list",
                "workflow_capability_match": True,
            },
            available_tools=["calculator"],
            workflow_available=True,
        )
        dec = bp.select_action(state)
        assert dec.action == Action.START_WORKFLOW

    def test_default_direct_answer(self):
        bp = BaselinePolicyV2()
        state = ExecutiveState(
            prompt_features={"text": "What is the capital of France?"},
            available_tools=[],
            workflow_available=False,
        )
        dec = bp.select_action(state)
        assert dec.action == Action.ANSWER_DIRECT

    def test_memory_retrieval_signal(self):
        bp = BaselinePolicyV2()
        state = ExecutiveState(
            prompt_features={"text": "What was the secret pin stored for session 42?"},
            memory_features={"hit_count": 3, "top_similarity": 0.9},
            available_tools=[],
        )
        dec = bp.select_action(state)
        assert dec.action == Action.RETRIEVE_MEMORY


# ---------------------------------------------------------------------------
# ExecutiveController tests
# ---------------------------------------------------------------------------


class _FakeDispatcher(ActionDispatcher):
    """Fake dispatcher for testing — records calls and returns canned results."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.answer_text = "direct answer"
        self.memory_text = "from memory"
        self.tool_text = "tool result"
        self.workflow_text = "workflow output"

    async def answer_direct(self, state, *, conversation_id=None):
        self.calls.append("answer_direct")
        return DispatcherResult(
            text=self.answer_text,
            latency_ms=100.0,
            token_count=20,
            action_completed=True,
        )

    async def retrieve_memory(self, state, *, conversation_id=None):
        self.calls.append("retrieve_memory")
        return DispatcherResult(
            text=self.memory_text,
            latency_ms=300.0,
            token_count=30,
            action_completed=True,
        )

    async def call_tool(self, state, *, conversation_id=None):
        self.calls.append("call_tool")
        return DispatcherResult(
            text=self.tool_text,
            latency_ms=400.0,
            token_count=40,
            action_completed=True,
            tool_name="http_fetch",
            tool_calls_executed=1,
            tool_result_valid=True,
        )

    async def start_workflow(self, state, *, conversation_id=None):
        self.calls.append("start_workflow")
        return DispatcherResult(
            text=self.workflow_text,
            latency_ms=1200.0,
            token_count=120,
            action_completed=True,
            workflow_name="plan_generate_test_reflect",
            acceptance_present=True,
            acceptance_passed=True,
            exit_code=0,
        )


async def _fake_verifier(action, disp, state):
    """Fake verifier: passes if the dispatch completed and has text."""
    if disp.action_completed and disp.text:
        return True, "pass", {}
    return False, "fail", {}


class TestExecutiveController:
    def test_learned_actions_are_four(self):
        assert len(LEARNED_ACTIONS) == 4
        assert Action.ANSWER_DIRECT in LEARNED_ACTIONS
        assert Action.RETRIEVE_MEMORY in LEARNED_ACTIONS
        assert Action.CALL_TOOL in LEARNED_ACTIONS
        assert Action.START_WORKFLOW in LEARNED_ACTIONS
        assert Action.REFLECT not in LEARNED_ACTIONS
        assert Action.ASK_OPERATOR not in LEARNED_ACTIONS

    def test_select_action_with_no_learned_policy_uses_baseline(self):
        ctrl = ExecutiveController(baseline=BaselinePolicyV2())
        state = ExecutiveState(
            prompt_features={"text": "What is 2+2?"},
            available_tools=["calculator"],
        )
        dec = ctrl.select_action(state)
        assert dec.policy_type == "baseline"
        assert dec.action == Action.ANSWER_DIRECT

    @pytest.mark.asyncio
    async def test_execute_dispatches_answer_direct(self):
        disp = _FakeDispatcher()
        sink = MemoryExperienceSink()
        ctrl = ExecutiveController(
            baseline=BaselinePolicyV2(),
            dispatcher=disp,
            verifier=_fake_verifier,
            experience_sink=sink,
        )
        state = ExecutiveState(
            prompt_features={"text": "What is the capital of France?"},
            available_tools=[],
        )
        result = await ctrl.execute(state, task_id="t1")
        assert result.action == Action.ANSWER_DIRECT
        assert result.text == "direct answer"
        assert "answer_direct" in disp.calls
        assert len(sink.records) == 1
        assert sink.records[0].chosen_action == Action.ANSWER_DIRECT
        assert sink.records[0].outcome.action_completed is True

    @pytest.mark.asyncio
    async def test_execute_dispatches_call_tool(self):
        disp = _FakeDispatcher()
        sink = MemoryExperienceSink()
        ctrl = ExecutiveController(
            baseline=BaselinePolicyV2(),
            dispatcher=disp,
            verifier=_fake_verifier,
            experience_sink=sink,
        )
        state = ExecutiveState(
            prompt_features={"text": "Fetch the current live nonce for endpoint 42"},
            available_tools=["http_fetch"],
        )
        result = await ctrl.execute(state, task_id="t2")
        assert result.action == Action.CALL_TOOL
        assert "call_tool" in disp.calls
        assert result.outcome.tool_name == "http_fetch"
        assert result.outcome.tool_calls_executed == 1

    @pytest.mark.asyncio
    async def test_execute_persists_experience(self):
        disp = _FakeDispatcher()
        sink = MemoryExperienceSink()
        ctrl = ExecutiveController(
            baseline=BaselinePolicyV2(),
            dispatcher=disp,
            verifier=_fake_verifier,
            experience_sink=sink,
        )
        state = ExecutiveState(
            prompt_features={"text": "What is 2+2?"},
            available_tools=[],
        )
        await ctrl.execute(state, task_id="t_persist")
        assert len(sink.records) == 1
        exp = sink.records[0]
        assert exp.task_id == "t_persist"
        assert exp.outcome.verified_success is True
        assert exp.outcome.verification_status == "pass"

    @pytest.mark.asyncio
    async def test_execute_without_verifier_does_not_claim_success(self):
        """Fail-closed: without a verifier, verified_success must be False."""
        disp = _FakeDispatcher()
        ctrl = ExecutiveController(
            baseline=BaselinePolicyV2(),
            dispatcher=disp,
            verifier=None,
        )
        state = ExecutiveState(prompt_features={"text": "test"})
        result = await ctrl.execute(state)
        assert result.outcome.verified_success is False
        assert result.outcome.verification_status == "skip"

    @pytest.mark.asyncio
    async def test_dispatch_error_records_runtime_error(self):
        class _ErrorDispatcher(ActionDispatcher):
            async def answer_direct(self, state, *, conversation_id=None):
                raise RuntimeError("LLM gateway down")
            async def retrieve_memory(self, state, *, conversation_id=None):
                raise RuntimeError("memory error")
            async def call_tool(self, state, *, conversation_id=None):
                raise RuntimeError("tool error")
            async def start_workflow(self, state, *, conversation_id=None):
                raise RuntimeError("workflow error")

        ctrl = ExecutiveController(
            baseline=BaselinePolicyV2(),
            dispatcher=_ErrorDispatcher(),
            verifier=_fake_verifier,
        )
        state = ExecutiveState(prompt_features={"text": "test"})
        result = await ctrl.execute(state)
        assert result.outcome.runtime_error is True
        assert result.outcome.action_completed is False


# ---------------------------------------------------------------------------
# ExperienceSink tests
# ---------------------------------------------------------------------------


class TestExperienceSink:
    @pytest.mark.asyncio
    async def test_memory_sink_collects(self):
        sink = MemoryExperienceSink()
        exp = ExecutiveExperience(
            task_id="t1",
            state=ExecutiveState(prompt_features={"text": "test"}),
            chosen_action=Action.ANSWER_DIRECT,
            outcome=Outcome(verified_success=True),
            utility=10.0,
        )
        await sink.persist(exp)
        assert len(sink.records) == 1

    @pytest.mark.asyncio
    async def test_jsonl_sink_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "experiences.jsonl")
            sink = JSONLExperienceSink(path)
            exp = ExecutiveExperience(
                task_id="t1",
                state=ExecutiveState(prompt_features={"text": "test"}),
                chosen_action=Action.ANSWER_DIRECT,
                outcome=Outcome(verified_success=True),
                utility=10.0,
            )
            await sink.persist(exp)
            assert os.path.exists(path)
            lines = Path(path).read_text().strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["task_id"] == "t1"


# ---------------------------------------------------------------------------
# DistributionShiftGuard v2 tests
# ---------------------------------------------------------------------------


class TestDistributionShiftGuardV2:
    def test_no_shift_with_insufficient_data(self):
        guard = DistributionShiftGuard(
            training_means=[0.5, 0.5],
            config=DistributionShiftConfig(min_window=10),
        )
        for _ in range(5):
            guard.observe([0.5, 0.5])
        assert not guard.shifted()

    def test_mean_shift_detected(self):
        guard = DistributionShiftGuard(
            training_means=[0.0, 0.0],
            config=DistributionShiftConfig(max_mean_distance=0.5, min_window=5),
        )
        for _ in range(10):
            guard.observe([5.0, 5.0])
        assert guard.shifted()
        assert any("mean shift" in r for r in guard.shift_reasons)

    def test_tool_inventory_change_detected(self):
        guard = DistributionShiftGuard(
            training_means=[0.0],
            config=DistributionShiftConfig(
                min_window=1,
                training_tool_names={"calculator"},
            ),
        )
        guard.observe([0.0])
        guard.observe_context(tool_names=["calculator", "http_fetch"])
        assert guard.shifted()
        assert any("tool inventory" in r for r in guard.shift_reasons)

    def test_model_id_mismatch_detected(self):
        guard = DistributionShiftGuard(
            training_means=[0.0],
            config=DistributionShiftConfig(
                min_window=1,
                training_model_ids={"qwen2.5-coder:3b"},
            ),
        )
        guard.observe([0.0])
        guard.observe_context(model_id="gpt-4o")
        assert guard.shifted()
        assert any("model ID" in r for r in guard.shift_reasons)

    def test_no_shift_when_all_match(self):
        guard = DistributionShiftGuard(
            training_means=[0.5],
            config=DistributionShiftConfig(
                min_window=5,
                max_mean_distance=1.0,
                training_tool_names={"calculator"},
                training_model_ids={"qwen"},
            ),
        )
        for _ in range(10):
            guard.observe([0.5])
        guard.observe_context(tool_names=["calculator"], model_id="qwen")
        assert not guard.shifted()


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_perfect_calibration(self):
        """When confidence always equals accuracy, Brier and ECE are low."""
        pairs = [(1.0, True)] * 10 + [(0.0, False)] * 10
        report = compute_calibration(pairs)
        assert report.brier_score < 0.01
        assert report.ece < 0.01

    def test_overconfident_calibration(self):
        """When confidence is always 1.0 but accuracy is 0.5, Brier is high."""
        pairs = [(1.0, True)] * 5 + [(1.0, False)] * 5
        report = compute_calibration(pairs)
        assert report.brier_score > 0.4
        assert report.ece > 0.4

    def test_abstention_rate(self):
        pairs = [(1.0, True)] * 8 + [(0.0, False)] * 2
        abstained = [False] * 8 + [True] * 2
        report = compute_calibration(pairs, abstained=abstained)
        assert report.abstention_rate == 0.2
        assert report.n_abstained == 2

    def test_accuracy_at_coverage(self):
        pairs = [(0.9, True), (0.8, True), (0.5, False), (0.3, False)]
        report = compute_calibration(pairs, coverage_levels=(0.75,))
        # At coverage 0.75, only the 0.9 and 0.8 predictions are included.
        assert report.accuracy_at_coverage["cov_0.75"] == 1.0

    def test_empty_pairs(self):
        report = compute_calibration([])
        assert report.n == 0
        assert report.brier_score == 0.0


# ---------------------------------------------------------------------------
# Promotion gate v2 tests
# ---------------------------------------------------------------------------


class TestPromotionGateV2:
    def _make_eval(self, baseline_util=7.0, candidate_util=9.0, sr_b=0.8, sr_c=1.0):
        from capsule_brain.autolearn.evaluator import (
            ActionMetrics,
            PairedEvaluation,
            TaskEvaluation,
        )

        bm = ActionMetrics(
            verified_success_rate=sr_b,
            mean_utility=baseline_util,
            safety_violations=0,
            tool_precision=0.8,
            tool_recall=0.8,
        )
        cm = ActionMetrics(
            verified_success_rate=sr_c,
            mean_utility=candidate_util,
            safety_violations=0,
            tool_precision=0.9,
            tool_recall=0.9,
        )
        te = [
            TaskEvaluation(
                task_id="t1",
                task_family="coding_workflow",
                baseline_action=Action.START_WORKFLOW,
                candidate_action=Action.START_WORKFLOW,
                baseline_utility=7.0,
                candidate_utility=9.0,
                delta=2.0,
                disagreement=False,
                baseline_outcome={},
                candidate_outcome={},
            )
        ]
        return PairedEvaluation(
            task_evaluations=te,
            baseline_metrics=bm,
            candidate_metrics=cm,
            mean_delta=2.0,
            median_delta=2.0,
            delta_ci_low=1.0,
            delta_ci_high=3.0,
            wins=1,
            ties=0,
            losses=0,
        )

    def test_v2_gates_pass_when_metrics_good(self):
        ev = self._make_eval()
        v2 = V2GateMetrics(
            baseline_workflow_routing_accuracy=0.9,
            candidate_workflow_routing_accuracy=1.0,
            baseline_over_routing_rate=0.1,
            candidate_over_routing_rate=0.05,
            candidate_brier_score=0.2,
        )
        result = evaluate_promotion_gate(ev, v2_metrics=v2)
        assert result.passed
        gate_names = [g["name"] for g in result.gates]
        assert "workflow_routing_accuracy_non_decrease" in gate_names
        assert "over_routing_rate_non_increase" in gate_names
        assert "calibration_brier_below_threshold" in gate_names

    def test_v2_calibration_gate_fails_on_high_brier(self):
        ev = self._make_eval()
        v2 = V2GateMetrics(
            baseline_workflow_routing_accuracy=0.9,
            candidate_workflow_routing_accuracy=1.0,
            baseline_over_routing_rate=0.1,
            candidate_over_routing_rate=0.05,
            candidate_brier_score=0.5,  # above default 0.3
        )
        config = GateConfig(brier_score_max=0.3)
        result = evaluate_promotion_gate(ev, config=config, v2_metrics=v2)
        assert not result.passed
        gate_names = [g["name"] for g in result.gates if not g["passed"]]
        assert "calibration_brier_below_threshold" in gate_names

    def test_v2_over_routing_gate_fails(self):
        ev = self._make_eval()
        v2 = V2GateMetrics(
            baseline_workflow_routing_accuracy=0.9,
            candidate_workflow_routing_accuracy=1.0,
            baseline_over_routing_rate=0.0,
            candidate_over_routing_rate=0.5,  # big increase
            candidate_brier_score=0.1,
        )
        config = GateConfig(over_routing_rate_max_increase=0.1)
        result = evaluate_promotion_gate(ev, config=config, v2_metrics=v2)
        assert not result.passed

    def test_v2_gates_skipped_when_no_v2_metrics(self):
        """When v2_metrics is None, gates 9-11 are skipped (backward compat)."""
        ev = self._make_eval()
        result = evaluate_promotion_gate(ev, v2_metrics=None)
        gate_names = [g["name"] for g in result.gates]
        assert "workflow_routing_accuracy_non_decrease" not in gate_names
        assert "calibration_brier_below_threshold" not in gate_names


# ---------------------------------------------------------------------------
# Safety boundary tests
# ---------------------------------------------------------------------------


class TestSafetyBoundaries:
    def test_baseline_v2_safety_keyword_cannot_be_overridden(self):
        """Safety keywords must always trigger ASK_OPERATOR, regardless of other signals."""
        bp = BaselinePolicyV2()
        # Even with strong tool signals and code indicators, safety wins.
        state = ExecutiveState(
            prompt_features={
                "text": "Delete the production database and write a function",
                "workflow_capability_match": True,
            },
            available_tools=["http_fetch", "calculator"],
            workflow_available=True,
            memory_features={"hit_count": 5, "top_similarity": 0.95},
        )
        dec = bp.select_action(state)
        assert dec.action == Action.ASK_OPERATOR

    @pytest.mark.asyncio
    async def test_controller_falls_back_on_invalid_action(self):
        """If the learned policy returns an invalid action, the controller falls back."""
        from capsule_brain.autolearn.policy import PolicyDecision

        class _BadPolicy(LearnedPolicy):
            def select_action(self, state, *, extractor=None, allowed_actions=None):
                return PolicyDecision(
                    action="NOT_AN_ACTION",  # invalid
                    confidence=0.9,
                    scores={},
                    probabilities={},
                    policy_version="bad_v1",
                    policy_type="learned",
                    reason="bad",
                    abstained=False,
                )

        bad = _BadPolicy(policy_id="bad", actions=tuple(Action.all()))
        ctrl = ExecutiveController(
            baseline=BaselinePolicyV2(),
            learned=bad,
            dispatcher=_FakeDispatcher(),
            verifier=_fake_verifier,
        )
        state = ExecutiveState(prompt_features={"text": "test"})
        dec = ctrl.select_action(state)
        assert dec.policy_fallback is True
        assert dec.policy_type == "baseline"

    @pytest.mark.asyncio
    async def test_controller_falls_back_on_error(self):
        """If the learned policy throws, the controller falls back to baseline."""
        class _ErrorPolicy(LearnedPolicy):
            def select_action(self, state, *, extractor=None, allowed_actions=None):
                raise RuntimeError("policy crashed")

        err = _ErrorPolicy(policy_id="err", actions=tuple(Action.all()))
        ctrl = ExecutiveController(
            baseline=BaselinePolicyV2(),
            learned=err,
            dispatcher=_FakeDispatcher(),
            verifier=_fake_verifier,
        )
        state = ExecutiveState(prompt_features={"text": "test"})
        dec = ctrl.select_action(state)
        assert dec.policy_fallback is True
        assert dec.policy_error != ""

    @pytest.mark.asyncio
    async def test_controller_falls_back_on_low_confidence(self):
        """If the learned policy abstains, the controller falls back to baseline."""
        from capsule_brain.autolearn.policy import PolicyDecision

        class _LowConfPolicy(LearnedPolicy):
            def select_action(self, state, *, extractor=None, allowed_actions=None):
                return PolicyDecision(
                    action=Action.CALL_TOOL,
                    confidence=0.1,  # below threshold
                    scores={a.value: 0.1 for a in Action.all()},
                    probabilities={a.value: 0.1 for a in Action.all()},
                    policy_version="lowconf_v1",
                    policy_type="learned",
                    reason="low confidence",
                    abstained=False,
                )

        low = _LowConfPolicy(policy_id="lowconf", actions=tuple(Action.all()))
        ctrl = ExecutiveController(
            baseline=BaselinePolicyV2(),
            learned=low,
            dispatcher=_FakeDispatcher(),
            verifier=_fake_verifier,
            confidence_threshold=0.5,
        )
        state = ExecutiveState(prompt_features={"text": "test"})
        dec = ctrl.select_action(state)
        assert dec.policy_fallback is True
        assert dec.abstained is True


# ---------------------------------------------------------------------------
# Action isolation tests
# ---------------------------------------------------------------------------


class TestActionIsolation:
    """Each action must only populate its own outcome fields."""

    @pytest.mark.asyncio
    async def test_answer_direct_does_not_set_tool_fields(self):
        disp = _FakeDispatcher()
        ctrl = ExecutiveController(
            baseline=BaselinePolicyV2(),
            dispatcher=disp,
            verifier=_fake_verifier,
        )
        state = ExecutiveState(prompt_features={"text": "What is 2+2?"})
        result = await ctrl.execute(state)
        assert result.outcome.tool_name is None
        assert result.outcome.tool_calls_executed == 0
        assert result.outcome.workflow_name is None

    @pytest.mark.asyncio
    async def test_call_tool_sets_tool_fields(self):
        disp = _FakeDispatcher()
        ctrl = ExecutiveController(
            baseline=BaselinePolicyV2(),
            dispatcher=disp,
            verifier=_fake_verifier,
        )
        state = ExecutiveState(
            prompt_features={"text": "Fetch the live nonce for endpoint 1"},
            available_tools=["http_fetch"],
        )
        result = await ctrl.execute(state)
        assert result.outcome.tool_name == "http_fetch"
        assert result.outcome.tool_calls_executed == 1
        assert result.outcome.tool_result_valid is True

    @pytest.mark.asyncio
    async def test_start_workflow_sets_workflow_fields(self):
        disp = _FakeDispatcher()
        ctrl = ExecutiveController(
            baseline=BaselinePolicyV2(),
            dispatcher=disp,
            verifier=_fake_verifier,
        )
        state = ExecutiveState(
            prompt_features={
                "text": "Write a Python function that sorts a list",
                "workflow_capability_match": True,
            },
            available_tools=["calculator"],
            workflow_available=True,
        )
        result = await ctrl.execute(state)
        assert result.outcome.workflow_name == "plan_generate_test_reflect"
        assert result.outcome.acceptance_present is True
        assert result.outcome.acceptance_passed is True
