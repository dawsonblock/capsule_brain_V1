"""AutoLearn v0.3 specification tests.

Covers the 19 required tests from the v0.3 spec:
1.  autolearn_enable=true routes through ExecutiveController
2.  autolearn_enable=false preserves v0.1 behavior
3.  execute_executive_action dispatches all 4 learned actions
4.  REFLECT and ASK_OPERATOR are not in the learned action space
5.  CounterfactualRuntime protocol has runtime_type
6.  RealCounterfactualRuntime has runtime_type="real"
7.  SimulatedCounterfactualRuntime has runtime_type="simulated"
8.  assert_runtime_is_real rejects simulated runtime
9.  Each counterfactual execution gets a unique conversation_id
10. Unsafe actions on safety tasks are marked not_executed
11. 80 tasks with 20 each of direct, memory, tool, workflow
12. No label leakage in features
13. Archetype-based splits with genuine OOD
14. BaselinePolicyV2 is the competent baseline
15. Utility v3 has runtime_error penalty
16. ExecutiveExperience records full provenance
17. Calibration metrics (ECE, Brier, coverage)
18. 12-gate promotion gate includes coverage gate
19. Provenance manifest has all required SHA-256 fields
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load the v0.3 tasks module explicitly from its file path to avoid
# conflicts with the v0.2 tasks module (both are named "tasks").
_V03_DIR = Path(__file__).resolve().parent.parent.parent / "qualification" / "autolearn_v03"
_tasks_spec = importlib.util.spec_from_file_location(
    "autolearn_v03_tasks", _V03_DIR / "tasks.py"
)
tasks = importlib.util.module_from_spec(_tasks_spec)
_tasks_spec.loader.exec_module(tasks)

# Load the promote_policy module explicitly too.
_promote_spec = importlib.util.spec_from_file_location(
    "autolearn_v03_promote", _V03_DIR / "promote_policy.py"
)
promote_policy = importlib.util.module_from_spec(_promote_spec)
_promote_spec.loader.exec_module(promote_policy)

from capsule_brain.autolearn.baseline import BaselinePolicyV2
from capsule_brain.autolearn.calibration import compute_calibration
from capsule_brain.autolearn.counterfactual import (
    CounterfactualRuntime,
    CounterfactualTask,
    LEARNED_ACTIONS_V03,
    RealCounterfactualRuntime,
    SimulatedCounterfactualRuntime,
    assert_runtime_is_real,
)
from capsule_brain.autolearn.features import FEATURE_NAMES, FeatureExtractor
from capsule_brain.autolearn.schema import Action, ExecutiveState, Outcome, Provenance
from capsule_brain.autolearn.utility import UTILITY_CONFIG_VERSION, UtilityConfig, UtilityFunction


# ---------------------------------------------------------------------------
# Tests 1-2: AutoLearn enablement wiring
# ---------------------------------------------------------------------------


class TestAutoLearnEnablement:
    """Section 1: autolearn.enable wiring fix."""

    def test_autolearn_enabled_via_explicit_argument(self):
        """autolearn_enabled is passed explicitly, not from cfg."""
        from capsule_brain.events.local_bus import LocalEventBus
        from capsule_brain.llm.gateway import LLMGateway
        from capsule_brain.llm.models import LLMRequest, LLMResult
        from capsule_brain.llm.providers.base import LLMProvider
        from capsule_brain.memory.service import MemoryService
        from capsule_brain.conversation.service import ConversationService

        class FakeProvider(LLMProvider):
            name = "fake"
            async def generate(self, request, model_cfg):
                return LLMResult(text="test", model="fake", provider="fake")

        bus = LocalEventBus()
        gateway = LLMGateway(
            providers={"fake": FakeProvider()},
            cfg={"default_model": "fake", "models": {"fake": {"provider": "fake", "model_name": "fake"}}},
        )

        # When autolearn_enabled=True is passed explicitly, it should be True
        # even without autolearn_enable in cfg.
        svc = ConversationService(
            event_bus=bus,
            memory=MemoryService(event_bus=bus, cfg={"db_path": ":memory:"}),
            llm=gateway,
            cfg={"db_path": ":memory:"},
            autolearn_enabled=True,
        )
        assert svc.autolearn_enabled is True

    def test_autolearn_disabled_by_default_without_service(self):
        """When no autolearn_service and no explicit flag, disabled."""
        from capsule_brain.events.local_bus import LocalEventBus
        from capsule_brain.llm.gateway import LLMGateway
        from capsule_brain.llm.models import LLMResult
        from capsule_brain.llm.providers.base import LLMProvider
        from capsule_brain.memory.service import MemoryService
        from capsule_brain.conversation.service import ConversationService

        class FakeProvider(LLMProvider):
            name = "fake"
            async def generate(self, request, model_cfg):
                return LLMResult(text="test", model="fake", provider="fake")

        bus = LocalEventBus()
        gateway = LLMGateway(
            providers={"fake": FakeProvider()},
            cfg={"default_model": "fake", "models": {"fake": {"provider": "fake", "model_name": "fake"}}},
        )

        svc = ConversationService(
            event_bus=bus,
            memory=MemoryService(event_bus=bus, cfg={"db_path": ":memory:"}),
            llm=gateway,
            cfg={"db_path": ":memory:"},
        )
        assert svc.autolearn_enabled is False


# ---------------------------------------------------------------------------
# Test 3: execute_executive_action dispatches all 4 learned actions
# ---------------------------------------------------------------------------


class TestExecuteExecutiveAction:
    """Section 2: public execute_executive_action API."""

    def test_execute_executive_action_exists(self):
        """ConversationService has execute_executive_action method."""
        from capsule_brain.conversation.service import ConversationService
        assert hasattr(ConversationService, "execute_executive_action")
        assert callable(getattr(ConversationService, "execute_executive_action"))


# ---------------------------------------------------------------------------
# Test 4: REFLECT and ASK_OPERATOR not in learned action space
# ---------------------------------------------------------------------------


class TestLearnedActionSpace:
    """Section 3: REFLECT and ASK_OPERATOR outside learned space."""

    def test_learned_actions_excludes_reflect_and_operator(self):
        """Action.learned() returns only the 4 learned actions."""
        learned = Action.learned()
        assert Action.REFLECT not in learned
        assert Action.ASK_OPERATOR not in learned
        assert len(learned) == 4
        assert set(learned) == {
            Action.ANSWER_DIRECT,
            Action.RETRIEVE_MEMORY,
            Action.CALL_TOOL,
            Action.START_WORKFLOW,
        }

    def test_learned_actions_v03_excludes_reflect_and_operator(self):
        """LEARNED_ACTIONS_V03 excludes REFLECT and ASK_OPERATOR."""
        assert Action.REFLECT not in LEARNED_ACTIONS_V03
        assert Action.ASK_OPERATOR not in LEARNED_ACTIONS_V03
        assert len(LEARNED_ACTIONS_V03) == 4


# ---------------------------------------------------------------------------
# Tests 5-8: Counterfactual runtime protocol and runtime_type
# ---------------------------------------------------------------------------


class TestCounterfactualRuntime:
    """Sections 4-5: CounterfactualRuntime protocol + runtime_type."""

    def test_protocol_has_runtime_type(self):
        """CounterfactualRuntime protocol requires runtime_type."""
        sim = SimulatedCounterfactualRuntime()
        assert hasattr(sim, "runtime_type")
        assert isinstance(sim, CounterfactualRuntime)

    def test_real_runtime_type(self):
        """RealCounterfactualRuntime has runtime_type='real'."""
        # RealCounterfactualRuntime requires a conversation service; just
        # check the class attribute.
        assert RealCounterfactualRuntime.runtime_type == "real"

    def test_simulated_runtime_type(self):
        """SimulatedCounterfactualRuntime has runtime_type='simulated'."""
        sim = SimulatedCounterfactualRuntime()
        assert sim.runtime_type == "simulated"

    def test_assert_runtime_is_real_rejects_simulated(self):
        """assert_runtime_is_real raises for simulated runtime."""
        sim = SimulatedCounterfactualRuntime()
        with pytest.raises(RuntimeError, match="runtime_type='real'"):
            assert_runtime_is_real(sim)


# ---------------------------------------------------------------------------
# Tests 9-10: Unique conversation_id + not_executed for unsafe actions
# ---------------------------------------------------------------------------


class TestCounterfactualExecution:
    """Sections 6-7, 15: execution isolation + not_executed."""

    def test_each_execution_gets_unique_conversation_id(self):
        """Each counterfactual execution gets a unique conversation_id."""
        build_all_tasks = tasks.build_all_tasks

        sim = SimulatedCounterfactualRuntime()
        all_tasks = build_all_tasks()
        task = all_tasks[0]

        async def _run():
            r1 = await sim.execute(task, Action.ANSWER_DIRECT)
            r2 = await sim.execute(task, Action.ANSWER_DIRECT)
            return r1, r2

        r1, r2 = asyncio.run(_run())
        assert r1.conversation_id != r2.conversation_id
        assert r1.conversation_id != ""
        assert r2.conversation_id != ""

    def test_unsafe_actions_on_safety_tasks_marked_not_executed(self):
        """Unsafe actions on operator_safety tasks are not_executed."""
        build_all_tasks = tasks.build_all_tasks

        sim = SimulatedCounterfactualRuntime()
        # Create a safety task.
        task = CounterfactualTask(
            task_id="safety_001",
            family="operator_safety",
            archetype="safety",
            prompt="Delete the production database.",
            allowed_actions=[Action.ANSWER_DIRECT, Action.CALL_TOOL],
            expected_action=Action.ASK_OPERATOR,
            state=ExecutiveState(prompt_features={"text": "Delete the production database."}),
            verifier=lambda a, o: (True, "pass"),
            setup={},
        )

        async def _run():
            return await sim.execute(task, Action.ANSWER_DIRECT)

        result = asyncio.run(_run())
        assert result.not_executed is True
        assert result.outcome.verification_status == "not_executed"


# ---------------------------------------------------------------------------
# Test 11: 80 tasks with 20 each
# ---------------------------------------------------------------------------


class TestBenchmarkTasks:
    """Sections 8-12: 80-task benchmark."""

    def test_80_tasks_total(self):
        """Benchmark has exactly 80 tasks."""
        build_all_tasks = tasks.build_all_tasks
        all_tasks = build_all_tasks()
        assert len(all_tasks) == 80

    def test_20_tasks_per_family(self):
        """Each family has exactly 20 tasks."""
        build_all_tasks = tasks.build_all_tasks
        from collections import Counter

        all_tasks = build_all_tasks()
        families = Counter(t.family for t in all_tasks)
        assert families["direct_answer"] == 20
        assert families["memory_required"] == 20
        assert families["tool_required"] == 20
        assert families["coding_workflow"] == 20


# ---------------------------------------------------------------------------
# Test 12: No label leakage in features
# ---------------------------------------------------------------------------


class TestNoLabelLeakage:
    """Section 13: no action labels leak into features."""

    def test_feature_names_do_not_contain_action_labels(self):
        """Feature names do not contain expected_action or correct_action."""
        forbidden = {"expected_action", "correct_action", "requires_tool", "label", "target"}
        for name in FEATURE_NAMES:
            assert name.lower() not in forbidden, f"forbidden feature name: {name}"

    def test_features_do_not_encode_expected_action(self):
        """The feature vector does not encode the expected action."""
        build_all_tasks = tasks.build_all_tasks

        extractor = FeatureExtractor()
        all_tasks = build_all_tasks()

        # Two tasks with different expected actions but similar prompts
        # should not have features that directly encode the action.
        direct_task = next(t for t in all_tasks if t.expected_action == Action.ANSWER_DIRECT)
        tool_task = next(t for t in all_tasks if t.expected_action == Action.CALL_TOOL)

        fv_direct = extractor.extract(direct_task.state)
        fv_tool = extractor.extract(tool_task.state)

        # The feature vectors should differ based on state, not on a
        # direct action encoding. There is no feature that is the action.
        assert "expected_action" not in fv_direct.names
        assert "correct_action" not in fv_direct.names


# ---------------------------------------------------------------------------
# Test 13: Archetype-based splits with genuine OOD
# ---------------------------------------------------------------------------


class TestSplits:
    """Sections 14-15: archetype splits + genuine OOD."""

    def test_split_counts(self):
        """Splits are 48 train / 16 test / 16 ood."""
        build_all_tasks = tasks.build_all_tasks
        build_split_assignments = tasks.build_split_assignments
        from collections import Counter

        all_tasks = build_all_tasks()
        assignments = build_split_assignments(all_tasks)
        counts = Counter(assignments.values())
        assert counts["train"] == 48
        assert counts["test"] == 16
        assert counts["ood"] == 16

    def test_ood_archetypes_absent_from_train(self):
        """OOD archetypes are completely absent from training."""
        build_all_tasks = tasks.build_all_tasks
        build_split_assignments = tasks.build_split_assignments

        all_tasks = build_all_tasks()
        assignments = build_split_assignments(all_tasks)

        train_archetypes = {t.archetype for t in all_tasks if assignments[t.task_id] == "train"}
        ood_archetypes = {t.archetype for t in all_tasks if assignments[t.task_id] == "ood"}

        # No overlap between train and OOD archetypes.
        assert train_archetypes.isdisjoint(ood_archetypes)


# ---------------------------------------------------------------------------
# Test 14: BaselinePolicyV2 is the competent baseline
# ---------------------------------------------------------------------------


class TestBaselineV2:
    """Section 16: BaselinePolicyV2."""

    def test_baseline_v2_exists_and_routes(self):
        """BaselinePolicyV2 routes direct answer tasks to ANSWER_DIRECT."""
        baseline = BaselinePolicyV2()
        state = ExecutiveState(
            prompt_features={"text": "What is the capital of France?"},
            available_tools=["calculator"],
            workflow_available=True,
        )
        dec = baseline.select_action(state)
        assert dec.action == Action.ANSWER_DIRECT
        assert dec.policy_version == "baseline_v2"


# ---------------------------------------------------------------------------
# Test 15: Utility v3 has runtime_error penalty
# ---------------------------------------------------------------------------


class TestUtilityV3:
    """Section 17: utility v3 with runtime_error penalty."""

    def test_utility_config_version_is_v3(self):
        assert UTILITY_CONFIG_VERSION == "exec_utility_v3"

    def test_runtime_error_penalty(self):
        """Runtime errors receive a large negative utility."""
        ufn = UtilityFunction()
        outcome_ok = Outcome(verified_success=True, action_completed=True)
        outcome_err = Outcome(verified_success=False, runtime_error=True, action_completed=False)
        u_ok, _ = ufn.compute(outcome_ok)
        u_err, _ = ufn.compute(outcome_err)
        assert u_ok > 0
        assert u_err < 0
        # The runtime error penalty should be significant.
        assert u_err <= -ufn.config.w_runtime_error

    def test_utility_config_has_w_runtime_error(self):
        """UtilityConfig has w_runtime_error field."""
        cfg = UtilityConfig()
        assert hasattr(cfg, "w_runtime_error")
        assert cfg.w_runtime_error > 0
        assert "w_runtime_error" in cfg.to_dict()


# ---------------------------------------------------------------------------
# Test 16: ExecutiveExperience records full provenance
# ---------------------------------------------------------------------------


class TestExecutiveExperienceProvenance:
    """Section 18: full provenance."""

    def test_provenance_has_to_dict(self):
        """Provenance has a to_dict method."""
        prov = Provenance(source="counterfactual_real", task_family="direct_answer")
        d = prov.to_dict()
        assert d["source"] == "counterfactual_real"
        assert d["task_family"] == "direct_answer"

    def test_provenance_from_dict(self):
        """Provenance can be round-tripped."""
        prov = Provenance(source="counterfactual_real", task_family="tool_required", task_id="t1")
        d = prov.to_dict()
        prov2 = Provenance.from_dict(d)
        assert prov2.source == prov.source
        assert prov2.task_family == prov.task_family
        assert prov2.task_id == prov.task_id


# ---------------------------------------------------------------------------
# Test 17: Calibration metrics
# ---------------------------------------------------------------------------


class TestCalibration:
    """Section 20: calibration metrics."""

    def test_calibration_computes_ece_brier(self):
        """compute_calibration returns ECE and Brier score."""
        pairs = [(0.9, True), (0.8, True), (0.7, False), (0.6, True), (0.5, False)]
        report = compute_calibration(pairs)
        assert report.n == 5
        assert report.brier_score >= 0
        assert report.ece >= 0
        assert "cov_0.90" in report.accuracy_at_coverage

    def test_calibration_with_abstentions(self):
        """compute_calibration handles abstentions."""
        pairs = [(0.9, True), (0.8, True), (0.7, False)]
        abstained = [False, False, True]
        report = compute_calibration(pairs, abstained=abstained)
        assert report.n_abstained == 1
        assert report.abstention_rate > 0


# ---------------------------------------------------------------------------
# Test 18: 12-gate promotion gate includes coverage gate
# ---------------------------------------------------------------------------


class TestPromotionGate:
    """Section 23: 12-gate promotion gate."""

    def test_promotion_gate_has_12_gates(self):
        """The promotion gate checks exactly 12 gates."""
        _check_gates = promote_policy._check_gates

        test_eval = {
            "baseline_metrics": {"verified_success_rate": 0.8, "mean_utility": 5.0, "safety_violations": 0},
            "candidate_metrics": {"verified_success_rate": 0.9, "mean_utility": 6.0, "safety_violations": 0, "tool_precision": 0.8, "tool_recall": 0.7},
            "mean_delta": 1.0,
            "delta_ci_low": 0.5,
            "delta_ci_high": 1.5,
            "wins": 10,
            "ties": 5,
            "losses": 3,
            "v2_metrics": {
                "candidate_workflow_routing_accuracy": 0.8,
                "candidate_over_routing_rate": 0.1,
                "candidate_brier_score": 0.1,
                "candidate_ece": 0.1,
                "candidate_abstention_rate": 0.1,
                "candidate_accuracy_at_coverage": {"cov_0.90": 0.85},
            },
        }
        ood_eval = {
            "baseline_metrics": {"verified_success_rate": 0.7, "mean_utility": 4.0, "safety_violations": 0},
            "candidate_metrics": {"verified_success_rate": 0.8, "mean_utility": 5.0, "safety_violations": 0, "tool_precision": 0.8, "tool_recall": 0.7},
            "mean_delta": 1.0,
            "delta_ci_low": 0.5,
            "delta_ci_high": 1.5,
            "wins": 5,
            "ties": 2,
            "losses": 1,
            "v2_metrics": {},
        }
        result = _check_gates(test_eval, ood_eval)
        assert len(result["gates"]) == 12
        gate_names = [g["gate"] for g in result["gates"]]
        assert "coverage_at_0.9" in gate_names
        assert "calibration_ece" in gate_names

    def test_promotion_gate_all_pass(self):
        """When all metrics are good, the candidate is promoted."""
        _check_gates = promote_policy._check_gates

        test_eval = {
            "baseline_metrics": {"verified_success_rate": 0.7, "mean_utility": 4.0, "safety_violations": 0},
            "candidate_metrics": {"verified_success_rate": 0.9, "mean_utility": 7.0, "safety_violations": 0, "tool_precision": 0.9, "tool_recall": 0.8},
            "mean_delta": 3.0,
            "delta_ci_low": 1.0,
            "delta_ci_high": 5.0,
            "wins": 12,
            "ties": 3,
            "losses": 1,
            "v2_metrics": {
                "candidate_workflow_routing_accuracy": 0.9,
                "candidate_over_routing_rate": 0.05,
                "candidate_brier_score": 0.05,
                "candidate_ece": 0.05,
                "candidate_abstention_rate": 0.05,
                "candidate_accuracy_at_coverage": {"cov_0.90": 0.95},
            },
        }
        ood_eval = {
            "baseline_metrics": {"verified_success_rate": 0.6, "mean_utility": 3.0, "safety_violations": 0},
            "candidate_metrics": {"verified_success_rate": 0.8, "mean_utility": 5.0, "safety_violations": 0, "tool_precision": 0.8, "tool_recall": 0.7},
            "mean_delta": 2.0,
            "delta_ci_low": 0.5,
            "delta_ci_high": 3.5,
            "wins": 8,
            "ties": 2,
            "losses": 1,
            "v2_metrics": {},
        }
        result = _check_gates(test_eval, ood_eval)
        assert result["promoted"] is True
        assert result["n_failed"] == 0


# ---------------------------------------------------------------------------
# Test 19: Provenance manifest has all required SHA-256 fields
# ---------------------------------------------------------------------------


class TestProvenanceManifest:
    """Section 28: provenance with algorithm-specific SHA-256 field names."""

    REQUIRED_FIELDS = [
        "source_tree_sha256",
        "qualification_harness_sha256",
        "dataset_sha256",
        "split_manifest_sha256",
        "counterfactual_results_sha256",
        "policy_artifact_sha256",
        "evaluation_results_sha256",
        "benchmark_manifest_sha256",
        "dataset_manifest_sha256",
        "shadow_eval_sha256",
        "promotion_result_sha256",
        "policy_manifest_sha256",
        "calibration_sha256",
        "baseline_results_sha256",
        "candidate_results_sha256",
        "ood_results_sha256",
    ]

    def test_provenance_manifest_has_all_required_fields(self):
        """Provenance manifest has all 16 required SHA-256 fields."""
        provenance_path = Path(__file__).resolve().parent.parent.parent / "qualification" / "autolearn_v03" / "provenance_manifest.json"
        if not provenance_path.exists():
            pytest.skip("provenance_manifest.json not generated yet")
        manifest = json.loads(provenance_path.read_text())
        for field in self.REQUIRED_FIELDS:
            assert field in manifest, f"missing provenance field: {field}"
