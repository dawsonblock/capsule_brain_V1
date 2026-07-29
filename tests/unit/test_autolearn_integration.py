"""Integration tests for AutoLearn v0.1.

The 10 required integration scenarios from the spec:
1. direct-answer task routes direct
2. secret-nonce task routes tool
3. persisted-memory task routes retrieval
4. coding task routes workflow
5. failed artifact routes reflection
6. ambiguous task abstains/falls back
7. learned-policy failure uses baseline
8. candidate cannot bypass verification
9. rejected candidate never becomes active
10. promoted policy survives process restart
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from capsule_brain.autolearn.baseline import BaselinePolicy
from capsule_brain.autolearn.dataset import (
    CounterfactualRow,
    build_dataset,
    build_split_manifest,
)
from capsule_brain.autolearn.evaluator import evaluate_paired
from capsule_brain.autolearn.features import FEATURE_NAMES, FeatureExtractor
from capsule_brain.autolearn.learner import LearnerConfig, SupervisedRouterLearner
from capsule_brain.autolearn.policy import LearnedPolicy, PolicyLoadError
from capsule_brain.autolearn.promotion import GateConfig, evaluate_promotion_gate
from capsule_brain.autolearn.registry import PolicyRegistry
from capsule_brain.autolearn.schema import Action, ExecutiveState, Outcome
from capsule_brain.autolearn.service import AutoLearnService, ExecutiveController
from capsule_brain.autolearn.utility import UtilityConfig, UtilityFunction
from capsule_brain.autolearn.shadow import ShadowEvaluator

from capsule_brain.autolearn.schema import (
    ActionMetadata,
    ExecutiveExperience,
    Provenance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(
    prompt: str,
    *,
    available_tools: list[str] | None = None,
    workflow_available: bool = False,
    memory_hit_count: int = 0,
    top_similarity: float = 0.0,
    previous_attempt_failed: bool = False,
    verification_failure_type: str = "none",
    workflow_capability_match: bool = False,
    model_id: str = "qwen2.5-coder:3b",
) -> ExecutiveState:
    return ExecutiveState(
        prompt_features={
            "text": prompt,
            "previous_attempt_failed": previous_attempt_failed,
            "verification_failure_type": verification_failure_type,
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


def _train_simple_policy(
    train_examples: list,
    val_examples: list,
    policy_id: str = "test_policy",
) -> LearnedPolicy:
    """Train a small policy from given examples."""
    cfg = LearnerConfig(
        actions=tuple(Action.all()),
        n_epochs=300,
        learning_rate=0.05,
        l2=0.1,
        confidence_threshold=0.4,
    )
    learner = SupervisedRouterLearner(cfg)
    policy, _ = learner.train(
        train_examples, val_examples,
        policy_id=policy_id,
    )
    return policy


# ---------------------------------------------------------------------------
# 1. Direct-answer task routes direct
# ---------------------------------------------------------------------------


class TestDirectAnswerRoutes:
    def test_baseline_routes_direct_for_simple_factual(self):
        baseline = BaselinePolicy()
        state = _state(
            "What is the capital of France?",
            available_tools=["secret_nonce_tool"],
            workflow_available=True,
        )
        dec = baseline.select_action(state)
        assert dec.action == Action.ANSWER_DIRECT

    def test_learned_routes_direct_for_simple_factual(self):
        from capsule_brain.autolearn.learner import TrainingExample
        # Train a policy on examples where direct-answer tasks -> ANSWER_DIRECT.
        ext = FeatureExtractor()
        train = []
        for prompt in ["What is 2+2?", "What is the capital of France?", "What is H2O?"]:
            fv = ext.extract(_state(prompt, available_tools=["t"], workflow_available=True))
            train.append(TrainingExample(
                features=fv.as_list(), best_action=Action.ANSWER_DIRECT,
                utility_margin=5.0, weight=5.0, task_id=prompt[:10], split="train",
            ))
        val = [train[0]]
        policy = _train_simple_policy(train, val)
        dec = policy.select_action(_state("What is the capital of Italy?", available_tools=["t"], workflow_available=True))
        assert dec.action == Action.ANSWER_DIRECT


# ---------------------------------------------------------------------------
# 2. Secret-nonce task routes tool
# ---------------------------------------------------------------------------


class TestSecretNonceRoutesTool:
    def test_baseline_routes_tool_for_nonce_request(self):
        baseline = BaselinePolicy()
        state = _state(
            "Fetch the current one-time access nonce for service slot 42 "
            "using the secret_nonce_tool. The nonce is live.",
            available_tools=["secret_nonce_tool"],
            workflow_available=True,
        )
        dec = baseline.select_action(state)
        assert dec.action == Action.CALL_TOOL

    def test_learned_routes_tool_for_nonce_request(self):
        from capsule_brain.autolearn.learner import TrainingExample
        ext = FeatureExtractor()
        train = []
        for i in range(10):
            fv = ext.extract(_state(
                f"Fetch the current nonce for slot {i} using the tool.",
                available_tools=["secret_nonce_tool"], workflow_available=True,
            ))
            train.append(TrainingExample(
                features=fv.as_list(), best_action=Action.CALL_TOOL,
                utility_margin=5.0, weight=5.0, task_id=f"tool_{i}", split="train",
            ))
        val = [train[0]]
        policy = _train_simple_policy(train, val)
        dec = policy.select_action(_state(
            "Fetch the live nonce for slot 99 using the secret_nonce_tool.",
            available_tools=["secret_nonce_tool"], workflow_available=True,
        ))
        assert dec.action == Action.CALL_TOOL


# ---------------------------------------------------------------------------
# 3. Persisted-memory task routes retrieval
# ---------------------------------------------------------------------------


class TestMemoryRoutesRetrieval:
    def test_baseline_routes_retrieval_for_memory_task(self):
        baseline = BaselinePolicy()
        state = _state(
            "Recall the configuration pin I stored earlier from memory.",
            available_tools=["secret_nonce_tool"],
            workflow_available=True,
            memory_hit_count=1,
            top_similarity=0.9,
        )
        dec = baseline.select_action(state)
        assert dec.action == Action.RETRIEVE_MEMORY

    def test_learned_routes_retrieval_for_memory_task(self):
        from capsule_brain.autolearn.learner import TrainingExample
        ext = FeatureExtractor()
        train = []
        for i in range(20):
            fv = ext.extract(_state(
                f"Recall the pin for project {i} from memory.",
                available_tools=["t"], workflow_available=True,
                memory_hit_count=1, top_similarity=0.9,
            ))
            train.append(TrainingExample(
                features=fv.as_list(), best_action=Action.RETRIEVE_MEMORY,
                utility_margin=5.0, weight=5.0, task_id=f"mem_{i}", split="train",
            ))
        val = [train[0]]
        cfg = LearnerConfig(
            actions=tuple(Action.all()), n_epochs=600,
            learning_rate=0.05, l2=0.1, confidence_threshold=0.4,
        )
        learner = SupervisedRouterLearner(cfg)
        policy, _ = learner.train(train, val, policy_id="mem_test")
        dec = policy.select_action(_state(
            "Recall the stored pin for project 99 from memory.",
            available_tools=["t"], workflow_available=True,
            memory_hit_count=1, top_similarity=0.9,
        ))
        assert dec.action == Action.RETRIEVE_MEMORY


# ---------------------------------------------------------------------------
# 4. Coding task routes workflow
# ---------------------------------------------------------------------------


class TestCodingRoutesWorkflow:
    def test_baseline_routes_workflow_for_coding_task(self):
        baseline = BaselinePolicy()
        state = _state(
            "def add(a, b): return a + b. Write a Python function that passes "
            "the acceptance test. Use the coding workflow.",
            available_tools=["secret_nonce_tool"],
            workflow_available=True,
            workflow_capability_match=True,
        )
        dec = baseline.select_action(state)
        assert dec.action == Action.START_WORKFLOW

    def test_learned_routes_workflow_for_coding_task(self):
        from capsule_brain.autolearn.learner import TrainingExample
        ext = FeatureExtractor()
        train = []
        for i in range(10):
            fv = ext.extract(_state(
                f"def f{i}(x): return x. Write a Python function. Use the workflow.",
                available_tools=["t"], workflow_available=True,
                workflow_capability_match=True,
            ))
            train.append(TrainingExample(
                features=fv.as_list(), best_action=Action.START_WORKFLOW,
                utility_margin=5.0, weight=5.0, task_id=f"wf_{i}", split="train",
            ))
        val = [train[0]]
        policy = _train_simple_policy(train, val)
        dec = policy.select_action(_state(
            "Write a Python function reverse_string(s). Use the coding workflow.",
            available_tools=["t"], workflow_available=True,
            workflow_capability_match=True,
        ))
        assert dec.action == Action.START_WORKFLOW


# ---------------------------------------------------------------------------
# 5. Failed artifact routes reflection
# ---------------------------------------------------------------------------


class TestFailedArtifactRoutesReflection:
    def test_baseline_routes_reflect_for_failed_artifact(self):
        baseline = BaselinePolicy()
        state = _state(
            "The previous implementation of add failed its acceptance test. "
            "Reflect on the failure and produce a corrected implementation.",
            available_tools=["secret_nonce_tool"],
            workflow_available=True,
            previous_attempt_failed=True,
            verification_failure_type="acceptance",
            workflow_capability_match=True,
        )
        dec = baseline.select_action(state)
        assert dec.action == Action.REFLECT


# ---------------------------------------------------------------------------
# 6. Ambiguous task abstains/falls back
# ---------------------------------------------------------------------------


class TestAmbiguousTaskAbstains:
    def test_uniform_policy_abstains_on_ambiguous_task(self):
        # A policy with all-zero weights produces uniform probabilities.
        # With a high confidence threshold, it should abstain.
        policy = LearnedPolicy(policy_id="test", confidence_threshold=0.9)
        dec = policy.select_action(_state("hello"))
        assert dec.abstained is True

    def test_controller_falls_back_on_abstention(self):
        policy = LearnedPolicy(policy_id="test", confidence_threshold=0.99)
        controller = ExecutiveController(
            baseline=BaselinePolicy(), learned=policy, confidence_threshold=0.99,
        )
        dec = controller.select_action(_state("hello"))
        assert dec.policy_fallback is True
        assert dec.abstained is True
        assert dec.policy_type == "baseline"


# ---------------------------------------------------------------------------
# 7. Learned-policy failure uses baseline
# ---------------------------------------------------------------------------


class TestLearnedPolicyFailureUsesBaseline:
    def test_broken_policy_falls_back_to_baseline(self):
        class BrokenPolicy(LearnedPolicy):
            def select_action(self, *a, **kw):
                raise RuntimeError("model crashed")

        broken = BrokenPolicy(policy_id="broken")
        controller = ExecutiveController(baseline=BaselinePolicy(), learned=broken)
        dec = controller.select_action(_state("What is 2+2?"))
        assert dec.policy_type == "baseline"
        assert dec.policy_fallback is True
        assert "crashed" in dec.policy_error

    def test_service_falls_back_when_policy_load_fails(self, tmp_path):
        # Write a corrupt active policy pointer.
        root = tmp_path / "policies"
        root.mkdir(parents=True, exist_ok=True)
        (root / "active.json").write_text("not valid json")
        svc = AutoLearnService(cfg={"policy_root": str(root)})
        asyncio.run(svc.start())
        assert svc.controller.has_learned_policy is False
        dec = svc.select_action(_state("hello"))
        assert dec.policy_type == "baseline"
        asyncio.run(svc.stop())


# ---------------------------------------------------------------------------
# 8. Candidate cannot bypass verification
# ---------------------------------------------------------------------------


class TestCannotBypassVerification:
    def test_policy_cannot_mark_experience_successful_without_verifier(self):
        # The ExecutiveExperience schema requires an Outcome with
        # verification_status. The utility function only awards success_term
        # when verified_success is True, which the verifier determines —
        # not the policy. The policy only chooses an action; it does not
        # judge the outcome.
        ufn = UtilityFunction(UtilityConfig())
        # A policy-chosen action with no verifier evidence:
        unverified = Outcome(verified_success=False, verification_status="skip")
        u_unverified, c = ufn.compute(unverified)
        assert c["success_term"] == 0.0
        assert u_unverified <= 0  # no success reward; zero or negative from costs
        # The same action WITH verifier evidence:
        verified = Outcome(verified_success=True, verification_status="pass")
        u_verified, _ = ufn.compute(verified)
        assert u_verified > u_unverified

    def test_policy_does_not_execute_actions(self):
        # The ExecutiveController.select_action returns a decision; it does
        # not execute the action. The caller is responsible for dispatch.
        controller = ExecutiveController(baseline=BaselinePolicy(), learned=None)
        dec = controller.select_action(_state("hello"))
        assert hasattr(dec, "action")
        assert hasattr(dec, "to_dict")
        # No side effects — the controller is pure.
        assert dec.state is not None


# ---------------------------------------------------------------------------
# 9. Rejected candidate never becomes active
# ---------------------------------------------------------------------------


class TestRejectedCandidateNeverActive:
    def test_rejected_candidate_cannot_be_activated(self, tmp_path):
        registry = PolicyRegistry(tmp_path / "policies")
        policy = LearnedPolicy(policy_id="cand1")
        registry.register(policy, training_data_digest="a", split_manifest_digest="b")
        registry.reject("cand1", reason="failed gate")
        with pytest.raises(RuntimeError, match="cannot activate rejected"):
            registry.set_active("cand1", reason="attempt")

    def test_rejected_candidate_not_loaded_as_active(self, tmp_path):
        registry = PolicyRegistry(tmp_path / "policies")
        policy = LearnedPolicy(policy_id="cand1")
        registry.register(policy, training_data_digest="a", split_manifest_digest="b")
        registry.reject("cand1", reason="failed gate")
        # No active pointer was set.
        assert registry.get_active_id() is None
        assert registry.load_active() is None


# ---------------------------------------------------------------------------
# 10. Promoted policy survives process restart
# ---------------------------------------------------------------------------


class TestPromotedPolicySurvivesRestart:
    def test_active_policy_loaded_after_restart(self, tmp_path):
        # First "process": register and promote a policy.
        registry = PolicyRegistry(tmp_path / "policies")
        policy = LearnedPolicy(policy_id="p1", policy_version="v1")
        registry.register(policy, training_data_digest="a", split_manifest_digest="b")
        registry.set_active("p1", reason="passed all gates")

        # Second "process": a new registry instance reads from the same root.
        registry2 = PolicyRegistry(tmp_path / "policies")
        assert registry2.get_active_id() == "p1"
        active = registry2.load_active()
        assert active is not None
        assert active[0].policy_id == "p1"
        assert active[1].promotion_status == "active"

    def test_autolearn_service_loads_promoted_policy_on_start(self, tmp_path):
        # First process: register and promote.
        registry = PolicyRegistry(tmp_path / "policies")
        policy = LearnedPolicy(policy_id="p1", policy_version="v1")
        registry.register(policy, training_data_digest="a", split_manifest_digest="b")
        registry.set_active("p1", reason="passed")

        # Second process: AutoLearnService starts and loads the active policy.
        svc = AutoLearnService(cfg={"policy_root": str(tmp_path / "policies")})
        asyncio.run(svc.start())
        assert svc.active_policy_id == "p1"
        assert svc.controller.has_learned_policy is True
        asyncio.run(svc.stop())

    def test_promoted_policy_artifact_is_immutable(self, tmp_path):
        registry = PolicyRegistry(tmp_path / "policies")
        policy = LearnedPolicy(policy_id="p1", policy_version="v1")
        registry.register(policy, training_data_digest="a", split_manifest_digest="b")
        registry.set_active("p1", reason="passed")
        # The artifact file should not change after promotion (only the
        # manifest and active pointer change).
        artifact_path = registry.artifact_path("p1")
        artifact_before = artifact_path.read_text()
        # Re-promote (no-op since already active, but rewrites manifest).
        registry.set_active("p1", reason="re-promoted")
        artifact_after = artifact_path.read_text()
        assert artifact_before == artifact_after


# ---------------------------------------------------------------------------
# End-to-end: train, evaluate, promote
# ---------------------------------------------------------------------------


class TestEndToEndPromotion:
    def test_full_train_evaluate_promote_cycle(self, tmp_path):
        """End-to-end: build counterfactuals, train, evaluate, promote."""
        from capsule_brain.autolearn.learner import TrainingExample

        # Build a small counterfactual dataset.
        ext = FeatureExtractor()
        ufn = UtilityFunction(UtilityConfig())
        rows: list[CounterfactualRow] = []
        for i in range(30):
            task_id = f"t{i:03d}"
            family = "direct_answer" if i < 15 else "tool_required"
            prompt = f"What is {i}?" if i < 15 else f"Fetch nonce {i} using the tool."
            state = _state(prompt, available_tools=["tool"], workflow_available=True)
            for action in [Action.ANSWER_DIRECT, Action.CALL_TOOL]:
                if family == "direct_answer":
                    outcome = Outcome(
                        verified_success=(action == Action.ANSWER_DIRECT),
                        verification_status="pass" if action == Action.ANSWER_DIRECT else "fail",
                        latency_ms=200 if action == Action.ANSWER_DIRECT else 400,
                    )
                else:
                    outcome = Outcome(
                        verified_success=(action == Action.CALL_TOOL),
                        verification_status="pass" if action == Action.CALL_TOOL else "fail",
                        latency_ms=400 if action == Action.CALL_TOOL else 200,
                    )
                u, _ = ufn.compute(outcome)
                exp = ExecutiveExperience(
                    task_id=task_id, state=state, chosen_action=action, outcome=outcome, utility=u,
                )
                rows.append(CounterfactualRow(task_id=task_id, task_family=family, action=action, experience=exp, utility=u))

        split = build_split_manifest(rows, train_families=["direct_answer"], test_families=["tool_required"])
        built = build_dataset(rows, split, learner_config=LearnerConfig(actions=tuple(Action.all()), n_epochs=300, learning_rate=0.05, l2=0.1))
        splits = {s: [] for s in ("train", "validation", "test", "ood")}
        for e in built.examples:
            splits.setdefault(e.split, []).append(e)

        train = splits["train"]
        test = splits["test"]
        assert len(train) > 0
        assert len(test) > 0

        # Train.
        cfg = LearnerConfig(actions=tuple(Action.all()), n_epochs=300, learning_rate=0.05, l2=0.1, confidence_threshold=0.4)
        learner = SupervisedRouterLearner(cfg)
        policy, metrics = learner.train(train, train[:2], policy_id="e2e_cand")

        # Register.
        registry = PolicyRegistry(tmp_path / "policies")
        manifest = registry.register(
            policy,
            training_data_digest="e2e",
            split_manifest_digest=split.digest,
            metrics=metrics.to_dict(),
        )

        # Evaluate on test.
        test_ids = [e.task_id for e in test]
        states = {}
        for r in rows:
            if r.task_id in test_ids:
                states[r.task_id] = r.experience.state
        families = {e.task_id: e.task_family for e in test}
        outcomes = {}
        for r in rows:
            if r.task_id in test_ids:
                outcomes[(r.task_id, r.action.value)] = {
                    "verified_success": r.experience.outcome.verified_success,
                    "verification_status": r.experience.outcome.verification_status,
                    "latency_ms": r.experience.outcome.latency_ms,
                    "token_count": r.experience.outcome.token_count,
                    "tool_failures": r.experience.outcome.tool_failures,
                    "workflow_iterations": r.experience.outcome.workflow_iterations,
                    "operator_intervention": r.experience.outcome.operator_intervention,
                    "safety_violation": r.experience.outcome.safety_violation,
                }

        eval_result = evaluate_paired(
            task_ids=test_ids, task_families=families, states=states,
            outcomes_by_action=outcomes,
            baseline=BaselinePolicy(), candidate=policy,
            utility_config=UtilityConfig(),
            bootstrap_iterations=500, bootstrap_seed=0,
        )

        # Promote.
        gate_result = evaluate_promotion_gate(eval_result, config=GateConfig(
            tool_precision_min=0.0, tool_recall_min=0.0, ood_score_floor=-100.0,
        ))
        if gate_result.passed:
            registry.set_active(policy.policy_id, reason=gate_result.reason)
            assert registry.get_active_id() == policy.policy_id
        else:
            registry.reject(policy.policy_id, reason=gate_result.reason)
            assert registry.get_active_id() is None
