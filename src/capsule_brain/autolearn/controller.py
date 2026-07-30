"""ExecutiveController: real runtime integration for AutoLearn v0.2.

This is the single integration point that closes the audit gap. The
controller sits on the real request path:

    User
      -> ConversationService
        -> ExecutiveController.execute(state)
          -> AutoLearnService.select_action(state)   # chooses
          -> action dispatcher                        # executes
          -> existing Capsule Brain services          # real services
          -> verifier outcome                         # judges
          -> ExecutiveExperience persisted
        <- ActionResult

AutoLearn chooses; Capsule Brain services execute; the verifier judges. The
controller never moves execution logic into AutoLearn and never lets the
learned policy touch the safety envelope.

The four production actions v0.2 dispatches through real services:

- ANSWER_DIRECT    -> LLMGateway.generate() with no tools / no memory.
- RETRIEVE_MEMORY  -> MemoryService retrieval, then LLMGateway.generate()
                      with retrieved evidence supplied as context.
- CALL_TOOL        -> LLMGateway.generate_with_tools() through the real
                      ToolRegistry; real tool handlers execute.
- START_WORKFLOW   -> WorkflowRunnerService.start_run() with an independent
                      verification/acceptance contract.

REFLECT and ASK_OPERATOR are handled by the baseline safety path and are
NOT routed through the learned policy until the first four route end to end.

Fail-safe contract: if the learned policy fails to load, has an
incompatible feature schema, produces an invalid action, or throws any
exception, the controller falls back to the baseline policy. It never fails
open. Every fallback is recorded with ``policy_fallback = true``.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from capsule_brain.autolearn.baseline import BaselinePolicy, BaselinePolicyV2
from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.policy import LearnedPolicy, PolicyLoadError
from capsule_brain.autolearn.schema import (
    Action,
    ActionMetadata,
    ExecutiveExperience,
    ExecutiveState,
    Outcome,
    Provenance,
)
from capsule_brain.autolearn.shadow import DistributionShiftGuard
from capsule_brain.autolearn.utility import UtilityConfig, UtilityFunction
from capsule_brain.safety.executive_guard import (
    ExecutiveSafetyGuard,
    SafetyDecision,
    SafetyDisposition,
)

log = logging.getLogger(__name__)

# The four actions v0.2 dispatches through real services.
LEARNED_ACTIONS: tuple[Action, ...] = (
    Action.ANSWER_DIRECT,
    Action.RETRIEVE_MEMORY,
    Action.CALL_TOOL,
    Action.START_WORKFLOW,
)


@dataclass(slots=True)
class ActionResult:
    """The real runtime outcome of executing a chosen action."""

    action: Action
    text: str = ""
    outcome: Outcome = field(default_factory=Outcome)
    experience: ExecutiveExperience | None = None
    decision: "ExecutiveDecision | None" = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutiveDecision:
    """The final action the runtime should execute, plus audit metadata."""

    action: Action
    reason: str
    policy_version: str
    policy_type: str  # "baseline" | "learned"
    confidence: float
    scores: dict[str, float]
    policy_fallback: bool = False
    policy_error: str = ""
    distribution_shift_detected: bool = False
    abstained: bool = False
    shadow_action: Action | None = None
    feature_vector: list[float] = field(default_factory=list)
    state: ExecutiveState | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "policy_type": self.policy_type,
            "confidence": self.confidence,
            "scores": dict(self.scores),
            "policy_fallback": self.policy_fallback,
            "policy_error": self.policy_error,
            "distribution_shift_detected": self.distribution_shift_detected,
            "abstained": self.abstained,
            "shadow_action": self.shadow_action.value if self.shadow_action else None,
            "feature_vector": list(self.feature_vector),
        }


# ---------------------------------------------------------------------------
# Action dispatcher interface
# ---------------------------------------------------------------------------
#
# The controller does not import ConversationService / MemoryService etc.
# directly. Instead it receives an ``ActionDispatcher`` that knows how to
# execute each of the four learned actions through the real Capsule Brain
# service stack. This keeps the controller testable and avoids a circular
# dependency between ConversationService and ExecutiveController.
#
# Each dispatcher method returns a DispatcherResult. The controller wraps
# that into an Outcome, runs the verifier, and persists the experience.


@dataclass(slots=True)
class DispatcherResult:
    """Raw result of dispatching one action through real services."""

    text: str = ""
    latency_ms: float = 0.0
    token_count: int = 0
    tool_failures: int = 0
    workflow_iterations: int = 0
    operator_intervention: bool = False
    safety_violation: bool = False
    runtime_error: bool = False
    action_completed: bool = False
    # Tool-specific.
    tool_name: str | None = None
    tool_calls_executed: int = 0
    tool_result_valid: bool = False
    final_answer_grounded: bool = False
    # Workflow-specific.
    workflow_name: str | None = None
    workflow_run_id: str | None = None
    acceptance_present: bool = False
    acceptance_passed: bool = False
    exit_code: int | None = None
    # Free-form metadata for the verifier / experience.
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionDispatcher:
    """Abstract interface for dispatching actions through real services.

    The concrete implementation lives in the runtime (ConversationService
    provides this). The controller calls these methods; they execute through
    the actual LLMGateway / MemoryService / ToolRegistry / WorkflowRunner.

    v0.3.2: Each method receives an explicit, immutable ``request_context``
    (a ``RequestContext``) so that no per-request data is read from a mutable
    service-level field. This makes concurrent request handling safe.
    """

    async def answer_direct(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: Any = None,
    ) -> DispatcherResult:
        raise NotImplementedError

    async def retrieve_memory(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: Any = None,
    ) -> DispatcherResult:
        raise NotImplementedError

    async def call_tool(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: Any = None,
    ) -> DispatcherResult:
        raise NotImplementedError

    async def start_workflow(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        request_context: Any = None,
    ) -> DispatcherResult:
        raise NotImplementedError


# A verifier callable: (action, dispatcher_result, state) -> (passed, status, outcome_overrides).
VerifierFn = Callable[
    [Action, DispatcherResult, ExecutiveState],
    Awaitable[tuple[bool, str, dict[str, Any]]],
]


class ExecutiveController:
    """Selects an action, dispatches it through real services, and records experience.

    The controller is the real request-path integration point. Unlike the
    v0.1 pure controller (which only chose an action), this one:
    1. Builds ExecutiveState from the runtime context (caller-supplied).
    2. Calls the active policy to select an action.
    3. Validates the action against the allowed set.
    4. Dispatches the action through the ActionDispatcher (real services).
    5. Runs the independent verifier to judge the outcome.
    6. Constructs and persists an ExecutiveExperience.
    7. Returns the final ActionResult.

    When no learned policy is active, or the policy abstains / errors /
    detects distribution shift, the controller falls back to the baseline
    policy. The fallback action is still dispatched through real services
    and still produces a real ExecutiveExperience — the only difference is
    the policy that chose it.
    """

    def __init__(
        self,
        baseline: BaselinePolicy | None = None,
        learned: LearnedPolicy | None = None,
        *,
        dispatcher: ActionDispatcher | None = None,
        verifier: VerifierFn | None = None,
        extractor: FeatureExtractor | None = None,
        confidence_threshold: float = 0.5,
        shift_guard: DistributionShiftGuard | None = None,
        utility_config: UtilityConfig | None = None,
        experience_sink: "ExperienceSink | None" = None,
        shadow_mode: bool = False,
        allowed_actions: list[Action] | None = None,
    ) -> None:
        self.baseline = baseline or BaselinePolicyV2()
        self.learned = learned
        self.dispatcher = dispatcher
        self.verifier = verifier
        self.extractor = extractor or FeatureExtractor()
        self.confidence_threshold = confidence_threshold
        self.shift_guard = shift_guard
        self.utility = UtilityFunction(utility_config or UtilityConfig())
        self.experience_sink = experience_sink
        self.shadow_mode = shadow_mode
        self._default_allowed = allowed_actions or list(LEARNED_ACTIONS)
        self._fallbacks = 0
        self._decisions = 0
        self._experiences_persisted = 0
        # v0.3.1: Immutable safety guard runs BEFORE the learned policy.
        # The learned policy can never override safety decisions.
        self.safety_guard = ExecutiveSafetyGuard()

    @property
    def has_learned_policy(self) -> bool:
        return self.learned is not None

    def set_learned_policy(self, policy: LearnedPolicy | None) -> None:
        self.learned = policy

    def set_dispatcher(self, dispatcher: ActionDispatcher) -> None:
        self.dispatcher = dispatcher

    def set_verifier(self, verifier: VerifierFn) -> None:
        self.verifier = verifier

    def set_experience_sink(self, sink: "ExperienceSink") -> None:
        self.experience_sink = sink

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def build_state(
        self,
        *,
        prompt: str,
        conversation_depth: int = 0,
        memory_hit_count: int = 0,
        top_memory_similarity: float = 0.0,
        available_tools: list[str] | None = None,
        workflow_available: bool = False,
        workflow_capability_match: bool = False,
        previous_attempt_failed: bool = False,
        verification_failure_type: str = "none",
        model_id: str | None = None,
        estimated_difficulty: float = 0.5,
    ) -> ExecutiveState:
        """Build an ExecutiveState from real runtime context.

        Features describe runtime state only — never the answer. No
        ``requires_tool`` / ``correct_action`` labels are permitted.
        """
        return ExecutiveState(
            prompt_features={
                "text": prompt,
                "previous_attempt_failed": previous_attempt_failed,
                "verification_failure_type": verification_failure_type,
                "estimated_difficulty": estimated_difficulty,
                "workflow_capability_match": workflow_capability_match,
            },
            conversation_features={"depth": conversation_depth},
            memory_features={
                "hit_count": memory_hit_count,
                "top_similarity": top_memory_similarity,
            },
            available_tools=list(available_tools or []),
            workflow_available=workflow_available,
            model_id=model_id,
            context_length=len(prompt),
        )

    # ------------------------------------------------------------------
    # Action selection (pure — no dispatch)
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: ExecutiveState,
        *,
        allowed_actions: list[Action] | None = None,
    ) -> ExecutiveDecision:
        """Select an action using the active policy, with fail-safe fallback.

        This is the pure selection path (no dispatch). ``execute`` calls this
        internally and then dispatches the chosen action.
        """
        self._decisions += 1
        allowed = allowed_actions or self._default_allowed
        fv = self.extractor.extract(state)
        fv_list = fv.as_list()

        # v0.3.1: Immutable safety guard runs BEFORE the learned policy.
        # The learned policy can never override ASK_OPERATOR or SAFE_REFUSAL.
        safety = self.safety_guard.evaluate(state)
        if safety.disposition == SafetyDisposition.REQUIRE_OPERATOR:
            return ExecutiveDecision(
                action=safety.required_action or Action.ASK_OPERATOR,
                reason=f"immutable safety: {safety.reason_code}",
                policy_version="immutable_safety",
                policy_type="immutable_safety",
                confidence=1.0,
                scores={},
                policy_fallback=False,
                policy_error="",
                feature_vector=fv_list,
                state=state,
            )
        if safety.disposition == SafetyDisposition.REFUSE:
            return ExecutiveDecision(
                action=safety.required_action or Action.SAFE_REFUSAL,
                reason=f"immutable safety: {safety.reason_code}",
                policy_version="immutable_safety",
                policy_type="immutable_safety",
                confidence=1.0,
                scores={},
                policy_fallback=False,
                policy_error="",
                feature_vector=fv_list,
                state=state,
            )

        # Distribution-shift guard.
        shift_detected = False
        if self.shift_guard is not None:
            self.shift_guard.observe(fv_list)
            try:
                shift_detected = self.shift_guard.shifted()
            except Exception as exc:
                log.warning("shift_guard error: %s", exc)
                shift_detected = False

        if shift_detected:
            return self._fallback_decision(
                state, fv_list, allowed,
                reason="distribution shift detected; using baseline",
                distribution_shift_detected=True,
            )

        if self.learned is None:
            return self._baseline_decision(
                state, fv_list, allowed, reason="no learned policy active"
            )

        try:
            c_dec = self.learned.select_action(
                state,
                extractor=self.extractor,
                allowed_actions=allowed,
            )
        except PolicyLoadError as exc:
            return self._fallback_decision(
                state, fv_list, allowed,
                reason=f"learned policy load error: {exc}",
                error=str(exc),
            )
        except Exception as exc:
            return self._fallback_decision(
                state, fv_list, allowed,
                reason=f"learned policy error: {exc}",
                error=str(exc),
            )

        if not _is_valid_action(c_dec.action, allowed):
            return self._fallback_decision(
                state, fv_list, allowed,
                reason=f"learned policy chose invalid/disallowed action {c_dec.action}",
                error="invalid_action",
            )

        if c_dec.abstained or c_dec.confidence < self.confidence_threshold:
            base_dec = self.baseline.select_action(state, allowed_actions=allowed)
            return ExecutiveDecision(
                action=base_dec.action,
                reason=f"learned policy abstained (conf={c_dec.confidence:.3f}); baseline fallback",
                policy_version=self.baseline.version,
                policy_type="baseline",
                confidence=max(base_dec.scores.values()) if base_dec.scores else 0.0,
                scores=base_dec.scores,
                policy_fallback=True,
                policy_error="abstention",
                abstained=True,
                shadow_action=c_dec.action if self.shadow_mode else None,
                feature_vector=fv_list,
                state=state,
            )

        return ExecutiveDecision(
            action=c_dec.action,
            reason=c_dec.reason,
            policy_version=self.learned.policy_version,
            policy_type="learned",
            confidence=c_dec.confidence,
            scores=c_dec.scores,
            policy_fallback=False,
            shadow_action=c_dec.action if self.shadow_mode else None,
            feature_vector=fv_list,
            state=state,
        )

    # ------------------------------------------------------------------
    # Full execute: select + dispatch + verify + persist
    # ------------------------------------------------------------------

    async def execute(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        allowed_actions: list[Action] | None = None,
        task_id: str = "",
        task_family: str = "production",
        shadow_execute_baseline: bool = False,
        request_context: Any = None,
    ) -> ActionResult:
        """Select an action, dispatch it through real services, verify, persist.

        This is the real request-path entry point. When ``shadow_mode`` is
        true and ``shadow_execute_baseline`` is true, the baseline action is
        executed in production and the candidate action is recorded but NOT
        executed (unless this is a controlled benchmark).

        v0.3.2: ``request_context`` is an immutable ``RequestContext`` threaded
        explicitly through the dispatch path so no mutable service-level
        request state is used.
        """
        if self.dispatcher is None:
            raise RuntimeError("ExecutiveController has no ActionDispatcher")

        allowed = allowed_actions or self._default_allowed
        decision = self.select_action(state, allowed_actions=allowed)

        # In shadow mode, execute the baseline action but record the
        # candidate's preference as shadow_action.
        execute_action = decision.action
        if self.shadow_mode and shadow_execute_baseline:
            base_dec = self.baseline.select_action(state, allowed_actions=allowed)
            execute_action = base_dec.action
            decision.shadow_action = decision.action if decision.policy_type == "learned" else None

        # Dispatch the chosen action through real services.
        started = time.perf_counter()
        try:
            disp = await self._dispatch(
                execute_action,
                state,
                conversation_id=conversation_id,
                request_context=request_context,
            )
        except Exception as exc:
            log.warning("action dispatch error for %s: %s", execute_action.value, exc)
            disp = DispatcherResult(
                runtime_error=True,
                action_completed=False,
                metadata={"dispatch_error": str(exc)},
            )
        # Measure latency from the dispatch itself if the dispatcher didn't.
        if disp.latency_ms <= 0:
            disp.latency_ms = (time.perf_counter() - started) * 1000.0

        # Run the independent verifier (if configured).
        verified_success = False
        verification_status = "skip"
        verifier_overrides: dict[str, Any] = {}
        if self.verifier is not None:
            try:
                verified_success, verification_status, verifier_overrides = await self.verifier(
                    execute_action, disp, state
                )
            except Exception as exc:
                log.warning("verifier error: %s", exc)
                verification_status = "error"
                verified_success = False
                verifier_overrides = {"verifier_error": str(exc)}
        else:
            # No verifier: cannot claim verified success. This is the
            # fail-closed path — unverified output never gets success_term.
            verification_status = "skip"
            verified_success = False

        # Build the outcome from the real dispatch + verifier judgment.
        outcome = _build_outcome(
            execute_action, disp, verified_success, verification_status, verifier_overrides
        )
        utility, components = self.utility.compute(outcome)

        # Construct and persist the ExecutiveExperience.
        exp = ExecutiveExperience(
            task_id=task_id,
            state=state,
            chosen_action=execute_action,
            action_metadata=ActionMetadata(
                tool_name=outcome.tool_name,
                workflow_name=outcome.workflow_name,
                model=state.model_id,
            ),
            outcome=outcome,
            utility=utility,
            utility_components=components,
            policy_version=decision.policy_version,
            provenance=Provenance(
                source="shadow" if self.shadow_mode else "production",
                policy_version=decision.policy_version,
                task_family=task_family,
                task_id=task_id or None,
                extra={
                    "policy_type": decision.policy_type,
                    "policy_fallback": decision.policy_fallback,
                    "policy_error": decision.policy_error,
                    "confidence": decision.confidence,
                    "abstained": decision.abstained,
                    "distribution_shift_detected": decision.distribution_shift_detected,
                    "shadow_action": decision.shadow_action.value if decision.shadow_action else None,
                    "disagreement": (
                        decision.shadow_action is not None
                        and decision.shadow_action != execute_action
                    ),
                },
            ),
        )
        if self.experience_sink is not None:
            try:
                await self.experience_sink.persist(exp)
                self._experiences_persisted += 1
            except Exception as exc:
                log.warning("experience persist error: %s", exc)

        return ActionResult(
            action=execute_action,
            text=disp.text,
            outcome=outcome,
            experience=exp,
            decision=decision,
            extra=dict(disp.metadata),
        )

    async def _dispatch(
        self,
        action: Action,
        state: ExecutiveState,
        *,
        conversation_id: str | None,
        request_context: Any = None,
    ) -> DispatcherResult:
        """Dispatch one action through the real ActionDispatcher."""
        if action == Action.ANSWER_DIRECT:
            return await self.dispatcher.answer_direct(
                state, conversation_id=conversation_id, request_context=request_context
            )
        if action == Action.RETRIEVE_MEMORY:
            return await self.dispatcher.retrieve_memory(
                state, conversation_id=conversation_id, request_context=request_context
            )
        if action == Action.CALL_TOOL:
            return await self.dispatcher.call_tool(
                state, conversation_id=conversation_id, request_context=request_context
            )
        if action == Action.START_WORKFLOW:
            return await self.dispatcher.start_workflow(
                state, conversation_id=conversation_id, request_context=request_context
            )
        if action == Action.REFLECT:
            # v0.2 does not dispatch REFLECT through the learned policy.
            # The baseline safety path handles it; here we return a no-op
            # so the experience records that the action was not executed
            # through real services.
            return DispatcherResult(
                action_completed=False,
                metadata={"reason": "REFLECT not dispatched in v0.2 learned path"},
            )
        if action == Action.ASK_OPERATOR:
            return DispatcherResult(
                action_completed=False,
                operator_intervention=True,
                metadata={"reason": "ASK_OPERATOR escalates; no execution"},
            )
        return DispatcherResult(
            runtime_error=True,
            metadata={"reason": f"unknown action {action}"},
        )

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------

    def _baseline_decision(
        self,
        state: ExecutiveState,
        fv_list: list[float],
        allowed: list[Action],
        *,
        reason: str,
    ) -> ExecutiveDecision:
        dec = self.baseline.select_action(state, allowed_actions=allowed)
        return ExecutiveDecision(
            action=dec.action,
            reason=reason,
            policy_version=self.baseline.version,
            policy_type="baseline",
            confidence=max(dec.scores.values()) if dec.scores else 0.0,
            scores=dec.scores,
            feature_vector=fv_list,
            state=state,
        )

    def _fallback_decision(
        self,
        state: ExecutiveState,
        fv_list: list[float],
        allowed: list[Action],
        *,
        reason: str,
        error: str = "",
        distribution_shift_detected: bool = False,
    ) -> ExecutiveDecision:
        self._fallbacks += 1
        dec = self.baseline.select_action(state, allowed_actions=allowed)
        return ExecutiveDecision(
            action=dec.action,
            reason=reason,
            policy_version=self.baseline.version,
            policy_type="baseline",
            confidence=max(dec.scores.values()) if dec.scores else 0.0,
            scores=dec.scores,
            policy_fallback=True,
            policy_error=error,
            distribution_shift_detected=distribution_shift_detected,
            feature_vector=fv_list,
            state=state,
        )


# ---------------------------------------------------------------------------
# Experience sink
# ---------------------------------------------------------------------------


class ExperienceSink:
    """Abstract sink for persisting ExecutiveExperience records."""

    async def persist(self, experience: ExecutiveExperience) -> None:
        raise NotImplementedError


class JSONLExperienceSink(ExperienceSink):
    """Append experiences to a JSONL file. Used by the qualification harness."""

    def __init__(self, path: str) -> None:
        self.path = path
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    async def persist(self, experience: ExecutiveExperience) -> None:
        import json
        with open(self.path, "a") as f:
            f.write(json.dumps(experience.to_dict(), sort_keys=True) + "\n")


class MemoryExperienceSink(ExperienceSink):
    """In-memory sink for tests. Collects experiences in a list."""

    def __init__(self) -> None:
        self.records: list[ExecutiveExperience] = []

    async def persist(self, experience: ExecutiveExperience) -> None:
        self.records.append(experience)


class ExperienceStoreSink(ExperienceSink):
    """Production sink that persists ExecutiveExperience via an
    ExperienceStore (converting to ExperienceRecord). Used by bootstrap.py
    to wire the runtime's experience_store into the ExecutiveController.
    """

    def __init__(self, experience_store: Any) -> None:
        self._store = experience_store

    async def persist(self, experience: ExecutiveExperience) -> None:
        from capsule_brain.learning.models import ExperienceRecord

        record = ExperienceRecord(
            id=experience.experience_id,
            conversation_id=experience.task_id,
            prompt_text=str(experience.state.prompt_features.get("text", "")),
            response_text=str(experience.outcome.verification_status),
            route=experience.chosen_action.value,
            model_alias=experience.action_metadata.model,
            latency_ms=experience.outcome.latency_ms,
            verifier=experience.outcome.__dict__ if hasattr(experience.outcome, "__dict__") else {
                "verified_success": experience.outcome.verified_success,
                "verification_status": experience.outcome.verification_status,
            },
            metadata={
                "task_id": experience.task_id,
                "chosen_action": experience.chosen_action.value,
                "utility": experience.utility,
                "policy_version": experience.policy_version,
                "provenance": experience.provenance.to_dict(),
            },
        )
        await self._store.create_experience(record)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_action(action: Action, allowed: list[Action] | None) -> bool:
    if not isinstance(action, Action):
        return False
    if allowed is None:
        return True
    return action in set(allowed)


def _build_outcome(
    action: Action,
    disp: DispatcherResult,
    verified_success: bool,
    verification_status: str,
    overrides: dict[str, Any],
) -> Outcome:
    """Build an Outcome from the real dispatch result + verifier judgment.

    The verifier is ground truth for ``verified_success``. The dispatcher
    supplies the runtime fields (latency, tokens, tool_calls, acceptance).
    Overrides from the verifier can set over-routing flags.
    """
    return Outcome(
        verified_success=verified_success,
        verification_status=verification_status,
        latency_ms=float(overrides.get("latency_ms", disp.latency_ms) or 0.0),
        token_count=int(overrides.get("token_count", disp.token_count) or 0),
        tool_failures=int(overrides.get("tool_failures", disp.tool_failures) or 0),
        workflow_iterations=int(overrides.get("workflow_iterations", disp.workflow_iterations) or 0),
        operator_intervention=bool(overrides.get("operator_intervention", disp.operator_intervention)),
        safety_violation=bool(overrides.get("safety_violation", disp.safety_violation)),
        runtime_error=bool(overrides.get("runtime_error", disp.runtime_error)),
        action_completed=bool(overrides.get("action_completed", disp.action_completed)),
        tool_name=overrides.get("tool_name", disp.tool_name),
        tool_calls_executed=int(overrides.get("tool_calls_executed", disp.tool_calls_executed) or 0),
        tool_result_valid=bool(overrides.get("tool_result_valid", disp.tool_result_valid)),
        final_answer_grounded=bool(overrides.get("final_answer_grounded", disp.final_answer_grounded)),
        workflow_name=overrides.get("workflow_name", disp.workflow_name),
        workflow_run_id=overrides.get("workflow_run_id", disp.workflow_run_id),
        acceptance_present=bool(overrides.get("acceptance_present", disp.acceptance_present)),
        acceptance_passed=bool(overrides.get("acceptance_passed", disp.acceptance_passed)),
        exit_code=overrides.get("exit_code", disp.exit_code),
        unnecessary_tool_use=bool(overrides.get("unnecessary_tool_use", False)),
        unnecessary_workflow_use=bool(overrides.get("unnecessary_workflow_use", False)),
    )
