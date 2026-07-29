"""Canonical ExecutiveExperience schema for AutoLearn v0.2.

Each record captures a single (state, chosen_action, verified_outcome)
triple plus the scalar utility derived from the outcome. The full outcome
vector is always preserved — learning never trains from raw success labels
alone.

v0.2 adds the runtime/verification fields required to distinguish *real*
execution from simulated evidence:
- ``runtime_error`` / ``action_completed`` flag whether the action actually
  ran through the Capsule Brain service stack (not a simulator).
- Tool-specific fields (``tool_name``, ``tool_calls_executed``,
  ``tool_result_valid``, ``final_answer_grounded``) record the real tool
  dispatch path.
- Workflow-specific fields (``workflow_name``, ``workflow_run_id``,
  ``acceptance_present``, ``acceptance_passed``, ``exit_code``) record the
  real workflow acceptance path.

``verified_success`` is never inferred from LLM prose; it is set only by the
independent verifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

# Schema version for the ExecutiveExperience record itself. Bump when the
# on-disk shape of an ExecutiveExperience changes in a backward-incompatible
# way. The feature schema has its own independent version (see features.py).
SCHEMA_VERSION = "exec_experience_v2"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Action(str, Enum):
    """The supported executive actions for AutoLearn v0.2.

    Deliberately small. New actions must not be added until these route
    reliably on the counterfactual benchmark. v0.2 only integrates the first
    four into the learned policy; REFLECT and ASK_OPERATOR remain available
    for the baseline safety path but are not learned until the first four
    route end to end.
    """

    ANSWER_DIRECT = "ANSWER_DIRECT"
    RETRIEVE_MEMORY = "RETRIEVE_MEMORY"
    CALL_TOOL = "CALL_TOOL"
    START_WORKFLOW = "START_WORKFLOW"
    REFLECT = "REFLECT"
    ASK_OPERATOR = "ASK_OPERATOR"

    @classmethod
    def all(cls) -> list["Action"]:
        return [a for a in Action]

    @classmethod
    def learned(cls) -> list["Action"]:
        """The four actions v0.2 trains and dispatches through real services."""
        return [
            Action.ANSWER_DIRECT,
            Action.RETRIEVE_MEMORY,
            Action.CALL_TOOL,
            Action.START_WORKFLOW,
        ]


@dataclass(slots=True)
class ExecutiveState:
    """Deterministic, interpretable state observed before action selection."""

    prompt_features: dict[str, Any] = field(default_factory=dict)
    conversation_features: dict[str, Any] = field(default_factory=dict)
    memory_features: dict[str, Any] = field(default_factory=dict)
    available_tools: list[str] = field(default_factory=list)
    workflow_available: bool = False
    model_id: str | None = None
    context_length: int = 0


@dataclass(slots=True)
class ActionMetadata:
    """What was actually done for the chosen action."""

    tool_name: str | None = None
    workflow_name: str | None = None
    model: str | None = None


@dataclass(slots=True)
class Outcome:
    """Full verified outcome vector. Never reduce this to a single label.

    v0.2 fields:
    - ``runtime_error``: the action dispatch raised before completing.
    - ``action_completed``: the action ran through real services to completion.
    - tool-specific: ``tool_name``, ``tool_calls_executed``,
      ``tool_result_valid``, ``final_answer_grounded``.
    - workflow-specific: ``workflow_name``, ``workflow_run_id``,
      ``acceptance_present``, ``acceptance_passed``, ``exit_code``.

    ``verified_success`` is set ONLY by the independent verifier, never
    inferred from LLM prose.
    """

    verified_success: bool = False
    verification_status: str = "skip"  # pass | fail | error | skip
    latency_ms: float = 0.0
    token_count: int = 0
    tool_failures: int = 0
    workflow_iterations: int = 0
    operator_intervention: bool = False
    safety_violation: bool = False
    # v0.2 runtime/verification fields.
    runtime_error: bool = False
    action_completed: bool = False
    # Tool-specific (populated for CALL_TOOL).
    tool_name: str | None = None
    tool_calls_executed: int = 0
    tool_result_valid: bool = False
    final_answer_grounded: bool = False
    # Workflow-specific (populated for START_WORKFLOW).
    workflow_name: str | None = None
    workflow_run_id: str | None = None
    acceptance_present: bool = False
    acceptance_passed: bool = False
    exit_code: int | None = None
    # Over-routing flags (set by the verifier/harness). A tool/workflow was
    # "unnecessary" when the action succeeded but the task archetype did not
    # require that action. The utility function applies a penalty.
    unnecessary_tool_use: bool = False
    unnecessary_workflow_use: bool = False


@dataclass(slots=True)
class Provenance:
    """Where this experience came from."""

    source: str = "production"  # production | counterfactual | shadow | baseline
    policy_version: str = "baseline_v1"
    task_family: str = "unknown"
    task_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutiveExperience:
    """A single (state, action, verified outcome, utility) learning record."""

    experience_id: str = field(default_factory=lambda: uuid4().hex)
    task_id: str = ""
    timestamp: str = field(default_factory=_utc_now_iso)

    state: ExecutiveState = field(default_factory=ExecutiveState)
    chosen_action: Action = Action.ANSWER_DIRECT
    action_metadata: ActionMetadata = field(default_factory=ActionMetadata)
    outcome: Outcome = field(default_factory=Outcome)

    utility: float = 0.0
    utility_components: dict[str, float] = field(default_factory=dict)

    policy_version: str = "baseline_v1"
    schema_version: str = SCHEMA_VERSION
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "state": {
                "prompt_features": dict(self.state.prompt_features),
                "conversation_features": dict(self.state.conversation_features),
                "memory_features": dict(self.state.memory_features),
                "available_tools": list(self.state.available_tools),
                "workflow_available": self.state.workflow_available,
                "model_id": self.state.model_id,
                "context_length": self.state.context_length,
            },
            "chosen_action": self.chosen_action.value,
            "action_metadata": {
                "tool_name": self.action_metadata.tool_name,
                "workflow_name": self.action_metadata.workflow_name,
                "model": self.action_metadata.model,
            },
            "outcome": {
                "verified_success": self.outcome.verified_success,
                "verification_status": self.outcome.verification_status,
                "latency_ms": self.outcome.latency_ms,
                "token_count": self.outcome.token_count,
                "tool_failures": self.outcome.tool_failures,
                "workflow_iterations": self.outcome.workflow_iterations,
                "operator_intervention": self.outcome.operator_intervention,
                "safety_violation": self.outcome.safety_violation,
                "runtime_error": self.outcome.runtime_error,
                "action_completed": self.outcome.action_completed,
                "tool_name": self.outcome.tool_name,
                "tool_calls_executed": self.outcome.tool_calls_executed,
                "tool_result_valid": self.outcome.tool_result_valid,
                "final_answer_grounded": self.outcome.final_answer_grounded,
                "workflow_name": self.outcome.workflow_name,
                "workflow_run_id": self.outcome.workflow_run_id,
                "acceptance_present": self.outcome.acceptance_present,
                "acceptance_passed": self.outcome.acceptance_passed,
                "exit_code": self.outcome.exit_code,
                "unnecessary_tool_use": self.outcome.unnecessary_tool_use,
                "unnecessary_workflow_use": self.outcome.unnecessary_workflow_use,
            },
            "utility": self.utility,
            "utility_components": dict(self.utility_components),
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
            "provenance": {
                "source": self.provenance.source,
                "policy_version": self.provenance.policy_version,
                "task_family": self.provenance.task_family,
                "task_id": self.provenance.task_id,
                "extra": dict(self.provenance.extra),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutiveExperience":
        st = data.get("state", {}) or {}
        state = ExecutiveState(
            prompt_features=dict(st.get("prompt_features", {}) or {}),
            conversation_features=dict(st.get("conversation_features", {}) or {}),
            memory_features=dict(st.get("memory_features", {}) or {}),
            available_tools=list(st.get("available_tools", []) or []),
            workflow_available=bool(st.get("workflow_available", False)),
            model_id=st.get("model_id"),
            context_length=int(st.get("context_length", 0) or 0),
        )
        am = data.get("action_metadata", {}) or {}
        meta = ActionMetadata(
            tool_name=am.get("tool_name"),
            workflow_name=am.get("workflow_name"),
            model=am.get("model"),
        )
        oc = data.get("outcome", {}) or {}
        outcome = Outcome(
            verified_success=bool(oc.get("verified_success", False)),
            verification_status=str(oc.get("verification_status", "skip")),
            latency_ms=float(oc.get("latency_ms", 0.0) or 0.0),
            token_count=int(oc.get("token_count", 0) or 0),
            tool_failures=int(oc.get("tool_failures", 0) or 0),
            workflow_iterations=int(oc.get("workflow_iterations", 0) or 0),
            operator_intervention=bool(oc.get("operator_intervention", False)),
            safety_violation=bool(oc.get("safety_violation", False)),
            runtime_error=bool(oc.get("runtime_error", False)),
            action_completed=bool(oc.get("action_completed", False)),
            tool_name=oc.get("tool_name"),
            tool_calls_executed=int(oc.get("tool_calls_executed", 0) or 0),
            tool_result_valid=bool(oc.get("tool_result_valid", False)),
            final_answer_grounded=bool(oc.get("final_answer_grounded", False)),
            workflow_name=oc.get("workflow_name"),
            workflow_run_id=oc.get("workflow_run_id"),
            acceptance_present=bool(oc.get("acceptance_present", False)),
            acceptance_passed=bool(oc.get("acceptance_passed", False)),
            exit_code=oc.get("exit_code"),
            unnecessary_tool_use=bool(oc.get("unnecessary_tool_use", False)),
            unnecessary_workflow_use=bool(oc.get("unnecessary_workflow_use", False)),
        )
        pv = data.get("provenance", {}) or {}
        prov = Provenance(
            source=str(pv.get("source", "production")),
            policy_version=str(pv.get("policy_version", "baseline_v1")),
            task_family=str(pv.get("task_family", "unknown")),
            task_id=pv.get("task_id"),
            extra=dict(pv.get("extra", {}) or {}),
        )
        return cls(
            experience_id=str(data.get("experience_id") or uuid4().hex),
            task_id=str(data.get("task_id", "")),
            timestamp=str(data.get("timestamp") or _utc_now_iso()),
            state=state,
            chosen_action=Action(str(data.get("chosen_action", "ANSWER_DIRECT"))),
            action_metadata=meta,
            outcome=outcome,
            utility=float(data.get("utility", 0.0) or 0.0),
            utility_components=dict(data.get("utility_components", {}) or {}),
            policy_version=str(data.get("policy_version", "baseline_v1")),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            provenance=prov,
        )
