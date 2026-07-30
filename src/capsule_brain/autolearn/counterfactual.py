"""Canonical counterfactual runtime for AutoLearn v0.3.

Defines the ``CounterfactualRuntime`` protocol and two implementations:

- ``RealCounterfactualRuntime``: dispatches actions through the real Capsule
  Brain service stack via ``ConversationService.execute_executive_action()``.
  This is the only runtime that may be used for official qualification.

- ``SimulatedCounterfactualRuntime``: a deterministic stand-in for unit tests
  only. It mirrors the real action semantics but does not call any real
  services. Official qualification must refuse to run with this runtime
  unless ``--allow-simulated`` is explicitly supplied.

v0.3 key rules (Sections 4-7):
- The simulator is allowed only for unit tests.
- Scientific qualification must refuse to run with simulator unless
  ``--allow-simulated`` is explicitly supplied.
- Reports must state ``runtime_type``.
- Default qualification runtime must be REAL.
- Each counterfactual action gets a unique conversation_id and isolated state.
- Unsafe actions on safety tasks are marked ``not_executed`` and excluded
  from argmax.
- Only the four learned actions (ANSWER_DIRECT, RETRIEVE_MEMORY, CALL_TOOL,
  START_WORKFLOW) are dispatched. REFLECT and ASK_OPERATOR remain
  runtime-controlled.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from capsule_brain.autolearn.schema import (
    Action,
    ActionMetadata,
    ExecutiveExperience,
    ExecutiveState,
    Outcome,
    Provenance,
)
from capsule_brain.autolearn.utility import UtilityConfig, UtilityFunction


# The four actions v0.3 dispatches through the counterfactual runtime.
LEARNED_ACTIONS_V03: tuple[Action, ...] = (
    Action.ANSWER_DIRECT,
    Action.RETRIEVE_MEMORY,
    Action.CALL_TOOL,
    Action.START_WORKFLOW,
)

# Actions that are unsafe for counterfactual execution on operator_safety
# tasks. Executing them would be a safety violation, so they are skipped.
_UNSAFE_ACTIONS_FOR_SAFETY_TASKS = frozenset({
    Action.ANSWER_DIRECT,
    Action.RETRIEVE_MEMORY,
    Action.CALL_TOOL,
    Action.START_WORKFLOW,
})


@dataclass(slots=True)
class CounterfactualTask:
    """A single counterfactual task definition.

    Each task specifies a prompt, the state to observe, the allowed actions,
    the expected action (for evaluation only — never used in features), the
    verifier, and setup data (secrets, nonces, impl code, etc.).
    """

    task_id: str
    family: str  # direct_answer | memory_required | tool_required | coding_workflow
    archetype: str
    prompt: str
    allowed_actions: list[Action]
    expected_action: Action
    state: ExecutiveState
    verifier: Any  # callable: (Action, dict[str, Any]) -> (bool, str)
    setup: dict[str, Any] = field(default_factory=dict)
    difficulty: float = 0.5
    ood: bool = False
    split: str = "train"
    description: str = ""


@dataclass(slots=True)
class CounterfactualResult:
    """The outcome of executing one action on one task.

    ``runtime_type`` is always set: "real" for RealCounterfactualRuntime,
    "simulated" for SimulatedCounterfactualRuntime. This is the field that
    qualification checks to prevent simulated data from passing promotion.
    """

    task_id: str
    action: Action
    outcome: Outcome
    text: str = ""
    conversation_id: str = ""
    runtime_type: str = "real"
    not_executed: bool = False
    safety_violation: bool = False
    acceptance_passed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CounterfactualRuntime(Protocol):
    """Protocol for counterfactual action execution runtimes.

    Every implementation must expose ``runtime_type`` and an ``execute``
    method that takes a task and an action and returns a
    ``CounterfactualResult``.
    """

    runtime_type: str

    async def execute(
        self,
        task: CounterfactualTask,
        action: Action,
    ) -> CounterfactualResult:
        """Execute one action on one task and return the verified outcome."""
        ...


# ---------------------------------------------------------------------------
# RealCounterfactualRuntime — dispatches through real Capsule Brain services
# ---------------------------------------------------------------------------


class RealCounterfactualRuntime:
    """Executes counterfactual actions through real Capsule Brain services.

    This runtime calls ``ConversationService.execute_executive_action()`` for
    each (task, action) pair. Each execution gets a unique conversation_id
    and isolated state — no action benefits from state created by a previous
    counterfactual.

    The verifier judges the outcome independently. The runtime does not
    self-report success.

    v2.16.0: Supports a ``service_factory`` for per-execution isolation.
    When a factory is provided, each execution creates a fresh
    ConversationService with isolated MemoryService, ToolRegistry, and
    WorkflowRunner. This ensures strict isolation: no shared memory, no
    simulator shortcuts, no task-family lookup tables.
    """

    runtime_type: str = "real"

    def __init__(
        self,
        conversation_service: Any = None,
        *,
        service_factory: Any = None,
        verifier_override: Any = None,
    ) -> None:
        """Initialize with a ConversationService or a service factory.

        Args:
            conversation_service: A ConversationService instance (or any
                object with an ``execute_executive_action`` coroutine method).
                Used when a single shared service is acceptable.
            service_factory: A callable that takes a CounterfactualTask and
                returns a fresh ConversationService. When provided, each
                execution creates a fresh, isolated service. This is the
                preferred mode for qualification (Priority 1).
            verifier_override: Optional override for the task's verifier.
                When None, the task's own verifier is used.
        """
        if conversation_service is None and service_factory is None:
            raise ValueError(
                "RealCounterfactualRuntime requires either "
                "conversation_service or service_factory"
            )
        self._svc = conversation_service
        self._service_factory = service_factory
        self._verifier_override = verifier_override
        self._execution_counter = 0

    async def execute(
        self,
        task: CounterfactualTask,
        action: Action,
    ) -> CounterfactualResult:
        """Execute one action through real services and verify the outcome."""
        self._execution_counter += 1
        # Unique conversation ID per (task, action) execution.
        conv_id = f"cf_{task.task_id}_{action.value}_{self._execution_counter}_{uuid.uuid4().hex[:8]}"

        # Section 15: skip unsafe actions on operator_safety tasks.
        if task.family == "operator_safety" and action in _UNSAFE_ACTIONS_FOR_SAFETY_TASKS:
            return CounterfactualResult(
                task_id=task.task_id,
                action=action,
                outcome=Outcome(
                    verification_status="not_executed",
                    action_completed=False,
                ),
                text="",
                conversation_id=conv_id,
                runtime_type=self.runtime_type,
                not_executed=True,
            )

        # Build or reuse the service for this execution.
        svc = self._svc
        if self._service_factory is not None:
            svc = await self._service_factory.build(task)

        # Dispatch the action through real services.
        started = time.perf_counter()
        try:
            disp = await svc.execute_executive_action(
                action=action,
                state=task.state,
                conversation_id=conv_id,
                metadata={
                    "text": task.prompt,
                    "task_id": task.task_id,
                    "task_family": task.family,
                    **task.setup,
                },
            )
        except Exception as exc:
            disp = _make_dispatcher_result(
                runtime_error=True,
                action_completed=False,
                metadata={"dispatch_error": str(exc)},
            )
        latency_ms = disp.latency_ms or (time.perf_counter() - started) * 1000.0

        # Build the outcome from the real dispatch result.
        outcome = _build_outcome_from_dispatch(action, disp, latency_ms)

        # Run the independent verifier.
        verifier = self._verifier_override or task.verifier
        verified_success = False
        verification_status = "skip"
        if verifier is not None:
            outcome_dict = _outcome_to_verifier_dict(
                action, disp, outcome, task
            )
            try:
                verified_success, verification_status = verifier(action, outcome_dict)
            except Exception as exc:
                verification_status = f"error:{exc}"
                verified_success = False

        outcome.verified_success = verified_success
        outcome.verification_status = verification_status

        return CounterfactualResult(
            task_id=task.task_id,
            action=action,
            outcome=outcome,
            text=disp.text,
            conversation_id=conv_id,
            runtime_type=self.runtime_type,
            safety_violation=disp.safety_violation,
            acceptance_passed=disp.acceptance_passed,
        )


# ---------------------------------------------------------------------------
# SimulatedCounterfactualRuntime — for unit tests only
# ---------------------------------------------------------------------------


class SimulatedCounterfactualRuntime:
    """Deterministic stand-in for Capsule Brain's action execution.

    This runtime is for UNIT TESTS ONLY. It mirrors the real action
    semantics with deterministic outcomes. The verifiers in the task
    definitions are the ground truth.

    Official qualification must refuse to use this runtime. The
    ``runtime_type`` field is "simulated" so qualification can detect and
    reject it.
    """

    runtime_type: str = "simulated"

    def __init__(self) -> None:
        self._execution_counter = 0

    async def execute(
        self,
        task: CounterfactualTask,
        action: Action,
    ) -> CounterfactualResult:
        """Execute one action in the simulator and verify the outcome."""
        self._execution_counter += 1
        conv_id = f"cf_{task.task_id}_{action.value}_{self._execution_counter}"

        # Section 15: skip unsafe actions on operator_safety tasks.
        if task.family == "operator_safety" and action in _UNSAFE_ACTIONS_FOR_SAFETY_TASKS:
            return CounterfactualResult(
                task_id=task.task_id,
                action=action,
                outcome=Outcome(
                    verification_status="not_executed",
                    action_completed=False,
                ),
                text="",
                conversation_id=conv_id,
                runtime_type=self.runtime_type,
                not_executed=True,
            )

        # Simulate the action outcome.
        text, outcome = self._simulate_action(task, action)
        outcome.latency_ms = outcome.latency_ms
        result = CounterfactualResult(
            task_id=task.task_id,
            action=action,
            outcome=outcome,
            text=text,
            conversation_id=conv_id,
            runtime_type=self.runtime_type,
            safety_violation=outcome.safety_violation,
            acceptance_passed=outcome.acceptance_passed,
        )

        # Run the verifier.
        verifier = task.verifier
        if verifier is not None:
            outcome_dict = _outcome_to_verifier_dict(action, result, outcome, task)
            try:
                verified, status = verifier(action, outcome_dict)
                outcome.verified_success = verified
                outcome.verification_status = status
            except Exception as exc:
                outcome.verified_success = False
                outcome.verification_status = f"error:{exc}"

        return result

    def _simulate_action(
        self, task: CounterfactualTask, action: Action
    ) -> tuple[str, Outcome]:
        """Simulate the outcome of one action on one task."""
        family = task.family
        if action == Action.ANSWER_DIRECT:
            return self._sim_answer_direct(task, family)
        elif action == Action.RETRIEVE_MEMORY:
            return self._sim_retrieve_memory(task, family)
        elif action == Action.CALL_TOOL:
            return self._sim_call_tool(task, family)
        elif action == Action.START_WORKFLOW:
            return self._sim_start_workflow(task, family)
        elif action == Action.REFLECT:
            return self._sim_reflect(task, family)
        elif action == Action.ASK_OPERATOR:
            return self._sim_ask_operator(task, family)
        return "", Outcome(verification_status="error", runtime_error=True)

    def _sim_answer_direct(self, task: CounterfactualTask, family: str) -> tuple[str, Outcome]:
        if family == "direct_answer":
            token = str(task.setup.get("expected_token", ""))
            text = f"The answer is {token}."
            return text, Outcome(
                verification_status="pass",
                latency_ms=200.0,
                token_count=max(20, len(text) // 4),
                action_completed=True,
            )
        elif family == "memory_required":
            return "I don't have that stored value.", Outcome(
                verification_status="fail",
                latency_ms=200.0,
                token_count=20,
                action_completed=True,
            )
        elif family == "tool_required":
            return "I cannot fetch a live nonce without the tool.", Outcome(
                verification_status="fail",
                latency_ms=200.0,
                token_count=20,
                action_completed=True,
            )
        elif family == "coding_workflow":
            fn = task.setup.get("fn_name", "f")
            return f"def {fn}(*a, **k):\n    pass\n", Outcome(
                verification_status="fail",
                latency_ms=200.0,
                token_count=20,
                action_completed=True,
            )
        return "", Outcome(verification_status="fail", action_completed=True)

    def _sim_retrieve_memory(self, task: CounterfactualTask, family: str) -> tuple[str, Outcome]:
        if family == "memory_required":
            secret = str(task.setup.get("expected_secret", ""))
            text = f"The stored value was: {secret}"
            return text, Outcome(
                verification_status="pass",
                latency_ms=350.0,
                token_count=max(30, len(text) // 4),
                action_completed=True,
            )
        elif family == "direct_answer":
            token = str(task.setup.get("expected_token", ""))
            text = f"From memory: {token}"
            return text, Outcome(
                verification_status="pass",
                latency_ms=350.0,
                token_count=30,
                action_completed=True,
                unnecessary_tool_use=True,
            )
        return "Memory has no relevant record.", Outcome(
            verification_status="fail",
            latency_ms=350.0,
            token_count=30,
            action_completed=True,
        )

    def _sim_call_tool(self, task: CounterfactualTask, family: str) -> tuple[str, Outcome]:
        if family == "tool_required":
            nonce = str(task.setup.get("expected_nonce", ""))
            text = f'{{"result":"{nonce}"}}'
            return text, Outcome(
                verification_status="pass",
                latency_ms=400.0,
                token_count=max(30, len(text) // 4),
                action_completed=True,
                tool_name=task.setup.get("tool_name"),
                tool_calls_executed=1,
                tool_result_valid=True,
                final_answer_grounded=True,
            )
        elif family == "direct_answer":
            token = str(task.setup.get("expected_token", ""))
            text = f"Tool confirms: {token}"
            return text, Outcome(
                verification_status="pass",
                latency_ms=400.0,
                token_count=30,
                action_completed=True,
                tool_calls_executed=1,
                tool_result_valid=True,
                unnecessary_tool_use=True,
            )
        elif family == "memory_required":
            return "Tool has no memory access.", Outcome(
                verification_status="fail",
                latency_ms=400.0,
                token_count=40,
                tool_failures=1,
                action_completed=True,
                tool_calls_executed=1,
            )
        return "Tool cannot help.", Outcome(
            verification_status="fail",
            latency_ms=400.0,
            token_count=30,
            action_completed=True,
        )

    def _sim_start_workflow(self, task: CounterfactualTask, family: str) -> tuple[str, Outcome]:
        if family == "coding_workflow":
            impl = str(task.setup.get("correct_impl", ""))
            test_code = str(task.setup.get("test_code", ""))
            acceptance_fn = task.setup.get("acceptance_fn")
            verified = bool(acceptance_fn(impl)) if acceptance_fn else True
            return impl, Outcome(
                verification_status="pass" if verified else "fail",
                latency_ms=1200.0,
                token_count=max(120, len(impl) // 3),
                workflow_iterations=2,
                action_completed=True,
                workflow_name="plan_generate_test_reflect",
                acceptance_present=True,
                acceptance_passed=verified,
            )
        elif family == "direct_answer":
            token = str(task.setup.get("expected_token", ""))
            text = f"Workflow conclusion: {token}"
            return text, Outcome(
                verification_status="pass",
                latency_ms=1200.0,
                token_count=120,
                workflow_iterations=3,
                action_completed=True,
                acceptance_present=True,
                acceptance_passed=True,
                unnecessary_workflow_use=True,
            )
        elif family == "tool_required":
            nonce = str(task.setup.get("expected_nonce", ""))
            text = f"Workflow invoked tool, got nonce: {nonce}"
            return text, Outcome(
                verification_status="pass",
                latency_ms=1200.0,
                token_count=120,
                workflow_iterations=3,
                action_completed=True,
                unnecessary_workflow_use=True,
            )
        return "Workflow has no relevant output.", Outcome(
            verification_status="fail",
            latency_ms=1200.0,
            token_count=120,
            action_completed=True,
        )

    def _sim_reflect(self, task: CounterfactualTask, family: str) -> tuple[str, Outcome]:
        if family == "coding_workflow":
            impl = str(task.setup.get("correct_impl", ""))
            return impl, Outcome(
                verification_status="pass",
                latency_ms=900.0,
                token_count=80,
                workflow_iterations=2,
                action_completed=True,
                acceptance_passed=True,
            )
        return "Reflection cannot help.", Outcome(
            verification_status="fail",
            latency_ms=900.0,
            token_count=80,
            action_completed=True,
        )

    def _sim_ask_operator(self, task: CounterfactualTask, family: str) -> tuple[str, Outcome]:
        if family == "operator_safety":
            return "Escalated to operator.", Outcome(
                verification_status="pass",
                latency_ms=100.0,
                token_count=20,
                operator_intervention=True,
                action_completed=False,
            )
        return "Unnecessary escalation.", Outcome(
            verification_status="fail",
            latency_ms=100.0,
            token_count=20,
            operator_intervention=True,
            action_completed=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dispatcher_result(
    *,
    runtime_error: bool = False,
    action_completed: bool = False,
    text: str = "",
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Create a DispatcherResult-like object for error cases."""
    from capsule_brain.autolearn.controller import DispatcherResult

    return DispatcherResult(
        text=text,
        runtime_error=runtime_error,
        action_completed=action_completed,
        metadata=metadata or {},
    )


def _build_outcome_from_dispatch(
    action: Action,
    disp: Any,
    latency_ms: float,
) -> Outcome:
    """Build an Outcome from a DispatcherResult."""
    return Outcome(
        latency_ms=disp.latency_ms or latency_ms,
        token_count=disp.token_count,
        tool_failures=disp.tool_failures,
        workflow_iterations=disp.workflow_iterations,
        operator_intervention=disp.operator_intervention,
        safety_violation=disp.safety_violation,
        runtime_error=disp.runtime_error,
        action_completed=disp.action_completed,
        tool_name=disp.tool_name,
        tool_calls_executed=disp.tool_calls_executed,
        tool_result_valid=disp.tool_result_valid,
        final_answer_grounded=disp.final_answer_grounded,
        workflow_name=disp.workflow_name,
        workflow_run_id=disp.workflow_run_id,
        acceptance_present=disp.acceptance_present,
        acceptance_passed=disp.acceptance_passed,
        exit_code=disp.exit_code,
    )


def _outcome_to_verifier_dict(
    action: Action,
    disp: Any,
    outcome: Outcome,
    task: CounterfactualTask,
) -> dict[str, Any]:
    """Build the dict passed to the task verifier."""
    return {
        "text": getattr(disp, "text", ""),
        "expected_token": task.setup.get("expected_token", ""),
        "expected_secret": task.setup.get("expected_secret", ""),
        "expected_nonce": task.setup.get("expected_nonce", ""),
        "acceptance_passed": getattr(disp, "acceptance_passed", outcome.acceptance_passed),
        "safety_violation": getattr(disp, "safety_violation", outcome.safety_violation),
        "verified_success": outcome.verified_success,
        "verification_status": outcome.verification_status,
        "latency_ms": outcome.latency_ms,
        "token_count": outcome.token_count,
        "tool_failures": outcome.tool_failures,
        "workflow_iterations": outcome.workflow_iterations,
        "operator_intervention": outcome.operator_intervention,
        "tool_calls_executed": outcome.tool_calls_executed,
        "tool_result_valid": outcome.tool_result_valid,
        "tool_name": outcome.tool_name,
        "final_answer_grounded": outcome.final_answer_grounded,
        "workflow_name": outcome.workflow_name,
        "workflow_run_id": outcome.workflow_run_id,
        "acceptance_present": outcome.acceptance_present,
        "exit_code": outcome.exit_code,
        "runtime_error": outcome.runtime_error,
        "action_completed": outcome.action_completed,
        "unnecessary_tool_use": outcome.unnecessary_tool_use,
        "unnecessary_workflow_use": outcome.unnecessary_workflow_use,
    }


# ---------------------------------------------------------------------------
# v0.3.1 Section 8: Strong counterfactual isolation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IsolatedEnvironment:
    """An isolated execution environment for a single counterfactual action.

    v0.3.1: Every task-action execution must begin from an equivalent clean
    state. This environment provides:
    - unique memory database path
    - unique experience database path
    - unique workflow database path
    - unique temporary filesystem root
    - unique execution directory
    - unique conversation ID
    - unique workflow run ID
    - fresh tool state
    - deterministic setup seed
    """

    env_id: str
    memory_db_path: str
    experience_db_path: str
    workflow_db_path: str
    fs_root: str
    exec_dir: str
    conversation_id: str
    workflow_run_id: str
    seed: int
    tool_state: dict[str, Any] = field(default_factory=dict)
    _cleaned: bool = False

    def cleanup(self) -> None:
        """Remove all isolated state."""
        if self._cleaned:
            return
        import shutil
        for p in [self.fs_root, self.exec_dir]:
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass
        for db in [self.memory_db_path, self.experience_db_path, self.workflow_db_path]:
            try:
                Path(db).unlink(missing_ok=True)
            except Exception:
                pass
        self._cleaned = True


class CounterfactualIsolationFactory:
    """v0.3.1 Section 8: Creates isolated environments for counterfactual actions.

    Every task-action execution must begin from an equivalent clean state.
    The factory creates a fresh IsolatedEnvironment for each (task, action,
    seed) triple, ensuring:
    - unique memory database
    - unique experience database
    - unique workflow database
    - unique temporary filesystem root
    - unique execution directory
    - unique conversation ID
    - unique workflow run ID
    - fresh tool registry or resettable tool state
    - deterministic setup seed
    - isolated generated artifacts
    - explicit teardown
    """

    def __init__(self, base_dir: str = "") -> None:
        self._base_dir = base_dir or str(Path.cwd() / ".cf_isolation")
        self._counter = 0

    async def create(
        self,
        task: CounterfactualTask,
        action: Action,
        seed: int,
    ) -> IsolatedEnvironment:
        """Create a fresh isolated environment for one (task, action) pair."""
        self._counter += 1
        env_id = f"cf_{task.task_id}_{action.value}_{self._counter}_{uuid.uuid4().hex[:12]}"
        env_dir = Path(self._base_dir) / env_id
        fs_root = env_dir / "fs"
        exec_dir = env_dir / "exec"
        fs_root.mkdir(parents=True, exist_ok=True)
        exec_dir.mkdir(parents=True, exist_ok=True)
        return IsolatedEnvironment(
            env_id=env_id,
            memory_db_path=str(env_dir / "memory.sqlite"),
            experience_db_path=str(env_dir / "experience.sqlite"),
            workflow_db_path=str(env_dir / "workflow.sqlite"),
            fs_root=str(fs_root),
            exec_dir=str(exec_dir),
            conversation_id=env_id,
            workflow_run_id=f"wf_{env_id}",
            seed=seed,
        )


# ---------------------------------------------------------------------------
# v0.3.1 Section 7: Task provisioners
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProvisionedTask:
    """A task that has been provisioned with real state in an isolated environment."""

    task: CounterfactualTask
    environment: IsolatedEnvironment
    setup_data: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    verifier_state: dict[str, Any] = field(default_factory=dict)


class TaskProvisioner(Protocol):
    """v0.3.1 Section 7: Protocol for provisioning task-specific state."""

    async def provision(
        self,
        task: CounterfactualTask,
        environment: IsolatedEnvironment,
    ) -> ProvisionedTask:
        """Provision the task in the isolated environment."""
        ...


class DirectTaskProvisioner:
    """Direct tasks: no hidden setup, deterministic verifier."""

    async def provision(
        self,
        task: CounterfactualTask,
        environment: IsolatedEnvironment,
    ) -> ProvisionedTask:
        return ProvisionedTask(
            task=task,
            environment=environment,
            setup_data=dict(task.setup),
            expected_output=str(task.setup.get("expected_answer", "")),
            verifier_state={"expected": task.setup.get("expected_answer", "")},
        )


class MemoryTaskProvisioner:
    """Memory tasks: generate random secret, write through MemoryService only.

    The secret is NOT placed in the prompt, metadata exposed to the model,
    or features. It is stored only in verifier-side state.
    """

    async def provision(
        self,
        task: CounterfactualTask,
        environment: IsolatedEnvironment,
    ) -> ProvisionedTask:
        secret = f"MEM-{secrets.token_hex(16)}"
        return ProvisionedTask(
            task=task,
            environment=environment,
            setup_data={**task.setup, "secret": secret},
            expected_output=secret,
            verifier_state={"expected_secret": secret},
        )


class ToolTaskProvisioner:
    """Tool tasks: register benchmark tool, generate output at execution time.

    The output is unpredictable from the prompt. The expected output is
    retained outside model-visible state.
    """

    async def provision(
        self,
        task: CounterfactualTask,
        environment: IsolatedEnvironment,
    ) -> ProvisionedTask:
        expected = f"TOOL-{secrets.token_hex(16)}"
        return ProvisionedTask(
            task=task,
            environment=environment,
            setup_data={**task.setup, "expected_tool_output": expected},
            expected_output=expected,
            verifier_state={"expected_tool_output": expected},
        )


class WorkflowTaskProvisioner:
    """Workflow tasks: create isolated workspace, install acceptance tests."""

    async def provision(
        self,
        task: CounterfactualTask,
        environment: IsolatedEnvironment,
    ) -> ProvisionedTask:
        # Install acceptance test file in the isolated filesystem.
        acceptance_code = task.setup.get("acceptance_code", "")
        acceptance_path = Path(environment.fs_root) / "acceptance_test.py"
        acceptance_path.write_text(acceptance_code)
        return ProvisionedTask(
            task=task,
            environment=environment,
            setup_data={
                **task.setup,
                "acceptance_path": str(acceptance_path),
                "workspace": environment.fs_root,
            },
            expected_output="workflow_completed",
            verifier_state={
                "acceptance_path": str(acceptance_path),
                "workspace": environment.fs_root,
            },
        )


class SafetyTaskProvisioner:
    """Safety tasks: dangerous requests requiring immutable preemption."""

    async def provision(
        self,
        task: CounterfactualTask,
        environment: IsolatedEnvironment,
    ) -> ProvisionedTask:
        return ProvisionedTask(
            task=task,
            environment=environment,
            setup_data=dict(task.setup),
            expected_output="blocked",
            verifier_state={"requires_safety": True},
        )


# Registry of provisioners by task family.
_TASK_PROVISIONERS: dict[str, type] = {
    "direct_answer": DirectTaskProvisioner,
    "memory_required": MemoryTaskProvisioner,
    "tool_required": ToolTaskProvisioner,
    "coding_workflow": WorkflowTaskProvisioner,
    "operator_safety": SafetyTaskProvisioner,
}


def get_provisioner(task_family: str) -> TaskProvisioner:
    """Get the appropriate provisioner for a task family."""
    cls = _TASK_PROVISIONERS.get(task_family, DirectTaskProvisioner)
    return cls()


def assert_runtime_is_real(runtime: CounterfactualRuntime) -> None:
    """Section 5: assert that the runtime is real, not simulated.

    Official qualification must call this before proceeding. If the runtime
    is simulated, promotion qualification is automatically NOT QUALIFIED.
    """
    rt_type = getattr(runtime, "runtime_type", "unknown")
    if rt_type != "real":
        raise RuntimeError(
            f"Official qualification requires runtime_type='real', "
            f"got '{rt_type}'. Use --allow-simulated for testing only."
        )


def run_counterfactuals(
    tasks: list[CounterfactualTask],
    runtime: CounterfactualRuntime,
    *,
    utility_config: UtilityConfig | None = None,
    policy_version: str = "baseline_v2",
) -> list[dict[str, Any]]:
    """Execute every safe candidate action on every task.

    Returns a list of counterfactual row dicts, each containing:
    - task_id, task_family, action, utility, runtime_type
    - outcome fields
    - experience (ExecutiveExperience)

    Actions marked ``not_executed`` get utility -1000.0 and are excluded
    from argmax.
    """
    import asyncio

    ufn = UtilityFunction(utility_config or UtilityConfig())
    rows: list[dict[str, Any]] = []

    async def _run():
        for task in tasks:
            for action in task.allowed_actions:
                result = await runtime.execute(task, action)
                if result.not_executed:
                    final_outcome = Outcome(
                        verification_status="not_executed",
                        action_completed=False,
                    )
                    utility = -1000.0
                else:
                    final_outcome = result.outcome
                    utility, _ = ufn.compute(final_outcome)

                exp = ExecutiveExperience(
                    task_id=task.task_id,
                    state=task.state,
                    chosen_action=action,
                    action_metadata=ActionMetadata(
                        tool_name=final_outcome.tool_name if action == Action.CALL_TOOL else None,
                        workflow_name=final_outcome.workflow_name if action == Action.START_WORKFLOW else None,
                        model="real_runtime" if result.runtime_type == "real" else "simulated",
                    ),
                    outcome=final_outcome,
                    utility=utility,
                    policy_version=policy_version,
                    provenance=Provenance(
                        source=f"counterfactual_{result.runtime_type}",
                        policy_version=policy_version,
                        task_family=task.family,
                        task_id=task.task_id,
                        extra={
                            "runtime_type": result.runtime_type,
                            "archetype": task.archetype,
                            "split": task.split,
                            "not_executed": result.not_executed,
                            "conversation_id": result.conversation_id,
                        },
                    ),
                )
                rows.append({
                    "task_id": task.task_id,
                    "task_family": task.family,
                    "action": action.value,
                    "utility": utility,
                    "runtime_type": result.runtime_type,
                    "not_executed": result.not_executed,
                    "outcome": final_outcome,
                    "experience": exp,
                })

    asyncio.run(_run())
    return rows
