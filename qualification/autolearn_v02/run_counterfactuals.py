"""Counterfactual execution harness for AutoLearn v0.2.

Executes every safe candidate action on every task using the
SimulatedCounterfactualRuntime and records the verified utility. Produces
counterfactuals.json with SHA-256 provenance.

v0.2 differences from v0.1:
- Uses the v0.2 task set (430 tasks, 44 archetypes).
- Records v2 outcome fields (runtime_error, action_completed, tool-specific,
  workflow-specific, over-routing flags).
- Computes SHA-256 hashes of the counterfactuals for provenance.
- Supports RealCounterfactualRuntime for live-LLM mode.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.dataset import CounterfactualRow
from capsule_brain.autolearn.schema import (
    Action,
    ActionMetadata,
    ExecutiveExperience,
    ExecutiveState,
    Outcome,
    Provenance,
)
from capsule_brain.autolearn.utility import UtilityConfig, UtilityFunction

from tasks import RoutingTask, build_all_tasks


@dataclass(slots=True)
class ActionResult:
    action: Action
    outcome: Outcome
    text: str = ""
    acceptance_passed: bool = False
    safety_violation: bool = False
    not_executed: bool = False
    conversation_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# Actions that are unsafe for counterfactual execution on operator_safety
# tasks. Executing them would be a safety violation, so they are skipped.
_UNSAFE_ACTIONS_FOR_SAFETY_TASKS = frozenset({
    Action.ANSWER_DIRECT,
    Action.RETRIEVE_MEMORY,
    Action.CALL_TOOL,
    Action.START_WORKFLOW,
})


class SimulatedCounterfactualRuntime:
    """Deterministic stand-in for Capsule Brain's action execution (v0.2).

    Mirrors the real action semantics with v2 outcome fields. The verifiers
    in tasks.py are the ground truth.

    Isolation: each call to execute() gets a unique conversation_id and
    workflow_run_id so counterfactual actions cannot contaminate each other.
    State counters are per-execution, not shared across actions.
    """

    def __init__(self) -> None:
        self._execution_counter = 0

    def execute(self, task: RoutingTask, action: Action) -> ActionResult:
        self._execution_counter += 1
        # Unique conversation ID per (task, action) execution.
        conv_id = f"cf_{task.task_id}_{action.value}_{self._execution_counter}"
        # Section 15: skip unsafe actions on operator_safety tasks.
        if task.family == "operator_safety" and action in _UNSAFE_ACTIONS_FOR_SAFETY_TASKS:
            return ActionResult(
                action=action,
                outcome=Outcome(
                    verification_status="not_executed",
                    safety_violation=False,
                    action_completed=False,
                ),
                text="",
                not_executed=True,
                conversation_id=conv_id,
            )
        if action == Action.ANSWER_DIRECT:
            result = self._answer_direct(task)
        elif action == Action.RETRIEVE_MEMORY:
            result = self._retrieve_memory(task)
        elif action == Action.CALL_TOOL:
            result = self._call_tool(task)
        elif action == Action.START_WORKFLOW:
            result = self._start_workflow(task)
        elif action == Action.REFLECT:
            result = self._reflect(task)
        elif action == Action.ASK_OPERATOR:
            result = self._ask_operator(task)
        else:
            result = ActionResult(
                action=action,
                outcome=Outcome(verification_status="error", runtime_error=True),
                text="",
            )
        result.conversation_id = conv_id
        # Assign unique workflow_run_id for workflow actions.
        if action == Action.START_WORKFLOW and result.outcome.workflow_run_id is None:
            result.outcome.workflow_run_id = f"wf_{self._execution_counter}"
        return result

    def _answer_direct(self, task: RoutingTask) -> ActionResult:
        text = ""
        verified = False
        if task.family == "direct_answer":
            token = str(task.setup.get("expected_token", ""))
            text = f"The answer is {token}."
            verified = True
        elif task.family == "memory_required":
            text = "I don't have that stored value available."
            verified = False
        elif task.family == "tool_required":
            text = "I cannot fetch a live nonce without the tool."
            verified = False
        elif task.family == "coding_workflow":
            text = f"def {task.setup.get('fn_name','f')}(*a, **k):\n    pass\n"
            verified = False
        elif task.family == "reflection_req":
            text = str(task.setup.get("faulty_artifact", ""))
            verified = False
        elif task.family == "operator_safety":
            text = "I cannot perform that action; it requires operator approval."
            verified = False
        return ActionResult(
            action=Action.ANSWER_DIRECT,
            outcome=Outcome(
                verified_success=verified,
                verification_status="pass" if verified else "fail",
                latency_ms=200.0,
                token_count=max(20, len(text) // 4),
                action_completed=True,
            ),
            text=text,
        )

    def _retrieve_memory(self, task: RoutingTask) -> ActionResult:
        text = ""
        verified = False
        unnecessary = False
        if task.family == "memory_required":
            secret = str(task.setup.get("expected_secret", ""))
            text = f"The stored pin was: {secret}"
            verified = True
        elif task.family == "direct_answer":
            token = str(task.setup.get("expected_token", ""))
            text = f"From memory: {token}"
            verified = True
            unnecessary = True  # correct but unnecessary
        elif task.family == "tool_required":
            text = "Memory has no record of a live nonce."
            verified = False
        elif task.family == "coding_workflow":
            text = "Memory has no matching implementation."
            verified = False
        elif task.family == "reflection_req":
            text = str(task.setup.get("faulty_artifact", ""))
            verified = False
        elif task.family == "operator_safety":
            text = "I cannot perform that action; it requires operator approval."
            verified = False
        return ActionResult(
            action=Action.RETRIEVE_MEMORY,
            outcome=Outcome(
                verified_success=verified,
                verification_status="pass" if verified else "fail",
                latency_ms=350.0,
                token_count=max(30, len(text) // 4),
                action_completed=True,
                unnecessary_tool_use=unnecessary,
            ),
            text=text,
        )

    def _call_tool(self, task: RoutingTask) -> ActionResult:
        text = ""
        verified = False
        tool_failures = 0
        unnecessary = False
        if task.family == "tool_required":
            nonce = str(task.setup.get("expected_nonce", ""))
            text = f"Tool returned nonce: {nonce}"
            verified = True
        elif task.family == "direct_answer":
            token = str(task.setup.get("expected_token", ""))
            text = f"Tool confirms: {token}"
            verified = True
            unnecessary = True  # correct but unnecessary
        elif task.family == "memory_required":
            text = "Tool has no memory access."
            verified = False
            tool_failures = 1
        elif task.family == "coding_workflow":
            text = "Tool cannot synthesize a passing implementation."
            verified = False
        elif task.family == "reflection_req":
            text = str(task.setup.get("faulty_artifact", ""))
            verified = False
        elif task.family == "operator_safety":
            return ActionResult(
                action=Action.CALL_TOOL,
                outcome=Outcome(
                    verified_success=False,
                    verification_status="fail",
                    latency_ms=400.0,
                    token_count=40,
                    tool_failures=1,
                    safety_violation=True,
                    action_completed=True,
                ),
                text="Refusing to execute destructive tool action.",
                safety_violation=True,
            )
        return ActionResult(
            action=Action.CALL_TOOL,
            outcome=Outcome(
                verified_success=verified,
                verification_status="pass" if verified else "fail",
                latency_ms=400.0,
                token_count=max(30, len(text) // 4),
                tool_failures=tool_failures,
                action_completed=True,
                tool_name=task.setup.get("tool_name"),
                tool_calls_executed=1,
                tool_result_valid=verified,
                unnecessary_tool_use=unnecessary,
            ),
            text=text,
        )

    def _start_workflow(self, task: RoutingTask) -> ActionResult:
        text = ""
        verified = False
        iterations = 1
        unnecessary = False
        if task.family == "coding_workflow":
            impl = str(task.setup.get("correct_impl", ""))
            text = impl
            # Run the acceptance test.
            test_code = str(task.setup.get("test_code", ""))
            acceptance_fn = task.setup.get("acceptance_fn")
            if acceptance_fn is not None:
                verified = bool(acceptance_fn(impl))
            else:
                verified = True
            iterations = 2
        elif task.family == "reflection_req":
            text = str(task.setup.get("faulty_artifact", ""))
            verified = False
        elif task.family == "direct_answer":
            token = str(task.setup.get("expected_token", ""))
            text = f"Workflow conclusion: {token}"
            verified = True
            unnecessary = True  # correct but heavy cost
            iterations = 3
        elif task.family == "memory_required":
            text = "Workflow has no memory access."
            verified = False
        elif task.family == "tool_required":
            nonce = str(task.setup.get("expected_nonce", ""))
            text = f"Workflow invoked tool, got nonce: {nonce}"
            verified = True
            unnecessary = True  # correct but heavy cost
            iterations = 3
        elif task.family == "operator_safety":
            return ActionResult(
                action=Action.START_WORKFLOW,
                outcome=Outcome(
                    verified_success=False,
                    verification_status="fail",
                    latency_ms=1500.0,
                    token_count=200,
                    safety_violation=True,
                    action_completed=True,
                ),
                text="Refusing to run destructive workflow.",
                safety_violation=True,
            )
        return ActionResult(
            action=Action.START_WORKFLOW,
            outcome=Outcome(
                verified_success=verified,
                verification_status="pass" if verified else "fail",
                latency_ms=1200.0,
                token_count=max(120, len(text) // 3),
                workflow_iterations=iterations,
                action_completed=True,
                workflow_name="plan_generate_test_reflect",
                acceptance_present=True,
                acceptance_passed=verified,
                unnecessary_workflow_use=unnecessary,
            ),
            text=text,
            acceptance_passed=verified,
        )

    def _reflect(self, task: RoutingTask) -> ActionResult:
        text = ""
        verified = False
        if task.family == "reflection_req":
            # Reflection produces the correct implementation.
            fn = task.setup.get("fn_name", "f")
            # Use the test code to verify.
            test_code = str(task.setup.get("test_code", ""))
            # Generate a correct implementation by running the test and
            # extracting the expected behavior. For the simulated runtime,
            # we use a known-correct implementation.
            correct_impls = {
                "add": "def add(a, b):\n    return a + b\n",
                "is_even": "def is_even(n):\n    return n % 2 == 0\n",
                "reverse_string": "def reverse_string(s):\n    return s[::-1]\n",
                "factorial": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n",
                "max_of_two": "def max_of_two(a, b):\n    return a if a >= b else b\n",
            }
            text = correct_impls.get(fn, f"def {fn}(*a, **k):\n    pass\n")
            # Run the acceptance test.
            try:
                namespace: dict[str, Any] = {}
                exec(text, namespace)
                exec(test_code, namespace)
                verified = True
            except Exception:
                verified = False
        elif task.family == "coding_workflow":
            impl = str(task.setup.get("correct_impl", ""))
            text = impl
            verified = True
        elif task.family == "direct_answer":
            token = str(task.setup.get("expected_token", ""))
            text = f"After reflection: {token}"
            verified = True
        elif task.family == "memory_required":
            text = "Reflection cannot recover un-stored memory."
            verified = False
        elif task.family == "tool_required":
            text = "Reflection cannot produce a live nonce."
            verified = False
        elif task.family == "operator_safety":
            text = "I cannot perform that action; it requires operator approval."
            verified = False
        return ActionResult(
            action=Action.REFLECT,
            outcome=Outcome(
                verified_success=verified,
                verification_status="pass" if verified else "fail",
                latency_ms=900.0,
                token_count=max(80, len(text) // 3),
                workflow_iterations=2,
                action_completed=True,
            ),
            text=text,
            acceptance_passed=verified,
        )

    def _ask_operator(self, task: RoutingTask) -> ActionResult:
        text = "Escalated to operator for approval."
        verified = True
        if task.family != "operator_safety":
            verified = False
        return ActionResult(
            action=Action.ASK_OPERATOR,
            outcome=Outcome(
                verified_success=verified,
                verification_status="pass" if verified else "fail",
                latency_ms=100.0,
                token_count=20,
                operator_intervention=True,
                action_completed=False,
            ),
            text=text,
        )


# Backward-compatible alias.
DeterministicRuntime = SimulatedCounterfactualRuntime


def _correct_implementation(fn_name: str) -> str:
    impls = {
        "add": "def add(a, b):\n    return a + b\n",
        "is_even": "def is_even(n):\n    return n % 2 == 0\n",
        "max_of_two": "def max_of_two(a, b):\n    return a if a >= b else b\n",
        "reverse_string": "def reverse_string(s):\n    return s[::-1]\n",
    }
    return impls.get(fn_name, f"def {fn_name}(*a, **k):\n    pass\n")


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def run_counterfactuals(
    tasks: list[RoutingTask] | None = None,
    *,
    utility_config: UtilityConfig | None = None,
    policy_version: str = "baseline_v2",
    runtime: SimulatedCounterfactualRuntime | None = None,
) -> list[CounterfactualRow]:
    """Execute every safe candidate action on every task and record U(task, a)."""
    tasks = tasks or build_all_tasks()
    rt = runtime or SimulatedCounterfactualRuntime()
    ufn = UtilityFunction(utility_config or UtilityConfig())
    rows: list[CounterfactualRow] = []

    for task in tasks:
        for action in task.allowed_actions:
            result = rt.execute(task, action)
            # Section 15: not_executed actions get zero utility and are
            # excluded from argmax (they cannot be the best action).
            if result.not_executed:
                final_outcome = Outcome(
                    verification_status="not_executed",
                    action_completed=False,
                )
                utility = -1000.0  # below any executed action
                exp = ExecutiveExperience(
                    task_id=task.task_id,
                    state=task.state,
                    chosen_action=action,
                    action_metadata=ActionMetadata(model="simulated_runtime_v2"),
                    outcome=final_outcome,
                    utility=utility,
                    policy_version=policy_version,
                    provenance=Provenance(
                        source="counterfactual_v2",
                        policy_version=policy_version,
                        task_family=task.family,
                        task_id=task.task_id,
                        extra={
                            "expected_action": task.expected_action.value,
                            "archetype": task.archetype,
                            "split": task.split,
                            "not_executed": True,
                        },
                    ),
                )
                rows.append(CounterfactualRow(
                    task_id=task.task_id,
                    task_family=task.family,
                    action=action,
                    experience=exp,
                    utility=utility,
                ))
                continue
            outcome_dict = {
                "text": result.text,
                "expected_token": task.setup.get("expected_token", ""),
                "expected_secret": task.setup.get("expected_secret", ""),
                "expected_nonce": task.setup.get("expected_nonce", ""),
                "acceptance_passed": result.acceptance_passed,
                "safety_violation": result.safety_violation,
                "verified_success": result.outcome.verified_success,
                "verification_status": result.outcome.verification_status,
                "latency_ms": result.outcome.latency_ms,
                "token_count": result.outcome.token_count,
                "tool_failures": result.outcome.tool_failures,
                "workflow_iterations": result.outcome.workflow_iterations,
                "operator_intervention": result.outcome.operator_intervention,
                "tool_calls_executed": result.outcome.tool_calls_executed,
                "tool_result_valid": result.outcome.tool_result_valid,
                "tool_name": result.outcome.tool_name,
                "final_answer_grounded": result.outcome.final_answer_grounded,
                "workflow_name": result.outcome.workflow_name,
                "workflow_run_id": result.outcome.workflow_run_id,
                "acceptance_present": result.outcome.acceptance_present,
                "exit_code": result.outcome.exit_code,
                "runtime_error": result.outcome.runtime_error,
                "action_completed": result.outcome.action_completed,
                "unnecessary_tool_use": result.outcome.unnecessary_tool_use,
                "unnecessary_workflow_use": result.outcome.unnecessary_workflow_use,
            }
            passed, status = task.verifier(action, outcome_dict)
            final_outcome = Outcome(
                verified_success=passed,
                verification_status=status,
                latency_ms=result.outcome.latency_ms,
                token_count=result.outcome.token_count,
                tool_failures=result.outcome.tool_failures,
                workflow_iterations=result.outcome.workflow_iterations,
                operator_intervention=result.outcome.operator_intervention,
                safety_violation=result.safety_violation,
                runtime_error=result.outcome.runtime_error,
                action_completed=result.outcome.action_completed,
                tool_name=result.outcome.tool_name,
                tool_calls_executed=result.outcome.tool_calls_executed,
                tool_result_valid=result.outcome.tool_result_valid,
                final_answer_grounded=result.outcome.final_answer_grounded,
                workflow_name=result.outcome.workflow_name,
                workflow_run_id=result.outcome.workflow_run_id,
                acceptance_present=result.outcome.acceptance_present,
                acceptance_passed=result.outcome.acceptance_passed,
                exit_code=result.outcome.exit_code,
                unnecessary_tool_use=result.outcome.unnecessary_tool_use,
                unnecessary_workflow_use=result.outcome.unnecessary_workflow_use,
            )
            utility, _ = ufn.compute(final_outcome)
            exp = ExecutiveExperience(
                task_id=task.task_id,
                state=task.state,
                chosen_action=action,
                action_metadata=ActionMetadata(
                    tool_name=task.setup.get("tool_name") if action == Action.CALL_TOOL else None,
                    workflow_name="plan_generate_test_reflect" if action == Action.START_WORKFLOW else None,
                    model="simulated_runtime_v2",
                ),
                outcome=final_outcome,
                utility=utility,
                policy_version=policy_version,
                provenance=Provenance(
                    source="counterfactual_v2",
                    policy_version=policy_version,
                    task_family=task.family,
                    task_id=task.task_id,
                    extra={
                        "expected_action": task.expected_action.value,
                        "archetype": task.archetype,
                        "split": task.split,
                    },
                ),
            )
            rows.append(CounterfactualRow(
                task_id=task.task_id,
                task_family=task.family,
                action=action,
                experience=exp,
                utility=utility,
            ))
    return rows


def save_counterfactuals(rows: list[CounterfactualRow], path: str | Path) -> str:
    """Save counterfactuals with SHA-256 provenance. Returns the digest."""
    data = {
        "n_rows": len(rows),
        "rows": [
            {
                "task_id": r.task_id,
                "task_family": r.task_family,
                "action": r.action.value,
                "utility": r.utility,
                "outcome": {
                    "verified_success": r.experience.outcome.verified_success,
                    "verification_status": r.experience.outcome.verification_status,
                    "latency_ms": r.experience.outcome.latency_ms,
                    "token_count": r.experience.outcome.token_count,
                    "tool_failures": r.experience.outcome.tool_failures,
                    "workflow_iterations": r.experience.outcome.workflow_iterations,
                    "operator_intervention": r.experience.outcome.operator_intervention,
                    "safety_violation": r.experience.outcome.safety_violation,
                    "runtime_error": r.experience.outcome.runtime_error,
                    "action_completed": r.experience.outcome.action_completed,
                    "tool_name": r.experience.outcome.tool_name,
                    "tool_calls_executed": r.experience.outcome.tool_calls_executed,
                    "tool_result_valid": r.experience.outcome.tool_result_valid,
                    "final_answer_grounded": r.experience.outcome.final_answer_grounded,
                    "workflow_name": r.experience.outcome.workflow_name,
                    "workflow_run_id": r.experience.outcome.workflow_run_id,
                    "acceptance_present": r.experience.outcome.acceptance_present,
                    "acceptance_passed": r.experience.outcome.acceptance_passed,
                    "exit_code": r.experience.outcome.exit_code,
                    "unnecessary_tool_use": r.experience.outcome.unnecessary_tool_use,
                    "unnecessary_workflow_use": r.experience.outcome.unnecessary_workflow_use,
                },
                "expected_action": r.experience.provenance.extra.get("expected_action", ""),
                "archetype": r.experience.provenance.extra.get("archetype", ""),
            }
            for r in rows
        ],
    }
    text = json.dumps(data, sort_keys=True, indent=2)
    Path(path).write_text(text)
    return _compute_sha256(text)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    tasks = build_all_tasks()
    rows = run_counterfactuals(tasks)
    digest = save_counterfactuals(rows, out_dir / "counterfactuals.json")
    print(f"Built {len(rows)} counterfactual rows from {len(tasks)} tasks.")
    print(f"SHA-256: {digest}")
    # Quick summary.
    by_family: dict[str, dict[str, float]] = {}
    for r in rows:
        by_family.setdefault(r.task_family, {})
        by_family[r.task_family].setdefault(r.action.value, 0.0)
        by_family[r.task_family][r.action.value] += r.utility
    for fam, acts in sorted(by_family.items()):
        best = max(acts, key=acts.get)
        print(f"  {fam}: best={best} sum_utility={acts[best]:.1f}")


if __name__ == "__main__":
    main()
