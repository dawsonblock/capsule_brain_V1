"""Run counterfactual actions through the real Capsule Brain runtime (v0.3).

Section 4-7: This script executes every safe candidate action on every task
through the RealCounterfactualRuntime (or SimulatedCounterfactualRuntime
with --allow-simulated). Each execution gets a unique conversation_id and
isolated state. Unsafe actions on safety tasks are marked not_executed and
excluded from argmax.

Writes counterfactuals.json with:
- runtime_type ("real" or "simulated")
- n_rows, n_tasks
- rows: [{task_id, task_family, action, utility, runtime_type, not_executed, outcome, ...}]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from capsule_brain.autolearn.counterfactual import (
    CounterfactualRuntime,
    RealCounterfactualRuntime,
    SimulatedCounterfactualRuntime,
    assert_runtime_is_real,
    run_counterfactuals,
)
from capsule_brain.autolearn.schema import Action, Outcome
from capsule_brain.autolearn.utility import UtilityConfig, UtilityFunction

from tasks import build_all_tasks


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _outcome_to_dict(outcome: Outcome) -> dict:
    return {
        "verified_success": outcome.verified_success,
        "verification_status": outcome.verification_status,
        "latency_ms": outcome.latency_ms,
        "token_count": outcome.token_count,
        "tool_failures": outcome.tool_failures,
        "workflow_iterations": outcome.workflow_iterations,
        "operator_intervention": outcome.operator_intervention,
        "safety_violation": outcome.safety_violation,
        "runtime_error": outcome.runtime_error,
        "action_completed": outcome.action_completed,
        "tool_name": outcome.tool_name,
        "tool_calls_executed": outcome.tool_calls_executed,
        "tool_result_valid": outcome.tool_result_valid,
        "final_answer_grounded": outcome.final_answer_grounded,
        "workflow_name": outcome.workflow_name,
        "workflow_run_id": outcome.workflow_run_id,
        "acceptance_present": outcome.acceptance_present,
        "acceptance_passed": outcome.acceptance_passed,
        "exit_code": outcome.exit_code,
        "unnecessary_tool_use": outcome.unnecessary_tool_use,
        "unnecessary_workflow_use": outcome.unnecessary_workflow_use,
    }


def _build_real_runtime() -> RealCounterfactualRuntime:
    """Build a RealCounterfactualRuntime with a QualificationServiceFactory.

    v2.16.0 Priority 1: This is a GENUINE real runtime, not a placeholder.
    It uses a QualificationServiceFactory that creates a fresh, isolated
    ConversationService per task execution with:
    - Fresh MemoryService (``:memory:`` SQLite) pre-populated with task data
    - Real LLMGateway with a deterministic QualificationProvider
    - Real ToolRegistry with tools from task setup
    - Real WorkflowRunnerService with a real execution service

    Each execution dispatches through the real
    ConversationService.execute_executive_action() method. No simulator
    shortcuts. No task-family lookup tables. No shared state.
    """
    from capsule_brain.autolearn.qualification_runtime import (
        QualificationServiceFactory,
    )

    factory = QualificationServiceFactory()
    return RealCounterfactualRuntime(service_factory=factory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run counterfactual actions (v0.3)")
    parser.add_argument(
        "--runtime",
        choices=["real", "simulated"],
        default="real",
        help="Runtime: 'real' (default, official) or 'simulated' (testing only).",
    )
    parser.add_argument(
        "--allow-simulated",
        action="store_true",
        help="(Deprecated, use --runtime simulated) Allow the simulated runtime.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (default: counterfactuals.json in script dir)",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    out_path = Path(args.out) if args.out else out_dir / "counterfactuals.json"

    tasks = build_all_tasks()
    print(f"loaded {len(tasks)} tasks")

    # Build the runtime.
    use_simulated = args.allow_simulated or args.runtime == "simulated"
    runtime: CounterfactualRuntime
    if use_simulated:
        runtime = SimulatedCounterfactualRuntime()
        print("WARNING: using simulated runtime — NOT for official qualification")
    else:
        runtime = _build_real_runtime()
        # Section 5: assert the runtime is real before proceeding.
        try:
            assert_runtime_is_real(runtime)
        except RuntimeError:
            print("ERROR: official qualification requires runtime_type='real'")
            print("Use --runtime simulated for testing only.")
            sys.exit(1)

    print(f"runtime_type: {runtime.runtime_type}")

    # Run counterfactuals.
    utility_config = UtilityConfig()
    rows = run_counterfactuals(tasks, runtime, utility_config=utility_config)

    # Serialize.
    rows_data = []
    for r in rows:
        rows_data.append({
            "task_id": r["task_id"],
            "task_family": r["task_family"],
            "action": r["action"],
            "utility": r["utility"],
            "runtime_type": r["runtime_type"],
            "not_executed": r["not_executed"],
            "outcome": _outcome_to_dict(r["outcome"]),
        })

    # Group by task for summary.
    by_task: dict[str, list] = {}
    for r in rows_data:
        by_task.setdefault(r["task_id"], []).append(r)

    output = {
        "runtime_type": runtime.runtime_type,
        "n_rows": len(rows_data),
        "n_tasks": len(by_task),
        "utility_config_version": utility_config.config_version,
        "rows": rows_data,
    }
    text = json.dumps(output, sort_keys=True, indent=2)
    out_path.write_text(text)
    print(f"wrote {out_path}")
    print(f"rows: {len(rows_data)}, tasks: {len(by_task)}")
    print(f"counterfactuals SHA-256: {_compute_sha256(text)}")

    # Verify no simulated rows leaked into real qualification.
    if not use_simulated:
        sim_count = sum(1 for r in rows_data if r["runtime_type"] == "simulated")
        if sim_count > 0:
            print(f"ERROR: {sim_count} simulated rows in real qualification!")
            sys.exit(1)


if __name__ == "__main__":
    main()
