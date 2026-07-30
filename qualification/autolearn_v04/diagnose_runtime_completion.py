"""Runtime completion diagnostics (Section 9).

A mandatory stage that runs each action on a balanced diagnostic subset
before building the learning dataset. Reports completion rates by action,
family, and provider. If completion rate for any required action is below
threshold, Gate A must be BLOCKED.

This stage distinguishes failure causes instead of compressing every
failure into utility -1000:
    - dispatcher failure
    - provider output failure
    - tool selection failure
    - tool execution failure
    - final grounding failure
    - workflow startup failure
    - workflow generation failure
    - acceptance failure
    - result-schema mismatch
    - timeout
    - unsupported action
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import AUTOLEARN_QUALIFICATION_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .schemas import (
    OutcomeAvailability,
    VerificationOutcome,
    read_json,
    write_json,
)


def diagnose_runtime_completion(config: QualificationConfig) -> dict[str, Any]:
    """Run completion diagnostics on the existing counterfactual outcomes.

    Reads counterfactual_outcomes.json and produces a detailed breakdown
    by action, family, and failure cause. Does NOT re-execute tasks.
    """
    artifacts = Path(config.artifacts_dir)
    outcomes_path = artifacts / "counterfactual_outcomes.json"
    if not outcomes_path.exists():
        return _blocked_result("counterfactual_outcomes.json missing")

    outcomes = read_json(outcomes_path)
    bm = read_json(artifacts / "benchmark_manifest.json")
    task_by_id = {t["task_id"]: t for t in bm["tasks"]}

    # Aggregate by action.
    by_action: dict[str, dict[str, int]] = defaultdict(lambda: {
        "attempted": 0,
        "action_completed": 0,
        "verified_success": 0,
        "verification_failure": 0,
        "runtime_error": 0,
        "timeout": 0,
        "unsupported": 0,
        "not_executed": 0,
    })
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {
        "attempted": 0,
        "action_completed": 0,
        "verified_success": 0,
        "verification_failure": 0,
        "runtime_error": 0,
        "timeout": 0,
        "not_executed": 0,
    })
    by_cause: dict[str, int] = defaultdict(int)
    latencies_by_action: dict[str, list[float]] = defaultdict(list)
    tokens_by_action: dict[str, list[int]] = defaultdict(int)

    for o in outcomes:
        action = o.get("action_id", "unknown")
        task = task_by_id.get(o.get("task_id", ""))
        family = task.get("family", "unknown") if task else "unknown"
        avail = o.get("availability", "not_executed")
        verif = o.get("verification")
        meta = o.get("execution_metadata", {})
        latency = meta.get("latency_ms", 0.0)
        tokens = meta.get("token_count", 0)

        by_action[action]["attempted"] += 1
        by_family[family]["attempted"] += 1
        latencies_by_action[action].append(float(latency))
        if tokens:
            tokens_by_action[action] += int(tokens)

        if avail == "executed":
            by_action[action]["action_completed"] += 1
            by_family[family]["action_completed"] += 1
            if verif == "success":
                by_action[action]["verified_success"] += 1
                by_family[family]["verified_success"] += 1
            elif verif == "failure":
                by_action[action]["verification_failure"] += 1
                by_family[family]["verification_failure"] += 1
                # Classify the failure cause.
                cause = _classify_failure(o, task)
                by_cause[cause] += 1
        elif avail == "execution_error":
            by_action[action]["runtime_error"] += 1
            by_family[family]["runtime_error"] += 1
            cause = _classify_failure(o, task)
            by_cause[cause] += 1
        elif avail == "timeout":
            by_action[action]["timeout"] += 1
            by_family[family]["timeout"] += 1
            by_cause["timeout"] += 1
        elif avail == "not_executed":
            by_action[action]["not_executed"] += 1
            by_family[family]["not_executed"] += 1
            by_cause["not_executed"] += 1
        else:
            by_action[action]["unsupported"] += 1
            by_cause["unsupported"] += 1

    # Compute mean latency per action.
    mean_latency: dict[str, float] = {}
    for action, lats in latencies_by_action.items():
        mean_latency[action] = sum(lats) / len(lats) if lats else 0.0

    # Compute completion rates.
    completion_rates: dict[str, float] = {}
    for action, stats in by_action.items():
        if stats["attempted"] > 0:
            completion_rates[action] = stats["action_completed"] / stats["attempted"]
        else:
            completion_rates[action] = 0.0

    # Check thresholds.
    min_completion = config.min_action_completion
    failed_actions = [
        a for a, rate in completion_rates.items() if rate < min_completion
    ]
    all_pass = len(failed_actions) == 0

    result = {
        "schema_version": "runtime-completion-diagnostics/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "status": "PASS" if all_pass else "BLOCKED",
        "provider": config.provider,
        "min_action_completion": min_completion,
        "by_action": {a: dict(s) for a, s in by_action.items()},
        "by_family": {f: dict(s) for f, s in by_family.items()},
        "by_cause": dict(by_cause),
        "completion_rates": completion_rates,
        "mean_latency_ms": mean_latency,
        "total_tokens_by_action": dict(tokens_by_action),
        "failed_actions": failed_actions,
        "n_outcomes": len(outcomes),
        "gate_a_blocked": not all_pass,
        "blocking_reason": (
            f"Actions below completion threshold: {failed_actions}"
            if failed_actions else ""
        ),
    }
    write_json(artifacts / "runtime_completion_diagnostics.json", result)
    return result


def _classify_failure(outcome: dict, task: dict | None) -> str:
    """Classify a failure into a specific cause."""
    if not task:
        return "unknown_task"
    family = task.get("family", "unknown")
    meta = outcome.get("execution_metadata", {})
    evidence = meta.get("verification_evidence", {})
    error_type = outcome.get("error_type", "")

    if error_type == "timeout":
        return "timeout"
    if error_type == "execution_error":
        return "dispatcher_failure"
    if family == "tool_required":
        if not evidence.get("tool_result_valid"):
            if evidence.get("invocations", 0) == 0:
                return "tool_selection_failure"
            return "tool_execution_failure"
    if family == "workflow_required":
        if not evidence.get("acceptance_contract_present"):
            return "workflow_startup_failure"
        if not evidence.get("test_passed"):
            return "acceptance_failure"
    if family == "direct_answer":
        observed = evidence.get("observed", "")
        if not observed:
            return "provider_output_failure"
        return "result_schema_mismatch"
    if family == "memory_required":
        return "final_grounding_failure"
    return "other"


def _blocked_result(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "runtime-completion-diagnostics/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "status": "BLOCKED",
        "reason": reason,
        "gate_a_blocked": True,
        "by_action": {},
        "by_family": {},
        "by_cause": {},
        "completion_rates": {},
        "failed_actions": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose runtime completion before policy training.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "infrastructure", "scientific"], default="smoke")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    args = parser.parse_args()
    config = QualificationConfig(mode=args.mode, runtime=args.runtime, artifacts_dir=args.artifacts_dir)
    result = diagnose_runtime_completion(config)
    print(f"runtime completion: {result['status']}")
    if result.get("failed_actions"):
        print(f"  failed actions: {result['failed_actions']}")
    for action, rate in result.get("completion_rates", {}).items():
        print(f"  {action}: {rate:.2%}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
