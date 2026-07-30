"""Route-level qualification diagnostics for v0.4.1.

Before training, report for every action:
    attempted, completed, verified, unverifiable,
    runtime errors, timeouts, schema failures,
    provider-output failures, tool-selection failures,
    tool-execution failures, grounding failures,
    workflow startup failures, workflow generation failures,
    acceptance failures.

Also report by task family and split.

Define minimum completion thresholds.
Do not train if a route is operationally broken.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from . import AUTOLEARN_QUALIFICATION_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .schemas import read_json, write_json


def compute_route_diagnostics(config: QualificationConfig) -> dict[str, Any]:
    """Compute route-level diagnostics from counterfactual outcomes."""
    artifacts = Path(config.artifacts_dir)

    outcomes = read_json(artifacts / "counterfactual_outcomes.json")
    bm = read_json(artifacts / "benchmark_manifest.json")
    task_by_id = {t["task_id"]: t for t in bm["tasks"]}

    # Per-action stats.
    action_stats: dict[str, dict[str, int]] = defaultdict(lambda: {
        "attempted": 0,
        "completed": 0,
        "verified": 0,
        "unverifiable": 0,
        "runtime_errors": 0,
        "timeouts": 0,
        "schema_failures": 0,
        "provider_output_failures": 0,
        "tool_selection_failures": 0,
        "tool_execution_failures": 0,
        "grounding_failures": 0,
        "workflow_startup_failures": 0,
        "workflow_generation_failures": 0,
        "acceptance_failures": 0,
    })

    # Per-family-split stats.
    family_split_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"attempted": 0, "completed": 0, "verified": 0})
    )

    for o in outcomes:
        action = o.get("action_id", "UNKNOWN")
        task = task_by_id.get(o.get("task_id", {}))
        family = task.get("family", "unknown") if task else "unknown"
        split = task.get("split", "unknown") if task else "unknown"

        stats = action_stats[action]
        stats["attempted"] += 1

        availability = o.get("availability", "unknown")
        verification = o.get("verification", "unknown")

        if availability == "executed":
            stats["completed"] += 1
        elif availability == "runtime_error":
            stats["runtime_errors"] += 1
        elif availability == "timeout":
            stats["timeouts"] += 1
        elif availability == "schema_error":
            stats["schema_failures"] += 1

        if verification == "success":
            stats["verified"] += 1
        elif verification == "failure":
            stats["acceptance_failures"] += 1
            # Classify failure type.
            evidence = o.get("execution_metadata", {}).get("verification_evidence", {})
            reason = evidence.get("reason", "")
            if "tool" in reason.lower():
                stats["tool_execution_failures"] += 1
            elif "workflow" in reason.lower():
                stats["workflow_generation_failures"] += 1
            elif "grounding" in reason.lower():
                stats["grounding_failures"] += 1
            elif "provider" in reason.lower():
                stats["provider_output_failures"] += 1
        elif verification == "unverifiable":
            stats["unverifiable"] += 1

        # Family-split stats.
        fs = family_split_stats[family][split]
        fs["attempted"] += 1
        if availability == "executed":
            fs["completed"] += 1
        if verification == "success":
            fs["verified"] += 1

    # Compute rates and check thresholds.
    min_completion = config.min_action_completion
    min_verification = 0.10  # at least 10% of attempted should be verified
    max_runtime_error = 0.50  # more than 50% runtime errors is broken

    route_assessment: dict[str, Any] = {}
    for action, stats in action_stats.items():
        attempted = stats["attempted"]
        if attempted == 0:
            route_assessment[action] = {"status": "not_attempted", "rates": {}}
            continue
        completion_rate = stats["completed"] / attempted
        verification_rate = stats["verified"] / attempted
        runtime_error_rate = stats["runtime_errors"] / attempted

        status = "pass"
        if completion_rate < min_completion:
            status = "broken"
        elif runtime_error_rate > max_runtime_error:
            status = "broken"
        elif verification_rate < min_verification:
            status = "degraded"

        route_assessment[action] = {
            "status": status,
            "rates": {
                "completion": completion_rate,
                "verification": verification_rate,
                "runtime_error": runtime_error_rate,
            },
            "counts": stats,
        }

    any_broken = any(r["status"] == "broken" for r in route_assessment.values())

    result = {
        "schema_version": "route-diagnostics/1",
        "protocol_version": PROTOCOL_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "action_stats": {k: dict(v) for k, v in action_stats.items()},
        "family_split_stats": {
            family: {split: dict(stats) for split, stats in splits.items()}
            for family, splits in family_split_stats.items()
        },
        "route_assessment": route_assessment,
        "any_broken": any_broken,
        "thresholds": {
            "min_completion": min_completion,
            "min_verification": min_verification,
            "max_runtime_error": max_runtime_error,
        },
    }
    write_json(artifacts / "route_diagnostics.json", result)
    return result
