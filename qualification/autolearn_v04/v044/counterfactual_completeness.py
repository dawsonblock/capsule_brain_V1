"""Counterfactual completeness analysis for v0.4.4."""
from __future__ import annotations

from typing import Any


def audit_counterfactual_completeness(
    counterfactual_outcomes: list[dict],
    benchmark_manifest: dict,
    action_matrix: dict | None = None,
) -> dict[str, Any]:
    """Audit counterfactual completeness.

    Validates that all eligible actions have counterfactual outcomes.
    """
    eligible_actions = set(benchmark_manifest.get("eligible_actions", []))
    n_tasks = benchmark_manifest.get("n_tasks", 0)

    # Group outcomes by task.
    task_outcomes: dict[str, set[str]] = {}
    for outcome in counterfactual_outcomes:
        tid = outcome.get("task_id", "")
        action = outcome.get("action", "")
        if tid and action:
            task_outcomes.setdefault(tid, set()).add(action)

    n_complete = 0
    n_incomplete = 0
    incomplete_task_ids: list[str] = []

    for tid, actions in task_outcomes.items():
        if eligible_actions.issubset(actions):
            n_complete += 1
        else:
            n_incomplete += 1
            incomplete_task_ids.append(tid)

    # If no task outcomes, use action matrix results.
    if not task_outcomes and action_matrix:
        n_complete = action_matrix.get("n_tasks_complete", 0)
        n_incomplete = action_matrix.get("n_tasks_incomplete", 0)
        incomplete_task_ids = action_matrix.get("excluded_task_ids", [])

    status = "PASS" if n_incomplete == 0 else "FAIL"

    return {
        "status": status,
        "n_tasks_total": n_tasks,
        "n_tasks_complete": n_complete,
        "n_tasks_incomplete": n_incomplete,
        "incomplete_task_ids": incomplete_task_ids,
        "eligible_actions": list(eligible_actions),
    }
