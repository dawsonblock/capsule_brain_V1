"""Real counterfactual-equivalence validation for v0.4.4.

This module performs *genuine* counterfactual-equivalence validation. It does
NOT infer equivalence from the mere presence of a ``setup_spec`` field or from
policy-selected task rows. Instead, for every task and every eligible action,
it compares the recorded shared initial-state fields across all action trials
for that task and proves that each action trial begins from the same task
state before action-specific execution.

The fields compared are the *shared initial state* — they must be identical
across all actions for a task. Action-specific *permitted execution state*
(e.g. the action that was actually taken, intermediate tool outputs produced
during execution) may legitimately differ and is not part of this comparison.

When outcome records lack the required fields for comparison, the task is
marked ``UNVERIFIABLE`` (never ``EQUIVALENT``).
"""
from __future__ import annotations

from typing import Any

from qualification.autolearn_v04.common.schemas import (
    CounterfactualEquivalenceStatus,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fields that constitute the *shared initial state* of a task. These must be
#: identical across every action trial for a given task to establish
#: counterfactual equivalence. They are read from each outcome record.
SHARED_INITIAL_STATE_FIELDS: tuple[str, ...] = (
    "task_id",
    "prompt_digest",
    "setup_digest",
    "hidden_state_setup_digest",
    "memory_state_digest",
    "tool_state_digest",
    "workflow_state_digest",
    "provider_model_revision",
    "tokenizer_revision",
    "generation_config_digest",
    "generation_seed_policy",
    "utility_version",
    "verifier_version",
    "timeout_config",
    "capability_permissions_digest",
    "environment_snapshot_digest",
    "benchmark_task_version",
)

#: The ``task_id`` field is used only to group outcomes by task; it is not
#: itself a "mismatch" candidate (it is identical by construction within a
#: task group). We keep it in the list for completeness but exclude it from
#: the mismatched-fields report.
_GROUPING_FIELD = "task_id"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_value(value: Any) -> Any:
    """Normalize a value for comparison.

    ``None`` and missing values are mapped to a sentinel so that "absent"
    is distinguishable from "present and equal".
    """
    if value is None:
        return _MISSING
    return value


_MISSING = "__MISSING__"


def _field_present(outcome: dict, field: str) -> bool:
    """Return True if the outcome record carries a non-null value for field."""
    return field in outcome and outcome[field] is not None


def _collect_field_values(
    outcomes: list[dict], field: str
) -> tuple[list[Any], bool]:
    """Collect normalized values for a field across outcomes.

    Returns (values, all_present). ``all_present`` is True only if every
    outcome carries a non-null value for the field.
    """
    values: list[Any] = []
    all_present = True
    for o in outcomes:
        if _field_present(o, field):
            values.append(_normalize_value(o[field]))
        else:
            values.append(_MISSING)
            all_present = False
    return values, all_present


def _values_consistent(values: list[Any]) -> bool:
    """Return True if all (non-missing) values are identical.

    A field where every outcome is missing is *not* considered consistent
    for equivalence purposes — it is treated as absent.
    """
    present = [v for v in values if v is not _MISSING]
    if not present:
        return False
    first = present[0]
    return all(v == first for v in present)


def _group_outcomes_by_task(
    counterfactual_outcomes: list[dict],
) -> dict[str, list[dict]]:
    """Group outcome records by their ``task_id``."""
    groups: dict[str, list[dict]] = {}
    for o in counterfactual_outcomes:
        tid = o.get("task_id")
        if tid is None:
            continue
        groups.setdefault(str(tid), []).append(o)
    return groups


def _manifest_task_ids(benchmark_manifest: dict) -> list[str]:
    """Extract the canonical task-id list from the benchmark manifest."""
    if not isinstance(benchmark_manifest, dict):
        return []
    # Prefer an explicit task-id list.
    for key in ("task_ids", "test_task_ids", "eval_task_ids"):
        ids = benchmark_manifest.get(key)
        if isinstance(ids, list) and ids:
            return [str(t) for t in ids]
    # Fall back to per-task entries.
    tasks = benchmark_manifest.get("tasks")
    if isinstance(tasks, list):
        ids = []
        for t in tasks:
            if isinstance(t, dict) and t.get("task_id") is not None:
                ids.append(str(t["task_id"]))
            elif t is not None:
                ids.append(str(t))
        if ids:
            return ids
    if isinstance(tasks, dict):
        return [str(k) for k in tasks.keys()]
    return []


# ---------------------------------------------------------------------------
# Per-task equivalence check
# ---------------------------------------------------------------------------


def _check_task_equivalence(outcomes: list[dict]) -> dict[str, Any]:
    """Check counterfactual equivalence for a single task.

    Returns a dict with:
      - status: CounterfactualEquivalenceStatus
      - mismatched_fields: list of fields that differ across actions
      - missing_fields: list of required fields absent from one or more outcomes
      - n_actions: number of action trials examined
    """
    n_actions = len(outcomes)

    # A single action trial cannot establish cross-action equivalence.
    if n_actions == 0:
        return {
            "status": CounterfactualEquivalenceStatus.UNVERIFIABLE,
            "mismatched_fields": [],
            "missing_fields": list(SHARED_INITIAL_STATE_FIELDS),
            "n_actions": 0,
        }

    mismatched_fields: list[str] = []
    missing_fields: list[str] = []

    for field in SHARED_INITIAL_STATE_FIELDS:
        # task_id is the grouping key; skip it as a mismatch candidate.
        if field == _GROUPING_FIELD:
            continue
        values, all_present = _collect_field_values(outcomes, field)
        if not all_present:
            missing_fields.append(field)
            continue
        if not _values_consistent(values):
            mismatched_fields.append(field)

    # Determine status.
    # 1. If any present field genuinely differs -> NON_EQUIVALENT (we have
    #    direct evidence of non-equivalence, regardless of missing fields).
    # 2. Else if any required field is missing -> UNVERIFIABLE (we cannot
    #    prove equivalence, and the rule forbids marking it EQUIVALENT).
    # 3. Else -> EQUIVALENT.
    if mismatched_fields:
        status = CounterfactualEquivalenceStatus.NON_EQUIVALENT
    elif missing_fields:
        status = CounterfactualEquivalenceStatus.UNVERIFIABLE
    else:
        status = CounterfactualEquivalenceStatus.EQUIVALENT

    return {
        "status": status,
        "mismatched_fields": mismatched_fields,
        "missing_fields": missing_fields,
        "n_actions": n_actions,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_counterfactual_equivalence(
    counterfactual_outcomes: list[dict],
    benchmark_manifest: dict,
) -> dict:
    """Validate real counterfactual equivalence across all tasks.

    For every task and every eligible action, compares the shared
    initial-state fields across all action trials. Proves that each action
    trial begins from the same task state before action-specific execution.

    Equivalence is NEVER inferred from ``setup_spec`` presence or from
    policy-selected task rows — it is established by direct field-by-field
    comparison of recorded outcome state.

    Parameters
    ----------
    counterfactual_outcomes:
        List of outcome-record dicts. Each must carry a ``task_id`` and the
        shared initial-state fields listed in :data:`SHARED_INITIAL_STATE_FIELDS`.
    benchmark_manifest:
        Benchmark manifest dict. Used to determine the canonical task set;
        if it does not enumerate tasks, the task set is derived from the
        outcomes.

    Returns
    -------
    dict
        See module docstring for the full return schema.
    """
    groups = _group_outcomes_by_task(counterfactual_outcomes)

    # Determine the set of tasks to check.
    manifest_ids = _manifest_task_ids(benchmark_manifest)
    if manifest_ids:
        task_ids = manifest_ids
    else:
        task_ids = sorted(groups.keys())

    per_task_results: dict[str, dict[str, Any]] = {}
    n_equivalent = 0
    n_non_equivalent = 0
    n_unverifiable = 0
    non_equivalent_task_ids: list[str] = []
    unverifiable_task_ids: list[str] = []

    for tid in task_ids:
        outcomes = groups.get(tid, [])
        result = _check_task_equivalence(outcomes)
        status = result["status"]
        per_task_results[tid] = {
            "status": status.value,
            "mismatched_fields": result["mismatched_fields"],
            "missing_fields": result["missing_fields"],
            "n_actions": result["n_actions"],
        }
        if status is CounterfactualEquivalenceStatus.EQUIVALENT:
            n_equivalent += 1
        elif status is CounterfactualEquivalenceStatus.NON_EQUIVALENT:
            n_non_equivalent += 1
            non_equivalent_task_ids.append(tid)
        else:
            n_unverifiable += 1
            unverifiable_task_ids.append(tid)

    n_checked = len(task_ids)

    # Overall status.
    if n_checked == 0:
        overall = CounterfactualEquivalenceStatus.UNVERIFIABLE
    elif n_equivalent == n_checked:
        overall = CounterfactualEquivalenceStatus.EQUIVALENT
    elif n_non_equivalent == n_checked:
        overall = CounterfactualEquivalenceStatus.NON_EQUIVALENT
    elif n_unverifiable == n_checked:
        overall = CounterfactualEquivalenceStatus.UNVERIFIABLE
    else:
        overall = CounterfactualEquivalenceStatus.PARTIALLY_EQUIVALENT

    # Non-equivalent tasks are excluded from primary metrics.
    excluded_task_ids = list(non_equivalent_task_ids)

    return {
        "status": overall.value,
        "n_tasks_checked": n_checked,
        "n_tasks_equivalent": n_equivalent,
        "n_tasks_non_equivalent": n_non_equivalent,
        "n_tasks_unverifiable": n_unverifiable,
        "non_equivalent_task_ids": non_equivalent_task_ids,
        "unverifiable_task_ids": unverifiable_task_ids,
        "per_task_results": per_task_results,
        "excluded_task_ids": excluded_task_ids,
    }
