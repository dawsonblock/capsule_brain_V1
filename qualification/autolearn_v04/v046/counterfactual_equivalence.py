"""Counterfactual-equivalence validation for v0.4.6.

Repairs the counterfactual-equivalence check so that empty strings are
**not** treated as matching digests.  For each task, all actions sharing
the same initial state must have identical mandatory digests.  An empty
or missing digest makes the task UNVERIFIABLE rather than silently
EQUIVALENT.

Mandatory digests (shared initial state across actions for a task):
    prompt_digest, setup_digest, hidden_setup_digest,
    environment_snapshot_digest, memory_state_digest,
    tool_state_digest, workflow_state_digest,
    capability_permissions_digest, timeout_config_digest,
    generation_config_digest, utility_config_digest,
    model_revision, tokenizer_revision, verifier_version

Per-task logic:
    - Any mandatory digest missing/empty → UNVERIFIABLE
    - Mandatory digests differ across actions → NON_EQUIVALENT
    - All mandatory digests nonempty and match → EQUIVALENT

Final report:
    PASS    — all primary tasks EQUIVALENT
    FAIL    — one or more primary tasks NON_EQUIVALENT
    BLOCKED — one or more required task states UNVERIFIABLE
"""
from __future__ import annotations

from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.common.audit_result import (
    AuditResult,
    make_pass,
    make_fail,
    make_blocked,
    aggregate_status,
)

# Digests that must be non-empty and identical across all actions for a task.
MANDATORY_DIGESTS: tuple[str, ...] = (
    "prompt_digest",
    "setup_digest",
    "hidden_setup_digest",
    "environment_snapshot_digest",
    "memory_state_digest",
    "tool_state_digest",
    "workflow_state_digest",
    "capability_permissions_digest",
    "timeout_config_digest",
    "generation_config_digest",
    "utility_config_digest",
    "model_revision",
    "tokenizer_revision",
    "verifier_version",
)


def _is_empty(value: Any) -> bool:
    """Return True if *value* is None or an empty string."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _classify_task(rows: list[dict]) -> tuple[str, list[str]]:
    """Classify a single task's counterfactual equivalence.

    Returns (status_string, reasons).
    status_string is one of "EQUIVALENT", "NON_EQUIVALENT", "UNVERIFIABLE".
    """
    reasons: list[str] = []

    # Check for missing/empty mandatory digests.
    for field in MANDATORY_DIGESTS:
        for idx, row in enumerate(rows):
            val = row.get(field)
            if _is_empty(val):
                reasons.append(
                    f"task {rows[0].get('task_id', '?')}: "
                    f"mandatory digest '{field}' is missing/empty in action "
                    f"'{row.get('eligible_action', row.get('executed_action', '?'))}'"
                )
                return "UNVERIFIABLE", reasons

    # All mandatory digests are non-empty — check they match across actions.
    for field in MANDATORY_DIGESTS:
        values = set()
        for row in rows:
            val = row.get(field)
            values.add(str(val))
        if len(values) > 1:
            reasons.append(
                f"task {rows[0].get('task_id', '?')}: "
                f"digest '{field}' differs across actions: {sorted(values)}"
            )
            return "NON_EQUIVALENT", reasons

    return "EQUIVALENT", reasons


def validate_counterfactual_equivalence(
    counterfactual_outcomes: list[dict],
    split_manifest: dict,
) -> dict:
    """Validate counterfactual equivalence across actions per task.

    Parameters
    ----------
    counterfactual_outcomes:
        List of per-action outcome dicts.  Each must carry ``task_id``,
        ``eligible_action`` (or ``executed_action``), and the mandatory
        digest fields.
    split_manifest:
        The split manifest dict.  Used to determine which tasks are
        "primary" (experience + validation splits) and which are
        excluded (safety, ood).

    Returns
    -------
    dict
        status, n_tasks_checked, n_equivalent, n_non_equivalent,
        n_unverifiable, excluded_task_ids, unverifiable_task_ids,
        reason, checks
    """
    checks: list[dict[str, Any]] = []

    # --- BLOCKED: no counterfactual outcomes ---
    if not counterfactual_outcomes:
        return {
            "status": AuditStatus.BLOCKED.value,
            "n_tasks_checked": 0,
            "n_equivalent": 0,
            "n_non_equivalent": 0,
            "n_unverifiable": 0,
            "excluded_task_ids": [],
            "unverifiable_task_ids": [],
            "reason": "no counterfactual outcomes provided",
            "checks": [],
        }

    # --- Determine primary vs excluded splits ---
    # Primary splits: experience, validation, test.
    # Excluded splits: safety, ood.
    excluded_splits = {"safety", "ood"}
    excluded_task_ids: list[str] = []

    splits = {}
    if isinstance(split_manifest, dict):
        splits = split_manifest.get("splits", {})
    for split_name, split_info in splits.items():
        if split_name in excluded_splits:
            tids = split_info.get("task_ids", []) if isinstance(split_info, dict) else []
            excluded_task_ids.extend(tids)

    # Also exclude any task whose own split field is in excluded_splits.
    for row in counterfactual_outcomes:
        row_split = row.get("split", "")
        if row_split in excluded_splits:
            tid = row.get("task_id")
            if tid and tid not in excluded_task_ids:
                excluded_task_ids.append(tid)

    excluded_set = set(excluded_task_ids)

    # --- Group rows by task_id ---
    by_task: dict[str, list[dict]] = {}
    for row in counterfactual_outcomes:
        tid = row.get("task_id")
        if tid is None:
            continue
        by_task.setdefault(tid, []).append(row)

    # --- Classify each task ---
    task_statuses: dict[str, str] = {}
    unverifiable_task_ids: list[str] = []
    n_equivalent = 0
    n_non_equivalent = 0
    n_unverifiable = 0

    for tid in sorted(by_task.keys()):
        rows = by_task[tid]
        status, reasons = _classify_task(rows)
        task_statuses[tid] = status

        check_entry: dict[str, Any] = {
            "task_id": tid,
            "n_actions": len(rows),
            "status": status,
        }
        if reasons:
            check_entry["reasons"] = reasons
        checks.append(check_entry)

        if tid in excluded_set:
            # Excluded tasks don't count toward the final verdict.
            continue

        if status == "EQUIVALENT":
            n_equivalent += 1
        elif status == "NON_EQUIVALENT":
            n_non_equivalent += 1
        elif status == "UNVERIFIABLE":
            n_unverifiable += 1
            unverifiable_task_ids.append(tid)

    n_tasks_checked = n_equivalent + n_non_equivalent + n_unverifiable

    # --- Determine final status ---
    # BLOCKED takes priority over FAIL (can't verify if evidence is missing).
    if n_unverifiable > 0:
        final_status = AuditStatus.BLOCKED
        reason = (
            f"{n_unverifiable} primary task(s) UNVERIFIABLE due to "
            f"missing/empty mandatory digests"
        )
    elif n_non_equivalent > 0:
        final_status = AuditStatus.FAIL
        reason = (
            f"{n_non_equivalent} primary task(s) NON_EQUIVALENT — "
            f"mandatory digests differ across actions"
        )
    else:
        final_status = AuditStatus.PASS
        reason = (
            f"all {n_equivalent} primary task(s) EQUIVALENT — "
            f"mandatory digests match across actions"
        )

    return {
        "status": final_status.value,
        "n_tasks_checked": n_tasks_checked,
        "n_equivalent": n_equivalent,
        "n_non_equivalent": n_non_equivalent,
        "n_unverifiable": n_unverifiable,
        "excluded_task_ids": sorted(excluded_set),
        "unverifiable_task_ids": unverifiable_task_ids,
        "reason": reason,
        "checks": checks,
    }
