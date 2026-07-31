"""Complete eligible-action matrix validation for v0.4.4.

This module validates the *complete* eligible-action matrix. It does NOT
infer completeness from policy-selected task rows: every task's full set of
eligible actions is audited against the actions that were actually measured,
executed, and verified.

For every task the validator computes:

* ``eligible_actions``    — the full set of actions the task admits.
* ``measured_actions``    — actions with a measured (utility-bearing) outcome.
* ``executed_actions``    — actions whose outcome was actually executed.
* ``verifiable_actions``  — measured actions that passed verification.
* ``missing_actions``     — eligible actions with no measured outcome and no
  predeclared infeasibility.
* ``unavailable_actions`` — eligible actions explicitly and predeclaredly
  infeasible (excused from the completeness requirement).

Primary oracle eligibility requires ``measured_actions == eligible_actions``
unless an action is explicitly and predeclaredly infeasible (in which case it
is recorded in ``unavailable_actions`` and excused).

Only ``COMPLETE`` tasks that are *also* counterfactually equivalent may
enter the primary metrics (oracle utility, oracle-sham headroom, candidate
recovered headroom, high-margin analysis, and primary Gate A bootstrap).
"""
from __future__ import annotations

from typing import Any

from qualification.autolearn_v04.common.schemas import (
    ActionMatrixStatus,
    CounterfactualEquivalenceStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest_eligible_actions(benchmark_manifest: dict, task_id: str) -> list[str]:
    """Return the eligible-action list for a task from the benchmark manifest.

    Resolution order:
      1. Per-task entry under ``tasks`` (dict keyed by task_id or list of
         dicts each carrying ``task_id`` and ``eligible_actions``).
      2. Per-task entry under ``task_actions`` (dict keyed by task_id).
      3. Global ``eligible_actions`` list on the manifest.
    """
    if not isinstance(benchmark_manifest, dict):
        return []

    # 1a. tasks as dict keyed by task_id.
    tasks = benchmark_manifest.get("tasks")
    if isinstance(tasks, dict) and task_id in tasks:
        entry = tasks[task_id]
        if isinstance(entry, dict):
            actions = entry.get("eligible_actions") or entry.get("allowed_actions")
            if isinstance(actions, list):
                return [str(a) for a in actions]

    # 1b. tasks as list of dicts.
    if isinstance(tasks, list):
        for t in tasks:
            if isinstance(t, dict) and str(t.get("task_id")) == task_id:
                actions = t.get("eligible_actions") or t.get("allowed_actions")
                if isinstance(actions, list):
                    return [str(a) for a in actions]

    # 2. task_actions dict.
    task_actions = benchmark_manifest.get("task_actions")
    if isinstance(task_actions, dict) and task_id in task_actions:
        entry = task_actions[task_id]
        if isinstance(entry, dict):
            actions = entry.get("eligible_actions") or entry.get("allowed_actions")
            if isinstance(actions, list):
                return [str(a) for a in actions]
        elif isinstance(entry, list):
            return [str(a) for a in entry]

    # 3. global eligible_actions.
    global_actions = benchmark_manifest.get("eligible_actions")
    if isinstance(global_actions, list):
        return [str(a) for a in global_actions]

    return []


def _manifest_unavailable_actions(benchmark_manifest: dict, task_id: str) -> list[str]:
    """Return the predeclared infeasible (unavailable) actions for a task.

    Looks for ``unavailable_actions`` / ``infeasible_actions`` on a per-task
    entry or globally on the manifest.
    """
    if not isinstance(benchmark_manifest, dict):
        return []

    def _from_entry(entry: Any) -> list[str]:
        if isinstance(entry, dict):
            ua = entry.get("unavailable_actions") or entry.get("infeasible_actions")
            if isinstance(ua, list):
                return [str(a) for a in ua]
        return []

    tasks = benchmark_manifest.get("tasks")
    if isinstance(tasks, dict) and task_id in tasks:
        ua = _from_entry(tasks[task_id])
        if ua:
            return ua
    if isinstance(tasks, list):
        for t in tasks:
            if isinstance(t, dict) and str(t.get("task_id")) == task_id:
                ua = _from_entry(t)
                if ua:
                    return ua

    task_actions = benchmark_manifest.get("task_actions")
    if isinstance(task_actions, dict) and task_id in task_actions:
        ua = _from_entry(task_actions[task_id])
        if ua:
            return ua

    global_ua = benchmark_manifest.get("unavailable_actions") or benchmark_manifest.get(
        "infeasible_actions"
    )
    if isinstance(global_ua, list):
        return [str(a) for a in global_ua]

    return []


def _manifest_task_ids(benchmark_manifest: dict) -> list[str]:
    """Extract the canonical task-id list from the benchmark manifest."""
    if not isinstance(benchmark_manifest, dict):
        return []
    for key in ("task_ids", "test_task_ids", "eval_task_ids"):
        ids = benchmark_manifest.get(key)
        if isinstance(ids, list) and ids:
            return [str(t) for t in ids]
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
    task_actions = benchmark_manifest.get("task_actions")
    if isinstance(task_actions, dict):
        return [str(k) for k in task_actions.keys()]
    return []


def _action_id(outcome: dict) -> str | None:
    """Extract the action id from an outcome record."""
    for key in ("action_id", "action", "eligible_action"):
        v = outcome.get(key)
        if v is not None:
            return str(v)
    return None


def _is_measured(outcome: dict) -> bool:
    """Return True if the outcome represents a measured (utility-bearing) trial."""
    status = outcome.get("status")
    if isinstance(status, str) and status.upper() == "MEASURED":
        return True
    availability = outcome.get("availability")
    if isinstance(availability, str) and availability.lower() == "executed":
        return True
    if outcome.get("utility") is not None:
        return True
    if outcome.get("utility_value") is not None:
        return True
    return False


def _is_executed(outcome: dict) -> bool:
    """Return True if the outcome was actually executed (not simulated/skipped)."""
    availability = outcome.get("availability")
    if isinstance(availability, str) and availability.lower() == "executed":
        return True
    if outcome.get("executed") is True:
        return True
    status = outcome.get("status")
    if isinstance(status, str) and status.upper() == "MEASURED":
        return True
    return False


def _is_verifiable(outcome: dict) -> bool:
    """Return True if the outcome is measured and passed verification."""
    if not _is_measured(outcome):
        return False
    if outcome.get("verifier_passed") is True:
        return True
    if outcome.get("verified") is True:
        return True
    status = outcome.get("status")
    if isinstance(status, str) and status.upper() == "MEASURED":
        # A measured outcome with no explicit verifier flag is treated as
        # verifiable only if it carries a utility value.
        return outcome.get("utility") is not None or outcome.get("utility_value") is not None
    return False


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


# ---------------------------------------------------------------------------
# Per-task audit
# ---------------------------------------------------------------------------


def _audit_task(
    task_id: str,
    outcomes: list[dict],
    benchmark_manifest: dict,
    equivalence_status: str | None,
) -> dict[str, Any]:
    """Audit the eligible-action matrix for a single task."""
    eligible = _manifest_eligible_actions(benchmark_manifest, task_id)
    unavailable = _manifest_unavailable_actions(benchmark_manifest, task_id)

    measured: list[str] = []
    executed: list[str] = []
    verifiable: list[str] = []
    seen_measured: set[str] = set()
    seen_executed: set[str] = set()
    seen_verifiable: set[str] = set()

    for o in outcomes:
        aid = _action_id(o)
        if aid is None:
            continue
        if _is_measured(o) and aid not in seen_measured:
            measured.append(aid)
            seen_measured.add(aid)
        if _is_executed(o) and aid not in seen_executed:
            executed.append(aid)
            seen_executed.add(aid)
        if _is_verifiable(o) and aid not in seen_verifiable:
            verifiable.append(aid)
            seen_verifiable.add(aid)

    eligible_set = set(eligible)
    unavailable_set = set(unavailable)
    measured_set = set(measured)

    # Missing = eligible actions neither measured nor predeclared unavailable.
    missing_set = eligible_set - measured_set - unavailable_set
    missing = sorted(missing_set)

    # Unavailable = eligible actions predeclared infeasible.
    unavailable_resolved = sorted(eligible_set & unavailable_set)

    # Determine status.
    if not eligible_set:
        # Cannot determine the eligible-action set -> unverifiable.
        status = ActionMatrixStatus.UNVERIFIABLE
    elif missing_set:
        status = ActionMatrixStatus.INCOMPLETE
    elif equivalence_status is not None and equivalence_status not in (
        CounterfactualEquivalenceStatus.EQUIVALENT.value,
    ):
        # Measured == eligible but the task is not counterfactually
        # equivalent, so its action matrix is not comparable.
        status = ActionMatrixStatus.NONCOMPARABLE
    else:
        status = ActionMatrixStatus.COMPLETE

    return {
        "status": status,
        "eligible": sorted(eligible_set),
        "measured": sorted(measured_set),
        "executed": sorted(seen_executed),
        "verifiable": sorted(seen_verifiable),
        "missing": missing,
        "unavailable": unavailable_resolved,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_action_matrix(
    counterfactual_outcomes: list[dict],
    benchmark_manifest: dict,
    equivalence_report: dict | None = None,
) -> dict:
    """Validate the complete eligible-action matrix.

    For every task, audits ``eligible_actions`` against ``measured_actions``,
    ``executed_actions``, and ``verifiable_actions``. Completeness is NEVER
    inferred from policy-selected task rows — the full eligible set is taken
    from the benchmark manifest.

    Only ``COMPLETE`` *and* counterfactually equivalent tasks may enter the
    primary metrics. All other tasks are excluded with a recorded reason.

    Parameters
    ----------
    counterfactual_outcomes:
        List of outcome-record dicts.
    benchmark_manifest:
        Benchmark manifest dict carrying the eligible-action matrix.
    equivalence_report:
        Optional output of :func:`validate_counterfactual_equivalence`. When
        provided, tasks that are not counterfactually equivalent are marked
        ``NONCOMPARABLE`` and excluded from primary metrics.

    Returns
    -------
    dict
        See module docstring for the full return schema.
    """
    groups = _group_outcomes_by_task(counterfactual_outcomes)

    # Equivalence lookup.
    eq_per_task: dict[str, str] = {}
    if isinstance(equivalence_report, dict):
        for tid, res in (equivalence_report.get("per_task_results") or {}).items():
            if isinstance(res, dict):
                eq_per_task[str(tid)] = str(res.get("status", ""))

    # Determine the set of tasks to audit.
    manifest_ids = _manifest_task_ids(benchmark_manifest)
    if manifest_ids:
        task_ids = manifest_ids
    else:
        task_ids = sorted(groups.keys())

    per_task_results: dict[str, dict[str, Any]] = {}
    n_complete = 0
    n_incomplete = 0
    n_noncomparable = 0
    n_unverifiable = 0
    excluded_task_ids: list[str] = []
    exclusion_reasons: dict[str, str] = {}

    for tid in task_ids:
        outcomes = groups.get(tid, [])
        eq_status = eq_per_task.get(tid)
        result = _audit_task(tid, outcomes, benchmark_manifest, eq_status)
        status = result["status"]

        per_task_results[tid] = {
            "status": status.value,
            "eligible": result["eligible"],
            "measured": result["measured"],
            "missing": result["missing"],
            "unavailable": result["unavailable"],
        }

        if status is ActionMatrixStatus.COMPLETE:
            n_complete += 1
        elif status is ActionMatrixStatus.INCOMPLETE:
            n_incomplete += 1
        elif status is ActionMatrixStatus.NONCOMPARABLE:
            n_noncomparable += 1
        else:
            n_unverifiable += 1

        # Exclusion logic: only COMPLETE + EQUIVALENT tasks enter primary
        # metrics. Everything else is excluded with a reason.
        is_equivalent = eq_status == CounterfactualEquivalenceStatus.EQUIVALENT.value
        if status is ActionMatrixStatus.COMPLETE and is_equivalent:
            continue
        excluded_task_ids.append(tid)
        if status is ActionMatrixStatus.INCOMPLETE:
            reason = "incomplete_action_matrix"
        elif status is ActionMatrixStatus.NONCOMPARABLE:
            reason = "not_counterfactually_equivalent"
        elif status is ActionMatrixStatus.UNVERIFIABLE:
            reason = "unverifiable_action_matrix"
        elif not is_equivalent:
            reason = "not_counterfactually_equivalent"
        else:
            reason = "excluded"
        exclusion_reasons[tid] = reason

    n_total = len(task_ids)

    # Overall status.
    if n_total == 0:
        overall = ActionMatrixStatus.UNVERIFIABLE
    elif n_complete == n_total:
        overall = ActionMatrixStatus.COMPLETE
    elif n_incomplete == n_total:
        overall = ActionMatrixStatus.INCOMPLETE
    elif n_noncomparable == n_total:
        overall = ActionMatrixStatus.NONCOMPARABLE
    elif n_unverifiable == n_total:
        overall = ActionMatrixStatus.UNVERIFIABLE
    else:
        # Mixed: default to INCOMPLETE if any incomplete, else the dominant
        # non-complete category.
        if n_incomplete > 0:
            overall = ActionMatrixStatus.INCOMPLETE
        elif n_noncomparable > 0:
            overall = ActionMatrixStatus.NONCOMPARABLE
        else:
            overall = ActionMatrixStatus.UNVERIFIABLE

    return {
        "status": overall.value,
        "n_tasks_total": n_total,
        "n_tasks_complete": n_complete,
        "n_tasks_incomplete": n_incomplete,
        "n_tasks_noncomparable": n_noncomparable,
        "n_tasks_unverifiable": n_unverifiable,
        "per_task_results": per_task_results,
        "excluded_task_ids": excluded_task_ids,
        "exclusion_reasons": exclusion_reasons,
    }
