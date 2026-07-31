"""High-margin recovery analysis for v0.4.4.

Measures how much of the *high-margin* headroom the candidate recovers
relative to the sham reference.  A task is "high-margin" when its
oracle-minus-sham headroom meets or exceeds a configurable threshold
(default ``0.15``).  On that subset we compute:

* ``oracle_high_margin_headroom``     — sum of (oracle - sham) over high-margin
                                        tasks; the total recoverable headroom.
* ``candidate_high_margin_recovery``  — sum of (candidate - sham) over the
                                        same tasks; what the candidate gained.
* ``sham_high_margin_recovery``       — sum of (sham - sham) = 0 by definition,
                                        reported for symmetry and to surface
                                        any data inconsistency.
* ``recovery_fraction``               — candidate recovery / oracle headroom.

Status is ``PASS`` when the candidate recovers a positive fraction of the
high-margin headroom, ``FAIL`` when it recovers none or a negative amount,
and ``NOT_RUN`` when there are no high-margin tasks to evaluate.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _per_task_utilities(results: dict, n_tasks: int) -> list[float | None]:
    """Extract per-task utilities from a policy results dict.

    Mirrors the canonical evaluator: a ``per_task`` list of records each
    carrying ``task_id`` and ``utility`` is preferred; otherwise the aggregate
    ``mean_utility`` is broadcast across all tasks.
    """
    per_task = results.get("per_task", [])
    if per_task:
        utils: list[float | None] = []
        for tr in per_task[:n_tasks]:
            u = tr.get("utility")
            if u is not None:
                u = float(u)
            utils.append(u)
        while len(utils) < n_tasks:
            utils.append(None)
        return utils

    mean_util = results.get("mean_utility")
    if mean_util is not None:
        mean_util = float(mean_util)
    return [mean_util] * n_tasks


def _resolve_n_tasks(
    candidate_results: dict,
    oracle_results: dict,
    sham_results: dict,
) -> int:
    """Determine the task count from the supplied results dicts."""
    for rd in (candidate_results, oracle_results, sham_results):
        per_task = rd.get("per_task", [])
        if per_task:
            return len(per_task)
        n = rd.get("n_tasks")
        if isinstance(n, int) and n > 0:
            return n
    return 30


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_high_margin_recovery(
    candidate_results: dict,
    oracle_results: dict,
    sham_results: dict,
    threshold: float = 0.15,
) -> dict:
    """Evaluate high-margin headroom recovery by the candidate.

    Args:
        candidate_results: Candidate policy results dict.
        oracle_results: Oracle policy results dict.
        sham_results: Sham policy results dict.
        threshold: Minimum per-task oracle-minus-sham headroom for a task to
            count as "high-margin" (default 0.15).

    Returns:
        Dict with ``status``, ``high_margin_task_count``,
        ``candidate_high_margin_recovery``, ``sham_high_margin_recovery``,
        ``oracle_high_margin_headroom``, and ``recovery_fraction``.
    """
    n_tasks = _resolve_n_tasks(candidate_results, oracle_results, sham_results)

    cand_utils = _per_task_utilities(candidate_results, n_tasks)
    oracle_utils = _per_task_utilities(oracle_results, n_tasks)
    sham_utils = _per_task_utilities(sham_results, n_tasks)

    high_margin_task_count = 0
    candidate_high_margin_recovery = 0.0
    sham_high_margin_recovery = 0.0
    oracle_high_margin_headroom = 0.0

    for i in range(n_tasks):
        o = oracle_utils[i]
        s = sham_utils[i]
        c = cand_utils[i]
        if o is None or s is None:
            continue
        headroom = float(o) - float(s)
        if headroom < threshold:
            continue

        high_margin_task_count += 1
        oracle_high_margin_headroom += headroom
        if c is not None:
            candidate_high_margin_recovery += float(c) - float(s)
        # By definition sham minus sham is zero; computed for symmetry.
        if s is not None:
            sham_high_margin_recovery += float(s) - float(s)

    if high_margin_task_count == 0:
        status = "NOT_RUN"
        recovery_fraction = 0.0
    else:
        if oracle_high_margin_headroom > 0.0:
            recovery_fraction = (
                candidate_high_margin_recovery / oracle_high_margin_headroom
            )
        else:
            recovery_fraction = 0.0
        status = "PASS" if recovery_fraction > 0.0 else "FAIL"

    return {
        "status": status,
        "high_margin_task_count": high_margin_task_count,
        "candidate_high_margin_recovery": candidate_high_margin_recovery,
        "sham_high_margin_recovery": sham_high_margin_recovery,
        "oracle_high_margin_headroom": oracle_high_margin_headroom,
        "recovery_fraction": recovery_fraction,
        "threshold": threshold,
    }
