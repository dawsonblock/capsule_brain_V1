"""Action confusion and utility-weighted confusion analyses.

Produces standard action confusion matrices (true best action vs selected
action) and utility-weighted confusion matrices (sum of lost utility per
cell) for every policy on the control ladder.

For each policy we report:

* **Standard confusion matrix** -- rows are the oracle best action, columns
  are the action the policy actually selected.  The diagonal counts correct
  selections; off-diagonal cells count substitutions.

* **Utility-weighted confusion matrix** -- each cell holds the total lost
  utility (regret) caused by selecting the predicted action instead of the
  oracle action::

      regret_i = U_i(best) - U_i(selected)

  Only non-negative regret is accumulated (selecting an action that happens
  to outperform the measured best contributes zero).

Regret is further decomposed by true action, selected action, task family,
and utility-margin bucket.  Summary statistics include the mean, median,
95th percentile, and a catastrophic-regret count (regret > 5.0).
"""
from __future__ import annotations

import numpy as np
from typing import Any

from .data import LoadedRun
from .policy_controls import (
    evaluate_policy_on_tasks,
    make_candidate_selector,
    make_baseline_selector,
    make_majority_selector,
    make_frequency_selector,
    make_sham_global_selector,
    make_sham_family_selector,
    make_sham_margin_selector,
    make_sham_feature_selector,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Regret values above this threshold are flagged as "catastrophic".
CATASTROPHIC_REGRET_THRESHOLD: float = 5.0

#: Margin-bucket upper edges (exclusive except for the last which is +inf).
_MARGIN_EDGES: list[float] = [0.5, 1.0, 2.0, 5.0, np.inf]

#: Human-readable labels for each margin bucket.
_MARGIN_LABELS: list[str] = [
    "tight(<0.5)",
    "small(0.5-1)",
    "medium(1-2)",
    "large(2-5)",
    "huge(>=5)",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _margin_bucket(margin: float) -> str:
    """Classify a utility margin into a bucket label.

    Parameters
    ----------
    margin:
        Difference between the best and second-best measured utility
        for a task.

    Returns
    -------
    str
        Bucket label from :data:`_MARGIN_LABELS`.
    """
    for i, edge in enumerate(_MARGIN_EDGES):
        if margin < edge:
            return _MARGIN_LABELS[i]
    return _MARGIN_LABELS[-1]


def _compute_task_margins(run: LoadedRun, task_ids: list[str]) -> dict[str, float]:
    """Compute the utility margin (best - second best) for each task.

    Tasks with fewer than two executed outcomes receive a margin of 0.
    """
    margins: dict[str, float] = {}
    for tid in task_ids:
        outcomes = run.outcomes_by_task.get(tid, [])
        executed = [
            o for o in outcomes
            if o.availability == "executed" and o.utility is not None
        ]
        if len(executed) < 2:
            margins[tid] = 0.0
            continue
        utils = sorted((o.utility for o in executed), reverse=True)
        margins[tid] = float(utils[0] - utils[1])
    return margins


def _regret_stats(regrets: np.ndarray) -> dict[str, Any]:
    """Compute summary statistics for an array of regret values."""
    n = len(regrets)
    if n == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "catastrophic_count": 0,
            "catastrophic_rate": 0.0,
            "n": 0,
        }
    return {
        "mean": float(np.mean(regrets)),
        "median": float(np.median(regrets)),
        "p95": float(np.percentile(regrets, 95)),
        "catastrophic_count": int(np.sum(regrets > CATASTROPHIC_REGRET_THRESHOLD)),
        "catastrophic_rate": float(np.mean(regrets > CATASTROPHIC_REGRET_THRESHOLD)),
        "n": n,
    }


def _group_regrets(
    per_task: list[dict[str, Any]],
    regrets: np.ndarray,
    key: str,
) -> dict[str, dict[str, Any]]:
    """Group regrets by a per-task attribute and compute stats per group.

    Parameters
    ----------
    per_task:
        List of per-task result dicts (from
        :func:`evaluate_policy_on_tasks`).
    regrets:
        Per-task regret values, aligned with *per_task*.
    key:
        Attribute name to group by (e.g. ``"best_action"``,
        ``"selected_action"``, ``"family"``).

    Returns
    -------
    dict
        Mapping ``{group_value: regret_stats}``.
    """
    groups: dict[str, list[float]] = {}
    for pt, reg in zip(per_task, regrets):
        k = str(pt.get(key, "unknown"))
        groups.setdefault(k, []).append(float(reg))

    result: dict[str, dict[str, Any]] = {}
    for k, vals in groups.items():
        result[k] = _regret_stats(np.array(vals))
    return result


def _margin_bucket_regrets(
    per_task: list[dict[str, Any]],
    regrets: np.ndarray,
    margins: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Group regrets by utility-margin bucket and compute stats per bucket."""
    groups: dict[str, list[float]] = {}
    for pt, reg in zip(per_task, regrets):
        tid = pt["task_id"]
        bucket = _margin_bucket(margins.get(tid, 0.0))
        groups.setdefault(bucket, []).append(float(reg))

    result: dict[str, dict[str, Any]] = {}
    for k, vals in groups.items():
        result[k] = _regret_stats(np.array(vals))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_confusion_analysis(
    run: LoadedRun,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compute action confusion and utility-weighted confusion analyses.

    Evaluates the candidate, baseline, majority, frequency, and all four
    sham variants on the given tasks and builds:

    1. **Standard confusion matrices** -- counts of (true best action,
       selected action) pairs.
    2. **Utility-weighted confusion matrices** -- sum of regret per cell,
       plus regret decomposed by true action, selected action, family,
       and margin bucket.

    Parameters
    ----------
    run:
        Loaded run data.
    task_ids:
        Tasks to evaluate on.  Defaults to the test split.

    Returns
    -------
    dict
        Keys:

        * ``"candidate_confusion_matrix"`` -- content for
          ``candidate_confusion_matrix.json``.
        * ``"utility_weighted_confusion_matrix"`` -- content for
          ``utility_weighted_confusion_matrix.json``.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    actions = run.all_actions
    action_idx = {a: i for i, a in enumerate(actions)}
    n_actions = len(actions)

    # Pre-compute utility margins for margin-bucket regret analysis.
    margins = _compute_task_margins(run, task_ids)

    # -- Build the policy ladder -------------------------------------------
    policies: dict[str, Any] = {
        "P_candidate": make_candidate_selector(run),
        "P_baseline": make_baseline_selector(run),
        "P_majority": make_majority_selector(run),
        "P_frequency": make_frequency_selector(run, seed=42),
        "P_sham_global": make_sham_global_selector(run, seed=12345),
        "P_sham_family": make_sham_family_selector(run, seed=12345),
        "P_sham_margin": make_sham_margin_selector(run, seed=12345),
        "P_sham_feature": make_sham_feature_selector(run, seed=12345),
    }

    # -- Output containers --------------------------------------------------
    confusion_payload: dict[str, Any] = {
        "actions": actions,
        "n_tasks": len(task_ids),
        "description": (
            "Standard action confusion matrices. Rows = true best action, "
            "columns = selected action. Diagonal = correct selections."
        ),
        "policies": {},
    }

    weighted_payload: dict[str, Any] = {
        "actions": actions,
        "n_tasks": len(task_ids),
        "regret_threshold": CATASTROPHIC_REGRET_THRESHOLD,
        "margin_buckets": _MARGIN_LABELS,
        "description": (
            "Utility-weighted confusion matrices. Each cell contains the "
            "sum of lost utility (regret) from selecting the predicted "
            "action instead of the oracle action. "
            "regret_i = U_i(best) - U_i(selected)."
        ),
        "policies": {},
    }

    # -- Evaluate each policy -----------------------------------------------
    for name, selector in policies.items():
        res = evaluate_policy_on_tasks(run, task_ids, selector, name)
        per_task = res["per_task"]

        # --- Standard confusion matrix ---
        confusion = np.zeros((n_actions, n_actions), dtype=int)
        for pt in per_task:
            ba = pt["best_action"]
            sa = pt["selected_action"]
            if ba in action_idx and sa in action_idx:
                confusion[action_idx[ba], action_idx[sa]] += 1

        diagonal = int(np.trace(confusion))
        total = int(confusion.sum())
        accuracy = diagonal / total if total > 0 else 0.0

        confusion_payload["policies"][name] = {
            "matrix": confusion.tolist(),
            "diagonal_count": diagonal,
            "off_diagonal_count": total - diagonal,
            "total": total,
            "accuracy": accuracy,
        }

        # --- Utility-weighted confusion matrix ---
        weighted = np.zeros((n_actions, n_actions))
        regrets = np.array([
            pt["oracle_utility"] - pt["utility"]
            for pt in per_task
        ])

        for pt, reg in zip(per_task, regrets):
            ba = pt["best_action"]
            sa = pt["selected_action"]
            if ba in action_idx and sa in action_idx:
                # Only accumulate non-negative regret (clipped at zero).
                weighted[action_idx[ba], action_idx[sa]] += max(0.0, float(reg))

        weighted_payload["policies"][name] = {
            "matrix": weighted.tolist(),
            "total_regret": float(weighted.sum()),
            "regret_stats": _regret_stats(regrets),
            "regret_by_true_action": _group_regrets(per_task, regrets, "best_action"),
            "regret_by_selected_action": _group_regrets(per_task, regrets, "selected_action"),
            "regret_by_family": _group_regrets(per_task, regrets, "family"),
            "regret_by_margin_bucket": _margin_bucket_regrets(per_task, regrets, margins),
        }

    return {
        "candidate_confusion_matrix": confusion_payload,
        "utility_weighted_confusion_matrix": weighted_payload,
    }
