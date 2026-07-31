"""Soft-utility target computation for v0.4.4 evidence-complete analysis.

Computes soft-utility training targets for multinomial logistic
regression learners.  Instead of using a hard one-hot label at the
argmax (best) action, the soft-utility target converts the measured
counterfactual utilities into a probability distribution via a
temperature-scaled softmax::

    q_i(a) = softmax(U_{i,a} / tau)

This allows the learner to respect near-ties and graded utility
differences rather than collapsing to a single action.

The soft targets are derived from the **oracle** counterfactual
outcomes (the measured utilities for each eligible action on each task).
The candidate results are used to determine which actions were actually
selected and to provide task-level context.

**Important**: Soft targets are computed on the training/experience
split only.  D_test is never used for target construction.

The output of :func:`compute_soft_utility_targets` is a JSON-serializable
dict suitable for writing to ``soft_utility_results.json``.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default temperature for the soft-utility softmax.
DEFAULT_TAU: float = 0.1

#: Default number of tasks to consider.
DEFAULT_N_TASKS: int = 30

#: Minimum utility value used when an action's utility is unavailable.
#: Using 0.0 keeps unavailable actions neutral in the softmax.
UNAVAILABLE_UTILITY_FILL: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _extract_per_task_utilities(
    results_dict: dict[str, Any],
) -> dict[str, dict[str, float | None]]:
    """Extract per-task per-action utilities from a results dict.

    Returns a mapping ``task_id -> {action -> utility}``.

    The results dict may carry:
      - ``per_task``: a list of records with ``task_id``, ``action``/
        ``selected_action``, and ``utility``.
      - ``counterfactual_outcomes``: a list of outcome records with
        ``task_id``, ``action``, and ``utility``.
    """
    per_task_utils: dict[str, dict[str, float | None]] = {}

    # Try counterfactual_outcomes first (richest source).
    outcomes = results_dict.get("counterfactual_outcomes", [])
    if outcomes:
        for rec in outcomes:
            tid = str(rec.get("task_id", ""))
            action = str(rec.get("action", ""))
            if not tid or not action:
                continue
            util = rec.get("utility")
            if tid not in per_task_utils:
                per_task_utils[tid] = {}
            per_task_utils[tid][action] = (
                float(util) if util is not None else None
            )

    # Fall back to per_task records.
    if not per_task_utils:
        per_task = results_dict.get("per_task", [])
        for tr in per_task:
            tid = str(tr.get("task_id", ""))
            action = str(
                tr.get("selected_action", tr.get("action", ""))
            )
            if not tid or not action:
                continue
            util = tr.get("utility")
            if tid not in per_task_utils:
                per_task_utils[tid] = {}
            per_task_utils[tid][action] = (
                float(util) if util is not None else None
            )

    return per_task_utils


def _extract_eligible_actions(
    results_dict: dict[str, Any],
) -> list[str]:
    """Extract the sorted list of eligible actions from a results dict."""
    # Try the top-level field.
    actions = results_dict.get("eligible_actions")
    if isinstance(actions, list) and actions:
        return sorted(str(a) for a in actions)

    # Derive from per-task utilities.
    per_task_utils = _extract_per_task_utilities(results_dict)
    all_actions: set[str] = set()
    for action_map in per_task_utils.values():
        all_actions.update(action_map.keys())
    return sorted(all_actions)


def _build_utility_matrix(
    per_task_utils: dict[str, dict[str, float | None]],
    actions: list[str],
    task_ids: list[str],
) -> np.ndarray:
    """Build a utility matrix [n_tasks, n_actions] from per-task utilities.

    Unavailable utilities are filled with ``UNAVAILABLE_UTILITY_FILL``.
    """
    action_idx = {a: i for i, a in enumerate(actions)}
    U = np.full((len(task_ids), len(actions)), UNAVAILABLE_UTILITY_FILL, dtype=float)
    for i, tid in enumerate(task_ids):
        action_map = per_task_utils.get(tid, {})
        for action, util in action_map.items():
            if action in action_idx and util is not None:
                U[i, action_idx[action]] = float(util)
    return U


def _compute_soft_targets(
    utility_matrix: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Compute soft-utility targets: q_i(a) = softmax(U_{i,a} / tau)."""
    scaled = utility_matrix / tau
    return _softmax(scaled)


def _target_distribution(
    soft_targets: np.ndarray,
    actions: list[str],
) -> dict[str, float]:
    """Compute the aggregate target distribution across all tasks.

    This is the mean of the per-task soft targets, representing the
    expected proportion of probability mass assigned to each action.
    """
    if soft_targets.size == 0:
        return {a: 0.0 for a in actions}
    mean_probs = soft_targets.mean(axis=0)
    return {a: float(mean_probs[i]) for i, a in enumerate(actions)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_soft_utility_targets(
    candidate_results: dict,
    oracle_results: dict,
    n_tasks: int = 30,
) -> dict:
    """Compute soft-utility targets for logistic regression learners.

    Converts the oracle's measured counterfactual utilities into
    temperature-scaled softmax targets for each task.  These targets
    can be used to train a multinomial logistic regression that
    respects graded utility differences and near-ties, rather than
    collapsing to a single hard label.

    Args:
        candidate_results: The candidate policy results dict.  Used to
            determine the task set and eligible actions.
        oracle_results: The oracle policy results dict containing the
            measured counterfactual utilities per (task, action).
        n_tasks: The expected number of tasks (default 30).  Used as a
            fallback when the task count is not derivable from the
            results.

    Returns:
        A JSON-serializable dict with:
          - status: "PASS" | "NOT_RUN"
          - soft_targets: list of {task_id, action, soft_target}
          - target_distribution: dict
    """
    # --- Extract per-task utilities from the oracle -----------------------
    oracle_per_task = _extract_per_task_utilities(oracle_results)

    if not oracle_per_task:
        # No oracle utilities available — cannot compute soft targets.
        return {
            "status": "NOT_RUN",
            "soft_targets": [],
            "target_distribution": {},
            "n_tasks": 0,
            "tau": DEFAULT_TAU,
            "reason": "No per-task oracle utilities available.",
        }

    # --- Determine the task set and eligible actions -----------------------
    # Prefer the candidate's task set; fall back to the oracle's.
    candidate_per_task = _extract_per_task_utilities(candidate_results)
    task_ids = list(candidate_per_task.keys()) if candidate_per_task else list(oracle_per_task.keys())

    # Limit to n_tasks if more are available.
    if len(task_ids) > n_tasks:
        task_ids = task_ids[:n_tasks]

    # Determine eligible actions from both sources.
    actions = _extract_eligible_actions(oracle_results)
    if not actions:
        actions = _extract_eligible_actions(candidate_results)
    if not actions:
        return {
            "status": "NOT_RUN",
            "soft_targets": [],
            "target_distribution": {},
            "n_tasks": len(task_ids),
            "tau": DEFAULT_TAU,
            "reason": "No eligible actions could be determined.",
        }

    # --- Build the utility matrix and compute soft targets -----------------
    U = _build_utility_matrix(oracle_per_task, actions, task_ids)
    soft = _compute_soft_targets(U, DEFAULT_TAU)

    # --- Assemble the per-task soft target list ----------------------------
    soft_targets: list[dict[str, Any]] = []
    for i, tid in enumerate(task_ids):
        for j, action in enumerate(actions):
            soft_targets.append({
                "task_id": tid,
                "action": action,
                "soft_target": float(soft[i, j]),
            })

    # --- Aggregate target distribution -------------------------------------
    target_dist = _target_distribution(soft, actions)

    return {
        "status": "PASS",
        "soft_targets": soft_targets,
        "target_distribution": target_dist,
        "n_tasks": len(task_ids),
        "n_actions": len(actions),
        "tau": DEFAULT_TAU,
        "actions": actions,
    }
