"""Class-balance analysis for v0.4.4 evidence-complete analysis.

Audits whether the candidate policy's action distribution exhibits class
imbalance that could bias the multinomial logistic regression learner
toward the majority action.

The analysis examines the action distribution recorded in the candidate
results and the eligible-action set declared in the benchmark manifest.
It computes:

  - the per-action distribution (counts and probabilities)
  - the minority and majority class counts
  - the imbalance ratio (majority / minority)
  - a status classification and a recommendation

Status thresholds:
  - **BALANCED**: imbalance ratio < 3.0
  - **IMBALANCED**: imbalance ratio in [3.0, 10.0)
  - **SEVERELY_IMBALANCED**: imbalance ratio >= 10.0

The output of :func:`analyze_class_balance` is a JSON-serializable dict
suitable for writing to ``class_balance_results.json``.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Imbalance ratio threshold for IMBALANCED classification.
IMBALANCED_RATIO_THRESHOLD: float = 3.0

#: Imbalance ratio threshold for SEVERELY_IMBALANCED classification.
SEVERELY_IMBALANCED_RATIO_THRESHOLD: float = 10.0

#: Default number of tasks to consider.
DEFAULT_N_TASKS: int = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_action_distribution(
    candidate_results: dict[str, Any],
) -> dict[str, float]:
    """Extract the action distribution from candidate results.

    Falls back to deriving it from ``per_task`` records when the
    precomputed ``action_distribution`` field is absent.
    """
    dist = candidate_results.get("action_distribution")
    if isinstance(dist, dict):
        return {str(k): float(v) for k, v in dist.items()}

    per_task = candidate_results.get("per_task", [])
    if per_task:
        actions = [
            tr.get("selected_action", tr.get("action", "unknown"))
            for tr in per_task
        ]
        counts = Counter(actions)
        n = len(actions)
        return {a: counts.get(a, 0) / n for a in counts} if n else {}
    return {}


def _extract_action_counts(
    candidate_results: dict[str, Any],
) -> dict[str, int]:
    """Extract raw action counts from candidate results.

    Falls back to deriving them from ``per_task`` records.
    """
    per_task = candidate_results.get("per_task", [])
    if per_task:
        actions = [
            tr.get("selected_action", tr.get("action", "unknown"))
            for tr in per_task
        ]
        return dict(Counter(actions))

    # If only the distribution is available, estimate counts from
    # n_tasks.
    dist = candidate_results.get("action_distribution", {})
    n_tasks = candidate_results.get("n_tasks", 0)
    if isinstance(dist, dict) and n_tasks:
        return {
            str(k): int(round(float(v) * n_tasks))
            for k, v in dist.items()
        }
    return {}


def _get_eligible_actions(
    benchmark_manifest: dict[str, Any],
) -> list[str]:
    """Extract the eligible-action set from the benchmark manifest."""
    actions = benchmark_manifest.get("eligible_actions")
    if isinstance(actions, list):
        return [str(a) for a in actions]
    # Fall back to per-task eligible actions.
    tasks = benchmark_manifest.get("tasks", {})
    all_actions: set[str] = set()
    if isinstance(tasks, dict):
        for task_info in tasks.values():
            if isinstance(task_info, dict):
                ea = task_info.get("eligible_actions", [])
                if isinstance(ea, list):
                    all_actions.update(str(a) for a in ea)
    elif isinstance(tasks, list):
        for task_info in tasks:
            if isinstance(task_info, dict):
                ea = task_info.get("eligible_actions", [])
                if isinstance(ea, list):
                    all_actions.update(str(a) for a in ea)
    return sorted(all_actions)


def _classify_imbalance(imbalance_ratio: float) -> str:
    """Classify the imbalance ratio into a status string."""
    if imbalance_ratio >= SEVERELY_IMBALANCED_RATIO_THRESHOLD:
        return "SEVERELY_IMBALANCED"
    if imbalance_ratio >= IMBALANCED_RATIO_THRESHOLD:
        return "IMBALANCED"
    return "BALANCED"


def _build_recommendation(
    status: str,
    minority_action: str | None,
    majority_action: str | None,
    imbalance_ratio: float,
    minority_count: int,
    majority_count: int,
) -> str:
    """Build a human-readable recommendation based on the imbalance status."""
    if status == "BALANCED":
        return (
            f"Action distribution is balanced (imbalance ratio "
            f"{imbalance_ratio:.2f}). No class-weighting intervention "
            f"is required."
        )
    if status == "IMBALANCED":
        return (
            f"Action distribution is imbalanced (imbalance ratio "
            f"{imbalance_ratio:.2f}). The majority action "
            f"'{majority_action}' ({majority_count} selections) "
            f"dominates the minority action '{minority_action}' "
            f"({minority_count} selections). Consider applying "
            f"inverse-frequency class weighting or effective-number "
            f"weighting to mitigate learner collapse toward the "
            f"majority action."
        )
    # SEVERELY_IMBALANCED
    return (
        f"Action distribution is severely imbalanced (imbalance ratio "
        f"{imbalance_ratio:.2f}). The majority action "
        f"'{majority_action}' ({majority_count} selections) heavily "
        f"dominates the minority action '{minority_action}' "
        f"({minority_count} selections). The learner is at high risk "
        f"of route collapse. Apply class-weighting (inverse-frequency "
        f"or effective-number) and verify that the predicted action "
        f"entropy on validation exceeds the collapse threshold. If "
        f"weighting does not resolve the collapse, collect additional "
        f"evidence for under-represented actions."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_class_balance(
    candidate_results: dict,
    benchmark_manifest: dict,
    n_tasks: int = 30,
) -> dict:
    """Analyse class balance in the candidate action distribution.

    Handles class imbalance explicitly by computing the action
    distribution, identifying the minority and majority classes, and
    classifying the severity of the imbalance.

    Args:
        candidate_results: The candidate policy results dict (with
            ``action_distribution``, ``per_task``, ``n_tasks``, etc.).
        benchmark_manifest: The benchmark manifest dict (with
            ``eligible_actions`` and/or per-task ``tasks``).
        n_tasks: The expected number of tasks (default 30).  Used as a
            fallback when ``n_tasks`` is not present in the results.

    Returns:
        A JSON-serializable dict with:
          - status: "BALANCED" | "IMBALANCED" | "SEVERELY_IMBALANCED"
          - action_distribution: dict
          - minority_class_count: int
          - majority_class_count: int
          - imbalance_ratio: float
          - recommendation: str
    """
    # --- Determine the effective task count --------------------------------
    effective_n_tasks = candidate_results.get("n_tasks", n_tasks)

    # --- Extract the action distribution and counts ------------------------
    action_dist = _extract_action_distribution(candidate_results)
    action_counts = _extract_action_counts(candidate_results)

    # If we have no counts but have a distribution, estimate counts.
    if not action_counts and action_dist:
        action_counts = {
            a: int(round(p * effective_n_tasks))
            for a, p in action_dist.items()
        }

    # Ensure all eligible actions are represented (even with zero count).
    eligible_actions = _get_eligible_actions(benchmark_manifest)
    if eligible_actions:
        for a in eligible_actions:
            if a not in action_counts:
                action_counts[a] = 0
            if a not in action_dist:
                action_dist[a] = 0.0

    # --- Compute minority / majority ---------------------------------------
    if action_counts:
        majority_action = max(action_counts, key=action_counts.get)
        minority_action = min(action_counts, key=action_counts.get)
        majority_count = int(action_counts[majority_action])
        minority_count = int(action_counts[minority_action])
    else:
        majority_action = None
        minority_action = None
        majority_count = 0
        minority_count = 0

    # --- Imbalance ratio ---------------------------------------------------
    # Guard against division by zero.  When the minority count is zero
    # the imbalance is effectively infinite — we clamp to a large value.
    if minority_count > 0:
        imbalance_ratio = float(majority_count) / float(minority_count)
    elif majority_count > 0:
        # Minority has zero samples — severe imbalance.
        imbalance_ratio = float("inf")
    else:
        # No data at all.
        imbalance_ratio = 1.0

    # For classification, treat inf as severely imbalanced.
    classification_ratio = (
        imbalance_ratio
        if math.isfinite(imbalance_ratio)
        else SEVERELY_IMBALANCED_RATIO_THRESHOLD
    )

    status = _classify_imbalance(classification_ratio)

    recommendation = _build_recommendation(
        status,
        minority_action,
        majority_action,
        imbalance_ratio if math.isfinite(imbalance_ratio) else float(
            SEVERELY_IMBALANCED_RATIO_THRESHOLD
        ),
        minority_count,
        majority_count,
    )

    return {
        "status": status,
        "action_distribution": action_dist,
        "action_counts": action_counts,
        "minority_class": minority_action,
        "majority_class": majority_action,
        "minority_class_count": minority_count,
        "majority_class_count": majority_count,
        "imbalance_ratio": (
            float(imbalance_ratio) if math.isfinite(imbalance_ratio) else float("inf")
        ),
        "recommendation": recommendation,
        "n_tasks": effective_n_tasks,
        "n_eligible_actions": len(eligible_actions) if eligible_actions else len(action_counts),
    }
