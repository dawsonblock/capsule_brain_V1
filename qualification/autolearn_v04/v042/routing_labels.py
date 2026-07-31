"""Conditional routing-value label classification.

Classifies every task into one of five routing-label statuses based on
the structure of its measured counterfactual utilities:

  CLEAR_BEST         — the best action's margin over the runner-up is
                       at least ``clear_best_threshold``.
  AMBIGUOUS          — the top two actions are within ``tie_band`` of
                       each other, so the best action is not
                       identifiable from measured utilities alone.
  NO_USABLE_ACTION   — every eligible action is either unverifiable or
                       has utility below ``utility_floor``.
  SAFETY_CONTROLLED  — the task belongs to the safety split or carries
                       an adversarial ``risk_class``; routing decisions
                       are deferred to the safety controller.
  INCOMPLETE         — counterfactual evidence is missing (no executed
                       outcomes for the task).

The primary router is trained only on CLEAR_BEST tasks.  AMBIGUOUS
tasks may optionally be included using soft utility targets.

A benchmark dominated by AMBIGUOUS tasks has weak routing
identifiability — the best action cannot be reliably distinguished
from near-ties, so a learned router has limited signal to exploit.

The output of :func:`classify_routing_labels` is the content for
``routing_label_status.json``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .data import LoadedRun


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Adversarial risk classes that trigger SAFETY_CONTROLLED.
ADVERSARIAL_RISK_CLASSES: set[str] = {"adversarial", "adversarial_unverified"}

#: Splits analysed for label-status proportions.
ANALYZED_SPLITS: list[str] = ["experience", "validation", "test", "ood", "safety"]

#: The five possible routing-label statuses, in priority order.
LABEL_STATUSES: list[str] = [
    "SAFETY_CONTROLLED",
    "INCOMPLETE",
    "NO_USABLE_ACTION",
    "AMBIGUOUS",
    "CLEAR_BEST",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sorted_utility_pairs(run: LoadedRun, task_id: str) -> list[tuple[str, float, str]]:
    """Return executed (action, utility, verification) triples sorted by utility desc.

    Only outcomes with ``availability == "executed"`` and a non-None
    utility are considered.
    """
    outcomes = run.outcomes_by_task.get(task_id, [])
    pairs = [
        (o.action_id, o.utility, o.verification)
        for o in outcomes
        if o.availability == "executed" and o.utility is not None
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _is_safety_controlled(run: LoadedRun, task_id: str) -> bool:
    """Whether a task is safety-controlled (safety split or adversarial risk)."""
    task = run.tasks.get(task_id)
    if task is None:
        return False
    if task.split == "safety":
        return True
    if task.risk_class and task.risk_class.lower() in ADVERSARIAL_RISK_CLASSES:
        return True
    return False


def _classify_single_task(
    run: LoadedRun,
    task_id: str,
    clear_best_threshold: float,
    tie_band: float,
    utility_floor: float,
) -> dict[str, Any]:
    """Classify a single task into a routing-label status.

    Returns a dict with the status, best/runner-up actions, margin, and
    eligibility information.
    """
    task = run.tasks.get(task_id)

    # SAFETY_CONTROLLED takes priority — routing is deferred.
    if _is_safety_controlled(run, task_id):
        return {
            "task_id": task_id,
            "status": "SAFETY_CONTROLLED",
            "best_action": None,
            "runner_up_action": None,
            "margin": None,
            "best_utility": None,
            "runner_up_utility": None,
            "n_executed": 0,
            "n_usable": 0,
            "split": task.split if task else "unknown",
            "family": task.family if task else "unknown",
            "risk_class": task.risk_class if task else "",
        }

    pairs = _sorted_utility_pairs(run, task_id)

    # INCOMPLETE — no executed outcomes at all.
    if not pairs:
        return {
            "task_id": task_id,
            "status": "INCOMPLETE",
            "best_action": None,
            "runner_up_action": None,
            "margin": None,
            "best_utility": None,
            "runner_up_utility": None,
            "n_executed": 0,
            "n_usable": 0,
            "split": task.split if task else "unknown",
            "family": task.family if task else "unknown",
            "risk_class": task.risk_class if task else "",
        }

    # Filter to usable actions: verified (not "failed") and above floor.
    usable = [
        (a, u, v)
        for a, u, v in pairs
        if v != "failed" and u >= utility_floor
    ]

    # NO_USABLE_ACTION — all eligible actions unverifiable or below floor.
    if not usable:
        return {
            "task_id": task_id,
            "status": "NO_USABLE_ACTION",
            "best_action": pairs[0][0] if pairs else None,
            "runner_up_action": pairs[1][0] if len(pairs) > 1 else None,
            "margin": None,
            "best_utility": pairs[0][1] if pairs else None,
            "runner_up_utility": pairs[1][1] if len(pairs) > 1 else None,
            "n_executed": len(pairs),
            "n_usable": 0,
            "split": task.split if task else "unknown",
            "family": task.family if task else "unknown",
            "risk_class": task.risk_class if task else "",
        }

    best_action, best_util, _ = usable[0]
    runner_up_action, runner_up_util, _ = (
        usable[1] if len(usable) > 1 else (None, best_util, None)
    )
    margin = best_util - runner_up_util

    # AMBIGUOUS — top two within tie_band.
    if margin is not None and margin < tie_band:
        status = "AMBIGUOUS"
    # CLEAR_BEST — margin >= clear_best_threshold.
    elif margin is not None and margin >= clear_best_threshold:
        status = "CLEAR_BEST"
    else:
        # Margin is in the gap between tie_band and clear_best_threshold.
        # Default to AMBIGUOUS since the best action is not clearly separable.
        status = "AMBIGUOUS"

    return {
        "task_id": task_id,
        "status": status,
        "best_action": best_action,
        "runner_up_action": runner_up_action,
        "margin": float(margin) if margin is not None else None,
        "best_utility": float(best_util),
        "runner_up_utility": float(runner_up_util) if runner_up_util is not None else None,
        "n_executed": len(pairs),
        "n_usable": len(usable),
        "split": task.split if task else "unknown",
        "family": task.family if task else "unknown",
        "risk_class": task.risk_class if task else "",
    }


def _split_task_ids(run: LoadedRun, split: str) -> list[str]:
    """Return task IDs belonging to a given split."""
    return [tid for tid, t in run.tasks.items() if t.split == split]


def _compute_proportions(labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute count and proportion of each label status."""
    n = len(labels)
    counts = Counter(l["status"] for l in labels)
    return {
        "n_tasks": n,
        "counts": {s: counts.get(s, 0) for s in LABEL_STATUSES},
        "proportions": {s: counts.get(s, 0) / n if n > 0 else 0.0 for s in LABEL_STATUSES},
    }


# ---------------------------------------------------------------------------
# Primary router training recommendations
# ---------------------------------------------------------------------------


def _router_training_recommendation(
    all_labels: list[dict[str, Any]],
    split_proportions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Generate training recommendations for the primary router.

    The primary router is trained only on CLEAR_BEST tasks.  AMBIGUOUS
    tasks may optionally be included using soft utility targets.
    """
    clear_best = [l for l in all_labels if l["status"] == "CLEAR_BEST"]
    ambiguous = [l for l in all_labels if l["status"] == "AMBIGUOUS"]

    # Use experience-split CLEAR_BEST for primary training.
    exp_clear = [l for l in clear_best if l["split"] == "experience"]
    exp_ambiguous = [l for l in ambiguous if l["split"] == "experience"]

    overall_props = split_proportions.get("all", {}).get("proportions", {})
    ambiguous_prop = overall_props.get("AMBIGUOUS", 0.0)
    clear_best_prop = overall_props.get("CLEAR_BEST", 0.0)

    if ambiguous_prop > 0.5:
        identifiability = "weak"
        ident_note = (
            "A benchmark dominated by AMBIGUOUS tasks has weak routing "
            "identifiability — the best action cannot be reliably "
            "distinguished from near-ties."
        )
    elif clear_best_prop > 0.5:
        identifiability = "strong"
        ident_note = (
            "A majority of tasks have a clear best action, providing "
            "strong routing identifiability for the primary router."
        )
    else:
        identifiability = "moderate"
        ident_note = (
            "Routing identifiability is moderate — a mix of clear and "
            "ambiguous tasks.  Soft utility targets for AMBIGUOUS tasks "
            "may improve router calibration."
        )

    return {
        "primary_training_set": {
            "source_split": "experience",
            "clear_best_count": len(exp_clear),
            "ambiguous_count": len(exp_ambiguous),
            "total_usable": len(exp_clear) + len(exp_ambiguous),
        },
        "strategy": {
            "primary": "Train on CLEAR_BEST tasks only.",
            "optional_extension": (
                "Include AMBIGUOUS tasks using soft utility targets "
                "(temperature-weighted distribution over top actions)."
            ),
        },
        "routing_identifiability": identifiability,
        "identifiability_note": ident_note,
        "ambiguous_proportion": float(ambiguous_prop),
        "clear_best_proportion": float(clear_best_prop),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def classify_routing_labels(
    run: LoadedRun,
    clear_best_threshold: float = 0.25,
    tie_band: float = 0.05,
    utility_floor: float = 0.0,
) -> dict[str, Any]:
    """Classify every task into a routing-label status.

    For each task the best and runner-up actions are determined from
    measured counterfactual utilities.  The task is then assigned one
    of five statuses (see module docstring).

    Args:
        run: loaded run data.
        clear_best_threshold: minimum margin for CLEAR_BEST status.
        tie_band: maximum margin between top two actions for AMBIGUOUS.
        utility_floor: minimum utility for an action to be considered
            usable.

    Returns:
        Dict with per-task labels, per-split and overall proportions,
        and router training recommendations.  Suitable for writing to
        ``routing_label_status.json``.
    """
    all_task_ids = list(run.tasks.keys())
    all_labels = [
        _classify_single_task(
            run, tid, clear_best_threshold, tie_band, utility_floor,
        )
        for tid in all_task_ids
    ]

    # Per-split proportions.
    split_proportions: dict[str, dict[str, Any]] = {}
    for split in ANALYZED_SPLITS:
        split_ids = _split_task_ids(run, split)
        split_labels = [l for l in all_labels if l["task_id"] in set(split_ids)]
        split_proportions[split] = _compute_proportions(split_labels)

    # Overall proportions.
    split_proportions["all"] = _compute_proportions(all_labels)

    # Router training recommendation.
    recommendation = _router_training_recommendation(all_labels, split_proportions)

    # Per-family breakdown (useful for diagnosing family-specific ambiguity).
    family_labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for l in all_labels:
        family_labels[l["family"]].append(l)
    family_proportions: dict[str, dict[str, Any]] = {}
    for fam, labels in sorted(family_labels.items()):
        family_proportions[fam] = _compute_proportions(labels)

    return {
        "thresholds": {
            "clear_best_threshold": clear_best_threshold,
            "tie_band": tie_band,
            "utility_floor": utility_floor,
        },
        "label_statuses": LABEL_STATUSES,
        "overall": split_proportions["all"],
        "by_split": {s: split_proportions[s] for s in ANALYZED_SPLITS if s in split_proportions},
        "by_family": family_proportions,
        "per_task": {l["task_id"]: l for l in all_labels},
        "router_training_recommendation": recommendation,
    }
