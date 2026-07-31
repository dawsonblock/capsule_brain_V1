"""Best-action imbalance analysis across splits.

Analyzes best-action distributions across D_experience, D_validation,
D_test, and D_ood, reporting per-action statistics, distribution-level
imbalance metrics, and cross-split distribution comparisons.

The output of :func:`analyze_imbalance` is the content for
``best_action_distribution.json``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .data import LoadedRun


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The canonical action set used throughout the benchmark.
CANONICAL_ACTIONS: list[str] = [
    "ANSWER_DIRECT",
    "RETRIEVE_MEMORY",
    "CALL_TOOL",
    "START_WORKFLOW",
]

#: The four splits analyzed for best-action imbalance.
SPLITS: list[str] = ["experience", "validation", "test", "ood"]

#: Controls the candidate must beat for promotion, listed in the
#: promotion research diagnostic field.
PROMOTION_CONTROLS: list[str] = [
    "P_random",
    "P_majority",
    "P_frequency",
    "P_family_majority",
    "P_archetype_majority",
    "P_sham_global",
    "P_sham_family",
    "P_sham_margin",
    "P_sham_feature",
    "P_baseline",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sorted_utility_pairs(run: LoadedRun, task_id: str) -> list[tuple[str, float]]:
    """Return executed (action, utility) pairs sorted by utility descending."""
    outcomes = run.outcomes_by_task.get(task_id, [])
    pairs = [
        (o.action_id, o.utility)
        for o in outcomes
        if o.availability == "executed" and o.utility is not None
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _safe_mean(values: list[float] | np.ndarray) -> float:
    """Mean of a list/array, returning 0.0 for empty input."""
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# Per-split analysis
# ---------------------------------------------------------------------------


def _split_task_ids(run: LoadedRun, split: str) -> list[str]:
    """Return task IDs belonging to a given split."""
    return [tid for tid, t in run.tasks.items() if t.split == split]


def _analyze_split(
    run: LoadedRun,
    split: str,
    task_ids: list[str],
) -> dict[str, Any]:
    """Compute per-action and distribution-level statistics for one split.

    For each action reports:
      - count
      - percentage
      - mean best-action utility
      - mean second-best utility
      - mean utility margin (best - second-best)
      - family composition (which families contribute)
      - archetype composition

    Also computes distribution-level imbalance metrics:
      - class entropy
      - effective number of actions
      - majority-action rate
      - Gini coefficient
    """
    # Collect best-action assignments and per-task utility info.
    best_actions: list[str] = []
    best_utils: list[float] = []
    second_best_utils: list[float] = []
    margins: list[float] = []

    # Track family/archetype composition per best action.
    action_families: dict[str, set[str]] = defaultdict(set)
    action_archetypes: dict[str, set[str]] = defaultdict(set)
    action_family_counts: dict[str, Counter] = defaultdict(Counter)
    action_archetype_counts: dict[str, Counter] = defaultdict(Counter)

    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        pairs = _sorted_utility_pairs(run, tid)
        if not pairs:
            continue
        best_action = pairs[0][0]
        best_util = pairs[0][1]
        second_util = pairs[1][1] if len(pairs) > 1 else best_util
        margin = best_util - second_util

        best_actions.append(best_action)
        best_utils.append(best_util)
        second_best_utils.append(second_util)
        margins.append(margin)

        action_families[best_action].add(task.family)
        action_archetypes[best_action].add(task.archetype)
        action_family_counts[best_action][task.family] += 1
        action_archetype_counts[best_action][task.archetype] += 1

    n = len(best_actions)
    counts = Counter(best_actions)

    # --- Per-action statistics -------------------------------------------
    per_action: dict[str, Any] = {}
    for action in CANONICAL_ACTIONS:
        c = counts.get(action, 0)
        pct = c / n if n > 0 else 0.0

        # Indices where this action is the best.
        idxs = [i for i, a in enumerate(best_actions) if a == action]
        if idxs:
            mean_best = _safe_mean([best_utils[i] for i in idxs])
            mean_second = _safe_mean([second_best_utils[i] for i in idxs])
            mean_margin = _safe_mean([margins[i] for i in idxs])
        else:
            mean_best = 0.0
            mean_second = 0.0
            mean_margin = 0.0

        per_action[action] = {
            "count": c,
            "percentage": pct,
            "mean_best_action_utility": mean_best,
            "mean_second_best_utility": mean_second,
            "mean_utility_margin": mean_margin,
            "family_composition": {
                "families": sorted(action_families.get(action, set())),
                "counts": dict(action_family_counts.get(action, {})),
            },
            "archetype_composition": {
                "archetypes": sorted(action_archetypes.get(action, set())),
                "counts": dict(action_archetype_counts.get(action, {})),
            },
        }

    # --- Distribution-level imbalance metrics ----------------------------
    probs = np.array(
        [counts.get(a, 0) / n for a in CANONICAL_ACTIONS],
        dtype=float,
    ) if n > 0 else np.zeros(len(CANONICAL_ACTIONS))

    # Class entropy: H = -sum(p_a * log(p_a))
    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero))) if len(nonzero) > 0 else 0.0

    # Effective number of actions: N_eff = exp(H)
    n_eff = float(np.exp(entropy)) if entropy > 0 else 1.0

    # Majority-action rate
    majority_rate = float(probs.max()) if n > 0 else 0.0

    # Gini coefficient over the action counts.
    gini = _gini_coefficient(probs)

    return {
        "split": split,
        "n_tasks": n,
        "per_action": per_action,
        "distribution": {
            "counts": {a: counts.get(a, 0) for a in CANONICAL_ACTIONS},
            "probabilities": {a: float(p) for a, p in zip(CANONICAL_ACTIONS, probs)},
            "class_entropy": entropy,
            "effective_number_of_actions": n_eff,
            "majority_action_rate": majority_rate,
            "majority_action": CANONICAL_ACTIONS[int(np.argmax(probs))] if n > 0 else None,
            "gini_coefficient": gini,
        },
    }


def _gini_coefficient(probs: np.ndarray) -> float:
    """Compute the Gini coefficient over a probability vector.

    Gini = (1 / (2 * n^2 * mean)) * sum_i sum_j |x_i - x_j|

    For a perfectly uniform distribution Gini = 0; for a degenerate
    distribution (one action only) Gini approaches 1.
    """
    n = len(probs)
    if n == 0:
        return 0.0
    mean = probs.mean()
    if abs(mean) < 1e-12:
        return 0.0
    diff_sum = float(np.sum(np.abs(probs[:, None] - probs[None, :])))
    return diff_sum / (2.0 * n * n * mean)


# ---------------------------------------------------------------------------
# Cross-split comparison
# ---------------------------------------------------------------------------


def _total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance: TV(P, Q) = 0.5 * sum |p_i - q_i|."""
    return float(0.5 * np.sum(np.abs(p - q)))


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base e) between two distributions.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q).
    """
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _compare_splits(
    split_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare best-action distributions pairwise across all splits."""
    comparisons: dict[str, Any] = {}

    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i + 1:]:
            r1 = split_results.get(s1)
            r2 = split_results.get(s2)
            if r1 is None or r2 is None:
                continue
            p = np.array(
                [r1["distribution"]["probabilities"][a] for a in CANONICAL_ACTIONS],
                dtype=float,
            )
            q = np.array(
                [r2["distribution"]["probabilities"][a] for a in CANONICAL_ACTIONS],
                dtype=float,
            )
            key = f"{s1}_vs_{s2}"
            comparisons[key] = {
                "total_variation_distance": _total_variation_distance(p, q),
                "jensen_shannon_divergence": _jensen_shannon_divergence(p, q),
            }

    return comparisons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_imbalance(run: LoadedRun) -> dict[str, Any]:
    """Analyze best-action imbalance across all splits.

    For each of the four splits (experience, validation, test, ood) this
    reports per-action statistics (count, percentage, mean best-action
    utility, mean second-best utility, mean utility margin, family and
    archetype composition) plus distribution-level imbalance metrics
    (class entropy, effective number of actions, majority-action rate,
    Gini coefficient).

    Best-action distributions are then compared pairwise across splits
    using total variation distance and Jensen-Shannon divergence.

    A promotion research diagnostic field lists the control policies
    that the candidate must beat to qualify for promotion.

    Args:
        run: Loaded run data with tasks, outcomes, and derived fields.

    Returns:
        Content for ``best_action_distribution.json``.
    """
    split_results: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        task_ids = _split_task_ids(run, split)
        split_results[split] = _analyze_split(run, split, task_ids)

    cross_split = _compare_splits(split_results)

    return {
        "splits": split_results,
        "cross_split_comparison": cross_split,
        "canonical_actions": CANONICAL_ACTIONS,
        "promotion_research_diagnostic": {
            "controls_the_candidate_must_beat": PROMOTION_CONTROLS,
            "description": (
                "For promotion, the candidate policy must demonstrate "
                "utility above each of these control policies on the "
                "held-out test split, with bootstrap confidence intervals "
                "excluding zero. The sham controls are the hardest "
                "identifiability barriers: if the candidate cannot beat "
                "a label-shuffled or feature-shuffled sham, the learned "
                "routing signal is not distinguishable from noise."
            ),
        },
    }
