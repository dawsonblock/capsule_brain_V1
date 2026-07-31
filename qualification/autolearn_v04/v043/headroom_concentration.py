"""Oracle headroom concentration analysis for v0.4.3.

Determines whether the aggregate oracle-sham headroom
(``oracle_mean - sham_mean``) is *broad* — spread across many tasks that
each contribute meaningfully — or *concentrated* — dominated by a small
cluster of tasks.

Concentration is assessed along several axes:

* per-task headroom (``oracle_utility_i - sham_utility_i``)
* headroom grouped by task family, archetype, best action,
  utility-margin bucket, and crossover type
* cumulative headroom captured by the top 1 / 5 / 10 tasks and the
  top 10% / 25% / 50% of tasks
* Herfindahl-Hirschman index, Gini coefficient, top-10-share, and
  effective headroom task count

The output of :func:`analyze_headroom_concentration` is a JSON-serializable
``headroom_concentration`` dict.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import LoadedRun, TaskInfo
from .canonical_evaluator import (
    PolicyDecisionFn,
    evaluate_policy_from_counterfactuals,
    make_oracle_fn,
    make_executed_sham_fn,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Fraction of total headroom from a single archetype above which the run
#: is flagged as CONCENTRATED.
ARCHETYPE_CONCENTRATION_FRACTION: float = 0.50

#: HHI threshold below which headroom is considered broad.
HHI_BROAD_THRESHOLD: float = 0.25

#: Effective headroom task count above which headroom is considered broad.
EFFECTIVE_COUNT_BROAD_THRESHOLD: float = 4.0

#: Utility-margin bucket thresholds.  ``margin`` is
#: ``oracle_utility - second_best_utility``.
MARGIN_BUCKET_DEFINITIONS: dict[str, tuple[float | None, float | None]] = {
    "NEAR_TIE": (None, 0.1),
    "WEAK": (0.1, 0.5),
    "MEANINGFUL": (0.5, 2.0),
    "HIGH_VALUE": (2.0, None),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assign_margin_bucket(margin: float) -> str:
    """Assign a utility margin to one of the predeclared buckets."""
    for name, (lo, hi) in MARGIN_BUCKET_DEFINITIONS.items():
        if lo is None and hi is not None:
            if margin < hi:
                return name
        elif lo is not None and hi is None:
            if margin >= lo:
                return name
        elif lo is not None and hi is not None:
            if lo <= margin < hi:
                return name
    return "HIGH_VALUE"


def _sorted_eligible_utilities(run: LoadedRun, task_id: str) -> list[tuple[str, float]]:
    """Return eligible (action, utility) pairs sorted by utility descending."""
    task = run.tasks.get(task_id)
    if task is None:
        return []
    allowed = set(task.allowed_actions)
    pairs = [
        (o.action_id, o.utility)
        for o in run.outcomes_by_task.get(task_id, [])
        if o.availability == "executed" and o.utility is not None and o.action_id in allowed
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a non-negative array.

    A value of 0 indicates perfect equality; 1 indicates maximal
    inequality (all mass on a single unit).
    """
    if len(values) == 0:
        return 0.0
    v = np.asarray(values, dtype=float)
    if v.sum() <= 0:
        return 0.0
    v_sorted = np.sort(v)
    n = len(v_sorted)
    cumsum = np.cumsum(v_sorted)
    # Gini using the standard trapezoidal formula.
    return float((n + 1 - 2.0 * cumsum.sum() / cumsum[-1]) / n)


def _herfindahl_index(shares: np.ndarray) -> float:
    """Herfindahl-Hirschman index from a vector of shares."""
    if len(shares) == 0:
        return 0.0
    return float(np.sum(shares ** 2))


# ---------------------------------------------------------------------------
# Per-task headroom
# ---------------------------------------------------------------------------


def _compute_per_task_headroom(
    run: LoadedRun,
    task_ids: list[str],
    sham_fn: PolicyDecisionFn,
) -> list[dict[str, Any]]:
    """Compute per-task oracle-sham headroom for every task with outcomes.

    For each task ``i``::

        headroom_i = oracle_utility_i - sham_utility_i

    where ``sham_utility_i`` is the utility of the action selected by the
    sham policy.  The utility margin (oracle minus second-best) and the
    resulting margin bucket are also recorded.
    """
    oracle_fn = make_oracle_fn(run)
    records: list[dict[str, Any]] = []
    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        oracle_action = oracle_fn(tid, task)
        # Oracle utility: utility of the oracle action if executed, else
        # fall back to the canonical max-executed-utility accessor.
        oracle_utility = run.get_utility_for_action(tid, oracle_action) if oracle_action else None
        if oracle_utility is None:
            oracle_utility = run.get_oracle_utility(tid)

        sham_action = sham_fn(tid, task)
        sham_utility = run.get_utility_for_action(tid, sham_action) if sham_action else None
        if sham_utility is None:
            # No executed outcome for the sham action — skip this task so
            # that headroom is only computed on comparable tasks.
            continue

        headroom = oracle_utility - sham_utility

        # Utility margin (oracle minus second-best eligible utility).
        pairs = _sorted_eligible_utilities(run, tid)
        best_util = pairs[0][1] if pairs else oracle_utility
        second_util = pairs[1][1] if len(pairs) > 1 else best_util
        margin = best_util - second_util
        bucket = _assign_margin_bucket(margin)

        records.append({
            "task_id": tid,
            "split": task.split,
            "family": task.family,
            "archetype": task.archetype,
            "crossover_type": task.crossover_type,
            "oracle_action": oracle_action,
            "oracle_utility": float(oracle_utility),
            "sham_action": sham_action,
            "sham_utility": float(sham_utility),
            "headroom": float(headroom),
            "best_utility": float(best_util),
            "second_best_utility": float(second_util),
            "margin": float(margin),
            "margin_bucket": bucket,
        })
    return records


# ---------------------------------------------------------------------------
# Grouped headroom
# ---------------------------------------------------------------------------


def _group_headroom(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    """Aggregate per-task headroom by a categorical key."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(str(r.get(key, "unknown")), []).append(r)

    out: dict[str, dict[str, Any]] = {}
    total = float(sum(r["headroom"] for r in records))
    for name, items in sorted(groups.items()):
        headrooms = np.array([r["headroom"] for r in items], dtype=float)
        share = float(headrooms.sum() / total) if total > 0 else 0.0
        out[name] = {
            "n_tasks": len(items),
            "total_headroom": float(headrooms.sum()),
            "mean_headroom": float(headrooms.mean()) if len(headrooms) else 0.0,
            "share_of_total": share,
            "mean_oracle_utility": float(np.mean([r["oracle_utility"] for r in items])),
            "mean_sham_utility": float(np.mean([r["sham_utility"] for r in items])),
        }
    return out


# ---------------------------------------------------------------------------
# Cumulative headroom
# ---------------------------------------------------------------------------


def _cumulative_headroom(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute cumulative headroom captured by top-k tasks and top fractions."""
    if not records:
        return {
            "top_1": 0.0,
            "top_5": 0.0,
            "top_10": 0.0,
            "top_10_percent": 0.0,
            "top_25_percent": 0.0,
            "top_50_percent": 0.0,
            "total_headroom": 0.0,
            "n_tasks": 0,
            "ranked_task_ids": [],
        }

    ranked = sorted(records, key=lambda r: r["headroom"], reverse=True)
    headrooms = np.array([r["headroom"] for r in ranked], dtype=float)
    total = float(headrooms.sum())
    n = len(ranked)
    cumsum = np.cumsum(headrooms)

    def _frac(k: int) -> float:
        if total == 0:
            return 0.0
        return float(cumsum[min(k, n) - 1] / total) if k > 0 else 0.0

    def _pct(p: float) -> float:
        if total == 0 or n == 0:
            return 0.0
        k = max(1, int(np.ceil(p * n)))
        return float(cumsum[min(k, n) - 1] / total)

    return {
        "top_1": _frac(1),
        "top_5": _frac(5),
        "top_10": _frac(10),
        "top_10_percent": _pct(0.10),
        "top_25_percent": _pct(0.25),
        "top_50_percent": _pct(0.50),
        "total_headroom": total,
        "n_tasks": n,
        "ranked_task_ids": [r["task_id"] for r in ranked],
    }


# ---------------------------------------------------------------------------
# Concentration metrics
# ---------------------------------------------------------------------------


def _concentration_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute HHI, Gini, top-10-share, and effective headroom task count."""
    if not records:
        return {
            "hhi": 0.0,
            "gini": 0.0,
            "top_10_share": 0.0,
            "effective_headroom_task_count": 0.0,
        }

    headrooms = np.array([r["headroom"] for r in records], dtype=float)
    total = float(headrooms.sum())
    n = len(headrooms)

    if total <= 0:
        return {
            "hhi": 0.0,
            "gini": 0.0,
            "top_10_share": 0.0,
            "effective_headroom_task_count": float(n),
        }

    # Only positive headroom contributes to concentration shares.
    positive = headrooms[headrooms > 0]
    pos_total = float(positive.sum()) if positive.size else total
    shares = np.where(headrooms > 0, headrooms / pos_total, 0.0)

    hhi = _herfindahl_index(shares)
    gini = _gini_coefficient(np.abs(headrooms))

    k = max(1, int(np.ceil(0.10 * n)))
    top_10_share = float(headrooms[np.argsort(headrooms)[::-1][:k]].sum() / total)

    effective_count = 1.0 / hhi if hhi > 0 else float(n)

    return {
        "hhi": float(hhi),
        "gini": float(gini),
        "top_10_share": top_10_share,
        "effective_headroom_task_count": float(effective_count),
    }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def _interpretation(
    concentration: dict[str, Any],
    by_archetype: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    """Return (concentration_status, interpretation_text)."""
    hhi = concentration["hhi"]
    eff = concentration["effective_headroom_task_count"]

    # Archetype concentration flag.
    max_archetype_share = 0.0
    dominant_archetype = ""
    for name, info in by_archetype.items():
        if info["share_of_total"] > max_archetype_share:
            max_archetype_share = info["share_of_total"]
            dominant_archetype = name

    archetype_concentrated = max_archetype_share >= ARCHETYPE_CONCENTRATION_FRACTION

    if hhi < HHI_BROAD_THRESHOLD and eff > EFFECTIVE_COUNT_BROAD_THRESHOLD and not archetype_concentrated:
        status = "BROAD"
        text = (
            f"Headroom is broad: HHI={hhi:.3f} (<{HHI_BROAD_THRESHOLD}), "
            f"effective task count={eff:.1f} (>{EFFECTIVE_COUNT_BROAD_THRESHOLD}). "
            f"Many tasks contribute meaningfully to oracle-sham headroom."
        )
    elif archetype_concentrated:
        status = "CONCENTRATED"
        text = (
            f"Headroom is concentrated: archetype '{dominant_archetype}' alone "
            f"accounts for {max_archetype_share:.1%} of total headroom "
            f"(>= {ARCHETYPE_CONCENTRATION_FRACTION:.0%}). HHI={hhi:.3f}."
        )
    elif hhi >= HHI_BROAD_THRESHOLD:
        status = "CONCENTRATED"
        text = (
            f"Headroom is concentrated: HHI={hhi:.3f} (>={HHI_BROAD_THRESHOLD}), "
            f"effective task count={eff:.1f}. A small cluster of tasks dominates."
        )
    else:
        status = "MODERATE"
        text = (
            f"Headroom is moderate: HHI={hhi:.3f}, effective task count={eff:.1f}. "
            f"Neither clearly broad nor strongly concentrated."
        )

    return status, text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_headroom_concentration(
    run: LoadedRun,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Validate oracle headroom concentration.

    Args:
        run: The loaded run with counterfactual outcomes and policies.
        task_ids: Optional subset of task IDs.  Defaults to the test split.

    Returns:
        A JSON-serializable ``headroom_concentration`` dict with
        per-task headroom, grouped headroom, cumulative headroom,
        concentration metrics, interpretation, and concentration status.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # Evaluate oracle and sham through the canonical evaluator so that
    # aggregate means are consistent with the rest of v0.4.3.
    oracle_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_oracle_fn(run), "oracle"
    )
    sham_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_executed_sham_fn(run), "sham"
    )

    aggregate_headroom = oracle_result.mean_utility - sham_result.mean_utility

    # Per-task headroom using the executed sham policy.
    sham_fn = make_executed_sham_fn(run)
    records = _compute_per_task_headroom(run, task_ids, sham_fn)

    # Grouped headroom.
    by_family = _group_headroom(records, "family")
    by_archetype = _group_headroom(records, "archetype")
    by_best_action = _group_headroom(records, "oracle_action")
    by_margin_bucket = _group_headroom(records, "margin_bucket")
    by_crossover = _group_headroom(records, "crossover_type")

    by_group: dict[str, dict[str, Any]] = {
        "family": by_family,
        "archetype": by_archetype,
        "best_action": by_best_action,
        "margin_bucket": by_margin_bucket,
        "crossover_type": by_crossover,
    }

    cumulative = _cumulative_headroom(records)
    concentration = _concentration_metrics(records)
    status, text = _interpretation(concentration, by_archetype)

    return {
        "aggregate_oracle_mean": float(oracle_result.mean_utility),
        "aggregate_sham_mean": float(sham_result.mean_utility),
        "aggregate_headroom": float(aggregate_headroom),
        "n_tasks": len(records),
        "per_task_headroom": records,
        "by_group": by_group,
        "cumulative_headroom": cumulative,
        "concentration_metrics": concentration,
        "interpretation": text,
        "concentration_status": status,
        "archetype_concentration_fraction": ARCHETYPE_CONCENTRATION_FRACTION,
        "margin_bucket_definitions": {
            name: {"min": lo, "max": hi}
            for name, (lo, hi) in MARGIN_BUCKET_DEFINITIONS.items()
        },
    }
