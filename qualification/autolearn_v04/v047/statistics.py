"""Paired cluster bootstrap statistics for v0.4.7 (Section 10.3).

Wraps and extends the ``grouped_paired_bootstrap`` from
``qualification.autolearn_v04.statistics`` with a richer, dict-based
result that carries win/tie/loss rates and a strict practical-effect
pass rule.

Pure Python only — no numpy — consistent with the rest of the codebase.
"""
from __future__ import annotations

import math
import random
from typing import Any

from qualification.autolearn_v04.statistics import grouped_paired_bootstrap


def paired_cluster_bootstrap(
    paired_rows: list[dict],
    *,
    cluster_key: str = "task_group_id",
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Paired cluster bootstrap by task group.

    Resamples CLUSTERS (task groups) with replacement, not individual rows.
    All rows belonging to a resampled cluster are included together, which
    preserves within-group correlation structure.

    Args:
        paired_rows: list of dicts with keys ``task_id``, ``task_group_id``,
            ``family``, ``delta`` (float paired difference),
            ``selected_utility`` (float), ``comparison_utility`` (float).
        cluster_key: the row key identifying the cluster (default
            ``task_group_id``).
        n_resamples: number of bootstrap resamples (>= 10_000 recommended).
        confidence_level: two-sided confidence level for the interval.
        seed: deterministic seed for the RNG.

    Returns:
        dict with keys: comparison, n_tasks, n_task_groups, mean_delta,
        median_delta, standard_error, confidence_level, lower_bound,
        upper_bound, practical_threshold, passes, win_rate, tie_rate,
        loss_rate.
    """
    if not paired_rows:
        return {
            "comparison": "",
            "n_tasks": 0,
            "n_task_groups": 0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "standard_error": 0.0,
            "confidence_level": confidence_level,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "practical_threshold": 0.0,
            "passes": False,
            "win_rate": 0.0,
            "tie_rate": 0.0,
            "loss_rate": 0.0,
        }

    # Group rows by cluster key, preserving cluster order of first appearance.
    cluster_to_rows: dict[Any, list[dict]] = {}
    cluster_order: list[Any] = []
    for row in paired_rows:
        ck = row.get(cluster_key)
        if ck not in cluster_to_rows:
            cluster_to_rows[ck] = []
            cluster_order.append(ck)
        cluster_to_rows[ck].append(row)

    unique_clusters = cluster_order
    n_clusters = len(unique_clusters)
    n_tasks = len(paired_rows)

    # Observed statistics from the original (unresampled) deltas.
    deltas = [float(row["delta"]) for row in paired_rows]
    observed_mean = sum(deltas) / n_tasks

    # Win/tie/loss rates from the original deltas.
    wins = sum(1 for d in deltas if d > 0)
    ties = sum(1 for d in deltas if d == 0)
    losses = sum(1 for d in deltas if d < 0)
    win_rate = wins / n_tasks
    tie_rate = ties / n_tasks
    loss_rate = losses / n_tasks

    # Cluster bootstrap: resample clusters with replacement.
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_resamples):
        sampled_deltas: list[float] = []
        for _ in range(n_clusters):
            ck = unique_clusters[rng.randrange(n_clusters)]
            for row in cluster_to_rows[ck]:
                sampled_deltas.append(float(row["delta"]))
        boot_means.append(sum(sampled_deltas) / len(sampled_deltas))

    boot_means.sort()
    alpha = 1.0 - confidence_level
    idx_low = max(0, min(n_resamples - 1, int((alpha / 2.0) * n_resamples)))
    idx_high = max(0, min(n_resamples - 1, int((1.0 - alpha / 2.0) * n_resamples)))
    lower_bound = boot_means[idx_low]
    upper_bound = boot_means[idx_high]
    median_delta = boot_means[n_resamples // 2]

    # Standard error of the bootstrap distribution (population std).
    variance = sum((bm - observed_mean) ** 2 for bm in boot_means) / n_resamples
    standard_error = math.sqrt(variance)

    # Practical threshold defaults to 0.0 here; gate evaluators override it
    # via the config and recompute the strict pass decision.
    practical_threshold = 0.0
    passes = lower_bound > practical_threshold

    return {
        "comparison": "",
        "n_tasks": n_tasks,
        "n_task_groups": n_clusters,
        "mean_delta": observed_mean,
        "median_delta": median_delta,
        "standard_error": standard_error,
        "confidence_level": confidence_level,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "practical_threshold": practical_threshold,
        "passes": passes,
        "win_rate": win_rate,
        "tie_rate": tie_rate,
        "loss_rate": loss_rate,
    }


def _rows_by_task(results: dict) -> dict[str, dict]:
    """Index ``task_rows`` from a results dict by ``task_id``."""
    by_task: dict[str, dict] = {}
    for row in results.get("task_rows", []):
        by_task[row["task_id"]] = row
    return by_task


def _make_paired_rows(
    selected: dict,
    comparison: dict,
    *,
    comparison_name: str,
) -> list[dict]:
    """Build paired rows by matching ``task_id`` across two result sets.

    Each row carries: task_id, task_group_id, family, delta,
    selected_utility, comparison_utility.
    """
    sel_by_task = _rows_by_task(selected)
    cmp_by_task = _rows_by_task(comparison)
    paired: list[dict] = []
    for task_id, sel_row in sel_by_task.items():
        cmp_row = cmp_by_task.get(task_id)
        if cmp_row is None:
            continue
        sel_util = float(sel_row.get("selected_utility", 0.0))
        cmp_util = float(cmp_row.get("selected_utility", 0.0))
        paired.append({
            "task_id": task_id,
            "task_group_id": sel_row.get(
                "task_group_id", sel_row.get("group_id", task_id)
            ),
            "family": sel_row.get("family", ""),
            "delta": sel_util - cmp_util,
            "selected_utility": sel_util,
            "comparison_utility": cmp_util,
        })
    return paired


def compute_paired_deltas(
    candidate_results: list[dict],
    baseline_results: list[dict],
    sham_results: list[dict],
    oracle_results: list[dict],
) -> dict:
    """Compute paired deltas for all comparisons.

    The four ``*_results`` arguments may each be either a single results
    dict (with a ``task_rows`` list) or a list of such dicts. When a list
    is provided, rows are merged across its entries by ``task_id`` (later
    entries do not overwrite earlier ones).

    Returns a dict with keys:
        - candidate_vs_baseline: list of paired rows
        - candidate_vs_sham: list of paired rows
        - oracle_vs_baseline: list of paired rows
        - oracle_vs_sham: list of paired rows

    Each row is matched by ``task_id`` and carries the paired difference
    (selected - comparison) plus the raw utilities.
    """
    def _merge(maybe_list):
        if isinstance(maybe_list, dict):
            return maybe_list
        merged_rows: list[dict] = []
        seen: set[str] = set()
        for entry in maybe_list:
            for row in entry.get("task_rows", []):
                tid = row.get("task_id")
                if tid in seen:
                    continue
                seen.add(tid)
                merged_rows.append(row)
        return {"task_rows": merged_rows}

    cand = _merge(candidate_results)
    base = _merge(baseline_results)
    sham = _merge(sham_results)
    oracle = _merge(oracle_results)

    return {
        "candidate_vs_baseline": _make_paired_rows(
            cand, base, comparison_name="baseline"
        ),
        "candidate_vs_sham": _make_paired_rows(
            cand, sham, comparison_name="sham"
        ),
        "oracle_vs_baseline": _make_paired_rows(
            oracle, base, comparison_name="baseline"
        ),
        "oracle_vs_sham": _make_paired_rows(
            oracle, sham, comparison_name="sham"
        ),
    }


__all__ = [
    "paired_cluster_bootstrap",
    "compute_paired_deltas",
    "grouped_paired_bootstrap",
]
