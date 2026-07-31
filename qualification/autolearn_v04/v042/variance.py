"""Catastrophic failure and variance analysis.

Computes paired utility deltas between the candidate policy and its
controls (baseline and sham), then reports robust distributional
summaries and leave-one-out / leave-one-cluster-out influence
diagnostics.  Outliers are deliberately *not* removed: the primary
metric is preserved while robust summaries are reported as
diagnostics so that catastrophic failures remain visible.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from .data import LoadedRun
from .bootstrap import task_paired_bootstrap
from .policy_controls import (
    make_candidate_selector,
    make_baseline_selector,
    make_sham_global_selector,
    evaluate_policy_on_tasks,
)


# Threshold (in utility units) below which a paired delta is flagged as
# a catastrophic-negative outcome and above which it is flagged as a
# large-positive outcome.
CATASTROPHIC_THRESHOLD: float = 5.0

# Fraction to trim from each tail when computing the trimmed mean.
TRIM_FRACTION: float = 0.10

# Number of top-influence tasks to report.
TOP_K_INFLUENCE: int = 10


def _robust_summary(values: np.ndarray) -> dict[str, float | int]:
    """Return a robust distributional summary of ``values``.

    Includes central-tendency, spread, tail, and count diagnostics.
    No values are discarded; this is purely a descriptive summary.
    """
    if values.size == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "trimmed_mean_10pct": 0.0,
            "std": 0.0,
            "mad": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p5": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "catastrophic_negative_count": 0,
            "large_positive_count": 0,
        }
    median = float(np.median(values))
    # Median absolute deviation (about the median), scaled to be a
    # consistent estimator of the standard deviation under normality.
    mad = float(np.median(np.abs(values - median)) * 1.4826)

    # Trimmed mean: drop the lowest and highest ``TRIM_FRACTION`` of
    # observations before averaging.
    n = values.size
    k = int(np.floor(n * TRIM_FRACTION))
    if 2 * k < n:
        trimmed = float(np.sort(values)[k:n - k].mean()) if k > 0 else float(values.mean())
    else:
        trimmed = float(np.median(values))

    return {
        "n": int(n),
        "mean": float(values.mean()),
        "median": median,
        "trimmed_mean_10pct": trimmed,
        "std": float(values.std(ddof=1)) if n > 1 else 0.0,
        "mad": mad,
        "min": float(values.min()),
        "max": float(values.max()),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "catastrophic_negative_count": int(np.sum(values < -CATASTROPHIC_THRESHOLD)),
        "large_positive_count": int(np.sum(values > CATASTROPHIC_THRESHOLD)),
    }


def _leave_one_out_mean(deltas: np.ndarray) -> np.ndarray:
    """For each index, the mean of ``deltas`` with that index excluded.

    Uses the algebraic identity:
        mean_without_i = (sum - x_i) / (n - 1)
    """
    n = deltas.size
    if n <= 1:
        return np.zeros(n)
    total = deltas.sum()
    return (total - deltas) / (n - 1)


def _leave_one_cluster_out_mean(
    deltas: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[str, float]:
    """Mean delta when each entire cluster is held out in turn."""
    out: dict[str, float] = {}
    n = deltas.size
    if n == 0:
        return out
    total = deltas.sum()
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        n_remaining = n - int(np.sum(mask))
        if n_remaining <= 0:
            out[str(c)] = float(deltas.mean())
        else:
            out[str(c)] = float((total - deltas[mask].sum()) / n_remaining)
    return out


def _top_k_influence(
    deltas: np.ndarray,
    task_ids: list[str],
    overall_mean: float,
    k: int = TOP_K_INFLUENCE,
) -> list[dict[str, Any]]:
    """Rank tasks by how much their removal shifts the mean delta.

    Influence is measured as the absolute change in the mean:
        |mean_without_i - overall_mean|
    The most influential tasks (largest shift) are returned first.
    """
    n = deltas.size
    if n <= 1:
        return []
    loo = _leave_one_out_mean(deltas)
    shifts = np.abs(loo - overall_mean)
    order = np.argsort(-shifts)  # descending
    k = min(k, n)
    return [
        {
            "task_id": task_ids[idx],
            "delta": float(deltas[idx]),
            "mean_without": float(loo[idx]),
            "mean_shift": float(loo[idx] - overall_mean),
            "abs_mean_shift": float(shifts[idx]),
        }
        for idx in order[:k]
    ]


def analyze_variance(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze variance and catastrophic failures of paired utility deltas.

    Computes, for the requested tasks (defaulting to the test split):
        delta_i     = U_candidate_i - U_baseline_i
        delta_sham_i = U_candidate_i - U_sham_i

    Reports robust distributional summaries and influence diagnostics.
    Outliers are *not* removed; the primary metric (mean delta) is
    preserved while robust summaries flag catastrophic behaviour.

    Returns a dict with two top-level keys suitable for writing to
    ``variance_analysis.json`` and ``influence_analysis.json``.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # ------------------------------------------------------------------
    # Evaluate policies on the selected tasks.
    # ------------------------------------------------------------------
    candidate_sel = make_candidate_selector(run)
    baseline_sel = make_baseline_selector(run)
    sham_sel = make_sham_global_selector(run, seed=12345)

    candidate_res = evaluate_policy_on_tasks(run, task_ids, candidate_sel, "candidate")
    baseline_res = evaluate_policy_on_tasks(run, task_ids, baseline_sel, "baseline")
    sham_res = evaluate_policy_on_tasks(run, task_ids, sham_sel, "sham")

    u_candidate = np.array(candidate_res["utilities"], dtype=float)
    u_baseline = np.array(baseline_res["utilities"], dtype=float)
    u_sham = np.array(sham_res["utilities"], dtype=float)

    # Per-task metadata (family / archetype) aligned to the evaluated
    # task ordering.  ``evaluate_policy_on_tasks`` skips tasks it cannot
    # score, so we derive the effective task list from its per_task
    # output to guarantee alignment.
    per_task = candidate_res["per_task"]
    effective_task_ids = [pt["task_id"] for pt in per_task]
    families = np.array([pt["family"] for pt in per_task])
    archetypes = np.array([pt["archetype"] for pt in per_task])

    # Paired deltas.
    delta = u_candidate - u_baseline
    delta_sham = u_candidate - u_sham

    overall_mean = float(delta.mean()) if delta.size else 0.0
    overall_mean_sham = float(delta_sham.mean()) if delta_sham.size else 0.0

    # ------------------------------------------------------------------
    # Variance analysis.
    # ------------------------------------------------------------------
    variance_analysis: dict[str, Any] = {
        "n_tasks": int(delta.size),
        "catastrophic_threshold": CATASTROPHIC_THRESHOLD,
        "trim_fraction": TRIM_FRACTION,
        "seed": seed,
        "delta_candidate_baseline": _robust_summary(delta),
        "delta_candidate_sham": _robust_summary(delta_sham),
        "primary_metric": {
            "name": "mean_delta_candidate_baseline",
            "value": overall_mean,
            "note": "Primary metric preserved without outlier removal.",
        },
        "bootstrap_candidate_baseline": task_paired_bootstrap(
            delta, n_bootstrap=5000, seed=seed,
        ),
        "bootstrap_candidate_sham": task_paired_bootstrap(
            delta_sham, n_bootstrap=5000, seed=seed,
        ),
    }

    # ------------------------------------------------------------------
    # Influence diagnostics.
    # ------------------------------------------------------------------
    influence_analysis: dict[str, Any] = {
        "n_tasks": int(delta.size),
        "overall_mean_delta": overall_mean,
        "overall_mean_delta_sham": overall_mean_sham,
        "leave_one_out": {
            "task_ids": effective_task_ids,
            "mean_without_task": _leave_one_out_mean(delta).tolist(),
            "mean_without_task_sham": _leave_one_out_mean(delta_sham).tolist(),
        },
        "leave_one_archetype_out": _leave_one_cluster_out_mean(delta, archetypes),
        "leave_one_archetype_out_sham": _leave_one_cluster_out_mean(delta_sham, archetypes),
        "leave_one_family_out": _leave_one_cluster_out_mean(delta, families),
        "leave_one_family_out_sham": _leave_one_cluster_out_mean(delta_sham, families),
        "top_k_influence_tasks": _top_k_influence(
            delta, effective_task_ids, overall_mean,
        ),
        "top_k_influence_tasks_sham": _top_k_influence(
            delta_sham, effective_task_ids, overall_mean_sham,
        ),
        "note": (
            "Outliers are not removed. Influence is reported as the "
            "change in the mean delta when a task or cluster is held "
            "out, preserving the primary metric."
        ),
    }

    return {
        "variance_analysis": variance_analysis,
        "influence_analysis": influence_analysis,
    }
