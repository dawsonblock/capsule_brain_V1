"""Bootstrap and confidence interval utilities for v0.4.2 diagnostics.

Provides task-level, cluster-level, and stratified bootstrap methods.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Any


def task_paired_bootstrap(
    deltas: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Standard task-level paired bootstrap.

    Args:
        deltas: per-task paired utility deltas.
        n_bootstrap: number of bootstrap replicates.
        confidence: confidence level (0.95 = 95% CI).
        seed: random seed.

    Returns:
        dict with mean, ci_low, ci_high, std_error.
    """
    if len(deltas) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0}
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = deltas[idx].mean()
    alpha = (1 - confidence) / 2
    ci_low = float(np.percentile(means, 100 * alpha))
    ci_high = float(np.percentile(means, 100 * (1 - alpha)))
    return {
        "mean": float(deltas.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "std_error": float(deltas.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
    }


def cluster_paired_bootstrap(
    deltas: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Cluster-aware paired bootstrap.

    Resamples whole clusters (archetypes, families, etc.) with replacement,
    then includes all tasks in selected clusters.

    Args:
        deltas: per-task paired utility deltas.
        cluster_ids: per-task cluster assignment.
        n_bootstrap: number of bootstrap replicates.
        confidence: confidence level.
        seed: random seed.

    Returns:
        dict with mean, ci_low, ci_high, std_error.
    """
    if len(deltas) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0}
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_to_indices: dict[Any, np.ndarray] = {}
    for c in unique_clusters:
        cluster_to_indices[c] = np.where(cluster_ids == c)[0]

    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
        means[i] = deltas[sampled_indices].mean()

    alpha = (1 - confidence) / 2
    return {
        "mean": float(deltas.mean()),
        "ci_low": float(np.percentile(means, 100 * alpha)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha))),
        "std_error": float(deltas.std(ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0,
    }


def stratified_paired_bootstrap(
    deltas: np.ndarray,
    strata_ids: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Family-stratified bootstrap — resamples within each stratum independently.

    Preserves stratum proportions. Each stratum is bootstrapped separately
    and the results are combined with original stratum weights.
    """
    if len(deltas) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0}
    rng = np.random.default_rng(seed)
    unique_strata = np.unique(strata_ids)
    n = len(deltas)

    # Compute stratum weights.
    stratum_weights = {}
    for s in unique_strata:
        stratum_weights[s] = np.sum(strata_ids == s) / n

    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        total = 0.0
        for s in unique_strata:
            s_indices = np.where(strata_ids == s)[0]
            if len(s_indices) == 0:
                continue
            sampled = rng.choice(s_indices, size=len(s_indices), replace=True)
            total += deltas[sampled].mean() * stratum_weights[s]
        means[i] = total

    alpha = (1 - confidence) / 2
    return {
        "mean": float(deltas.mean()),
        "ci_low": float(np.percentile(means, 100 * alpha)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha))),
        "std_error": float(deltas.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
    }


def bootstrap_ratio(
    numerator_deltas: np.ndarray,
    denominator: float,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap a ratio (numerator_mean / denominator) with CI.

    Used for headroom recovery fractions.
    """
    if abs(denominator) < 1e-10:
        return {"ratio": 0.0, "ci_low": 0.0, "ci_high": 0.0, "mean_numerator": 0.0}
    base = task_paired_bootstrap(numerator_deltas, n_bootstrap, confidence, seed)
    rng = np.random.default_rng(seed)
    n = len(numerator_deltas)
    ratios = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ratios[i] = numerator_deltas[idx].mean() / denominator
    alpha = (1 - confidence) / 2
    return {
        "ratio": float(numerator_deltas.mean() / denominator),
        "ci_low": float(np.percentile(ratios, 100 * alpha)),
        "ci_high": float(np.percentile(ratios, 100 * (1 - alpha))),
        "mean_numerator": base["mean"],
    }
