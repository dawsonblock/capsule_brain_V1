"""Statistical tests for v0.4.0 qualification (Section 10).

Adds:
  * grouped (stratified) paired bootstrap by task group (Section 10.3).
  * Holm-Bonferroni multiple-comparison correction (Section 10.5).
  * AUROC / AUPRC / Brier / ECE for Gate B (Section 12.11).

Reuses the core BCa bootstrap, permutation test, and sign test from the
shared autolearn statistics module.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from capsule_brain.autolearn.statistics import (
    BootstrapResult,
    SignTestResult,
    paired_bootstrap,
    paired_permutation_test,
    sign_test,
)


@dataclass(slots=True)
class GroupedBootstrapResult:
    """Result of a grouped (stratified) paired bootstrap."""
    mean: float
    median: float
    std_error: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    n_groups: int
    n_samples: int
    epsilon: float = 0.0
    passes_threshold: bool = False
    bootstrap_samples: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 8),
            "median": round(self.median, 8),
            "std_error": round(self.std_error, 8),
            "ci_low": round(self.ci_low, 8),
            "ci_high": round(self.ci_high, 8),
            "n_bootstrap": self.n_bootstrap,
            "n_groups": self.n_groups,
            "n_samples": self.n_samples,
            "epsilon": round(self.epsilon, 8),
            "passes_threshold": self.passes_threshold,
        }


def grouped_paired_bootstrap(
    deltas: list[float],
    groups: list[str],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    epsilon: float = 0.0,
    seed: int = 42,
) -> GroupedBootstrapResult:
    """Paired stratified bootstrap by task group (Section 10.3).

    Resamples GROUPS (with replacement), not individual action rows. All
    deltas belonging to a resampled group are included together. This
    preserves within-group correlation structure.

    Args:
        deltas: per-task paired differences.
        groups: per-task group id (same length as deltas).
        n_bootstrap: at least 10,000 (Section 10.3).
        confidence: 0.95 two-sided.
        epsilon: practical improvement threshold; passes_threshold is True
            iff ci_low > epsilon.
        seed: fixed seed recorded in provenance.
    """
    if not deltas:
        return GroupedBootstrapResult(
            mean=0.0, median=0.0, std_error=0.0,
            ci_low=0.0, ci_high=0.0,
            n_bootstrap=n_bootstrap, n_groups=0, n_samples=0,
            epsilon=epsilon, passes_threshold=False,
        )
    if len(deltas) != len(groups):
        raise ValueError(
            f"deltas ({len(deltas)}) and groups ({len(groups)}) must be equal length"
        )

    # Group deltas by group id.
    group_to_deltas: dict[str, list[float]] = {}
    for d, g in zip(deltas, groups):
        group_to_deltas.setdefault(g, []).append(d)
    unique_groups = list(group_to_deltas.keys())
    n_groups = len(unique_groups)

    rng = random.Random(seed)
    observed_mean = sum(deltas) / len(deltas)

    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        # Resample groups with replacement.
        sampled_deltas: list[float] = []
        for _ in range(n_groups):
            g = unique_groups[rng.randrange(n_groups)]
            sampled_deltas.extend(group_to_deltas[g])
        boot_means.append(sum(sampled_deltas) / len(sampled_deltas))

    boot_means.sort()
    alpha = 1.0 - confidence
    idx_low = max(0, min(n_bootstrap - 1, int((alpha / 2) * n_bootstrap)))
    idx_high = max(0, min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap)))
    ci_low = boot_means[idx_low]
    ci_high = boot_means[idx_high]
    median = boot_means[n_bootstrap // 2]
    std_error = math.sqrt(
        sum((bm - observed_mean) ** 2 for bm in boot_means) / n_bootstrap
    )
    passes = ci_low > epsilon

    return GroupedBootstrapResult(
        mean=observed_mean,
        median=median,
        std_error=std_error,
        ci_low=ci_low,
        ci_high=ci_high,
        n_bootstrap=n_bootstrap,
        n_groups=n_groups,
        n_samples=len(deltas),
        epsilon=epsilon,
        passes_threshold=passes,
        bootstrap_samples=boot_means,
    )


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction (Section 10.5)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HolmCorrectionResult:
    """Result of Holm-Bonferroni multiple-comparison correction."""
    n_tests: int
    corrected: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tests": self.n_tests,
            "corrected": list(self.corrected),
        }


def holm_bonferroni(p_values: list[float], *, alpha: float = 0.05) -> HolmCorrectionResult:
    """Apply Holm-Bonferroni step-down correction.

    Returns per-test adjusted p-values and reject flags.
    """
    n = len(p_values)
    indexed = list(enumerate(p_values))
    indexed.sort(key=lambda x: x[1])  # ascending p
    corrected: list[dict[str, Any]] = []
    rejected = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_p = min(1.0, p * (n - rank))
        reject = adjusted_p <= alpha
        corrected.append({
            "test_index": orig_idx,
            "raw_p_value": p,
            "adjusted_p_value": adjusted_p,
            "rejected": reject,
        })
    # Holm step-down: once we fail to reject, all subsequent (larger p) also fail.
    stopped = False
    for entry in corrected:
        if stopped:
            entry["rejected"] = False
        elif not entry["rejected"]:
            stopped = True
    for entry in corrected:
        if entry["rejected"]:
            rejected[entry["test_index"]] = True
    return HolmCorrectionResult(n_tests=n, corrected=corrected)


# ---------------------------------------------------------------------------
# Gate B metrics (Section 12.11)
# ---------------------------------------------------------------------------


def auroc(scores: list[float], labels: list[int]) -> float:
    """Area under the ROC curve (trapezoidal Mann-Whitney U).

    labels: 1 = success, 0 = failure. Higher score => more likely success.
    """
    if not scores or len(scores) != len(labels):
        return 0.5
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    # Rank-sum approach with tie handling.
    ranks: list[float] = []
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # average of ranks i+1..j
        for k in range(i, j):
            ranks.append(avg_rank)
        i = j
    sum_ranks_pos = sum(r for (s, l), r in zip(paired, ranks) if l == 1)
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auroc_val = u / (n_pos * n_neg)
    return auroc_val


def auprc(scores: list[float], labels: list[int]) -> float:
    """Area under the precision-recall curve (trapezoidal)."""
    if not scores or len(scores) != len(labels):
        return 0.0
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(1 for l in labels if l == 1)
    if n_pos == 0:
        return 0.0
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for s, l in paired:
        if l == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return area


def brier_score(scores: list[float], labels: list[int]) -> float:
    """Mean squared error between predicted probabilities and labels."""
    if not scores:
        return 1.0
    return sum((s - l) ** 2 for s, l in zip(scores, labels)) / len(scores)


def expected_calibration_error(
    scores: list[float], labels: list[int], *, n_bins: int = 10
) -> float:
    """Expected Calibration Error (Section 12.11)."""
    if not scores:
        return 1.0
    bins = [[] for _ in range(n_bins)]
    for s, l in zip(scores, labels):
        idx = min(n_bins - 1, int(s * n_bins))
        bins[idx].append((s, l))
    ece = 0.0
    n = len(scores)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(s for s, _ in b) / len(b)
        avg_acc = sum(l for _, l in b) / len(b)
        ece += (len(b) / n) * abs(avg_conf - avg_acc)
    return ece


def balanced_accuracy(scores: list[float], labels: list[int], *, threshold: float = 0.5) -> float:
    """Balanced accuracy at a threshold."""
    if not scores:
        return 0.0
    tp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 1)
    tn = sum(1 for s, l in zip(scores, labels) if s < threshold and l == 0)
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    tpr = tp / n_pos if n_pos else 0.0
    tnr = tn / n_neg if n_neg else 0.0
    return (tpr + tnr) / 2.0


def bootstrap_auroc_ci(
    scores: list[float],
    labels: list[int],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for AUROC. Returns (point_estimate, ci_low, ci_high)."""
    if not scores:
        return 0.5, 0.5, 0.5
    point = auroc(scores, labels)
    n = len(scores)
    rng = random.Random(seed)
    boot_aurocs: list[float] = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        bs = [scores[i] for i in idx]
        bl = [labels[i] for i in idx]
        boot_aurocs.append(auroc(bs, bl))
    boot_aurocs.sort()
    alpha = 1.0 - confidence
    ci_low = boot_aurocs[max(0, min(n_bootstrap - 1, int((alpha / 2) * n_bootstrap)))]
    ci_high = boot_aurocs[max(0, min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap)))]
    return point, ci_low, ci_high
