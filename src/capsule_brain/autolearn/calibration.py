"""Calibration metrics for AutoLearn v0.2.

Computes Brier score, Expected Calibration Error (ECE), accuracy-at-coverage,
and abstention rate from the held-out evaluation. These are reported
alongside the promotion gate so a candidate that is accurate but
over-confident (or under-confident) is visible before promotion.

All metrics are computed from the (confidence, correct) pairs produced by
the evaluator. ``correct`` is True only when the verifier passed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CalibrationReport:
    """Calibration metrics for a set of (confidence, correct) pairs."""

    brier_score: float = 0.0
    ece: float = 0.0  # Expected Calibration Error
    n_bins: int = 10
    abstention_rate: float = 0.0
    accuracy_at_coverage: dict[str, float] = field(default_factory=dict)
    n: int = 0
    n_abstained: int = 0
    bins: list[dict[str, Any]] = field(default_factory=list)
    mean_confidence: float = 0.0
    mean_accuracy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "brier_score": self.brier_score,
            "ece": self.ece,
            "n_bins": self.n_bins,
            "abstention_rate": self.abstention_rate,
            "accuracy_at_coverage": dict(self.accuracy_at_coverage),
            "n": self.n,
            "n_abstained": self.n_abstained,
            "bins": list(self.bins),
            "mean_confidence": self.mean_confidence,
            "mean_accuracy": self.mean_accuracy,
        }


def compute_calibration(
    pairs: list[tuple[float, bool]],
    *,
    abstained: list[bool] | None = None,
    n_bins: int = 10,
    coverage_levels: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95),
) -> CalibrationReport:
    """Compute calibration metrics from (confidence, correct) pairs.

    Args:
        pairs: list of (confidence, correct) where confidence is the
            policy's confidence in its chosen action and correct is whether
            the verifier passed.
        abstained: parallel list of abstention flags. If None, no
            abstentions are assumed.
        n_bins: number of equal-width confidence bins for ECE.
        coverage_levels: confidence thresholds for accuracy-at-coverage.

    Returns:
        CalibrationReport with Brier, ECE, abstention rate, and
        accuracy-at-coverage.
    """
    if not pairs:
        return CalibrationReport(n_bins=n_bins)

    abst_flags = abstained or [False] * len(pairs)
    n_total = len(pairs)
    n_abstained = sum(1 for a in abst_flags if a)
    abstention_rate = n_abstained / n_total if n_total > 0 else 0.0

    # Non-abstained pairs for Brier/ECE.
    active_pairs = [
        (conf, correct) for (conf, correct), abst in zip(pairs, abst_flags) if not abst
    ]
    n_active = len(active_pairs)
    if n_active == 0:
        return CalibrationReport(
            n_bins=n_bins,
            abstention_rate=abstention_rate,
            n=n_total,
            n_abstained=n_abstained,
        )

    confidences = [p[0] for p in active_pairs]
    corrects = [1.0 if p[1] else 0.0 for p in active_pairs]
    mean_conf = sum(confidences) / n_active
    mean_acc = sum(corrects) / n_active

    # Brier score: mean((confidence - correct)^2).
    brier = sum((c - r) ** 2 for c, r in zip(confidences, corrects)) / n_active

    # ECE: sum over bins of (|bin|/n) * |mean_conf_bin - mean_acc_bin|.
    bin_size = 1.0 / n_bins
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for b in range(n_bins):
        lo = b * bin_size
        hi = (b + 1) * bin_size
        in_bin = [
            (c, r) for c, r in zip(confidences, corrects)
            if lo <= c < hi or (b == n_bins - 1 and c == hi)
        ]
        if not in_bin:
            bins.append({
                "bin_lo": round(lo, 4),
                "bin_hi": round(hi, 4),
                "count": 0,
                "mean_confidence": 0.0,
                "mean_accuracy": 0.0,
                "gap": 0.0,
            })
            continue
        bc = [p[0] for p in in_bin]
        br = [p[1] for p in in_bin]
        mean_conf_bin = sum(bc) / len(bc)
        mean_acc_bin = sum(br) / len(br)
        gap = abs(mean_conf_bin - mean_acc_bin)
        ece += (len(in_bin) / n_active) * gap
        bins.append({
            "bin_lo": round(lo, 4),
            "bin_hi": round(hi, 4),
            "count": len(in_bin),
            "mean_confidence": round(mean_conf_bin, 4),
            "mean_accuracy": round(mean_acc_bin, 4),
            "gap": round(gap, 4),
        })

    # Accuracy-at-coverage: for each coverage level, compute accuracy on
    # the subset of predictions with confidence >= coverage level.
    accuracy_at_coverage: dict[str, float] = {}
    for cov in coverage_levels:
        subset = [r for c, r in zip(confidences, corrects) if c >= cov]
        if subset:
            accuracy_at_coverage[f"cov_{cov:.2f}"] = sum(subset) / len(subset)
        else:
            accuracy_at_coverage[f"cov_{cov:.2f}"] = 0.0

    return CalibrationReport(
        brier_score=brier,
        ece=ece,
        n_bins=n_bins,
        abstention_rate=abstention_rate,
        accuracy_at_coverage=accuracy_at_coverage,
        n=n_total,
        n_abstained=n_abstained,
        bins=bins,
        mean_confidence=mean_conf,
        mean_accuracy=mean_acc,
    )
