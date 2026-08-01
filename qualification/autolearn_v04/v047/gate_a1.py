"""Gate A1 — Routing headroom (v0.4.7).

Evaluates whether the benchmark offers sufficient routing headroom by
comparing the ORACLE against the baseline and the sham. The oracle is
the upper-bound policy that always selects the best available action;
if it cannot beat baseline/sham by a practically meaningful margin, the
benchmark itself lacks routing signal and downstream candidate gates
cannot be interpreted.

A Gate A1 failure means the benchmark offers insufficient routing
headroom. It does NOT mean the learner is defective.
"""
from __future__ import annotations

from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.v047.config import GateA1Config, StatisticsConfig
from qualification.autolearn_v04.v047.gate_schema import ComparisonResult, GateA1Result
from qualification.autolearn_v04.v047.statistics import (
    compute_paired_deltas,
    paired_cluster_bootstrap,
)


def _comparison_to_result(
    stats: dict,
    *,
    comparison_name: str,
    practical_threshold: float,
) -> ComparisonResult:
    """Promote a bootstrap stats dict into a ComparisonResult.

    The strict pass rule (``lower_bound > practical_threshold``) is
    recomputed here using the gate-specific threshold so that a zero or
    positive point estimate cannot pass when the lower confidence bound
    does not exceed the threshold.
    """
    passes = stats["lower_bound"] > practical_threshold
    return ComparisonResult(
        comparison=comparison_name,
        n_tasks=stats["n_tasks"],
        n_task_groups=stats["n_task_groups"],
        mean_delta=stats["mean_delta"],
        median_delta=stats["median_delta"],
        standard_error=stats["standard_error"],
        confidence_level=stats["confidence_level"],
        lower_bound=stats["lower_bound"],
        upper_bound=stats["upper_bound"],
        practical_threshold=practical_threshold,
        passes=passes,
        win_rate=stats["win_rate"],
        tie_rate=stats["tie_rate"],
        loss_rate=stats["loss_rate"],
    )


def evaluate_gate_a1(
    oracle_results: dict,
    baseline_results: dict,
    sham_results: dict,
    config: GateA1Config,
    statistics_config: StatisticsConfig,
) -> GateA1Result:
    """Evaluate routing headroom.

    Compares oracle against baseline and sham. Passes only when the LCB
    of each comparison exceeds the configured practical-effect threshold.

    A Gate A1 failure means the benchmark offers insufficient routing
    headroom. It does NOT mean the learner is defective.
    """
    # Build paired rows for the two oracle comparisons.
    deltas = compute_paired_deltas(
        candidate_results=[],
        baseline_results=baseline_results,
        sham_results=sham_results,
        oracle_results=oracle_results,
    )
    oracle_vs_baseline_rows = deltas["oracle_vs_baseline"]
    oracle_vs_sham_rows = deltas["oracle_vs_sham"]

    reasons: list[str] = []

    # Oracle vs baseline.
    ob_stats = paired_cluster_bootstrap(
        oracle_vs_baseline_rows,
        cluster_key=statistics_config.cluster_key,
        n_resamples=statistics_config.bootstrap_resamples,
        confidence_level=config.confidence_level,
        seed=42,
    )
    ob_threshold = config.oracle_vs_baseline_min_effect
    ob_result = _comparison_to_result(
        ob_stats,
        comparison_name="oracle_vs_baseline",
        practical_threshold=ob_threshold,
    )
    if not ob_result.passes:
        reasons.append(
            f"oracle_vs_baseline LCB {ob_result.lower_bound:.6f} "
            f"does not exceed threshold {ob_threshold:.6f}"
        )

    # Oracle vs sham.
    os_stats = paired_cluster_bootstrap(
        oracle_vs_sham_rows,
        cluster_key=statistics_config.cluster_key,
        n_resamples=statistics_config.bootstrap_resamples,
        confidence_level=config.confidence_level,
        seed=42,
    )
    os_threshold = config.oracle_vs_sham_min_effect
    os_result = _comparison_to_result(
        os_stats,
        comparison_name="oracle_vs_sham",
        practical_threshold=os_threshold,
    )
    if not os_result.passes:
        reasons.append(
            f"oracle_vs_sham LCB {os_result.lower_bound:.6f} "
            f"does not exceed threshold {os_threshold:.6f}"
        )

    # Gate passes only when both comparisons pass.
    if not oracle_vs_baseline_rows and not oracle_vs_sham_rows:
        status = AuditStatus.BLOCKED.value
        reasons.append("no paired task rows available for oracle comparisons")
    elif ob_result.passes and os_result.passes:
        status = AuditStatus.PASS.value
    else:
        status = AuditStatus.FAIL.value

    return GateA1Result(
        status=status,
        oracle_vs_baseline=ob_result.to_dict(),
        oracle_vs_sham=os_result.to_dict(),
        reasons=reasons,
    )


__all__ = ["evaluate_gate_a1"]
