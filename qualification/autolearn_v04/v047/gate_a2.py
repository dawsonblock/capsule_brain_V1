"""Gate A2 — Candidate causal effectiveness (v0.4.7).

Evaluates whether the candidate policy demonstrates causal effectiveness
by beating BOTH the baseline and the sham with a lower confidence bound
that exceeds the configured practical-effect thresholds.

Pass criteria (strict, LCB-based):
  1. LCB(delta_candidate_vs_baseline) > epsilon_0
  2. LCB(delta_candidate_vs_sham) > epsilon_S

A zero or positive point estimate must NOT pass when the lower
confidence bound does not exceed the threshold.
"""
from __future__ import annotations

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.v047.config import GateA2Config, StatisticsConfig
from qualification.autolearn_v04.v047.gate_schema import ComparisonResult, GateA2Result
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


def evaluate_gate_a2(
    candidate_results: dict,
    baseline_results: dict,
    sham_results: dict,
    config: GateA2Config,
    statistics_config: StatisticsConfig,
    matched_pair_flip: dict | None = None,
) -> GateA2Result:
    """Evaluate candidate causal effectiveness.

    The candidate must beat BOTH baseline and sham. Passes only when:
      - LCB(delta_candidate_vs_baseline) > epsilon_0
      - LCB(delta_candidate_vs_sham) > epsilon_S

    Additionally, if matched-pair flip data is provided, the candidate
    must achieve flip accuracy > 0.5 (the policy must pick different
    actions for matched pairs where the optimal route flips).

    Do NOT permit a zero or positive point estimate to pass when the
    lower confidence bound does not exceed the threshold.
    """
    deltas = compute_paired_deltas(
        candidate_results=candidate_results,
        baseline_results=baseline_results,
        sham_results=sham_results,
        oracle_results=[],
    )
    cand_vs_base_rows = deltas["candidate_vs_baseline"]
    cand_vs_sham_rows = deltas["candidate_vs_sham"]

    reasons: list[str] = []

    # Candidate vs baseline.
    cb_stats = paired_cluster_bootstrap(
        cand_vs_base_rows,
        cluster_key=statistics_config.cluster_key,
        n_resamples=statistics_config.bootstrap_resamples,
        confidence_level=config.confidence_level,
        seed=42,
    )
    cb_threshold = config.candidate_vs_baseline_min_effect
    cb_result = _comparison_to_result(
        cb_stats,
        comparison_name="candidate_vs_baseline",
        practical_threshold=cb_threshold,
    )
    if not cb_result.passes:
        reasons.append(
            f"candidate_vs_baseline LCB {cb_result.lower_bound:.6f} "
            f"does not exceed epsilon_0 {cb_threshold:.6f}"
        )

    # Candidate vs sham.
    cs_stats = paired_cluster_bootstrap(
        cand_vs_sham_rows,
        cluster_key=statistics_config.cluster_key,
        n_resamples=statistics_config.bootstrap_resamples,
        confidence_level=config.confidence_level,
        seed=42,
    )
    cs_threshold = config.candidate_vs_sham_min_effect
    cs_result = _comparison_to_result(
        cs_stats,
        comparison_name="candidate_vs_sham",
        practical_threshold=cs_threshold,
    )
    if not cs_result.passes:
        reasons.append(
            f"candidate_vs_sham LCB {cs_result.lower_bound:.6f} "
            f"does not exceed epsilon_S {cs_threshold:.6f}"
        )

    # Matched-pair flip accuracy check.
    flip_data: dict = {}
    flip_ok = True
    if matched_pair_flip is not None:
        flip_data = matched_pair_flip
        flip_acc = matched_pair_flip.get("flip_accuracy", 0.0)
        flip_threshold = matched_pair_flip.get("threshold", 0.5)
        flip_ok = flip_acc > flip_threshold
        if not flip_ok:
            reasons.append(
                f"matched_pair_flip accuracy {flip_acc:.4f} "
                f"does not exceed threshold {flip_threshold:.4f}"
            )

    # Gate passes only when all comparisons pass.
    if not cand_vs_base_rows and not cand_vs_sham_rows:
        status = AuditStatus.BLOCKED.value
        reasons.append("no paired task rows available for candidate comparisons")
    elif cb_result.passes and cs_result.passes and flip_ok:
        status = AuditStatus.PASS.value
    else:
        status = AuditStatus.FAIL.value

    return GateA2Result(
        status=status,
        candidate_vs_baseline=cb_result.to_dict(),
        candidate_vs_sham=cs_result.to_dict(),
        matched_pair_flip=flip_data,
        reasons=reasons,
    )


__all__ = ["evaluate_gate_a2"]
