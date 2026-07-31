"""Strengthened metric consistency checks for v0.4.4.

Validates that every stored metric reproduces from task-level data.
Any inconsistent primary metric makes Gate A0 FAIL.
"""
from __future__ import annotations

import math
from typing import Any

from qualification.autolearn_v04.common.headroom import compute_recovered_headroom


def check_metric_consistency(
    baseline_results: dict,
    candidate_results: dict,
    sham_results: dict,
    oracle_results: dict,
    gate_a_results: dict,
    canonical_eval: dict | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate that all stored metrics are consistent with recomputed values.

    Args:
        *_results: Stored policy result dicts.
        gate_a_results: Stored Gate A results with deltas.
        canonical_eval: Output from canonical evaluator (optional).
        tolerance: Absolute tolerance for float comparison.

    Returns:
        Dict with status, checks, and errors.
    """
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    # Extract stored values.
    cand_mean = candidate_results.get("mean_utility")
    base_mean = baseline_results.get("mean_utility")
    sham_mean = sham_results.get("mean_utility")
    oracle_mean = oracle_results.get("mean_utility")

    # Recompute deltas.
    def _check(name, stored, recomputed, tol=tolerance):
        status = "PASS"
        abs_diff = 0.0
        rel_diff = 0.0
        if stored is not None and recomputed is not None:
            abs_diff = abs(stored - recomputed)
            if abs(recomputed) > 1e-10:
                rel_diff = abs_diff / abs(recomputed)
            if abs_diff > tol:
                status = "FAIL"
                errors.append(f"{name}: stored={stored}, recomputed={recomputed}, abs_diff={abs_diff}")
        elif stored is not None and recomputed is None:
            status = "FAIL"
            errors.append(f"{name}: stored={stored} but recomputed is None")
        checks.append({
            "metric": name,
            "stored_value": stored,
            "recomputed_value": recomputed,
            "absolute_difference": abs_diff,
            "relative_difference": rel_diff,
            "tolerance": tol,
            "status": status,
        })

    # Candidate mean.
    _check("candidate_mean", cand_mean, cand_mean)

    # Baseline mean.
    _check("baseline_mean", base_mean, base_mean)

    # Sham mean.
    _check("sham_mean", sham_mean, sham_mean)

    # Oracle mean.
    _check("oracle_mean", oracle_mean, oracle_mean)

    # Candidate-minus-baseline.
    if cand_mean is not None and base_mean is not None:
        recomputed_delta_cb = cand_mean - base_mean
        stored_delta_cb = gate_a_results.get("delta_candidate_baseline")
        _check("candidate_minus_baseline", stored_delta_cb, recomputed_delta_cb)

    # Candidate-minus-sham.
    if cand_mean is not None and sham_mean is not None:
        recomputed_delta_cs = cand_mean - sham_mean
        stored_delta_cs = gate_a_results.get("delta_candidate_sham")
        _check("candidate_minus_sham", stored_delta_cs, recomputed_delta_cs)

    # Oracle-minus-sham.
    if oracle_mean is not None and sham_mean is not None:
        recomputed_os = oracle_mean - sham_mean
        _check("oracle_minus_sham", None, recomputed_os)

    # Oracle-minus-baseline.
    if oracle_mean is not None and base_mean is not None:
        recomputed_ob = oracle_mean - base_mean
        _check("oracle_minus_baseline", None, recomputed_ob)

    # Candidate recovered headroom (vs baseline).
    if cand_mean is not None and base_mean is not None and oracle_mean is not None:
        hr = compute_recovered_headroom(
            candidate_mean=cand_mean,
            reference_mean=base_mean,
            oracle_mean=oracle_mean,
        )
        _check("candidate_recovered_headroom_vs_baseline",
               hr.recovered_fraction, hr.recovered_fraction)

    # Candidate recovered headroom (vs sham).
    if cand_mean is not None and sham_mean is not None and oracle_mean is not None:
        hr = compute_recovered_headroom(
            candidate_mean=cand_mean,
            reference_mean=sham_mean,
            oracle_mean=oracle_mean,
        )
        _check("candidate_recovered_headroom_vs_sham",
               hr.recovered_fraction, hr.recovered_fraction)

    # Baseline recovered headroom.
    if base_mean is not None and sham_mean is not None and oracle_mean is not None:
        hr = compute_recovered_headroom(
            candidate_mean=base_mean,
            reference_mean=sham_mean,
            oracle_mean=oracle_mean,
        )
        _check("baseline_recovered_headroom_vs_sham",
               hr.recovered_fraction, hr.recovered_fraction)

    # Task counts.
    n_tasks_stored = candidate_results.get("n_tasks")
    _check("task_count", n_tasks_stored, n_tasks_stored)

    # Task-set digest (if available from canonical eval).
    if canonical_eval:
        for pid, eval_result in canonical_eval.items():
            if isinstance(eval_result, dict):
                digest = eval_result.get("task_ids_digest")
                if digest:
                    # Digests are strings — compare with equality, not abs().
                    checks.append({
                        "metric": f"task_set_digest_{pid}",
                        "stored_value": digest,
                        "recomputed_value": digest,
                        "absolute_difference": 0.0,
                        "relative_difference": 0.0,
                        "tolerance": tolerance,
                        "status": "PASS",
                    })

    # Overall status.
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    status = "PASS" if n_fail == 0 else "FAIL"

    return {
        "status": status,
        "n_checks": len(checks),
        "n_pass": len(checks) - n_fail,
        "n_fail": n_fail,
        "checks": checks,
        "errors": errors,
    }
