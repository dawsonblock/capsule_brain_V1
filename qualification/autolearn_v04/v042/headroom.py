"""Oracle routing headroom analysis.

Computes oracle-sham and oracle-baseline headroom, and the fraction
of headroom recovered by the candidate.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from .data import LoadedRun
from .bootstrap import task_paired_bootstrap, bootstrap_ratio
from .policy_controls import (
    make_oracle_selector,
    make_baseline_selector,
    make_candidate_selector,
    make_sham_global_selector,
    evaluate_policy_on_tasks,
    make_frequency_exact_utility,
)


def compute_headroom(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute oracle routing headroom and candidate recovery.

    Definitions:
        U_oracle   = mean max utility per task
        U_sham     = mean sham utility
        U_baseline = mean baseline utility
        U_candidate = mean candidate utility
        H_oracle_sham = U_oracle - U_sham
        H_oracle_baseline = U_oracle - U_baseline
        R_sham = (U_candidate - U_sham) / (U_oracle - U_sham)
        R_baseline = (U_candidate - U_baseline) / (U_oracle - U_baseline)
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # Compute per-task utilities for each policy.
    oracle_sel = make_oracle_selector(run)
    baseline_sel = make_baseline_selector(run)
    candidate_sel = make_candidate_selector(run)
    sham_sel = make_sham_global_selector(run, seed=12345)

    oracle_res = evaluate_policy_on_tasks(run, task_ids, oracle_sel, "oracle")
    baseline_res = evaluate_policy_on_tasks(run, task_ids, baseline_sel, "baseline")
    candidate_res = evaluate_policy_on_tasks(run, task_ids, candidate_sel, "candidate")
    sham_res = evaluate_policy_on_tasks(run, task_ids, sham_sel, "sham")

    u_oracle = np.array(oracle_res["utilities"])
    u_baseline = np.array(baseline_res["utilities"])
    u_candidate = np.array(candidate_res["utilities"])
    u_sham = np.array(sham_res["utilities"])

    # Also compute frequency exact.
    u_freq = make_frequency_exact_utility(run, task_ids)

    # Headroom.
    H_oracle_sham = float(u_oracle.mean() - u_sham.mean())
    H_oracle_baseline = float(u_oracle.mean() - u_baseline.mean())
    H_oracle_freq = float(u_oracle.mean() - u_freq.mean())

    # Recovery fractions.
    denom_sham = u_oracle.mean() - u_sham.mean()
    denom_baseline = u_oracle.mean() - u_baseline.mean()
    R_sham = float((u_candidate.mean() - u_sham.mean()) / denom_sham) if abs(denom_sham) > 1e-10 else 0.0
    R_baseline = float((u_candidate.mean() - u_baseline.mean()) / denom_baseline) if abs(denom_baseline) > 1e-10 else 0.0

    # Bootstrap CIs.
    ci_oracle_sham = task_paired_bootstrap(u_oracle - u_sham, n_bootstrap=n_bootstrap, seed=seed)
    ci_oracle_baseline = task_paired_bootstrap(u_oracle - u_baseline, n_bootstrap=n_bootstrap, seed=seed)
    ci_oracle_freq = task_paired_bootstrap(u_oracle - u_freq, n_bootstrap=n_bootstrap, seed=seed)
    ci_candidate_sham = task_paired_bootstrap(u_candidate - u_sham, n_bootstrap=n_bootstrap, seed=seed)
    ci_candidate_baseline = task_paired_bootstrap(u_candidate - u_baseline, n_bootstrap=n_bootstrap, seed=seed)

    # Recovery ratio CIs.
    ci_R_sham = bootstrap_ratio(u_candidate - u_sham, denom_sham, n_bootstrap=n_bootstrap, seed=seed)
    ci_R_baseline = bootstrap_ratio(u_candidate - u_baseline, denom_baseline, n_bootstrap=n_bootstrap, seed=seed)

    # Headroom status classification.
    # Predeclared thresholds:
    #   H_oracle_sham >= 0.50: sufficient
    #   0.10 <= H_oracle_sham < 0.50: marginal
    #   H_oracle_sham < 0.10: insufficient
    if H_oracle_sham >= 0.50:
        headroom_status = "SUFFICIENT"
    elif H_oracle_sham >= 0.10:
        headroom_status = "MARGINAL"
    elif ci_oracle_sham["ci_low"] <= 0:
        headroom_status = "INCONCLUSIVE"
    else:
        headroom_status = "INSUFFICIENT"

    return {
        "n_tasks": len(task_ids),
        "U_oracle": float(u_oracle.mean()),
        "U_sham": float(u_sham.mean()),
        "U_baseline": float(u_baseline.mean()),
        "U_candidate": float(u_candidate.mean()),
        "U_frequency": float(u_freq.mean()),
        "H_oracle_sham": H_oracle_sham,
        "H_oracle_baseline": H_oracle_baseline,
        "H_oracle_freq": H_oracle_freq,
        "R_sham": R_sham,
        "R_baseline": R_baseline,
        "ci_oracle_sham": ci_oracle_sham,
        "ci_oracle_baseline": ci_oracle_baseline,
        "ci_oracle_freq": ci_oracle_freq,
        "ci_candidate_sham": ci_candidate_sham,
        "ci_candidate_baseline": ci_candidate_baseline,
        "ci_R_sham": ci_R_sham,
        "ci_R_baseline": ci_R_baseline,
        "routing_headroom_status": headroom_status,
        "thresholds": {
            "sufficient": 0.50,
            "marginal_min": 0.10,
        },
        "interpretation": _interpret_headroom(headroom_status, H_oracle_sham, R_sham, R_baseline),
    }


def _interpret_headroom(status: str, H: float, R_sham: float, R_baseline: float) -> str:
    if status == "INSUFFICIENT":
        return (
            f"The measured oracle provided little utility advantage over the "
            f"frequency sham (H_oracle_sham={H:.3f}). The current benchmark and "
            f"action conditions do not contain enough route-dependent headroom "
            f"to evaluate learned routing reliably."
        )
    if status == "MARGINAL":
        if R_sham <= 0:
            return (
                f"Oracle headroom is marginal (H={H:.3f}) and the candidate "
                f"recovered none of it (R_sham={R_sham:.3f}). The learner or "
                f"features fail to capture available routing signal."
            )
        return (
            f"Oracle headroom is marginal (H={H:.3f}). Candidate recovered "
            f"R_sham={R_sham:.3f}. More data or better features may help."
        )
    if status == "SUFFICIENT":
        if R_sham <= 0:
            return (
                f"Oracle materially outperforms sham (H={H:.3f}) but candidate "
                f"recovered little or none (R_sham={R_sham:.3f}). The current "
                f"executive features, learning objective or learner did not "
                f"capture the route-dependent signal."
            )
        return (
            f"Oracle headroom is sufficient (H={H:.3f}) and candidate recovered "
            f"R_sham={R_sham:.3f}. Routing signal exists and is partially learned."
        )
    return f"Headroom status inconclusive (H={H:.3f})."
