"""Layered Gate A sub-gates for v0.4.2.

Decomposes Gate A into diagnostic sub-gates:
  A0 — Infrastructure validity
  A1 — Routing headroom
  A2 — Learnability
  A3 — Candidate versus baseline
  A4 — Candidate versus sham
  A5 — High-margin recovery

Final Gate A PASS requires all of A0-A5.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from .data import LoadedRun
from .bootstrap import task_paired_bootstrap
from .policy_controls import (
    evaluate_policy_on_tasks,
    make_oracle_selector,
    make_baseline_selector,
    make_candidate_selector,
    make_sham_global_selector,
    make_majority_selector,
    make_frequency_selector,
    make_frequency_exact_utility,
)


def evaluate_gate_a_layers(
    run: LoadedRun,
    headroom_analysis: dict[str, Any] | None = None,
    margin_analysis: dict[str, Any] | None = None,
    feature_audit: dict[str, Any] | None = None,
    task_ids: list[str] | None = None,
    epsilon_baseline: float = 0.01,
    epsilon_sham: float = 0.01,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate the layered Gate A sub-gates.

    Args:
        run: loaded run data.
        headroom_analysis: output from headroom.compute_headroom().
        margin_analysis: output from margin_analysis.analyze_margins().
        feature_audit: output from feature_audit.audit_features().
        task_ids: tasks to evaluate on (default: test split).
        epsilon_baseline: practical threshold for candidate-baseline.
        epsilon_sham: practical threshold for candidate-sham.
        n_bootstrap: bootstrap iterations.
        seed: random seed.

    Returns:
        dict with A0-A5 sub-gate results and overall Gate A decision.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # Evaluate all policies.
    oracle_sel = make_oracle_selector(run)
    baseline_sel = make_baseline_selector(run)
    candidate_sel = make_candidate_selector(run)
    sham_sel = make_sham_global_selector(run, seed=12345)
    majority_sel = make_majority_selector(run)
    freq_sel = make_frequency_selector(run, seed=seed)

    oracle_res = evaluate_policy_on_tasks(run, task_ids, oracle_sel, "oracle")
    baseline_res = evaluate_policy_on_tasks(run, task_ids, baseline_sel, "baseline")
    candidate_res = evaluate_policy_on_tasks(run, task_ids, candidate_sel, "candidate")
    sham_res = evaluate_policy_on_tasks(run, task_ids, sham_sel, "sham")
    majority_res = evaluate_policy_on_tasks(run, task_ids, majority_sel, "majority")
    freq_res = evaluate_policy_on_tasks(run, task_ids, freq_sel, "frequency")

    u_oracle = np.array(oracle_res["utilities"])
    u_baseline = np.array(baseline_res["utilities"])
    u_candidate = np.array(candidate_res["utilities"])
    u_sham = np.array(sham_res["utilities"])
    u_majority = np.array(majority_res["utilities"])
    u_freq_exact = make_frequency_exact_utility(run, task_ids)

    gates = {}

    # A0 — Infrastructure validity
    gates["A0_infrastructure_validity"] = _gate_a0(run)

    # A1 — Routing headroom
    gates["A1_routing_headroom"] = _gate_a1(
        u_oracle, u_sham, u_freq_exact, headroom_analysis, seed)

    # A2 — Learnability
    gates["A2_learnability"] = _gate_a2(
        run, u_candidate, u_majority, u_freq_exact, feature_audit, seed)

    # A3 — Candidate versus baseline
    gates["A3_candidate_vs_baseline"] = _gate_a3(
        u_candidate, u_baseline, epsilon_baseline, n_bootstrap, seed)

    # A4 — Candidate versus sham
    gates["A4_candidate_vs_sham"] = _gate_a4(
        u_candidate, u_sham, epsilon_sham, n_bootstrap, seed)

    # A5 — High-margin recovery
    gates["A5_high_margin_recovery"] = _gate_a5(
        run, margin_analysis, task_ids, n_bootstrap, seed)

    # Overall decision.
    all_pass = all(g.get("status") == "PASS" for g in gates.values())
    any_blocked = any(g.get("status") == "BLOCKED" for g in gates.values())
    any_fail = any(g.get("status") == "FAIL" for g in gates.values())

    if all_pass:
        overall = "PASS"
    elif any_blocked:
        overall = "BLOCKED"
    elif any_fail:
        overall = "FAIL"
    else:
        overall = "INCONCLUSIVE"

    return {
        "gate_a_layered": {
            "overall_status": overall,
            "sub_gates": gates,
            "requirements": (
                "Final Gate A PASS requires: A0, A1, A2, A3, A4, A5."
            ),
            "epsilon_baseline": epsilon_baseline,
            "epsilon_sham": epsilon_sham,
        },
        "policy_utilities": {
            "oracle": float(u_oracle.mean()),
            "candidate": float(u_candidate.mean()),
            "baseline": float(u_baseline.mean()),
            "sham": float(u_sham.mean()),
            "majority": float(u_majority.mean()),
            "frequency_exact": float(u_freq_exact.mean()),
        },
    }


def _gate_a0(run: LoadedRun) -> dict[str, Any]:
    """A0 — Infrastructure validity."""
    checks = {}

    # Real provider.
    provider_class = run.qualification_report.get("provider", "unknown")
    checks["real_provider"] = {
        "passed": provider_class in ("modal_gpu", "local_transformers"),
        "value": provider_class,
    }

    # Valid counterfactuals.
    n_outcomes = len(run.outcomes)
    checks["valid_counterfactuals"] = {
        "passed": n_outcomes > 0,
        "value": n_outcomes,
    }

    # Valid provenance.
    has_provenance = bool(run.qualification_report.get("provenance"))
    checks["valid_provenance"] = {
        "passed": has_provenance,
        "value": has_provenance,
    }

    # Route completion.
    diag = run.runtime_diagnostics
    checks["route_completion"] = {
        "passed": diag.get("status") == "PASS",
        "value": diag.get("status"),
    }

    # No test leakage (safety tasks excluded from training).
    train_ids = {ex["task_id"] for ex in run.train_examples}
    test_ids = set(run.test_task_ids)
    leakage = train_ids & test_ids
    checks["no_test_leakage"] = {
        "passed": len(leakage) == 0,
        "value": len(leakage),
    }

    # Positive evidence weights.
    n_positive = sum(1 for ex in run.train_examples if ex.get("weight", 0) > 0)
    checks["positive_evidence_weights"] = {
        "passed": n_positive > 0,
        "value": n_positive,
    }

    all_pass = all(c["passed"] for c in checks.values())
    return {
        "status": "PASS" if all_pass else "BLOCKED",
        "checks": checks,
    }


def _gate_a1(
    u_oracle: np.ndarray,
    u_sham: np.ndarray,
    u_freq: np.ndarray,
    headroom_analysis: dict[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    """A1 — Routing headroom. Oracle must materially beat frequency sham."""
    H_oracle_sham = float(u_oracle.mean() - u_sham.mean())
    H_oracle_freq = float(u_oracle.mean() - u_freq.mean())

    ci_oracle_sham = task_paired_bootstrap(u_oracle - u_sham, seed=seed)
    ci_oracle_freq = task_paired_bootstrap(u_oracle - u_freq, seed=seed)

    # PASS requires oracle LCB > 0.10 over sham.
    threshold = 0.10
    passed = ci_oracle_sham["ci_low"] > threshold

    return {
        "status": "PASS" if passed else "FAIL",
        "H_oracle_sham": H_oracle_sham,
        "H_oracle_freq": H_oracle_freq,
        "ci_oracle_sham": ci_oracle_sham,
        "ci_oracle_freq": ci_oracle_freq,
        "threshold": threshold,
        "headroom_status": headroom_analysis.get("routing_headroom_status") if headroom_analysis else "UNKNOWN",
    }


def _gate_a2(
    run: LoadedRun,
    u_candidate: np.ndarray,
    u_majority: np.ndarray,
    u_freq: np.ndarray,
    feature_audit: dict[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    """A2 — Learnability. Features must predict best action beyond
    family/frequency controls on validation."""
    # Candidate must beat majority and frequency.
    ci_cand_majority = task_paired_bootstrap(u_candidate - u_majority, seed=seed)
    ci_cand_freq = task_paired_bootstrap(u_candidate - u_freq, seed=seed)

    beats_majority = ci_cand_majority["ci_low"] > 0
    beats_frequency = ci_cand_freq["ci_low"] > 0

    # Check feature signal beyond family.
    features_beyond_family = True
    if feature_audit:
        family_only = feature_audit.get("family_only_model", {}).get("test_utility")
        full_model = feature_audit.get("full_model", {}).get("test_utility")
        if family_only is not None and full_model is not None:
            features_beyond_family = full_model > family_only

    passed = beats_majority and beats_frequency and features_beyond_family

    return {
        "status": "PASS" if passed else "FAIL",
        "beats_majority": beats_majority,
        "beats_frequency": beats_frequency,
        "features_beyond_family": features_beyond_family,
        "ci_candidate_vs_majority": ci_cand_majority,
        "ci_candidate_vs_frequency": ci_cand_freq,
    }


def _gate_a3(
    u_candidate: np.ndarray,
    u_baseline: np.ndarray,
    epsilon: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """A3 — Candidate versus baseline. LCB95 > epsilon."""
    ci = task_paired_bootstrap(u_candidate - u_baseline, n_bootstrap=n_bootstrap, seed=seed)
    passed = ci["ci_low"] > epsilon
    return {
        "status": "PASS" if passed else "FAIL",
        "mean_delta": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "epsilon": epsilon,
    }


def _gate_a4(
    u_candidate: np.ndarray,
    u_sham: np.ndarray,
    epsilon: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """A4 — Candidate versus sham. LCB95 > epsilon."""
    ci = task_paired_bootstrap(u_candidate - u_sham, n_bootstrap=n_bootstrap, seed=seed)
    passed = ci["ci_low"] > epsilon
    return {
        "status": "PASS" if passed else "FAIL",
        "mean_delta": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "epsilon": epsilon,
    }


def _gate_a5(
    run: LoadedRun,
    margin_analysis: dict[str, Any] | None,
    task_ids: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """A5 — High-margin recovery. Candidate must recover configured fraction
    of high-margin oracle headroom."""
    if margin_analysis is None:
        return {
            "status": "BLOCKED",
            "reason": "margin_analysis not provided",
        }

    high_margin = margin_analysis.get("Gate_A_high_margin", {})
    if not high_margin:
        return {
            "status": "BLOCKED",
            "reason": "Gate_A_high_margin not found in margin analysis",
        }

    u_candidate_hm = high_margin.get("candidate_utility", 0.0)
    u_sham_hm = high_margin.get("sham_utility", 0.0)
    u_oracle_hm = high_margin.get("oracle_utility", 0.0)

    headroom_hm = u_oracle_hm - u_sham_hm
    if abs(headroom_hm) < 1e-10:
        return {
            "status": "FAIL",
            "reason": "No high-margin headroom available",
            "headroom": headroom_hm,
        }

    recovery = (u_candidate_hm - u_sham_hm) / headroom_hm
    threshold = 0.10  # Candidate must recover at least 10% of high-margin headroom
    passed = recovery > threshold

    return {
        "status": "PASS" if passed else "FAIL",
        "candidate_utility": u_candidate_hm,
        "sham_utility": u_sham_hm,
        "oracle_utility": u_oracle_hm,
        "headroom": headroom_hm,
        "recovery": recovery,
        "threshold": threshold,
    }
