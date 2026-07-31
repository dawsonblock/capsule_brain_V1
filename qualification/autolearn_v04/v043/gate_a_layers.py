"""Corrected Gate A layered logic for v0.4.3.

Decomposes Gate A into six diagnostic sub-gates:

* **A0 — Infrastructure validity**: PASS only if all infrastructure and
  lineage checks pass (from ``gate_a0_result`` ``overall_status``).
* **A1 — Routing headroom**: PASS if oracle-sham headroom exceeds the
  threshold (0.05) AND is not excessively concentrated.
* **A2 — Feature learnability**: PASS if executive features improve
  validation expected utility beyond frequency and family-only controls.
* **A3 — Candidate versus baseline**: PASS if candidate-minus-baseline
  LCB exceeds the practical threshold (0.01).
* **A4 — Candidate versus sham**: PASS if candidate-minus-sham LCB
  exceeds the threshold (0.0) AND the candidate beats the sham-seed
  distribution (candidate percentile > 50%).
* **A5 — High-margin recovery**: PASS if the candidate recovers >= 10%
  of high-margin oracle-sham headroom with nonnegative LCB.

Final Gate A PASS requires A0 through A5 to all PASS.

No scale decision may override a failed A0 or A2.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import LoadedRun
from .config import (
    A1_HEADROOM_THRESHOLD,
    A3_PRACTICAL_THRESHOLD,
    A4_PRACTICAL_THRESHOLD,
    A4_SHAM_PERCENTILE_THRESHOLD,
)
from ..v042.bootstrap import task_paired_bootstrap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_from_result(result: dict[str, Any] | None, key: str = "overall_status") -> str:
    """Safely extract a status string from a result dict."""
    if not result:
        return "NOT_RUN"
    return str(result.get(key, "NOT_RUN"))


def _task_level_utilities(
    canonical_results: dict[str, Any], policy_id: str
) -> tuple[list[str], np.ndarray]:
    """Extract task-level utilities for a policy from canonical_results.

    ``canonical_results`` may either map policy_id -> PolicyEvalResult or
    policy_id -> dict (result_to_dict output).  Returns (task_ids, utilities)
    aligned by task_id.
    """
    result = canonical_results.get(policy_id) if canonical_results else None
    if result is None:
        return [], np.array([])
    task_results = _get_task_results(result)
    pairs: list[tuple[str, float]] = []
    for tr in task_results:
        u = tr.get("selected_utility")
        if u is None:
            continue
        try:
            pairs.append((tr["task_id"], float(u)))
        except (TypeError, ValueError):
            continue
    pairs.sort(key=lambda x: x[0])
    tids = [p[0] for p in pairs]
    utils = np.array([p[1] for p in pairs], dtype=float)
    return tids, utils


def _get_task_results(result: Any) -> list[dict[str, Any]]:
    """Extract task_results from either a PolicyEvalResult or a dict."""
    if hasattr(result, "task_results"):
        tr = result.task_results
        return [
            {
                "task_id": r.task_id,
                "selected_utility": r.selected_utility,
            }
            for r in tr
        ]
    if isinstance(result, dict):
        return result.get("task_results", [])
    return []


def _aligned_deltas(
    canonical_results: dict[str, Any],
    policy_a: str,
    policy_b: str,
) -> np.ndarray:
    """Compute per-task paired deltas (policy_a - policy_b) on the common
    set of complete tasks."""
    tids_a, utils_a = _task_level_utilities(canonical_results, policy_a)
    tids_b, utils_b = _task_level_utilities(canonical_results, policy_b)
    if len(tids_a) == 0 or len(tids_b) == 0:
        return np.array([])
    map_b = dict(zip(tids_b, utils_b))
    deltas: list[float] = []
    for tid, ua in zip(tids_a, utils_a):
        ub = map_b.get(tid)
        if ub is None:
            continue
        deltas.append(float(ua - ub))
    return np.array(deltas, dtype=float)


# ---------------------------------------------------------------------------
# Sub-gates
# ---------------------------------------------------------------------------


def _gate_a0(gate_a0_result: dict[str, Any] | None) -> dict[str, Any]:
    """A0 — Infrastructure validity."""
    if not gate_a0_result:
        return {
            "status": "BLOCKED",
            "key_metric": None,
            "threshold": "PASS",
            "reason": "gate_a0_result not provided",
        }
    overall = str(gate_a0_result.get("overall_status", "NOT_RUN"))
    if overall == "PASS":
        status = "PASS"
        reason = "All infrastructure and lineage checks pass."
    elif overall == "NOT_RUN":
        status = "NOT_RUN"
        reason = "Gate A0 was not run."
    else:
        status = "BLOCKED"
        reason = f"Gate A0 overall_status={overall}; infrastructure/lineage unresolved."
    return {
        "status": status,
        "key_metric": overall,
        "threshold": "PASS",
        "reason": reason,
    }


def _gate_a1(
    headroom_concentration_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """A1 — Routing headroom."""
    threshold = A1_HEADROOM_THRESHOLD
    if not headroom_concentration_result:
        return {
            "status": "NOT_RUN",
            "key_metric": None,
            "threshold": threshold,
            "reason": "headroom_concentration_result not provided",
        }
    aggregate_headroom = headroom_concentration_result.get("aggregate_headroom")
    concentration_status = headroom_concentration_result.get(
        "concentration_status", "UNKNOWN"
    )
    if aggregate_headroom is None:
        return {
            "status": "NOT_RUN",
            "key_metric": None,
            "threshold": threshold,
            "reason": "aggregate_headroom missing from headroom_concentration_result",
        }
    headroom_ok = float(aggregate_headroom) > threshold
    not_concentrated = concentration_status != "CONCENTRATED"
    passed = headroom_ok and not_concentrated
    if passed:
        reason = (
            f"Oracle-sham headroom={aggregate_headroom:.4f} (> {threshold}) "
            f"and concentration_status={concentration_status} (not CONCENTRATED)."
        )
    elif not headroom_ok:
        reason = (
            f"Oracle-sham headroom={aggregate_headroom:.4f} "
            f"does not exceed threshold {threshold}."
        )
    else:
        reason = (
            f"Headroom is excessively concentrated "
            f"(concentration_status={concentration_status})."
        )
    return {
        "status": "PASS" if passed else "FAIL",
        "key_metric": float(aggregate_headroom),
        "threshold": threshold,
        "concentration_status": concentration_status,
        "reason": reason,
    }


def _gate_a2(
    feature_signal_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """A2 — Feature learnability."""
    if not feature_signal_result:
        return {
            "status": "NOT_RUN",
            "key_metric": None,
            "threshold": "FEATURE_SIGNAL_PRESENT",
            "reason": "feature_signal_result not provided",
        }
    conclusion = feature_signal_result.get("learnability_conclusion", "UNKNOWN")
    passed = conclusion == "FEATURE_SIGNAL_PRESENT"
    if passed:
        reason = (
            "Executive features improve validation expected utility beyond "
            "frequency and family-only controls."
        )
    else:
        reason = (
            f"Feature signal absent or inconclusive "
            f"(learnability_conclusion={conclusion})."
        )
    return {
        "status": "PASS" if passed else "FAIL",
        "key_metric": conclusion,
        "threshold": "FEATURE_SIGNAL_PRESENT",
        "reason": reason,
    }


def _gate_a3(
    canonical_results: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """A3 — Candidate versus baseline."""
    threshold = A3_PRACTICAL_THRESHOLD
    deltas = _aligned_deltas(canonical_results, "candidate_executed", "baseline")
    if len(deltas) == 0:
        return {
            "status": "NOT_RUN",
            "key_metric": None,
            "threshold": threshold,
            "reason": "No aligned candidate/baseline task utilities available.",
        }
    ci = task_paired_bootstrap(deltas, n_bootstrap=n_bootstrap, seed=seed)
    passed = ci["ci_low"] > threshold
    reason = (
        f"Candidate-minus-baseline LCB={ci['ci_low']:.6f} "
        f"{'>' if passed else '<='} threshold {threshold}."
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "key_metric": float(ci["ci_low"]),
        "threshold": threshold,
        "ci": ci,
        "n_tasks": int(len(deltas)),
        "reason": reason,
    }


def _gate_a4(
    canonical_results: dict[str, Any],
    sham_ensemble_result: dict[str, Any] | None,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """A4 — Candidate versus sham."""
    threshold = A4_PRACTICAL_THRESHOLD
    percentile_threshold = A4_SHAM_PERCENTILE_THRESHOLD
    deltas = _aligned_deltas(canonical_results, "candidate_executed", "sham")
    if len(deltas) == 0:
        return {
            "status": "NOT_RUN",
            "key_metric": None,
            "threshold": threshold,
            "reason": "No aligned candidate/sham task utilities available.",
        }
    ci = task_paired_bootstrap(deltas, n_bootstrap=n_bootstrap, seed=seed)
    lcb_ok = ci["ci_low"] > threshold

    # Candidate percentile within the sham-seed distribution.
    candidate_percentile: float | None = None
    percentile_ok = False
    if sham_ensemble_result:
        cp = sham_ensemble_result.get("candidate_percentile")
        if cp is not None:
            if isinstance(cp, dict):
                # Take the mean across sham types.
                vals = [float(v) for v in cp.values() if v is not None]
                if vals:
                    candidate_percentile = float(np.mean(vals))
            elif isinstance(cp, (int, float)):
                candidate_percentile = float(cp)
        if candidate_percentile is None:
            # Fallback: compute from candidate_utility vs sham_seed_utilities.
            cand_util = sham_ensemble_result.get("candidate_utility")
            seed_utils = sham_ensemble_result.get("sham_seed_utilities", [])
            if cand_util is not None and seed_utils:
                arr = np.array(seed_utils, dtype=float)
                candidate_percentile = float(
                    np.mean(arr < cand_util) * 100.0
                )
        if candidate_percentile is not None:
            percentile_ok = candidate_percentile > percentile_threshold

    passed = lcb_ok and percentile_ok
    parts = [f"LCB={ci['ci_low']:.6f} {'>' if lcb_ok else '<='} {threshold}"]
    if candidate_percentile is not None:
        parts.append(
            f"candidate percentile={candidate_percentile:.1f}% "
            f"{'>' if percentile_ok else '<='} {percentile_threshold}%"
        )
    else:
        parts.append("sham-seed distribution not available; percentile check skipped")
    reason = "; ".join(parts) + "."
    return {
        "status": "PASS" if passed else "FAIL",
        "key_metric": float(ci["ci_low"]),
        "threshold": threshold,
        "ci": ci,
        "candidate_percentile": candidate_percentile,
        "percentile_threshold": percentile_threshold,
        "n_tasks": int(len(deltas)),
        "reason": reason,
    }


def _gate_a5(
    high_margin_result: dict[str, Any] | None,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """A5 — High-margin recovery."""
    min_recovery = 0.10
    if not high_margin_result:
        return {
            "status": "NOT_RUN",
            "key_metric": None,
            "threshold": min_recovery,
            "reason": "high_margin_result not provided",
        }
    gate_a5 = high_margin_result.get("gate_a5", high_margin_result)
    if not gate_a5:
        return {
            "status": "NOT_RUN",
            "key_metric": None,
            "threshold": min_recovery,
            "reason": "gate_a5 entry missing from high_margin_result",
        }
    recovery = gate_a5.get("recovery")
    lcb = gate_a5.get("lcb")
    headroom = gate_a5.get("headroom")
    if recovery is None:
        return {
            "status": "NOT_RUN",
            "key_metric": None,
            "threshold": min_recovery,
            "reason": "recovery missing from high_margin gate_a5 entry",
        }
    recovery_ok = float(recovery) >= min_recovery
    lcb_ok = (lcb is None) or (float(lcb) >= 0.0)
    passed = recovery_ok and lcb_ok
    reason = (
        f"High-margin recovery={float(recovery):.4f} "
        f"{'>=' if recovery_ok else '<'} {min_recovery}"
        + (f", LCB={float(lcb):.6f} {'>=' if lcb_ok else '<'} 0.0" if lcb is not None else "")
        + "."
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "key_metric": float(recovery),
        "threshold": min_recovery,
        "recovery": float(recovery),
        "lcb": float(lcb) if lcb is not None else None,
        "headroom": float(headroom) if headroom is not None else None,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_gate_a_v043(
    run: LoadedRun,
    gate_a0_result: dict[str, Any],
    headroom_concentration_result: dict[str, Any],
    feature_signal_result: dict[str, Any],
    canonical_results: dict[str, Any],
    high_margin_result: dict[str, Any],
    sham_ensemble_result: dict[str, Any],
    metric_consistency_result: dict[str, Any],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate the corrected Gate A layered sub-gates for v0.4.3.

    Args:
        run: The loaded run.
        gate_a0_result: Output of the Gate A0 infrastructure audit.
        headroom_concentration_result: Output of
            ``analyze_headroom_concentration``.
        feature_signal_result: Output of the feature-signal / learnability
            analysis.
        canonical_results: Dict mapping policy_id to PolicyEvalResult (or
            its dict form) from the canonical evaluator.
        high_margin_result: Output of the high-margin recovery analysis
            (must contain a ``gate_a5`` entry).
        sham_ensemble_result: Output of the sham-seed ensemble analysis.
        metric_consistency_result: Output of the metric-consistency
            cross-check (used to confirm evidence validity alongside A0).
        n_bootstrap: Number of bootstrap replicates.
        seed: Random seed.

    Returns:
        A JSON-serializable ``gate_a_v043_results`` dict with ``sub_gates``
        (A0-A5), ``overall_status``, ``n_pass``, ``n_fail``, ``n_blocked``,
        and ``interpretation``.
    """
    sub_gates: dict[str, dict[str, Any]] = {}

    # A0 — Infrastructure validity.
    a0 = _gate_a0(gate_a0_result)
    # Fold metric-consistency into A0: if metrics drifted, A0 is blocked.
    if metric_consistency_result:
        mc_status = str(metric_consistency_result.get("overall_status", "PASS"))
        if mc_status not in ("PASS", "CONSISTENT", "NOT_RUN"):
            a0 = {
                "status": "BLOCKED",
                "key_metric": a0.get("key_metric"),
                "threshold": "PASS",
                "metric_consistency_status": mc_status,
                "reason": (
                    f"Metric consistency check failed ({mc_status}); "
                    "evidence validity unresolved."
                ),
            }
        else:
            a0["metric_consistency_status"] = mc_status
    sub_gates["A0"] = a0

    # A1 — Routing headroom.
    sub_gates["A1"] = _gate_a1(headroom_concentration_result)

    # A2 — Feature learnability.
    sub_gates["A2"] = _gate_a2(feature_signal_result)

    # A3 — Candidate versus baseline.
    sub_gates["A3"] = _gate_a3(canonical_results, n_bootstrap, seed)

    # A4 — Candidate versus sham.
    sub_gates["A4"] = _gate_a4(
        canonical_results, sham_ensemble_result, n_bootstrap, seed
    )

    # A5 — High-margin recovery.
    sub_gates["A5"] = _gate_a5(high_margin_result, n_bootstrap, seed)

    # Overall decision.
    statuses = {k: v["status"] for k, v in sub_gates.items()}
    n_pass = sum(1 for s in statuses.values() if s == "PASS")
    n_fail = sum(1 for s in statuses.values() if s == "FAIL")
    n_blocked = sum(1 for s in statuses.values() if s == "BLOCKED")
    n_not_run = sum(1 for s in statuses.values() if s == "NOT_RUN")

    if n_blocked > 0:
        overall = "BLOCKED"
    elif n_not_run > 0:
        overall = "NOT_RUN"
    elif n_fail > 0:
        overall = "FAIL"
    elif n_pass == len(sub_gates):
        overall = "PASS"
    else:
        overall = "INCONCLUSIVE"

    # Build interpretation.
    failed = [k for k, s in statuses.items() if s == "FAIL"]
    blocked = [k for k, s in statuses.items() if s == "BLOCKED"]
    not_run = [k for k, s in statuses.items() if s == "NOT_RUN"]
    if overall == "PASS":
        interpretation = (
            "Gate A PASS: all sub-gates A0-A5 pass. Router-side evidence "
            "is sufficient."
        )
    elif overall == "BLOCKED":
        interpretation = (
            f"Gate A BLOCKED: sub-gate(s) {blocked} are blocked. "
            "Infrastructure or evidence validity is unresolved."
        )
    elif overall == "NOT_RUN":
        interpretation = (
            f"Gate A NOT_RUN: sub-gate(s) {not_run} were not run. "
            "Insufficient evidence to decide."
        )
    elif overall == "FAIL":
        interpretation = (
            f"Gate A FAIL: sub-gate(s) {failed} failed. "
            "No scale decision may override a failed A0 or A2."
        )
    else:
        interpretation = "Gate A is inconclusive."

    return {
        "sub_gates": sub_gates,
        "overall_status": overall,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_blocked": n_blocked,
        "n_not_run": n_not_run,
        "interpretation": interpretation,
        "requirements": "Final Gate A PASS requires A0, A1, A2, A3, A4, A5.",
        "no_scale_override_for_failed_A0_or_A2": True,
    }
