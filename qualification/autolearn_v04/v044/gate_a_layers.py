"""Gate A layered evaluation for v0.4.4.

Corrected Gate A layers with proper sub-gate logic.
"""
from __future__ import annotations

from typing import Any


def evaluate_gate_a_v044(
    gate_a0: dict,
    candidate_results: dict,
    baseline_results: dict,
    sham_results: dict,
    oracle_results: dict,
    gate_a_results: dict,
    feature_signal: dict,
    headroom_concentration: dict,
    high_margin_recovery: dict,
    raw_vs_executed: dict,
    candidate_stability: dict,
    metric_consistency: dict,
    scale_decision: dict | None = None,
    gate_a_epsilon: float = 0.01,
    gate_a_alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate Gate A with corrected layered logic.

    Gate A layers:
    - A0: Evidence validity (from gate_a0 module)
    - A1: Routing headroom (oracle > baseline + epsilon)
    - A2: Feature signal (beyond family/frequency)
    - A3: Candidate vs baseline (LCB > epsilon)
    - A4: Candidate vs sham (candidate > sham)
    - A5: High-margin recovery
    """
    sub_gates: dict[str, dict[str, Any]] = {}

    # A0: Evidence validity
    a0_status = gate_a0.get("status", "BLOCKED")
    sub_gates["A0_evidence_validity"] = {
        "status": a0_status,
        "observed": gate_a0.get("n_pass", 0),
        "expected": gate_a0.get("n_sub_gates", 18),
        "reason": "Evidence validity" + (" passed" if a0_status == "PASS" else " failed"),
    }

    # If A0 is not PASS, all other gates are BLOCKED.
    if a0_status != "PASS":
        for name in ["A1_routing_headroom", "A2_feature_signal", "A3_candidate_vs_baseline",
                      "A4_candidate_vs_sham", "A5_high_margin_recovery"]:
            sub_gates[name] = {
                "status": "BLOCKED",
                "observed": "N/A",
                "expected": "A0 PASS required",
                "reason": "A0 did not pass",
            }
        return {
            "overall_status": "BLOCKED",
            "sub_gates": sub_gates,
            "gate_a_verdict": "BLOCKED",
        }

    # A1: Routing headroom
    oracle_mean = oracle_results.get("mean_utility", 0)
    baseline_mean = baseline_results.get("mean_utility", 0)
    routing_headroom = oracle_mean - baseline_mean
    a1_pass = routing_headroom > gate_a_epsilon
    sub_gates["A1_routing_headroom"] = {
        "status": "PASS" if a1_pass else "FAIL",
        "observed": routing_headroom,
        "expected": f"> {gate_a_epsilon}",
        "reason": f"Oracle-baseline headroom = {routing_headroom:.4f}",
    }

    # A2: Feature signal
    fs_status = feature_signal.get("status", "INCONCLUSIVE")
    a2_pass = fs_status in ("FEATURE_SIGNAL_PRESENT", "FEATURE_SIGNAL_WEAK")
    sub_gates["A2_feature_signal"] = {
        "status": "PASS" if a2_pass else "FAIL",
        "observed": fs_status,
        "expected": "FEATURE_SIGNAL_PRESENT or FEATURE_SIGNAL_WEAK",
        "reason": feature_signal.get("conclusion", "Feature signal analysis"),
    }

    # A3: Candidate vs baseline
    cand_mean = candidate_results.get("mean_utility", 0)
    delta_cb = cand_mean - baseline_mean
    delta_cb_lcb = gate_a_results.get("delta_candidate_baseline_lcb", delta_cb)
    a3_pass = delta_cb_lcb > gate_a_epsilon
    sub_gates["A3_candidate_vs_baseline"] = {
        "status": "PASS" if a3_pass else "FAIL",
        "observed": delta_cb_lcb,
        "expected": f"> {gate_a_epsilon}",
        "reason": f"Candidate-baseline LCB = {delta_cb_lcb:.4f}",
    }

    # A4: Candidate vs sham
    sham_mean = sham_results.get("mean_utility", 0)
    delta_cs = cand_mean - sham_mean
    delta_cs_lcb = gate_a_results.get("delta_candidate_sham_lcb", delta_cs)
    a4_pass = delta_cs > 0 and delta_cs_lcb > 0
    sub_gates["A4_candidate_vs_sham"] = {
        "status": "PASS" if a4_pass else "FAIL",
        "observed": delta_cs,
        "expected": "> 0",
        "reason": f"Candidate-sham delta = {delta_cs:.4f}, LCB = {delta_cs_lcb:.4f}",
    }

    # A5: High-margin recovery
    hmr_status = high_margin_recovery.get("status", "NOT_RUN")
    a5_pass = hmr_status == "PASS"
    sub_gates["A5_high_margin_recovery"] = {
        "status": "PASS" if a5_pass else "FAIL",
        "observed": high_margin_recovery.get("recovery_fraction", 0),
        "expected": "> 0",
        "reason": f"High-margin recovery: {hmr_status}",
    }

    # Overall Gate A verdict.
    all_pass = all(g["status"] == "PASS" for g in sub_gates.values())
    overall = "PASS" if all_pass else "FAIL"

    return {
        "overall_status": overall,
        "sub_gates": sub_gates,
        "gate_a_verdict": overall,
        "routing_headroom": routing_headroom,
        "candidate_recovered_headroom_vs_baseline": delta_cb / routing_headroom if routing_headroom > 0 else None,
        "candidate_recovered_headroom_vs_sham": delta_cs / (oracle_mean - sham_mean) if (oracle_mean - sham_mean) > 0 else None,
    }
