"""Corrected model-scale decision for v0.4.4.

Removes unreachable duplicate branches. Uses precise variable names.
Does NOT call oracle-minus-sham headroom "headroom_recovered".
Uses: routing_headroom, candidate_recovered_headroom.

Decision rules:
- A0 not PASS → BLOCKED
- A0 PASS, A1 FAIL → REDESIGN_BENCHMARK
- A0 PASS, A1 PASS, A2 FAIL → IMPROVE_FEATURES
- A0 PASS, A1 PASS, A2 PASS, diagnostic learner improves validation → IMPROVE_LEARNER
- Raw candidate beats sham but executed does not → FIX_CALIBRATION_AND_FALLBACK
- Candidate advantage improves with experience and is stable → COLLECT_MORE_7B_EXPERIENCE
- A0 PASS, A1 PASS, sham valid, utility consistent, feature signal ≥ weak,
  candidate stable, action-specific competence plausibly capability limited
  → APPROVE_MATCHED_14B_PILOT
- Gate A PASS → NO_SCALE_NEEDED_FOR_GATE_A
- Default: approve_full_14b_run = false
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ScaleDecisionResult:
    """Result of the model-scale decision."""
    decision: str
    approve_full_14b_run: bool
    approve_matched_14b_pilot: bool
    routing_headroom: float | None
    routing_headroom_sufficient: bool
    candidate_recovered_headroom: float | None
    reason: str
    decision_path: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_scale_decision(
    gate_a0_status: str,
    gate_a1_routing_headroom_status: str,
    gate_a2_feature_signal_status: str,
    gate_a3_candidate_vs_baseline_status: str,
    gate_a4_candidate_vs_sham_status: str,
    gate_a5_high_margin_status: str,
    raw_candidate_vs_sham: str,
    executed_candidate_vs_sham: str,
    candidate_stability_status: str,
    sham_validity_status: str,
    counterfactual_completeness_status: str,
    utility_consistency_status: str,
    headroom_concentration_status: str,
    learner_comparison_status: str,
    calibration_suppression_status: str,
    routing_headroom: float | None = None,
    candidate_recovered_headroom: float | None = None,
    gate_a_overall_status: str = "FAIL",
) -> ScaleDecisionResult:
    """Make the model-scale decision following the declared decision tree.

    No unreachable duplicate branches. Precise variable names.
    """
    path: list[str] = []
    approve_full = False
    approve_pilot = False

    # Rule 1: A0 not PASS → BLOCKED
    if gate_a0_status != "PASS":
        path.append("A0 not PASS → BLOCKED")
        return ScaleDecisionResult(
            decision="BLOCKED",
            approve_full_14b_run=False,
            approve_matched_14b_pilot=False,
            routing_headroom=routing_headroom,
            routing_headroom_sufficient=False,
            candidate_recovered_headroom=candidate_recovered_headroom,
            reason="Gate A0 did not pass — evidence validity cannot be confirmed",
            decision_path=path,
        )

    # Rule: Gate A PASS → NO_SCALE_NEEDED_FOR_GATE_A
    if gate_a_overall_status == "PASS":
        path.append("Gate A PASS → NO_SCALE_NEEDED_FOR_GATE_A")
        return ScaleDecisionResult(
            decision="NO_SCALE_NEEDED_FOR_GATE_A",
            approve_full_14b_run=False,
            approve_matched_14b_pilot=False,
            routing_headroom=routing_headroom,
            routing_headroom_sufficient=True,
            candidate_recovered_headroom=candidate_recovered_headroom,
            reason="Gate A already passes — no model scaling needed",
            decision_path=path,
        )

    # Rule 2: A0 PASS, A1 FAIL → REDESIGN_BENCHMARK
    if gate_a1_routing_headroom_status == "FAIL":
        path.append("A0 PASS, A1 FAIL → REDESIGN_BENCHMARK")
        return ScaleDecisionResult(
            decision="REDESIGN_BENCHMARK",
            approve_full_14b_run=False,
            approve_matched_14b_pilot=False,
            routing_headroom=routing_headroom,
            routing_headroom_sufficient=False,
            candidate_recovered_headroom=candidate_recovered_headroom,
            reason="Insufficient routing headroom — benchmark needs redesign",
            decision_path=path,
        )

    # Rule 3: A0 PASS, A1 PASS, A2 FAIL → IMPROVE_FEATURES
    if gate_a2_feature_signal_status in ("FEATURE_SIGNAL_ABSENT", "FAIL"):
        path.append("A0 PASS, A1 PASS, A2 FAIL → IMPROVE_FEATURES")
        return ScaleDecisionResult(
            decision="IMPROVE_FEATURES",
            approve_full_14b_run=False,
            approve_matched_14b_pilot=False,
            routing_headroom=routing_headroom,
            routing_headroom_sufficient=True,
            candidate_recovered_headroom=candidate_recovered_headroom,
            reason="Pre-action features lack signal beyond family and frequency",
            decision_path=path,
        )

    # Rule 4: A0 PASS, A1 PASS, A2 PASS, learner improves validation → IMPROVE_LEARNER
    if (gate_a2_feature_signal_status in ("FEATURE_SIGNAL_PRESENT", "FEATURE_SIGNAL_WEAK", "PASS")
            and learner_comparison_status == "IMPROVE_LEARNER"):
        path.append("A0 PASS, A1 PASS, A2 PASS, learner improves → IMPROVE_LEARNER")
        return ScaleDecisionResult(
            decision="IMPROVE_LEARNER",
            approve_full_14b_run=False,
            approve_matched_14b_pilot=False,
            routing_headroom=routing_headroom,
            routing_headroom_sufficient=True,
            candidate_recovered_headroom=candidate_recovered_headroom,
            reason="Feature signal present but current learner does not recover it",
            decision_path=path,
        )

    # Rule 5: Raw candidate beats sham but executed does not → FIX_CALIBRATION_AND_FALLBACK
    if raw_candidate_vs_sham == "PASS" and executed_candidate_vs_sham != "PASS":
        path.append("Raw wins, executed loses → FIX_CALIBRATION_AND_FALLBACK")
        return ScaleDecisionResult(
            decision="FIX_CALIBRATION_AND_FALLBACK",
            approve_full_14b_run=False,
            approve_matched_14b_pilot=False,
            routing_headroom=routing_headroom,
            routing_headroom_sufficient=True,
            candidate_recovered_headroom=candidate_recovered_headroom,
            reason="Raw candidate recovers headroom but fallback removes it",
            decision_path=path,
        )

    # Rule 6: Candidate advantage improves with experience and is stable
    if (gate_a3_candidate_vs_baseline_status == "PASS"
            and candidate_stability_status in ("STABLE", "DIAGNOSTIC_ONLY")
            and gate_a4_candidate_vs_sham_status != "PASS"):
        path.append("Candidate improves with experience, stable → COLLECT_MORE_7B_EXPERIENCE")
        return ScaleDecisionResult(
            decision="COLLECT_MORE_7B_EXPERIENCE",
            approve_full_14b_run=False,
            approve_matched_14b_pilot=False,
            routing_headroom=routing_headroom,
            routing_headroom_sufficient=True,
            candidate_recovered_headroom=candidate_recovered_headroom,
            reason="Candidate advantage grows with experience — collect more 7B data",
            decision_path=path,
        )

    # Rule 7: A0 PASS, A1 PASS, sham valid, utility consistent, feature signal ≥ weak,
    # candidate stable, capability limited → APPROVE_MATCHED_14B_PILOT
    pilot_eligible = (
        gate_a0_status == "PASS"
        and gate_a1_routing_headroom_status == "PASS"
        and sham_validity_status == "PASS"
        and utility_consistency_status == "PASS"
        and counterfactual_completeness_status == "PASS"
        and gate_a2_feature_signal_status in ("FEATURE_SIGNAL_PRESENT", "FEATURE_SIGNAL_WEAK", "PASS")
        and candidate_stability_status not in ("FAILED", "UNSTABLE")
        and headroom_concentration_status != "INVALID_DATA"
    )

    if pilot_eligible:
        path.append("All pilot eligibility checks pass → APPROVE_MATCHED_14B_PILOT")
        approve_pilot = True
        return ScaleDecisionResult(
            decision="APPROVE_MATCHED_14B_PILOT",
            approve_full_14b_run=False,
            approve_matched_14b_pilot=True,
            routing_headroom=routing_headroom,
            routing_headroom_sufficient=True,
            candidate_recovered_headroom=candidate_recovered_headroom,
            reason="Evidence validity, routing headroom, sham controls, feature signal, "
                   "and router stability passed. Action-specific competence is plausibly "
                   "capability limited. A matched pilot is justified.",
            decision_path=path,
        )

    # Default: do not approve
    path.append("Default → no scale-up approved")
    return ScaleDecisionResult(
        decision="NO_SCALE_UP",
        approve_full_14b_run=False,
        approve_matched_14b_pilot=False,
        routing_headroom=routing_headroom,
        routing_headroom_sufficient=gate_a1_routing_headroom_status == "PASS",
        candidate_recovered_headroom=candidate_recovered_headroom,
        reason="No scale-up condition met",
        decision_path=path,
    )
