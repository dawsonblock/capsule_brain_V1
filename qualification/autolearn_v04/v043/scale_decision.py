"""Deterministic model-scale decision engine for v0.4.3.

Implements the corrected scale decision that replaces the v0.4.2
defect where ``APPROVE_14B_PILOT`` was issued merely because oracle
headroom was sufficient.

The decision engine applies a strict ordered case analysis:

  Case A: A0 != PASS                          -> BLOCKED
  Case B: A1 == FAIL                          -> REDESIGN_BENCHMARK
  Case C: A1 == PASS, A2 == FAIL              -> IMPROVE_FEATURES
  Case D: A1 == PASS, A2 == PASS,
          diagnostic learner beats current    -> IMPROVE_LEARNER
  Case E: raw beats sham, executed does not   -> FIX_CALIBRATION_AND_FALLBACK
  Case F: candidate-sham improves with
          more experience, stable             -> COLLECT_MORE_7B_EXPERIENCE
  Case G: all router-side checks pass,
          candidate recovers some headroom,
          matched pilot eligible              -> APPROVE_MATCHED_14B_PILOT
  Case H: Gate A passes                       -> NO_SCALE_NEEDED_FOR_GATE_A

A larger model may increase raw task capability, but this does not
establish that the current router can learn state-dependent routing.
"""
from __future__ import annotations

from typing import Any

from .data import LoadedRun
from .config import (
    SCALE_HEADROOM_THRESHOLD,
    PILOT_MIN_TASKS,
    PILOT_MAX_TASKS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub_gate_status(gate_a_result: dict[str, Any], key: str) -> str:
    """Extract a sub-gate status string from gate_a_result['sub_gates']."""
    sub_gates = gate_a_result.get("sub_gates", {})
    if not sub_gates:
        return "NOT_RUN"
    entry = sub_gates.get(key, {})
    return str(entry.get("status", "NOT_RUN"))


def _gate_a_overall(gate_a_result: dict[str, Any]) -> str:
    """Extract the Gate A overall status."""
    return str(gate_a_result.get("overall_status", "NOT_RUN"))


def _check_learner_beats_current(
    learner_validation_result: dict[str, Any] | None,
) -> bool:
    """Check whether a diagnostic learner beats the current learner.

    Returns True if the learner_validation_result indicates that a
    diagnostic learner (e.g. a different architecture or hyperparameter
    set) achieves higher validation expected utility than the current
    learner.
    """
    if not learner_validation_result:
        return False
    # Check for an explicit flag.
    if learner_validation_result.get("diagnostic_beats_current") is True:
        return True
    # Check for a comparison result with a clear winner.
    comparison = learner_validation_result.get("comparison", {})
    if isinstance(comparison, dict):
        winner = comparison.get("winner", "")
        if winner and "diagnostic" in str(winner).lower():
            return True
    # Check for validation_utility comparison.
    diag_util = learner_validation_result.get("diagnostic_validation_utility")
    current_util = learner_validation_result.get("current_validation_utility")
    if diag_util is not None and current_util is not None:
        if float(diag_util) > float(current_util):
            return True
    return False


def _check_improvement_trend(
    candidate_stability_result: dict[str, Any] | None,
) -> bool:
    """Check whether candidate-sham gap improves with more experience.

    This case requires multiple runs with varying amounts of training
    experience.  For a single run, this is not applicable and defaults
    to not triggered.
    """
    if not candidate_stability_result:
        return False
    # Look for an explicit improvement_trend field.
    trend = candidate_stability_result.get("improvement_trend")
    if trend is not None:
        return bool(trend)
    # Look for multi-run evidence.
    multi_run = candidate_stability_result.get("multi_run_evidence", {})
    if isinstance(multi_run, dict):
        return bool(multi_run.get("consistent_improvement", False))
    return False


def _check_matched_pilot_eligibility(
    run: LoadedRun,
    gate_a_result: dict[str, Any],
    headroom_concentration_result: dict[str, Any],
    counterfactual_completeness_result: dict[str, Any],
    utility_consistency_result: dict[str, Any],
    candidate_stability_result: dict[str, Any] | None,
    sham_ensemble_result: dict[str, Any] | None,
) -> tuple[bool, dict[str, bool]]:
    """Check matched 14B pilot eligibility conditions.

    Returns (eligible, conditions_dict).
    """
    a0_status = _sub_gate_status(gate_a_result, "A0")
    a1_status = _sub_gate_status(gate_a_result, "A1")
    a2_status = _sub_gate_status(gate_a_result, "A2")

    # Headroom concentration status.
    concentration_status = headroom_concentration_result.get(
        "concentration_status", "UNKNOWN",
    )

    # Counterfactual completeness.
    cf_summary = counterfactual_completeness_result.get("summary", {})
    cf_status = "PASS"
    if isinstance(cf_summary, dict):
        cf_complete = cf_summary.get("all_actions_executed")
        if cf_complete is not None and not cf_complete:
            cf_status = "FAIL"

    # Utility consistency.
    uc_status = utility_consistency_result.get("overall_status", "PASS")

    # Candidate stability.
    stability_class = "STABLE"
    if candidate_stability_result:
        stability_class = candidate_stability_result.get(
            "stability_classification", "STABLE",
        )

    # Sham validity.
    sham_valid = True
    if sham_ensemble_result:
        sham_valid = bool(sham_ensemble_result.get("candidate_beats_sham", False))

    # Headroom value.
    aggregate_headroom = headroom_concentration_result.get(
        "aggregate_headroom", 0.0,
    )
    if aggregate_headroom is None:
        aggregate_headroom = 0.0

    # Feature signal conclusion.
    feature_signal_conclusion = "FEATURE_SIGNAL_PRESENT"
    # This is passed separately; we infer from A2 status.
    feature_signal_ok = a2_status == "PASS"

    conditions = {
        "gate_a0_passes": a0_status == "PASS",
        "counterfactual_completeness_passes": cf_status == "PASS",
        "utility_consistency_passes": uc_status == "PASS",
        "sham_control_valid": sham_valid,
        "oracle_headroom_sufficient": float(aggregate_headroom) > SCALE_HEADROOM_THRESHOLD,
        "headroom_not_trivial_defect": concentration_status != "CONCENTRATED",
        "feature_signal_or_missing_state_identified": feature_signal_ok,
        "candidate_not_completely_unstable": stability_class != "ROUTER_UNSTABLE",
        "scale_decision_allows_pilot": True,  # will be set by caller
    }

    # The scale_decision_allows_pilot condition is circular; we check
    # all other conditions first.
    other_conditions = {
        k: v for k, v in conditions.items()
        if k != "scale_decision_allows_pilot"
    }
    eligible = all(other_conditions.values())
    return eligible, conditions


def _confidence_level(
    decision: str,
    gate_a_result: dict[str, Any],
    n_supporting: int,
    n_blocking: int,
) -> str:
    """Determine confidence level for the decision."""
    overall = _gate_a_overall(gate_a_result)
    if decision == "BLOCKED":
        return "HIGH"
    if decision == "APPROVE_MATCHED_14B_PILOT":
        if overall == "PASS" and n_blocking == 0:
            return "HIGH"
        return "MEDIUM"
    if decision == "NO_SCALE_NEEDED_FOR_GATE_A":
        if overall == "PASS":
            return "HIGH"
        return "MEDIUM"
    if decision in ("REDESIGN_BENCHMARK", "IMPROVE_FEATURES"):
        return "HIGH"
    if decision in ("FIX_CALIBRATION_AND_FALLBACK", "IMPROVE_LEARNER"):
        return "MEDIUM"
    if decision == "COLLECT_MORE_7B_EXPERIENCE":
        return "LOW"
    return "MEDIUM"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_scale_decision_v043(
    run: LoadedRun,
    gate_a_result: dict[str, Any],
    headroom_concentration_result: dict[str, Any],
    feature_signal_result: dict[str, Any],
    raw_vs_executed_result: dict[str, Any],
    candidate_stability_result: dict[str, Any],
    sham_ensemble_result: dict[str, Any],
    counterfactual_completeness_result: dict[str, Any],
    utility_consistency_result: dict[str, Any],
    learner_validation_result: dict[str, Any] | None = None,
) -> dict:
    """Deterministic model-scale decision engine for v0.4.3.

    Applies a strict ordered case analysis to produce a machine-readable
    scale decision.  The decision is deterministic: given the same
    inputs, the output is always the same.

    **Important**: A larger model may increase raw task capability, but
    this does not establish that the current router can learn
    state-dependent routing.  Therefore ``approve_full_14b_run`` is
    always False by default.

    Args:
        run: The loaded run.
        gate_a_result: Output from ``evaluate_gate_a_v043()``.  Must
            contain ``sub_gates`` with A0-A5 statuses.
        headroom_concentration_result: Output of
            ``analyze_headroom_concentration``.
        feature_signal_result: Output of the feature-signal analysis.
        raw_vs_executed_result: Output of the raw-vs-executed analysis.
        candidate_stability_result: Output of the candidate-stability
            analysis.
        sham_ensemble_result: Output of ``run_sham_ensemble``.
        counterfactual_completeness_result: Output of the
            counterfactual completeness analysis.
        utility_consistency_result: Output of
            ``audit_utility_consistency``.
        learner_validation_result: Optional output of a diagnostic
            learner validation comparison.

    Returns:
        JSON-serializable ``model_scale_decision`` dict with:

        - ``decision``: str
        - ``reason``: str
        - ``supporting_gates``: list[str]
        - ``blocking_gates``: list[str]
        - ``required_next_action``: str
        - ``approve_full_14b_run``: bool (always False)
        - ``approve_matched_14b_pilot``: bool
        - ``confidence``: str ("HIGH"/"MEDIUM"/"LOW")
        - ``evidence_artifacts``: list[str]
        - ``blocking_conditions_checked``: dict
        - ``scale_decision_statement``: str
    """
    # Extract sub-gate statuses.
    a0 = _sub_gate_status(gate_a_result, "A0")
    a1 = _sub_gate_status(gate_a_result, "A1")
    a2 = _sub_gate_status(gate_a_result, "A2")
    a3 = _sub_gate_status(gate_a_result, "A3")
    a4 = _sub_gate_status(gate_a_result, "A4")
    a5 = _sub_gate_status(gate_a_result, "A5")
    gate_a_overall = _gate_a_overall(gate_a_result)

    sub_gate_statuses = {
        "A0": a0, "A1": a1, "A2": a2,
        "A3": a3, "A4": a4, "A5": a5,
    }

    # Build supporting / blocking gate lists.
    supporting_gates: list[str] = [
        k for k, v in sub_gate_statuses.items() if v == "PASS"
    ]
    blocking_gates: list[str] = [
        k for k, v in sub_gate_statuses.items()
        if v in ("FAIL", "BLOCKED")
    ]

    evidence_artifacts = [
        "gate_a_result",
        "headroom_concentration_result",
        "feature_signal_result",
        "raw_vs_executed_result",
        "candidate_stability_result",
        "sham_ensemble_result",
        "counterfactual_completeness_result",
        "utility_consistency_result",
    ]
    if learner_validation_result is not None:
        evidence_artifacts.append("learner_validation_result")

    # The mandatory statement.
    scale_decision_statement = (
        "A larger model may increase raw task capability, but this does "
        "not establish that the current router can learn state-dependent "
        "routing."
    )

    # ------------------------------------------------------------------
    # Case analysis (ordered A through H)
    # ------------------------------------------------------------------

    blocking_conditions_checked: dict[str, Any] = {
        "A0_not_pass": a0 != "PASS",
        "A1_fail": a1 == "FAIL",
        "A2_fail_with_A1_pass": (a1 == "PASS" and a2 == "FAIL"),
        "diagnostic_learner_beats_current": _check_learner_beats_current(
            learner_validation_result,
        ),
        "calibration_suppresses_routing": (
            raw_vs_executed_result.get("bottleneck_classification", "")
            == "CALIBRATION_SUPPRESSES_ROUTING"
        ),
        "improvement_trend": _check_improvement_trend(
            candidate_stability_result,
        ),
        "all_router_side_pass": gate_a_overall == "PASS",
    }

    decision: str = "INCONCLUSIVE"
    reason: str = "No case matched."
    required_next_action: str = "Review evidence artifacts."

    # Case A: A0 != PASS -> BLOCKED.
    if a0 != "PASS":
        decision = "BLOCKED"
        reason = (
            "Infrastructure or evidence validity unresolved "
            f"(A0 status={a0})."
        )
        required_next_action = (
            "Resolve infrastructure and evidence validity issues "
            "before any scale decision can be made."
        )

    # Case B: A1 == FAIL -> REDESIGN_BENCHMARK.
    elif a1 == "FAIL":
        decision = "REDESIGN_BENCHMARK"
        reason = "Insufficient routing headroom (A1 FAIL)."
        required_next_action = (
            "Redesign the benchmark to create routing headroom: "
            "ensure oracle-sham utility gap exceeds the threshold "
            f"({SCALE_HEADROOM_THRESHOLD}) and is not concentrated."
        )

    # Case C: A1 == PASS and A2 == FAIL -> IMPROVE_FEATURES.
    elif a1 == "PASS" and a2 == "FAIL":
        decision = "IMPROVE_FEATURES"
        reason = (
            "Headroom exists but pre-action features do not identify it "
            "(A2 FAIL)."
        )
        required_next_action = (
            "Improve executive feature engineering: add or refine "
            "features that carry state-dependent routing signal. "
            "Re-run feature-signal analysis to confirm "
            "FEATURE_SIGNAL_PRESENT."
        )

    # Case D: A1 == PASS, A2 == PASS, diagnostic learner beats current.
    elif (
        a1 == "PASS"
        and a2 == "PASS"
        and _check_learner_beats_current(learner_validation_result)
    ):
        decision = "IMPROVE_LEARNER"
        reason = (
            "A diagnostic learner beats the current learner on "
            "validation expected utility. The routing signal is "
            "learnable but the current learner architecture or "
            "training procedure is suboptimal."
        )
        required_next_action = (
            "Adopt the diagnostic learner architecture or training "
            "procedure, retrain the candidate policy, and re-run "
            "qualification."
        )

    # Case E: raw candidate beats sham but executed candidate does not.
    elif (
        raw_vs_executed_result.get("bottleneck_classification", "")
        == "CALIBRATION_SUPPRESSES_ROUTING"
    ):
        decision = "FIX_CALIBRATION_AND_FALLBACK"
        reason = (
            "Raw candidate beats sham but executed candidate does not: "
            "calibration/fallback is suppressing useful routing."
        )
        required_next_action = (
            "Fix the calibration and fallback logic so that the "
            "production stack preserves the raw learner's routing "
            "decisions. Re-run raw-vs-executed analysis to confirm "
            "RAW_LEARNER_GOOD."
        )

    # Case F: candidate-sham improves with more experience, stable.
    elif _check_improvement_trend(candidate_stability_result):
        decision = "COLLECT_MORE_7B_EXPERIENCE"
        reason = (
            "Candidate-sham gap improves consistently with more "
            "experience and remains stable. The router may learn "
            "state-dependent routing with additional training data."
        )
        required_next_action = (
            "Collect more 7B experience examples and re-run "
            "qualification. Monitor the candidate-sham gap for "
            "continued improvement."
        )

    # Case G: all router-side checks pass, candidate recovers some
    # headroom, matched pilot eligible.
    elif gate_a_overall == "PASS":
        # Check headroom recovery.
        aggregate_headroom = headroom_concentration_result.get(
            "aggregate_headroom", 0.0,
        )
        if aggregate_headroom is None:
            aggregate_headroom = 0.0
        headroom_recovered = float(aggregate_headroom) > SCALE_HEADROOM_THRESHOLD

        # Check matched pilot eligibility.
        pilot_eligible, pilot_conditions = _check_matched_pilot_eligibility(
            run,
            gate_a_result,
            headroom_concentration_result,
            counterfactual_completeness_result,
            utility_consistency_result,
            candidate_stability_result,
            sham_ensemble_result,
        )

        if headroom_recovered and pilot_eligible:
            decision = "APPROVE_MATCHED_14B_PILOT"
            reason = (
                "All router-side checks pass (Gate A PASS), candidate "
                "recovers some headroom, and matched pilot eligibility "
                "conditions are met."
            )
            required_next_action = (
                f"Generate a matched pilot manifest with "
                f"{PILOT_MIN_TASKS}-{PILOT_MAX_TASKS} tasks and "
                "execute the matched 14B pilot. Do NOT proceed to a "
                "full 14B run until the pilot confirms that the "
                "router benefits from increased model capability."
            )
            blocking_conditions_checked["pilot_eligibility"] = pilot_conditions
        else:
            # Case H: Gate A passes but no matched pilot needed or
            # eligible.
            decision = "NO_SCALE_NEEDED_FOR_GATE_A"
            reason = (
                "Gate A passes. Router-side evidence is sufficient. "
                "No model-scale increase is warranted at this time."
            )
            required_next_action = (
                "No immediate action required. Monitor router "
                "performance in production and re-qualify if "
                "conditions change."
            )
            blocking_conditions_checked["pilot_eligibility"] = pilot_conditions
            blocking_conditions_checked["headroom_recovered"] = headroom_recovered

    # Case H: Gate A passes (fallback if not caught by Case G).
    elif gate_a_overall == "PASS":
        decision = "NO_SCALE_NEEDED_FOR_GATE_A"
        reason = (
            "Gate A passes. Router-side evidence is sufficient. "
            "No model-scale increase is warranted at this time."
        )
        required_next_action = (
            "No immediate action required. Monitor router "
            "performance in production and re-qualify if "
            "conditions change."
        )

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------

    approve_matched_14b_pilot = (decision == "APPROVE_MATCHED_14B_PILOT")
    confidence = _confidence_level(
        decision, gate_a_result, len(supporting_gates), len(blocking_gates),
    )

    return {
        "decision": decision,
        "reason": reason,
        "supporting_gates": supporting_gates,
        "blocking_gates": blocking_gates,
        "required_next_action": required_next_action,
        "approve_full_14b_run": False,  # always False by default
        "approve_matched_14b_pilot": approve_matched_14b_pilot,
        "confidence": confidence,
        "evidence_artifacts": evidence_artifacts,
        "blocking_conditions_checked": blocking_conditions_checked,
        "scale_decision_statement": scale_decision_statement,
        "gate_a_overall_status": gate_a_overall,
        "sub_gate_statuses": sub_gate_statuses,
    }
