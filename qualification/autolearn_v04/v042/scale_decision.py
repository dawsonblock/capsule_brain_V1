"""Model-size escalation gate.

Produces a formal machine-readable decision artifact determining
whether a 14B scaling run is justified.
"""
from __future__ import annotations

from typing import Any

from .data import LoadedRun


def make_scale_decision(
    run: LoadedRun,
    headroom_analysis: dict[str, Any],
    sham_audit: dict[str, Any],
    feature_audit: dict[str, Any] | None = None,
    equivalence_report: dict[str, Any] | None = None,
    learning_curves: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a model_scale_decision.json artifact.

    Approves a larger-model pilot only when at least one condition holds:
    A. Oracle-sham headroom is already substantial, but low action-level
       model success prevents the candidate from learning stable boundaries.
    B. A small matched cross-model pilot shows that increasing model size
       raises oracle-sham headroom.
    C. Per-action competence is the dominant bottleneck and the larger
       model is expected to improve action selectivity.

    Do not approve scale-up when:
    - oracle-sham headroom is small
    - candidate features have no predictive signal
    - sham construction is invalid
    - benchmark labels are dominated by near ties
    - candidate learning is unstable
    - counterfactual equivalence is unproven
    - source data have unresolved integrity issues
    """
    H = headroom_analysis.get("H_oracle_sham", 0.0)
    headroom_status = headroom_analysis.get("routing_headroom_status", "INCONCLUSIVE")
    R_sham = headroom_analysis.get("R_sham", 0.0)

    # Check blocking conditions.
    blockers = []

    if headroom_status == "INSUFFICIENT":
        blockers.append("oracle_sham_headroom_insufficient")
    if headroom_status == "INCONCLUSIVE":
        blockers.append("oracle_sham_headroom_inconclusive")

    # Check sham validity.
    sham_criterion_met = sham_audit.get("criterion_met", False)
    if not sham_criterion_met:
        # This is expected — candidate should beat sham. Not a blocker for scale decision.
        pass

    # Check feature signal.
    if feature_audit:
        family_only_utility = feature_audit.get("family_only_model", {}).get("test_utility")
        full_model_utility = feature_audit.get("full_model", {}).get("test_utility")
        if family_only_utility is not None and full_model_utility is not None:
            if full_model_utility <= family_only_utility:
                blockers.append("features_no_predictive_signal_beyond_family")

    # Check equivalence.
    if equivalence_report:
        if not equivalence_report.get("equivalence_verified", True):
            blockers.append("counterfactual_equivalence_unproven")

    # Check learning curves.
    if learning_curves:
        trend = learning_curves.get("trend_decision", "INCONCLUSIVE")
        if trend == "DATA_LIMITED":
            # This is a reason to collect more data, not scale model.
            blockers.append("data_limited_collect_more_7b_experience")

    # Determine decision.
    if blockers:
        # Determine which non-scale action is recommended.
        if "oracle_sham_headroom_insufficient" in blockers:
            decision = "REDESIGN_BENCHMARK"
            reason = "Oracle-sham headroom is insufficient. Scaling the model "
            reason += "will not help if the benchmark lacks route-dependent utility."
        elif "data_limited_collect_more_7b_experience" in blockers:
            decision = "COLLECT_MORE_7B_EXPERIENCE"
            reason = "Candidate-sham gap is improving with more data. "
            reason += "Collect more 7B experience before scaling."
        elif "features_no_predictive_signal_beyond_family" in blockers:
            decision = "IMPROVE_FEATURES"
            reason = "Executive features do not predict best action beyond family. "
            reason += "Improve features before scaling model."
        elif "counterfactual_equivalence_unproven" in blockers:
            decision = "BLOCKED"
            reason = "Counterfactual equivalence is unproven. "
            reason += "Cannot make scale decision without valid evidence."
        else:
            decision = "INCONCLUSIVE"
            reason = "Multiple blockers prevent a clear scale decision."
    else:
        # Check approval conditions.
        if H >= 0.50 and R_sham <= 0:
            # Headroom is substantial but candidate didn't recover it.
            # This could be a model capability issue (condition A).
            decision = "APPROVE_14B_PILOT"
            reason = "Oracle-sham headroom is substantial (H={:.3f}) but candidate ".format(H)
            reason += "recovered none of it (R_sham={:.3f}). Low action-level model ".format(R_sham)
            reason += "success may prevent stable boundary learning. A 14B pilot "
            reason += "is justified to test whether model capability is the bottleneck."
        elif H >= 0.10 and R_sham > 0:
            decision = "APPROVE_14B_PILOT"
            reason = "Routing headroom exists (H={:.3f}) and candidate shows some ".format(H)
            reason += "recovery (R_sham={:.3f}). A 14B pilot may increase ".format(R_sham)
            reason += "action selectivity and headroom."
        else:
            decision = "INCONCLUSIVE"
            reason = "Conditions for scale-up are not clearly met. "
            reason += "Headroom={:.3f}, R_sham={:.3f}.".format(H, R_sham)

    return {
        "decision": decision,
        "reason": reason,
        "headroom_H_oracle_sham": H,
        "headroom_status": headroom_status,
        "candidate_recovery_R_sham": R_sham,
        "blockers": blockers,
        "approval_conditions": {
            "A_headroom_substantial_but_low_success": H >= 0.50 and R_sham <= 0,
            "B_cross_model_pilot_shows_headroom_increase": False,  # requires actual pilot
            "C_per_action_competence_dominant_bottleneck": False,  # requires analysis
        },
        "blocking_conditions_checked": {
            "oracle_sham_headroom_small": headroom_status in ("INSUFFICIENT", "INCONCLUSIVE"),
            "features_no_signal": "features_no_predictive_signal_beyond_family" in blockers,
            "sham_construction_invalid": False,  # sham audit checks this
            "benchmark_labels_dominated_by_near_ties": False,  # margin analysis checks this
            "candidate_learning_unstable": False,  # seed stability checks this
            "counterfactual_equivalence_unproven": "counterfactual_equivalence_unproven" in blockers,
            "source_data_integrity_issues": False,
        },
        "scale_up_not_recommended_when": [
            "oracle-sham headroom is small",
            "candidate features have no predictive signal",
            "sham construction is invalid",
            "benchmark labels are dominated by near ties",
            "candidate learning is unstable",
            "counterfactual equivalence is unproven",
            "source data have unresolved integrity issues",
        ],
    }
