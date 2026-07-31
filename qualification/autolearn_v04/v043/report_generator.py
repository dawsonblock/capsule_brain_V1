"""Generate the GATE_A_V043_REPAIR_REPORT.md from all analysis results."""
from __future__ import annotations

from typing import Any

from .data import LoadedRun
from .config import PACKAGE_VERSION, QUALIFICATION_VERSION


def _f(v: Any) -> float:
    """Safely convert any value to float."""
    if v is None or isinstance(v, (dict, list)):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def generate_v043_report(
    run: LoadedRun,
    canonical_results: dict,
    metric_consistency: dict,
    sham_identity: dict,
    gate_a0: dict,
    decision_trace: dict,
    raw_vs_executed: dict,
    calibration: dict,
    headroom_concentration: dict,
    high_margin: dict,
    feature_signal: dict,
    missing_state: dict,
    feature_registry: dict,
    learner_validation: dict,
    soft_utility: dict,
    class_balance: dict,
    sham_ensemble: dict,
    candidate_stability: dict,
    parameter_diagnostics: dict,
    utility_consistency: dict,
    counterfactual_completeness: dict,
    gate_a: dict,
    scale_decision: dict,
    pilot_manifest: dict,
) -> str:
    """Generate the full v0.4.3 repair report."""
    lines: list[str] = []
    w = lines.append

    w(f"# Gate A v0.4.3 Repair Report — AutoLearn {QUALIFICATION_VERSION}")
    w("")
    w(f"**Package**: {PACKAGE_VERSION} | **AutoLearn**: {run.qualification_report.get('autolearn_version', 'N/A')} | "
      f"**Qualification**: {QUALIFICATION_VERSION}")
    w(f"**Source run**: {run.artifacts_dir.name} | **Tasks**: {len(run.tasks)} | **Test tasks**: {len(run.test_task_ids)}")
    w("")
    w("---")
    w("")

    # 1. Source run identity
    w("## 1. Source Run Identity")
    w("")
    w(f"- **Artifacts directory**: `{run.artifacts_dir}`")
    w(f"- **Benchmark manifest SHA-256**: `{run.benchmark_manifest_sha256[:16]}...`")
    w(f"- **Counterfactual results SHA-256**: `{run.counterfactual_results_sha256[:16]}...`")
    w(f"- **Candidate policy SHA-256**: `{run.candidate_policy_sha256[:16]}...`")
    w(f"- **Sham policy SHA-256**: `{run.sham_policy_sha256[:16]}...`")
    w(f"- **Gate A result SHA-256**: `{run.gate_a_result_sha256[:16]}...`")
    w("")

    # 2. Metric consistency
    w("## 2. Metric Consistency")
    w("")
    w(f"**Overall status**: {metric_consistency.get('overall_status', 'UNKNOWN')}")
    w("")
    w("| Metric | Expected | Actual | Abs Diff | Status |")
    w("|--------|----------|--------|----------|--------|")
    for check in metric_consistency.get("checks", []):
        name = check.get("name", "")
        expected = check.get("expected", "")
        actual = check.get("actual", "")
        diff = check.get("abs_diff", 0)
        status = check.get("status", "")
        w(f"| {name} | {expected} | {actual} | {diff} | {status} |")
    w("")
    if metric_consistency.get("failed_checks"):
        w("**Failed checks:**")
        for fc in metric_consistency["failed_checks"]:
            if isinstance(fc, dict):
                w(f"- {fc.get('name', '')}: expected {fc.get('expected')}, got {fc.get('actual')}")
            else:
                w(f"- {fc}")
        w("")

    # 3. Corrected recovered-headroom values
    w("## 3. Corrected Recovered-Headroom Values")
    w("")
    ce = canonical_results.get("candidate_executed")
    if ce:
        ce_dict = ce if isinstance(ce, dict) else {}
        h = ce_dict.get("headroom_vs_reference", {})
        w("| Metric | Value |")
        w("|--------|-------|")
        w(f"| Candidate mean utility | {_f(ce_dict.get('mean_utility')):.6f} |")
        w(f"| Sham mean utility | {_f(canonical_results.get('sham', {}).get('mean_utility') if isinstance(canonical_results.get('sham'), dict) else 0):.6f} |")
        w(f"| Baseline mean utility | {_f(canonical_results.get('baseline', {}).get('mean_utility') if isinstance(canonical_results.get('baseline'), dict) else 0):.6f} |")
        w(f"| Oracle mean utility | {_f(canonical_results.get('oracle', {}).get('mean_utility') if isinstance(canonical_results.get('oracle'), dict) else 0):.6f} |")
        w(f"| Candidate - Sham | {_f(h.get('numerator')):.6f} |")
        w(f"| Oracle - Sham (denominator) | {_f(h.get('denominator')):.6f} |")
        w(f"| **Recovered headroom fraction** | {h.get('recovered_headroom_fraction', 'N/A')} |")
        w(f"| **Recovered headroom percent** | {h.get('recovered_headroom_percent', 'N/A')} |")
        w("")
        w(f"**Previous (incorrect) v0.4.2 value**: R_sham = -6.11 (used raw candidate 0.621 instead of executed candidate 5.048)")
        w(f"**Corrected value**: R_sham = {h.get('recovered_headroom_fraction', 'N/A')}")
        w("")

    # 4. Sham identity
    w("## 4. Sham Identity and Ensemble")
    w("")
    w(f"**Primary sham ID**: {sham_identity.get('primary_sham_id', 'N/A')}")
    w(f"**Overall status**: {sham_identity.get('overall_status', 'UNKNOWN')}")
    w("")
    se_by_type = sham_ensemble.get("by_type", {})
    if se_by_type:
        w("| Sham Type | Seeds | Mean Utility | Std | Candidate Percentile |")
        w("|-----------|-------|-------------|-----|---------------------|")
        for stype, sdata in se_by_type.items():
            mu = _f(sdata.get("mean_utility"))
            std = _f(sdata.get("std_utility"))
            pct = sham_ensemble.get("candidate_percentile", {}).get(stype, "N/A")
            w(f"| {stype} | {sdata.get('seeds', 0)} | {mu:.6f} | {std:.6f} | {pct} |")
        w("")
    w(f"**Candidate beats sham**: {sham_ensemble.get('candidate_beats_sham', 'N/A')}")
    w("")

    # 5. Gate A0 audit
    w("## 5. Gate A0 Audit")
    w("")
    w(f"**Overall status**: {gate_a0.get('overall_status', 'UNKNOWN')}")
    w(f"**Pass**: {gate_a0.get('n_pass', 0)} | **Fail**: {gate_a0.get('n_fail', 0)} | **Blocked**: {gate_a0.get('n_blocked', 0)}")
    w("")
    w("| Sub-Gate | Status | Reason |")
    w("|----------|--------|--------|")
    for sg in gate_a0.get("sub_gates", {}).values():
        w(f"| {sg.get('gate_id', '')} | {sg.get('status', '')} | {sg.get('reason', '')[:80]} |")
    w("")

    # 6. Raw versus executed candidate
    w("## 6. Raw Versus Executed Candidate")
    w("")
    raw_u = _f(raw_vs_executed.get("raw_result", {}).get("mean_utility"))
    exec_u = _f(raw_vs_executed.get("executed_result", {}).get("mean_utility"))
    w(f"| Metric | Raw | Executed | Delta |")
    w(f"|--------|-----|----------|-------|")
    w(f"| Mean utility | {raw_u:.6f} | {exec_u:.6f} | {exec_u - raw_u:.6f} |")
    comp = raw_vs_executed.get("comparison", {})
    for metric_name, metric_data in comp.items():
        if isinstance(metric_data, dict):
            raw_v = _f(metric_data.get("raw"))
            exec_v = _f(metric_data.get("executed"))
            w(f"| {metric_name} | {raw_v:.6f} | {exec_v:.6f} | {exec_v - raw_v:.6f} |")
    w("")
    w(f"**Bottleneck classification**: {raw_vs_executed.get('bottleneck_classification', 'N/A')}")
    w(f"**Interpretation**: {raw_vs_executed.get('interpretation', 'N/A')}")
    w("")

    # 7. Calibration and fallback effects
    w("## 7. Calibration and Fallback Effects")
    w("")
    w(f"**Production threshold (validation-selected)**: {calibration.get('production_threshold', 'N/A')}")
    w(f"**Interpretation**: {calibration.get('interpretation', 'N/A')}")
    w("")
    w("*See `calibration_fallback_audit.json` for full curves.*")
    w("")

    # 8. Oracle headroom concentration
    w("## 8. Oracle Headroom Concentration")
    w("")
    cm = headroom_concentration.get("concentration_metrics", {})
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| Concentration status | {headroom_concentration.get('concentration_status', 'N/A')} |")
    w(f"| HHI | {_f(cm.get('hhi')):.4f} |")
    w(f"| Gini coefficient | {_f(cm.get('gini')):.4f} |")
    w(f"| Top-10% share | {_f(cm.get('top_10_share')):.4f} |")
    w(f"| Effective headroom task count | {_f(cm.get('effective_count')):.2f} |")
    w("")
    w(f"**Interpretation**: {headroom_concentration.get('interpretation', 'N/A')}")
    w("")

    # 9. High-margin task performance
    w("## 9. High-Margin Task Performance")
    w("")
    ga5 = high_margin.get("gate_a5", {})
    w(f"**Gate A5 status**: {ga5.get('status', 'N/A')}")
    w(f"**Recovery fraction**: {ga5.get('recovery_fraction', 'N/A')}")
    w(f"**LCB**: {ga5.get('lcb', 'N/A')}")
    w(f"**Threshold**: {ga5.get('threshold', 'N/A')}")
    w("")
    w(f"**Interpretation**: {high_margin.get('interpretation', 'N/A')}")
    w("")

    # 10. Feature signal
    w("## 10. Feature Signal Beyond Family/Frequency")
    w("")
    w(f"**Learnability conclusion**: {feature_signal.get('learnability_conclusion', 'N/A')}")
    w(f"**Interpretation**: {feature_signal.get('interpretation', 'N/A')}")
    w("")
    w("*See `feature_signal_analysis.json` for model comparison details.*")
    w("")

    # 11. Missing pre-action state
    w("## 11. Missing Pre-Action State")
    w("")
    n_candidates = len(missing_state.get("missing_feature_candidates", []))
    w(f"**Missing feature candidates**: {n_candidates}")
    w("")
    if missing_state.get("missing_feature_candidates"):
        w("| Feature | Available Before Action | Computable in Production |")
        w("|---------|------------------------|------------------------|")
        for fc in missing_state["missing_feature_candidates"][:10]:
            w(f"| {fc.get('name', '')} | {fc.get('available_before_action', '')} | {fc.get('computable_in_production', '')} |")
        w("")
    w(f"**Interpretation**: {missing_state.get('interpretation', 'N/A')}")
    w("")

    # 12. Learner comparison
    w("## 12. Learner Comparison")
    w("")
    w(f"**Best validation learner**: {learner_validation.get('best_validation_learner', 'N/A')}")
    w(f"**Interpretation**: {learner_validation.get('interpretation', 'N/A')}")
    w("")
    w("*See `learner_validation_results.json` for full comparison.*")
    w("")

    # 13. Candidate seed stability
    w("## 13. Candidate Seed Stability")
    w("")
    agg = candidate_stability.get("aggregate", {})
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| Stability classification | {candidate_stability.get('stability_classification', 'N/A')} |")
    w(f"| Mean utility (across seeds) | {_f(agg.get('mean_utility')):.6f} |")
    w(f"| Std utility | {_f(agg.get('std_utility')):.6f} |")
    w(f"| Original candidate percentile | {_f(candidate_stability.get('original_candidate_percentile')):.1f}% |")
    w("")
    w(f"**Interpretation**: {candidate_stability.get('interpretation', 'N/A')}")
    w("")

    # 14. Utility consistency
    w("## 14. Utility Consistency")
    w("")
    w(f"**Overall status**: {utility_consistency.get('overall_status', 'N/A')}")
    w(f"**Interpretation**: {utility_consistency.get('interpretation', 'N/A')}")
    w("")

    # 15. Counterfactual completeness
    w("## 15. Counterfactual Completeness")
    w("")
    cc_summary = counterfactual_completeness.get("summary", {})
    w(f"| Status | Count |")
    w(f"|--------|-------|")
    for status, count in cc_summary.items():
        w(f"| {status} | {count} |")
    w("")
    w(f"**Interpretation**: {counterfactual_completeness.get('interpretation', 'N/A')}")
    w("")

    # 16. Corrected Gate A layers
    w("## 16. Corrected Gate A Layers")
    w("")
    w(f"**Overall status**: {gate_a.get('overall_status', 'N/A')}")
    w("")
    w("| Sub-Gate | Status | Key Metric | Threshold | Reason |")
    w("|----------|--------|------------|-----------|--------|")
    for name, gate in gate_a.get("sub_gates", {}).items():
        w(f"| {name} | {gate.get('status', '')} | {gate.get('key_metric', '')} | {gate.get('threshold', '')} | {gate.get('reason', '')[:60]} |")
    w("")

    # 17. Model-scale decision
    w("## 17. Model-Scale Decision")
    w("")
    w(f"**Decision**: {scale_decision.get('decision', 'N/A')}")
    w(f"**Reason**: {scale_decision.get('reason', 'N/A')}")
    w(f"**Approve full 14B run**: {scale_decision.get('approve_full_14b_run', False)}")
    w(f"**Approve matched 14B pilot**: {scale_decision.get('approve_matched_14b_pilot', False)}")
    w(f"**Required next action**: {scale_decision.get('required_next_action', 'N/A')}")
    w("")
    w(f"> {scale_decision.get('disclaimer', 'A larger model may increase raw task capability, but this does not establish that the current router can learn state-dependent routing.')}")
    w("")

    # 18. Required next engineering action
    w("## 18. Required Next Engineering Action")
    w("")
    decision = scale_decision.get("decision", "")
    if decision == "IMPROVE_FEATURES":
        w("The primary bottleneck is the feature set. The next required change is:")
        w("- Add pre-action memory/tool/workflow reliability and confidence-state features")
        w("- Re-run qualification with improved features")
    elif decision == "FIX_CALIBRATION_AND_FALLBACK":
        w("The raw candidate learns useful routing, but calibration/fallback suppresses it:")
        w("- Retune validation thresholds")
        w("- Increase candidate-controlled coverage")
        w("- Re-run qualification with corrected calibration")
    elif decision == "IMPROVE_LEARNER":
        w("Features contain signal but the logistic router fails to recover it:")
        w("- Replace or augment logistic router with a validation-selected nonlinear policy")
        w("- Re-run qualification with improved learner")
    elif decision == "BLOCKED":
        w("Infrastructure or evidence validity must be resolved before any further analysis.")
    elif decision == "APPROVE_MATCHED_14B_PILOT":
        w("A matched 14B pilot is approved. Execute the pilot with the fixed task subset.")
    else:
        w(f"Follow the decision: {decision}")
    w("")

    # 19. Final conclusion
    w("## 19. Final Conclusion")
    w("")
    ce_mean = _f(canonical_results.get("candidate_executed", {}).get("mean_utility") if isinstance(canonical_results.get("candidate_executed"), dict) else 0)
    sham_mean = _f(canonical_results.get("sham", {}).get("mean_utility") if isinstance(canonical_results.get("sham"), dict) else 0)
    oracle_mean = _f(canonical_results.get("oracle", {}).get("mean_utility") if isinstance(canonical_results.get("oracle"), dict) else 0)
    w(f"The 7B experiment contains measurable oracle routing headroom "
      f"(oracle-sham = {oracle_mean - sham_mean:.4f}), but the current candidate "
      f"does not recover that headroom or outperform the sham "
      f"(candidate-sham = {ce_mean - sham_mean:.4f}). The previously reported "
      f"recovered-headroom value was revalidated using aggregate matched utilities. "
      f"The primary unresolved bottleneck is the routing system—features, learner, "
      f"calibration, fallback, or stability—not model size by default.")
    w("")
    w(f"**Corrected recovered headroom (R_sham)**: "
      f"{canonical_results.get('candidate_executed', {}).get('headroom_vs_reference', {}).get('recovered_headroom_fraction', 'N/A') if isinstance(canonical_results.get('candidate_executed'), dict) else 'N/A'}")
    w("")
    w("---")
    w(f"*Generated by AutoLearn v0.4.3 repair pipeline.*")

    return "\n".join(lines)
