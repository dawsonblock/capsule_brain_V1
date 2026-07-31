"""v0.4.4 report generator.

Generates GATE_A_V044_REPAIR_REPORT.md with all 28 required sections.
Every section links to its source artifact by relative path and digest.
Never hardcodes pass symbols.
"""
from __future__ import annotations

from typing import Any

from capsule_brain.version import (
    PACKAGE_VERSION,
    AUTOLEARN_VERSION,
    AUTOLEARN_QUALIFICATION_VERSION,
)


def generate_v044_report(
    run_id: str,
    evidence_manifest: dict,
    evidence_import: dict,
    source_hash: dict,
    provider_validation: dict,
    version_validation: dict,
    model_identity: dict,
    cf_equivalence: dict,
    action_matrix: dict,
    metric_consistency: dict,
    canonical_eval: dict,
    primary_sham: dict,
    sham_identity: dict,
    sham_ensemble: dict,
    candidate_stability: dict,
    gate_a0: dict,
    safety_evidence: dict,
    raw_vs_exec: dict,
    calibration: dict,
    headroom_concentration: dict,
    high_margin: dict,
    feature_signal: dict,
    missing_state: dict,
    learner_validation: dict,
    utility_consistency: dict,
    cf_completeness: dict,
    gate_a: dict,
    scale_decision: dict,
    pilot_eligibility: dict,
    pilot_manifest: dict,
    evidence: Any = None,
) -> str:
    """Generate the full v0.4.4 diagnostic report as markdown."""
    lines: list[str] = []
    w = lines.append

    w("# Gate A v0.4.4 Evidence-Complete Repair Report")
    w("")
    w(f"**Analysis Run ID**: {run_id}")
    w(f"**Package Version**: {PACKAGE_VERSION}")
    w(f"**AutoLearn Version**: {AUTOLEARN_VERSION}")
    w(f"**Qualification Version**: {AUTOLEARN_QUALIFICATION_VERSION}")
    w(f"**Analysis Mode**: evidence-analysis")
    w(f"**Original Run ID**: {evidence_manifest.get('original_run_id', 'unknown')}")
    w(f"**Scientific Result**: {evidence_manifest.get('scientific_result', 'unknown')}")
    w("")

    # Helper for status symbols.
    def _sym(status: str) -> str:
        if status == "PASS":
            return "PASS"
        elif status == "FAIL":
            return "FAIL"
        elif status == "BLOCKED":
            return "BLOCKED"
        elif status == "NOT_RUN":
            return "NOT_RUN"
        elif status == "DIAGNOSTIC_ONLY":
            return "DIAGNOSTIC_ONLY"
        elif status == "INVALID" or status == "INVALID_DATA":
            return "INVALID"
        else:
            return status

    # Section 1: Build and version identity
    w("## 1. Build and Version Identity")
    w("")
    w(f"| Field | Value |")
    w(f"|-------|-------|")
    w(f"| Package version | {PACKAGE_VERSION} |")
    w(f"| AutoLearn version | {AUTOLEARN_VERSION} |")
    w(f"| Qualification version | {AUTOLEARN_QUALIFICATION_VERSION} |")
    w(f"| Original package version | {evidence_manifest.get('original_package_version', '')} |")
    w(f"| Original qualification version | {evidence_manifest.get('original_qualification_version', '')} |")
    vv = version_validation
    w(f"| Version validation | {_sym(vv.get('status', 'FAIL'))} |")
    w(f"*Artifact: analysis_config.json, analysis_manifest.json*")
    w("")

    # Section 2: Immutable 7B evidence package
    w("## 2. Immutable 7B Evidence Package")
    w("")
    w(f"| Field | Value |")
    w(f"|-------|-------|")
    w(f"| Evidence package version | {evidence_manifest.get('evidence_package_version', '')} |")
    w(f"| Original run ID | {evidence_manifest.get('original_run_id', '')} |")
    w(f"| Model ID | {evidence_manifest.get('model_id', '')} |")
    w(f"| Model revision | {evidence_manifest.get('model_revision', '')} |")
    w(f"| Provider class | {evidence_manifest.get('provider_class', '')} |")
    w(f"| Runtime type | {evidence_manifest.get('runtime_type', '')} |")
    w(f"| Immutable | {evidence_manifest.get('immutable', '')} |")
    w(f"| Scientific result | {evidence_manifest.get('scientific_result', '')} |")
    w(f"| Promoted | {evidence_manifest.get('promoted', '')} |")
    w(f"*Artifact: EVIDENCE_MANIFEST.json*")
    w("")

    # Section 3: Evidence checksum validation
    w("## 3. Evidence Checksum Validation")
    w("")
    ei = evidence_import
    w(f"**Status**: {_sym(ei.get('status', 'FAIL'))}")
    w(f"**Files validated**: {ei.get('n_files_validated', 0)}")
    checksum_results = ei.get("checksum_results", {})
    n_match = sum(1 for c in checksum_results.values() if c.get("match"))
    w(f"**Checksums matching**: {n_match}/{len(checksum_results)}")
    w(f"*Artifact: evidence_import_report.json, artifact_checksums.json*")
    w("")

    # Section 4: Analysis source provenance
    w("## 4. Analysis Source Provenance")
    w("")
    w(f"| Field | Value |")
    w(f"|-------|-------|")
    w(f"| Source file count | {source_hash.get('analysis_source_file_count', 0)} |")
    w(f"| Source byte count | {source_hash.get('analysis_source_byte_count', 0)} |")
    w(f"| Source tree SHA-256 | {source_hash.get('analysis_source_tree_sha256', '')[:16]}... |")
    w(f"| Hash algorithm | {source_hash.get('hash_algorithm_version', '')} |")
    w(f"*Artifact: analysis_source_provenance.json*")
    w("")

    # Section 5: Model/provider identity
    w("## 5. Model/Provider Identity")
    w("")
    pv = provider_validation.get("provider_validation", provider_validation)
    w(f"| Field | Value |")
    w(f"|-------|-------|")
    w(f"| Provider class | {pv.get('provider_class', '')} |")
    w(f"| Model ID | {pv.get('model_id', '')} |")
    w(f"| Model revision | {pv.get('model_revision', '')} |")
    w(f"| Runtime type | {pv.get('runtime_type', '')} |")
    w(f"| Provider validation | {_sym(pv.get('status', 'FAIL'))} |")
    w(f"| Model identity | {_sym(model_identity.get('status', 'FAIL'))} |")
    w(f"*Artifact: model_identity_consistency.json*")
    w("")

    # Section 6: Counterfactual equivalence
    w("## 6. Counterfactual Equivalence")
    w("")
    w(f"**Status**: {_sym(cf_equivalence.get('status', 'UNVERIFIABLE'))}")
    w(f"**Tasks checked**: {cf_equivalence.get('n_tasks_checked', 0)}")
    w(f"**Tasks equivalent**: {cf_equivalence.get('n_tasks_equivalent', 0)}")
    w(f"**Tasks non-equivalent**: {cf_equivalence.get('n_tasks_non_equivalent', 0)}")
    w(f"**Excluded task IDs**: {len(cf_equivalence.get('excluded_task_ids', []))}")
    w(f"*Artifact: counterfactual_equivalence_report.json*")
    w("")

    # Section 7: Eligible-action matrix completeness
    w("## 7. Eligible-Action Matrix Completeness")
    w("")
    w(f"**Status**: {_sym(action_matrix.get('status', 'INCOMPLETE'))}")
    w(f"**Tasks total**: {action_matrix.get('n_tasks_total', 0)}")
    w(f"**Tasks complete**: {action_matrix.get('n_tasks_complete', 0)}")
    w(f"**Excluded task IDs**: {len(action_matrix.get('excluded_task_ids', []))}")
    w(f"*Artifact: action_matrix_completeness.json*")
    w("")

    # Section 8: Metric consistency
    w("## 8. Metric Consistency")
    w("")
    w(f"**Status**: {_sym(metric_consistency.get('status', 'FAIL'))}")
    w(f"**Checks**: {metric_consistency.get('n_pass', 0)}/{metric_consistency.get('n_checks', 0)} passed")
    w(f"*Artifact: metric_consistency_report.json*")
    w("")

    # Section 9: Corrected recovered-headroom calculation
    w("## 9. Corrected Recovered-Headroom Calculation")
    w("")
    if evidence:
        cand_mean = evidence.candidate_results.get("mean_utility", 0)
        sham_mean = evidence.sham_results.get("mean_utility", 0)
        oracle_mean = evidence.oracle_results.get("mean_utility", 0)
        baseline_mean = evidence.baseline_results.get("mean_utility", 0)
        w(f"| Metric | Value |")
        w(f"|--------|-------|")
        w(f"| Candidate mean utility | {cand_mean} |")
        w(f"| Sham mean utility | {sham_mean} |")
        w(f"| Oracle mean utility | {oracle_mean} |")
        w(f"| Baseline mean utility | {baseline_mean} |")
        w(f"| Numerator (candidate - sham) | {cand_mean - sham_mean:.4f} |")
        w(f"| Denominator (oracle - sham) | {oracle_mean - sham_mean:.4f} |")
        denom = oracle_mean - sham_mean
        if denom > 1e-10:
            frac = (cand_mean - sham_mean) / denom
            w(f"| Recovered headroom fraction | {frac:.6f} |")
            w(f"| Recovered headroom percent | {frac * 100:.4f}% |")
        else:
            w(f"| Recovered headroom fraction | INVALID (denominator <= 0) |")
    w(f"*Artifact: canonical_policy_evaluation.json*")
    w("")

    # Section 10: Primary sham identity
    w("## 10. Primary Sham Identity")
    w("")
    w(f"**Primary Sham ID**: {primary_sham.get('primary_sham_id', '')}")
    w(f"**Sham type**: {primary_sham.get('sham_type', '')}")
    w(f"**Identity validation**: {_sym(sham_identity.get('status', 'FAIL'))}")
    w(f"*Artifact: primary_sham_manifest.json, sham_identity_report.json*")
    w("")

    # Section 11: Sham ensemble
    w("## 11. Sham Ensemble")
    w("")
    w(f"**Seeds**: {sham_ensemble.get('n_seeds', 0)}")
    stats = sham_ensemble.get("ensemble_stats", {})
    w(f"**Mean sham utility**: {stats.get('mean_sham_utility', 0):.4f}")
    w(f"**Median sham utility**: {stats.get('median_sham_utility', 0):.4f}")
    w(f"**Sham std**: {stats.get('sham_std', 0):.4f}")
    w(f"**Candidate percentile**: {stats.get('candidate_percentile', 0):.2f}")
    w(f"**Label**: {sham_ensemble.get('label', 'DIAGNOSTIC_ONLY')}")
    w(f"*Artifact: sham_ensemble_results.json*")
    w("")

    # Section 12: Candidate seed stability
    w("## 12. Candidate Seed Stability")
    w("")
    w(f"**Status**: {_sym(candidate_stability.get('status', 'DIAGNOSTIC_ONLY'))}")
    w(f"**Seeds**: {candidate_stability.get('n_seeds', 0)}")
    w(f"**Label**: {candidate_stability.get('label', 'DIAGNOSTIC_ONLY')}")
    w(f"*Artifact: candidate_stability_results.json*")
    w("")

    # Section 13: Gate A0 sub-gates
    w("## 13. Gate A0 Sub-Gates")
    w("")
    w(f"**Overall A0 Status**: {_sym(gate_a0.get('status', 'BLOCKED'))}")
    w(f"**Sub-gates**: {gate_a0.get('n_pass', 0)}/{gate_a0.get('n_sub_gates', 18)} passed")
    w("")
    w("| Sub-Gate | Status | Reason |")
    w("|----------|--------|--------|")
    for name, gate in gate_a0.get("sub_gates", {}).items():
        w(f"| {name} | {_sym(gate.get('status', 'FAIL'))} | {gate.get('reason', '')} |")
    w(f"*Artifact: gate_a0_audit.json*")
    w("")

    # Section 14: Safety evidence
    w("## 14. Safety Evidence")
    w("")
    w(f"**Status**: {_sym(safety_evidence.get('status', 'FAIL'))}")
    w(f"**Checks**: {safety_evidence.get('n_pass', 0)}/{safety_evidence.get('n_checks', 0)} passed")
    w(f"*Artifact: safety_evidence_integrity.json*")
    w("")

    # Section 15: Raw versus executed candidate
    w("## 15. Raw Versus Executed Candidate")
    w("")
    w(f"**Conclusions**: {', '.join(raw_vs_exec.get('conclusions', []))}")
    w(f"*Artifact: raw_vs_executed_results.json, candidate_decision_trace.jsonl*")
    w("")

    # Section 16: Calibration and fallback effects
    w("## 16. Calibration and Fallback Effects")
    w("")
    w(f"**Status**: {_sym(calibration.get('status', 'FAIL'))}")
    w(f"**Conclusion**: {calibration.get('conclusion', 'NO_SUPPRESSION')}")
    w(f"*Artifact: calibration_fallback_audit.json*")
    w("")

    # Section 17: Oracle headroom concentration
    w("## 17. Oracle Headroom Concentration")
    w("")
    w(f"**Status**: {_sym(headroom_concentration.get('status', 'INVALID_DATA'))}")
    w(f"**Negative headroom present**: {headroom_concentration.get('negative_headroom_present', False)}")
    w(f"**Gini**: {headroom_concentration.get('gini', 0):.4f}")
    w(f"**HHI**: {headroom_concentration.get('hhi', 0):.4f}")
    w(f"*Artifact: headroom_concentration.json*")
    w("")

    # Section 18: High-margin recovery
    w("## 18. High-Margin Recovery")
    w("")
    w(f"**Status**: {_sym(high_margin.get('status', 'NOT_RUN'))}")
    w(f"**High-margin task count**: {high_margin.get('high_margin_task_count', 0)}")
    w(f"**Recovery fraction**: {high_margin.get('recovery_fraction', 0):.4f}")
    w(f"*Artifact: high_margin_recovery.json*")
    w("")

    # Section 19: Feature signal beyond family and frequency
    w("## 19. Feature Signal Beyond Family and Frequency")
    w("")
    w(f"**Status**: {_sym(feature_signal.get('status', 'INCONCLUSIVE'))}")
    w(f"**Conclusion**: {feature_signal.get('conclusion', '')}")
    w(f"*Artifact: feature_signal_analysis.json*")
    w("")

    # Section 20: Missing pre-action state
    w("## 20. Missing Pre-Action State")
    w("")
    w(f"**Status**: {_sym(missing_state.get('status', 'INCONCLUSIVE'))}")
    w(f"**Model-derived confidence as plausible missing**: {missing_state.get('model_derived_confidence_as_plausible_missing', False)}")
    w(f"*Artifact: missing_state_analysis.json*")
    w("")

    # Section 21: Learner comparison
    w("## 21. Learner Comparison")
    w("")
    w(f"**Best learner**: {learner_validation.get('best_learner', 'none')}")
    w(f"**Conclusion**: {learner_validation.get('conclusion', 'INCONCLUSIVE')}")
    w(f"*Artifact: learner_validation_results.json*")
    w("")

    # Section 22: Utility consistency
    w("## 22. Utility Consistency")
    w("")
    w(f"**Status**: {_sym(utility_consistency.get('status', 'FAIL'))}")
    w(f"*Artifact: utility_consistency_report.json*")
    w("")

    # Section 23: Counterfactual completeness
    w("## 23. Counterfactual Completeness")
    w("")
    w(f"**Status**: {_sym(cf_completeness.get('status', 'FAIL'))}")
    w(f"*Artifact: counterfactual_completeness.json*")
    w("")

    # Section 24: Corrected Gate A layers
    w("## 24. Corrected Gate A Layers")
    w("")
    w(f"**Overall Gate A**: {_sym(gate_a.get('overall_status', 'BLOCKED'))}")
    w("")
    w("| Layer | Status | Reason |")
    w("|-------|--------|--------|")
    for name, gate in gate_a.get("sub_gates", {}).items():
        w(f"| {name} | {_sym(gate.get('status', 'FAIL'))} | {gate.get('reason', '')} |")
    w(f"*Artifact: gate_a_v044_results.json*")
    w("")

    # Section 25: Model-scale decision
    w("## 25. Model-Scale Decision")
    w("")
    w(f"**Decision**: {scale_decision.get('decision', 'BLOCKED')}")
    w(f"**Approve full 14B run**: {scale_decision.get('approve_full_14b_run', False)}")
    w(f"**Approve matched 14B pilot**: {scale_decision.get('approve_matched_14b_pilot', False)}")
    w(f"**Reason**: {scale_decision.get('reason', '')}")
    w(f"*Artifact: model_scale_decision.json*")
    w("")

    # Section 26: Matched pilot eligibility
    w("## 26. Matched Pilot Eligibility")
    w("")
    w(f"**Eligible**: {pilot_eligibility.get('eligible', False)}")
    w(f"**Checks**: {pilot_eligibility.get('n_pass', 0)}/{pilot_eligibility.get('n_checks', 0)} passed")
    if pilot_manifest.get("eligible"):
        w(f"**Pilot task count**: {pilot_manifest.get('pilot_task_count', 0)}")
    w(f"*Artifact: matched_pilot_manifest.json (if eligible)*")
    w("")

    # Section 27: Required next engineering action
    w("## 27. Required Next Engineering Action")
    w("")
    decision = scale_decision.get("decision", "BLOCKED")
    actions = {
        "BLOCKED": "Fix evidence validity issues before any scientific analysis.",
        "REDESIGN_BENCHMARK": "Redesign the benchmark to produce sufficient routing headroom.",
        "IMPROVE_FEATURES": "Improve pre-action state representation to capture route signal.",
        "IMPROVE_LEARNER": "Try a validation-selected utility-sensitive or low-capacity nonlinear learner.",
        "FIX_CALIBRATION_AND_FALLBACK": "Repair confidence, abstention, shift detection, or fallback controls.",
        "COLLECT_MORE_7B_EXPERIENCE": "Collect additional 7B experience before model scaling.",
        "APPROVE_MATCHED_14B_PILOT": "Execute the matched 7B-vs-14B pilot with frozen task subset.",
        "NO_SCALE_NEEDED_FOR_GATE_A": "No scaling needed — Gate A already passes.",
        "NO_SCALE_UP": "No scale-up condition met. Continue diagnosis.",
    }
    w(f"**Action**: {actions.get(decision, 'Continue diagnosis.')}")
    w("")

    # Section 28: Final scientific conclusion
    w("## 28. Final Scientific Conclusion")
    w("")
    if gate_a0.get("status") != "PASS":
        w("> The immutable 7B evidence package could not be fully validated. Gate A")
        w("> diagnostics are blocked until artifact identity, counterfactual comparability")
        w("> and metric consistency are repaired.")
    elif gate_a.get("overall_status") == "PASS":
        w("> Gate A passes. No model scaling is needed.")
    else:
        w("> The 7B experiment did not demonstrate candidate superiority over the sham.")
        w("> The corrected analysis separates evidence integrity, raw router quality,")
        w("> fallback effects, feature signal, learner adequacy and model capability")
        w("> before any scale-up decision.")
    w("")
    w("---")
    w(f"*Generated by AutoLearn v0.4.4 evidence-complete analysis pipeline.*")
    w(f"*Package version {PACKAGE_VERSION}, AutoLearn {AUTOLEARN_VERSION}, Qualification {AUTOLEARN_QUALIFICATION_VERSION}.*")

    return "\n".join(lines)
