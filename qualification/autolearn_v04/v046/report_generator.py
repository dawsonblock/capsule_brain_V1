"""v0.4.6 evidence repair report generator.

Generates GATE_A_V046_EVIDENCE_REPAIR_REPORT.md with all required sections.
Every BLOCKED section identifies the exact missing evidence.
No synthetic numeric conclusions when evidence is unavailable.

Allowed report language:
    - For synthetic: "The synthetic routing fixture is structurally complete
      and suitable for testing evidence schemas, policy evaluation, Gate A
      arithmetic and audit plumbing. It is not real-model evidence, is not
      scientifically claim-eligible, cannot be promoted and cannot support
      model-scale decisions."
    - For unavailable historical: "The original Modal 7B task-level artifacts
      are not available in this repository. The historical Gate A result cannot
      be reproduced from the current source package. Synthetic fixtures are
      maintained separately and are not used as substitutes."
"""
from __future__ import annotations

from typing import Any

from capsule_brain.version import (
    AUTOLEARN_QUALIFICATION_VERSION,
    AUTOLEARN_VERSION,
    PACKAGE_VERSION,
)


# Allowed report language — must not be paraphrased.
SYNTHETIC_LANGUAGE = (
    "The synthetic routing fixture is structurally complete and suitable for "
    "testing evidence schemas, policy evaluation, Gate A arithmetic and audit "
    "plumbing. It is not real-model evidence, is not scientifically "
    "claim-eligible, cannot be promoted and cannot support model-scale "
    "decisions."
)

UNAVAILABLE_HISTORICAL_LANGUAGE = (
    "The original Modal 7B task-level artifacts are not available in this "
    "repository. The historical Gate A result cannot be reproduced from the "
    "current source package. Synthetic fixtures are maintained separately and "
    "are not used as substitutes."
)


def _sym(status: str) -> str:
    """Return the status symbol unchanged (no mapping)."""
    return status if status else "BLOCKED"


def _status(report: dict | None, default: str = "BLOCKED") -> str:
    if report is None:
        return default
    if not isinstance(report, dict):
        return default
    return report.get("status", default)


def _reason(report: dict | None, default: str = "") -> str:
    if report is None or not isinstance(report, dict):
        return default
    return report.get("reason", default)


def generate_v046_report(
    run_id: str,
    evidence_manifest: dict,
    evidence_origin: dict,
    fixture_validation: dict | None,
    scientific_validation: dict | None,
    cross_origin: dict | None,
    experience_normalized: dict | None,
    evidence_weights: dict | None,
    verifier_registry: dict | None,
    cf_equivalence: dict | None,
    action_matrix: dict | None,
    split_access: dict | None,
    candidate_parity: dict | None,
    sham_parity: dict | None,
    artifact_lineage: dict | None,
    oracle_discrepancy: dict | None,
    historical_identity: dict | None,
    analysis_identity: dict | None,
    cross_version: dict | None,
    task_split: dict | None,
    safety_evidence: dict | None,
    utility_consistency: dict | None,
    metric_consistency: dict | None,
    gate_a0: dict | None,
    scale_decision: dict | None,
) -> str:
    """Generate the full v0.4.6 evidence repair report as markdown.

    Parameters are the artifact dicts produced by each pipeline stage.
    Any may be None if the stage was skipped or produced no output.
    """
    lines: list[str] = []
    w = lines.append

    origin = evidence_origin or {}
    origin_str = str(origin.get("origin", "")).upper()
    is_synthetic = origin_str == "SYNTHETIC"

    w("# Gate A v0.4.6 Evidence Repair Report")
    w("")
    w(f"**Analysis Run ID**: {run_id}")
    w(f"**Package Version**: {PACKAGE_VERSION}")
    w(f"**AutoLearn Version**: {AUTOLEARN_VERSION}")
    w(f"**Qualification Version**: {AUTOLEARN_QUALIFICATION_VERSION}")
    w(f"**Analysis Mode**: evidence-analysis")
    w(f"**Source Evidence Run ID**: {evidence_manifest.get('original_run_id', 'unknown')}")
    w(f"**Evidence Origin**: {origin_str or 'unknown'}")
    w(f"**Scientific Result**: {evidence_manifest.get('scientific_result', 'unknown')}")
    w("")

    # --- Section 1: Build identity ---
    w("## 1. Build Identity")
    w("")
    w("| Field | Value |")
    w("|-------|-------|")
    w(f"| Package version | {PACKAGE_VERSION} |")
    w(f"| AutoLearn version | {AUTOLEARN_VERSION} |")
    w(f"| Qualification version | {AUTOLEARN_QUALIFICATION_VERSION} |")
    w(f"| Original package version | {evidence_manifest.get('original_package_version', '')} |")
    w(f"| Original qualification version | {evidence_manifest.get('original_qualification_version', '')} |")
    w("")

    # --- Section 2: Evidence origin ---
    w("## 2. Evidence Origin")
    w("")
    w(f"**Status**: {_sym(origin.get('status', 'BLOCKED'))}")
    w("| Field | Value |")
    w("|-------|-------|")
    w(f"| Origin | {origin.get('origin', '')} |")
    w(f"| Runtime type | {origin.get('runtime_type', '')} |")
    w(f"| Model ID | {origin.get('model_id', '')} |")
    w(f"| Tokenizer ID | {origin.get('tokenizer_id', '')} |")
    w(f"| Provider class | {origin.get('provider_class', '')} |")
    w(f"| Scientific claim eligible | {origin.get('scientific_claim_eligible', '')} |")
    w(f"| Promotable | {origin.get('promotable', '')} |")
    w(f"| Supports Gate A | {origin.get('supports_gate_a', '')} |")
    w(f"| Supports Gate B | {origin.get('supports_gate_b', '')} |")
    if origin.get("misclassification_violations"):
        w(f"**Misclassification violations**: {origin['misclassification_violations']}")
    w(f"*Reason*: {origin.get('reason', '')}")
    w("")

    # --- Section 3: Fixture validation ---
    w("## 3. Fixture Validation")
    w("")
    if fixture_validation:
        w(f"**Structural validity**: {_sym(fixture_validation.get('structural_validity', 'BLOCKED'))}")
        w(f"**Evidence origin**: {fixture_validation.get('evidence_origin', '')}")
        w(f"**Scientific claim eligibility**: {fixture_validation.get('scientific_claim_eligibility', '')}")
        w(f"**Gate A eligible**: {fixture_validation.get('gate_a_eligible', '')}")
        w(f"**Promotable**: {fixture_validation.get('promotable', '')}")
        checks = fixture_validation.get("checks", {})
        if isinstance(checks, dict) and checks:
            w("")
            w("| Check | Status | Observed | Expected | Reason |")
            w("|------|--------|----------|----------|--------|")
            for cname, cval in checks.items():
                if isinstance(cval, dict):
                    w(f"| {cname} | {cval.get('status', '')} | {cval.get('observed', '')} | {cval.get('expected', '')} | {cval.get('reason', '')} |")
        w(f"*Reason*: {fixture_validation.get('reason', '')}")
    else:
        w("Not applicable — evidence origin is not SYNTHETIC.")
    w("")

    # --- Section 4: Scientific evidence validation ---
    w("## 4. Scientific Evidence Validation")
    w("")
    if scientific_validation:
        for dim_name, dim_key in [
            ("Structural completeness", "structural_completeness"),
            ("Origin authenticity", "origin_authenticity"),
            ("Scientific eligibility", "scientific_eligibility"),
            ("Gate A eligibility", "gate_a_eligibility"),
        ]:
            dim = scientific_validation.get(dim_key, {})
            if isinstance(dim, dict):
                w(f"### {dim_name}")
                w(f"**Status**: {_sym(dim.get('status', 'BLOCKED'))}")
                w(f"*Reason*: {dim.get('reason', '')}")
                w("")
    else:
        w("Not applicable — evidence origin is not REAL_MODEL.")
    w("")

    # --- Section 5: Cross-origin duplication ---
    w("## 5. Cross-Origin Duplication")
    w("")
    w(f"**Status**: {_sym(_status(cross_origin))}")
    if cross_origin:
        w(f"**Packages scanned**: {cross_origin.get('packages_scanned', 0)}")
        violations = cross_origin.get("violations", [])
        w(f"**Violations**: {len(violations)}")
        if violations:
            w("")
            w("| Severity | Package A | Origin A | Package B | Origin B | Matching digests |")
            w("|----------|-----------|----------|-----------|----------|------------------|")
            for v in violations:
                w(f"| {v.get('severity', '')} | {v.get('package_a', '')} | {v.get('origin_a', '')} | {v.get('package_b', '')} | {v.get('origin_b', '')} | {v.get('matching_digests', [])} |")
    w(f"*Reason*: {_reason(cross_origin)}")
    w("")

    # --- Section 6: Experience normalization ---
    w("## 6. Experience Normalization")
    w("")
    if experience_normalized:
        w(f"**Status**: {_sym(experience_normalized.get('status', 'BLOCKED'))}")
        w(f"**Normalized rows**: {experience_normalized.get('n_normalized_rows', 0)}")
        w(f"**Schema version**: {experience_normalized.get('schema_version', '')}")
        w(f"*Reason*: {experience_normalized.get('reason', '')}")
    else:
        w("**Status**: BLOCKED")
        w("*Reason*: experience normalization not performed")
    w("")

    # --- Section 7: Evidence weights ---
    w("## 7. Evidence Weight Audit")
    w("")
    w(f"**Status**: {_sym(_status(evidence_weights))}")
    if evidence_weights:
        w(f"**Experience task rows found**: {evidence_weights.get('experience_task_rows_found', 0)}")
        w(f"**Experience action rows found**: {evidence_weights.get('experience_action_rows_found', 0)}")
        w(f"**Rows with quality fields**: {evidence_weights.get('rows_containing_quality_fields', 0)}")
        w(f"**Positive final weights**: {evidence_weights.get('positive_final_weights', 0)}")
        w(f"**Zero weights**: {evidence_weights.get('zero_weights', 0)}")
        w(f"**Negative weights**: {evidence_weights.get('negative_weights', 0)}")
        w(f"**Non-finite weights**: {evidence_weights.get('nonfinite_weights', 0)}")
        w(f"**Mean q_total**: {evidence_weights.get('mean_q_total', 0.0)}")
    w(f"*Reason*: {_reason(evidence_weights)}")
    w("")

    # --- Section 8: Verifier registry ---
    w("## 8. Verifier Registry Audit")
    w("")
    w(f"**Status**: {_sym(_status(verifier_registry))}")
    if verifier_registry:
        w(f"**Used verifier count**: {verifier_registry.get('used_verifier_count', 0)}")
        w(f"**Registered verifier count**: {verifier_registry.get('registered_verifier_count', 0)}")
        unknown = verifier_registry.get("unknown_verifiers", [])
        w(f"**Unknown verifiers**: {len(unknown)}")
        if unknown:
            for u in unknown:
                w(f"  - {u}")
    w(f"*Reason*: {_reason(verifier_registry)}")
    w("")

    # --- Section 9: Counterfactual equivalence ---
    w("## 9. Counterfactual Equivalence")
    w("")
    w(f"**Status**: {_sym(_status(cf_equivalence))}")
    if cf_equivalence:
        w(f"**Tasks checked**: {cf_equivalence.get('n_tasks_checked', 0)}")
        w(f"**Tasks equivalent**: {cf_equivalence.get('n_equivalent', 0)}")
        w(f"**Tasks non-equivalent**: {cf_equivalence.get('n_non_equivalent', 0)}")
        w(f"**Tasks unverifiable**: {cf_equivalence.get('n_unverifiable', 0)}")
        unverifiable = cf_equivalence.get("unverifiable_task_ids", [])
        if unverifiable:
            w(f"**Unverifiable task IDs**: {unverifiable}")
    w(f"*Reason*: {_reason(cf_equivalence)}")
    w("")

    # --- Section 10: Action matrix ---
    w("## 10. Action Matrix Completeness")
    w("")
    w(f"**Status**: {_sym(_status(action_matrix))}")
    if action_matrix:
        w(f"**Tasks total**: {action_matrix.get('n_tasks_total', 0)}")
        w(f"**Tasks complete**: {action_matrix.get('n_tasks_complete', 0)}")
        w(f"**Eligible actions**: {action_matrix.get('n_eligible_actions', 0)}")
        excluded = action_matrix.get("excluded_task_ids", [])
        if excluded:
            w(f"**Excluded task IDs**: {excluded}")
    w(f"*Reason*: {_reason(action_matrix)}")
    w("")

    # --- Section 11: Split access ---
    w("## 11. Split Access Audit")
    w("")
    w(f"**Status**: {_sym(_status(split_access))}")
    if split_access:
        w(f"**Stages audited**: {split_access.get('n_stages_audited', 0)}")
        forbidden = split_access.get("forbidden_accesses", [])
        w(f"**Forbidden accesses**: {len(forbidden)}")
        if forbidden:
            for fa in forbidden:
                w(f"  - {fa.get('stage', '')} -> {fa.get('split', '')}: {fa.get('reason', '')}")
    w(f"*Reason*: {_reason(split_access)}")
    w("")

    # --- Section 12: Serialization parity ---
    w("## 12. Serialization Parity")
    w("")
    w("### Candidate")
    w(f"**Status**: {_sym(_status(candidate_parity))}")
    if candidate_parity:
        w(f"**Policy hash**: {candidate_parity.get('policy_hash', '')}")
        w(f"**Max abs diff**: {candidate_parity.get('max_abs_diff', 0.0)}")
        w(f"**Selected action parity**: {candidate_parity.get('selected_action_parity', False)}")
    w(f"*Reason*: {_reason(candidate_parity)}")
    w("")
    w("### Sham")
    w(f"**Status**: {_sym(_status(sham_parity))}")
    if sham_parity:
        w(f"**Policy hash**: {sham_parity.get('policy_hash', '')}")
        w(f"**Max abs diff**: {sham_parity.get('max_abs_diff', 0.0)}")
        w(f"**Selected action parity**: {sham_parity.get('selected_action_parity', False)}")
    w(f"*Reason*: {_reason(sham_parity)}")
    w("")

    # --- Section 13: Artifact lineage ---
    w("## 13. Artifact Lineage")
    w("")
    w(f"**Status**: {_sym(_status(artifact_lineage))}")
    if artifact_lineage:
        w(f"**Artifacts**: {artifact_lineage.get('n_artifacts', 0)}")
        w(f"**Valid**: {artifact_lineage.get('n_valid', 0)}")
        w(f"**Missing parents**: {artifact_lineage.get('n_missing_parents', 0)}")
        w(f"**Cycles**: {artifact_lineage.get('n_cycles', 0)}")
        w(f"**Cross-origin**: {artifact_lineage.get('n_cross_origin', 0)}")
    w(f"*Reason*: {_reason(artifact_lineage)}")
    w("")

    # --- Section 14: Oracle discrepancy ---
    w("## 14. Oracle Discrepancy")
    w("")
    w(f"**Status**: {_sym(_status(oracle_discrepancy))}")
    if oracle_discrepancy:
        w(f"**Classification**: {oracle_discrepancy.get('classification', '')}")
        w(f"**Oracle mean**: {oracle_discrepancy.get('oracle_mean', '')}")
        w(f"**Sham mean**: {oracle_discrepancy.get('sham_mean', '')}")
        w(f"**Oracle minus sham**: {oracle_discrepancy.get('oracle_minus_sham', '')}")
        w(f"**Discrepancy found**: {oracle_discrepancy.get('discrepancy_found', '')}")
    w(f"*Reason*: {_reason(oracle_discrepancy)}")
    w("")

    # --- Section 15: Historical identity ---
    w("## 15. Historical Identity")
    w("")
    w(f"**Status**: {_sym(_status(historical_identity))}")
    if historical_identity:
        w(f"**Original package version**: {historical_identity.get('original_package_version', '')}")
        w(f"**Original qualification version**: {historical_identity.get('original_qualification_version', '')}")
        w(f"**Original commit SHA**: {historical_identity.get('original_commit_sha', '')}")
        w(f"**Original source tree SHA-256**: {historical_identity.get('original_source_tree_sha256', '')}")
        errs = historical_identity.get("errors", [])
        if errs:
            w(f"**Errors**: {'; '.join(errs)}")
    w(f"*Reason*: {_reason(historical_identity)}")
    w("")

    # --- Section 16: Analysis identity ---
    w("## 16. Current Analysis Identity")
    w("")
    w(f"**Status**: {_sym(_status(analysis_identity))}")
    if analysis_identity:
        w(f"**Analysis package version**: {analysis_identity.get('analysis_package_version', '')}")
        w(f"**Analysis qualification version**: {analysis_identity.get('analysis_qualification_version', '')}")
        sha = analysis_identity.get('analysis_source_tree_sha256', '')
        if sha:
            w(f"**Analysis source tree SHA-256**: {sha[:16]}...")
        w(f"**Config digest**: {analysis_identity.get('config_digest', '')[:16]}...")
    w(f"*Reason*: {_reason(analysis_identity)}")
    w("")

    # --- Section 17: Cross-version lineage ---
    w("## 17. Cross-Version Lineage")
    w("")
    w(f"**Status**: {_sym(_status(cross_version))}")
    if cross_version:
        w(f"**Historical version**: {cross_version.get('historical_version', '')}")
        w(f"**Analysis version**: {cross_version.get('analysis_version', '')}")
        w(f"**Version gap valid**: {cross_version.get('version_gap_valid', False)}")
        w(f"**Lineage complete**: {cross_version.get('lineage_complete', False)}")
        errs = cross_version.get("errors", [])
        if errs:
            w(f"**Errors**: {'; '.join(errs)}")
    w(f"*Reason*: {_reason(cross_version)}")
    w("")

    # --- Section 18: Task/split consistency ---
    w("## 18. Task/Split Consistency")
    w("")
    w(f"**Status**: {_sym(_status(task_split))}")
    if task_split:
        w(f"**Benchmark task count**: {task_split.get('benchmark_task_count', 0)}")
        w(f"**Unknown task IDs**: {len(task_split.get('unknown_task_ids', []))}")
        w(f"**Duplicate task IDs**: {len(task_split.get('duplicate_task_ids', []))}")
        w(f"**Test in experience**: {len(task_split.get('test_in_experience', []))}")
    w(f"*Reason*: {_reason(task_split)}")
    w("")

    # --- Section 19: Safety evidence ---
    w("## 19. Safety Evidence Integrity")
    w("")
    w(f"**Status**: {_sym(_status(safety_evidence))}")
    if safety_evidence:
        w(f"**Safety task count**: {safety_evidence.get('safety_task_count', 0)}")
        errs = safety_evidence.get("errors", [])
        if errs:
            w(f"**Errors**: {'; '.join(errs[:5])}")
    w(f"*Reason*: {_reason(safety_evidence)}")
    w("")

    # --- Section 20: Utility consistency ---
    w("## 20. Utility Consistency")
    w("")
    w(f"**Status**: {_sym(_status(utility_consistency))}")
    if utility_consistency:
        w(f"**Checks**: {utility_consistency.get('n_checks', 0)}")
        w(f"**Failures**: {utility_consistency.get('n_fail', 0)}")
    w(f"*Reason*: {_reason(utility_consistency)}")
    w("")

    # --- Section 21: Metric consistency ---
    w("## 21. Metric Consistency")
    w("")
    w(f"**Status**: {_sym(_status(metric_consistency))}")
    if metric_consistency:
        w(f"**Checks**: {metric_consistency.get('n_checks', 0)}")
        w(f"**Failures**: {metric_consistency.get('n_fail', 0)}")
    w(f"*Reason*: {_reason(metric_consistency)}")
    w("")

    # --- Section 22: Gate A0 ---
    w("## 22. Gate A0")
    w("")
    w(f"**Status**: {_sym(_status(gate_a0))}")
    if gate_a0:
        w(f"**Sub-gates**: {gate_a0.get('n_sub_gates', 0)}")
        w(f"**Pass**: {gate_a0.get('n_pass', 0)}")
        w(f"**Fail**: {gate_a0.get('n_fail', 0)}")
        w(f"**Blocked**: {gate_a0.get('n_blocked', 0)}")
        w(f"**Not applicable**: {gate_a0.get('n_not_applicable', 0)}")
        sub_gates = gate_a0.get("sub_gates", {})
        if isinstance(sub_gates, dict) and sub_gates:
            w("")
            w("| Sub-gate | Status | Reason |")
            w("|----------|--------|--------|")
            for sg_name in sorted(sub_gates.keys()):
                sg = sub_gates[sg_name]
                if isinstance(sg, dict):
                    w(f"| {sg_name} | {sg.get('status', '')} | {sg.get('reason', '')} |")
        w(f"*Overall reason*: {gate_a0.get('overall_reason', '')}")
    w("")

    # --- Section 23: Scale decision ---
    w("## 23. Scale Decision")
    w("")
    w(f"**Decision**: {scale_decision.get('decision', 'BLOCKED') if scale_decision else 'BLOCKED'}")
    if scale_decision:
        w(f"**Approve matched 14B pilot**: {scale_decision.get('approve_matched_14b_pilot', False)}")
        w(f"**Approve full 14B run**: {scale_decision.get('approve_full_14b_run', False)}")
        w(f"*Reason*: {scale_decision.get('reason', '')}")
    w("")

    # --- Section 24: Conclusions ---
    w("## 24. Conclusions")
    w("")
    if is_synthetic:
        w(SYNTHETIC_LANGUAGE)
    else:
        # Check if historical evidence is unavailable
        hist_status = _status(historical_identity)
        if hist_status in ("BLOCKED", "FAIL"):
            w(UNAVAILABLE_HISTORICAL_LANGUAGE)
        else:
            a0_status = _status(gate_a0)
            if a0_status == "PASS":
                w("Gate A0 passed. Evidence is eligible for scientific claims.")
            else:
                w(f"Gate A0 status: {a0_status}. Evidence does not meet all Gate A0 requirements.")
    w("")

    return "\n".join(lines)
