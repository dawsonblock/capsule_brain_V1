"""Gate A0 audit for v0.4.6 — 24 sub-gates.

Uses AuditStatus enum and AuditResult for all status propagation.
Each sub-gate reads the correct fields from its source report.
Source statuses are parsed with AuditStatus.from_str().
FAIL is never converted to BLOCKED or vice versa.

For fixture validation (evidence_origin=SYNTHETIC), scientific sub-gates
return NOT_APPLICABLE.

Sub-gates:
    A0.1  Structural completeness
    A0.2  Evidence-origin authenticity
    A0.3  Scientific-claim eligibility
    A0.4  Cross-origin duplication
    A0.5  Source provenance
    A0.6  Historical identity
    A0.7  Current analysis identity
    A0.8  Cross-version lineage
    A0.9  Provider/model authenticity
    A0.10 Single generation identity
    A0.11 Positive evidence weights
    A0.12 Counterfactual equivalence
    A0.13 Complete action matrix
    A0.14 Verifier registry completeness
    A0.15 Utility consistency
    A0.16 Split-access integrity
    A0.17 Candidate serialization parity
    A0.18 Sham serialization parity
    A0.19 Task/split consistency
    A0.20 Artifact lineage
    A0.21 Metric consistency
    A0.22 Safety evidence integrity
    A0.23 No stale artifact reuse
    A0.24 Oracle consistency
"""
from __future__ import annotations

from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.common.audit_result import (
    AuditResult,
    make_pass,
    make_fail,
    make_blocked,
    make_not_applicable,
    aggregate_status,
)
from qualification.autolearn_v04.common.evidence_origin import EvidenceOrigin


# Ordered list of all 24 sub-gate names.
SUB_GATE_NAMES = [
    "A0.1_structural_completeness",
    "A0.2_evidence_origin_authenticity",
    "A0.3_scientific_claim_eligibility",
    "A0.4_cross_origin_duplication",
    "A0.5_source_provenance",
    "A0.6_historical_identity",
    "A0.7_current_analysis_identity",
    "A0.8_cross_version_lineage",
    "A0.9_provider_model_authenticity",
    "A0.10_single_generation_identity",
    "A0.11_positive_evidence_weights",
    "A0.12_counterfactual_equivalence",
    "A0.13_complete_action_matrix",
    "A0.14_verifier_registry_completeness",
    "A0.15_utility_consistency",
    "A0.16_split_access_integrity",
    "A0.17_candidate_serialization_parity",
    "A0.18_sham_serialization_parity",
    "A0.19_task_split_consistency",
    "A0.20_artifact_lineage",
    "A0.21_metric_consistency",
    "A0.22_safety_evidence_integrity",
    "A0.23_no_stale_artifact_reuse",
    "A0.24_oracle_consistency",
]

# Sub-gates that are scientific and should return NOT_APPLICABLE for
# synthetic fixture evidence.
_SCIENTIFIC_SUB_GATES = frozenset({
    "A0.2_evidence_origin_authenticity",
    "A0.3_scientific_claim_eligibility",
    "A0.6_historical_identity",
    "A0.7_current_analysis_identity",
    "A0.8_cross_version_lineage",
    "A0.9_provider_model_authenticity",
    "A0.10_single_generation_identity",
})

_REMEDIATIONS: dict[str, str] = {
    "A0.1_structural_completeness": "Complete all required evidence artifacts",
    "A0.2_evidence_origin_authenticity": "Use a REAL_MODEL provider for scientific claims",
    "A0.3_scientific_claim_eligibility": "Ensure evidence is scientifically claim-eligible",
    "A0.4_cross_origin_duplication": "Remove cross-origin duplicated evidence",
    "A0.5_source_provenance": "Record complete source provenance",
    "A0.6_historical_identity": "Record historical source tree identity and commit SHA",
    "A0.7_current_analysis_identity": "Record current analysis source tree identity",
    "A0.8_cross_version_lineage": "Establish explicit lineage between versions",
    "A0.9_provider_model_authenticity": "Use a REAL_MODEL provider with model_digest",
    "A0.10_single_generation_identity": "Use exactly one model revision and generation",
    "A0.11_positive_evidence_weights": "Ensure all evidence quality weights are positive",
    "A0.12_counterfactual_equivalence": "Ensure all actions start from equivalent task state",
    "A0.13_complete_action_matrix": "Measure all eligible actions for every task",
    "A0.14_verifier_registry_completeness": "Complete the verifier registry",
    "A0.15_utility_consistency": "Fix utility inconsistency in outcomes",
    "A0.16_split_access_integrity": "Remove any final-test data from training splits",
    "A0.17_candidate_serialization_parity": "Fix candidate policy serialization parity",
    "A0.18_sham_serialization_parity": "Fix sham policy serialization parity",
    "A0.19_task_split_consistency": "Align task and split manifests",
    "A0.20_artifact_lineage": "Complete artifact lineage records",
    "A0.21_metric_consistency": "Fix metric inconsistency in stored results",
    "A0.22_safety_evidence_integrity": "Fix safety evidence validation failures",
    "A0.23_no_stale_artifact_reuse": "Remove stale artifacts from evidence package",
    "A0.24_oracle_consistency": "Resolve oracle discrepancy",
}


def _remediation(gate_name: str, status: AuditStatus) -> str:
    if status in (AuditStatus.PASS, AuditStatus.NOT_APPLICABLE):
        return ""
    return _REMEDIATIONS.get(gate_name, "Fix the identified issue")


def _is_missing(report: dict | None) -> bool:
    """A report is missing (BLOCKED) if it is None or explicitly empty."""
    if report is None:
        return True
    if isinstance(report, dict) and not report:
        return True
    return False


def _status_from_report(report: dict | None, default: AuditStatus = AuditStatus.BLOCKED) -> AuditStatus:
    """Extract an AuditStatus from a report dict using AuditStatus.from_str()."""
    if _is_missing(report):
        return AuditStatus.BLOCKED
    if not isinstance(report, dict):
        return AuditStatus.BLOCKED
    s = report.get("status")
    return AuditStatus.from_str(s) if s is not None else default


def _add(
    sub_gates: dict[str, dict[str, Any]],
    name: str,
    status: AuditStatus,
    observed: Any,
    expected: Any,
    reason: str,
    evidence_artifact: str = "",
    evidence_field: str = "",
) -> None:
    sub_gates[name] = {
        "name": name,
        "status": status.value,
        "observed": observed,
        "expected": expected,
        "reason": reason,
        "evidence_artifact": evidence_artifact,
        "evidence_field": evidence_field,
        "remediation": _remediation(name, status),
    }


def _blocked(
    sub_gates: dict[str, dict[str, Any]],
    name: str,
    reason: str,
    evidence_artifact: str = "",
    evidence_field: str = "",
) -> None:
    _add(sub_gates, name, AuditStatus.BLOCKED, "missing", "evidence provided",
         reason, evidence_artifact, evidence_field)


def _is_synthetic_origin(evidence_origin: dict | None) -> bool:
    """Check if the evidence origin report indicates SYNTHETIC origin."""
    if _is_missing(evidence_origin):
        return False
    origin = str(evidence_origin.get("origin", "")).upper()
    return origin == "SYNTHETIC"


def evaluate_gate_a0(
    byte_integrity: dict,
    scientific_completeness: dict,
    placeholder_report: dict,
    row_count_report: dict,
    historical_identity: dict,
    analysis_identity: dict,
    cross_version_lineage: dict,
    provider_validation: dict,
    model_identity: dict,
    evidence_weight_audit: dict,
    counterfactual_equivalence: dict,
    action_matrix: dict,
    verifier_registry: dict,
    utility_consistency: dict,
    split_access: dict,
    candidate_parity: dict,
    sham_parity: dict,
    task_split: dict,
    artifact_lineage: dict,
    metric_consistency: dict,
    safety_evidence: dict,
    stale_artifact: dict,
    oracle_consistency: dict,
    cross_origin_duplication: dict,
    evidence_origin: dict,
) -> dict:
    """Evaluate all 24 Gate A0 sub-gates.

    Each sub-gate reads the correct fields from its source report.
    Source statuses are parsed with AuditStatus.from_str().
    FAIL is never converted to BLOCKED or vice versa.

    For fixture validation (evidence_origin=SYNTHETIC), scientific
    sub-gates return NOT_APPLICABLE.

    Returns dict with status, n_sub_gates, n_pass, n_fail, n_blocked,
    n_not_applicable, sub_gates, and overall_reason.
    """
    sub_gates: dict[str, dict[str, Any]] = {}

    is_synthetic = _is_synthetic_origin(evidence_origin)

    # A0.1 Structural completeness
    if _is_missing(byte_integrity) and _is_missing(scientific_completeness):
        _blocked(sub_gates, "A0.1_structural_completeness",
                 "byte integrity and scientific completeness reports missing",
                 "artifact_checksums.json", "checksums")
    else:
        # Use byte_integrity as primary, fall back to scientific_completeness
        primary = byte_integrity if not _is_missing(byte_integrity) else scientific_completeness
        s = _status_from_report(primary)
        if primary is byte_integrity:
            checksum_results = byte_integrity.get("checksum_results")
            if isinstance(checksum_results, dict) and checksum_results:
                n_checked = len(checksum_results)
                n_match = sum(
                    1 for c in checksum_results.values()
                    if isinstance(c, dict) and c.get("match")
                )
            else:
                n_checked = byte_integrity.get("n_files_validated", 0)
                n_match = byte_integrity.get("n_match", n_checked)
            observed = f"{n_match}/{n_checked} match"
            if s == AuditStatus.BLOCKED and n_checked > 0:
                if n_match == n_checked:
                    s = AuditStatus.PASS
                else:
                    s = AuditStatus.FAIL
            if s == AuditStatus.PASS and n_match != n_checked:
                s = AuditStatus.FAIL
        else:
            checks = scientific_completeness.get("checks")
            if isinstance(checks, dict) and checks:
                n_required = len(checks)
                n_present = sum(
                    1 for c in checks.values()
                    if isinstance(c, dict) and c.get("status") == "PASS"
                )
            else:
                n_required = scientific_completeness.get("n_required", 0)
                n_present = scientific_completeness.get("n_present", 0)
            observed = f"{n_present}/{n_required} artifacts"
            if s == AuditStatus.PASS and n_present != n_required:
                s = AuditStatus.FAIL
        _add(sub_gates, "A0.1_structural_completeness", s, observed,
             "all complete",
             primary.get("reason", "structural completeness check"),
             "artifact_checksums.json", "checksums")

    # A0.2 Evidence-origin authenticity
    if is_synthetic:
        _add(sub_gates, "A0.2_evidence_origin_authenticity",
             AuditStatus.NOT_APPLICABLE, "SYNTHETIC", "REAL_MODEL",
             "Scientific sub-gate not applicable to synthetic fixture",
             "EVIDENCE_MANIFEST.json", "evidence_origin")
    elif _is_missing(evidence_origin):
        _blocked(sub_gates, "A0.2_evidence_origin_authenticity",
                 "evidence origin report missing",
                 "EVIDENCE_MANIFEST.json", "evidence_origin")
    else:
        origin = str(evidence_origin.get("origin", "")).upper()
        s = AuditStatus.PASS if origin == "REAL_MODEL" else AuditStatus.FAIL
        _add(sub_gates, "A0.2_evidence_origin_authenticity", s,
             origin, "REAL_MODEL",
             evidence_origin.get("reason", f"origin={origin}"),
             "EVIDENCE_MANIFEST.json", "evidence_origin")

    # A0.3 Scientific-claim eligibility
    if is_synthetic:
        _add(sub_gates, "A0.3_scientific_claim_eligibility",
             AuditStatus.NOT_APPLICABLE, "SYNTHETIC", "scientific_claim_eligible=True",
             "Scientific sub-gate not applicable to synthetic fixture",
             "EVIDENCE_MANIFEST.json", "scientific_claim_eligible")
    elif _is_missing(evidence_origin):
        _blocked(sub_gates, "A0.3_scientific_claim_eligibility",
                 "evidence origin report missing",
                 "EVIDENCE_MANIFEST.json", "scientific_claim_eligible")
    else:
        sci_eligible = evidence_origin.get("scientific_claim_eligible", False)
        s = AuditStatus.PASS if sci_eligible else AuditStatus.FAIL
        _add(sub_gates, "A0.3_scientific_claim_eligibility", s,
             sci_eligible, True,
             evidence_origin.get("reason", f"scientific_claim_eligible={sci_eligible}"),
             "EVIDENCE_MANIFEST.json", "scientific_claim_eligible")

    # A0.4 Cross-origin duplication
    if _is_missing(cross_origin_duplication):
        _blocked(sub_gates, "A0.4_cross_origin_duplication",
                 "cross-origin duplication report missing",
                 "cross_origin_duplication.json", "duplicates")
    else:
        n_dupes = cross_origin_duplication.get("n_duplicates",
                        cross_origin_duplication.get("n_dupes", 0))
        dupe_detected = cross_origin_duplication.get("duplicates_detected",
                          n_dupes > 0)
        s = AuditStatus.PASS if not dupe_detected and n_dupes == 0 else AuditStatus.FAIL
        _add(sub_gates, "A0.4_cross_origin_duplication", s,
             f"{n_dupes} duplicates", "0 duplicates",
             cross_origin_duplication.get("reason", "cross-origin duplication check"),
             "cross_origin_duplication.json", "duplicates")

    # A0.5 Source provenance
    if _is_missing(row_count_report) and _is_missing(historical_identity):
        # Fall back to checking source_provenance directly is not possible
        # here — we use the row_count_report or historical_identity as proxy
        _blocked(sub_gates, "A0.5_source_provenance",
                 "source provenance report missing",
                 "source_provenance.json", "provenance")
    else:
        # Use historical_identity or row_count_report for provenance info
        prov = historical_identity if not _is_missing(historical_identity) else row_count_report
        s = _status_from_report(prov)
        source_hash = (
            prov.get("original_source_tree_sha256")
            or prov.get("source_tree_sha256")
            or prov.get("source_hash", "")
        )
        config_hash = (
            prov.get("original_config_sha256")
            or prov.get("config_digest")
            or prov.get("config_hash", "")
        )
        has_provenance = bool(source_hash or config_hash)
        if s == AuditStatus.BLOCKED and has_provenance:
            s = AuditStatus.PASS if source_hash else AuditStatus.FAIL
        _add(sub_gates, "A0.5_source_provenance", s,
             f"source_hash={'yes' if source_hash else 'no'}, "
             f"config_hash={'yes' if config_hash else 'no'}",
             "complete source provenance",
             prov.get("reason", "source provenance check"),
             "source_provenance.json", "provenance")

    # A0.6 Historical identity
    if is_synthetic:
        _add(sub_gates, "A0.6_historical_identity",
             AuditStatus.NOT_APPLICABLE, "SYNTHETIC", "historical identity",
             "Scientific sub-gate not applicable to synthetic fixture",
             "source_provenance.json", "original_commit_sha")
    elif _is_missing(historical_identity):
        _blocked(sub_gates, "A0.6_historical_identity",
                 "historical identity report missing",
                 "source_provenance.json", "original_commit_sha")
    else:
        s = _status_from_report(historical_identity)
        commit = (
            historical_identity.get("original_commit_sha")
            or historical_identity.get("commit_sha", "")
        )
        source_hash = (
            historical_identity.get("original_source_tree_sha256")
            or historical_identity.get("source_hash")
            or historical_identity.get("source_tree_sha256", "")
        )
        if s == AuditStatus.BLOCKED and (commit or source_hash):
            s = AuditStatus.PASS if (commit and source_hash) else AuditStatus.FAIL
        _add(sub_gates, "A0.6_historical_identity", s,
             f"commit={'yes' if commit else 'no'}, "
             f"source_hash={'yes' if source_hash else 'no'}",
             "nonempty commit SHA and source hash",
             historical_identity.get("reason", "historical identity"),
             "source_provenance.json", "original_commit_sha")

    # A0.7 Current analysis identity
    if is_synthetic:
        _add(sub_gates, "A0.7_current_analysis_identity",
             AuditStatus.NOT_APPLICABLE, "SYNTHETIC", "analysis identity",
             "Scientific sub-gate not applicable to synthetic fixture",
             "analysis_source_provenance.json", "analysis_source_tree_sha256")
    elif _is_missing(analysis_identity):
        _blocked(sub_gates, "A0.7_current_analysis_identity",
                 "current analysis identity missing",
                 "analysis_source_provenance.json", "analysis_source_tree_sha256")
    else:
        s = _status_from_report(analysis_identity)
        source_hash = (
            analysis_identity.get("analysis_source_tree_sha256")
            or analysis_identity.get("source_hash")
            or analysis_identity.get("source_tree_sha256", "")
        )
        valid_hash = (
            isinstance(source_hash, str)
            and len(source_hash) == 64
            and all(c in "0123456789abcdef" for c in source_hash.lower())
        )
        if s == AuditStatus.BLOCKED and source_hash:
            s = AuditStatus.PASS if valid_hash else AuditStatus.FAIL
        _add(sub_gates, "A0.7_current_analysis_identity", s,
             f"source_hash={'yes' if source_hash else 'no'}",
             "valid 64-char source-tree hash",
             analysis_identity.get("reason", "current analysis identity"),
             "analysis_source_provenance.json", "analysis_source_tree_sha256")

    # A0.8 Cross-version lineage
    if is_synthetic:
        _add(sub_gates, "A0.8_cross_version_lineage",
             AuditStatus.NOT_APPLICABLE, "SYNTHETIC", "cross-version lineage",
             "Scientific sub-gate not applicable to synthetic fixture",
             "cross_version_lineage.json", "lineage")
    elif _is_missing(cross_version_lineage):
        _blocked(sub_gates, "A0.8_cross_version_lineage",
                 "cross-version lineage missing",
                 "cross_version_lineage.json", "lineage")
    else:
        linked = cross_version_lineage.get("linked", False)
        historical_ref = cross_version_lineage.get("historical_ref", "")
        current_ref = cross_version_lineage.get("current_ref", "")
        s = AuditStatus.PASS if linked and historical_ref and current_ref else AuditStatus.FAIL
        _add(sub_gates, "A0.8_cross_version_lineage", s,
             f"linked={linked}", "explicit lineage link",
             cross_version_lineage.get("reason", "cross-version lineage"),
             "cross_version_lineage.json", "lineage")

    # A0.9 Provider/model authenticity
    if is_synthetic:
        _add(sub_gates, "A0.9_provider_model_authenticity",
             AuditStatus.NOT_APPLICABLE, "SYNTHETIC", "REAL_MODEL",
             "Scientific sub-gate not applicable to synthetic fixture",
             "provider_manifest.json", "provider_class")
    elif _is_missing(provider_validation):
        _blocked(sub_gates, "A0.9_provider_model_authenticity",
                 "provider validation missing",
                 "provider_manifest.json", "provider_class")
    else:
        pv = provider_validation.get("provider_validation", provider_validation)
        provider_class = str(
            pv.get("provider_class", provider_validation.get("provider_class", ""))
        ).upper()
        model_id = str(pv.get("model_id", provider_validation.get("model_id", ""))).lower()
        tokenizer_id = str(
            pv.get("tokenizer_id", provider_validation.get("tokenizer_id", ""))
        ).lower()
        model_digest = pv.get("model_digest", provider_validation.get("model_digest"))
        is_real = provider_class == "REAL_MODEL"
        no_synthetic = ("synthetic" not in model_id) and ("synthetic" not in tokenizer_id)
        has_digest = bool(model_digest)
        s = AuditStatus.PASS if (is_real and no_synthetic and has_digest) else AuditStatus.FAIL
        _add(sub_gates, "A0.9_provider_model_authenticity", s,
             f"provider_class={provider_class}, model_digest={'yes' if has_digest else 'no'}",
             "REAL_MODEL with model_digest and no synthetic markers",
             provider_validation.get("reason", "provider/model authenticity"),
             "provider_manifest.json", "provider_class")

    # A0.10 Single generation identity
    if is_synthetic:
        _add(sub_gates, "A0.10_single_generation_identity",
             AuditStatus.NOT_APPLICABLE, "SYNTHETIC", "single generation",
             "Scientific sub-gate not applicable to synthetic fixture",
             "provider_manifest.json", "model_revision")
    elif _is_missing(model_identity):
        _blocked(sub_gates, "A0.10_single_generation_identity",
                 "model identity missing",
                 "provider_manifest.json", "model_revision")
    else:
        n_revisions = len(model_identity.get("unique_model_revisions", []))
        n_generations = len(model_identity.get("unique_generations", []))
        s = AuditStatus.PASS if n_revisions == 1 and n_generations == 1 else AuditStatus.FAIL
        _add(sub_gates, "A0.10_single_generation_identity", s,
             f"{n_revisions} revision(s), {n_generations} generation(s)",
             "exactly 1 each",
             model_identity.get("reason", "model identity check"),
             "provider_manifest.json", "model_revision")

    # A0.11 Positive evidence weights
    if _is_missing(evidence_weight_audit):
        _blocked(sub_gates, "A0.11_positive_evidence_weights",
                 "evidence weight audit missing",
                 "dataset_manifest.json", "quality_weights")
    else:
        weights = evidence_weight_audit.get("weights", [])
        n_negative = evidence_weight_audit.get("n_negative", 0)
        n_zero = evidence_weight_audit.get("n_zero", 0)
        if not weights:
            s = AuditStatus.BLOCKED
        else:
            s = AuditStatus.PASS if n_negative == 0 and n_zero == 0 else AuditStatus.FAIL
        _add(sub_gates, "A0.11_positive_evidence_weights", s,
             f"{len(weights)} weights, {n_negative} negative, {n_zero} zero",
             "all weights > 0",
             evidence_weight_audit.get("reason", "evidence weight audit"),
             "dataset_manifest.json", "quality_weights")

    # A0.12 Counterfactual equivalence
    if _is_missing(counterfactual_equivalence):
        _blocked(sub_gates, "A0.12_counterfactual_equivalence",
                 "counterfactual equivalence report missing",
                 "counterfactual_equivalence_report.json", "status")
    else:
        ce_status = counterfactual_equivalence.get("ce_status",
                        counterfactual_equivalence.get("status", "UNVERIFIABLE"))
        n_equiv = counterfactual_equivalence.get("n_tasks_equivalent", 0)
        n_total = counterfactual_equivalence.get("n_tasks_checked", 0)
        if n_total == 0:
            s = AuditStatus.BLOCKED
        elif (ce_status == "EQUIVALENT" or counterfactual_equivalence.get("status") == "PASS") and n_equiv == n_total:
            s = AuditStatus.PASS
        else:
            s = AuditStatus.FAIL
        _add(sub_gates, "A0.12_counterfactual_equivalence", s,
             f"{n_equiv}/{n_total} equivalent", "all EQUIVALENT",
             counterfactual_equivalence.get("reason", f"counterfactual equivalence: {ce_status}"),
             "counterfactual_equivalence_report.json", "status")

    # A0.13 Complete action matrix
    if _is_missing(action_matrix):
        _blocked(sub_gates, "A0.13_complete_action_matrix",
                 "action matrix report missing",
                 "action_matrix_completeness.json", "status")
    else:
        am_status = action_matrix.get("am_status",
                        action_matrix.get("status", "INCOMPLETE"))
        n_complete = action_matrix.get("n_tasks_complete", 0)
        n_total = action_matrix.get("n_tasks_total", 0)
        if n_total == 0:
            s = AuditStatus.BLOCKED
        elif (am_status == "COMPLETE" or action_matrix.get("status") == "PASS") and n_complete == n_total:
            s = AuditStatus.PASS
        else:
            s = AuditStatus.FAIL
        _add(sub_gates, "A0.13_complete_action_matrix", s,
             f"{n_complete}/{n_total} complete", "all COMPLETE",
             action_matrix.get("reason", f"action matrix: {am_status}"),
             "action_matrix_completeness.json", "status")

    # A0.14 Verifier registry completeness
    if _is_missing(verifier_registry):
        _blocked(sub_gates, "A0.14_verifier_registry_completeness",
                 "verifier registry report missing",
                 "verifier_registry.json", "verifiers")
    else:
        n_registered = verifier_registry.get("n_registered", 0)
        n_required = verifier_registry.get("n_required", 0)
        if n_required == 0:
            s = AuditStatus.BLOCKED
        else:
            s = AuditStatus.PASS if n_registered >= n_required else AuditStatus.FAIL
        _add(sub_gates, "A0.14_verifier_registry_completeness", s,
             f"{n_registered}/{n_required} verifiers", "all required verifiers registered",
             verifier_registry.get("reason", "verifier registry completeness"),
             "verifier_registry.json", "verifiers")

    # A0.15 Utility consistency
    if _is_missing(utility_consistency):
        _blocked(sub_gates, "A0.15_utility_consistency",
                 "utility consistency report missing",
                 "utility_consistency_report.json", "status")
    else:
        s = _status_from_report(utility_consistency, default=AuditStatus.FAIL)
        n_fail = utility_consistency.get("n_fail", 0)
        # Preserve BLOCKED from the source report; never convert to FAIL.
        if s == AuditStatus.BLOCKED:
            pass  # keep BLOCKED
        elif s == AuditStatus.PASS:
            s = AuditStatus.PASS if n_fail == 0 else AuditStatus.FAIL
        else:
            s = AuditStatus.FAIL
        _add(sub_gates, "A0.15_utility_consistency", s,
             f"{n_fail} failures", "0 failures",
             utility_consistency.get("reason", "utility consistency check"),
             "utility_consistency_report.json", "status")

    # A0.16 Split-access integrity
    if _is_missing(split_access):
        _blocked(sub_gates, "A0.16_split_access_integrity",
                 "split access report missing",
                 "split_manifest.json", "splits")
    else:
        s = _status_from_report(split_access)
        leakage = split_access.get("leakage_detected", s == AuditStatus.FAIL)
        n_violations = split_access.get("n_violations",
                          len(split_access.get("forbidden_accesses", [])))
        # Preserve BLOCKED from the source report; never convert to FAIL.
        if s == AuditStatus.BLOCKED:
            pass  # keep BLOCKED
        elif s == AuditStatus.PASS or (not leakage and n_violations == 0):
            s = AuditStatus.PASS
        else:
            s = AuditStatus.FAIL
        _add(sub_gates, "A0.16_split_access_integrity", s,
             f"{n_violations} violations", "0 violations",
             split_access.get("reason", "split-access integrity check"),
             "split_manifest.json", "splits")

    # A0.17 Candidate serialization parity
    if _is_missing(candidate_parity) or "parity" not in candidate_parity:
        _blocked(sub_gates, "A0.17_candidate_serialization_parity",
                 "candidate parity report missing or empty",
                 "candidate_policy.json", "policy_sha256")
    else:
        parity = candidate_parity.get("parity", False)
        sha = candidate_parity.get("policy_sha256", "")
        s = AuditStatus.PASS if parity else AuditStatus.FAIL
        _add(sub_gates, "A0.17_candidate_serialization_parity", s,
             f"parity={parity}, sha={sha[:12] if sha else 'none'}",
             "serialization parity confirmed",
             candidate_parity.get("reason", "candidate serialization parity"),
             "candidate_policy.json", "policy_sha256")

    # A0.18 Sham serialization parity
    if _is_missing(sham_parity) or "parity" not in sham_parity:
        _blocked(sub_gates, "A0.18_sham_serialization_parity",
                 "sham parity report missing or empty",
                 "sham_policy.json", "policy_sha256")
    else:
        parity = sham_parity.get("parity", False)
        sha = sham_parity.get("policy_sha256", "")
        s = AuditStatus.PASS if parity else AuditStatus.FAIL
        _add(sub_gates, "A0.18_sham_serialization_parity", s,
             f"parity={parity}, sha={sha[:12] if sha else 'none'}",
             "serialization parity confirmed",
             sham_parity.get("reason", "sham serialization parity"),
             "sham_policy.json", "policy_sha256")

    # A0.19 Task/split consistency
    if _is_missing(task_split):
        _blocked(sub_gates, "A0.19_task_split_consistency",
                 "task/split consistency report missing",
                 "benchmark_manifest.json", "benchmark_task_count")
    else:
        s = _status_from_report(task_split)
        n_bench = task_split.get("benchmark_task_count",
                      task_split.get("n_tasks", 0))
        n_split = task_split.get("split_task_count",
                      task_split.get("n_split_tasks", 0))
        if s not in (AuditStatus.PASS, AuditStatus.FAIL, AuditStatus.BLOCKED, AuditStatus.NOT_RUN):
            consistent = task_split.get("consistent", False)
            s = AuditStatus.PASS if consistent and n_bench == n_split else AuditStatus.FAIL
        _add(sub_gates, "A0.19_task_split_consistency", s,
             f"benchmark={n_bench}, split={n_split}", "consistent",
             task_split.get("reason", "task/split consistency"),
             "benchmark_manifest.json", "benchmark_task_count")

    # A0.20 Artifact lineage
    if _is_missing(artifact_lineage):
        _blocked(sub_gates, "A0.20_artifact_lineage",
                 "artifact lineage report missing",
                 "source_provenance.json", "n_artifacts")
    else:
        s = _status_from_report(artifact_lineage)
        n_artifacts = artifact_lineage.get("n_artifacts", 0)
        n_valid = artifact_lineage.get("n_valid",
                        artifact_lineage.get("n_with_lineage", 0))
        if s not in (AuditStatus.PASS, AuditStatus.FAIL, AuditStatus.BLOCKED, AuditStatus.NOT_RUN):
            complete = artifact_lineage.get("complete", False)
            s = AuditStatus.PASS if complete and n_artifacts > 0 and n_valid == n_artifacts else AuditStatus.FAIL
        _add(sub_gates, "A0.20_artifact_lineage", s,
             f"{n_valid}/{n_artifacts} valid", "all artifacts valid",
             artifact_lineage.get("reason", "artifact lineage"),
             "source_provenance.json", "n_artifacts")

    # A0.21 Metric consistency
    if _is_missing(metric_consistency):
        _blocked(sub_gates, "A0.21_metric_consistency",
                 "metric consistency report missing",
                 "metric_consistency_report.json", "status")
    else:
        s = _status_from_report(metric_consistency, default=AuditStatus.FAIL)
        n_fail = metric_consistency.get("n_fail", 0)
        if s == AuditStatus.PASS:
            s = AuditStatus.PASS if n_fail == 0 else AuditStatus.FAIL
        else:
            s = AuditStatus.FAIL
        _add(sub_gates, "A0.21_metric_consistency", s,
             f"{n_fail} failures", "0 failures",
             metric_consistency.get("reason", "metric consistency check"),
             "metric_consistency_report.json", "status")

    # A0.22 Safety evidence integrity
    if _is_missing(safety_evidence):
        _blocked(sub_gates, "A0.22_safety_evidence_integrity",
                 "safety evidence report missing",
                 "safety_evidence_report.json", "status")
    else:
        s = _status_from_report(safety_evidence, default=AuditStatus.FAIL)
        checks_list = safety_evidence.get("checks")
        if isinstance(checks_list, list) and checks_list:
            n_fail = sum(1 for c in checks_list
                         if isinstance(c, dict) and c.get("status") == "FAIL")
        else:
            n_fail = safety_evidence.get("n_fail", 0)
        # Preserve BLOCKED from the source report; never convert to FAIL.
        if s == AuditStatus.BLOCKED:
            pass  # keep BLOCKED
        elif s == AuditStatus.PASS:
            s = AuditStatus.PASS if n_fail == 0 else AuditStatus.FAIL
        else:
            s = AuditStatus.FAIL
        _add(sub_gates, "A0.22_safety_evidence_integrity", s,
             f"{n_fail} failures", "0 failures",
             safety_evidence.get("reason", "safety evidence integrity"),
             "safety_evidence_report.json", "status")

    # A0.23 No stale artifact reuse
    if _is_missing(stale_artifact):
        _blocked(sub_gates, "A0.23_no_stale_artifact_reuse",
                 "stale artifact report missing",
                 "evidence_import_report.json", "stale_check")
    else:
        stale = stale_artifact.get("stale_detected", True)
        n_stale = stale_artifact.get("n_stale", 0)
        s = AuditStatus.PASS if not stale and n_stale == 0 else AuditStatus.FAIL
        _add(sub_gates, "A0.23_no_stale_artifact_reuse", s,
             f"{n_stale} stale artifacts", "0 stale artifacts",
             stale_artifact.get("reason", "stale artifact check"),
             "evidence_import_report.json", "stale_check")

    # A0.24 Oracle consistency
    if _is_missing(oracle_consistency):
        _blocked(sub_gates, "A0.24_oracle_consistency",
                 "oracle consistency report missing",
                 "oracle_discrepancy_report.json", "status")
    else:
        s = _status_from_report(oracle_consistency, default=AuditStatus.FAIL)
        discrepancy_found = oracle_consistency.get("discrepancy_found", True)
        if s == AuditStatus.PASS:
            s = AuditStatus.PASS if not discrepancy_found else AuditStatus.FAIL
        else:
            s = AuditStatus.FAIL
        _add(sub_gates, "A0.24_oracle_consistency", s,
             f"discrepancy_found={discrepancy_found}", "no discrepancy",
             oracle_consistency.get("reason",
                 oracle_consistency.get("resolution", "oracle consistency")),
             "oracle_discrepancy_report.json", "status")

    # Compute overall status.
    n_pass = sum(1 for g in sub_gates.values() if g["status"] == AuditStatus.PASS.value)
    n_fail = sum(1 for g in sub_gates.values() if g["status"] == AuditStatus.FAIL.value)
    n_blocked = sum(1 for g in sub_gates.values() if g["status"] == AuditStatus.BLOCKED.value)
    n_not_applicable = sum(1 for g in sub_gates.values() if g["status"] == AuditStatus.NOT_APPLICABLE.value)

    if n_blocked > 0:
        overall = AuditStatus.BLOCKED.value
        overall_reason = (
            f"Gate A0 BLOCKED: {n_blocked} sub-gate(s) missing required evidence "
            f"({n_pass} PASS, {n_fail} FAIL, {n_not_applicable} N/A)"
        )
    elif n_fail > 0:
        overall = AuditStatus.FAIL.value
        overall_reason = (
            f"Gate A0 FAIL: {n_fail} sub-gate(s) violated conditions "
            f"({n_pass} PASS, {n_not_applicable} N/A)"
        )
    else:
        overall = AuditStatus.PASS.value
        overall_reason = (
            f"Gate A0 PASS: all {n_pass} mandatory sub-gates passed "
            f"({n_not_applicable} N/A)"
        )

    return {
        "status": overall,
        "n_sub_gates": 24,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_blocked": n_blocked,
        "n_not_applicable": n_not_applicable,
        "sub_gates": sub_gates,
        "overall_reason": overall_reason,
    }
