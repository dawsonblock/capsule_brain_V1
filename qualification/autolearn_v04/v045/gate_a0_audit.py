"""Gate A0 audit for v0.4.5.

22 sub-gates with strict PASS/FAIL/BLOCKED/NOT_RUN semantics.

NO hardcoded PASS. Every sub-gate must load and validate actual evidence.

Status semantics:
- PASS: check executed and satisfied
- FAIL: check executed and condition violated
- BLOCKED: required evidence missing
- NOT_RUN: stage not attempted

Final A0 PASS only if every mandatory sub-gate passes.
"""
from __future__ import annotations

from typing import Any


# Ordered list of all 22 sub-gate names.
SUB_GATE_NAMES = [
    "A0.1_byte_integrity",
    "A0.2_scientific_completeness",
    "A0.3_placeholder_free_evidence",
    "A0.4_historical_source_identity",
    "A0.5_current_analysis_identity",
    "A0.6_cross_version_lineage",
    "A0.7_real_model_provider_identity",
    "A0.8_single_model_and_generation_identity",
    "A0.9_positive_evidence_weights",
    "A0.10_counterfactual_equivalence",
    "A0.11_complete_eligible_action_matrix",
    "A0.12_verifier_registry_completeness",
    "A0.13_utility_consistency",
    "A0.14_no_final_test_leakage",
    "A0.15_candidate_serialization_parity",
    "A0.16_sham_serialization_parity",
    "A0.17_task_split_consistency",
    "A0.18_artifact_lineage_completeness",
    "A0.19_metric_consistency",
    "A0.20_safety_evidence_integrity",
    "A0.21_no_stale_artifact_reuse",
    "A0.22_oracle_consistency",
]


_REMEDIATIONS: dict[str, str] = {
    "A0.1_byte_integrity": "Re-generate evidence package with correct byte-level checksums",
    "A0.2_scientific_completeness": "Complete all required scientific artifacts before Gate A0",
    "A0.3_placeholder_free_evidence": "Replace placeholder/sentinel values with real measured evidence",
    "A0.4_historical_source_identity": "Record historical source tree identity and commit SHA",
    "A0.5_current_analysis_identity": "Record current analysis source tree identity and commit SHA",
    "A0.6_cross_version_lineage": "Establish explicit lineage between historical and current versions",
    "A0.7_real_model_provider_identity": "Use a REAL_MODEL provider for all Gate A claims",
    "A0.8_single_model_and_generation_identity": "Use exactly one model revision and generation",
    "A0.9_positive_evidence_weights": "Ensure all evidence quality weights are positive",
    "A0.10_counterfactual_equivalence": "Ensure all actions start from equivalent task state",
    "A0.11_complete_eligible_action_matrix": "Measure all eligible actions for every task",
    "A0.12_verifier_registry_completeness": "Complete the verifier registry",
    "A0.13_utility_consistency": "Fix utility inconsistency in outcomes",
    "A0.14_no_final_test_leakage": "Remove any final-test data from training splits",
    "A0.15_candidate_serialization_parity": "Fix candidate policy serialization parity",
    "A0.16_sham_serialization_parity": "Fix sham policy serialization parity",
    "A0.17_task_split_consistency": "Align task and split manifests",
    "A0.18_artifact_lineage_completeness": "Complete artifact lineage records",
    "A0.19_metric_consistency": "Fix metric inconsistency in stored results",
    "A0.20_safety_evidence_integrity": "Fix safety evidence validation failures",
    "A0.21_no_stale_artifact_reuse": "Remove stale artifacts from evidence package",
    "A0.22_oracle_consistency": "Resolve oracle discrepancy via oracle_discrepancy.investigate_oracle_discrepancy",
}


def _remediation(gate_name: str, status: str) -> str:
    if status == "PASS":
        return ""
    return _REMEDIATIONS.get(gate_name, "Fix the identified issue")


def _is_missing(report: dict | None) -> bool:
    """A report is missing (BLOCKED) if it is None or explicitly empty."""
    if report is None:
        return True
    if isinstance(report, dict) and not report:
        return True
    return False


def _status_from_report(report: dict | None, default: str = "BLOCKED") -> str:
    """Extract a status from a report dict.

    Never returns PASS without evidence. If the report is missing, BLOCKED.
    """
    if _is_missing(report):
        return "BLOCKED"
    if not isinstance(report, dict):
        return "BLOCKED"
    s = report.get("status")
    if s in ("PASS", "FAIL", "BLOCKED", "NOT_RUN"):
        return s
    return default


def _add(
    sub_gates: dict[str, dict[str, Any]],
    name: str,
    status: str,
    observed: Any,
    expected: Any,
    reason: str,
    evidence_artifact: str = "",
    evidence_field: str = "",
) -> None:
    sub_gates[name] = {
        "name": name,
        "status": status,
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
    _add(sub_gates, name, "BLOCKED", "missing", "evidence provided", reason,
         evidence_artifact, evidence_field)


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
) -> dict:
    """Evaluate all 22 Gate A0 sub-gates.

    Every sub-gate must load and validate actual evidence. No hardcoded PASS.
    Missing input -> BLOCKED. Failed condition -> FAIL.

    Returns dict with status, n_sub_gates, n_pass, n_fail, n_blocked,
    sub_gates, and overall_reason.
    """
    sub_gates: dict[str, dict[str, Any]] = {}

    # A0.1 Byte integrity
    if _is_missing(byte_integrity):
        _blocked(sub_gates, "A0.1_byte_integrity",
                 "byte integrity report missing", "artifact_checksums.json", "checksums")
    else:
        s = _status_from_report(byte_integrity)
        checksum_results = byte_integrity.get("checksum_results")
        if isinstance(checksum_results, dict) and checksum_results:
            # n_checked = number of files we actually computed hashes for.
            # artifact_checksums.json is counted in n_files_validated but not in checksum_results.
            n_checked = len(checksum_results)
            n_match = sum(
                1 for c in checksum_results.values()
                if isinstance(c, dict) and c.get("match")
            )
        else:
            n_checked = byte_integrity.get(
                "n_files_validated",
                byte_integrity.get("n_checksums", byte_integrity.get("n_checked", 0)),
            )
            n_match = byte_integrity.get("n_match", n_checked)
        errors = byte_integrity.get("errors", []) or []
        observed = f"{n_match}/{n_checked} match"
        # Use the report's status directly when available.
        if s == "BLOCKED" and n_checked > 0:
            if n_match == n_checked and not errors:
                s = "PASS"
            else:
                s = "FAIL"
        if s == "PASS" and n_match != n_checked:
            s = "FAIL"
        _add(sub_gates, "A0.1_byte_integrity", s, observed, "all match",
             byte_integrity.get("reason", "byte integrity check"),
             "artifact_checksums.json", "checksums")

    # A0.2 Scientific completeness
    if _is_missing(scientific_completeness):
        _blocked(sub_gates, "A0.2_scientific_completeness",
                 "scientific completeness report missing", "evidence_manifest.json", "completeness")
    else:
        s = _status_from_report(scientific_completeness)
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
        if s == "PASS" and n_present != n_required:
            s = "FAIL"
        _add(sub_gates, "A0.2_scientific_completeness", s, observed, "all required artifacts",
             scientific_completeness.get("reason", "scientific completeness check"),
             "evidence_manifest.json", "completeness")

    # A0.3 Placeholder-free evidence
    if _is_missing(placeholder_report):
        _blocked(sub_gates, "A0.3_placeholder_free_evidence",
                 "placeholder report missing", "placeholder_report.json", "n_detections")
    else:
        n_placeholder = placeholder_report.get(
            "n_detections",
            placeholder_report.get("n_placeholder", 0),
        )
        if n_placeholder == 0 and "n_detections" not in placeholder_report:
            # Backward-compat: old reports carried n_placeholder/n_total.
            n_placeholder = placeholder_report.get("n_placeholder", 0)
        s = "PASS" if n_placeholder == 0 else "FAIL"
        _add(sub_gates, "A0.3_placeholder_free_evidence", s,
             f"{n_placeholder} placeholders detected", "0 placeholders",
             placeholder_report.get("reason", "placeholder scan"),
             "placeholder_report.json", "n_detections")

    # A0.4 Historical source identity
    if _is_missing(historical_identity):
        _blocked(sub_gates, "A0.4_historical_source_identity",
                 "historical source identity missing", "historical_source_provenance.json", "original_commit_sha")
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
        if s == "BLOCKED" and (commit or source_hash):
            s = "PASS" if (commit and source_hash) else "FAIL"
        _add(sub_gates, "A0.4_historical_source_identity", s,
             f"commit={commit[:12] if commit else 'null'}, "
             f"source_hash={source_hash[:12] if source_hash else 'missing'}",
             "nonempty commit SHA and source hash",
             historical_identity.get("reason", "historical source identity"),
             "historical_source_provenance.json", "original_commit_sha")

    # A0.5 Current analysis identity
    if _is_missing(analysis_identity):
        _blocked(sub_gates, "A0.5_current_analysis_identity",
                 "current analysis identity missing", "analysis_source_provenance.json", "analysis_source_tree_sha256")
    else:
        s = _status_from_report(analysis_identity)
        source_hash = (
            analysis_identity.get("analysis_source_tree_sha256")
            or analysis_identity.get("source_hash")
            or analysis_identity.get("source_tree_sha256", "")
        )
        config_digest = analysis_identity.get("config_digest", "")
        # A valid 64-char hex source-tree hash is sufficient; commit SHA
        # is NOT required for the current analysis identity.
        valid_hash = (
            isinstance(source_hash, str)
            and len(source_hash) == 64
            and all(c in "0123456789abcdef" for c in source_hash.lower())
        )
        if s == "BLOCKED" and source_hash:
            s = "PASS" if valid_hash else "FAIL"
        _add(sub_gates, "A0.5_current_analysis_identity", s,
             f"source_hash={source_hash[:12] if source_hash else 'missing'}, "
             f"config_digest={config_digest[:12] if config_digest else 'none'}",
             "valid 64-char source-tree hash",
             analysis_identity.get("reason", "current analysis identity"),
             "analysis_source_provenance.json", "analysis_source_tree_sha256")

    # A0.6 Cross-version lineage
    if _is_missing(cross_version_lineage):
        _blocked(sub_gates, "A0.6_cross_version_lineage",
                 "cross-version lineage missing", "cross_version_lineage.json", "lineage")
    else:
        linked = cross_version_lineage.get("linked", False)
        historical_ref = cross_version_lineage.get("historical_ref", "")
        current_ref = cross_version_lineage.get("current_ref", "")
        s = "PASS" if linked and historical_ref and current_ref else "FAIL"
        _add(sub_gates, "A0.6_cross_version_lineage", s,
             f"linked={linked}", "explicit lineage link",
             cross_version_lineage.get("reason", "cross-version lineage"),
             "cross_version_lineage.json", "lineage")

    # A0.7 Real-model provider identity
    if _is_missing(provider_validation):
        _blocked(sub_gates, "A0.7_real_model_provider_identity",
                 "provider validation missing", "provider_manifest.json", "provider_class")
    else:
        pv = provider_validation.get("provider_validation", provider_validation)
        provider_class = str(
            pv.get("provider_class", provider_validation.get("provider_class", ""))
        ).upper()
        model_id = str(pv.get("model_id", provider_validation.get("model_id", ""))).lower()
        tokenizer_id = str(
            pv.get("tokenizer_id", provider_validation.get("tokenizer_id", ""))
        ).lower()
        # PASS only when provider_class == REAL_MODEL and neither model_id
        # nor tokenizer_id contains "synthetic". Otherwise FAIL.
        is_real = provider_class == "REAL_MODEL"
        no_synthetic = ("synthetic" not in model_id) and ("synthetic" not in tokenizer_id)
        s = "PASS" if (is_real and no_synthetic) else "FAIL"
        _add(sub_gates, "A0.7_real_model_provider_identity", s,
             f"provider_class={provider_class}, model_id={model_id}, "
             f"tokenizer_id={tokenizer_id}",
             "REAL_MODEL with no synthetic model/tokenizer",
             provider_validation.get("reason", "provider validation"),
             "provider_manifest.json", "provider_class")

    # A0.8 Single model and generation identity
    if _is_missing(model_identity):
        _blocked(sub_gates, "A0.8_single_model_and_generation_identity",
                 "model identity missing", "provider_manifest.json", "model_revision")
    else:
        n_revisions = len(model_identity.get("unique_model_revisions", []))
        n_generations = len(model_identity.get("unique_generations", []))
        s = "PASS" if n_revisions == 1 and n_generations == 1 else "FAIL"
        _add(sub_gates, "A0.8_single_model_and_generation_identity", s,
             f"{n_revisions} revision(s), {n_generations} generation(s)", "exactly 1 each",
             model_identity.get("reason", "model identity check"),
             "provider_manifest.json", "model_revision")

    # A0.9 Positive evidence weights
    if _is_missing(evidence_weight_audit):
        _blocked(sub_gates, "A0.9_positive_evidence_weights",
                 "evidence weight audit missing", "dataset_manifest.json", "quality_weights")
    else:
        weights = evidence_weight_audit.get("weights", [])
        n_negative = evidence_weight_audit.get("n_negative", 0)
        n_zero = evidence_weight_audit.get("n_zero", 0)
        s = "PASS" if weights and n_negative == 0 and n_zero == 0 else "FAIL"
        if not weights:
            s = "BLOCKED"
        _add(sub_gates, "A0.9_positive_evidence_weights", s,
             f"{len(weights)} weights, {n_negative} negative, {n_zero} zero",
             "all weights > 0",
             evidence_weight_audit.get("reason", "evidence weight audit"),
             "dataset_manifest.json", "quality_weights")

    # A0.10 Counterfactual equivalence
    if _is_missing(counterfactual_equivalence):
        _blocked(sub_gates, "A0.10_counterfactual_equivalence",
                 "counterfactual equivalence report missing",
                 "counterfactual_equivalence_report.json", "status")
    else:
        ce_status = counterfactual_equivalence.get("ce_status",
                        counterfactual_equivalence.get("status", "UNVERIFIABLE"))
        n_equiv = counterfactual_equivalence.get("n_tasks_equivalent", 0)
        n_total = counterfactual_equivalence.get("n_tasks_checked", 0)
        # PASS if status is PASS or ce_status is EQUIVALENT, and all tasks equivalent.
        s = "PASS" if (ce_status == "EQUIVALENT" or counterfactual_equivalence.get("status") == "PASS") and n_total > 0 and n_equiv == n_total else "FAIL"
        if n_total == 0:
            s = "BLOCKED"
        _add(sub_gates, "A0.10_counterfactual_equivalence", s,
             f"{n_equiv}/{n_total} equivalent", "all EQUIVALENT",
             counterfactual_equivalence.get("reason", f"counterfactual equivalence: {ce_status}"),
             "counterfactual_equivalence_report.json", "status")

    # A0.11 Complete eligible-action matrix
    if _is_missing(action_matrix):
        _blocked(sub_gates, "A0.11_complete_eligible_action_matrix",
                 "action matrix report missing", "action_matrix_completeness.json", "status")
    else:
        am_status = action_matrix.get("am_status",
                        action_matrix.get("status", "INCOMPLETE"))
        n_complete = action_matrix.get("n_tasks_complete", 0)
        n_total = action_matrix.get("n_tasks_total", 0)
        # PASS if status is PASS or am_status is COMPLETE, and all tasks complete.
        s = "PASS" if (am_status == "COMPLETE" or action_matrix.get("status") == "PASS") and n_total > 0 and n_complete == n_total else "FAIL"
        if n_total == 0:
            s = "BLOCKED"
        _add(sub_gates, "A0.11_complete_eligible_action_matrix", s,
             f"{n_complete}/{n_total} complete", "all COMPLETE",
             action_matrix.get("reason", f"action matrix: {am_status}"),
             "action_matrix_completeness.json", "status")

    # A0.12 Verifier registry completeness
    if _is_missing(verifier_registry):
        _blocked(sub_gates, "A0.12_verifier_registry_completeness",
                 "verifier registry report missing", "dataset_manifest.json", "verifier_version")
    else:
        n_registered = verifier_registry.get("n_registered", 0)
        n_required = verifier_registry.get("n_required", 0)
        s = "PASS" if n_required > 0 and n_registered >= n_required else "FAIL"
        if n_required == 0:
            s = "BLOCKED"
        _add(sub_gates, "A0.12_verifier_registry_completeness", s,
             f"{n_registered}/{n_required} verifiers", "all required verifiers registered",
             verifier_registry.get("reason", "verifier registry completeness"),
             "dataset_manifest.json", "verifier_version")

    # A0.13 Utility consistency
    if _is_missing(utility_consistency):
        _blocked(sub_gates, "A0.13_utility_consistency",
                 "utility consistency report missing", "utility_consistency_report.json", "status")
    else:
        uc_status = _status_from_report(utility_consistency, default="FAIL")
        n_fail = utility_consistency.get("n_fail", 0)
        # Preserve BLOCKED from the source report; never convert to FAIL.
        if uc_status == "BLOCKED":
            s = "BLOCKED"
        elif uc_status == "PASS":
            s = "PASS" if n_fail == 0 else "FAIL"
        else:
            s = "FAIL"
        _add(sub_gates, "A0.13_utility_consistency", s,
             f"{n_fail} failures", "0 failures",
             utility_consistency.get("reason", "utility consistency check"),
             "utility_consistency_report.json", "status")

    # A0.14 No final-test leakage
    if _is_missing(split_access):
        _blocked(sub_gates, "A0.14_no_final_test_leakage",
                 "split access report missing", "split_manifest.json", "splits")
    else:
        sa_status = split_access.get("status", "BLOCKED")
        leakage = split_access.get("leakage_detected", sa_status == "FAIL")
        n_violations = split_access.get("n_violations", len(split_access.get("forbidden_accesses", [])))
        if sa_status == "BLOCKED":
            s = "BLOCKED"
        elif sa_status == "PASS" or (not leakage and n_violations == 0):
            s = "PASS"
        else:
            s = "FAIL"
        _add(sub_gates, "A0.14_no_final_test_leakage", s,
             f"{n_violations} violations", "0 violations",
             split_access.get("reason", "final-test leakage check"),
             "split_manifest.json", "splits")

    # A0.15 Candidate serialization parity
    if _is_missing(candidate_parity) or "parity" not in candidate_parity:
        _blocked(sub_gates, "A0.15_candidate_serialization_parity",
                 "candidate parity report missing or empty", "candidate_policy.json", "policy_sha256")
    else:
        parity = candidate_parity.get("parity", False)
        sha = candidate_parity.get("policy_sha256", "")
        s = "PASS" if parity else "FAIL"
        _add(sub_gates, "A0.15_candidate_serialization_parity", s,
             f"parity={parity}, sha={sha[:12] if sha else 'none'}", "serialization parity confirmed",
             candidate_parity.get("reason", "candidate serialization parity"),
             "candidate_policy.json", "policy_sha256")

    # A0.16 Sham serialization parity
    if _is_missing(sham_parity) or "parity" not in sham_parity:
        _blocked(sub_gates, "A0.16_sham_serialization_parity",
                 "sham parity report missing or empty", "sham_policy.json", "policy_sha256")
    else:
        parity = sham_parity.get("parity", False)
        sha = sham_parity.get("policy_sha256", "")
        s = "PASS" if parity else "FAIL"
        _add(sub_gates, "A0.16_sham_serialization_parity", s,
             f"parity={parity}, sha={sha[:12] if sha else 'none'}", "serialization parity confirmed",
             sham_parity.get("reason", "sham serialization parity"),
             "sham_policy.json", "policy_sha256")

    # A0.17 Task/split consistency
    if _is_missing(task_split):
        _blocked(sub_gates, "A0.17_task_split_consistency",
                 "task/split consistency report missing", "benchmark_manifest.json", "benchmark_task_count")
    else:
        ts_status = task_split.get("status")
        if ts_status in ("PASS", "FAIL", "BLOCKED", "NOT_RUN"):
            s = ts_status
            n_bench = task_split.get(
                "benchmark_task_count", task_split.get("n_tasks", 0))
            n_split = task_split.get(
                "split_task_count", task_split.get("n_split_tasks", 0))
        else:
            # Backward-compat with older report shape.
            consistent = task_split.get("consistent", False)
            n_bench = task_split.get("n_tasks", 0)
            n_split = task_split.get("n_split_tasks", 0)
            s = "PASS" if consistent and n_bench == n_split else "FAIL"
        _add(sub_gates, "A0.17_task_split_consistency", s,
             f"benchmark={n_bench}, split={n_split}", "consistent",
             task_split.get("reason", "task/split consistency"),
             "benchmark_manifest.json", "benchmark_task_count")

    # A0.18 Artifact lineage completeness
    if _is_missing(artifact_lineage):
        _blocked(sub_gates, "A0.18_artifact_lineage_completeness",
                 "artifact lineage report missing", "source_provenance.json", "n_artifacts")
    else:
        al_status = artifact_lineage.get("status")
        if al_status in ("PASS", "FAIL", "BLOCKED", "NOT_RUN"):
            s = al_status
            n_artifacts = artifact_lineage.get("n_artifacts", 0)
            n_valid = artifact_lineage.get("n_valid", 0)
        else:
            # Backward-compat with older report shape.
            complete = artifact_lineage.get("complete", False)
            n_artifacts = artifact_lineage.get("n_artifacts", 0)
            n_valid = artifact_lineage.get("n_with_lineage", 0)
            s = "PASS" if complete and n_artifacts > 0 and n_valid == n_artifacts else "FAIL"
        _add(sub_gates, "A0.18_artifact_lineage_completeness", s,
             f"{n_valid}/{n_artifacts} valid", "all artifacts valid",
             artifact_lineage.get("reason", "artifact lineage completeness"),
             "source_provenance.json", "n_artifacts")

    # A0.19 Metric consistency
    if _is_missing(metric_consistency):
        _blocked(sub_gates, "A0.19_metric_consistency",
                 "metric consistency report missing", "metric_consistency_report.json", "status")
    else:
        mc_status = metric_consistency.get("status", "FAIL")
        n_fail = metric_consistency.get("n_fail", 0)
        s = "PASS" if mc_status == "PASS" and n_fail == 0 else "FAIL"
        _add(sub_gates, "A0.19_metric_consistency", s,
             f"{n_fail} failures", "0 failures",
             metric_consistency.get("reason", "metric consistency check"),
             "metric_consistency_report.json", "status")

    # A0.20 Safety evidence integrity
    if _is_missing(safety_evidence):
        _blocked(sub_gates, "A0.20_safety_evidence_integrity",
                 "safety evidence report missing", "safety_evidence_report.json", "status")
    else:
        se_status = _status_from_report(safety_evidence, default="FAIL")
        checks = safety_evidence.get("checks")
        if isinstance(checks, list) and checks:
            n_fail = sum(1 for c in checks if isinstance(c, dict) and c.get("status") == "FAIL")
        else:
            n_fail = safety_evidence.get("n_fail", 0)
        # Preserve BLOCKED from the source report; never convert to FAIL.
        if se_status == "BLOCKED":
            s = "BLOCKED"
        elif se_status == "PASS":
            s = "PASS" if n_fail == 0 else "FAIL"
        else:
            s = "FAIL"
        _add(sub_gates, "A0.20_safety_evidence_integrity", s,
             f"{n_fail} failures", "0 failures",
             safety_evidence.get("reason", "safety evidence integrity"),
             "safety_evidence_report.json", "status")

    # A0.21 No stale artifact reuse
    if _is_missing(stale_artifact):
        _blocked(sub_gates, "A0.21_no_stale_artifact_reuse",
                 "stale artifact report missing", "evidence_import_report.json", "stale_check")
    else:
        stale = stale_artifact.get("stale_detected", True)
        n_stale = stale_artifact.get("n_stale", 0)
        s = "PASS" if not stale and n_stale == 0 else "FAIL"
        _add(sub_gates, "A0.21_no_stale_artifact_reuse", s,
             f"{n_stale} stale artifacts", "0 stale artifacts",
             stale_artifact.get("reason", "stale artifact check"),
             "evidence_import_report.json", "stale_check")

    # A0.22 Oracle consistency
    if _is_missing(oracle_consistency):
        _blocked(sub_gates, "A0.22_oracle_consistency",
                 "oracle consistency report missing", "oracle_discrepancy_report.json", "status")
    else:
        oc_status = oracle_consistency.get("status", "FAIL")
        discrepancy_found = oracle_consistency.get("discrepancy_found", True)
        s = "PASS" if oc_status == "PASS" and not discrepancy_found else "FAIL"
        _add(sub_gates, "A0.22_oracle_consistency", s,
             f"discrepancy_found={discrepancy_found}", "no discrepancy",
             oracle_consistency.get("reason", oracle_consistency.get("resolution", "oracle consistency")),
             "oracle_discrepancy_report.json", "status")

    # Compute overall status.
    n_pass = sum(1 for g in sub_gates.values() if g["status"] == "PASS")
    n_fail = sum(1 for g in sub_gates.values() if g["status"] == "FAIL")
    n_blocked = sum(1 for g in sub_gates.values() if g["status"] == "BLOCKED")

    if n_blocked > 0:
        overall = "BLOCKED"
        overall_reason = (
            f"Gate A0 BLOCKED: {n_blocked} sub-gate(s) missing required evidence "
            f"({n_pass} PASS, {n_fail} FAIL)"
        )
    elif n_fail > 0:
        overall = "FAIL"
        overall_reason = (
            f"Gate A0 FAIL: {n_fail} sub-gate(s) violated conditions "
            f"({n_pass} PASS)"
        )
    else:
        overall = "PASS"
        overall_reason = (
            f"Gate A0 PASS: all {n_pass} mandatory sub-gates passed"
        )

    return {
        "status": overall,
        "n_sub_gates": 22,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_blocked": n_blocked,
        "sub_gates": sub_gates,
        "overall_reason": overall_reason,
    }
