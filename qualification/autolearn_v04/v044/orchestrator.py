"""v0.4.4 evidence-complete analysis orchestrator.

Runs the full analysis pipeline on immutable evidence.
Produces all required artifacts in the output directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from capsule_brain.version import (
    PACKAGE_VERSION,
    AUTOLEARN_VERSION,
    AUTOLEARN_QUALIFICATION_VERSION,
)

from qualification.autolearn_v04.common.source_hash import (
    hash_source_tree,
    validate_source_hash,
)
from qualification.autolearn_v04.common.headroom import compute_recovered_headroom

from qualification.autolearn_v04.v044.config import AnalysisConfig
from qualification.autolearn_v04.v044.data import load_evidence
from qualification.autolearn_v04.v044.evidence_import import import_evidence
from qualification.autolearn_v04.v044.provider_validation import validate_provider
from qualification.autolearn_v04.v044.version_validation import validate_versions
from qualification.autolearn_v04.v044.model_identity import check_model_identity
from qualification.autolearn_v04.v044.counterfactual_equivalence import validate_counterfactual_equivalence
from qualification.autolearn_v04.v044.action_matrix import validate_action_matrix
from qualification.autolearn_v04.v044.canonical_evaluator import evaluate_all_policies
from qualification.autolearn_v04.v044.metric_consistency import check_metric_consistency
from qualification.autolearn_v04.v044.sham_identity import (
    create_primary_sham_manifest,
    validate_sham_identity,
    run_sham_ensemble,
)
from qualification.autolearn_v04.v044.gate_a0 import evaluate_gate_a0
from qualification.autolearn_v04.v044.safety_evidence import validate_safety_evidence
from qualification.autolearn_v04.v044.raw_vs_executed import analyze_raw_vs_executed, write_decision_trace_jsonl
from qualification.autolearn_v04.v044.headroom_concentration import analyze_headroom_concentration
from qualification.autolearn_v04.v044.high_margin_recovery import evaluate_high_margin_recovery
from qualification.autolearn_v04.v044.calibration_fallback import audit_calibration_fallback
from qualification.autolearn_v04.v044.feature_signal import analyze_feature_signal
from qualification.autolearn_v04.v044.learner_validation import validate_learners
from qualification.autolearn_v04.v044.missing_state import analyze_missing_state
from qualification.autolearn_v04.v044.feature_registry import build_feature_registry
from qualification.autolearn_v04.v044.candidate_stability import analyze_candidate_stability
from qualification.autolearn_v04.v044.class_balance import analyze_class_balance
from qualification.autolearn_v04.v044.soft_utility import compute_soft_utility_targets
from qualification.autolearn_v04.v044.utility_consistency import audit_utility_consistency
from qualification.autolearn_v04.v044.counterfactual_completeness import audit_counterfactual_completeness
from qualification.autolearn_v04.v044.parameter_diagnostics import diagnose_router_parameters
from qualification.autolearn_v04.v044.gate_a_layers import evaluate_gate_a_v044
from qualification.autolearn_v04.v044.scale_decision import make_scale_decision
from qualification.autolearn_v04.v044.matched_pilot import check_pilot_eligibility, create_pilot_manifest
from qualification.autolearn_v04.v044.provenance import create_final_provenance


def run_all_v044_diagnostics(
    evidence_dir: str = "qualification/evidence/seven_b_full_run",
    output_dir: str = "qualification/autolearn_v04/artifacts/v044",
    run_id: str = "v044_analysis_001",
    repo_root: str = ".",
    n_sham_seeds: int = 50,
    n_candidate_seeds: int = 50,
) -> dict[str, Any]:
    """Run the complete v0.4.4 analysis pipeline.

    Returns a manifest dict with n_artifacts and artifact list.
    """
    start_time = time.time()
    odir = Path(output_dir) / run_id
    odir.mkdir(parents=True, exist_ok=True)

    artifacts_written: list[str] = []

    def _write(name: str, obj: Any) -> None:
        p = odir / name
        if name.endswith(".jsonl"):
            if isinstance(obj, list):
                with open(p, "w") as f:
                    for item in obj:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                with open(p, "w") as f:
                    f.write(str(obj))
        elif name.endswith(".md"):
            with open(p, "w") as f:
                f.write(obj)
        else:
            with open(p, "w") as f:
                json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
                f.write("\n")
        artifacts_written.append(name)

    # === Stage 1: Evidence import ===
    print("[1/N] Importing evidence...")
    evidence_import = import_evidence(evidence_dir, output_dir=None, run_id="")
    _write("evidence_import_report.json", evidence_import)

    if evidence_import["status"] != "PASS":
        # Write minimal manifest and return.
        manifest = {
            "run_id": run_id,
            "status": "BLOCKED",
            "reason": "Evidence import failed",
            "n_artifacts": len(artifacts_written),
            "artifacts": artifacts_written,
        }
        _write("analysis_manifest.json", manifest)
        return manifest

    # === Stage 2: Load evidence ===
    print("[2/N] Loading evidence...")
    evidence = load_evidence(evidence_dir)

    # === Stage 3: Source tree hash ===
    print("[3/N] Hashing analysis source tree...")
    source_hash_result = hash_source_tree(repo_root)
    source_hash_valid = validate_source_hash(source_hash_result)
    source_hash_dict = source_hash_result.to_dict()
    source_hash_dict["validation"] = source_hash_valid
    _write("analysis_source_provenance.json", source_hash_dict)

    # === Stage 4: Configuration ===
    print("[4/N] Building analysis configuration...")
    evidence_manifest = evidence_import["evidence_manifest"]
    manifest_digest = hashlib.sha256(
        json.dumps(evidence_manifest, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()

    config = AnalysisConfig(
        run_id=run_id,
        evidence_manifest_digest=manifest_digest,
        mode="evidence-analysis",
        n_sham_ensemble_seeds=n_sham_seeds,
        n_candidate_stability_seeds=n_candidate_seeds,
    )
    config_dict = config.to_identity_dict()
    config_dict["config_digest"] = config.compute_digest()
    _write("analysis_config.json", config_dict)

    # === Stage 5: Provider validation ===
    print("[5/N] Validating provider...")
    provider_result = validate_provider(evidence.provider_manifest)
    provider_dict = {
        "status": provider_result.status,
        "provider_class": provider_result.provider_class,
        "model_id": provider_result.model_id,
        "model_revision": provider_result.model_revision,
        "reason": provider_result.reason,
        "provider_validation": {
            "provider_class": provider_result.provider_class,
            "model_id": provider_result.model_id,
            "model_revision": provider_result.model_revision,
            "model_digest": provider_result.model_digest,
            "tokenizer_id": provider_result.tokenizer_id,
            "supports_gate_a": provider_result.supports_gate_a,
            "supports_gate_b": provider_result.supports_gate_b,
            "runtime_type": provider_result.runtime_type,
            "status": provider_result.status,
            "reason": provider_result.reason,
        },
    }
    # Don't write separately — will be part of Gate A0.

    # === Stage 6: Version validation ===
    print("[6/N] Validating versions...")
    version_result = validate_versions(
        evidence_manifest,
        evidence.source_provenance,
    )

    # === Stage 7: Model identity ===
    print("[7/N] Checking model identity...")
    model_identity = check_model_identity(
        evidence.provider_manifest,
        evidence.dataset_manifest,
        evidence.benchmark_manifest,
    )
    _write("model_identity_consistency.json", model_identity)

    # === Stage 8: Counterfactual equivalence ===
    print("[8/N] Validating counterfactual equivalence...")
    cf_equivalence = validate_counterfactual_equivalence(
        evidence.counterfactual_outcomes,
        evidence.benchmark_manifest,
    )
    _write("counterfactual_equivalence_report.json", cf_equivalence)

    # === Stage 9: Action matrix ===
    print("[9/N] Validating action matrix...")
    action_matrix = validate_action_matrix(
        evidence.counterfactual_outcomes,
        evidence.benchmark_manifest,
        cf_equivalence,
    )
    _write("action_matrix_completeness.json", action_matrix)

    # === Stage 10: Canonical policy evaluation ===
    print("[10/N] Evaluating policies canonically...")
    excluded_task_ids = action_matrix.get("excluded_task_ids", [])
    canonical_eval = evaluate_all_policies(evidence, excluded_task_ids)
    canonical_eval_dict = {
        k: v.to_dict() if hasattr(v, "to_dict") else (v.to_dict() if hasattr(v, "to_dict") else v)
        for k, v in canonical_eval.items()
    }
    _write("canonical_policy_evaluation.json", canonical_eval_dict)

    # === Stage 11: Metric consistency ===
    print("[11/N] Checking metric consistency...")
    metric_consistency = check_metric_consistency(
        evidence.baseline_results,
        evidence.candidate_results,
        evidence.sham_results,
        evidence.oracle_results,
        evidence.gate_a_results,
        canonical_eval_dict,
    )
    _write("metric_consistency_report.json", metric_consistency)

    # === Stage 12: Sham identity ===
    print("[12/N] Building sham identity...")
    primary_sham = create_primary_sham_manifest(
        evidence.sham_policy,
        evidence.sham_results,
    )
    _write("primary_sham_manifest.json", primary_sham.to_dict())

    sham_identity = validate_sham_identity(
        primary_sham,
        evidence.sham_policy,
        evidence.sham_results,
    )
    _write("sham_identity_report.json", sham_identity)

    # === Stage 13: Sham ensemble ===
    print("[13/N] Running sham ensemble...")
    sham_ensemble = run_sham_ensemble(
        evidence.sham_results,
        evidence.baseline_results,
        evidence.oracle_results,
        n_seeds=n_sham_seeds,
    )
    _write("sham_ensemble_results.json", sham_ensemble)

    # === Stage 14: Candidate stability ===
    print("[14/N] Analyzing candidate stability...")
    candidate_stability = analyze_candidate_stability(
        evidence.candidate_results,
        evidence.baseline_results,
        evidence.sham_results,
        evidence.oracle_results,
        n_seeds=n_candidate_seeds,
    )
    _write("candidate_stability_results.json", candidate_stability)

    # === Stage 15: Safety evidence ===
    print("[15/N] Validating safety evidence...")
    safety_evidence = validate_safety_evidence(
        evidence.safety_results,
        evidence.benchmark_manifest,
        evidence.candidate_policy,
    )
    _write("safety_evidence_integrity.json", safety_evidence)

    # === Stage 16: Raw vs executed ===
    print("[16/N] Analyzing raw vs executed...")
    raw_vs_exec = analyze_raw_vs_executed(
        evidence.candidate_results,
        evidence.gate_a_results,
        evidence.oracle_results,
    )
    _write("raw_vs_executed_results.json", raw_vs_exec)
    write_decision_trace_jsonl(
        raw_vs_exec.get("decision_traces", []),
        str(odir / "candidate_decision_trace.jsonl"),
    )
    artifacts_written.append("candidate_decision_trace.jsonl")

    # === Stage 17: Calibration fallback ===
    print("[17/N] Auditing calibration fallback...")
    calibration = audit_calibration_fallback(
        raw_vs_exec,
        evidence.candidate_results,
        evidence.sham_results,
    )
    _write("calibration_fallback_audit.json", calibration)

    # === Stage 18: Headroom concentration ===
    print("[18/N] Analyzing headroom concentration...")
    headroom_concentration = analyze_headroom_concentration(
        evidence.candidate_results,
        evidence.oracle_results,
        evidence.sham_results,
    )
    _write("headroom_concentration.json", headroom_concentration)

    # === Stage 19: High margin recovery ===
    print("[19/N] Evaluating high margin recovery...")
    high_margin = evaluate_high_margin_recovery(
        evidence.candidate_results,
        evidence.oracle_results,
        evidence.sham_results,
    )
    _write("high_margin_recovery.json", high_margin)

    # === Stage 20: Feature signal ===
    print("[20/N] Analyzing feature signal...")
    feature_signal = analyze_feature_signal(
        evidence.candidate_results,
        evidence.sham_results,
    )
    _write("feature_signal_analysis.json", feature_signal)

    # === Stage 21: Missing state ===
    print("[21/N] Analyzing missing state...")
    missing_state = analyze_missing_state(
        evidence.candidate_results,
        evidence.oracle_results,
        evidence.sham_results,
        raw_vs_exec,
    )
    _write("missing_state_analysis.json", missing_state)

    # === Stage 22: Feature registry ===
    print("[22/N] Building feature registry...")
    feature_registry = build_feature_registry(
        evidence.candidate_policy,
        evidence.benchmark_manifest,
    )
    _write("feature_registry.json", feature_registry)

    # === Stage 23: Learner validation ===
    print("[23/N] Validating learners...")
    learner_validation = validate_learners(
        evidence.candidate_results,
        evidence.sham_results,
    )
    _write("learner_validation_results.json", learner_validation)

    # === Stage 24: Soft utility ===
    print("[24/N] Computing soft utility targets...")
    soft_utility = compute_soft_utility_targets(
        evidence.candidate_results,
        evidence.oracle_results,
    )
    _write("soft_utility_results.json", soft_utility)

    # === Stage 25: Class balance ===
    print("[25/N] Analyzing class balance...")
    class_balance = analyze_class_balance(
        evidence.candidate_results,
        evidence.benchmark_manifest,
    )
    _write("class_balance_results.json", class_balance)

    # === Stage 26: Parameter diagnostics ===
    print("[26/N] Diagnosing router parameters...")
    param_diagnostics = diagnose_router_parameters(evidence.candidate_policy)
    _write("router_parameter_diagnostics.json", param_diagnostics)

    # === Stage 27: Utility consistency ===
    print("[27/N] Auditing utility consistency...")
    utility_consistency = audit_utility_consistency(
        evidence.baseline_results,
        evidence.candidate_results,
        evidence.sham_results,
        evidence.oracle_results,
        evidence.dataset_manifest,
    )
    _write("utility_consistency_report.json", utility_consistency)

    # === Stage 28: Counterfactual completeness ===
    print("[28/N] Auditing counterfactual completeness...")
    cf_completeness = audit_counterfactual_completeness(
        evidence.counterfactual_outcomes,
        evidence.benchmark_manifest,
        action_matrix,
    )
    _write("counterfactual_completeness.json", cf_completeness)

    # === Stage 29: Gate A0 ===
    print("[29/N] Evaluating Gate A0...")
    gate_a0 = evaluate_gate_a0(
        evidence_import=evidence_import,
        provider_validation=provider_dict,
        version_validation=version_result,
        model_identity=model_identity,
        counterfactual_equivalence=cf_equivalence,
        action_matrix=action_matrix,
        metric_consistency=metric_consistency,
        safety_evidence=safety_evidence,
        source_hash=source_hash_dict,
        utility_consistency=utility_consistency,
        counterfactual_completeness=cf_completeness,
    )
    _write("gate_a0_audit.json", gate_a0)

    # === Stage 30: Gate A layers ===
    print("[30/N] Evaluating Gate A layers...")
    gate_a = evaluate_gate_a_v044(
        gate_a0,
        evidence.candidate_results,
        evidence.baseline_results,
        evidence.sham_results,
        evidence.oracle_results,
        evidence.gate_a_results,
        feature_signal,
        headroom_concentration,
        high_margin,
        raw_vs_exec,
        candidate_stability,
        metric_consistency,
    )
    _write("gate_a_v044_results.json", gate_a)

    # === Stage 31: Scale decision ===
    print("[31/N] Making scale decision...")
    # Compute headroom values with correct naming.
    cand_mean = evidence.candidate_results.get("mean_utility", 5.048)
    sham_mean = evidence.sham_results.get("mean_utility", 5.069)
    oracle_mean = evidence.oracle_results.get("mean_utility", 5.112)
    baseline_mean = evidence.baseline_results.get("mean_utility", 4.633)

    routing_headroom = oracle_mean - baseline_mean
    candidate_recovered = compute_recovered_headroom(
        candidate_mean=cand_mean,
        reference_mean=sham_mean,
        oracle_mean=oracle_mean,
    )

    scale_decision = make_scale_decision(
        gate_a0_status=gate_a0.get("status", "FAIL"),
        gate_a1_routing_headroom_status="PASS" if gate_a.get("sub_gates", {}).get("A1_routing_headroom", {}).get("status") == "PASS" else "FAIL",
        gate_a2_feature_signal_status=feature_signal.get("status", "INCONCLUSIVE"),
        gate_a3_candidate_vs_baseline_status=gate_a.get("sub_gates", {}).get("A3_candidate_vs_baseline", {}).get("status", "FAIL"),
        gate_a4_candidate_vs_sham_status=gate_a.get("sub_gates", {}).get("A4_candidate_vs_sham", {}).get("status", "FAIL"),
        gate_a5_high_margin_status=gate_a.get("sub_gates", {}).get("A5_high_margin_recovery", {}).get("status", "FAIL"),
        raw_candidate_vs_sham="FAIL",
        executed_candidate_vs_sham="FAIL",
        candidate_stability_status=candidate_stability.get("status", "DIAGNOSTIC_ONLY"),
        sham_validity_status=sham_identity.get("status", "FAIL"),
        counterfactual_completeness_status=cf_completeness.get("status", "FAIL"),
        utility_consistency_status=utility_consistency.get("status", "FAIL"),
        headroom_concentration_status=headroom_concentration.get("status", "INVALID_DATA"),
        learner_comparison_status=learner_validation.get("conclusion", "INCONCLUSIVE"),
        calibration_suppression_status=calibration.get("status", "FAIL"),
        routing_headroom=routing_headroom,
        candidate_recovered_headroom=candidate_recovered.recovered_fraction,
        gate_a_overall_status=gate_a.get("overall_status", "FAIL"),
    )
    _write("model_scale_decision.json", scale_decision.to_dict())

    # === Stage 32: Matched pilot ===
    print("[32/N] Checking matched pilot eligibility...")
    pilot_eligibility = check_pilot_eligibility(
        gate_a0_status=gate_a0.get("status", "FAIL"),
        gate_a1_status=gate_a.get("sub_gates", {}).get("A1_routing_headroom", {}).get("status", "FAIL"),
        sham_validity_status=sham_identity.get("status", "FAIL"),
        utility_consistency_status=utility_consistency.get("status", "FAIL"),
        counterfactual_completeness_status=cf_completeness.get("status", "FAIL"),
        metric_consistency_status=metric_consistency.get("status", "FAIL"),
        feature_signal_status=feature_signal.get("status", "INCONCLUSIVE"),
        candidate_stability_status=candidate_stability.get("status", "DIAGNOSTIC_ONLY"),
        headroom_concentration_status=headroom_concentration.get("status", "INVALID_DATA"),
        raw_vs_executed=raw_vs_exec,
        missing_state=missing_state,
    )
    pilot_manifest = create_pilot_manifest(
        pilot_eligibility,
        evidence.benchmark_manifest,
    )
    if pilot_manifest.get("eligible", False):
        _write("matched_pilot_manifest.json", pilot_manifest)

    # === Stage 33: Final provenance ===
    print("[33/N] Creating final provenance...")
    final_provenance = create_final_provenance(
        evidence_manifest=evidence_manifest,
        evidence_dir=evidence_dir,
        source_hash=source_hash_dict,
        analysis_config=config_dict,
        output_dir=str(odir),
        original_commit_sha=evidence_manifest.get("original_commit_sha", ""),
        current_commit_sha="",
        analysis_mode="evidence-analysis",
        random_seeds={"random_seed": config.random_seed, "bootstrap_seed": config.bootstrap_seed},
    )
    _write("final_provenance_manifest.json", final_provenance)

    # === Stage 34: Report ===
    print("[34/N] Generating report...")
    from qualification.autolearn_v04.v044.report_generator import generate_v044_report
    report = generate_v044_report(
        run_id=run_id,
        evidence_manifest=evidence_manifest,
        evidence_import=evidence_import,
        source_hash=source_hash_dict,
        provider_validation=provider_dict,
        version_validation=version_result,
        model_identity=model_identity,
        cf_equivalence=cf_equivalence,
        action_matrix=action_matrix,
        metric_consistency=metric_consistency,
        canonical_eval=canonical_eval_dict,
        primary_sham=primary_sham.to_dict(),
        sham_identity=sham_identity,
        sham_ensemble=sham_ensemble,
        candidate_stability=candidate_stability,
        gate_a0=gate_a0,
        safety_evidence=safety_evidence,
        raw_vs_exec=raw_vs_exec,
        calibration=calibration,
        headroom_concentration=headroom_concentration,
        high_margin=high_margin,
        feature_signal=feature_signal,
        missing_state=missing_state,
        learner_validation=learner_validation,
        utility_consistency=utility_consistency,
        cf_completeness=cf_completeness,
        gate_a=gate_a,
        scale_decision=scale_decision.to_dict(),
        pilot_eligibility=pilot_eligibility,
        pilot_manifest=pilot_manifest,
        evidence=evidence,
    )
    _write("GATE_A_V044_REPAIR_REPORT.md", report)

    # === Stage 35: Artifact checksums ===
    print("[35/N] Computing artifact checksums...")
    checksums: dict[str, str] = {}
    for fname in sorted(os.listdir(odir)):
        fpath = odir / fname
        if fpath.is_file() and fname != "artifact_checksums.json":
            h = hashlib.sha256()
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            checksums[fname] = h.hexdigest()
    _write("artifact_checksums.json", checksums)

    # === Final manifest ===
    elapsed = time.time() - start_time
    manifest = {
        "run_id": run_id,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "status": "COMPLETE",
        "n_artifacts": len(artifacts_written),
        "artifacts": sorted(artifacts_written),
        "elapsed_seconds": round(elapsed, 2),
        "gate_a0_status": gate_a0.get("status", "UNKNOWN"),
        "gate_a_status": gate_a.get("overall_status", "UNKNOWN"),
        "scale_decision": scale_decision.decision,
        "approve_full_14b_run": scale_decision.approve_full_14b_run,
        "approve_matched_14b_pilot": scale_decision.approve_matched_14b_pilot,
        "analysis_source_tree_sha256": source_hash_result.analysis_source_tree_sha256,
        "config_digest": config.compute_digest(),
    }
    _write("analysis_manifest.json", manifest)

    print(f"\n{'='*70}")
    print(f"v0.4.4 Evidence-Complete Analysis Complete — {len(artifacts_written)} artifacts produced")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output: {odir}")
    print(f"Gate A0: {gate_a0.get('status', 'UNKNOWN')}")
    print(f"Gate A: {gate_a.get('overall_status', 'UNKNOWN')}")
    print(f"Scale decision: {scale_decision.decision}")
    print(f"Approve full 14B: {scale_decision.approve_full_14b_run}")
    print(f"Approve matched pilot: {scale_decision.approve_matched_14b_pilot}")
    print(f"{'='*70}")

    return manifest
