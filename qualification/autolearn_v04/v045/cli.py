"""Canonical CLI for v0.4.5 evidence repair analysis.

Usage:
    python -m qualification.autolearn_v04.v045.cli analyze \
        --evidence-dir qualification/evidence/seven_b_full_run \
        --output-dir qualification/autolearn_v04/artifacts/v045 \
        --run-id <run_id>

Commands:
    validate-evidence     — validate evidence package (byte + scientific)
    analyze               — run full analysis pipeline
    report                — generate report from existing outputs
    verify-output         — verify all required artifacts present
    create-evidence-package — create evidence package from original run
    create-pilot-manifest — create pilot manifest if eligible
    list-missing-evidence — list missing/placeholder evidence

Exit codes:
    0: pipeline completed (regardless of scientific PASS/FAIL)
    1: pipeline integrity failure
    2: evidence invalid or incomplete
    3: configuration error
    4: unexpected runtime error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from capsule_brain.version import PACKAGE_VERSION, AUTOLEARN_QUALIFICATION_VERSION

from qualification.autolearn_v04.common.schemas import ExitCode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qualification.autolearn_v04.v045.cli",
        description="v0.4.5 evidence repair analysis CLI",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"capsule-brain {PACKAGE_VERSION} / qualification {AUTOLEARN_QUALIFICATION_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate-evidence
    p_validate = subparsers.add_parser("validate-evidence", help="Validate evidence package")
    p_validate.add_argument("--evidence-dir", required=True)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Run full analysis pipeline")
    p_analyze.add_argument("--evidence-dir", required=True)
    p_analyze.add_argument("--output-dir", required=True)
    p_analyze.add_argument("--run-id", default="v045_analysis_001")
    p_analyze.add_argument("--repo-root", default=".")
    p_analyze.add_argument("--n-sham-seeds", type=int, default=50)
    p_analyze.add_argument("--n-candidate-seeds", type=int, default=50)
    p_analyze.add_argument("--force", action="store_true")
    p_analyze.add_argument("--resume", action="store_true")

    # report
    p_report = subparsers.add_parser("report", help="Generate report from existing outputs")
    p_report.add_argument("--output-dir", required=True)
    p_report.add_argument("--run-id", required=True)

    # verify-output
    p_verify = subparsers.add_parser("verify-output", help="Verify all required artifacts present")
    p_verify.add_argument("--output-dir", required=True)
    p_verify.add_argument("--run-id", required=True)

    # create-evidence-package
    p_create = subparsers.add_parser("create-evidence-package", help="Create evidence package from run")
    p_create.add_argument("--source-run-dir", required=True)
    p_create.add_argument("--output-dir", default="qualification/evidence/seven_b_full_run")
    p_create.add_argument("--run-id", default="scientific_full_7b_001")

    # create-pilot-manifest
    p_pilot = subparsers.add_parser("create-pilot-manifest", help="Create pilot manifest")
    p_pilot.add_argument("--output-dir", required=True)
    p_pilot.add_argument("--run-id", required=True)

    # list-missing-evidence
    p_list = subparsers.add_parser("list-missing-evidence", help="List missing/placeholder evidence")
    p_list.add_argument("--evidence-dir", required=True)

    args = parser.parse_args(argv)

    try:
        if args.command == "validate-evidence":
            return _cmd_validate_evidence(args)
        elif args.command == "analyze":
            return _cmd_analyze(args)
        elif args.command == "report":
            return _cmd_report(args)
        elif args.command == "verify-output":
            return _cmd_verify_output(args)
        elif args.command == "create-evidence-package":
            return _cmd_create_evidence(args)
        elif args.command == "create-pilot-manifest":
            return _cmd_create_pilot(args)
        elif args.command == "list-missing-evidence":
            return _cmd_list_missing(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return int(ExitCode.CONFIG_ERROR)
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        return int(ExitCode.RUNTIME_ERROR)


def _cmd_validate_evidence(args) -> int:
    from qualification.autolearn_v04.v045.evidence_import import validate_evidence_package
    from qualification.autolearn_v04.v045.placeholder_detection import detect_placeholders

    result = validate_evidence_package(args.evidence_dir)
    placeholders = detect_placeholders(args.evidence_dir)

    byte_status = result.get("byte_integrity", {}).get("status", "FAIL")
    sci_status = result.get("scientific_completeness", {}).get("status", "FAIL")
    overall = result.get("overall_eligibility", "FAIL")

    print(f"Checksum integrity: {byte_status}")
    print(f"Scientific completeness: {sci_status}")
    print(f"Overall evidence eligibility: {overall}")
    print(f"Placeholder detections: {placeholders.get('n_detections', 0)}")

    if overall == "PASS":
        return 0
    else:
        return 2  # evidence invalid or incomplete


def _cmd_analyze(args) -> int:
    from qualification.autolearn_v04.v045.orchestrator import run_all_v045_diagnostics

    output_path = Path(args.output_dir) / args.run_id
    if output_path.exists() and not args.force:
        existing = list(output_path.iterdir())
        if existing and not args.resume:
            print(f"Output directory already exists: {output_path}")
            print("Use --force to overwrite or --resume to continue.")
            return int(ExitCode.CONFIG_ERROR)

    manifest = run_all_v045_diagnostics(
        evidence_dir=args.evidence_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        repo_root=args.repo_root,
        n_sham_seeds=args.n_sham_seeds,
        n_candidate_seeds=args.n_candidate_seeds,
    )

    status = manifest.get("overall_evidence_eligibility", "FAIL")
    gate_a0 = manifest.get("gate_a0_status", "BLOCKED")
    decision = manifest.get("scale_decision", "BLOCKED")

    print(f"\n{'='*60}")
    print(f"Analysis complete")
    print(f"Evidence eligibility: {status}")
    print(f"Gate A0: {gate_a0}")
    print(f"Scale decision: {decision}")
    print(f"Approve full 14B: {manifest.get('approve_full_14b_run', False)}")
    print(f"Approve matched pilot: {manifest.get('approve_matched_14b_pilot', False)}")
    print(f"{'='*60}")

    # Exit code 0 if pipeline completed, regardless of scientific PASS/FAIL.
    # Exit code 2 if evidence is invalid/incomplete.
    if status == "PASS":
        return 0
    else:
        return 2


def _cmd_report(args) -> int:
    print("Report generation from existing outputs requires running 'analyze' first.")
    return int(ExitCode.CONFIG_ERROR)


def _cmd_verify_output(args) -> int:
    output_path = Path(args.output_dir) / args.run_id

    required = [
        "analysis_manifest.json", "analysis_source_provenance.json",
        "analysis_config.json", "byte_integrity_report.json",
        "scientific_completeness_report.json", "placeholder_detection_report.json",
        "row_count_integrity.json", "historical_identity_report.json",
        "analysis_identity_report.json", "cross_version_lineage_report.json",
        "model_identity_consistency.json", "task_split_consistency.json",
        "counterfactual_equivalence_report.json", "action_matrix_completeness.json",
        "verifier_registry_audit.json", "evidence_weight_audit.json",
        "split_access_audit.json", "candidate_serialization_parity.json",
        "sham_serialization_parity.json", "artifact_lineage_report.json",
        "safety_evidence_integrity.json", "utility_consistency_report.json",
        "metric_consistency_report.json", "oracle_discrepancy_report.json",
        "canonical_policy_evaluation.json", "raw_vs_executed_results.json",
        "feature_signal_analysis.json", "learner_validation_results.json",
        "candidate_stability_results.json", "sham_ensemble_results.json",
        "gate_a0_v045.json", "gate_a_v045_results.json",
        "model_scale_decision.json", "final_provenance_manifest.json",
        "GATE_A_V045_EVIDENCE_REPAIR_REPORT.md", "artifact_checksums.json",
    ]

    missing = []
    for name in required:
        if not (output_path / name).exists():
            missing.append(name)

    if missing:
        print(f"Missing {len(missing)} artifacts:")
        for m in missing:
            print(f"  - {m}")
        return int(ExitCode.INTEGRITY_FAILURE)
    else:
        print(f"All {len(required)} required artifacts present.")
        return 0


def _cmd_create_evidence(args) -> int:
    source_path = Path(args.source_run_dir)
    if not source_path.exists():
        print(f"Source run directory does not exist: {source_path}", file=sys.stderr)
        return int(ExitCode.CONFIG_ERROR)

    print(f"Creating evidence package from: {source_path}")
    print("This command requires the actual Modal artifact directory.")
    print("If the source directory is incomplete, the command will report missing files.")

    # Check for required source files
    required = [
        "counterfactual_outcomes.jsonl", "executive_experiences.jsonl",
        "candidate_policy.json", "sham_policy.json",
        "benchmark_manifest.json", "split_manifest.json",
        "provider_manifest.json", "source_provenance.json",
    ]
    missing = [f for f in required if not (source_path / f).exists()]
    if missing:
        print(f"\nMissing required source files:")
        for m in missing:
            print(f"  - {m}")
        return 2

    print("All required source files present. Evidence package creation not yet implemented.")
    return int(ExitCode.CONFIG_ERROR)


def _cmd_create_pilot(args) -> int:
    print("Pilot manifest creation requires valid evidence and 'analyze' first.")
    return int(ExitCode.CONFIG_ERROR)


def _cmd_list_missing(args) -> int:
    from qualification.autolearn_v04.v045.evidence_import import validate_evidence_package
    from qualification.autolearn_v04.v045.placeholder_detection import detect_placeholders
    from qualification.autolearn_v04.v045.config import REQUIRED_EVIDENCE_FILES

    ev_path = Path(args.evidence_dir)

    print("=== Missing Evidence Report ===")
    print()

    # Missing files
    missing_files = []
    for f in REQUIRED_EVIDENCE_FILES:
        if not (ev_path / f).exists():
            missing_files.append(f)
    if missing_files:
        print("Missing files:")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("All required files present.")
    print()

    # Placeholder files
    placeholders = detect_placeholders(ev_path)
    if placeholders.get("detections"):
        print("Placeholder files:")
        for d in placeholders["detections"]:
            print(f"  - {d.get('file', '')}: {d.get('reason', '')}")
    else:
        print("No placeholder markers detected.")
    print()

    # Row count mismatches
    from qualification.autolearn_v04.v045.row_count_integrity import check_row_counts
    row_counts = check_row_counts(ev_path)
    cf = row_counts.get("counterfactual_rows", {})
    if not cf.get("match", False):
        print(f"Row count mismatch: counterfactual actual={cf.get('actual', 0)}, declared={cf.get('declared', 0)}")
    print()

    # Empty task ID lists
    split_manifest_path = ev_path / "split_manifest.json"
    if split_manifest_path.exists():
        sm = json.loads(split_manifest_path.read_text())
        for split_name, split_data in sm.get("splits", {}).items():
            count = split_data.get("count", 0)
            task_ids = split_data.get("task_ids", [])
            if count > 0 and len(task_ids) == 0:
                print(f"Empty task IDs: split '{split_name}' declares count={count} but has 0 task IDs")
    print()

    # Unknown source identity
    sp_path = ev_path / "source_provenance.json"
    if sp_path.exists():
        sp = json.loads(sp_path.read_text())
        commit = sp.get("original_commit_sha", "unknown")
        if commit == "unknown" or commit is None:
            print(f"Unknown source identity: original_commit_sha = {commit}")
    print()

    # Missing policy state
    for policy_file in ["candidate_policy.json", "sham_policy.json"]:
        p_path = ev_path / policy_file
        if p_path.exists():
            p = json.loads(p_path.read_text())
            if "weights" not in p:
                print(f"Missing policy state: {policy_file} has no 'weights' field")
    print()

    # Missing decision traces
    if not (ev_path / "candidate_decision_trace.jsonl").exists():
        print("Missing decision traces: candidate_decision_trace.jsonl absent")
    print()

    # Missing feature data
    if not (ev_path / "feature_schema.json").exists():
        print("Missing feature data: feature_schema.json absent")
    print()

    # Missing safety rows
    safety_jsonl = ev_path / "safety_results.jsonl"
    safety_json = ev_path / "safety_results.json"
    if safety_jsonl.exists():
        with open(safety_jsonl) as f:
            count = sum(1 for line in f if line.strip())
        if count == 0:
            print("Missing safety rows: safety_results.jsonl is empty")
    elif safety_json.exists():
        print("Missing safety rows: safety_results.json is summary-only (task-level rows required)")
    else:
        print("Missing safety rows: no safety results file found")

    return 0


if __name__ == "__main__":
    sys.exit(main())
