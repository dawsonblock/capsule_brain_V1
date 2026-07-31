"""Canonical CLI for v0.4.4 analysis.

Usage:
    python -m qualification.autolearn_v04.v044.cli analyze \
        --evidence-dir qualification/evidence/fixtures/synthetic_7b_routing \
        --output-dir qualification/autolearn_v04/artifacts/v044/<run_id> \
        --run-id <run_id>

Commands:
    validate-evidence  — validate evidence package only
    analyze             — run full analysis pipeline
    report              — generate report from existing outputs
    verify-output       — verify all required artifacts present
    create-evidence-package — create evidence package from run data
    create-pilot-manifest — create pilot manifest if eligible

Exit codes:
    0: pipeline completed (regardless of scientific PASS/FAIL)
    1: pipeline integrity failure
    2: evidence validation failure
    3: configuration error
    4: unexpected runtime error
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from capsule_brain.version import PACKAGE_VERSION, AUTOLEARN_QUALIFICATION_VERSION

from qualification.autolearn_v04.common.schemas import ExitCode


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="qualification.autolearn_v04.v044.cli",
        description="v0.4.4 evidence-complete Gate A repair analysis CLI",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"capsule-brain {PACKAGE_VERSION} / qualification {AUTOLEARN_QUALIFICATION_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate-evidence
    p_validate = subparsers.add_parser("validate-evidence", help="Validate evidence package")
    p_validate.add_argument("--evidence-dir", required=True)
    p_validate.add_argument("--output-dir", default=None)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Run full analysis pipeline")
    p_analyze.add_argument("--evidence-dir", required=True)
    p_analyze.add_argument("--output-dir", required=True)
    p_analyze.add_argument("--run-id", default="v044_analysis_001")
    p_analyze.add_argument("--repo-root", default=".")
    p_analyze.add_argument("--n-sham-seeds", type=int, default=50)
    p_analyze.add_argument("--n-candidate-seeds", type=int, default=50)
    p_analyze.add_argument("--force", action="store_true", help="Overwrite existing output")
    p_analyze.add_argument("--resume", action="store_true", help="Resume from existing output")

    # report
    p_report = subparsers.add_parser("report", help="Generate report from existing outputs")
    p_report.add_argument("--output-dir", required=True)
    p_report.add_argument("--run-id", required=True)

    # verify-output
    p_verify = subparsers.add_parser("verify-output", help="Verify all required artifacts present")
    p_verify.add_argument("--output-dir", required=True)
    p_verify.add_argument("--run-id", required=True)

    # create-evidence-package
    p_create = subparsers.add_parser("create-evidence-package", help="Create evidence package")
    p_create.add_argument("--evidence-dir", default="qualification/evidence/fixtures/synthetic_7b_routing")

    # create-pilot-manifest
    p_pilot = subparsers.add_parser("create-pilot-manifest", help="Create pilot manifest")
    p_pilot.add_argument("--output-dir", required=True)
    p_pilot.add_argument("--run-id", required=True)

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
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return int(ExitCode.CONFIG_ERROR)
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        return int(ExitCode.RUNTIME_ERROR)


def _cmd_validate_evidence(args) -> int:
    """Validate evidence package."""
    from qualification.autolearn_v04.v044.evidence_import import validate_evidence_package

    result = validate_evidence_package(args.evidence_dir)
    status = result.get("status", "FAIL")

    print(f"Evidence validation: {status}")
    print(f"Files validated: {result.get('n_files_validated', 0)}")

    errors = result.get("errors", [])
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")

    if status == "PASS":
        return 0
    elif status == "BLOCKED":
        return int(ExitCode.INTEGRITY_FAILURE)
    else:
        return int(ExitCode.EVIDENCE_FAILURE)


def _cmd_analyze(args) -> int:
    """Run full analysis pipeline."""
    from qualification.autolearn_v04.v044.orchestrator import run_all_v044_diagnostics

    # Check for existing output.
    output_path = Path(args.output_dir) / args.run_id
    if output_path.exists() and not args.force:
        existing = list(output_path.iterdir())
        if existing and not args.resume:
            print(f"Output directory already exists: {output_path}")
            print("Use --force to overwrite or --resume to continue.")
            return int(ExitCode.CONFIG_ERROR)

    manifest = run_all_v044_diagnostics(
        evidence_dir=args.evidence_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        repo_root=args.repo_root,
        n_sham_seeds=args.n_sham_seeds,
        n_candidate_seeds=args.n_candidate_seeds,
    )

    status = manifest.get("status", "UNKNOWN")
    n_artifacts = manifest.get("n_artifacts", 0)
    gate_a0 = manifest.get("gate_a0_status", "UNKNOWN")
    gate_a = manifest.get("gate_a_status", "UNKNOWN")
    decision = manifest.get("scale_decision", "UNKNOWN")

    print(f"\n{'='*60}")
    print(f"Analysis complete: {status}")
    print(f"Artifacts: {n_artifacts}")
    print(f"Gate A0: {gate_a0}")
    print(f"Gate A: {gate_a}")
    print(f"Scale decision: {decision}")
    print(f"Approve full 14B: {manifest.get('approve_full_14b_run', False)}")
    print(f"Approve matched pilot: {manifest.get('approve_matched_14b_pilot', False)}")
    print(f"{'='*60}")

    # Exit code 0 if pipeline completed, regardless of scientific PASS/FAIL.
    if status == "COMPLETE":
        return 0
    elif status == "BLOCKED":
        # Evidence validation failure.
        return int(ExitCode.EVIDENCE_FAILURE)
    else:
        return int(ExitCode.INTEGRITY_FAILURE)


def _cmd_report(args) -> int:
    """Generate report from existing outputs."""
    print("Report generation from existing outputs is not yet implemented.")
    print("Use 'analyze' to run the full pipeline which includes report generation.")
    return int(ExitCode.CONFIG_ERROR)


def _cmd_verify_output(args) -> int:
    """Verify all required artifacts are present."""
    output_path = Path(args.output_dir) / args.run_id

    required = [
        "analysis_manifest.json", "analysis_source_provenance.json",
        "analysis_config.json", "evidence_import_report.json",
        "model_identity_consistency.json", "counterfactual_equivalence_report.json",
        "action_matrix_completeness.json", "metric_consistency_report.json",
        "primary_sham_manifest.json", "sham_identity_report.json",
        "sham_ensemble_results.json", "candidate_stability_results.json",
        "gate_a0_audit.json", "safety_evidence_integrity.json",
        "candidate_decision_trace.jsonl", "raw_vs_executed_results.json",
        "calibration_fallback_audit.json", "headroom_concentration.json",
        "high_margin_recovery.json", "feature_signal_analysis.json",
        "missing_state_analysis.json", "feature_registry.json",
        "learner_validation_results.json", "soft_utility_results.json",
        "class_balance_results.json", "router_parameter_diagnostics.json",
        "utility_consistency_report.json", "counterfactual_completeness.json",
        "canonical_policy_evaluation.json", "gate_a_v044_results.json",
        "model_scale_decision.json", "final_provenance_manifest.json",
        "GATE_A_V044_REPAIR_REPORT.md", "artifact_checksums.json",
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
    """Create evidence package."""
    from qualification.autolearn_v04.v044.create_evidence import create_evidence_package

    result = create_evidence_package(args.evidence_dir)
    print(f"Evidence package created: {result}")
    return 0


def _cmd_create_pilot(args) -> int:
    """Create pilot manifest."""
    print("Pilot manifest creation requires running 'analyze' first.")
    return int(ExitCode.CONFIG_ERROR)


if __name__ == "__main__":
    sys.exit(main())
