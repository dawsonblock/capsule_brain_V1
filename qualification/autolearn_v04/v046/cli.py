"""Canonical CLI for v0.4.6 evidence repair analysis.

Usage:
    python -m qualification.autolearn_v04.v046.cli analyze \
        --evidence-dir qualification/evidence/fixtures/synthetic_7b_routing \
        --output-dir qualification/autolearn_v04/artifacts/v046 \
        --run-id <run_id>

Commands:
    validate-fixture               — validate a synthetic fixture evidence package
    validate-evidence              — validate evidence for scientific claims
    scan-evidence-origins          — scan a root directory for evidence origins
    detect-cross-origin-duplicates — detect cross-origin duplication
    recover-modal-evidence         — recover modal evidence from a source dir
    list-missing-modal-evidence    — list missing modal evidence files
    analyze                        — run full analysis pipeline
    report                         — generate report from existing outputs
    verify-output                  — verify all required artifacts present

Exit codes:
    0: success (pipeline completed, regardless of scientific PASS/FAIL)
    1: pipeline runtime error
    2: invalid scientific evidence
    3: configuration error
    4: cross-origin integrity violation
    5: required historical evidence unavailable

Fixture validation may exit 0 while stating scientific_eligibility=false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from capsule_brain.version import PACKAGE_VERSION, AUTOLEARN_QUALIFICATION_VERSION


# ---------------------------------------------------------------------------
# Exit codes (v0.4.6 has a richer set than the common ExitCode enum)
# ---------------------------------------------------------------------------

EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_INVALID_SCIENTIFIC_EVIDENCE = 2
EXIT_CONFIG_ERROR = 3
EXIT_CROSS_ORIGIN_VIOLATION = 4
EXIT_HISTORICAL_UNAVAILABLE = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qualification.autolearn_v04.v046.cli",
        description="v0.4.6 evidence repair analysis CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"capsule-brain {PACKAGE_VERSION} / qualification {AUTOLEARN_QUALIFICATION_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate-fixture
    p_vf = subparsers.add_parser(
        "validate-fixture", help="Validate a synthetic fixture evidence package"
    )
    p_vf.add_argument("--evidence-dir", required=True)

    # validate-evidence
    p_ve = subparsers.add_parser(
        "validate-evidence", help="Validate evidence for scientific claims"
    )
    p_ve.add_argument("--evidence-dir", required=True)

    # scan-evidence-origins
    p_scan = subparsers.add_parser(
        "scan-evidence-origins", help="Scan a root directory for evidence origins"
    )
    p_scan.add_argument("--root", required=True)

    # detect-cross-origin-duplicates
    p_dup = subparsers.add_parser(
        "detect-cross-origin-duplicates",
        help="Detect cross-origin duplication across evidence packages",
    )
    p_dup.add_argument("--root", required=True)

    # recover-modal-evidence
    p_rec = subparsers.add_parser(
        "recover-modal-evidence", help="Recover modal evidence from a source directory"
    )
    p_rec.add_argument("--source-dir", required=True)
    p_rec.add_argument("--output-dir", required=True)
    p_rec.add_argument("--expected-run-id", default="scientific_full_7b_001")

    # list-missing-modal-evidence
    p_list = subparsers.add_parser(
        "list-missing-modal-evidence",
        help="List missing modal evidence files",
    )
    p_list.add_argument("--source-dir", required=True)

    # analyze
    p_analyze = subparsers.add_parser(
        "analyze", help="Run full v0.4.6 analysis pipeline"
    )
    p_analyze.add_argument("--evidence-dir", required=True)
    p_analyze.add_argument("--output-dir", required=True)
    p_analyze.add_argument("--run-id", default="v046_analysis_001")
    p_analyze.add_argument("--repo-root", default=".")
    p_analyze.add_argument("--n-sham-seeds", type=int, default=50)
    p_analyze.add_argument("--n-candidate-seeds", type=int, default=50)
    p_analyze.add_argument("--force", action="store_true")

    # report
    p_report = subparsers.add_parser(
        "report", help="Generate report from existing outputs"
    )
    p_report.add_argument("--output-dir", required=True)
    p_report.add_argument("--run-id", required=True)

    # verify-output
    p_verify = subparsers.add_parser(
        "verify-output", help="Verify all required artifacts present"
    )
    p_verify.add_argument("--output-dir", required=True)
    p_verify.add_argument("--run-id", required=True)

    args = parser.parse_args(argv)

    try:
        if args.command == "validate-fixture":
            return _cmd_validate_fixture(args)
        elif args.command == "validate-evidence":
            return _cmd_validate_evidence(args)
        elif args.command == "scan-evidence-origins":
            return _cmd_scan_evidence_origins(args)
        elif args.command == "detect-cross-origin-duplicates":
            return _cmd_detect_cross_origin_duplicates(args)
        elif args.command == "recover-modal-evidence":
            return _cmd_recover_modal_evidence(args)
        elif args.command == "list-missing-modal-evidence":
            return _cmd_list_missing_modal_evidence(args)
        elif args.command == "analyze":
            return _cmd_analyze(args)
        elif args.command == "report":
            return _cmd_report(args)
        elif args.command == "verify-output":
            return _cmd_verify_output(args)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return EXIT_CONFIG_ERROR
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _cmd_validate_fixture(args) -> int:
    from qualification.autolearn_v04.v046.fixture_validation import validate_fixture

    result = validate_fixture(args.evidence_dir)

    structural = result.get("structural_validity", "FAIL")
    origin = result.get("evidence_origin", "")
    sci_eligible = result.get("scientific_claim_eligibility", "")
    gate_a_eligible = result.get("gate_a_eligible", False)
    promotable = result.get("promotable", False)

    print(f"Structural validity: {structural}")
    print(f"Evidence origin: {origin}")
    print(f"Scientific claim eligibility: {sci_eligible}")
    print(f"Gate A eligible: {gate_a_eligible}")
    print(f"Promotable: {promotable}")
    print(f"Reason: {result.get('reason', '')}")

    checks = result.get("checks", {})
    if isinstance(checks, dict):
        n_checks = len(checks)
        n_fail = sum(1 for c in checks.values() if c.get("status") == "FAIL")
        print(f"Checks: {n_checks} total, {n_fail} failed")

    # Fixture validation may exit 0 while stating scientific_eligibility=false
    if structural == "PASS":
        return EXIT_SUCCESS
    else:
        return EXIT_CONFIG_ERROR


def _cmd_validate_evidence(args) -> int:
    from qualification.autolearn_v04.v046.scientific_evidence_validation import (
        validate_scientific_evidence,
    )

    result = validate_scientific_evidence(args.evidence_dir)

    structural = result.get("structural_completeness", {})
    origin_auth = result.get("origin_authenticity", {})
    scientific = result.get("scientific_eligibility", {})
    gate_a = result.get("gate_a_eligibility", {})

    print(f"Structural completeness: {structural.get('status', 'BLOCKED')}")
    print(f"Origin authenticity: {origin_auth.get('status', 'BLOCKED')}")
    print(f"Scientific eligibility: {scientific.get('status', 'BLOCKED')}")
    print(f"Gate A eligibility: {gate_a.get('status', 'BLOCKED')}")

    sci_status = scientific.get("status", "FAIL")
    if sci_status == "PASS":
        return EXIT_SUCCESS
    else:
        return EXIT_INVALID_SCIENTIFIC_EVIDENCE


def _cmd_scan_evidence_origins(args) -> int:
    from qualification.autolearn_v04.v046.evidence_origin_audit import audit_evidence_origin

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"Root directory does not exist: {args.root}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    # Find all evidence packages (directories with EVIDENCE_MANIFEST.json)
    import os

    packages = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "EVIDENCE_MANIFEST.json" in filenames:
            packages.append(dirpath)

    if not packages:
        print(f"No evidence packages found under {args.root}")
        return EXIT_SUCCESS

    print(f"Found {len(packages)} evidence package(s):")
    print()
    for pkg in sorted(packages):
        result = audit_evidence_origin(pkg)
        print(f"  {pkg}")
        print(f"    Origin: {result.get('origin', '')}")
        print(f"    Status: {result.get('status', '')}")
        print(f"    Scientific claim eligible: {result.get('scientific_claim_eligible', '')}")
        print(f"    Promotable: {result.get('promotable', '')}")
        print(f"    Reason: {result.get('reason', '')}")
        print()

    return EXIT_SUCCESS


def _cmd_detect_cross_origin_duplicates(args) -> int:
    from qualification.autolearn_v04.v046.cross_origin_duplication import (
        detect_cross_origin_duplicates,
    )

    result = detect_cross_origin_duplicates(args.root)

    status = result.get("status", "BLOCKED")
    packages_scanned = result.get("packages_scanned", 0)
    violations = result.get("violations", [])

    print(f"Status: {status}")
    print(f"Packages scanned: {packages_scanned}")
    print(f"Violations: {len(violations)}")

    if violations:
        print()
        for v in violations:
            print(f"  [{v.get('severity', '')}] {v.get('reason', '')}")
            print(f"    Package A: {v.get('package_a', '')} (origin={v.get('origin_a', '')})")
            print(f"    Package B: {v.get('package_b', '')} (origin={v.get('origin_b', '')})")
            print(f"    Matching digests: {v.get('matching_digests', [])}")
            print()

    if status == "FAIL":
        return EXIT_CROSS_ORIGIN_VIOLATION
    return EXIT_SUCCESS


def _cmd_recover_modal_evidence(args) -> int:
    from qualification.autolearn_v04.v046.modal_evidence_recovery import (
        recover_modal_evidence,
    )

    result = recover_modal_evidence(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        expected_run_id=args.expected_run_id,
    )

    status = result.get("status", "MISSING")
    recovered = result.get("recovered_files", {})
    missing = result.get("missing_files", [])

    print(f"Status: {status}")
    print(f"Recovered files: {len(recovered)}")
    print(f"Missing files: {len(missing)}")

    if recovered:
        print()
        print("Recovered:")
        for name, path in sorted(recovered.items()):
            print(f"  {name}: {path}")

    if missing:
        print()
        print("Missing:")
        for name in missing:
            print(f"  - {name}")

    print(f"\nReason: {result.get('reason', '')}")

    if status == "RECOVERED":
        return EXIT_SUCCESS
    else:
        return EXIT_HISTORICAL_UNAVAILABLE


def _cmd_list_missing_modal_evidence(args) -> int:
    from qualification.autolearn_v04.v046.modal_evidence_recovery import (
        REQUIRED_MODAL_FILES,
        _find_file,
    )

    source_path = Path(args.source_dir)
    if not source_path.exists() or not source_path.is_dir():
        print(f"Source directory does not exist: {args.source_dir}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    print("=== Missing Modal Evidence Report ===")
    print()

    missing = []
    found = []
    for canonical_name, patterns in REQUIRED_MODAL_FILES:
        found_path = _find_file(source_path, patterns)
        if found_path is not None:
            found.append((canonical_name, str(found_path)))
        else:
            missing.append(canonical_name)

    print(f"Found: {len(found)} files")
    for name, path in found:
        print(f"  {name}: {path}")

    print()
    print(f"Missing: {len(missing)} files")
    for name in missing:
        print(f"  - {name}")

    if missing:
        return EXIT_HISTORICAL_UNAVAILABLE
    return EXIT_SUCCESS


def _cmd_analyze(args) -> int:
    from qualification.autolearn_v04.v046.orchestrator import run_all_v046_diagnostics

    output_path = Path(args.output_dir) / args.run_id
    if output_path.exists() and not args.force:
        existing = list(output_path.iterdir())
        if existing:
            print(f"Output directory already exists: {output_path}")
            print("Use --force to overwrite.")
            return EXIT_CONFIG_ERROR

    manifest = run_all_v046_diagnostics(
        evidence_dir=args.evidence_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        repo_root=args.repo_root,
        n_sham_seeds=args.n_sham_seeds,
        n_candidate_seeds=args.n_candidate_seeds,
        force=args.force,
    )

    origin = manifest.get("evidence_origin", "")
    status = manifest.get("overall_evidence_eligibility", "FAIL")
    gate_a0 = manifest.get("gate_a0_status", "BLOCKED")
    decision = manifest.get("scale_decision", "BLOCKED")

    print(f"\n{'='*60}")
    print(f"Analysis complete")
    print(f"Evidence origin: {origin}")
    print(f"Evidence eligibility: {status}")
    print(f"Gate A0: {gate_a0}")
    print(f"Scale decision: {decision}")
    print(f"Approve full 14B: {manifest.get('approve_full_14b_run', False)}")
    print(f"Approve matched pilot: {manifest.get('approve_matched_14b_pilot', False)}")
    print(f"{'='*60}")

    # Exit code 0 if pipeline completed, regardless of scientific PASS/FAIL.
    # For synthetic fixtures, the pipeline completing is success.
    if origin == "SYNTHETIC":
        return EXIT_SUCCESS
    if status == "PASS":
        return EXIT_SUCCESS
    return EXIT_INVALID_SCIENTIFIC_EVIDENCE


def _cmd_report(args) -> int:
    output_path = Path(args.output_dir) / args.run_id

    if not output_path.exists():
        print(f"Output directory does not exist: {output_path}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    report_path = output_path / "GATE_A_V046_EVIDENCE_REPAIR_REPORT.md"
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
        print(content)
        return EXIT_SUCCESS

    # Try to regenerate from artifacts
    manifest_path = output_path / "analysis_manifest.json"
    if not manifest_path.exists():
        print(f"Analysis manifest not found: {manifest_path}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    print(
        "Report generation from existing outputs requires running 'analyze' first. "
        "The report file is generated during the analyze stage."
    )
    return EXIT_CONFIG_ERROR


def _cmd_verify_output(args) -> int:
    output_path = Path(args.output_dir) / args.run_id

    # Base required artifacts (always required regardless of origin)
    required = [
        "analysis_manifest.json",
        "analysis_source_provenance.json",
        "analysis_config.json",
        "evidence_origin_audit.json",
        "cross_origin_duplication_report.json",
        "experience_normalization_report.json",
        "normalized_experience_rows.json",
        "evidence_weight_audit.json",
        "verifier_registry_audit.json",
        "counterfactual_equivalence_report.json",
        "action_matrix_completeness.json",
        "split_access_audit.json",
        "candidate_serialization_parity.json",
        "sham_serialization_parity.json",
        "artifact_lineage_report.json",
        "oracle_discrepancy_report.json",
        "historical_identity_report.json",
        "analysis_identity_report.json",
        "cross_version_lineage_report.json",
        "model_identity_consistency.json",
        "task_split_consistency.json",
        "safety_evidence_integrity.json",
        "utility_consistency_report.json",
        "metric_consistency_report.json",
        "gate_a0_v046.json",
        "model_scale_decision.json",
        "final_provenance_manifest.json",
        "artifact_checksums.json",
        "GATE_A_V046_EVIDENCE_REPAIR_REPORT.md",
    ]

    # Determine evidence origin from analysis manifest to add origin-specific
    # artifacts.
    manifest_path = output_path / "analysis_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            origin = str(manifest.get("evidence_origin", "")).upper()
        except (OSError, json.JSONDecodeError):
            origin = ""
    else:
        origin = ""

    # SYNTHETIC evidence produces fixture_validation_report.json;
    # REAL_MODEL evidence produces scientific_evidence_validation.json.
    if origin == "SYNTHETIC":
        required.append("fixture_validation_report.json")
    elif origin == "REAL_MODEL":
        required.append("scientific_evidence_validation.json")
    else:
        # Unknown origin — require both if they exist
        required.append("fixture_validation_report.json")
        required.append("scientific_evidence_validation.json")

    missing = []
    for name in required:
        if not (output_path / name).exists():
            missing.append(name)

    if missing:
        print(f"Missing {len(missing)} artifacts:")
        for m in missing:
            print(f"  - {m}")
        return EXIT_RUNTIME_ERROR
    else:
        print(f"All {len(required)} required artifacts present.")
        return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
