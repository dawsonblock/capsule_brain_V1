"""Validate cross-version lineage between a historical run and current analysis.

A historical run produced under an earlier Capsule Brain package version (for
example 2.15.5) may legitimately be analyzed under a later version (for example
2.15.9).  Differing versions are valid *when both identities are complete*.

This module verifies that:

    1.  the evidence manifest names the original run
    2.  the analysis manifest names the current analyzer
    3.  original artifact hashes are preserved
    4.  normalization lineage is complete (if normalization occurred)
    5.  versions are different but both identities are complete (valid)

A missing original source identity must produce ``BLOCKED`` rather than
``FAIL`` — the lineage cannot be established at all.  Version differences alone
must never produce ``FAIL``.
"""
from __future__ import annotations

from typing import Any


def _is_nonempty_str(value: Any) -> bool:
    """Return True when *value* is a non-empty string."""
    return isinstance(value, str) and value.strip() != ""


def _record(
    check: str,
    status: str,
    observed: Any,
    expected: Any,
    reason: str,
    checks: list[dict[str, Any]],
) -> None:
    """Append a single check record to *checks*."""
    checks.append(
        {
            "check": check,
            "status": status,
            "observed": observed,
            "expected": expected,
            "reason": reason,
        }
    )


def validate_cross_version_lineage(
    historical_identity: dict,
    analysis_identity: dict,
    evidence_manifest: dict,
) -> dict:
    """Validate cross-version lineage between historical run and current analysis.

    Parameters
    ----------
    historical_identity:
        The result dict returned by
        :func:`historical_identity.validate_historical_identity`.
    analysis_identity:
        The result dict returned by
        :func:`analysis_identity.validate_analysis_identity`.
    evidence_manifest:
        The immutable evidence manifest produced by the original run.  Used to
        confirm that original artifact hashes are preserved and that the
        manifest names the original run.

    Returns
    -------
    dict
        A result dict with ``status`` ("PASS" | "FAIL" | "BLOCKED"),
        ``historical_version``, ``analysis_version``, ``version_gap_valid``,
        ``lineage_complete``, a per-check ``checks`` list, and an ``errors``
        list.
    """
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    historical_identity = historical_identity or {}
    analysis_identity = analysis_identity or {}
    evidence_manifest = evidence_manifest or {}

    historical_status = historical_identity.get("status", "")
    analysis_status = analysis_identity.get("status", "")

    historical_version = historical_identity.get("original_package_version", "")
    analysis_version = analysis_identity.get("analysis_package_version", "")

    # --- If historical identity is BLOCKED, lineage cannot be established ----
    if historical_status == "BLOCKED":
        _record(
            "historical_identity_established",
            "BLOCKED",
            historical_status,
            "PASS or FAIL (complete identity)",
            "historical identity is BLOCKED; original source identity is missing and lineage cannot be established",
            checks,
        )
        errors.append(
            "historical identity is BLOCKED; original source identity is missing and lineage cannot be established"
        )
        return {
            "status": "BLOCKED",
            "historical_version": historical_version,
            "analysis_version": analysis_version,
            "version_gap_valid": False,
            "lineage_complete": False,
            "checks": checks,
            "errors": errors,
        }

    # --- Check 1: evidence manifest names original run ----------------------
    original_run_name = (
        evidence_manifest.get("original_run_name")
        or evidence_manifest.get("run_name")
        or evidence_manifest.get("original_run_id")
        or evidence_manifest.get("run_id")
        or ""
    )
    if _is_nonempty_str(original_run_name):
        _record(
            "evidence_manifest_names_original_run",
            "PASS",
            original_run_name,
            "non-empty string",
            "",
            checks,
        )
    else:
        _record(
            "evidence_manifest_names_original_run",
            "FAIL",
            original_run_name,
            "non-empty string",
            "evidence manifest does not name the original run (original_run_name/run_name missing)",
            checks,
        )
        errors.append(
            "evidence manifest does not name the original run (original_run_name/run_name missing)"
        )

    # --- Check 2: analysis manifest names current analyzer -----------------
    analysis_run_name = (
        analysis_identity.get("analysis_run_name")
        or analysis_identity.get("run_name")
        or analysis_identity.get("run_id")
        or ""
    )
    if _is_nonempty_str(analysis_run_name):
        _record(
            "analysis_manifest_names_current_analyzer",
            "PASS",
            analysis_run_name,
            "non-empty string",
            "",
            checks,
        )
    else:
        _record(
            "analysis_manifest_names_current_analyzer",
            "FAIL",
            analysis_run_name,
            "non-empty string",
            "analysis identity does not name the current analyzer (analysis_run_name/run_name missing)",
            checks,
        )
        errors.append(
            "analysis identity does not name the current analyzer (analysis_run_name/run_name missing)"
        )

    # --- Check 3: original artifact hashes preserved ------------------------
    original_artifact_hashes = evidence_manifest.get("artifact_hashes") or evidence_manifest.get("original_artifact_hashes")
    artifact_checksums = evidence_manifest.get("artifact_checksums")
    if isinstance(original_artifact_hashes, dict) and len(original_artifact_hashes) > 0:
        _record(
            "original_artifact_hashes_preserved",
            "PASS",
            f"{len(original_artifact_hashes)} artifact hashes",
            "non-empty dict of artifact hashes",
            "",
            checks,
        )
    elif isinstance(artifact_checksums, dict) and len(artifact_checksums) > 0:
        _record(
            "original_artifact_hashes_preserved",
            "PASS",
            f"{len(artifact_checksums)} artifact checksums",
            "non-empty dict of artifact checksums",
            "",
            checks,
        )
    elif _is_nonempty_str(original_artifact_hashes) or _is_nonempty_str(artifact_checksums):
        _record(
            "original_artifact_hashes_preserved",
            "PASS",
            original_artifact_hashes or artifact_checksums,
            "non-empty artifact hash reference",
            "",
            checks,
        )
    else:
        _record(
            "original_artifact_hashes_preserved",
            "FAIL",
            original_artifact_hashes,
            "non-empty dict of artifact hashes",
            "evidence manifest does not preserve original artifact hashes",
            checks,
        )
        errors.append("evidence manifest does not preserve original artifact hashes")

    # --- Check 4: normalization lineage complete (if normalization occurred) -
    normalization_applied = evidence_manifest.get("normalization_applied")
    normalization_lineage = (
        evidence_manifest.get("normalization_lineage")
        or evidence_manifest.get("normalization_metadata")
    )
    if normalization_applied is None:
        # No normalization flag present — treat as no normalization occurred.
        _record(
            "normalization_lineage_complete",
            "PASS",
            "not applicable (normalization_applied not set)",
            "complete lineage if normalization occurred",
            "no normalization flag present; lineage check skipped",
            checks,
        )
    elif not normalization_applied:
        _record(
            "normalization_lineage_complete",
            "PASS",
            normalization_applied,
            "complete lineage if normalization occurred",
            "normalization_applied is false; lineage check skipped",
            checks,
        )
    else:
        # Normalization occurred — require a non-empty lineage record.
        if isinstance(normalization_lineage, dict) and len(normalization_lineage) > 0:
            _record(
                "normalization_lineage_complete",
                "PASS",
                f"{len(normalization_lineage)} lineage entries",
                "non-empty normalization lineage",
                "",
                checks,
            )
        elif _is_nonempty_str(normalization_lineage):
            _record(
                "normalization_lineage_complete",
                "PASS",
                normalization_lineage,
                "non-empty normalization lineage",
                "",
                checks,
            )
        else:
            _record(
                "normalization_lineage_complete",
                "FAIL",
                normalization_lineage,
                "non-empty normalization lineage",
                "normalization_applied is true but normalization_lineage is missing or empty",
                checks,
            )
            errors.append(
                "normalization_applied is true but normalization_lineage is missing or empty"
            )

    # --- Check 5: versions differ but both identities complete (valid) -----
    versions_differ = (
        _is_nonempty_str(historical_version)
        and _is_nonempty_str(analysis_version)
        and historical_version != analysis_version
    )
    historical_complete = historical_status == "PASS"
    analysis_complete = analysis_status == "PASS"

    if versions_differ and historical_complete and analysis_complete:
        _record(
            "cross_version_gap_valid",
            "PASS",
            f"{historical_version} -> {analysis_version}",
            "different versions with both identities complete",
            "versions differ but both historical and analysis identities are complete; this is valid",
            checks,
        )
    elif not versions_differ and historical_complete and analysis_complete:
        _record(
            "cross_version_gap_valid",
            "PASS",
            f"{historical_version} -> {analysis_version}",
            "same or different versions with both identities complete",
            "versions match and both identities are complete; this is valid",
            checks,
        )
    elif not historical_complete:
        _record(
            "cross_version_gap_valid",
            "FAIL",
            f"historical={historical_status}",
            "historical identity complete (PASS)",
            "historical identity is not complete; version gap cannot be validated",
            checks,
        )
        errors.append(
            "historical identity is not complete; version gap cannot be validated"
        )
    elif not analysis_complete:
        _record(
            "cross_version_gap_valid",
            "FAIL",
            f"analysis={analysis_status}",
            "analysis identity complete (PASS)",
            "analysis identity is not complete; version gap cannot be validated",
            checks,
        )
        errors.append(
            "analysis identity is not complete; version gap cannot be validated"
        )
    else:
        _record(
            "cross_version_gap_valid",
            "FAIL",
            f"{historical_version} -> {analysis_version}",
            "both identities complete",
            "version gap is not valid",
            checks,
        )
        errors.append("version gap is not valid")

    version_gap_valid = all(
        c["status"] == "PASS"
        for c in checks
        if c["check"] == "cross_version_gap_valid"
    )

    # --- Aggregate status --------------------------------------------------
    all_passed = all(c["status"] == "PASS" for c in checks)
    status = "PASS" if all_passed else "FAIL"
    lineage_complete = all_passed

    return {
        "status": status,
        "historical_version": historical_version,
        "analysis_version": analysis_version,
        "version_gap_valid": version_gap_valid,
        "lineage_complete": lineage_complete,
        "checks": checks,
        "errors": errors,
    }
