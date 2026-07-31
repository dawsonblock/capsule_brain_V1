"""Exact version validation for v0.4.4 evidence analysis.

This module enforces strict version identity across the evidence package,
the source provenance, and the currently-running analysis code.  Version
strings must match *exactly* — nonempty strings are not sufficient.

Required checks:
    1.  evidence package version is supported
    2.  run package version == expected original package version
    3.  current analysis package version == 2.15.8
    4.  current qualification version == 0.4.4
    5.  source manifest records the current analysis version
    6.  original run version matches the immutable evidence manifest
"""
from __future__ import annotations

from typing import Any

from capsule_brain.version import PACKAGE_VERSION, AUTOLEARN_QUALIFICATION_VERSION


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EVIDENCE_PACKAGE_VERSIONS: frozenset[str] = frozenset({"1.0.0"})

EXPECTED_ANALYSIS_PACKAGE_VERSION = "2.15.8"
EXPECTED_ANALYSIS_QUALIFICATION_VERSION = "0.4.4"


def _check(
    name: str,
    observed: Any,
    expected: Any,
    reason: str,
    checks: dict[str, dict[str, Any]],
) -> bool:
    """Record a single check and return whether it passed."""
    passed = observed == expected
    checks[name] = {
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "expected": expected,
        "reason": reason if not passed else "",
    }
    return passed


def validate_versions(
    evidence_manifest: dict[str, Any],
    source_provenance: dict[str, Any],
    expected_original_package_version: str = "2.15.5",
) -> dict[str, Any]:
    """Validate version identity across evidence, provenance, and code.

    Returns a dict with ``status`` ("PASS" or "FAIL"), the original run
    package version, the current analysis package and qualification
    versions, and a per-check breakdown.
    """
    checks: dict[str, dict[str, Any]] = {}

    # --- Extract observed values -------------------------------------------
    evidence_pkg_version = evidence_manifest.get("evidence_package_version", "")
    evidence_original_pkg = evidence_manifest.get("original_package_version", "")
    evidence_original_qual = evidence_manifest.get("original_qualification_version", "")

    source_original_pkg = source_provenance.get("original_package_version", "")
    source_original_qual = source_provenance.get("original_qualification_version", "")
    # The source provenance must record the version of the code currently
    # performing the analysis (not the original run version).
    source_analysis_pkg = (
        source_provenance.get("analysis_package_version")
        or source_provenance.get("package_version")
        or ""
    )

    # --- Check 1: evidence package version is supported --------------------
    _check(
        "evidence_package_version_supported",
        evidence_pkg_version,
        True,  # expected is "is supported"
        f"evidence_package_version '{evidence_pkg_version}' is not in "
        f"supported set {sorted(SUPPORTED_EVIDENCE_PACKAGE_VERSIONS)}",
        checks,
    )
    # Override with proper expected value for readability.
    checks["evidence_package_version_supported"]["expected"] = (
        f"one of {sorted(SUPPORTED_EVIDENCE_PACKAGE_VERSIONS)}"
    )
    if evidence_pkg_version not in SUPPORTED_EVIDENCE_PACKAGE_VERSIONS:
        checks["evidence_package_version_supported"]["status"] = "FAIL"
    else:
        checks["evidence_package_version_supported"]["status"] = "PASS"
        checks["evidence_package_version_supported"]["reason"] = ""

    # --- Check 2: run package version == expected original -----------------
    _check(
        "run_package_version_matches_expected",
        evidence_original_pkg,
        expected_original_package_version,
        f"evidence original_package_version '{evidence_original_pkg}' != "
        f"expected '{expected_original_package_version}'",
        checks,
    )

    # --- Check 3: current analysis package version == 2.15.8 ---------------
    _check(
        "analysis_package_version_current",
        PACKAGE_VERSION,
        EXPECTED_ANALYSIS_PACKAGE_VERSION,
        f"current PACKAGE_VERSION '{PACKAGE_VERSION}' != "
        f"expected '{EXPECTED_ANALYSIS_PACKAGE_VERSION}'",
        checks,
    )

    # --- Check 4: current qualification version == 0.4.4 -------------------
    _check(
        "analysis_qualification_version_current",
        AUTOLEARN_QUALIFICATION_VERSION,
        EXPECTED_ANALYSIS_QUALIFICATION_VERSION,
        f"current AUTOLEARN_QUALIFICATION_VERSION "
        f"'{AUTOLEARN_QUALIFICATION_VERSION}' != "
        f"expected '{EXPECTED_ANALYSIS_QUALIFICATION_VERSION}'",
        checks,
    )

    # --- Check 5: source manifest matches current version ------------------
    _check(
        "source_manifest_matches_current_version",
        source_analysis_pkg,
        PACKAGE_VERSION,
        f"source_provenance analysis_package_version '{source_analysis_pkg}' "
        f"!= current PACKAGE_VERSION '{PACKAGE_VERSION}'",
        checks,
    )

    # --- Check 6: original run version matches immutable evidence manifest --
    _check(
        "original_run_version_matches_evidence_manifest",
        source_original_pkg,
        evidence_original_pkg,
        f"source_provenance original_package_version '{source_original_pkg}' "
        f"!= evidence_manifest original_package_version "
        f"'{evidence_original_pkg}'",
        checks,
    )

    # Also verify qualification version consistency between source and evidence.
    _check(
        "original_qualification_version_matches_evidence_manifest",
        source_original_qual,
        evidence_original_qual,
        f"source_provenance original_qualification_version "
        f"'{source_original_qual}' != evidence_manifest "
        f"original_qualification_version '{evidence_original_qual}'",
        checks,
    )

    # --- Aggregate status ---------------------------------------------------
    all_passed = all(c["status"] == "PASS" for c in checks.values())
    status = "PASS" if all_passed else "FAIL"

    return {
        "status": status,
        "original_run_package_version": evidence_original_pkg,
        "analysis_package_version": PACKAGE_VERSION,
        "analysis_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "checks": checks,
    }
