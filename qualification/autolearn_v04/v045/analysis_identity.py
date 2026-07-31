"""Validate the current analysis identity.

This module verifies that the code performing the *current* analysis is the
expected Capsule Brain 2.15.9 / qualification 0.4.5 build, and that the
analysis source tree and configuration have been fingerprinted.

Required checks:
    1.  current package version == 2.15.9
    2.  current qualification version == 0.4.5
    3.  current analysis source-tree hash is present
    4.  current dependency/config digest is present
"""
from __future__ import annotations

from typing import Any

from capsule_brain.version import (
    AUTOLEARN_QUALIFICATION_VERSION,
    AUTOLEARN_VERSION,
    PACKAGE_VERSION,
)

# Expected identity for the current analysis build.
EXPECTED_PACKAGE_VERSION = "2.15.9"
EXPECTED_QUALIFICATION_VERSION = "0.4.5"


def _is_nonempty_str(value: Any) -> bool:
    """Return True when *value* is a non-empty string."""
    return isinstance(value, str) and value.strip() != ""


def _is_valid_sha256(value: Any) -> bool:
    """Return True when *value* looks like a 64-char hex sha256 digest."""
    if not isinstance(value, str):
        return False
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except (TypeError, ValueError):
        return False
    return True


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


def validate_analysis_identity(
    analysis_source_hash: dict,
    config_digest: str,
) -> dict:
    """Validate the current analysis identity.

    Parameters
    ----------
    analysis_source_hash:
        A dict describing the current analysis source tree.  Expected keys
        include ``source_tree_sha256`` (a 64-char hex digest) and optionally
        ``source_file_count`` / ``source_byte_count``.
    config_digest:
        A hex digest (typically sha256) of the dependency/config payload used
        for this analysis run.

    Returns
    -------
    dict
        A result dict with ``status`` ("PASS" | "FAIL"), the analysis package
        and qualification versions, the source-tree hash, the config digest, a
        per-check ``checks`` list, and an ``errors`` list.
    """
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    analysis_source_hash = analysis_source_hash or {}
    source_tree_sha256 = analysis_source_hash.get("source_tree_sha256", "")

    # --- Check 1: current package version == 2.15.9 ------------------------
    if PACKAGE_VERSION == EXPECTED_PACKAGE_VERSION:
        _record(
            "analysis_package_version_current",
            "PASS",
            PACKAGE_VERSION,
            EXPECTED_PACKAGE_VERSION,
            "",
            checks,
        )
    else:
        _record(
            "analysis_package_version_current",
            "FAIL",
            PACKAGE_VERSION,
            EXPECTED_PACKAGE_VERSION,
            f"current PACKAGE_VERSION '{PACKAGE_VERSION}' != expected '{EXPECTED_PACKAGE_VERSION}'",
            checks,
        )
        errors.append(
            f"current PACKAGE_VERSION '{PACKAGE_VERSION}' != expected '{EXPECTED_PACKAGE_VERSION}'"
        )

    # --- Check 2: current qualification version == 0.4.5 -------------------
    if AUTOLEARN_QUALIFICATION_VERSION == EXPECTED_QUALIFICATION_VERSION:
        _record(
            "analysis_qualification_version_current",
            "PASS",
            AUTOLEARN_QUALIFICATION_VERSION,
            EXPECTED_QUALIFICATION_VERSION,
            "",
            checks,
        )
    else:
        _record(
            "analysis_qualification_version_current",
            "FAIL",
            AUTOLEARN_QUALIFICATION_VERSION,
            EXPECTED_QUALIFICATION_VERSION,
            f"current AUTOLEARN_QUALIFICATION_VERSION '{AUTOLEARN_QUALIFICATION_VERSION}' != expected '{EXPECTED_QUALIFICATION_VERSION}'",
            checks,
        )
        errors.append(
            f"current AUTOLEARN_QUALIFICATION_VERSION '{AUTOLEARN_QUALIFICATION_VERSION}' != expected '{EXPECTED_QUALIFICATION_VERSION}'"
        )

    # --- Check 3: analysis source-tree hash present ------------------------
    if _is_valid_sha256(source_tree_sha256):
        _record(
            "analysis_source_tree_sha256_present",
            "PASS",
            source_tree_sha256,
            "64-char hex sha256",
            "",
            checks,
        )
    else:
        _record(
            "analysis_source_tree_sha256_present",
            "FAIL",
            source_tree_sha256,
            "64-char hex sha256",
            "analysis source_tree_sha256 is missing or not a valid 64-char hex digest",
            checks,
        )
        errors.append(
            "analysis source_tree_sha256 is missing or not a valid 64-char hex digest"
        )

    # --- Check 4: dependency/config digest present -------------------------
    if _is_nonempty_str(config_digest):
        _record(
            "analysis_config_digest_present",
            "PASS",
            config_digest,
            "non-empty string",
            "",
            checks,
        )
    else:
        _record(
            "analysis_config_digest_present",
            "FAIL",
            config_digest,
            "non-empty string",
            "config_digest is missing or empty",
            checks,
        )
        errors.append("config_digest is missing or empty")

    # --- Aggregate status --------------------------------------------------
    all_passed = all(c["status"] == "PASS" for c in checks)
    status = "PASS" if all_passed else "FAIL"

    return {
        "status": status,
        "analysis_package_version": PACKAGE_VERSION,
        "analysis_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "analysis_autolearn_version": AUTOLEARN_VERSION,
        "analysis_source_tree_sha256": source_tree_sha256,
        "config_digest": config_digest,
        "checks": checks,
        "errors": errors,
    }
