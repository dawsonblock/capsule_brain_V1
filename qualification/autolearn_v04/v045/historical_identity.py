"""Validate the historical execution identity of the original scientific run.

A historical run produced under an earlier Capsule Brain package version (for
example 2.15.5) must carry enough provenance to establish *which* code produced
its evidence.  This module verifies that the evidence manifest and source
provenance together record:

    1.  the original package version
    2.  the original qualification version
    3.  a source-tree hash (sha256) of the original source tree, *or* a commit
        SHA identifying it
    4.  the original provider/model identity

If ``commit_sha`` is null but ``source_tree_sha256`` is valid and
``source_file_count`` is greater than zero, the identity is still acceptable —
the source-tree hash is a sufficient fingerprint on its own.  If
``commit_sha`` is the literal string ``"unknown"`` and no source-tree hash is
present, the run is ``BLOCKED`` because its origin cannot be established.
"""
from __future__ import annotations

from typing import Any

# Sentinel value indicating the commit SHA was never recorded.
_UNKNOWN_COMMIT = "unknown"


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


def validate_historical_identity(
    evidence_manifest: dict,
    source_provenance: dict,
) -> dict:
    """Validate the historical execution identity of the original run.

    Parameters
    ----------
    evidence_manifest:
        The immutable evidence manifest produced by the original run.  Expected
        keys include ``original_package_version``,
        ``original_qualification_version``, and ``provider_model_identity``.
    source_provenance:
        The source provenance record describing the original source tree.
        Expected keys include ``commit_sha``, ``source_tree_sha256``,
        ``source_file_count``, and ``source_byte_count``.

    Returns
    -------
    dict
        A result dict with ``status`` ("PASS" | "FAIL" | "BLOCKED"), the
        extracted original identity fields, a per-check ``checks`` list, and an
        ``errors`` list.
    """
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    # --- Extract observed values -------------------------------------------
    original_package_version = evidence_manifest.get(
        "original_package_version"
    ) or source_provenance.get("original_package_version") or ""
    original_qualification_version = evidence_manifest.get(
        "original_qualification_version"
    ) or source_provenance.get("original_qualification_version") or ""
    original_commit_sha = source_provenance.get("commit_sha")
    if original_commit_sha is None:
        original_commit_sha = evidence_manifest.get("commit_sha")
    original_source_tree_sha256 = (
        source_provenance.get("source_tree_sha256")
        or evidence_manifest.get("source_tree_sha256")
        or ""
    )
    original_source_file_count = (
        source_provenance.get("source_file_count")
        if source_provenance.get("source_file_count") is not None
        else evidence_manifest.get("source_file_count")
    )
    if original_source_file_count is None:
        original_source_file_count = 0
    original_source_byte_count = (
        source_provenance.get("source_byte_count")
        if source_provenance.get("source_byte_count") is not None
        else evidence_manifest.get("source_byte_count")
    )
    if original_source_byte_count is None:
        original_source_byte_count = 0
    provider_model_identity = (
        evidence_manifest.get("provider_model_identity")
        or source_provenance.get("provider_model_identity")
        or evidence_manifest.get("provider_model")
        or source_provenance.get("provider_model")
        or ""
    )

    blocked = False

    # --- Check 1: original package version present -------------------------
    if _is_nonempty_str(original_package_version):
        _record(
            "original_package_version_present",
            "PASS",
            original_package_version,
            "non-empty string",
            "",
            checks,
        )
    else:
        _record(
            "original_package_version_present",
            "FAIL",
            original_package_version,
            "non-empty string",
            "original_package_version is missing or empty",
            checks,
        )
        errors.append("original_package_version is missing or empty")

    # --- Check 2: original qualification version present -------------------
    if _is_nonempty_str(original_qualification_version):
        _record(
            "original_qualification_version_present",
            "PASS",
            original_qualification_version,
            "non-empty string",
            "",
            checks,
        )
    else:
        _record(
            "original_qualification_version_present",
            "FAIL",
            original_qualification_version,
            "non-empty string",
            "original_qualification_version is missing or empty",
            checks,
        )
        errors.append("original_qualification_version is missing or empty")

    # --- Check 3: source-tree hash present (or commit SHA) -----------------
    source_tree_valid = _is_valid_sha256(original_source_tree_sha256)
    commit_is_null = original_commit_sha is None
    commit_is_unknown = isinstance(original_commit_sha, str) and original_commit_sha.strip() == _UNKNOWN_COMMIT
    commit_is_valid = (
        _is_nonempty_str(original_commit_sha) and not commit_is_unknown
    )

    if source_tree_valid:
        _record(
            "original_source_tree_sha256_present",
            "PASS",
            original_source_tree_sha256,
            "64-char hex sha256",
            "",
            checks,
        )
    elif commit_is_valid:
        _record(
            "original_source_tree_sha256_present",
            "PASS",
            original_source_tree_sha256,
            "64-char hex sha256 or valid commit_sha",
            "source_tree_sha256 missing but commit_sha is present",
            checks,
        )
    else:
        _record(
            "original_source_tree_sha256_present",
            "FAIL",
            original_source_tree_sha256,
            "64-char hex sha256 or valid commit_sha",
            "source_tree_sha256 is missing/invalid and no valid commit_sha is available",
            checks,
        )
        errors.append(
            "source_tree_sha256 is missing/invalid and no valid commit_sha is available"
        )

    # --- Check 4: original commit SHA (may be null) ------------------------
    if commit_is_null:
        # null is acceptable only when source-tree hash is valid and files > 0.
        if source_tree_valid and original_source_file_count > 0:
            _record(
                "original_commit_sha_present",
                "PASS",
                original_commit_sha,
                "non-empty string or null (with valid source-tree hash)",
                "commit_sha is null but source_tree_sha256 is valid and source_file_count > 0",
                checks,
            )
        else:
            _record(
                "original_commit_sha_present",
                "FAIL",
                original_commit_sha,
                "non-empty string or null (with valid source-tree hash)",
                "commit_sha is null and source-tree hash is missing/invalid or source_file_count is 0",
                checks,
            )
            errors.append(
                "commit_sha is null and source-tree hash is missing/invalid or source_file_count is 0"
            )
    elif commit_is_unknown:
        # "unknown" with no source-tree hash => BLOCKED.
        if source_tree_valid and original_source_file_count > 0:
            _record(
                "original_commit_sha_present",
                "PASS",
                original_commit_sha,
                "non-empty string or valid source-tree hash",
                "commit_sha is 'unknown' but source_tree_sha256 is valid and source_file_count > 0",
                checks,
            )
        else:
            _record(
                "original_commit_sha_present",
                "BLOCKED",
                original_commit_sha,
                "non-empty string or valid source-tree hash",
                "commit_sha is 'unknown' and no valid source_tree_sha256 is present; origin cannot be established",
                checks,
            )
            errors.append(
                "commit_sha is 'unknown' and no valid source_tree_sha256 is present; origin cannot be established"
            )
            blocked = True
    elif commit_is_valid:
        _record(
            "original_commit_sha_present",
            "PASS",
            original_commit_sha,
            "non-empty string",
            "",
            checks,
        )
    else:
        _record(
            "original_commit_sha_present",
            "FAIL",
            original_commit_sha,
            "non-empty string",
            "commit_sha is present but not a valid non-empty string",
            checks,
        )
        errors.append("commit_sha is present but not a valid non-empty string")

    # --- Check 5: provider/model identity present --------------------------
    if _is_nonempty_str(provider_model_identity):
        _record(
            "original_provider_model_identity_present",
            "PASS",
            provider_model_identity,
            "non-empty string",
            "",
            checks,
        )
    else:
        _record(
            "original_provider_model_identity_present",
            "FAIL",
            provider_model_identity,
            "non-empty string",
            "provider_model_identity is missing or empty",
            checks,
        )
        errors.append("provider_model_identity is missing or empty")

    # --- Aggregate status --------------------------------------------------
    if blocked:
        status = "BLOCKED"
    elif all(c["status"] == "PASS" for c in checks):
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "status": status,
        "original_package_version": original_package_version,
        "original_qualification_version": original_qualification_version,
        "original_commit_sha": original_commit_sha,
        "original_source_tree_sha256": original_source_tree_sha256,
        "original_source_file_count": original_source_file_count,
        "original_source_byte_count": original_source_byte_count,
        "checks": checks,
        "errors": errors,
    }
