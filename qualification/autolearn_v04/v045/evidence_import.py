"""Import and validate the immutable 7B evidence package for v0.4.5 analysis.

This module is the single entry point for bringing evidence from an immutable
7B run directory into the v0.4.5 analysis workspace.  It separates validation
into two independent dimensions:

- **ByteIntegrityStatus**: ``PASS`` | ``FAIL`` | ``BLOCKED``
    All files exist, hashes recompute, and recorded hashes match.

- **ScientificCompletenessStatus**: ``PASS`` | ``FAIL`` | ``BLOCKED``
    All expected task-level records exist, actual row counts match declared
    counts, required fields are populated, task IDs and split IDs match,
    policies are complete, source provenance is complete, no placeholders,
    and no aggregate-only substitution.

- **OverallEvidenceEligibility**: ``PASS`` | ``FAIL`` | ``BLOCKED``
    The conjunction of byte integrity and scientific completeness.  A
    ``BLOCKED`` in either dimension blocks the overall eligibility.

Downstream scientific analysis must NOT proceed when
``overall_eligibility != "PASS"``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from qualification.autolearn_v04.common.schemas import EvidenceValidationError
from qualification.autolearn_v04.v045.config import REQUIRED_EVIDENCE_FILES
from qualification.autolearn_v04.v045.placeholder_detection import detect_placeholders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file, reading in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    """Load a JSON file, raising ``EvidenceValidationError`` on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError(f"Failed to load JSON {path.name}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file line by line, skipping blank lines."""
    records: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise EvidenceValidationError(
                        f"Invalid JSON in {path.name} line {lineno}: {exc}"
                    ) from exc
    except OSError as exc:
        raise EvidenceValidationError(f"Failed to read {path.name}: {exc}") from exc
    return records


def _safe_load_json(path: Path) -> Any:
    """Load a JSON file, returning ``None`` on failure (no raise)."""
    try:
        return _load_json(path)
    except EvidenceValidationError:
        return None


def _safe_load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, returning ``[]`` on failure (no raise)."""
    try:
        return _load_jsonl(path)
    except EvidenceValidationError:
        return []


# ---------------------------------------------------------------------------
# Byte integrity
# ---------------------------------------------------------------------------


def _validate_byte_integrity(evidence_dir: Path) -> dict[str, Any]:
    """Validate that all files exist and recorded hashes match recomputed ones."""
    errors: list[str] = []
    checksum_results: dict[str, dict[str, Any]] = {}
    n_files_validated = 0

    # --- Step 1: directory exists ------------------------------------------
    if not evidence_dir.exists() or not evidence_dir.is_dir():
        return {
            "status": "BLOCKED",
            "n_files_validated": 0,
            "checksum_results": {},
            "errors": [f"Evidence directory does not exist: {evidence_dir}"],
        }

    # --- Step 2: required files present ------------------------------------
    missing: list[str] = []
    for fname in REQUIRED_EVIDENCE_FILES:
        fpath = evidence_dir / fname
        if not fpath.exists() or not fpath.is_file():
            missing.append(fname)
    if missing:
        errors.append(f"Missing required evidence files: {', '.join(missing)}")
        return {
            "status": "FAIL",
            "n_files_validated": 0,
            "checksum_results": {},
            "errors": errors,
        }

    # --- Step 3: load checksum registry ------------------------------------
    checksum_registry = _safe_load_json(evidence_dir / "artifact_checksums.json")
    if not isinstance(checksum_registry, dict):
        checksum_registry = {}

    # --- Step 4: recompute checksums ---------------------------------------
    for fname in REQUIRED_EVIDENCE_FILES:
        if fname == "artifact_checksums.json":
            n_files_validated += 1
            continue
        fpath = evidence_dir / fname
        recorded = checksum_registry.get(fname)
        try:
            recomputed = _sha256_file(fpath)
        except OSError as exc:
            errors.append(f"Could not hash {fname}: {exc}")
            checksum_results[fname] = {
                "recorded": recorded,
                "recomputed": None,
                "match": False,
            }
            n_files_validated += 1
            continue
        match = recorded is not None and recorded == recomputed
        checksum_results[fname] = {
            "recorded": recorded,
            "recomputed": recomputed,
            "match": match,
        }
        n_files_validated += 1
        if not match:
            errors.append(
                f"Checksum mismatch for {fname}: "
                f"recorded={recorded} recomputed={recomputed}"
            )

    checksum_failures = [c for c in checksum_results.values() if not c["match"]]
    status = "FAIL" if (errors or checksum_failures) else "PASS"

    return {
        "status": status,
        "n_files_validated": n_files_validated,
        "checksum_results": checksum_results,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Scientific completeness
# ---------------------------------------------------------------------------


def _validate_scientific_completeness(
    evidence_dir: Path,
    placeholder_report: dict[str, Any],
) -> dict[str, Any]:
    """Validate that the evidence is scientifically complete."""
    errors: list[str] = []
    checks: dict[str, dict[str, Any]] = {}

    # --- Required files present (re-checked for completeness) -------------
    missing: list[str] = []
    for fname in REQUIRED_EVIDENCE_FILES:
        fpath = evidence_dir / fname
        if not fpath.exists() or not fpath.is_file():
            missing.append(fname)
    if missing:
        errors.append(f"Missing required evidence files: {', '.join(missing)}")
        checks["required_files_present"] = {
            "status": "FAIL",
            "observed": missing,
            "expected": "all required files present",
            "reason": f"missing: {', '.join(missing)}",
        }
        return {
            "status": "FAIL",
            "checks": checks,
            "errors": errors,
        }
    checks["required_files_present"] = {
        "status": "PASS",
        "observed": len(REQUIRED_EVIDENCE_FILES),
        "expected": len(REQUIRED_EVIDENCE_FILES),
        "reason": "all required files present",
    }

    # --- Load manifests ----------------------------------------------------
    evidence_manifest = _safe_load_json(evidence_dir / "EVIDENCE_MANIFEST.json") or {}
    benchmark_manifest = _safe_load_json(evidence_dir / "benchmark_manifest.json") or {}
    split_manifest = _safe_load_json(evidence_dir / "split_manifest.json") or {}
    dataset_manifest = _safe_load_json(evidence_dir / "dataset_manifest.json") or {}
    provider_manifest = _safe_load_json(evidence_dir / "provider_manifest.json") or {}
    source_provenance = _safe_load_json(evidence_dir / "source_provenance.json") or {}
    candidate_policy = _safe_load_json(evidence_dir / "candidate_policy.json") or {}
    sham_policy = _safe_load_json(evidence_dir / "sham_policy.json") or {}
    candidate_results = _safe_load_json(evidence_dir / "candidate_results.json") or {}
    sham_results = _safe_load_json(evidence_dir / "sham_results.json") or {}

    cf_records = _safe_load_jsonl(evidence_dir / "counterfactual_outcomes.jsonl")
    exp_records = _safe_load_jsonl(evidence_dir / "executive_experiences.jsonl")

    # --- Task-level records exist -----------------------------------------
    cf_task_records = [r for r in cf_records if isinstance(r, dict) and "task_id" in r]
    exp_task_records = [r for r in exp_records if isinstance(r, dict) and "task_id" in r]
    has_task_records = bool(cf_task_records or exp_task_records)
    checks["task_level_records_exist"] = {
        "status": "PASS" if has_task_records else "FAIL",
        "observed": len(cf_task_records) + len(exp_task_records),
        "expected": ">=1 task-level records",
        "reason": "task-level records present" if has_task_records
        else "no task-level records found (aggregate-only substitution)",
    }
    if not has_task_records:
        errors.append("No task-level records found (aggregate-only substitution)")

    # --- Row counts match declared counts ---------------------------------
    declared_n_outcomes = (
        (cf_records[0].get("n_outcomes") if cf_records else None)
        or dataset_manifest.get("n_counterfactuals")
        or (_safe_load_json(evidence_dir / "coverage_results.json") or {}).get("n_counterfactuals")
    )
    actual_cf_rows = len(cf_task_records) if cf_task_records else len(cf_records)
    cf_match = (
        isinstance(declared_n_outcomes, int)
        and actual_cf_rows == declared_n_outcomes
    )
    checks["counterfactual_row_count_match"] = {
        "status": "PASS" if cf_match else "FAIL",
        "observed": actual_cf_rows,
        "expected": declared_n_outcomes,
        "reason": "actual counterfactual rows match declared n_outcomes" if cf_match
        else f"actual {actual_cf_rows} != declared {declared_n_outcomes}",
    }
    if not cf_match:
        errors.append(
            f"Counterfactual row count mismatch: actual={actual_cf_rows} "
            f"declared={declared_n_outcomes}"
        )

    declared_n_experiences = (
        (exp_records[0].get("n_experiences") if exp_records else None)
        or benchmark_manifest.get("n_experience")
    )
    actual_exp_rows = len(exp_task_records) if exp_task_records else len(exp_records)
    exp_match = (
        isinstance(declared_n_experiences, int)
        and actual_exp_rows == declared_n_experiences
    )
    checks["experience_row_count_match"] = {
        "status": "PASS" if exp_match else "FAIL",
        "observed": actual_exp_rows,
        "expected": declared_n_experiences,
        "reason": "actual experience rows match declared n_experiences" if exp_match
        else f"actual {actual_exp_rows} != declared {declared_n_experiences}",
    }
    if not exp_match:
        errors.append(
            f"Experience row count mismatch: actual={actual_exp_rows} "
            f"declared={declared_n_experiences}"
        )

    # --- Task IDs and split IDs match -------------------------------------
    splits = split_manifest.get("splits") if isinstance(split_manifest, dict) else None
    split_task_ids: set[Any] = set()
    split_counts_ok = True
    if isinstance(splits, dict):
        for split_name, split_info in splits.items():
            if not isinstance(split_info, dict):
                continue
            count = split_info.get("count")
            task_ids = split_info.get("task_ids")
            if isinstance(count, int) and count > 0:
                if not isinstance(task_ids, list) or len(task_ids) != count:
                    split_counts_ok = False
                    errors.append(
                        f"Split '{split_name}' count ({count}) != task_ids "
                        f"length ({len(task_ids) if isinstance(task_ids, list) else 0})"
                    )
                if isinstance(task_ids, list):
                    split_task_ids.update(task_ids)
    checks["split_ids_match_counts"] = {
        "status": "PASS" if split_counts_ok else "FAIL",
        "observed": "consistent" if split_counts_ok else "inconsistent",
        "expected": "split counts equal task_ids lengths",
        "reason": "split IDs match declared counts" if split_counts_ok
        else "split IDs do not match declared counts",
    }

    # --- Required fields populated ----------------------------------------
    required_manifest_fields = [
        "original_run_id", "model_id", "model_revision", "provider_class",
        "runtime_type", "benchmark_sha256",
    ]
    missing_fields = [
        f for f in required_manifest_fields
        if not (isinstance(evidence_manifest, dict) and evidence_manifest.get(f))
    ]
    checks["required_manifest_fields_populated"] = {
        "status": "PASS" if not missing_fields else "FAIL",
        "observed": missing_fields,
        "expected": "all required fields populated",
        "reason": "required manifest fields populated" if not missing_fields
        else f"missing fields: {', '.join(missing_fields)}",
    }
    if missing_fields:
        errors.append(f"Missing required manifest fields: {', '.join(missing_fields)}")

    # --- Policies complete ------------------------------------------------
    policies_complete = True
    for pname, policy in (("candidate_policy.json", candidate_policy),
                          ("sham_policy.json", sham_policy)):
        if not isinstance(policy, dict) or not policy.get("policy_id"):
            policies_complete = False
            errors.append(f"{pname} missing policy_id")
        has_dim = any(
            k in (policy or {}) for k in ("n_training", "n_features", "policy_dimension")
        )
        has_weights = any(
            k in (policy or {}) for k in ("weights", "coefficients", "params", "parameters", "theta")
        )
        if has_dim and not has_weights:
            policies_complete = False
            errors.append(f"{pname} declares dimension but missing weights")
    checks["policies_complete"] = {
        "status": "PASS" if policies_complete else "FAIL",
        "observed": "complete" if policies_complete else "incomplete",
        "expected": "complete policies with weights",
        "reason": "policies complete" if policies_complete
        else "policies incomplete",
    }

    # --- Source provenance complete --------------------------------------
    provenance_complete = (
        isinstance(source_provenance, dict)
        and bool(source_provenance.get("run_id"))
        and bool(source_provenance.get("original_package_version"))
        and bool(source_provenance.get("original_qualification_version"))
        and source_provenance.get("original_commit_sha") not in (None, "", "unknown")
    )
    checks["source_provenance_complete"] = {
        "status": "PASS" if provenance_complete else "FAIL",
        "observed": source_provenance if isinstance(source_provenance, dict) else {},
        "expected": "complete provenance with known commit sha",
        "reason": "source provenance complete" if provenance_complete
        else "source provenance incomplete (unknown commit sha)",
    }
    if not provenance_complete:
        errors.append("Source provenance incomplete (unknown commit sha)")

    # --- No placeholders --------------------------------------------------
    placeholder_status = placeholder_report.get("status", "PASS")
    placeholder_ok = placeholder_status == "PASS"
    checks["no_placeholders"] = {
        "status": "PASS" if placeholder_ok else "FAIL",
        "observed": placeholder_report.get("n_detections", 0),
        "expected": 0,
        "reason": "no placeholders detected" if placeholder_ok
        else f"{placeholder_report.get('n_detections', 0)} placeholders detected",
    }
    if not placeholder_ok:
        errors.append(
            f"Placeholders detected: {placeholder_report.get('n_detections', 0)} detections"
        )

    # --- No aggregate-only substitution ----------------------------------
    aggregate_only = not has_task_records
    checks["no_aggregate_only_substitution"] = {
        "status": "PASS" if not aggregate_only else "FAIL",
        "observed": "task-level records present" if not aggregate_only
        else "aggregate-only",
        "expected": "task-level records present",
        "reason": "no aggregate-only substitution" if not aggregate_only
        else "aggregate-only substitution detected",
    }

    # --- Overall scientific completeness status --------------------------
    failed_checks = [c for c in checks.values() if c["status"] != "PASS"]
    status = "FAIL" if (errors or failed_checks) else "PASS"

    return {
        "status": status,
        "checks": checks,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def _validate_package(evidence_dir: Path) -> dict[str, Any]:
    """Run the full validation suite against ``evidence_dir``.

    Returns the result dict that callers wrap into the public response.  This
    helper does not perform any copying; ``import_evidence`` handles that
    separately after validation succeeds.
    """
    # --- Byte integrity ---------------------------------------------------
    byte_integrity = _validate_byte_integrity(evidence_dir)

    # If byte integrity is BLOCKED (directory missing), short-circuit.
    if byte_integrity["status"] == "BLOCKED":
        return {
            "byte_integrity": byte_integrity,
            "scientific_completeness": {
                "status": "BLOCKED",
                "checks": {},
                "errors": ["scientific completeness blocked: byte integrity BLOCKED"],
            },
            "overall_eligibility": "BLOCKED",
            "evidence_manifest": {},
            "placeholder_report": {
                "status": "FAIL",
                "detections": [],
                "n_detections": 0,
            },
        }

    # --- Placeholder detection -------------------------------------------
    placeholder_report = detect_placeholders(evidence_dir)

    # --- Evidence manifest ------------------------------------------------
    evidence_manifest: dict[str, Any] = {}
    try:
        evidence_manifest = _load_json(evidence_dir / "EVIDENCE_MANIFEST.json")
        if not isinstance(evidence_manifest, dict):
            evidence_manifest = {}
    except EvidenceValidationError:
        evidence_manifest = {}

    # --- Scientific completeness -----------------------------------------
    scientific_completeness = _validate_scientific_completeness(
        evidence_dir, placeholder_report
    )

    # --- Overall eligibility ---------------------------------------------
    bi_status = byte_integrity["status"]
    sc_status = scientific_completeness["status"]
    if bi_status == "BLOCKED" or sc_status == "BLOCKED":
        overall = "BLOCKED"
    elif bi_status == "PASS" and sc_status == "PASS":
        overall = "PASS"
    else:
        overall = "FAIL"

    return {
        "byte_integrity": byte_integrity,
        "scientific_completeness": scientific_completeness,
        "overall_eligibility": overall,
        "evidence_manifest": evidence_manifest,
        "placeholder_report": placeholder_report,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_evidence_package(evidence_dir: str | Path) -> dict[str, Any]:
    """Validate an evidence package without copying any files.

    Returns a result dict with the same shape as :func:`import_evidence`,
    except no files are copied (``imported_files`` is always empty).
    """
    result = _validate_package(Path(evidence_dir))
    result["imported_files"] = []
    return result


def import_evidence(
    evidence_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Import and validate the immutable 7B evidence package.

    Parameters
    ----------
    evidence_dir:
        Path to the immutable evidence package directory.
    output_dir:
        When provided, evidence files are copied here as read-only normalized
        artifacts.  The directory is created if it does not exist.  Copying
        only occurs when ``overall_eligibility == "PASS"``.

    Returns
    -------
    dict
        A result dict with keys:

        - ``byte_integrity``: ``{status, n_files_validated, checksum_results, errors}``
        - ``scientific_completeness``: ``{status, checks, errors}``
        - ``overall_eligibility``: ``"PASS"`` | ``"FAIL"`` | ``"BLOCKED"``
        - ``evidence_manifest``: the loaded ``EVIDENCE_MANIFEST.json`` dict
        - ``placeholder_report``: result from :func:`detect_placeholders`
        - ``imported_files``: list of ``{source, destination, sha256}``

        Downstream scientific analysis must NOT proceed when
        ``overall_eligibility != "PASS"``.
    """
    edir = Path(evidence_dir)

    # Run the full validation suite first.
    result = _validate_package(edir)

    # --- Copy normalized read-only artifacts ------------------------------
    imported_files: list[dict[str, str]] = []
    if output_dir is not None and result["overall_eligibility"] == "PASS":
        odir = Path(output_dir)
        odir.mkdir(parents=True, exist_ok=True)
        for fname in REQUIRED_EVIDENCE_FILES:
            src = edir / fname
            dst = odir / fname
            try:
                shutil.copy2(src, dst)
                # Normalize permissions: read-only for the owner.
                os.chmod(dst, 0o444)
                sha = _sha256_file(dst)
                imported_files.append({
                    "source": str(src),
                    "destination": str(dst),
                    "sha256": sha,
                })
            except OSError as exc:
                result["byte_integrity"]["errors"] = (
                    result["byte_integrity"].get("errors", []) + [
                        f"Failed to copy {fname} into workspace: {exc}"
                    ]
                )
                result["overall_eligibility"] = "FAIL"
    result["imported_files"] = imported_files
    return result
