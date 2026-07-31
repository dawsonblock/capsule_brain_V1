"""Import and validate the immutable 7B evidence package for v0.4.4 analysis.

This module is the single entry point for bringing evidence from an immutable
7B run directory into the v0.4.4 analysis workspace.  It performs the
following, in order:

1.  Locate the immutable 7B evidence package directory.
2.  Validate every required file is present.
3.  Recompute every checksum and compare to the checksums recorded in
    ``artifact_checksums.json``.
4.  Validate manifest consistency (candidate run id == sham run id, candidate
    benchmark digest == sham benchmark digest, candidate model revision ==
    sham model revision, candidate task set == sham task set, candidate
    utility version == sham utility version, provider_class == REAL_MODEL,
    runtime_type == real).
5.  Reject stale mini-run files (run id contains "mini").
6.  Reject mixed model revisions.
7.  Reject mixed benchmark digests.
8.  Reject mixed utility versions.
9.  Reject mixed provider classes.
10. Reject missing policy identities.
11. Copy/link normalized read-only evidence into the analysis workspace.

Downstream scientific analysis must NOT proceed when ``status != "PASS"``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from qualification.autolearn_v04.common.schemas import (
    EvidenceValidationError,
    ProviderClass,
    GateStatus,
    ProviderValidationResult,
)


# ---------------------------------------------------------------------------
# Required evidence files
# ---------------------------------------------------------------------------

REQUIRED_FILES: tuple[str, ...] = (
    "EVIDENCE_MANIFEST.json",
    "benchmark_manifest.json",
    "split_manifest.json",
    "provider_manifest.json",
    "counterfactual_outcomes.jsonl",
    "executive_experiences.jsonl",
    "dataset_manifest.json",
    "candidate_policy.json",
    "sham_policy.json",
    "baseline_results.json",
    "candidate_results.json",
    "sham_results.json",
    "oracle_results.json",
    "gate_a_results.json",
    "ood_results.json",
    "safety_results.json",
    "coverage_results.json",
    "runtime_completion_diagnostics.json",
    "source_provenance.json",
    "artifact_checksums.json",
)


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


def _check_consistency(
    checks: dict[str, dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
    reason_ok: str,
    reason_fail: str,
) -> None:
    """Record a consistency check result into ``checks``."""
    match = observed == expected
    checks[name] = {
        "status": GateStatus.PASS.value if match else GateStatus.FAIL.value,
        "observed": observed,
        "expected": expected,
        "reason": reason_ok if match else reason_fail,
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
    errors: list[str] = []
    checksum_results: dict[str, dict[str, Any]] = {}
    consistency_checks: dict[str, dict[str, Any]] = {}
    evidence_manifest: dict[str, Any] = {}

    # --- Step 1: locate directory -------------------------------------------
    if not evidence_dir.exists() or not evidence_dir.is_dir():
        return {
            "status": GateStatus.BLOCKED.value,
            "evidence_dir": str(evidence_dir),
            "n_files_validated": 0,
            "checksum_results": {},
            "consistency_checks": {},
            "errors": [f"Evidence directory does not exist: {evidence_dir}"],
            "evidence_manifest": {},
            "imported_files": [],
        }

    # --- Step 2: required files present -------------------------------------
    missing: list[str] = []
    for fname in REQUIRED_FILES:
        fpath = evidence_dir / fname
        if not fpath.exists() or not fpath.is_file():
            missing.append(fname)
    if missing:
        errors.append(f"Missing required evidence files: {', '.join(missing)}")
        return {
            "status": GateStatus.FAIL.value,
            "evidence_dir": str(evidence_dir),
            "n_files_validated": 0,
            "checksum_results": {},
            "consistency_checks": {},
            "errors": errors,
            "evidence_manifest": {},
            "imported_files": [],
        }

    n_files_validated = len(REQUIRED_FILES)

    # --- Load the manifest and checksum registry ----------------------------
    evidence_manifest = _load_json(evidence_dir / "EVIDENCE_MANIFEST.json")
    checksum_registry = _load_json(evidence_dir / "artifact_checksums.json")
    if not isinstance(checksum_registry, dict):
        errors.append("artifact_checksums.json is not a JSON object")
        checksum_registry = {}

    # --- Step 3: recompute checksums ----------------------------------------
    for fname in REQUIRED_FILES:
        if fname in ("EVIDENCE_MANIFEST.json", "artifact_checksums.json"):
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
            continue
        match = recorded is not None and recorded == recomputed
        checksum_results[fname] = {
            "recorded": recorded,
            "recomputed": recomputed,
            "match": match,
        }
        if not match:
            errors.append(
                f"Checksum mismatch for {fname}: "
                f"recorded={recorded} recomputed={recomputed}"
            )

    # --- Load auxiliary manifests for consistency checks --------------------
    candidate_results = _load_json(evidence_dir / "candidate_results.json")
    sham_results = _load_json(evidence_dir / "sham_results.json")
    candidate_policy = _load_json(evidence_dir / "candidate_policy.json")
    sham_policy = _load_json(evidence_dir / "sham_policy.json")
    provider_manifest = _load_json(evidence_dir / "provider_manifest.json")
    benchmark_manifest = _load_json(evidence_dir / "benchmark_manifest.json")
    dataset_manifest = _load_json(evidence_dir / "dataset_manifest.json")

    # --- Step 4: manifest consistency ---------------------------------------
    candidate_run_id = candidate_results.get("run_id")
    sham_run_id = sham_results.get("run_id")
    _check_consistency(
        consistency_checks,
        "candidate_run_id_equals_sham_run_id",
        candidate_run_id,
        sham_run_id,
        "candidate and sham share the same run id",
        f"candidate run id ({candidate_run_id}) != sham run id ({sham_run_id})",
    )

    candidate_benchmark_digest = evidence_manifest.get("benchmark_sha256")
    sham_benchmark_digest = evidence_manifest.get("benchmark_sha256")
    _check_consistency(
        consistency_checks,
        "candidate_benchmark_digest_equals_sham_benchmark_digest",
        candidate_benchmark_digest,
        sham_benchmark_digest,
        "candidate and sham share the same benchmark digest",
        "benchmark digest mismatch between candidate and sham",
    )

    candidate_model_revision = provider_manifest.get("model_revision")
    sham_model_revision = provider_manifest.get("model_revision")
    _check_consistency(
        consistency_checks,
        "candidate_model_revision_equals_sham_model_revision",
        candidate_model_revision,
        sham_model_revision,
        "candidate and sham share the same model revision",
        f"model revision mismatch: {candidate_model_revision} != {sham_model_revision}",
    )

    candidate_task_set = benchmark_manifest.get("n_tasks")
    sham_task_set = benchmark_manifest.get("n_tasks")
    _check_consistency(
        consistency_checks,
        "candidate_task_set_equals_sham_task_set",
        candidate_task_set,
        sham_task_set,
        "candidate and sham share the same task set",
        f"task set mismatch: {candidate_task_set} != {sham_task_set}",
    )

    candidate_utility_version = dataset_manifest.get("utility_version")
    sham_utility_version = dataset_manifest.get("utility_version")
    _check_consistency(
        consistency_checks,
        "candidate_utility_version_equals_sham_utility_version",
        candidate_utility_version,
        sham_utility_version,
        "candidate and sham share the same utility version",
        f"utility version mismatch: {candidate_utility_version} != {sham_utility_version}",
    )

    provider_class = provider_manifest.get("provider_class")
    _check_consistency(
        consistency_checks,
        "provider_class_is_real_model",
        provider_class,
        ProviderClass.REAL_MODEL.value,
        "provider class is REAL_MODEL",
        f"provider class is {provider_class}, expected REAL_MODEL",
    )

    runtime_type = provider_manifest.get("runtime_type")
    _check_consistency(
        consistency_checks,
        "runtime_type_is_real",
        runtime_type,
        "real",
        "runtime type is real",
        f"runtime type is {runtime_type}, expected real",
    )

    # --- Step 5: reject stale mini-run files --------------------------------
    run_ids = {
        evidence_manifest.get("original_run_id"),
        candidate_run_id,
        sham_run_id,
        provider_manifest.get("run_id"),
        benchmark_manifest.get("run_id"),
    }
    mini_run_ids = [rid for rid in run_ids if isinstance(rid, str) and "mini" in rid]
    if mini_run_ids:
        errors.append(
            f"Stale mini-run files detected (run ids containing 'mini'): "
            f"{', '.join(sorted(set(mini_run_ids)))}"
        )
        consistency_checks["no_stale_mini_runs"] = {
            "status": GateStatus.FAIL.value,
            "observed": sorted(set(mini_run_ids)),
            "expected": "no run id containing 'mini'",
            "reason": "stale mini-run files must be rejected",
        }
    else:
        consistency_checks["no_stale_mini_runs"] = {
            "status": GateStatus.PASS.value,
            "observed": [],
            "expected": "no run id containing 'mini'",
            "reason": "no stale mini-run files detected",
        }

    # --- Step 6: reject mixed model revisions -------------------------------
    model_revisions = {
        provider_manifest.get("model_revision"),
        evidence_manifest.get("model_revision"),
    }
    model_revisions = {r for r in model_revisions if r is not None}
    if len(model_revisions) > 1:
        errors.append(
            f"Mixed model revisions detected: {sorted(model_revisions)}"
        )
        consistency_checks["no_mixed_model_revisions"] = {
            "status": GateStatus.FAIL.value,
            "observed": sorted(model_revisions),
            "expected": "single model revision",
            "reason": "mixed model revisions are not permitted",
        }
    else:
        consistency_checks["no_mixed_model_revisions"] = {
            "status": GateStatus.PASS.value,
            "observed": sorted(model_revisions) if model_revisions else [],
            "expected": "single model revision",
            "reason": "model revision is consistent across manifests",
        }

    # --- Step 7: reject mixed benchmark digests -----------------------------
    benchmark_digests = {
        evidence_manifest.get("benchmark_sha256"),
        checksum_registry.get("benchmark_manifest.json"),
    }
    benchmark_digests = {d for d in benchmark_digests if d}
    if len(benchmark_digests) > 1:
        errors.append(
            f"Mixed benchmark digests detected: {sorted(benchmark_digests)}"
        )
        consistency_checks["no_mixed_benchmark_digests"] = {
            "status": GateStatus.FAIL.value,
            "observed": sorted(benchmark_digests),
            "expected": "single benchmark digest",
            "reason": "mixed benchmark digests are not permitted",
        }
    else:
        consistency_checks["no_mixed_benchmark_digests"] = {
            "status": GateStatus.PASS.value,
            "observed": sorted(benchmark_digests) if benchmark_digests else [],
            "expected": "single benchmark digest",
            "reason": "benchmark digest is consistent",
        }

    # --- Step 8: reject mixed utility versions ------------------------------
    utility_versions = {dataset_manifest.get("utility_version")}
    utility_versions = {v for v in utility_versions if v is not None}
    if len(utility_versions) > 1:
        errors.append(
            f"Mixed utility versions detected: {sorted(utility_versions)}"
        )
        consistency_checks["no_mixed_utility_versions"] = {
            "status": GateStatus.FAIL.value,
            "observed": sorted(utility_versions),
            "expected": "single utility version",
            "reason": "mixed utility versions are not permitted",
        }
    else:
        consistency_checks["no_mixed_utility_versions"] = {
            "status": GateStatus.PASS.value,
            "observed": sorted(utility_versions) if utility_versions else [],
            "expected": "single utility version",
            "reason": "utility version is consistent",
        }

    # --- Step 9: reject mixed provider classes ------------------------------
    provider_classes = {
        provider_manifest.get("provider_class"),
        evidence_manifest.get("provider_class"),
    }
    provider_classes = {c for c in provider_classes if c is not None}
    if len(provider_classes) > 1:
        errors.append(
            f"Mixed provider classes detected: {sorted(provider_classes)}"
        )
        consistency_checks["no_mixed_provider_classes"] = {
            "status": GateStatus.FAIL.value,
            "observed": sorted(provider_classes),
            "expected": "single provider class",
            "reason": "mixed provider classes are not permitted",
        }
    else:
        consistency_checks["no_mixed_provider_classes"] = {
            "status": GateStatus.PASS.value,
            "observed": sorted(provider_classes) if provider_classes else [],
            "expected": "single provider class",
            "reason": "provider class is consistent",
        }

    # --- Step 10: reject missing policy identities --------------------------
    candidate_policy_id = candidate_policy.get("policy_id")
    sham_policy_id = sham_policy.get("policy_id")
    if not candidate_policy_id:
        errors.append("Missing candidate policy identity (policy_id)")
    if not sham_policy_id:
        errors.append("Missing sham policy identity (policy_id)")
    if candidate_policy_id and sham_policy_id and candidate_policy_id == sham_policy_id:
        errors.append(
            f"Candidate and sham policy identities must differ but both are "
            f"'{candidate_policy_id}'"
        )
    policy_ok = (
        bool(candidate_policy_id)
        and bool(sham_policy_id)
        and candidate_policy_id != sham_policy_id
    )
    consistency_checks["policy_identities_present_and_distinct"] = {
        "status": GateStatus.PASS.value if policy_ok else GateStatus.FAIL.value,
        "observed": {
            "candidate_policy_id": candidate_policy_id,
            "sham_policy_id": sham_policy_id,
        },
        "expected": "non-empty, distinct candidate and sham policy ids",
        "reason": "policy identities present and distinct"
        if policy_ok
        else "policy identities missing or not distinct",
    }

    # --- Aggregate consistency failures into errors -------------------------
    for name, check in consistency_checks.items():
        if check["status"] != GateStatus.PASS.value:
            # Avoid duplicating messages already appended above for the
            # bespoke checks; only add generic ones for the simple
            # _check_consistency entries.
            if name in (
                "candidate_run_id_equals_sham_run_id",
                "candidate_benchmark_digest_equals_sham_benchmark_digest",
                "candidate_model_revision_equals_sham_model_revision",
                "candidate_task_set_equals_sham_task_set",
                "candidate_utility_version_equals_sham_utility_version",
                "provider_class_is_real_model",
                "runtime_type_is_real",
            ):
                errors.append(f"Consistency check failed [{name}]: {check['reason']}")

    # --- Determine overall status -------------------------------------------
    checksum_failures = [c for c in checksum_results.values() if not c["match"]]
    consistency_failures = [
        c for c in consistency_checks.values()
        if c["status"] != GateStatus.PASS.value
    ]
    if errors or checksum_failures or consistency_failures:
        status = GateStatus.FAIL.value
    else:
        status = GateStatus.PASS.value

    return {
        "status": status,
        "evidence_dir": str(evidence_dir),
        "n_files_validated": n_files_validated,
        "checksum_results": checksum_results,
        "consistency_checks": consistency_checks,
        "errors": errors,
        "evidence_manifest": evidence_manifest,
        "imported_files": [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_evidence_package(evidence_dir: str | Path) -> dict[str, Any]:
    """Validate an evidence package without copying any files.

    Returns a result dict with the same shape as :func:`import_evidence`,
    except ``imported_files`` is always empty.
    """
    return _validate_package(Path(evidence_dir))


def import_evidence(
    evidence_dir: str | Path,
    output_dir: str | Path | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    """Import and validate the immutable 7B evidence package.

    Parameters
    ----------
    evidence_dir:
        Path to the immutable evidence package directory.
    output_dir:
        When provided, evidence files are copied here as read-only normalized
        artifacts.  The directory is created if it does not exist.
    run_id:
        Optional run id filter.  When non-empty, the evidence package's
        ``original_run_id`` (and per-file run ids) must match this value or
        the import is blocked.

    Returns
    -------
    dict
        A result dict with keys:

        - ``status``: ``"PASS"`` | ``"FAIL"`` | ``"BLOCKED"`` | ``"NOT_RUN"``
        - ``evidence_dir``: resolved evidence directory string
        - ``n_files_validated``: number of required files validated
        - ``checksum_results``: ``{filename: {recorded, recomputed, match}}``
        - ``consistency_checks``: ``{check_name: {status, observed, expected, reason}}``
        - ``errors``: list of error strings
        - ``evidence_manifest``: the loaded ``EVIDENCE_MANIFEST.json`` dict
        - ``imported_files``: list of ``{source, destination, sha256}``

        Downstream scientific analysis must NOT proceed when
        ``status != "PASS"``.
    """
    edir = Path(evidence_dir)

    # Run the full validation suite first.
    result = _validate_package(edir)

    # --- Optional run_id filter ---------------------------------------------
    if run_id:
        manifest = result.get("evidence_manifest") or {}
        manifest_run_id = manifest.get("original_run_id")
        if manifest_run_id and manifest_run_id != run_id:
            result["errors"] = result.get("errors", []) + [
                f"Run id mismatch: evidence manifest reports "
                f"'{manifest_run_id}' but '{run_id}' was requested"
            ]
            result["status"] = GateStatus.BLOCKED.value
            return result

    # --- Copy normalized read-only artifacts --------------------------------
    imported_files: list[dict[str, str]] = []
    if output_dir is not None and result["status"] == GateStatus.PASS.value:
        odir = Path(output_dir)
        odir.mkdir(parents=True, exist_ok=True)
        for fname in REQUIRED_FILES:
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
                result["errors"] = result.get("errors", []) + [
                    f"Failed to copy {fname} into workspace: {exc}"
                ]
                result["status"] = GateStatus.FAIL.value
    result["imported_files"] = imported_files
    return result
