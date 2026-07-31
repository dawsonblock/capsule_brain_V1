"""Fixture-specific validation for v0.4.6.

This module validates the synthetic fixture evidence package.  It must
NOT fail simply because strings contain "synthetic", "fixture", or
"simulated" — those are expected in fixture evidence.

The fixture is structurally valid when:
  - all required evidence files exist
  - row counts match declared counts
  - split consistency holds
  - the action matrix is complete (every task has all eligible actions)
  - policy schema is valid
  - safety rows are present
  - checksum integrity holds
  - origin is classified as SYNTHETIC
  - the fixture is non-promotable

Because the fixture is SYNTHETIC, scientific_claim_eligibility is
NOT_APPLICABLE, gate_a_eligible is False, and promotable is False.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.common.evidence_origin import (
    EvidenceOrigin,
    classify_evidence_origin,
    check_anti_misclassification,
)
from qualification.autolearn_v04.v045.config import REQUIRED_EVIDENCE_FILES
from qualification.autolearn_v04.v046.config import (
    CANONICAL_ACTIONS,
    PLACEHOLDER_MARKERS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _safe_load_json(path: Path) -> Any:
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _safe_load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return _load_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return []


def _check(
    checks: dict[str, dict[str, Any]],
    name: str,
    status: str,
    observed: Any,
    expected: Any,
    reason: str,
) -> None:
    checks[name] = {
        "status": status,
        "observed": observed,
        "expected": expected,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_required_files(evidence_dir: Path, checks: dict) -> bool:
    """Check that all required fixture files exist."""
    missing = []
    for fname in REQUIRED_EVIDENCE_FILES:
        if not (evidence_dir / fname).exists():
            missing.append(fname)
    if missing:
        _check(checks, "required_fixture_files", "FAIL",
               f"missing: {', '.join(missing)}", "all required files present",
               f"Missing required evidence files: {', '.join(missing)}")
        return False
    _check(checks, "required_fixture_files", "PASS",
           f"{len(REQUIRED_EVIDENCE_FILES)} files present",
           f"{len(REQUIRED_EVIDENCE_FILES)} files",
           "All required fixture files present")
    return True


def _check_row_counts(
    evidence_dir: Path,
    manifest: dict,
    checks: dict,
) -> None:
    """Check that row counts match declared counts in the manifest."""
    declared_cf = manifest.get("n_counterfactual_outcomes", 0)
    declared_exp = manifest.get("n_experiences", 0)
    declared_tasks = manifest.get("n_tasks", 0)

    cf_rows = _safe_load_jsonl(evidence_dir / "counterfactual_outcomes.jsonl")
    exp_rows = _safe_load_jsonl(evidence_dir / "executive_experiences.jsonl")

    cf_count = len(cf_rows)
    exp_count = len(exp_rows)

    if cf_count == declared_cf:
        _check(checks, "row_count_counterfactuals", "PASS",
               f"{cf_count} rows", f"{declared_cf} rows",
               "Counterfactual row count matches manifest")
    else:
        _check(checks, "row_count_counterfactuals", "FAIL",
               f"{cf_count} rows", f"{declared_cf} rows",
               f"Counterfactual row count mismatch: {cf_count} != {declared_cf}")

    if exp_count == declared_exp:
        _check(checks, "row_count_experiences", "PASS",
               f"{exp_count} rows", f"{declared_exp} rows",
               "Experience row count matches manifest")
    else:
        _check(checks, "row_count_experiences", "FAIL",
               f"{exp_count} rows", f"{declared_exp} rows",
               f"Experience row count mismatch: {exp_count} != {declared_exp}")

    # Check benchmark manifest total_tasks matches
    bench = _safe_load_json(evidence_dir / "benchmark_manifest.json")
    if isinstance(bench, dict):
        bench_tasks = bench.get("total_tasks", 0)
        if bench_tasks == declared_tasks:
            _check(checks, "row_count_tasks", "PASS",
                   f"{bench_tasks} tasks", f"{declared_tasks} tasks",
                   "Task count matches manifest")
        else:
            _check(checks, "row_count_tasks", "FAIL",
                   f"{bench_tasks} tasks", f"{declared_tasks} tasks",
                   f"Task count mismatch: {bench_tasks} != {declared_tasks}")
    else:
        _check(checks, "row_count_tasks", "FAIL",
               "benchmark_manifest not loadable", f"{declared_tasks} tasks",
               "Cannot load benchmark_manifest.json")


def _check_split_consistency(evidence_dir: Path, checks: dict) -> None:
    """Check that split manifest is internally consistent."""
    split = _safe_load_json(evidence_dir / "split_manifest.json")
    if not isinstance(split, dict):
        _check(checks, "split_consistency", "FAIL",
               "split_manifest not loadable", "valid split manifest",
               "Cannot load split_manifest.json")
        return

    splits = split.get("splits", {})
    if not isinstance(splits, dict) or not splits:
        _check(checks, "split_consistency", "FAIL",
               "no splits defined", "non-empty splits dict",
               "split_manifest has no splits")
        return

    all_ok = True
    total = 0
    for split_name, split_info in splits.items():
        if not isinstance(split_info, dict):
            all_ok = False
            continue
        count = split_info.get("count", 0)
        task_ids = split_info.get("task_ids", [])
        if count != len(task_ids):
            all_ok = False
        total += len(task_ids)

    if all_ok:
        _check(checks, "split_consistency", "PASS",
               f"{len(splits)} splits, {total} total tasks",
               "all split counts match task_id lists",
               "Split manifest is internally consistent")
    else:
        _check(checks, "split_consistency", "FAIL",
               f"{len(splits)} splits", "all split counts match task_id lists",
               "Split manifest has count/task_id mismatches")


def _check_action_matrix(evidence_dir: Path, checks: dict) -> None:
    """Check that every task in counterfactual outcomes has all 4 actions."""
    cf_rows = _safe_load_jsonl(evidence_dir / "counterfactual_outcomes.jsonl")
    if not cf_rows:
        _check(checks, "action_matrix_completeness", "FAIL",
               "0 counterfactual rows", "all tasks with all actions",
               "No counterfactual outcome rows")
        return

    # Group by task_id
    task_actions: dict[str, set[str]] = {}
    for row in cf_rows:
        tid = row.get("task_id", "")
        action = row.get("eligible_action", row.get("executed_action", ""))
        if tid and action:
            task_actions.setdefault(tid, set()).add(action)

    n_total = len(task_actions)
    n_complete = 0
    for tid, actions in task_actions.items():
        if len(actions) >= len(CANONICAL_ACTIONS):
            n_complete += 1

    if n_complete == n_total and n_total > 0:
        _check(checks, "action_matrix_completeness", "PASS",
               f"{n_complete}/{n_total} tasks complete",
               f"all tasks have {len(CANONICAL_ACTIONS)} actions",
               "Action matrix is complete")
    else:
        _check(checks, "action_matrix_completeness", "FAIL",
               f"{n_complete}/{n_total} tasks complete",
               f"all tasks have {len(CANONICAL_ACTIONS)} actions",
               f"Action matrix incomplete: {n_complete}/{n_total} tasks have all actions")


def _check_policy_schema(evidence_dir: Path, checks: dict) -> None:
    """Check that candidate and sham policies have valid schema."""
    candidate = _safe_load_json(evidence_dir / "candidate_policy.json")
    sham = _safe_load_json(evidence_dir / "sham_policy.json")

    required_fields = {"policy_id", "policy_type", "model_id", "weights", "bias",
                       "action_ordering", "policy_sha256"}

    all_ok = True
    for name, policy in (("candidate", candidate), ("sham", sham)):
        if not isinstance(policy, dict):
            all_ok = False
            continue
        missing = required_fields - set(policy.keys())
        if missing:
            all_ok = False

    if all_ok:
        _check(checks, "policy_schema", "PASS",
               "candidate and sham policies valid",
               "all required policy fields present",
               "Policy schemas are valid")
    else:
        _check(checks, "policy_schema", "FAIL",
               "policy schema incomplete", "all required policy fields present",
               "Policy schema validation failed")


def _check_safety_rows(evidence_dir: Path, checks: dict) -> None:
    """Check that safety_results.jsonl has rows."""
    safety = _safe_load_jsonl(evidence_dir / "safety_results.jsonl")
    if len(safety) > 0:
        _check(checks, "safety_rows", "PASS",
               f"{len(safety)} safety rows", ">0 safety rows",
               "Safety evidence rows present")
    else:
        _check(checks, "safety_rows", "FAIL",
               f"{len(safety)} safety rows", ">0 safety rows",
               "No safety evidence rows")


def _check_checksum_integrity(evidence_dir: Path, checks: dict) -> None:
    """Check that recorded checksums match recomputed checksums."""
    checksum_registry = _safe_load_json(evidence_dir / "artifact_checksums.json")
    if not isinstance(checksum_registry, dict):
        _check(checks, "checksum_integrity", "FAIL",
               "artifact_checksums not loadable", "valid checksum registry",
               "Cannot load artifact_checksums.json")
        return

    n_checked = 0
    n_match = 0
    mismatches = []
    for fname in REQUIRED_EVIDENCE_FILES:
        fpath = evidence_dir / fname
        if not fpath.exists():
            continue
        if fname == "artifact_checksums.json":
            n_checked += 1
            n_match += 1
            continue
        recorded = checksum_registry.get(fname)
        try:
            recomputed = _sha256_file(fpath)
        except OSError:
            mismatches.append(fname)
            n_checked += 1
            continue
        n_checked += 1
        if recorded is not None and recorded == recomputed:
            n_match += 1
        else:
            mismatches.append(fname)

    if n_match == n_checked and not mismatches:
        _check(checks, "checksum_integrity", "PASS",
               f"{n_match}/{n_checked} match", "all match",
               "All checksums verified")
    else:
        _check(checks, "checksum_integrity", "FAIL",
               f"{n_match}/{n_checked} match", "all match",
               f"Checksum mismatches: {', '.join(mismatches)}")


def _check_origin_classification(
    evidence_dir: Path,
    evidence_manifest: dict,
    provider_manifest: dict,
    checks: dict,
) -> str:
    """Check that evidence origin is classified as SYNTHETIC."""
    result = classify_evidence_origin(provider_manifest, evidence_manifest)
    origin = result.origin

    if origin == EvidenceOrigin.SYNTHETIC:
        _check(checks, "origin_classification", "PASS",
               origin.value, "SYNTHETIC",
               "Evidence origin correctly classified as SYNTHETIC")
    else:
        _check(checks, "origin_classification", "FAIL",
               origin.value, "SYNTHETIC",
               f"Evidence origin misclassified: {origin.value}")

    # Also check anti-misclassification
    violations = check_anti_misclassification(provider_manifest, evidence_manifest)
    if not violations:
        _check(checks, "anti_misclassification", "PASS",
               "0 violations", "0 violations",
               "No anti-misclassification violations")
    else:
        _check(checks, "anti_misclassification", "FAIL",
               f"{len(violations)} violations", "0 violations",
               f"Anti-misclassification violations: {'; '.join(violations)}")

    return origin.value


def _check_non_promotability(evidence_manifest: dict, checks: dict) -> None:
    """Check that the fixture is marked as non-promotable."""
    promotable = evidence_manifest.get("promotable", True)
    scientific_eligible = evidence_manifest.get("scientific_claim_eligible", True)

    if not promotable and not scientific_eligible:
        _check(checks, "non_promotability", "PASS",
               f"promotable={promotable}, scientific_claim_eligible={scientific_eligible}",
               "promotable=False, scientific_claim_eligible=False",
               "Fixture correctly marked as non-promotable")
    else:
        _check(checks, "non_promotability", "FAIL",
               f"promotable={promotable}, scientific_claim_eligible={scientific_eligible}",
               "promotable=False, scientific_claim_eligible=False",
               "Fixture must be non-promotable and non-scientific")


def _check_no_real_placeholders(evidence_dir: Path, checks: dict) -> None:
    """Check for real placeholder markers (NOT 'synthetic')."""
    # We scan key JSON files for placeholder markers, but NOT "synthetic"
    # which is expected in fixture evidence.
    found = []
    files_to_scan = [
        "EVIDENCE_MANIFEST.json",
        "provider_manifest.json",
        "source_provenance.json",
        "candidate_policy.json",
        "sham_policy.json",
    ]
    for fname in files_to_scan:
        fpath = evidence_dir / fname
        if not fpath.exists():
            continue
        try:
            content = fpath.read_text(encoding="utf-8").lower()
        except OSError:
            continue
        for marker in PLACEHOLDER_MARKERS:
            if marker.lower() in content:
                found.append(f"{fname}: '{marker}'")

    if not found:
        _check(checks, "no_real_placeholders", "PASS",
               "0 placeholder markers", "0 placeholder markers",
               "No real placeholder markers detected (synthetic excluded)")
    else:
        _check(checks, "no_real_placeholders", "FAIL",
               f"{len(found)} markers found", "0 placeholder markers",
               f"Placeholder markers found: {'; '.join(found)}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_fixture(evidence_dir: str) -> dict:
    """Validate a synthetic fixture evidence directory.

    Returns a dict with:
        structural_validity: "PASS" | "FAIL"
        evidence_origin: "SYNTHETIC"
        scientific_claim_eligibility: "NOT_APPLICABLE"
        gate_a_eligible: False
        promotable: False
        checks: dict of check_name -> {status, observed, expected, reason}
        reason: summary string
    """
    evidence_path = Path(evidence_dir)
    checks: dict[str, dict[str, Any]] = {}

    # Check directory exists
    if not evidence_path.exists() or not evidence_path.is_dir():
        _check(checks, "directory_exists", "FAIL",
               "missing", "valid directory",
               f"Evidence directory does not exist: {evidence_dir}")
        return {
            "structural_validity": "FAIL",
            "evidence_origin": "SYNTHETIC",
            "scientific_claim_eligibility": "NOT_APPLICABLE",
            "gate_a_eligible": False,
            "promotable": False,
            "checks": checks,
            "reason": f"Evidence directory does not exist: {evidence_dir}",
        }

    _check(checks, "directory_exists", "PASS",
           str(evidence_dir), "valid directory",
           "Evidence directory exists")

    # Check required files
    files_ok = _check_required_files(evidence_path, checks)

    # Load manifests
    evidence_manifest = _safe_load_json(evidence_path / "EVIDENCE_MANIFEST.json") or {}
    provider_manifest = _safe_load_json(evidence_path / "provider_manifest.json") or {}

    if files_ok:
        _check_row_counts(evidence_path, evidence_manifest, checks)
        _check_split_consistency(evidence_path, checks)
        _check_action_matrix(evidence_path, checks)
        _check_policy_schema(evidence_path, checks)
        _check_safety_rows(evidence_path, checks)
        _check_checksum_integrity(evidence_path, checks)
        _check_no_real_placeholders(evidence_path, checks)

    origin_str = _check_origin_classification(
        evidence_path, evidence_manifest, provider_manifest, checks
    )
    _check_non_promotability(evidence_manifest, checks)

    # Determine overall structural validity
    n_fail = sum(1 for c in checks.values() if c["status"] == "FAIL")
    structural_validity = "PASS" if n_fail == 0 else "FAIL"

    if structural_validity == "PASS":
        reason = (
            f"Fixture validation PASS: {len(checks)} checks passed, "
            f"origin=SYNTHETIC, non-promotable"
        )
    else:
        failed_names = [k for k, v in checks.items() if v["status"] == "FAIL"]
        reason = (
            f"Fixture validation FAIL: {n_fail} check(s) failed "
            f"({', '.join(failed_names)})"
        )

    return {
        "structural_validity": structural_validity,
        "evidence_origin": "SYNTHETIC",
        "scientific_claim_eligibility": "NOT_APPLICABLE",
        "gate_a_eligible": False,
        "promotable": False,
        "checks": checks,
        "reason": reason,
    }
