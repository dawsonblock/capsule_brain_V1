"""Cross-origin duplication detection for v0.4.6.

Scans all evidence packages under a root directory and compares
benchmark, split, task-set, policy, and counterfactual digests across
packages with **different** origins.

A high-severity violation occurs when the same or near-identical
evidence appears under different origins (e.g. the same benchmark
digest in both a SYNTHETIC and a REAL_MODEL package).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.common.evidence_origin import EvidenceOrigin

# Digest fields to compare across packages with different origins.
DIGEST_FIELDS: tuple[str, ...] = (
    "benchmark_sha256",
    "split_sha256",
    "counterfactuals_sha256",
    "candidate_policy_sha256",
    "sham_policy_sha256",
)

# File names that contain the digests we compare.
EVIDENCE_MANIFEST_NAME = "EVIDENCE_MANIFEST.json"


def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _compute_file_digest(path: Path) -> str | None:
    """Compute SHA-256 of a file, returning None on failure."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _scan_package(pkg_dir: Path) -> dict[str, Any] | None:
    """Scan a single evidence package directory.

    Returns a dict with origin and digest values, or None if the
    directory does not look like an evidence package.
    """
    manifest_path = pkg_dir / EVIDENCE_MANIFEST_NAME
    manifest = _load_json(manifest_path)
    if manifest is None:
        return None

    origin = str(manifest.get("evidence_origin", "")).upper()
    if not origin:
        # Try provider_manifest as fallback.
        prov = _load_json(pkg_dir / "provider_manifest.json")
        if prov:
            origin = str(prov.get("provider_class", "")).upper()

    digests: dict[str, str] = {}
    for field in DIGEST_FIELDS:
        val = manifest.get(field)
        if isinstance(val, str) and val:
            digests[field] = val

    # Also compute file-level digests for key artifacts.
    for fname in (
        "counterfactual_outcomes.jsonl",
        "candidate_policy.json",
        "sham_policy.json",
        "benchmark_manifest.json",
        "split_manifest.json",
    ):
        fpath = pkg_dir / fname
        if fpath.exists():
            fd = _compute_file_digest(fpath)
            if fd:
                digests[f"_file:{fname}"] = fd

    run_id = manifest.get("original_run_id") or manifest.get("run_id", "")

    return {
        "package_dir": str(pkg_dir),
        "origin": origin,
        "run_id": str(run_id),
        "digests": digests,
        "manifest": manifest,
    }


def _origins_differ(o1: str, o2: str) -> bool:
    """Return True if two origin strings represent different origins."""
    if not o1 or not o2:
        return False
    return o1.upper() != o2.upper()


def detect_cross_origin_duplicates(evidence_root: str) -> dict:
    """Scan all evidence packages under *evidence_root* for cross-origin
    duplication.

    Parameters
    ----------
    evidence_root:
        Path to a directory containing one or more evidence packages.

    Returns
    -------
    dict
        status (PASS/FAIL), violations list, packages_scanned, reason
    """
    root = Path(evidence_root)
    violations: list[dict[str, Any]] = []

    if not root.exists() or not root.is_dir():
        return {
            "status": AuditStatus.BLOCKED.value,
            "violations": [],
            "packages_scanned": 0,
            "reason": f"evidence root does not exist or is not a directory: {evidence_root}",
        }

    # Scan for evidence packages.  A package is any directory that
    # contains an EVIDENCE_MANIFEST.json.
    packages: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if EVIDENCE_MANIFEST_NAME in filenames:
            info = _scan_package(Path(dirpath))
            if info is not None:
                packages.append(info)

    packages_scanned = len(packages)

    if packages_scanned == 0:
        return {
            "status": AuditStatus.BLOCKED.value,
            "violations": [],
            "packages_scanned": 0,
            "reason": "no evidence packages found under evidence root",
        }

    # Compare every pair of packages with different origins.
    for i in range(len(packages)):
        for j in range(i + 1, len(packages)):
            p1 = packages[i]
            p2 = packages[j]

            if not _origins_differ(p1["origin"], p2["origin"]):
                continue

            # Compare digests.
            shared_digest_keys = set(p1["digests"].keys()) & set(p2["digests"].keys())
            matching_digests: list[str] = []
            for key in sorted(shared_digest_keys):
                if p1["digests"][key] == p2["digests"][key]:
                    matching_digests.append(key)

            if matching_digests:
                violations.append({
                    "severity": "HIGH",
                    "package_a": p1["package_dir"],
                    "origin_a": p1["origin"],
                    "run_id_a": p1["run_id"],
                    "package_b": p2["package_dir"],
                    "origin_b": p2["origin"],
                    "run_id_b": p2["run_id"],
                    "matching_digests": matching_digests,
                    "reason": (
                        f"same or near-identical evidence found under "
                        f"different origins: {p1['origin']} and {p2['origin']} "
                        f"share {len(matching_digests)} digest(s)"
                    ),
                })

    if violations:
        return {
            "status": AuditStatus.FAIL.value,
            "violations": violations,
            "packages_scanned": packages_scanned,
            "reason": (
                f"{len(violations)} cross-origin duplication violation(s) detected"
            ),
        }

    return {
        "status": AuditStatus.PASS.value,
        "violations": [],
        "packages_scanned": packages_scanned,
        "reason": (
            f"no cross-origin duplication detected across "
            f"{packages_scanned} package(s)"
        ),
    }
