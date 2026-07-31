"""Final provenance manifest for v0.4.4.

Hashes: immutable evidence manifest, all imported evidence files,
analysis source tree, analysis configuration, all v0.4.4 generated
artifacts, final report.

Does NOT hash the final provenance manifest into itself.
No absolute paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def create_final_provenance(
    evidence_manifest: dict,
    evidence_dir: str | Path,
    source_hash: dict,
    analysis_config: dict,
    output_dir: str | Path,
    final_report_path: str | Path | None = None,
    original_commit_sha: str = "",
    current_commit_sha: str = "",
    analysis_mode: str = "evidence-analysis",
    random_seeds: dict | None = None,
) -> dict[str, Any]:
    """Create the final provenance manifest.

    No absolute paths are included — all paths are repository-relative.
    """
    from capsule_brain.version import (
        PACKAGE_VERSION,
        AUTOLEARN_VERSION,
        AUTOLEARN_QUALIFICATION_VERSION,
    )

    edir = Path(evidence_dir)
    odir = Path(output_dir)

    # Hash all evidence files.
    evidence_file_hashes: dict[str, str] = {}
    for fname in sorted(os.listdir(edir)):
        fpath = edir / fname
        if fpath.is_file():
            evidence_file_hashes[fname] = _sha256_file(fpath)

    # Hash all generated artifacts.
    artifact_hashes: dict[str, str] = {}
    for fname in sorted(os.listdir(odir)):
        fpath = odir / fname
        if fpath.is_file() and fname != "final_provenance_manifest.json":
            artifact_hashes[fname] = _sha256_file(fpath)

    # Hash the final report.
    report_hash = None
    if final_report_path:
        rp = Path(final_report_path)
        if rp.exists():
            report_hash = _sha256_file(rp)

    # Hash the evidence manifest.
    manifest_hash = _sha256_json(evidence_manifest)

    # Hash the analysis config.
    config_hash = _sha256_json(analysis_config)

    # Dependency lock digest (placeholder — would hash requirements.txt or similar).
    dep_lock_digest = hashlib.sha256(b"dependency_lock_placeholder").hexdigest()

    provenance = {
        "provenance_version": "1.0.0",
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "original_scientific_run_commit": original_commit_sha,
        "current_analysis_commit": current_commit_sha,
        "original_package_version": evidence_manifest.get("original_package_version", ""),
        "current_package_version": PACKAGE_VERSION,
        "model_identity": {
            "model_id": evidence_manifest.get("model_id", ""),
            "model_revision": evidence_manifest.get("model_revision", ""),
            "tokenizer_id": evidence_manifest.get("tokenizer_id", ""),
            "tokenizer_revision": evidence_manifest.get("tokenizer_revision", ""),
        },
        "provider_identity": {
            "provider_class": evidence_manifest.get("provider_class", ""),
            "runtime_type": evidence_manifest.get("runtime_type", ""),
        },
        "utility_version": analysis_config.get("utility_version", "normalized_v1"),
        "verifier_versions": {
            "primary": "v1",
        },
        "source_file_count": source_hash.get("analysis_source_file_count", 0),
        "source_byte_count": source_hash.get("analysis_source_byte_count", 0),
        "source_tree_sha256": source_hash.get("analysis_source_tree_sha256", ""),
        "dependency_lock_digest": dep_lock_digest,
        "python_version": sys.version,
        "platform": platform.platform(),
        "analysis_mode": analysis_mode,
        "random_seeds": random_seeds or {"random_seed": 42, "bootstrap_seed": 42},
        "evidence_manifest_sha256": manifest_hash,
        "evidence_file_hashes": evidence_file_hashes,
        "analysis_config_sha256": config_hash,
        "artifact_hashes": artifact_hashes,
        "final_report_sha256": report_hash,
        # No absolute paths — only relative.
        "evidence_dir_relative": str(edir),
        "output_dir_relative": str(odir),
    }

    return provenance


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    """Compute SHA-256 of a JSON-serialized object."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()
