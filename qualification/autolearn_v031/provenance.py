"""Provenance computation for AutoLearn v0.3.1.

v0.3.1 Section 25: Correct provenance.

- source_tree_sha256: hash normalized repository files directly (NOT git hash)
- qualification_source_sha256: hash qualification source only
- Exclude: .git, __pycache__, .pytest_cache, build, dist, generated outputs,
  SQLite WAL/SHM, temp files, venvs
- Sort paths lexicographically
- Hash path + normalized file bytes
- Use repository-relative artifact paths only
- No absolute local paths
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from qualification.autolearn_v031.schemas import ProvenanceManifest

# Directories and file patterns to exclude from source tree hashing.
_EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", "build", "dist",
    ".venv", "venv", "env", ".tox", ".mypy_cache", ".ruff_cache",
    "node_modules", ".eggs", ".cache",
}

_EXCLUDE_FILE_SUFFIXES = {
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll",
    ".sqlite", ".sqlite-wal", ".sqlite-shm", ".db", ".db-wal", ".db-shm",
    ".log", ".tmp", ".swp", ".swo",
}

_EXCLUDE_FILE_NAMES = {
    ".DS_Store", "Thumbs.db", ".coverage", "coverage.xml",
}

# Generated qualification outputs are excluded from source tree hash.
_GENERATED_OUTPUT_DIRS = {
    "artifacts", ".cf_isolation", "data",
}


def _should_exclude_path(path: Path, root: Path) -> bool:
    """Check if a path should be excluded from source tree hashing."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    parts = rel.parts
    # Exclude if any part is in _EXCLUDE_DIRS.
    for part in parts:
        if part in _EXCLUDE_DIRS:
            return True
        if part in _GENERATED_OUTPUT_DIRS:
            return True
    # Exclude by file suffix.
    if path.suffix in _EXCLUDE_FILE_SUFFIXES:
        return True
    # Exclude by file name.
    if path.name in _EXCLUDE_FILE_NAMES:
        return True
    return False


def compute_source_tree_sha256(root: str | Path) -> str:
    """Compute SHA-256 over normalized repository source files.

    This is NOT a git commit hash. It hashes the actual file contents,
    sorted lexicographically by relative path.
    """
    root_path = Path(root).resolve()
    h = hashlib.sha256()
    files: list[Path] = []
    for path in root_path.rglob("*"):
        if path.is_file() and not _should_exclude_path(path, root_path):
            files.append(path)
    # Sort by relative path string for determinism.
    files.sort(key=lambda p: str(p.relative_to(root_path)))
    for path in files:
        rel = str(path.relative_to(root_path))
        # Hash the path + file bytes.
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(path.read_bytes())
        except Exception:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()


def compute_qualification_source_sha256(root: str | Path) -> str:
    """Compute SHA-256 over qualification source files only."""
    root_path = Path(root).resolve()
    qual_dir = root_path / "qualification" / "autolearn_v031"
    if not qual_dir.exists():
        return hashlib.sha256(b"").hexdigest()
    h = hashlib.sha256()
    files: list[Path] = []
    for path in qual_dir.rglob("*.py"):
        if path.is_file():
            files.append(path)
    files.sort(key=lambda p: str(p.relative_to(root_path)))
    for path in files:
        rel = str(path.relative_to(root_path))
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def compute_artifact_sha256(artifact_path: str | Path) -> str:
    """Compute SHA-256 of a single artifact file."""
    path = Path(artifact_path)
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_json_sha256(data: Any) -> str:
    """Compute SHA-256 of JSON-serialized data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_provenance_manifest(
    *,
    root: str | Path,
    artifact_hashes: dict[str, str] | None = None,
    model_id: str = "",
    model_digest: str = "",
    container_image_digest: str = "",
    runtime_config_digest: str = "",
    random_seeds: tuple[str, ...] = (),
    dependency_lock_digest: str = "",
    package_version: str = "",
) -> ProvenanceManifest:
    """Build a complete provenance manifest."""
    root = Path(root)
    artifact_hashes = artifact_hashes or {}
    return ProvenanceManifest(
        source_tree_sha256=compute_source_tree_sha256(root),
        qualification_source_sha256=compute_qualification_source_sha256(root),
        benchmark_manifest_sha256=artifact_hashes.get("benchmark_manifest", ""),
        split_manifest_sha256=artifact_hashes.get("split_manifest", ""),
        counterfactual_results_sha256=artifact_hashes.get("counterfactual_results", ""),
        executive_experiences_sha256=artifact_hashes.get("executive_experiences", ""),
        dataset_sha256=artifact_hashes.get("dataset", ""),
        baseline_policy_sha256=artifact_hashes.get("baseline_policy", ""),
        sham_policy_sha256=artifact_hashes.get("sham_policy", ""),
        candidate_policy_sha256=artifact_hashes.get("candidate_policy", ""),
        validation_results_sha256=artifact_hashes.get("validation_results", ""),
        test_results_sha256=artifact_hashes.get("test_results", ""),
        ood_results_sha256=artifact_hashes.get("ood_results", ""),
        promotion_result_sha256=artifact_hashes.get("promotion_result", ""),
        post_promotion_results_sha256=artifact_hashes.get("post_promotion_results", ""),
        report_sha256=artifact_hashes.get("report", ""),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        package_version=package_version,
        model_id=model_id,
        model_digest=model_digest,
        container_image_digest=container_image_digest,
        runtime_config_digest=runtime_config_digest,
        random_seeds=random_seeds,
        dependency_lock_digest=dependency_lock_digest,
    )


def verify_provenance(manifest: ProvenanceManifest, root: str | Path) -> bool:
    """Verify that the source tree hash matches the current repository."""
    actual = compute_source_tree_sha256(root)
    return actual == manifest.source_tree_sha256
