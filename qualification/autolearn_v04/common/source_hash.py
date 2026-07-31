"""Canonical analysis source-tree hasher.

Hashes the complete analysis source tree, not just a single file.
Excludes generated artifacts, caches, evidence, and build directories.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


# Directories to exclude (relative prefixes or absolute dir names).
EXCLUDED_DIR_NAMES = frozenset({
    "__pycache__",
    ".pytest_cache",
    ".git",
    "build",
    "dist",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
})

# Prefixes to exclude (repository-relative).
EXCLUDED_PREFIXES = (
    "qualification/evidence/",
    "qualification/autolearn_v04/artifacts/",
    "qualification/archive/",
    "qualification/autolearn_v04/v042/__pycache__/",
    "qualification/autolearn_v04/v043/__pycache__/",
    "qualification/autolearn_v04/v044/__pycache__/",
    "qualification/autolearn_v04/common/__pycache__/",
)

# File patterns to exclude.
EXCLUDED_SUFFIXES = frozenset({
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite-wal",
    ".sqlite-shm",
    ".log",
})

# Empty SHA-256 (hash of empty input).
SHA256_EMPTY = hashlib.sha256(b"").hexdigest()

# Hash algorithm version for reproducibility.
HASH_ALGORITHM_VERSION = "sha256-tree-v1"


@dataclass(frozen=True)
class SourceTreeHashResult:
    """Result of hashing the analysis source tree."""

    analysis_source_file_count: int
    analysis_source_byte_count: int
    analysis_source_tree_sha256: str
    included_roots: list[str]
    excluded_prefixes: list[str]
    hash_algorithm_version: str
    file_list: list[str]  # sorted list of relative paths included

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Don't include full file list in dict by default (too large).
        d.pop("file_list", None)
        return d


def hash_source_tree(
    repo_root: str | Path,
    included_roots: list[str] | None = None,
) -> SourceTreeHashResult:
    """Hash the complete analysis source tree.

    Args:
        repo_root: Repository root directory.
        included_roots: List of repository-relative directories to include.
            Defaults to the standard v0.4.4 analysis source set.

    Returns:
        SourceTreeHashResult with file count, byte count, and digest.
    """
    if included_roots is None:
        included_roots = [
            "qualification/autolearn_v04/v044/",
            "qualification/autolearn_v04/common/",
            "qualification/autolearn_v04/__init__.py",
            "src/capsule_brain/",
            "pyproject.toml",
            "qualification/QUALIFICATION_MANIFEST.json",
        ]

    root = Path(repo_root)
    if not root.exists():
        raise FileNotFoundError(f"Repository root not found: {root}")

    files: list[tuple[str, bytes]] = []
    relative_paths: list[str] = []

    for inc in included_roots:
        inc_path = root / inc
        if inc_path.is_file():
            rel = inc
            if _is_excluded(rel):
                continue
            data = inc_path.read_bytes()
            files.append((rel, data))
            relative_paths.append(rel)
        elif inc_path.is_dir():
            for dirpath, dirnames, filenames in os.walk(inc_path):
                # Filter excluded dirs in-place.
                dirnames[:] = [
                    d for d in dirnames
                    if d not in EXCLUDED_DIR_NAMES
                ]
                for fname in sorted(filenames):
                    fpath = Path(dirpath) / fname
                    rel = str(fpath.relative_to(root))
                    if _is_excluded(rel):
                        continue
                    if fname.endswith(".pyc"):
                        continue
                    data = fpath.read_bytes()
                    files.append((rel, data))
                    relative_paths.append(rel)

    # Sort by relative path for deterministic ordering.
    files.sort(key=lambda x: x[0])
    relative_paths.sort()

    # Compute combined hash.
    h = hashlib.sha256()
    byte_count = 0
    for rel, data in files:
        h.update(rel.encode())
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
        byte_count += len(data)

    digest = h.hexdigest()

    return SourceTreeHashResult(
        analysis_source_file_count=len(files),
        analysis_source_byte_count=byte_count,
        analysis_source_tree_sha256=digest,
        included_roots=included_roots,
        excluded_prefixes=list(EXCLUDED_PREFIXES),
        hash_algorithm_version=HASH_ALGORITHM_VERSION,
        file_list=relative_paths,
    )


def _is_excluded(rel_path: str) -> bool:
    """Check if a relative path should be excluded."""
    for prefix in EXCLUDED_PREFIXES:
        if rel_path.startswith(prefix):
            return True
    for suffix in EXCLUDED_SUFFIXES:
        if rel_path.endswith(suffix):
            return True
    return False


def validate_source_hash(result: SourceTreeHashResult) -> dict[str, Any]:
    """Validate that a source hash result is not empty or invalid."""
    errors: list[str] = []
    if result.analysis_source_file_count == 0:
        errors.append("file_count is zero")
    if result.analysis_source_byte_count == 0:
        errors.append("byte_count is zero")
    if result.analysis_source_tree_sha256 == SHA256_EMPTY:
        errors.append("digest is SHA256_EMPTY")
    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }
