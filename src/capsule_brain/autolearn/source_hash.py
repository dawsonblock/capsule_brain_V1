"""Deterministic source hashing for AutoLearn v2.16.0 qualification.

Priority 8: Computes canonical SHA-256 over the source tree using
deterministic ordering, independent of Git. ZIP distributions have no
Git history, so provenance must not rely on ``git rev-parse``.

Hashes are computed over:
- src/
- qualification/
- tests/
- configs/
- pyproject.toml

Files are sorted by relative path. Content is read as bytes and fed
through SHA-256. The final hash is a SHA-256 of all per-file hashes
concatenated in sorted order.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


# Directories and files to include in the source hash.
SOURCE_PATHS: tuple[str, ...] = (
    "src",
    "qualification",
    "tests",
    "configs",
)

# Individual files to include.
SOURCE_FILES: tuple[str, ...] = (
    "pyproject.toml",
)

# File extensions to skip (binary, cache, etc.)
SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".so", ".dylib", ".dll",
    ".sqlite", ".db", ".json",  # artifacts, not source
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".zip", ".tar", ".gz",
})

# Directories to skip.
SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".pytest_cache", ".git",
    "node_modules", ".mypy_cache", ".ruff_cache",
    "data",  # runtime data, not source
})


def _should_skip(path: Path) -> bool:
    """Check if a file should be skipped."""
    if path.suffix in SKIP_EXTENSIONS:
        return True
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    # Skip .gitignore and other dotfiles in qualification dirs.
    if path.name == ".gitignore":
        return True
    return False


def _collect_files(root: Path) -> list[Path]:
    """Collect all source files in deterministic order."""
    files: list[Path] = []

    # Collect from directories.
    for src_path in SOURCE_PATHS:
        dir_path = root / src_path
        if not dir_path.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(dir_path):
            # Skip excluded directories in-place.
            dirnames[:] = sorted(
                d for d in dirnames if d not in SKIP_DIRS
            )
            for filename in sorted(filenames):
                filepath = Path(dirpath) / filename
                if not _should_skip(filepath):
                    files.append(filepath)

    # Collect individual files.
    for src_file in SOURCE_FILES:
        filepath = root / src_file
        if filepath.exists():
            files.append(filepath)

    # Sort by relative path for deterministic ordering.
    files.sort(key=lambda p: str(p.relative_to(root)))
    return files


def compute_source_tree_hash(root: str | Path) -> str:
    """Compute a deterministic SHA-256 over the source tree.

    Args:
        root: Root directory of the project.

    Returns:
        A hex SHA-256 string representing the canonical source tree hash.
    """
    root_path = Path(root).resolve()
    files = _collect_files(root_path)

    h = hashlib.sha256()
    for filepath in files:
        rel = str(filepath.relative_to(root_path))
        # Include the relative path in the hash so file moves are detected.
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")  # separator
        try:
            content = filepath.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            h.update(file_hash.encode("utf-8"))
        except Exception:
            h.update(b"ERROR")
        h.update(b"\x01")  # record separator

    return h.hexdigest()


def compute_file_hash(path: str | Path) -> str:
    """Compute SHA-256 of a single file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_string_hash(text: str) -> str:
    """Compute SHA-256 of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_provenance(
    *,
    runtime_type: str,
    evaluation_runtime_type: str,
    simulated_rows: int,
    dataset_hash: str,
    expected_dataset_hash: str,
    source_tree_hash: str,
    expected_source_tree_hash: str,
    train_test_overlap: bool,
    policy_lineage_valid: bool,
) -> dict:
    """Priority 7: Verify all provenance conditions for promotion.

    Returns a dict with per-check pass/fail and an overall ``valid`` flag.
    Simulation may never promote a policy — even perfect simulator
    performance must fail qualification.
    """
    checks = []

    def _check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    _check(
        "runtime_is_real",
        runtime_type == "real",
        f"runtime_type={runtime_type}",
    )
    _check(
        "evaluation_runtime_is_real",
        evaluation_runtime_type == "real",
        f"evaluation_runtime_type={evaluation_runtime_type}",
    )
    _check(
        "simulated_rows_zero",
        simulated_rows == 0,
        f"simulated_rows={simulated_rows}",
    )
    _check(
        "dataset_hash_verified",
        dataset_hash == expected_dataset_hash,
        f"dataset_hash={dataset_hash[:16]}..., expected={expected_dataset_hash[:16]}...",
    )
    _check(
        "source_tree_hash_verified",
        source_tree_hash == expected_source_tree_hash,
        f"source_tree_hash={source_tree_hash[:16]}..., expected={expected_source_tree_hash[:16]}...",
    )
    _check(
        "no_train_test_overlap",
        not train_test_overlap,
        f"train_test_overlap={train_test_overlap}",
    )
    _check(
        "policy_lineage_valid",
        policy_lineage_valid,
        f"policy_lineage_valid={policy_lineage_valid}",
    )

    all_passed = all(c["passed"] for c in checks)
    return {
        "valid": all_passed,
        "checks": checks,
        "n_passed": sum(1 for c in checks if c["passed"]),
        "n_failed": sum(1 for c in checks if not c["passed"]),
    }
