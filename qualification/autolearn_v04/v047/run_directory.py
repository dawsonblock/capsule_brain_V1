"""Strict, immutable run-directory contract for v0.4.7.

A v0.4.7 run directory is the single source of truth for a qualification
run.  It must contain a fixed set of manifest files, JSONL evidence
streams, subdirectories for each policy, and a checksum file that
cryptographically binds every artifact in the tree.

The contract is immutable: once ``write_checksums_file`` has been
written, the directory must not change.  ``verify_checksums`` detects
any post-hoc mutation.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Required files in a v0.4.7 run directory (relative to run root).
REQUIRED_FILES = [
    "RUN_MANIFEST.json",
    "SOURCE_MANIFEST.json",
    "ENVIRONMENT_MANIFEST.json",
    "PROVIDER_MANIFEST.json",
    "MODEL_MANIFEST.json",
    "TOKENIZER_MANIFEST.json",
    "GENERATION_CONFIG.json",
    "UTILITY_CONFIG.json",
    "VERIFIER_REGISTRY.json",
    "BENCHMARK_MANIFEST.json",
    "SPLIT_MANIFEST.json",
    "SPLIT_ACCESS_LOG.jsonl",
    "COUNTERFACTUAL_EQUIVALENCE.jsonl",
    "COUNTERFACTUAL_OUTCOMES.jsonl",
    "NORMALIZED_EXPERIENCES.jsonl",
    "SAFETY_EXPERIENCES.jsonl",
    "TRAINING_MANIFEST.json",
    "ARTIFACT_DAG.json",
    "CHECKSUMS.sha256",
    "FINAL_REPORT.md",
]

# Required subdirectories.
REQUIRED_DIRS = [
    "BASELINE_POLICY",
    "CANDIDATE_POLICY",
    "SHAM_POLICY",
    "ORACLE_POLICY",
    "EVALUATION",
    "GATE_RESULTS",
]


# ---------------------------------------------------------------------------
# Directory creation / validation
# ---------------------------------------------------------------------------


def create_run_directory(root: Path, run_id: str, force: bool = False) -> Path:
    """Create a run directory under ``root / run_id``.

    Fails if the directory already exists and is non-empty, unless
    ``force=True`` in which case the existing directory is removed first.

    All required subdirectories are created.  Required files are NOT
    created here; the orchestrator is responsible for writing them.
    """
    root = Path(root)
    run_dir = root / run_id

    if run_dir.exists():
        if any(run_dir.iterdir()):
            if not force:
                raise FileExistsError(
                    f"Run directory already exists and is non-empty: {run_dir} "
                    "(pass force=True to overwrite)"
                )
            import shutil

            shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for d in REQUIRED_DIRS:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

    return run_dir


def validate_run_directory(run_dir: Path) -> dict:
    """Validate that a run directory has all required files and dirs.

    Returns a dict with::

        {
            "valid": bool,
            "missing": list[str],   # missing required files
            "missing_dirs": list[str],
            "reasons": list[str],   # human-readable explanations
        }
    """
    run_dir = Path(run_dir)
    missing: list[str] = []
    missing_dirs: list[str] = []
    reasons: list[str] = []

    if not run_dir.exists() or not run_dir.is_dir():
        return {
            "valid": False,
            "missing": list(REQUIRED_FILES),
            "missing_dirs": list(REQUIRED_DIRS),
            "reasons": [f"run directory does not exist: {run_dir}"],
        }

    for name in REQUIRED_FILES:
        p = run_dir / name
        if not p.exists() or not p.is_file():
            missing.append(name)
            reasons.append(f"missing required file: {name}")

    for name in REQUIRED_DIRS:
        p = run_dir / name
        if not p.exists() or not p.is_dir():
            missing_dirs.append(name)
            reasons.append(f"missing required directory: {name}")

    return {
        "valid": len(missing) == 0 and len(missing_dirs) == 0,
        "missing": missing,
        "missing_dirs": missing_dirs,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_checksums(run_dir: Path) -> dict:
    """Compute SHA-256 checksums for all files in the run directory.

    Returns a dict mapping the repository-relative path (relative to
    ``run_dir``) to the lowercase hex SHA-256 digest.

    The CHECKSUMS.sha256 file itself is excluded from the computation
    because it is the output of this function.
    """
    run_dir = Path(run_dir)
    checksums: dict[str, str] = {}

    if not run_dir.exists() or not run_dir.is_dir():
        return checksums

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir).as_posix()
        if rel == "CHECKSUMS.sha256":
            continue
        checksums[rel] = _sha256_file(path)

    return checksums


def write_checksums_file(run_dir: Path) -> None:
    """Write ``CHECKSUMS.sha256`` with all current file checksums.

    The file uses the standard ``<sha256>  <relative-path>`` format
    (two-space separator) so it can be verified with ``sha256sum -c``.
    """
    run_dir = Path(run_dir)
    checksums = compute_checksums(run_dir)
    lines = [f"{digest}  {rel}" for rel, digest in sorted(checksums.items())]
    (run_dir / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksums(run_dir: Path) -> dict:
    """Verify all checksums recorded in ``CHECKSUMS.sha256``.

    Returns a dict with::

        {
            "valid": bool,
            "mismatches": list[str],   # paths whose digest changed
            "missing": list[str],      # paths listed but no longer present
            "extra": list[str],        # files present but not in checksums
        }
    """
    run_dir = Path(run_dir)
    checksums_path = run_dir / "CHECKSUMS.sha256"
    mismatches: list[str] = []
    missing: list[str] = []
    extra: list[str] = []

    recorded: dict[str, str] = {}
    if checksums_path.exists():
        for line in checksums_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: "<sha256>  <path>"
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            digest, rel = parts
            recorded[rel.strip()] = digest.strip()

    # Check recorded files.
    for rel, expected_digest in recorded.items():
        p = run_dir / rel
        if not p.exists():
            missing.append(rel)
            continue
        actual = _sha256_file(p)
        if actual != expected_digest:
            mismatches.append(rel)

    # Detect extra files not in the manifest.
    current = compute_checksums(run_dir)
    for rel in current:
        if rel not in recorded:
            extra.append(rel)

    return {
        "valid": len(mismatches) == 0 and len(missing) == 0,
        "mismatches": mismatches,
        "missing": missing,
        "extra": extra,
    }


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_manifest(run_id: str, config: dict, source_digest: str) -> dict:
    """Build ``RUN_MANIFEST.json`` with run identity.

    The run manifest binds a run to its configuration and source tree.
    Any change to ``config`` or ``source_digest`` produces a different
    run identity.
    """
    config_digest = _sha256_json(config)
    return {
        "run_id": run_id,
        "protocol_version": "0.4.7",
        "source_digest": source_digest,
        "config_digest": config_digest,
        "run_identity": _sha256_str(
            f"{run_id}|{source_digest}|{config_digest}"
        ),
        "created_at": _now(),
    }


def build_source_manifest(
    git_commit: str | None,
    git_dirty: bool | None,
    tree_digest: str,
    files: list[dict],
) -> dict:
    """Build ``SOURCE_MANIFEST.json``.

    If ``git_commit`` is ``None`` the manifest records
    ``source_identity_method: "TREE_MANIFEST"`` and relies on the
    ``tree_digest`` for reproducibility.  Otherwise it records the git
    commit and dirty flag.
    """
    if git_commit is None:
        return {
            "source_identity_method": "TREE_MANIFEST",
            "git_commit": None,
            "git_dirty": None,
            "tree_digest": tree_digest,
            "file_count": len(files),
            "files": files,
            "created_at": _now(),
        }
    return {
        "source_identity_method": "GIT_COMMIT",
        "git_commit": git_commit,
        "git_dirty": bool(git_dirty) if git_dirty is not None else None,
        "tree_digest": tree_digest,
        "file_count": len(files),
        "files": files,
        "created_at": _now(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
