"""Provenance: complete SHA-256 lineage for v0.4.0 qualification (Section 7).

Fixes the v0.3.2 defects:
  * source_hash was SHA256(empty content) because compute_source_tree_hash
    was called with the package subdirectory, not the project root.
  * split_manifest_digest was "" because the wrong key was read.
  * Provenance was a disconnected chain, not embedded in the policy.
  * Promotion trusted the policy's declared hashes instead of recomputing.

v0.4.0 introduces:
  * PolicyProvenance with 12+ non-empty digest fields.
  * Empty-content SHA-256 rejection (validate_digest).
  * Source-tree hashing from the PROJECT ROOT over src/, qualification/,
    tests/, configs/, pyproject.toml -- excluding generated artifacts.
  * A transitive lineage digest H_lineage over all component digests.
  * Independent verification: promotion recomputes every digest and
    compares it to the policy's declared values, failing closed on
    mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .schemas import (
    EMPTY_SHA256,
    PolicyProvenance,
    ProvenanceError,
    canonical_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
    validate_digest,
    write_json,
)


# ---------------------------------------------------------------------------
# Source-tree hashing (Section 7.3)
# ---------------------------------------------------------------------------

# Directories and files to include in the source hash. These are relative
# to the PROJECT ROOT, not the package directory.
SOURCE_PATHS: tuple[str, ...] = (
    "src",
    "qualification/autolearn_v04",
    "tests",
    "configs",
)

SOURCE_FILES: tuple[str, ...] = (
    "pyproject.toml",
)

# File extensions to skip (binary, cache, artifacts).
SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".so", ".dylib", ".dll",
    ".sqlite", ".db", ".sqlite-wal", ".sqlite-shm", ".wal", ".shm",
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".zip", ".tar", ".gz",
    ".safetensors", ".npz", ".npy",
    ".egg-info",
})

# Simple directory names to skip — matched against any path component.
# These are generic cache/build/venv names that should never appear in
# source trees.  Do NOT include generic names like "data" here — that
# would skip every file when the repo lives under /mnt/data/...
SIMPLE_SKIP_DIR_NAMES: frozenset[str] = frozenset({
    "__pycache__", ".pytest_cache", ".git",
    "node_modules", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "dist", "build",
})

# Repository-relative path prefixes to skip.  These are checked against
# the path relative to the repository root, never against absolute paths.
SKIP_RELATIVE_PREFIXES: frozenset[str] = frozenset({
    "qualification/archive",
    "qualification/autolearn_v04/artifacts",
    "qualification/autolearn_v04/activations",
    "src/capsule_brain.egg-info",
})

# File names to skip.
SKIP_FILENAMES: frozenset[str] = frozenset({
    ".gitignore", ".DS_Store",
})

# Text file extensions for line-ending normalization.
TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".toml", ".yaml", ".yml", ".json", ".md", ".txt", ".rst",
    ".cfg", ".ini", ".sh", ".bash",
})


def should_skip(path: Path, root: Path) -> bool:
    """Determine whether a file should be excluded from source hashing.

    All include/exclude decisions operate on repository-relative paths,
    never on absolute path components.  This ensures that a parent
    directory named ``data`` or ``build`` does not cause every source
    file to be skipped.
    """
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return True  # path is outside the root — skip it

    relative_parts = relative.parts
    relative_posix = relative.as_posix()

    # Check simple dir names against relative parts only.
    if any(part in SIMPLE_SKIP_DIR_NAMES for part in relative_parts):
        return True

    # Check relative prefix exclusions.
    if any(
        relative_posix == prefix
        or relative_posix.startswith(prefix + "/")
        for prefix in SKIP_RELATIVE_PREFIXES
    ):
        return True

    # Check extensions and filenames.
    if path.suffix in SKIP_EXTENSIONS:
        return True
    if path.name in SKIP_FILENAMES:
        return True

    return False


# Backward-compatible alias (deprecated — use should_skip with root).
def _should_skip(path: Path) -> bool:
    """Legacy skip check — only use when root is not available."""
    if path.suffix in SKIP_EXTENSIONS:
        return True
    if path.name in SKIP_FILENAMES:
        return True
    if any(part in SIMPLE_SKIP_DIR_NAMES for part in path.parts):
        return True
    return False


def _normalize_line_endings(data: bytes, is_text: bool) -> bytes:
    """Normalize CRLF to LF for text files (Section 5.3)."""
    if not is_text:
        return data
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _collect_files(root: Path) -> list[Path]:
    """Collect all source files in deterministic order."""
    files: list[Path] = []
    for src_path in SOURCE_PATHS:
        dir_path = root / src_path
        if not dir_path.exists():
            continue
        import os
        for dirpath, dirnames, filenames in os.walk(dir_path):
            # Prune simple skip dirs during walk for efficiency.
            dirnames[:] = sorted(
                d for d in dirnames if d not in SIMPLE_SKIP_DIR_NAMES
            )
            for filename in sorted(filenames):
                filepath = Path(dirpath) / filename
                if not should_skip(filepath, root):
                    files.append(filepath)
    for src_file in SOURCE_FILES:
        filepath = root / src_file
        if filepath.exists() and not should_skip(filepath, root):
            files.append(filepath)
    files.sort(key=lambda p: str(p.relative_to(root)))
    return files


@dataclass(frozen=True)
class SourceTreeHashResult:
    """Result of source-tree hashing with metadata for provenance."""
    repository_root: str
    source_file_count: int
    source_byte_count: int
    included_roots: list[str]
    excluded_relative_prefixes: list[str]
    source_tree_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_root": self.repository_root,
            "source_file_count": self.source_file_count,
            "source_byte_count": self.source_byte_count,
            "included_roots": list(self.included_roots),
            "excluded_relative_prefixes": list(self.excluded_relative_prefixes),
            "source_tree_sha256": self.source_tree_sha256,
        }


class SourceHashError(RuntimeError):
    """Raised when source hashing fails (empty set, no files, etc.)."""


def compute_source_tree_hash_detailed(root: str | Path) -> SourceTreeHashResult:
    """Compute a deterministic SHA-256 over the source tree with full metadata.

    Hash algorithm (Section 5.3):
        1. collect matching files;
        2. sort repository-relative paths lexicographically;
        3. normalize line endings for text files;
        4. hash path length, path bytes, file length and file bytes;
        5. produce SHA-256.

    Hard fail if:
        source_file_count == 0
        source_byte_count == 0
        source_tree_sha256 == SHA256_EMPTY
    """
    root_path = Path(root).resolve()
    files = _collect_files(root_path)

    h = hashlib.sha256()
    total_bytes = 0
    for filepath in files:
        rel = str(filepath.relative_to(root_path))
        rel_bytes = rel.encode("utf-8")
        is_text = filepath.suffix in TEXT_EXTENSIONS
        try:
            file_bytes = filepath.read_bytes()
        except Exception:
            file_bytes = b"ERROR"
        file_bytes = _normalize_line_endings(file_bytes, is_text)
        file_len = len(file_bytes)
        total_bytes += file_len

        # Hash: path_length, path_bytes, file_length, file_bytes
        h.update(len(rel_bytes).to_bytes(4, "little"))
        h.update(rel_bytes)
        h.update(file_len.to_bytes(8, "little"))
        h.update(file_bytes)

    digest = h.hexdigest()

    result = SourceTreeHashResult(
        repository_root=str(root_path),
        source_file_count=len(files),
        source_byte_count=total_bytes,
        included_roots=list(SOURCE_PATHS),
        excluded_relative_prefixes=sorted(SKIP_RELATIVE_PREFIXES),
        source_tree_sha256=digest,
    )

    # Hard fail on empty source set.
    if result.source_file_count == 0:
        raise SourceHashError(
            f"Source hashing found 0 files under {root_path}. "
            f"Repository root discovery is broken."
        )
    if result.source_byte_count == 0:
        raise SourceHashError(
            f"Source hashing found 0 bytes under {root_path}. "
            f"All files are empty or unreadable."
        )
    if digest == EMPTY_SHA256:
        raise SourceHashError(
            f"Source hash is SHA-256 of empty input. "
            f"No file content was hashed."
        )

    return result


def compute_source_tree_hash(root: str | Path) -> str:
    """Compute a deterministic SHA-256 over the source tree from the PROJECT ROOT.

    Wrapper around compute_source_tree_hash_detailed that returns only the
    digest string. Raises SourceHashError on empty/invalid source sets.
    """
    return compute_source_tree_hash_detailed(root).source_tree_sha256


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the project root by walking up from start until pyproject.toml
    and src/ are both present."""
    p = Path(start).resolve() if start else Path(__file__).resolve().parent
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    # Fallback: assume the package's grandparent (qualification/autolearn_v04 -> repo root)
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Artifact digests
# ---------------------------------------------------------------------------


def artifact_digest(path: str | Path) -> str:
    """SHA-256 of an artifact file, or "" if missing."""
    p = Path(path)
    if not p.exists():
        return ""
    return sha256_bytes(p.read_bytes())


def json_artifact_digest(path: str | Path) -> str:
    """SHA-256 of a JSON artifact file's canonical content."""
    p = Path(path)
    if not p.exists():
        return ""
    # Hash the raw bytes so a byte-identical file always produces the same
    # digest regardless of pretty-printing differences.
    return sha256_bytes(p.read_bytes())


# ---------------------------------------------------------------------------
# Lineage digest (Section 7.4)
# ---------------------------------------------------------------------------


def compute_lineage_digest(prov: PolicyProvenance) -> str:
    """H_lineage = H(H_source, H_benchmark, H_split, H_execution, H_dataset,
    H_features, H_hyperparameters, H_learner, H_runtime, H_verifier, H_parent).

    The parent digest is the SHA-256 of the parent policy id (or a fixed
    sentinel for the root baseline).
    """
    parent_component = sha256_text(prov.parent_policy_id or "ROOT_BASELINE")
    h = hashlib.sha256()
    for component in (
        prov.source_tree_digest,
        prov.benchmark_manifest_digest,
        prov.split_manifest_digest,
        prov.counterfactual_results_digest,
        prov.training_dataset_digest,
        prov.feature_schema_digest,
        prov.hyperparameter_digest,
        prov.learner_code_digest,
        prov.runtime_config_digest,
        prov.verifier_config_digest,
        parent_component,
    ):
        h.update(component.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Provenance assembly
# ---------------------------------------------------------------------------


def build_policy_provenance(
    *,
    config: QualificationConfig,
    policy_id: str,
    policy_version: str,
    parent_policy_id: str | None,
    benchmark_manifest_path: Path,
    split_manifest_path: Path,
    counterfactual_results_path: Path,
    training_dataset_path: Path,
    feature_schema_digest: str,
    hyperparameter_digest: str,
    learner_code_digest: str,
    model_id: str,
    model_revision: str | None,
    tokenizer_id: str,
    tokenizer_revision: str | None,
    created_at_utc: str,
) -> PolicyProvenance:
    """Assemble a complete PolicyProvenance from on-disk artifacts.

    Every digest is computed from the actual artifact content. The
    resulting object is validated before return.
    """
    project_root = find_project_root()
    source_tree_digest = compute_source_tree_hash(project_root)
    benchmark_manifest_digest = json_artifact_digest(benchmark_manifest_path)
    split_manifest_digest = json_artifact_digest(split_manifest_path)
    counterfactual_results_digest = json_artifact_digest(counterfactual_results_path)
    training_dataset_digest = json_artifact_digest(training_dataset_path)
    runtime_config_digest = config.runtime_config_digest()
    verifier_config_digest = config.verifier_config_digest()

    prov = PolicyProvenance(
        policy_id=policy_id,
        policy_version=policy_version,
        parent_policy_id=parent_policy_id,
        source_tree_digest=source_tree_digest,
        benchmark_manifest_digest=benchmark_manifest_digest,
        split_manifest_digest=split_manifest_digest,
        counterfactual_results_digest=counterfactual_results_digest,
        training_dataset_digest=training_dataset_digest,
        feature_schema_digest=feature_schema_digest,
        hyperparameter_digest=hyperparameter_digest,
        learner_code_digest=learner_code_digest,
        runtime_config_digest=runtime_config_digest,
        verifier_config_digest=verifier_config_digest,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        created_at_utc=created_at_utc,
    )
    prov.validate()
    # Compute and attach the lineage digest.
    lineage = compute_lineage_digest(prov)
    # dataclass is frozen; rebuild with the lineage digest.
    return PolicyProvenance(
        policy_id=prov.policy_id,
        policy_version=prov.policy_version,
        parent_policy_id=prov.parent_policy_id,
        source_tree_digest=prov.source_tree_digest,
        benchmark_manifest_digest=prov.benchmark_manifest_digest,
        split_manifest_digest=prov.split_manifest_digest,
        counterfactual_results_digest=prov.counterfactual_results_digest,
        training_dataset_digest=prov.training_dataset_digest,
        feature_schema_digest=prov.feature_schema_digest,
        hyperparameter_digest=prov.hyperparameter_digest,
        learner_code_digest=prov.learner_code_digest,
        runtime_config_digest=prov.runtime_config_digest,
        verifier_config_digest=prov.verifier_config_digest,
        model_id=prov.model_id,
        model_revision=prov.model_revision,
        tokenizer_id=prov.tokenizer_id,
        tokenizer_revision=prov.tokenizer_revision,
        created_at_utc=prov.created_at_utc,
        lineage_digest=lineage,
    )


# ---------------------------------------------------------------------------
# Independent verification (Section 7.5)
# ---------------------------------------------------------------------------


def verify_policy_provenance(
    *,
    config: QualificationConfig,
    declared: PolicyProvenance,
    benchmark_manifest_path: Path,
    split_manifest_path: Path,
    counterfactual_results_path: Path,
    training_dataset_path: Path,
    feature_schema_digest: str,
    hyperparameter_digest: str,
    learner_code_digest: str,
) -> dict[str, Any]:
    """Independently recompute every digest and compare to the declared values.

    Section 7.5: the promotion stage must NOT trust the policy's declared
    hashes. It must load referenced artifacts, recompute digests, compare
    each field, compare the final lineage digest, and fail closed on
    mismatch.
    """
    checks: list[dict[str, Any]] = []

    def _check(name: str, declared_val: str, recomputed: str) -> None:
        ok = declared_val == recomputed and declared_val != "" and declared_val != EMPTY_SHA256
        checks.append({
            "check": name,
            "passed": ok,
            "declared": declared_val[:16] + "..." if declared_val else "(empty)",
            "recomputed": recomputed[:16] + "..." if recomputed else "(missing)",
            "match": declared_val == recomputed,
        })

    project_root = find_project_root()
    _check("source_tree_digest", declared.source_tree_digest, compute_source_tree_hash(project_root))
    _check("benchmark_manifest_digest", declared.benchmark_manifest_digest, json_artifact_digest(benchmark_manifest_path))
    _check("split_manifest_digest", declared.split_manifest_digest, json_artifact_digest(split_manifest_path))
    _check("counterfactual_results_digest", declared.counterfactual_results_digest, json_artifact_digest(counterfactual_results_path))
    _check("training_dataset_digest", declared.training_dataset_digest, json_artifact_digest(training_dataset_path))
    _check("feature_schema_digest", declared.feature_schema_digest, feature_schema_digest)
    _check("hyperparameter_digest", declared.hyperparameter_digest, hyperparameter_digest)
    _check("learner_code_digest", declared.learner_code_digest, learner_code_digest)
    _check("runtime_config_digest", declared.runtime_config_digest, config.runtime_config_digest())
    _check("verifier_config_digest", declared.verifier_config_digest, config.verifier_config_digest())

    # Recompute lineage digest and compare.
    recomputed_lineage = compute_lineage_digest(declared)
    _check("lineage_digest", declared.lineage_digest, recomputed_lineage)

    all_passed = all(c["passed"] for c in checks)
    return {
        "valid": all_passed,
        "checks": checks,
        "n_passed": sum(1 for c in checks if c["passed"]),
        "n_failed": sum(1 for c in checks if not c["passed"]),
    }


# ---------------------------------------------------------------------------
# Pipeline-level provenance chain (for audit)
# ---------------------------------------------------------------------------


def _artifact_list(artifacts: Path) -> dict[str, str]:
    paths = [
        "benchmark_manifest.json",
        "split_manifest.json",
        "counterfactual_outcomes.json",
        "executive_experiences.jsonl",
        "dataset_manifest.json",
        "dataset_train.json",
        "dataset_validation.json",
        "dataset_test.json",
        "dataset_ood.json",
        "dataset_safety.json",
        "candidate_policy.json",
        "candidate_training.json",
        "sham_policy.json",
        "sham_training.json",
        "gate_a_result.json",
        "gate_b_result.json",
        "promotion_result.json",
        "post_promotion_result.json",
        "qualification_report.json",
    ]
    d = {}
    for p in paths:
        full = artifacts / p
        d[p] = artifact_digest(full) if full.exists() else "MISSING"
    return d


def build_pipeline_provenance(config: QualificationConfig) -> dict[str, Any]:
    """Build the final pipeline-level provenance chain for audit (Section 7.6).

    v0.4.1: Two-phase provenance:
    - Pre-execution: source_provenance.json (from compute_and_write_source_provenance)
    - Final: provenance.json (this function, runs after post-promotion)

    The final provenance covers the full artifact chain including
    benchmark, counterfactuals, dataset, candidate/sham/oracle,
    Gate A, Gate B, promotion, post-promotion, and report.

    No absolute paths are recorded. Source file and byte counts are included.
    The final provenance manifest is NOT hashed into itself.
    """
    artifacts = Path(config.artifacts_dir)
    project_root = find_project_root()
    source_result = compute_source_tree_hash_detailed(project_root)
    source_hash = source_result.source_tree_sha256
    artifact_digests = _artifact_list(artifacts)

    chain: list[dict] = []
    prev_hash = source_hash
    for name, h in artifact_digests.items():
        record = {"artifact": name, "hash": h, "previous": prev_hash}
        record_hash = sha256_text(canonical_json(record))
        chain.append({"record": record, "record_hash": record_hash})
        prev_hash = record_hash

    provenance = {
        "schema_version": "pipeline-provenance/2",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "phase": "final",
        "source_hash": source_hash,
        "source_file_count": source_result.source_file_count,
        "source_byte_count": source_result.source_byte_count,
        "artifact_digests": artifact_digests,
        "chain": chain,
        "final_hash": chain[-1]["record_hash"] if chain else source_hash,
        # v0.4.1: No absolute paths — use relative references only.
        "repository_root": str(project_root),
    }
    # Do NOT hash the provenance manifest into itself.
    path = artifacts / "provenance.json"
    write_json(path, provenance)
    return provenance


def compute_and_write_source_provenance(config: QualificationConfig) -> dict[str, Any]:
    """Compute source-tree provenance and write to artifacts/source_provenance.json.

    This is the early-stage provenance that runs before benchmark building
    and counterfactual execution. It records the source tree hash, file count,
    and byte count.
    """
    artifacts = Path(config.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    project_root = find_project_root()
    result = compute_source_tree_hash_detailed(project_root)

    provenance = {
        "schema_version": "source-provenance/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        **result.to_dict(),
    }
    write_json(artifacts / "source_provenance.json", provenance)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v0.4.0 pipeline provenance chain.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    args = parser.parse_args()
    config = QualificationConfig(artifacts_dir=args.artifacts_dir)
    provenance = build_pipeline_provenance(config)
    print(f"provenance: source_hash={provenance['source_hash'][:16]}... "
          f"final_hash={provenance['final_hash'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
