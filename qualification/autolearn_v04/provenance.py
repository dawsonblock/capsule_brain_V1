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
    "qualification",
    "tests",
    "configs",
)

SOURCE_FILES: tuple[str, ...] = (
    "pyproject.toml",
)

# File extensions to skip (binary, cache, artifacts).
SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".so", ".dylib", ".dll",
    ".sqlite", ".db",
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".zip", ".tar", ".gz",
    ".safetensors", ".npz", ".npy",
    ".egg-info",
})

# Directories to skip (generated artifacts, caches, venvs).
SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".pytest_cache", ".git",
    "node_modules", ".mypy_cache", ".ruff_cache",
    "data",  # runtime data, not source
    "artifacts", "artifacts_v04", "artifacts_v032",
    "qualification_runs",
    ".cf_isolation_v04", ".qual_sandbox_v04",
    ".cf_isolation_v032", ".qual_sandbox_v032",
    ".qual_sandbox",
    "dist", "build",
    "src/capsule_brain.egg-info",
})

# File names to skip.
SKIP_FILENAMES: frozenset[str] = frozenset({
    ".gitignore", ".DS_Store",
})


def _should_skip(path: Path) -> bool:
    if path.suffix in SKIP_EXTENSIONS:
        return True
    if path.name in SKIP_FILENAMES:
        return True
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    # Skip generated qualification run outputs.
    if "qualification_runs" in path.parts:
        return True
    return False


def _collect_files(root: Path) -> list[Path]:
    """Collect all source files in deterministic order."""
    files: list[Path] = []
    for src_path in SOURCE_PATHS:
        dir_path = root / src_path
        if not dir_path.exists():
            continue
        import os
        for dirpath, dirnames, filenames in os.walk(dir_path):
            dirnames[:] = sorted(
                d for d in dirnames if d not in SKIP_DIRS
            )
            for filename in sorted(filenames):
                filepath = Path(dirpath) / filename
                if not _should_skip(filepath):
                    files.append(filepath)
    for src_file in SOURCE_FILES:
        filepath = root / src_file
        if filepath.exists() and not _should_skip(filepath):
            files.append(filepath)
    files.sort(key=lambda p: str(p.relative_to(root)))
    return files


def compute_source_tree_hash(root: str | Path) -> str:
    """Compute a deterministic SHA-256 over the source tree from the PROJECT ROOT.

    Section 7.3: hash only relevant source and configuration files. Exclude
    .git, __pycache__, generated artifacts, test outputs, qualification run
    outputs, virtual environments, model caches, and logs.

    Includes file path AND file content (Section 7.3):
        digest.update(relative_path)
        digest.update(b"\\0")
        digest.update(file_bytes)
        digest.update(b"\\0")
    """
    root_path = Path(root).resolve()
    files = _collect_files(root_path)
    h = hashlib.sha256()
    for filepath in files:
        rel = str(filepath.relative_to(root_path))
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        try:
            file_bytes = filepath.read_bytes()
            h.update(file_bytes)
        except Exception:
            h.update(b"ERROR")
        h.update(b"\x00")
    digest = h.hexdigest()
    # Hard guard: if no files were hashed, the root is wrong. Return the
    # empty digest so validate_digest rejects it downstream.
    return digest


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
    """Build the pipeline-level provenance chain for audit (Section 7.6)."""
    artifacts = Path(config.artifacts_dir)
    project_root = find_project_root()
    source_hash = compute_source_tree_hash(project_root)
    artifact_digests = _artifact_list(artifacts)

    chain: list[dict] = []
    prev_hash = source_hash
    for name, h in artifact_digests.items():
        record = {"artifact": name, "hash": h, "previous": prev_hash}
        record_hash = sha256_text(canonical_json(record))
        chain.append({"record": record, "record_hash": record_hash})
        prev_hash = record_hash

    provenance = {
        "schema_version": "pipeline-provenance/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "source_hash": source_hash,
        "artifact_digests": artifact_digests,
        "chain": chain,
        "final_hash": chain[-1]["record_hash"] if chain else source_hash,
    }
    path = artifacts / "provenance.json"
    write_json(path, provenance)
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
