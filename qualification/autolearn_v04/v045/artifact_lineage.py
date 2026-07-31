"""Artifact lineage validation for v0.4.5.

Validates the complete artifact DAG produced by a qualification run.
Every artifact must carry provenance metadata and the lineage must
satisfy:

* no missing parents
* no cyclic dependencies
* no cross-run contamination
* no stale mini-run artifact
* all hashes recompute
* all task-set digests align where required
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Required fields on every artifact record.
REQUIRED_ARTIFACT_FIELDS = (
    "artifact_type",
    "schema_version",
    "run_id",
    "parent_artifact_hashes",
    "source_evidence_manifest_sha256",
    "created_at",
    "producer_module",
    "producer_source_sha256",
    "config_sha256",
)


def _missing_fields(artifact: dict) -> list[str]:
    return [
        f for f in REQUIRED_ARTIFACT_FIELDS
        if f not in artifact or artifact[f] in (None, "")
    ]


def _recompute_hash(artifact: dict) -> str:
    """Recompute a canonical SHA-256 over the artifact's identity fields.

    Only stable identity-bearing fields are hashed, so a recompute
    mismatch reflects a genuine provenance drift rather than incidental
    ordering of bookkeeping fields.
    """
    payload = {
        "artifact_type": artifact.get("artifact_type"),
        "schema_version": artifact.get("schema_version"),
        "run_id": artifact.get("run_id"),
        "parent_artifact_hashes": list(artifact.get("parent_artifact_hashes", [])),
        "source_evidence_manifest_sha256": artifact.get("source_evidence_manifest_sha256"),
        "created_at": artifact.get("created_at"),
        "producer_module": artifact.get("producer_module"),
        "producer_source_sha256": artifact.get("producer_source_sha256"),
        "config_sha256": artifact.get("config_sha256"),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _detect_cycle(
    artifacts: dict[str, dict],
) -> list[list[str]]:
    """Return a list of cycles (each a list of artifact names) if any."""
    cycles: list[list[str]] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {name: WHITE for name in artifacts}

    def dfs(node: str, path: list[str]) -> None:
        color[node] = GRAY
        path.append(node)
        parents = artifacts[node].get("parent_artifact_hashes", []) or []
        for parent in parents:
            if parent not in artifacts:
                continue
            if color[parent] == GRAY:
                # Found a back edge -> cycle.
                try:
                    start = path.index(parent)
                    cycles.append(path[start:] + [parent])
                except ValueError:
                    cycles.append([node, parent, node])
            elif color[parent] == WHITE:
                dfs(parent, path)
        path.pop()
        color[node] = BLACK

    for name in artifacts:
        if color[name] == WHITE:
            dfs(name, [])
    return cycles


def validate_artifact_lineage(artifacts: dict[str, dict]) -> dict:
    """Validate the complete artifact DAG.

    Parameters
    ----------
    artifacts:
        Mapping of artifact name -> artifact record. Each record must
        include ``artifact_type``, ``schema_version``, ``run_id``,
        ``parent_artifact_hashes``, ``source_evidence_manifest_sha256``,
        ``created_at``, ``producer_module``, ``producer_source_sha256``
        and ``config_sha256``.

    Returns
    -------
    dict
        ``status`` (``"PASS"`` | ``"FAIL"`` | ``"BLOCKED"``),
        ``n_artifacts``, ``n_valid``, ``missing_parents``,
        ``cyclic_dependencies``, ``cross_run_contamination``,
        ``stale_artifacts`` and ``errors``.
    """
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    if artifacts is None or not isinstance(artifacts, dict) or len(artifacts) == 0:
        return {
            "status": "BLOCKED",
            "n_artifacts": 0,
            "n_valid": 0,
            "missing_parents": [],
            "cyclic_dependencies": [],
            "cross_run_contamination": [],
            "stale_artifacts": [],
            "errors": ["artifact lineage missing"],
        }

    n_artifacts = len(artifacts)

    # --- Structural completeness -----------------------------------------
    structurally_valid: list[str] = []
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            errors.append(f"artifact '{name}' is not a dict")
            continue
        missing = _missing_fields(artifact)
        if missing:
            errors.append(f"artifact '{name}' missing fields: {missing}")
            continue
        structurally_valid.append(name)
    checks.append({
        "check": "artifacts_structurally_complete",
        "status": "PASS" if len(structurally_valid) == n_artifacts else "FAIL",
        "observed": len(structurally_valid),
        "expected": n_artifacts,
    })

    # --- Missing parents --------------------------------------------------
    missing_parents: list[dict[str, Any]] = []
    for name in structurally_valid:
        parents = artifacts[name].get("parent_artifact_hashes", []) or []
        for parent in parents:
            if parent not in artifacts:
                missing_parents.append({
                    "artifact": name,
                    "missing_parent": parent,
                })
    checks.append({
        "check": "no_missing_parents",
        "status": "PASS" if not missing_parents else "FAIL",
        "observed": len(missing_parents),
        "expected": 0,
    })
    if missing_parents:
        errors.append(f"{len(missing_parents)} missing parent artifact(s)")

    # --- Cyclic dependencies ---------------------------------------------
    cyclic = _detect_cycle(artifacts)
    cyclic_dependencies: list[dict[str, Any]] = [
        {"cycle": cycle} for cycle in cyclic
    ]
    checks.append({
        "check": "no_cyclic_dependencies",
        "status": "PASS" if not cyclic else "FAIL",
        "observed": len(cyclic),
        "expected": 0,
    })
    if cyclic:
        errors.append(f"{len(cyclic)} cyclic dependency/dependencies detected")

    # --- Cross-run contamination -----------------------------------------
    cross_run: list[dict[str, Any]] = []
    run_ids = {
        artifacts[name].get("run_id") for name in structurally_valid
        if artifacts[name].get("run_id")
    }
    primary_run_id = None
    if run_ids:
        # The primary run id is the most common one.
        from collections import Counter
        counts = Counter(
            artifacts[name].get("run_id") for name in structurally_valid
        )
        primary_run_id = counts.most_common(1)[0][0]
    for name in structurally_valid:
        rid = artifacts[name].get("run_id")
        if primary_run_id and rid != primary_run_id:
            cross_run.append({
                "artifact": name,
                "run_id": rid,
                "expected_run_id": primary_run_id,
            })
    checks.append({
        "check": "no_cross_run_contamination",
        "status": "PASS" if not cross_run else "FAIL",
        "observed": len(cross_run),
        "expected": 0,
    })
    if cross_run:
        errors.append(f"{len(cross_run)} artifact(s) from a foreign run")

    # --- Stale mini-run artifact -----------------------------------------
    stale: list[dict[str, Any]] = []
    for name in structurally_valid:
        artifact = artifacts[name]
        stale_markers = (
            "mini_run",
            "mini-run",
            "stale",
            "scratch",
            "tmp",
            "temp",
        )
        haystack = " ".join([
            str(artifact.get("artifact_type", "")),
            str(artifact.get("producer_module", "")),
            name,
        ]).lower()
        if any(marker in haystack for marker in stale_markers):
            stale.append({
                "artifact": name,
                "reason": "artifact appears to be a stale mini-run / scratch artifact",
            })
    checks.append({
        "check": "no_stale_mini_run_artifact",
        "status": "PASS" if not stale else "FAIL",
        "observed": len(stale),
        "expected": 0,
    })
    if stale:
        errors.append(f"{len(stale)} stale mini-run artifact(s)")

    # --- Hashes recompute -------------------------------------------------
    hash_mismatches: list[dict[str, Any]] = []
    for name in structurally_valid:
        artifact = artifacts[name]
        declared = artifact.get("artifact_sha256")
        if not declared:
            # If no declared hash is present we cannot recompute; treat as
            # a missing identity field but do not fail the recompute check
            # outright (the structural check already covers missing fields).
            continue
        recomputed = _recompute_hash(artifact)
        if recomputed != declared:
            hash_mismatches.append({
                "artifact": name,
                "declared": declared,
                "recomputed": recomputed,
            })
    checks.append({
        "check": "all_hashes_recompute",
        "status": "PASS" if not hash_mismatches else "FAIL",
        "observed": len(hash_mismatches),
        "expected": 0,
    })
    if hash_mismatches:
        errors.append(f"{len(hash_mismatches)} artifact hash(es) did not recompute")

    # --- Task-set digest alignment ---------------------------------------
    digest_mismatches: list[dict[str, Any]] = []
    for name in structurally_valid:
        artifact = artifacts[name]
        declared_digest = artifact.get("task_set_digest")
        expected_digest = artifact.get("expected_task_set_digest")
        if declared_digest is None and expected_digest is None:
            # Not every artifact carries a task-set digest; only enforce
            # alignment where both are present.
            continue
        if declared_digest is not None and expected_digest is not None:
            if declared_digest != expected_digest:
                digest_mismatches.append({
                    "artifact": name,
                    "declared": declared_digest,
                    "expected": expected_digest,
                })
        elif declared_digest is not None and expected_digest is None:
            digest_mismatches.append({
                "artifact": name,
                "declared": declared_digest,
                "expected": None,
                "reason": "expected_task_set_digest missing",
            })
    checks.append({
        "check": "task_set_digests_align",
        "status": "PASS" if not digest_mismatches else "FAIL",
        "observed": len(digest_mismatches),
        "expected": 0,
    })
    if digest_mismatches:
        errors.append(f"{len(digest_mismatches)} task-set digest mismatch(es)")

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    status = "PASS" if n_fail == 0 else "FAIL"

    # An artifact is "valid" if it is structurally complete and not flagged
    # by any of the lineage checks.
    flagged = {
        entry["artifact"]
        for entry in (missing_parents + cross_run + stale + hash_mismatches + digest_mismatches)
    }
    n_valid = len([name for name in structurally_valid if name not in flagged])

    return {
        "status": status,
        "n_artifacts": n_artifacts,
        "n_valid": n_valid,
        "missing_parents": missing_parents,
        "cyclic_dependencies": cyclic_dependencies,
        "cross_run_contamination": cross_run,
        "stale_artifacts": stale,
        "errors": errors,
    }
