"""Artifact lineage validation for v0.4.6.

Builds and validates the artifact-lineage DAG.  Every artifact must
carry provenance metadata and the lineage must satisfy:

    - no missing parents
    - no cycles
    - no cross-origin parentage
    - no cross-run contamination
    - no synthetic parent in real run
    - all hashes recompute
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.common.evidence_origin import EvidenceOrigin

# Required fields on every artifact record.
REQUIRED_ARTIFACT_FIELDS = (
    "artifact_type",
    "run_id",
    "parent_artifact_hashes",
    "created_at",
    "producer_module",
)

# Fields used for hash recomputation.
HASH_FIELDS = (
    "artifact_type",
    "run_id",
    "parent_artifact_hashes",
    "created_at",
    "producer_module",
)


def _missing_fields(artifact: dict) -> list[str]:
    return [
        f for f in REQUIRED_ARTIFACT_FIELDS
        if f not in artifact or artifact[f] in (None, "")
    ]


def _recompute_hash(artifact: dict) -> str:
    """Recompute a canonical SHA-256 over the artifact's identity fields."""
    payload = {}
    for field in HASH_FIELDS:
        val = artifact.get(field)
        if isinstance(val, list):
            payload[field] = list(val)
        else:
            payload[field] = val
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _detect_cycle(artifacts: dict[str, dict]) -> list[list[str]]:
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


def _get_origin(artifact: dict) -> str:
    """Extract the evidence origin from an artifact."""
    origin = artifact.get("evidence_origin")
    if origin:
        return str(origin).upper()
    origin = artifact.get("origin")
    if origin:
        return str(origin).upper()
    return ""


def validate_artifact_lineage(
    artifacts: dict,
    evidence_origin: str,
) -> dict:
    """Validate the complete artifact DAG.

    Parameters
    ----------
    artifacts:
        Mapping of artifact name -> artifact record.  Each record must
        include ``artifact_type``, ``run_id``, ``parent_artifact_hashes``,
        ``created_at``, and ``producer_module``.
    evidence_origin:
        The expected evidence origin for this package (e.g. "SYNTHETIC",
        "REAL_MODEL").

    Returns
    -------
    dict
        status, n_artifacts, n_valid, n_missing_parents, n_cycles,
        n_cross_origin, reason
    """
    expected_origin = str(evidence_origin).upper()

    if artifacts is None or not isinstance(artifacts, dict) or len(artifacts) == 0:
        return {
            "status": AuditStatus.BLOCKED.value,
            "n_artifacts": 0,
            "n_valid": 0,
            "n_missing_parents": 0,
            "n_cycles": 0,
            "n_cross_origin": 0,
            "reason": "artifact lineage missing",
        }

    n_artifacts = len(artifacts)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    # --- Structural completeness ---
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

    # --- Missing parents ---
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

    # --- Cycles ---
    cyclic = _detect_cycle(artifacts)
    checks.append({
        "check": "no_cycles",
        "status": "PASS" if not cyclic else "FAIL",
        "observed": len(cyclic),
        "expected": 0,
    })
    if cyclic:
        errors.append(f"{len(cyclic)} cyclic dependency/dependencies detected")

    # --- Cross-origin parentage ---
    cross_origin: list[dict[str, Any]] = []
    for name in structurally_valid:
        artifact = artifacts[name]
        art_origin = _get_origin(artifact)
        if not art_origin:
            continue
        parents = artifact.get("parent_artifact_hashes", []) or []
        for parent in parents:
            if parent not in artifacts:
                continue
            parent_origin = _get_origin(artifacts[parent])
            if parent_origin and art_origin and parent_origin != art_origin:
                cross_origin.append({
                    "artifact": name,
                    "origin": art_origin,
                    "parent": parent,
                    "parent_origin": parent_origin,
                })
    checks.append({
        "check": "no_cross_origin_parentage",
        "status": "PASS" if not cross_origin else "FAIL",
        "observed": len(cross_origin),
        "expected": 0,
    })
    if cross_origin:
        errors.append(f"{len(cross_origin)} cross-origin parentage violation(s)")

    # --- Cross-run contamination ---
    from collections import Counter
    run_ids = [
        artifacts[name].get("run_id") for name in structurally_valid
        if artifacts[name].get("run_id")
    ]
    cross_run: list[dict[str, Any]] = []
    primary_run_id = None
    if run_ids:
        counts = Counter(run_ids)
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

    # --- No synthetic parent in real run ---
    synthetic_in_real: list[dict[str, Any]] = []
    if expected_origin == EvidenceOrigin.REAL_MODEL.value:
        for name in structurally_valid:
            parents = artifacts[name].get("parent_artifact_hashes", []) or []
            for parent in parents:
                if parent not in artifacts:
                    continue
                parent_origin = _get_origin(artifacts[parent])
                if parent_origin == EvidenceOrigin.SYNTHETIC.value:
                    synthetic_in_real.append({
                        "artifact": name,
                        "parent": parent,
                        "parent_origin": parent_origin,
                    })
    checks.append({
        "check": "no_synthetic_parent_in_real_run",
        "status": "PASS" if not synthetic_in_real else "FAIL",
        "observed": len(synthetic_in_real),
        "expected": 0,
    })
    if synthetic_in_real:
        errors.append(
            f"{len(synthetic_in_real)} synthetic parent(s) in real run"
        )

    # --- Hash recomputation ---
    hash_mismatches: list[dict[str, Any]] = []
    for name in structurally_valid:
        artifact = artifacts[name]
        stored_hash = artifact.get("artifact_hash") or artifact.get("hash")
        if stored_hash:
            recomputed = _recompute_hash(artifact)
            if recomputed != stored_hash:
                hash_mismatches.append({
                    "artifact": name,
                    "stored_hash": stored_hash,
                    "recomputed_hash": recomputed,
                })
    checks.append({
        "check": "all_hashes_recompute",
        "status": "PASS" if not hash_mismatches else "FAIL",
        "observed": len(hash_mismatches),
        "expected": 0,
    })
    if hash_mismatches:
        errors.append(f"{len(hash_mismatches)} hash mismatch(es)")

    # --- Determine final status ---
    n_valid = sum(1 for c in checks if c["status"] == "PASS")

    if errors:
        # FAIL takes priority
        return {
            "status": AuditStatus.FAIL.value,
            "n_artifacts": n_artifacts,
            "n_valid": n_valid,
            "n_missing_parents": len(missing_parents),
            "n_cycles": len(cyclic),
            "n_cross_origin": len(cross_origin),
            "reason": "; ".join(errors),
        }

    return {
        "status": AuditStatus.PASS.value,
        "n_artifacts": n_artifacts,
        "n_valid": n_valid,
        "n_missing_parents": 0,
        "n_cycles": 0,
        "n_cross_origin": 0,
        "reason": (
            f"all {n_artifacts} artifact(s) pass lineage validation "
            f"(origin={expected_origin})"
        ),
    }
