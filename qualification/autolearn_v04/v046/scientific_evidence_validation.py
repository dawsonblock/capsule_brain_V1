"""Strict scientific evidence validation for v0.4.6.

This module validates evidence for scientific claims.  It must REJECT:
  - synthetic origin
  - simulated origin
  - infrastructure-only origin
  - missing model_digest
  - missing historical source identity
  - missing task-level rows
  - missing equivalence digests
  - missing utility identity
  - incomplete policy lineage
  - missing split-access audit
  - missing serialization parity

The output distinguishes four independent dimensions:
  - Structural completeness
  - Origin authenticity
  - Scientific eligibility
  - Gate A eligibility

It does NOT produce a single aggregate PASS/FAIL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.common.evidence_origin import (
    EvidenceOrigin,
    classify_evidence_origin,
)
from qualification.autolearn_v04.common.evidence_structure import (
    EvidenceStructureLevel,
    classify_evidence,
)
from qualification.autolearn_v04.v045.config import REQUIRED_EVIDENCE_FILES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _safe_load_json(path: Path) -> Any:
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _safe_load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return _load_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return []


def _status_str(status: AuditStatus) -> str:
    return status.value


# ---------------------------------------------------------------------------
# Dimension 1: Structural completeness
# ---------------------------------------------------------------------------

def _validate_structural_completeness(evidence_dir: Path) -> dict[str, Any]:
    """Validate structural completeness: files, rows, schema."""
    checks: dict[str, dict[str, Any]] = {}

    # Required files
    missing = []
    for fname in REQUIRED_EVIDENCE_FILES:
        if not (evidence_dir / fname).exists():
            missing.append(fname)
    if missing:
        checks["required_files"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": f"missing: {', '.join(missing)}",
            "expected": "all required files present",
            "reason": f"Missing required files: {', '.join(missing)}",
        }
    else:
        checks["required_files"] = {
            "status": _status_str(AuditStatus.PASS),
            "observed": f"{len(REQUIRED_EVIDENCE_FILES)} files present",
            "expected": f"{len(REQUIRED_EVIDENCE_FILES)} files",
            "reason": "All required evidence files present",
        }

    # Task-level rows
    cf_rows = _safe_load_jsonl(evidence_dir / "counterfactual_outcomes.jsonl")
    if len(cf_rows) > 0:
        checks["task_level_rows"] = {
            "status": _status_str(AuditStatus.PASS),
            "observed": f"{len(cf_rows)} counterfactual rows",
            "expected": ">0 task-level rows",
            "reason": "Task-level counterfactual rows present",
        }
    else:
        checks["task_level_rows"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": "0 counterfactual rows",
            "expected": ">0 task-level rows",
            "reason": "Missing task-level rows",
        }

    # Equivalence digests (counterfactual rows should have environment_snapshot_digest)
    if cf_rows:
        n_with_equiv = sum(
            1 for r in cf_rows
            if r.get("environment_snapshot_digest") or r.get("prompt_digest")
        )
        if n_with_equiv == len(cf_rows):
            checks["equivalence_digests"] = {
                "status": _status_str(AuditStatus.PASS),
                "observed": f"{n_with_equiv}/{len(cf_rows)} rows with digests",
                "expected": "all rows with equivalence digests",
                "reason": "Equivalence digests present in all rows",
            }
        else:
            checks["equivalence_digests"] = {
                "status": _status_str(AuditStatus.FAIL),
                "observed": f"{n_with_equiv}/{len(cf_rows)} rows with digests",
                "expected": "all rows with equivalence digests",
                "reason": "Missing equivalence digests in some rows",
            }
    else:
        checks["equivalence_digests"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": "0 rows",
            "expected": "all rows with equivalence digests",
            "reason": "No counterfactual rows to check for equivalence digests",
        }

    # Utility identity (utility_version present)
    has_utility = False
    if cf_rows:
        has_utility = any(r.get("utility_version") for r in cf_rows)
    if has_utility:
        checks["utility_identity"] = {
            "status": _status_str(AuditStatus.PASS),
            "observed": "utility_version present",
            "expected": "utility_version present",
            "reason": "Utility identity present in counterfactual rows",
        }
    else:
        checks["utility_identity"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": "utility_version missing",
            "expected": "utility_version present",
            "reason": "Missing utility identity",
        }

    # Policy lineage (source_provenance exists and has required fields)
    provenance = _safe_load_json(evidence_dir / "source_provenance.json")
    if isinstance(provenance, dict):
        has_lineage = bool(
            provenance.get("original_source_tree_sha256")
            or provenance.get("source_tree_sha256")
        )
        if has_lineage:
            checks["policy_lineage"] = {
                "status": _status_str(AuditStatus.PASS),
                "observed": "source provenance present",
                "expected": "complete policy lineage",
                "reason": "Policy lineage (source provenance) present",
            }
        else:
            checks["policy_lineage"] = {
                "status": _status_str(AuditStatus.FAIL),
                "observed": "source tree hash missing",
                "expected": "complete policy lineage",
                "reason": "Incomplete policy lineage",
            }
    else:
        checks["policy_lineage"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": "source_provenance missing",
            "expected": "complete policy lineage",
            "reason": "Missing policy lineage",
        }

    # Serialization parity (policies have policy_sha256)
    candidate = _safe_load_json(evidence_dir / "candidate_policy.json")
    sham = _safe_load_json(evidence_dir / "sham_policy.json")
    has_parity = (
        isinstance(candidate, dict) and bool(candidate.get("policy_sha256"))
        and isinstance(sham, dict) and bool(sham.get("policy_sha256"))
    )
    if has_parity:
        checks["serialization_parity"] = {
            "status": _status_str(AuditStatus.PASS),
            "observed": "both policies have policy_sha256",
            "expected": "serialization parity digests present",
            "reason": "Serialization parity digests present",
        }
    else:
        checks["serialization_parity"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": "missing policy_sha256",
            "expected": "serialization parity digests present",
            "reason": "Missing serialization parity",
        }

    # Split-access audit (split_manifest exists with splits)
    split = _safe_load_json(evidence_dir / "split_manifest.json")
    if isinstance(split, dict) and isinstance(split.get("splits"), dict) and split["splits"]:
        checks["split_access_audit"] = {
            "status": _status_str(AuditStatus.PASS),
            "observed": "split manifest with splits present",
            "expected": "split-access audit present",
            "reason": "Split-access audit present",
        }
    else:
        checks["split_access_audit"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": "split manifest missing or empty",
            "expected": "split-access audit present",
            "reason": "Missing split-access audit",
        }

    n_fail = sum(1 for c in checks.values() if c["status"] == _status_str(AuditStatus.FAIL))
    overall = _status_str(AuditStatus.PASS) if n_fail == 0 else _status_str(AuditStatus.FAIL)

    return {
        "status": overall,
        "checks": checks,
        "n_checks": len(checks),
        "n_fail": n_fail,
        "reason": f"Structural completeness: {overall} ({n_fail} failures)",
    }


# ---------------------------------------------------------------------------
# Dimension 2: Origin authenticity
# ---------------------------------------------------------------------------

def _validate_origin_authenticity(
    evidence_dir: Path,
    evidence_manifest: dict,
    provider_manifest: dict,
) -> dict[str, Any]:
    """Validate origin authenticity — must be REAL_MODEL for scientific claims."""
    checks: dict[str, dict[str, Any]] = {}

    result = classify_evidence_origin(provider_manifest, evidence_manifest)
    origin = result.origin

    # Must NOT be synthetic
    if origin == EvidenceOrigin.SYNTHETIC:
        checks["not_synthetic"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": "SYNTHETIC",
            "expected": "not SYNTHETIC",
            "reason": "Synthetic origin rejected for scientific evidence",
        }
    else:
        checks["not_synthetic"] = {
            "status": _status_str(AuditStatus.PASS),
            "observed": origin.value,
            "expected": "not SYNTHETIC",
            "reason": "Origin is not synthetic",
        }

    # Must NOT be simulated
    if origin == EvidenceOrigin.SIMULATED:
        checks["not_simulated"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": "SIMULATED",
            "expected": "not SIMULATED",
            "reason": "Simulated origin rejected for scientific evidence",
        }
    else:
        checks["not_simulated"] = {
            "status": _status_str(AuditStatus.PASS),
            "observed": origin.value,
            "expected": "not SIMULATED",
            "reason": "Origin is not simulated",
        }

    # Must NOT be infrastructure-only
    if origin == EvidenceOrigin.INFRASTRUCTURE:
        checks["not_infrastructure_only"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": "INFRASTRUCTURE",
            "expected": "not INFRASTRUCTURE-only",
            "reason": "Infrastructure-only origin rejected for scientific evidence",
        }
    else:
        checks["not_infrastructure_only"] = {
            "status": _status_str(AuditStatus.PASS),
            "observed": origin.value,
            "expected": "not INFRASTRUCTURE-only",
            "reason": "Origin is not infrastructure-only",
        }

    # Must be REAL_MODEL
    if origin == EvidenceOrigin.REAL_MODEL:
        checks["is_real_model"] = {
            "status": _status_str(AuditStatus.PASS),
            "observed": "REAL_MODEL",
            "expected": "REAL_MODEL",
            "reason": "Origin is REAL_MODEL",
        }
    else:
        checks["is_real_model"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": origin.value,
            "expected": "REAL_MODEL",
            "reason": f"Origin is {origin.value}, not REAL_MODEL",
        }

    # model_digest must be present
    model_digest = provider_manifest.get("model_digest")
    if model_digest:
        checks["model_digest"] = {
            "status": _status_str(AuditStatus.PASS),
            "observed": model_digest[:12] + "...",
            "expected": "non-null model_digest",
            "reason": "model_digest present",
        }
    else:
        checks["model_digest"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": "null",
            "expected": "non-null model_digest",
            "reason": "Missing model_digest",
        }

    # Historical source identity (commit SHA or source tree hash)
    provenance = _safe_load_json(evidence_dir / "source_provenance.json")
    if isinstance(provenance, dict):
        commit = provenance.get("original_commit_sha")
        source_hash = provenance.get("original_source_tree_sha256")
        if commit or source_hash:
            checks["historical_source_identity"] = {
                "status": _status_str(AuditStatus.PASS),
                "observed": f"commit={'yes' if commit else 'no'}, source_hash={'yes' if source_hash else 'no'}",
                "expected": "historical source identity present",
                "reason": "Historical source identity present",
            }
        else:
            checks["historical_source_identity"] = {
                "status": _status_str(AuditStatus.FAIL),
                "observed": "both missing",
                "expected": "historical source identity present",
                "reason": "Missing historical source identity",
            }
    else:
        checks["historical_source_identity"] = {
            "status": _status_str(AuditStatus.FAIL),
            "observed": "source_provenance missing",
            "expected": "historical source identity present",
            "reason": "Missing historical source identity",
        }

    n_fail = sum(1 for c in checks.values() if c["status"] == _status_str(AuditStatus.FAIL))
    overall = _status_str(AuditStatus.PASS) if n_fail == 0 else _status_str(AuditStatus.FAIL)

    return {
        "status": overall,
        "origin": origin.value,
        "checks": checks,
        "n_checks": len(checks),
        "n_fail": n_fail,
        "reason": f"Origin authenticity: {overall} ({n_fail} failures)",
    }


# ---------------------------------------------------------------------------
# Dimension 3: Scientific eligibility
# ---------------------------------------------------------------------------

def _validate_scientific_eligibility(
    structural: dict,
    origin_auth: dict,
    evidence_manifest: dict,
    provider_manifest: dict,
) -> dict[str, Any]:
    """Validate scientific eligibility — requires structural + origin + model_digest."""
    checks: dict[str, dict[str, Any]] = {}

    # Structural must pass
    checks["structural_complete"] = {
        "status": structural["status"],
        "observed": structural["status"],
        "expected": "PASS",
        "reason": "Structural completeness must pass for scientific eligibility",
    }

    # Origin must be REAL_MODEL
    checks["origin_real_model"] = {
        "status": origin_auth["status"],
        "observed": origin_auth.get("origin", "UNKNOWN"),
        "expected": "REAL_MODEL",
        "reason": "Origin must be REAL_MODEL for scientific eligibility",
    }

    # model_digest must be present
    model_digest = provider_manifest.get("model_digest")
    checks["model_digest_present"] = {
        "status": _status_str(AuditStatus.PASS) if model_digest else _status_str(AuditStatus.FAIL),
        "observed": "present" if model_digest else "missing",
        "expected": "present",
        "reason": "model_digest required for scientific eligibility",
    }

    # Evidence manifest must declare scientific_claim_eligible=True
    sci_eligible = evidence_manifest.get("scientific_claim_eligible", False)
    checks["manifest_scientific_eligible"] = {
        "status": _status_str(AuditStatus.PASS) if sci_eligible else _status_str(AuditStatus.FAIL),
        "observed": sci_eligible,
        "expected": True,
        "reason": "Evidence manifest must declare scientific_claim_eligible=True",
    }

    # Determine structure level for classify_evidence
    if structural["status"] == _status_str(AuditStatus.PASS):
        structure_level = EvidenceStructureLevel.ACTION_LEVEL_COMPLETE
    else:
        structure_level = EvidenceStructureLevel.NONE

    origin = EvidenceOrigin.UNKNOWN
    try:
        origin = EvidenceOrigin(origin_auth.get("origin", "UNKNOWN"))
    except ValueError:
        origin = EvidenceOrigin.UNKNOWN

    classification = classify_evidence(
        structure_level=structure_level,
        origin=origin,
        structural_pass=(structural["status"] == _status_str(AuditStatus.PASS)),
    )

    checks["classification"] = {
        "status": _status_str(AuditStatus.PASS) if classification.scientific_claim_eligible else _status_str(AuditStatus.FAIL),
        "observed": classification.to_dict(),
        "expected": "scientific_claim_eligible=True",
        "reason": classification.reason,
    }

    n_fail = sum(1 for c in checks.values() if c["status"] == _status_str(AuditStatus.FAIL))
    overall = _status_str(AuditStatus.PASS) if n_fail == 0 else _status_str(AuditStatus.FAIL)

    return {
        "status": overall,
        "scientific_claim_eligible": n_fail == 0,
        "checks": checks,
        "n_checks": len(checks),
        "n_fail": n_fail,
        "reason": f"Scientific eligibility: {overall} ({n_fail} failures)",
    }


# ---------------------------------------------------------------------------
# Dimension 4: Gate A eligibility
# ---------------------------------------------------------------------------

def _validate_gate_a_eligibility(
    scientific: dict,
    evidence_manifest: dict,
    provider_manifest: dict,
) -> dict[str, Any]:
    """Validate Gate A eligibility — requires scientific eligibility + action-level structure."""
    checks: dict[str, dict[str, Any]] = {}

    # Scientific eligibility must pass
    checks["scientific_eligible"] = {
        "status": scientific["status"],
        "observed": scientific["status"],
        "expected": "PASS",
        "reason": "Scientific eligibility must pass for Gate A",
    }

    # Provider must support Gate A
    supports_gate_a = provider_manifest.get("supports_gate_a", False)
    checks["provider_supports_gate_a"] = {
        "status": _status_str(AuditStatus.PASS) if supports_gate_a else _status_str(AuditStatus.FAIL),
        "observed": supports_gate_a,
        "expected": True,
        "reason": "Provider must support Gate A",
    }

    # Evidence manifest must not be synthetic
    origin = str(evidence_manifest.get("evidence_origin", "")).upper()
    checks["not_synthetic_origin"] = {
        "status": _status_str(AuditStatus.PASS) if origin != "SYNTHETIC" else _status_str(AuditStatus.FAIL),
        "observed": origin,
        "expected": "not SYNTHETIC",
        "reason": "Evidence origin must not be SYNTHETIC for Gate A",
    }

    n_fail = sum(1 for c in checks.values() if c["status"] == _status_str(AuditStatus.FAIL))
    overall = _status_str(AuditStatus.PASS) if n_fail == 0 else _status_str(AuditStatus.FAIL)

    return {
        "status": overall,
        "gate_a_eligible": n_fail == 0,
        "checks": checks,
        "n_checks": len(checks),
        "n_fail": n_fail,
        "reason": f"Gate A eligibility: {overall} ({n_fail} failures)",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_scientific_evidence(evidence_dir: str) -> dict:
    """Validate evidence for scientific claims.

    Returns a dict with four independent dimensions:
        structural_completeness: dict
        origin_authenticity: dict
        scientific_eligibility: dict
        gate_a_eligibility: dict

    Does NOT produce a single aggregate PASS/FAIL.
    """
    evidence_path = Path(evidence_dir)

    evidence_manifest = _safe_load_json(evidence_path / "EVIDENCE_MANIFEST.json") or {}
    provider_manifest = _safe_load_json(evidence_path / "provider_manifest.json") or {}

    structural = _validate_structural_completeness(evidence_path)
    origin_auth = _validate_origin_authenticity(
        evidence_path, evidence_manifest, provider_manifest
    )
    scientific = _validate_scientific_eligibility(
        structural, origin_auth, evidence_manifest, provider_manifest
    )
    gate_a = _validate_gate_a_eligibility(
        scientific, evidence_manifest, provider_manifest
    )

    return {
        "structural_completeness": structural,
        "origin_authenticity": origin_auth,
        "scientific_eligibility": scientific,
        "gate_a_eligibility": gate_a,
    }
