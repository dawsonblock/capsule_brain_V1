"""Evidence origin audit for v0.4.6.

Audits the evidence origin from a directory by loading
EVIDENCE_MANIFEST.json and provider_manifest.json, classifying the
origin using the common ``classify_evidence_origin`` function, and
checking anti-misclassification rules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.common.evidence_origin import (
    EvidenceOrigin,
    classify_evidence_origin,
    check_anti_misclassification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def audit_evidence_origin(evidence_dir: str) -> dict[str, Any]:
    """Audit the evidence origin from a directory.

    Loads EVIDENCE_MANIFEST.json and provider_manifest.json, classifies
    the evidence origin, and checks anti-misclassification rules.

    Returns a dict with:
        origin, runtime_type, model_id, tokenizer_id, provider_class,
        scientific_claim_eligible, promotable, supports_gate_a,
        supports_gate_b, status, reason, misclassification_violations
    """
    evidence_path = Path(evidence_dir)

    evidence_manifest = _safe_load_json(evidence_path / "EVIDENCE_MANIFEST.json")
    provider_manifest = _safe_load_json(evidence_path / "provider_manifest.json")

    if not isinstance(evidence_manifest, dict):
        evidence_manifest = {}
    if not isinstance(provider_manifest, dict):
        provider_manifest = {}

    # Classify evidence origin
    result = classify_evidence_origin(provider_manifest, evidence_manifest)

    # Check anti-misclassification rules
    violations = check_anti_misclassification(provider_manifest, evidence_manifest)

    # Determine audit status
    if violations:
        status = AuditStatus.INVALID.value
        reason = (
            f"Evidence origin audit: {len(violations)} misclassification "
            f"violation(s) detected"
        )
    elif result.status == "FAIL":
        status = AuditStatus.FAIL.value
        reason = result.reason
    elif result.status == "BLOCKED":
        status = AuditStatus.BLOCKED.value
        reason = result.reason
    else:
        status = AuditStatus.PASS.value
        reason = result.reason

    return {
        "origin": result.origin.value,
        "runtime_type": result.runtime_type,
        "model_id": result.model_id,
        "tokenizer_id": result.tokenizer_id,
        "provider_class": result.provider_class,
        "scientific_claim_eligible": result.scientific_claim_eligible,
        "promotable": result.promotable,
        "supports_gate_a": result.supports_gate_a,
        "supports_gate_b": result.supports_gate_b,
        "status": status,
        "reason": reason,
        "misclassification_violations": violations,
    }
