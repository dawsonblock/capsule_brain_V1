"""Evidence-origin enforcement for v0.4.7.

Supported evidence classes: REAL_MODEL, SYNTHETIC, UNAVAILABLE

Rules:
- Synthetic evidence may test pipeline correctness but may NEVER satisfy
  a real-model causal efficacy claim.
- Unavailable evidence may preserve historical metadata only.
- Reports must label synthetic results prominently.
- Mixed-origin runs must be rejected unless explicitly supported.
- Evidence origin must propagate into every derived artifact.
- Promotion must require REAL_MODEL.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class EvidenceOrigin(str, Enum):
    REAL_MODEL = "REAL_MODEL"
    SYNTHETIC = "SYNTHETIC"
    UNAVAILABLE = "UNAVAILABLE"


def require_scientific_evidence(
    manifest: dict[str, Any],
    *,
    required_origin: EvidenceOrigin = EvidenceOrigin.REAL_MODEL,
) -> None:
    """Raise ValueError if the evidence manifest does not satisfy the required origin.

    This is the enforcement point for the scientific rule:
    "Synthetic evidence may never satisfy a real-model causal efficacy claim."
    """
    actual = str(manifest.get("evidence_origin", "")).upper()
    required = required_origin.value

    if actual != required:
        raise ValueError(
            f"Required evidence origin '{required}' but got '{actual}'. "
            f"Synthetic evidence cannot satisfy real-model causal efficacy claims."
        )

    # Additional check: if REAL_MODEL is required, verify the manifest
    # actually has model identity fields.
    if required == EvidenceOrigin.REAL_MODEL.value:
        model_id = manifest.get("model_id", "")
        if not model_id or model_id.startswith("synthetic"):
            raise ValueError(
                f"Evidence origin is REAL_MODEL but model_id is '{model_id}'. "
                f"Synthetic model IDs cannot satisfy real-model claims."
            )


def can_promote(evidence_origin: str, require_real_model: bool = True) -> bool:
    """Check if evidence of this origin can be used for promotion."""
    origin = str(evidence_origin).upper()
    if require_real_model:
        return origin == EvidenceOrigin.REAL_MODEL.value
    return origin in (EvidenceOrigin.REAL_MODEL.value, EvidenceOrigin.SYNTHETIC.value)


def label_for_report(evidence_origin: str) -> str:
    """Return the prominent label for reports."""
    origin = str(evidence_origin).upper()
    if origin == EvidenceOrigin.SYNTHETIC.value:
        return "SYNTHETIC EVIDENCE — PIPELINE VALIDATION ONLY — NOT A REAL-MODEL RESULT"
    if origin == EvidenceOrigin.UNAVAILABLE.value:
        return "UNVERIFIED HISTORICAL CLAIM — SOURCE ARTIFACTS NOT INCLUDED"
    if origin == EvidenceOrigin.REAL_MODEL.value:
        return "REAL-MODEL EVIDENCE"
    return f"UNKNOWN EVIDENCE ORIGIN: {evidence_origin}"
