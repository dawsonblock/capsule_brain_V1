"""Canonical evidence-origin typing for v0.4.6.

Every evidence package has exactly one evidence origin that determines its
scientific eligibility, promotability and gate support.

Invariants enforced:
    SYNTHETIC: scientific_claim_eligible=False, promotable=False,
              supports_gate_a=False, supports_gate_b=False,
              runtime_type=synthetic, provider_class=SYNTHETIC
    SIMULATED: scientific_claim_eligible=False, promotable=False,
              runtime_type=simulated, provider_class=SIMULATED
    INFRASTRUCTURE: scientific_claim_eligible=False (unless a specific gate
              permits only infrastructure claims), cannot support scientific
              Gate A
    REAL_MODEL: scientific_claim_eligible may be True only after model identity,
              task-level execution evidence and provenance pass
    UNKNOWN: fail closed
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EvidenceOrigin(str, Enum):
    """Canonical evidence origin types."""
    REAL_MODEL = "REAL_MODEL"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    SIMULATED = "SIMULATED"
    SYNTHETIC = "SYNTHETIC"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EvidenceOriginResult:
    """Result of evidence-origin classification."""
    origin: EvidenceOrigin
    runtime_type: str
    model_id: str
    tokenizer_id: str
    provider_class: str
    scientific_claim_eligible: bool
    promotable: bool
    supports_gate_a: bool
    supports_gate_b: bool
    status: str  # "PASS" | "FAIL" | "BLOCKED"
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin.value,
            "runtime_type": self.runtime_type,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "provider_class": self.provider_class,
            "scientific_claim_eligible": self.scientific_claim_eligible,
            "promotable": self.promotable,
            "supports_gate_a": self.supports_gate_a,
            "supports_gate_b": self.supports_gate_b,
            "status": self.status,
            "reason": self.reason,
        }


def _is_synthetic_marker(value: str) -> bool:
    """Check if a string contains synthetic markers."""
    if not isinstance(value, str):
        return False
    lower = value.lower()
    return "synthetic" in lower


def classify_evidence_origin(
    provider_manifest: dict[str, Any],
    evidence_manifest: dict[str, Any] | None = None,
) -> EvidenceOriginResult:
    """Classify the evidence origin from provider and evidence manifests.

    Enforces canonical invariants and anti-misclassification rules.
    """
    evidence_manifest = evidence_manifest or {}
    provider_class = str(provider_manifest.get("provider_class", "")).upper()
    runtime_type = str(provider_manifest.get("runtime_type", "")).lower()
    model_id = str(provider_manifest.get("model_id", ""))
    tokenizer_id = str(provider_manifest.get("tokenizer_id", ""))
    declared_origin = str(evidence_manifest.get("evidence_origin", "")).upper()

    # --- Detect origin ---
    origin = EvidenceOrigin.UNKNOWN

    if declared_origin:
        try:
            origin = EvidenceOrigin(declared_origin)
        except ValueError:
            origin = EvidenceOrigin.UNKNOWN

    # Override based on provider_class if it's more specific
    if provider_class == "SYNTHETIC" or _is_synthetic_marker(model_id) or _is_synthetic_marker(tokenizer_id):
        origin = EvidenceOrigin.SYNTHETIC
    elif provider_class == "SIMULATED" or runtime_type == "simulated":
        origin = EvidenceOrigin.SIMULATED
    elif provider_class == "REAL_MODEL":
        if _is_synthetic_marker(model_id) or _is_synthetic_marker(tokenizer_id):
            origin = EvidenceOrigin.SYNTHETIC  # misclassified
        else:
            origin = EvidenceOrigin.REAL_MODEL
    elif provider_class == "INFRASTRUCTURE":
        origin = EvidenceOrigin.INFRASTRUCTURE

    # --- Enforce invariants ---
    if origin == EvidenceOrigin.SYNTHETIC:
        return EvidenceOriginResult(
            origin=EvidenceOrigin.SYNTHETIC,
            runtime_type="synthetic",
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            provider_class="SYNTHETIC",
            scientific_claim_eligible=False,
            promotable=False,
            supports_gate_a=False,
            supports_gate_b=False,
            status="PASS",
            reason="synthetic origin correctly classified",
        )

    if origin == EvidenceOrigin.SIMULATED:
        return EvidenceOriginResult(
            origin=EvidenceOrigin.SIMULATED,
            runtime_type="simulated",
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            provider_class="SIMULATED",
            scientific_claim_eligible=False,
            promotable=False,
            supports_gate_a=False,
            supports_gate_b=False,
            status="PASS",
            reason="simulated origin correctly classified",
        )

    if origin == EvidenceOrigin.INFRASTRUCTURE:
        return EvidenceOriginResult(
            origin=EvidenceOrigin.INFRASTRUCTURE,
            runtime_type=runtime_type or "infrastructure",
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            provider_class="INFRASTRUCTURE",
            scientific_claim_eligible=False,
            promotable=False,
            supports_gate_a=False,
            supports_gate_b=False,
            status="PASS",
            reason="infrastructure origin correctly classified",
        )

    if origin == EvidenceOrigin.REAL_MODEL:
        # Check for misclassification
        if _is_synthetic_marker(model_id) or _is_synthetic_marker(tokenizer_id):
            return EvidenceOriginResult(
                origin=EvidenceOrigin.SYNTHETIC,
                runtime_type="synthetic",
                model_id=model_id,
                tokenizer_id=tokenizer_id,
                provider_class="SYNTHETIC",
                scientific_claim_eligible=False,
                promotable=False,
                supports_gate_a=False,
                supports_gate_b=False,
                status="FAIL",
                reason="misclassification: provider_class=REAL_MODEL but model_id or tokenizer_id contains 'synthetic'",
            )
        # Real model — check if model_digest is present for scientific eligibility
        model_digest = provider_manifest.get("model_digest")
        scientific_eligible = bool(model_digest)
        return EvidenceOriginResult(
            origin=EvidenceOrigin.REAL_MODEL,
            runtime_type="real",
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            provider_class="REAL_MODEL",
            scientific_claim_eligible=scientific_eligible,
            promotable=False,  # only after full validation
            supports_gate_a=scientific_eligible,
            supports_gate_b=False,
            status="PASS" if scientific_eligible else "BLOCKED",
            reason="real-model origin; scientific eligibility requires model_digest and full validation" if not scientific_eligible else "real-model origin with model digest",
        )

    # UNKNOWN — fail closed
    return EvidenceOriginResult(
        origin=EvidenceOrigin.UNKNOWN,
        runtime_type=runtime_type,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        provider_class=provider_class,
        scientific_claim_eligible=False,
        promotable=False,
        supports_gate_a=False,
        supports_gate_b=False,
        status="FAIL",
        reason="unknown evidence origin — failing closed",
    )


def check_anti_misclassification(
    provider_manifest: dict[str, Any],
    evidence_manifest: dict[str, Any],
) -> list[str]:
    """Check anti-misclassification rules. Returns list of violations."""
    violations: list[str] = []
    provider_class = str(provider_manifest.get("provider_class", "")).upper()
    runtime_type = str(provider_manifest.get("runtime_type", "")).lower()
    model_id = str(provider_manifest.get("model_id", "")).lower()
    tokenizer_id = str(provider_manifest.get("tokenizer_id", "")).lower()
    evidence_origin = str(evidence_manifest.get("evidence_origin", "")).upper()
    scientific_claim_eligible = evidence_manifest.get("scientific_claim_eligible", False)
    supports_gate_a = evidence_manifest.get("supports_gate_a", False)
    promotable = evidence_manifest.get("promotable", False)

    if provider_class == "REAL_MODEL" and "synthetic" in model_id:
        violations.append("provider_class=REAL_MODEL but model_id contains 'synthetic'")
    if provider_class == "REAL_MODEL" and "synthetic" in tokenizer_id:
        violations.append("provider_class=REAL_MODEL but tokenizer_id contains 'synthetic'")
    if runtime_type == "real" and evidence_origin == "SYNTHETIC":
        violations.append("runtime_type=real but evidence_origin=SYNTHETIC")
    if scientific_claim_eligible and evidence_origin != "REAL_MODEL":
        violations.append(f"scientific_claim_eligible=True but evidence_origin={evidence_origin}")
    if supports_gate_a and evidence_origin != "REAL_MODEL":
        violations.append(f"supports_gate_a=True but evidence_origin={evidence_origin}")
    if promotable and evidence_origin != "REAL_MODEL":
        violations.append(f"promotable=True but evidence_origin={evidence_origin}")
    if provider_class == "REAL_MODEL" and not provider_manifest.get("model_digest"):
        violations.append("provider_class=REAL_MODEL but model_digest is missing")

    return violations
