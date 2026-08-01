"""Gate A4 — Promotion eligibility for v0.4.7.

Determines whether a candidate policy is eligible for shadow or active
promotion based on the full gate hierarchy (A0-A3) plus safety,
artifact-binding, and evidence-origin checks.
"""
from __future__ import annotations

from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.v047.config import PromotionConfig
from qualification.autolearn_v04.v047.gate_schema import GateA4Result


def _is_pass(status: str) -> bool:
    return AuditStatus.from_str(status) == AuditStatus.PASS


def evaluate_gate_a4(
    gate_a0_status: str,
    gate_a1_status: str,
    gate_a2_status: str,
    gate_a3_status: str,
    safety_status: str,
    artifact_binding_status: str,
    evidence_origin: str,
    config: PromotionConfig,
) -> GateA4Result:
    """Evaluate promotion eligibility.

    Promotion requires:
    - Gate A0 = PASS
    - Gate A1 = PASS
    - Gate A2 = PASS
    - Gate A3 = PASS
    - Safety = PASS
    - Artifact binding = PASS
    - Evidence origin = REAL_MODEL (if require_real_model_evidence)

    Two promotion modes:
    - SHADOW_ELIGIBLE: all gates pass -> shadow_eligible=True
    - ACTIVE_ELIGIBLE: requires post-shadow validation (not checked here)

    Default: shadow_eligible if all gates pass, active_eligible=False
    """
    blocking_reasons: list[str] = []

    if not _is_pass(gate_a0_status):
        blocking_reasons.append(f"Gate A0 not PASS (got {gate_a0_status})")
    if not _is_pass(gate_a1_status):
        blocking_reasons.append(f"Gate A1 not PASS (got {gate_a1_status})")
    if not _is_pass(gate_a2_status):
        blocking_reasons.append(f"Gate A2 not PASS (got {gate_a2_status})")
    if not _is_pass(gate_a3_status):
        blocking_reasons.append(f"Gate A3 not PASS (got {gate_a3_status})")
    if not _is_pass(safety_status):
        blocking_reasons.append(f"Safety not PASS (got {safety_status})")
    if not _is_pass(artifact_binding_status):
        blocking_reasons.append(
            f"Artifact binding not PASS (got {artifact_binding_status})"
        )

    if config.require_real_model_evidence:
        origin = str(evidence_origin or "").upper()
        if origin != "REAL_MODEL":
            blocking_reasons.append(
                f"evidence_origin is {evidence_origin!r}, required REAL_MODEL"
            )

    shadow_eligible = len(blocking_reasons) == 0
    # Active eligibility requires post-shadow validation, not checked here.
    active_eligible = False

    status = AuditStatus.PASS.value if shadow_eligible else AuditStatus.FAIL.value

    return GateA4Result(
        status=status,
        shadow_eligible=shadow_eligible,
        active_eligible=active_eligible,
        blocking_reasons=blocking_reasons,
    )
