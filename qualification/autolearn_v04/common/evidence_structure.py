"""Evidence structure level — independent of origin.

Structure level measures how granular the data is:
    NONE = 0: no evidence
    AGGREGATE_ONLY = 1: only summary statistics
    TASK_LEVEL = 2: per-task rows
    ACTION_LEVEL_COMPLETE = 3: all eligible actions per task

Origin measures authenticity:
    INFRASTRUCTURE / SYNTHETIC / SIMULATED / REAL_MODEL / UNKNOWN

These are independent dimensions. A synthetic fixture can have
ACTION_LEVEL_COMPLETE structure but SYNTHETIC origin.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from qualification.autolearn_v04.common.evidence_origin import EvidenceOrigin


class EvidenceStructureLevel(IntEnum):
    """Evidence structure granularity. Higher = more granular."""
    NONE = 0
    AGGREGATE_ONLY = 1
    TASK_LEVEL = 2
    ACTION_LEVEL_COMPLETE = 3


@dataclass(frozen=True)
class EvidenceClassification:
    """Two-dimensional evidence classification."""
    structure_level: EvidenceStructureLevel
    origin: EvidenceOrigin
    structural_status: str  # "PASS" | "FAIL" | "BLOCKED"
    scientific_claim_eligible: bool
    gate_a_eligible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure_level": self.structure_level.name,
            "origin": self.origin.value,
            "structural_status": self.structural_status,
            "scientific_claim_eligible": self.scientific_claim_eligible,
            "gate_a_eligible": self.gate_a_eligible,
            "reason": self.reason,
        }


def classify_evidence(
    structure_level: EvidenceStructureLevel,
    origin: EvidenceOrigin,
    structural_pass: bool = True,
) -> EvidenceClassification:
    """Classify evidence along both structure and origin dimensions.

    A synthetic fixture with complete structure:
        structure_level = ACTION_LEVEL_COMPLETE
        origin = SYNTHETIC
        structural_status = PASS
        scientific_claim_eligible = False
        gate_a_eligible = False

    A real but incomplete run:
        structure_level = TASK_LEVEL
        origin = REAL_MODEL
        structural_status = BLOCKED
        scientific_claim_eligible = False
        gate_a_eligible = False

    A valid scientific run:
        structure_level = ACTION_LEVEL_COMPLETE
        origin = REAL_MODEL
        structural_status = PASS
        scientific_claim_eligible = True
        gate_a_eligible = True
    """
    if not structural_pass:
        structural_status = "BLOCKED"
    elif structure_level == EvidenceStructureLevel.NONE:
        structural_status = "BLOCKED"
    else:
        structural_status = "PASS"

    # Scientific eligibility requires REAL_MODEL origin AND complete structure
    scientific_eligible = (
        origin == EvidenceOrigin.REAL_MODEL
        and structural_status == "PASS"
        and structure_level >= EvidenceStructureLevel.TASK_LEVEL
    )

    # Gate A eligibility requires REAL_MODEL origin AND action-level structure
    gate_a_eligible = (
        origin == EvidenceOrigin.REAL_MODEL
        and structural_status == "PASS"
        and structure_level >= EvidenceStructureLevel.ACTION_LEVEL_COMPLETE
    )

    if origin != EvidenceOrigin.REAL_MODEL:
        reason = f"structure={structure_level.name}, origin={origin.value} — not scientifically claim-eligible"
    elif structural_status != "PASS":
        reason = f"structure={structure_level.name}, origin=REAL_MODEL — structure incomplete"
    elif not gate_a_eligible:
        reason = f"structure={structure_level.name}, origin=REAL_MODEL — needs ACTION_LEVEL_COMPLETE for Gate A"
    else:
        reason = f"structure={structure_level.name}, origin=REAL_MODEL — scientifically claim-eligible"

    return EvidenceClassification(
        structure_level=structure_level,
        origin=origin,
        structural_status=structural_status,
        scientific_claim_eligible=scientific_eligible,
        gate_a_eligible=gate_a_eligible,
        reason=reason,
    )
