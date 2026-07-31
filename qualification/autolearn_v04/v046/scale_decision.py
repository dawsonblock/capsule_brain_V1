"""Model-scale decision logic for v0.4.6.

Determines whether evidence is eligible for the scale-decision tree and,
if so, what scale decision to make.

Rules:
    - Synthetic fixture: decision = NOT_APPLICABLE
    - Missing historical evidence: decision = BLOCKED
    - Real evidence with failed Gate A0: decision = BLOCKED
    - Only valid real evidence with PASS Gate A0 may enter the
      scale-decision tree

A synthetic fixture must NEVER produce:
    APPROVE_MATCHED_14B_PILOT
    APPROVE_FULL_14B_RUN
    NO_SCALE_NEEDED_FOR_GATE_A
"""
from __future__ import annotations

from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus


# Scale decision constants
DECISION_NOT_APPLICABLE = "NOT_APPLICABLE"
DECISION_BLOCKED = "BLOCKED"
DECISION_APPROVE_MATCHED_14B_PILOT = "APPROVE_MATCHED_14B_PILOT"
DECISION_APPROVE_FULL_14B_RUN = "APPROVE_FULL_14B_RUN"
DECISION_NO_SCALE_NEEDED_FOR_GATE_A = "NO_SCALE_NEEDED_FOR_GATE_A"

# Decisions that a synthetic fixture must NEVER produce
_FORBIDDEN_FOR_SYNTHETIC = frozenset({
    DECISION_APPROVE_MATCHED_14B_PILOT,
    DECISION_APPROVE_FULL_14B_RUN,
    DECISION_NO_SCALE_NEEDED_FOR_GATE_A,
})


def make_scale_decision(
    gate_a0_status: str,
    evidence_origin: str,
    gate_a_status: str = "BLOCKED",
) -> dict[str, Any]:
    """Determine the model-scale decision.

    Args:
        gate_a0_status: Status of Gate A0 (PASS, FAIL, BLOCKED, etc.)
        evidence_origin: Evidence origin (SYNTHETIC, REAL_MODEL, etc.)
        gate_a_status: Status of Gate A (defaults to BLOCKED)

    Returns:
        dict with decision, approve_matched_14b_pilot,
        approve_full_14b_run, reason
    """
    origin = str(evidence_origin).upper()
    a0 = AuditStatus.from_str(gate_a0_status)
    gate_a = AuditStatus.from_str(gate_a_status)

    # Default: no approvals
    approve_matched_14b_pilot = False
    approve_full_14b_run = False

    # --- Rule 1: Synthetic fixture -> NOT_APPLICABLE ---
    if origin == "SYNTHETIC":
        decision = DECISION_NOT_APPLICABLE
        reason = (
            "Synthetic fixture evidence is not eligible for scale decisions; "
            "decision=NOT_APPLICABLE"
        )
        return {
            "decision": decision,
            "approve_matched_14b_pilot": approve_matched_14b_pilot,
            "approve_full_14b_run": approve_full_14b_run,
            "reason": reason,
        }

    # --- Rule 2: Missing historical evidence -> BLOCKED ---
    # Non-REAL_MODEL origins (SIMULATED, INFRASTRUCTURE, UNKNOWN) lack
    # historical evidence.
    if origin != "REAL_MODEL":
        decision = DECISION_BLOCKED
        reason = (
            f"Evidence origin={origin} lacks historical evidence; "
            f"scale decision BLOCKED"
        )
        return {
            "decision": decision,
            "approve_matched_14b_pilot": approve_matched_14b_pilot,
            "approve_full_14b_run": approve_full_14b_run,
            "reason": reason,
        }

    # --- Rule 3: Real evidence with failed Gate A0 -> BLOCKED ---
    if a0 != AuditStatus.PASS:
        decision = DECISION_BLOCKED
        reason = (
            f"Real evidence but Gate A0 status={a0.value} (not PASS); "
            f"scale decision BLOCKED"
        )
        return {
            "decision": decision,
            "approve_matched_14b_pilot": approve_matched_14b_pilot,
            "approve_full_14b_run": approve_full_14b_run,
            "reason": reason,
        }

    # --- Rule 4: Only valid real evidence with PASS Gate A0 may enter
    # the scale-decision tree ---
    # Gate A must also pass for any scale approval
    if gate_a != AuditStatus.PASS:
        decision = DECISION_BLOCKED
        reason = (
            f"Gate A0 PASS but Gate A status={gate_a.value} (not PASS); "
            f"scale decision BLOCKED"
        )
        return {
            "decision": decision,
            "approve_matched_14b_pilot": approve_matched_14b_pilot,
            "approve_full_14b_run": approve_full_14b_run,
            "reason": reason,
        }

    # Gate A0 and Gate A both PASS with REAL_MODEL origin.
    # Enter the scale-decision tree.
    # For now, the default for a 7B model that passes is NO_SCALE_NEEDED.
    # A 14B pilot would require additional headroom evidence.
    decision = DECISION_NO_SCALE_NEEDED_FOR_GATE_A
    reason = (
        "Real evidence with PASS Gate A0 and PASS Gate A; "
        "no scale needed for current Gate A qualification"
    )

    # Safety check: ensure we never return a forbidden decision for synthetic
    # (this path is only reachable for REAL_MODEL, so this is a no-op guard)
    if origin == "SYNTHETIC" and decision in _FORBIDDEN_FOR_SYNTHETIC:
        decision = DECISION_NOT_APPLICABLE
        reason = "Safety override: synthetic fixture cannot receive scale approval"

    return {
        "decision": decision,
        "approve_matched_14b_pilot": approve_matched_14b_pilot,
        "approve_full_14b_run": approve_full_14b_run,
        "reason": reason,
    }
