"""Gate A0 — Evidence admissibility (v0.4.7 wrapper).

Gate A0 determines whether a run is scientifically evaluable.
It does NOT evaluate whether the candidate policy is useful.

This module wraps the existing v0.4.6 Gate A0 evaluation (24 sub-gates)
and returns a v0.4.7 GateResult.
"""
from __future__ import annotations

from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.v047.gate_schema import GateResult


def evaluate_gate_a0_v047(
    gate_a0_v046_result: dict[str, Any],
    evidence_origin: str,
) -> GateResult:
    """Wrap the v0.4.6 Gate A0 result into a v0.4.7 GateResult.

    The v0.4.6 result has:
    - status: PASS | FAIL | BLOCKED
    - n_sub_gates: int
    - n_pass, n_fail, n_blocked, n_not_applicable: int
    - sub_gates: dict[sub_gate_name, sub_gate_result]
    - overall_reason: str

    This wrapper preserves all of that information and adds the
    evidence origin for context.
    """
    status = gate_a0_v046_result.get("status", AuditStatus.BLOCKED.value)
    n_pass = gate_a0_v046_result.get("n_pass", 0)
    n_fail = gate_a0_v046_result.get("n_fail", 0)
    n_blocked = gate_a0_v046_result.get("n_blocked", 0)
    n_not_applicable = gate_a0_v046_result.get("n_not_applicable", 0)
    overall_reason = gate_a0_v046_result.get("overall_reason", "")
    sub_gates = gate_a0_v046_result.get("sub_gates", {})

    reasons: list[str] = []
    if status != AuditStatus.PASS.value:
        reasons.append(overall_reason or f"Gate A0 status is {status}")
        # Add specific failure reasons
        for sg_name, sg_result in sub_gates.items():
            sg_status = sg_result.get("status", "")
            if sg_status in (AuditStatus.FAIL.value, AuditStatus.BLOCKED.value):
                reasons.append(f"{sg_name}: {sg_result.get('reason', sg_status)}")

    checks = {
        "n_sub_gates": gate_a0_v046_result.get("n_sub_gates", 24),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_blocked": n_blocked,
        "n_not_applicable": n_not_applicable,
        "sub_gates": sub_gates,
        "evidence_origin": evidence_origin,
    }

    return GateResult(
        gate_name="gate_a0_admissibility",
        status=status,
        reasons=reasons,
        checks=checks,
    )
