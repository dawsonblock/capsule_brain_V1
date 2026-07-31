"""Typed audit-result schema for v0.4.6.

Every audit stage serializes this structure. Gate A0 consumes these objects
directly, preventing stale schema keys from silently producing zeros.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus


@dataclass(frozen=True)
class AuditResult:
    """Typed result of a single audit check."""
    gate_id: str
    name: str
    status: AuditStatus
    observed: Any
    expected: Any
    reason: str
    evidence_artifact: str | None = None
    evidence_field: str | None = None
    remediation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "name": self.name,
            "status": self.status.value,
            "observed": self.observed,
            "expected": self.expected,
            "reason": self.reason,
            "evidence_artifact": self.evidence_artifact,
            "evidence_field": self.evidence_field,
            "remediation": self.remediation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AuditResult":
        return cls(
            gate_id=d.get("gate_id", ""),
            name=d.get("name", ""),
            status=AuditStatus.from_str(d.get("status", "")),
            observed=d.get("observed"),
            expected=d.get("expected"),
            reason=d.get("reason", ""),
            evidence_artifact=d.get("evidence_artifact"),
            evidence_field=d.get("evidence_field"),
            remediation=d.get("remediation"),
        )


def make_pass(
    gate_id: str, name: str, observed: Any, expected: Any, reason: str = "",
    evidence_artifact: str | None = None, evidence_field: str | None = None,
) -> AuditResult:
    """Create a PASS AuditResult."""
    return AuditResult(
        gate_id=gate_id, name=name, status=AuditStatus.PASS,
        observed=observed, expected=expected, reason=reason,
        evidence_artifact=evidence_artifact, evidence_field=evidence_field,
    )


def make_fail(
    gate_id: str, name: str, observed: Any, expected: Any, reason: str = "",
    evidence_artifact: str | None = None, evidence_field: str | None = None,
    remediation: str | None = None,
) -> AuditResult:
    """Create a FAIL AuditResult."""
    return AuditResult(
        gate_id=gate_id, name=name, status=AuditStatus.FAIL,
        observed=observed, expected=expected, reason=reason,
        evidence_artifact=evidence_artifact, evidence_field=evidence_field,
        remediation=remediation,
    )


def make_blocked(
    gate_id: str, name: str, reason: str = "",
    evidence_artifact: str | None = None, evidence_field: str | None = None,
) -> AuditResult:
    """Create a BLOCKED AuditResult."""
    return AuditResult(
        gate_id=gate_id, name=name, status=AuditStatus.BLOCKED,
        observed=None, expected="evidence present", reason=reason,
        evidence_artifact=evidence_artifact, evidence_field=evidence_field,
    )


def make_not_applicable(
    gate_id: str, name: str, reason: str = "",
) -> AuditResult:
    """Create a NOT_APPLICABLE AuditResult."""
    return AuditResult(
        gate_id=gate_id, name=name, status=AuditStatus.NOT_APPLICABLE,
        observed=None, expected=None, reason=reason,
    )


def aggregate_status(results: list[AuditResult]) -> AuditStatus:
    """Aggregate a list of AuditResults into a single status.

    FAIL takes priority over BLOCKED. PASS only if all pass or are N/A.
    """
    if not results:
        return AuditStatus.NOT_RUN

    has_fail = False
    has_blocked = False
    has_not_run = False
    has_invalid = False

    for r in results:
        if r.status == AuditStatus.FAIL:
            has_fail = True
        elif r.status == AuditStatus.BLOCKED:
            has_blocked = True
        elif r.status == AuditStatus.NOT_RUN:
            has_not_run = True
        elif r.status == AuditStatus.INVALID:
            has_invalid = True

    if has_invalid:
        return AuditStatus.INVALID
    if has_fail:
        return AuditStatus.FAIL
    if has_blocked:
        return AuditStatus.BLOCKED
    if has_not_run:
        return AuditStatus.NOT_RUN
    return AuditStatus.PASS
