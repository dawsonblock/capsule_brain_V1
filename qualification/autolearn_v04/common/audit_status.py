"""Canonical audit status enum for v0.4.6.

Every audit stage must return this status. Gate A0 must propagate source
statuses directly.

Status semantics:
    PASS: check executed and satisfied
    FAIL: check executed and condition violated
    BLOCKED: required evidence missing, cannot check
    NOT_RUN: stage not attempted
    INVALID: evidence is invalid or misclassified
    DIAGNOSTIC_ONLY: result is diagnostic only, not a gate decision
    NOT_APPLICABLE: check does not apply to this evidence type
"""
from __future__ import annotations

from enum import Enum


class AuditStatus(str, Enum):
    """Canonical audit status."""
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"
    INVALID = "INVALID"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    NOT_APPLICABLE = "NOT_APPLICABLE"

    @classmethod
    def from_str(cls, value: str) -> "AuditStatus":
        """Parse a string into an AuditStatus, defaulting to NOT_RUN."""
        if not isinstance(value, str):
            return cls.NOT_RUN
        try:
            return cls(value.upper())
        except ValueError:
            return cls.NOT_RUN

    def is_blocking(self) -> bool:
        """Return True if this status blocks the overall gate."""
        return self in (AuditStatus.FAIL, AuditStatus.BLOCKED, AuditStatus.INVALID)

    def is_passing(self) -> bool:
        """Return True if this status is considered passing."""
        return self in (AuditStatus.PASS, AuditStatus.NOT_APPLICABLE, AuditStatus.DIAGNOSTIC_ONLY)
