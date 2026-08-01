"""Structured gate result schema for v0.4.7.

Replaces the ambiguous ``gate_a_status: "PASS"`` with an explicit
hierarchy of gates, each with its own status, reasons, and checks.

Scientific rule: Gate A0 PASS does NOT imply Gate A2 PASS.
``legacy_gate_a_status`` is computed conservatively: it is PASS only
when both Gate A2 and Gate A3 pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

# Reuse the existing AuditStatus enum.
from qualification.autolearn_v04.common.audit_status import AuditStatus


@dataclass
class GateResult:
    """Result of a single gate evaluation."""

    gate_name: str
    status: str  # PASS | FAIL | BLOCKED | NOT_RUN
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return self.status == AuditStatus.PASS.value

    @property
    def blocked(self) -> bool:
        return self.status == AuditStatus.BLOCKED.value

    @property
    def failed(self) -> bool:
        return self.status == AuditStatus.FAIL.value


@dataclass
class ComparisonResult:
    """Statistical comparison between two policies."""

    comparison: str
    n_tasks: int = 0
    n_task_groups: int = 0
    mean_delta: float = 0.0
    median_delta: float = 0.0
    standard_error: float = 0.0
    confidence_level: float = 0.95
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    practical_threshold: float = 0.0
    passes: bool = False
    win_rate: float = 0.0
    tie_rate: float = 0.0
    loss_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateA1Result:
    """Gate A1 — Routing headroom."""

    status: str = AuditStatus.NOT_RUN.value
    oracle_vs_baseline: dict[str, Any] = field(default_factory=dict)
    oracle_vs_sham: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateA2Result:
    """Gate A2 — Candidate causal effectiveness."""

    status: str = AuditStatus.NOT_RUN.value
    candidate_vs_baseline: dict[str, Any] = field(default_factory=dict)
    candidate_vs_sham: dict[str, Any] = field(default_factory=dict)
    matched_pair_flip: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateA3Result:
    """Gate A3 — Robustness and replication."""

    status: str = AuditStatus.NOT_RUN.value
    replicate_summary: dict[str, Any] = field(default_factory=dict)
    family_summary: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateA4Result:
    """Gate A4 — Promotion eligibility."""

    status: str = AuditStatus.NOT_RUN.value
    shadow_eligible: bool = False
    active_eligible: bool = False
    blocking_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualificationVerdict:
    """Top-level structured verdict replacing ambiguous gate_a_status.

    This is the canonical v0.4.7 result object.  ``legacy_gate_a_status``
    is provided for backward compatibility and is computed conservatively:
    PASS only when both Gate A2 and Gate A3 pass.
    """

    protocol_version: str = "0.4.7"
    run_id: str = ""
    evidence_origin: str = ""

    # Individual gate results.
    gate_a0_admissibility: dict[str, Any] = field(default_factory=dict)
    gate_a1_headroom: dict[str, Any] = field(default_factory=dict)
    gate_a2_effectiveness: dict[str, Any] = field(default_factory=dict)
    gate_a3_robustness: dict[str, Any] = field(default_factory=dict)
    gate_a4_promotion: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    # --- Legacy compatibility ------------------------------------------------

    @property
    def legacy_gate_a_status(self) -> str:
        """Conservative legacy mapping.

        A Gate A0 pass must NEVER map to legacy Gate A success.
        Legacy PASS requires both Gate A2 and Gate A3 to pass.
        """
        a2_status = self.gate_a2_effectiveness.get("status", AuditStatus.NOT_RUN.value)
        a3_status = self.gate_a3_robustness.get("status", AuditStatus.NOT_RUN.value)
        if a2_status == AuditStatus.PASS.value and a3_status == AuditStatus.PASS.value:
            return "PASS"
        if (a2_status == AuditStatus.BLOCKED.value
                or a3_status == AuditStatus.BLOCKED.value):
            return "BLOCKED"
        return "FAIL"

    @property
    def shadow_eligible(self) -> bool:
        return self.gate_a4_promotion.get("shadow_eligible", False)

    @property
    def active_eligible(self) -> bool:
        return self.gate_a4_promotion.get("active_eligible", False)

    def to_machine_verdict(self) -> dict[str, Any]:
        """Machine-readable verdict per the spec."""
        return {
            "release": "2.15.11",
            "autolearn": "0.3.10",
            "qualification": "0.4.7",
            "protocol": "0.4.7",
            "gate_a0_admissibility": self.gate_a0_admissibility.get("status", "NOT_RUN"),
            "gate_a1_headroom": self.gate_a1_headroom.get("status", "NOT_RUN"),
            "gate_a2_effectiveness": self.gate_a2_effectiveness.get("status", "NOT_RUN"),
            "gate_a3_robustness": self.gate_a3_robustness.get("status", "NOT_RUN"),
            "gate_a4_promotion": self.gate_a4_promotion.get("status", "NOT_RUN"),
            "shadow_eligible": self.shadow_eligible,
            "active_eligible": self.active_eligible,
            "evidence_origin": self.evidence_origin,
            "legacy_gate_a_status": self.legacy_gate_a_status,
        }
