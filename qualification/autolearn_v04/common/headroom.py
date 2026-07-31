"""Canonical recovered-headroom calculation.

This is the SINGLE canonical implementation. No other module may
reimplement this formula.

The formula is::

    R = (candidate_mean - reference_mean) / (oracle_mean - reference_mean)

Key rules:
- Uses AGGREGATE means, NOT task-level ratios.
- Task-level ratios are never averaged.
- Negative or zero denominator → INVALID (returns None).
- Fraction and percent are separate fields, never confused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any


# Minimum denominator below which headroom is invalid.
MINIMUM_DENOMINATOR = 1e-10


@dataclass(frozen=True)
class RecoveredHeadroomResult:
    """Canonical result of recovered-headroom calculation."""

    status: str  # "VALID" | "INVALID_ORACLE_RELATION" | "NON_FINITE"
    candidate_mean: float
    reference_mean: float
    oracle_mean: float
    numerator: float
    denominator: float
    recovered_fraction: float | None
    recovered_percent: float | None
    task_count: int
    task_ids_digest: str
    reference_policy_id: str
    reference_policy_sha256: str
    candidate_policy_sha256: str
    oracle_artifact_sha256: str
    utility_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_recovered_headroom(
    candidate_mean: float,
    reference_mean: float,
    oracle_mean: float,
    *,
    task_count: int = 0,
    task_ids_digest: str = "",
    reference_policy_id: str = "",
    reference_policy_sha256: str = "",
    candidate_policy_sha256: str = "",
    oracle_artifact_sha256: str = "",
    utility_version: str = "normalized_v1",
) -> RecoveredHeadroomResult:
    """Compute recovered headroom using the canonical aggregate formula.

    R = (candidate_mean - reference_mean) / (oracle_mean - reference_mean)

    Returns INVALID_ORACLE_RELATION when denominator <= MINIMUM_DENOMINATOR.
    A negative oracle-minus-reference indicates:
    - wrong task set
    - incomplete action matrix
    - utility inconsistency
    - incorrect oracle
    - data mismatch
    """
    numerator = candidate_mean - reference_mean
    denominator = oracle_mean - reference_mean

    # Check finiteness.
    vals = [candidate_mean, reference_mean, oracle_mean]
    if not all(_is_finite(v) for v in vals):
        return RecoveredHeadroomResult(
            status="NON_FINITE",
            candidate_mean=candidate_mean,
            reference_mean=reference_mean,
            oracle_mean=oracle_mean,
            numerator=numerator,
            denominator=denominator,
            recovered_fraction=None,
            recovered_percent=None,
            task_count=task_count,
            task_ids_digest=task_ids_digest,
            reference_policy_id=reference_policy_id,
            reference_policy_sha256=reference_policy_sha256,
            candidate_policy_sha256=candidate_policy_sha256,
            oracle_artifact_sha256=oracle_artifact_sha256,
            utility_version=utility_version,
        )

    # Reject nonpositive denominator.
    # Do NOT use abs(denominator) — a negative oracle-minus-reference
    # is an integrity error, not a valid headroom calculation.
    if denominator <= MINIMUM_DENOMINATOR:
        return RecoveredHeadroomResult(
            status="INVALID_ORACLE_RELATION",
            candidate_mean=candidate_mean,
            reference_mean=reference_mean,
            oracle_mean=oracle_mean,
            numerator=numerator,
            denominator=denominator,
            recovered_fraction=None,
            recovered_percent=None,
            task_count=task_count,
            task_ids_digest=task_ids_digest,
            reference_policy_id=reference_policy_id,
            reference_policy_sha256=reference_policy_sha256,
            candidate_policy_sha256=candidate_policy_sha256,
            oracle_artifact_sha256=oracle_artifact_sha256,
            utility_version=utility_version,
        )

    fraction = numerator / denominator
    percent = fraction * 100.0

    return RecoveredHeadroomResult(
        status="VALID",
        candidate_mean=candidate_mean,
        reference_mean=reference_mean,
        oracle_mean=oracle_mean,
        numerator=numerator,
        denominator=denominator,
        recovered_fraction=fraction,
        recovered_percent=percent,
        task_count=task_count,
        task_ids_digest=task_ids_digest,
        reference_policy_id=reference_policy_id,
        reference_policy_sha256=reference_policy_sha256,
        candidate_policy_sha256=candidate_policy_sha256,
        oracle_artifact_sha256=oracle_artifact_sha256,
        utility_version=utility_version,
    )


def _is_finite(v: Any) -> bool:
    """Check if a value is a finite number."""
    if v is None or isinstance(v, bool):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    import math
    return math.isfinite(f)


def compute_task_ids_digest(task_ids: list[str]) -> str:
    """Compute a stable SHA-256 digest over a sorted list of task IDs."""
    payload = json.dumps(sorted(task_ids), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()
