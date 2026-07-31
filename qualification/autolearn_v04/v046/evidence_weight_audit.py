"""Evidence weight audit for v0.4.6.

Audits evidence weights in normalized experience task rows and
action-level rows.  Reports **actual counts** — never 0 when rows
exist but quality fields are missing.

Key fix: if rows exist but quality fields (q_total, final_weight, etc.)
are missing, the audit reports status=FAIL with
reason=REQUIRED_WEIGHT_FIELDS_MISSING, and the row count reflects the
actual number of rows found.
"""
from __future__ import annotations

import math
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus

# Quality / weight fields that must be present on every task row.
QUALITY_FIELDS: tuple[str, ...] = (
    "q_verifier",
    "q_execution",
    "q_counterfactual",
    "q_isolation",
    "q_provenance",
    "q_total",
    "final_weight",
)

# Reason string used when rows exist but quality fields are missing.
REQUIRED_WEIGHT_FIELDS_MISSING = "REQUIRED_WEIGHT_FIELDS_MISSING"


def _is_finite(value: Any) -> bool:
    """Return True if value is a finite number."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _has_quality_fields(row: dict) -> bool:
    """Return True if the row contains all quality fields with finite values."""
    for field in QUALITY_FIELDS:
        val = row.get(field)
        if val is None or not _is_finite(val):
            return False
    return True


def audit_evidence_weights(
    experience_task_rows: list[dict],
    experience_action_rows: list[dict],
    candidate_policy: dict,
) -> dict:
    """Audit evidence weights in experience data.

    Parameters
    ----------
    experience_task_rows:
        Normalized task-level experience rows (output of
        ``normalize_experience_rows``).
    experience_action_rows:
        Action-level experience rows (raw input).
    candidate_policy:
        The candidate policy dict (for verifier registry lookup).

    Returns
    -------
    dict
        experience_task_rows_found, experience_action_rows_found,
        rows_containing_quality_fields, positive_final_weights,
        zero_weights, negative_weights, nonfinite_weights,
        unknown_verifiers, effective_total_weight, mean_q_total,
        min_q_total, max_q_total, status, reason
    """
    n_task_rows = len(experience_task_rows) if experience_task_rows else 0
    n_action_rows = len(experience_action_rows) if experience_action_rows else 0

    # --- BLOCKED: no rows at all ---
    if n_task_rows == 0 and n_action_rows == 0:
        return {
            "experience_task_rows_found": 0,
            "experience_action_rows_found": 0,
            "rows_containing_quality_fields": 0,
            "positive_final_weights": 0,
            "zero_weights": 0,
            "negative_weights": 0,
            "nonfinite_weights": 0,
            "unknown_verifiers": [],
            "effective_total_weight": 0.0,
            "mean_q_total": 0.0,
            "min_q_total": 0.0,
            "max_q_total": 0.0,
            "weights": [],
            "n_negative": 0,
            "n_zero": 0,
            "status": AuditStatus.BLOCKED.value,
            "reason": "no experience rows found (task or action level)",
        }

    # --- Determine which rows to audit ---
    # Prefer task-level rows; fall back to action-level if task rows absent.
    rows_to_audit = experience_task_rows if n_task_rows > 0 else experience_action_rows

    rows_with_quality = 0
    positive_weights = 0
    zero_weights = 0
    negative_weights = 0
    nonfinite_weights = 0
    unknown_verifiers: list[str] = []
    effective_total = 0.0
    q_totals: list[float] = []
    weights_list: list[float] = []

    # Collect known verifier names from candidate policy if available.
    known_verifiers: set[str] = set()
    if isinstance(candidate_policy, dict):
        vr = candidate_policy.get("verifier_registry")
        if isinstance(vr, dict):
            known_verifiers.update(vr.keys())

    for row in rows_to_audit:
        if not isinstance(row, dict):
            continue

        has_quality = _has_quality_fields(row)
        if has_quality:
            rows_with_quality += 1

        # Check final_weight.
        fw = row.get("final_weight")
        if _is_finite(fw):
            fw_val = float(fw)
            effective_total += fw_val
            if fw_val > 0:
                positive_weights += 1
                if has_quality:
                    weights_list.append(fw_val)
            elif fw_val == 0:
                zero_weights += 1
            else:
                negative_weights += 1
        else:
            nonfinite_weights += 1

        # Check q_total.
        qt = row.get("q_total")
        if _is_finite(qt):
            q_totals.append(float(qt))

        # Check verifier names.
        vnames = row.get("verifier_names")
        if isinstance(vnames, list):
            for vn in vnames:
                if vn and known_verifiers and vn not in known_verifiers:
                    if vn not in unknown_verifiers:
                        unknown_verifiers.append(vn)
        else:
            vn = row.get("verifier_name")
            if vn and known_verifiers and vn not in known_verifiers:
                if vn not in unknown_verifiers:
                    unknown_verifiers.append(vn)

    # --- Compute statistics ---
    if q_totals:
        mean_q = sum(q_totals) / len(q_totals)
        min_q = min(q_totals)
        max_q = max(q_totals)
    else:
        mean_q = 0.0
        min_q = 0.0
        max_q = 0.0

    # --- Determine status ---
    # If rows exist but none have quality fields → FAIL.
    if rows_with_quality == 0:
        return {
            "experience_task_rows_found": n_task_rows,
            "experience_action_rows_found": n_action_rows,
            "rows_containing_quality_fields": 0,
            "positive_final_weights": positive_weights,
            "zero_weights": zero_weights,
            "negative_weights": negative_weights,
            "nonfinite_weights": nonfinite_weights,
            "unknown_verifiers": unknown_verifiers,
            "effective_total_weight": effective_total,
            "mean_q_total": mean_q,
            "min_q_total": min_q,
            "max_q_total": max_q,
            "weights": [],
            "n_negative": negative_weights,
            "n_zero": zero_weights,
            "status": AuditStatus.FAIL.value,
            "reason": REQUIRED_WEIGHT_FIELDS_MISSING,
        }

    # Check for negative or non-finite weights.
    errors: list[str] = []
    if negative_weights > 0:
        errors.append(f"{negative_weights} row(s) with negative final_weight")
    if nonfinite_weights > 0:
        errors.append(f"{nonfinite_weights} row(s) with non-finite final_weight")
    if unknown_verifiers:
        errors.append(f"unknown verifier(s): {unknown_verifiers}")

    if errors:
        return {
            "experience_task_rows_found": n_task_rows,
            "experience_action_rows_found": n_action_rows,
            "rows_containing_quality_fields": rows_with_quality,
            "positive_final_weights": positive_weights,
            "zero_weights": zero_weights,
            "negative_weights": negative_weights,
            "nonfinite_weights": nonfinite_weights,
            "unknown_verifiers": unknown_verifiers,
            "effective_total_weight": effective_total,
            "mean_q_total": mean_q,
            "min_q_total": min_q,
            "max_q_total": max_q,
            "weights": weights_list,
            "n_negative": negative_weights,
            "n_zero": zero_weights,
            "status": AuditStatus.FAIL.value,
            "reason": "; ".join(errors),
        }

    return {
        "experience_task_rows_found": n_task_rows,
        "experience_action_rows_found": n_action_rows,
        "rows_containing_quality_fields": rows_with_quality,
        "positive_final_weights": positive_weights,
        "zero_weights": zero_weights,
        "negative_weights": negative_weights,
        "nonfinite_weights": nonfinite_weights,
        "unknown_verifiers": unknown_verifiers,
        "effective_total_weight": effective_total,
        "mean_q_total": mean_q,
        "min_q_total": min_q,
        "max_q_total": max_q,
        "weights": weights_list,
        "n_negative": negative_weights,
        "n_zero": zero_weights,
        "status": AuditStatus.PASS.value,
        "reason": (
            f"all {rows_with_quality} row(s) have valid quality fields "
            f"and positive weights"
        ),
    }
