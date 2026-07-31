"""Evidence weight audit for v0.4.5.

Validates that the experience file contains real training examples with
positive evidence weights and complete provenance.

Each experience row must include:
  - task_id, split, feature_vector or feature reference, best_action,
    eligible_actions
  - measured_action_utilities, utility_margin
  - q_verifier, q_execution, q_counterfactual, q_isolation, q_provenance,
    q_total, final_weight
  - verifier_name, run_id

Required checks:
  - experience split only (not safety controlled)
  - q_total > 0
  - final_weight > 0
  - finite values
  - known verifier
  - complete provenance
  - at least two comparable actions
  - feature dimension matches policy (if policy provided)
"""
from __future__ import annotations

import math
from typing import Any

from .verifier_registry_audit import DEFAULT_VERIFIER_REGISTRY


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: tuple[str, ...] = (
    "task_id",
    "split",
    "best_action",
    "eligible_actions",
    "measured_action_utilities",
    "utility_margin",
    "q_verifier",
    "q_execution",
    "q_counterfactual",
    "q_isolation",
    "q_provenance",
    "q_total",
    "final_weight",
    "verifier_name",
    "run_id",
)

Q_FIELDS: tuple[str, ...] = (
    "q_verifier",
    "q_execution",
    "q_counterfactual",
    "q_isolation",
    "q_provenance",
    "q_total",
    "final_weight",
)

# Splits that are safety-controlled and must not appear in experience data.
SAFETY_SPLITS: frozenset[str] = frozenset({"safety", "safety_controlled"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_finite(value: Any) -> bool:
    """Return True if value is a finite number."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _get_feature_vector(row: dict) -> list[float] | None:
    """Extract a feature vector from an experience row."""
    fv = row.get("feature_vector")
    if isinstance(fv, list):
        return [float(x) for x in fv if _is_finite(x)]
    ref = row.get("feature_reference") or row.get("features")
    if isinstance(ref, list):
        return [float(x) for x in ref if _is_finite(x)]
    return None


def _policy_feature_dim(candidate_policy: dict) -> int | None:
    """Determine the expected feature dimension from the candidate policy."""
    if not isinstance(candidate_policy, dict):
        return None
    names = candidate_policy.get("feature_names")
    if isinstance(names, list) and names:
        return len(names)
    transform = candidate_policy.get("feature_transform")
    if isinstance(transform, dict):
        names = transform.get("feature_names")
        if isinstance(names, list) and names:
            return len(names)
    weights = candidate_policy.get("weights")
    if isinstance(weights, list) and weights:
        first = weights[0]
        if isinstance(first, list):
            return len(first)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_evidence_weights(
    experience_rows: list[dict],
    candidate_policy: dict | None = None,
    min_q_total: float = 0.0,
    min_weight: float = 0.0,
) -> dict:
    """Audit evidence weights in the experience file.

    Validates that every experience row carries the required fields, has
    positive ``q_total`` and ``final_weight``, finite values, a known
    verifier, complete provenance, at least two comparable actions, and a
    feature dimension consistent with the candidate policy (when provided).

    Returns a dict with status ("PASS" | "FAIL" | "BLOCKED"), row counts,
    weight statistics, per-row checks, and error messages.
    """
    errors: list[str] = []

    # --- BLOCKED check: no experience rows ---
    if not experience_rows:
        errors.append("experience_rows is empty")
        return {
            "status": "BLOCKED",
            "experience_row_count": 0,
            "positive_weight_row_count": 0,
            "zero_weight_row_count": 0,
            "effective_weight_sum": 0.0,
            "mean_q_total": 0.0,
            "min_q_total": 0.0,
            "max_q_total": 0.0,
            "checks": [],
            "errors": errors,
        }

    known_verifiers = set(DEFAULT_VERIFIER_REGISTRY.keys())
    if isinstance(candidate_policy, dict):
        extra = candidate_policy.get("verifier_registry")
        if isinstance(extra, dict):
            known_verifiers.update(extra.keys())

    expected_dim = _policy_feature_dim(candidate_policy or {})

    checks: list[dict[str, Any]] = []
    q_totals: list[float] = []
    positive_weight_count = 0
    zero_weight_count = 0
    effective_weight_sum = 0.0

    for idx, row in enumerate(experience_rows):
        if not isinstance(row, dict):
            errors.append(f"Experience row {idx} is not a dict")
            checks.append({
                "row_index": idx,
                "task_id": None,
                "check": "row_is_dict",
                "status": "FAIL",
                "detail": "row is not a dict",
            })
            continue

        tid = row.get("task_id")
        row_errors: list[str] = []

        # --- Required fields ---
        missing = [f for f in REQUIRED_FIELDS if f not in row or row[f] is None]
        if missing:
            row_errors.append(f"missing required fields: {missing}")
            checks.append({
                "row_index": idx,
                "task_id": tid,
                "check": "required_fields_present",
                "status": "FAIL",
                "detail": f"missing: {missing}",
            })
        else:
            checks.append({
                "row_index": idx,
                "task_id": tid,
                "check": "required_fields_present",
                "status": "PASS",
                "detail": "",
            })

        # --- Feature vector present ---
        fv = _get_feature_vector(row)
        has_features = fv is not None and len(fv) > 0
        checks.append({
            "row_index": idx,
            "task_id": tid,
            "check": "feature_vector_present",
            "status": "PASS" if has_features else "FAIL",
            "detail": "" if has_features else "no feature_vector or feature reference",
        })
        if not has_features:
            row_errors.append("missing feature_vector or feature reference")

        # --- Experience split only (not safety controlled) ---
        split = str(row.get("split", ""))
        split_ok = split and split not in SAFETY_SPLITS
        checks.append({
            "row_index": idx,
            "task_id": tid,
            "check": "experience_split_only",
            "status": "PASS" if split_ok else "FAIL",
            "detail": f"split={split!r}",
        })
        if not split_ok:
            row_errors.append(f"row uses safety-controlled split '{split}'")

        # --- Finite q/weight values ---
        non_finite = [f for f in Q_FIELDS if not _is_finite(row.get(f))]
        checks.append({
            "row_index": idx,
            "task_id": tid,
            "check": "finite_values",
            "status": "PASS" if not non_finite else "FAIL",
            "detail": "" if not non_finite else f"non-finite: {non_finite}",
        })
        if non_finite:
            row_errors.append(f"non-finite values: {non_finite}")

        # --- q_total > min_q_total ---
        q_total = row.get("q_total")
        q_total_ok = _is_finite(q_total) and float(q_total) > min_q_total
        checks.append({
            "row_index": idx,
            "task_id": tid,
            "check": "q_total_positive",
            "status": "PASS" if q_total_ok else "FAIL",
            "detail": f"q_total={q_total}",
        })
        if not q_total_ok:
            row_errors.append(f"q_total ({q_total}) is not > {min_q_total}")
        else:
            q_totals.append(float(q_total))

        # --- final_weight > min_weight ---
        fw = row.get("final_weight")
        fw_ok = _is_finite(fw) and float(fw) > min_weight
        checks.append({
            "row_index": idx,
            "task_id": tid,
            "check": "final_weight_positive",
            "status": "PASS" if fw_ok else "FAIL",
            "detail": f"final_weight={fw}",
        })
        if not fw_ok:
            row_errors.append(f"final_weight ({fw}) is not > {min_weight}")
            zero_weight_count += 1
        else:
            positive_weight_count += 1
            effective_weight_sum += float(fw)

        # --- Known verifier ---
        vname = row.get("verifier_name")
        verifier_ok = bool(vname) and str(vname) in known_verifiers
        checks.append({
            "row_index": idx,
            "task_id": tid,
            "check": "known_verifier",
            "status": "PASS" if verifier_ok else "FAIL",
            "detail": f"verifier_name={vname!r}",
        })
        if not verifier_ok:
            row_errors.append(f"unknown verifier '{vname}'")

        # --- Complete provenance ---
        provenance_fields = ("run_id", "q_provenance", "q_isolation")
        prov_missing = [f for f in provenance_fields if not row.get(f)]
        checks.append({
            "row_index": idx,
            "task_id": tid,
            "check": "complete_provenance",
            "status": "PASS" if not prov_missing else "FAIL",
            "detail": "" if not prov_missing else f"missing: {prov_missing}",
        })
        if prov_missing:
            row_errors.append(f"incomplete provenance: {prov_missing}")

        # --- At least two comparable actions ---
        eligible = row.get("eligible_actions")
        utilities = row.get("measured_action_utilities")
        n_eligible = len(eligible) if isinstance(eligible, list) else 0
        n_utilities = len(utilities) if isinstance(utilities, dict) else (
            len(utilities) if isinstance(utilities, list) else 0
        )
        comparable_ok = n_eligible >= 2 and n_utilities >= 2
        checks.append({
            "row_index": idx,
            "task_id": tid,
            "check": "at_least_two_comparable_actions",
            "status": "PASS" if comparable_ok else "FAIL",
            "detail": f"eligible={n_eligible}, utilities={n_utilities}",
        })
        if not comparable_ok:
            row_errors.append(
                f"need at least two comparable actions "
                f"(eligible={n_eligible}, utilities={n_utilities})"
            )

        # --- Feature dimension matches policy ---
        if expected_dim is not None and fv is not None:
            dim_ok = len(fv) == expected_dim
            checks.append({
                "row_index": idx,
                "task_id": tid,
                "check": "feature_dimension_matches_policy",
                "status": "PASS" if dim_ok else "FAIL",
                "detail": f"got={len(fv)}, expected={expected_dim}",
            })
            if not dim_ok:
                row_errors.append(
                    f"feature dimension {len(fv)} != policy {expected_dim}"
                )

        for msg in row_errors:
            errors.append(f"Row {idx} (task_id={tid}): {msg}")

    # --- Aggregate statistics ---
    if q_totals:
        mean_q = sum(q_totals) / len(q_totals)
        min_q = min(q_totals)
        max_q = max(q_totals)
    else:
        mean_q = 0.0
        min_q = 0.0
        max_q = 0.0

    status = "PASS" if not errors else "FAIL"

    return {
        "status": status,
        "experience_row_count": len(experience_rows),
        "positive_weight_row_count": positive_weight_count,
        "zero_weight_row_count": zero_weight_count,
        "effective_weight_sum": float(effective_weight_sum),
        "mean_q_total": float(mean_q),
        "min_q_total": float(min_q),
        "max_q_total": float(max_q),
        "checks": checks,
        "errors": errors,
    }
