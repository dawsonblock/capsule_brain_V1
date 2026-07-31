"""Serialization parity audit for v0.4.5.

Validates that a candidate (or sham) policy round-trips through
serialization without behavioural drift. The audit consumes a
``parity_report`` produced by the training/evaluation pipeline that
compares the in-memory policy against the reloaded policy.

Required checks:
* original policy logits vs reloaded policy logits (max abs diff <= tol)
* selected actions match
* feature transform hash matches
* action ordering matches
* weight shape matches
* bias shape matches
* normalization state matches
* policy SHA-256 matches
"""
from __future__ import annotations

from typing import Any

# Default tolerance for floating point logit comparison.
DEFAULT_TOLERANCE = 1e-9


def _check(
    checks: list[dict[str, Any]],
    errors: list[str],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    reason: str,
) -> bool:
    checks.append({
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "expected": expected,
        "reason": reason,
    })
    if not passed:
        errors.append(f"{name}: {reason} (observed={observed}, expected={expected})")
    return passed


def audit_serialization_parity(
    parity_report: dict | None,
    policy_type: str = "candidate",
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Audit candidate/sham policy serialization parity.

    Parameters
    ----------
    parity_report:
        A dict produced by the pipeline comparing the original policy to
        the reloaded policy. If ``None`` the audit is BLOCKED.
    policy_type:
        ``"candidate"`` or ``"sham"``. Used only for labelling.
    tolerance:
        Maximum acceptable absolute logit difference.

    Returns
    -------
    dict
        ``status`` (``"PASS"`` | ``"FAIL"`` | ``"BLOCKED"``),
        ``max_abs_logit_difference``, ``selected_actions_match``,
        ``feature_transform_hash_match``, ``action_ordering_match``,
        ``weight_shape_match``, ``policy_sha256_match``, ``checks`` and
        ``errors``.
    """
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    blocked_result = {
        "status": "BLOCKED",
        "max_abs_logit_difference": None,
        "selected_actions_match": None,
        "feature_transform_hash_match": None,
        "action_ordering_match": None,
        "weight_shape_match": None,
        "bias_shape_match": None,
        "normalization_state_match": None,
        "policy_sha256_match": None,
        "checks": [],
        "errors": ["serialization parity artifact missing"],
    }

    if parity_report is None:
        return blocked_result

    if not isinstance(parity_report, dict):
        blocked_result["errors"] = ["serialization parity artifact must be a dict"]
        return blocked_result

    label = policy_type or "candidate"

    # --- Logit difference -------------------------------------------------
    max_abs_diff = parity_report.get("max_abs_logit_difference")
    logit_ok = (
        isinstance(max_abs_diff, (int, float))
        and float(max_abs_diff) <= float(tolerance)
    )
    _check(
        checks, errors,
        f"{label}_logit_parity",
        logit_ok,
        max_abs_diff,
        f"<= {tolerance}",
        "original policy logits vs reloaded policy logits must agree within tolerance",
    )

    # --- Selected actions match ------------------------------------------
    selected_match = parity_report.get("selected_actions_match")
    selected_ok = selected_match is True
    _check(
        checks, errors,
        f"{label}_selected_actions_match",
        selected_ok,
        selected_match,
        True,
        "selected actions must match between original and reloaded policy",
    )

    # --- Feature transform hash match ------------------------------------
    ft_hash_match = parity_report.get("feature_transform_hash_match")
    ft_ok = ft_hash_match is True
    _check(
        checks, errors,
        f"{label}_feature_transform_hash_match",
        ft_ok,
        ft_hash_match,
        True,
        "feature transform hash must match between original and reloaded policy",
    )

    # --- Action ordering match -------------------------------------------
    action_order_match = parity_report.get("action_ordering_match")
    ao_ok = action_order_match is True
    _check(
        checks, errors,
        f"{label}_action_ordering_match",
        ao_ok,
        action_order_match,
        True,
        "action ordering must match between original and reloaded policy",
    )

    # --- Weight shape match ----------------------------------------------
    weight_shape_match = parity_report.get("weight_shape_match")
    ws_ok = weight_shape_match is True
    _check(
        checks, errors,
        f"{label}_weight_shape_match",
        ws_ok,
        weight_shape_match,
        True,
        "weight shape must match between original and reloaded policy",
    )

    # --- Bias shape match ------------------------------------------------
    bias_shape_match = parity_report.get("bias_shape_match")
    bs_ok = bias_shape_match is True
    _check(
        checks, errors,
        f"{label}_bias_shape_match",
        bs_ok,
        bias_shape_match,
        True,
        "bias shape must match between original and reloaded policy",
    )

    # --- Normalization state match --------------------------------------
    norm_match = parity_report.get("normalization_state_match")
    ns_ok = norm_match is True
    _check(
        checks, errors,
        f"{label}_normalization_state_match",
        ns_ok,
        norm_match,
        True,
        "normalization state must match between original and reloaded policy",
    )

    # --- Policy SHA-256 match -------------------------------------------
    sha_match = parity_report.get("policy_sha256_match")
    sha_ok = sha_match is True
    _check(
        checks, errors,
        f"{label}_policy_sha256_match",
        sha_ok,
        sha_match,
        True,
        "policy SHA-256 must match between original and reloaded policy",
    )

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    status = "PASS" if n_fail == 0 else "FAIL"

    return {
        "status": status,
        "max_abs_logit_difference": max_abs_diff,
        "selected_actions_match": bool(selected_ok),
        "feature_transform_hash_match": bool(ft_ok),
        "action_ordering_match": bool(ao_ok),
        "weight_shape_match": bool(ws_ok),
        "bias_shape_match": bool(bs_ok),
        "normalization_state_match": bool(ns_ok),
        "policy_sha256_match": bool(sha_ok),
        "checks": checks,
        "errors": errors,
    }
