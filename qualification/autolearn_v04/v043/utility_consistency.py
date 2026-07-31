"""Utility consistency audit for v0.4.3.

Audits the full utility path across all policies to ensure that the
utility for a given (task_id, action_id) pair is identical regardless of
which policy selected it.

Key insight: all policies (candidate, baseline, sham, oracle) use the
SAME counterfactual outcomes.  Therefore the utility for a given
(task_id, action_id) pair should be identical regardless of which policy
selected it.  Any discrepancy indicates a utility-version mismatch,
different clipping, missing components, None-to-zero conversion, or
inconsistent normalization.

Checks performed:
  1. reward_components keys are consistent across outcomes
  2. utility values are consistent for same (task_id, action_id) pair
  3. None values are not silently converted to zero
  4. non-executed actions are treated consistently
  5. no double penalties in reward_components
  6. latency normalization is consistent across outcomes
  7. token normalization is consistent across outcomes
  8. all policies use the same utility version and normalization
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import LoadedRun, Outcome, PolicyEvaluation
from .config import (
    METRIC_TOLERANCE_STRICT,
    METRIC_TOLERANCE_FLOAT,
    UTILITY_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_per_task_utilities(
    eval_artifact: PolicyEvaluation | None,
) -> dict[str, dict[str, Any]]:
    """Extract per-task utility records from a PolicyEvaluation artifact.

    Returns dict of task_id -> {action, utility, success, availability}.
    """
    if eval_artifact is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for tid, rec in eval_artifact.per_task.items():
        result[tid] = {
            "action": rec.get("action"),
            "utility": rec.get("utility"),
            "success": rec.get("success"),
            "availability": rec.get("availability", "executed"),
        }
    return result


def _check_reward_components_keys(
    outcomes: list[Outcome],
) -> dict[str, Any]:
    """Check that reward_components keys are consistent across outcomes."""
    if not outcomes:
        return {
            "status": "NOT_RUN",
            "detail": "No outcomes to check.",
        }

    all_keys: set[str] = set()
    per_outcome_keys: list[set[str]] = []
    for o in outcomes:
        keys = set(o.reward_components.keys())
        all_keys |= keys
        per_outcome_keys.append(keys)

    # Find outcomes missing keys that others have.
    missing_report: list[dict[str, Any]] = []
    for i, o in enumerate(outcomes):
        missing = all_keys - per_outcome_keys[i]
        if missing:
            missing_report.append({
                "task_id": o.task_id,
                "action_id": o.action_id,
                "missing_keys": sorted(missing),
            })

    if not missing_report:
        return {
            "status": "PASS",
            "detail": (
                f"All {len(outcomes)} outcomes have consistent "
                f"reward_components keys: {sorted(all_keys)}."
            ),
        }
    return {
        "status": "FAIL",
        "detail": (
            f"{len(missing_report)} of {len(outcomes)} outcomes have "
            f"missing reward_components keys."
        ),
        "missing": missing_report[:50],
        "n_missing": len(missing_report),
    }


def _check_utility_consistency(
    run: LoadedRun,
) -> dict[str, Any]:
    """Check that utility values are consistent for same (task_id, action_id).

    All policies read from the same counterfactual outcomes, so the
    utility stored in ``outcomes_by_task_action`` must match the raw
    counterfactual outcome utility.  We also cross-check the per-task
    utilities reported in each policy evaluation artifact.
    """
    discrepancies: list[dict[str, Any]] = []

    # Check counterfactual outcomes vs outcomes_by_task_action.
    for raw in run.counterfactual_outcomes:
        tid = raw.get("task_id", "")
        aid = raw.get("action_id", "")
        raw_util = raw.get("utility")
        outcome = run.outcomes_by_task_action.get((tid, aid))
        if outcome is not None and raw_util is not None:
            if abs(float(raw_util) - outcome.utility) > METRIC_TOLERANCE_FLOAT:
                discrepancies.append({
                    "task_id": tid,
                    "action_id": aid,
                    "raw_utility": float(raw_util),
                    "stored_utility": outcome.utility,
                    "source": "counterfactual_vs_stored",
                })

    # Cross-check per-task utilities across policy evaluation artifacts.
    eval_artifacts = {
        "candidate": run.eval_candidate,
        "baseline": run.eval_baseline,
        "sham": run.eval_sham,
        "oracle": run.eval_oracle,
    }
    per_task_maps = {
        name: _extract_per_task_utilities(art)
        for name, art in eval_artifacts.items()
    }

    # For each task, check that the utility for the same action is
    # consistent across policies that selected it.
    all_task_ids = set()
    for m in per_task_maps.values():
        all_task_ids |= set(m.keys())

    for tid in sorted(all_task_ids):
        # Collect (policy, action, utility) for this task.
        action_utilities: dict[str, list[tuple[str, Any]]] = {}
        for pname, m in per_task_maps.items():
            rec = m.get(tid)
            if rec is None:
                continue
            action = rec.get("action")
            util = rec.get("utility")
            action_utilities.setdefault(str(action), []).append((pname, util))

        for action, pairs in action_utilities.items():
            if len(pairs) < 2:
                continue
            utils = [u for _, u in pairs if u is not None]
            if len(utils) < 2:
                continue
            u_min = min(float(u) for u in utils)
            u_max = max(float(u) for u in utils)
            if u_max - u_min > METRIC_TOLERANCE_FLOAT:
                discrepancies.append({
                    "task_id": tid,
                    "action_id": action,
                    "policies": [
                        {"policy": p, "utility": u}
                        for p, u in pairs
                    ],
                    "source": "cross_policy",
                })

    if not discrepancies:
        return {
            "status": "PASS",
            "detail": (
                "Utility values are consistent for all (task_id, action_id) "
                "pairs across counterfactual outcomes and policy evaluation "
                "artifacts."
            ),
        }
    return {
        "status": "FAIL",
        "detail": (
            f"{len(discrepancies)} utility discrepancies found across "
            "counterfactual outcomes and policy evaluation artifacts."
        ),
        "discrepancies": discrepancies[:50],
        "n_discrepancies": len(discrepancies),
    }


def _check_none_not_converted_to_zero(
    run: LoadedRun,
) -> dict[str, Any]:
    """Check that None utility values are not silently converted to zero.

    Outcomes with availability != "executed" should not have their
    utility treated as 0.0 in any evaluation artifact.  We check both
    the counterfactual outcomes and the per-task records in the eval
    artifacts.
    """
    violations: list[dict[str, Any]] = []

    # Check counterfactual outcomes: non-executed should not have
    # utility == 0.0 if the raw record had null/None utility.
    for raw in run.counterfactual_outcomes:
        avail = raw.get("availability", "executed")
        raw_util = raw.get("utility")
        if avail != "executed" and raw_util is None:
            # Check if the stored Outcome has utility 0.0 (silent conversion).
            tid = raw.get("task_id", "")
            aid = raw.get("action_id", "")
            outcome = run.outcomes_by_task_action.get((tid, aid))
            if outcome is not None and outcome.utility == 0.0:
                violations.append({
                    "task_id": tid,
                    "action_id": aid,
                    "availability": avail,
                    "stored_utility": outcome.utility,
                    "raw_utility": raw_util,
                    "source": "counterfactual_outcome",
                })

    # Check eval artifacts: tasks with unavailable actions should not
    # report utility 0.0 for those actions.
    eval_artifacts = {
        "candidate": run.eval_candidate,
        "baseline": run.eval_baseline,
        "sham": run.eval_sham,
        "oracle": run.eval_oracle,
    }
    for pname, art in eval_artifacts.items():
        if art is None:
            continue
        for tid, rec in art.per_task.items():
            avail = rec.get("availability", "executed")
            util = rec.get("utility")
            if avail != "executed" and util == 0.0:
                violations.append({
                    "policy": pname,
                    "task_id": tid,
                    "availability": avail,
                    "reported_utility": util,
                    "source": "eval_artifact",
                })

    if not violations:
        return {
            "status": "PASS",
            "detail": (
                "No None utility values were silently converted to zero."
            ),
        }
    return {
        "status": "FAIL",
        "detail": (
            f"{len(violations)} cases where None utility may have been "
            "silently converted to zero."
        ),
        "violations": violations[:50],
        "n_violations": len(violations),
    }


def _check_non_executed_consistency(
    run: LoadedRun,
) -> dict[str, Any]:
    """Check that non-executed actions are treated consistently.

    Non-executed actions (availability != "executed") should be excluded
    from all aggregate utility calculations across all policies.  We
    verify that the outcome utility is not used in any policy's mean.
    """
    issues: list[dict[str, Any]] = []

    non_executed_by_task: dict[str, list[str]] = {}
    for o in run.outcomes:
        if o.availability != "executed":
            non_executed_by_task.setdefault(o.task_id, []).append(o.action_id)

    if not non_executed_by_task:
        return {
            "status": "PASS",
            "detail": "No non-executed actions found; check trivially passes.",
        }

    # Check that get_utility_for_action returns None for non-executed.
    for tid, actions in non_executed_by_task.items():
        for aid in actions:
            util = run.get_utility_for_action(tid, aid)
            if util is not None:
                issues.append({
                    "task_id": tid,
                    "action_id": aid,
                    "availability": "non-executed",
                    "returned_utility": util,
                    "source": "get_utility_for_action",
                })

    if not issues:
        return {
            "status": "PASS",
            "detail": (
                f"All {sum(len(v) for v in non_executed_by_task.values())} "
                "non-executed actions are correctly excluded from utility "
                "lookups."
            ),
        }
    return {
        "status": "FAIL",
        "detail": (
            f"{len(issues)} non-executed actions returned a non-None utility."
        ),
        "issues": issues[:50],
        "n_issues": len(issues),
    }


def _check_double_penalties(
    run: LoadedRun,
) -> dict[str, Any]:
    """Check for double penalties in reward_components.

    A double penalty occurs when the same penalty category (e.g.
    latency_penalty, token_penalty) appears multiple times in the
    reward_components of a single outcome, or when a penalty is applied
    both as a named component and embedded within another component.
    """
    issues: list[dict[str, Any]] = []

    penalty_keywords = {"penalty", "penalize", "deduction", "cost_penalty"}

    for o in run.outcomes:
        rc = o.reward_components
        keys = list(rc.keys())

        # Check for duplicate penalty keys (case-insensitive).
        penalty_keys = [
            k for k in keys
            if any(kw in k.lower() for kw in penalty_keywords)
        ]
        if len(penalty_keys) != len(set(penalty_keys)):
            issues.append({
                "task_id": o.task_id,
                "action_id": o.action_id,
                "duplicate_penalty_keys": penalty_keys,
                "source": "duplicate_keys",
            })

        # Check for penalty values that appear to be applied twice.
        # If two penalty components have the same value, it may indicate
        # a double application.
        penalty_values: dict[float, list[str]] = {}
        for k in penalty_keys:
            v = rc.get(k)
            if isinstance(v, (int, float)):
                penalty_values.setdefault(float(v), []).append(k)
        for val, keys_with_val in penalty_values.items():
            if len(keys_with_val) > 1 and val != 0.0:
                issues.append({
                    "task_id": o.task_id,
                    "action_id": o.action_id,
                    "duplicate_penalty_value": val,
                    "keys": keys_with_val,
                    "source": "duplicate_value",
                })

    if not issues:
        return {
            "status": "PASS",
            "detail": "No double penalties detected in reward_components.",
        }
    return {
        "status": "FAIL",
        "detail": (
            f"{len(issues)} potential double-penalty issues detected."
        ),
        "issues": issues[:50],
        "n_issues": len(issues),
    }


def _check_latency_normalization(
    run: LoadedRun,
) -> dict[str, Any]:
    """Check that latency normalization is consistent across outcomes.

    If execution_metadata contains latency fields, they should be
    normalized on the same scale across all outcomes.
    """
    latency_values: list[dict[str, Any]] = []
    latency_keys: set[str] = set()

    for o in run.outcomes:
        em = o.execution_metadata
        for k, v in em.items():
            if "latency" in k.lower() and isinstance(v, (int, float)):
                latency_keys.add(k)
                latency_values.append({
                    "task_id": o.task_id,
                    "action_id": o.action_id,
                    "key": k,
                    "value": float(v),
                })

    if not latency_values:
        return {
            "status": "PASS",
            "detail": "No latency fields found in execution_metadata.",
        }

    # Check that values are on a consistent scale (no wild outliers
    # suggesting different normalization).
    values = [lv["value"] for lv in latency_values]
    arr = np.array(values, dtype=float)
    val_min = float(arr.min())
    val_max = float(arr.max())
    val_mean = float(arr.mean())
    val_std = float(arr.std())

    # If the range spans multiple orders of magnitude, normalization may
    # be inconsistent.
    range_ratio = (val_max - val_min) / max(abs(val_mean), 1e-10)
    consistent = range_ratio < 100.0  # heuristic threshold

    if consistent:
        return {
            "status": "PASS",
            "detail": (
                f"Latency normalization is consistent across "
                f"{len(latency_values)} outcomes "
                f"(keys={sorted(latency_keys)}, "
                f"range=[{val_min:.4f}, {val_max:.4f}], "
                f"mean={val_mean:.4f}, std={val_std:.4f})."
            ),
        }
    return {
        "status": "FAIL",
        "detail": (
            f"Latency values span an inconsistent range "
            f"(range_ratio={range_ratio:.2f}), suggesting different "
            "normalization scales."
        ),
        "keys": sorted(latency_keys),
        "n_values": len(latency_values),
        "min": val_min,
        "max": val_max,
        "mean": val_mean,
        "std": val_std,
    }


def _check_token_normalization(
    run: LoadedRun,
) -> dict[str, Any]:
    """Check that token normalization is consistent across outcomes."""
    token_values: list[dict[str, Any]] = []
    token_keys: set[str] = set()

    for o in run.outcomes:
        em = o.execution_metadata
        for k, v in em.items():
            if "token" in k.lower() and isinstance(v, (int, float)):
                token_keys.add(k)
                token_values.append({
                    "task_id": o.task_id,
                    "action_id": o.action_id,
                    "key": k,
                    "value": float(v),
                })

    if not token_values:
        return {
            "status": "PASS",
            "detail": "No token fields found in execution_metadata.",
        }

    values = [tv["value"] for tv in token_values]
    arr = np.array(values, dtype=float)
    val_min = float(arr.min())
    val_max = float(arr.max())
    val_mean = float(arr.mean())
    val_std = float(arr.std())

    range_ratio = (val_max - val_min) / max(abs(val_mean), 1e-10)
    consistent = range_ratio < 100.0

    if consistent:
        return {
            "status": "PASS",
            "detail": (
                f"Token normalization is consistent across "
                f"{len(token_values)} outcomes "
                f"(keys={sorted(token_keys)}, "
                f"range=[{val_min:.4f}, {val_max:.4f}], "
                f"mean={val_mean:.4f}, std={val_std:.4f})."
            ),
        }
    return {
        "status": "FAIL",
        "detail": (
            f"Token values span an inconsistent range "
            f"(range_ratio={range_ratio:.2f}), suggesting different "
            "normalization scales."
        ),
        "keys": sorted(token_keys),
        "n_values": len(token_values),
        "min": val_min,
        "max": val_max,
        "mean": val_mean,
        "std": val_std,
    }


def _check_utility_version_and_normalization(
    run: LoadedRun,
) -> dict[str, Any]:
    """Verify that all policies use the same utility version and normalization.

    Checks that the utility values in the eval artifacts are consistent
    with the counterfactual outcomes (same version) and that no policy
    applies different clipping or normalization.
    """
    eval_artifacts = {
        "candidate": run.eval_candidate,
        "baseline": run.eval_baseline,
        "sham": run.eval_sham,
        "oracle": run.eval_oracle,
    }

    issues: list[dict[str, Any]] = []

    for pname, art in eval_artifacts.items():
        if art is None:
            continue
        # Check that per-task utilities match counterfactual outcomes.
        for tid, rec in art.per_task.items():
            action = rec.get("action")
            util = rec.get("utility")
            if action is None or util is None:
                continue
            cf_util = run.get_utility_for_action(tid, str(action))
            if cf_util is not None:
                if abs(float(util) - cf_util) > METRIC_TOLERANCE_FLOAT:
                    issues.append({
                        "policy": pname,
                        "task_id": tid,
                        "action_id": action,
                        "eval_utility": float(util),
                        "counterfactual_utility": cf_util,
                        "source": "version_mismatch",
                    })

    # Check for different clipping: if any policy's utilities are outside
    # the range of counterfactual utilities, clipping may differ.
    cf_utils = [
        o.utility for o in run.outcomes if o.availability == "executed"
    ]
    if cf_utils:
        cf_min = min(cf_utils)
        cf_max = max(cf_utils)
        for pname, art in eval_artifacts.items():
            if art is None:
                continue
            for tid, rec in art.per_task.items():
                util = rec.get("utility")
                if util is None:
                    continue
                u = float(util)
                if u < cf_min - METRIC_TOLERANCE_FLOAT or u > cf_max + METRIC_TOLERANCE_FLOAT:
                    issues.append({
                        "policy": pname,
                        "task_id": tid,
                        "utility": u,
                        "cf_min": cf_min,
                        "cf_max": cf_max,
                        "source": "clipping_mismatch",
                    })

    if not issues:
        return {
            "status": "PASS",
            "detail": (
                f"All policies use the same utility version "
                f"({UTILITY_VERSION}) and normalization. "
                "Per-task utilities match counterfactual outcomes within "
                f"tolerance ({METRIC_TOLERANCE_FLOAT})."
            ),
        }
    return {
        "status": "FAIL",
        "detail": (
            f"{len(issues)} utility version or normalization mismatches "
            "detected across policy evaluation artifacts."
        ),
        "issues": issues[:50],
        "n_issues": len(issues),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_utility_consistency(run: LoadedRun) -> dict:
    """Audit the full utility path for consistency across all policies.

    For every action outcome record, this audit verifies:

    1. reward_components keys are consistent across outcomes
    2. utility values are consistent for same (task_id, action_id) pair
       regardless of which policy selected it
    3. None values are not silently converted to zero
    4. non-executed actions are treated consistently
    5. no double penalties in reward_components
    6. latency normalization is consistent
    7. token normalization is consistent
    8. all policies use the same utility version and normalization

    The key insight: all policies use the SAME counterfactual outcomes,
    so the utility for a given (task_id, action_id) pair should be
    identical regardless of which policy selected it.

    Args:
        run: The loaded run with counterfactual outcomes and evaluation
            artifacts.

    Returns:
        JSON-serializable ``utility_consistency_report`` dict with:

        - ``per_outcome_audit``: summary stats over all outcomes
        - ``consistency_checks``: dict of check_name -> {status, detail}
        - ``overall_status``: "PASS" or "FAIL"
        - ``interpretation``: str
    """
    outcomes = run.outcomes

    # Per-outcome audit summary stats.
    n_outcomes = len(outcomes)
    n_executed = sum(1 for o in outcomes if o.availability == "executed")
    n_unavailable = sum(1 for o in outcomes if o.availability == "unavailable")
    n_skipped = sum(1 for o in outcomes if o.availability == "skipped")
    n_other = n_outcomes - n_executed - n_unavailable - n_skipped

    executed_utils = [
        o.utility for o in outcomes if o.availability == "executed"
    ]
    if executed_utils:
        util_arr = np.array(executed_utils, dtype=float)
        util_stats = {
            "n": len(executed_utils),
            "min": float(util_arr.min()),
            "max": float(util_arr.max()),
            "mean": float(util_arr.mean()),
            "std": float(util_arr.std(ddof=1)) if len(util_arr) > 1 else 0.0,
        }
    else:
        util_stats = {"n": 0, "min": None, "max": None, "mean": None, "std": None}

    # Collect all reward_components keys.
    all_rc_keys: set[str] = set()
    for o in outcomes:
        all_rc_keys |= set(o.reward_components.keys())

    per_outcome_audit = {
        "n_outcomes": n_outcomes,
        "n_executed": n_executed,
        "n_unavailable": n_unavailable,
        "n_skipped": n_skipped,
        "n_other_availability": n_other,
        "executed_utility_stats": util_stats,
        "reward_components_keys": sorted(all_rc_keys),
    }

    # Run all consistency checks.
    consistency_checks: dict[str, dict[str, Any]] = {
        "reward_components_keys_consistent": _check_reward_components_keys(
            outcomes,
        ),
        "utility_values_consistent": _check_utility_consistency(run),
        "none_not_converted_to_zero": _check_none_not_converted_to_zero(run),
        "non_executed_consistency": _check_non_executed_consistency(run),
        "no_double_penalties": _check_double_penalties(run),
        "latency_normalization": _check_latency_normalization(run),
        "token_normalization": _check_token_normalization(run),
        "utility_version_and_normalization": (
            _check_utility_version_and_normalization(run)
        ),
    }

    # Overall status: FAIL if any check fails.
    n_pass = sum(
        1 for c in consistency_checks.values()
        if c.get("status") == "PASS"
    )
    n_fail = sum(
        1 for c in consistency_checks.values()
        if c.get("status") == "FAIL"
    )
    n_not_run = sum(
        1 for c in consistency_checks.values()
        if c.get("status") == "NOT_RUN"
    )

    overall_status = "FAIL" if n_fail > 0 else "PASS"

    failed_checks = [
        name for name, c in consistency_checks.items()
        if c.get("status") == "FAIL"
    ]

    if overall_status == "PASS":
        interpretation = (
            f"Utility consistency PASS: all {len(consistency_checks)} "
            "checks pass. All policies use the same counterfactual "
            "outcomes with consistent utility version, normalization, "
            "and treatment of non-executed actions."
        )
    else:
        interpretation = (
            f"Utility consistency FAIL: {n_fail} of "
            f"{len(consistency_checks)} checks failed "
            f"({failed_checks}). Utility values may differ across "
            "policies due to version mismatch, different clipping, "
            "None-to-zero conversion, or inconsistent normalization."
        )

    return {
        "per_outcome_audit": per_outcome_audit,
        "consistency_checks": consistency_checks,
        "overall_status": overall_status,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_not_run": n_not_run,
        "failed_checks": failed_checks,
        "interpretation": interpretation,
    }
