"""Canonical policy evaluation for v0.4.5.

ONE evaluator for all policies: baseline, candidate, sham, oracle.

Key rule: REAL rows only. NO aggregate-to-task expansion. If task-level
policy decisions are absent, the aggregate summary may be displayed but
all task-level analyses are BLOCKED.

Evidence levels:
- AGGREGATE_ONLY: only summary means available
- TASK_LEVEL: task-level decisions available
- ACTION_LEVEL_COMPLETE: complete action matrix available
"""
from __future__ import annotations

import math
from typing import Any

from qualification.autolearn_v04.common.evidence_levels import (
    EvidenceLevel,
    detect_evidence_level,
)
from qualification.autolearn_v04.common.headroom import compute_recovered_headroom


# Analyses that are BLOCKED when evidence level is below TASK_LEVEL or
# ACTION_LEVEL_COMPLETE, per the canonical evidence-level registry.
_TASK_LEVEL_ANALYSES = [
    "candidate_vs_baseline_bootstrap",
    "candidate_vs_sham_bootstrap",
    "family_analysis",
    "archetype_analysis",
    "confusion_matrix",
    "raw_vs_executed",
    "feature_signal",
]

_ACTION_LEVEL_ANALYSES = [
    "oracle_routing_headroom",
    "headroom",
    "headroom_concentration",
    "high_margin_recovery",
]


def _is_finite(v: Any) -> bool:
    if v is None or isinstance(v, bool):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _safe_float(v: Any) -> float | None:
    if not _is_finite(v):
        return None
    return float(v)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _extract_mean(results: dict) -> float | None:
    if not isinstance(results, dict):
        return None
    for key in ("mean_utility", "mean", "aggregate_mean", "summary_mean"):
        v = _safe_float(results.get(key))
        if v is not None:
            return v
    per_task = results.get("per_task") or results.get("task_results") or []
    utils = [_safe_float(r.get("utility")) for r in per_task]
    utils = [u for u in utils if u is not None]
    return _mean(utils)


def _extract_task_rows(
    counterfactual_outcomes: list[dict],
    policy_name: str,
) -> list[dict]:
    """Extract task-level rows for a given policy from counterfactual outcomes.

    Returns rows that carry a task_id and a measured utility. Rows without a
    task_id are NOT considered task-level and are ignored.
    """
    rows: list[dict] = []
    for o in counterfactual_outcomes:
        policy = o.get("policy") or o.get("policy_id") or o.get("policy_name")
        if policy != policy_name:
            continue
        tid = o.get("task_id") or o.get("id") or o.get("task")
        if tid is None:
            continue
        rows.append(o)
    return rows


def _policy_mean_from_rows(rows: list[dict]) -> float | None:
    utils = [_safe_float(r.get("utility")) for r in rows]
    utils = [u for u in utils if u is not None]
    return _mean(utils)


def _build_oracle_rows(counterfactual_outcomes: list[dict]) -> list[dict]:
    """Construct oracle rows from REAL measured outcomes only.

    Oracle = maximum measured utility among complete eligible actions, per
    task. This NEVER expands aggregate means into synthetic task rows.
    """
    per_task: dict[str, list[dict]] = {}
    for o in counterfactual_outcomes:
        tid = o.get("task_id") or o.get("id") or o.get("task")
        if tid is None:
            continue
        util = _safe_float(o.get("utility"))
        if util is None:
            continue
        per_task.setdefault(str(tid), []).append(o)

    oracle_rows: list[dict] = []
    for tid, rows in per_task.items():
        best = max(rows, key=lambda r: _safe_float(r.get("utility")) or float("-inf"))
        best_util = _safe_float(best.get("utility"))
        oracle_rows.append(
            {
                "task_id": tid,
                "policy": "oracle",
                "selected_action": best.get("selected_action")
                or best.get("eligible_action")
                or best.get("action")
                or "",
                "utility": best_util,
                "oracle_construction_method": "task_level_best_action",
                "source": "counterfactual_outcomes",
            }
        )
    return oracle_rows


def _policy_status(rows: list[dict], evidence_level: EvidenceLevel) -> str:
    if rows:
        return "MEASURED"
    if evidence_level >= EvidenceLevel.AGGREGATE_ONLY:
        return "AGGREGATE_ONLY"
    return "BLOCKED"


def evaluate_policies_canonically(
    counterfactual_outcomes: list[dict],
    candidate_results: dict,
    sham_results: dict,
    baseline_results: dict,
    oracle_results: dict,
    split_manifest: dict | None = None,
    benchmark_manifest: dict | None = None,
) -> dict:
    """Evaluate all policies canonically from REAL rows only.

    For each policy (baseline, candidate, sham, oracle):
    - If task-level rows exist in counterfactual_outcomes, compute metrics
      from actual rows.
    - If only aggregate summary exists, report AGGREGATE_ONLY level and
      BLOCK all task-level analyses.
    - NEVER expand aggregate means into synthetic task rows.

    Oracle: select maximum measured utility among complete eligible actions.
    Do NOT construct oracle from aggregate mean.

    Returns a dict with evidence_level, policies, headroom, blocked_analyses,
    and errors.
    """
    errors: list[str] = []

    # Detect evidence level from available artifacts.
    evidence_level = detect_evidence_level(
        counterfactual_outcomes,
        policy_results=candidate_results,
        split_manifest=split_manifest,
    )

    # Build per-policy rows from REAL counterfactual outcomes.
    baseline_rows = _extract_task_rows(counterfactual_outcomes, "baseline")
    candidate_rows = _extract_task_rows(counterfactual_outcomes, "candidate")
    sham_rows = _extract_task_rows(counterfactual_outcomes, "sham")

    # Oracle is constructed from task-level best action, never from aggregate.
    oracle_rows: list[dict] = []
    if evidence_level >= EvidenceLevel.TASK_LEVEL:
        oracle_rows = _build_oracle_rows(counterfactual_outcomes)

    policies: dict[str, dict[str, Any]] = {}

    for name, rows, results in (
        ("baseline", baseline_rows, baseline_results),
        ("candidate", candidate_rows, candidate_results),
        ("sham", sham_rows, sham_results),
    ):
        if rows:
            mean_utility = _policy_mean_from_rows(rows)
            task_count = len(rows)
            policies[name] = {
                "mean_utility": mean_utility,
                "task_count": task_count,
                "task_rows": rows,
                "status": _policy_status(rows, evidence_level),
            }
        else:
            # Aggregate-only fallback: display summary, BLOCK task-level.
            mean_utility = _extract_mean(results)
            policies[name] = {
                "mean_utility": mean_utility,
                "task_count": 0,
                "task_rows": None,
                "status": "AGGREGATE_ONLY" if mean_utility is not None else "BLOCKED",
            }
            if mean_utility is None:
                errors.append(f"{name}: no task-level rows and no aggregate mean available")

    # Oracle policy.
    if oracle_rows:
        oracle_mean = _policy_mean_from_rows(oracle_rows)
        policies["oracle"] = {
            "mean_utility": oracle_mean,
            "task_count": len(oracle_rows),
            "task_rows": oracle_rows,
            "status": "MEASURED",
            "oracle_construction_method": "task_level_best_action",
        }
    else:
        # Do NOT construct oracle from aggregate mean.
        oracle_mean_reported = _extract_mean(oracle_results)
        policies["oracle"] = {
            "mean_utility": None,  # explicitly not using aggregate mean
            "task_count": 0,
            "task_rows": None,
            "status": "BLOCKED",
            "oracle_construction_method": "none",
            "reported_aggregate_mean": oracle_mean_reported,
        }
        errors.append(
            "oracle: task-level rows unavailable; oracle not constructed from aggregate mean"
        )

    # Headroom: BLOCKED unless ACTION_LEVEL_COMPLETE.
    headroom: dict[str, Any] = {"status": "BLOCKED"}
    if evidence_level >= EvidenceLevel.ACTION_LEVEL_COMPLETE:
        candidate_mean = policies["candidate"].get("mean_utility")
        baseline_mean = policies["baseline"].get("mean_utility")
        oracle_mean = policies["oracle"].get("mean_utility")
        if candidate_mean is not None and baseline_mean is not None and oracle_mean is not None:
            hr = compute_recovered_headroom(
                candidate_mean=candidate_mean,
                reference_mean=baseline_mean,
                oracle_mean=oracle_mean,
                task_count=policies["candidate"].get("task_count", 0),
            )
            headroom = hr.to_dict()
            headroom["status"] = hr.status
        else:
            headroom = {
                "status": "BLOCKED",
                "reason": "missing candidate/baseline/oracle means for headroom",
            }
            errors.append("headroom BLOCKED: missing required means")
    else:
        headroom = {
            "status": "BLOCKED",
            "reason": f"requires ACTION_LEVEL_COMPLETE, achieved {evidence_level.name}",
        }

    # Blocked analyses.
    blocked_analyses: list[str] = []
    if evidence_level < EvidenceLevel.TASK_LEVEL:
        blocked_analyses.extend(_TASK_LEVEL_ANALYSES)
    if evidence_level < EvidenceLevel.ACTION_LEVEL_COMPLETE:
        blocked_analyses.extend(_ACTION_LEVEL_ANALYSES)

    # If evidence level is below TASK_LEVEL, all task-level analyses BLOCKED.
    if evidence_level < EvidenceLevel.TASK_LEVEL:
        for a in _TASK_LEVEL_ANALYSES:
            if a not in blocked_analyses:
                blocked_analyses.append(a)

    # Deduplicate.
    blocked_analyses = list(dict.fromkeys(blocked_analyses))

    return {
        "evidence_level": evidence_level.name,
        "policies": policies,
        "headroom": headroom,
        "blocked_analyses": blocked_analyses,
        "errors": errors,
    }
