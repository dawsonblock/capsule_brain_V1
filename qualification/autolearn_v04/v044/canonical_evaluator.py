"""Canonical policy evaluator for v0.4.4.

ONE evaluator for all policies: random, majority, frequency, baseline,
candidate raw, candidate executed, primary sham, sham ensemble members, oracle.

All summary metrics in reports must derive from these canonical outputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from qualification.autolearn_v04.common.headroom import (
    compute_recovered_headroom,
    compute_task_ids_digest,
    RecoveredHeadroomResult,
)
from qualification.autolearn_v04.common.schemas import (
    OutcomeStatus,
    MissingUtilityError,
    InvalidOutcomeError,
)


@dataclass(frozen=True)
class TaskLevelResult:
    """Per-task result from canonical policy evaluation."""
    task_id: str
    selected_action: str
    selected_action_available: bool
    selected_utility: float | None
    verified_success: bool | None
    oracle_action: str
    oracle_utility: float | None
    regret: float | None
    family: str
    archetype: str
    margin_bucket: str
    headroom_contribution: float | None
    outcome_status: str


@dataclass(frozen=True)
class PolicyEvalResult:
    """Aggregate result from canonical policy evaluation."""
    policy_id: str
    n_tasks: int
    mean_utility: float | None
    median_utility: float | None
    verified_success_rate: float | None
    wins: int
    ties: int
    losses: int
    mean_regret: float | None
    median_regret: float | None
    p95_regret: float | None
    action_distribution: dict[str, float]
    family_metrics: dict[str, dict[str, float]]
    archetype_metrics: dict[str, dict[str, float]]
    margin_metrics: dict[str, dict[str, float]]
    task_ids_digest: str
    task_results: list[TaskLevelResult]
    headroom_vs_reference: RecoveredHeadroomResult | None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "policy_id": self.policy_id,
            "n_tasks": self.n_tasks,
            "mean_utility": self.mean_utility,
            "median_utility": self.median_utility,
            "verified_success_rate": self.verified_success_rate,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "mean_regret": self.mean_regret,
            "median_regret": self.median_regret,
            "p95_regret": self.p95_regret,
            "action_distribution": self.action_distribution,
            "family_metrics": self.family_metrics,
            "archetype_metrics": self.archetype_metrics,
            "margin_metrics": self.margin_metrics,
            "task_ids_digest": self.task_ids_digest,
            "headroom_vs_reference": self.headroom_vs_reference.to_dict() if self.headroom_vs_reference else None,
        }
        return d


def evaluate_policy_from_results(
    policy_id: str,
    results_dict: dict[str, Any],
    task_ids: list[str],
    oracle_results: dict[str, Any] | None = None,
    reference_mean: float | None = None,
    reference_policy_id: str = "",
    task_families: dict[str, str] | None = None,
    task_archetypes: dict[str, str] | None = None,
    task_margins: dict[str, str] | None = None,
) -> PolicyEvalResult:
    """Evaluate a policy from its results dict.

    This is the canonical evaluator. All policy metrics must go through this.
    """
    task_families = task_families or {}
    task_archetypes = task_archetypes or {}
    task_margins = task_margins or {}

    # Extract per-task utilities from results.
    # Results dict may contain "per_task" list or just aggregate values.
    per_task = results_dict.get("per_task", [])
    task_utils: dict[str, float | None] = {}
    task_actions: dict[str, str] = {}
    task_success: dict[str, bool | None] = {}

    if per_task:
        for tr in per_task:
            tid = tr.get("task_id", "")
            util = tr.get("utility")
            # Strict utility validation
            if tr.get("outcome_status", "EXECUTED") == "EXECUTED":
                if util is None:
                    raise MissingUtilityError(
                        f"Task {tid}: EXECUTED outcome has None utility"
                    )
                if "utility" not in tr:
                    raise MissingUtilityError(
                        f"Task {tid}: EXECUTED outcome missing utility field"
                    )
            task_utils[tid] = util
            task_actions[tid] = tr.get("selected_action", tr.get("action", "unknown"))
            task_success[tid] = tr.get("verified_success")
    else:
        # Use aggregate mean if no per-task data.
        mean_util = results_dict.get("mean_utility")
        for tid in task_ids:
            task_utils[tid] = mean_util
            task_actions[tid] = "unknown"
            task_success[tid] = None

    # Oracle per-task utilities.
    oracle_utils: dict[str, float | None] = {}
    if oracle_results:
        oracle_per_task = oracle_results.get("per_task", [])
        if oracle_per_task:
            for tr in oracle_per_task:
                tid = tr.get("task_id", "")
                oracle_utils[tid] = tr.get("utility")
        else:
            for tid in task_ids:
                oracle_utils[tid] = oracle_results.get("mean_utility")

    # Build task-level results.
    task_results: list[TaskLevelResult] = []
    valid_utils: list[float] = []
    valid_regrets: list[float] = []
    wins = ties = losses = 0
    action_counts: dict[str, int] = {}

    for tid in task_ids:
        util = task_utils.get(tid)
        oracle_util = oracle_utils.get(tid)
        action = task_actions.get(tid, "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1

        regret = None
        if util is not None and oracle_util is not None:
            regret = oracle_util - util
            valid_regrets.append(regret)
            if regret < -1e-9:
                wins += 1
            elif regret > 1e-9:
                losses += 1
            else:
                ties += 1

        if util is not None:
            valid_utils.append(util)

        headroom_contrib = None
        if util is not None and oracle_util is not None:
            headroom_contrib = oracle_util - util

        family = task_families.get(tid, "unknown")
        archetype = task_archetypes.get(tid, "unknown")
        margin = task_margins.get(tid, "unknown")

        outcome_status = "MEASURED" if util is not None else "UNAVAILABLE"

        task_results.append(TaskLevelResult(
            task_id=tid,
            selected_action=action,
            selected_action_available=util is not None,
            selected_utility=util,
            verified_success=task_success.get(tid),
            oracle_action="",
            oracle_utility=oracle_util,
            regret=regret,
            family=family,
            archetype=archetype,
            margin_bucket=margin,
            headroom_contribution=headroom_contrib,
            outcome_status=outcome_status,
        ))

    # Aggregate metrics.
    n_tasks = len(task_ids)
    mean_utility = sum(valid_utils) / len(valid_utils) if valid_utils else None
    median_utility = _median(valid_utils) if valid_utils else None

    n_success = sum(1 for s in task_success.values() if s is True)
    verified_success_rate = n_success / n_tasks if n_tasks > 0 else None

    mean_regret = sum(valid_regrets) / len(valid_regrets) if valid_regrets else None
    median_regret = _median(valid_regrets) if valid_regrets else None
    p95_regret = _percentile(valid_regrets, 95) if valid_regrets else None

    action_distribution = {
        a: c / n_tasks for a, c in action_counts.items()
    } if n_tasks > 0 else {}

    # Family/archetype/margin metrics.
    family_metrics = _group_metrics(task_results, "family")
    archetype_metrics = _group_metrics(task_results, "archetype")
    margin_metrics = _group_metrics(task_results, "margin_bucket")

    task_ids_digest = compute_task_ids_digest(task_ids)

    # Headroom vs reference.
    headroom = None
    if mean_utility is not None and reference_mean is not None:
        oracle_mean = oracle_results.get("mean_utility") if oracle_results else None
        if oracle_mean is not None:
            headroom = compute_recovered_headroom(
                candidate_mean=mean_utility,
                reference_mean=reference_mean,
                oracle_mean=oracle_mean,
                task_count=n_tasks,
                task_ids_digest=task_ids_digest,
                reference_policy_id=reference_policy_id,
            )

    return PolicyEvalResult(
        policy_id=policy_id,
        n_tasks=n_tasks,
        mean_utility=mean_utility,
        median_utility=median_utility,
        verified_success_rate=verified_success_rate,
        wins=wins,
        ties=ties,
        losses=losses,
        mean_regret=mean_regret,
        median_regret=median_regret,
        p95_regret=p95_regret,
        action_distribution=action_distribution,
        family_metrics=family_metrics,
        archetype_metrics=archetype_metrics,
        margin_metrics=margin_metrics,
        task_ids_digest=task_ids_digest,
        task_results=task_results,
        headroom_vs_reference=headroom,
    )


def evaluate_all_policies(
    evidence: Any,
    excluded_task_ids: list[str] | None = None,
) -> dict[str, PolicyEvalResult]:
    """Evaluate all policies from loaded evidence using the canonical evaluator.

    Args:
        evidence: LoadedEvidence object from data.load_evidence().
        excluded_task_ids: Task IDs to exclude (non-equivalent or incomplete).

    Returns:
        Dict of policy_id -> PolicyEvalResult.
    """
    excluded = set(excluded_task_ids or [])

    # Get test task IDs from split manifest.
    test_task_ids = evidence.split_manifest.get("splits", {}).get("test", {}).get("task_ids", [])
    if not test_task_ids:
        # Use count to generate placeholder IDs.
        n_test = evidence.split_manifest.get("splits", {}).get("test", {}).get("count", 0)
        test_task_ids = [f"test_{i:04d}" for i in range(n_test)]

    # Filter out excluded tasks.
    test_task_ids = [t for t in test_task_ids if t not in excluded]

    results = {}

    # Evaluate each policy.
    policy_results_map = {
        "frozen_baseline": evidence.baseline_results,
        "candidate_executed": evidence.candidate_results,
        "sham_permuted": evidence.sham_results,
        "oracle": evidence.oracle_results,
    }

    # Sham mean for headroom reference.
    sham_mean = evidence.sham_results.get("mean_utility")
    baseline_mean = evidence.baseline_results.get("mean_utility")

    for pid, rdict in policy_results_map.items():
        ref_mean = None
        ref_id = ""
        if pid in ("candidate_executed", "sham_permuted"):
            ref_mean = baseline_mean
            ref_id = "frozen_baseline"
        elif pid == "oracle":
            ref_mean = baseline_mean
            ref_id = "frozen_baseline"

        result = evaluate_policy_from_results(
            policy_id=pid,
            results_dict=rdict,
            task_ids=test_task_ids,
            oracle_results=evidence.oracle_results,
            reference_mean=ref_mean,
            reference_policy_id=ref_id,
        )
        results[pid] = result

    # Compute candidate headroom vs sham.
    cand = results.get("candidate_executed")
    if cand and cand.mean_utility is not None and sham_mean is not None:
        oracle_mean = evidence.oracle_results.get("mean_utility")
        if oracle_mean is not None:
            cand_headroom_vs_sham = compute_recovered_headroom(
                candidate_mean=cand.mean_utility,
                reference_mean=sham_mean,
                oracle_mean=oracle_mean,
                task_count=cand.n_tasks,
                task_ids_digest=cand.task_ids_digest,
                reference_policy_id="sham_permuted",
            )
            # Attach as additional field in dict output.
            results["candidate_executed_vs_sham"] = cand_headroom_vs_sham

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _group_metrics(task_results: list[TaskLevelResult], group_field: str) -> dict[str, dict[str, float]]:
    """Compute metrics grouped by a field."""
    groups: dict[str, list[TaskLevelResult]] = {}
    for tr in task_results:
        key = getattr(tr, group_field, "unknown")
        groups.setdefault(key, []).append(tr)

    metrics = {}
    for key, items in groups.items():
        utils = [t.selected_utility for t in items if t.selected_utility is not None]
        regrets = [t.regret for t in items if t.regret is not None]
        metrics[key] = {
            "n_tasks": len(items),
            "mean_utility": sum(utils) / len(utils) if utils else 0.0,
            "mean_regret": sum(regrets) / len(regrets) if regrets else 0.0,
        }
    return metrics
