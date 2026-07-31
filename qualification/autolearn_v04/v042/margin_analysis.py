"""Utility-margin analysis and margin-bucket diagnostics.

For each task we sort the eligible action utilities and compute three
margin quantities:

  - ``margin_1_2``  = U_i(1) - U_i(2)  (best minus second-best)
  - ``margin_1_mean`` = U_i(1) - mean(U_i(other))  (best minus mean of rest)
  - ``oracle_advantage_over_sham`` = U_i(best) - E_{a~P_sham}[U_i(a)]

Tasks are then assigned to predeclared margin buckets and, within each
bucket, we report policy utilities (candidate, baseline, sham, oracle),
candidate-minus-sham and candidate-minus-baseline deltas with bootstrap
confidence bounds, classification accuracy, and headroom recovered.

A secondary Gate A diagnostic — ``Gate_A_high_margin`` — evaluates the
candidate only on meaningful + high_value tasks.  This does **not**
replace the full Gate A but helps identify whether the router works
where routing matters.

The output of :func:`analyze_margins` is the content for
``utility_margin_analysis.json``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import LoadedRun
from .bootstrap import task_paired_bootstrap
from .policy_controls import (
    make_oracle_selector,
    make_baseline_selector,
    make_candidate_selector,
    make_sham_global_selector,
    evaluate_policy_on_tasks,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Predeclared margin buckets defined by margin_1_2 thresholds.
MARGIN_BUCKETS: dict[str, dict[str, float | None]] = {
    "near_tie": {"min": None, "max": 0.05},
    "weak": {"min": 0.05, "max": 0.25},
    "meaningful": {"min": 0.25, "max": 1.0},
    "high_value": {"min": 1.0, "max": None},
}

#: Buckets where routing actually matters — used for the secondary
#: Gate A diagnostic.
HIGH_MARGIN_BUCKETS: list[str] = ["meaningful", "high_value"]


# ---------------------------------------------------------------------------
# Per-task margin computation
# ---------------------------------------------------------------------------


def _sorted_eligible_utilities(run: LoadedRun, task_id: str) -> list[tuple[str, float]]:
    """Return eligible (action, utility) pairs sorted by utility descending.

    Only actions that are both in the task's ``allowed_actions`` and have
    an executed outcome are considered.
    """
    task = run.tasks.get(task_id)
    if task is None:
        return []
    allowed = set(task.allowed_actions)
    outcomes = run.outcomes_by_task.get(task_id, [])
    pairs = [
        (o.action_id, o.utility)
        for o in outcomes
        if o.availability == "executed"
        and o.utility is not None
        and o.action_id in allowed
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _sham_expected_utility(
    run: LoadedRun,
    task_id: str,
    sham_selector,
) -> float:
    """Compute E_{a~P_sham}[U_i(a)] for a single task.

    Because the sham selector is deterministic given a seed, we approximate
    the expectation by evaluating the selector once.  For a more faithful
    expectation the caller would use a stochastic sham; here we use the
    trained sham policy's argmax as a point estimate, consistent with the
    rest of the v0.4.2 control ladder.
    """
    task = run.tasks.get(task_id)
    if task is None:
        return 0.0
    action = sham_selector(task_id, task)
    if action is None:
        return 0.0
    u = run.get_utility_for_action(task_id, action)
    return u if u is not None else 0.0


def _compute_task_margins(
    run: LoadedRun,
    task_ids: list[str],
    sham_selector,
) -> list[dict[str, Any]]:
    """Compute margin quantities for every task with executed outcomes.

    Returns a list of per-task dicts with:
      - task_id, split, family, archetype
      - sorted_utilities (descending)
      - best_action, best_utility, second_best_utility
      - margin_1_2, margin_1_mean
      - oracle_advantage_over_sham
      - bucket assignment
    """
    records: list[dict[str, Any]] = []
    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        pairs = _sorted_eligible_utilities(run, tid)
        if not pairs:
            continue

        utils = [u for _, u in pairs]
        best_action, best_util = pairs[0]
        second_util = pairs[1][1] if len(pairs) > 1 else best_util

        margin_1_2 = best_util - second_util
        if len(pairs) > 1:
            other_utils = utils[1:]
            margin_1_mean = best_util - float(np.mean(other_utils))
        else:
            margin_1_mean = 0.0

        sham_exp = _sham_expected_utility(run, tid, sham_selector)
        oracle_advantage = best_util - sham_exp

        bucket = _assign_bucket(margin_1_2)

        records.append({
            "task_id": tid,
            "split": task.split,
            "family": task.family,
            "archetype": task.archetype,
            "sorted_utilities": [{"action": a, "utility": u} for a, u in pairs],
            "best_action": best_action,
            "best_utility": best_util,
            "second_best_utility": second_util,
            "margin_1_2": float(margin_1_2),
            "margin_1_mean": float(margin_1_mean),
            "oracle_advantage_over_sham": float(oracle_advantage),
            "bucket": bucket,
        })
    return records


def _assign_bucket(margin_1_2: float) -> str:
    """Assign a margin value to one of the predeclared buckets."""
    for name, bounds in MARGIN_BUCKETS.items():
        lo = bounds["min"]
        hi = bounds["max"]
        if lo is not None and hi is not None:
            if lo <= margin_1_2 < hi:
                return name
        elif lo is None and hi is not None:
            if margin_1_2 < hi:
                return name
        elif lo is not None and hi is None:
            if margin_1_2 >= lo:
                return name
    # Fallback (should not happen with the current bucket definitions).
    return "high_value"


# ---------------------------------------------------------------------------
# Bucket-level analysis
# ---------------------------------------------------------------------------


def _analyze_bucket(
    run: LoadedRun,
    bucket_name: str,
    bucket_task_ids: list[str],
    oracle_res: dict[str, Any],
    baseline_res: dict[str, Any],
    candidate_res: dict[str, Any],
    sham_res: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Compute policy utilities and diagnostics for one margin bucket.

    Reports:
      - task count
      - mean candidate / baseline / sham / oracle utility
      - candidate-minus-sham and candidate-minus-baseline (mean + bootstrap CI)
      - classification accuracy (fraction where candidate picks the best action)
      - headroom recovered: (U_candidate - U_sham) / (U_oracle - U_sham)
    """
    n = len(bucket_task_ids)
    if n == 0:
        return {
            "bucket": bucket_name,
            "n_tasks": 0,
            "candidate_utility": 0.0,
            "baseline_utility": 0.0,
            "sham_utility": 0.0,
            "oracle_utility": 0.0,
            "candidate_minus_sham": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0},
            "candidate_minus_baseline": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0},
            "classification_accuracy": 0.0,
            "headroom_recovered": 0.0,
            "task_ids": [],
        }

    # Map task_id -> utility for each policy.
    def _util_map(res: dict[str, Any]) -> dict[str, float]:
        return {pt["task_id"]: pt["utility"] for pt in res.get("per_task", [])}

    oracle_map = _util_map(oracle_res)
    baseline_map = _util_map(baseline_res)
    candidate_map = _util_map(candidate_res)
    sham_map = _util_map(sham_res)

    # Also map task_id -> best_action and selected_action for accuracy.
    candidate_selected = {pt["task_id"]: pt["selected_action"] for pt in candidate_res.get("per_task", [])}
    best_action_map = {pt["task_id"]: pt["best_action"] for pt in candidate_res.get("per_task", [])}

    u_cand = np.array([candidate_map.get(tid, 0.0) for tid in bucket_task_ids])
    u_base = np.array([baseline_map.get(tid, 0.0) for tid in bucket_task_ids])
    u_sham = np.array([sham_map.get(tid, 0.0) for tid in bucket_task_ids])
    u_oracle = np.array([oracle_map.get(tid, 0.0) for tid in bucket_task_ids])

    # Bootstrap CIs.
    ci_cand_sham = task_paired_bootstrap(
        u_cand - u_sham, n_bootstrap=n_bootstrap, seed=seed,
    )
    ci_cand_base = task_paired_bootstrap(
        u_cand - u_base, n_bootstrap=n_bootstrap, seed=seed,
    )

    # Classification accuracy: fraction of tasks where candidate == best.
    correct = 0
    for tid in bucket_task_ids:
        sel = candidate_selected.get(tid)
        best = best_action_map.get(tid)
        if sel is not None and best is not None and sel == best:
            correct += 1
    accuracy = correct / n if n > 0 else 0.0

    # Headroom recovered.
    denom = u_oracle.mean() - u_sham.mean()
    if abs(denom) > 1e-10:
        headroom_recovered = float((u_cand.mean() - u_sham.mean()) / denom)
    else:
        headroom_recovered = 0.0

    return {
        "bucket": bucket_name,
        "n_tasks": n,
        "candidate_utility": float(u_cand.mean()),
        "baseline_utility": float(u_base.mean()),
        "sham_utility": float(u_sham.mean()),
        "oracle_utility": float(u_oracle.mean()),
        "candidate_minus_sham": ci_cand_sham,
        "candidate_minus_baseline": ci_cand_base,
        "classification_accuracy": accuracy,
        "headroom_recovered": headroom_recovered,
        "task_ids": bucket_task_ids,
    }


# ---------------------------------------------------------------------------
# Secondary Gate A diagnostic (high-margin only)
# ---------------------------------------------------------------------------


def _gate_a_high_margin(
    run: LoadedRun,
    high_margin_task_ids: list[str],
    oracle_res: dict[str, Any],
    baseline_res: dict[str, Any],
    candidate_res: dict[str, Any],
    sham_res: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate Gate A only on meaningful + high_value tasks.

    This is a *secondary* diagnostic — it does not replace the full Gate A
    evaluation.  Its purpose is to show whether the candidate router works
    on the subset of tasks where routing actually matters (i.e. where the
    oracle margin is large enough that action selection has real utility
    consequences).
    """
    if len(high_margin_task_ids) == 0:
        return {
            "n_tasks": 0,
            "candidate_utility": 0.0,
            "sham_utility": 0.0,
            "baseline_utility": 0.0,
            "oracle_utility": 0.0,
            "candidate_vs_sham": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0},
            "candidate_vs_baseline": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0},
            "headroom_recovered_vs_sham": 0.0,
            "headroom_recovered_vs_baseline": 0.0,
            "classification_accuracy": 0.0,
            "note": "No tasks in meaningful + high_value buckets.",
        }

    def _util_map(res: dict[str, Any]) -> dict[str, float]:
        return {pt["task_id"]: pt["utility"] for pt in res.get("per_task", [])}

    oracle_map = _util_map(oracle_res)
    baseline_map = _util_map(baseline_res)
    candidate_map = _util_map(candidate_res)
    sham_map = _util_map(sham_res)

    u_cand = np.array([candidate_map.get(tid, 0.0) for tid in high_margin_task_ids])
    u_base = np.array([baseline_map.get(tid, 0.0) for tid in high_margin_task_ids])
    u_sham = np.array([sham_map.get(tid, 0.0) for tid in high_margin_task_ids])
    u_oracle = np.array([oracle_map.get(tid, 0.0) for tid in high_margin_task_ids])

    ci_vs_sham = task_paired_bootstrap(
        u_cand - u_sham, n_bootstrap=n_bootstrap, seed=seed,
    )
    ci_vs_baseline = task_paired_bootstrap(
        u_cand - u_base, n_bootstrap=n_bootstrap, seed=seed,
    )

    denom_sham = u_oracle.mean() - u_sham.mean()
    denom_base = u_oracle.mean() - u_base.mean()
    r_sham = float((u_cand.mean() - u_sham.mean()) / denom_sham) if abs(denom_sham) > 1e-10 else 0.0
    r_base = float((u_cand.mean() - u_base.mean()) / denom_base) if abs(denom_base) > 1e-10 else 0.0

    # Classification accuracy on high-margin tasks.
    candidate_selected = {pt["task_id"]: pt["selected_action"] for pt in candidate_res.get("per_task", [])}
    best_action_map = {pt["task_id"]: pt["best_action"] for pt in candidate_res.get("per_task", [])}
    correct = sum(
        1 for tid in high_margin_task_ids
        if candidate_selected.get(tid) is not None
        and candidate_selected.get(tid) == best_action_map.get(tid)
    )
    accuracy = correct / len(high_margin_task_ids) if high_margin_task_ids else 0.0

    return {
        "n_tasks": len(high_margin_task_ids),
        "candidate_utility": float(u_cand.mean()),
        "sham_utility": float(u_sham.mean()),
        "baseline_utility": float(u_base.mean()),
        "oracle_utility": float(u_oracle.mean()),
        "candidate_vs_sham": ci_vs_sham,
        "candidate_vs_baseline": ci_vs_baseline,
        "headroom_recovered_vs_sham": r_sham,
        "headroom_recovered_vs_baseline": r_base,
        "classification_accuracy": accuracy,
        "note": (
            "Secondary Gate A diagnostic restricted to meaningful + "
            "high_value margin buckets. Does NOT replace the full Gate A."
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_margins(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze utility margins and produce margin-bucket diagnostics.

    For each task the eligible action utilities are sorted descending and
    three margin quantities are computed:

      - ``margin_1_2`` — best minus second-best utility.
      - ``margin_1_mean`` — best minus the mean of all remaining utilities.
      - ``oracle_advantage_over_sham`` — best utility minus the expected
        sham-policy utility for that task.

    Tasks are assigned to one of four predeclared buckets
    (``near_tie``, ``weak``, ``meaningful``, ``high_value``) based on
    ``margin_1_2``.  Within each bucket the candidate, baseline, sham,
    and oracle policy utilities are reported, along with
    candidate-minus-sham and candidate-minus-baseline bootstrap
    confidence bounds, classification accuracy, and headroom recovered.

    A secondary ``Gate_A_high_margin`` diagnostic evaluates the candidate
    only on the ``meaningful`` and ``high_value`` buckets.

    Args:
        run: Loaded run data.
        task_ids: Tasks to analyze.  Defaults to the test split.
        n_bootstrap: Number of bootstrap replicates for confidence bounds.
        seed: Random seed for bootstrap.

    Returns:
        Content for ``utility_margin_analysis.json``.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # Build policy selectors and evaluate once over the full task set.
    oracle_sel = make_oracle_selector(run)
    baseline_sel = make_baseline_selector(run)
    candidate_sel = make_candidate_selector(run)
    sham_sel = make_sham_global_selector(run, seed=12345)

    oracle_res = evaluate_policy_on_tasks(run, task_ids, oracle_sel, "oracle")
    baseline_res = evaluate_policy_on_tasks(run, task_ids, baseline_sel, "baseline")
    candidate_res = evaluate_policy_on_tasks(run, task_ids, candidate_sel, "candidate")
    sham_res = evaluate_policy_on_tasks(run, task_ids, sham_sel, "sham_global")

    # Compute per-task margins.
    task_margins = _compute_task_margins(run, task_ids, sham_sel)

    # Group task IDs by bucket.
    bucket_task_ids: dict[str, list[str]] = {name: [] for name in MARGIN_BUCKETS}
    for rec in task_margins:
        bucket_task_ids[rec["bucket"]].append(rec["task_id"])

    # Analyze each bucket.
    bucket_results: dict[str, Any] = {}
    for name in MARGIN_BUCKETS:
        bucket_results[name] = _analyze_bucket(
            run,
            name,
            bucket_task_ids[name],
            oracle_res,
            baseline_res,
            candidate_res,
            sham_res,
            n_bootstrap,
            seed,
        )

    # Secondary Gate A diagnostic on high-margin tasks.
    high_margin_ids = []
    for name in HIGH_MARGIN_BUCKETS:
        high_margin_ids.extend(bucket_task_ids[name])

    gate_a_high = _gate_a_high_margin(
        run,
        high_margin_ids,
        oracle_res,
        baseline_res,
        candidate_res,
        sham_res,
        n_bootstrap,
        seed,
    )

    return {
        "n_tasks": len(task_margins),
        "margin_definitions": {
            "margin_1_2": "U_i(1) - U_i(2)",
            "margin_1_mean": "U_i(1) - mean(U_i(other))",
            "oracle_advantage_over_sham": "U_i(best) - E_{a~P_sham}[U_i(a)]",
        },
        "bucket_thresholds": {
            name: {"min": b["min"], "max": b["max"]}
            for name, b in MARGIN_BUCKETS.items()
        },
        "buckets": bucket_results,
        "Gate_A_high_margin": gate_a_high,
        "per_task": task_margins,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }
