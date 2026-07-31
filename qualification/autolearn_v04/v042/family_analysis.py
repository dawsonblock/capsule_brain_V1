"""Per-family and per-archetype analysis.

Breaks down policy performance, action distributions, utility, regret,
and failure rates by task family and by archetype.  Groups with fewer
than :data:`MIN_SAMPLE_SIZE` tasks are flagged as exploratory and their
action distributions are reported with shrinkage smoothing.

For every family and archetype we report:

* task count
* per-action verification success rates
* best-action (oracle) distribution
* candidate / baseline / sham / oracle action distributions
* candidate / baseline / sham / oracle mean utility
* candidate-minus-sham and candidate-minus-baseline utility deltas
* candidate regret (mean, median, p95)
* mean utility margin (best minus second-best)
* verification failure rate
* runtime failure rate

A family-regression table comparing candidate vs baseline per family
with bootstrap confidence intervals is also produced.
"""
from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Any

from .data import LoadedRun
from .policy_controls import (
    evaluate_policy_on_tasks,
    make_candidate_selector,
    make_baseline_selector,
    make_sham_global_selector,
    make_oracle_selector,
)
from .bootstrap import task_paired_bootstrap


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Groups with fewer than this many tasks are flagged exploratory.
MIN_SAMPLE_SIZE: int = 5

#: Pseudocount added to each action bucket for shrinkage smoothing.
SHRINKAGE_PRIOR: float = 1.0

#: Bootstrap iterations for the family-regression table.
_N_BOOTSTRAP: int = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shrinkage_distribution(
    counts: Counter,
    actions: list[str],
    prior: float = SHRINKAGE_PRIOR,
) -> dict[str, float]:
    """Compute a shrinkage-smoothed categorical distribution.

    Adds *prior* pseudocounts to every action bucket (Laplace / add-one
    style smoothing) so that small-sample groups do not produce
    degenerate zero-probability estimates.

    Parameters
    ----------
    counts:
        Raw action counts.
    actions:
        Full action vocabulary.
    prior:
        Pseudocount per action.

    Returns
    -------
    dict
        Smoothed probability for each action.
    """
    total = sum(counts.values()) + prior * len(actions)
    if total == 0:
        return {a: 0.0 for a in actions}
    return {a: (counts.get(a, 0) + prior) / total for a in actions}


def _action_success_rates(
    run: LoadedRun,
    task_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Per-action verification success rate among executed outcomes.

    Parameters
    ----------
    run:
        Loaded run data.
    task_ids:
        Tasks in the group.

    Returns
    -------
    dict
        ``{action: {"n": int, "success_count": int, "success_rate": float}}``
    """
    action_total: Counter = Counter()
    action_success: Counter = Counter()
    for tid in task_ids:
        for o in run.outcomes_by_task.get(tid, []):
            if o.availability != "executed":
                continue
            action_total[o.action_id] += 1
            if o.verification == "success":
                action_success[o.action_id] += 1

    result: dict[str, dict[str, Any]] = {}
    for a in run.all_actions:
        t = action_total.get(a, 0)
        s = action_success.get(a, 0)
        result[a] = {
            "n": t,
            "success_count": s,
            "success_rate": s / t if t > 0 else 0.0,
        }
    return result


def _compute_margin(run: LoadedRun, task_id: str) -> float:
    """Utility margin: best minus second-best executed utility.

    Returns 0.0 when fewer than two executed outcomes are available.
    """
    outcomes = run.outcomes_by_task.get(task_id, [])
    executed = [
        o for o in outcomes
        if o.availability == "executed" and o.utility is not None
    ]
    if len(executed) < 2:
        return 0.0
    utils = sorted((o.utility for o in executed), reverse=True)
    return float(utils[0] - utils[1])


def _failure_rates(run: LoadedRun, task_ids: list[str]) -> dict[str, Any]:
    """Verification and runtime failure rates for a group of tasks.

    * **verification_failure_rate** -- fraction of all outcomes where
      verification did not return ``"success"``.
    * **runtime_failure_rate** -- fraction of outcomes where the action
      was not executed (``availability != "executed"``).
    """
    n_outcomes = 0
    n_verify_fail = 0
    n_runtime_fail = 0
    for tid in task_ids:
        for o in run.outcomes_by_task.get(tid, []):
            n_outcomes += 1
            if o.verification != "success":
                n_verify_fail += 1
            if o.availability != "executed":
                n_runtime_fail += 1

    if n_outcomes == 0:
        return {
            "verification_failure_rate": 0.0,
            "runtime_failure_rate": 0.0,
            "n_outcomes": 0,
        }
    return {
        "verification_failure_rate": n_verify_fail / n_outcomes,
        "runtime_failure_rate": n_runtime_fail / n_outcomes,
        "n_outcomes": n_outcomes,
    }


def _index_by_task(res: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index per-task results by task_id for O(1) lookup."""
    return {pt["task_id"]: pt for pt in res["per_task"]}


def _analyze_group(
    run: LoadedRun,
    task_ids: list[str],
    actions: list[str],
    candidate_res: dict[str, Any],
    baseline_res: dict[str, Any],
    sham_res: dict[str, Any],
    oracle_res: dict[str, Any],
) -> dict[str, Any]:
    """Compute all metrics for a single family or archetype group.

    Parameters
    ----------
    run:
        Loaded run data.
    task_ids:
        Task IDs belonging to the group.
    actions:
        Full action vocabulary (for distribution alignment).
    candidate_res, baseline_res, sham_res, oracle_res:
        Full policy evaluation results (evaluated on the superset of
        tasks).  Per-task entries are indexed by task_id internally.
    """
    n = len(task_ids)
    exploratory = n < MIN_SAMPLE_SIZE

    cand_idx = _index_by_task(candidate_res)
    base_idx = _index_by_task(baseline_res)
    sham_idx = _index_by_task(sham_res)
    oracle_idx = _index_by_task(oracle_res)

    # -- Best-action (oracle) distribution --------------------------------
    best_action_counts: Counter = Counter()
    for tid in task_ids:
        ba = run.get_best_action(tid)
        if ba is not None:
            best_action_counts[ba] += 1

    # -- Policy action distributions --------------------------------------
    def _action_dist(idx: dict[str, dict[str, Any]]) -> Counter:
        c: Counter = Counter()
        for tid in task_ids:
            pt = idx.get(tid)
            if pt is not None:
                c[pt["selected_action"]] += 1
        return c

    cand_dist = _action_dist(cand_idx)
    base_dist = _action_dist(base_idx)
    sham_dist = _action_dist(sham_idx)
    oracle_dist = _action_dist(oracle_idx)

    # -- Policy mean utilities --------------------------------------------
    def _mean_utility(idx: dict[str, dict[str, Any]]) -> float:
        vals = [idx[tid]["utility"] for tid in task_ids if tid in idx]
        return float(np.mean(vals)) if vals else 0.0

    u_cand = _mean_utility(cand_idx)
    u_base = _mean_utility(base_idx)
    u_sham = _mean_utility(sham_idx)
    u_oracle = _mean_utility(oracle_idx)

    # -- Candidate regret --------------------------------------------------
    cand_regrets = np.array([
        cand_idx[tid]["oracle_utility"] - cand_idx[tid]["utility"]
        for tid in task_ids
        if tid in cand_idx
    ])
    if len(cand_regrets) > 0:
        candidate_regret: dict[str, Any] = {
            "mean": float(cand_regrets.mean()),
            "median": float(np.median(cand_regrets)),
            "p95": float(np.percentile(cand_regrets, 95)),
            "n": len(cand_regrets),
        }
    else:
        candidate_regret = {"mean": 0.0, "median": 0.0, "p95": 0.0, "n": 0}

    # -- Mean utility margin -----------------------------------------------
    margins = [_compute_margin(run, tid) for tid in task_ids]
    mean_margin = float(np.mean(margins)) if margins else 0.0

    # -- Failure rates -----------------------------------------------------
    failure = _failure_rates(run, task_ids)

    # -- Action success rates ---------------------------------------------
    action_success = _action_success_rates(run, task_ids)

    return {
        "n_tasks": n,
        "exploratory": exploratory,
        "action_success_rates": action_success,
        "best_action_distribution": {
            "counts": dict(best_action_counts),
            "shrinkage": _shrinkage_distribution(best_action_counts, actions),
        },
        "policy_action_distributions": {
            "candidate": {
                "counts": dict(cand_dist),
                "shrinkage": _shrinkage_distribution(cand_dist, actions),
            },
            "baseline": {
                "counts": dict(base_dist),
                "shrinkage": _shrinkage_distribution(base_dist, actions),
            },
            "sham": {
                "counts": dict(sham_dist),
                "shrinkage": _shrinkage_distribution(sham_dist, actions),
            },
            "oracle": {
                "counts": dict(oracle_dist),
                "shrinkage": _shrinkage_distribution(oracle_dist, actions),
            },
        },
        "policy_utilities": {
            "candidate": u_cand,
            "baseline": u_base,
            "sham": u_sham,
            "oracle": u_oracle,
        },
        "candidate_minus_sham": u_cand - u_sham,
        "candidate_minus_baseline": u_cand - u_base,
        "candidate_regret": candidate_regret,
        "mean_utility_margin": mean_margin,
        "verification_failure_rate": failure["verification_failure_rate"],
        "runtime_failure_rate": failure["runtime_failure_rate"],
        "n_outcomes": failure["n_outcomes"],
    }


def _family_regression_table(
    run: LoadedRun,
    task_ids: list[str],
    candidate_res: dict[str, Any],
    baseline_res: dict[str, Any],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Per-family candidate-vs-baseline comparison with bootstrap CIs.

    For each family, computes the mean per-task utility delta
    (candidate - baseline) and a 95% bootstrap confidence interval.
    Rows are flagged as exploratory when the family has fewer than
    :data:`MIN_SAMPLE_SIZE` tasks.
    """
    cand_idx = _index_by_task(candidate_res)
    base_idx = _index_by_task(baseline_res)

    # Group task_ids by family.
    family_tasks: dict[str, list[str]] = {}
    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        family_tasks.setdefault(task.family, []).append(tid)

    table: list[dict[str, Any]] = []
    for fam in sorted(family_tasks):
        tids = family_tasks[fam]
        deltas = np.array([
            cand_idx[tid]["utility"] - base_idx[tid]["utility"]
            for tid in tids
            if tid in cand_idx and tid in base_idx
        ])
        if len(deltas) == 0:
            continue

        ci = task_paired_bootstrap(
            deltas, n_bootstrap=_N_BOOTSTRAP, seed=seed,
        )
        table.append({
            "family": fam,
            "n_tasks": len(tids),
            "exploratory": len(tids) < MIN_SAMPLE_SIZE,
            "mean_delta": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "std_error": ci["std_error"],
            "candidate_better": ci["ci_low"] > 0,
            "baseline_better": ci["ci_high"] < 0,
        })
    return table


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_families(
    run: LoadedRun,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze performance by task family and by archetype.

    Evaluates the candidate, baseline, sham (global), and oracle policies
    on the given tasks, then groups results by ``family`` and
    ``archetype`` to produce per-group metrics and a family-regression
    table.

    Parameters
    ----------
    run:
        Loaded run data.
    task_ids:
        Tasks to analyze.  Defaults to the test split.

    Returns
    -------
    dict
        Keys:

        * ``"family_analysis"`` -- content for ``family_analysis.json``.
        * ``"archetype_analysis"`` -- content for
          ``archetype_analysis.json``.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    actions = run.all_actions

    # -- Evaluate policies once on the full task set -----------------------
    candidate_res = evaluate_policy_on_tasks(
        run, task_ids, make_candidate_selector(run), "candidate",
    )
    baseline_res = evaluate_policy_on_tasks(
        run, task_ids, make_baseline_selector(run), "baseline",
    )
    sham_res = evaluate_policy_on_tasks(
        run, task_ids, make_sham_global_selector(run, seed=12345), "sham",
    )
    oracle_res = evaluate_policy_on_tasks(
        run, task_ids, make_oracle_selector(run), "oracle",
    )

    # -- Group tasks by family and archetype -------------------------------
    family_tasks: dict[str, list[str]] = {}
    archetype_tasks: dict[str, list[str]] = {}
    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        family_tasks.setdefault(task.family, []).append(tid)
        archetype_tasks.setdefault(task.archetype, []).append(tid)

    # -- Family analysis ---------------------------------------------------
    family_results: dict[str, Any] = {}
    for fam in sorted(family_tasks):
        family_results[fam] = _analyze_group(
            run, family_tasks[fam], actions,
            candidate_res, baseline_res, sham_res, oracle_res,
        )

    family_regression = _family_regression_table(
        run, task_ids, candidate_res, baseline_res,
    )

    # -- Archetype analysis ------------------------------------------------
    archetype_results: dict[str, Any] = {}
    for arch in sorted(archetype_tasks):
        archetype_results[arch] = _analyze_group(
            run, archetype_tasks[arch], actions,
            candidate_res, baseline_res, sham_res, oracle_res,
        )

    return {
        "family_analysis": {
            "n_tasks": len(task_ids),
            "min_sample_size": MIN_SAMPLE_SIZE,
            "shrinkage_prior": SHRINKAGE_PRIOR,
            "families": family_results,
            "family_regression_table": family_regression,
        },
        "archetype_analysis": {
            "n_tasks": len(task_ids),
            "min_sample_size": MIN_SAMPLE_SIZE,
            "shrinkage_prior": SHRINKAGE_PRIOR,
            "archetypes": archetype_results,
        },
    }
