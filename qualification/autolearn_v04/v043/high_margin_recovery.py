"""High-margin task recovery analysis for v0.4.3.

Re-evaluates the candidate policy on high-margin tasks — those where the
oracle margin (``oracle_utility - second_best_utility``) is large enough
that action selection has real utility consequences.

Tasks are assigned to predeclared margin buckets:

* ``NEAR_TIE``    — margin < 0.1
* ``WEAK``        — 0.1 <= margin < 0.5
* ``MEANINGFUL``  — 0.5 <= margin < 2.0
* ``HIGH_VALUE``  — margin >= 2.0

For ``MEANINGFUL`` and ``HIGH_VALUE`` tasks the module reports candidate
raw / executed utility, baseline utility, primary sham utility, frequency
utility, oracle utility, candidate-minus-sham and candidate-minus-baseline
deltas, headroom recovered, classification accuracy, utility regret, and
bootstrap confidence intervals.

Gate A5 definition: the candidate executed policy must recover at least
10% of oracle-sham headroom on ``MEANINGFUL + HIGH_VALUE`` tasks with a
nonnegative lower confidence bound.

The output of :func:`evaluate_high_margin_recovery` is a JSON-serializable
``high_margin_recovery`` dict.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import LoadedRun, TaskInfo
from .canonical_evaluator import (
    PolicyDecisionFn,
    evaluate_policy_from_counterfactuals,
    make_oracle_fn,
    make_baseline_fn,
    make_raw_candidate_fn,
    make_executed_candidate_fn,
    make_executed_sham_fn,
    make_frequency_fn,
    compute_headroom_metrics,
)
from .config import (
    DEFAULT_N_BOOTSTRAP,
    A5_MIN_RECOVERY_FRACTION,
    A5_MARGIN_BUCKETS,
    MINIMUM_DENOMINATOR,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Predeclared margin bucket thresholds.  ``margin`` is
#: ``oracle_utility - second_best_utility``.
MARGIN_BUCKET_DEFINITIONS: dict[str, tuple[float | None, float | None]] = {
    "NEAR_TIE": (None, 0.1),
    "WEAK": (0.1, 0.5),
    "MEANINGFUL": (0.5, 2.0),
    "HIGH_VALUE": (2.0, None),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assign_margin_bucket(margin: float) -> str:
    """Assign a utility margin to one of the predeclared buckets."""
    for name, (lo, hi) in MARGIN_BUCKET_DEFINITIONS.items():
        if lo is None and hi is not None:
            if margin < hi:
                return name
        elif lo is not None and hi is None:
            if margin >= lo:
                return name
        elif lo is not None and hi is not None:
            if lo <= margin < hi:
                return name
    return "HIGH_VALUE"


def _sorted_eligible_utilities(run: LoadedRun, task_id: str) -> list[tuple[str, float]]:
    """Return eligible (action, utility) pairs sorted by utility descending."""
    task = run.tasks.get(task_id)
    if task is None:
        return []
    allowed = set(task.allowed_actions)
    pairs = [
        (o.action_id, o.utility)
        for o in run.outcomes_by_task.get(task_id, [])
        if o.availability == "executed" and o.utility is not None and o.action_id in allowed
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _task_margin(run: LoadedRun, task_id: str) -> tuple[float, str, float, float]:
    """Return (margin, best_action, best_utility, second_best_utility)."""
    pairs = _sorted_eligible_utilities(run, task_id)
    if not pairs:
        return 0.0, "", 0.0, 0.0
    best_action, best_util = pairs[0]
    second_util = pairs[1][1] if len(pairs) > 1 else best_util
    return float(best_util - second_util), best_action, float(best_util), float(second_util)


def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Standard task-level paired bootstrap confidence interval."""
    if len(values) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0}
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    alpha = (1 - confidence) / 2
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(means, 100 * alpha)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha))),
        "std_error": float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
    }


def _bootstrap_recovery_fraction(
    candidate_utils: np.ndarray,
    sham_utils: np.ndarray,
    oracle_utils: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the headroom recovery fraction.

    The recovery fraction is the aggregate ratio
    ``(mean(candidate) - mean(sham)) / (mean(oracle) - mean(sham))``
    computed on each bootstrap resample.
    """
    n = len(candidate_utils)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0}
    rng = np.random.default_rng(seed)
    fractions = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        c = candidate_utils[idx].mean()
        s = sham_utils[idx].mean()
        o = oracle_utils[idx].mean()
        denom = o - s
        fractions[i] = (c - s) / denom if abs(denom) > MINIMUM_DENOMINATOR else 0.0
    point_c = candidate_utils.mean()
    point_s = sham_utils.mean()
    point_o = oracle_utils.mean()
    point_denom = point_o - point_s
    point_frac = (point_c - point_s) / point_denom if abs(point_denom) > MINIMUM_DENOMINATOR else 0.0
    return {
        "mean": float(point_frac),
        "ci_low": float(np.percentile(fractions, 2.5)),
        "ci_high": float(np.percentile(fractions, 97.5)),
        "std_error": float(fractions.std(ddof=1) / np.sqrt(n_bootstrap)) if n_bootstrap > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-task policy utilities
# ---------------------------------------------------------------------------


def _per_task_utilities(
    run: LoadedRun,
    task_ids: list[str],
    policy_fn: PolicyDecisionFn,
) -> dict[str, dict[str, Any]]:
    """Return task_id -> {action, utility} for a policy decision function."""
    out: dict[str, dict[str, Any]] = {}
    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        action = policy_fn(tid, task)
        utility = run.get_utility_for_action(tid, action) if action else None
        out[tid] = {"action": action, "utility": utility}
    return out


# ---------------------------------------------------------------------------
# Bucket analysis
# ---------------------------------------------------------------------------


def _analyze_bucket(
    run: LoadedRun,
    bucket_name: str,
    bucket_task_ids: list[str],
    candidate_raw_map: dict[str, dict[str, Any]],
    candidate_exec_map: dict[str, dict[str, Any]],
    baseline_map: dict[str, dict[str, Any]],
    sham_map: dict[str, dict[str, Any]],
    frequency_map: dict[str, dict[str, Any]],
    oracle_map: dict[str, dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Compute recovery metrics for a single margin bucket."""
    n = len(bucket_task_ids)
    if n == 0:
        return {
            "bucket": bucket_name,
            "n_tasks": 0,
            "candidate_raw_utility": 0.0,
            "candidate_executed_utility": 0.0,
            "baseline_utility": 0.0,
            "primary_sham_utility": 0.0,
            "frequency_utility": 0.0,
            "oracle_utility": 0.0,
            "candidate_minus_sham": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0},
            "candidate_minus_baseline": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0},
            "headroom_recovered": None,
            "classification_accuracy": 0.0,
            "utility_regret": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std_error": 0.0},
            "task_ids": [],
        }

    def _arr(field: str, m: dict[str, dict[str, Any]]) -> np.ndarray:
        return np.array(
            [m.get(tid, {}).get(field) or 0.0 for tid in bucket_task_ids],
            dtype=float,
        )

    u_raw = _arr("utility", candidate_raw_map)
    u_exec = _arr("utility", candidate_exec_map)
    u_base = _arr("utility", baseline_map)
    u_sham = _arr("utility", sham_map)
    u_freq = _arr("utility", frequency_map)
    u_oracle = _arr("utility", oracle_map)

    ci_cand_sham = _bootstrap_ci(u_exec - u_sham, n_bootstrap=n_bootstrap, seed=seed)
    ci_cand_base = _bootstrap_ci(u_exec - u_base, n_bootstrap=n_bootstrap, seed=seed)
    regret = u_oracle - u_exec
    ci_regret = _bootstrap_ci(regret, n_bootstrap=n_bootstrap, seed=seed)

    # Classification accuracy: fraction where candidate executed == oracle.
    correct = 0
    for tid in bucket_task_ids:
        sel = candidate_exec_map.get(tid, {}).get("action")
        ora = oracle_map.get(tid, {}).get("action")
        if sel is not None and ora is not None and sel == ora:
            correct += 1
    accuracy = correct / n if n > 0 else 0.0

    headroom = compute_headroom_metrics(
        candidate_mean=float(u_exec.mean()),
        reference_mean=float(u_sham.mean()),
        oracle_mean=float(u_oracle.mean()),
        n_tasks=n,
        reference_policy_id="sham",
        candidate_policy_id="candidate_executed",
    )

    return {
        "bucket": bucket_name,
        "n_tasks": n,
        "candidate_raw_utility": float(u_raw.mean()),
        "candidate_executed_utility": float(u_exec.mean()),
        "baseline_utility": float(u_base.mean()),
        "primary_sham_utility": float(u_sham.mean()),
        "frequency_utility": float(u_freq.mean()),
        "oracle_utility": float(u_oracle.mean()),
        "candidate_minus_sham": ci_cand_sham,
        "candidate_minus_baseline": ci_cand_base,
        "headroom_recovered": headroom,
        "classification_accuracy": accuracy,
        "utility_regret": ci_regret,
        "task_ids": bucket_task_ids,
    }


# ---------------------------------------------------------------------------
# Gate A5
# ---------------------------------------------------------------------------


def _gate_a5(
    high_margin_task_ids: list[str],
    candidate_exec_map: dict[str, dict[str, Any]],
    sham_map: dict[str, dict[str, Any]],
    oracle_map: dict[str, dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate Gate A5 on the MEANINGFUL + HIGH_VALUE task subset."""
    if not high_margin_task_ids:
        return {
            "status": "FAIL",
            "recovery_fraction": 0.0,
            "lcb": 0.0,
            "threshold": A5_MIN_RECOVERY_FRACTION,
            "n_tasks": 0,
            "reason": "no_high_margin_tasks",
        }

    u_exec = np.array(
        [candidate_exec_map.get(tid, {}).get("utility") or 0.0 for tid in high_margin_task_ids],
        dtype=float,
    )
    u_sham = np.array(
        [sham_map.get(tid, {}).get("utility") or 0.0 for tid in high_margin_task_ids],
        dtype=float,
    )
    u_oracle = np.array(
        [oracle_map.get(tid, {}).get("utility") or 0.0 for tid in high_margin_task_ids],
        dtype=float,
    )

    ci = _bootstrap_recovery_fraction(u_exec, u_sham, u_oracle, n_bootstrap, seed)
    recovery_fraction = ci["mean"]
    lcb = ci["ci_low"]

    passed = (recovery_fraction >= A5_MIN_RECOVERY_FRACTION) and (lcb >= 0.0)

    return {
        "status": "PASS" if passed else "FAIL",
        "recovery_fraction": float(recovery_fraction),
        "lcb": float(lcb),
        "ucb": float(ci["ci_high"]),
        "std_error": float(ci["std_error"]),
        "threshold": A5_MIN_RECOVERY_FRACTION,
        "n_tasks": len(high_margin_task_ids),
        "n_bootstrap": n_bootstrap,
        "candidate_mean": float(u_exec.mean()),
        "sham_mean": float(u_sham.mean()),
        "oracle_mean": float(u_oracle.mean()),
        "reason": (
            "recovery_fraction >= threshold and lower confidence bound >= 0"
            if passed else
            "recovery_fraction below threshold or lower confidence bound negative"
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_high_margin_recovery(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Re-evaluate high-margin tasks and assess Gate A5.

    Args:
        run: The loaded run with counterfactual outcomes and policies.
        task_ids: Optional subset of task IDs.  Defaults to the test split.
        seed: Random seed for bootstrap and stochastic policies.

    Returns:
        A JSON-serializable ``high_margin_recovery`` dict with
        margin_buckets, gate_a5, and interpretation.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    n_bootstrap = DEFAULT_N_BOOTSTRAP

    # Build policy decision functions.
    oracle_fn = make_oracle_fn(run)
    baseline_fn = make_baseline_fn(run)
    candidate_raw_fn = make_raw_candidate_fn(run)
    candidate_exec_fn = make_executed_candidate_fn(run)
    sham_fn = make_executed_sham_fn(run)
    frequency_fn = make_frequency_fn(run, seed=seed)

    # Per-task utilities for each policy.
    oracle_map = _per_task_utilities(run, task_ids, oracle_fn)
    baseline_map = _per_task_utilities(run, task_ids, baseline_fn)
    candidate_raw_map = _per_task_utilities(run, task_ids, candidate_raw_fn)
    candidate_exec_map = _per_task_utilities(run, task_ids, candidate_exec_fn)
    sham_map = _per_task_utilities(run, task_ids, sham_fn)
    frequency_map = _per_task_utilities(run, task_ids, frequency_fn)

    # Assign margin buckets and collect per-task records.
    bucket_tasks: dict[str, list[str]] = {name: [] for name in MARGIN_BUCKET_DEFINITIONS}
    per_task: list[dict[str, Any]] = []
    for tid in task_ids:
        if tid not in oracle_map:
            continue
        margin, best_action, best_util, second_util = _task_margin(run, tid)
        bucket = _assign_margin_bucket(margin)
        bucket_tasks[bucket].append(tid)
        per_task.append({
            "task_id": tid,
            "family": run.tasks[tid].family if tid in run.tasks else "",
            "archetype": run.tasks[tid].archetype if tid in run.tasks else "",
            "margin": margin,
            "margin_bucket": bucket,
            "best_action": best_action,
            "best_utility": best_util,
            "second_best_utility": second_util,
            "oracle_utility": oracle_map.get(tid, {}).get("utility"),
            "candidate_executed_utility": candidate_exec_map.get(tid, {}).get("utility"),
            "candidate_raw_utility": candidate_raw_map.get(tid, {}).get("utility"),
            "sham_utility": sham_map.get(tid, {}).get("utility"),
            "baseline_utility": baseline_map.get(tid, {}).get("utility"),
            "frequency_utility": frequency_map.get(tid, {}).get("utility"),
        })

    # Analyze each bucket.
    margin_buckets: dict[str, Any] = {}
    for name in MARGIN_BUCKET_DEFINITIONS:
        margin_buckets[name] = _analyze_bucket(
            run,
            name,
            bucket_tasks[name],
            candidate_raw_map,
            candidate_exec_map,
            baseline_map,
            sham_map,
            frequency_map,
            oracle_map,
            n_bootstrap,
            seed,
        )

    # Gate A5 on MEANINGFUL + HIGH_VALUE tasks.
    high_margin_ids: list[str] = []
    for bucket_name in A5_MARGIN_BUCKETS:
        high_margin_ids.extend(bucket_tasks.get(bucket_name, []))

    gate_a5 = _gate_a5(
        high_margin_ids,
        candidate_exec_map,
        sham_map,
        oracle_map,
        n_bootstrap,
        seed,
    )

    # Interpretation.
    n_high = len(high_margin_ids)
    if n_high == 0:
        interpretation = (
            "No MEANINGFUL or HIGH_VALUE tasks were found; Gate A5 cannot be "
            "evaluated and is reported as FAIL."
        )
    elif gate_a5["status"] == "PASS":
        interpretation = (
            f"Gate A5 PASS: the candidate executed policy recovers "
            f"{gate_a5['recovery_fraction']:.1%} of oracle-sham headroom on "
            f"{n_high} MEANINGFUL+HIGH_VALUE tasks "
            f"(threshold {A5_MIN_RECOVERY_FRACTION:.0%}, LCB {gate_a5['lcb']:.3f})."
        )
    else:
        interpretation = (
            f"Gate A5 FAIL: the candidate executed policy recovers "
            f"{gate_a5['recovery_fraction']:.1%} of oracle-sham headroom on "
            f"{n_high} MEANINGFUL+HIGH_VALUE tasks "
            f"(threshold {A5_MIN_RECOVERY_FRACTION:.0%}, LCB {gate_a5['lcb']:.3f}). "
            f"Either the recovery fraction is below threshold or the lower "
            f"confidence bound is negative."
        )

    return {
        "margin_buckets": margin_buckets,
        "gate_a5": gate_a5,
        "interpretation": interpretation,
        "per_task": per_task,
        "margin_bucket_definitions": {
            name: {"min": lo, "max": hi}
            for name, (lo, hi) in MARGIN_BUCKET_DEFINITIONS.items()
        },
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }
