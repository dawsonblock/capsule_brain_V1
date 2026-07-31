"""Bootstrap power and sample-size analysis for v0.4.2 diagnostics.

Estimates how many independent tasks would be required for the lower bound
of a 95% confidence interval on the paired utility delta to exceed a small
threshold ``epsilon`` (default 0.01), under several assumed true effect sizes.

The analysis uses the observed paired-delta distribution (candidate - baseline
and candidate - sham) to characterise per-task variance, then simulates
drawing ``n`` tasks with a specified true mean effect and that observed
variance to find the ``n`` at which LCB95 > epsilon in 80% and 90% of
simulations.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from .data import LoadedRun
from .bootstrap import task_paired_bootstrap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lcb95(deltas: np.ndarray) -> float:
    """Lower bound of a 95% confidence interval on the mean of ``deltas``.

    Uses a fast normal approximation instead of full bootstrap for speed
    inside the power simulation loop. The production CI construction uses
    the full task_paired_bootstrap.
    """
    if len(deltas) == 0:
        return 0.0
    n = len(deltas)
    mean = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
    if std == 0.0:
        return mean  # No variance — LCB equals mean.
    # t-distribution 95% lower bound (approximate with z for n > 30).
    from scipy import stats as _stats
    if n > 30:
        z = 1.645  # one-sided 95%
    else:
        z = float(_stats.t.ppf(0.95, df=n - 1))
    return mean - z * std / np.sqrt(n)


def _simulate_power(
    true_effect: float,
    observed_std: float,
    n: int,
    n_sims: int,
    epsilon: float,
    rng: np.random.Generator,
) -> float:
    """Simulate the probability that LCB95 > epsilon for a given ``n``.

    Args:
        true_effect: assumed true mean paired delta.
        observed_std: observed per-task standard deviation of the delta.
        n: number of independent tasks to simulate.
        n_sims: number of simulation replicates.
        epsilon: threshold for the lower confidence bound.
        rng: numpy random generator.

    Returns:
        Fraction of simulations in which LCB95 > epsilon.
    """
    if n <= 0:
        return 0.0
    successes = 0
    for _ in range(n_sims):
        sample = rng.normal(loc=true_effect, scale=observed_std, size=n)
        if _lcb95(sample) > epsilon:
            successes += 1
    return successes / n_sims


def _find_required_n(
    true_effect: float,
    observed_std: float,
    target_power: float,
    n_sims: int,
    epsilon: float,
    rng: np.random.Generator,
    max_n: int = 10000,
) -> int:
    """Find the smallest ``n`` achieving at least ``target_power``.

    Uses a coarse-to-fine search: first doubles ``n`` until the target power
    is reached (or ``max_n`` is hit), then binary-searches between the last
    failing and first passing ``n``.

    Args:
        true_effect: assumed true mean paired delta.
        observed_std: observed per-task standard deviation of the delta.
        target_power: desired power (e.g. 0.80 or 0.90).
        n_sims: number of simulation replicates per ``n``.
        epsilon: threshold for the lower confidence bound.
        rng: numpy random generator.
        max_n: upper bound on the search.

    Returns:
        Smallest ``n`` achieving the target power, or ``max_n`` if never
        reached.
    """
    if observed_std <= 0:
        # No variance: a single task suffices if the effect exceeds epsilon.
        return 1 if true_effect > epsilon else max_n

    # Coarse doubling search.
    lo = 1
    hi = 1
    while hi < max_n:
        power = _simulate_power(true_effect, observed_std, hi, n_sims, epsilon, rng)
        if power >= target_power:
            break
        lo = hi + 1
        hi *= 2
    if hi >= max_n:
        # Check max_n itself.
        power = _simulate_power(true_effect, observed_std, max_n, n_sims, epsilon, rng)
        if power < target_power:
            return max_n

    # Binary search between lo-1 (last failing) and hi (first passing).
    best = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        power = _simulate_power(true_effect, observed_std, mid, n_sims, epsilon, rng)
        if power >= target_power:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return int(best)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_power_analysis(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute bootstrap power and sample-size analysis.

    Uses the observed paired-delta distributions (candidate - baseline and
    candidate - sham) to estimate per-task variance, then simulates how many
    independent tasks would be required for LCB95 > epsilon = 0.01 under
    several assumed true effect sizes.

    Args:
        run: loaded run data.
        task_ids: tasks to use for the observed delta distribution (default:
            test split).
        n_bootstrap: number of bootstrap replicates for the observed CI.
        seed: random seed for reproducibility.

    Returns:
        dict suitable for serialization as ``power_analysis.json``.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Build the observed paired-delta distributions from measured utilities.
    # ------------------------------------------------------------------
    # We reuse the utility matrix machinery from the run to get per-task
    # utilities for candidate, baseline, and sham, then form paired deltas.
    task_ids_used, actions, U = run.get_utility_matrix(task_ids)
    action_idx = {a: i for i, a in enumerate(actions)}

    # Candidate per-task utility: apply the stored candidate policy.
    cand_policy = run.candidate_policy
    cand_W = np.array(cand_policy.get("weights", []))
    cand_b = np.array(cand_policy.get("bias", []))
    cand_actions = cand_policy.get("actions", actions)

    # Sham per-task utility: apply the stored sham policy.
    sham_policy = run.sham_policy
    sham_W = np.array(sham_policy.get("weights", []))
    sham_b = np.array(sham_policy.get("bias", []))
    sham_actions = sham_policy.get("actions", actions)

    # Build a feature lookup for the selected tasks.
    feature_lookup: dict[str, np.ndarray] = {}
    for ex in run.test_examples:
        feature_lookup[ex["task_id"]] = np.array(ex.get("features", []))

    def _policy_utilities(W: np.ndarray, b: np.ndarray, pol_actions: list[str]) -> np.ndarray:
        """Per-task utility for a trained linear policy on the selected tasks."""
        pidx = {a: i for i, a in enumerate(pol_actions)}
        utils: list[float] = []
        for tid in task_ids_used:
            task = run.tasks.get(tid)
            if task is None:
                utils.append(0.0)
                continue
            feats = feature_lookup.get(tid)
            eligible = [a for a in task.allowed_actions if a in pidx]
            if feats is None or not eligible or W.size == 0:
                u = run.get_utility_for_action(tid, eligible[0]) if eligible else 0.0
                utils.append(u if u is not None else 0.0)
                continue
            logits = feats @ W.T + b
            elig_logits = np.array([logits[pidx[a]] for a in eligible])
            chosen = eligible[int(np.argmax(elig_logits))]
            u = run.get_utility_for_action(tid, chosen)
            utils.append(u if u is not None else 0.0)
        return np.array(utils)

    u_candidate = _policy_utilities(cand_W, cand_b, cand_actions)
    u_sham = _policy_utilities(sham_W, sham_b, sham_actions)

    # Baseline per-task utility: ANSWER_DIRECT (or first eligible action).
    baseline_utils: list[float] = []
    for tid in task_ids_used:
        task = run.tasks.get(tid)
        if task is None:
            baseline_utils.append(0.0)
            continue
        chosen = "ANSWER_DIRECT" if "ANSWER_DIRECT" in task.allowed_actions else (
            task.allowed_actions[0] if task.allowed_actions else None
        )
        if chosen is None:
            baseline_utils.append(0.0)
            continue
        u = run.get_utility_for_action(tid, chosen)
        baseline_utils.append(u if u is not None else 0.0)
    u_baseline = np.array(baseline_utils)

    # Paired deltas.
    delta_candidate_baseline = u_candidate - u_baseline
    delta_candidate_sham = u_candidate - u_sham

    # Observed statistics.
    obs_cb = task_paired_bootstrap(delta_candidate_baseline, n_bootstrap=n_bootstrap, seed=seed)
    obs_cs = task_paired_bootstrap(delta_candidate_sham, n_bootstrap=n_bootstrap, seed=seed)

    std_cb = float(delta_candidate_baseline.std(ddof=1)) if len(delta_candidate_baseline) > 1 else 0.0
    std_cs = float(delta_candidate_sham.std(ddof=1)) if len(delta_candidate_sham) > 1 else 0.0

    # ------------------------------------------------------------------
    # Power simulation across assumed true effect sizes.
    # ------------------------------------------------------------------
    epsilon = 0.01
    effect_sizes = [0.01, 0.05, 0.10, 0.25, 0.415]
    target_powers = [0.80, 0.90]
    n_sims = 200  # simulation replicates per (effect, n, power) cell

    results: dict[str, Any] = {
        "epsilon": epsilon,
        "n_observed_tasks": len(task_ids_used),
        "observed_delta_candidate_baseline": {
            "mean": obs_cb["mean"],
            "ci_low": obs_cb["ci_low"],
            "ci_high": obs_cb["ci_high"],
            "std": std_cb,
        },
        "observed_delta_candidate_sham": {
            "mean": obs_cs["mean"],
            "ci_low": obs_cs["ci_low"],
            "ci_high": obs_cs["ci_high"],
            "std": std_cs,
        },
        "effect_sizes": effect_sizes,
        "target_powers": target_powers,
        "n_sims_per_cell": n_sims,
        "candidate_vs_baseline": [],
        "candidate_vs_sham": [],
    }

    for effect in effect_sizes:
        row_cb: dict[str, Any] = {
            "assumed_true_effect": effect,
            "observed_std": std_cb,
            "required_n": {},
        }
        for tp in target_powers:
            n_req = _find_required_n(
                true_effect=effect,
                observed_std=std_cb,
                target_power=tp,
                n_sims=n_sims,
                epsilon=epsilon,
                rng=rng,
            )
            row_cb["required_n"][f"power_{int(tp * 100)}"] = n_req
        results["candidate_vs_baseline"].append(row_cb)

    for effect in effect_sizes:
        row_cs: dict[str, Any] = {
            "assumed_true_effect": effect,
            "observed_std": std_cs,
            "required_n": {},
        }
        for tp in target_powers:
            n_req = _find_required_n(
                true_effect=effect,
                observed_std=std_cs,
                target_power=tp,
                n_sims=n_sims,
                epsilon=epsilon,
                rng=rng,
            )
            row_cs["required_n"][f"power_{int(tp * 100)}"] = n_req
        results["candidate_vs_sham"].append(row_cs)

    # ------------------------------------------------------------------
    # Summary interpretation.
    # ------------------------------------------------------------------
    results["interpretation"] = _interpret_power(results)
    return results


def _interpret_power(results: dict[str, Any]) -> str:
    """Human-readable summary of the power analysis."""
    obs_cb = results["observed_delta_candidate_baseline"]["mean"]
    obs_cs = results["observed_delta_candidate_sham"]["mean"]
    n_obs = results["n_observed_tasks"]

    # Required n for the observed effect size (closest entry).
    def _closest(rows: list[dict[str, Any]], observed: float) -> dict[str, Any] | None:
        if not rows:
            return None
        return min(rows, key=lambda r: abs(r["assumed_true_effect"] - observed))

    cb_row = _closest(results["candidate_vs_baseline"], obs_cb)
    cs_row = _closest(results["candidate_vs_sham"], obs_cs)

    parts: list[str] = []
    parts.append(
        f"Observed candidate-baseline delta = {obs_cb:.4f} over {n_obs} tasks; "
        f"candidate-sham delta = {obs_cs:.4f}."
    )
    if cb_row is not None:
        n80 = cb_row["required_n"].get("power_80", "?")
        n90 = cb_row["required_n"].get("power_90", "?")
        parts.append(
            f"For a true effect near the observed candidate-baseline delta "
            f"({cb_row['assumed_true_effect']:.3f}), ~{n80} tasks are needed "
            f"for 80% power and ~{n90} for 90% power (LCB95 > {results['epsilon']})."
        )
    if cs_row is not None:
        n80 = cs_row["required_n"].get("power_80", "?")
        n90 = cs_row["required_n"].get("power_90", "?")
        parts.append(
            f"For a true effect near the observed candidate-sham delta "
            f"({cs_row['assumed_true_effect']:.3f}), ~{n80} tasks are needed "
            f"for 80% power and ~{n90} for 90% power."
        )
    return " ".join(parts)
