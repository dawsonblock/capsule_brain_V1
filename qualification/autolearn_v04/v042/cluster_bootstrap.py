"""Cluster-aware confidence intervals for paired utility deltas.

Compares four bootstrap schemes for the same set of paired deltas:

  1. task-level paired bootstrap          — resample individual tasks
  2. archetype-cluster paired bootstrap   — resample whole archetypes
  3. generator-family cluster bootstrap   — resample whole families
  4. family-stratified bootstrap          — resample within each family

For each scheme we compute confidence intervals for four contrasts:

  - candidate - baseline
  - candidate - sham
  - oracle    - sham
  - oracle    - baseline

The *primary* interval is the archetype-cluster bootstrap.  If the
primary interval's conclusion (sign / exclusion of zero) differs
materially from the other methods, ``cluster_sensitive`` is set to
``True`` so that downstream consumers know the inference is fragile to
clustering assumptions.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from .data import LoadedRun
from .bootstrap import (
    task_paired_bootstrap,
    cluster_paired_bootstrap,
    stratified_paired_bootstrap,
)
from .policy_controls import (
    make_candidate_selector,
    make_baseline_selector,
    make_sham_global_selector,
    evaluate_policy_on_tasks,
)


# Name of the bootstrap method designated as the primary interval.
PRIMARY_METHOD: str = "archetype_cluster"


def _bootstrap_all_methods(
    deltas: np.ndarray,
    archetype_ids: np.ndarray,
    family_ids: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Run all four bootstrap variants on a single delta vector."""
    return {
        "task": task_paired_bootstrap(deltas, n_bootstrap=n_bootstrap, seed=seed),
        "archetype_cluster": cluster_paired_bootstrap(
            deltas, archetype_ids, n_bootstrap=n_bootstrap, seed=seed,
        ),
        "family_cluster": cluster_paired_bootstrap(
            deltas, family_ids, n_bootstrap=n_bootstrap, seed=seed,
        ),
        "family_stratified": stratified_paired_bootstrap(
            deltas, family_ids, n_bootstrap=n_bootstrap, seed=seed,
        ),
    }


def _conclusion(ci: dict[str, float]) -> str:
    """Summarise whether a CI excludes zero and on which side.

    Returns one of ``"positive"``, ``"negative"``, or ``"inconclusive"``.
    """
    if ci["ci_low"] > 0:
        return "positive"
    if ci["ci_high"] < 0:
        return "negative"
    return "inconclusive"


def _is_materially_different(
    primary: dict[str, float],
    others: list[dict[str, float]],
) -> bool:
    """Decide whether non-primary intervals disagree materially.

    A material disagreement is any of:
      - a different sign conclusion (positive / negative / inconclusive)
      - a CI boundary crossing zero when the primary does not (or vice
        versa) — captured by the conclusion check above.
    """
    primary_conclusion = _conclusion(primary)
    for other in others:
        if _conclusion(other) != primary_conclusion:
            return True
    return False


def compute_cluster_cis(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute cluster-aware confidence intervals for paired deltas.

    Evaluates the candidate, baseline, sham, and oracle policies on the
    requested tasks (defaulting to the test split) and builds
    per-task utility vectors.  Four bootstrap schemes are then applied
    to each of the four contrasts and the results are compared.

    Returns a dict suitable for writing to
    ``cluster_bootstrap_results.json``.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # ------------------------------------------------------------------
    # Evaluate policies.
    # ------------------------------------------------------------------
    candidate_sel = make_candidate_selector(run)
    baseline_sel = make_baseline_selector(run)
    sham_sel = make_sham_global_selector(run, seed=12345)

    candidate_res = evaluate_policy_on_tasks(run, task_ids, candidate_sel, "candidate")
    baseline_res = evaluate_policy_on_tasks(run, task_ids, baseline_sel, "baseline")
    sham_res = evaluate_policy_on_tasks(run, task_ids, sham_sel, "sham")

    # Oracle utility is the best measured utility per task, aligned to
    # the effective task ordering produced by the policy evaluator.
    per_task = candidate_res["per_task"]
    u_candidate = np.array(candidate_res["utilities"], dtype=float)
    u_baseline = np.array(baseline_res["utilities"], dtype=float)
    u_sham = np.array(sham_res["utilities"], dtype=float)
    u_oracle = np.array(
        [run.get_oracle_utility(pt["task_id"]) for pt in per_task], dtype=float,
    )

    archetype_ids = np.array([pt["archetype"] for pt in per_task])
    family_ids = np.array([pt["family"] for pt in per_task])

    # ------------------------------------------------------------------
    # Four contrasts.
    # ------------------------------------------------------------------
    contrasts: dict[str, np.ndarray] = {
        "candidate_baseline": u_candidate - u_baseline,
        "candidate_sham": u_candidate - u_sham,
        "oracle_sham": u_oracle - u_sham,
        "oracle_baseline": u_oracle - u_baseline,
    }

    # ------------------------------------------------------------------
    # Bootstrap each contrast with all four methods.
    # ------------------------------------------------------------------
    results: dict[str, Any] = {
        "n_tasks": int(u_candidate.size),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "primary_method": PRIMARY_METHOD,
        "contrasts": {},
    }

    # Track whether any contrast is cluster-sensitive.
    any_sensitive = False

    for contrast_name, deltas in contrasts.items():
        method_cis = _bootstrap_all_methods(
            deltas, archetype_ids, family_ids, n_bootstrap, seed,
        )
        primary_ci = method_cis[PRIMARY_METHOD]
        other_cis = [v for k, v in method_cis.items() if k != PRIMARY_METHOD]
        sensitive = _is_materially_different(primary_ci, other_cis)
        if sensitive:
            any_sensitive = True

        results["contrasts"][contrast_name] = {
            "mean_delta": float(deltas.mean()) if deltas.size else 0.0,
            "intervals": method_cis,
            "primary_interval": primary_ci,
            "conclusions": {
                name: _conclusion(ci) for name, ci in method_cis.items()
            },
            "cluster_sensitive": sensitive,
        }

    results["cluster_sensitive"] = any_sensitive
    results["note"] = (
        "Primary interval is the archetype-cluster paired bootstrap. "
        "cluster_sensitive is True when any contrast's primary "
        "conclusion (sign / zero-exclusion) disagrees with at least "
        "one alternative bootstrap method."
    )

    return results
