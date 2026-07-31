"""Action-specific competence analysis for v0.4.2 diagnostics.

Measures per-action and per-family competence: success rates conditional
on selected, oracle, and wrong actions; action selectivity; and the
action-dependence index (ADI / NADI) with bootstrap confidence intervals.

All analysis consumes existing counterfactual outcomes — no new model
executions are required.
"""
from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict
from typing import Any

from .data import LoadedRun
from .bootstrap import task_paired_bootstrap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_feature_lookup(run: LoadedRun) -> dict[str, list[float]]:
    """Build a task_id -> features lookup from all available examples."""
    lookup: dict[str, list[float]] = {}
    for ex in (
        run.test_examples
        + run.ood_examples
        + run.train_examples
        + run.validation_examples
    ):
        lookup[ex["task_id"]] = ex.get("features", [])
    return lookup


def _get_candidate_selected_actions(
    run: LoadedRun,
    task_ids: list[str],
) -> dict[str, str | None]:
    """Determine the action selected by the candidate policy for each task.

    Applies the trained candidate weights to the task's feature vector and
    picks the highest-logit eligible action.
    """
    policy = run.candidate_policy
    weights = np.array(policy.get("weights", []))
    bias = np.array(policy.get("bias", []))
    actions = policy.get("actions", run.all_actions)
    action_idx = {a: i for i, a in enumerate(actions)}
    feat_lookup = _build_feature_lookup(run)

    selected: dict[str, str | None] = {}
    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            selected[tid] = None
            continue
        feats = feat_lookup.get(tid)
        if feats is None or not weights.size:
            selected[tid] = task.allowed_actions[0] if task.allowed_actions else None
            continue
        feats_arr = np.array(feats)
        if len(feats_arr) != weights.shape[1]:
            selected[tid] = task.allowed_actions[0] if task.allowed_actions else None
            continue
        logits = feats_arr @ weights.T + bias
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            selected[tid] = task.allowed_actions[0] if task.allowed_actions else None
            continue
        eligible_logits = np.array([logits[action_idx[a]] for a in eligible])
        selected[tid] = eligible[int(np.argmax(eligible_logits))]
    return selected


def _is_success(run: LoadedRun, task_id: str, action_id: str) -> bool:
    """Check whether a (task, action) outcome was verified as a success."""
    o = run.outcomes_by_task_action.get((task_id, action_id))
    if o is None:
        return False
    return o.verification == "success"


def _get_failure_reason(run: LoadedRun, task_id: str, action_id: str) -> str:
    """Extract a failure reason from an outcome's metadata."""
    o = run.outcomes_by_task_action.get((task_id, action_id))
    if o is None:
        return "no_outcome"
    if o.verification == "success":
        return "success"
    # Try execution_metadata for a failure reason.
    meta = o.execution_metadata or {}
    reason = (
        meta.get("failure_reason")
        or meta.get("reason")
        or meta.get("error")
        or meta.get("failure_category")
    )
    if reason:
        return str(reason)
    # Try reward_components.
    rc = o.reward_components or {}
    reason = rc.get("failure_reason") or rc.get("reason")
    if reason:
        return str(reason)
    return o.verification


# ---------------------------------------------------------------------------
# Core competence analysis
# ---------------------------------------------------------------------------

def _compute_action_family_competence(
    run: LoadedRun,
    task_ids: list[str],
    actions: list[str],
    families: list[str],
    selected_actions: dict[str, str | None],
) -> dict[str, Any]:
    """Compute per-action, per-family competence metrics.

    For each (action, family) pair:
      - P(success | a, family)          — success rate of action a on tasks in family
      - P(success | selected=a, family) — success rate when candidate selected a
      - P(success | oracle, family)     — success rate of the oracle (best) action
      - P(success | wrong, family)      — success rate of non-oracle actions
      - mean utility by action/family
      - verification failure reasons
    """
    results: dict[str, Any] = {}

    for family in families:
        family_tasks = [
            tid for tid in task_ids
            if run.tasks.get(tid) and run.tasks[tid].family == family
        ]
        if not family_tasks:
            continue

        family_results: dict[str, Any] = {}
        for action in actions:
            # P(success | a, family) and mean utility.
            successes_a = 0
            total_a = 0
            utils_a: list[float] = []
            for tid in family_tasks:
                u = run.get_utility_for_action(tid, action)
                if u is not None:
                    total_a += 1
                    utils_a.append(u)
                    if _is_success(run, tid, action):
                        successes_a += 1
            p_success_a = successes_a / total_a if total_a > 0 else 0.0
            mean_util_a = float(np.mean(utils_a)) if utils_a else 0.0

            # P(success | selected=a, family).
            sel_successes = 0
            sel_total = 0
            for tid in family_tasks:
                if selected_actions.get(tid) == action:
                    sel_total += 1
                    if _is_success(run, tid, action):
                        sel_successes += 1
            p_success_selected = sel_successes / sel_total if sel_total > 0 else 0.0

            # Failure reasons for this action in this family.
            failure_reasons: dict[str, int] = defaultdict(int)
            for tid in family_tasks:
                o = run.outcomes_by_task_action.get((tid, action))
                if o is not None and o.verification != "success":
                    failure_reasons[_get_failure_reason(run, tid, action)] += 1

            family_results[action] = {
                "p_success": p_success_a,
                "n_attempts": total_a,
                "p_success_selected": p_success_selected,
                "n_selected": sel_total,
                "mean_utility": mean_util_a,
                "failure_reasons": dict(failure_reasons),
            }

        # P(success | oracle, family) and P(success | wrong, family).
        oracle_successes = 0
        wrong_successes = 0
        wrong_total = 0
        oracle_total = 0
        for tid in family_tasks:
            best = run.get_best_action(tid)
            if best is None:
                continue
            oracle_total += 1
            if _is_success(run, tid, best):
                oracle_successes += 1
            for act in actions:
                if act == best:
                    continue
                u = run.get_utility_for_action(tid, act)
                if u is not None:
                    wrong_total += 1
                    if _is_success(run, tid, act):
                        wrong_successes += 1

        family_results["_family_summary"] = {
            "n_tasks": len(family_tasks),
            "p_success_oracle": oracle_successes / oracle_total if oracle_total > 0 else 0.0,
            "p_success_wrong": wrong_successes / wrong_total if wrong_total > 0 else 0.0,
            "n_oracle_evaluated": oracle_total,
            "n_wrong_evaluated": wrong_total,
        }

        results[family] = family_results

    return results


def _compute_action_selectivity(
    run: LoadedRun,
    task_ids: list[str],
    actions: list[str],
    families: list[str],
) -> dict[str, Any]:
    """Compute action selectivity per family.

    S_family = max_a P(success | a, family) - mean_{b != a*} P(success | b, family)

    where a* is the argmax action. This measures how much the best action
    stands out from the rest within a family.
    """
    selectivity: dict[str, Any] = {}

    for family in families:
        family_tasks = [
            tid for tid in task_ids
            if run.tasks.get(tid) and run.tasks[tid].family == family
        ]
        if not family_tasks:
            continue

        p_success_by_action: dict[str, float] = {}
        for action in actions:
            successes = 0
            total = 0
            for tid in family_tasks:
                u = run.get_utility_for_action(tid, action)
                if u is not None:
                    total += 1
                    if _is_success(run, tid, action):
                        successes += 1
            p_success_by_action[action] = successes / total if total > 0 else 0.0

        if not p_success_by_action:
            continue

        best_action = max(p_success_by_action, key=p_success_by_action.get)
        best_p = p_success_by_action[best_action]
        other_ps = [p for a, p in p_success_by_action.items() if a != best_action]
        mean_other = float(np.mean(other_ps)) if other_ps else 0.0

        selectivity[family] = {
            "best_action": best_action,
            "p_success_best": best_p,
            "mean_p_success_others": mean_other,
            "selectivity": best_p - mean_other,
            "p_success_by_action": p_success_by_action,
        }

    return selectivity


def _compute_action_dependence_index(
    run: LoadedRun,
    task_ids: list[str],
    actions: list[str],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute the Action-Dependence Index (ADI) and Normalized ADI (NADI).

    ADI = mean_i [max_a U_i,a - mean_a U_i,a]

    NADI = ADI / scale(U), where scale(U) = std of all utilities.

    Bootstrap intervals are computed by resampling tasks.
    """
    _, _, U = run.get_utility_matrix(task_ids)
    n_tasks, n_actions_arr = U.shape

    if n_tasks == 0 or n_actions_arr == 0:
        return {
            "adi": 0.0,
            "nadi": 0.0,
            "scale": 0.0,
            "bootstrap": {},
            "n_tasks": 0,
        }

    # Per-task ADI contributions.
    max_per_task = U.max(axis=1)
    mean_per_task = U.mean(axis=1)
    adi_per_task = max_per_task - mean_per_task
    adi = float(adi_per_task.mean())

    # Scale = std of all utilities.
    all_utils = U.flatten()
    scale = float(np.std(all_utils, ddof=1)) if len(all_utils) > 1 else 0.0
    nadi = float(adi / scale) if scale > 1e-12 else 0.0

    # Bootstrap ADI.
    rng = np.random.default_rng(seed)
    adi_samples = np.empty(n_bootstrap)
    nadi_samples = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_tasks, size=n_tasks)
        U_b = U[idx]
        max_b = U_b.max(axis=1)
        mean_b = U_b.mean(axis=1)
        adi_b = float((max_b - mean_b).mean())
        adi_samples[b] = adi_b
        scale_b = float(np.std(U_b.flatten(), ddof=1)) if U_b.size > 1 else 0.0
        nadi_samples[b] = adi_b / scale_b if scale_b > 1e-12 else 0.0

    alpha = 0.025  # 95% CI
    adi_ci = {
        "mean": adi,
        "ci_low": float(np.percentile(adi_samples, 100 * alpha)),
        "ci_high": float(np.percentile(adi_samples, 100 * (1 - alpha))),
        "std_error": float(np.std(adi_samples, ddof=1)) if n_bootstrap > 1 else 0.0,
    }
    nadi_ci = {
        "mean": nadi,
        "ci_low": float(np.percentile(nadi_samples, 100 * alpha)),
        "ci_high": float(np.percentile(nadi_samples, 100 * (1 - alpha))),
        "std_error": float(np.std(nadi_samples, ddof=1)) if n_bootstrap > 1 else 0.0,
    }

    return {
        "adi": adi,
        "nadi": nadi,
        "scale": scale,
        "n_tasks": n_tasks,
        "n_actions": n_actions_arr,
        "per_task_adi": adi_per_task.tolist(),
        "bootstrap": {
            "n_bootstrap": n_bootstrap,
            "adi_ci": adi_ci,
            "nadi_ci": nadi_ci,
        },
        "description": (
            "ADI = mean_i [max_a U_i,a - mean_a U_i,a] measures how much "
            "utility varies across actions within a task, averaged over "
            "tasks. NADI = ADI / std(U) normalizes by the overall utility "
            "scale. Higher values indicate stronger action-dependence "
            "(routing matters)."
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_competence(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze action-specific competence across task families.

    Args:
        run: loaded run data with outcomes, tasks, and candidate policy.
        task_ids: tasks to analyze (default: test split).
        n_bootstrap: number of bootstrap replicates for ADI/NADI intervals.
        seed: random seed for bootstrap.

    Returns:
        Content for ``action_competence_analysis.json`` containing:

        * per-action, per-family success rates and utilities
        * action selectivity per family
        * action-dependence index (ADI) and normalized ADI (NADI) with
          bootstrap confidence intervals
        * verification failure reason breakdown
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    actions = run.all_actions
    families = run.all_families

    # Determine candidate-selected actions.
    selected_actions = _get_candidate_selected_actions(run, task_ids)

    # Per-action, per-family competence.
    action_family = _compute_action_family_competence(
        run, task_ids, actions, families, selected_actions
    )

    # Action selectivity per family.
    selectivity = _compute_action_selectivity(run, task_ids, actions, families)

    # Action-dependence index.
    adi_results = _compute_action_dependence_index(
        run, task_ids, actions, n_bootstrap=n_bootstrap, seed=seed
    )

    # Overall action distribution from candidate selections.
    selected_dist = Counter(selected_actions.values())
    action_distribution = {a: selected_dist.get(a, 0) for a in actions}

    # Overall verification failure reasons.
    all_failure_reasons: dict[str, int] = defaultdict(int)
    for tid in task_ids:
        for o in run.outcomes_by_task.get(tid, []):
            if o.availability == "executed" and o.verification != "success":
                reason = _get_failure_reason(run, tid, o.action_id)
                all_failure_reasons[reason] += 1

    return {
        "n_tasks": len(task_ids),
        "n_actions": len(actions),
        "n_families": len(families),
        "actions": actions,
        "families": families,
        "candidate_action_distribution": action_distribution,
        "action_family_competence": action_family,
        "action_selectivity": selectivity,
        "action_dependence_index": adi_results,
        "verification_failure_reasons": dict(all_failure_reasons),
        "definitions": {
            "p_success_selected": "P(success | candidate selected action a, family f)",
            "p_success_oracle": "P(success | oracle/best action, family f)",
            "p_success_wrong": "P(success | non-oracle action, family f)",
            "selectivity": "S_family = max_a P(success|a,f) - mean_{b!=a*} P(success|b,f)",
            "adi": "ADI = mean_i [max_a U_i,a - mean_a U_i,a]",
            "nadi": "NADI = ADI / std(all utilities)",
        },
    }
