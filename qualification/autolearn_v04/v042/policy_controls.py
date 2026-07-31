"""Policy control ladder for v0.4.2 diagnostics.

Evaluates a ladder of control policies on the same held-out test tasks
using measured counterfactual utilities. No new model executions.

Policies:
  P_random          — uniform random action selection
  P_majority        — always pick most frequent best action in D_experience
  P_frequency       — sample from marginal best-action distribution
  P_family_majority — most frequent best action within task family
  P_archetype_majority — most frequent best action within archetype
  P_sham_global     — learner trained on globally shuffled labels
  P_sham_family     — learner trained on within-family shuffled labels
  P_sham_margin     — learner trained on margin-bin-preserving shuffled labels
  P_sham_feature    — learner trained on shuffled features
  P_baseline        — frozen deterministic baseline
  P_candidate       — learned state-dependent policy
  P_oracle          — best measured action per task
"""
from __future__ import annotations

import json
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Any

from .data import LoadedRun
from .bootstrap import task_paired_bootstrap


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------


def evaluate_policy_on_tasks(
    run: LoadedRun,
    task_ids: list[str],
    action_selector: callable,
    policy_name: str,
) -> dict[str, Any]:
    """Evaluate a policy on a set of tasks using measured utilities.

    Args:
        run: loaded run data.
        task_ids: tasks to evaluate on.
        action_selector: function(task_id, task_info) -> action_name.
        policy_name: name of the policy.

    Returns:
        dict with mean_utility, per-task utilities, action distribution, etc.
    """
    utilities = []
    selected_actions = []
    best_actions = []
    successes = 0
    action_dist = Counter()
    per_task = []

    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        action = action_selector(tid, task)
        if action is None:
            continue
        u = run.get_utility_for_action(tid, action)
        if u is None:
            u = 0.0
        best = run.get_best_action(tid)
        o = run.outcomes_by_task_action.get((tid, action))
        verified = o.verification if o else "unknown"

        utilities.append(u)
        selected_actions.append(action)
        best_actions.append(best or "")
        action_dist[action] += 1
        if verified == "success":
            successes += 1

        per_task.append({
            "task_id": tid,
            "selected_action": action,
            "best_action": best,
            "utility": u,
            "oracle_utility": run.get_oracle_utility(tid),
            "verified": verified,
            "family": task.family,
            "archetype": task.archetype,
        })

    utilities = np.array(utilities)
    n = len(utilities)

    # Confusion matrix.
    all_actions = run.all_actions
    action_idx = {a: i for i, a in enumerate(all_actions)}
    confusion = np.zeros((len(all_actions), len(all_actions)), dtype=int)
    for sa, ba in zip(selected_actions, best_actions):
        if ba in action_idx and sa in action_idx:
            confusion[action_idx[ba], action_idx[sa]] += 1

    # Utility-weighted confusion (regret).
    regret_matrix = np.zeros((len(all_actions), len(all_actions)))
    for pt in per_task:
        sa = pt["selected_action"]
        ba = pt["best_action"]
        if ba in action_idx and sa in action_idx:
            regret = pt["oracle_utility"] - pt["utility"]
            regret_matrix[action_idx[ba], action_idx[sa]] += max(0, regret)

    result = {
        "policy_name": policy_name,
        "n_tasks": n,
        "mean_utility": float(utilities.mean()) if n > 0 else 0.0,
        "median_utility": float(np.median(utilities)) if n > 0 else 0.0,
        "utility_variance": float(utilities.var()) if n > 0 else 0.0,
        "verified_success_count": successes,
        "verified_success_rate": successes / n if n > 0 else 0.0,
        "action_distribution": {a: action_dist.get(a, 0) for a in all_actions},
        "action_distribution_pct": {a: action_dist.get(a, 0) / n if n > 0 else 0.0 for a in all_actions},
        "confusion_matrix": {
            "actions": all_actions,
            "matrix": confusion.tolist(),
        },
        "utility_weighted_confusion": {
            "actions": all_actions,
            "matrix": regret_matrix.tolist(),
        },
        "per_task": per_task,
        "utilities": utilities.tolist(),
    }

    # Regret statistics.
    regrets = np.array([
        pt["oracle_utility"] - pt["utility"]
        for pt in per_task
    ])
    result["regret"] = {
        "mean": float(regrets.mean()) if n > 0 else 0.0,
        "median": float(np.median(regrets)) if n > 0 else 0.0,
        "p95": float(np.percentile(regrets, 95)) if n > 0 else 0.0,
        "catastrophic_count": int(np.sum(regrets > 5.0)),
        "by_true_action": {},
        "by_selected_action": {},
        "by_family": {},
    }
    for pt, reg in zip(per_task, regrets):
        ba = pt["best_action"]
        sa = pt["selected_action"]
        fam = pt["family"]
        result["regret"]["by_true_action"].setdefault(ba, []).append(reg)
        result["regret"]["by_selected_action"].setdefault(sa, []).append(reg)
        result["regret"]["by_family"].setdefault(fam, []).append(reg)
    for key in ("by_true_action", "by_selected_action", "by_family"):
        for k, vals in result["regret"][key].items():
            result["regret"][key][k] = {
                "mean": float(np.mean(vals)),
                "count": len(vals),
            }

    return result


def compute_wins_ties_losses(
    policy_utils: np.ndarray,
    reference_utils: np.ndarray,
) -> dict[str, int]:
    """Compute wins/ties/losses of policy vs reference."""
    diffs = policy_utils - reference_utils
    return {
        "wins": int(np.sum(diffs > 1e-9)),
        "ties": int(np.sum(np.abs(diffs) <= 1e-9)),
        "losses": int(np.sum(diffs < -1e-9)),
    }


# ---------------------------------------------------------------------------
# Policy selectors
# ---------------------------------------------------------------------------


def make_random_selector(run: LoadedRun, seed: int = 42) -> callable:
    """Uniform random action among eligible actions."""
    rng = np.random.default_rng(seed)
    def selector(tid, task):
        eligible = [a for a in task.allowed_actions if a in run.all_actions]
        if not eligible:
            return None
        return eligible[rng.integers(0, len(eligible))]
    return selector


def make_majority_selector(run: LoadedRun) -> callable:
    """Always pick the most frequent best action in D_experience."""
    best_actions = [ex["best_action"] for ex in run.train_examples]
    counts = Counter(best_actions)
    majority_action = counts.most_common(1)[0][0] if counts else run.all_actions[0]
    def selector(tid, task):
        return majority_action if majority_action in task.allowed_actions else task.allowed_actions[0]
    return selector


def make_frequency_selector(run: LoadedRun, seed: int = 42) -> callable:
    """Sample from marginal best-action distribution in D_experience."""
    best_actions = [ex["best_action"] for ex in run.train_examples]
    counts = Counter(best_actions)
    total = sum(counts.values())
    probs = {a: counts.get(a, 0) / total for a in run.all_actions}
    rng = np.random.default_rng(seed)
    def selector(tid, task):
        eligible = [a for a in task.allowed_actions if a in run.all_actions]
        if not eligible:
            return None
        p = np.array([probs.get(a, 0) for a in eligible])
        p = p / p.sum()
        return eligible[rng.choice(len(eligible), p=p)]
    return selector


def make_frequency_exact_utility(run: LoadedRun, task_ids: list[str]) -> np.ndarray:
    """Exact expected utility for frequency policy (no Monte Carlo noise).

    U_frequency(task_i) = sum_a p(a) * U_i(a)
    """
    best_actions = [ex["best_action"] for ex in run.train_examples]
    counts = Counter(best_actions)
    total = sum(counts.values())
    probs = {a: counts.get(a, 0) / total for a in run.all_actions}

    utilities = []
    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            utilities.append(0.0)
            continue
        u = 0.0
        for a in task.allowed_actions:
            if a in probs:
                ua = run.get_utility_for_action(tid, a)
                if ua is not None:
                    u += probs[a] * ua
        utilities.append(u)
    return np.array(utilities)


def make_family_majority_selector(run: LoadedRun) -> callable:
    """Most frequent best action within task family (from D_experience)."""
    family_action_counts: dict[str, Counter] = {}
    for ex in run.train_examples:
        fam = ex.get("task_family", "unknown")
        family_action_counts.setdefault(fam, Counter())[ex["best_action"]] += 1

    def selector(tid, task):
        fam = task.family
        counts = family_action_counts.get(fam)
        if counts:
            best = counts.most_common(1)[0][0]
            if best in task.allowed_actions:
                return best
        # Fall back to global majority.
        global_counts = Counter(ex["best_action"] for ex in run.train_examples)
        if global_counts:
            best = global_counts.most_common(1)[0][0]
            if best in task.allowed_actions:
                return best
        return task.allowed_actions[0] if task.allowed_actions else None
    return selector


def make_archetype_majority_selector(run: LoadedRun) -> callable:
    """Most frequent best action within archetype (from D_experience).

    Diagnostic only — does not access test labels. Uses the best-action
    distribution from D_experience tasks that share the same archetype.
    If no training tasks share the archetype, falls back to global majority.
    """
    archetype_action_counts: dict[str, Counter] = {}
    for ex in run.train_examples:
        tid = ex["task_id"]
        task = run.tasks.get(tid)
        if task:
            archetype_action_counts.setdefault(task.archetype, Counter())[ex["best_action"]] += 1

    def selector(tid, task):
        counts = archetype_action_counts.get(task.archetype)
        if counts:
            best = counts.most_common(1)[0][0]
            if best in task.allowed_actions:
                return best
        # Fall back to global majority.
        global_counts = Counter(ex["best_action"] for ex in run.train_examples)
        if global_counts:
            best = global_counts.most_common(1)[0][0]
            if best in task.allowed_actions:
                return best
        return task.allowed_actions[0] if task.allowed_actions else None
    return selector


def make_oracle_selector(run: LoadedRun) -> callable:
    """Best measured action per task."""
    def selector(tid, task):
        return run.get_best_action(tid)
    return selector


def make_baseline_selector(run: LoadedRun) -> callable:
    """Frozen deterministic baseline policy.

    The baseline always selects ANSWER_DIRECT (the default action
    when no routing policy is applied).
    """
    def selector(tid, task):
        if "ANSWER_DIRECT" in task.allowed_actions:
            return "ANSWER_DIRECT"
        return task.allowed_actions[0] if task.allowed_actions else None
    return selector


def make_candidate_selector(run: LoadedRun) -> callable:
    """Learned candidate policy — apply trained weights to features."""
    policy = run.candidate_policy
    weights = np.array(policy.get("weights", []))
    bias = np.array(policy.get("bias", []))
    actions = policy.get("actions", run.all_actions)
    feature_names = policy.get("feature_names", [])
    action_idx = {a: i for i, a in enumerate(actions)}

    def selector(tid, task):
        # Find the training example for this task to get features.
        ex = None
        for e in run.test_examples + run.ood_examples:
            if e["task_id"] == tid:
                ex = e
                break
        if ex is None:
            # Try all examples.
            for e in run.train_examples + run.validation_examples:
                if e["task_id"] == tid:
                    ex = e
                    break
        if ex is None or not weights.size:
            return task.allowed_actions[0] if task.allowed_actions else None

        features = np.array(ex.get("features", []))
        if len(features) != weights.shape[1]:
            return task.allowed_actions[0] if task.allowed_actions else None

        logits = features @ weights.T + bias
        # Only consider eligible actions.
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            return task.allowed_actions[0] if task.allowed_actions else None
        eligible_logits = np.array([logits[action_idx[a]] for a in eligible])
        return eligible[int(np.argmax(eligible_logits))]

    return selector


def make_sham_global_selector(run: LoadedRun, seed: int = 12345) -> callable:
    """Sham policy trained on globally shuffled best-action labels."""
    return _make_sham_selector(run, shuffle_mode="global", seed=seed)


def make_sham_family_selector(run: LoadedRun, seed: int = 12345) -> callable:
    """Sham policy trained on within-family shuffled labels."""
    return _make_sham_selector(run, shuffle_mode="family", seed=seed)


def make_sham_margin_selector(run: LoadedRun, seed: int = 12345) -> callable:
    """Sham policy trained on margin-bin-preserving shuffled labels."""
    return _make_sham_selector(run, shuffle_mode="margin", seed=seed)


def make_sham_feature_selector(run: LoadedRun, seed: int = 12345) -> callable:
    """Sham policy trained on shuffled feature vectors."""
    return _make_sham_selector(run, shuffle_mode="feature", seed=seed)


def _make_sham_selector(run: LoadedRun, shuffle_mode: str, seed: int) -> callable:
    """Train a sham policy with the specified shuffle mode and return selector."""
    from .sham_trainer import train_sham_policy
    sham_policy = train_sham_policy(run, shuffle_mode=shuffle_mode, seed=seed)
    weights = np.array(sham_policy.get("weights", []))
    bias = np.array(sham_policy.get("bias", []))
    actions = sham_policy.get("actions", run.all_actions)
    action_idx = {a: i for i, a in enumerate(actions)}

    def selector(tid, task):
        ex = None
        for e in run.test_examples + run.ood_examples:
            if e["task_id"] == tid:
                ex = e
                break
        if ex is None:
            for e in run.train_examples + run.validation_examples:
                if e["task_id"] == tid:
                    ex = e
                    break
        if ex is None or not weights.size:
            return task.allowed_actions[0] if task.allowed_actions else None

        features = np.array(ex.get("features", []))
        if len(features) != weights.shape[1]:
            return task.allowed_actions[0] if task.allowed_actions else None

        logits = features @ weights.T + bias
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            return task.allowed_actions[0] if task.allowed_actions else None
        eligible_logits = np.array([logits[action_idx[a]] for a in eligible])
        return eligible[int(np.argmax(eligible_logits))]

    return selector


# ---------------------------------------------------------------------------
# Full ladder evaluation
# ---------------------------------------------------------------------------


def evaluate_policy_ladder(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    n_frequency_mc: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate the full policy control ladder.

    Args:
        run: loaded run data.
        task_ids: tasks to evaluate on (default: test split).
        n_frequency_mc: Monte Carlo draws for frequency policy.
        seed: base random seed.

    Returns:
        policy_control_ladder.json content.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # Also evaluate on OOD for completeness.
    ood_ids = run.ood_task_ids

    policies = {
        "P_random": make_random_selector(run, seed=seed),
        "P_majority": make_majority_selector(run),
        "P_frequency": make_frequency_selector(run, seed=seed),
        "P_family_majority": make_family_majority_selector(run),
        "P_archetype_majority": make_archetype_majority_selector(run),
        "P_sham_global": make_sham_global_selector(run, seed=12345),
        "P_sham_family": make_sham_family_selector(run, seed=12345),
        "P_sham_margin": make_sham_margin_selector(run, seed=12345),
        "P_sham_feature": make_sham_feature_selector(run, seed=12345),
        "P_baseline": make_baseline_selector(run),
        "P_candidate": make_candidate_selector(run),
        "P_oracle": make_oracle_selector(run),
    }

    results = {}
    for name, selector in policies.items():
        results[name] = evaluate_policy_on_tasks(run, task_ids, selector, name)

    # Add exact frequency utility (no MC noise).
    freq_exact = make_frequency_exact_utility(run, task_ids)
    results["P_frequency_exact"] = {
        "policy_name": "P_frequency_exact",
        "n_tasks": len(task_ids),
        "mean_utility": float(freq_exact.mean()),
        "utilities": freq_exact.tolist(),
        "note": "Exact expected utility: sum_a p(a) * U_i(a). No Monte Carlo noise.",
    }

    # Compute wins/ties/losses vs baseline and vs sham.
    baseline_utils = np.array(results["P_baseline"]["utilities"])
    sham_utils = np.array(results["P_sham_global"]["utilities"])
    for name, res in results.items():
        if "utilities" not in res or name in ("P_oracle", "P_baseline"):
            continue
        pu = np.array(res["utilities"])
        if len(pu) == len(baseline_utils):
            res["vs_baseline"] = compute_wins_ties_losses(pu, baseline_utils)
        if len(pu) == len(sham_utils) and name != "P_sham_global":
            res["vs_sham"] = compute_wins_ties_losses(pu, sham_utils)

    # Bootstrap CIs for key comparisons.
    candidate_utils = np.array(results["P_candidate"].get("utilities", []))
    oracle_utils = np.array(results["P_oracle"].get("utilities", []))

    comparisons = {}
    if len(candidate_utils) > 0 and len(baseline_utils) > 0:
        comparisons["candidate_vs_baseline"] = task_paired_bootstrap(
            candidate_utils - baseline_utils, seed=seed)
    if len(candidate_utils) > 0 and len(sham_utils) > 0:
        comparisons["candidate_vs_sham"] = task_paired_bootstrap(
            candidate_utils - sham_utils, seed=seed)
    if len(oracle_utils) > 0 and len(sham_utils) > 0:
        comparisons["oracle_vs_sham"] = task_paired_bootstrap(
            oracle_utils - sham_utils, seed=seed)
    if len(oracle_utils) > 0 and len(baseline_utils) > 0:
        comparisons["oracle_vs_baseline"] = task_paired_bootstrap(
            oracle_utils - baseline_utils, seed=seed)

    # Frequency exact comparisons.
    if len(freq_exact) > 0 and len(candidate_utils) > 0:
        comparisons["candidate_vs_frequency"] = task_paired_bootstrap(
            candidate_utils - freq_exact, seed=seed)
    if len(freq_exact) > 0 and len(baseline_utils) > 0:
        comparisons["frequency_vs_baseline"] = task_paired_bootstrap(
            freq_exact - baseline_utils, seed=seed)

    return {
        "policies": results,
        "comparisons": comparisons,
        "n_tasks": len(task_ids),
        "task_ids": task_ids,
        "seed": seed,
    }
