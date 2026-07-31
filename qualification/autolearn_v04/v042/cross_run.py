"""Cross-run comparison between mini and full 7B runs.

Loads both the mini and full counterfactual runs and compares them on
task composition, best-action distribution, utility-margin distribution,
model generation configuration, provider/benchmark versions, candidate
learner settings, sham construction, calibration thresholds, random
seeds, runtime completion, verification success rates, and failure
types.

Where possible, policies are re-evaluated on the *exact overlapping task
subset* (tasks that appear in both runs by ``task_id``) so that
candidate-baseline deltas are compared on identical task sets rather
than on different compositions.

The specific research question addressed is whether the apparent
increase in candidate-baseline delta (mini: +0.093, full: +0.415) is
caused by genuine candidate learning improvement or by baseline weakness
on the newly added tasks present only in the full run.

The output of :func:`compare_runs` is the content for
``mini_full_comparison.json``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import LoadedRun, load_run, load_two_runs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The canonical action set used throughout the benchmark.
CANONICAL_ACTIONS: list[str] = [
    "ANSWER_DIRECT",
    "RETRIEVE_MEMORY",
    "CALL_TOOL",
    "START_WORKFLOW",
]

#: Predeclared mini-run candidate-baseline delta for the research question.
MINI_DELTA: float = 0.093

#: Predeclared full-run candidate-baseline delta for the research question.
FULL_DELTA: float = 0.415


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float] | np.ndarray) -> float:
    """Mean of a list/array, returning 0.0 for empty input."""
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _sorted_utility_pairs(run: LoadedRun, task_id: str) -> list[tuple[str, float]]:
    """Return executed (action, utility) pairs sorted by utility descending."""
    outcomes = run.outcomes_by_task.get(task_id, [])
    pairs = [
        (o.action_id, o.utility)
        for o in outcomes
        if o.availability == "executed" and o.utility is not None
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _distribution(counts: Counter, keys: list[str], n: int) -> dict[str, Any]:
    """Build a distribution dict (counts, percentages) over a fixed key set."""
    return {
        "counts": {k: counts.get(k, 0) for k in keys},
        "percentages": {k: counts.get(k, 0) / n if n > 0 else 0.0 for k in keys},
        "n": n,
    }


def _total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance: TV(P, Q) = 0.5 * sum |p_i - q_i|."""
    return float(0.5 * np.sum(np.abs(p - q)))


# ---------------------------------------------------------------------------
# Composition and distribution comparisons
# ---------------------------------------------------------------------------


def _task_composition(run: LoadedRun) -> dict[str, Any]:
    """Family and archetype distribution for a run."""
    families = Counter(t.family for t in run.tasks.values())
    archetypes = Counter(t.archetype for t in run.tasks.values())
    splits = Counter(t.split for t in run.tasks.values())
    n = len(run.tasks)
    return {
        "n_tasks": n,
        "family_distribution": _distribution(families, run.all_families, n),
        "archetype_distribution": _distribution(archetypes, run.all_archetypes, n),
        "split_distribution": _distribution(splits, run.all_splits, n),
    }


def _best_action_distribution(run: LoadedRun) -> dict[str, Any]:
    """Best-action distribution across all tasks with measured outcomes."""
    best_actions: list[str] = []
    for tid in run.tasks:
        pairs = _sorted_utility_pairs(run, tid)
        if pairs:
            best_actions.append(pairs[0][0])
    counts = Counter(best_actions)
    n = len(best_actions)
    return {
        "n_tasks_with_outcomes": n,
        "distribution": _distribution(counts, CANONICAL_ACTIONS, n),
    }


def _utility_margin_distribution(run: LoadedRun) -> dict[str, Any]:
    """Utility-margin (best - second-best) distribution across tasks."""
    margins: list[float] = []
    for tid in run.tasks:
        pairs = _sorted_utility_pairs(run, tid)
        if pairs:
            best = pairs[0][1]
            second = pairs[1][1] if len(pairs) > 1 else best
            margins.append(best - second)
    arr = np.array(margins) if margins else np.array([0.0])
    return {
        "n_tasks": len(margins),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "fraction_zero_margin": float(np.mean(arr <= 1e-9)) if len(arr) else 0.0,
    }


# ---------------------------------------------------------------------------
# Configuration and provenance comparisons
# ---------------------------------------------------------------------------


def _extract_gen_config(run: LoadedRun) -> dict[str, Any]:
    """Extract model generation configuration from the benchmark manifest."""
    bm = run.benchmark_manifest
    gen = bm.get("generation_config", {})
    return {
        "model_id": gen.get("model_id", bm.get("model_id", "unknown")),
        "model_revision": gen.get("model_revision", bm.get("model_revision", "unknown")),
        "dtype": gen.get("dtype", bm.get("dtype", "unknown")),
        "device": gen.get("device", bm.get("device", "unknown")),
        "max_new_tokens": gen.get("max_new_tokens", bm.get("max_new_tokens", "unknown")),
        "temperature": gen.get("temperature", bm.get("temperature", "unknown")),
        "top_p": gen.get("top_p", bm.get("top_p", "unknown")),
    }


def _extract_provider_version(run: LoadedRun) -> dict[str, Any]:
    """Extract provider version info from the benchmark manifest."""
    bm = run.benchmark_manifest
    return {
        "provider": bm.get("provider", "unknown"),
        "provider_version": bm.get("provider_version", "unknown"),
        "api_version": bm.get("api_version", "unknown"),
    }


def _extract_benchmark_version(run: LoadedRun) -> dict[str, Any]:
    """Extract benchmark version info from the benchmark manifest."""
    bm = run.benchmark_manifest
    return {
        "benchmark_version": bm.get("benchmark_version", "unknown"),
        "benchmark_revision": bm.get("benchmark_revision", "unknown"),
        "benchmark_manifest_sha256": run.benchmark_manifest_sha256,
        "split_manifest_sha256": run.split_manifest_sha256,
        "counterfactual_results_sha256": run.counterfactual_results_sha256,
        "dataset_sha256": run.dataset_sha256,
    }


def _extract_learner_settings(run: LoadedRun) -> dict[str, Any]:
    """Extract candidate learner settings from candidate training metadata."""
    ct = run.candidate_training
    cp = run.candidate_policy
    return {
        "n_epochs": ct.get("n_epochs", cp.get("n_epochs", "unknown")),
        "learning_rate": ct.get("lr", cp.get("lr", "unknown")),
        "l2_regularization": ct.get("l2", cp.get("l2", "unknown")),
        "seed": ct.get("seed", cp.get("seed", "unknown")),
        "n_train_examples": ct.get("n_train", cp.get("n_train", "unknown")),
        "feature_names": cp.get("feature_names", []),
        "n_features": len(cp.get("feature_names", [])),
        "architecture": cp.get("architecture", "multinomial_logistic"),
    }


def _extract_sham_construction(run: LoadedRun) -> dict[str, Any]:
    """Extract sham construction details from sham training metadata."""
    st = run.sham_training
    sp = run.sham_policy
    return {
        "shuffle_mode": st.get("shuffle_mode", sp.get("shuffle_mode", "unknown")),
        "seed": st.get("seed", sp.get("seed", "unknown")),
        "n_seeds": st.get("n_seeds", 1),
        "n_train": st.get("n_train", sp.get("n_train", "unknown")),
        "architecture": sp.get("architecture", "multinomial_logistic"),
    }


def _extract_calibration_thresholds(run: LoadedRun) -> dict[str, Any]:
    """Extract calibration / gate thresholds from qualification report."""
    qr = run.qualification_report
    ga = run.gate_a_result
    return {
        "gate_a_threshold": ga.get("threshold", qr.get("gate_a_threshold", "unknown")),
        "promotion_threshold": run.promotion_result.get("threshold", qr.get("promotion_threshold", "unknown")),
        "headroom_sufficient": qr.get("headroom_sufficient_threshold", 0.50),
        "headroom_marginal_min": qr.get("headroom_marginal_min_threshold", 0.10),
        "confidence_level": qr.get("confidence_level", 0.95),
    }


def _extract_random_seeds(run: LoadedRun) -> dict[str, Any]:
    """Extract all random seeds used across the run."""
    bm = run.benchmark_manifest
    ct = run.candidate_training
    st = run.sham_training
    return {
        "benchmark_seed": bm.get("benchmark_seed", "unknown"),
        "generator_seed": bm.get("generator_seed", "unknown"),
        "candidate_seed": ct.get("seed", "unknown"),
        "sham_seed": st.get("seed", "unknown"),
        "split_seed": run.split_manifest.get("split_seed", "unknown"),
    }


# ---------------------------------------------------------------------------
# Runtime and verification comparisons
# ---------------------------------------------------------------------------


def _runtime_completion(run: LoadedRun) -> dict[str, Any]:
    """Runtime completion statistics from runtime diagnostics."""
    rd = run.runtime_diagnostics
    total = rd.get("total_tasks", len(run.tasks))
    completed = rd.get("completed_tasks", rd.get("n_completed", 0))
    failed = rd.get("failed_tasks", rd.get("n_failed", 0))
    return {
        "total_tasks": total,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "completion_rate": completed / total if total > 0 else 0.0,
        "wall_time_seconds": rd.get("wall_time_seconds", rd.get("elapsed_seconds", "unknown")),
        "n_retries": rd.get("n_retries", "unknown"),
    }


def _verification_success_rates(run: LoadedRun) -> dict[str, Any]:
    """Verification success rates per action and overall."""
    per_action: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})
    overall = {"success": 0, "total": 0}
    for o in run.outcomes:
        if o.availability != "executed":
            continue
        per_action[o.action_id]["total"] += 1
        overall["total"] += 1
        if o.verification == "success":
            per_action[o.action_id]["success"] += 1
            overall["success"] += 1
    return {
        "overall_success_rate": overall["success"] / overall["total"] if overall["total"] > 0 else 0.0,
        "overall_success_count": overall["success"],
        "overall_total": overall["total"],
        "per_action": {
            a: {
                "success_rate": per_action[a]["success"] / per_action[a]["total"] if per_action[a]["total"] > 0 else 0.0,
                "success_count": per_action[a]["success"],
                "total": per_action[a]["total"],
            }
            for a in CANONICAL_ACTIONS
        },
    }


def _failure_types(run: LoadedRun) -> dict[str, Any]:
    """Categorize failure types from outcomes."""
    failure_counts: Counter = Counter()
    for o in run.outcomes:
        if o.availability != "executed":
            continue
        if o.verification != "success":
            failure_counts[o.verification] += 1
    total_failures = sum(failure_counts.values())
    return {
        "total_failures": total_failures,
        "failure_type_counts": dict(failure_counts),
        "failure_type_percentages": {
            k: v / total_failures if total_failures > 0 else 0.0
            for k, v in failure_counts.items()
        },
    }


# ---------------------------------------------------------------------------
# Overlapping task subset analysis
# ---------------------------------------------------------------------------


def _find_overlapping_tasks(mini: LoadedRun, full: LoadedRun) -> list[str]:
    """Return task_ids present in both runs."""
    return sorted(set(mini.tasks.keys()) & set(full.tasks.keys()))


def _evaluate_policy_on_overlap(
    run: LoadedRun,
    task_ids: list[str],
    action_selector: Any,
) -> np.ndarray:
    """Evaluate a policy selector on the overlapping task subset.

    Returns a per-task utility array aligned to ``task_ids``.
    """
    utilities: list[float] = []
    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            utilities.append(0.0)
            continue
        action = action_selector(tid, task)
        if action is None:
            utilities.append(0.0)
            continue
        u = run.get_utility_for_action(tid, action)
        utilities.append(u if u is not None else 0.0)
    return np.array(utilities)


def _evaluate_baseline_on_overlap(
    run: LoadedRun, task_ids: list[str]
) -> np.ndarray:
    """Evaluate the frozen deterministic baseline (ANSWER_DIRECT) on overlap."""
    def selector(tid, task):
        if "ANSWER_DIRECT" in task.allowed_actions:
            return "ANSWER_DIRECT"
        return task.allowed_actions[0] if task.allowed_actions else None
    return _evaluate_policy_on_overlap(run, task_ids, selector)


def _evaluate_candidate_on_overlap(
    run: LoadedRun, task_ids: list[str]
) -> np.ndarray:
    """Evaluate the learned candidate policy on the overlapping subset."""
    from .policy_controls import make_candidate_selector
    selector = make_candidate_selector(run)
    return _evaluate_policy_on_overlap(run, task_ids, selector)


def _evaluate_oracle_on_overlap(
    run: LoadedRun, task_ids: list[str]
) -> np.ndarray:
    """Evaluate the oracle (best measured action) on the overlapping subset."""
    def selector(tid, task):
        return run.get_best_action(tid)
    return _evaluate_policy_on_overlap(run, task_ids, selector)


def _overlap_analysis(
    mini: LoadedRun, full: LoadedRun, overlap_ids: list[str]
) -> dict[str, Any]:
    """Re-evaluate all policies on the exact overlapping task subset."""
    if not overlap_ids:
        return {
            "n_overlap_tasks": 0,
            "note": "No overlapping task_ids between mini and full runs.",
        }

    # Per-run policy evaluations on the overlap.
    mini_baseline = _evaluate_baseline_on_overlap(mini, overlap_ids)
    mini_candidate = _evaluate_candidate_on_overlap(mini, overlap_ids)
    mini_oracle = _evaluate_oracle_on_overlap(mini, overlap_ids)

    full_baseline = _evaluate_baseline_on_overlap(full, overlap_ids)
    full_candidate = _evaluate_candidate_on_overlap(full, overlap_ids)
    full_oracle = _evaluate_oracle_on_overlap(full, overlap_ids)

    mini_delta = float((mini_candidate - mini_baseline).mean())
    full_delta = float((full_candidate - full_baseline).mean())

    return {
        "n_overlap_tasks": len(overlap_ids),
        "overlap_task_ids": overlap_ids,
        "mini": {
            "U_baseline": float(mini_baseline.mean()),
            "U_candidate": float(mini_candidate.mean()),
            "U_oracle": float(mini_oracle.mean()),
            "candidate_baseline_delta": mini_delta,
            "candidate_oracle_gap": float((mini_oracle - mini_candidate).mean()),
            "baseline_oracle_gap": float((mini_oracle - mini_baseline).mean()),
        },
        "full": {
            "U_baseline": float(full_baseline.mean()),
            "U_candidate": float(full_candidate.mean()),
            "U_oracle": float(full_oracle.mean()),
            "candidate_baseline_delta": full_delta,
            "candidate_oracle_gap": float((full_oracle - full_candidate).mean()),
            "baseline_oracle_gap": float((full_oracle - full_baseline).mean()),
        },
        "delta_change_on_overlap": full_delta - mini_delta,
    }


# ---------------------------------------------------------------------------
# Delta attribution analysis
# ---------------------------------------------------------------------------


def _delta_attribution(
    mini: LoadedRun, full: LoadedRun, overlap_ids: list[str]
) -> dict[str, Any]:
    """Attribute the delta increase to candidate improvement vs baseline weakness.

    The apparent increase in candidate-baseline delta is:
        mini: +0.093, full: +0.415

    To determine the cause we compare:
      1. The delta on the *overlapping* task subset (same tasks, both runs).
         If the overlap delta is similar between runs, the increase is driven
         by the newly added tasks in the full run.
      2. The baseline utility on full-only tasks vs overlap tasks.
         If the baseline is much weaker on full-only tasks, the delta increase
         is caused by baseline weakness on newly added tasks.
      3. The candidate utility on full-only tasks vs overlap tasks.
         If the candidate is much stronger on full-only tasks, the delta
         increase is caused by genuine candidate learning improvement.
    """
    full_only_ids = sorted(set(full.tasks.keys()) - set(mini.tasks.keys()))
    mini_only_ids = sorted(set(mini.tasks.keys()) - set(full.tasks.keys()))

    # Baseline and candidate on full-only tasks.
    full_only_baseline = _evaluate_baseline_on_overlap(full, full_only_ids) if full_only_ids else np.array([])
    full_only_candidate = _evaluate_candidate_on_overlap(full, full_only_ids) if full_only_ids else np.array([])
    full_only_oracle = _evaluate_oracle_on_overlap(full, full_only_ids) if full_only_ids else np.array([])

    # Baseline and candidate on overlap (full run).
    overlap_baseline_full = _evaluate_baseline_on_overlap(full, overlap_ids) if overlap_ids else np.array([])
    overlap_candidate_full = _evaluate_candidate_on_overlap(full, overlap_ids) if overlap_ids else np.array([])
    overlap_oracle_full = _evaluate_oracle_on_overlap(full, overlap_ids) if overlap_ids else np.array([])

    # Baseline and candidate on overlap (mini run).
    overlap_baseline_mini = _evaluate_baseline_on_overlap(mini, overlap_ids) if overlap_ids else np.array([])
    overlap_candidate_mini = _evaluate_candidate_on_overlap(mini, overlap_ids) if overlap_ids else np.array([])

    # Deltas.
    overlap_delta_mini = float((overlap_candidate_mini - overlap_baseline_mini).mean()) if len(overlap_ids) else 0.0
    overlap_delta_full = float((overlap_candidate_full - overlap_baseline_full).mean()) if len(overlap_ids) else 0.0
    full_only_delta = float((full_only_candidate - full_only_baseline).mean()) if len(full_only_ids) else 0.0

    # Baseline weakness indicator: how much lower is baseline on full-only
    # tasks compared to overlap tasks?
    baseline_overlap_mean = float(overlap_baseline_full.mean()) if len(overlap_ids) else 0.0
    baseline_full_only_mean = float(full_only_baseline.mean()) if len(full_only_ids) else 0.0
    baseline_weakness = baseline_overlap_mean - baseline_full_only_mean

    # Candidate improvement indicator: how much higher is candidate on
    # full-only tasks compared to overlap tasks?
    candidate_overlap_mean = float(overlap_candidate_full.mean()) if len(overlap_ids) else 0.0
    candidate_full_only_mean = float(full_only_candidate.mean()) if len(full_only_ids) else 0.0
    candidate_improvement = candidate_full_only_mean - candidate_overlap_mean

    # Determine the dominant cause.
    if baseline_weakness > candidate_improvement and baseline_weakness > 0.01:
        dominant_cause = "baseline_weakness_on_new_tasks"
        explanation = (
            f"The increase in candidate-baseline delta (mini: +{MINI_DELTA:.3f}, "
            f"full: +{FULL_DELTA:.3f}) is primarily caused by baseline weakness "
            f"on newly added tasks. The baseline utility on full-only tasks "
            f"({baseline_full_only_mean:.3f}) is substantially lower than on "
            f"overlapping tasks ({baseline_overlap_mean:.3f}), while the "
            f"candidate improvement on new tasks is smaller "
            f"({candidate_improvement:.3f})."
        )
    elif candidate_improvement > baseline_weakness and candidate_improvement > 0.01:
        dominant_cause = "candidate_learning_improvement"
        explanation = (
            f"The increase in candidate-baseline delta (mini: +{MINI_DELTA:.3f}, "
            f"full: +{FULL_DELTA:.3f}) is primarily caused by genuine candidate "
            f"learning improvement on newly added tasks. The candidate utility "
            f"on full-only tasks ({candidate_full_only_mean:.3f}) is substantially "
            f"higher than on overlapping tasks ({candidate_overlap_mean:.3f}), "
            f"while baseline weakness is smaller ({baseline_weakness:.3f})."
        )
    else:
        dominant_cause = "inconclusive"
        explanation = (
            f"The cause of the delta increase (mini: +{MINI_DELTA:.3f}, "
            f"full: +{FULL_DELTA:.3f}) could not be clearly attributed. "
            f"Baseline weakness ({baseline_weakness:.3f}) and candidate "
            f"improvement ({candidate_improvement:.3f}) are both small or "
            f"similar in magnitude."
        )

    return {
        "research_question": {
            "mini_delta": MINI_DELTA,
            "full_delta": FULL_DELTA,
            "delta_increase": FULL_DELTA - MINI_DELTA,
        },
        "overlap_delta_mini": overlap_delta_mini,
        "overlap_delta_full": overlap_delta_full,
        "full_only_delta": full_only_delta,
        "n_full_only_tasks": len(full_only_ids),
        "n_mini_only_tasks": len(mini_only_ids),
        "full_only_task_ids": full_only_ids,
        "mini_only_task_ids": mini_only_ids,
        "baseline_utility": {
            "overlap_mean": baseline_overlap_mean,
            "full_only_mean": baseline_full_only_mean,
            "weakness_on_new_tasks": baseline_weakness,
        },
        "candidate_utility": {
            "overlap_mean": candidate_overlap_mean,
            "full_only_mean": candidate_full_only_mean,
            "improvement_on_new_tasks": candidate_improvement,
        },
        "oracle_utility": {
            "overlap_mean": float(overlap_oracle_full.mean()) if len(overlap_ids) else 0.0,
            "full_only_mean": float(full_only_oracle.mean()) if len(full_only_ids) else 0.0,
        },
        "dominant_cause": dominant_cause,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Side-by-side comparison builder
# ---------------------------------------------------------------------------


def _side_by_side(mini: LoadedRun, full: LoadedRun) -> dict[str, Any]:
    """Build a side-by-side comparison of all configuration and distribution fields."""
    return {
        "task_composition": {
            "mini": _task_composition(mini),
            "full": _task_composition(full),
        },
        "best_action_distribution": {
            "mini": _best_action_distribution(mini),
            "full": _best_action_distribution(full),
        },
        "utility_margin_distribution": {
            "mini": _utility_margin_distribution(mini),
            "full": _utility_margin_distribution(full),
        },
        "model_generation_config": {
            "mini": _extract_gen_config(mini),
            "full": _extract_gen_config(full),
        },
        "provider_version": {
            "mini": _extract_provider_version(mini),
            "full": _extract_provider_version(full),
        },
        "benchmark_version": {
            "mini": _extract_benchmark_version(mini),
            "full": _extract_benchmark_version(full),
        },
        "candidate_learner_settings": {
            "mini": _extract_learner_settings(mini),
            "full": _extract_learner_settings(full),
        },
        "sham_construction": {
            "mini": _extract_sham_construction(mini),
            "full": _extract_sham_construction(full),
        },
        "calibration_thresholds": {
            "mini": _extract_calibration_thresholds(mini),
            "full": _extract_calibration_thresholds(full),
        },
        "random_seeds": {
            "mini": _extract_random_seeds(mini),
            "full": _extract_random_seeds(full),
        },
        "runtime_completion": {
            "mini": _runtime_completion(mini),
            "full": _runtime_completion(full),
        },
        "verification_success_rates": {
            "mini": _verification_success_rates(mini),
            "full": _verification_success_rates(full),
        },
        "failure_types": {
            "mini": _failure_types(mini),
            "full": _failure_types(full),
        },
    }


def _distribution_comparisons(mini: LoadedRun, full: LoadedRun) -> dict[str, Any]:
    """Compute distribution-level comparison metrics between runs."""
    # Best-action distribution TV distance.
    mini_bd = _best_action_distribution(mini)["distribution"]["percentages"]
    full_bd = _best_action_distribution(full)["distribution"]["percentages"]
    p = np.array([mini_bd.get(a, 0.0) for a in CANONICAL_ACTIONS])
    q = np.array([full_bd.get(a, 0.0) for a in CANONICAL_ACTIONS])
    best_action_tv = _total_variation_distance(p, q)

    # Family distribution TV distance.
    all_fams = sorted(set(mini.all_families) | set(full.all_families))
    mini_fam = Counter(t.family for t in mini.tasks.values())
    full_fam = Counter(t.family for t in full.tasks.values())
    n_mini = len(mini.tasks) or 1
    n_full = len(full.tasks) or 1
    p_fam = np.array([mini_fam.get(f, 0) / n_mini for f in all_fams])
    q_fam = np.array([full_fam.get(f, 0) / n_full for f in all_fams])
    family_tv = _total_variation_distance(p_fam, q_fam)

    # Archetype distribution TV distance.
    all_archs = sorted(set(mini.all_archetypes) | set(full.all_archetypes))
    mini_arch = Counter(t.archetype for t in mini.tasks.values())
    full_arch = Counter(t.archetype for t in full.tasks.values())
    p_arch = np.array([mini_arch.get(a, 0) / n_mini for a in all_archs])
    q_arch = np.array([full_arch.get(a, 0) / n_full for a in all_archs])
    archetype_tv = _total_variation_distance(p_arch, q_arch)

    return {
        "best_action_tv_distance": best_action_tv,
        "family_tv_distance": family_tv,
        "archetype_tv_distance": archetype_tv,
        "shared_families": sorted(set(mini.all_families) & set(full.all_families)),
        "shared_archetypes": sorted(set(mini.all_archetypes) & set(full.all_archetypes)),
        "mini_only_families": sorted(set(mini.all_families) - set(full.all_families)),
        "full_only_families": sorted(set(full.all_families) - set(mini.all_families)),
        "mini_only_archetypes": sorted(set(mini.all_archetypes) - set(full.all_archetypes)),
        "full_only_archetypes": sorted(set(full.all_archetypes) - set(mini.all_archetypes)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_runs(mini_dir: str | Path, full_dir: str | Path) -> dict[str, Any]:
    """Compare mini and full 7B runs and produce the comparison report.

    Loads both runs, compares them on task composition, best-action
    distribution, utility-margin distribution, model generation
    configuration, provider version, benchmark version, candidate learner
    settings, sham construction, calibration thresholds, random seeds,
    runtime completion, verification success rates, and failure types.

    Re-evaluates all policies on the exact overlapping task subset (tasks
    present in both runs by ``task_id``) to determine whether the apparent
    increase in candidate-baseline delta (mini: +0.093, full: +0.415) is
    caused by candidate learning improvement or baseline weakness on
    newly added tasks.

    Args:
        mini_dir: Path to the mini run artifacts directory.
        full_dir: Path to the full run artifacts directory.

    Returns:
        Content for ``mini_full_comparison.json``.
    """
    mini, full = load_two_runs(mini_dir, full_dir)

    overlap_ids = _find_overlapping_tasks(mini, full)

    side_by_side = _side_by_side(mini, full)
    dist_comparisons = _distribution_comparisons(mini, full)
    overlap = _overlap_analysis(mini, full, overlap_ids)
    attribution = _delta_attribution(mini, full, overlap_ids)

    return {
        "mini_artifacts_dir": str(mini.artifacts_dir),
        "full_artifacts_dir": str(full.artifacts_dir),
        "mini_n_tasks": len(mini.tasks),
        "full_n_tasks": len(full.tasks),
        "n_overlapping_tasks": len(overlap_ids),
        "side_by_side": side_by_side,
        "distribution_comparisons": dist_comparisons,
        "overlap_analysis": overlap,
        "delta_attribution": attribution,
        "summary": {
            "mini_delta": MINI_DELTA,
            "full_delta": FULL_DELTA,
            "delta_increase": FULL_DELTA - MINI_DELTA,
            "dominant_cause": attribution["dominant_cause"],
            "n_overlap_tasks": len(overlap_ids),
            "n_full_only_tasks": attribution["n_full_only_tasks"],
            "n_mini_only_tasks": attribution["n_mini_only_tasks"],
        },
    }
