"""Multiple-seed stability analysis for v0.4.2 diagnostics.

Repeats candidate and sham policy training across many learner seeds
to characterize the stability of the learned executive policy.  No new
model executions are required — every policy is trained on the *same*
fixed counterfactual outcomes and evaluated against the measured
test-split utilities.

The output of :func:`compute_seed_stability` is the content for
``seed_stability_results.json``.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .data import LoadedRun
from .sham_trainer import (
    candidate_ensemble,
    sham_ensemble,
    train_candidate_policy,
    train_sham_policy,
)
from .policy_controls import (
    evaluate_policy_on_tasks,
    make_baseline_selector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy (natural log) of a probability vector."""
    p = np.asarray(probs, dtype=float)
    nonzero = p[p > 0]
    if nonzero.size == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log(nonzero)))


def _effective_coverage(action_counts: Counter, n_tasks: int) -> float:
    """Effective coverage = exp(entropy) / n_actions.

    Ranges from 1/n_actions (degenerate) to 1.0 (uniform).
    """
    if n_tasks == 0:
        return 0.0
    probs = np.array(
        [action_counts.get(a, 0) / n_tasks for a in sorted(action_counts)],
        dtype=float,
    )
    if probs.sum() == 0:
        return 0.0
    probs = probs / probs.sum()
    n_actions = len(probs)
    if n_actions <= 1:
        return 1.0
    return float(np.exp(_entropy(probs)) / n_actions)


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(P || Q) with smoothing to avoid division by zero."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _action_distribution(
    selected_actions: list[str],
    all_actions: list[str],
) -> dict[str, float]:
    """Return a normalised action distribution over *all_actions*."""
    n = len(selected_actions)
    counts = Counter(selected_actions)
    return {a: counts.get(a, 0) / n for a in all_actions} if n > 0 else {a: 0.0 for a in all_actions}


def _summarise(values: np.ndarray) -> dict[str, float]:
    """Summary statistics for a 1-D array of scalars."""
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "p05": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def _build_test_feature_matrix(run: LoadedRun) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """Collect test-split features.

    Returns:
        task_ids:   ordered list of test task IDs that have features.
        features:   [n_tasks, n_features] matrix.
        tid_to_ex:  mapping from task_id to the source example dict.
    """
    tid_to_ex: dict[str, dict] = {}
    for ex in run.test_examples:
        tid = ex.get("task_id")
        if tid is not None and "features" in ex:
            tid_to_ex[tid] = ex

    # Only keep tasks that actually exist in the run.
    test_ids = [tid for tid in run.test_task_ids if tid in tid_to_ex]
    if not test_ids:
        return [], np.empty((0, 0)), {}

    feat_list = [np.asarray(tid_to_ex[tid]["features"], dtype=float) for tid in test_ids]
    features = np.stack(feat_list, axis=0)
    return test_ids, features, tid_to_ex


def evaluate_trained_policy_on_test(
    run: LoadedRun,
    policy: dict[str, Any],
    task_ids: list[str] | None = None,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Evaluate a trained policy dictionary on the test split.

    This helper mirrors the logic in
    :func:`policy_controls.make_candidate_selector` but operates on an
    *arbitrary* trained policy dict (candidate or sham) and returns
    rich per-task information including softmax probabilities.

    Steps:
      1. Get test example features from ``run.test_examples``.
      2. Apply trained weights to produce logits.
      3. Select the eligible action with the highest logit.
      4. Look up measured utility for the (task, action) pair from
         ``run.outcomes_by_task_action``.

    Args:
        run:          loaded run data.
        policy:       dict with ``weights``, ``bias``, ``actions``.
        task_ids:     optional subset of test task IDs (default: all test).
        temperature:  softmax temperature for probability reporting.

    Returns:
        dict with per-task utilities, selected actions, max probabilities,
        action distribution, and aggregate statistics.
    """
    weights = np.array(policy.get("weights", []), dtype=float)
    bias = np.array(policy.get("bias", []), dtype=float)
    actions = policy.get("actions", run.all_actions)
    action_idx = {a: i for i, a in enumerate(actions)}

    if task_ids is None:
        task_ids = run.test_task_ids

    # Build feature matrix for the requested tasks.
    tid_to_ex: dict[str, dict] = {}
    for ex in run.test_examples:
        tid = ex.get("task_id")
        if tid is not None:
            tid_to_ex[tid] = ex

    per_task: list[dict[str, Any]] = []
    utilities: list[float] = []
    selected_actions: list[str] = []
    max_probs: list[float] = []

    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        ex = tid_to_ex.get(tid)
        if ex is None or not weights.size:
            # Fall back to first eligible action.
            eligible = [a for a in task.allowed_actions if a in run.all_actions]
            if not eligible:
                continue
            action = eligible[0]
            u = run.get_utility_for_action(tid, action)
            if u is None:
                u = 0.0
            per_task.append({
                "task_id": tid,
                "selected_action": action,
                "utility": u,
                "max_probability": 1.0 / len(eligible),
                "oracle_utility": run.get_oracle_utility(tid),
            })
            utilities.append(u)
            selected_actions.append(action)
            max_probs.append(1.0 / len(eligible))
            continue

        features = np.asarray(ex.get("features", []), dtype=float)
        if features.shape[0] != weights.shape[1]:
            eligible = [a for a in task.allowed_actions if a in run.all_actions]
            if not eligible:
                continue
            action = eligible[0]
            u = run.get_utility_for_action(tid, action)
            if u is None:
                u = 0.0
            per_task.append({
                "task_id": tid,
                "selected_action": action,
                "utility": u,
                "max_probability": 1.0 / len(eligible),
                "oracle_utility": run.get_oracle_utility(tid),
            })
            utilities.append(u)
            selected_actions.append(action)
            max_probs.append(1.0 / len(eligible))
            continue

        logits = features @ weights.T + bias

        # Restrict to eligible actions.
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            eligible = [a for a in task.allowed_actions if a in run.all_actions]
        if not eligible:
            continue

        eligible_indices = [action_idx[a] for a in eligible]
        eligible_logits = logits[eligible_indices]

        # Softmax over eligible actions only.
        with np.errstate(divide="ignore"):
            scaled = eligible_logits / max(temperature, 1e-12)
        probs = _softmax(scaled)
        best_local = int(np.argmax(probs))
        action = eligible[best_local]
        max_prob = float(probs[best_local])

        u = run.get_utility_for_action(tid, action)
        if u is None:
            u = 0.0

        per_task.append({
            "task_id": tid,
            "selected_action": action,
            "utility": u,
            "max_probability": max_prob,
            "oracle_utility": run.get_oracle_utility(tid),
            "eligible_probs": {a: float(probs[j]) for j, a in enumerate(eligible)},
        })
        utilities.append(u)
        selected_actions.append(action)
        max_probs.append(max_prob)

    utilities_arr = np.array(utilities)
    max_probs_arr = np.array(max_probs)
    action_counts = Counter(selected_actions)
    n = len(utilities)

    return {
        "n_tasks": n,
        "mean_utility": float(utilities_arr.mean()) if n > 0 else 0.0,
        "utilities": utilities_arr.tolist(),
        "selected_actions": selected_actions,
        "action_counts": dict(action_counts),
        "action_distribution": _action_distribution(selected_actions, run.all_actions),
        "action_entropy": _entropy(
            np.array([action_counts.get(a, 0) / n for a in run.all_actions], dtype=float)
        ) if n > 0 else 0.0,
        "effective_coverage": _effective_coverage(action_counts, n),
        "mean_max_probability": float(max_probs_arr.mean()) if n > 0 else 0.0,
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_seed_stability(
    run: LoadedRun,
    n_seeds: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute multiple-seed stability for candidate and sham policies.

    Repeats candidate training across ``n_seeds`` learner seeds and each
    sham design across ``n_seeds`` sham seeds.  Every trained policy is
    evaluated on the test split using fixed measured counterfactual
    utilities — no new model executions are performed.

    The **primary** candidate result uses the predeclared ``seed``
    (default 42).  All seeds are reported; we do not cherry-pick the
    best.

    Args:
        run:      loaded run data.
        n_seeds:  number of seeds to train per design.
        seed:     predeclared primary candidate seed.

    Returns:
        Content for ``seed_stability_results.json``.
    """
    test_ids = run.test_task_ids
    all_actions = run.all_actions

    # ------------------------------------------------------------------
    # Baseline (deterministic — seed-independent)
    # ------------------------------------------------------------------
    baseline_selector = make_baseline_selector(run)
    baseline_res = evaluate_policy_on_tasks(
        run, test_ids, baseline_selector, "P_baseline",
    )
    baseline_utils = np.array(baseline_res["utilities"])
    baseline_mean = float(baseline_utils.mean()) if baseline_utils.size > 0 else 0.0
    baseline_action_dist = np.array(
        [baseline_res["action_distribution_pct"].get(a, 0.0) for a in all_actions],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Oracle (deterministic — seed-independent)
    # ------------------------------------------------------------------
    oracle_utils = np.array([
        run.get_oracle_utility(tid) for tid in test_ids
    ])
    oracle_mean = float(oracle_utils.mean()) if oracle_utils.size > 0 else 0.0

    # ------------------------------------------------------------------
    # Candidate ensemble — n_seeds learner seeds
    # ------------------------------------------------------------------
    # The primary seed is the predeclared ``seed``; the remaining seeds
    # are derived deterministically so the full ensemble is reproducible.
    candidate_seeds = [seed + i for i in range(n_seeds)]
    candidate_policies = [
        train_candidate_policy(run, seed=s) for s in candidate_seeds
    ]

    candidate_records: list[dict[str, Any]] = []
    cand_test_utils: list[float] = []
    cand_baseline_deltas: list[float] = []
    cand_action_entropies: list[float] = []
    cand_eff_coverages: list[float] = []
    cand_max_probs: list[float] = []
    cand_action_dists: list[np.ndarray] = []

    for s, pol in zip(candidate_seeds, candidate_policies):
        ev = evaluate_trained_policy_on_test(run, pol, task_ids=test_ids)
        u = ev["mean_utility"]
        utils_arr = np.array(ev["utilities"])
        delta_bl = float((utils_arr - baseline_utils).mean()) if utils_arr.size > 0 else 0.0
        ad = np.array(
            [ev["action_distribution"].get(a, 0.0) for a in all_actions],
            dtype=float,
        )
        cand_test_utils.append(u)
        cand_baseline_deltas.append(delta_bl)
        cand_action_entropies.append(ev["action_entropy"])
        cand_eff_coverages.append(ev["effective_coverage"])
        cand_max_probs.append(ev["mean_max_probability"])
        cand_action_dists.append(ad)
        candidate_records.append({
            "seed": s,
            "mean_utility": u,
            "candidate_baseline_delta": delta_bl,
            "action_entropy": ev["action_entropy"],
            "effective_coverage": ev["effective_coverage"],
            "mean_max_probability": ev["mean_max_probability"],
            "action_distribution": ev["action_distribution"],
            "is_primary": s == seed,
        })

    cand_test_utils_arr = np.array(cand_test_utils)
    cand_baseline_deltas_arr = np.array(cand_baseline_deltas)
    cand_action_entropies_arr = np.array(cand_action_entropies)
    cand_eff_coverages_arr = np.array(cand_eff_coverages)
    cand_max_probs_arr = np.array(cand_max_probs)
    cand_action_dists_arr = np.array(cand_action_dists) if cand_action_dists else np.empty((0, len(all_actions)))

    # ------------------------------------------------------------------
    # Sham ensemble — n_seeds sham seeds per shuffle mode
    # ------------------------------------------------------------------
    sham_modes = ["global", "family", "margin", "feature"]
    sham_results: dict[str, Any] = {}

    # We will compute candidate-sham deltas for each sham mode using
    # the *primary* candidate seed's per-task utilities.
    primary_idx = candidate_seeds.index(seed) if seed in candidate_seeds else 0
    primary_cand_utils = np.array(candidate_records[primary_idx]["mean_utility"])  # scalar

    # For per-task candidate-sham delta we need the primary candidate's
    # per-task utilities.
    primary_ev = evaluate_trained_policy_on_test(
        run, candidate_policies[primary_idx], task_ids=test_ids,
    )
    primary_cand_per_task = np.array(primary_ev["utilities"])

    for mode in sham_modes:
        sham_seeds = [10000 + i for i in range(n_seeds)]
        sham_policies = [
            train_sham_policy(run, shuffle_mode=mode, seed=s) for s in sham_seeds
        ]

        sham_records: list[dict[str, Any]] = []
        sham_utils_list: list[float] = []
        sham_action_dists_list: list[np.ndarray] = []
        sham_per_task_means: list[np.ndarray] = []

        for s, pol in zip(sham_seeds, sham_policies):
            ev = evaluate_trained_policy_on_test(run, pol, task_ids=test_ids)
            u = ev["mean_utility"]
            ad = np.array(
                [ev["action_distribution"].get(a, 0.0) for a in all_actions],
                dtype=float,
            )
            sham_utils_list.append(u)
            sham_action_dists_list.append(ad)
            sham_per_task_means.append(np.array(ev["utilities"]))
            sham_records.append({
                "seed": s,
                "mean_utility": u,
                "action_entropy": ev["action_entropy"],
                "effective_coverage": ev["effective_coverage"],
                "action_distribution": ev["action_distribution"],
            })

        sham_utils_arr = np.array(sham_utils_list)
        sham_action_dists_arr = np.array(sham_action_dists_list) if sham_action_dists_list else np.empty((0, len(all_actions)))

        # Candidate-sham delta for each candidate seed vs mean sham utility.
        cand_sham_deltas = cand_test_utils_arr - sham_utils_arr.mean()

        # Candidate percentile relative to sham distribution.
        if sham_utils_arr.size > 0:
            cand_percentiles = [
                float(np.mean(sham_utils_arr < cu) * 100.0) for cu in cand_test_utils_arr
            ]
        else:
            cand_percentiles = [0.0] * len(cand_test_utils_arr)

        # Primary candidate per-task delta vs each sham seed.
        primary_sham_deltas = [
            float((primary_cand_per_task - spt).mean())
            for spt in sham_per_task_means
        ]

        sham_results[mode] = {
            "n_seeds": n_seeds,
            "seeds": sham_seeds,
            "mean_sham_utility": float(sham_utils_arr.mean()) if sham_utils_arr.size > 0 else 0.0,
            "std_sham_utility": float(sham_utils_arr.std(ddof=1)) if sham_utils_arr.size > 1 else 0.0,
            "interval_95": [
                float(np.percentile(sham_utils_arr, 2.5)),
                float(np.percentile(sham_utils_arr, 97.5)),
            ] if sham_utils_arr.size > 0 else [0.0, 0.0],
            "min_sham_utility": float(sham_utils_arr.min()) if sham_utils_arr.size > 0 else 0.0,
            "max_sham_utility": float(sham_utils_arr.max()) if sham_utils_arr.size > 0 else 0.0,
            "mean_action_distribution": {
                a: float(sham_action_dists_arr[:, j].mean())
                for j, a in enumerate(all_actions)
            } if sham_action_dists_arr.size > 0 else {a: 0.0 for a in all_actions},
            "std_action_distribution": {
                a: float(sham_action_dists_arr[:, j].std(ddof=1))
                for j, a in enumerate(all_actions)
            } if sham_action_dists_arr.size > 1 else {a: 0.0 for a in all_actions},
            "per_seed": sham_records,
            "candidate_percentile_vs_sham": {
                "mean": float(np.mean(cand_percentiles)) if cand_percentiles else 0.0,
                "primary_seed": cand_percentiles[primary_idx] if cand_percentiles else 0.0,
                "per_seed": cand_percentiles,
            },
            "candidate_sham_delta_summary": _summarise(cand_sham_deltas),
            "primary_candidate_sham_delta_per_sham_seed": primary_sham_deltas,
        }

    # ------------------------------------------------------------------
    # Cross-cutting candidate distribution summaries
    # ------------------------------------------------------------------
    # KL from baseline action distribution (per candidate seed).
    kl_from_baseline = [
        _kl_divergence(ad, baseline_action_dist)
        for ad in cand_action_dists
    ]
    kl_from_baseline_arr = np.array(kl_from_baseline)

    # Headroom recovered per candidate seed.
    # R = (U_candidate - U_baseline) / (U_oracle - U_baseline)
    denom = oracle_mean - baseline_mean
    headroom_recovered = [
        float((cu - baseline_mean) / denom) if abs(denom) > 1e-10 else 0.0
        for cu in cand_test_utils
    ]
    headroom_recovered_arr = np.array(headroom_recovered)

    # Candidate-sham delta (primary candidate vs mean sham per mode).
    candidate_sham_delta_by_mode = {
        mode: float(cand_test_utils_arr[primary_idx] - sham_results[mode]["mean_sham_utility"])
        for mode in sham_modes
    }

    return {
        "n_seeds": n_seeds,
        "primary_seed": seed,
        "n_test_tasks": len(test_ids),
        "test_task_ids": test_ids,
        "baseline_mean_utility": baseline_mean,
        "oracle_mean_utility": oracle_mean,
        "candidate": {
            "seeds": candidate_seeds,
            "primary_seed": seed,
            "primary_mean_utility": float(cand_test_utils_arr[primary_idx]),
            "test_utility_distribution": _summarise(cand_test_utils_arr),
            "candidate_baseline_delta_distribution": _summarise(cand_baseline_deltas_arr),
            "action_entropy_distribution": _summarise(cand_action_entropies_arr),
            "effective_coverage_distribution": _summarise(cand_eff_coverages_arr),
            "calibration_mean_max_probability": _summarise(cand_max_probs_arr),
            "kl_from_baseline_distribution": _summarise(kl_from_baseline_arr),
            "headroom_recovered_distribution": _summarise(headroom_recovered_arr),
            "candidate_sham_delta_by_mode": candidate_sham_delta_by_mode,
            "per_seed": candidate_records,
            "note": (
                "The primary candidate result uses the predeclared seed "
                f"({seed}). All {n_seeds} seeds are reported; no "
                "cherry-picking of the best seed is performed."
            ),
        },
        "sham": sham_results,
        "summary": {
            "candidate_test_utility_mean": float(cand_test_utils_arr.mean()) if cand_test_utils_arr.size > 0 else 0.0,
            "candidate_test_utility_std": float(cand_test_utils_arr.std(ddof=1)) if cand_test_utils_arr.size > 1 else 0.0,
            "candidate_test_utility_95_interval": [
                float(np.percentile(cand_test_utils_arr, 2.5)),
                float(np.percentile(cand_test_utils_arr, 97.5)),
            ] if cand_test_utils_arr.size > 0 else [0.0, 0.0],
            "candidate_test_utility_min": float(cand_test_utils_arr.min()) if cand_test_utils_arr.size > 0 else 0.0,
            "candidate_test_utility_max": float(cand_test_utils_arr.max()) if cand_test_utils_arr.size > 0 else 0.0,
            "primary_candidate_test_utility": float(cand_test_utils_arr[primary_idx]) if cand_test_utils_arr.size > 0 else 0.0,
            "best_seed_test_utility": float(cand_test_utils_arr.max()) if cand_test_utils_arr.size > 0 else 0.0,
            "best_seed": int(candidate_seeds[int(np.argmax(cand_test_utils_arr))]) if cand_test_utils_arr.size > 0 else None,
            "worst_seed_test_utility": float(cand_test_utils_arr.min()) if cand_test_utils_arr.size > 0 else 0.0,
            "worst_seed": int(candidate_seeds[int(np.argmin(cand_test_utils_arr))]) if cand_test_utils_arr.size > 0 else None,
            "seed_spread": float(cand_test_utils_arr.max() - cand_test_utils_arr.min()) if cand_test_utils_arr.size > 0 else 0.0,
            "coefficient_of_variation": (
                float(cand_test_utils_arr.std(ddof=1) / abs(cand_test_utils_arr.mean()))
                if cand_test_utils_arr.size > 1 and abs(cand_test_utils_arr.mean()) > 1e-12 else 0.0
            ),
        },
    }
