"""Sham ensemble analysis for v0.4.3.

Runs multiple sham policy seeds across six sham types and compares the
candidate's executed utility against the sham-seed distribution.

Sham types retained from v0.4.2:
  - global_label_shuffle: globally shuffle best-action labels
  - family_conditioned_shuffle: shuffle within task family
  - margin_bin_shuffle: shuffle within utility-margin bins
  - feature_shuffle: shuffle feature vectors, keep labels
  - frequency_control: sample actions per D_experience best-action distribution
  - majority_control: always select the majority best-action

For each sham type, 50 seeds are run.  Each seed trains a sham policy
(with shuffled labels for the first four types, or uses a control
decision function for the last two) and evaluates it on the test split
using the raw argmax of the trained weights (not the production stack,
since we cannot reconstruct the full production policy for arbitrary
seeds).

The candidate percentile -- what fraction of sham seeds have utility
below the candidate's executed mean utility -- is the primary comparison
metric.  A candidate below the 50th percentile is not meaningfully
better than sham.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .data import LoadedRun, TaskInfo
from .canonical_evaluator import (
    PolicyDecisionFn,
    evaluate_policy_from_counterfactuals,
    make_majority_fn,
    make_frequency_fn,
    make_oracle_fn,
    make_baseline_fn,
)
from ..v042.sham_trainer import train_sham_policy


# ---------------------------------------------------------------------------
# Sham type registry
# ---------------------------------------------------------------------------

# Maps sham type name to train_sham_policy shuffle_mode.
# ``None`` entries use a dedicated control decision function instead of
# training a shuffled-label logistic model.
_SHUFFLE_MODE_MAP: dict[str, str | None] = {
    "global_label_shuffle": "global",
    "family_conditioned_shuffle": "family",
    "margin_bin_shuffle": "margin",
    "feature_shuffle": "feature",
    "frequency_control": None,
    "majority_control": None,
}

SHAM_TYPES: list[str] = list(_SHUFFLE_MODE_MAP.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_policy_fn(
    policy: dict[str, Any],
    run: LoadedRun,
) -> PolicyDecisionFn:
    """Build a raw-argmax decision function from a trained policy dict.

    This mirrors ``make_raw_candidate_fn`` but accepts an arbitrary
    policy dict (e.g. a trained sham policy) rather than
    ``run.candidate_policy``.
    """
    W = np.array(policy.get("weights", []))
    b = np.array(policy.get("bias", []))
    actions = policy.get("actions", run.all_actions)
    action_idx = {a: i for i, a in enumerate(actions)}

    feature_lookup: dict[str, np.ndarray] = {}
    for ex in (
        run.test_examples
        + run.train_examples
        + run.validation_examples
        + run.ood_examples
    ):
        feature_lookup[ex["task_id"]] = np.array(ex.get("features", []))

    def fn(tid: str, task: TaskInfo) -> str | None:
        feats = feature_lookup.get(tid)
        if feats is None or W.size == 0:
            return task.allowed_actions[0] if task.allowed_actions else None
        if len(feats) != W.shape[1]:
            return task.allowed_actions[0] if task.allowed_actions else None
        logits = feats @ W.T + b
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            return task.allowed_actions[0] if task.allowed_actions else None
        elig_logits = np.array([logits[action_idx[a]] for a in eligible])
        return eligible[int(np.argmax(elig_logits))]

    return fn


def _compute_training_loss(
    policy: dict[str, Any],
    run: LoadedRun,
) -> float:
    """Compute mean cross-entropy loss of a trained policy on training data."""
    examples = run.train_examples
    if not examples:
        return 0.0
    W = np.array(policy.get("weights", []))
    b = np.array(policy.get("bias", []))
    if W.size == 0:
        return 0.0
    actions = policy.get("actions", run.all_actions)
    features = np.array([ex["features"] for ex in examples])
    labels = np.array([ex["best_action"] for ex in examples])
    label_idx = {a: i for i, a in enumerate(actions)}
    y = np.array([label_idx.get(a, 0) for a in labels])
    logits = features @ W + b
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_exp_sum = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - log_exp_sum
    n = len(y)
    return float(-log_probs[np.arange(n), y].mean())


def _normal_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Normal-approximation confidence interval (mean +/- z * SE)."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    z = 1.959963984540054  # 95% two-sided
    half = z * std / math.sqrt(len(arr))
    return mean - half, mean + half


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sham_ensemble(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    n_seeds: int = 50,
    base_seed: int = 10000,
) -> dict:
    """Run the sham-seed ensemble across all six sham types.

    For each sham type, ``n_seeds`` seeds are run.  Each seed:

    1. Trains a sham policy with shuffled labels using
       ``train_sham_policy`` from ``v042.sham_trainer`` (for the four
       shuffle-based types), or uses a control decision function
       (frequency / majority).
    2. Evaluates on test tasks using the raw argmax of the trained
       weights (not the production stack, since we cannot reconstruct
       the full production policy for arbitrary seeds).
    3. Records: utility, action distribution, headroom recovery, regret,
       training loss.

    The primary sham comparison uses the mean sham policy performance
    across seeds.  The candidate percentile is the fraction of sham
    seeds whose utility is below the candidate's executed mean utility
    (from ``run.eval_candidate.mean_utility``).  A candidate below the
    50th percentile is not meaningfully better than sham.

    Args:
        run: The loaded run with counterfactual outcomes and evaluation
            artifacts.
        task_ids: Test task IDs to evaluate on.  Defaults to
            ``run.test_task_ids``.
        n_seeds: Number of seeds per sham type (default 50).
        base_seed: Base random seed (default 10000).

    Returns:
        JSON-serializable ``sham_ensemble_results`` dict with:

        - ``by_type``: dict of sham_type -> {seeds, mean_utility,
          std_utility, ci_low, ci_high, action_distribution}
        - ``candidate_percentile``: dict of sham_type -> percentile
        - ``primary_sham_mean``: float (mean utility of
          global_label_shuffle)
        - ``candidate_beats_sham``: bool
        - ``candidate_utility``: float
        - ``sham_seed_utilities``: list of floats (primary sham type)
        - ``interpretation``: str
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # Evaluate oracle and baseline once (for headroom reference).
    oracle_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_oracle_fn(run), "oracle",
    )
    baseline_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_baseline_fn(run), "baseline",
    )

    # Candidate executed mean utility from the actual evaluation artifact.
    candidate_utility: float = (
        run.eval_candidate.mean_utility
        if run.eval_candidate is not None
        else 0.0
    )

    by_type: dict[str, dict[str, Any]] = {}
    candidate_percentile: dict[str, float] = {}
    primary_sham_seed_utilities: list[float] = []

    for sham_type, shuffle_mode in _SHUFFLE_MODE_MAP.items():
        seed_records: list[dict[str, Any]] = []
        seed_utilities: list[float] = []

        for i in range(n_seeds):
            seed = base_seed + i

            if shuffle_mode is not None:
                # Train a sham policy with shuffled labels.
                policy = train_sham_policy(
                    run, shuffle_mode=shuffle_mode, seed=seed,
                )
                if "error" in policy:
                    seed_records.append({
                        "seed": seed,
                        "utility": 0.0,
                        "action_distribution": {},
                        "headroom_recovery": None,
                        "regret": None,
                        "training_loss": 0.0,
                        "error": policy["error"],
                    })
                    seed_utilities.append(0.0)
                    continue
                fn = _make_raw_policy_fn(policy, run)
                training_loss = _compute_training_loss(policy, run)
            elif sham_type == "frequency_control":
                fn = make_frequency_fn(run, seed=seed)
                training_loss = 0.0
            elif sham_type == "majority_control":
                fn = make_majority_fn(run)
                training_loss = 0.0
            else:
                continue

            result = evaluate_policy_from_counterfactuals(
                run, task_ids, fn, f"sham_{sham_type}_seed{seed}",
                reference_result=baseline_result,
                oracle_result=oracle_result,
            )

            headroom_recovery: float | None = None
            if result.headroom_vs_reference:
                headroom_recovery = result.headroom_vs_reference.get(
                    "recovered_headroom_fraction",
                )

            record = {
                "seed": seed,
                "utility": float(result.mean_utility),
                "action_distribution": dict(result.action_distribution),
                "headroom_recovery": (
                    float(headroom_recovery)
                    if headroom_recovery is not None
                    else None
                ),
                "regret": (
                    float(result.mean_regret)
                    if result.mean_regret is not None
                    else None
                ),
                "training_loss": float(training_loss),
            }
            seed_records.append(record)
            seed_utilities.append(float(result.mean_utility))

        # Compute statistics across seeds.
        utilities_arr = np.array(seed_utilities, dtype=float)
        mean_util = float(utilities_arr.mean()) if len(utilities_arr) else 0.0
        std_util = (
            float(utilities_arr.std(ddof=1))
            if len(utilities_arr) > 1
            else 0.0
        )
        ci_low, ci_high = _normal_ci(seed_utilities)

        # Mean action distribution across seeds.
        action_dist_values: dict[str, list[int]] = {}
        for rec in seed_records:
            ad = rec["action_distribution"]
            for a, c in ad.items():
                action_dist_values.setdefault(a, []).append(c)
        mean_action_dist: dict[str, float] = {}
        for a, counts in action_dist_values.items():
            mean_action_dist[a] = float(np.mean(counts))

        by_type[sham_type] = {
            "seeds": seed_records,
            "mean_utility": mean_util,
            "std_utility": std_util,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "action_distribution": mean_action_dist,
        }

        # Candidate percentile: fraction of sham seeds below candidate utility.
        if seed_utilities:
            percentile = float(
                np.mean(utilities_arr < candidate_utility) * 100.0
            )
        else:
            percentile = 0.0
        candidate_percentile[sham_type] = percentile

        # Track primary sham type seed utilities for downstream consumers.
        if sham_type == "global_label_shuffle":
            primary_sham_seed_utilities = seed_utilities

    # Primary sham mean: mean utility of the global_label_shuffle type.
    primary_sham_mean = float(
        by_type.get("global_label_shuffle", {}).get("mean_utility", 0.0)
    )

    # Candidate beats sham: candidate above 50th percentile for primary sham.
    primary_percentile = candidate_percentile.get(
        "global_label_shuffle", 0.0,
    )
    candidate_beats_sham = primary_percentile > 50.0

    # Interpretation.
    if candidate_beats_sham:
        interpretation = (
            f"Candidate executed mean utility ({candidate_utility:.4f}) "
            f"exceeds the primary sham mean ({primary_sham_mean:.4f}); "
            f"candidate percentile={primary_percentile:.1f}% (> 50%). "
            "The candidate is meaningfully better than sham."
        )
    else:
        interpretation = (
            f"Candidate executed mean utility ({candidate_utility:.4f}) "
            f"does not exceed the primary sham mean "
            f"({primary_sham_mean:.4f}); "
            f"candidate percentile={primary_percentile:.1f}% (<= 50%). "
            "A candidate below the 50th percentile is not meaningfully "
            "better than sham."
        )

    return {
        "by_type": by_type,
        "candidate_percentile": candidate_percentile,
        "primary_sham_mean": primary_sham_mean,
        "candidate_beats_sham": candidate_beats_sham,
        "candidate_utility": float(candidate_utility),
        "sham_seed_utilities": primary_sham_seed_utilities,
        "interpretation": interpretation,
        "n_seeds": n_seeds,
        "base_seed": base_seed,
    }
