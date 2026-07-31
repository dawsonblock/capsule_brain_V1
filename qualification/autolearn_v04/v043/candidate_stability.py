"""Candidate training-stability validation for v0.4.3.

Repeats candidate policy training across many learner seeds on the *same*
fixed counterfactual evidence (identical training data, different random
initialisation seeds) to characterise whether the production candidate is
a typical member of the seed ensemble or an outlier.

For each seed we train a multinomial logistic regression with the same
architecture as the production candidate (same features, same actions)
using :func:`train_logistic` from the v0.4.2 ``sham_trainer``.  Each
trained model is evaluated on the test split using the raw argmax over
eligible actions — we deliberately do **not** reconstruct the full
production policy stack for arbitrary seeds, because the production
``LearnedPolicy.select_action`` includes confidence gating, abstention,
and fallback logic that depends on calibration artifacts not available
for synthetic seeds.

The output of :func:`analyze_candidate_stability` is a JSON-serializable
``candidate_stability_results`` dict.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .data import LoadedRun, TaskInfo
from .canonical_evaluator import (
    PolicyDecisionFn,
    evaluate_policy_from_counterfactuals,
    make_oracle_fn,
    make_executed_sham_fn,
    make_baseline_fn,
    compute_headroom_metrics,
)
from .config import (
    CANDIDATE_N_SEEDS,
    CANDIDATE_BASE_SEED,
    MINIMUM_DENOMINATOR,
    HEADROOM_PERCENTAGE_SCALE,
)

# Import the shared logistic trainer from v0.4.2 so that every seed uses
# the exact same training algorithm as the production candidate.
from qualification.autolearn_v04.v042.sham_trainer import train_logistic


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Standard-deviation threshold above which the candidate is classified
#: as ROUTER_UNSTABLE and model-scale attribution is blocked.
ROUTER_UNSTABLE_STD_THRESHOLD: float = 0.5

#: Multiplier for the "outlier" classification (beyond N std of the mean).
OUTLIER_STD_MULTIPLIER: float = 2.0

#: Multiplier for the "typical" classification (within N std of the mean).
TYPICAL_STD_MULTIPLIER: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _summarise(values: np.ndarray) -> dict[str, float]:
    """Summary statistics for a 1-D array of scalars."""
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _action_distribution(
    selected_actions: list[str],
    all_actions: list[str],
) -> dict[str, float]:
    """Return a normalised action distribution over *all_actions*."""
    n = len(selected_actions)
    counts = Counter(selected_actions)
    if n == 0:
        return {a: 0.0 for a in all_actions}
    return {a: counts.get(a, 0) / n for a in all_actions}


def _build_train_feature_matrix(
    run: LoadedRun,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect training-split features and labels.

    Returns:
        features: [n_train, n_features] matrix.
        labels:   [n_train] array of best-action strings.
        actions:  sorted list of all action names.
    """
    examples = run.train_examples
    if not examples:
        return np.empty((0, 0)), np.array([], dtype=object), list(run.all_actions)
    features = np.array([ex["features"] for ex in examples], dtype=float)
    labels = np.array([ex["best_action"] for ex in examples], dtype=object)
    actions = list(run.all_actions)
    return features, labels, actions


def _build_test_feature_lookup(
    run: LoadedRun,
) -> dict[str, np.ndarray]:
    """Build a task_id -> feature-vector lookup for the test split."""
    lookup: dict[str, np.ndarray] = {}
    for ex in run.test_examples:
        tid = ex.get("task_id")
        if tid is not None and "features" in ex:
            lookup[tid] = np.asarray(ex["features"], dtype=float)
    return lookup


def _make_seed_policy_fn(
    W: np.ndarray,
    b: np.ndarray,
    actions: list[str],
    feature_lookup: dict[str, np.ndarray],
) -> PolicyDecisionFn:
    """Build a raw-argmax policy decision function for a trained seed model.

    The function selects the eligible action with the highest logit,
    matching the behaviour of ``make_raw_candidate_fn`` in the canonical
    evaluator.
    """
    action_idx = {a: i for i, a in enumerate(actions)}

    def fn(tid: str, task: TaskInfo) -> str | None:
        feats = feature_lookup.get(tid)
        if feats is None or W.size == 0:
            return task.allowed_actions[0] if task.allowed_actions else None
        if len(feats) != W.shape[1]:
            return task.allowed_actions[0] if task.allowed_actions else None
        logits = feats @ W + b
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            return task.allowed_actions[0] if task.allowed_actions else None
        elig_logits = np.array([logits[action_idx[a]] for a in eligible])
        return eligible[int(np.argmax(elig_logits))]

    return fn


def _evaluate_seed_model(
    run: LoadedRun,
    W: np.ndarray,
    b: np.ndarray,
    actions: list[str],
    task_ids: list[str],
    feature_lookup: dict[str, np.ndarray],
    sham_mean: float,
    baseline_mean: float,
    oracle_mean: float,
) -> dict[str, Any]:
    """Evaluate a single seed model on the test split via raw argmax.

    Returns a per-seed record with mean utility, action distribution,
    candidate-minus-sham, candidate-minus-baseline, headroom recovery,
    and coefficient variance.
    """
    policy_fn = _make_seed_policy_fn(W, b, actions, feature_lookup)
    result = evaluate_policy_from_counterfactuals(
        run, task_ids, policy_fn, "candidate_seed"
    )

    # Action distribution over all actions.
    selected = [
        r.selected_action for r in result.task_results if r.selected_action is not None
    ]
    action_dist = _action_distribution(selected, actions)

    # Coefficient variance: mean across features of the variance across
    # actions for that feature.  This captures how consistently each
    # feature influences the action logits.
    if W.size > 0:
        per_feature_var = np.var(W, axis=1)  # variance across actions
        coeff_var = float(np.mean(per_feature_var)) if per_feature_var.size else 0.0
    else:
        coeff_var = 0.0

    mean_utility = float(result.mean_utility)
    cand_minus_sham = mean_utility - sham_mean
    cand_minus_baseline = mean_utility - baseline_mean

    # Headroom recovery fraction (candidate over sham, relative to oracle).
    denom = oracle_mean - sham_mean
    if abs(denom) > MINIMUM_DENOMINATOR:
        headroom_recovery = float((mean_utility - sham_mean) / denom)
    else:
        headroom_recovery = None

    return {
        "mean_utility": mean_utility,
        "action_distribution": action_dist,
        "candidate_minus_sham": float(cand_minus_sham),
        "candidate_minus_baseline": float(cand_minus_baseline),
        "headroom_recovery": headroom_recovery,
        "coefficient_variance": coeff_var,
        "n_complete": int(result.n_complete),
        "n_tasks": int(result.n_tasks),
    }


def _percentile_of_value(values: np.ndarray, value: float) -> float:
    """Percentile rank of *value* within *values* (0-100)."""
    if values.size == 0:
        return 50.0
    below = int(np.sum(values < value))
    equal = int(np.sum(values == value))
    return float((below + 0.5 * equal) / values.size * 100.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_candidate_stability(
    run: LoadedRun,
    n_seeds: int = CANDIDATE_N_SEEDS,
    base_seed: int = CANDIDATE_BASE_SEED,
) -> dict[str, Any]:
    """Validate candidate training stability across many seeds.

    Trains at least ``n_seeds`` candidate logistic-regression models on
    fixed evidence (same training data, different initialisation seeds)
    and evaluates each on the test split using raw argmax routing.

    The original production candidate's mean utility (from the evaluation
    artifact) is compared against the seed ensemble to determine whether
    it is typical, above/below median, or an outlier.

    If the candidate performance standard deviation across seeds exceeds
    :data:`ROUTER_UNSTABLE_STD_THRESHOLD`, the run is classified as
    ``ROUTER_UNSTABLE`` and model-scale attribution should be blocked.

    Args:
        run:       A LoadedRun from data.py.
        n_seeds:   Number of candidate seeds to train (default 50).
        base_seed: Base seed; seed *i* = base_seed + *i*.

    Returns:
        A JSON-serializable ``candidate_stability_results`` dict with
        per_seed records, aggregate statistics, the original candidate's
        percentile, a stability classification, and an interpretation.
    """
    task_ids = run.test_task_ids
    actions = list(run.all_actions)

    # --- Build fixed training evidence ---
    train_features, train_labels, train_actions = _build_train_feature_matrix(run)

    # --- Build test feature lookup ---
    feature_lookup = _build_test_feature_lookup(run)

    # --- Reference policy means (for candidate-minus-* metrics) ---
    # We evaluate sham, baseline, and oracle through the canonical
    # evaluator so that all comparisons use the same aggregate formula.
    oracle_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_oracle_fn(run), "oracle"
    )
    baseline_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_baseline_fn(run), "baseline"
    )
    sham_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_executed_sham_fn(run), "sham"
    )

    oracle_mean = float(oracle_result.mean_utility)
    baseline_mean = float(baseline_result.mean_utility)
    sham_mean = float(sham_result.mean_utility)

    # --- Train and evaluate each seed ---
    per_seed: list[dict[str, Any]] = []

    if train_features.size == 0 or len(train_labels) == 0:
        # No training data — cannot train seed models.
        return {
            "per_seed": [],
            "aggregate": {},
            "original_candidate_percentile": 50.0,
            "stability_classification": "UNSTABLE",
            "interpretation": (
                "No training examples available; candidate stability "
                "cannot be assessed. Classified as UNSTABLE."
            ),
            "n_seeds_requested": n_seeds,
            "n_seeds_trained": 0,
            "reference_means": {
                "oracle": oracle_mean,
                "baseline": baseline_mean,
                "sham": sham_mean,
            },
        }

    for i in range(n_seeds):
        seed = base_seed + i
        try:
            W, b = train_logistic(
                train_features,
                train_labels,
                train_actions,
                seed=seed,
            )
        except Exception:
            continue

        record = _evaluate_seed_model(
            run, W, b, train_actions, task_ids, feature_lookup,
            sham_mean, baseline_mean, oracle_mean,
        )
        record["seed"] = seed
        per_seed.append(record)

    n_trained = len(per_seed)
    if n_trained == 0:
        return {
            "per_seed": [],
            "aggregate": {},
            "original_candidate_percentile": 50.0,
            "stability_classification": "UNSTABLE",
            "interpretation": (
                "All seed training attempts failed; candidate stability "
                "cannot be assessed. Classified as UNSTABLE."
            ),
            "n_seeds_requested": n_seeds,
            "n_seeds_trained": 0,
            "reference_means": {
                "oracle": oracle_mean,
                "baseline": baseline_mean,
                "sham": sham_mean,
            },
        }

    # --- Aggregate across seeds ---
    mean_utils = np.array([r["mean_utility"] for r in per_seed], dtype=float)
    cms = np.array([r["candidate_minus_sham"] for r in per_seed], dtype=float)
    cmb = np.array([r["candidate_minus_baseline"] for r in per_seed], dtype=float)
    headrooms = np.array(
        [r["headroom_recovery"] for r in per_seed if r["headroom_recovery"] is not None],
        dtype=float,
    )
    coeff_vars = np.array(
        [r["coefficient_variance"] for r in per_seed], dtype=float
    )

    # Action distribution: mean and std per action across seeds.
    action_keys = actions
    action_means: dict[str, float] = {}
    action_stds: dict[str, float] = {}
    for a in action_keys:
        vals = np.array(
            [r["action_distribution"].get(a, 0.0) for r in per_seed], dtype=float
        )
        action_means[a] = float(vals.mean()) if vals.size else 0.0
        action_stds[a] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0

    # Effective coverage: fraction of seeds where candidate beats sham.
    beats_sham = int(np.sum(cms > 0))
    effective_coverage = float(beats_sham / n_trained) if n_trained > 0 else 0.0

    aggregate = {
        "mean_utility": _summarise(mean_utils),
        "candidate_minus_sham": _summarise(cms),
        "candidate_minus_baseline": _summarise(cmb),
        "headroom_recovery": _summarise(headrooms) if headrooms.size else {},
        "action_distribution": {
            "mean": action_means,
            "std": action_stds,
        },
        "coefficient_variance": {
            "mean": float(coeff_vars.mean()) if coeff_vars.size else 0.0,
            "std": float(coeff_vars.std(ddof=1)) if coeff_vars.size > 1 else 0.0,
        },
        "effective_coverage": effective_coverage,
        "n_seeds": n_trained,
    }

    # --- Original candidate percentile ---
    original_mean: float | None = None
    if run.eval_candidate is not None:
        original_mean = float(run.eval_candidate.mean_utility)
    if original_mean is None and run.candidate_policy:
        # Fall back to raw candidate evaluation.
        from .canonical_evaluator import make_raw_candidate_fn
        raw_result = evaluate_policy_from_counterfactuals(
            run, task_ids, make_raw_candidate_fn(run), "candidate_raw"
        )
        original_mean = float(raw_result.mean_utility)

    if original_mean is not None and mean_utils.size > 0:
        original_percentile = _percentile_of_value(mean_utils, original_mean)
    else:
        original_percentile = 50.0

    # --- Stability classification ---
    util_std = float(mean_utils.std(ddof=1)) if mean_utils.size > 1 else 0.0
    util_mean = float(mean_utils.mean()) if mean_utils.size else 0.0

    if util_std > ROUTER_UNSTABLE_STD_THRESHOLD:
        stability_classification = "ROUTER_UNSTABLE"
    elif util_std < 1e-9:
        stability_classification = "STABLE"
    else:
        stability_classification = "STABLE"

    # --- Original candidate position relative to ensemble ---
    if original_mean is not None and mean_utils.size > 0:
        diff = original_mean - util_mean
        z_score = diff / util_std if util_std > 1e-12 else 0.0
        if abs(z_score) > OUTLIER_STD_MULTIPLIER:
            original_position = "outlier"
        elif abs(z_score) <= TYPICAL_STD_MULTIPLIER:
            original_position = "typical"
        elif original_mean > float(np.median(mean_utils)):
            original_position = "above_median"
        else:
            original_position = "below_median"
    else:
        original_position = "unknown"
        z_score = None

    # --- Interpretation ---
    if stability_classification == "ROUTER_UNSTABLE":
        interpretation = (
            f"Candidate performance is ROUTER_UNSTABLE: std={util_std:.4f} "
            f"across {n_trained} seeds (threshold={ROUTER_UNSTABLE_STD_THRESHOLD}). "
            f"Model-scale attribution should be blocked because the learned "
            f"router is not reproducible across initialisation seeds."
        )
    elif original_position == "outlier":
        interpretation = (
            f"Original candidate is an OUTLIER (z={z_score:.2f} if z_score is not None else 'N/A') "
            f"relative to the {n_trained}-seed ensemble "
            f"(mean={util_mean:.4f}, std={util_std:.4f}). The production "
            f"candidate may be cherry-picked or atypical."
        )
    elif original_position == "typical":
        interpretation = (
            f"Original candidate is TYPICAL (within {TYPICAL_STD_MULTIPLIER} std of the "
            f"{n_trained}-seed ensemble mean). std={util_std:.4f}. "
            f"The learned router is reproducible across seeds."
        )
    else:
        interpretation = (
            f"Original candidate is {original_position.replace('_', ' ')} "
            f"relative to the {n_trained}-seed ensemble "
            f"(mean={util_mean:.4f}, std={util_std:.4f}, "
            f"percentile={original_percentile:.1f}). "
            f"std={util_std:.4f} is below the ROUTER_UNSTABLE threshold."
        )

    return {
        "per_seed": per_seed,
        "aggregate": aggregate,
        "original_candidate_percentile": float(original_percentile),
        "original_candidate_mean_utility": original_mean,
        "original_candidate_position": original_position,
        "original_candidate_z_score": float(z_score) if z_score is not None else None,
        "stability_classification": stability_classification,
        "router_unstable_std_threshold": ROUTER_UNSTABLE_STD_THRESHOLD,
        "interpretation": interpretation,
        "n_seeds_requested": n_seeds,
        "n_seeds_trained": n_trained,
        "reference_means": {
            "oracle": oracle_mean,
            "baseline": baseline_mean,
            "sham": sham_mean,
        },
    }
