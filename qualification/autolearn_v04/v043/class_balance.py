"""Class-balance analysis for the v0.4.3 repair release.

Audits whether the multinomial logistic regression learner collapses
toward the majority action and evaluates a panel of configurable
class-weighting strategies to mitigate imbalance.

Weighting options evaluated (all trained on D_experience, evaluated on
D_validation):

  1. **no_class_weighting** — uniform weights (baseline).
  2. **inverse_frequency** — inverse-frequency class weights clipped
     to ``[0.1, 10.0]``.
  3. **effective_number** — effective-number-of-samples weighting
     (Cui et al., 2019) with ``beta = 0.99``.
  4. **utility_margin** — per-sample weights based on utility margin
     (``weight = max(0, margin) + 1.0``).
  5. **class_balanced_utility** — inverse-frequency class weights
     combined with utility-margin sample weights.

For each option, the module records class weights, effective sample
weights, weighted class totals, raw action distribution, and predicted
action distribution on validation.

Rejection criteria:
  - **Route collapse**: predicted action entropy < 0.1.
  - **Seed instability**: std of validation utility across 3 seeds > 0.5.
  - **Accuracy-utility conflict**: action accuracy improves while
    expected utility decreases relative to no-weighting.

The best weighting option is selected on D_validation only.

The output of :func:`analyze_class_balance` is a JSON-serializable dict
suitable for writing to ``class_balance_results.json``.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import LoadedRun, TaskInfo
from .canonical_evaluator import (
    PolicyDecisionFn,
    compute_headroom_metrics,
    evaluate_policy_from_counterfactuals,
    make_oracle_fn,
    make_executed_sham_fn,
    make_baseline_fn,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Clip bounds for inverse-frequency weights.
INVERSE_FREQ_CLIP_MIN: float = 0.1
INVERSE_FREQ_CLIP_MAX: float = 10.0

#: Beta for effective-number-of-samples weighting.
EFFECTIVE_NUMBER_BETA: float = 0.99

#: Entropy threshold below which a predicted distribution is "collapsed".
COLLAPSE_ENTROPY_THRESHOLD: float = 0.1

#: Std threshold for seed instability.
SEED_INSTABILITY_STD_THRESHOLD: float = 0.5

#: Number of seeds used for stability checking.
N_STABILITY_SEEDS: int = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_features_labels(
    examples: list[dict[str, Any]],
    actions: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix, label indices, and task IDs from examples.

    Returns (features [n, d], label_indices [n], task_ids [n]).
    """
    features = np.array([ex["features"] for ex in examples], dtype=float)
    action_idx = {a: i for i, a in enumerate(actions)}
    labels = np.array([action_idx.get(ex["best_action"], 0) for ex in examples])
    task_ids = [ex["task_id"] for ex in examples]
    return features, labels, task_ids


def _compute_raw_action_distribution(
    train_labels: np.ndarray,
    actions: list[str],
) -> dict[str, Any]:
    """Compute the raw best-action distribution in the training data."""
    n = len(train_labels)
    counts = Counter(int(l) for l in train_labels)
    dist: dict[str, int] = {}
    probs: dict[str, float] = {}
    for i, a in enumerate(actions):
        c = counts.get(i, 0)
        dist[a] = c
        probs[a] = float(c / n) if n > 0 else 0.0
    return {
        "counts": dist,
        "probabilities": probs,
        "n_total": n,
    }


def _compute_inverse_freq_weights(
    train_labels: np.ndarray,
    actions: list[str],
) -> dict[int, float]:
    """Compute inverse-frequency class weights clipped to [0.1, 10.0].

    weight_c = n / (n_c * n_classes), clipped.
    """
    n = len(train_labels)
    n_classes = len(actions)
    counts = Counter(int(l) for l in train_labels)
    weights: dict[int, float] = {}
    for i in range(n_classes):
        c = counts.get(i, 0)
        if c == 0:
            w = INVERSE_FREQ_CLIP_MAX
        else:
            w = n / (c * n_classes)
        w = float(np.clip(w, INVERSE_FREQ_CLIP_MIN, INVERSE_FREQ_CLIP_MAX))
        weights[i] = w
    return weights


def _compute_effective_number_weights(
    train_labels: np.ndarray,
    actions: list[str],
    beta: float = EFFECTIVE_NUMBER_BETA,
) -> dict[int, float]:
    """Compute effective-number-of-samples class weights.

    weight_c = (1 - beta) / (1 - beta^n_c)

    (Cui et al., "Class-Balanced Loss Based on Effective Number of
    Samples", CVPR 2019.)
    """
    counts = Counter(int(l) for l in train_labels)
    weights: dict[int, float] = {}
    for i in range(len(actions)):
        c = counts.get(i, 0)
        if c == 0:
            w = 1.0
        else:
            w = (1.0 - beta) / (1.0 - beta ** c)
        weights[i] = float(w)
    return weights


def _compute_utility_margin_sample_weights(
    examples: list[dict[str, Any]],
) -> np.ndarray:
    """Compute per-sample utility-margin weights.

    weight = max(0, utility_margin) + 1.0.
    """
    weights = np.ones(len(examples), dtype=float)
    for i, ex in enumerate(examples):
        margin = ex.get("utility_margin", 0.0)
        weights[i] = max(0.0, float(margin)) + 1.0
    return weights


def _compute_action_entropy(action_dist: dict[str, int], n: int) -> float:
    """Compute the entropy of an action distribution (natural log)."""
    if n == 0:
        return 0.0
    probs = np.array([c / n for c in action_dist.values()], dtype=float)
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log(nonzero))) if len(nonzero) > 0 else 0.0


def _compute_action_accuracy(task_results: list[Any]) -> float:
    """Compute the fraction of tasks where predicted action matches oracle."""
    if not task_results:
        return 0.0
    correct = sum(
        1 for r in task_results
        if r.selected_action is not None and r.selected_action == r.oracle_action
    )
    return float(correct / len(task_results))


def _build_feature_lookup(run: LoadedRun) -> dict[str, np.ndarray]:
    """Build a task_id -> features lookup from all available examples."""
    lookup: dict[str, np.ndarray] = {}
    for ex in (
        run.train_examples
        + run.validation_examples
        + run.test_examples
        + run.ood_examples
        + run.safety_examples
    ):
        tid = ex.get("task_id", "")
        if tid:
            lookup[tid] = np.array(ex.get("features", []), dtype=float)
    return lookup


def _make_sklearn_policy_fn(
    model: Any,
    feature_lookup: dict[str, np.ndarray],
    actions: list[str],
) -> PolicyDecisionFn:
    """Create a PolicyDecisionFn from a trained sklearn model."""

    def fn(tid: str, task: TaskInfo) -> str | None:
        feats = feature_lookup.get(tid)
        if feats is None or feats.size == 0:
            return task.allowed_actions[0] if task.allowed_actions else None
        pred = int(model.predict(feats.reshape(1, -1))[0])
        if pred < 0 or pred >= len(actions):
            pred = 0
        pred_action = actions[pred]
        if pred_action in task.allowed_actions:
            return pred_action
        eligible = [a for a in task.allowed_actions if a in actions]
        if eligible:
            return eligible[0]
        return task.allowed_actions[0] if task.allowed_actions else None

    return fn


# ---------------------------------------------------------------------------
# Weighting option evaluation
# ---------------------------------------------------------------------------


def _train_and_evaluate_option(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    feature_lookup: dict[str, np.ndarray],
    val_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
    class_weights: dict[int, float] | None,
    sample_weights: np.ndarray | None,
    seed: int,
    sham_result: Any,
    baseline_result: Any,
    oracle_result: Any,
    option_name: str,
) -> tuple[Any, dict[str, Any]]:
    """Train a logistic regression with the given weights and evaluate on validation.

    Returns (trained_model, metrics_dict).
    """
    # Build sklearn class_weight dict (uses label values as keys).
    sklearn_cw: dict[int, float] | None = None
    if class_weights is not None:
        sklearn_cw = {int(k): float(v) for k, v in class_weights.items()}

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight=sklearn_cw,
        random_state=seed,
    )
    if sample_weights is not None:
        model.fit(train_features, train_labels, sample_weight=sample_weights)
    else:
        model.fit(train_features, train_labels)

    # Evaluate on validation.
    policy_fn = _make_sklearn_policy_fn(model, feature_lookup, actions)
    result = evaluate_policy_from_counterfactuals(
        run, val_task_ids, policy_fn, option_name,
        reference_result=sham_result, oracle_result=oracle_result,
    )

    mean_utility = result.mean_utility
    sham_mean = sham_result.mean_utility
    baseline_mean = baseline_result.mean_utility
    oracle_mean = oracle_result.mean_utility

    headroom_vs_sham = compute_headroom_metrics(
        candidate_mean=mean_utility,
        reference_mean=sham_mean,
        oracle_mean=oracle_mean,
        n_tasks=result.n_complete,
        task_ids_digest=result.task_ids_digest,
        reference_policy_id="sham",
        candidate_policy_id=option_name,
    )

    headroom_vs_baseline = compute_headroom_metrics(
        candidate_mean=mean_utility,
        reference_mean=baseline_mean,
        oracle_mean=oracle_mean,
        n_tasks=result.n_complete,
        task_ids_digest=result.task_ids_digest,
        reference_policy_id="baseline",
        candidate_policy_id=option_name,
    )

    action_accuracy = _compute_action_accuracy(result.task_results)
    action_entropy = _compute_action_entropy(
        result.action_distribution, result.n_tasks,
    )

    # Compute effective sample weights and weighted class totals.
    n_train = len(train_labels)
    eff_weights = np.ones(n_train, dtype=float)
    if class_weights is not None:
        for i in range(n_train):
            eff_weights[i] *= class_weights.get(int(train_labels[i]), 1.0)
    if sample_weights is not None:
        eff_weights *= sample_weights

    weighted_totals: dict[str, float] = {}
    for idx, a in enumerate(actions):
        mask = train_labels == idx
        weighted_totals[a] = float(eff_weights[mask].sum())

    metrics = {
        "validation_expected_utility": float(mean_utility),
        "validation_candidate_minus_sham": float(mean_utility - sham_mean),
        "validation_candidate_minus_baseline": float(mean_utility - baseline_mean),
        "validation_regret": float(result.mean_regret) if result.mean_regret is not None else None,
        "validation_headroom_recovered_vs_sham": headroom_vs_sham,
        "validation_headroom_recovered_vs_baseline": headroom_vs_baseline,
        "validation_action_accuracy": float(action_accuracy),
        "validation_action_entropy": float(action_entropy),
        "validation_action_distribution": dict(result.action_distribution),
        "n_validation_tasks": result.n_tasks,
        "n_validation_complete": result.n_complete,
        "class_weights": (
            {actions[k]: float(v) for k, v in class_weights.items()}
            if class_weights is not None
            else None
        ),
        "effective_sample_weights": {
            "mean": float(eff_weights.mean()),
            "std": float(eff_weights.std()),
            "min": float(eff_weights.min()),
            "max": float(eff_weights.max()),
        },
        "weighted_class_totals": weighted_totals,
    }

    return model, metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_class_balance(run: LoadedRun, seed: int = 42) -> dict[str, Any]:
    """Audit class-balance and evaluate weighting strategies for the learner.

    Trains logistic regression under five class-weighting configurations
    on D_experience, evaluates each on D_validation, checks rejection
    criteria (route collapse, seed instability, accuracy-utility
    conflict), and selects the best option on validation only.

    Args:
        run: loaded run data.
        seed: random seed for reproducibility.

    Returns:
        JSON-serializable dict with per-option metrics, the best
        option, rejection notes, and an interpretation.  Suitable for
        writing to ``class_balance_results.json``.
    """
    actions = run.all_actions

    # ------------------------------------------------------------------
    # Prepare data splits.
    # ------------------------------------------------------------------
    train_examples = run.train_examples
    val_examples = run.validation_examples

    if not train_examples or not val_examples:
        return {
            "error": "Missing training or validation examples.",
            "n_train": len(train_examples),
            "n_validation": len(val_examples),
            "seed": seed,
        }

    train_features, train_labels, _ = _extract_features_labels(
        train_examples, actions,
    )
    _, _, val_task_ids = _extract_features_labels(val_examples, actions)

    feature_lookup = _build_feature_lookup(run)

    # Raw action distribution.
    raw_dist = _compute_raw_action_distribution(train_labels, actions)

    # ------------------------------------------------------------------
    # Evaluate reference policies on validation.
    # ------------------------------------------------------------------
    oracle_val = evaluate_policy_from_counterfactuals(
        run, val_task_ids, make_oracle_fn(run), "oracle",
    )
    sham_val = evaluate_policy_from_counterfactuals(
        run, val_task_ids, make_executed_sham_fn(run), "sham",
        reference_result=None, oracle_result=oracle_val,
    )
    baseline_val = evaluate_policy_from_counterfactuals(
        run, val_task_ids, make_baseline_fn(run), "baseline",
        reference_result=None, oracle_result=oracle_val,
    )

    # ------------------------------------------------------------------
    # Pre-compute weighting strategies.
    # ------------------------------------------------------------------
    inverse_freq_cw = _compute_inverse_freq_weights(train_labels, actions)
    effective_number_cw = _compute_effective_number_weights(train_labels, actions)
    utility_margin_sw = _compute_utility_margin_sample_weights(train_examples)

    # Define the five options.
    option_specs: list[tuple[str, dict[int, float] | None, np.ndarray | None]] = [
        ("no_class_weighting", None, None),
        ("inverse_frequency", inverse_freq_cw, None),
        ("effective_number", effective_number_cw, None),
        ("utility_margin", None, utility_margin_sw),
        ("class_balanced_utility", inverse_freq_cw, utility_margin_sw),
    ]

    # ------------------------------------------------------------------
    # Evaluate each option.
    # ------------------------------------------------------------------
    options: dict[str, dict[str, Any]] = {}
    rejection_notes: list[str] = []

    # First, evaluate no_class_weighting as the reference for
    # accuracy-utility conflict checks.
    no_weight_metrics: dict[str, Any] | None = None

    for option_name, cw, sw in option_specs:
        try:
            # Train and evaluate with the primary seed.
            _, metrics = _train_and_evaluate_option(
                train_features, train_labels, feature_lookup,
                val_task_ids, run, actions,
                cw, sw, seed,
                sham_val, baseline_val, oracle_val,
                option_name,
            )

            # Add metadata about the weighting configuration.
            metrics["raw_action_distribution"] = raw_dist

            # ----------------------------------------------------------
            # Rejection check 1: route collapse.
            # ----------------------------------------------------------
            pred_entropy = metrics["validation_action_entropy"]
            is_collapsed = pred_entropy < COLLAPSE_ENTROPY_THRESHOLD
            metrics["is_collapsed"] = bool(is_collapsed)
            if is_collapsed:
                rejection_notes.append(
                    f"{option_name}: rejected — predicted action entropy "
                    f"({pred_entropy:.4f}) below collapse threshold "
                    f"({COLLAPSE_ENTROPY_THRESHOLD})."
                )

            # ----------------------------------------------------------
            # Rejection check 2: seed instability.
            # ----------------------------------------------------------
            seed_utils: list[float] = [metrics["validation_expected_utility"]]
            for s in range(1, N_STABILITY_SEEDS):
                try:
                    _, seed_metrics = _train_and_evaluate_option(
                        train_features, train_labels, feature_lookup,
                        val_task_ids, run, actions,
                        cw, sw, seed + s,
                        sham_val, baseline_val, oracle_val,
                        f"{option_name}_seed_{s}",
                    )
                    seed_utils.append(seed_metrics["validation_expected_utility"])
                except Exception:
                    seed_utils.append(0.0)

            seed_std = float(np.std(seed_utils))
            metrics["seed_stability"] = {
                "seed_utilities": [float(u) for u in seed_utils],
                "std": seed_std,
                "is_unstable": bool(seed_std > SEED_INSTABILITY_STD_THRESHOLD),
            }
            if seed_std > SEED_INSTABILITY_STD_THRESHOLD:
                rejection_notes.append(
                    f"{option_name}: rejected — seed instability std "
                    f"({seed_std:.4f}) exceeds threshold "
                    f"({SEED_INSTABILITY_STD_THRESHOLD})."
                )

            # ----------------------------------------------------------
            # Rejection check 3: accuracy-utility conflict.
            # ----------------------------------------------------------
            if option_name == "no_class_weighting":
                no_weight_metrics = metrics
            elif no_weight_metrics is not None:
                acc_gain = (
                    metrics["validation_action_accuracy"]
                    - no_weight_metrics["validation_action_accuracy"]
                )
                util_delta = (
                    metrics["validation_expected_utility"]
                    - no_weight_metrics["validation_expected_utility"]
                )
                metrics["accuracy_utility_conflict"] = bool(
                    acc_gain > 0 and util_delta < 0
                )
                if acc_gain > 0 and util_delta < 0:
                    rejection_notes.append(
                        f"{option_name}: rejected — action accuracy improves "
                        f"({acc_gain:+.4f}) while utility decreases "
                        f"({util_delta:+.4f}) relative to no_class_weighting."
                    )
            else:
                metrics["accuracy_utility_conflict"] = False

            options[option_name] = metrics

        except Exception as exc:
            options[option_name] = {
                "error": str(exc),
                "raw_action_distribution": raw_dist,
            }
            rejection_notes.append(
                f"{option_name}: rejected — training or evaluation error: {exc}"
            )

    # ------------------------------------------------------------------
    # Select best option on validation only (excluding rejected options).
    # ------------------------------------------------------------------
    valid_options: dict[str, float] = {}
    for name, data in options.items():
        if "error" in data:
            continue
        if data.get("is_collapsed", False):
            continue
        if data.get("seed_stability", {}).get("is_unstable", False):
            continue
        if data.get("accuracy_utility_conflict", False):
            continue
        valid_options[name] = data["validation_expected_utility"]

    if valid_options:
        best_option = max(valid_options, key=valid_options.get)
        best_val_util = valid_options[best_option]
    else:
        best_option = None
        best_val_util = None

    # ------------------------------------------------------------------
    # Interpretation.
    # ------------------------------------------------------------------
    # Check if the learner collapses toward the majority action.
    majority_action = max(
        raw_dist["counts"], key=raw_dist["counts"].get
    ) if raw_dist["counts"] else None
    majority_fraction = (
        raw_dist["probabilities"].get(majority_action, 0.0)
        if majority_action
        else 0.0
    )

    # Check predicted distribution for no_class_weighting.
    no_weight_pred_entropy = (
        options.get("no_class_weighting", {}).get("validation_action_entropy", 0.0)
    )
    no_weight_collapsed = (
        options.get("no_class_weighting", {}).get("is_collapsed", False)
    )

    if no_weight_collapsed:
        collapse_conclusion = "learner_collapses"
        collapse_note = (
            f"The unweighted learner collapses toward the majority action "
            f"({majority_action}, {majority_fraction:.1%} of training data). "
            f"Predicted action entropy is {no_weight_pred_entropy:.4f} on "
            f"validation.  Class weighting may help."
        )
    else:
        collapse_conclusion = "no_collapse"
        collapse_note = (
            f"The unweighted learner does not collapse toward the majority "
            f"action.  Predicted action entropy is "
            f"{no_weight_pred_entropy:.4f} on validation."
        )

    # Check if any weighting option helps.
    no_weight_util = (
        options.get("no_class_weighting", {}).get("validation_expected_utility", 0.0)
    )
    if best_option and best_option != "no_class_weighting":
        best_gain = best_val_util - no_weight_util if best_val_util is not None else 0.0
        if best_gain > 0.05:
            weighting_conclusion = "weighting_helps"
            weighting_note = (
                f"The best weighting option ({best_option}) materially "
                f"improves validation utility by {best_gain:.3f} over "
                f"no_class_weighting."
            )
        elif best_gain > 0.01:
            weighting_conclusion = "weighting_marginal"
            weighting_note = (
                f"The best weighting option ({best_option}) marginally "
                f"improves validation utility by {best_gain:.3f}."
            )
        else:
            weighting_conclusion = "weighting_not_needed"
            weighting_note = (
                "No weighting option materially improves over "
                "no_class_weighting on validation."
            )
    else:
        best_gain = 0.0
        weighting_conclusion = "weighting_not_needed"
        weighting_note = (
            "No weighting option improves over no_class_weighting on "
            "validation, or all alternatives were rejected."
        )

    return {
        "seed": seed,
        "n_train": len(train_examples),
        "n_validation": len(val_examples),
        "actions": actions,
        "raw_action_distribution": raw_dist,
        "options": options,
        "best_option": best_option,
        "best_validation_utility": float(best_val_util) if best_val_util is not None else None,
        "rejection_notes": rejection_notes,
        "interpretation": {
            "collapse_conclusion": collapse_conclusion,
            "collapse_note": collapse_note,
            "weighting_conclusion": weighting_conclusion,
            "weighting_note": weighting_note,
            "majority_action": majority_action,
            "majority_action_fraction": float(majority_fraction),
            "best_option": best_option,
            "best_gain_over_no_weighting": float(best_gain),
            "n_rejected": len(rejection_notes),
        },
    }
