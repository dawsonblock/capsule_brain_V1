"""Learner-capacity comparison for routing policies.

Compares the current multinomial logistic regression policy against a
panel of controlled diagnostic learners to determine whether the
feature-to-action boundary is nonlinear and whether learner capacity
is the binding constraint.

Learners evaluated:
  1. Multinomial logistic regression (current primary candidate)
  2. Class-weighted logistic regression
  3. Utility-weighted logistic regression (weight examples by utility margin)
  4. Shallow decision tree (max_depth=3)
  5. Small gradient-boosted tree (max_depth=2, n_estimators=50)
  6. Nearest-centroid classifier
  7. Family-conditioned logistic model (family one-hot + features)

All learners use identical splits: train on D_experience, select
hyperparameters on D_validation, then report final held-out test
utility on D_test.  The best learner is selected on D_validation
only — D_test is never used for model selection.

Evaluation protocol for a trained model on test tasks:
  1. Get test example features from ``run.test_examples``.
  2. Apply the model to predict an action.
  3. Look up measured utility for (task, action) from
     ``run.outcomes_by_task_action``.

Interpretation:
  - If a nonlinear shallow model materially beats logistic regression,
    the feature-to-action boundary is nonlinear.
  - If no learner beats frequency controls, the problem is likely
    headroom or features rather than learner capacity.

The output of :func:`compare_learners` is the content for
``learner_comparison.json``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestCentroid

from .data import LoadedRun
from .sham_trainer import train_logistic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_features_labels(
    examples: list[dict],
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


def _build_family_one_hot(
    examples: list[dict],
    all_families: list[str],
) -> np.ndarray:
    """Build a family one-hot encoding matrix [n, n_families]."""
    fam_idx = {f: i for i, f in enumerate(all_families)}
    n = len(examples)
    one_hot = np.zeros((n, len(all_families)), dtype=float)
    for i, ex in enumerate(examples):
        fam = ex.get("task_family", "unknown")
        if fam in fam_idx:
            one_hot[i, fam_idx[fam]] = 1.0
    return one_hot


def _compute_utility_weights(
    run: LoadedRun,
    examples: list[dict],
) -> np.ndarray:
    """Compute per-example utility-margin weights.

    Weight = max(0, utility_margin) + 1.0  (so every example gets at
    least weight 1.0, and clear-best examples get more).
    """
    weights = np.ones(len(examples), dtype=float)
    for i, ex in enumerate(examples):
        margin = ex.get("utility_margin", 0.0)
        weights[i] = max(0.0, margin) + 1.0
    return weights


def _evaluate_model_utility(
    model: Any,
    test_features: np.ndarray,
    test_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
) -> dict[str, Any]:
    """Evaluate a trained sklearn model on test tasks.

    Applies the model to predict actions, then looks up measured
    utility for each (task, action) pair from
    ``run.outcomes_by_task_action``.

    Returns a dict with mean_utility, per-task utilities, and action
    distribution.
    """
    predictions = model.predict(test_features)
    pred_actions = [actions[int(p)] for p in predictions]

    utilities: list[float] = []
    per_task: list[dict[str, Any]] = []
    action_dist: dict[str, int] = {a: 0 for a in actions}

    for tid, pred_action in zip(test_task_ids, pred_actions):
        task = run.tasks.get(tid)
        if task is None:
            utilities.append(0.0)
            continue

        # Respect allowed_actions constraint.
        if pred_action not in task.allowed_actions:
            eligible = [a for a in task.allowed_actions if a in actions]
            if eligible:
                pred_action = eligible[0]
            else:
                pred_action = task.allowed_actions[0] if task.allowed_actions else pred_action

        u = run.get_utility_for_action(tid, pred_action)
        if u is None:
            u = 0.0

        utilities.append(u)
        action_dist[pred_action] = action_dist.get(pred_action, 0) + 1
        per_task.append({
            "task_id": tid,
            "predicted_action": pred_action,
            "utility": u,
            "oracle_utility": run.get_oracle_utility(tid),
            "family": task.family,
        })

    utilities_arr = np.array(utilities, dtype=float)
    n = len(utilities_arr)

    return {
        "mean_utility": float(utilities_arr.mean()) if n > 0 else 0.0,
        "median_utility": float(np.median(utilities_arr)) if n > 0 else 0.0,
        "utility_std": float(utilities_arr.std()) if n > 0 else 0.0,
        "n_tasks": n,
        "action_distribution": action_dist,
        "utilities": utilities_arr.tolist(),
        "per_task": per_task,
    }


def _evaluate_logistic_utility(
    weights: np.ndarray,
    bias: np.ndarray,
    test_features: np.ndarray,
    test_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
) -> dict[str, Any]:
    """Evaluate a custom logistic model (weights + bias) on test tasks."""
    logits = test_features @ weights + bias
    predictions = np.argmax(logits, axis=1)
    pred_actions = [actions[int(p)] for p in predictions]

    utilities: list[float] = []
    per_task: list[dict[str, Any]] = []
    action_dist: dict[str, int] = {a: 0 for a in actions}

    for tid, pred_action in zip(test_task_ids, pred_actions):
        task = run.tasks.get(tid)
        if task is None:
            utilities.append(0.0)
            continue

        if pred_action not in task.allowed_actions:
            eligible = [a for a in task.allowed_actions if a in actions]
            if eligible:
                pred_action = eligible[0]
            else:
                pred_action = task.allowed_actions[0] if task.allowed_actions else pred_action

        u = run.get_utility_for_action(tid, pred_action)
        if u is None:
            u = 0.0

        utilities.append(u)
        action_dist[pred_action] = action_dist.get(pred_action, 0) + 1
        per_task.append({
            "task_id": tid,
            "predicted_action": pred_action,
            "utility": u,
            "oracle_utility": run.get_oracle_utility(tid),
            "family": task.family,
        })

    utilities_arr = np.array(utilities, dtype=float)
    n = len(utilities_arr)

    return {
        "mean_utility": float(utilities_arr.mean()) if n > 0 else 0.0,
        "median_utility": float(np.median(utilities_arr)) if n > 0 else 0.0,
        "utility_std": float(utilities_arr.std()) if n > 0 else 0.0,
        "n_tasks": n,
        "action_distribution": action_dist,
        "utilities": utilities_arr.tolist(),
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Individual learner trainers
# ---------------------------------------------------------------------------


def _train_multinomial_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Train multinomial logistic regression via sklearn."""
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(train_features, train_labels)
    val_util = _evaluate_model_utility(model, val_features, val_task_ids, run, actions)
    return model, {"val_mean_utility": val_util["mean_utility"], "n_train": len(train_labels)}


def _train_class_weighted_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Train class-weighted logistic regression (balanced)."""
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(train_features, train_labels)
    val_util = _evaluate_model_utility(model, val_features, val_task_ids, run, actions)
    return model, {"val_mean_utility": val_util["mean_utility"], "n_train": len(train_labels)}


def _train_utility_weighted_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
    seed: int,
    train_examples: list[dict],
) -> tuple[Any, dict[str, Any]]:
    """Train utility-weighted logistic regression.

    Each example is weighted by its utility margin so that clear-best
    examples have more influence on the decision boundary.
    """
    sample_weights = _compute_utility_weights(run, train_examples)
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(train_features, train_labels, sample_weight=sample_weights)
    val_util = _evaluate_model_utility(model, val_features, val_task_ids, run, actions)
    return model, {
        "val_mean_utility": val_util["mean_utility"],
        "n_train": len(train_labels),
        "mean_sample_weight": float(sample_weights.mean()),
    }


def _train_decision_tree(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Train a shallow decision tree (max_depth=3)."""
    model = DecisionTreeClassifier(max_depth=3, random_state=seed)
    model.fit(train_features, train_labels)
    val_util = _evaluate_model_utility(model, val_features, val_task_ids, run, actions)
    return model, {"val_mean_utility": val_util["mean_utility"], "n_train": len(train_labels)}


def _train_gradient_boosted_tree(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Train a small gradient-boosted tree (max_depth=2, n_estimators=50)."""
    model = GradientBoostingClassifier(
        max_depth=2,
        n_estimators=50,
        random_state=seed,
    )
    model.fit(train_features, train_labels)
    val_util = _evaluate_model_utility(model, val_features, val_task_ids, run, actions)
    return model, {"val_mean_utility": val_util["mean_utility"], "n_train": len(train_labels)}


def _train_nearest_centroid(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Train a nearest-centroid classifier."""
    model = NearestCentroid()
    model.fit(train_features, train_labels)
    val_util = _evaluate_model_utility(model, val_features, val_task_ids, run, actions)
    return model, {"val_mean_utility": val_util["mean_utility"], "n_train": len(train_labels)}


def _train_family_conditioned_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
    seed: int,
    train_examples: list[dict],
    val_examples: list[dict],
    all_families: list[str],
) -> tuple[Any, dict[str, Any]]:
    """Train family-conditioned logistic model (family one-hot + features)."""
    train_fam = _build_family_one_hot(train_examples, all_families)
    val_fam = _build_family_one_hot(val_examples, all_families)
    train_aug = np.hstack([train_features, train_fam])
    val_aug = np.hstack([val_features, val_fam])

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(train_aug, train_labels)
    val_util = _evaluate_model_utility(model, val_aug, val_task_ids, run, actions)
    return model, {
        "val_mean_utility": val_util["mean_utility"],
        "n_train": len(train_labels),
        "n_features_original": train_features.shape[1],
        "n_features_family": train_fam.shape[1],
        "n_features_total": train_aug.shape[1],
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compare_learners(run: LoadedRun, seed: int = 42) -> dict[str, Any]:
    """Compare learner capacity across a panel of diagnostic classifiers.

    Trains each learner on D_experience, selects on D_validation, and
    reports final held-out test utility on D_test.  The best learner
    is chosen by validation utility only — D_test is never used for
    selection.

    Args:
        run: loaded run data.
        seed: random seed for reproducibility.

    Returns:
        Dict with per-learner validation/test utilities, the selected
        best learner, and an interpretation of the results.  Suitable
        for writing to ``learner_comparison.json``.
    """
    actions = run.all_actions
    all_families = run.all_families

    # ------------------------------------------------------------------
    # Prepare data splits.
    # ------------------------------------------------------------------
    train_examples = run.train_examples
    val_examples = run.validation_examples
    test_examples = run.test_examples

    if not train_examples or not test_examples:
        return {
            "error": "Missing training or test examples.",
            "n_train": len(train_examples),
            "n_test": len(test_examples),
        }

    train_features, train_labels, _ = _extract_features_labels(train_examples, actions)
    val_features, _, val_task_ids = _extract_features_labels(val_examples, actions)
    test_features, _, test_task_ids = _extract_features_labels(test_examples, actions)

    # ------------------------------------------------------------------
    # Train and evaluate each learner.
    # ------------------------------------------------------------------
    learners: dict[str, dict[str, Any]] = {}

    # 1. Multinomial logistic regression (current primary candidate).
    model, meta = _train_multinomial_logistic(
        train_features, train_labels, val_features, val_task_ids,
        run, actions, seed,
    )
    test_util = _evaluate_model_utility(model, test_features, test_task_ids, run, actions)
    learners["multinomial_logistic"] = {
        "description": "Multinomial logistic regression (current primary candidate).",
        "is_primary": True,
        "validation": meta,
        "test": test_util,
    }

    # 2. Class-weighted logistic regression.
    model, meta = _train_class_weighted_logistic(
        train_features, train_labels, val_features, val_task_ids,
        run, actions, seed,
    )
    test_util = _evaluate_model_utility(model, test_features, test_task_ids, run, actions)
    learners["class_weighted_logistic"] = {
        "description": "Class-weighted logistic regression (balanced).",
        "validation": meta,
        "test": test_util,
    }

    # 3. Utility-weighted logistic regression.
    model, meta = _train_utility_weighted_logistic(
        train_features, train_labels, val_features, val_task_ids,
        run, actions, seed, train_examples,
    )
    test_util = _evaluate_model_utility(model, test_features, test_task_ids, run, actions)
    learners["utility_weighted_logistic"] = {
        "description": "Utility-weighted logistic regression (weight by utility margin).",
        "validation": meta,
        "test": test_util,
    }

    # 4. Shallow decision tree.
    model, meta = _train_decision_tree(
        train_features, train_labels, val_features, val_task_ids,
        run, actions, seed,
    )
    test_util = _evaluate_model_utility(model, test_features, test_task_ids, run, actions)
    learners["shallow_decision_tree"] = {
        "description": "Shallow decision tree (max_depth=3).",
        "validation": meta,
        "test": test_util,
    }

    # 5. Small gradient-boosted tree.
    model, meta = _train_gradient_boosted_tree(
        train_features, train_labels, val_features, val_task_ids,
        run, actions, seed,
    )
    test_util = _evaluate_model_utility(model, test_features, test_task_ids, run, actions)
    learners["gradient_boosted_tree"] = {
        "description": "Small gradient-boosted tree (max_depth=2, n_estimators=50).",
        "validation": meta,
        "test": test_util,
    }

    # 6. Nearest-centroid classifier.
    model, meta = _train_nearest_centroid(
        train_features, train_labels, val_features, val_task_ids,
        run, actions, seed,
    )
    test_util = _evaluate_model_utility(model, test_features, test_task_ids, run, actions)
    learners["nearest_centroid"] = {
        "description": "Nearest-centroid classifier.",
        "validation": meta,
        "test": test_util,
    }

    # 7. Family-conditioned logistic model.
    model, meta = _train_family_conditioned_logistic(
        train_features, train_labels, val_features, val_task_ids,
        run, actions, seed, train_examples, val_examples, all_families,
    )
    # For test evaluation, augment test features with family one-hot.
    test_fam = _build_family_one_hot(test_examples, all_families)
    test_aug = np.hstack([test_features, test_fam])
    test_util = _evaluate_model_utility(model, test_aug, test_task_ids, run, actions)
    learners["family_conditioned_logistic"] = {
        "description": "Family-conditioned logistic model (family one-hot + features).",
        "validation": meta,
        "test": test_util,
    }

    # ------------------------------------------------------------------
    # Also evaluate the current logistic policy from sham_trainer for
    # architectural parity reference.
    # ------------------------------------------------------------------
    W, b = train_logistic(
        train_features,
        np.array([actions[i] for i in train_labels]),
        actions,
        seed=seed,
    )
    custom_val = _evaluate_logistic_utility(
        W, b, val_features, val_task_ids, run, actions,
    )
    custom_test = _evaluate_logistic_utility(
        W, b, test_features, test_task_ids, run, actions,
    )
    learners["custom_logistic_reference"] = {
        "description": "Custom logistic regression from sham_trainer (architectural parity).",
        "validation": {"val_mean_utility": custom_val["mean_utility"], "n_train": len(train_labels)},
        "test": custom_test,
    }

    # ------------------------------------------------------------------
    # Select best learner on validation only.
    # ------------------------------------------------------------------
    val_utils = {
        name: data["validation"]["val_mean_utility"]
        for name, data in learners.items()
    }
    best_learner = max(val_utils, key=val_utils.get)
    best_val_util = val_utils[best_learner]

    # Primary candidate (multinomial logistic) for comparison.
    primary_val = val_utils.get("multinomial_logistic", 0.0)
    primary_test = learners["multinomial_logistic"]["test"]["mean_utility"]
    best_test = learners[best_learner]["test"]["mean_utility"]

    # Nonlinear learners.
    nonlinear_names = ["shallow_decision_tree", "gradient_boosted_tree", "nearest_centroid"]
    best_nonlinear_val = max(
        (val_utils[n] for n in nonlinear_names if n in val_utils),
        default=0.0,
    )
    best_nonlinear_name = max(
        ((n, val_utils[n]) for n in nonlinear_names if n in val_utils),
        key=lambda x: x[1],
        default=(None, 0.0),
    )[0]
    best_nonlinear_test = (
        learners[best_nonlinear_name]["test"]["mean_utility"]
        if best_nonlinear_name
        else 0.0
    )

    # ------------------------------------------------------------------
    # Interpretation.
    # ------------------------------------------------------------------
    nonlinear_gain = best_nonlinear_test - primary_test
    if nonlinear_gain > 0.05:
        boundary_interpretation = "nonlinear"
        boundary_note = (
            f"A nonlinear shallow model ({best_nonlinear_name}) materially "
            f"beats logistic regression by {nonlinear_gain:.3f} utility on "
            f"test.  The feature-to-action boundary is likely nonlinear."
        )
    elif nonlinear_gain > 0.01:
        boundary_interpretation = "possibly_nonlinear"
        boundary_note = (
            f"A nonlinear shallow model ({best_nonlinear_name}) slightly "
            f"beats logistic regression by {nonlinear_gain:.3f}.  There may "
            f"be mild nonlinearity in the feature-to-action boundary."
        )
    else:
        boundary_interpretation = "linear"
        boundary_note = (
            "No nonlinear shallow model materially beats logistic "
            "regression.  The feature-to-action boundary appears "
            "approximately linear."
        )

    # Check if any learner beats a simple frequency baseline.
    # (Using majority action utility as a proxy.)
    from collections import Counter
    best_action_counts = Counter(ex["best_action"] for ex in train_examples)
    majority_action = best_action_counts.most_common(1)[0][0] if best_action_counts else actions[0]
    freq_test_utils = []
    for tid in test_task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        action = majority_action if majority_action in task.allowed_actions else (
            task.allowed_actions[0] if task.allowed_actions else majority_action
        )
        u = run.get_utility_for_action(tid, action)
        freq_test_utils.append(u if u is not None else 0.0)
    freq_test_util = float(np.mean(freq_test_utils)) if freq_test_utils else 0.0

    if best_test <= freq_test_util + 0.01:
        capacity_interpretation = "not_learner_capacity"
        capacity_note = (
            "No learner beats the frequency/majority control on test.  "
            "The problem is likely headroom or features rather than "
            "learner capacity."
        )
    else:
        capacity_interpretation = "learner_capacity_matters"
        capacity_note = (
            f"The best learner ({best_learner}) beats the frequency "
            f"control by {best_test - freq_test_util:.3f} on test.  "
            f"Learner capacity contributes to routing performance."
        )

    return {
        "seed": seed,
        "n_train": len(train_examples),
        "n_validation": len(val_examples),
        "n_test": len(test_examples),
        "actions": actions,
        "learners": learners,
        "selection": {
            "method": "Best validation utility (D_test not used for selection).",
            "best_learner": best_learner,
            "best_val_utility": float(best_val_util),
            "best_test_utility": float(best_test),
            "primary_learner": "multinomial_logistic",
            "primary_val_utility": float(primary_val),
            "primary_test_utility": float(primary_test),
        },
        "frequency_baseline": {
            "majority_action": majority_action,
            "test_mean_utility": freq_test_util,
        },
        "interpretation": {
            "boundary_type": boundary_interpretation,
            "boundary_note": boundary_note,
            "capacity_conclusion": capacity_interpretation,
            "capacity_note": capacity_note,
            "nonlinear_gain_over_logistic": float(nonlinear_gain),
            "best_nonlinear_learner": best_nonlinear_name,
            "best_nonlinear_test_utility": float(best_nonlinear_test),
        },
    }
