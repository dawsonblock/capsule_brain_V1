"""Learner validation for the v0.4.3 repair release.

Keeps multinomial logistic regression as the primary production learner
and adds a panel of validation-only diagnostic learners to determine
whether a different learner architecture would materially improve
routing utility on held-out validation tasks.

Learners evaluated (all trained on D_experience, evaluated on D_validation):

  1. Multinomial logistic regression (primary candidate)
  2. Class-weighted multinomial logistic regression
  3. Utility-weighted logistic regression (weight samples by utility margin)
  4. Soft-utility-target logistic regression (soft targets from utilities)
  5. Shallow decision tree (max_depth=3)
  6. Small gradient-boosted tree (max_depth=2, n_estimators=50)
  7. Nearest-centroid classifier
  8. Family-conditioned logistic model (one-hot family + features)

The best learner is selected on D_validation only.  D_test is never
used for model selection.  Test results for the selected learner are
reported under an explicit EXPLORATORY label for diagnostic purposes
only.

The output of :func:`validate_learners` is a JSON-serializable dict
suitable for writing to ``learner_validation_results.json``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestCentroid

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
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


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


def _build_family_one_hot(
    examples: list[dict[str, Any]],
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
    examples: list[dict[str, Any]],
) -> np.ndarray:
    """Compute per-example utility-margin weights.

    Weight = max(0, utility_margin) + 1.0 so that clear-best examples
    get more influence while every example receives at least weight 1.0.
    """
    weights = np.ones(len(examples), dtype=float)
    for i, ex in enumerate(examples):
        margin = ex.get("utility_margin", 0.0)
        weights[i] = max(0.0, float(margin)) + 1.0
    return weights


def _build_utility_matrix(
    run: LoadedRun,
    task_ids: list[str],
    actions: list[str],
) -> np.ndarray:
    """Build a utility matrix [n_tasks, n_actions] from measured outcomes.

    Unavailable outcomes are filled with 0.0 (neutral for softmax).
    """
    action_idx = {a: i for i, a in enumerate(actions)}
    U = np.zeros((len(task_ids), len(actions)), dtype=float)
    for i, tid in enumerate(task_ids):
        for outcome in run.outcomes_by_task.get(tid, []):
            if (
                outcome.action_id in action_idx
                and outcome.availability == "executed"
                and outcome.utility is not None
            ):
                U[i, action_idx[outcome.action_id]] = outcome.utility
    return U


def _build_feature_lookup(
    run: LoadedRun,
) -> dict[str, np.ndarray]:
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


def _build_family_feature_lookup(
    run: LoadedRun,
    all_families: list[str],
) -> dict[str, np.ndarray]:
    """Build a task_id -> augmented features lookup (features + family one-hot)."""
    fam_idx = {f: i for i, f in enumerate(all_families)}
    lookup: dict[str, np.ndarray] = {}
    for ex in (
        run.train_examples
        + run.validation_examples
        + run.test_examples
        + run.ood_examples
        + run.safety_examples
    ):
        tid = ex.get("task_id", "")
        if not tid:
            continue
        feats = np.array(ex.get("features", []), dtype=float)
        fam = ex.get("task_family", "unknown")
        fam_one_hot = np.zeros(len(all_families), dtype=float)
        if fam in fam_idx:
            fam_one_hot[fam_idx[fam]] = 1.0
        lookup[tid] = np.concatenate([feats, fam_one_hot])
    return lookup


def _make_policy_fn(
    predict_fn: Callable[[np.ndarray], int],
    feature_lookup: dict[str, np.ndarray],
    actions: list[str],
) -> PolicyDecisionFn:
    """Create a PolicyDecisionFn from a predict function and feature lookup.

    The predict_fn takes a 1-D feature vector and returns an action index.
    The resulting policy respects allowed_actions constraints.
    """

    def fn(tid: str, task: TaskInfo) -> str | None:
        feats = feature_lookup.get(tid)
        if feats is None or feats.size == 0:
            return task.allowed_actions[0] if task.allowed_actions else None
        pred_idx = int(predict_fn(feats))
        if pred_idx < 0 or pred_idx >= len(actions):
            pred_idx = 0
        pred_action = actions[pred_idx]
        if pred_action in task.allowed_actions:
            return pred_action
        eligible = [a for a in task.allowed_actions if a in actions]
        if eligible:
            return eligible[0]
        return task.allowed_actions[0] if task.allowed_actions else None

    return fn


def _compute_action_entropy(action_dist: dict[str, int], n: int) -> float:
    """Compute the entropy of an action distribution (natural log)."""
    if n == 0:
        return 0.0
    probs = np.array([c / n for c in action_dist.values()], dtype=float)
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log(nonzero))) if len(nonzero) > 0 else 0.0


def _compute_route_collapse(action_dist: dict[str, int], n: int) -> dict[str, Any]:
    """Compute route-collapse metrics from an action distribution."""
    if n == 0:
        return {
            "n_unique_actions_predicted": 0,
            "predicts_single_action": False,
            "predicted_action_entropy": 0.0,
            "is_collapsed": False,
            "max_action_fraction": 0.0,
        }
    entropy = _compute_action_entropy(action_dist, n)
    n_unique = len(action_dist)
    max_count = max(action_dist.values()) if action_dist else 0
    max_fraction = max_count / n
    return {
        "n_unique_actions_predicted": n_unique,
        "predicts_single_action": n_unique == 1,
        "predicted_action_entropy": float(entropy),
        "is_collapsed": bool(entropy < 0.1),
        "max_action_fraction": float(max_fraction),
    }


def _compute_action_accuracy(
    task_results: list[Any],
) -> float:
    """Compute the fraction of tasks where predicted action matches oracle."""
    if not task_results:
        return 0.0
    correct = sum(
        1 for r in task_results if r.selected_action is not None and r.selected_action == r.oracle_action
    )
    return float(correct / len(task_results))


# ---------------------------------------------------------------------------
# Custom soft-target logistic trainer
# ---------------------------------------------------------------------------


def _train_soft_target_logistic(
    features: np.ndarray,
    soft_targets: np.ndarray,
    n_actions: int,
    n_epochs: int = 200,
    lr: float = 0.01,
    l2: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Train multinomial logistic regression with soft targets via GD.

    Mirrors the gradient-descent approach from v042 ``sham_trainer.train_logistic``
    but replaces one-hot targets with arbitrary soft target distributions.

    Returns (weights [n_features, n_actions], bias [n_actions]).
    """
    rng = np.random.default_rng(seed)
    n, d = features.shape
    W = rng.standard_normal((d, n_actions)) * 0.01
    b = np.zeros(n_actions)

    for _ in range(n_epochs):
        logits = features @ W + b
        probs = _softmax(logits)
        diff = probs - soft_targets
        grad_W = features.T @ diff / n + l2 * W
        grad_b = diff.mean(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b

    return W, b


# ---------------------------------------------------------------------------
# Learner trainers
# ---------------------------------------------------------------------------


def _train_multinomial_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
) -> tuple[Any, Callable[[np.ndarray], int]]:
    """Train multinomial logistic regression via sklearn."""
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(train_features, train_labels)
    return model, lambda feats: int(model.predict(feats.reshape(1, -1))[0])


def _train_class_weighted_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
) -> tuple[Any, Callable[[np.ndarray], int]]:
    """Train class-weighted logistic regression (balanced)."""
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(train_features, train_labels)
    return model, lambda feats: int(model.predict(feats.reshape(1, -1))[0])


def _train_utility_weighted_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    sample_weights: np.ndarray,
    seed: int,
) -> tuple[Any, Callable[[np.ndarray], int]]:
    """Train utility-weighted logistic regression (sample weights by margin)."""
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(train_features, train_labels, sample_weight=sample_weights)
    return model, lambda feats: int(model.predict(feats.reshape(1, -1))[0])


def _train_soft_utility_logistic(
    train_features: np.ndarray,
    utility_matrix: np.ndarray,
    n_actions: int,
    tau: float,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], Callable[[np.ndarray], int]]:
    """Train soft-utility-target logistic regression via custom GD."""
    scaled_u = utility_matrix / tau
    soft_targets = _softmax(scaled_u)
    W, b = _train_soft_target_logistic(
        train_features, soft_targets, n_actions, seed=seed,
    )
    return (W, b), lambda feats: int(np.argmax(feats @ W + b))


def _train_decision_tree(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
) -> tuple[Any, Callable[[np.ndarray], int]]:
    """Train a shallow decision tree (max_depth=3)."""
    model = DecisionTreeClassifier(max_depth=3, random_state=seed)
    model.fit(train_features, train_labels)
    return model, lambda feats: int(model.predict(feats.reshape(1, -1))[0])


def _train_gradient_boosted_tree(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
) -> tuple[Any, Callable[[np.ndarray], int]]:
    """Train a small gradient-boosted tree (max_depth=2, n_estimators=50)."""
    model = GradientBoostingClassifier(
        max_depth=2,
        n_estimators=50,
        random_state=seed,
    )
    model.fit(train_features, train_labels)
    return model, lambda feats: int(model.predict(feats.reshape(1, -1))[0])


def _train_nearest_centroid(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
) -> tuple[Any, Callable[[np.ndarray], int]]:
    """Train a nearest-centroid classifier."""
    model = NearestCentroid()
    model.fit(train_features, train_labels)
    return model, lambda feats: int(model.predict(feats.reshape(1, -1))[0])


def _train_family_conditioned_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_fam_one_hot: np.ndarray,
    seed: int,
) -> tuple[Any, Callable[[np.ndarray], int]]:
    """Train family-conditioned logistic model (family one-hot + features)."""
    train_aug = np.hstack([train_features, train_fam_one_hot])
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(train_aug, train_labels)
    return model, lambda feats: int(model.predict(feats.reshape(1, -1))[0])


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_learner_metrics(
    learner_result: Any,
    sham_result: Any,
    baseline_result: Any,
    oracle_result: Any,
    actions: list[str],
    val_task_ids: list[str],
    task_ids_digest: str,
) -> dict[str, Any]:
    """Compute the full metric suite for a learner on validation."""
    mean_utility = learner_result.mean_utility
    sham_mean = sham_result.mean_utility
    baseline_mean = baseline_result.mean_utility
    oracle_mean = oracle_result.mean_utility

    # Headroom recovered vs sham.
    headroom_vs_sham = compute_headroom_metrics(
        candidate_mean=mean_utility,
        reference_mean=sham_mean,
        oracle_mean=oracle_mean,
        n_tasks=learner_result.n_complete,
        task_ids_digest=task_ids_digest,
        reference_policy_id="sham",
        candidate_policy_id=learner_result.policy_id,
    )

    # Headroom recovered vs baseline.
    headroom_vs_baseline = compute_headroom_metrics(
        candidate_mean=mean_utility,
        reference_mean=baseline_mean,
        oracle_mean=oracle_mean,
        n_tasks=learner_result.n_complete,
        task_ids_digest=task_ids_digest,
        reference_policy_id="baseline",
        candidate_policy_id=learner_result.policy_id,
    )

    # Action accuracy and entropy.
    action_accuracy = _compute_action_accuracy(learner_result.task_results)
    action_entropy = _compute_action_entropy(
        learner_result.action_distribution, learner_result.n_tasks,
    )
    route_collapse = _compute_route_collapse(
        learner_result.action_distribution, learner_result.n_tasks,
    )

    return {
        "validation_expected_utility": float(mean_utility),
        "validation_candidate_minus_sham": float(mean_utility - sham_mean),
        "validation_candidate_minus_baseline": float(mean_utility - baseline_mean),
        "validation_regret": float(learner_result.mean_regret) if learner_result.mean_regret is not None else None,
        "validation_headroom_recovered_vs_sham": headroom_vs_sham,
        "validation_headroom_recovered_vs_baseline": headroom_vs_baseline,
        "validation_action_accuracy": float(action_accuracy),
        "validation_action_entropy": float(action_entropy),
        "route_collapse_metrics": route_collapse,
        "validation_action_distribution": dict(learner_result.action_distribution),
        "n_validation_tasks": learner_result.n_tasks,
        "n_validation_complete": learner_result.n_complete,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate_learners(run: LoadedRun, seed: int = 42) -> dict[str, Any]:
    """Validate a panel of diagnostic learners on held-out validation data.

    Trains eight learners on D_experience, evaluates each on D_validation,
    and selects the best learner by validation expected utility only.
    D_test is never used for selection; test results for the selected
    learner are reported under an explicit EXPLORATORY label.

    Args:
        run: loaded run data.
        seed: random seed for reproducibility.

    Returns:
        JSON-serializable dict with per-learner validation metrics,
        the best validation learner, exploratory test results, and
        an interpretation.  Suitable for writing to
        ``learner_validation_results.json``.
    """
    actions = run.all_actions
    all_families = run.all_families
    n_actions = len(actions)

    # ------------------------------------------------------------------
    # Prepare data splits.
    # ------------------------------------------------------------------
    train_examples = run.train_examples
    val_examples = run.validation_examples
    test_examples = run.test_examples

    if not train_examples or not val_examples:
        return {
            "error": "Missing training or validation examples.",
            "n_train": len(train_examples),
            "n_validation": len(val_examples),
            "seed": seed,
        }

    train_features, train_labels, train_task_ids = _extract_features_labels(
        train_examples, actions,
    )
    val_features, _, val_task_ids = _extract_features_labels(
        val_examples, actions,
    )
    test_features, _, test_task_ids = _extract_features_labels(
        test_examples, actions,
    )

    # Feature lookups for policy decision functions.
    feature_lookup = _build_feature_lookup(run)
    family_feature_lookup = _build_family_feature_lookup(run, all_families)

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

    val_digest = oracle_val.task_ids_digest

    # ------------------------------------------------------------------
    # Pre-compute shared data for learners.
    # ------------------------------------------------------------------
    utility_weights = _compute_utility_weights(train_examples)
    train_utility_matrix = _build_utility_matrix(run, train_task_ids, actions)
    train_fam_one_hot = _build_family_one_hot(train_examples, all_families)

    # ------------------------------------------------------------------
    # Train and evaluate each learner.
    # ------------------------------------------------------------------
    learners: dict[str, dict[str, Any]] = {}

    # 1. Multinomial logistic regression (primary candidate).
    try:
        model, predict_fn = _train_multinomial_logistic(
            train_features, train_labels, seed,
        )
        policy_fn = _make_policy_fn(predict_fn, feature_lookup, actions)
        result = evaluate_policy_from_counterfactuals(
            run, val_task_ids, policy_fn, "multinomial_logistic",
            reference_result=sham_val, oracle_result=oracle_val,
        )
        learners["multinomial_logistic"] = {
            "description": "Multinomial logistic regression (primary candidate).",
            "is_primary": True,
            **_compute_learner_metrics(
                result, sham_val, baseline_val, oracle_val,
                actions, val_task_ids, val_digest,
            ),
        }
    except Exception as exc:
        learners["multinomial_logistic"] = {
            "description": "Multinomial logistic regression (primary candidate).",
            "is_primary": True,
            "error": str(exc),
        }

    # 2. Class-weighted logistic regression.
    try:
        model, predict_fn = _train_class_weighted_logistic(
            train_features, train_labels, seed,
        )
        policy_fn = _make_policy_fn(predict_fn, feature_lookup, actions)
        result = evaluate_policy_from_counterfactuals(
            run, val_task_ids, policy_fn, "class_weighted_logistic",
            reference_result=sham_val, oracle_result=oracle_val,
        )
        learners["class_weighted_logistic"] = {
            "description": "Class-weighted logistic regression (balanced).",
            **_compute_learner_metrics(
                result, sham_val, baseline_val, oracle_val,
                actions, val_task_ids, val_digest,
            ),
        }
    except Exception as exc:
        learners["class_weighted_logistic"] = {
            "description": "Class-weighted logistic regression (balanced).",
            "error": str(exc),
        }

    # 3. Utility-weighted logistic regression.
    try:
        model, predict_fn = _train_utility_weighted_logistic(
            train_features, train_labels, utility_weights, seed,
        )
        policy_fn = _make_policy_fn(predict_fn, feature_lookup, actions)
        result = evaluate_policy_from_counterfactuals(
            run, val_task_ids, policy_fn, "utility_weighted_logistic",
            reference_result=sham_val, oracle_result=oracle_val,
        )
        learners["utility_weighted_logistic"] = {
            "description": "Utility-weighted logistic regression (weight by utility margin).",
            "mean_sample_weight": float(utility_weights.mean()),
            **_compute_learner_metrics(
                result, sham_val, baseline_val, oracle_val,
                actions, val_task_ids, val_digest,
            ),
        }
    except Exception as exc:
        learners["utility_weighted_logistic"] = {
            "description": "Utility-weighted logistic regression (weight by utility margin).",
            "error": str(exc),
        }

    # 4. Soft-utility-target logistic regression.
    try:
        soft_tau = 0.1
        (W, b), predict_fn = _train_soft_utility_logistic(
            train_features, train_utility_matrix, n_actions,
            tau=soft_tau, seed=seed,
        )
        policy_fn = _make_policy_fn(predict_fn, feature_lookup, actions)
        result = evaluate_policy_from_counterfactuals(
            run, val_task_ids, policy_fn, "soft_utility_logistic",
            reference_result=sham_val, oracle_result=oracle_val,
        )
        learners["soft_utility_logistic"] = {
            "description": "Soft-utility-target logistic regression (soft targets from utilities).",
            "tau": soft_tau,
            **_compute_learner_metrics(
                result, sham_val, baseline_val, oracle_val,
                actions, val_task_ids, val_digest,
            ),
        }
    except Exception as exc:
        learners["soft_utility_logistic"] = {
            "description": "Soft-utility-target logistic regression (soft targets from utilities).",
            "error": str(exc),
        }

    # 5. Shallow decision tree.
    try:
        model, predict_fn = _train_decision_tree(
            train_features, train_labels, seed,
        )
        policy_fn = _make_policy_fn(predict_fn, feature_lookup, actions)
        result = evaluate_policy_from_counterfactuals(
            run, val_task_ids, policy_fn, "shallow_decision_tree",
            reference_result=sham_val, oracle_result=oracle_val,
        )
        learners["shallow_decision_tree"] = {
            "description": "Shallow decision tree (max_depth=3).",
            **_compute_learner_metrics(
                result, sham_val, baseline_val, oracle_val,
                actions, val_task_ids, val_digest,
            ),
        }
    except Exception as exc:
        learners["shallow_decision_tree"] = {
            "description": "Shallow decision tree (max_depth=3).",
            "error": str(exc),
        }

    # 6. Small gradient-boosted tree.
    try:
        model, predict_fn = _train_gradient_boosted_tree(
            train_features, train_labels, seed,
        )
        policy_fn = _make_policy_fn(predict_fn, feature_lookup, actions)
        result = evaluate_policy_from_counterfactuals(
            run, val_task_ids, policy_fn, "gradient_boosted_tree",
            reference_result=sham_val, oracle_result=oracle_val,
        )
        learners["gradient_boosted_tree"] = {
            "description": "Small gradient-boosted tree (max_depth=2, n_estimators=50).",
            **_compute_learner_metrics(
                result, sham_val, baseline_val, oracle_val,
                actions, val_task_ids, val_digest,
            ),
        }
    except Exception as exc:
        learners["gradient_boosted_tree"] = {
            "description": "Small gradient-boosted tree (max_depth=2, n_estimators=50).",
            "error": str(exc),
        }

    # 7. Nearest-centroid classifier.
    try:
        model, predict_fn = _train_nearest_centroid(
            train_features, train_labels, seed,
        )
        policy_fn = _make_policy_fn(predict_fn, feature_lookup, actions)
        result = evaluate_policy_from_counterfactuals(
            run, val_task_ids, policy_fn, "nearest_centroid",
            reference_result=sham_val, oracle_result=oracle_val,
        )
        learners["nearest_centroid"] = {
            "description": "Nearest-centroid classifier.",
            **_compute_learner_metrics(
                result, sham_val, baseline_val, oracle_val,
                actions, val_task_ids, val_digest,
            ),
        }
    except Exception as exc:
        learners["nearest_centroid"] = {
            "description": "Nearest-centroid classifier.",
            "error": str(exc),
        }

    # 8. Family-conditioned logistic model.
    try:
        model, predict_fn = _train_family_conditioned_logistic(
            train_features, train_labels, train_fam_one_hot, seed,
        )
        policy_fn = _make_policy_fn(predict_fn, family_feature_lookup, actions)
        result = evaluate_policy_from_counterfactuals(
            run, val_task_ids, policy_fn, "family_conditioned_logistic",
            reference_result=sham_val, oracle_result=oracle_val,
        )
        learners["family_conditioned_logistic"] = {
            "description": "Family-conditioned logistic model (family one-hot + features).",
            "n_features_original": int(train_features.shape[1]),
            "n_features_family": int(train_fam_one_hot.shape[1]),
            "n_features_total": int(train_features.shape[1] + train_fam_one_hot.shape[1]),
            **_compute_learner_metrics(
                result, sham_val, baseline_val, oracle_val,
                actions, val_task_ids, val_digest,
            ),
        }
    except Exception as exc:
        learners["family_conditioned_logistic"] = {
            "description": "Family-conditioned logistic model (family one-hot + features).",
            "error": str(exc),
        }

    # ------------------------------------------------------------------
    # Select best learner on validation only.
    # ------------------------------------------------------------------
    val_utils: dict[str, float] = {}
    for name, data in learners.items():
        if "error" not in data:
            val_utils[name] = data["validation_expected_utility"]

    if not val_utils:
        return {
            "seed": seed,
            "n_train": len(train_examples),
            "n_validation": len(val_examples),
            "n_test": len(test_examples),
            "actions": actions,
            "learners": learners,
            "best_validation_learner": None,
            "test_results_exploratory": {},
            "interpretation": {
                "conclusion": "all_learners_failed",
                "note": "All learners encountered errors during training or evaluation.",
            },
        }

    best_learner = max(val_utils, key=val_utils.get)
    best_val_util = val_utils[best_learner]
    primary_val_util = val_utils.get("multinomial_logistic", 0.0)

    # ------------------------------------------------------------------
    # Exploratory test evaluation for the best validation learner.
    # ------------------------------------------------------------------
    test_results_exploratory: dict[str, Any] = {
        "EXPLORATORY": True,
        "note": (
            "Test results are EXPLORATORY only and were NOT used for "
            "learner selection.  Selection was based on D_validation only."
        ),
        "best_validation_learner": best_learner,
    }

    if test_examples:
        # Evaluate reference policies on test.
        oracle_test = evaluate_policy_from_counterfactuals(
            run, test_task_ids, make_oracle_fn(run), "oracle",
        )
        sham_test = evaluate_policy_from_counterfactuals(
            run, test_task_ids, make_executed_sham_fn(run), "sham",
            reference_result=None, oracle_result=oracle_test,
        )
        baseline_test = evaluate_policy_from_counterfactuals(
            run, test_task_ids, make_baseline_fn(run), "baseline",
            reference_result=None, oracle_result=oracle_test,
        )
        test_digest = oracle_test.task_ids_digest

        # Re-train the best learner and evaluate on test.
        try:
            if best_learner == "multinomial_logistic":
                _, pred_fn = _train_multinomial_logistic(train_features, train_labels, seed)
                fn = _make_policy_fn(pred_fn, feature_lookup, actions)
            elif best_learner == "class_weighted_logistic":
                _, pred_fn = _train_class_weighted_logistic(train_features, train_labels, seed)
                fn = _make_policy_fn(pred_fn, feature_lookup, actions)
            elif best_learner == "utility_weighted_logistic":
                _, pred_fn = _train_utility_weighted_logistic(
                    train_features, train_labels, utility_weights, seed,
                )
                fn = _make_policy_fn(pred_fn, feature_lookup, actions)
            elif best_learner == "soft_utility_logistic":
                _, pred_fn = _train_soft_utility_logistic(
                    train_features, train_utility_matrix, n_actions,
                    tau=0.1, seed=seed,
                )
                fn = _make_policy_fn(pred_fn, feature_lookup, actions)
            elif best_learner == "shallow_decision_tree":
                _, pred_fn = _train_decision_tree(train_features, train_labels, seed)
                fn = _make_policy_fn(pred_fn, feature_lookup, actions)
            elif best_learner == "gradient_boosted_tree":
                _, pred_fn = _train_gradient_boosted_tree(train_features, train_labels, seed)
                fn = _make_policy_fn(pred_fn, feature_lookup, actions)
            elif best_learner == "nearest_centroid":
                _, pred_fn = _train_nearest_centroid(train_features, train_labels, seed)
                fn = _make_policy_fn(pred_fn, feature_lookup, actions)
            elif best_learner == "family_conditioned_logistic":
                _, pred_fn = _train_family_conditioned_logistic(
                    train_features, train_labels, train_fam_one_hot, seed,
                )
                fn = _make_policy_fn(pred_fn, family_feature_lookup, actions)
            else:
                fn = None

            if fn is not None:
                best_test = evaluate_policy_from_counterfactuals(
                    run, test_task_ids, fn, best_learner,
                    reference_result=sham_test, oracle_result=oracle_test,
                )
                test_headroom_vs_baseline = compute_headroom_metrics(
                    candidate_mean=best_test.mean_utility,
                    reference_mean=baseline_test.mean_utility,
                    oracle_mean=oracle_test.mean_utility,
                    n_tasks=best_test.n_complete,
                    task_ids_digest=test_digest,
                    reference_policy_id="baseline",
                    candidate_policy_id=best_learner,
                )
                test_results_exploratory.update({
                    "test_expected_utility": float(best_test.mean_utility),
                    "test_candidate_minus_sham": float(best_test.mean_utility - sham_test.mean_utility),
                    "test_candidate_minus_baseline": float(best_test.mean_utility - baseline_test.mean_utility),
                    "test_regret": float(best_test.mean_regret) if best_test.mean_regret is not None else None,
                    "test_headroom_recovered_vs_sham": best_test.headroom_vs_reference,
                    "test_headroom_recovered_vs_baseline": test_headroom_vs_baseline,
                    "test_action_distribution": dict(best_test.action_distribution),
                    "test_n_tasks": best_test.n_tasks,
                    "test_n_complete": best_test.n_complete,
                    "test_action_accuracy": _compute_action_accuracy(best_test.task_results),
                })
        except Exception as exc:
            test_results_exploratory["error"] = str(exc)

    # ------------------------------------------------------------------
    # Interpretation.
    # ------------------------------------------------------------------
    gain_over_primary = best_val_util - primary_val_util

    # Check for route collapse in the best learner.
    best_data = learners[best_learner]
    best_collapsed = (
        best_data.get("route_collapse_metrics", {}).get("is_collapsed", False)
        if "error" not in best_data
        else False
    )

    # Check if any nonlinear learner beats the primary.
    nonlinear_names = [
        "shallow_decision_tree", "gradient_boosted_tree", "nearest_centroid",
    ]
    best_nonlinear_val = max(
        (val_utils[n] for n in nonlinear_names if n in val_utils),
        default=0.0,
    )
    best_nonlinear_name = max(
        ((n, val_utils[n]) for n in nonlinear_names if n in val_utils),
        key=lambda x: x[1],
        default=(None, 0.0),
    )[0]

    if gain_over_primary > 0.05:
        selection_conclusion = "diagnostic_learner_wins"
        selection_note = (
            f"The best validation learner ({best_learner}) materially beats "
            f"the primary multinomial logistic regression by "
            f"{gain_over_primary:.3f} utility on validation.  Consider "
            f"selecting this learner for a new qualification run."
        )
    elif gain_over_primary > 0.01:
        selection_conclusion = "diagnostic_learner_marginal"
        selection_note = (
            f"The best validation learner ({best_learner}) marginally beats "
            f"the primary by {gain_over_primary:.3f}.  The improvement is "
            f"small; the primary learner remains a reasonable choice."
        )
    else:
        selection_conclusion = "primary_sufficient"
        selection_note = (
            "No diagnostic learner materially beats the primary multinomial "
            "logistic regression on validation.  The current learner "
            "architecture is sufficient."
        )

    if best_nonlinear_name and best_nonlinear_val > primary_val_util + 0.05:
        boundary_conclusion = "nonlinear"
        boundary_note = (
            f"A nonlinear shallow model ({best_nonlinear_name}) beats logistic "
            f"regression by {best_nonlinear_val - primary_val_util:.3f} on "
            f"validation.  The feature-to-action boundary may be nonlinear."
        )
    elif best_nonlinear_name and best_nonlinear_val > primary_val_util + 0.01:
        boundary_conclusion = "possibly_nonlinear"
        boundary_note = (
            f"A nonlinear shallow model ({best_nonlinear_name}) slightly beats "
            f"logistic regression by {best_nonlinear_val - primary_val_util:.3f}. "
            f"There may be mild nonlinearity."
        )
    else:
        boundary_conclusion = "linear"
        boundary_note = (
            "No nonlinear shallow model materially beats logistic regression "
            "on validation.  The feature-to-action boundary appears "
            "approximately linear."
        )

    collapse_note = (
        f"The best learner ({best_learner}) shows route collapse "
        f"(predicted action entropy < 0.1).  This is a concern for "
        f"production deployment."
        if best_collapsed
        else
        f"The best learner ({best_learner}) does not show route collapse."
    )

    return {
        "seed": seed,
        "n_train": len(train_examples),
        "n_validation": len(val_examples),
        "n_test": len(test_examples),
        "actions": actions,
        "learners": learners,
        "best_validation_learner": best_learner,
        "best_validation_utility": float(best_val_util),
        "primary_validation_utility": float(primary_val_util),
        "test_results_exploratory": test_results_exploratory,
        "interpretation": {
            "selection_conclusion": selection_conclusion,
            "selection_note": selection_note,
            "boundary_type": boundary_conclusion,
            "boundary_note": boundary_note,
            "route_collapse_note": collapse_note,
            "best_learner_collapsed": bool(best_collapsed),
            "gain_over_primary": float(gain_over_primary),
            "best_nonlinear_learner": best_nonlinear_name,
            "best_nonlinear_val_utility": float(best_nonlinear_val),
        },
    }
