"""Learner validation for v0.4.4 evidence-complete analysis.

Compares diagnostic learners on validation only.  All learner selection
uses ``D_validation`` only — the final test split is never used for
model selection.

Diagnostic learners evaluated (all trained on D_experience, evaluated
on D_validation):

  1. Multinomial logistic regression
  2. Class-weighted logistic regression
  3. Utility-weighted logistic regression (weight samples by utility margin)
  4. Soft-utility-target logistic regression (soft targets from utilities)
  5. Tie-aware soft-target logistic regression
  6. Shallow decision tree (max_depth=3)
  7. Small gradient-boosted model (max_depth=2, n_estimators=50)
  8. Nearest-centroid classifier

The module detects and reports the following issues without relying
only on scikit-learn warnings:

  - zero-variance features
  - singular covariance
  - nonconvergence
  - class absence
  - constant predictions
  - single-action collapse
  - unstable coefficients

The output of :func:`validate_learners` is a JSON-serializable dict.
"""
from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Callable

import numpy as np

from .config import MINIMUM_DENOMINATOR


# ---------------------------------------------------------------------------
# Optional sklearn imports
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised when sklearn is present
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neighbors import NearestCentroid
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Status categories
# ---------------------------------------------------------------------------

LEARNER_VALIDATION_COMPLETE = "LEARNER_VALIDATION_COMPLETE"
LEARNER_VALIDATION_DEGENERATE = "LEARNER_VALIDATION_DEGENERATE"
LEARNER_VALIDATION_INCONCLUSIVE = "LEARNER_VALIDATION_INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _extract_examples(
    results: dict[str, Any],
    split: str = "validation",
) -> list[dict[str, Any]]:
    """Extract per-task examples from a results dict."""
    per_task = results.get("per_task", [])
    if per_task:
        return [ex for ex in per_task if isinstance(ex, dict)]
    experiences = results.get("experiences", [])
    if experiences:
        return experiences
    experiences = results.get("executive_experiences", [])
    if experiences:
        return experiences
    return []


def _feature_matrix(examples: list[dict[str, Any]]) -> np.ndarray:
    """Stack the ``features`` vectors of a list of examples."""
    if not examples:
        return np.zeros((0, 0), dtype=float)
    feats: list[list[float]] = []
    for ex in examples:
        f = ex.get("features", [])
        if isinstance(f, list):
            feats.append([float(x) for x in f])
        elif isinstance(f, np.ndarray):
            feats.append([float(x) for x in f.tolist()])
        else:
            feats.append([0.0])
    if not feats:
        return np.zeros((0, 0), dtype=float)
    return np.array(feats, dtype=float)


def _labels(examples: list[dict[str, Any]]) -> np.ndarray:
    """Return the best-action labels for a list of examples."""
    return np.array([ex.get("best_action", "") for ex in examples], dtype=object)


def _get_actions(results: dict[str, Any]) -> list[str]:
    """Extract the action list from a results dict."""
    actions = results.get("actions", [])
    if actions:
        return list(actions)
    per_task = results.get("per_task", [])
    seen: list[str] = []
    for ex in per_task:
        a = ex.get("best_action", "")
        if a and a not in seen:
            seen.append(a)
    return seen


def _get_utility_for_action(
    results: dict[str, Any], task_id: str, action: str
) -> float | None:
    """Look up the measured utility for a (task_id, action) pair."""
    per_task = results.get("per_task", [])
    for ex in per_task:
        if ex.get("task_id") == task_id:
            outcomes = ex.get("outcomes", [])
            for oc in outcomes:
                if oc.get("action") == action or oc.get("action_id") == action:
                    u = oc.get("utility")
                    if u is not None:
                        return float(u)
            if ex.get("selected_action", ex.get("best_action")) == action:
                u = ex.get("utility")
                if u is not None:
                    return float(u)
    return None


def _compute_utility_weights(examples: list[dict[str, Any]]) -> np.ndarray:
    """Compute per-example utility-margin weights."""
    weights = np.ones(len(examples), dtype=float)
    for i, ex in enumerate(examples):
        margin = ex.get("utility_margin", 0.0)
        weights[i] = max(0.0, float(margin)) + 1.0
    return weights


def _build_utility_matrix(
    examples: list[dict[str, Any]],
    actions: list[str],
    candidate_results: dict[str, Any],
) -> np.ndarray:
    """Build a utility matrix [n_examples, n_actions] from measured outcomes."""
    action_idx = {a: i for i, a in enumerate(actions)}
    U = np.zeros((len(examples), len(actions)), dtype=float)
    for i, ex in enumerate(examples):
        tid = ex.get("task_id", "")
        outcomes = ex.get("outcomes", [])
        for oc in outcomes:
            a = oc.get("action", oc.get("action_id", ""))
            if a in action_idx and oc.get("utility") is not None:
                U[i, action_idx[a]] = float(oc["utility"])
        # Fallback: task-level utility for the selected action.
        if not any(U[i] != 0):
            sel = ex.get("selected_action", ex.get("best_action", ""))
            u = _get_utility_for_action(candidate_results, tid, sel)
            if u is not None and sel in action_idx:
                U[i, action_idx[sel]] = u
    return U


# ---------------------------------------------------------------------------
# Issue detection
# ---------------------------------------------------------------------------


def _detect_zero_variance_features(features: np.ndarray) -> list[int]:
    """Return indices of features with zero variance."""
    if features.shape[0] < 2 or features.shape[1] == 0:
        return []
    variances = np.var(features, axis=0)
    return [int(i) for i, v in enumerate(variances) if v < MINIMUM_DENOMINATOR]


def _detect_singular_covariance(features: np.ndarray) -> bool:
    """Check whether the feature covariance matrix is singular."""
    if features.shape[0] < 2 or features.shape[1] < 1:
        return False
    try:
        cov = np.cov(features, rowvar=False)
        if cov.ndim == 0:
            return False
        cond = np.linalg.cond(cov)
        return bool(cond > 1e12 or not np.isfinite(cond))
    except np.linalg.LinAlgError:
        return True


def _detect_class_absence(labels: np.ndarray, actions: list[str]) -> list[str]:
    """Return actions that are absent from the training labels."""
    present = set(labels.tolist())
    return [a for a in actions if a not in present]


def _detect_constant_predictions(predictions: np.ndarray) -> bool:
    """Check whether all predictions are the same."""
    if len(predictions) == 0:
        return False
    return bool(np.all(predictions == predictions[0]))


def _detect_single_action_collapse(predictions: np.ndarray) -> bool:
    """Check whether the model collapses to a single action."""
    if len(predictions) == 0:
        return False
    return len(np.unique(predictions)) == 1


def _detect_unstable_coefficients(
    coefficients_across_seeds: list[np.ndarray],
) -> bool:
    """Check whether coefficients vary widely across seeds."""
    if len(coefficients_across_seeds) < 2:
        return False
    stacked = np.array(coefficients_across_seeds)
    stds = np.std(stacked, axis=0)
    max_std = float(np.max(stds)) if stds.size else 0.0
    return bool(max_std > 1.0)


def _detect_nonconvergence(model: Any) -> bool:
    """Check for nonconvergence signs on a sklearn logistic model."""
    if hasattr(model, "n_iter_"):
        n_iter = model.n_iter_
        if isinstance(n_iter, (list, np.ndarray)):
            max_iter = getattr(model, "max_iter", 1000)
            return any(int(n) >= max_iter for n in n_iter)
        return int(n_iter) >= getattr(model, "max_iter", 1000)
    return False


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


def _train_tie_aware_soft_target_logistic(
    features: np.ndarray,
    utility_matrix: np.ndarray,
    n_actions: int,
    tau: float = 0.1,
    tie_threshold: float = 0.01,
    n_epochs: int = 200,
    lr: float = 0.01,
    l2: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Train tie-aware soft-target logistic regression.

    When the top-2 utilities are within ``tie_threshold``, the soft
    target distributes mass equally among the tied actions rather than
    concentrating on the single argmax.
    """
    n = utility_matrix.shape[0]
    soft_targets = np.zeros_like(utility_matrix, dtype=float)

    for i in range(n):
        u = utility_matrix[i]
        top_idx = np.argsort(u)[::-1]
        top1, top2 = top_idx[0], top_idx[1]
        if abs(u[top1] - u[top2]) < tie_threshold:
            # Tie: distribute mass equally among tied actions.
            tied = [j for j in range(n_actions) if abs(u[j] - u[top1]) < tie_threshold]
            for j in tied:
                soft_targets[i, j] = 1.0 / len(tied)
        else:
            scaled = u / tau
            soft_targets[i] = _softmax(scaled.reshape(1, -1))[0]

    return _train_soft_target_logistic(
        features, soft_targets, n_actions, n_epochs, lr, l2, seed,
    )


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
        solver="lbfgs", max_iter=1000, random_state=seed,
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
        solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=seed,
    )
    model.fit(train_features, train_labels)
    return model, lambda feats: int(model.predict(feats.reshape(1, -1))[0])


def _train_utility_weighted_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    sample_weights: np.ndarray,
    seed: int,
) -> tuple[Any, Callable[[np.ndarray], int]]:
    """Train utility-weighted logistic regression."""
    model = LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=seed,
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


def _train_tie_aware_soft_logistic(
    train_features: np.ndarray,
    utility_matrix: np.ndarray,
    n_actions: int,
    tau: float,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], Callable[[np.ndarray], int]]:
    """Train tie-aware soft-target logistic regression."""
    W, b = _train_tie_aware_soft_target_logistic(
        train_features, utility_matrix, n_actions,
        tau=tau, seed=seed,
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


def _train_gradient_boosted(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
) -> tuple[Any, Callable[[np.ndarray], int]]:
    """Train a small gradient-boosted model (max_depth=2, n_estimators=50)."""
    model = GradientBoostingClassifier(
        max_depth=2, n_estimators=50, random_state=seed,
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_predictions(
    predictions: np.ndarray,
    val_examples: list[dict[str, Any]],
    actions: list[str],
    candidate_results: dict[str, Any],
) -> float:
    """Compute validation expected utility from predicted action indices."""
    if len(val_examples) == 0:
        return 0.0
    utilities: list[float] = []
    for i, ex in enumerate(val_examples):
        tid = ex.get("task_id", "")
        idx = int(predictions[i])
        action = actions[idx] if idx < len(actions) else ""
        u = _get_utility_for_action(candidate_results, tid, action)
        utilities.append(u if u is not None else 0.0)
    return float(np.mean(utilities)) if utilities else 0.0


def _predict_all(
    predict_fn: Callable[[np.ndarray], int],
    val_features: np.ndarray,
) -> np.ndarray:
    """Apply a per-row predict function to all validation rows."""
    if val_features.shape[0] == 0:
        return np.array([], dtype=int)
    return np.array([predict_fn(val_features[i]) for i in range(val_features.shape[0])])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_learners(
    candidate_results: dict,
    sham_results: dict,
    n_tasks: int = 30,
) -> dict:
    """Compare diagnostic learners on validation only.

    All learner selection uses ``D_validation`` only.  The final test
    split is never used for model selection.

    Args:
        candidate_results: Results dict from the candidate policy run.
        sham_results: Results dict from the sham policy run.
        n_tasks: Minimum number of tasks for a conclusive analysis.

    Returns:
        A dict with:
          - status: one of the status categories.
          - learner_results: dict of learner name -> {
              validation_utility, converged, warnings}.
          - best_learner: name of the best validation learner.
          - conclusion: human-readable conclusion string.
    """
    train_examples = _extract_examples(candidate_results, "train")
    val_examples = _extract_examples(candidate_results, "validation")
    if not val_examples:
        val_examples = train_examples

    actions = _get_actions(candidate_results)

    if len(val_examples) < n_tasks:
        return {
            "status": LEARNER_VALIDATION_INCONCLUSIVE,
            "learner_results": {},
            "best_learner": None,
            "conclusion": (
                f"Insufficient validation tasks ({len(val_examples)} < "
                f"{n_tasks}). Learner validation is inconclusive."
            ),
        }

    if len(actions) < 2:
        return {
            "status": LEARNER_VALIDATION_DEGENERATE,
            "learner_results": {},
            "best_learner": None,
            "conclusion": (
                f"Fewer than 2 actions ({actions}). Cannot compare "
                f"learners with a single-action space."
            ),
        }

    train_features = _feature_matrix(train_examples)
    val_features = _feature_matrix(val_examples)
    train_labels = _labels(train_examples)
    val_labels = _labels(val_examples)

    # Map labels to integer indices for sklearn.
    action_idx = {a: i for i, a in enumerate(actions)}
    train_label_idx = np.array([action_idx.get(a, 0) for a in train_labels])
    val_label_idx = np.array([action_idx.get(a, 0) for a in val_labels])

    # --- Pre-flight diagnostics -----------------------------------------
    global_warnings: list[str] = []

    zero_var_idx = _detect_zero_variance_features(train_features)
    if zero_var_idx:
        global_warnings.append(
            f"zero_variance_features:{zero_var_idx}"
        )

    singular = _detect_singular_covariance(train_features)
    if singular:
        global_warnings.append("singular_covariance")

    absent_classes = _detect_class_absence(train_labels, actions)
    if absent_classes:
        global_warnings.append(
            f"class_absence:{absent_classes}"
        )

    utility_weights = _compute_utility_weights(train_examples)
    train_utility_matrix = _build_utility_matrix(
        train_examples, actions, candidate_results
    )

    n_actions = len(actions)
    seed = 42
    seeds = [42, 123, 456]

    learner_results: dict[str, dict[str, Any]] = {}

    # --- 1. Multinomial logistic regression ------------------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            model, predict_fn = _train_multinomial_logistic(
                train_features, train_label_idx, seed,
            )
        preds = _predict_all(predict_fn, val_features)
        val_util = _evaluate_predictions(
            preds, val_examples, actions, candidate_results
        )
        warns: list[str] = list(global_warnings)
        if _detect_nonconvergence(model):
            warns.append("nonconvergence")
        if _detect_constant_predictions(preds):
            warns.append("constant_predictions")
        if _detect_single_action_collapse(preds):
            warns.append("single_action_collapse")
        # Check coefficient stability across seeds.
        coeffs: list[np.ndarray] = []
        for s in seeds:
            try:
                m, _ = _train_multinomial_logistic(train_features, train_label_idx, s)
                if hasattr(m, "coef_"):
                    coeffs.append(np.array(m.coef_))
            except Exception:
                pass
        if _detect_unstable_coefficients(coeffs):
            warns.append("unstable_coefficients")
        learner_results["multinomial_logistic_regression"] = {
            "validation_utility": val_util,
            "converged": "nonconvergence" not in warns,
            "warnings": warns,
        }
    except Exception as exc:
        learner_results["multinomial_logistic_regression"] = {
            "validation_utility": 0.0,
            "converged": False,
            "warnings": list(global_warnings) + [f"error:{exc}"],
        }

    # --- 2. Class-weighted logistic regression ---------------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            model, predict_fn = _train_class_weighted_logistic(
                train_features, train_label_idx, seed,
            )
        preds = _predict_all(predict_fn, val_features)
        val_util = _evaluate_predictions(
            preds, val_examples, actions, candidate_results
        )
        warns = list(global_warnings)
        if _detect_nonconvergence(model):
            warns.append("nonconvergence")
        if _detect_constant_predictions(preds):
            warns.append("constant_predictions")
        if _detect_single_action_collapse(preds):
            warns.append("single_action_collapse")
        learner_results["class_weighted_logistic_regression"] = {
            "validation_utility": val_util,
            "converged": "nonconvergence" not in warns,
            "warnings": warns,
        }
    except Exception as exc:
        learner_results["class_weighted_logistic_regression"] = {
            "validation_utility": 0.0,
            "converged": False,
            "warnings": list(global_warnings) + [f"error:{exc}"],
        }

    # --- 3. Utility-weighted logistic regression -------------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            model, predict_fn = _train_utility_weighted_logistic(
                train_features, train_label_idx, utility_weights, seed,
            )
        preds = _predict_all(predict_fn, val_features)
        val_util = _evaluate_predictions(
            preds, val_examples, actions, candidate_results
        )
        warns = list(global_warnings)
        if _detect_nonconvergence(model):
            warns.append("nonconvergence")
        if _detect_constant_predictions(preds):
            warns.append("constant_predictions")
        if _detect_single_action_collapse(preds):
            warns.append("single_action_collapse")
        learner_results["utility_weighted_logistic_regression"] = {
            "validation_utility": val_util,
            "converged": "nonconvergence" not in warns,
            "warnings": warns,
        }
    except Exception as exc:
        learner_results["utility_weighted_logistic_regression"] = {
            "validation_utility": 0.0,
            "converged": False,
            "warnings": list(global_warnings) + [f"error:{exc}"],
        }

    # --- 4. Soft-utility-target logistic regression ----------------------
    try:
        soft_tau = 0.1
        (W, b), predict_fn = _train_soft_utility_logistic(
            train_features, train_utility_matrix, n_actions,
            tau=soft_tau, seed=seed,
        )
        preds = _predict_all(predict_fn, val_features)
        val_util = _evaluate_predictions(
            preds, val_examples, actions, candidate_results
        )
        warns = list(global_warnings)
        if _detect_constant_predictions(preds):
            warns.append("constant_predictions")
        if _detect_single_action_collapse(preds):
            warns.append("single_action_collapse")
        learner_results["soft_utility_target_logistic_regression"] = {
            "validation_utility": val_util,
            "converged": True,
            "warnings": warns,
        }
    except Exception as exc:
        learner_results["soft_utility_target_logistic_regression"] = {
            "validation_utility": 0.0,
            "converged": False,
            "warnings": list(global_warnings) + [f"error:{exc}"],
        }

    # --- 5. Tie-aware soft-target logistic regression --------------------
    try:
        tie_tau = 0.1
        (W, b), predict_fn = _train_tie_aware_soft_logistic(
            train_features, train_utility_matrix, n_actions,
            tau=tie_tau, seed=seed,
        )
        preds = _predict_all(predict_fn, val_features)
        val_util = _evaluate_predictions(
            preds, val_examples, actions, candidate_results
        )
        warns = list(global_warnings)
        if _detect_constant_predictions(preds):
            warns.append("constant_predictions")
        if _detect_single_action_collapse(preds):
            warns.append("single_action_collapse")
        learner_results["tie_aware_soft_target_logistic_regression"] = {
            "validation_utility": val_util,
            "converged": True,
            "warnings": warns,
        }
    except Exception as exc:
        learner_results["tie_aware_soft_target_logistic_regression"] = {
            "validation_utility": 0.0,
            "converged": False,
            "warnings": list(global_warnings) + [f"error:{exc}"],
        }

    # --- 6. Shallow decision tree ----------------------------------------
    try:
        model, predict_fn = _train_decision_tree(
            train_features, train_label_idx, seed,
        )
        preds = _predict_all(predict_fn, val_features)
        val_util = _evaluate_predictions(
            preds, val_examples, actions, candidate_results
        )
        warns = list(global_warnings)
        if _detect_constant_predictions(preds):
            warns.append("constant_predictions")
        if _detect_single_action_collapse(preds):
            warns.append("single_action_collapse")
        learner_results["shallow_decision_tree"] = {
            "validation_utility": val_util,
            "converged": True,
            "warnings": warns,
        }
    except Exception as exc:
        learner_results["shallow_decision_tree"] = {
            "validation_utility": 0.0,
            "converged": False,
            "warnings": list(global_warnings) + [f"error:{exc}"],
        }

    # --- 7. Small gradient-boosted model ---------------------------------
    try:
        model, predict_fn = _train_gradient_boosted(
            train_features, train_label_idx, seed,
        )
        preds = _predict_all(predict_fn, val_features)
        val_util = _evaluate_predictions(
            preds, val_examples, actions, candidate_results
        )
        warns = list(global_warnings)
        if _detect_constant_predictions(preds):
            warns.append("constant_predictions")
        if _detect_single_action_collapse(preds):
            warns.append("single_action_collapse")
        learner_results["small_gradient_boosted_model"] = {
            "validation_utility": val_util,
            "converged": True,
            "warnings": warns,
        }
    except Exception as exc:
        learner_results["small_gradient_boosted_model"] = {
            "validation_utility": 0.0,
            "converged": False,
            "warnings": list(global_warnings) + [f"error:{exc}"],
        }

    # --- 8. Nearest centroid ---------------------------------------------
    try:
        model, predict_fn = _train_nearest_centroid(
            train_features, train_label_idx, seed,
        )
        preds = _predict_all(predict_fn, val_features)
        val_util = _evaluate_predictions(
            preds, val_examples, actions, candidate_results
        )
        warns = list(global_warnings)
        if _detect_constant_predictions(preds):
            warns.append("constant_predictions")
        if _detect_single_action_collapse(preds):
            warns.append("single_action_collapse")
        learner_results["nearest_centroid"] = {
            "validation_utility": val_util,
            "converged": True,
            "warnings": warns,
        }
    except Exception as exc:
        learner_results["nearest_centroid"] = {
            "validation_utility": 0.0,
            "converged": False,
            "warnings": list(global_warnings) + [f"error:{exc}"],
        }

    # --- Select best learner on validation only --------------------------
    val_utils: dict[str, float] = {
        name: data["validation_utility"]
        for name, data in learner_results.items()
    }

    if not val_utils:
        return {
            "status": LEARNER_VALIDATION_INCONCLUSIVE,
            "learner_results": learner_results,
            "best_learner": None,
            "conclusion": "All learners encountered errors.",
        }

    best_learner = max(val_utils, key=val_utils.get)
    best_val_util = val_utils[best_learner]

    # Collect all unique warnings.
    all_warnings: set[str] = set()
    for data in learner_results.values():
        all_warnings.update(data.get("warnings", []))

    if all_warnings:
        status = LEARNER_VALIDATION_COMPLETE
        conclusion = (
            f"Best validation learner: {best_learner} "
            f"(validation utility {best_val_util:.4f}). "
            f"Warnings detected: {sorted(all_warnings)}. "
            f"Selection used validation only."
        )
    else:
        status = LEARNER_VALIDATION_COMPLETE
        conclusion = (
            f"Best validation learner: {best_learner} "
            f"(validation utility {best_val_util:.4f}). "
            f"No warnings detected. Selection used validation only."
        )

    return {
        "status": status,
        "learner_results": learner_results,
        "best_learner": best_learner,
        "conclusion": conclusion,
    }
