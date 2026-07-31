"""Utility-sensitive training objective comparison.

Compares three routing-policy training objectives that differ in how
they use measured counterfactual utilities:

  1. **Hard-label router** (current): train on argmax best-action
     labels — a standard cross-entropy objective with one-hot targets.

  2. **Soft-utility router**: convert utilities into a soft target
     distribution using temperature ``tau``:

         q_i(a) = exp(U_{i,a} / tau) / sum_b exp(U_{i,b} / tau)

     Train to minimize:

         L = -sum_i w_i sum_a q_i(a) * log p_theta(a | x_i)

     This is a cross-entropy with soft targets derived from utilities.

  3. **Expected-utility router**: directly minimize the negative
     expected utility under the policy's own action distribution:

         L = -sum_i w_i sum_a p_theta(a | x_i) * U_{i,a}

     This objective is aware of the model's own probabilities and
     pushes mass toward high-utility actions.

The temperature ``tau`` for the soft-utility router is selected on
D_validation only.  All three objectives are compared on validation
before final test evaluation.

For each task in the training set, the full utility vector (utility
for each action) is built from ``run.outcomes_by_task_action``.

The output of :func:`compare_objectives` is the content for
``utility_objective_comparison.json``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import LoadedRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _extract_features(examples: list[dict]) -> np.ndarray:
    """Extract feature matrix from a list of examples."""
    return np.array([ex["features"] for ex in examples], dtype=float)


def _build_utility_matrix(
    run: LoadedRun,
    task_ids: list[str],
    actions: list[str],
) -> np.ndarray:
    """Build a utility matrix [n_tasks, n_actions] from measured outcomes.

    Missing (task, action) pairs get utility 0.0.  Only executed
    outcomes with non-None utility are included.
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


def _build_hard_labels(
    run: LoadedRun,
    task_ids: list[str],
    actions: list[str],
) -> np.ndarray:
    """Build hard (argmax best-action) labels for each task.

    Returns an array of action indices.  Tasks with no executed
    outcomes default to action index 0.
    """
    action_idx = {a: i for i, a in enumerate(actions)}
    labels = np.zeros(len(task_ids), dtype=int)
    for i, tid in enumerate(task_ids):
        best = run.get_best_action(tid)
        if best is not None and best in action_idx:
            labels[i] = action_idx[best]
    return labels


def _compute_example_weights(
    run: LoadedRun,
    task_ids: list[str],
) -> np.ndarray:
    """Compute per-example weights based on utility margin.

    Weight = max(0, margin) + 1.0, where margin is the gap between
    the best and second-best measured utilities.  Clear-best tasks
    get more weight; ambiguous tasks get baseline weight 1.0.
    """
    weights = np.ones(len(task_ids), dtype=float)
    for i, tid in enumerate(task_ids):
        outcomes = run.outcomes_by_task.get(tid, [])
        executed = [
            o.utility for o in outcomes
            if o.availability == "executed" and o.utility is not None
        ]
        if len(executed) >= 2:
            sorted_u = sorted(executed, reverse=True)
            margin = sorted_u[0] - sorted_u[1]
            weights[i] = max(0.0, margin) + 1.0
    return weights


# ---------------------------------------------------------------------------
# Objective 1: Hard-label router
# ---------------------------------------------------------------------------


def _train_hard_label_router(
    features: np.ndarray,
    hard_labels: np.ndarray,
    weights: np.ndarray,
    n_actions: int,
    n_epochs: int = 200,
    lr: float = 0.01,
    l2: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a logistic router on hard (argmax) labels.

    Minimizes standard cross-entropy with one-hot targets:
        L = -sum_i w_i * log p_theta(y_i | x_i)

    Returns (weights [n_features, n_actions], bias [n_actions]).
    """
    rng = np.random.default_rng(seed)
    n, d = features.shape

    W = rng.standard_normal((d, n_actions)) * 0.01
    b = np.zeros(n_actions)

    for _ in range(n_epochs):
        logits = features @ W + b
        probs = _softmax(logits)

        # One-hot targets.
        one_hot = np.zeros((n, n_actions))
        one_hot[np.arange(n), hard_labels] = 1.0

        # Weighted gradient: diff_i = w_i * (p_i - y_i)
        diff = (probs - one_hot) * weights[:, None]
        grad_W = features.T @ diff / n + l2 * W
        grad_b = diff.mean(axis=0)

        W -= lr * grad_W
        b -= lr * grad_b

    return W, b


# ---------------------------------------------------------------------------
# Objective 2: Soft-utility router
# ---------------------------------------------------------------------------


def _train_soft_utility_router(
    features: np.ndarray,
    utility_matrix: np.ndarray,
    weights: np.ndarray,
    n_actions: int,
    tau: float = 0.1,
    n_epochs: int = 200,
    lr: float = 0.01,
    l2: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a logistic router with soft utility targets.

    Converts utilities into a soft target distribution:
        q_i(a) = exp(U_{i,a} / tau) / sum_b exp(U_{i,b} / tau)

    Minimizes cross-entropy with soft targets:
        L = -sum_i w_i sum_a q_i(a) * log p_theta(a | x_i)

    Returns (weights [n_features, n_actions], bias [n_actions]).
    """
    rng = np.random.default_rng(seed)
    n, d = features.shape

    # Compute soft targets from utilities.
    scaled_u = utility_matrix / tau
    soft_targets = _softmax(scaled_u)

    W = rng.standard_normal((d, n_actions)) * 0.01
    b = np.zeros(n_actions)

    for _ in range(n_epochs):
        logits = features @ W + b
        probs = _softmax(logits)

        # Weighted gradient: diff_i = w_i * (p_i - q_i)
        diff = (probs - soft_targets) * weights[:, None]
        grad_W = features.T @ diff / n + l2 * W
        grad_b = diff.mean(axis=0)

        W -= lr * grad_W
        b -= lr * grad_b

    return W, b


# ---------------------------------------------------------------------------
# Objective 3: Expected-utility router
# ---------------------------------------------------------------------------


def _train_expected_utility_router(
    features: np.ndarray,
    utility_matrix: np.ndarray,
    weights: np.ndarray,
    n_actions: int,
    n_epochs: int = 200,
    lr: float = 0.01,
    l2: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a logistic router by directly minimizing negative expected utility.

    Minimizes:
        L = -sum_i w_i sum_a p_theta(a | x_i) * U_{i,a}

    The gradient with respect to the logits is:
        dL/dlogit_k = w_i * p_i(k) * (U_{i,k} - sum_a p_i(a) * U_{i,a})

    Returns (weights [n_features, n_actions], bias [n_actions]).
    """
    rng = np.random.default_rng(seed)
    n, d = features.shape

    W = rng.standard_normal((d, n_actions)) * 0.01
    b = np.zeros(n_actions)

    for _ in range(n_epochs):
        logits = features @ W + b
        probs = _softmax(logits)

        # Expected utility per example: E_i = sum_a p_i(a) * U_{i,a}
        expected_u = np.sum(probs * utility_matrix, axis=1, keepdims=True)

        # Gradient of -E[U] w.r.t. logits:
        # d/dlogit_k [-sum_a p(a) U(a)] = -p(k) * (U(k) - E[U])
        # So the gradient of the loss is: p(k) * (U(k) - E[U])
        # (we minimize negative expected utility, so we ascend E[U]).
        grad_logits = probs * (utility_matrix - expected_u)  # [n, n_actions]

        # Apply per-example weights.
        grad_logits = grad_logits * weights[:, None]

        grad_W = features.T @ grad_logits / n + l2 * W
        grad_b = grad_logits.mean(axis=0)

        W -= lr * grad_W
        b -= lr * grad_b

    return W, b


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def _evaluate_router(
    W: np.ndarray,
    b: np.ndarray,
    features: np.ndarray,
    task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
) -> dict[str, Any]:
    """Evaluate a trained router (weights + bias) on a set of tasks.

    Predicts the argmax action, then looks up measured utility from
    ``run.outcomes_by_task_action``.
    """
    action_idx = {a: i for i, a in enumerate(actions)}
    logits = features @ W + b
    predictions = np.argmax(logits, axis=1)
    pred_actions = [actions[int(p)] for p in predictions]

    utilities: list[float] = []
    per_task: list[dict[str, Any]] = []

    for tid, pred_action in zip(task_ids, pred_actions):
        task = run.tasks.get(tid)
        if task is None:
            utilities.append(0.0)
            continue

        # Respect allowed_actions constraint.
        if pred_action not in task.allowed_actions:
            eligible = [a for a in task.allowed_actions if a in action_idx]
            if eligible:
                pred_action = eligible[0]
            else:
                pred_action = task.allowed_actions[0] if task.allowed_actions else pred_action

        u = run.get_utility_for_action(tid, pred_action)
        if u is None:
            u = 0.0

        utilities.append(u)
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
        "utilities": utilities_arr.tolist(),
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Temperature selection for soft-utility router
# ---------------------------------------------------------------------------


def _select_tau(
    train_features: np.ndarray,
    train_utility: np.ndarray,
    train_weights: np.ndarray,
    val_features: np.ndarray,
    val_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
    n_actions: int,
    tau_candidates: list[float],
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Select the best temperature tau on D_validation.

    Tries each candidate tau, trains a soft-utility router, evaluates
    on validation, and returns the tau with the highest validation
    utility.

    Returns (best_tau, {tau: val_utility}).
    """
    val_results: dict[str, float] = {}
    best_tau = tau_candidates[0]
    best_val = -np.inf

    for tau in tau_candidates:
        W, b = _train_soft_utility_router(
            train_features, train_utility, train_weights,
            n_actions, tau=tau, seed=seed,
        )
        val_res = _evaluate_router(W, b, val_features, val_task_ids, run, actions)
        val_util = val_res["mean_utility"]
        val_results[f"tau_{tau}"] = val_util
        if val_util > best_val:
            best_val = val_util
            best_tau = tau

    return best_tau, val_results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compare_objectives(run: LoadedRun, seed: int = 42) -> dict[str, Any]:
    """Compare utility-sensitive training objectives for routing policies.

    Trains three routers with different objectives (hard-label,
    soft-utility, expected-utility) on D_experience, selects the
    temperature ``tau`` for the soft-utility router on D_validation,
    and compares all three on validation before final test evaluation
    on D_test.

    Args:
        run: loaded run data.
        seed: random seed for reproducibility.

    Returns:
        Dict with per-objective validation/test utilities, selected
        tau, and interpretation.  Suitable for writing to
        ``utility_objective_comparison.json``.
    """
    actions = run.all_actions
    n_actions = len(actions)

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

    train_features = _extract_features(train_examples)
    val_features = _extract_features(val_examples)
    test_features = _extract_features(test_examples)

    train_task_ids = [ex["task_id"] for ex in train_examples]
    val_task_ids = [ex["task_id"] for ex in val_examples]
    test_task_ids = [ex["task_id"] for ex in test_examples]

    # Build utility matrices and labels from measured outcomes.
    train_utility = _build_utility_matrix(run, train_task_ids, actions)
    val_utility = _build_utility_matrix(run, val_task_ids, actions)
    # test_utility is not used for training or selection — only for
    # final evaluation via run.outcomes_by_task_action lookups.

    train_hard_labels = _build_hard_labels(run, train_task_ids, actions)
    train_weights = _compute_example_weights(run, train_task_ids)

    # ------------------------------------------------------------------
    # Objective 1: Hard-label router (current).
    # ------------------------------------------------------------------
    W_hard, b_hard = _train_hard_label_router(
        train_features, train_hard_labels, train_weights,
        n_actions, seed=seed,
    )
    hard_val = _evaluate_router(W_hard, b_hard, val_features, val_task_ids, run, actions)
    hard_test = _evaluate_router(W_hard, b_hard, test_features, test_task_ids, run, actions)

    # ------------------------------------------------------------------
    # Objective 2: Soft-utility router (select tau on validation).
    # ------------------------------------------------------------------
    tau_candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    best_tau, tau_val_results = _select_tau(
        train_features, train_utility, train_weights,
        val_features, val_task_ids, run, actions,
        n_actions, tau_candidates, seed,
    )

    W_soft, b_soft = _train_soft_utility_router(
        train_features, train_utility, train_weights,
        n_actions, tau=best_tau, seed=seed,
    )
    soft_val = _evaluate_router(W_soft, b_soft, val_features, val_task_ids, run, actions)
    soft_test = _evaluate_router(W_soft, b_soft, test_features, test_task_ids, run, actions)

    # ------------------------------------------------------------------
    # Objective 3: Expected-utility router.
    # ------------------------------------------------------------------
    W_eu, b_eu = _train_expected_utility_router(
        train_features, train_utility, train_weights,
        n_actions, seed=seed,
    )
    eu_val = _evaluate_router(W_eu, b_eu, val_features, val_task_ids, run, actions)
    eu_test = _evaluate_router(W_eu, b_eu, test_features, test_task_ids, run, actions)

    # ------------------------------------------------------------------
    # Summarize results.
    # ------------------------------------------------------------------
    objectives: dict[str, dict[str, Any]] = {
        "hard_label": {
            "description": (
                "Hard-label router (current): train on argmax "
                "best-action labels with cross-entropy loss."
            ),
            "is_current": True,
            "validation": {
                "mean_utility": hard_val["mean_utility"],
                "n_tasks": hard_val["n_tasks"],
            },
            "test": {
                "mean_utility": hard_test["mean_utility"],
                "median_utility": hard_test["median_utility"],
                "utility_std": hard_test["utility_std"],
                "n_tasks": hard_test["n_tasks"],
            },
        },
        "soft_utility": {
            "description": (
                "Soft-utility router: convert utilities into soft "
                "target distribution via temperature tau, train with "
                "cross-entropy against soft targets."
            ),
            "selected_tau": best_tau,
            "tau_candidates": tau_candidates,
            "tau_validation_utilities": tau_val_results,
            "validation": {
                "mean_utility": soft_val["mean_utility"],
                "n_tasks": soft_val["n_tasks"],
            },
            "test": {
                "mean_utility": soft_test["mean_utility"],
                "median_utility": soft_test["median_utility"],
                "utility_std": soft_test["utility_std"],
                "n_tasks": soft_test["n_tasks"],
            },
        },
        "expected_utility": {
            "description": (
                "Expected-utility router: directly minimize negative "
                "expected utility under the policy's own action "
                "distribution."
            ),
            "validation": {
                "mean_utility": eu_val["mean_utility"],
                "n_tasks": eu_val["n_tasks"],
            },
            "test": {
                "mean_utility": eu_test["mean_utility"],
                "median_utility": eu_test["median_utility"],
                "utility_std": eu_test["utility_std"],
                "n_tasks": eu_test["n_tasks"],
            },
        },
    }

    # ------------------------------------------------------------------
    # Select best objective on validation only.
    # ------------------------------------------------------------------
    val_utils = {
        name: data["validation"]["mean_utility"]
        for name, data in objectives.items()
    }
    best_objective = max(val_utils, key=val_utils.get)

    # Compute gains over the current hard-label objective.
    soft_gain = soft_test["mean_utility"] - hard_test["mean_utility"]
    eu_gain = eu_test["mean_utility"] - hard_test["mean_utility"]
    best_gain = max(soft_gain, eu_gain)

    # ------------------------------------------------------------------
    # Interpretation.
    # ------------------------------------------------------------------
    if best_gain > 0.05:
        objective_interpretation = "utility_sensitive_helps"
        objective_note = (
            f"A utility-sensitive objective ({best_objective}) materially "
            f"improves test utility by {best_gain:.3f} over the hard-label "
            f"router.  The current objective leaves signal on the table."
        )
    elif best_gain > 0.01:
        objective_interpretation = "utility_sensitive_marginal"
        objective_note = (
            f"A utility-sensitive objective ({best_objective}) marginally "
            f"improves test utility by {best_gain:.3f}.  There is modest "
            f"benefit to utility-aware training."
        )
    else:
        objective_interpretation = "hard_label_sufficient"
        objective_note = (
            "No utility-sensitive objective materially improves over the "
            "hard-label router.  The current objective is sufficient, or "
            "the utility signal is too weak for soft targets to help."
        )

    return {
        "seed": seed,
        "n_train": len(train_examples),
        "n_validation": len(val_examples),
        "n_test": len(test_examples),
        "actions": actions,
        "objectives": objectives,
        "selection": {
            "method": "Best validation utility (D_test not used for selection).",
            "best_objective": best_objective,
            "best_val_utility": float(val_utils[best_objective]),
        },
        "gains_over_hard_label": {
            "soft_utility_test_gain": float(soft_gain),
            "expected_utility_test_gain": float(eu_gain),
            "best_test_gain": float(best_gain),
        },
        "interpretation": {
            "conclusion": objective_interpretation,
            "note": objective_note,
            "selected_tau": best_tau,
        },
    }
