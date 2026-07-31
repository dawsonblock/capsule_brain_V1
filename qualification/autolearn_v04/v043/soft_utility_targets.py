"""Soft-utility target comparison for the v0.4.3 repair release.

Compares three routing-policy training objectives that differ in how
they use measured counterfactual utilities to form training targets:

  1. **Hard-label target**: one-hot at the argmax action (current
     production objective).
  2. **Soft-utility target**: convert utilities into a soft target
     distribution via temperature ``tau``::

         q_i(a) = softmax(U_{i,a} / tau)

     The temperature is selected on D_validation only from the
     candidate set ``{0.01, 0.05, 0.1, 0.2, 0.5, 1.0}``.
  3. **Tie-aware target**: if the top-two utility margin is below
     ``tie_threshold`` (0.1), distribute probability between the top
     actions; otherwise use a sharper (one-hot) target.

All three models are trained with custom gradient descent mirroring
the ``train_logistic`` function from v042 ``sham_trainer.py``::

    W shape: (n_features, n_actions)
    logits  = features @ W + b
    probs   = softmax(logits)
    gradient (hard)  = features.T @ (probs - one_hot)      / n
    gradient (soft)  = features.T @ (probs - soft_targets) / n

Models are trained on D_experience and evaluated on D_validation.
D_test is never used for selection.

The output of :func:`compare_soft_targets` is a JSON-serializable dict
suitable for writing to ``soft_utility_results.json``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

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

#: Candidate temperatures for the soft-utility router.
TAU_CANDIDATES: list[float] = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

#: Margin below which a task is considered a near-tie.
TIE_THRESHOLD: float = 0.1

#: Temperature used for distributing probability in tie-aware targets.
TIE_SOFTMAX_TAU: float = 0.5

#: Practical improvement threshold for production replacement justification.
PRODUCTION_REPLACEMENT_THRESHOLD: float = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _extract_features(examples: list[dict[str, Any]]) -> np.ndarray:
    """Extract feature matrix from a list of examples."""
    return np.array([ex["features"] for ex in examples], dtype=float)


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
        oracle_action = run.get_oracle_action(tid)
        if oracle_action is not None and oracle_action in action_idx:
            labels[i] = action_idx[oracle_action]
    return labels


def _build_hard_targets(
    hard_labels: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    """Build one-hot targets from hard labels."""
    n = len(hard_labels)
    one_hot = np.zeros((n, n_actions), dtype=float)
    one_hot[np.arange(n), hard_labels] = 1.0
    return one_hot


def _build_soft_utility_targets(
    utility_matrix: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Build soft-utility targets: q_i(a) = softmax(U_{i,a} / tau)."""
    scaled_u = utility_matrix / tau
    return _softmax(scaled_u)


def _build_tie_aware_targets(
    utility_matrix: np.ndarray,
    tie_threshold: float = TIE_THRESHOLD,
    tie_softmax_tau: float = TIE_SOFTMAX_TAU,
) -> np.ndarray:
    """Build tie-aware targets.

    For each task:
      - If the top-two utility margin < ``tie_threshold``, distribute
        probability between the top-2 actions using softmax of their
        utilities with ``tie_softmax_tau``.
      - Otherwise, use a one-hot (sharper) target at the argmax action.
    """
    n, n_actions = utility_matrix.shape
    targets = np.zeros_like(utility_matrix)

    for i in range(n):
        u = utility_matrix[i]
        sorted_idx = np.argsort(u)[::-1]
        top1 = int(sorted_idx[0])
        top2 = int(sorted_idx[1]) if n_actions > 1 else top1
        margin = u[top1] - u[top2]

        if margin < tie_threshold and n_actions > 1:
            # Distribute probability between top-2 actions.
            top_u = np.array([u[top1], u[top2]])
            top_probs = _softmax(top_u / tie_softmax_tau)
            targets[i, top1] = top_probs[0]
            targets[i, top2] = top_probs[1]
        else:
            # Sharper target: one-hot at argmax.
            targets[i, top1] = 1.0

    return targets


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


def _make_logistic_policy_fn(
    W: np.ndarray,
    b: np.ndarray,
    feature_lookup: dict[str, np.ndarray],
    actions: list[str],
) -> PolicyDecisionFn:
    """Create a PolicyDecisionFn from logistic weights and bias."""

    def fn(tid: str, task: TaskInfo) -> str | None:
        feats = feature_lookup.get(tid)
        if feats is None or feats.size == 0:
            return task.allowed_actions[0] if task.allowed_actions else None
        if len(feats) != W.shape[0]:
            return task.allowed_actions[0] if task.allowed_actions else None
        logits = feats @ W + b
        pred_idx = int(np.argmax(logits))
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


# ---------------------------------------------------------------------------
# Custom gradient-descent trainers
# ---------------------------------------------------------------------------


def _train_logistic_gd(
    features: np.ndarray,
    targets: np.ndarray,
    n_actions: int,
    n_epochs: int = 200,
    lr: float = 0.01,
    l2: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Train multinomial logistic regression with arbitrary targets via GD.

    This mirrors ``train_logistic`` from v042 ``sham_trainer.py`` but
    accepts any target distribution (one-hot for hard labels, soft
    distributions for soft-utility or tie-aware targets).

    W shape: (n_features, n_actions)
    logits  = features @ W + b
    probs   = softmax(logits)
    gradient = features.T @ (probs - targets) / n  (+ L2 on W)

    Returns (weights [n_features, n_actions], bias [n_actions]).
    """
    rng = np.random.default_rng(seed)
    n, d = features.shape
    W = rng.standard_normal((d, n_actions)) * 0.01
    b = np.zeros(n_actions)

    for _ in range(n_epochs):
        logits = features @ W + b
        probs = _softmax(logits)
        diff = probs - targets
        grad_W = features.T @ diff / n + l2 * W
        grad_b = diff.mean(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b

    return W, b


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_model_on_validation(
    W: np.ndarray,
    b: np.ndarray,
    feature_lookup: dict[str, np.ndarray],
    val_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
    sham_result: Any,
    baseline_result: Any,
    oracle_result: Any,
    model_id: str,
) -> dict[str, Any]:
    """Evaluate a trained logistic model on validation and compute metrics."""
    policy_fn = _make_logistic_policy_fn(W, b, feature_lookup, actions)
    result = evaluate_policy_from_counterfactuals(
        run, val_task_ids, policy_fn, model_id,
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
        candidate_policy_id=model_id,
    )

    headroom_vs_baseline = compute_headroom_metrics(
        candidate_mean=mean_utility,
        reference_mean=baseline_mean,
        oracle_mean=oracle_mean,
        n_tasks=result.n_complete,
        task_ids_digest=result.task_ids_digest,
        reference_policy_id="baseline",
        candidate_policy_id=model_id,
    )

    return {
        "validation_expected_utility": float(mean_utility),
        "validation_candidate_minus_sham": float(mean_utility - sham_mean),
        "validation_candidate_minus_baseline": float(mean_utility - baseline_mean),
        "validation_regret": float(result.mean_regret) if result.mean_regret is not None else None,
        "validation_headroom_recovered_vs_sham": headroom_vs_sham,
        "validation_headroom_recovered_vs_baseline": headroom_vs_baseline,
        "validation_action_distribution": dict(result.action_distribution),
        "n_validation_tasks": result.n_tasks,
        "n_validation_complete": result.n_complete,
    }


# ---------------------------------------------------------------------------
# Temperature selection
# ---------------------------------------------------------------------------


def _select_tau(
    train_features: np.ndarray,
    train_utility: np.ndarray,
    n_actions: int,
    feature_lookup: dict[str, np.ndarray],
    val_task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
    sham_result: Any,
    baseline_result: Any,
    oracle_result: Any,
    tau_candidates: list[float],
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Select the best temperature tau on D_validation.

    Trains a soft-utility router for each candidate tau, evaluates on
    validation, and returns the tau with the highest validation utility.

    Returns (best_tau, {tau_str: val_utility}).
    """
    val_results: dict[str, float] = {}
    best_tau = tau_candidates[0]
    best_val = -np.inf

    for tau in tau_candidates:
        soft_targets = _build_soft_utility_targets(train_utility, tau)
        W, b = _train_logistic_gd(
            train_features, soft_targets, n_actions, seed=seed,
        )
        metrics = _evaluate_model_on_validation(
            W, b, feature_lookup, val_task_ids, run, actions,
            sham_result, baseline_result, oracle_result,
            f"soft_utility_tau_{tau}",
        )
        val_util = metrics["validation_expected_utility"]
        val_results[f"tau_{tau}"] = val_util
        if val_util > best_val:
            best_val = val_util
            best_tau = tau

    return best_tau, val_results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compare_soft_targets(run: LoadedRun, seed: int = 42) -> dict[str, Any]:
    """Compare soft-utility training targets for routing policies.

    Trains three logistic routers with different target formulations
    (hard-label, soft-utility, tie-aware) on D_experience using custom
    gradient descent, selects the temperature ``tau`` for the
    soft-utility router on D_validation, and compares all three on
    validation.

    Args:
        run: loaded run data.
        seed: random seed for reproducibility.

    Returns:
        JSON-serializable dict with per-model validation metrics,
        the best tau, an interpretation, and a flag indicating
        whether production replacement is justified.  Suitable for
        writing to ``soft_utility_results.json``.
    """
    actions = run.all_actions
    n_actions = len(actions)

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

    train_features = _extract_features(train_examples)
    train_task_ids = [ex["task_id"] for ex in train_examples]
    val_task_ids = [ex["task_id"] for ex in val_examples]

    # Build utility matrix and labels from measured outcomes.
    train_utility = _build_utility_matrix(run, train_task_ids, actions)
    train_hard_labels = _build_hard_labels(run, train_task_ids, actions)

    # Feature lookup for policy decision functions.
    feature_lookup = _build_feature_lookup(run)

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
    # Model 1: Hard-label router (current production objective).
    # ------------------------------------------------------------------
    hard_targets = _build_hard_targets(train_hard_labels, n_actions)
    W_hard, b_hard = _train_logistic_gd(
        train_features, hard_targets, n_actions, seed=seed,
    )
    hard_metrics = _evaluate_model_on_validation(
        W_hard, b_hard, feature_lookup, val_task_ids, run, actions,
        sham_val, baseline_val, oracle_val, "hard_label",
    )

    # ------------------------------------------------------------------
    # Model 2: Soft-utility router (select tau on validation).
    # ------------------------------------------------------------------
    best_tau, tau_val_results = _select_tau(
        train_features, train_utility, n_actions,
        feature_lookup, val_task_ids, run, actions,
        sham_val, baseline_val, oracle_val,
        TAU_CANDIDATES, seed,
    )

    soft_targets = _build_soft_utility_targets(train_utility, best_tau)
    W_soft, b_soft = _train_logistic_gd(
        train_features, soft_targets, n_actions, seed=seed,
    )
    soft_metrics = _evaluate_model_on_validation(
        W_soft, b_soft, feature_lookup, val_task_ids, run, actions,
        sham_val, baseline_val, oracle_val, "soft_utility",
    )

    # ------------------------------------------------------------------
    # Model 3: Tie-aware soft router.
    # ------------------------------------------------------------------
    tie_targets = _build_tie_aware_targets(train_utility)
    W_tie, b_tie = _train_logistic_gd(
        train_features, tie_targets, n_actions, seed=seed,
    )
    tie_metrics = _evaluate_model_on_validation(
        W_tie, b_tie, feature_lookup, val_task_ids, run, actions,
        sham_val, baseline_val, oracle_val, "tie_aware_soft",
    )

    # ------------------------------------------------------------------
    # Summarize results.
    # ------------------------------------------------------------------
    models: dict[str, dict[str, Any]] = {
        "hard_label": {
            "description": (
                "Hard-label router (current): train on one-hot argmax "
                "best-action labels with cross-entropy loss."
            ),
            "is_current": True,
            **hard_metrics,
        },
        "soft_utility": {
            "description": (
                "Soft-utility router: convert utilities into soft target "
                "distribution via temperature tau, train with cross-entropy "
                "against soft targets."
            ),
            "selected_tau": float(best_tau),
            "tau_candidates": TAU_CANDIDATES,
            "tau_validation_utilities": tau_val_results,
            **soft_metrics,
        },
        "tie_aware_soft": {
            "description": (
                "Tie-aware soft router: distribute probability between top "
                "actions when the utility margin is small; use sharper "
                "(one-hot) target otherwise."
            ),
            "tie_threshold": TIE_THRESHOLD,
            "tie_softmax_tau": TIE_SOFTMAX_TAU,
            **tie_metrics,
        },
    }

    # ------------------------------------------------------------------
    # Select best model on validation only.
    # ------------------------------------------------------------------
    val_utils = {
        name: data["validation_expected_utility"]
        for name, data in models.items()
    }
    best_model = max(val_utils, key=val_utils.get)

    # Compute gains over the current hard-label objective.
    soft_gain = soft_metrics["validation_expected_utility"] - hard_metrics["validation_expected_utility"]
    tie_gain = tie_metrics["validation_expected_utility"] - hard_metrics["validation_expected_utility"]
    best_gain = max(soft_gain, tie_gain)

    # ------------------------------------------------------------------
    # Production replacement justification.
    # ------------------------------------------------------------------
    production_replacement_justified = bool(
        best_gain > PRODUCTION_REPLACEMENT_THRESHOLD
        and best_model != "hard_label"
    )

    # ------------------------------------------------------------------
    # Interpretation.
    # ------------------------------------------------------------------
    if best_gain > 0.05:
        conclusion = "soft_targets_help"
        note = (
            f"A soft-target model ({best_model}) materially improves "
            f"validation utility by {best_gain:.3f} over the hard-label "
            f"router.  The current objective leaves signal on the table."
        )
    elif best_gain > 0.01:
        conclusion = "soft_targets_marginal"
        note = (
            f"A soft-target model ({best_model}) marginally improves "
            f"validation utility by {best_gain:.3f}.  There is modest "
            f"benefit to utility-aware training."
        )
    else:
        conclusion = "hard_label_sufficient"
        note = (
            "No soft-target model materially improves over the hard-label "
            "router on validation.  The current objective is sufficient, "
            "or the utility signal is too weak for soft targets to help."
        )

    return {
        "seed": seed,
        "n_train": len(train_examples),
        "n_validation": len(val_examples),
        "actions": actions,
        "models": models,
        "best_tau": float(best_tau),
        "best_model": best_model,
        "gains_over_hard_label": {
            "soft_utility_validation_gain": float(soft_gain),
            "tie_aware_validation_gain": float(tie_gain),
            "best_validation_gain": float(best_gain),
        },
        "interpretation": {
            "conclusion": conclusion,
            "note": note,
            "selected_tau": float(best_tau),
            "best_model": best_model,
        },
        "production_replacement_justified": production_replacement_justified,
    }
