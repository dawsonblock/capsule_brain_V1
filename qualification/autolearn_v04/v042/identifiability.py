"""Benchmark identifiability audit.

Trains a series of diagnostic models on restricted feature sets to test
whether the benchmark's best-action signal is identifiable from surface
features alone, or whether it is template-driven (determinable from
family or archetype without genuine state-dependent reasoning).

Diagnostic models trained:
  - family only (one-hot encoded)
  - archetype only (one-hot encoded)
  - prompt length only
  - available-action mask only
  - action-cost priors only (if available)
  - marginal action frequency
  - full executive features

Validation is used for model selection; test is used for final
comparison. If family-only or archetype-only achieves nearly the same
utility as the full candidate, the benchmark may be template-driven.

Additionally tests whether hidden environment state changes the best
action within the same prompt archetype. Looks for crossover examples
(``crossover_type != "none"``) where the same or near-identical visible
prompt has a different hidden environment condition and a different
measured optimal action.

The output of :func:`audit_identifiability` is the content for
``benchmark_identifiability.json``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .data import LoadedRun
from .sham_trainer import train_logistic

try:
    from sklearn.linear_model import LogisticRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


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

#: Utility threshold below which a diagnostic model is considered to
#: achieve "nearly the same" utility as the full candidate.
TEMPLATE_UTILITY_THRESHOLD: float = 0.05

#: Fraction of the full candidate's utility that a diagnostic model must
#: achieve to be flagged as a potential template signal.
TEMPLATE_FRACTION_THRESHOLD: float = 0.90


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


def _get_best_action(run: LoadedRun, task_id: str) -> str | None:
    """Get the measured best action for a task."""
    pairs = _sorted_utility_pairs(run, task_id)
    return pairs[0][0] if pairs else None


def _one_hot_encode(
    values: list[str], categories: list[str]
) -> np.ndarray:
    """One-hot encode a list of string values against a category list."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    encoded = np.zeros((len(values), len(categories)))
    for i, v in enumerate(values):
        if v in cat_idx:
            encoded[i, cat_idx[v]] = 1.0
    return encoded


def _evaluate_policy_utility(
    run: LoadedRun,
    task_ids: list[str],
    predictions: list[str],
) -> float:
    """Evaluate mean utility of predicted actions on the given tasks."""
    utilities: list[float] = []
    for tid, pred_action in zip(task_ids, predictions):
        task = run.tasks.get(tid)
        if task is None:
            continue
        if pred_action not in task.allowed_actions:
            # If predicted action is not allowed, fall back to first allowed.
            pred_action = task.allowed_actions[0] if task.allowed_actions else None
        if pred_action is None:
            utilities.append(0.0)
            continue
        u = run.get_utility_for_action(tid, pred_action)
        utilities.append(u if u is not None else 0.0)
    return _safe_mean(utilities)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


def _extract_family_features(
    run: LoadedRun, task_ids: list[str]
) -> np.ndarray:
    """One-hot encode task family."""
    families = [run.tasks[tid].family for tid in task_ids]
    return _one_hot_encode(families, run.all_families)


def _extract_archetype_features(
    run: LoadedRun, task_ids: list[str]
) -> np.ndarray:
    """One-hot encode task archetype."""
    archetypes = [run.tasks[tid].archetype for tid in task_ids]
    return _one_hot_encode(archetypes, run.all_archetypes)


def _extract_prompt_length_features(
    run: LoadedRun, task_ids: list[str]
) -> np.ndarray:
    """Extract prompt length as a single feature (normalized)."""
    lengths = np.array([len(run.tasks[tid].prompt) for tid in task_ids])
    if lengths.std() > 1e-10:
        lengths = (lengths - lengths.mean()) / lengths.std()
    return lengths.reshape(-1, 1)


def _extract_action_mask_features(
    run: LoadedRun, task_ids: list[str]
) -> np.ndarray:
    """Extract available-action mask (binary vector over canonical actions)."""
    features = np.zeros((len(task_ids), len(CANONICAL_ACTIONS)))
    for i, tid in enumerate(task_ids):
        task = run.tasks[tid]
        for j, action in enumerate(CANONICAL_ACTIONS):
            if action in task.allowed_actions:
                features[i, j] = 1.0
    return features


def _extract_action_cost_features(
    run: LoadedRun, task_ids: list[str]
) -> np.ndarray | None:
    """Extract action-cost priors if available in setup_spec.

    Returns None if no cost information is present in any task.
    """
    cost_keys = ["action_costs", "costs", "action_priors"]
    has_costs = False
    for tid in task_ids:
        task = run.tasks[tid]
        for key in cost_keys:
            if key in task.setup_spec:
                has_costs = True
                break
        if has_costs:
            break

    if not has_costs:
        return None

    features = np.zeros((len(task_ids), len(CANONICAL_ACTIONS)))
    for i, tid in enumerate(task_ids):
        task = run.tasks[tid]
        costs: dict[str, float] = {}
        for key in cost_keys:
            if key in task.setup_spec:
                val = task.setup_spec[key]
                if isinstance(val, dict):
                    costs.update(val)
                elif isinstance(val, list):
                    for j, c in enumerate(val):
                        if j < len(CANONICAL_ACTIONS):
                            costs[CANONICAL_ACTIONS[j]] = float(c)
                break
        for j, action in enumerate(CANONICAL_ACTIONS):
            features[i, j] = costs.get(action, 0.0)

    # Normalize.
    if features.std() > 1e-10:
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)
    return features


def _extract_full_features(
    run: LoadedRun, task_ids: list[str]
) -> np.ndarray:
    """Extract the full executive feature vector from dataset examples."""
    # Build a lookup from task_id to example features.
    feature_lookup: dict[str, list[float]] = {}
    for ex in run.train_examples + run.validation_examples + run.test_examples + run.ood_examples:
        tid = ex.get("task_id")
        if tid and "features" in ex:
            feature_lookup[tid] = ex["features"]

    # Determine feature dimension.
    dim = 0
    for f in feature_lookup.values():
        dim = max(dim, len(f))

    features = np.zeros((len(task_ids), dim))
    for i, tid in enumerate(task_ids):
        f = feature_lookup.get(tid)
        if f:
            features[i, :len(f)] = f
    return features


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------


def _train_sklearn_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    actions: list[str],
    C_values: list[float] | None = None,
) -> tuple[Any, float, list[str]]:
    """Train sklearn LogisticRegression with validation-based C selection.

    Returns (best_model, best_val_accuracy, test_predictions).
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    best_model = None
    best_val_acc = -1.0

    for C in C_values:
        model = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        try:
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_acc = float(np.mean(val_pred == y_val))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
        except Exception:
            continue

    if best_model is None:
        # Fallback: default C.
        best_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        try:
            best_model.fit(X_train, y_train)
        except Exception:
            # Return dummy predictions.
            majority = Counter(y_train).most_common(1)[0][0] if len(y_train) else actions[0]
            return None, 0.0, [majority] * len(X_test)

    test_pred = best_model.predict(X_test)
    # Map sklearn class indices to action names using classes_ (which may
    # be a subset of all actions if some actions never appear in training).
    classes = list(best_model.classes_)
    test_pred_actions = []
    for p in test_pred:
        if p in classes:
            idx = classes.index(p)
            if idx < len(actions):
                test_pred_actions.append(actions[idx])
            else:
                test_pred_actions.append(actions[0])
        else:
            test_pred_actions.append(actions[0])
    return best_model, best_val_acc, test_pred_actions


def _train_numpy_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    actions: list[str],
) -> tuple[dict[str, Any], float, list[str]]:
    """Train numpy-based logistic regression with validation-based selection.

    Uses :func:`train_logistic` from ``.sham_trainer``.
    """
    lr_values = [0.001, 0.01, 0.1]
    l2_values = [0.001, 0.01, 0.1]
    best_W, best_b = None, None
    best_val_acc = -1.0

    for lr in lr_values:
        for l2 in l2_values:
            W, b = train_logistic(
                X_train, y_train, actions,
                n_epochs=200, lr=lr, l2=l2, seed=42,
            )
            # Validate.
            logits = X_val @ W + b
            val_pred_idx = np.argmax(logits, axis=1)
            val_pred = np.array([actions[i] for i in val_pred_idx])
            val_acc = float(np.mean(val_pred == y_val))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_W, best_b = W, b

    if best_W is None:
        W, b = train_logistic(X_train, y_train, actions, seed=42)
        best_W, best_b = W, b

    # Test predictions.
    test_logits = X_test @ best_W + best_b
    test_pred_idx = np.argmax(test_logits, axis=1)
    test_pred_actions = [actions[i] for i in test_pred_idx]

    return {
        "weights": best_W.tolist(),
        "bias": best_b.tolist(),
    }, best_val_acc, test_pred_actions


def _train_diagnostic_model(
    run: LoadedRun,
    feature_extractor: Any,
    model_name: str,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    actions: list[str],
) -> dict[str, Any]:
    """Train a diagnostic model with the given feature extractor.

    Uses validation for model selection, test for final evaluation.
    Falls back to numpy logistic regression if sklearn is unavailable.
    """
    # Extract features.
    X_train = feature_extractor(run, train_ids)
    X_val = feature_extractor(run, val_ids)
    X_test = feature_extractor(run, test_ids)

    # Extract labels (best action).
    y_train = np.array([_get_best_action(run, tid) or actions[0] for tid in train_ids])
    y_val = np.array([_get_best_action(run, tid) or actions[0] for tid in val_ids])
    y_test = np.array([_get_best_action(run, tid) or actions[0] for tid in test_ids])

    # Check for degenerate case (single class).
    if len(set(y_train)) < 2:
        majority = Counter(y_train).most_common(1)[0][0]
        test_pred = [majority] * len(test_ids)
        val_acc = float(np.mean(y_val == majority)) if len(y_val) else 0.0
        test_acc = float(np.mean(y_test == majority)) if len(y_test) else 0.0
        test_utility = _evaluate_policy_utility(run, test_ids, test_pred)
        return {
            "model_name": model_name,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "n_features": X_train.shape[1],
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "test_utility": test_utility,
            "test_predictions": test_pred,
            "degenerate": True,
            "single_class": majority,
            "backend": "degenerate",
        }

    # Train model.
    if _HAS_SKLEARN:
        model, val_acc, test_pred = _train_sklearn_logistic(
            X_train, y_train, X_val, y_val, X_test, actions,
        )
        backend = "sklearn"
        model_info = {"C": getattr(model, "C", 1.0)} if model else {}
    else:
        model_info, val_acc, test_pred = _train_numpy_logistic(
            X_train, y_train, X_val, y_val, X_test, actions,
        )
        backend = "numpy"

    test_acc = float(np.mean(np.array(test_pred) == y_test)) if len(y_test) else 0.0
    test_utility = _evaluate_policy_utility(run, test_ids, test_pred)

    return {
        "model_name": model_name,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_features": X_train.shape[1],
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_utility": test_utility,
        "test_predictions": test_pred,
        "degenerate": False,
        "backend": backend,
        "model_info": model_info,
    }


# ---------------------------------------------------------------------------
# Marginal action frequency baseline
# ---------------------------------------------------------------------------


def _marginal_frequency_baseline(
    run: LoadedRun,
    train_ids: list[str],
    test_ids: list[str],
) -> dict[str, Any]:
    """Marginal action frequency baseline (no features, just distribution)."""
    best_actions = [_get_best_action(run, tid) for tid in train_ids]
    best_actions = [a for a in best_actions if a is not None]
    counts = Counter(best_actions)
    total = sum(counts.values())
    probs = {a: counts.get(a, 0) / total for a in CANONICAL_ACTIONS} if total else {}

    # Predict the most frequent action for all test tasks.
    majority = counts.most_common(1)[0][0] if counts else CANONICAL_ACTIONS[0]
    test_pred = [majority] * len(test_ids)
    test_utility = _evaluate_policy_utility(run, test_ids, test_pred)

    y_test = np.array([_get_best_action(run, tid) or CANONICAL_ACTIONS[0] for tid in test_ids])
    test_acc = float(np.mean(np.array(test_pred) == y_test)) if len(y_test) else 0.0

    return {
        "model_name": "marginal_action_frequency",
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "n_features": 0,
        "test_accuracy": test_acc,
        "test_utility": test_utility,
        "majority_action": majority,
        "action_distribution": probs,
        "backend": "marginal",
        "degenerate": len(counts) < 2,
    }


# ---------------------------------------------------------------------------
# Crossover analysis
# ---------------------------------------------------------------------------


def _crossover_analysis(run: LoadedRun) -> dict[str, Any]:
    """Analyze crossover examples where hidden state changes the best action.

    Looks for tasks with ``crossover_type != "none"`` and groups them by
    archetype to find cases where the same (or near-identical) visible
    prompt has a different hidden environment condition and a different
    measured optimal action.
    """
    crossover_tasks = [
        tid for tid, t in run.tasks.items()
        if t.crossover_type and t.crossover_type != "none"
    ]

    # Group by archetype + prompt to find near-identical visible prompts.
    archetype_prompt_groups: dict[str, list[str]] = defaultdict(list)
    for tid in crossover_tasks:
        task = run.tasks[tid]
        # Use a truncated prompt hash as a proxy for "near-identical".
        prompt_key = task.prompt[:200]  # First 200 chars as visible prompt proxy.
        key = f"{task.archetype}::{prompt_key}"
        archetype_prompt_groups[key].append(tid)

    # Find groups with multiple tasks (same visible prompt, different tasks).
    crossover_groups: list[dict[str, Any]] = []
    for key, tids in archetype_prompt_groups.items():
        if len(tids) < 2:
            continue
        # Check if best actions differ.
        best_actions = {}
        for tid in tids:
            ba = _get_best_action(run, tid)
            if ba:
                best_actions[tid] = ba

        unique_actions = set(best_actions.values())
        if len(unique_actions) > 1:
            # This is a true crossover: same visible prompt, different
            # hidden state, different optimal action.
            archetype = key.split("::")[0]
            crossover_groups.append({
                "group_key": key,
                "archetype": archetype,
                "task_ids": tids,
                "best_actions": best_actions,
                "crossover_types": [run.tasks[tid].crossover_type for tid in tids],
                "n_tasks": len(tids),
                "n_unique_best_actions": len(unique_actions),
                "unique_best_actions": sorted(unique_actions),
            })

    # Also collect all crossover tasks with their best actions.
    crossover_details: list[dict[str, Any]] = []
    for tid in crossover_tasks:
        task = run.tasks[tid]
        ba = _get_best_action(run, tid)
        crossover_details.append({
            "task_id": tid,
            "family": task.family,
            "archetype": task.archetype,
            "crossover_type": task.crossover_type,
            "group_id": task.group_id,
            "best_action": ba,
            "prompt_length": len(task.prompt),
        })

    # Crossover type distribution.
    crossover_type_counts = Counter(run.tasks[tid].crossover_type for tid in crossover_tasks)

    return {
        "n_crossover_tasks": len(crossover_tasks),
        "n_crossover_groups_with_action_change": len(crossover_groups),
        "crossover_type_distribution": dict(crossover_type_counts),
        "crossover_groups": crossover_groups,
        "crossover_task_details": crossover_details,
        "interpretation": _interpret_crossover(crossover_tasks, crossover_groups),
    }


def _interpret_crossover(
    crossover_tasks: list[str],
    crossover_groups: list[dict[str, Any]],
) -> str:
    """Generate a human-readable interpretation of crossover findings."""
    if not crossover_tasks:
        return (
            "No crossover tasks found (all tasks have crossover_type='none'). "
            "The benchmark does not appear to test whether hidden environment "
            "state changes the optimal action within the same prompt archetype."
        )
    if not crossover_groups:
        return (
            f"Found {len(crossover_tasks)} crossover tasks but no groups where "
            "the same visible prompt has different hidden state AND different "
            "optimal action. The hidden state may not be changing the best "
            "action, or the visible prompts may not be near-identical enough."
        )
    return (
        f"Found {len(crossover_tasks)} crossover tasks and "
        f"{len(crossover_groups)} groups where the same/near-identical visible "
        "prompt has different hidden environment conditions and different "
        "measured optimal actions. This indicates the benchmark does test "
        "state-dependent routing beyond surface template features."
    )


# ---------------------------------------------------------------------------
# Template-driven detection
# ---------------------------------------------------------------------------


def _template_detection(
    diagnostic_results: dict[str, Any],
    candidate_utility: float,
) -> dict[str, Any]:
    """Determine if the benchmark may be template-driven.

    If family-only or archetype-only achieves nearly the same utility as
    the full candidate, the benchmark may be template-driven.
    """
    family_utility = diagnostic_results.get("family_only", {}).get("test_utility", 0.0)
    archetype_utility = diagnostic_results.get("archetype_only", {}).get("test_utility", 0.0)

    # Absolute difference.
    family_gap = candidate_utility - family_utility
    archetype_gap = candidate_utility - archetype_utility

    # Fraction of candidate utility achieved.
    family_fraction = family_utility / candidate_utility if abs(candidate_utility) > 1e-10 else 0.0
    archetype_fraction = archetype_utility / candidate_utility if abs(candidate_utility) > 1e-10 else 0.0

    is_template_driven = (
        family_gap < TEMPLATE_UTILITY_THRESHOLD
        or archetype_gap < TEMPLATE_UTILITY_THRESHOLD
        or family_fraction >= TEMPLATE_FRACTION_THRESHOLD
        or archetype_fraction >= TEMPLATE_FRACTION_THRESHOLD
    )

    if is_template_driven:
        reason = (
            f"Family-only (utility={family_utility:.3f}, fraction={family_fraction:.3f}) "
            f"or archetype-only (utility={archetype_utility:.3f}, fraction={archetype_fraction:.3f}) "
            f"achieves nearly the same utility as the full candidate "
            f"(utility={candidate_utility:.3f}). The benchmark may be "
            f"template-driven: the best action is largely determined by "
            f"surface family/archetype rather than state-dependent reasoning."
        )
    else:
        reason = (
            f"Family-only (utility={family_utility:.3f}, fraction={family_fraction:.3f}) "
            f"and archetype-only (utility={archetype_utility:.3f}, fraction={archetype_fraction:.3f}) "
            f"do not achieve the same utility as the full candidate "
            f"(utility={candidate_utility:.3f}). The benchmark appears to "
            f"require state-dependent reasoning beyond surface templates."
        )

    return {
        "is_template_driven": is_template_driven,
        "candidate_utility": candidate_utility,
        "family_only_utility": family_utility,
        "archetype_only_utility": archetype_utility,
        "family_gap": family_gap,
        "archetype_gap": archetype_gap,
        "family_fraction_of_candidate": family_fraction,
        "archetype_fraction_of_candidate": archetype_fraction,
        "thresholds": {
            "utility_threshold": TEMPLATE_UTILITY_THRESHOLD,
            "fraction_threshold": TEMPLATE_FRACTION_THRESHOLD,
        },
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_identifiability(run: LoadedRun) -> dict[str, Any]:
    """Audit benchmark identifiability with diagnostic feature-restricted models.

    Trains diagnostic models using progressively richer feature sets:
    family only, archetype only, prompt length only, available-action
    mask only, action-cost priors only (if available), marginal action
    frequency, and full executive features. Validation is used for model
    selection; test is used for final comparison.

    If family-only or archetype-only achieves nearly the same utility as
    the full candidate, the benchmark may be template-driven.

    Also tests whether hidden environment state changes the best action
    within the same prompt archetype by looking for crossover examples
    (``crossover_type != "none"``) where the same/near-identical visible
    prompt has different hidden environment conditions and different
    measured optimal actions.

    Args:
        run: Loaded run data with tasks, outcomes, and dataset examples.

    Returns:
        Content for ``benchmark_identifiability.json``.
    """
    actions = run.all_actions if run.all_actions else CANONICAL_ACTIONS

    # Determine train/val/test task IDs from dataset splits.
    train_ids = [ex["task_id"] for ex in run.train_examples if "task_id" in ex]
    val_ids = [ex["task_id"] for ex in run.validation_examples if "task_id" in ex]
    test_ids = [ex["task_id"] for ex in run.test_examples if "task_id" in ex]

    # Filter to tasks that actually exist in the run and have outcomes.
    train_ids = [tid for tid in train_ids if tid in run.tasks and run.outcomes_by_task.get(tid)]
    val_ids = [tid for tid in val_ids if tid in run.tasks and run.outcomes_by_task.get(tid)]
    test_ids = [tid for tid in test_ids if tid in run.tasks and run.outcomes_by_task.get(tid)]

    # If no explicit splits, fall back to using all tasks with outcomes.
    if not train_ids:
        all_ids = [tid for tid in run.tasks if run.outcomes_by_task.get(tid)]
        n = len(all_ids)
        if n >= 3:
            train_ids = all_ids[: n // 2]
            val_ids = all_ids[n // 2 : 3 * n // 4]
            test_ids = all_ids[3 * n // 4 :]
        elif n > 0:
            train_ids = all_ids
            val_ids = all_ids
            test_ids = all_ids

    # Train diagnostic models.
    feature_extractors: dict[str, Any] = {
        "family_only": _extract_family_features,
        "archetype_only": _extract_archetype_features,
        "prompt_length_only": _extract_prompt_length_features,
        "available_action_mask_only": _extract_action_mask_features,
        "full_executive_features": _extract_full_features,
    }

    diagnostic_results: dict[str, Any] = {}

    for name, extractor in feature_extractors.items():
        diagnostic_results[name] = _train_diagnostic_model(
            run, extractor, name, train_ids, val_ids, test_ids, actions,
        )

    # Action-cost priors (if available).
    cost_features = _extract_action_cost_features(run, train_ids)
    if cost_features is not None:
        diagnostic_results["action_cost_priors_only"] = _train_diagnostic_model(
            run, _extract_action_cost_features, "action_cost_priors_only",
            train_ids, val_ids, test_ids, actions,
        )
    else:
        diagnostic_results["action_cost_priors_only"] = {
            "model_name": "action_cost_priors_only",
            "available": False,
            "reason": "no action cost fields found in setup_spec",
        }

    # Marginal action frequency baseline.
    diagnostic_results["marginal_action_frequency"] = _marginal_frequency_baseline(
        run, train_ids, test_ids,
    )

    # Candidate utility for comparison (full executive features model).
    candidate_utility = diagnostic_results.get("full_executive_features", {}).get("test_utility", 0.0)

    # Template-driven detection.
    template_analysis = _template_detection(diagnostic_results, candidate_utility)

    # Crossover analysis.
    crossover = _crossover_analysis(run)

    # Summary table of diagnostic model utilities.
    utility_summary: list[dict[str, Any]] = []
    for name in [
        "family_only",
        "archetype_only",
        "prompt_length_only",
        "available_action_mask_only",
        "action_cost_priors_only",
        "marginal_action_frequency",
        "full_executive_features",
    ]:
        res = diagnostic_results.get(name, {})
        utility_summary.append({
            "model": name,
            "test_utility": res.get("test_utility", 0.0),
            "test_accuracy": res.get("test_accuracy", 0.0),
            "n_features": res.get("n_features", 0),
            "available": res.get("available", True),
        })

    return {
        "n_train_tasks": len(train_ids),
        "n_val_tasks": len(val_ids),
        "n_test_tasks": len(test_ids),
        "train_task_ids": train_ids,
        "val_task_ids": val_ids,
        "test_task_ids": test_ids,
        "sklearn_available": _HAS_SKLEARN,
        "diagnostic_models": diagnostic_results,
        "utility_summary": utility_summary,
        "template_analysis": template_analysis,
        "crossover_analysis": crossover,
        "summary": {
            "is_template_driven": template_analysis["is_template_driven"],
            "candidate_utility": candidate_utility,
            "family_only_utility": template_analysis["family_only_utility"],
            "archetype_only_utility": template_analysis["archetype_only_utility"],
            "n_crossover_tasks": crossover["n_crossover_tasks"],
            "n_crossover_groups_with_action_change": crossover["n_crossover_groups_with_action_change"],
            "description": (
                "Diagnostic models with restricted feature sets are trained "
                "to test whether the benchmark's best-action signal is "
                "identifiable from surface features alone. If family-only or "
                "archetype-only models achieve nearly the same utility as the "
                "full candidate, the benchmark may be template-driven. "
                "Crossover analysis tests whether hidden environment state "
                "changes the optimal action within the same prompt archetype."
            ),
        },
    }
