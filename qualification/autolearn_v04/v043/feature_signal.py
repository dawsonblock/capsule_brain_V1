"""Feature signal validation for v0.4.3.

Validates whether the full executive state adds predictive value beyond
simple controls (intercept, action frequency, task family, task
archetype).

A ladder of nine models is trained on ``D_experience`` and selected on
``D_validation``:

1. intercept only (``DummyClassifier``)
2. marginal action frequency (predict most common action)
3. family only (one-hot family -> logistic)
4. archetype only (one-hot archetype -> logistic)
5. family + archetype (one-hot both -> logistic)
6. features excluding family (all features minus family-derived ones)
7. features excluding archetype
8. full feature model (all features -> logistic)
9. shuffled feature model (feature shuffle sham)

For each model the module reports action accuracy, balanced accuracy,
cross entropy, expected utility, regret, headroom recovered,
candidate-minus-frequency, and candidate-minus-family-majority.

The primary learnability conclusion is one of:

* ``FEATURE_SIGNAL_PRESENT``  — features excluding family/archetype
  materially improve expected utility over family-only and frequency
  controls (improvement > 0.05).
* ``FEATURE_SIGNAL_WEAK``     — only small unstable gains
  (0.01 < improvement <= 0.05).
* ``FEATURE_SIGNAL_ABSENT``   — no gain (improvement <= 0.01).
* ``TEMPLATE_LEAKAGE_DOMINANT`` — family/archetype controls explain
  nearly all performance (family-only utility >= 0.95 * full feature
  utility).

The output of :func:`analyze_feature_signal` is a JSON-serializable
``feature_signal_analysis`` dict.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .data import LoadedRun, TaskInfo
from .canonical_evaluator import compute_headroom_metrics
from .config import MINIMUM_DENOMINATOR


# ---------------------------------------------------------------------------
# Optional sklearn imports
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised when sklearn is present
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.preprocessing import OneHotEncoder
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

FEATURE_SIGNAL_PRESENT_THRESHOLD: float = 0.05
FEATURE_SIGNAL_WEAK_THRESHOLD: float = 0.01
TEMPLATE_LEAKAGE_FRACTION: float = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _onehot_encode(
    values: list[str],
    categories: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """One-hot encode a list of categorical values."""
    if categories is None:
        categories = sorted(set(values))
    cat_idx = {c: i for i, c in enumerate(categories)}
    mat = np.zeros((len(values), len(categories)))
    for i, v in enumerate(values):
        if v in cat_idx:
            mat[i, cat_idx[v]] = 1.0
    return mat, categories


def _feature_matrix(examples: list[dict[str, Any]]) -> np.ndarray:
    """Stack the ``features`` vectors of a list of examples."""
    if not examples:
        return np.zeros((0, 0), dtype=float)
    return np.array([ex["features"] for ex in examples], dtype=float)


def _labels(examples: list[dict[str, Any]]) -> np.ndarray:
    """Return the best-action labels for a list of examples."""
    return np.array([ex.get("best_action", "") for ex in examples], dtype=object)


def _families(examples: list[dict[str, Any]]) -> list[str]:
    return [ex.get("task_family", "unknown") for ex in examples]


def _archetypes(run: LoadedRun, examples: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for ex in examples:
        tid = ex.get("task_id", "")
        task = run.tasks.get(tid)
        out.append(task.archetype if task is not None else "unknown")
    return out


def _family_derived_mask(feature_names: list[str]) -> np.ndarray:
    """Boolean mask of features whose name contains 'family'."""
    return np.array(
        ["family" in name.lower() for name in feature_names], dtype=bool
    )


def _archetype_derived_mask(feature_names: list[str]) -> np.ndarray:
    """Boolean mask of features whose name contains 'archetype'."""
    return np.array(
        ["archetype" in name.lower() for name in feature_names], dtype=bool
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def _train_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    actions: list[str],
    seed: int = 42,
) -> Any:
    """Train a sklearn LogisticRegression classifier.

    sklearn 1.8 removed the ``multi_class`` parameter; the default
    behaviour is multinomial for >2 classes.
    """
    if not _HAS_SKLEARN:
        raise RuntimeError("sklearn is required for feature_signal analysis")
    if len(actions) < 2:
        # Degenerate single-action case.
        model = DummyClassifier(strategy="constant", constant=actions[0] if actions else "A")
        model.fit(train_features, train_labels)
        return model
    model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(train_features, train_labels)
    return model


def _train_dummy(train_labels: np.ndarray, actions: list[str]) -> Any:
    """Intercept-only model: always predict the most common class."""
    if not _HAS_SKLEARN:
        raise RuntimeError("sklearn is required for feature_signal analysis")
    counts = Counter(train_labels.tolist())
    if counts:
        constant = counts.most_common(1)[0][0]
    else:
        constant = actions[0] if actions else "A"
    model = DummyClassifier(strategy="constant", constant=constant)
    # DummyClassifier needs an X with at least one feature.
    X = np.zeros((len(train_labels), 1))
    model.fit(X, train_labels)
    return model


def _predict_proba(model: Any, features: np.ndarray, actions: list[str]) -> np.ndarray:
    """Return predicted probabilities aligned to ``actions`` order."""
    classes = list(model.classes_)
    proba = model.predict_proba(features)
    out = np.zeros((features.shape[0], len(actions)), dtype=float)
    for j, cls in enumerate(classes):
        if cls in actions:
            out[:, actions.index(cls)] = proba[:, j]
    return out


def _predict_argmax(model: Any, features: np.ndarray, actions: list[str]) -> np.ndarray:
    """Return predicted action index (argmax) for each row."""
    proba = _predict_proba(model, features, actions)
    return np.argmax(proba, axis=1)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def _eligible_action_indices(
    run: LoadedRun, examples: list[dict[str, Any]], actions: list[str]
) -> list[list[int]]:
    """For each example return the indices of allowed actions present in ``actions``."""
    action_idx = {a: i for i, a in enumerate(actions)}
    out: list[list[int]] = []
    for ex in examples:
        tid = ex.get("task_id", "")
        task = run.tasks.get(tid)
        allowed = task.allowed_actions if task is not None else []
        out.append([action_idx[a] for a in allowed if a in action_idx])
    return out


def _evaluate_model(
    run: LoadedRun,
    model: Any,
    val_features: np.ndarray,
    val_examples: list[dict[str, Any]],
    val_labels: np.ndarray,
    actions: list[str],
    frequency_utility: float,
    family_majority_utility: float,
    oracle_mean: float,
    sham_mean: float,
) -> dict[str, Any]:
    """Evaluate a trained model on the validation set."""
    if len(val_examples) == 0:
        return {
            "action_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "cross_entropy": 0.0,
            "expected_utility": 0.0,
            "regret": 0.0,
            "headroom_recovered": None,
            "candidate_minus_frequency": 0.0,
            "candidate_minus_family_majority": 0.0,
        }

    proba = _predict_proba(model, val_features, actions)
    pred_idx = np.argmax(proba, axis=1)
    eligible = _eligible_action_indices(run, val_examples, actions)

    # Restrict argmax to eligible actions; if none eligible, fall back to
    # the global argmax.
    selected: list[int] = []
    for i, elig in enumerate(eligible):
        if elig:
            selected.append(int(elig[np.argmax(proba[i, elig])]))
        else:
            selected.append(int(pred_idx[i]))

    utilities: list[float] = []
    regrets: list[float] = []
    correct = 0
    label_idx = {a: i for i, a in enumerate(actions)}

    for i, ex in enumerate(val_examples):
        tid = ex.get("task_id", "")
        action = actions[selected[i]]
        u = run.get_utility_for_action(tid, action)
        if u is None:
            u = 0.0
        utilities.append(u)
        oracle_u = run.get_oracle_utility(tid)
        regrets.append(oracle_u - u)
        if val_labels[i] in label_idx and label_idx[val_labels[i]] == selected[i]:
            correct += 1

    utilities_arr = np.array(utilities, dtype=float)
    regrets_arr = np.array(regrets, dtype=float)
    expected_utility = float(utilities_arr.mean())
    regret = float(regrets_arr.mean())

    # Action accuracy.
    action_accuracy = correct / len(val_examples) if val_examples else 0.0

    # Balanced accuracy.
    unique_labels = sorted(set(val_labels.tolist()))
    per_class_acc: list[float] = []
    for cls in unique_labels:
        mask = val_labels == cls
        if mask.sum() == 0:
            continue
        cls_pred = np.array(selected)
        cls_true = np.array([label_idx.get(a, -1) for a in val_labels])
        cls_pred_idx = np.array([label_idx.get(actions[s], -1) for s in cls_pred])
        per_class_acc.append(float(np.mean(cls_true[mask] == cls_pred_idx[mask])))
    balanced_accuracy = float(np.mean(per_class_acc)) if per_class_acc else 0.0

    # Cross entropy.
    eps = 1e-12
    true_idx = np.array([label_idx.get(a, 0) for a in val_labels])
    p_true = proba[np.arange(len(val_labels)), true_idx]
    cross_entropy = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))

    # Headroom recovered on validation.
    headroom = compute_headroom_metrics(
        candidate_mean=expected_utility,
        reference_mean=sham_mean,
        oracle_mean=oracle_mean,
        n_tasks=len(val_examples),
        reference_policy_id="sham",
        candidate_policy_id="model",
    )

    return {
        "action_accuracy": float(action_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "cross_entropy": cross_entropy,
        "expected_utility": expected_utility,
        "regret": regret,
        "headroom_recovered": headroom,
        "candidate_minus_frequency": float(expected_utility - frequency_utility),
        "candidate_minus_family_majority": float(expected_utility - family_majority_utility),
    }


# ---------------------------------------------------------------------------
# Frequency / family-majority utility baselines
# ---------------------------------------------------------------------------


def _frequency_policy_utility(
    run: LoadedRun, examples: list[dict[str, Any]], actions: list[str]
) -> tuple[float, str]:
    """Utility of always picking the most common D_experience best action."""
    train_labels = _labels(run.train_examples)
    counts = Counter(train_labels.tolist())
    if not counts:
        return 0.0, actions[0] if actions else ""
    majority = counts.most_common(1)[0][0]
    utils: list[float] = []
    for ex in examples:
        tid = ex.get("task_id", "")
        task = run.tasks.get(tid)
        allowed = task.allowed_actions if task is not None else []
        action = majority if majority in allowed else (allowed[0] if allowed else majority)
        u = run.get_utility_for_action(tid, action)
        utils.append(u if u is not None else 0.0)
    return float(np.mean(utils)) if utils else 0.0, majority


def _family_majority_utility(
    run: LoadedRun, examples: list[dict[str, Any]], train_families: list[str]
) -> float:
    """Utility of a per-family majority-action policy."""
    train_labels = _labels(run.train_examples)
    family_to_majority: dict[str, str] = {}
    for fam, lab in zip(train_families, train_labels.tolist()):
        # Last-wins majority is fine for a simple control.
        family_to_majority[fam] = lab
    # Proper majority via Counter.
    fam_counts: dict[str, Counter] = {}
    for fam, lab in zip(train_families, train_labels.tolist()):
        fam_counts.setdefault(fam, Counter())[lab] += 1
    for fam, c in fam_counts.items():
        family_to_majority[fam] = c.most_common(1)[0][0]

    utils: list[float] = []
    for ex in examples:
        tid = ex.get("task_id", "")
        task = run.tasks.get(tid)
        allowed = task.allowed_actions if task is not None else []
        fam = ex.get("task_family", "unknown")
        action = family_to_majority.get(fam, allowed[0] if allowed else "")
        if action not in allowed:
            action = allowed[0] if allowed else action
        u = run.get_utility_for_action(tid, action)
        utils.append(u if u is not None else 0.0)
    return float(np.mean(utils)) if utils else 0.0


# ---------------------------------------------------------------------------
# Learnability conclusion
# ---------------------------------------------------------------------------


def _learnability_conclusion(
    models: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    """Determine the primary learnability conclusion."""
    full_util = models.get("full_feature_model", {}).get("expected_utility", 0.0)
    family_util = models.get("family_only", {}).get("expected_utility", 0.0)
    freq_util = models.get("marginal_action_frequency", {}).get("expected_utility", 0.0)
    excl_family_util = models.get("features_excluding_family", {}).get("expected_utility", 0.0)
    excl_arch_util = models.get("features_excluding_archetype", {}).get("expected_utility", 0.0)

    # Improvement of feature models over the best simple control.
    best_control = max(family_util, freq_util)
    feature_improvement = max(excl_family_util, excl_arch_util) - best_control

    # Template leakage check.
    template_leakage = (
        family_util >= TEMPLATE_LEAKAGE_FRACTION * full_util
        and full_util > 0
    )

    if template_leakage:
        conclusion = "TEMPLATE_LEAKAGE_DOMINANT"
        text = (
            f"Family/archetype controls explain nearly all performance: "
            f"family-only utility ({family_util:.3f}) >= "
            f"{TEMPLATE_LEAKAGE_FRACTION:.0%} of full feature utility "
            f"({full_util:.3f}). The learned policy may be exploiting "
            f"template metadata rather than genuine executive signal."
        )
    elif feature_improvement > FEATURE_SIGNAL_PRESENT_THRESHOLD:
        conclusion = "FEATURE_SIGNAL_PRESENT"
        text = (
            f"Features excluding family/archetype materially improve "
            f"expected utility over family-only and frequency controls "
            f"(improvement {feature_improvement:.3f} > "
            f"{FEATURE_SIGNAL_PRESENT_THRESHOLD}). Genuine executive "
            f"signal is present."
        )
    elif feature_improvement > FEATURE_SIGNAL_WEAK_THRESHOLD:
        conclusion = "FEATURE_SIGNAL_WEAK"
        text = (
            f"Only small unstable gains from features excluding "
            f"family/archetype (improvement {feature_improvement:.3f}, "
            f"{FEATURE_SIGNAL_WEAK_THRESHOLD} < improvement <= "
            f"{FEATURE_SIGNAL_PRESENT_THRESHOLD}). Signal is weak."
        )
    else:
        conclusion = "FEATURE_SIGNAL_ABSENT"
        text = (
            f"No gain from features excluding family/archetype "
            f"(improvement {feature_improvement:.3f} <= "
            f"{FEATURE_SIGNAL_WEAK_THRESHOLD}). No genuine executive "
            f"signal beyond simple controls."
        )

    return conclusion, text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_feature_signal(run: LoadedRun, seed: int = 42) -> dict[str, Any]:
    """Validate whether full executive state adds predictive value.

    Args:
        run: The loaded run with training/validation examples and
            counterfactual outcomes.
        seed: Random seed for model training.

    Returns:
        A JSON-serializable ``feature_signal_analysis`` dict with
        models, learnability_conclusion, and interpretation.
    """
    train_examples = run.train_examples
    val_examples = run.validation_examples or run.train_examples

    feature_names: list[str] = run.candidate_policy.get("feature_names", [])
    actions = run.candidate_policy.get("actions", run.all_actions)

    # Oracle / sham means on validation for headroom recovered.
    val_oracle = np.array(
        [run.get_oracle_utility(ex.get("task_id", "")) for ex in val_examples],
        dtype=float,
    )
    oracle_mean = float(val_oracle.mean()) if val_oracle.size else 0.0

    # Sham mean on validation: use the executed sham policy.
    from .canonical_evaluator import make_executed_sham_fn
    sham_fn = make_executed_sham_fn(run)
    sham_utils: list[float] = []
    for ex in val_examples:
        tid = ex.get("task_id", "")
        task = run.tasks.get(tid)
        if task is None:
            sham_utils.append(0.0)
            continue
        action = sham_fn(tid, task)
        u = run.get_utility_for_action(tid, action) if action else None
        sham_utils.append(u if u is not None else 0.0)
    sham_mean = float(np.mean(sham_utils)) if sham_utils else 0.0

    # Frequency and family-majority utility baselines (on validation).
    frequency_utility, majority_action = _frequency_policy_utility(
        run, val_examples, actions
    )
    train_families = _families(train_examples)
    family_majority_utility = _family_majority_utility(
        run, val_examples, train_families
    )

    # Feature matrices and labels.
    train_features = _feature_matrix(train_examples)
    val_features = _feature_matrix(val_examples)
    train_labels = _labels(train_examples)
    val_labels = _labels(val_examples)

    train_fam = _families(train_examples)
    val_fam = _families(val_examples)
    train_arch = _archetypes(run, train_examples)
    val_arch = _archetypes(run, val_examples)

    fam_onehot_train, fam_categories = _onehot_encode(train_fam)
    fam_onehot_val, _ = _onehot_encode(val_fam, fam_categories)
    arch_onehot_train, arch_categories = _onehot_encode(train_arch)
    arch_onehot_val, _ = _onehot_encode(val_arch, arch_categories)
    fam_arch_train = np.hstack([fam_onehot_train, arch_onehot_train])
    fam_arch_val = np.hstack([fam_onehot_val, arch_onehot_val])

    # Feature masks.
    fam_mask = _family_derived_mask(feature_names)
    arch_mask = _archetype_derived_mask(feature_names)
    non_fam_idx = [i for i, m in enumerate(fam_mask) if not m]
    non_arch_idx = [i for i, m in enumerate(arch_mask) if not m]

    models: dict[str, dict[str, Any]] = {}

    # --- 1. intercept only ---
    dummy = _train_dummy(train_labels, actions)
    dummy_val_X = np.zeros((len(val_examples), 1))
    models["intercept_only"] = _evaluate_model(
        run, dummy, dummy_val_X, val_examples, val_labels, actions,
        frequency_utility, family_majority_utility, oracle_mean, sham_mean,
    )
    models["intercept_only"]["n_features"] = 0

    # --- 2. marginal action frequency ---
    # Represented as a constant-feature logistic that always predicts the
    # majority action (same predictions as the frequency control).
    freq_model = _train_dummy(train_labels, actions)
    freq_val_X = np.zeros((len(val_examples), 1))
    freq_metrics = _evaluate_model(
        run, freq_model, freq_val_X, val_examples, val_labels, actions,
        frequency_utility, family_majority_utility, oracle_mean, sham_mean,
    )
    freq_metrics["expected_utility"] = frequency_utility
    freq_metrics["candidate_minus_frequency"] = 0.0
    freq_metrics["candidate_minus_family_majority"] = float(
        frequency_utility - family_majority_utility
    )
    freq_metrics["majority_action"] = majority_action
    freq_metrics["n_features"] = 0
    models["marginal_action_frequency"] = freq_metrics

    # Helper to train + evaluate a logistic model.
    def _train_eval(
        name: str,
        X_train: np.ndarray,
        X_val: np.ndarray,
        n_features: int,
    ) -> None:
        if X_train.shape[1] == 0 or len(np.unique(train_labels)) < 2:
            # Degenerate: fall back to dummy.
            model = _train_dummy(train_labels, actions)
            X_val_use = np.zeros((len(val_examples), 1))
        else:
            model = _train_logistic(X_train, train_labels, actions, seed=seed)
            X_val_use = X_val
        metrics = _evaluate_model(
            run, model, X_val_use, val_examples, val_labels, actions,
            frequency_utility, family_majority_utility, oracle_mean, sham_mean,
        )
        metrics["n_features"] = n_features
        models[name] = metrics

    # --- 3. family only ---
    _train_eval("family_only", fam_onehot_train, fam_onehot_val, fam_onehot_train.shape[1])

    # --- 4. archetype only ---
    _train_eval("archetype_only", arch_onehot_train, arch_onehot_val, arch_onehot_train.shape[1])

    # --- 5. family + archetype ---
    _train_eval(
        "family_plus_archetype",
        fam_arch_train, fam_arch_val, fam_arch_train.shape[1],
    )

    # --- 6. features excluding family ---
    if non_fam_idx and train_features.shape[1] == len(feature_names):
        X_tr = train_features[:, non_fam_idx]
        X_val = val_features[:, non_fam_idx]
    else:
        X_tr = np.zeros((len(train_examples), 0))
        X_val = np.zeros((len(val_examples), 0))
    _train_eval("features_excluding_family", X_tr, X_val, len(non_fam_idx))

    # --- 7. features excluding archetype ---
    if non_arch_idx and train_features.shape[1] == len(feature_names):
        X_tr = train_features[:, non_arch_idx]
        X_val = val_features[:, non_arch_idx]
    else:
        X_tr = np.zeros((len(train_examples), 0))
        X_val = np.zeros((len(val_examples), 0))
    _train_eval("features_excluding_archetype", X_tr, X_val, len(non_arch_idx))

    # --- 8. full feature model ---
    if train_features.shape[1] > 0:
        _train_eval(
            "full_feature_model",
            train_features, val_features, train_features.shape[1],
        )
    else:
        models["full_feature_model"] = _evaluate_model(
            run, _train_dummy(train_labels, actions),
            np.zeros((len(val_examples), 1)),
            val_examples, val_labels, actions,
            frequency_utility, family_majority_utility, oracle_mean, sham_mean,
        )
        models["full_feature_model"]["n_features"] = 0

    # --- 9. shuffled feature model (feature shuffle sham) ---
    rng = np.random.default_rng(seed)
    shuffled_train = train_features.copy()
    if shuffled_train.size > 0:
        rng.shuffle(shuffled_train)  # shuffle flat — breaks row/label alignment
    if shuffled_train.size > 0 and train_features.shape[1] > 0:
        _train_eval(
            "shuffled_feature_model",
            shuffled_train, val_features, train_features.shape[1],
        )
    else:
        models["shuffled_feature_model"] = _evaluate_model(
            run, _train_dummy(train_labels, actions),
            np.zeros((len(val_examples), 1)),
            val_examples, val_labels, actions,
            frequency_utility, family_majority_utility, oracle_mean, sham_mean,
        )
        models["shuffled_feature_model"]["n_features"] = 0

    # Learnability conclusion.
    conclusion, text = _learnability_conclusion(models)

    return {
        "models": models,
        "learnability_conclusion": conclusion,
        "interpretation": text,
        "feature_names": feature_names,
        "n_train_examples": len(train_examples),
        "n_validation_examples": len(val_examples),
        "actions": actions,
        "frequency_utility": frequency_utility,
        "family_majority_utility": family_majority_utility,
        "oracle_mean_validation": oracle_mean,
        "sham_mean_validation": sham_mean,
        "seed": seed,
        "thresholds": {
            "feature_signal_present": FEATURE_SIGNAL_PRESENT_THRESHOLD,
            "feature_signal_weak": FEATURE_SIGNAL_WEAK_THRESHOLD,
            "template_leakage_fraction": TEMPLATE_LEAKAGE_FRACTION,
        },
    }
