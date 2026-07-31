"""Feature utility and leakage audit for v0.4.2 diagnostics.

Audits every executive feature used by the learned candidate policy for
information leakage, predictive utility, and redundancy with task-family
or archetype metadata.

The audit produces content for two artifacts:

* ``feature_audit.json`` — per-feature classification, leakage rejection,
  and diagnostic summaries.
* ``feature_ablation_results.json`` — permutation importance, drop-column
  ablation, coefficient stability, correlation structure, and the
  family-residual test.

All diagnostics consume existing training/validation examples — no new
model executions are required.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from .data import LoadedRun
from .sham_trainer import train_logistic, train_sham_policy

# ---------------------------------------------------------------------------
# Optional sklearn imports — the codebase has sklearn installed, but we
# provide numpy fallbacks for robustness.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised when sklearn is present
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.dummy import DummyClassifier
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Leakage patterns — feature names containing any of these substrings are
# rejected as post-outcome / label-leaking / verifier-derived.
# ---------------------------------------------------------------------------
_LEAKAGE_PATTERNS: list[str] = [
    "expected action",
    "best action",
    "verifier result",
    "verified success",
    "expected answer",
    "utility",
    "post execution latency",
    "post execution token count",
    "tool result",
    "workflow acceptance",
    "post hoc failure category",
]


# ---------------------------------------------------------------------------
# Feature classification helpers
# ---------------------------------------------------------------------------

def _derive_definition(name: str) -> str:
    """Derive a human-readable definition from a feature name.

    Expands common abbreviations and replaces underscores with spaces.
    """
    expansions: dict[str, str] = {
        "n_": "number of ",
        "num_": "number of ",
        "len_": "length of ",
        "is_": "indicator: ",
        "has_": "indicator: has ",
        "avg_": "average ",
        "max_": "maximum ",
        "min_": "minimum ",
        "std_": "standard deviation of ",
        "pct_": "percentage of ",
        "freq_": "frequency of ",
        "cnt_": "count of ",
    }
    desc = name.lower().replace("_", " ")
    for abbrev, full in expansions.items():
        if desc.startswith(abbrev.replace("_", " ")):
            desc = full + desc[len(abbrev.replace("_", " ")):]
            break
    return desc.strip()


def _infer_source(name: str) -> str:
    """Infer the data source of a feature from its name."""
    lower = name.lower().replace("_", " ").replace("-", " ")
    if "family" in lower:
        return "task_family"
    if "archetype" in lower:
        return "task_archetype"
    if "verifier" in lower or "verified" in lower:
        return "verifier"
    if "utility" in lower:
        return "outcome"
    if "post execution" in lower:
        return "execution_metadata"
    if "tool" in lower:
        return "tool_output"
    if "token" in lower:
        return "execution_metadata"
    if "latency" in lower:
        return "execution_metadata"
    if "prompt" in lower:
        return "task_prompt"
    if "risk" in lower:
        return "task_metadata"
    if "crossover" in lower:
        return "task_metadata"
    if "generator" in lower or "template" in lower or "capability" in lower or "entity" in lower:
        return "task_metadata"
    return "task_features"


def _infer_availability(name: str) -> str:
    """Infer when a feature becomes available (pre- vs post-execution)."""
    lower = name.lower().replace("_", " ").replace("-", " ")
    post_markers = [
        "post execution",
        "verifier", "verified",
        "utility", "tool result",
        "workflow acceptance",
        "post hoc",
        "latency", "token count",
    ]
    if any(m in lower for m in post_markers):
        return "post-execution"
    return "pre-execution"


def _classify_feature(name: str) -> dict[str, Any]:
    """Classify a single executive feature by its name.

    Records:
      - name, definition, source, availability time
      - model-visible, post-outcome, family-derived, archetype-derived,
        verifier-derived, action-label proxy
      - rejected (with reason) if the name matches a leakage pattern
    """
    lower = name.lower()
    # Normalise underscores and hyphens to spaces so patterns like
    # "best action" also match feature names written as "best_action_hint"
    # or "post-execution latency" matching "post_execution_latency".
    normalised = lower.replace("_", " ").replace("-", " ")

    # Check leakage patterns against both the raw and normalised name.
    matched_pattern: str | None = None
    for pattern in _LEAKAGE_PATTERNS:
        if pattern in lower or pattern in normalised:
            matched_pattern = pattern
            break

    is_post_outcome = any(m in normalised for m in [
        "post execution", "verifier", "verified",
        "utility", "tool result", "workflow acceptance",
        "post hoc", "latency", "token count",
    ])
    is_family_derived = "family" in lower
    is_archetype_derived = "archetype" in lower
    is_verifier_derived = "verifier" in lower or "verified" in lower
    is_action_label_proxy = any(m in normalised for m in [
        "best action", "expected action",
        "action label", "selected action",
        "oracle action",
    ])

    return {
        "name": name,
        "definition": _derive_definition(name),
        "source": _infer_source(name),
        "availability_time": _infer_availability(name),
        "model_visible": True,  # all features in feature_names are model-visible
        "post_outcome": is_post_outcome,
        "family_derived": is_family_derived,
        "archetype_derived": is_archetype_derived,
        "verifier_derived": is_verifier_derived,
        "action_label_proxy": is_action_label_proxy,
        "rejected": matched_pattern is not None,
        "rejection_reason": (
            f"Feature name contains leakage pattern '{matched_pattern}'"
            if matched_pattern else None
        ),
    }


# ---------------------------------------------------------------------------
# Encoding and evaluation helpers
# ---------------------------------------------------------------------------

def _onehot_encode(
    values: list[str],
    categories: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """One-hot encode a list of categorical values.

    Returns (onehot_matrix, category_names).
    """
    if categories is None:
        categories = sorted(set(values))
    cat_idx = {c: i for i, c in enumerate(categories)}
    mat = np.zeros((len(values), len(categories)))
    for i, v in enumerate(values):
        if v in cat_idx:
            mat[i, cat_idx[v]] = 1.0
    return mat, categories


def _eval_model_utility(
    weights: np.ndarray,
    bias: np.ndarray,
    features: np.ndarray,
    examples: list[dict],
    run: LoadedRun,
    actions: list[str],
) -> float:
    """Evaluate mean utility of a logistic model on a set of examples.

    For each example, predicts the best eligible action and looks up the
    measured counterfactual utility.
    """
    action_idx = {a: i for i, a in enumerate(actions)}
    utilities: list[float] = []
    for i, ex in enumerate(examples):
        tid = ex["task_id"]
        task = run.tasks.get(tid)
        if task is None:
            continue
        logits = features[i] @ weights + bias
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            continue
        eligible_logits = np.array([logits[action_idx[a]] for a in eligible])
        chosen = eligible[int(np.argmax(eligible_logits))]
        u = run.get_utility_for_action(tid, chosen)
        utilities.append(u if u is not None else 0.0)
    return float(np.mean(utilities)) if utilities else 0.0


def _eval_model_accuracy(
    weights: np.ndarray,
    bias: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    actions: list[str],
) -> float:
    """Evaluate classification accuracy of a logistic model."""
    action_idx = {a: i for i, a in enumerate(actions)}
    if len(features) == 0:
        return 0.0
    logits = features @ weights + bias
    preds = np.array([actions[int(np.argmax(row))] for row in logits])
    label_arr = np.array([action_idx.get(a, -1) for a in labels])
    pred_arr = np.array([action_idx.get(a, -1) for a in preds])
    mask = (label_arr >= 0) & (pred_arr >= 0)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(label_arr[mask] == pred_arr[mask]))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Numpy fallback classifiers (used only if sklearn is unavailable)
# ---------------------------------------------------------------------------

def _numpy_logistic_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    actions: list[str],
) -> float:
    """Train a logistic model via train_logistic and return accuracy."""
    W, b = train_logistic(train_features, train_labels, actions)
    return _eval_model_accuracy(W, b, eval_features, eval_labels, actions)


def _numpy_mutual_info(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Simple numpy mutual information fallback.

    Computes MI between each feature and the label using binning.
    """
    n, d = features.shape
    mi = np.zeros(d)
    unique_labels = np.unique(labels)
    n_bins = min(10, max(2, int(np.sqrt(n))))
    for j in range(d):
        col = features[:, j]
        if np.std(col) < 1e-12:
            mi[j] = 0.0
            continue
        bins = np.linspace(col.min() - 1e-9, col.max() + 1e-9, n_bins + 1)
        binned = np.digitize(col, bins[1:-1])
        mi_val = 0.0
        for b in np.unique(binned):
            p_x = np.mean(binned == b)
            for lab in unique_labels:
                p_xy = np.mean((binned == b) & (labels == lab))
                if p_xy > 0:
                    p_y = np.mean(labels == lab)
                    mi_val += p_xy * np.log(p_xy / (p_x * p_y))
        mi[j] = max(0.0, mi_val)
    return mi


# ---------------------------------------------------------------------------
# Feature diagnostics
# ---------------------------------------------------------------------------

def _univariate_mutual_info(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Compute univariate mutual information between each feature and best action."""
    if len(features) == 0 or len(feature_names) == 0:
        return {name: 0.0 for name in feature_names}
    if _HAS_SKLEARN:
        mi = mutual_info_classif(features, labels, discrete_features=False, random_state=42)
    else:
        mi = _numpy_mutual_info(features, labels)
    return {name: float(val) for name, val in zip(feature_names, mi)}


def _permutation_importance(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
) -> dict[str, Any]:
    """Permutation importance: shuffle each feature, measure accuracy drop."""
    if len(train_features) == 0 or len(feature_names) == 0:
        return {"baseline_accuracy": 0.0, "importances": {}, "description": "no data"}
    rng = np.random.default_rng(seed)
    actions = sorted(set(train_labels))

    if _HAS_SKLEARN:
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(train_features, train_labels)
        baseline_acc = float(model.score(val_features, val_labels))
        importances: dict[str, float] = {}
        for j, name in enumerate(feature_names):
            shuffled = val_features.copy()
            rng.shuffle(shuffled[:, j])
            acc = float(model.score(shuffled, val_labels))
            importances[name] = baseline_acc - acc
    else:
        W, b = train_logistic(train_features, train_labels, actions, seed=seed)
        baseline_acc = _eval_model_accuracy(W, b, val_features, val_labels, actions)
        importances = {}
        for j, name in enumerate(feature_names):
            shuffled = val_features.copy()
            rng.shuffle(shuffled[:, j])
            acc = _eval_model_accuracy(W, b, shuffled, val_labels, actions)
            importances[name] = baseline_acc - acc

    return {
        "baseline_accuracy": baseline_acc,
        "importances": importances,
        "description": (
            "Permutation importance: accuracy drop when a single feature is "
            "shuffled. Positive values indicate the feature contributes to "
            "prediction; near-zero or negative values indicate noise or "
            "redundancy."
        ),
    }


def _drop_column_ablation(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_examples: list[dict],
    run: LoadedRun,
    actions: list[str],
    feature_names: list[str],
    seed: int = 42,
) -> dict[str, Any]:
    """Drop-column ablation: remove one feature, retrain, measure utility change."""
    if len(train_features) == 0 or len(feature_names) == 0:
        return {"baseline_utility": 0.0, "ablations": {}, "description": "no data"}

    n_features = len(feature_names)

    # Baseline model with all features.
    W_base, b_base = train_logistic(train_features, train_labels, actions, seed=seed)
    baseline_util = _eval_model_utility(W_base, b_base, val_features, val_examples, run, actions)

    ablations: dict[str, dict[str, float]] = {}
    for j in range(n_features):
        mask = [k for k in range(n_features) if k != j]
        if not mask:
            continue
        W_j, b_j = train_logistic(
            train_features[:, mask], train_labels, actions, seed=seed
        )
        util_j = _eval_model_utility(
            W_j, b_j, val_features[:, mask], val_examples, run, actions
        )
        ablations[feature_names[j]] = {
            "utility_without_feature": util_j,
            "utility_change": baseline_util - util_j,
            "relative_change": (
                (baseline_util - util_j) / abs(baseline_util)
                if abs(baseline_util) > 1e-12 else 0.0
            ),
        }

    return {
        "baseline_utility": baseline_util,
        "ablations": ablations,
        "description": (
            "Drop-column ablation: each feature is removed, the learner is "
            "retrained, and the change in mean validation utility is recorded. "
            "Positive utility_change means the feature was helping."
        ),
    }


def _coefficient_stability(
    features: np.ndarray,
    labels: np.ndarray,
    actions: list[str],
    feature_names: list[str],
    n_bootstrap: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Coefficient stability across bootstrap samples.

    Trains the learner on 20 bootstrap resamples and records the mean,
    std, and coefficient of variation of each weight.
    """
    if len(features) == 0 or len(feature_names) == 0:
        return {"n_bootstrap": n_bootstrap, "stability": {}, "description": "no data"}

    rng = np.random.default_rng(seed)
    n = len(features)
    n_features = len(feature_names)
    n_actions = len(actions)

    coef_samples = np.zeros((n_bootstrap, n_features, n_actions))
    for s in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        W_s, _ = train_logistic(features[idx], labels[idx], actions, seed=seed + s)
        coef_samples[s] = W_s

    mean_coefs = coef_samples.mean(axis=0)
    std_coefs = coef_samples.std(axis=0, ddof=1)
    cv_coefs = std_coefs / (np.abs(mean_coefs) + 1e-10)

    stability: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(feature_names):
        stability[name] = {
            "mean_coefficients": mean_coefs[j].tolist(),
            "std_coefficients": std_coefs[j].tolist(),
            "cv_coefficients": cv_coefs[j].tolist(),
            "mean_cv": float(np.mean(cv_coefs[j])),
            "max_cv": float(np.max(cv_coefs[j])),
        }

    return {
        "n_bootstrap": n_bootstrap,
        "stability": stability,
        "description": (
            "Coefficient stability across 20 bootstrap resamples. High "
            "coefficient of variation (CV) indicates the feature's weight is "
            "unstable across resamples, suggesting weak or spurious signal."
        ),
    }


def _feature_correlation(
    features: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Compute the feature correlation matrix."""
    if len(features) == 0 or len(feature_names) == 0:
        return {"matrix": [], "feature_names": [], "high_correlations": []}

    n_features = len(feature_names)
    if n_features == 1:
        return {
            "matrix": [[1.0]],
            "feature_names": feature_names,
            "high_correlations": [],
        }

    corr = np.corrcoef(features.T)
    # Replace NaN (zero-variance features) with 0.
    corr = np.nan_to_num(corr, nan=0.0)

    high_corr: list[dict[str, Any]] = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(corr[i, j]) >= 0.8:
                high_corr.append({
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "correlation": float(corr[i, j]),
                })

    return {
        "matrix": corr.tolist(),
        "feature_names": feature_names,
        "high_correlations": high_corr,
        "description": (
            "Pearson correlation between executive features. Pairs with "
            "|r| >= 0.8 are flagged as highly correlated (redundant)."
        ),
    }


def _conditioned_predictive_power(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    train_condition: np.ndarray,
    val_condition: np.ndarray,
    feature_names: list[str],
    condition_name: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Conditioned predictive power: does each feature help beyond a grouping?

    Trains a baseline model on the conditioning variable alone (e.g. one-hot
    family), then adds each executive feature individually and measures the
    accuracy improvement.
    """
    if len(train_features) == 0 or len(feature_names) == 0:
        return {"condition": condition_name, "baseline_accuracy": 0.0, "per_feature": {}}

    actions = sorted(set(train_labels))

    if _HAS_SKLEARN:
        baseline_model = LogisticRegression(max_iter=1000, random_state=seed)
        baseline_model.fit(train_condition, train_labels)
        baseline_acc = float(baseline_model.score(val_condition, val_labels))

        per_feature: dict[str, dict[str, float]] = {}
        for j, name in enumerate(feature_names):
            combined_train = np.hstack([train_condition, train_features[:, [j]]])
            combined_val = np.hstack([val_condition, val_features[:, [j]]])
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(combined_train, train_labels)
            acc = float(model.score(combined_val, val_labels))
            per_feature[name] = {
                "accuracy_with_feature": acc,
                "improvement": acc - baseline_acc,
            }
    else:
        baseline_acc = _numpy_logistic_accuracy(
            train_condition, train_labels, val_condition, val_labels, actions
        )
        per_feature = {}
        for j, name in enumerate(feature_names):
            combined_train = np.hstack([train_condition, train_features[:, [j]]])
            combined_val = np.hstack([val_condition, val_features[:, [j]]])
            acc = _numpy_logistic_accuracy(
                combined_train, train_labels, combined_val, val_labels, actions
            )
            per_feature[name] = {
                "accuracy_with_feature": acc,
                "improvement": acc - baseline_acc,
            }

    return {
        "condition": condition_name,
        "baseline_accuracy": baseline_acc,
        "per_feature": per_feature,
        "description": (
            f"Conditioned predictive power beyond {condition_name}. For each "
            f"feature, a model is trained with {condition_name} + that feature "
            f"and compared to the {condition_name}-only baseline. Positive "
            f"improvement means the feature adds signal beyond the grouping."
        ),
    }


# ---------------------------------------------------------------------------
# Family-residual test
# ---------------------------------------------------------------------------

def _family_residual_test(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    train_families: list[str],
    val_families: list[str],
    feature_names: list[str],
    actions: list[str],
    run: LoadedRun,
    train_examples: list[dict],
    val_examples: list[dict],
    seed: int = 42,
) -> dict[str, Any]:
    """Family-residual test.

    Fits five models and compares their accuracy:
      1. family-only       — one-hot encoded task family
      2. features-only      — executive features without family
      3. full               — family + executive features
      4. shuffled-features  — executive features with shuffled columns
      5. intercept-only     — no features (prior class distribution)
    """
    if len(train_features) == 0:
        return {"models": {}, "description": "no training data"}

    # One-hot encode family.
    all_families = sorted(set(train_families) | set(val_families))
    train_fam_oh, _ = _onehot_encode(train_families, all_families)
    val_fam_oh, _ = _onehot_encode(val_families, all_families)

    rng = np.random.default_rng(seed)

    # --- 1. Family-only model ---
    if _HAS_SKLEARN:
        fam_model = LogisticRegression(max_iter=1000, random_state=seed)
        fam_model.fit(train_fam_oh, train_labels)
        acc_family = float(fam_model.score(val_fam_oh, val_labels))
    else:
        acc_family = _numpy_logistic_accuracy(
            train_fam_oh, train_labels, val_fam_oh, val_labels, actions
        )

    # --- 2. Features-without-family model ---
    if _HAS_SKLEARN:
        feat_model = LogisticRegression(max_iter=1000, random_state=seed)
        feat_model.fit(train_features, train_labels)
        acc_features = float(feat_model.score(val_features, val_labels))
    else:
        W_f, b_f = train_logistic(train_features, train_labels, actions, seed=seed)
        acc_features = _eval_model_accuracy(W_f, b_f, val_features, val_labels, actions)

    # --- 3. Full model (family + features) ---
    train_combined = np.hstack([train_fam_oh, train_features])
    val_combined = np.hstack([val_fam_oh, val_features])
    if _HAS_SKLEARN:
        full_model = LogisticRegression(max_iter=1000, random_state=seed)
        full_model.fit(train_combined, train_labels)
        acc_full = float(full_model.score(val_combined, val_labels))
    else:
        W_full, b_full = train_logistic(train_combined, train_labels, actions, seed=seed)
        acc_full = _eval_model_accuracy(W_full, b_full, val_combined, val_labels, actions)

    # --- 4. Shuffled-feature model ---
    # Use train_sham_policy with feature shuffle mode for the learner-based
    # shuffled model, then also compute a simple shuffled-features accuracy.
    shuffled_train = train_features.copy()
    shuffled_val = val_features.copy()
    for j in range(train_features.shape[1]):
        rng.shuffle(shuffled_train[:, j])
        rng.shuffle(shuffled_val[:, j])
    if _HAS_SKLEARN:
        shuf_model = LogisticRegression(max_iter=1000, random_state=seed)
        shuf_model.fit(shuffled_train, train_labels)
        acc_shuffled = float(shuf_model.score(shuffled_val, val_labels))
    else:
        W_s, b_s = train_logistic(shuffled_train, train_labels, actions, seed=seed)
        acc_shuffled = _eval_model_accuracy(W_s, b_s, shuffled_val, val_labels, actions)

    # Also train a sham policy with feature-shuffle mode to use the import.
    sham_feat_policy = train_sham_policy(run, shuffle_mode="feature", seed=seed)
    sham_feat_acc = 0.0
    if sham_feat_policy.get("weights"):
        W_sh = np.array(sham_feat_policy["weights"])
        b_sh = np.array(sham_feat_policy["bias"])
        sham_feat_acc = _eval_model_accuracy(
            W_sh, b_sh, val_features, val_labels, actions
        )

    # --- 5. Intercept-only model ---
    if _HAS_SKLEARN:
        dummy = DummyClassifier(strategy="most_frequent", random_state=seed)
        dummy.fit(np.zeros((len(train_labels), 1)), train_labels)
        acc_intercept = float(dummy.score(np.zeros((len(val_labels), 1)), val_labels))
    else:
        # Predict the majority class.
        from collections import Counter
        counts = Counter(train_labels)
        majority = counts.most_common(1)[0][0] if counts else actions[0]
        acc_intercept = float(np.mean(val_labels == majority))

    # Utility evaluation for the full vs family-only models.
    n_fam = train_fam_oh.shape[1]
    W_full_util, b_full_util = train_logistic(
        train_combined, train_labels, actions, seed=seed
    )
    util_full = _eval_model_utility(
        W_full_util, b_full_util, val_combined, val_examples, run, actions
    )
    W_fam_util, b_fam_util = train_logistic(
        train_fam_oh, train_labels, actions, seed=seed
    )
    util_family = _eval_model_utility(
        W_fam_util, b_fam_util, val_fam_oh, val_examples, run, actions
    )

    return {
        "models": {
            "family_only": {
                "accuracy": acc_family,
                "utility": util_family,
                "n_features": n_fam,
                "description": "One-hot encoded task family only.",
            },
            "features_without_family": {
                "accuracy": acc_features,
                "n_features": len(feature_names),
                "description": "Executive features without family one-hot.",
            },
            "full": {
                "accuracy": acc_full,
                "utility": util_full,
                "n_features": n_fam + len(feature_names),
                "description": "Family one-hot + executive features.",
            },
            "shuffled_feature": {
                "accuracy": acc_shuffled,
                "sham_policy_accuracy": sham_feat_acc,
                "n_features": len(feature_names),
                "description": "Executive features with each column independently shuffled.",
            },
            "intercept_only": {
                "accuracy": acc_intercept,
                "n_features": 0,
                "description": "No features — predicts majority class.",
            },
        },
        "executive_features_beyond_family": {
            "accuracy_gain": acc_full - acc_family,
            "utility_gain": util_full - util_family,
            "features_alone_vs_family": acc_features - acc_family,
        },
        "description": (
            "Family-residual test: compares five models to determine whether "
            "executive features improve action prediction beyond what task "
            "family alone provides. If full > family_only, executive features "
            "carry residual signal. If features_without_family <= family_only, "
            "the features are redundant with family."
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def audit_features(run: LoadedRun) -> dict[str, Any]:
    """Audit every executive feature used by the learned candidate policy.

    Args:
        run: loaded run data with candidate_policy, training examples, and
            validation examples.

    Returns:
        A dict with two top-level keys:

        * ``feature_audit`` — content for ``feature_audit.json``.
        * ``feature_ablation_results`` — content for
          ``feature_ablation_results.json``.
    """
    feature_names: list[str] = run.candidate_policy.get("feature_names", [])
    actions = run.candidate_policy.get("actions", run.all_actions)

    # --- 1. Feature inventory and classification ---
    feature_inventory = [_classify_feature(name) for name in feature_names]
    rejected = [f for f in feature_inventory if f["rejected"]]
    accepted_names = [f["name"] for f in feature_inventory if not f["rejected"]]

    # --- 2. Prepare data for diagnostics ---
    train_examples = run.train_examples
    val_examples = run.validation_examples or run.train_examples

    diagnostics: dict[str, Any] = {}
    family_residual: dict[str, Any] = {}

    if train_examples and feature_names:
        features = np.array([ex["features"] for ex in train_examples])
        labels = np.array([ex["best_action"] for ex in train_examples])
        train_families = [ex.get("task_family", "unknown") for ex in train_examples]
        train_archetypes = [
            run.tasks.get(ex["task_id"], type("T", (), {"archetype": "unknown"})()).archetype
            if run.tasks.get(ex["task_id"]) else "unknown"
            for ex in train_examples
        ]

        val_features = np.array([ex["features"] for ex in val_examples])
        val_labels = np.array([ex["best_action"] for ex in val_examples])
        val_families = [ex.get("task_family", "unknown") for ex in val_examples]
        val_archetypes = [
            run.tasks.get(ex["task_id"], type("T", (), {"archetype": "unknown"})()).archetype
            if run.tasks.get(ex["task_id"]) else "unknown"
            for ex in val_examples
        ]

        # --- 3. Feature diagnostics ---
        diagnostics = {
            "univariate_mutual_info": _univariate_mutual_info(
                features, labels, feature_names
            ),
            "permutation_importance": _permutation_importance(
                features, labels, val_features, val_labels, feature_names
            ),
            "coefficient_stability": _coefficient_stability(
                features, labels, actions, feature_names
            ),
            "feature_correlation": _feature_correlation(features, feature_names),
            "family_conditioned_predictive_power": _conditioned_predictive_power(
                features, labels, val_features, val_labels,
                _onehot_encode(train_families)[0],
                _onehot_encode(val_families, _onehot_encode(train_families)[1])[0],
                feature_names, "family"
            ),
            "archetype_conditioned_predictive_power": _conditioned_predictive_power(
                features, labels, val_features, val_labels,
                _onehot_encode(train_archetypes)[0],
                _onehot_encode(val_archetypes, _onehot_encode(train_archetypes)[1])[0],
                feature_names, "archetype"
            ),
        }

        # --- 4. Drop-column ablation ---
        drop_column = _drop_column_ablation(
            features, labels, val_features, val_examples, run, actions, feature_names
        )

        # --- 5. Family-residual test ---
        family_residual = _family_residual_test(
            features, labels, val_features, val_labels,
            train_families, val_families, feature_names, actions,
            run, train_examples, val_examples
        )
    else:
        diagnostics = {"error": "no training examples or feature names available"}
        drop_column = {"error": "no training examples or feature names available"}
        family_residual = {"error": "no training examples or feature names available"}

    # --- Assemble feature_audit.json content ---
    feature_audit: dict[str, Any] = {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "feature_inventory": feature_inventory,
        "rejected_features": rejected,
        "accepted_features": accepted_names,
        "n_rejected": len(rejected),
        "n_accepted": len(accepted_names),
        "leakage_patterns_checked": _LEAKAGE_PATTERNS,
        "has_post_outcome_features": any(f["post_outcome"] for f in feature_inventory),
        "has_family_derived_features": any(f["family_derived"] for f in feature_inventory),
        "has_archetype_derived_features": any(
            f["archetype_derived"] for f in feature_inventory
        ),
        "has_verifier_derived_features": any(
            f["verifier_derived"] for f in feature_inventory
        ),
        "has_action_label_proxy": any(
            f["action_label_proxy"] for f in feature_inventory
        ),
        "diagnostics_summary": {
            "univariate_mutual_info": diagnostics.get("univariate_mutual_info", {}),
            "feature_correlation": {
                "high_correlations": (
                    diagnostics.get("feature_correlation", {})
                    .get("high_correlations", [])
                ),
            },
            "coefficient_stability_summary": {
                name: {"mean_cv": info.get("mean_cv", 0.0), "max_cv": info.get("max_cv", 0.0)}
                for name, info in (
                    diagnostics.get("coefficient_stability", {}).get("stability", {}).items()
                )
            },
        },
    }

    # --- Assemble feature_ablation_results.json content ---
    feature_ablation: dict[str, Any] = {
        "permutation_importance": diagnostics.get("permutation_importance", {}),
        "drop_column_ablation": drop_column,
        "coefficient_stability": diagnostics.get("coefficient_stability", {}),
        "feature_correlation": diagnostics.get("feature_correlation", {}),
        "family_conditioned_predictive_power": diagnostics.get(
            "family_conditioned_predictive_power", {}
        ),
        "archetype_conditioned_predictive_power": diagnostics.get(
            "archetype_conditioned_predictive_power", {}
        ),
        "family_residual_test": family_residual,
    }

    return {
        "feature_audit": feature_audit,
        "feature_ablation_results": feature_ablation,
    }
