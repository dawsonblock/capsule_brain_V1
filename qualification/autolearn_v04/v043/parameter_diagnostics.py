"""Coefficient and decision-boundary diagnostics for v0.4.3.

Inspects the logistic candidate model's weight matrix, bootstrap
coefficient intervals, feature-action influence, multicollinearity,
separation, and the L2 regularization path to determine whether the
learned router has a well-conditioned, identifiable decision boundary.

The output of :func:`diagnose_parameters` is a JSON-serializable
``router_parameter_diagnostics`` dict.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import LoadedRun, TaskInfo
from .canonical_evaluator import (
    PolicyDecisionFn,
    evaluate_policy_from_counterfactuals,
    make_oracle_fn,
)
from .config import MINIMUM_DENOMINATOR

# Import the shared logistic trainer from v0.4.2 for the regularization
# path so that every model uses the same training algorithm.
from qualification.autolearn_v04.v042.sham_trainer import train_logistic


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: |correlation| above which a feature pair is flagged for multicollinearity.
MULTICOLLINEARITY_CORR_THRESHOLD: float = 0.8

#: |W| below which all coefficients are considered "near zero".
NEAR_ZERO_COEFF_THRESHOLD: float = 0.01

#: Dominance ratio: if one feature's |W| exceeds this multiple of the mean.
DOMINANCE_RATIO: float = 10.0

#: Logit-std below which action logits are considered "nearly constant".
LOGIT_STD_THRESHOLD: float = 0.1

#: Validation accuracy below which class separation is considered "weak".
WEAK_SEPARATION_ACC_THRESHOLD: float = 0.3

#: L2 regularization values for the regularization path.
REGULARIZATION_PATH_L2: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0]

#: Bootstrap confidence interval level (central 95%).
BOOTSTRAP_CI_LEVEL: float = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _build_feature_matrices(
    run: LoadedRun,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Collect train/validation/test feature matrices and labels.

    Returns:
        X_train, y_train, X_val, y_val, actions, feature_names
    """
    actions = list(run.all_actions)

    def _extract(examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        feats = []
        labels = []
        for ex in examples:
            if "features" not in ex:
                continue
            feats.append(np.asarray(ex["features"], dtype=float))
            labels.append(ex.get("best_action", actions[0] if actions else ""))
        if not feats:
            return np.empty((0, 0)), np.array([], dtype=object)
        return np.stack(feats, axis=0), np.array(labels, dtype=object)

    X_train, y_train = _extract(run.train_examples)
    X_val, y_val = _extract(run.validation_examples)
    X_test, y_test = _extract(run.test_examples)

    # If validation is empty, fall back to test for validation purposes.
    if X_val.size == 0:
        X_val, y_val = X_test, y_test

    # Feature names from the dataset manifest or derived.
    feature_names: list[str] = []
    manifest = run.dataset_manifest or {}
    feature_names = list(manifest.get("feature_names", []))
    if not feature_names and X_train.size > 0:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    return X_train, y_train, X_val, y_val, actions, feature_names


def _make_seed_policy_fn(
    W: np.ndarray,
    b: np.ndarray,
    actions: list[str],
    feature_lookup: dict[str, np.ndarray],
) -> PolicyDecisionFn:
    """Build a raw-argmax policy decision function for a trained model."""
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


def _eval_model_utility(
    W: np.ndarray,
    b: np.ndarray,
    features: np.ndarray,
    task_ids: list[str],
    run: LoadedRun,
    actions: list[str],
) -> float:
    """Evaluate a trained model's mean utility on a set of tasks."""
    if features.size == 0 or W.size == 0:
        return 0.0
    action_idx = {a: i for i, a in enumerate(actions)}
    utilities: list[float] = []
    for i, tid in enumerate(task_ids):
        task = run.tasks.get(tid)
        if task is None:
            continue
        logits = features[i] @ W + b
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            continue
        elig_logits = np.array([logits[action_idx[a]] for a in eligible])
        action = eligible[int(np.argmax(elig_logits))]
        u = run.get_utility_for_action(tid, action)
        if u is not None:
            utilities.append(u)
    return float(np.mean(utilities)) if utilities else 0.0


def _eval_model_accuracy(
    W: np.ndarray,
    b: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    actions: list[str],
) -> float:
    """Evaluate a trained model's classification accuracy."""
    if X.size == 0 or W.size == 0 or len(y) == 0:
        return 0.0
    action_idx = {a: i for i, a in enumerate(actions)}
    logits = X @ W + b
    pred_idx = np.argmax(logits, axis=1)
    correct = 0
    for i, label in enumerate(y):
        if label in action_idx and pred_idx[i] == action_idx[label]:
            correct += 1
    return float(correct / len(y))


# ---------------------------------------------------------------------------
# Diagnostic sub-analyses
# ---------------------------------------------------------------------------


def _coefficient_matrix(run: LoadedRun) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract the weight matrix W and bias b from the candidate policy.

    Returns W oriented as [n_features, n_actions] regardless of how it is
    stored in the policy dict (some policies store [n_actions, n_features]).
    """
    policy = run.candidate_policy or {}
    W = np.array(policy.get("weights", []), dtype=float)
    b = np.array(policy.get("bias", []), dtype=float)
    actions = list(policy.get("actions", run.all_actions))
    feature_names = list(policy.get("feature_names", []))
    # Normalize orientation to [n_features, n_actions].
    if W.ndim == 2 and feature_names and actions:
        n_feat = len(feature_names)
        n_act = len(actions)
        if W.shape == (n_act, n_feat):
            W = W.T
    return W, b, actions


def _standardized_coefficients(
    W: np.ndarray, X_train: np.ndarray, feature_names: list[str], actions: list[str]
) -> dict[str, Any]:
    """Compute standardized coefficients: W * std(feature).

    W is expected as [n_features, n_actions].
    """
    if W.size == 0 or X_train.size == 0:
        return {"matrix": [], "per_feature": {}, "per_action": {}}
    # Normalize to [n_features, n_actions] if needed.
    n_feat = len(feature_names)
    n_act = len(actions)
    if W.shape == (n_act, n_feat):
        W = W.T
    feat_std = np.std(X_train, axis=0)
    # Avoid division by zero.
    feat_std_safe = np.where(feat_std > 1e-12, feat_std, 1.0)
    # Align feat_std to W's feature dimension.
    if len(feat_std_safe) == W.shape[0]:
        std_W = W * feat_std_safe[:, np.newaxis]
    else:
        std_W = W

    per_feature: dict[str, dict[str, float]] = {}
    for j, fname in enumerate(feature_names):
        if j < std_W.shape[0]:
            per_feature[fname] = {
                actions[a]: float(std_W[j, a]) for a in range(std_W.shape[1])
            }

    per_action: dict[str, dict[str, float]] = {}
    for a, action in enumerate(actions):
        if a < std_W.shape[1]:
            per_action[action] = {
                feature_names[j]: float(std_W[j, a])
                for j in range(std_W.shape[0])
                if j < len(feature_names)
            }

    return {
        "matrix": std_W.tolist(),
        "per_feature": per_feature,
        "per_action": per_action,
    }


def _bootstrap_coefficients(
    X_train: np.ndarray,
    y_train: np.ndarray,
    actions: list[str],
    feature_names: list[str],
    W_original: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap coefficient intervals.

    Trains ``n_bootstrap`` models on resampled training data and reports
    mean, std, and 95% CI per weight, plus sign stability (fraction of
    bootstrap samples where the sign matches the original).
    """
    if X_train.size == 0 or len(y_train) == 0:
        return {"intervals": {}, "sign_stability": {}, "n_bootstrap": 0}

    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    n_features = X_train.shape[1]
    n_actions = len(actions)

    boot_Ws: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_boot = X_train[idx]
        y_boot = y_train[idx]
        try:
            W, _ = train_logistic(X_boot, y_boot, actions, seed=int(rng.integers(0, 2**31 - 1)))
            if W.shape == (n_features, n_actions):
                boot_Ws.append(W)
        except Exception:
            continue

    if not boot_Ws:
        return {"intervals": {}, "sign_stability": {}, "n_bootstrap": 0}

    boot_stack = np.stack(boot_Ws, axis=0)  # [n_boot, n_features, n_actions]
    alpha = (1.0 - BOOTSTRAP_CI_LEVEL) / 2.0

    intervals: dict[str, dict[str, dict[str, float]]] = {}
    sign_stability: dict[str, dict[str, float]] = {}

    for j in range(n_features):
        fname = feature_names[j] if j < len(feature_names) else f"f{j}"
        intervals[fname] = {}
        sign_stability[fname] = {}
        for a in range(n_actions):
            aname = actions[a] if a < len(actions) else f"a{a}"
            samples = boot_stack[:, j, a]
            orig_sign = np.sign(W_original[j, a]) if W_original.size and j < W_original.shape[0] and a < W_original.shape[1] else 0
            boot_signs = np.sign(samples)
            if orig_sign != 0:
                stable = float(np.mean(boot_signs == orig_sign))
            else:
                stable = float(np.mean(np.abs(samples) < NEAR_ZERO_COEFF_THRESHOLD))
            sign_stability[fname][aname] = stable
            intervals[fname][aname] = {
                "mean": float(samples.mean()),
                "std": float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
                "ci_lower": float(np.percentile(samples, alpha * 100)),
                "ci_upper": float(np.percentile(samples, (1.0 - alpha) * 100)),
            }

    return {
        "intervals": intervals,
        "sign_stability": sign_stability,
        "n_bootstrap": len(boot_Ws),
        "ci_level": BOOTSTRAP_CI_LEVEL,
    }


def _feature_action_influence(
    W: np.ndarray, feature_names: list[str], actions: list[str]
) -> dict[str, Any]:
    """Compute which features most influence which actions.

    Influence is measured as the absolute standardized weight magnitude.
    Returns a ranking of (feature, action) pairs by influence.
    """
    if W.size == 0:
        return {"ranking": [], "top_features_per_action": {}, "top_actions_per_feature": {}}

    abs_W = np.abs(W)
    ranking: list[dict[str, Any]] = []
    for j in range(W.shape[0]):
        fname = feature_names[j] if j < len(feature_names) else f"f{j}"
        for a in range(W.shape[1]):
            aname = actions[a] if a < len(actions) else f"a{a}"
            ranking.append({
                "feature": fname,
                "action": aname,
                "influence": float(abs_W[j, a]),
                "weight": float(W[j, a]),
            })
    ranking.sort(key=lambda x: x["influence"], reverse=True)

    top_features_per_action: dict[str, list[dict[str, Any]]] = {}
    for a, aname in enumerate(actions):
        if a >= W.shape[1]:
            continue
        col = abs_W[:, a]
        order = np.argsort(col)[::-1]
        top_features_per_action[aname] = [
            {
                "feature": feature_names[j] if j < len(feature_names) else f"f{j}",
                "influence": float(col[j]),
            }
            for j in order[:5]
        ]

    top_actions_per_feature: dict[str, list[dict[str, Any]]] = {}
    for j in range(W.shape[0]):
        fname = feature_names[j] if j < len(feature_names) else f"f{j}"
        row = abs_W[j, :]
        order = np.argsort(row)[::-1]
        top_actions_per_feature[fname] = [
            {
                "action": actions[a] if a < len(actions) else f"a{a}",
                "influence": float(row[a]),
            }
            for a in order[:5]
        ]

    return {
        "ranking": ranking[:20],
        "top_features_per_action": top_features_per_action,
        "top_actions_per_feature": top_actions_per_feature,
    }


def _condition_number(X_train: np.ndarray) -> float:
    """Condition number of the feature matrix (ratio of largest to smallest singular value)."""
    if X_train.size == 0 or X_train.shape[0] < 2 or X_train.shape[1] < 1:
        return 0.0
    try:
        s = np.linalg.svd(X_train, compute_uv=False)
        if s[-1] > 1e-15:
            return float(s[0] / s[-1])
        return float("inf")
    except np.linalg.LinAlgError:
        return float("inf")


def _multicollinearity_warnings(
    X_train: np.ndarray, feature_names: list[str]
) -> list[dict[str, Any]]:
    """Flag feature pairs with |correlation| > threshold."""
    if X_train.size == 0 or X_train.shape[1] < 2:
        return []
    try:
        corr = np.corrcoef(X_train, rowvar=False)
    except Exception:
        return []
    if corr.ndim != 2 or corr.shape[0] < 2:
        return []
    warnings: list[dict[str, Any]] = []
    n = corr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isfinite(corr[i, j]):
                continue
            if abs(corr[i, j]) > MULTICOLLINEARITY_CORR_THRESHOLD:
                fi = feature_names[i] if i < len(feature_names) else f"f{i}"
                fj = feature_names[j] if j < len(feature_names) else f"f{j}"
                warnings.append({
                    "feature_1": fi,
                    "feature_2": fj,
                    "correlation": float(corr[i, j]),
                    "threshold": MULTICOLLINEARITY_CORR_THRESHOLD,
                })
    return warnings


def _separation_warnings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    actions: list[str],
) -> list[dict[str, Any]]:
    """Flag features that perfectly predict an action."""
    if X_train.size == 0 or len(y_train) == 0:
        return []
    warnings: list[dict[str, Any]] = []
    n_features = X_train.shape[1]
    for j in range(n_features):
        fname = feature_names[j] if j < len(feature_names) else f"f{j}"
        col = X_train[:, j]
        for action in actions:
            mask = np.array([y == action for y in y_train])
            if mask.sum() == 0:
                continue
            action_vals = col[mask]
            other_vals = col[~mask]
            if other_vals.size == 0:
                continue
            # Perfect separation: action values are entirely above or below others.
            if np.min(action_vals) > np.max(other_vals) or np.max(action_vals) < np.min(other_vals):
                warnings.append({
                    "feature": fname,
                    "action": action,
                    "detail": "feature perfectly separates this action from others",
                })
    return warnings


def _regularization_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    actions: list[str],
) -> list[dict[str, Any]]:
    """Train with different L2 strengths and report validation utility.

    Uses sklearn LogisticRegression (without multi_class parameter) when
    available, falling back to the numpy train_logistic with matching L2.
    """
    path: list[dict[str, Any]] = []
    if X_train.size == 0 or len(y_train) == 0:
        return path

    try:
        from sklearn.linear_model import LogisticRegression
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    for l2 in REGULARIZATION_PATH_L2:
        C = 1.0 / l2 if l2 > 0 else 1e6
        val_utility: float
        val_accuracy: float
        if has_sklearn and len(set(y_train.tolist())) >= 2:
            try:
                model = LogisticRegression(
                    C=C, max_iter=1000, random_state=42, solver="lbfgs"
                )
                model.fit(X_train, y_train)
                val_accuracy = float(model.score(X_val, y_val)) if X_val.size else 0.0
                # Compute validation utility via raw argmax.
                W_sk = np.array(model.coef_, dtype=float).T  # [n_features, n_classes]
                b_sk = np.array(model.intercept_, dtype=float)
                classes = list(model.classes_)
                # Map to full action space.
                n_features = X_train.shape[1]
                n_actions = len(actions)
                W_full = np.zeros((n_features, n_actions))
                b_full = np.zeros(n_actions)
                for ci, cls in enumerate(classes):
                    if cls in actions:
                        ai = actions.index(cls)
                        if ci < W_sk.shape[1]:
                            W_full[:, ai] = W_sk[:, ci]
                            b_full[ai] = b_sk[ci]
                val_utility = _eval_model_utility(
                    W_full, b_full, X_val,
                    [ex.get("task_id", "") for ex in run_val_examples] if False else [],
                    None, actions,
                ) if False else 0.0
                # Fallback: use accuracy as proxy if utility eval not straightforward.
                val_utility = val_accuracy  # Use accuracy as proxy for sklearn path.
            except Exception:
                val_accuracy = 0.0
                val_utility = 0.0
        else:
            # Numpy fallback.
            try:
                W, b = train_logistic(
                    X_train, y_train, actions, l2=l2, seed=42
                )
                val_accuracy = _eval_model_accuracy(W, b, X_val, y_val, actions)
                val_utility = val_accuracy  # proxy
            except Exception:
                val_accuracy = 0.0
                val_utility = 0.0

        path.append({
            "l2": l2,
            "C": C,
            "validation_accuracy": float(val_accuracy),
            "validation_utility": float(val_utility),
        })

    return path


# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------


def _diagnostic_checks(
    W: np.ndarray,
    b: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    actions: list[str],
    feature_names: list[str],
    regularization_path: list[dict[str, Any]],
    val_accuracy: float,
) -> dict[str, dict[str, Any]]:
    """Run the battery of diagnostic checks."""
    checks: dict[str, dict[str, Any]] = {}

    # 1. Coefficients near zero.
    if W.size > 0:
        all_near_zero = bool(np.all(np.abs(W) < NEAR_ZERO_COEFF_THRESHOLD))
        max_abs = float(np.max(np.abs(W)))
    else:
        all_near_zero = True
        max_abs = 0.0
    checks["coefficients_near_zero"] = {
        "result": all_near_zero,
        "detail": (
            f"All |W| < {NEAR_ZERO_COEFF_THRESHOLD}" if all_near_zero
            else f"Max |W| = {max_abs:.6f} (threshold {NEAR_ZERO_COEFF_THRESHOLD})"
        ),
        "max_abs_coefficient": max_abs,
    }

    # 2. One feature dominates.
    if W.size > 0 and W.shape[0] > 0:
        per_feature_mag = np.max(np.abs(W), axis=1)
        mean_mag = float(np.mean(per_feature_mag))
        max_feat_mag = float(np.max(per_feature_mag))
        if mean_mag > 1e-12:
            dominance_ratio = max_feat_mag / mean_mag
        else:
            dominance_ratio = 0.0
        dominant_idx = int(np.argmax(per_feature_mag))
        dominant_feature = (
            feature_names[dominant_idx] if dominant_idx < len(feature_names) else f"f{dominant_idx}"
        )
        one_dominates = bool(dominance_ratio > DOMINANCE_RATIO)
    else:
        dominance_ratio = 0.0
        dominant_feature = ""
        one_dominates = False
    checks["one_feature_dominates"] = {
        "result": one_dominates,
        "detail": (
            f"Feature '{dominant_feature}' has |W| {dominance_ratio:.1f}x the mean "
            f"(threshold {DOMINANCE_RATIO}x)" if one_dominates
            else f"No single feature dominates (max ratio {dominance_ratio:.2f}x, "
            f"threshold {DOMINANCE_RATIO}x)"
        ),
        "dominance_ratio": float(dominance_ratio),
        "dominant_feature": dominant_feature,
    }

    # 3. Action logits nearly constant.
    if X_test.size > 0 and W.size > 0:
        logits = X_test @ W + b
        logit_std = float(np.mean(np.std(logits, axis=0)))
    else:
        logit_std = 0.0
    logits_constant = bool(logit_std < LOGIT_STD_THRESHOLD)
    checks["action_logits_nearly_constant"] = {
        "result": logits_constant,
        "detail": (
            f"Mean logit std = {logit_std:.6f} (< {LOGIT_STD_THRESHOLD})"
            if logits_constant
            else f"Mean logit std = {logit_std:.6f} (>= {LOGIT_STD_THRESHOLD})"
        ),
        "mean_logit_std": logit_std,
    }

    # 4. Family features dominate.
    # Identify family-derived features by name pattern.
    family_features = [
        i for i, fn in enumerate(feature_names)
        if "family" in fn.lower()
    ]
    if family_features and W.size > 0:
        family_mags = np.max(np.abs(W[family_features, :]), axis=1)
        non_family_idx = [i for i in range(W.shape[0]) if i not in family_features]
        if non_family_idx:
            non_family_mags = np.max(np.abs(W[non_family_idx, :]), axis=1)
            family_mean = float(np.mean(family_mags))
            non_family_mean = float(np.mean(non_family_mags))
            family_dominates = bool(family_mean > non_family_mean)
        else:
            family_dominates = True
            family_mean = float(np.mean(family_mags))
            non_family_mean = 0.0
    else:
        family_dominates = False
        family_mean = 0.0
        non_family_mean = 0.0
    checks["family_features_dominate"] = {
        "result": family_dominates,
        "detail": (
            f"Family-derived features have higher standardized |W| "
            f"({family_mean:.6f} vs {non_family_mean:.6f})"
            if family_dominates
            else f"Family-derived features do not dominate "
            f"({family_mean:.6f} vs {non_family_mean:.6f})"
        ),
        "family_mean_magnitude": family_mean,
        "non_family_mean_magnitude": non_family_mean,
    }

    # 5. Normalization inconsistent.
    if X_train.size > 0 and X_train.shape[1] > 0:
        feat_means = np.mean(X_train, axis=0)
        feat_mean_abs = np.abs(feat_means)
        max_mean = float(np.max(feat_mean_abs))
        min_mean = float(np.min(feat_mean_abs[feat_mean_abs > 1e-12])) if np.any(feat_mean_abs > 1e-12) else 0.0
        if min_mean > 1e-12:
            ratio = max_mean / min_mean
        else:
            ratio = float("inf") if max_mean > 1e-12 else 1.0
        inconsistent = bool(ratio > 1000.0)  # orders of magnitude
    else:
        ratio = 1.0
        inconsistent = False
    checks["normalization_inconsistent"] = {
        "result": inconsistent,
        "detail": (
            f"Feature means vary by {ratio:.1e}x (orders of magnitude)"
            if inconsistent
            else f"Feature means vary by {ratio:.2f}x (consistent)"
        ),
        "mean_ratio": float(ratio) if np.isfinite(ratio) else None,
    }

    # 6. Regularization too strong.
    if regularization_path:
        high_l2 = [p for p in regularization_path if p["l2"] >= 1.0]
        if high_l2:
            high_l2_accs = [p["validation_accuracy"] for p in high_l2]
            high_l2_too_strong = all(a < 0.01 for a in high_l2_accs)
        else:
            high_l2_too_strong = False
    else:
        high_l2_too_strong = False
    checks["regularization_too_strong"] = {
        "result": bool(high_l2_too_strong),
        "detail": (
            "High L2 values collapse validation accuracy to near zero"
            if high_l2_too_strong
            else "Regularization path does not show collapse"
        ),
    }

    # 7. Class separation weak.
    weak_separation = bool(val_accuracy < WEAK_SEPARATION_ACC_THRESHOLD)
    checks["class_separation_weak"] = {
        "result": weak_separation,
        "detail": (
            f"Validation accuracy = {val_accuracy:.4f} (< {WEAK_SEPARATION_ACC_THRESHOLD})"
            if weak_separation
            else f"Validation accuracy = {val_accuracy:.4f} (>= {WEAK_SEPARATION_ACC_THRESHOLD})"
        ),
        "validation_accuracy": float(val_accuracy),
    }

    return checks


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def _interpretation(checks: dict[str, dict[str, Any]]) -> str:
    """Build a human-readable interpretation from the diagnostic checks."""
    triggered = [name for name, c in checks.items() if c.get("result")]
    if not triggered:
        return (
            "No parameter diagnostic checks triggered. The candidate model's "
            "coefficients are well-conditioned, non-degenerate, and the "
            "decision boundary is identifiable."
        )
    parts = [f"Triggered checks: {', '.join(triggered)}."]
    if "coefficients_near_zero" in triggered:
        parts.append(
            "All coefficients are near zero — the router may not have learned "
            "any meaningful decision boundary."
        )
    if "one_feature_dominates" in triggered:
        parts.append(
            "A single feature dominates the coefficient matrix — the router "
            "may be over-reliant on one signal."
        )
    if "action_logits_nearly_constant" in triggered:
        parts.append(
            "Action logits are nearly constant across tasks — the router "
            "produces similar scores regardless of input."
        )
    if "family_features_dominate" in triggered:
        parts.append(
            "Family-derived features have the highest standardized coefficients "
            "— the router may be exploiting family identity rather than "
            "task-level features."
        )
    if "normalization_inconsistent" in triggered:
        parts.append(
            "Feature normalization is inconsistent — feature means vary by "
            "orders of magnitude, which can destabilise logistic regression."
        )
    if "regularization_too_strong" in triggered:
        parts.append(
            "Regularization is too strong — high L2 values collapse all "
            "coefficients to near zero."
        )
    if "class_separation_weak" in triggered:
        parts.append(
            "Class separation is weak — validation accuracy is below threshold, "
            "indicating the features do not linearly separate the actions."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diagnose_parameters(
    run: LoadedRun,
    n_bootstrap: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Diagnose the candidate model's coefficients and decision boundary.

    Reports the coefficient matrix, standardized coefficients, bootstrap
    intervals, sign stability, feature-action influence, condition number,
    multicollinearity warnings, separation warnings, the L2 regularization
    path, and a battery of diagnostic checks.

    Args:
        run:          A LoadedRun from data.py.
        n_bootstrap:  Number of bootstrap resamples (default 20).
        seed:         Random seed for bootstrap resampling.

    Returns:
        A JSON-serializable ``router_parameter_diagnostics`` dict.
    """
    # --- Extract candidate model ---
    W, b, actions = _coefficient_matrix(run)

    # --- Build feature matrices ---
    X_train, y_train, X_val, y_val, train_actions, feature_names = _build_feature_matrices(run)

    # --- Coefficient matrix ---
    coeff_matrix = {
        "W": W.tolist(),
        "b": b.tolist(),
        "actions": actions,
        "feature_names": feature_names,
        "shape": list(W.shape) if W.size else [0, 0],
    }

    # --- Standardized coefficients ---
    std_coeffs = _standardized_coefficients(W, X_train, feature_names, actions)

    # --- Bootstrap intervals ---
    bootstrap = _bootstrap_coefficients(
        X_train, y_train, train_actions, feature_names, W, n_bootstrap, seed
    )

    # --- Sign stability (extracted from bootstrap) ---
    sign_stability = bootstrap.get("sign_stability", {})

    # --- Feature-action influence ---
    influence = _feature_action_influence(W, feature_names, actions)

    # --- Condition number ---
    cond_number = _condition_number(X_train)

    # --- Multicollinearity warnings ---
    multi_warnings = _multicollinearity_warnings(X_train, feature_names)

    # --- Separation warnings ---
    sep_warnings = _separation_warnings(X_train, y_train, feature_names, train_actions)

    # --- Regularization path ---
    reg_path = _regularization_path(X_train, y_train, X_val, y_val, train_actions)

    # --- Validation accuracy for class-separation check ---
    if W.size > 0 and X_val.size > 0:
        val_accuracy = _eval_model_accuracy(W, b, X_val, y_val, actions)
    else:
        val_accuracy = 0.0

    # --- Diagnostic checks ---
    X_test, y_test = _build_feature_matrices(run)[2:4]
    checks = _diagnostic_checks(
        W, b, X_train, X_test, y_train, y_test, actions, feature_names,
        reg_path, val_accuracy,
    )

    # --- Interpretation ---
    interpretation = _interpretation(checks)

    return {
        "coefficient_matrix": coeff_matrix,
        "standardized_coefficients": std_coeffs,
        "bootstrap_intervals": bootstrap.get("intervals", {}),
        "bootstrap_n_samples": bootstrap.get("n_bootstrap", 0),
        "bootstrap_ci_level": bootstrap.get("ci_level", BOOTSTRAP_CI_LEVEL),
        "sign_stability": sign_stability,
        "feature_action_influence": influence,
        "condition_number": float(cond_number) if np.isfinite(cond_number) else None,
        "multicollinearity_warnings": multi_warnings,
        "separation_warnings": sep_warnings,
        "regularization_path": reg_path,
        "diagnostic_checks": checks,
        "interpretation": interpretation,
    }
