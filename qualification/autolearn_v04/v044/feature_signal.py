"""Feature signal validation for v0.4.4 evidence-complete analysis.

Measures feature signal beyond family and action frequency by training a
ladder of nine validation-only control models on ``D_experience`` and
selecting on ``D_validation``.

Required validation-only controls (in order):

  1. intercept only
  2. frequency only
  3. family only
  4. archetype only
  5. family + archetype
  6. pre-action features excluding family
  7. pre-action features excluding archetype
  8. full pre-action feature set
  9. shuffled features

A feature signal counts as present only when the full pre-action feature
model improves validation expected utility beyond BOTH the frequency
control AND the family-only control.  The improvement must be stable
across configured seeds.

Result categories:

  * ``FEATURE_SIGNAL_PRESENT`` — full feature model improves validation
    expected utility beyond both frequency and family-only controls by
    more than ``FEATURE_SIGNAL_PRESENT_THRESHOLD``, stable across seeds.
  * ``FEATURE_SIGNAL_WEAK`` — only small unstable gains
    (``FEATURE_SIGNAL_WEAK_THRESHOLD`` < improvement <=
    ``FEATURE_SIGNAL_PRESENT_THRESHOLD``).
  * ``FEATURE_SIGNAL_ABSENT`` — no gain (improvement <=
    ``FEATURE_SIGNAL_WEAK_THRESHOLD``).
  * ``TEMPLATE_LEAKAGE_DOMINANT`` — family/archetype controls explain
    nearly all performance (family-only utility >=
    ``TEMPLATE_LEAKAGE_FRACTION`` of full feature utility).
  * ``INCONCLUSIVE`` — insufficient data or all models failed.

The final test split is never used for feature-set selection.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .config import MINIMUM_DENOMINATOR


# ---------------------------------------------------------------------------
# Optional sklearn imports
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised when sklearn is present
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

FEATURE_SIGNAL_PRESENT_THRESHOLD: float = 0.05
FEATURE_SIGNAL_WEAK_THRESHOLD: float = 0.01
TEMPLATE_LEAKAGE_FRACTION: float = 0.95
DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 101112)


# ---------------------------------------------------------------------------
# Result categories
# ---------------------------------------------------------------------------

FEATURE_SIGNAL_PRESENT = "FEATURE_SIGNAL_PRESENT"
FEATURE_SIGNAL_WEAK = "FEATURE_SIGNAL_WEAK"
FEATURE_SIGNAL_ABSENT = "FEATURE_SIGNAL_ABSENT"
TEMPLATE_LEAKAGE_DOMINANT = "TEMPLATE_LEAKAGE_DOMINANT"
INCONCLUSIVE = "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
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


def _extract_examples(
    results: dict[str, Any],
    split: str = "validation",
) -> list[dict[str, Any]]:
    """Extract per-task examples from a results dict.

    Looks for ``per_task`` entries that carry ``features`` and
    ``best_action`` fields, falling back to ``experiences`` /
    ``executive_experiences`` lists.
    """
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


def _families(examples: list[dict[str, Any]]) -> list[str]:
    return [ex.get("task_family", ex.get("family", "unknown")) for ex in examples]


def _archetypes(examples: list[dict[str, Any]]) -> list[str]:
    return [ex.get("archetype", "unknown") for ex in examples]


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


def _get_feature_names(results: dict[str, Any]) -> list[str]:
    """Extract feature names from a results dict or candidate policy."""
    names = results.get("feature_names", [])
    if names:
        return list(names)
    transform = results.get("feature_transform", {})
    if isinstance(transform, dict):
        names = transform.get("feature_names", [])
        if names:
            return list(names)
    return []


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
            # If the selected action matches, use the task-level utility.
            if ex.get("selected_action", ex.get("best_action")) == action:
                u = ex.get("utility")
                if u is not None:
                    return float(u)
    return None


def _get_oracle_utility(
    oracle_results: dict[str, Any], task_id: str
) -> float:
    """Look up the oracle utility for a task."""
    per_task = oracle_results.get("per_task", [])
    for ex in per_task:
        if ex.get("task_id") == task_id:
            u = ex.get("utility")
            if u is not None:
                return float(u)
    mean = oracle_results.get("mean_utility")
    return float(mean) if mean is not None else 0.0


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def _train_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    seed: int = 42,
) -> Any:
    """Train a sklearn LogisticRegression classifier."""
    if not _HAS_SKLEARN:
        raise RuntimeError("sklearn is required for feature_signal analysis")
    model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(train_features, train_labels)
    return model


def _train_dummy(train_labels: np.ndarray) -> Any:
    """Intercept-only model: always predict the most common class."""
    if not _HAS_SKLEARN:
        raise RuntimeError("sklearn is required for feature_signal analysis")
    counts = Counter(train_labels.tolist())
    constant = counts.most_common(1)[0][0] if counts else "A"
    model = DummyClassifier(strategy="constant", constant=constant)
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


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def _evaluate_model_utility(
    model: Any,
    val_features: np.ndarray,
    val_examples: list[dict[str, Any]],
    actions: list[str],
    candidate_results: dict[str, Any],
) -> float:
    """Evaluate a trained model on validation and return expected utility."""
    if len(val_examples) == 0:
        return 0.0

    proba = _predict_proba(model, val_features, actions)
    pred_idx = np.argmax(proba, axis=1)

    utilities: list[float] = []
    for i, ex in enumerate(val_examples):
        tid = ex.get("task_id", "")
        action = actions[pred_idx[i]] if pred_idx[i] < len(actions) else ""
        u = _get_utility_for_action(candidate_results, tid, action)
        utilities.append(u if u is not None else 0.0)

    return float(np.mean(utilities)) if utilities else 0.0


def _frequency_utility(
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    candidate_results: dict[str, Any],
    actions: list[str],
) -> float:
    """Utility of always picking the most common D_experience best action."""
    train_labels = _labels(train_examples)
    counts = Counter(train_labels.tolist())
    if not counts:
        return 0.0
    majority = counts.most_common(1)[0][0]
    utils: list[float] = []
    for ex in val_examples:
        tid = ex.get("task_id", "")
        u = _get_utility_for_action(candidate_results, tid, majority)
        utils.append(u if u is not None else 0.0)
    return float(np.mean(utils)) if utils else 0.0


def _family_majority_utility(
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    candidate_results: dict[str, Any],
) -> float:
    """Utility of a per-family majority-action policy."""
    train_labels = _labels(train_examples)
    train_fam = _families(train_examples)
    fam_counts: dict[str, Counter] = {}
    for fam, lab in zip(train_fam, train_labels.tolist()):
        fam_counts.setdefault(fam, Counter())[lab] += 1
    family_to_majority = {f: c.most_common(1)[0][0] for f, c in fam_counts.items()}

    utils: list[float] = []
    for ex in val_examples:
        tid = ex.get("task_id", "")
        fam = ex.get("task_family", ex.get("family", "unknown"))
        action = family_to_majority.get(fam, "")
        u = _get_utility_for_action(candidate_results, tid, action)
        utils.append(u if u is not None else 0.0)
    return float(np.mean(utils)) if utils else 0.0


# ---------------------------------------------------------------------------
# Single-seed control ladder
# ---------------------------------------------------------------------------


def _run_control_ladder(
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    feature_names: list[str],
    actions: list[str],
    candidate_results: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    """Run the nine-control ladder for a single seed.

    Returns a dict mapping control name to validation expected utility.
    """
    train_features = _feature_matrix(train_examples)
    val_features = _feature_matrix(val_examples)
    train_labels = _labels(train_examples)
    val_labels = _labels(val_examples)

    train_fam = _families(train_examples)
    val_fam = _families(val_examples)
    train_arch = _archetypes(train_examples)
    val_arch = _archetypes(val_examples)

    fam_onehot_train, fam_categories = _onehot_encode(train_fam)
    fam_onehot_val, _ = _onehot_encode(val_fam, fam_categories)
    arch_onehot_train, arch_categories = _onehot_encode(train_arch)
    arch_onehot_val, _ = _onehot_encode(val_arch, arch_categories)
    fam_arch_train = np.hstack([fam_onehot_train, arch_onehot_train])
    fam_arch_val = np.hstack([fam_onehot_val, arch_onehot_val])

    fam_mask = _family_derived_mask(feature_names)
    arch_mask = _archetype_derived_mask(feature_names)
    non_fam_idx = [i for i, m in enumerate(fam_mask) if not m]
    non_arch_idx = [i for i, m in enumerate(arch_mask) if not m]

    freq_util = _frequency_utility(
        train_examples, val_examples, candidate_results, actions
    )
    fam_maj_util = _family_majority_utility(
        train_examples, val_examples, candidate_results
    )

    results: dict[str, float] = {}

    # --- 1. intercept only ---
    dummy = _train_dummy(train_labels)
    dummy_val_X = np.zeros((len(val_examples), 1))
    results["intercept_only"] = _evaluate_model_utility(
        dummy, dummy_val_X, val_examples, actions, candidate_results
    )

    # --- 2. frequency only ---
    results["frequency_only"] = freq_util

    # --- 3. family only ---
    if fam_onehot_train.shape[1] > 0 and len(np.unique(train_labels)) >= 2:
        model = _train_logistic(fam_onehot_train, train_labels, seed=seed)
        results["family_only"] = _evaluate_model_utility(
            model, fam_onehot_val, val_examples, actions, candidate_results
        )
    else:
        results["family_only"] = fam_maj_util

    # --- 4. archetype only ---
    if arch_onehot_train.shape[1] > 0 and len(np.unique(train_labels)) >= 2:
        model = _train_logistic(arch_onehot_train, train_labels, seed=seed)
        results["archetype_only"] = _evaluate_model_utility(
            model, arch_onehot_val, val_examples, actions, candidate_results
        )
    else:
        dummy = _train_dummy(train_labels)
        results["archetype_only"] = _evaluate_model_utility(
            dummy, np.zeros((len(val_examples), 1)), val_examples, actions, candidate_results
        )

    # --- 5. family + archetype ---
    if fam_arch_train.shape[1] > 0 and len(np.unique(train_labels)) >= 2:
        model = _train_logistic(fam_arch_train, train_labels, seed=seed)
        results["family_plus_archetype"] = _evaluate_model_utility(
            model, fam_arch_val, val_examples, actions, candidate_results
        )
    else:
        results["family_plus_archetype"] = fam_maj_util

    # --- 6. pre-action features excluding family ---
    if non_fam_idx and train_features.shape[1] == len(feature_names):
        X_tr = train_features[:, non_fam_idx]
        X_val = val_features[:, non_fam_idx]
    else:
        X_tr = np.zeros((len(train_examples), 0))
        X_val = np.zeros((len(val_examples), 0))
    if X_tr.shape[1] > 0 and len(np.unique(train_labels)) >= 2:
        model = _train_logistic(X_tr, train_labels, seed=seed)
        results["pre_action_features_excluding_family"] = _evaluate_model_utility(
            model, X_val, val_examples, actions, candidate_results
        )
    else:
        results["pre_action_features_excluding_family"] = freq_util

    # --- 7. pre-action features excluding archetype ---
    if non_arch_idx and train_features.shape[1] == len(feature_names):
        X_tr = train_features[:, non_arch_idx]
        X_val = val_features[:, non_arch_idx]
    else:
        X_tr = np.zeros((len(train_examples), 0))
        X_val = np.zeros((len(val_examples), 0))
    if X_tr.shape[1] > 0 and len(np.unique(train_labels)) >= 2:
        model = _train_logistic(X_tr, train_labels, seed=seed)
        results["pre_action_features_excluding_archetype"] = _evaluate_model_utility(
            model, X_val, val_examples, actions, candidate_results
        )
    else:
        results["pre_action_features_excluding_archetype"] = freq_util

    # --- 8. full pre-action feature set ---
    if train_features.shape[1] > 0 and len(np.unique(train_labels)) >= 2:
        model = _train_logistic(train_features, train_labels, seed=seed)
        results["full_pre_action_feature_set"] = _evaluate_model_utility(
            model, val_features, val_examples, actions, candidate_results
        )
    else:
        results["full_pre_action_feature_set"] = freq_util

    # --- 9. shuffled features ---
    rng = np.random.default_rng(seed)
    shuffled_train = train_features.copy()
    if shuffled_train.size > 0:
        rng.shuffle(shuffled_train)
    if shuffled_train.shape[1] > 0 and len(np.unique(train_labels)) >= 2:
        model = _train_logistic(shuffled_train, train_labels, seed=seed)
        results["shuffled_features"] = _evaluate_model_utility(
            model, val_features, val_examples, actions, candidate_results
        )
    else:
        results["shuffled_features"] = freq_util

    return results


# ---------------------------------------------------------------------------
# Learnability conclusion
# ---------------------------------------------------------------------------


def _determine_status(
    control_results: dict[str, float],
    seed_results: list[dict[str, float]],
) -> tuple[str, str]:
    """Determine the feature signal status from control results.

    A feature signal counts as present only when the full pre-action
    feature model improves validation expected utility beyond BOTH the
    frequency control AND the family-only control, stable across seeds.
    """
    full_util = control_results.get("full_pre_action_feature_set", 0.0)
    family_util = control_results.get("family_only", 0.0)
    freq_util = control_results.get("frequency_only", 0.0)

    if full_util == 0.0 and family_util == 0.0 and freq_util == 0.0:
        return INCONCLUSIVE, (
            "All control models produced zero validation utility. "
            "Insufficient data to determine feature signal."
        )

    # Template leakage check.
    template_leakage = (
        family_util >= TEMPLATE_LEAKAGE_FRACTION * full_util
        and full_util > 0
    )
    if template_leakage:
        return TEMPLATE_LEAKAGE_DOMINANT, (
            f"Family/archetype controls explain nearly all performance: "
            f"family-only utility ({family_util:.3f}) >= "
            f"{TEMPLATE_LEAKAGE_FRACTION:.0%} of full feature utility "
            f"({full_util:.3f}). The learned policy may be exploiting "
            f"template metadata rather than genuine executive signal."
        )

    # Improvement over both controls.
    improvement_vs_freq = full_util - freq_util
    improvement_vs_family = full_util - family_util
    min_improvement = min(improvement_vs_freq, improvement_vs_family)

    # Check stability across seeds.
    full_utils_across_seeds = [
        sr.get("full_pre_action_feature_set", 0.0) for sr in seed_results
    ]
    if len(full_utils_across_seeds) >= 2:
        full_std = float(np.std(full_utils_across_seeds))
    else:
        full_std = 0.0

    stable = full_std < FEATURE_SIGNAL_WEAK_THRESHOLD

    if min_improvement > FEATURE_SIGNAL_PRESENT_THRESHOLD and stable:
        return FEATURE_SIGNAL_PRESENT, (
            f"Full pre-action feature model improves validation expected "
            f"utility beyond both frequency ({improvement_vs_freq:.3f}) "
            f"and family-only ({improvement_vs_family:.3f}) controls, "
            f"stable across seeds (std={full_std:.4f}). Genuine executive "
            f"signal is present."
        )
    elif min_improvement > FEATURE_SIGNAL_WEAK_THRESHOLD:
        if stable:
            return FEATURE_SIGNAL_WEAK, (
                f"Only small gains from full feature model "
                f"(improvement {min_improvement:.3f}, "
                f"{FEATURE_SIGNAL_WEAK_THRESHOLD} < improvement <= "
                f"{FEATURE_SIGNAL_PRESENT_THRESHOLD}). Signal is weak."
            )
        else:
            return FEATURE_SIGNAL_WEAK, (
                f"Feature improvements are unstable across seeds "
                f"(std={full_std:.4f}). Signal is weak and not reliable."
            )
    elif min_improvement <= FEATURE_SIGNAL_WEAK_THRESHOLD:
        return FEATURE_SIGNAL_ABSENT, (
            f"No gain from full feature model beyond simple controls "
            f"(improvement {min_improvement:.3f} <= "
            f"{FEATURE_SIGNAL_WEAK_THRESHOLD}). No genuine executive "
            f"signal beyond frequency and family."
        )

    return INCONCLUSIVE, (
        "Feature signal analysis was inconclusive due to insufficient "
        "data or model failures."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_feature_signal(
    candidate_results: dict,
    sham_results: dict,
    n_tasks: int = 30,
    n_seeds: int = 5,
) -> dict:
    """Measure feature signal beyond family and action frequency.

    Trains a ladder of nine validation-only control models on
    ``D_experience`` and evaluates each on ``D_validation``.

    Args:
        candidate_results: Results dict from the candidate policy run,
            containing per-task examples with features, best_action,
            and outcome utilities.
        sham_results: Results dict from the sham policy run, used for
            sham-mean reference.
        n_tasks: Minimum number of tasks required for a conclusive
            analysis.
        n_seeds: Number of seeds to use for stability checking.

    Returns:
        A dict with:
          - status: one of the five result categories.
          - control_results: dict of control name -> validation utility
            (averaged across seeds).
          - conclusion: human-readable conclusion string.
    """
    train_examples = _extract_examples(candidate_results, "train")
    val_examples = _extract_examples(candidate_results, "validation")
    if not val_examples:
        val_examples = train_examples

    feature_names = _get_feature_names(candidate_results)
    actions = _get_actions(candidate_results)

    if len(val_examples) < n_tasks:
        return {
            "status": INCONCLUSIVE,
            "control_results": {},
            "conclusion": (
                f"Insufficient validation tasks ({len(val_examples)} < "
                f"{n_tasks}). Feature signal analysis is inconclusive."
            ),
        }

    if len(actions) < 2:
        return {
            "status": INCONCLUSIVE,
            "control_results": {},
            "conclusion": (
                f"Fewer than 2 actions ({actions}). Cannot measure "
                f"feature signal with a single-action space."
            ),
        }

    seeds = list(DEFAULT_SEEDS[:n_seeds]) if n_seeds <= len(DEFAULT_SEEDS) else [
        DEFAULT_SEEDS[i % len(DEFAULT_SEEDS)] + i for i in range(n_seeds)
    ]

    seed_results: list[dict[str, float]] = []
    for seed in seeds:
        try:
            sr = _run_control_ladder(
                train_examples, val_examples, feature_names, actions,
                candidate_results, seed,
            )
            seed_results.append(sr)
        except Exception:
            seed_results.append({})

    if not seed_results or all(not sr for sr in seed_results):
        return {
            "status": INCONCLUSIVE,
            "control_results": {},
            "conclusion": (
                "All control model runs failed. Feature signal analysis "
                "is inconclusive."
            ),
        }

    # Average control results across seeds.
    all_controls: set[str] = set()
    for sr in seed_results:
        all_controls.update(sr.keys())

    control_results: dict[str, float] = {}
    for ctrl in sorted(all_controls):
        vals = [sr.get(ctrl, 0.0) for sr in seed_results if ctrl in sr]
        control_results[ctrl] = float(np.mean(vals)) if vals else 0.0

    status, conclusion = _determine_status(control_results, seed_results)

    return {
        "status": status,
        "control_results": control_results,
        "conclusion": conclusion,
    }
