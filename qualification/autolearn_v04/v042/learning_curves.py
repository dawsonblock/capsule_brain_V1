"""Learning-curve analysis for v0.4.2 diagnostics.

Retrains candidate and sham policies on nested subsets of the existing
experience pool and evaluates each on the held-out test split to determine
whether the candidate-minus-sham gap improves with more training data.

The analysis classifies the learner/benchmark regime into one of:

  DATA_LIMITED      — candidate-sham improves consistently with n
  FEATURE_LIMITED   — candidate-sham flat near zero
  HEADROOM_LIMITED  — oracle-sham is small at all sizes
  LEARNER_LIMITED   — train accuracy rises but test utility doesn't
  INCONCLUSIVE      — none of the above patterns is clearly dominant
"""
from __future__ import annotations

import numpy as np
from typing import Any

from .data import LoadedRun
from .sham_trainer import (
    train_candidate_policy,
    train_sham_policy,
    candidate_ensemble,
    sham_ensemble,
)
from .policy_controls import (
    evaluate_policy_on_tasks,
    make_baseline_selector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_feature_lookup(run: LoadedRun) -> dict[str, np.ndarray]:
    """Map task_id -> feature vector for every test example."""
    lookup: dict[str, np.ndarray] = {}
    for ex in run.test_examples:
        lookup[ex["task_id"]] = np.array(ex.get("features", []))
    return lookup


def evaluate_trained_policy_on_test(
    run: LoadedRun,
    weights: np.ndarray,
    bias: np.ndarray,
    actions: list[str],
    test_task_ids: list[str],
) -> dict[str, Any]:
    """Evaluate a trained linear policy on the test split.

    For each test task:
      1. Look up the example feature vector from ``run.test_examples``.
      2. Compute logits = features @ weights + bias.
      3. Restrict to actions eligible for that task (``task.allowed_actions``).
      4. Select the eligible action with the highest logit.
      5. Look up the measured utility for that (task, action) pair.

    Args:
        run: loaded run data.
        weights: trained weight matrix [n_features, n_actions].
        bias: trained bias vector [n_actions].
        actions: action names corresponding to weight/bias columns.
        test_task_ids: test task ids to evaluate on.

    Returns:
        dict with ``mean_utility``, ``utilities`` (per-task list aligned with
        ``test_task_ids``), ``selected_actions``, ``classification_accuracy``,
        and ``action_entropy``.
    """
    feature_lookup = _build_test_feature_lookup(run)
    action_idx = {a: i for i, a in enumerate(actions)}

    utilities: list[float] = []
    selected_actions: list[str] = []
    best_actions: list[str] = []
    action_counts: dict[str, int] = {a: 0 for a in actions}

    for tid in test_task_ids:
        task = run.tasks.get(tid)
        if task is None:
            utilities.append(0.0)
            selected_actions.append("")
            best_actions.append("")
            continue

        features = feature_lookup.get(tid)
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if features is None or not eligible or weights.size == 0:
            # Fall back to first eligible action (or zero).
            fallback = eligible[0] if eligible else ""
            u = run.get_utility_for_action(tid, fallback) if fallback else 0.0
            utilities.append(u if u is not None else 0.0)
            selected_actions.append(fallback)
            best_actions.append(run.get_best_action(tid) or "")
            if fallback:
                action_counts[fallback] = action_counts.get(fallback, 0) + 1
            continue

        logits = features @ weights + bias
        eligible_logits = np.array([logits[action_idx[a]] for a in eligible])
        chosen = eligible[int(np.argmax(eligible_logits))]

        u = run.get_utility_for_action(tid, chosen)
        utilities.append(u if u is not None else 0.0)
        selected_actions.append(chosen)
        best_actions.append(run.get_best_action(tid) or "")
        action_counts[chosen] = action_counts.get(chosen, 0) + 1

    utilities_arr = np.array(utilities)

    # Classification accuracy: fraction of tasks where selected == best.
    correct = sum(
        1 for sa, ba in zip(selected_actions, best_actions)
        if sa and ba and sa == ba
    )
    n_eval = sum(1 for sa in selected_actions if sa)
    classification_accuracy = correct / n_eval if n_eval > 0 else 0.0

    # Action entropy over selected actions.
    total = sum(action_counts.values())
    if total > 0:
        probs = np.array([action_counts.get(a, 0) / total for a in actions])
        nonzero = probs[probs > 0]
        action_entropy = float(-np.sum(nonzero * np.log(nonzero)))
    else:
        action_entropy = 0.0

    return {
        "mean_utility": float(utilities_arr.mean()) if len(utilities_arr) else 0.0,
        "utilities": utilities_arr.tolist(),
        "selected_actions": selected_actions,
        "classification_accuracy": classification_accuracy,
        "action_entropy": action_entropy,
        "n_evaluated": n_eval,
    }


def _oracle_utilities(run: LoadedRun, test_task_ids: list[str]) -> np.ndarray:
    """Per-task oracle (max measured) utility for the test split."""
    return np.array([run.get_oracle_utility(tid) for tid in test_task_ids])


def _baseline_utilities(run: LoadedRun, test_task_ids: list[str]) -> np.ndarray:
    """Per-task baseline utility for the test split."""
    baseline_sel = make_baseline_selector(run)
    res = evaluate_policy_on_tasks(run, test_task_ids, baseline_sel, "baseline")
    return np.array(res["utilities"])


def _fit_trend(xs: np.ndarray, ys: np.ndarray) -> dict[str, float]:
    """Cautious ordinary-least-squares linear trend.

    Returns slope, intercept, r_squared, and the slope's standard error.
    The fit is "cautious" in that we only trust a positive trend when
    the slope is at least one standard error above zero.
    """
    if len(xs) < 2:
        return {"slope": 0.0, "intercept": float(ys[0]) if len(ys) else 0.0,
                "r_squared": 0.0, "slope_se": float("inf")}
    x = xs.astype(float)
    y = ys.astype(float)
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = float(np.sum((x - x_mean) ** 2))
    sxy = float(np.sum((x - x_mean) * (y - y_mean)))
    syy = float(np.sum((y - y_mean) ** 2))
    if sxx < 1e-12:
        return {"slope": 0.0, "intercept": y_mean, "r_squared": 0.0,
                "slope_se": float("inf")}
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    residuals = y - (slope * x + intercept)
    ss_res = float(np.sum(residuals ** 2))
    r_squared = 1.0 - ss_res / syy if syy > 1e-12 else 0.0
    # Standard error of the slope.
    if n > 2 and ss_res > 0:
        sigma2 = ss_res / (n - 2)
        slope_se = float(np.sqrt(sigma2 / sxx))
    else:
        slope_se = 0.0 if ss_res <= 0 else float("inf")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "slope_se": slope_se,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_learning_curves(
    run: LoadedRun,
    n_seeds: int = 20,
    sizes: list[int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute learning curves over nested subsets of the experience pool.

    Uses the existing ``run.train_examples`` as the experience pool. The first
    ``N`` examples form the nested subset of size ``N`` (larger sets strictly
    contain smaller sets). Each subset is used to retrain ``n_seeds`` candidate
    and ``n_seeds`` sham policies, which are then evaluated on
    ``run.test_task_ids``.

    Args:
        run: loaded run data.
        n_seeds: number of candidate/sham seeds per size (default 20).
        sizes: nested subset sizes (default ``[10, 20, 30, 40, 50]`` or the
            largest available sizes that do not exceed the pool).
        seed: base random seed for reproducibility.

    Returns:
        dict suitable for serialization as ``learning_curves.json``.
    """
    pool = run.train_examples
    pool_size = len(pool)
    test_task_ids = run.test_task_ids

    # Determine the sizes to use.
    if sizes is None:
        default_sizes = [10, 20, 30, 40, 50]
        sizes = [s for s in default_sizes if s <= pool_size]
        if not sizes:
            sizes = [pool_size] if pool_size > 0 else []
    else:
        sizes = [s for s in sizes if s <= pool_size and s > 0]
    sizes = sorted(set(sizes))

    # Precompute oracle and baseline per-task utilities (constant across sizes).
    u_oracle = _oracle_utilities(run, test_task_ids)
    u_baseline = _baseline_utilities(run, test_task_ids)
    oracle_mean = float(u_oracle.mean()) if len(u_oracle) else 0.0
    baseline_mean = float(u_baseline.mean()) if len(u_baseline) else 0.0

    per_size: list[dict[str, Any]] = []
    candidate_sham_means: list[float] = []
    oracle_sham_means: list[float] = []
    train_acc_means: list[float] = []
    test_util_means: list[float] = []

    for size in sizes:
        # Build a temporary run-like view with the nested subset.
        subset = pool[:size]

        # We train by temporarily replacing run.train_examples. To avoid
        # mutating the caller's object, we swap and restore.
        original_train = run.train_examples
        run.train_examples = subset
        try:
            cand_policies = candidate_ensemble(run, n_seeds=n_seeds, base_seed=seed)
            sham_policies = sham_ensemble(run, n_seeds=n_seeds, base_seed=seed + 5000)
        finally:
            run.train_examples = original_train

        actions = run.all_actions

        # Evaluate each candidate and sham policy on the test split.
        cand_results: list[dict[str, Any]] = []
        sham_results: list[dict[str, Any]] = []
        cand_utilities: list[np.ndarray] = []
        sham_utilities: list[np.ndarray] = []
        cand_train_accs: list[float] = []
        sham_train_accs: list[float] = []

        for pol in cand_policies:
            W = np.array(pol.get("weights", []))
            b = np.array(pol.get("bias", []))
            res = evaluate_trained_policy_on_test(run, W, b, actions, test_task_ids)
            cand_results.append(res)
            cand_utilities.append(np.array(res["utilities"]))
            # Train accuracy: how often the policy picks the labelled best
            # action on the subset it was trained on.
            cand_train_accs.append(_train_accuracy(run, W, b, actions, subset))

        for pol in sham_policies:
            W = np.array(pol.get("weights", []))
            b = np.array(pol.get("bias", []))
            res = evaluate_trained_policy_on_test(run, W, b, actions, test_task_ids)
            sham_results.append(res)
            sham_utilities.append(np.array(res["utilities"]))
            sham_train_accs.append(_train_accuracy(run, W, b, actions, subset))

        # Aggregate across seeds (mean over seeds, then mean over tasks).
        cand_util_matrix = np.array(cand_utilities)  # [n_seeds, n_tasks]
        sham_util_matrix = np.array(sham_utilities)
        cand_util_per_seed = cand_util_matrix.mean(axis=1)  # [n_seeds]
        sham_util_per_seed = sham_util_matrix.mean(axis=1)

        candidate_mean = float(cand_util_per_seed.mean())
        sham_mean = float(sham_util_per_seed.mean())

        # Per-task paired deltas averaged across seeds.
        cand_minus_base = float((cand_util_matrix.mean(axis=0) - u_baseline).mean())
        cand_minus_sham = float((cand_util_matrix.mean(axis=0) - sham_util_matrix.mean(axis=0)).mean())
        oracle_minus_sham = float((u_oracle - sham_util_matrix.mean(axis=0)).mean())

        # Headroom recovered relative to oracle-sham.
        denom = oracle_minus_sham
        headroom_recovered = cand_minus_sham / denom if abs(denom) > 1e-10 else 0.0

        # Classification accuracy and action entropy averaged across seeds.
        cand_acc = float(np.mean([r["classification_accuracy"] for r in cand_results]))
        sham_acc = float(np.mean([r["classification_accuracy"] for r in sham_results]))
        cand_entropy = float(np.mean([r["action_entropy"] for r in cand_results]))
        sham_entropy = float(np.mean([r["action_entropy"] for r in sham_results]))
        train_acc = float(np.mean(cand_train_accs))
        sham_train_acc = float(np.mean(sham_train_accs))

        entry = {
            "size": size,
            "n_seeds": n_seeds,
            "candidate_utility": candidate_mean,
            "baseline_utility": baseline_mean,
            "sham_utility": sham_mean,
            "oracle_utility": oracle_mean,
            "candidate_minus_baseline": cand_minus_base,
            "candidate_minus_sham": cand_minus_sham,
            "oracle_minus_sham": oracle_minus_sham,
            "headroom_recovered": headroom_recovered,
            "candidate_classification_accuracy": cand_acc,
            "sham_classification_accuracy": sham_acc,
            "candidate_action_entropy": cand_entropy,
            "sham_action_entropy": sham_entropy,
            "candidate_train_accuracy": train_acc,
            "sham_train_accuracy": sham_train_acc,
            "candidate_utility_seeds": cand_util_per_seed.tolist(),
            "sham_utility_seeds": sham_util_per_seed.tolist(),
        }
        per_size.append(entry)
        candidate_sham_means.append(cand_minus_sham)
        oracle_sham_means.append(oracle_minus_sham)
        train_acc_means.append(train_acc)
        test_util_means.append(candidate_mean)

    # ------------------------------------------------------------------
    # Cautious trend model on candidate-minus-sham vs. size.
    # ------------------------------------------------------------------
    xs = np.array(sizes, dtype=float)
    ys_cs = np.array(candidate_sham_means, dtype=float)
    trend_cs = _fit_trend(xs, ys_cs)

    # Trend on oracle-minus-sham (for HEADROOM_LIMITED check).
    ys_os = np.array(oracle_sham_means, dtype=float)
    trend_os = _fit_trend(xs, ys_os)

    # Trend on train accuracy vs. test utility (for LEARNER_LIMITED check).
    trend_train = _fit_trend(xs, np.array(train_acc_means, dtype=float))
    trend_test = _fit_trend(xs, np.array(test_util_means, dtype=float))

    # ------------------------------------------------------------------
    # Decision category.
    # ------------------------------------------------------------------
    decision = _classify_regime(
        sizes=sizes,
        candidate_sham_means=candidate_sham_means,
        oracle_sham_means=oracle_sham_means,
        train_acc_means=train_acc_means,
        test_util_means=test_util_means,
        trend_cs=trend_cs,
        trend_os=trend_os,
        trend_train=trend_train,
        trend_test=trend_test,
    )

    return {
        "n_seeds": n_seeds,
        "sizes": sizes,
        "pool_size": pool_size,
        "n_test_tasks": len(test_task_ids),
        "oracle_utility": oracle_mean,
        "baseline_utility": baseline_mean,
        "per_size": per_size,
        "trend_candidate_minus_sham": trend_cs,
        "trend_oracle_minus_sham": trend_os,
        "trend_candidate_train_accuracy": trend_train,
        "trend_candidate_test_utility": trend_test,
        "decision": decision["category"],
        "decision_rationale": decision["rationale"],
    }


def _train_accuracy(
    run: LoadedRun,
    weights: np.ndarray,
    bias: np.ndarray,
    actions: list[str],
    subset: list[dict],
) -> float:
    """Fraction of training examples where the policy selects the labelled best action."""
    if not subset or weights.size == 0:
        return 0.0
    action_idx = {a: i for i, a in enumerate(actions)}
    correct = 0
    total = 0
    for ex in subset:
        features = np.array(ex.get("features", []))
        best = ex.get("best_action", "")
        task = run.tasks.get(ex.get("task_id", ""))
        allowed = task.allowed_actions if task else actions
        eligible = [a for a in allowed if a in action_idx]
        if not eligible or len(features) != weights.shape[1]:
            continue
        logits = features @ weights + bias
        eligible_logits = np.array([logits[action_idx[a]] for a in eligible])
        chosen = eligible[int(np.argmax(eligible_logits))]
        if chosen == best:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def _classify_regime(
    sizes: list[int],
    candidate_sham_means: list[float],
    oracle_sham_means: list[float],
    train_acc_means: list[float],
    test_util_means: list[float],
    trend_cs: dict[str, float],
    trend_os: dict[str, float],
    trend_train: dict[str, float],
    trend_test: dict[str, float],
) -> dict[str, str]:
    """Classify the learning regime into one of the decision categories."""
    if len(sizes) < 2:
        return {"category": "INCONCLUSIVE",
                "rationale": "Fewer than two subset sizes; cannot fit a trend."}

    cs = np.array(candidate_sham_means)
    os_ = np.array(oracle_sham_means)
    ta = np.array(train_acc_means)
    tu = np.array(test_util_means)

    # HEADROOM_LIMITED: oracle-sham is small at all sizes.
    max_oracle_sham = float(np.max(os_))
    if max_oracle_sham < 0.10:
        return {
            "category": "HEADROOM_LIMITED",
            "rationale": (
                f"Oracle-minus-sham headroom is small at every subset size "
                f"(max={max_oracle_sham:.4f} < 0.10). The benchmark does not "
                f"contain enough route-dependent headroom for data to help."
            ),
        }

    # DATA_LIMITED: candidate-sham improves consistently with n.
    # "Consistently" means a positive slope that is at least one standard
    # error above zero, and the largest value exceeds the smallest.
    slope_cs = trend_cs["slope"]
    se_cs = trend_cs["slope_se"]
    monotonic_up = bool(np.all(np.diff(cs) >= -1e-9)) and cs[-1] > cs[0]
    slope_significant = (
        se_cs != float("inf")
        and slope_cs > 0
        and (slope_cs - se_cs) > 0
    )
    if monotonic_up and slope_significant and cs[-1] - cs[0] > 0.01:
        return {
            "category": "DATA_LIMITED",
            "rationale": (
                f"Candidate-minus-sham improves consistently with training size "
                f"(slope={slope_cs:.5f} +/- {se_cs:.5f}, "
                f"delta={cs[-1] - cs[0]:.4f}). More experience data should "
                f"continue to widen the gap."
            ),
        }

    # LEARNER_LIMITED: train accuracy rises but test utility doesn't.
    train_rises = trend_train["slope"] > 0 and ta[-1] > ta[0]
    test_flat = abs(trend_test["slope"]) <= max(1e-4, trend_test.get("slope_se", 0.0))
    if train_rises and test_flat:
        return {
            "category": "LEARNER_LIMITED",
            "rationale": (
                f"Train accuracy rises with more data "
                f"(slope={trend_train['slope']:.5f}) but test utility is flat "
                f"(slope={trend_test['slope']:.5f}). The learner memorises the "
                f"training set without generalising to held-out tasks."
            ),
        }

    # FEATURE_LIMITED: candidate-sham flat near zero.
    max_cs = float(np.max(np.abs(cs)))
    flat = abs(trend_cs["slope"]) <= max(1e-4, se_cs)
    if flat and max_cs < 0.05:
        return {
            "category": "FEATURE_LIMITED",
            "rationale": (
                f"Candidate-minus-sham is flat near zero across all sizes "
                f"(max |gap|={max_cs:.4f}, slope={trend_cs['slope']:.5f}). "
                f"The features do not carry enough route-dependent signal for "
                f"additional data to help."
            ),
        }

    return {
        "category": "INCONCLUSIVE",
        "rationale": (
            f"No single regime pattern is clearly dominant "
            f"(cs_slope={trend_cs['slope']:.5f}, "
            f"max_oracle_sham={max_oracle_sham:.4f}, "
            f"train_slope={trend_train['slope']:.5f}, "
            f"test_slope={trend_test['slope']:.5f})."
        ),
    }
