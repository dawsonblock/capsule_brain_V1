"""Calibration and threshold audit for v0.4.2 diagnostics.

Inspects the candidate policy's confidence / abstention / shift
thresholds and temperature calibration, then traces how the action
distribution and utility change as each threshold gate is applied.

All threshold sweeps on the test set are **exploratory** and
**non-qualifying** — they are reported for diagnostic transparency only
and must not be used to select a post-hoc operating point for
qualification.

The output of :func:`audit_calibration` is the content for
``calibration_audit.json``.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .data import LoadedRun
from .policy_controls import evaluate_policy_on_tasks, make_candidate_selector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _action_distribution(
    selected_actions: list[str],
    all_actions: list[str],
) -> dict[str, float]:
    """Normalised action distribution over *all_actions*."""
    n = len(selected_actions)
    counts = Counter(selected_actions)
    if n == 0:
        return {a: 0.0 for a in all_actions}
    return {a: counts.get(a, 0) / n for a in all_actions}


def _mean_utility(
    run: LoadedRun,
    task_action_pairs: list[tuple[str, str]],
) -> float:
    """Mean measured utility over a list of (task_id, action_id) pairs."""
    if not task_action_pairs:
        return 0.0
    total = 0.0
    for tid, action in task_action_pairs:
        u = run.get_utility_for_action(tid, action)
        if u is None:
            u = 0.0
        total += u
    return total / len(task_action_pairs)


def _build_test_logits(
    run: LoadedRun,
    policy: dict[str, Any],
    temperature: float = 1.0,
) -> tuple[list[str], np.ndarray, np.ndarray, list[list[str]], dict[str, int]]:
    """Compute logits and softmax probabilities for every test task.

    Returns:
        task_ids:          ordered list of test task IDs with valid features.
        logits:            [n_tasks, n_actions] raw logits.
        probs:             [n_tasks, n_actions] softmax probabilities.
        eligible_lists:    per-task list of eligible action names.
        action_idx_map:    mapping from action name to column index.
    """
    weights = np.array(policy.get("weights", []), dtype=float)
    bias = np.array(policy.get("bias", []), dtype=float)
    actions = policy.get("actions", run.all_actions)
    action_idx = {a: i for i, a in enumerate(actions)}

    tid_to_ex: dict[str, dict] = {}
    for ex in run.test_examples:
        tid = ex.get("task_id")
        if tid is not None:
            tid_to_ex[tid] = ex

    task_ids: list[str] = []
    logits_rows: list[np.ndarray] = []
    eligible_lists: list[list[str]] = []

    for tid in run.test_task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        ex = tid_to_ex.get(tid)
        if ex is None or not weights.size:
            continue
        features = np.asarray(ex.get("features", []), dtype=float)
        if features.shape[0] != weights.shape[1]:
            continue

        logits = features @ weights.T + bias
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            eligible = [a for a in task.allowed_actions if a in run.all_actions]
        if not eligible:
            continue

        task_ids.append(tid)
        logits_rows.append(logits)
        eligible_lists.append(eligible)

    if not task_ids:
        return [], np.empty((0, 0)), np.empty((0, 0)), [], action_idx

    logits_mat = np.stack(logits_rows, axis=0)  # [n_tasks, n_actions]

    # Compute softmax over eligible actions only (mask ineligible to -inf).
    masked = np.full_like(logits_mat, -np.inf)
    for i, eligible in enumerate(eligible_lists):
        for a in eligible:
            j = action_idx.get(a)
            if j is not None:
                masked[i, j] = logits_mat[i, j]

    scaled = masked / max(temperature, 1e-12)
    probs = _softmax(scaled)
    # Replace any NaN (all-masked rows — shouldn't happen) with uniform.
    nan_rows = np.isnan(probs).any(axis=1)
    for i in np.where(nan_rows)[0]:
        n_elig = len(eligible_lists[i])
        probs[i, :] = 0.0
        for a in eligible_lists[i]:
            j = action_idx.get(a)
            if j is not None:
                probs[i, j] = 1.0 / n_elig

    return task_ids, logits_mat, probs, eligible_lists, action_idx


def _select_action(
    probs: np.ndarray,
    eligible: list[str],
    action_idx: dict[str, int],
    confidence_threshold: float = 0.0,
    abstention_threshold: float = 0.0,
) -> str | None:
    """Select an action given probabilities and thresholds.

    * If the max eligible probability is below ``confidence_threshold``
      the policy abstains (returns ``None``).
    * ``abstention_threshold`` is an absolute floor — if the max
      probability is below this the policy also abstains.

    Returns the selected action name or ``None`` for abstention.
    """
    if not eligible:
        return None
    eligible_probs = np.array([probs[action_idx[a]] for a in eligible])
    best_local = int(np.argmax(eligible_probs))
    best_prob = float(eligible_probs[best_local])

    if best_prob < confidence_threshold:
        return None
    if best_prob < abstention_threshold:
        return None
    return eligible[best_local]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def audit_calibration(
    run: LoadedRun,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Audit calibration and threshold behaviour of the candidate policy.

    Inspects the thresholds stored in ``run.candidate_policy`` and
    traces the action distribution and utility through four stages:

      1. **Raw** — candidate action distribution before any thresholding.
      2. **Post-abstention** — after the abstention threshold gate.
      3. **Post-shift-fallback** — after shift-threshold fallback to
         the deterministic baseline.
      4. **Final executed** — the distribution actually executed
         (equivalent to stage 3 in this offline analysis).

    Utility is calculated at each stage using measured counterfactual
    outcomes — no new model executions.

    A **risk-coverage curve** is produced by sweeping the confidence
    threshold from 0 to 1.  All such sweeps on the test set are labelled
    **exploratory and non-qualifying**.

    Args:
        run:       loaded run data.
        task_ids:  optional subset of test task IDs (default: all test).

    Returns:
        Content for ``calibration_audit.json``.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    all_actions = run.all_actions
    policy = run.candidate_policy

    # ------------------------------------------------------------------
    # Thresholds from candidate_policy (with sensible defaults)
    # ------------------------------------------------------------------
    confidence_threshold = float(policy.get("confidence_threshold", 0.0))
    abstention_threshold = float(policy.get("abstention_threshold", 0.0))
    shift_threshold = float(policy.get("shift_threshold", 0.0))
    temperature = float(policy.get("temperature", 1.0))
    effective_coverage = float(policy.get("effective_coverage", 0.0))

    thresholds = {
        "confidence_threshold": confidence_threshold,
        "abstention_threshold": abstention_threshold,
        "shift_threshold": shift_threshold,
        "temperature": temperature,
        "effective_coverage": effective_coverage,
        "source": "candidate_policy.json",
        "defaults_used": {
            "confidence_threshold": 0.0,
            "abstention_threshold": 0.0,
            "shift_threshold": 0.0,
            "temperature": 1.0,
            "effective_coverage": 0.0,
        },
    }

    # ------------------------------------------------------------------
    # Compute logits / probabilities for the requested tasks
    # ------------------------------------------------------------------
    eval_task_ids, logits_mat, probs_mat, eligible_lists, action_idx = \
        _build_test_logits(run, policy, temperature=temperature)

    # If task_ids was specified, filter to the intersection.
    if task_ids is not None:
        keep_set = set(task_ids)
        keep_indices = [i for i, tid in enumerate(eval_task_ids) if tid in keep_set]
        eval_task_ids = [eval_task_ids[i] for i in keep_indices]
        logits_mat = logits_mat[keep_indices] if logits_mat.size else logits_mat
        probs_mat = probs_mat[keep_indices] if probs_mat.size else probs_mat
        eligible_lists = [eligible_lists[i] for i in keep_indices]

    n_tasks = len(eval_task_ids)

    # ------------------------------------------------------------------
    # Stage 1: Raw candidate action distribution (no thresholding)
    # ------------------------------------------------------------------
    raw_actions: list[str] = []
    raw_pairs: list[tuple[str, str]] = []
    raw_max_probs: list[float] = []

    for i, tid in enumerate(eval_task_ids):
        eligible = eligible_lists[i]
        if not eligible:
            continue
        eligible_probs = np.array([probs_mat[i, action_idx[a]] for a in eligible])
        best_local = int(np.argmax(eligible_probs))
        action = eligible[best_local]
        raw_actions.append(action)
        raw_pairs.append((tid, action))
        raw_max_probs.append(float(eligible_probs[best_local]))

    raw_utility = _mean_utility(run, raw_pairs)
    raw_dist = _action_distribution(raw_actions, all_actions)

    # ------------------------------------------------------------------
    # Stage 2: Post-abstention distribution
    # ------------------------------------------------------------------
    # Tasks where max prob < abstention_threshold are abstained
    # (dropped from the distribution).
    post_abstain_actions: list[str] = []
    post_abstain_pairs: list[tuple[str, str]] = []
    abstained_count = 0

    for i, tid in enumerate(eval_task_ids):
        eligible = eligible_lists[i]
        if not eligible:
            continue
        eligible_probs = np.array([probs_mat[i, action_idx[a]] for a in eligible])
        best_local = int(np.argmax(eligible_probs))
        best_prob = float(eligible_probs[best_local])
        if best_prob < abstention_threshold:
            abstained_count += 1
            continue
        action = eligible[best_local]
        post_abstain_actions.append(action)
        post_abstain_pairs.append((tid, action))

    post_abstain_utility = _mean_utility(run, post_abstain_pairs)
    post_abstain_dist = _action_distribution(post_abstain_actions, all_actions)

    # ------------------------------------------------------------------
    # Stage 3: Post-shift-fallback distribution
    # ------------------------------------------------------------------
    # If the shift threshold is exceeded (measured by distribution shift
    # between raw and a reference — here approximated by the confidence
    # gap), the policy falls back to the deterministic baseline action
    # (ANSWER_DIRECT or first eligible).  In this offline analysis we
    # apply the confidence_threshold gate: tasks below the confidence
    # threshold fall back to the baseline selector.
    from .policy_controls import make_baseline_selector
    baseline_selector = make_baseline_selector(run)

    post_shift_actions: list[str] = []
    post_shift_pairs: list[tuple[str, str]] = []
    fallback_count = 0

    for i, tid in enumerate(eval_task_ids):
        task = run.tasks.get(tid)
        if task is None:
            continue
        eligible = eligible_lists[i]
        if not eligible:
            continue
        eligible_probs = np.array([probs_mat[i, action_idx[a]] for a in eligible])
        best_local = int(np.argmax(eligible_probs))
        best_prob = float(eligible_probs[best_local])

        if best_prob < confidence_threshold:
            # Fallback to baseline.
            fallback_action = baseline_selector(tid, task)
            if fallback_action is None:
                fallback_action = eligible[0]
            post_shift_actions.append(fallback_action)
            post_shift_pairs.append((tid, fallback_action))
            fallback_count += 1
        else:
            action = eligible[best_local]
            post_shift_actions.append(action)
            post_shift_pairs.append((tid, action))

    post_shift_utility = _mean_utility(run, post_shift_pairs)
    post_shift_dist = _action_distribution(post_shift_actions, all_actions)

    # ------------------------------------------------------------------
    # Stage 4: Final executed distribution
    # ------------------------------------------------------------------
    # In this offline analysis the final executed distribution is the
    # same as the post-shift-fallback distribution (no additional
    # runtime gates are applied).
    final_actions = post_shift_actions
    final_pairs = post_shift_pairs
    final_utility = post_shift_utility
    final_dist = post_shift_dist

    # ------------------------------------------------------------------
    # Also evaluate the standard candidate selector for reference
    # ------------------------------------------------------------------
    candidate_selector = make_candidate_selector(run)
    candidate_ref = evaluate_policy_on_tasks(
        run, eval_task_ids, candidate_selector, "P_candidate",
    )

    # ------------------------------------------------------------------
    # Risk-coverage curve: sweep confidence threshold from 0 to 1
    # ------------------------------------------------------------------
    sweep_thresholds = np.linspace(0.0, 1.0, 101)
    risk_coverage: list[dict[str, Any]] = []

    for thr in sweep_thresholds:
        selected: list[str] = []
        pairs: list[tuple[str, str]] = []
        abstained = 0

        for i, tid in enumerate(eval_task_ids):
            task = run.tasks.get(tid)
            if task is None:
                continue
            eligible = eligible_lists[i]
            if not eligible:
                continue
            action = _select_action(
                probs_mat[i], eligible, action_idx,
                confidence_threshold=thr,
                abstention_threshold=0.0,
            )
            if action is None:
                abstained += 1
                continue
            selected.append(action)
            pairs.append((tid, action))

        coverage = (n_tasks - abstained) / n_tasks if n_tasks > 0 else 0.0
        mean_u = _mean_utility(run, pairs)
        dist = _action_distribution(selected, all_actions)

        risk_coverage.append({
            "confidence_threshold": float(thr),
            "coverage": float(coverage),
            "n_selected": len(pairs),
            "n_abstained": abstained,
            "mean_utility": float(mean_u),
            "action_distribution": dist,
        })

    # Find the threshold that maximises utility on the sweep.
    best_sweep = max(risk_coverage, key=lambda r: r["mean_utility"]) if risk_coverage else None

    # ------------------------------------------------------------------
    # Calibration metrics
    # ------------------------------------------------------------------
    # Mean max probability (confidence).
    if raw_max_probs:
        max_probs_arr = np.array(raw_max_probs)
        calibration_metrics = {
            "mean_max_probability": float(max_probs_arr.mean()),
            "std_max_probability": float(max_probs_arr.std(ddof=1)) if max_probs_arr.size > 1 else 0.0,
            "median_max_probability": float(np.median(max_probs_arr)),
            "fraction_above_confidence_threshold": float(
                np.mean(max_probs_arr >= confidence_threshold)
            ),
            "fraction_above_abstention_threshold": float(
                np.mean(max_probs_arr >= abstention_threshold)
            ),
        }
    else:
        calibration_metrics = {
            "mean_max_probability": 0.0,
            "std_max_probability": 0.0,
            "median_max_probability": 0.0,
            "fraction_above_confidence_threshold": 0.0,
            "fraction_above_abstention_threshold": 0.0,
        }

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    return {
        "n_tasks": n_tasks,
        "task_ids": eval_task_ids,
        "thresholds": thresholds,
        "stages": {
            "raw_candidate": {
                "description": "Candidate action distribution before any thresholding.",
                "action_distribution": raw_dist,
                "mean_utility": float(raw_utility),
                "n_tasks": len(raw_pairs),
            },
            "post_abstention": {
                "description": (
                    "Distribution after abstention threshold gate. "
                    "Tasks with max probability below the abstention "
                    "threshold are dropped."
                ),
                "action_distribution": post_abstain_dist,
                "mean_utility": float(post_abstain_utility),
                "n_tasks": len(post_abstain_pairs),
                "n_abstained": abstained_count,
                "abstention_rate": float(abstained_count / n_tasks) if n_tasks > 0 else 0.0,
            },
            "post_shift_fallback": {
                "description": (
                    "Distribution after shift-threshold fallback. "
                    "Tasks below the confidence threshold fall back "
                    "to the deterministic baseline selector."
                ),
                "action_distribution": post_shift_dist,
                "mean_utility": float(post_shift_utility),
                "n_tasks": len(post_shift_pairs),
                "n_fallback": fallback_count,
                "fallback_rate": float(fallback_count / n_tasks) if n_tasks > 0 else 0.0,
            },
            "final_executed": {
                "description": (
                    "Final executed action distribution. In this "
                    "offline analysis this equals the post-shift-"
                    "fallback stage (no additional runtime gates)."
                ),
                "action_distribution": final_dist,
                "mean_utility": float(final_utility),
                "n_tasks": len(final_pairs),
            },
        },
        "calibration_metrics": calibration_metrics,
        "candidate_reference": {
            "mean_utility": candidate_ref["mean_utility"],
            "action_distribution": candidate_ref["action_distribution"],
            "action_distribution_pct": candidate_ref["action_distribution_pct"],
        },
        "risk_coverage_curve": {
            "description": (
                "Risk-coverage curve produced by sweeping the "
                "confidence threshold from 0 to 1. All test-set "
                "threshold sweeps are EXPLORATORY and NON-QUALIFYING — "
                "they are reported for diagnostic transparency only "
                "and must not be used to select a post-hoc operating "
                "point for qualification."
            ),
            "exploratory": True,
            "non_qualifying": True,
            "n_thresholds": len(sweep_thresholds),
            "points": risk_coverage,
            "best_utility_threshold": {
                "confidence_threshold": best_sweep["confidence_threshold"] if best_sweep else None,
                "mean_utility": best_sweep["mean_utility"] if best_sweep else None,
                "coverage": best_sweep["coverage"] if best_sweep else None,
                "warning": (
                    "This threshold was selected post-hoc on the test "
                    "set and is NON-QUALIFYING. It must not be used as "
                    "the operating point for qualification."
                ),
            } if best_sweep else None,
        },
        "labels": {
            "test_set_threshold_sweeps": "exploratory_and_non_qualifying",
            "note": (
                "All threshold sweeps on the test set are exploratory "
                "and non-qualifying. Operating-point selection must be "
                "performed on a separate validation split."
            ),
        },
    }
