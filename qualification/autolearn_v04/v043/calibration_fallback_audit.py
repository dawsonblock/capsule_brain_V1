"""Calibration and fallback audit for v0.4.3.

Analyzes the effect of the confidence threshold and temperature on
candidate routing, fallback rates, utility, and calibration quality.

All tuning is performed on VALIDATION ONLY.  The production threshold is
predeclared from validation.  Final-test results are labeled EXPLORATORY
and NON_QUALIFYING — they are retrospective diagnostics only and must not
be used for qualification decisions.

Produces risk-coverage, utility-coverage, accuracy-coverage, and
candidate-control-rate curves.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import LoadedRun, TaskInfo
from .canonical_evaluator import (
    evaluate_policy_from_counterfactuals,
    make_oracle_fn,
    make_baseline_fn,
    make_executed_sham_fn,
    make_raw_candidate_fn,
    _make_state_from_task,
)
from .config import MINIMUM_DENOMINATOR


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THRESHOLD_STEP = 0.05
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0
TEMPERATURE_GRID = [0.5, 1.0, 2.0]
CATASTROPHIC_UTILITY_THRESHOLD = 0.0  # utility below this is catastrophic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_production_confidences(
    run: LoadedRun,
    task_ids: list[str],
    temperature: float = 1.0,
) -> dict[str, float]:
    """Get the production LearnedPolicy confidence for each task.

    Reconstructs the LearnedPolicy with a given temperature and calls
    select_action() to get decision.confidence for each task.
    """
    from capsule_brain.autolearn.policy import LearnedPolicy
    from capsule_brain.autolearn.evaluator import Action

    policy_dict = dict(run.candidate_policy)
    policy_dict["temperature"] = temperature
    policy = LearnedPolicy.from_dict(policy_dict)

    confidences: dict[str, float] = {}
    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        state = _make_state_from_task(task)
        allowed = [Action(a) for a in task.allowed_actions]
        decision = policy.select_action(state, allowed_actions=allowed)
        confidences[tid] = float(decision.confidence)
    return confidences


def _get_production_actions(
    run: LoadedRun,
    task_ids: list[str],
    temperature: float = 1.0,
    confidence_threshold: float = 0.5,
) -> dict[str, str | None]:
    """Get the production action for each task with a given threshold.

    When confidence < threshold, the action falls back to baseline.
    """
    from capsule_brain.autolearn.policy import LearnedPolicy
    from capsule_brain.autolearn.evaluator import Action
    from capsule_brain.autolearn.baseline import BaselinePolicyV3

    policy_dict = dict(run.candidate_policy)
    policy_dict["temperature"] = temperature
    policy_dict["confidence_threshold"] = confidence_threshold
    policy = LearnedPolicy.from_dict(policy_dict)
    baseline = BaselinePolicyV3()

    actions: dict[str, str | None] = {}
    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue
        state = _make_state_from_task(task)
        allowed = [Action(a) for a in task.allowed_actions]
        decision = policy.select_action(state, allowed_actions=allowed)
        if decision.abstained or decision.confidence < confidence_threshold:
            # Fallback to baseline.
            base_dec = baseline.select_action(state, allowed_actions=allowed)
            actions[tid] = base_dec.action.value
        else:
            actions[tid] = decision.action.value
    return actions


def _brier_score(
    confidences: dict[str, float],
    oracle_actions: dict[str, str | None],
    selected_actions: dict[str, str | None],
) -> float:
    """Compute Brier score: mean squared error of confidence vs correctness.

    For each task, the "forecast" is the confidence that the selected
    action is correct, and the "outcome" is 1.0 if the selected action
    matches the oracle action, 0.0 otherwise.
    """
    sq_errors: list[float] = []
    for tid, conf in confidences.items():
        selected = selected_actions.get(tid)
        oracle = oracle_actions.get(tid)
        outcome = 1.0 if selected == oracle and selected is not None else 0.0
        sq_errors.append((conf - outcome) ** 2)
    return float(np.mean(sq_errors)) if sq_errors else 0.0


def _calibration_error(
    confidences: dict[str, float],
    oracle_actions: dict[str, str | None],
    selected_actions: dict[str, str | None],
    n_bins: int = 10,
) -> float:
    """Compute expected calibration error (ECE) with n_bins equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [
            tid for tid, c in confidences.items()
            if lo <= c < hi or (i == n_bins - 1 and c == hi)
        ]
        if not in_bin:
            continue
        bin_conf = np.mean([confidences[t] for t in in_bin])
        bin_acc = np.mean([
            1.0 if selected_actions.get(t) == oracle_actions.get(t)
            and selected_actions.get(t) is not None
            else 0.0
            for t in in_bin
        ])
        ece += (len(in_bin) / total) * abs(bin_conf - bin_acc)
    return float(ece)


def _action_distribution(actions: dict[str, str | None]) -> dict[str, int]:
    """Count actions in a dict of task_id -> action."""
    from collections import Counter
    c: Counter[str] = Counter()
    for a in actions.values():
        if a is not None:
            c[a] += 1
    return dict(c)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def _sweep_thresholds(
    run: LoadedRun,
    task_ids: list[str],
    confidences: dict[str, float],
    oracle_actions: dict[str, str | None],
    oracle_utilities: dict[str, float],
    sham_utility: float,
    baseline_utility: float,
    temperature: float,
) -> list[dict[str, Any]]:
    """Sweep confidence thresholds and compute metrics for each.

    For each threshold, determines which tasks the candidate controls
    (confidence > threshold) vs falls back.  Computes utility, candidate-
    minus-sham, candidate-minus-baseline, catastrophic failures, action
    distribution, calibration error, and Brier score.
    """
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP)
    curves: list[dict[str, Any]] = []

    for thr in thresholds:
        thr_f = float(round(thr, 4))

        # Determine candidate-controlled vs fallback tasks.
        candidate_controlled = [
            tid for tid in task_ids
            if tid in confidences and confidences[tid] > thr_f
        ]
        fallback_tasks = [
            tid for tid in task_ids
            if tid in confidences and confidences[tid] <= thr_f
        ]

        # Get actions for this threshold.
        actions = _get_production_actions(
            run, task_ids, temperature=temperature, confidence_threshold=thr_f,
        )

        # Compute utility.
        utilities: list[float] = []
        catastrophic_count = 0
        for tid in task_ids:
            action = actions.get(tid)
            if action is None:
                continue
            u = run.get_utility_for_action(tid, action)
            if u is not None:
                utilities.append(u)
                if u < CATASTROPHIC_UTILITY_THRESHOLD:
                    catastrophic_count += 1

        mean_utility = float(np.mean(utilities)) if utilities else 0.0
        candidate_minus_sham = mean_utility - sham_utility
        candidate_minus_baseline = mean_utility - baseline_utility

        # Coverage = fraction of tasks the candidate controls.
        coverage = (
            len(candidate_controlled) / len(task_ids)
            if task_ids else 0.0
        )

        # Accuracy.
        correct = sum(
            1 for tid in task_ids
            if actions.get(tid) is not None
            and actions.get(tid) == oracle_actions.get(tid)
        )
        total_with_action = sum(1 for tid in task_ids if actions.get(tid) is not None)
        accuracy = correct / total_with_action if total_with_action > 0 else 0.0

        # Calibration error and Brier score.
        cal_error = _calibration_error(confidences, oracle_actions, actions)
        brier = _brier_score(confidences, oracle_actions, actions)

        # Action distribution.
        action_dist = _action_distribution(actions)

        curves.append({
            "threshold": thr_f,
            "temperature": temperature,
            "candidate_controlled_count": len(candidate_controlled),
            "fallback_count": len(fallback_tasks),
            "coverage": coverage,
            "mean_utility": mean_utility,
            "candidate_minus_sham": candidate_minus_sham,
            "candidate_minus_baseline": candidate_minus_baseline,
            "catastrophic_failure_count": catastrophic_count,
            "accuracy": accuracy,
            "calibration_error": cal_error,
            "brier_score": brier,
            "action_distribution": action_dist,
        })

    return curves


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def audit_calibration_fallback(
    run: LoadedRun,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze calibration and fallback effects on VALIDATION ONLY.

    Sweeps confidence thresholds (0.0 to 1.0 in 0.05 increments) and
    temperatures (0.5, 1.0, 2.0) on validation tasks.  For each
    configuration, reports candidate-controlled count, fallback count,
    mean utility, candidate-minus-sham, candidate-minus-baseline,
    catastrophic failures, action distribution, calibration error, and
    Brier score.

    Produces risk-coverage, utility-coverage, accuracy-coverage, and
    candidate-control-rate curves.

    The production threshold is predeclared using validation only.  Final-
    test results are labeled EXPLORATORY and NON_QUALIFYING.

    Args:
        run: The loaded run with counterfactual outcomes and policies.
        seed: Random seed (for sham evaluation consistency).

    Returns:
        dict with validation_curves, production_threshold,
        final_test_exploratory, and interpretation.  All values are
        JSON-serializable.
    """
    # ------------------------------------------------------------------
    # VALIDATION ONLY — all tuning happens here.
    # ------------------------------------------------------------------
    val_task_ids = run.validation_task_ids

    # Evaluate sham and baseline on validation for reference utilities.
    val_oracle_result = evaluate_policy_from_counterfactuals(
        run, val_task_ids, make_oracle_fn(run), "oracle_val",
    )
    val_baseline_result = evaluate_policy_from_counterfactuals(
        run, val_task_ids, make_baseline_fn(run), "baseline_val",
        oracle_result=val_oracle_result,
    )
    val_sham_result = evaluate_policy_from_counterfactuals(
        run, val_task_ids, make_executed_sham_fn(run), "sham_val",
        reference_result=val_baseline_result, oracle_result=val_oracle_result,
    )

    val_sham_utility = val_sham_result.mean_utility
    val_baseline_utility = val_baseline_result.mean_utility

    # Oracle actions and utilities for validation.
    val_oracle_actions: dict[str, str | None] = {}
    val_oracle_utilities: dict[str, float] = {}
    for tid in val_task_ids:
        val_oracle_actions[tid] = run.get_oracle_action(tid)
        val_oracle_utilities[tid] = run.get_oracle_utility(tid)

    # Sweep temperatures and thresholds.
    all_val_curves: dict[str, list[dict[str, Any]]] = {}
    best_threshold: float = 0.5
    best_score: float = -1e18

    for temp in TEMPERATURE_GRID:
        temp_key = f"temp_{temp}"
        confidences = _get_production_confidences(
            run, val_task_ids, temperature=temp,
        )
        curves = _sweep_thresholds(
            run, val_task_ids, confidences,
            val_oracle_actions, val_oracle_utilities,
            val_sham_utility, val_baseline_utility,
            temperature=temp,
        )
        all_val_curves[temp_key] = curves

        # Select best threshold on validation: maximize candidate-minus-sham
        # subject to catastrophic_failure_count == 0 (if possible).
        for entry in curves:
            score = entry["candidate_minus_sham"]
            # Penalize catastrophic failures.
            if entry["catastrophic_failure_count"] > 0:
                score -= entry["catastrophic_failure_count"] * 0.5
            if score > best_score:
                best_score = score
                best_threshold = entry["threshold"]

    # Build named curves from the temperature=1.0 sweep (production default).
    default_curves = all_val_curves.get("temp_1.0", [])
    risk_coverage = [
        {"coverage": c["coverage"], "risk": 1.0 - c["mean_utility"],
         "threshold": c["threshold"]}
        for c in default_curves
    ]
    utility_coverage = [
        {"coverage": c["coverage"], "utility": c["mean_utility"],
         "threshold": c["threshold"]}
        for c in default_curves
    ]
    accuracy_coverage = [
        {"coverage": c["coverage"], "accuracy": c["accuracy"],
         "threshold": c["threshold"]}
        for c in default_curves
    ]
    candidate_control_rate = [
        {"threshold": c["threshold"],
         "control_rate": c["candidate_controlled_count"] / max(len(val_task_ids), 1),
         "fallback_rate": c["fallback_count"] / max(len(val_task_ids), 1)}
        for c in default_curves
    ]

    validation_curves: dict[str, Any] = {
        "threshold_sweeps": all_val_curves,
        "risk_coverage": risk_coverage,
        "utility_coverage": utility_coverage,
        "accuracy_coverage": accuracy_coverage,
        "candidate_control_rate": candidate_control_rate,
        "validation_sham_utility": val_sham_utility,
        "validation_baseline_utility": val_baseline_utility,
        "validation_oracle_utility": val_oracle_result.mean_utility,
        "n_validation_tasks": len(val_task_ids),
    }

    # ------------------------------------------------------------------
    # FINAL TEST — EXPLORATORY ONLY, NON_QUALIFYING.
    # ------------------------------------------------------------------
    test_task_ids = run.test_task_ids

    test_oracle_result = evaluate_policy_from_counterfactuals(
        run, test_task_ids, make_oracle_fn(run), "oracle_test",
    )
    test_baseline_result = evaluate_policy_from_counterfactuals(
        run, test_task_ids, make_baseline_fn(run), "baseline_test",
        oracle_result=test_oracle_result,
    )
    test_sham_result = evaluate_policy_from_counterfactuals(
        run, test_task_ids, make_executed_sham_fn(run), "sham_test",
        reference_result=test_baseline_result, oracle_result=test_oracle_result,
    )

    test_sham_utility = test_sham_result.mean_utility
    test_baseline_utility = test_baseline_result.mean_utility

    test_oracle_actions: dict[str, str | None] = {}
    test_oracle_utilities: dict[str, float] = {}
    for tid in test_task_ids:
        test_oracle_actions[tid] = run.get_oracle_action(tid)
        test_oracle_utilities[tid] = run.get_oracle_utility(tid)

    # Evaluate final test at the predeclared production threshold only.
    test_confidences = _get_production_confidences(
        run, test_task_ids, temperature=1.0,
    )
    test_curves = _sweep_thresholds(
        run, test_task_ids, test_confidences,
        test_oracle_actions, test_oracle_utilities,
        test_sham_utility, test_baseline_utility,
        temperature=1.0,
    )

    # Find the entry at the production threshold.
    production_entry: dict[str, Any] | None = None
    for entry in test_curves:
        if abs(entry["threshold"] - best_threshold) < 1e-6:
            production_entry = entry
            break
    if production_entry is None and test_curves:
        production_entry = test_curves[0]

    final_test_exploratory: dict[str, Any] = {
        "label": "EXPLORATORY",
        "qualifying": False,
        "warning": (
            "Final-test results are EXPLORATORY and NON_QUALIFYING. "
            "They are retrospective diagnostics only and must NOT be "
            "used for qualification decisions. The production threshold "
            "was predeclared on validation only."
        ),
        "production_threshold": best_threshold,
        "threshold_sweep": test_curves,
        "production_threshold_result": production_entry,
        "test_sham_utility": test_sham_utility,
        "test_baseline_utility": test_baseline_utility,
        "test_oracle_utility": test_oracle_result.mean_utility,
        "n_test_tasks": len(test_task_ids),
    }

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    interpretation = _build_interpretation(
        validation_curves=validation_curves,
        production_threshold=best_threshold,
        final_test_exploratory=final_test_exploratory,
    )

    return {
        "validation_curves": validation_curves,
        "production_threshold": best_threshold,
        "final_test_exploratory": final_test_exploratory,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def _build_interpretation(
    validation_curves: dict[str, Any],
    production_threshold: float,
    final_test_exploratory: dict[str, Any],
) -> str:
    """Build a human-readable interpretation of the calibration audit."""
    parts: list[str] = []

    n_val = validation_curves.get("n_validation_tasks", 0)
    parts.append(
        f"Calibration audit performed on {n_val} validation tasks. "
        f"Production threshold predeclared at {production_threshold:.2f} "
        f"(selected on validation only)."
    )

    # Analyze the default-temperature sweep.
    default_sweep = validation_curves.get("threshold_sweeps", {}).get("temp_1.0", [])
    if default_sweep:
        # Find the production threshold entry.
        prod_entry = None
        for entry in default_sweep:
            if abs(entry["threshold"] - production_threshold) < 1e-6:
                prod_entry = entry
                break
        if prod_entry:
            control = prod_entry["candidate_controlled_count"]
            fallback = prod_entry["fallback_count"]
            util = prod_entry["mean_utility"]
            cms = prod_entry["candidate_minus_sham"]
            cal_err = prod_entry["calibration_error"]
            brier = prod_entry["brier_score"]
            parts.append(
                f"At threshold={production_threshold:.2f}: candidate "
                f"controls {control}/{n_val} tasks, falls back on "
                f"{fallback}/{n_val}. Validation utility = {util:.4f}, "
                f"candidate-minus-sham = {cms:+.4f}, calibration error "
                f"= {cal_err:.4f}, Brier score = {brier:.4f}."
            )

            if fallback > control:
                parts.append(
                    f"The candidate falls back on the majority of "
                    f"validation tasks ({fallback}/{n_val}), indicating "
                    f"the confidence threshold is aggressive and the "
                    f"baseline handles most routing."
                )
            elif control > 0 and cms > 0:
                parts.append(
                    f"The candidate controls a majority of tasks and "
                    f"beats sham on validation, indicating the learned "
                    f"router adds value when it is confident."
                )

    # Final test exploratory warning.
    ft = final_test_exploratory.get("production_threshold_result")
    if ft:
        ft_util = ft.get("mean_utility", 0.0)
        ft_cms = ft.get("candidate_minus_sham", 0.0)
        parts.append(
            f"Final-test EXPLORATORY (NON_QUALIFYING): at the predeclared "
            f"threshold, test utility = {ft_util:.4f}, candidate-minus-"
            f"sham = {ft_cms:+.4f}."
        )

    return " ".join(parts)
