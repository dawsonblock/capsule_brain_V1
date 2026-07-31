"""Raw-router versus executed-router comparison for v0.4.3.

Compares the raw candidate (argmax of stored weights, subject only to
action eligibility) against the executed candidate (full production stack
including confidence, abstention, and fallback).

This isolates whether the learner itself produces useful routing, or
whether the production calibration/fallback stack is suppressing useful
routing decisions.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .data import LoadedRun
from .canonical_evaluator import (
    PolicyEvalResult,
    evaluate_policy_from_counterfactuals,
    make_raw_candidate_fn,
    make_executed_candidate_fn,
    make_executed_sham_fn,
    make_oracle_fn,
    make_baseline_fn,
    compute_headroom_metrics,
    result_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _action_accuracy(result: PolicyEvalResult, oracle_result: PolicyEvalResult) -> float:
    """Fraction of tasks where selected action == oracle action."""
    oracle_actions = {r.task_id: r.selected_action for r in oracle_result.task_results}
    matches = 0
    total = 0
    for r in result.task_results:
        if r.selected_action is not None and r.task_id in oracle_actions:
            total += 1
            if r.selected_action == oracle_actions[r.task_id]:
                matches += 1
    return matches / total if total > 0 else 0.0


def _mean_regret(result: PolicyEvalResult) -> float | None:
    """Mean regret (oracle_utility - selected_utility) over complete tasks."""
    regrets = [
        r.regret for r in result.task_results
        if r.regret is not None
    ]
    return float(np.mean(regrets)) if regrets else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compare_raw_vs_executed(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare raw-router versus executed-router performance.

    Evaluates four policy variants through the canonical evaluator:
    1. P_candidate_raw: argmax of weights, subject only to action eligibility.
    2. P_candidate_executed: full production stack (LearnedPolicy.select_action).
    3. Sham (executed): production stack on sham policy.
    4. Oracle: best measured action per task.

    Compares raw vs executed on: utility, candidate-vs-sham, action accuracy,
    regret, and headroom recovery.

    Args:
        run: The loaded run with counterfactual outcomes and policies.
        task_ids: Optional list of task IDs.  Defaults to all test tasks.
        seed: Random seed (unused currently but kept for API consistency).

    Returns:
        dict with raw_result, executed_result, comparison, interpretation,
        and bottleneck_classification.  All values are JSON-serializable.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # --- Evaluate all four variants ---

    # Oracle (needed for headroom and accuracy).
    oracle_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_oracle_fn(run), "oracle",
    )

    # Baseline (needed for headroom vs baseline).
    baseline_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_baseline_fn(run), "baseline",
        oracle_result=oracle_result,
    )

    # Sham (executed production stack on sham policy).
    sham_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_executed_sham_fn(run), "sham",
        reference_result=baseline_result, oracle_result=oracle_result,
    )

    # Raw candidate (argmax of weights, action-eligibility only).
    raw_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_raw_candidate_fn(run), "candidate_raw",
        reference_result=baseline_result, oracle_result=oracle_result,
    )

    # Executed candidate (full production stack).
    executed_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_executed_candidate_fn(run), "candidate_executed",
        reference_result=baseline_result, oracle_result=oracle_result,
    )

    # --- Headroom metrics (candidate vs sham) ---
    raw_headroom_vs_sham = compute_headroom_metrics(
        candidate_mean=raw_result.mean_utility,
        reference_mean=sham_result.mean_utility,
        oracle_mean=oracle_result.mean_utility,
        n_tasks=raw_result.n_complete,
        task_ids_digest=raw_result.task_ids_digest,
        reference_policy_id="sham",
        candidate_policy_id="candidate_raw",
    )
    raw_result.headroom_vs_reference = raw_headroom_vs_sham

    executed_headroom_vs_sham = compute_headroom_metrics(
        candidate_mean=executed_result.mean_utility,
        reference_mean=sham_result.mean_utility,
        oracle_mean=oracle_result.mean_utility,
        n_tasks=executed_result.n_complete,
        task_ids_digest=executed_result.task_ids_digest,
        reference_policy_id="sham",
        candidate_policy_id="candidate_executed",
    )
    executed_result.headroom_vs_reference = executed_headroom_vs_sham

    # --- Compute comparison metrics ---
    raw_utility = raw_result.mean_utility
    executed_utility = executed_result.mean_utility
    sham_utility = sham_result.mean_utility

    raw_vs_sham = raw_utility - sham_utility
    executed_vs_sham = executed_utility - sham_utility

    raw_accuracy = _action_accuracy(raw_result, oracle_result)
    executed_accuracy = _action_accuracy(executed_result, oracle_result)

    raw_regret = _mean_regret(raw_result)
    executed_regret = _mean_regret(executed_result)

    raw_headroom_frac = raw_headroom_vs_sham.get("recovered_headroom_fraction")
    executed_headroom_frac = executed_headroom_vs_sham.get("recovered_headroom_fraction")

    comparison: dict[str, dict[str, Any]] = {
        "mean_utility": {
            "raw": raw_utility,
            "executed": executed_utility,
            "delta": executed_utility - raw_utility,
        },
        "candidate_vs_sham": {
            "raw": raw_vs_sham,
            "executed": executed_vs_sham,
            "delta": executed_vs_sham - raw_vs_sham,
        },
        "action_accuracy": {
            "raw": raw_accuracy,
            "executed": executed_accuracy,
            "delta": executed_accuracy - raw_accuracy,
        },
        "mean_regret": {
            "raw": raw_regret,
            "executed": executed_regret,
            "delta": (
                executed_regret - raw_regret
                if raw_regret is not None and executed_regret is not None
                else None
            ),
        },
        "headroom_recovery_fraction": {
            "raw": raw_headroom_frac,
            "executed": executed_headroom_frac,
            "delta": (
                executed_headroom_frac - raw_headroom_frac
                if raw_headroom_frac is not None and executed_headroom_frac is not None
                else None
            ),
        },
    }

    # --- Bottleneck classification ---
    bottleneck_classification = _classify_bottleneck(
        raw_vs_sham=raw_vs_sham,
        executed_vs_sham=executed_vs_sham,
        raw_result=raw_result,
        executed_result=executed_result,
    )

    # --- Interpretation ---
    interpretation = _build_interpretation(
        raw_utility=raw_utility,
        executed_utility=executed_utility,
        sham_utility=sham_utility,
        raw_vs_sham=raw_vs_sham,
        executed_vs_sham=executed_vs_sham,
        raw_accuracy=raw_accuracy,
        executed_accuracy=executed_accuracy,
        raw_regret=raw_regret,
        executed_regret=executed_regret,
        raw_headroom_frac=raw_headroom_frac,
        executed_headroom_frac=executed_headroom_frac,
        bottleneck=bottleneck_classification,
    )

    return {
        "raw_result": result_to_dict(raw_result),
        "executed_result": result_to_dict(executed_result),
        "sham_result": result_to_dict(sham_result),
        "oracle_result": result_to_dict(oracle_result),
        "comparison": comparison,
        "interpretation": interpretation,
        "bottleneck_classification": bottleneck_classification,
    }


# ---------------------------------------------------------------------------
# Bottleneck classification
# ---------------------------------------------------------------------------


def _classify_bottleneck(
    raw_vs_sham: float,
    executed_vs_sham: float,
    raw_result: PolicyEvalResult,
    executed_result: PolicyEvalResult,
) -> str:
    """Classify the bottleneck limiting candidate performance.

    Classification logic:
    - RAW_LEARNER_GOOD: raw candidate beats sham and executed candidate
      also beats sham (the learner and calibration are both adequate).
    - CALIBRATION_SUPPRESSES_ROUTING: raw candidate beats sham but executed
      candidate does not (calibration/fallback is suppressing useful routing).
    - LEARNER_INADEQUATE: raw candidate does not beat sham (features,
      objective, or learner are inadequate).
    - BOTH_INADEQUATE: neither raw nor executed beats sham, and the raw
      candidate is highly unstable.
    """
    raw_beats_sham = raw_vs_sham > 0
    executed_beats_sham = executed_vs_sham > 0

    if raw_beats_sham and executed_beats_sham:
        return "RAW_LEARNER_GOOD"
    elif raw_beats_sham and not executed_beats_sham:
        return "CALIBRATION_SUPPRESSES_ROUTING"
    elif not raw_beats_sham and not executed_beats_sham:
        # Check instability: if raw candidate has high unavailable rate,
        # both learner and data may be inadequate.
        raw_unavailable_rate = raw_result.n_unavailable / max(raw_result.n_tasks, 1)
        if raw_unavailable_rate > 0.3:
            return "BOTH_INADEQUATE"
        return "LEARNER_INADEQUATE"
    else:
        # executed beats sham but raw does not — unusual, treat as learner
        # inadequate with calibration compensating.
        return "LEARNER_INADEQUATE"


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def _build_interpretation(
    raw_utility: float,
    executed_utility: float,
    sham_utility: float,
    raw_vs_sham: float,
    executed_vs_sham: float,
    raw_accuracy: float,
    executed_accuracy: float,
    raw_regret: float | None,
    executed_regret: float | None,
    raw_headroom_frac: float | None,
    executed_headroom_frac: float | None,
    bottleneck: str,
) -> str:
    """Build a human-readable interpretation of the raw vs executed comparison."""
    parts: list[str] = []

    parts.append(
        f"Raw candidate utility = {raw_utility:.4f}, executed candidate "
        f"utility = {executed_utility:.4f}, sham utility = {sham_utility:.4f}."
    )

    if raw_vs_sham > 0 and executed_vs_sham <= 0:
        parts.append(
            f"The raw candidate beats sham (+{raw_vs_sham:.4f}) but the "
            f"executed candidate does not ({executed_vs_sham:+.4f}), "
            f"meaning calibration/fallback is suppressing useful routing "
            f"decisions that the raw learner correctly identifies."
        )
    elif raw_vs_sham <= 0 and executed_vs_sham <= 0:
        parts.append(
            f"Neither the raw candidate ({raw_vs_sham:+.4f}) nor the "
            f"executed candidate ({executed_vs_sham:+.4f}) beats sham, "
            f"indicating that the features, objective, or learner are "
            f"inadequate — the routing signal is not learnable from the "
            f"current feature set."
        )
    elif raw_vs_sham > 0 and executed_vs_sham > 0:
        parts.append(
            f"Both the raw candidate (+{raw_vs_sham:.4f}) and the executed "
            f"candidate (+{executed_vs_sham:.4f}) beat sham, meaning the "
            f"learner produces useful routing and the production stack "
            f"preserves it."
        )
    else:
        parts.append(
            f"The executed candidate beats sham (+{executed_vs_sham:.4f}) "
            f"but the raw candidate does not ({raw_vs_sham:+.4f}), "
            f"indicating calibration compensates for a weak raw learner."
        )

    # Accuracy.
    parts.append(
        f"Action accuracy: raw = {raw_accuracy:.1%}, executed = "
        f"{executed_accuracy:.1%}."
    )

    # Regret.
    if raw_regret is not None and executed_regret is not None:
        parts.append(
            f"Mean regret: raw = {raw_regret:.4f}, executed = "
            f"{executed_regret:.4f}."
        )

    # Headroom.
    if raw_headroom_frac is not None and executed_headroom_frac is not None:
        parts.append(
            f"Headroom recovery: raw = {raw_headroom_frac:.1%}, executed "
            f"= {executed_headroom_frac:.1%}."
        )

    # Bottleneck.
    bottleneck_labels = {
        "RAW_LEARNER_GOOD": "the raw learner is adequate and calibration "
            "preserves useful routing",
        "CALIBRATION_SUPPRESSES_ROUTING": "calibration/fallback is "
            "suppressing useful routing that the raw learner correctly "
            "identifies",
        "LEARNER_INADEQUATE": "the features, objective, or learner are "
            "inadequate — the routing signal is not learnable",
        "BOTH_INADEQUATE": "both the learner and the data are inadequate "
            "— the raw candidate is unstable and does not beat sham",
    }
    parts.append(
        f"Bottleneck classification: {bottleneck} — "
        f"{bottleneck_labels.get(bottleneck, 'unknown')}."
    )

    return " ".join(parts)
