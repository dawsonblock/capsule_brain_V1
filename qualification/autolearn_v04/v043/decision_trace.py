"""Decision-trace reconstruction for v0.4.3.

Reconstructs the full candidate decision path for every evaluation task:
raw logits, probabilities, argmax, confidence, threshold/abstention/shift/
safety results, action masks, fallback metadata, the final production
action, the executed action, and the verified utility from counterfactual
outcomes.

This module is diagnostic only — it does not recompute qualification
metrics.  All traces are JSON-serializable so they can be emitted as JSONL.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .data import LoadedRun, TaskInfo
from .canonical_evaluator import (
    make_raw_candidate_fn,
    make_executed_candidate_fn,
    _make_state_from_task,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    if temperature <= 0:
        temperature = 1e-6
    scaled = logits / temperature
    m = np.max(scaled)
    exps = np.exp(scaled - m)
    total = np.sum(exps)
    if total <= 0:
        return np.full_like(scaled, 1.0 / len(scaled))
    return exps / total


def _build_feature_lookup(run: LoadedRun) -> dict[str, np.ndarray]:
    """Build a task_id -> feature vector lookup from all dataset splits."""
    lookup: dict[str, np.ndarray] = {}
    for ex in (
        run.test_examples
        + run.train_examples
        + run.validation_examples
        + run.ood_examples
        + run.safety_examples
    ):
        tid = ex.get("task_id", "")
        if tid:
            lookup[tid] = np.array(ex.get("features", []), dtype=float)
    return lookup


def _get_raw_logits(
    run: LoadedRun,
    task: TaskInfo,
    feature_lookup: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Compute raw logits for all actions from candidate weights.

    Returns (logits, actions) or (None, None) if features/weights missing.
    """
    policy = run.candidate_policy
    W = np.array(policy.get("weights", []), dtype=float)
    b = np.array(policy.get("bias", []), dtype=float)
    actions = policy.get("actions", run.all_actions)
    feats = feature_lookup.get(task.task_id)
    if feats is None or W.size == 0:
        return None, None
    if len(feats) != W.shape[1]:
        return None, None
    logits = feats @ W.T + b
    return logits, list(actions)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def reconstruct_decision_trace(
    run: LoadedRun,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Reconstruct the candidate decision path for every evaluation task.

    For each test task, reconstructs:
    - raw logits (from candidate_policy weights applied to features)
    - raw probabilities (softmax of logits)
    - raw selected action (argmax of logits among eligible actions)
    - confidence score (max probability among eligible)
    - confidence threshold result (compare with policy's threshold)
    - abstention result
    - shift detector result
    - safety guard result
    - allowed-action mask
    - fallback reason
    - fallback policy
    - final selected action (from production LearnedPolicy.select_action())
    - executed action (same as final for evaluation)
    - verified utility (from counterfactual outcomes)

    Also calculates action distributions and stage utilities.

    Args:
        run: The loaded run with counterfactual outcomes and policies.
        task_ids: Optional list of task IDs.  Defaults to all test tasks.

    Returns:
        dict with per_task, action_distributions, stage_utilities, and
        interpretation.  All values are JSON-serializable.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # Build the production LearnedPolicy for full decision path.
    from capsule_brain.autolearn.policy import LearnedPolicy
    from capsule_brain.autolearn.evaluator import Action

    prod_policy = LearnedPolicy.from_dict(run.candidate_policy)
    confidence_threshold = prod_policy.confidence_threshold

    # Raw candidate fn for the raw argmax action.
    raw_fn = make_raw_candidate_fn(run)

    # Feature lookup for raw logits.
    feature_lookup = _build_feature_lookup(run)

    per_task: list[dict[str, Any]] = []

    # Track action distributions at each stage.
    raw_action_dist: Counter[str] = Counter()
    post_mask_action_dist: Counter[str] = Counter()
    final_action_dist: Counter[str] = Counter()

    # Track utilities at each stage.
    raw_utilities: list[float] = []
    post_mask_utilities: list[float] = []
    final_utilities: list[float] = []

    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue

        # --- Raw logits and probabilities ---
        raw_logits_result = _get_raw_logits(run, task, feature_lookup)
        raw_logits: list[float] | None = None
        raw_probs: list[float] | None = None
        raw_actions_list: list[str] | None = None
        raw_selected_action: str | None = None
        raw_confidence: float | None = None

        if raw_logits_result[0] is not None:
            logits_arr, actions_list = raw_logits_result
            raw_logits = [float(x) for x in logits_arr]
            probs_arr = _softmax(logits_arr, prod_policy.temperature)
            raw_probs = [float(x) for x in probs_arr]
            raw_actions_list = actions_list

            # Raw argmax among eligible actions.
            action_idx = {a: i for i, a in enumerate(actions_list)}
            eligible = [a for a in task.allowed_actions if a in action_idx]
            if eligible:
                elig_logits = np.array([logits_arr[action_idx[a]] for a in eligible])
                elig_probs = np.array([probs_arr[action_idx[a]] for a in eligible])
                raw_selected_action = eligible[int(np.argmax(elig_logits))]
                raw_confidence = float(np.max(elig_probs))

        # --- Raw candidate action (from make_raw_candidate_fn) ---
        raw_candidate_action = raw_fn(tid, task)
        if raw_candidate_action is not None:
            raw_action_dist[raw_candidate_action] += 1

        # --- Post-action-mask action ---
        # This is the raw argmax restricted to eligible actions (same as
        # raw_selected_action above, but computed via the raw fn which
        # already applies the mask).
        post_mask_action = raw_candidate_action
        if post_mask_action is not None:
            post_mask_action_dist[post_mask_action] += 1

        # --- Production decision (full stack) ---
        state = _make_state_from_task(task)
        allowed = [Action(a) for a in task.allowed_actions]
        decision = prod_policy.select_action(state, allowed_actions=allowed)

        final_action = decision.action.value
        final_confidence = float(decision.confidence)
        final_action_dist[final_action] += 1

        # Abstention / fallback metadata.
        abstained = bool(decision.abstained)
        confidence_threshold_result = final_confidence >= confidence_threshold
        fallback_reason: str | None = None
        fallback_policy: str | None = None
        if abstained:
            fallback_reason = decision.reason
            fallback_policy = "baseline"
        elif not confidence_threshold_result:
            fallback_reason = "confidence below threshold"
            fallback_policy = "baseline"

        # Shift detector and safety guard: the LearnedPolicy itself does not
        # run these; they are controller-level.  In the evaluation context
        # (counterfactual replay) we report them as not-triggered / not-
        # applicable since the canonical evaluator calls LearnedPolicy
        # directly without the ExecutiveController wrapper.
        shift_detector_result = {
            "triggered": False,
            "note": "LearnedPolicy does not run shift detection; "
                    "ExecutiveController-level guard not applied in "
                    "counterfactual replay.",
        }
        safety_guard_result = {
            "triggered": False,
            "note": "LearnedPolicy does not run safety guard; "
                    "ExecutiveController-level guard not applied in "
                    "counterfactual replay.",
        }

        # Allowed-action mask.
        allowed_action_mask = {
            a: (a in task.allowed_actions) for a in (raw_actions_list or run.all_actions)
        }

        # --- Executed action (same as final for evaluation) ---
        executed_action = final_action

        # --- Verified utility from counterfactual outcomes ---
        final_utility = run.get_utility_for_action(tid, final_action)
        raw_utility = (
            run.get_utility_for_action(tid, raw_candidate_action)
            if raw_candidate_action is not None
            else None
        )
        post_mask_utility = raw_utility  # same as raw in this context

        oracle_action = run.get_oracle_action(tid)
        oracle_utility = run.get_oracle_utility(tid)

        # Collect utilities for stage means.
        if raw_utility is not None:
            raw_utilities.append(raw_utility)
        if post_mask_utility is not None:
            post_mask_utilities.append(post_mask_utility)
        if final_utility is not None:
            final_utilities.append(final_utility)

        trace: dict[str, Any] = {
            "task_id": tid,
            "split": task.split,
            "family": task.family,
            "archetype": task.archetype,
            "allowed_actions": task.allowed_actions,
            "raw_logits": raw_logits,
            "raw_probabilities": raw_probs,
            "raw_actions": raw_actions_list,
            "raw_selected_action": raw_selected_action,
            "raw_confidence": raw_confidence,
            "confidence_score": final_confidence,
            "confidence_threshold": confidence_threshold,
            "confidence_threshold_result": confidence_threshold_result,
            "abstention_result": {
                "abstained": abstained,
                "reason": decision.reason,
            },
            "shift_detector_result": shift_detector_result,
            "safety_guard_result": safety_guard_result,
            "allowed_action_mask": allowed_action_mask,
            "fallback_reason": fallback_reason,
            "fallback_policy": fallback_policy,
            "final_selected_action": final_action,
            "executed_action": executed_action,
            "verified_utility": final_utility,
            "raw_candidate_utility": raw_utility,
            "oracle_action": oracle_action,
            "oracle_utility": oracle_utility,
            "regret": (
                oracle_utility - final_utility
                if final_utility is not None
                else None
            ),
            "raw_candidate_action": raw_candidate_action,
            "post_mask_action": post_mask_action,
        }
        per_task.append(trace)

    # --- Action distributions ---
    action_distributions: dict[str, dict[str, int]] = {
        "raw_candidate": dict(raw_action_dist),
        "post_action_mask": dict(post_mask_action_dist),
        "final_executed": dict(final_action_dist),
    }

    # --- Stage utilities ---
    stage_utilities: dict[str, float | None] = {
        "raw_candidate": (
            float(np.mean(raw_utilities)) if raw_utilities else None
        ),
        "post_action_mask": (
            float(np.mean(post_mask_utilities)) if post_mask_utilities else None
        ),
        "final_executed": (
            float(np.mean(final_utilities)) if final_utilities else None
        ),
    }

    # --- Interpretation ---
    interpretation = _build_interpretation(
        per_task, action_distributions, stage_utilities
    )

    return {
        "per_task": per_task,
        "action_distributions": action_distributions,
        "stage_utilities": stage_utilities,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def _build_interpretation(
    per_task: list[dict[str, Any]],
    action_distributions: dict[str, dict[str, int]],
    stage_utilities: dict[str, float | None],
) -> str:
    """Build a human-readable interpretation of the decision trace."""
    n = len(per_task)
    if n == 0:
        return "No tasks to analyze."

    n_abstained = sum(
        1 for t in per_task if t.get("abstention_result", {}).get("abstained", False)
    )
    abstain_rate = n_abstained / n

    raw_u = stage_utilities.get("raw_candidate")
    final_u = stage_utilities.get("final_executed")

    parts: list[str] = []

    # Abstention summary.
    if abstain_rate > 0.5:
        parts.append(
            f"The learned policy abstains on {n_abstained}/{n} tasks "
            f"({abstain_rate:.0%}), indicating the confidence threshold is "
            f"suppressing most candidate routing decisions."
        )
    elif abstain_rate > 0.1:
        parts.append(
            f"The learned policy abstains on {n_abstained}/{n} tasks "
            f"({abstain_rate:.0%}), partially suppressing candidate routing."
        )
    else:
        parts.append(
            f"The learned policy abstains on only {n_abstained}/{n} tasks "
            f"({abstain_rate:.0%}), so the confidence threshold is not a "
            f"major bottleneck."
        )

    # Utility comparison.
    if raw_u is not None and final_u is not None:
        delta = final_u - raw_u
        if abs(delta) < 0.01:
            parts.append(
                f"Raw candidate utility ({raw_u:.4f}) and final executed "
                f"utility ({final_u:.4f}) are nearly identical, meaning "
                f"calibration/fallback does not materially change outcomes."
            )
        elif delta > 0:
            parts.append(
                f"Calibration/fallback improves utility from {raw_u:.4f} "
                f"(raw) to {final_u:.4f} (executed), recovering "
                f"{delta:.4f} utility points."
            )
        else:
            parts.append(
                f"Calibration/fallback reduces utility from {raw_u:.4f} "
                f"(raw) to {final_u:.4f} (executed), losing "
                f"{abs(delta):.4f} utility points — suggesting the "
                f"production stack is suppressing useful routing."
            )

    # Action distribution concentration.
    raw_dist = action_distributions.get("raw_candidate", {})
    final_dist = action_distributions.get("final_executed", {})
    if raw_dist and final_dist:
        raw_hhi = sum((c / n) ** 2 for c in raw_dist.values())
        final_hhi = sum((c / n) ** 2 for c in final_dist.values())
        if raw_hhi > 0.5:
            parts.append(
                f"The raw candidate is highly concentrated "
                f"(HHI={raw_hhi:.2f}), routing most tasks to a single action."
            )
        if final_hhi > raw_hhi + 0.1:
            parts.append(
                f"The final executed distribution is more concentrated "
                f"(HHI {raw_hhi:.2f} -> {final_hhi:.2f}), indicating "
                f"fallback narrows the action space."
            )

    return " ".join(parts)
