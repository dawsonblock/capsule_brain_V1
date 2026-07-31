"""Canonical policy evaluator for v0.4.3.

All policy metrics must be computed through this single evaluator to prevent
drift between reports.  The evaluator takes a set of tasks, a policy decision
function, and the counterfactual outcomes, and returns task-level and aggregate
metrics.

Key design decisions:
- Uses the actual counterfactual outcomes (not recomputed from weights).
- Returns both task-level and aggregate metrics.
- Computes headroom metrics using the canonical aggregate formula.
- Does NOT average task-level ratios.
- Handles missing/unavailable outcomes explicitly.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable
from dataclasses import dataclass, field

import numpy as np

from .data import LoadedRun, TaskInfo, Outcome
from .config import MINIMUM_DENOMINATOR, HEADROOM_PERCENTAGE_SCALE


# ---------------------------------------------------------------------------
# Policy decision function type
# ---------------------------------------------------------------------------

PolicyDecisionFn = Callable[[str, TaskInfo], str | None]


# ---------------------------------------------------------------------------
# Result classes
# ---------------------------------------------------------------------------


@dataclass
class TaskEvalResult:
    task_id: str
    split: str
    family: str
    archetype: str
    selected_action: str | None
    executed_action: str | None
    selected_utility: float | None
    oracle_utility: float
    oracle_action: str | None
    regret: float | None  # oracle_utility - selected_utility (None if unavailable)
    verified_success: bool | None
    availability: str  # "executed" | "unavailable" | "no_action"
    allowed_actions: list[str]
    is_complete: bool


@dataclass
class PolicyEvalResult:
    policy_id: str
    n_tasks: int
    n_complete: int
    n_unavailable: int
    task_results: list[TaskEvalResult]
    mean_utility: float
    mean_utility_complete_only: float
    success_rate: float
    mean_regret: float | None
    mean_oracle_utility: float
    # Headroom metrics (computed against a reference policy).
    headroom_vs_reference: dict[str, Any] = field(default_factory=dict)
    # Family-level metrics.
    family_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Action distribution.
    action_distribution: dict[str, int] = field(default_factory=dict)
    # Task IDs digest for consistency checking.
    task_ids_digest: str = ""


# ---------------------------------------------------------------------------
# Canonical evaluator
# ---------------------------------------------------------------------------


def evaluate_policy_from_counterfactuals(
    run: LoadedRun,
    task_ids: list[str],
    policy_fn: PolicyDecisionFn,
    policy_id: str,
    reference_result: PolicyEvalResult | None = None,
    oracle_result: PolicyEvalResult | None = None,
) -> PolicyEvalResult:
    """Evaluate a policy on a set of tasks using counterfactual outcomes.

    Args:
        run: The loaded run with counterfactual outcomes.
        task_ids: Task IDs to evaluate on.
        policy_fn: Function(task_id, task_info) -> selected_action.
        policy_id: Name of the policy (e.g. "candidate", "sham", "baseline").
        reference_result: Optional reference policy result for headroom calculation.
        oracle_result: Optional oracle policy result for headroom calculation.

    Returns:
        PolicyEvalResult with task-level and aggregate metrics.
    """
    task_results: list[TaskEvalResult] = []

    for tid in task_ids:
        task = run.tasks.get(tid)
        if task is None:
            continue

        # Get the policy's selected action.
        selected_action = policy_fn(tid, task)

        # Get the oracle action and utility.
        oracle_action = run.get_oracle_action(tid)
        oracle_utility = run.get_oracle_utility(tid)

        # Get the selected action's utility from counterfactuals.
        if selected_action is not None:
            selected_utility = run.get_utility_for_action(tid, selected_action)
            outcome = run.outcomes_by_task_action.get((tid, selected_action))
            verified_success = None
            availability = "unavailable"
            if outcome is not None:
                availability = outcome.availability
                verified_success = outcome.verification == "success"
            else:
                verified_success = None
        else:
            selected_utility = None
            verified_success = None
            availability = "no_action"

        # Regret.
        regret = None
        if selected_utility is not None:
            regret = oracle_utility - selected_utility

        is_complete = (selected_utility is not None)

        task_results.append(TaskEvalResult(
            task_id=tid,
            split=task.split,
            family=task.family,
            archetype=task.archetype,
            selected_action=selected_action,
            executed_action=selected_action,  # Same for non-production policies
            selected_utility=selected_utility,
            oracle_utility=oracle_utility,
            oracle_action=oracle_action,
            regret=regret,
            verified_success=verified_success,
            availability=availability,
            allowed_actions=task.allowed_actions,
            is_complete=is_complete,
        ))

    # Aggregate metrics.
    complete_results = [r for r in task_results if r.is_complete]
    utilities = np.array([r.selected_utility for r in complete_results if r.selected_utility is not None])
    oracle_utils = np.array([r.oracle_utility for r in complete_results])
    regrets = np.array([r.regret for r in complete_results if r.regret is not None])
    successes = [r.verified_success for r in complete_results if r.verified_success is not None]

    mean_utility = float(utilities.mean()) if len(utilities) > 0 else 0.0
    mean_utility_complete_only = mean_utility  # Same thing — we only use complete
    success_rate = float(np.mean(successes)) if successes else 0.0
    mean_regret = float(regrets.mean()) if len(regrets) > 0 else None
    mean_oracle = float(oracle_utils.mean()) if len(oracle_utils) > 0 else 0.0

    # Action distribution.
    action_dist: dict[str, int] = {}
    for r in task_results:
        if r.selected_action is not None:
            action_dist[r.selected_action] = action_dist.get(r.selected_action, 0) + 1

    # Family metrics.
    family_metrics: dict[str, dict[str, Any]] = {}
    families = sorted({r.family for r in complete_results})
    for fam in families:
        fam_results = [r for r in complete_results if r.family == fam]
        fam_utils = [r.selected_utility for r in fam_results if r.selected_utility is not None]
        fam_oracle = [r.oracle_utility for r in fam_results]
        family_metrics[fam] = {
            "n_tasks": len(fam_results),
            "mean_utility": float(np.mean(fam_utils)) if fam_utils else 0.0,
            "mean_oracle_utility": float(np.mean(fam_oracle)) if fam_oracle else 0.0,
        }

    # Task IDs digest.
    task_ids_sorted = sorted(task_ids)
    task_ids_digest = hashlib.sha256(
        json.dumps(task_ids_sorted).encode()
    ).hexdigest()

    # Headroom vs reference.
    headroom = {}
    if reference_result is not None and oracle_result is not None:
        headroom = compute_headroom_metrics(
            candidate_mean=mean_utility,
            reference_mean=reference_result.mean_utility,
            oracle_mean=oracle_result.mean_utility,
            n_tasks=len(complete_results),
            task_ids_digest=task_ids_digest,
            reference_policy_id=reference_result.policy_id,
            candidate_policy_id=policy_id,
        )

    return PolicyEvalResult(
        policy_id=policy_id,
        n_tasks=len(task_results),
        n_complete=len(complete_results),
        n_unavailable=len(task_results) - len(complete_results),
        task_results=task_results,
        mean_utility=mean_utility,
        mean_utility_complete_only=mean_utility_complete_only,
        success_rate=success_rate,
        mean_regret=mean_regret,
        mean_oracle_utility=mean_oracle,
        headroom_vs_reference=headroom,
        family_metrics=family_metrics,
        action_distribution=action_dist,
        task_ids_digest=task_ids_digest,
    )


# ---------------------------------------------------------------------------
# Headroom calculation — the canonical aggregate formula
# ---------------------------------------------------------------------------


def compute_headroom_metrics(
    candidate_mean: float,
    reference_mean: float,
    oracle_mean: float,
    n_tasks: int,
    task_ids_digest: str = "",
    reference_policy_id: str = "",
    candidate_policy_id: str = "",
    utility_scale: str = "normalized_v1",
) -> dict[str, Any]:
    """Compute recovered headroom using the canonical aggregate formula.

    R = (candidate_mean - reference_mean) / (oracle_mean - reference_mean)

    This is an AGGREGATE formula — it uses mean utilities, NOT task-level ratios.
    Task-level ratios are never averaged.

    Returns None for recovered_fraction if the denominator is too small.
    """
    numerator = candidate_mean - reference_mean
    denominator = oracle_mean - reference_mean

    # Check finiteness.
    if not all(np.isfinite([candidate_mean, reference_mean, oracle_mean])):
        return {
            "candidate_mean": candidate_mean,
            "reference_mean": reference_mean,
            "oracle_mean": oracle_mean,
            "numerator": numerator,
            "denominator": denominator,
            "recovered_headroom_fraction": None,
            "recovered_headroom_percent": None,
            "task_count": n_tasks,
            "task_ids_digest": task_ids_digest,
            "utility_scale": utility_scale,
            "reference_policy_id": reference_policy_id,
            "candidate_policy_id": candidate_policy_id,
            "error": "non_finite_values",
        }

    if abs(denominator) <= MINIMUM_DENOMINATOR:
        return {
            "candidate_mean": candidate_mean,
            "reference_mean": reference_mean,
            "oracle_mean": oracle_mean,
            "numerator": numerator,
            "denominator": denominator,
            "recovered_headroom_fraction": None,
            "recovered_headroom_percent": None,
            "task_count": n_tasks,
            "task_ids_digest": task_ids_digest,
            "utility_scale": utility_scale,
            "reference_policy_id": reference_policy_id,
            "candidate_policy_id": candidate_policy_id,
            "error": "denominator_too_small",
        }

    fraction = numerator / denominator
    percent = fraction * HEADROOM_PERCENTAGE_SCALE

    return {
        "candidate_mean": float(candidate_mean),
        "reference_mean": float(reference_mean),
        "oracle_mean": float(oracle_mean),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "recovered_headroom_fraction": float(fraction),
        "recovered_headroom_percent": float(percent),
        "task_count": n_tasks,
        "task_ids_digest": task_ids_digest,
        "utility_scale": utility_scale,
        "reference_policy_id": reference_policy_id,
        "candidate_policy_id": candidate_policy_id,
    }


# ---------------------------------------------------------------------------
# Built-in policy decision functions
# ---------------------------------------------------------------------------


def make_oracle_fn(run: LoadedRun) -> PolicyDecisionFn:
    """Oracle: select the action with highest measured utility."""
    def fn(tid: str, task: TaskInfo) -> str | None:
        return run.get_oracle_action(tid)
    return fn


def make_baseline_fn(run: LoadedRun) -> PolicyDecisionFn:
    """Baseline: use the production BaselinePolicyV3."""
    from capsule_brain.autolearn.baseline import BaselinePolicyV3
    from capsule_brain.autolearn.evaluator import Action

    policy = BaselinePolicyV3()

    def fn(tid: str, task: TaskInfo) -> str | None:
        state = _make_state_from_task(task)
        allowed = [Action(a) for a in task.allowed_actions]
        decision = policy.select_action(state, allowed_actions=allowed)
        return decision.action.value

    return fn


def make_majority_fn(run: LoadedRun) -> PolicyDecisionFn:
    """Majority: select the most common best-action in D_experience."""
    from collections import Counter
    best_actions = [ex.get("best_action", "") for ex in run.train_examples]
    counts = Counter(best_actions)
    majority_action = counts.most_common(1)[0][0] if counts else "ANSWER_DIRECT"

    def fn(tid: str, task: TaskInfo) -> str | None:
        if majority_action in task.allowed_actions:
            return majority_action
        return task.allowed_actions[0] if task.allowed_actions else None
    return fn


def make_raw_candidate_fn(run: LoadedRun) -> PolicyDecisionFn:
    """Raw candidate: argmax of stored weights (no production stack)."""
    policy = run.candidate_policy
    W = np.array(policy.get("weights", []))
    b = np.array(policy.get("bias", []))
    actions = policy.get("actions", run.all_actions)
    action_idx = {a: i for i, a in enumerate(actions)}

    # Build feature lookup.
    feature_lookup = {}
    for ex in run.test_examples + run.train_examples + run.validation_examples + run.ood_examples:
        feature_lookup[ex["task_id"]] = np.array(ex.get("features", []))

    def fn(tid: str, task: TaskInfo) -> str | None:
        feats = feature_lookup.get(tid)
        if feats is None or W.size == 0:
            return task.allowed_actions[0] if task.allowed_actions else None
        if len(feats) != W.shape[1]:
            return task.allowed_actions[0] if task.allowed_actions else None
        logits = feats @ W.T + b
        eligible = [a for a in task.allowed_actions if a in action_idx]
        if not eligible:
            return task.allowed_actions[0] if task.allowed_actions else None
        elig_logits = np.array([logits[action_idx[a]] for a in eligible])
        return eligible[int(np.argmax(elig_logits))]

    return fn


def _make_state_from_task(task: TaskInfo) -> Any:
    """Build an ExecutiveState from a TaskInfo for production policy evaluation."""
    from capsule_brain.autolearn.evaluator import ExecutiveState
    prompt = task.prompt
    setup = task.setup_spec or {}
    return ExecutiveState(
        prompt_features={
            "text": prompt,
            "structured_output_request": "json" in prompt.lower(),
            "estimated_difficulty": 0.5,
            "workflow_capability_match": task.family == "workflow_required",
        },
        conversation_features={"depth": 0},
        memory_features={
            "hit_count": 1 if task.family == "memory_required" and setup.get("secret") else 0,
            "top_similarity": 0.95 if task.family == "memory_required" and setup.get("secret") else 0.0,
        },
        available_tools=setup.get("available_tools", []),
        workflow_available=True,
        model_id="qual-grounded-v04",
        context_length=len(prompt),
    )


def make_executed_candidate_fn(run: LoadedRun) -> PolicyDecisionFn:
    """Executed candidate: use the production LearnedPolicy.select_action().

    This includes the full production stack: confidence, abstention, shift
    detection, fallback, and safety.  This matches the actual evaluation
    artifact mean utility.
    """
    from capsule_brain.autolearn.policy import LearnedPolicy
    from capsule_brain.autolearn.evaluator import Action

    policy = LearnedPolicy.from_dict(run.candidate_policy)

    def fn(tid: str, task: TaskInfo) -> str | None:
        state = _make_state_from_task(task)
        allowed = [Action(a) for a in task.allowed_actions]
        decision = policy.select_action(state, allowed_actions=allowed)
        return decision.action.value

    return fn


def make_executed_sham_fn(run: LoadedRun) -> PolicyDecisionFn:
    """Executed sham: use the production LearnedPolicy.select_action() on sham policy."""
    from capsule_brain.autolearn.policy import LearnedPolicy
    from capsule_brain.autolearn.evaluator import Action

    policy = LearnedPolicy.from_dict(run.sham_policy)

    def fn(tid: str, task: TaskInfo) -> str | None:
        state = _make_state_from_task(task)
        allowed = [Action(a) for a in task.allowed_actions]
        decision = policy.select_action(state, allowed_actions=allowed)
        return decision.action.value

    return fn


def make_frequency_fn(run: LoadedRun, seed: int = 42) -> PolicyDecisionFn:
    """Frequency: sample actions according to D_experience best-action distribution."""
    from collections import Counter
    rng = np.random.default_rng(seed)
    best_actions = [ex.get("best_action", "") for ex in run.train_examples]
    counts = Counter(best_actions)
    total = sum(counts.values())
    actions_list = list(counts.keys())
    probs = np.array([counts[a] / total for a in actions_list])

    def fn(tid: str, task: TaskInfo) -> str | None:
        eligible = [a for a in actions_list if a in task.allowed_actions]
        if not eligible:
            return task.allowed_actions[0] if task.allowed_actions else None
        elig_probs = np.array([counts.get(a, 0) / total for a in eligible])
        elig_probs = elig_probs / elig_probs.sum()
        return eligible[int(rng.choice(len(eligible), p=elig_probs))]

    return fn


def make_random_fn(run: LoadedRun, seed: int = 42) -> PolicyDecisionFn:
    """Random: uniform random eligible action."""
    rng = np.random.default_rng(seed)

    def fn(tid: str, task: TaskInfo) -> str | None:
        eligible = task.allowed_actions
        if not eligible:
            return None
        return eligible[int(rng.integers(len(eligible)))]

    return fn


# ---------------------------------------------------------------------------
# Evaluate all policies through the canonical evaluator
# ---------------------------------------------------------------------------


def evaluate_all_policies(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    seed: int = 42,
) -> dict[str, PolicyEvalResult]:
    """Evaluate all standard policies through the canonical evaluator.

    Returns a dict mapping policy_id to PolicyEvalResult.
    Policies: oracle, baseline, majority, frequency, random,
              candidate_raw, candidate_executed, sham_executed.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # Evaluate oracle first (needed for headroom).
    oracle_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_oracle_fn(run), "oracle")

    # Baseline.
    baseline_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_baseline_fn(run), "baseline",
        reference_result=None, oracle_result=oracle_result)

    # Majority.
    majority_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_majority_fn(run), "majority",
        reference_result=baseline_result, oracle_result=oracle_result)

    # Frequency.
    freq_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_frequency_fn(run, seed=seed), "frequency",
        reference_result=baseline_result, oracle_result=oracle_result)

    # Random.
    random_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_random_fn(run, seed=seed), "random",
        reference_result=baseline_result, oracle_result=oracle_result)

    # Candidate raw (argmax of weights, no production stack).
    candidate_raw_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_raw_candidate_fn(run), "candidate_raw",
        reference_result=baseline_result, oracle_result=oracle_result)

    # Candidate executed (actual actions from evaluation artifact).
    candidate_exec_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_executed_candidate_fn(run), "candidate_executed",
        reference_result=baseline_result, oracle_result=oracle_result)

    # Sham executed (actual actions from sham evaluation artifact).
    sham_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_executed_sham_fn(run), "sham",
        reference_result=baseline_result, oracle_result=oracle_result)

    # Now compute headroom metrics with the correct reference/oracle pairs.
    # Candidate vs sham (headroom over sham).
    candidate_exec_result.headroom_vs_reference = compute_headroom_metrics(
        candidate_mean=candidate_exec_result.mean_utility,
        reference_mean=sham_result.mean_utility,
        oracle_mean=oracle_result.mean_utility,
        n_tasks=len([r for r in candidate_exec_result.task_results if r.is_complete]),
        task_ids_digest=candidate_exec_result.task_ids_digest,
        reference_policy_id="sham",
        candidate_policy_id="candidate_executed",
    )

    # Candidate vs baseline (headroom over baseline).
    candidate_exec_result.headroom_vs_reference["vs_baseline"] = compute_headroom_metrics(
        candidate_mean=candidate_exec_result.mean_utility,
        reference_mean=baseline_result.mean_utility,
        oracle_mean=oracle_result.mean_utility,
        n_tasks=len([r for r in candidate_exec_result.task_results if r.is_complete]),
        task_ids_digest=candidate_exec_result.task_ids_digest,
        reference_policy_id="baseline",
        candidate_policy_id="candidate_executed",
    )

    # Raw candidate vs sham.
    candidate_raw_result.headroom_vs_reference = compute_headroom_metrics(
        candidate_mean=candidate_raw_result.mean_utility,
        reference_mean=sham_result.mean_utility,
        oracle_mean=oracle_result.mean_utility,
        n_tasks=len([r for r in candidate_raw_result.task_results if r.is_complete]),
        task_ids_digest=candidate_raw_result.task_ids_digest,
        reference_policy_id="sham",
        candidate_policy_id="candidate_raw",
    )

    return {
        "oracle": oracle_result,
        "baseline": baseline_result,
        "majority": majority_result,
        "frequency": freq_result,
        "random": random_result,
        "candidate_raw": candidate_raw_result,
        "candidate_executed": candidate_exec_result,
        "sham": sham_result,
    }


def result_to_dict(result: PolicyEvalResult) -> dict[str, Any]:
    """Convert a PolicyEvalResult to a JSON-serializable dict."""
    return {
        "policy_id": result.policy_id,
        "n_tasks": result.n_tasks,
        "n_complete": result.n_complete,
        "n_unavailable": result.n_unavailable,
        "mean_utility": result.mean_utility,
        "mean_utility_complete_only": result.mean_utility_complete_only,
        "success_rate": result.success_rate,
        "mean_regret": result.mean_regret,
        "mean_oracle_utility": result.mean_oracle_utility,
        "headroom_vs_reference": result.headroom_vs_reference,
        "family_metrics": result.family_metrics,
        "action_distribution": result.action_distribution,
        "task_ids_digest": result.task_ids_digest,
        "task_results": [
            {
                "task_id": r.task_id,
                "split": r.split,
                "family": r.family,
                "archetype": r.archetype,
                "selected_action": r.selected_action,
                "executed_action": r.executed_action,
                "selected_utility": r.selected_utility,
                "oracle_utility": r.oracle_utility,
                "oracle_action": r.oracle_action,
                "regret": r.regret,
                "verified_success": r.verified_success,
                "availability": r.availability,
                "is_complete": r.is_complete,
            }
            for r in result.task_results
        ],
    }
