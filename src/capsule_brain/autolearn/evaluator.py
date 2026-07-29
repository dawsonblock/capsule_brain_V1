"""Offline evaluator: paired utility deltas + bootstrap confidence intervals.

Evaluates baseline and candidate on identical held-out tasks and reports
paired differences. Promotion is never based on aggregate accuracy alone —
the gate (promotion.py) consumes these paired metrics.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicy
from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action, ExecutiveState, Outcome
from capsule_brain.autolearn.utility import UtilityConfig, UtilityFunction


@dataclass(slots=True)
class TaskEvaluation:
    """Per-task paired evaluation record."""

    task_id: str
    task_family: str
    baseline_action: Action
    candidate_action: Action
    baseline_utility: float
    candidate_utility: float
    delta: float
    disagreement: bool
    baseline_outcome: dict[str, Any]
    candidate_outcome: dict[str, Any]


@dataclass(slots=True)
class ActionMetrics:
    """Aggregate metrics for one policy over a set of tasks."""

    verified_success_rate: float = 0.0
    mean_utility: float = 0.0
    median_utility: float = 0.0
    tool_precision: float = 0.0
    tool_recall: float = 0.0
    workflow_routing_accuracy: float = 0.0
    unnecessary_reflection_rate: float = 0.0
    operator_escalation_rate: float = 0.0
    median_latency: float = 0.0
    p95_latency: float = 0.0
    tokens_per_successful_task: float = 0.0
    failure_rate: float = 0.0
    safety_violations: int = 0
    n: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified_success_rate": self.verified_success_rate,
            "mean_utility": self.mean_utility,
            "median_utility": self.median_utility,
            "tool_precision": self.tool_precision,
            "tool_recall": self.tool_recall,
            "workflow_routing_accuracy": self.workflow_routing_accuracy,
            "unnecessary_reflection_rate": self.unnecessary_reflection_rate,
            "operator_escalation_rate": self.operator_escalation_rate,
            "median_latency": self.median_latency,
            "p95_latency": self.p95_latency,
            "tokens_per_successful_task": self.tokens_per_successful_task,
            "failure_rate": self.failure_rate,
            "safety_violations": self.safety_violations,
            "n": self.n,
        }


@dataclass(slots=True)
class PairedEvaluation:
    task_evaluations: list[TaskEvaluation] = field(default_factory=list)
    baseline_metrics: ActionMetrics = field(default_factory=ActionMetrics)
    candidate_metrics: ActionMetrics = field(default_factory=ActionMetrics)
    mean_delta: float = 0.0
    median_delta: float = 0.0
    delta_ci_low: float = 0.0
    delta_ci_high: float = 0.0
    wins: int = 0
    ties: int = 0
    losses: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    bootstrap_iterations: int = 0
    bootstrap_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_evaluations": [
                {
                    "task_id": t.task_id,
                    "task_family": t.task_family,
                    "baseline_action": t.baseline_action.value,
                    "candidate_action": t.candidate_action.value,
                    "baseline_utility": t.baseline_utility,
                    "candidate_utility": t.candidate_utility,
                    "delta": t.delta,
                    "disagreement": t.disagreement,
                }
                for t in self.task_evaluations
            ],
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "mean_delta": self.mean_delta,
            "median_delta": self.median_delta,
            "delta_ci_low": self.delta_ci_low,
            "delta_ci_high": self.delta_ci_high,
            "wins": self.wins,
            "ties": self.ties,
            "losses": self.losses,
            "confusion": self.confusion,
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
        }


# A TaskOutcomeResolver maps (task_id, action) -> outcome dict. The harness
# supplies this; the evaluator itself is policy-only and does not execute
# the runtime. This keeps evaluation deterministic and reproducible.
TaskOutcomeResolver = Any  # callable: (task_id: str, action: Action) -> dict[str, Any]


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _median(vals: list[float]) -> float:
    return _percentile(sorted(vals), 0.5)


def _compute_action_metrics(
    task_ids: list[str],
    actions: dict[str, Action],
    outcomes: dict[str, dict[str, Any]],
    utilities: dict[str, float],
    *,
    expected_tool_task_ids: set[str] | None = None,
    expected_workflow_task_ids: set[str] | None = None,
) -> ActionMetrics:
    n = len(task_ids)
    if n == 0:
        return ActionMetrics()
    utils = [utilities[tid] for tid in task_ids]
    successes = sum(
        1 for tid in task_ids
        if outcomes[tid].get("verified_success", False)
    )
    failures = sum(
        1 for tid in task_ids
        if outcomes[tid].get("verification_status") == "fail"
    )
    safety = sum(
        1 for tid in task_ids
        if outcomes[tid].get("safety_violation", False)
    )
    latencies = sorted(outcomes[tid].get("latency_ms", 0.0) for tid in task_ids)
    tokens = [outcomes[tid].get("token_count", 0) for tid in task_ids]
    successful_tokens = sum(
        outcomes[tid].get("token_count", 0)
        for tid in task_ids
        if outcomes[tid].get("verified_success", False)
    )
    operator = sum(
        1 for tid in task_ids
        if outcomes[tid].get("operator_intervention", False)
    )
    reflections = sum(1 for tid in task_ids if actions[tid] == Action.REFLECT)

    # Tool precision/recall: precision = P(tool chosen | tool was required);
    # recall = P(tool chosen | tool required). We treat "tool required" as the
    # ground truth set when expected_tool_task_ids is provided.
    tool_precision = 0.0
    tool_recall = 0.0
    if expected_tool_task_ids is not None:
        chosen_tool = {tid for tid in task_ids if actions[tid] == Action.CALL_TOOL}
        tp = len(chosen_tool & expected_tool_task_ids)
        tool_precision = tp / len(chosen_tool) if chosen_tool else 0.0
        tool_recall = (
            tp / len(expected_tool_task_ids) if expected_tool_task_ids else 0.0
        )

    workflow_acc = 0.0
    if expected_workflow_task_ids is not None:
        chosen_wf = {tid for tid in task_ids if actions[tid] == Action.START_WORKFLOW}
        tp = len(chosen_wf & expected_workflow_task_ids)
        workflow_acc = (
            tp / len(expected_workflow_task_ids)
            if expected_workflow_task_ids
            else 0.0
        )

    return ActionMetrics(
        verified_success_rate=successes / n,
        mean_utility=sum(utils) / n,
        median_utility=_median(utils),
        tool_precision=tool_precision,
        tool_recall=tool_recall,
        workflow_routing_accuracy=workflow_acc,
        unnecessary_reflection_rate=reflections / n,
        operator_escalation_rate=operator / n,
        median_latency=_percentile(latencies, 0.5),
        p95_latency=_percentile(latencies, 0.95),
        tokens_per_successful_task=(
            successful_tokens / successes if successes > 0 else 0.0
        ),
        failure_rate=failures / n,
        safety_violations=safety,
        n=n,
    )


def evaluate_paired(
    *,
    task_ids: list[str],
    task_families: dict[str, str],
    states: dict[str, ExecutiveState],
    outcomes_by_action: dict[tuple[str, str], dict[str, Any]],
    baseline: BaselinePolicy,
    candidate: LearnedPolicy,
    utility_config: UtilityConfig | None = None,
    allowed_actions: list[Action] | None = None,
    expected_tool_task_ids: set[str] | None = None,
    expected_workflow_task_ids: set[str] | None = None,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 0,
    confidence_threshold: float = 0.5,
) -> PairedEvaluation:
    """Evaluate baseline vs candidate on identical held-out tasks.

    ``outcomes_by_action`` maps (task_id, action_value) -> outcome dict. The
    evaluator does not execute the runtime; it looks up the precomputed
    counterfactual outcome for whatever action each policy selects.
    """
    ufn = UtilityFunction(utility_config or UtilityConfig())
    extractor = FeatureExtractor()

    task_evaluations: list[TaskEvaluation] = []
    baseline_actions: dict[str, Action] = {}
    candidate_actions: dict[str, Action] = {}
    baseline_outcomes: dict[str, dict[str, Any]] = {}
    candidate_outcomes: dict[str, dict[str, Any]] = {}
    baseline_utils: dict[str, float] = {}
    candidate_utils: dict[str, float] = {}

    confusion: dict[str, dict[str, int]] = {
        a.value: {b.value: 0 for b in Action.all()} for a in Action.all()
    }

    for tid in task_ids:
        state = states[tid]
        b_dec = baseline.select_action(state, allowed_actions=allowed_actions)
        c_dec = candidate.select_action(
            state, extractor=extractor, allowed_actions=allowed_actions
        )
        # Candidate abstention falls back to baseline action.
        c_action = (
            b_dec.action if c_dec.abstained or c_dec.confidence < confidence_threshold
            else c_dec.action
        )
        baseline_actions[tid] = b_dec.action
        candidate_actions[tid] = c_action

        b_out = outcomes_by_action.get((tid, b_dec.action.value), {})
        c_out = outcomes_by_action.get((tid, c_action.value), {})

        # Compute utilities from the outcome vectors.
        b_oc = Outcome(
            verified_success=bool(b_out.get("verified_success", False)),
            verification_status=str(b_out.get("verification_status", "skip")),
            latency_ms=float(b_out.get("latency_ms", 0.0) or 0.0),
            token_count=int(b_out.get("token_count", 0) or 0),
            tool_failures=int(b_out.get("tool_failures", 0) or 0),
            workflow_iterations=int(b_out.get("workflow_iterations", 0) or 0),
            operator_intervention=bool(b_out.get("operator_intervention", False)),
            safety_violation=bool(b_out.get("safety_violation", False)),
        )
        c_oc = Outcome(
            verified_success=bool(c_out.get("verified_success", False)),
            verification_status=str(c_out.get("verification_status", "skip")),
            latency_ms=float(c_out.get("latency_ms", 0.0) or 0.0),
            token_count=int(c_out.get("token_count", 0) or 0),
            tool_failures=int(c_out.get("tool_failures", 0) or 0),
            workflow_iterations=int(c_out.get("workflow_iterations", 0) or 0),
            operator_intervention=bool(c_out.get("operator_intervention", False)),
            safety_violation=bool(c_out.get("safety_violation", False)),
        )
        b_u, _ = ufn.compute(b_oc)
        c_u, _ = ufn.compute(c_oc)

        baseline_outcomes[tid] = b_out
        candidate_outcomes[tid] = c_out
        baseline_utils[tid] = b_u
        candidate_utils[tid] = c_u

        confusion[b_dec.action.value][c_action.value] = (
            confusion[b_dec.action.value][c_action.value] + 1
        )

        task_evaluations.append(TaskEvaluation(
            task_id=tid,
            task_family=task_families.get(tid, "unknown"),
            baseline_action=b_dec.action,
            candidate_action=c_action,
            baseline_utility=b_u,
            candidate_utility=c_u,
            delta=c_u - b_u,
            disagreement=(b_dec.action != c_action),
            baseline_outcome=b_out,
            candidate_outcome=c_out,
        ))

    baseline_metrics = _compute_action_metrics(
        task_ids, baseline_actions, baseline_outcomes, baseline_utils,
        expected_tool_task_ids=expected_tool_task_ids,
        expected_workflow_task_ids=expected_workflow_task_ids,
    )
    candidate_metrics = _compute_action_metrics(
        task_ids, candidate_actions, candidate_outcomes, candidate_utils,
        expected_tool_task_ids=expected_tool_task_ids,
        expected_workflow_task_ids=expected_workflow_task_ids,
    )

    deltas = [t.delta for t in task_evaluations]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    median_delta = _median(deltas)
    ci_low, ci_high = bootstrap_ci(deltas, iterations=bootstrap_iterations, seed=bootstrap_seed)

    wins = sum(1 for d in deltas if d > 1e-9)
    losses = sum(1 for d in deltas if d < -1e-9)
    ties = len(deltas) - wins - losses

    return PairedEvaluation(
        task_evaluations=task_evaluations,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        mean_delta=mean_delta,
        median_delta=median_delta,
        delta_ci_low=ci_low,
        delta_ci_high=ci_high,
        wins=wins,
        ties=ties,
        losses=losses,
        confusion=confusion,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )


def bootstrap_ci(
    deltas: list[float],
    *,
    iterations: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for the mean of ``deltas``."""
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0
    if iterations <= 0:
        # Fall back to a normal approximation.
        mean = sum(deltas) / n
        var = sum((d - mean) ** 2 for d in deltas) / n
        sd = math.sqrt(var)
        return mean - 1.96 * sd, mean + 1.96 * sd
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = max(0, int((alpha / 2) * iterations))
    hi_idx = min(iterations - 1, int((1 - alpha / 2) * iterations))
    return means[lo_idx], means[hi_idx]
