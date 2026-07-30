"""Common evaluation logic for v0.4.0 (Section 10/11).

Evaluates a policy by replaying the real measured counterfactual utilities:
for each task in the split, the policy selects an action; the selected
action's measured utility is the task's utility. No live re-execution
during eval -- the outcome is the real measured one.

v0.4.0 fixes:
  * Missing rows produce NO utility (None), not -1000.
  * Evaluation returns a typed PolicyEvaluation with explicit completeness
    tracking. Incomplete splits are BLOCKED for official qualification.
  * Grouped bootstrap by task group (Section 10.3).
  * Oracle evaluation uses only EXECUTED outcomes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicyV3
from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action, ExecutiveState

from .config import QualificationConfig
from .run_counterfactuals import _state_from_task, load_typed_outcomes
from .schemas import (
    CounterfactualOutcome,
    EvaluationStatus,
    OutcomeAvailability,
    PolicyEvaluation,
    VerificationOutcome,
    is_counterfactually_complete_dict,
    read_json,
    write_json,
)
from .statistics import grouped_paired_bootstrap


def _action_allowed_for_task(action: Action, task: dict) -> bool:
    return action.value in task["allowed_actions"]


def _select_action(policy, state: ExecutiveState, task: dict) -> Action:
    allowed = [Action(a) for a in task["allowed_actions"]]
    decision = policy.select_action(state, allowed_actions=allowed)
    return decision.action


def _outcomes_by_task(outcomes: list[CounterfactualOutcome]) -> dict[str, list[CounterfactualOutcome]]:
    d: dict[str, list[CounterfactualOutcome]] = {}
    for o in outcomes:
        d.setdefault(o.task_id, []).append(o)
    return d


def _make_state(task: dict) -> ExecutiveState:
    return _state_from_task(task)


def _load_outcomes(config: QualificationConfig) -> list[CounterfactualOutcome]:
    return load_typed_outcomes(config)


def _load_benchmark(config: QualificationConfig) -> dict[str, dict]:
    artifacts = Path(config.artifacts_dir)
    bm = read_json(artifacts / "benchmark_manifest.json")
    return {t["task_id"]: t for t in bm["tasks"]}


def evaluate_policy_on_split(
    config: QualificationConfig,
    split: str,
    policy,
    *,
    baseline=None,
    oracle: bool = False,
    label: str = "policy",
) -> PolicyEvaluation:
    """Evaluate a policy on the named split using real measured utilities.

    Returns a typed PolicyEvaluation. If any task in the split lacks a
    complete set of EXECUTED outcomes, the evaluation status is INCOMPLETE
    (or BLOCKED for official qualification) and mean_utility is None unless
    explicitly labeled as a partial descriptive statistic.
    """
    artifacts = Path(config.artifacts_dir)
    tasks_by_id = _load_benchmark(config)
    all_outcomes = _load_outcomes(config)
    outcomes_by_task = _outcomes_by_task(all_outcomes)
    outcomes_by_task_action: dict[tuple[str, str], CounterfactualOutcome] = {}
    for o in all_outcomes:
        outcomes_by_task_action[(o.task_id, o.action_id)] = o

    split_tasks = [t for t in tasks_by_id.values() if t["split"] == split]
    if not split_tasks:
        return PolicyEvaluation(
            status=EvaluationStatus.NOT_RUN,
            policy_id=label,
            split_name=split,
            n_manifest_tasks=0,
            n_complete_tasks=0,
            n_missing_tasks=0,
            mean_utility=None,
            success_rate=None,
            metrics={},
            missing_task_ids=(),
            blocking_reasons=("no tasks in split",),
        )

    utilities: list[float] = []
    successes: list[int] = []
    per_family: dict[str, Any] = {}
    selected_counts: dict[str, int] = {}
    deltas: list[float] = []
    delta_groups: list[str] = []
    task_results: list[dict] = []
    missing_task_ids: list[str] = []
    blocking_reasons: list[str] = []

    for task in split_tasks:
        tid = task["task_id"]
        task_outcomes = outcomes_by_task.get(tid, [])
        state = _make_state(task)

        # Check counterfactual completeness for this task.
        complete = is_counterfactually_complete_dict(task, outcomes_by_task_action)
        if not complete:
            missing_task_ids.append(tid)
            blocking_reasons.append(f"task {tid} lacks complete EXECUTED outcomes")
            continue

        if oracle:
            # Oracle: pick the measured argmax utility action among EXECUTED outcomes.
            executed = [o for o in task_outcomes if o.availability is OutcomeAvailability.EXECUTED and o.utility is not None]
            if not executed:
                missing_task_ids.append(tid)
                blocking_reasons.append(f"task {tid} has no EXECUTED outcomes for oracle")
                continue
            best_outcome = max(executed, key=lambda o: o.utility)
            action = Action(best_outcome.action_id)
        else:
            action = _select_action(policy, state, task)

        # Find the selected action's outcome.
        selected_outcome = outcomes_by_task_action.get((tid, action.value))
        if selected_outcome is None or selected_outcome.availability is not OutcomeAvailability.EXECUTED:
            # The policy selected an action that was not executed. This is a
            # missing-evidence situation, NOT a -1000 failure.
            missing_task_ids.append(tid)
            blocking_reasons.append(f"task {tid}: selected action {action.value} not EXECUTED")
            continue

        u = float(selected_outcome.utility)
        success = selected_outcome.verification is VerificationOutcome.SUCCESS
        utilities.append(u)
        successes.append(1 if success else 0)
        selected_counts[action.value] = selected_counts.get(action.value, 0) + 1

        family = task["family"]
        if family not in per_family:
            per_family[family] = {"n": 0, "mean_utility": 0.0, "success_rate": 0.0, "selected": {}}
        per_family[family]["n"] += 1
        per_family[family]["selected"][action.value] = per_family[family]["selected"].get(action.value, 0) + 1

        if baseline is not None:
            base_action = _select_action(baseline, state, task)
            base_outcome = outcomes_by_task_action.get((tid, base_action.value))
            if base_outcome is not None and base_outcome.availability is OutcomeAvailability.EXECUTED and base_outcome.utility is not None:
                deltas.append(u - float(base_outcome.utility))
                delta_groups.append(task.get("group_id", tid))

        task_results.append({
            "task_id": tid,
            "family": family,
            "group_id": task.get("group_id", ""),
            "selected_action": action.value,
            "utility": u,
            "verified_success": success,
        })

    for fam, stats in per_family.items():
        fam_rows = [r for r in task_results if r["family"] == fam]
        if fam_rows:
            stats["mean_utility"] = sum(r["utility"] for r in fam_rows) / len(fam_rows)
            stats["success_rate"] = sum(1 for r in fam_rows if r["verified_success"]) / len(fam_rows)

    n_complete = len(utilities)
    n_missing = len(missing_task_ids)
    n_manifest = len(split_tasks)

    # Determine status.
    if n_missing > 0:
        status = EvaluationStatus.INCOMPLETE
        mean_u = None
        success_rate = None
    elif n_complete == 0:
        status = EvaluationStatus.BLOCKED
        mean_u = None
        success_rate = None
    else:
        status = EvaluationStatus.EVALUATED
        mean_u = sum(utilities) / len(utilities)
        success_rate = sum(successes) / len(successes)

    metrics: dict[str, float] = {}
    if deltas and status is EvaluationStatus.EVALUATED:
        boot = grouped_paired_bootstrap(
            deltas, delta_groups,
            n_bootstrap=config.bootstrap_replicates,
            epsilon=config.epsilon_utility,
            seed=config.bootstrap_seed,
        )
        metrics["vs_baseline_mean_delta"] = boot.mean
        metrics["vs_baseline_ci_low"] = boot.ci_low
        metrics["vs_baseline_ci_high"] = boot.ci_high
        metrics["vs_baseline_passes_threshold"] = 1.0 if boot.passes_threshold else 0.0

    # Tool/workflow precision/recall.
    if split in {"test", "ood"} and task_results:
        metrics.update(_tool_metrics(task_results))
        metrics.update(_workflow_metrics(task_results))

    # Action diversity.
    if selected_counts:
        total = sum(selected_counts.values())
        metrics["max_single_action_share"] = max(selected_counts.values()) / total

    return PolicyEvaluation(
        status=status,
        policy_id=label,
        split_name=split,
        n_manifest_tasks=n_manifest,
        n_complete_tasks=n_complete,
        n_missing_tasks=n_missing,
        mean_utility=mean_u,
        success_rate=success_rate,
        metrics=metrics,
        missing_task_ids=tuple(missing_task_ids),
        blocking_reasons=tuple(blocking_reasons),
    )


def _tool_metrics(task_results: list[dict]) -> dict[str, float]:
    tp = fp = fn = 0
    for r in task_results:
        is_tool = r["family"] == "tool_required"
        sel = r["selected_action"]
        if is_tool and sel == "CALL_TOOL":
            tp += 1
        elif not is_tool and sel == "CALL_TOOL":
            fp += 1
        elif is_tool and sel != "CALL_TOOL":
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tool_precision": precision, "tool_recall": recall}


def _workflow_metrics(task_results: list[dict]) -> dict[str, float]:
    tp = fp = fn = 0
    for r in task_results:
        is_wf = r["family"] == "workflow_required"
        sel = r["selected_action"]
        if is_wf and sel == "START_WORKFLOW":
            tp += 1
        elif not is_wf and sel == "START_WORKFLOW":
            fp += 1
        elif is_wf and sel != "START_WORKFLOW":
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"workflow_precision": precision, "workflow_recall": recall}


def evaluation_to_legacy_dict(ev: PolicyEvaluation) -> dict[str, Any]:
    """Convert a PolicyEvaluation to a dict for artifact serialization."""
    return ev.to_dict()
