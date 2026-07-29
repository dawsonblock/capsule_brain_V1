"""Evaluate the candidate policy against the baseline on held-out tasks (v0.2).

Reads the candidate policy, builds the held-out task set (test + ood splits),
looks up the precomputed counterfactual outcome for each (task, action) pair,
runs the paired evaluator, computes calibration metrics, and writes
evaluation.json with SHA-256 provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capsule_brain.autolearn.baseline import BaselinePolicyV2
from capsule_brain.autolearn.calibration import compute_calibration
from capsule_brain.autolearn.dataset import load_split_manifest
from capsule_brain.autolearn.evaluator import evaluate_paired
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.registry import PolicyRegistry
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.utility import UtilityConfig

from tasks import RoutingTask, build_all_tasks


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _build_outcome_lookup(tasks: list[RoutingTask]) -> dict[tuple[str, str], dict]:
    from run_counterfactuals import SimulatedCounterfactualRuntime

    runtime = SimulatedCounterfactualRuntime()
    lookup: dict[tuple[str, str], dict] = {}
    for task in tasks:
        for action in task.allowed_actions:
            result = runtime.execute(task, action)
            passed, status = task.verifier(action, {
                "text": result.text,
                "expected_token": task.setup.get("expected_token", ""),
                "expected_secret": task.setup.get("expected_secret", ""),
                "expected_nonce": task.setup.get("expected_nonce", ""),
                "acceptance_passed": result.acceptance_passed,
                "safety_violation": result.safety_violation,
                "verified_success": result.outcome.verified_success,
                "verification_status": result.outcome.verification_status,
                "latency_ms": result.outcome.latency_ms,
                "token_count": result.outcome.token_count,
                "tool_failures": result.outcome.tool_failures,
                "workflow_iterations": result.outcome.workflow_iterations,
                "operator_intervention": result.outcome.operator_intervention,
            })
            lookup[(task.task_id, action.value)] = {
                "verified_success": passed,
                "verification_status": status,
                "latency_ms": result.outcome.latency_ms,
                "token_count": result.outcome.token_count,
                "tool_failures": result.outcome.tool_failures,
                "workflow_iterations": result.outcome.workflow_iterations,
                "operator_intervention": result.outcome.operator_intervention,
                "safety_violation": result.safety_violation,
                "runtime_error": result.outcome.runtime_error,
                "action_completed": result.outcome.action_completed,
                "tool_name": result.outcome.tool_name,
                "tool_calls_executed": result.outcome.tool_calls_executed,
                "tool_result_valid": result.outcome.tool_result_valid,
                "final_answer_grounded": result.outcome.final_answer_grounded,
                "workflow_name": result.outcome.workflow_name,
                "workflow_run_id": result.outcome.workflow_run_id,
                "acceptance_present": result.outcome.acceptance_present,
                "acceptance_passed": result.outcome.acceptance_passed,
                "exit_code": result.outcome.exit_code,
                "unnecessary_tool_use": result.outcome.unnecessary_tool_use,
                "unnecessary_workflow_use": result.outcome.unnecessary_workflow_use,
            }
    return lookup


def _compute_v2_metrics(
    eval_result,
    tasks_by_id: dict[str, RoutingTask],
    task_ids: list[str],
    candidate: LearnedPolicy,
) -> dict:
    """Compute v2 metrics: workflow routing accuracy, over-routing rate, Brier."""
    from capsule_brain.autolearn.features import FeatureExtractor

    extractor = FeatureExtractor()
    # Workflow routing accuracy: fraction of coding_workflow tasks where
    # the candidate chose START_WORKFLOW.
    wf_tasks = [tid for tid in task_ids if tasks_by_id[tid].family == "coding_workflow"]
    wf_correct = 0
    for t in eval_result.task_evaluations:
        if t.task_family == "coding_workflow" and t.candidate_action == Action.START_WORKFLOW:
            wf_correct += 1
    wf_accuracy = wf_correct / len(wf_tasks) if wf_tasks else 0.0

    # Over-routing rate: fraction of tasks where the candidate used a
    # heavier action than necessary (tool/workflow on direct/memory tasks).
    over_routing = 0
    for t in eval_result.task_evaluations:
        if t.task_family == "direct_answer" and t.candidate_action in (Action.CALL_TOOL, Action.START_WORKFLOW):
            over_routing += 1
        elif t.task_family == "memory_required" and t.candidate_action == Action.START_WORKFLOW:
            over_routing += 1
    over_routing_rate = over_routing / len(task_ids) if task_ids else 0.0

    # Brier score from (confidence, correct) pairs.
    # "correct" means the candidate chose the expected action for the task,
    # not merely that it beat the baseline on utility.
    pairs: list[tuple[float, bool]] = []
    abstained: list[bool] = []
    for t in eval_result.task_evaluations:
        task = tasks_by_id.get(t.task_id)
        if task is None:
            continue
        try:
            c_dec = candidate.select_action(task.state, extractor=extractor)
            conf = c_dec.confidence
            correct = t.candidate_action == task.expected_action
            pairs.append((conf, correct))
            abstained.append(c_dec.abstained)
        except Exception:
            pairs.append((0.0, False))
            abstained.append(True)

    cal = compute_calibration(pairs, abstained=abstained)

    return {
        "candidate_workflow_routing_accuracy": wf_accuracy,
        "candidate_over_routing_rate": over_routing_rate,
        "candidate_brier_score": cal.brier_score,
        "candidate_ece": cal.ece,
        "candidate_abstention_rate": cal.abstention_rate,
        "calibration": cal.to_dict(),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    candidate_id = (out_dir / "candidate_policy_id.txt").read_text().strip()
    registry = PolicyRegistry(out_dir / "policies")
    candidate = registry.load_policy(candidate_id)
    print(f"loaded candidate {candidate.policy_id} ({candidate.policy_version})")

    split_manifest = load_split_manifest(out_dir / "split_manifest.json")
    tasks = build_all_tasks()
    tasks_by_id = {t.task_id: t for t in tasks}

    test_ids = [
        tid for tid, sp in split_manifest.assignments.items() if sp == "test"
    ]
    ood_ids = [
        tid for tid, sp in split_manifest.assignments.items() if sp == "ood"
    ]
    print(f"test={len(test_ids)} ood={len(ood_ids)}")

    outcome_lookup = _build_outcome_lookup(tasks)

    def _run_eval(ids: list[str], label: str) -> dict:
        states = {tid: tasks_by_id[tid].state for tid in ids}
        families = {tid: tasks_by_id[tid].family for tid in ids}
        eval_result = evaluate_paired(
            task_ids=ids,
            task_families=families,
            states=states,
            outcomes_by_action=outcome_lookup,
            baseline=BaselinePolicyV2(),
            candidate=candidate,
            utility_config=UtilityConfig(),
            allowed_actions=None,
            expected_tool_task_ids={tid for tid in ids if tasks_by_id[tid].family == "tool_required"},
            expected_workflow_task_ids={tid for tid in ids if tasks_by_id[tid].family == "coding_workflow"},
            bootstrap_iterations=2000,
            bootstrap_seed=0,
            confidence_threshold=candidate.confidence_threshold,
        )
        print(f"\n=== {label} ===")
        print(f"baseline mean_utility={eval_result.baseline_metrics.mean_utility:.3f}")
        print(f"candidate mean_utility={eval_result.candidate_metrics.mean_utility:.3f}")
        print(f"mean_delta={eval_result.mean_delta:.3f} "
              f"CI=[{eval_result.delta_ci_low:.3f}, {eval_result.delta_ci_high:.3f}]")
        print(f"wins={eval_result.wins} ties={eval_result.ties} losses={eval_result.losses}")
        print(f"baseline success_rate={eval_result.baseline_metrics.verified_success_rate:.3f} "
              f"candidate success_rate={eval_result.candidate_metrics.verified_success_rate:.3f}")
        print(f"baseline safety={eval_result.baseline_metrics.safety_violations} "
              f"candidate safety={eval_result.candidate_metrics.safety_violations}")

        # v2 metrics.
        v2_metrics = _compute_v2_metrics(eval_result, tasks_by_id, ids, candidate)
        print(f"workflow_routing_accuracy={v2_metrics['candidate_workflow_routing_accuracy']:.3f}")
        print(f"over_routing_rate={v2_metrics['candidate_over_routing_rate']:.3f}")
        print(f"brier_score={v2_metrics['candidate_brier_score']:.3f}")

        result_dict = eval_result.to_dict()
        result_dict["v2_metrics"] = v2_metrics
        return result_dict

    test_eval = _run_eval(test_ids, "TEST")
    ood_eval = _run_eval(ood_ids, "OOD") if ood_ids else {}

    eval_text = json.dumps({
        "candidate_policy_id": candidate.policy_id,
        "test": test_eval,
        "ood": ood_eval,
    }, sort_keys=True, indent=2)
    (out_dir / "evaluation.json").write_text(eval_text)
    print(f"\nwrote {out_dir / 'evaluation.json'}")
    print(f"evaluation SHA-256: {_compute_sha256(eval_text)}")


if __name__ == "__main__":
    main()
