"""Evaluate the candidate policy against the baseline on held-out tasks.

Reads the candidate policy from the PolicyRegistry, builds the held-out
task set (test + ood splits), looks up the precomputed counterfactual
outcome for each (task, action) pair, and runs the paired evaluator.
Writes evaluation.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from capsule_brain.autolearn.baseline import BaselinePolicy
from capsule_brain.autolearn.dataset import load_split_manifest
from capsule_brain.autolearn.evaluator import evaluate_paired
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.registry import PolicyRegistry
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.utility import UtilityConfig

from tasks import RoutingTask, build_all_tasks


def _build_outcome_lookup(tasks: list[RoutingTask]) -> dict[tuple[str, str], dict]:
    """Rebuild the deterministic outcome for each (task, action) pair.

    This mirrors run_counterfactuals.DeterministicRuntime so the evaluator
    does not need to re-execute the runtime; it looks up the precomputed
    outcome for whatever action each policy selects.
    """
    from run_counterfactuals import DeterministicRuntime

    runtime = DeterministicRuntime()
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
            }
    return lookup


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    candidate_id = (out_dir / "candidate_policy_id.txt").read_text().strip()
    registry = PolicyRegistry(out_dir / "policies")
    candidate = registry.load_policy(candidate_id)
    print(f"loaded candidate {candidate.policy_id} ({candidate.policy_version})")

    split_manifest = load_split_manifest(out_dir / "split_manifest.json")
    tasks = build_all_tasks()
    tasks_by_id = {t.task_id: t for t in tasks}

    # Held-out task sets: test split for the main evaluation. For OOD, use
    # the operator_safety tasks from the test split — these are the hardest
    # tasks with the most unusual feature profiles (safety-critical prompts
    # that require ASK_OPERATOR, a pattern that's qualitatively different
    # from the normal routing tasks).
    test_ids_all = [
        tid for tid, sp in split_manifest.assignments.items() if sp == "test"
    ]
    ood_ids = [
        tid for tid in test_ids_all
        if tasks_by_id[tid].family == "operator_safety"
    ]
    test_ids = [tid for tid in test_ids_all if tid not in ood_ids]
    print(f"test={len(test_ids)} ood={len(ood_ids)}")

    # If the hash-based split produced no test tasks (families were all
    # explicitly assigned), use validation families as the held-out set.
    if not test_ids:
        test_ids = [
            tid for tid, sp in split_manifest.assignments.items() if sp == "validation"
        ]
        print(f"no test split; using validation as held-out: {len(test_ids)}")

    outcome_lookup = _build_outcome_lookup(tasks)

    def _run_eval(ids: list[str], label: str) -> dict:
        states = {tid: tasks_by_id[tid].state for tid in ids}
        families = {tid: tasks_by_id[tid].family for tid in ids}
        eval_result = evaluate_paired(
            task_ids=ids,
            task_families=families,
            states=states,
            outcomes_by_action=outcome_lookup,
            baseline=BaselinePolicy(),
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
        return eval_result.to_dict()

    test_eval = _run_eval(test_ids, "TEST")
    ood_eval = _run_eval(ood_ids, "OOD") if ood_ids else {}

    (out_dir / "evaluation.json").write_text(json.dumps({
        "candidate_policy_id": candidate.policy_id,
        "test": test_eval,
        "ood": ood_eval,
    }, sort_keys=True, indent=2))
    print(f"\nwrote {out_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
