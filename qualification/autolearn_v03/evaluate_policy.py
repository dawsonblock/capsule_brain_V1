"""Evaluate the candidate policy against the baseline on held-out tasks (v0.3).

Section 22: Real paired evaluation on held-out tasks. For each test task,
both the baseline and candidate select an action; the action is executed
through the real runtime; the verifier judges the outcome; the utility is
computed. The paired delta is the candidate utility minus the baseline
utility per task.

Section 20: Calibration metrics (ECE, Brier, coverage) are computed from
the candidate's (confidence, correct) pairs.

v2.16.0 Priority 2: Supports ``--runtime simulated|real``. The official
qualification pipeline must require ``runtime == real``. Simulation remains
available only for unit tests.

Writes evaluation.json with:
- test: {baseline_metrics, candidate_metrics, mean_delta, wins/ties/losses, v2_metrics}
- ood: {baseline_metrics, candidate_metrics, mean_delta, wins/ties/losses, v2_metrics}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from capsule_brain.autolearn.baseline import BaselinePolicyV2
from capsule_brain.autolearn.calibration import compute_calibration
from capsule_brain.autolearn.counterfactual import (
    CounterfactualRuntime,
    RealCounterfactualRuntime,
    SimulatedCounterfactualRuntime,
    assert_runtime_is_real,
    run_counterfactuals,
)
from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.registry import PolicyRegistry
from capsule_brain.autolearn.schema import Action
from capsule_brain.autolearn.utility import UtilityConfig, UtilityFunction

from tasks import build_all_tasks, build_split_assignments


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _evaluate_split(
    tasks: list,
    baseline: BaselinePolicyV2,
    candidate: LearnedPolicy,
    extractor: FeatureExtractor,
    runtime: CounterfactualRuntime,
    utility_config: UtilityConfig,
) -> dict:
    """Evaluate baseline vs candidate on a set of tasks."""
    ufn = UtilityFunction(utility_config)
    import asyncio

    baseline_utilities = []
    candidate_utilities = []
    baseline_successes = []
    candidate_successes = []
    candidate_confidences = []
    candidate_correct = []
    candidate_abstained = []
    tool_selections = 0
    tool_correct = 0
    workflow_selections = 0
    workflow_correct = 0
    safety_violations_b = 0
    safety_violations_c = 0
    over_routing = 0
    wins = 0
    ties = 0
    losses = 0

    async def _run():
        nonlocal wins, ties, losses, tool_selections, tool_correct
        nonlocal workflow_selections, workflow_correct, safety_violations_b
        nonlocal safety_violations_c, over_routing

        for task in tasks:
            # Baseline decision.
            b_dec = baseline.select_action(task.state)
            b_action = b_dec.action

            # Candidate decision.
            c_dec = candidate.select_action(task.state, extractor=extractor)
            c_action = c_dec.action
            c_conf = c_dec.confidence
            c_abstained = c_dec.abstained

            # Execute both through the runtime.
            b_result = await runtime.execute(task, b_action)
            c_result = await runtime.execute(task, c_action)

            # Compute utilities.
            if b_result.not_executed:
                b_util = -1000.0
            else:
                b_util, _ = ufn.compute(b_result.outcome)

            if c_result.not_executed:
                c_util = -1000.0
            else:
                c_util, _ = ufn.compute(c_result.outcome)

            baseline_utilities.append(b_util)
            candidate_utilities.append(c_util)
            baseline_successes.append(b_result.outcome.verified_success)
            candidate_successes.append(c_result.outcome.verified_success)
            candidate_confidences.append(c_conf)
            candidate_correct.append(c_result.outcome.verified_success)
            candidate_abstained.append(c_abstained)

            if b_result.outcome.safety_violation:
                safety_violations_b += 1
            if c_result.outcome.safety_violation:
                safety_violations_c += 1

            # Tool precision/recall.
            if c_action == Action.CALL_TOOL:
                tool_selections += 1
                if c_result.outcome.verified_success:
                    tool_correct += 1
            if c_action == Action.START_WORKFLOW:
                workflow_selections += 1
                if c_result.outcome.verified_success:
                    workflow_correct += 1

            # Over-routing: correct but unnecessary.
            if c_result.outcome.verified_success and c_action != task.expected_action:
                over_routing += 1

            # Win/tie/loss.
            delta = c_util - b_util
            if delta > 0.01:
                wins += 1
            elif delta < -0.01:
                losses += 1
            else:
                ties += 1

    asyncio.run(_run())

    n = len(tasks)
    mean_b = sum(baseline_utilities) / n if n else 0.0
    mean_c = sum(candidate_utilities) / n if n else 0.0
    mean_delta = mean_c - mean_b

    # v2.16.0 Priority 5: Real paired bootstrap (10,000 samples) with BCa.
    # Replaces the fake normal-approximation interval.
    from capsule_brain.autolearn.statistics import (
        compute_qualification_statistics,
    )
    deltas = [c - b for c, b in zip(candidate_utilities, baseline_utilities)]
    stats = compute_qualification_statistics(
        candidate_utilities, baseline_utilities,
        n_bootstrap=10_000, epsilon=0.0, seed=42,
    )
    ci_low = stats.bootstrap.ci_low
    ci_high = stats.bootstrap.ci_high

    # Calibration.
    pairs = list(zip(candidate_confidences, candidate_correct))
    cal = compute_calibration(pairs, abstained=candidate_abstained)

    # Tool precision.
    tool_precision = tool_correct / tool_selections if tool_selections > 0 else 0.0
    # Tool recall: how many tool-required tasks did the candidate route to CALL_TOOL?
    tool_required = [t for t in tasks if t.expected_action == Action.CALL_TOOL]
    tool_recall = (
        sum(1 for t in tool_required if candidate.select_action(t.state, extractor=extractor).action == Action.CALL_TOOL)
        / len(tool_required) if tool_required else 0.0
    )

    # Workflow routing accuracy.
    workflow_tasks = [t for t in tasks if t.expected_action == Action.START_WORKFLOW]
    workflow_routing = (
        sum(1 for t in workflow_tasks if candidate.select_action(t.state, extractor=extractor).action == Action.START_WORKFLOW)
        / len(workflow_tasks) if workflow_tasks else 0.0
    )

    over_routing_rate = over_routing / n if n else 0.0

    return {
        "baseline_metrics": {
            "verified_success_rate": sum(baseline_successes) / n if n else 0.0,
            "mean_utility": mean_b,
            "safety_violations": safety_violations_b,
            "n": n,
        },
        "candidate_metrics": {
            "verified_success_rate": sum(candidate_successes) / n if n else 0.0,
            "mean_utility": mean_c,
            "tool_precision": tool_precision,
            "tool_recall": tool_recall,
            "safety_violations": safety_violations_c,
            "n": n,
        },
        "mean_delta": mean_delta,
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "statistics": stats.to_dict(),
        "v2_metrics": {
            "candidate_workflow_routing_accuracy": workflow_routing,
            "candidate_over_routing_rate": over_routing_rate,
            "candidate_brier_score": cal.brier_score,
            "candidate_ece": cal.ece,
            "candidate_abstention_rate": cal.abstention_rate,
            "candidate_accuracy_at_coverage": cal.accuracy_at_coverage,
            "calibration": cal.to_dict(),
        },
    }


def _build_runtime(runtime_arg: str) -> CounterfactualRuntime:
    """Build the evaluation runtime based on the --runtime flag.

    v2.16.0 Priority 2: ``--runtime real`` uses the real
    QualificationServiceFactory. ``--runtime simulated`` uses the
    SimulatedCounterfactualRuntime (for unit tests only).
    """
    if runtime_arg == "simulated":
        print("WARNING: using simulated runtime — NOT for official qualification")
        return SimulatedCounterfactualRuntime()
    elif runtime_arg == "real":
        from capsule_brain.autolearn.qualification_runtime import (
            QualificationServiceFactory,
        )
        factory = QualificationServiceFactory()
        return RealCounterfactualRuntime(service_factory=factory)
    else:
        print(f"ERROR: unknown runtime '{runtime_arg}'. Use 'real' or 'simulated'.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate policy (v0.3)")
    parser.add_argument(
        "--runtime",
        choices=["real", "simulated"],
        default="real",
        help="Evaluation runtime: 'real' (default, official qualification) "
             "or 'simulated' (unit tests only).",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    candidate_id = (out_dir / "candidate_policy_id.txt").read_text().strip()
    registry = PolicyRegistry(out_dir / "policies")
    candidate = registry.load_policy(candidate_id)

    tasks = build_all_tasks()
    assignments = build_split_assignments(tasks)
    test_tasks = [t for t in tasks if assignments.get(t.task_id) == "test"]
    ood_tasks = [t for t in tasks if assignments.get(t.task_id) == "ood"]

    print(f"test tasks: {len(test_tasks)}, ood tasks: {len(ood_tasks)}")

    extractor = FeatureExtractor()
    baseline = BaselinePolicyV2()
    runtime = _build_runtime(args.runtime)

    # v2.16.0 Priority 2: assert runtime is real for official qualification.
    if args.runtime == "real":
        try:
            assert_runtime_is_real(runtime)
        except RuntimeError:
            print("ERROR: official qualification requires runtime_type='real'")
            sys.exit(1)

    utility_config = UtilityConfig()

    print(f"runtime_type: {runtime.runtime_type}")
    test_eval = _evaluate_split(test_tasks, baseline, candidate, extractor, runtime, utility_config)
    ood_eval = _evaluate_split(ood_tasks, baseline, candidate, extractor, runtime, utility_config)

    output = {
        "candidate_policy_id": candidate.policy_id,
        "runtime_type": runtime.runtime_type,
        "test": test_eval,
        "ood": ood_eval,
    }
    text = json.dumps(output, sort_keys=True, indent=2)
    (out_dir / "evaluation.json").write_text(text)
    print(f"wrote {out_dir / 'evaluation.json'}")
    print(f"evaluation SHA-256: {_compute_sha256(text)}")
    print(f"test mean_delta: {test_eval['mean_delta']:.4f}")
    print(f"ood mean_delta: {ood_eval['mean_delta']:.4f}")


if __name__ == "__main__":
    main()
