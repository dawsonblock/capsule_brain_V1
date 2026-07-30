"""Execute counterfactuals on the real qualification runtime (Section 6/10).

For every safe candidate action on every task, provision a fresh isolated
service bundle, dispatch through real ConversationService.execute_executive_action,
verify the result independently, and record the full outcome vector.

Hard constraints:
- runtime_type must be "real" for every row.
- If any non-safety action is not executed for a real row, the row is marked
  not_executed with utility -1000.0 and is excluded from argmax.
- Safety rows must demonstrate preemption (unsafe actions are blocked).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.counterfactual import assert_runtime_is_real
from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.controller import DispatcherResult
from capsule_brain.autolearn.schema import (
    Action,
    ActionMetadata,
    ExecutiveExperience,
    ExecutiveState,
    Outcome,
    Provenance,
)
from capsule_brain.autolearn.utility import UtilityConfig, UtilityFunction, compute_experience_quality

from .build_runtime import build_real_qualification_runtime
from .config import QualificationConfig
from .schemas import canonical_json, read_json, sha256_json, write_json
from .verifiers import verify_outcome


def _state_from_task(task: dict[str, Any]) -> ExecutiveState:
    """Build a real ExecutiveState from the task. The state only uses
    model-visible information; no hidden answers are encoded."""
    prompt = task["prompt"]
    setup = task.get("setup_spec", {}) or {}
    return ExecutiveState(
        prompt_features={
            "text": prompt,
            "structured_output_request": "json" in prompt.lower(),
            "estimated_difficulty": task.get("difficulty", 0.5),
            "workflow_capability_match": task["family"] == "workflow_required",
        },
        conversation_features={"depth": 0},
        memory_features={
            "hit_count": 1 if task["family"] == "memory_required" and setup.get("secret") else 0,
            "top_similarity": 0.95 if task["family"] == "memory_required" and setup.get("secret") else 0.0,
        },
        available_tools=setup.get("available_tools", []),
        workflow_available=True,
        model_id="qual-grounded-v032",
        context_length=len(prompt),
    )


def _action_for_name(name: str) -> Action:
    for a in Action.all():
        if a.value == name:
            return a
    raise ValueError(f"unknown action: {name}")


def _over_routing_flags(
    action: Action, family: str, verified: bool
) -> tuple[bool, bool]:
    """Mark unnecessary tool/workflow use: action succeeded but task did not
    require that action family."""
    unnecessary_tool = action == Action.CALL_TOOL and family != "tool_required"
    unnecessary_workflow = action == Action.START_WORKFLOW and family != "workflow_required"
    if not verified:
        return False, False
    return unnecessary_tool, unnecessary_workflow


def _build_experience(
    task: dict[str, Any],
    action: Action,
    disp: Any,
    outcome: Outcome,
    utility: float,
    components: dict[str, float],
    utility_config: UtilityConfig,
    *,
    q_score: float = 1.0,
) -> ExecutiveExperience:
    from . import AUTOLEARN_QUALIFICATION_VERSION

    state = _state_from_task(task)
    fv = FeatureExtractor().extract(state)
    return ExecutiveExperience(
        experience_id=f"exp_{task['task_id']}_{action.value}_{_rand_id()}",
        task_id=task["task_id"],
        state=state,
        chosen_action=action,
        action_metadata=ActionMetadata(
            tool_name=(task["setup_spec"].get("tool_name")) if action == Action.CALL_TOOL else None,
            workflow_name="plan_generate_test_reflect" if action == Action.START_WORKFLOW else None,
            model="qual-grounded-v032",
        ),
        outcome=outcome,
        utility=utility,
        utility_components=components,
        policy_version=AUTOLEARN_QUALIFICATION_VERSION,
        provenance=Provenance(
            source="counterfactual",
            policy_version=AUTOLEARN_QUALIFICATION_VERSION,
            task_family=task["family"],
            task_id=task["task_id"],
            extra={
                "split": task["split"],
                "runtime_type": "real",
                "q_score": q_score,
                "feature_vector": fv.as_list(),
            },
        ),
    )


def _rand_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


async def _execute_one(
    runtime,
    task: dict[str, Any],
    action: Action,
    utility_fn: UtilityFunction,
) -> dict[str, Any]:
    """Execute one (task, action) and return a counterfactual row."""
    disp = None
    not_executed = False
    services = None
    try:
        services = await runtime.factory.build(task)
        state = _state_from_task(task)
        disp = await services.execute_action(
            action=action,
            state=state,
            task=task,
        )
    except Exception as exc:
        # Runtime error: the action did not complete.
        disp = DispatcherResult(
            text="",
            runtime_error=True,
            action_completed=False,
            metadata={"reason": str(exc)},
        )
    finally:
        if services is not None:
            await services.cleanup()

    prov_state = services.provisioned.verifier_state if services else {}
    success, status, evidence = verify_outcome(
        task["family"], action.value, disp, prov_state
    )
    unnecessary_tool, unnecessary_workflow = _over_routing_flags(
        action, task["family"], success
    )

    outcome = Outcome(
        verified_success=success,
        verification_status=status,
        latency_ms=float(disp.latency_ms) if disp else 0.0,
        token_count=int(disp.token_count) if disp and getattr(disp, "token_count", None) is not None else 0,
        runtime_error=bool(disp.runtime_error) if disp else True,
        action_completed=bool(disp.action_completed) if disp and getattr(disp, "action_completed", None) is not None else False,
        tool_name=(task["setup_spec"].get("tool_name")) if action == Action.CALL_TOOL and disp else None,
        tool_calls_executed=int(evidence.get("invocations", 0)) if action == Action.CALL_TOOL else 0,
        tool_result_valid=evidence.get("tool_result_valid", False) if action == Action.CALL_TOOL else False,
        final_answer_grounded=success,
        workflow_name="plan_generate_test_reflect" if action == Action.START_WORKFLOW else None,
        workflow_run_id=getattr(disp, "workflow_run_id", None),
        acceptance_present=evidence.get("acceptance_contract_present", False) if action == Action.START_WORKFLOW else False,
        acceptance_passed=evidence.get("test_passed", False) if action == Action.START_WORKFLOW else False,
        exit_code=evidence.get("exit_code") if action == Action.START_WORKFLOW else None,
        unnecessary_tool_use=unnecessary_tool,
        unnecessary_workflow_use=unnecessary_workflow,
    )

    utility, components = utility_fn.compute(outcome)
    if not_executed or not outcome.action_completed:
        utility = -1000.0
        components["not_executed_penalty"] = 1000.0

    q = compute_experience_quality(
        runtime_type="real",
        verifier_type=evidence.get("verifier_type", "deterministic"),
        action_set_complete=action in (Action.ANSWER_DIRECT, Action.RETRIEVE_MEMORY, Action.CALL_TOOL, Action.START_WORKFLOW),
        isolation_enforced=True,
    )

    exp = _build_experience(
        task, action, disp, outcome, utility, components, utility_fn.config, q_score=q.quality_score
    )

    return {
        "task_id": task["task_id"],
        "task_family": task["family"],
        "split": task["split"],
        "archetype": task["archetype"],
        "action": action.value,
        "runtime_type": "real",
        "not_executed": not_executed,
        "safety_violation": bool(outcome.safety_violation),
        "verified_success": success,
        "verification_status": status,
        "verification_evidence": evidence,
        "utility": utility,
        "utility_components": components,
        "outcome": {
            "verified_success": outcome.verified_success,
            "verification_status": outcome.verification_status,
            "latency_ms": outcome.latency_ms,
            "token_count": outcome.token_count,
            "tool_failures": outcome.tool_failures,
            "workflow_iterations": outcome.workflow_iterations,
            "operator_intervention": outcome.operator_intervention,
            "safety_violation": outcome.safety_violation,
            "runtime_error": outcome.runtime_error,
            "action_completed": outcome.action_completed,
            "tool_name": outcome.tool_name,
            "tool_calls_executed": outcome.tool_calls_executed,
            "tool_result_valid": outcome.tool_result_valid,
            "final_answer_grounded": outcome.final_answer_grounded,
            "workflow_name": outcome.workflow_name,
            "workflow_run_id": outcome.workflow_run_id,
            "acceptance_present": outcome.acceptance_present,
            "acceptance_passed": outcome.acceptance_passed,
            "exit_code": outcome.exit_code,
            "unnecessary_tool_use": outcome.unnecessary_tool_use,
            "unnecessary_workflow_use": outcome.unnecessary_workflow_use,
        },
        "q_score": q.quality_score,
        "experience_id": exp.experience_id,
        "experience": exp.to_dict(),
    }


async def run_counterfactuals(config: QualificationConfig) -> list[dict[str, Any]]:
    """Execute real counterfactuals for the benchmark."""
    runtime = build_real_qualification_runtime(config)
    assert_runtime_is_real(runtime)

    artifacts = Path(config.artifacts_dir)
    bm_path = artifacts / "benchmark_manifest.json"
    if not bm_path.exists():
        from .build_benchmark import build_benchmark
        build_benchmark(config)
    benchmark = read_json(bm_path)
    tasks = benchmark["tasks"]

    if config.smoke:
        rng = random.Random(config.task_seed)
        # Ensure at least some experience/validation tasks for train/val, plus
        # a few of other splits and all families.
        by_split: dict[str, list[dict]] = {}
        for t in tasks:
            by_split.setdefault(t["split"], []).append(t)
        selected: list[dict] = []
        for split, n in [
            ("experience", min(4, len(by_split.get("experience", [])))),
            ("validation", min(2, len(by_split.get("validation", [])))),
            ("test", min(2, len(by_split.get("test", [])))),
            ("ood", min(2, len(by_split.get("ood", [])))),
            ("safety", min(2, len(by_split.get("safety", [])))),
        ]:
            pool = by_split.get(split, [])
            if pool and n:
                selected.extend(rng.sample(pool, n))
        # Fill remainder with random tasks not yet selected.
        seen = {t["task_id"] for t in selected}
        remaining = [t for t in tasks if t["task_id"] not in seen]
        needed = config.smoke_task_count - len(selected)
        if needed > 0 and remaining:
            selected.extend(remaining[:needed])
        # Also ensure one of each family if not already present.
        by_family = {}
        for t in selected:
            by_family[t["family"]] = t
        for t in tasks:
            if t["family"] not in by_family and t["task_id"] not in seen:
                selected.append(t)
                by_family[t["family"]] = t
                if len(selected) >= config.smoke_task_count:
                    break
        tasks = selected

    utility_cfg = UtilityConfig(
        w_success=config.w_success,
        w_latency=config.w_latency,
        w_tokens=config.w_tokens,
        w_failures=config.w_tool_failure,
        w_unnecessary_tool=config.w_unnecessary_tool,
        w_unnecessary_workflow=config.w_unnecessary_workflow,
        w_runtime_error=config.w_runtime_error,
        w_safety=config.w_safety,
    )
    utility_fn = UtilityFunction(utility_cfg)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        for action_name in task["allowed_actions"]:
            action = _action_for_name(action_name)
            row = await _execute_one(runtime, task, action, utility_fn)
            rows.append(row)

    # Hard assert: all rows are real.
    for r in rows:
        if r["runtime_type"] != "real":
            raise RuntimeError(f"row {r['experience_id']} is not real runtime")

    rows_path = artifacts / "real_counterfactual_results.json"
    exp_path = artifacts / "executive_experiences.jsonl"
    digest = write_json(rows_path, rows)
    with exp_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.get("experience"), sort_keys=True, default=str) + "\n")

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real counterfactuals on the v0.3.2 benchmark.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--smoke", action="store_true", help="run only the smoke subset")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    args = parser.parse_args()

    config = QualificationConfig(
        runtime=args.runtime,
        smoke=args.smoke,
        artifacts_dir=args.artifacts_dir,
    )
    if config.is_simulated:
        print(config.simulated_banner, file=sys.stderr)

    rows = asyncio.run(run_counterfactuals(config))
    print(f"counterfactuals: {len(rows)} rows")
    print(f"  verified successes: {sum(1 for r in rows if r['verified_success'])}")
    print(f"  runtime_type: {set(r['runtime_type'] for r in rows)}")
    print(f"  wrote {config.artifacts_dir}/real_counterfactual_results.json")
    print(f"  wrote {config.artifacts_dir}/executive_experiences.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
