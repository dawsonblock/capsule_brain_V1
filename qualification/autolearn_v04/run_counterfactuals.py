"""Execute counterfactuals on the real qualification runtime (Section 6/10).

For every safe candidate action on every task, provision a fresh isolated
service bundle, dispatch through real ConversationService.execute_executive_action,
verify the result independently, and record a typed CounterfactualOutcome.

v0.4.0 hard guarantees (Section 5.1):
  * Missing executions are represented by OutcomeAvailability.NOT_EXECUTED,
    NOT by a -1000 utility placeholder.
  * The utility function only scores EXECUTED outcomes. Calling it on a
    non-EXECUTED outcome raises IncompleteEvidenceError.
  * Action execution order is randomized per task and recorded (Section 9.3).
  * Every row records its availability, verification, utility (or None),
    reward components, and execution metadata.

Smoke mode (Section 6): executes a small deterministic subset and evaluates
ONLY the smoke subset. It never produces inferential statistics or a
promotion verdict.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
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

from . import AUTOLEARN_QUALIFICATION_VERSION
from .build_runtime import build_real_qualification_runtime
from .config import QualificationConfig
from .schemas import (
    CounterfactualOutcome,
    IncompleteEvidenceError,
    OutcomeAvailability,
    VerificationOutcome,
    canonical_json,
    read_json,
    sha256_json,
    sha256_text,
    write_json,
)
from .verifiers import verify_outcome


def _state_from_task(task: dict[str, Any]) -> ExecutiveState:
    """Build a real ExecutiveState from the task (model-visible info only)."""
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
        model_id="qual-grounded-v04",
        context_length=len(prompt),
    )


def _action_for_name(name: str) -> Action:
    for a in Action.all():
        if a.value == name:
            return a
    raise ValueError(f"unknown action: {name}")


def _over_routing_flags(action: Action, family: str, verified: bool) -> tuple[bool, bool]:
    unnecessary_tool = action == Action.CALL_TOOL and family != "tool_required"
    unnecessary_workflow = action == Action.START_WORKFLOW and family != "workflow_required"
    if not verified:
        return False, False
    return unnecessary_tool, unnecessary_workflow


def _rand_id() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


async def _execute_one(
    runtime,
    task: dict[str, Any],
    action: Action,
    utility_fn: UtilityFunction,
) -> CounterfactualOutcome:
    """Execute one (task, action) and return a typed CounterfactualOutcome.

    NEVER returns a -1000 placeholder. If the action did not execute, the
    outcome has availability != EXECUTED and utility is None.
    """
    services = None
    error_type: str | None = None
    error_message: str | None = None
    disp: DispatcherResult | None = None

    # Pre-check: CALL_TOOL on a task with no tool registered is a
    # completed-but-failed execution (the model tried to call a tool, there
    # was none). This is NOT an execution error -- it is a real behavioral
    # outcome with a utility penalty for unnecessary tool use.
    setup = task.get("setup_spec", {}) or {}
    if action == Action.CALL_TOOL and not setup.get("tool_name"):
        disp = DispatcherResult(
            text="", runtime_error=False, action_completed=True,
            metadata={"reason": "no tool registered for this task",
                      "error_type": "no_tool_available"},
        )
    else:
        try:
            services = await runtime.factory.build(task)
            state = _state_from_task(task)
            disp = await services.execute_action(
                action=action,
                state=state,
                task=task,
            )
        except TimeoutError as exc:
            error_type = "timeout"
            error_message = str(exc)
            disp = DispatcherResult(
                text="", runtime_error=True, action_completed=False,
                metadata={"reason": str(exc), "error_type": "timeout"},
            )
        except Exception as exc:
            error_type = "execution_error"
            error_message = str(exc)
            disp = DispatcherResult(
                text="", runtime_error=True, action_completed=False,
                metadata={"reason": str(exc), "error_type": "execution_error"},
            )
        finally:
            if services is not None:
                await services.cleanup()

    # Determine availability from the dispatch result.
    if disp is None or getattr(disp, "runtime_error", False):
        availability = OutcomeAvailability.EXECUTION_ERROR
    elif not getattr(disp, "action_completed", False):
        # The action was dispatched but did not complete (e.g. safety preemption
        # blocked it, or it was not run). This is NOT_EXECUTED, not a failure.
        availability = OutcomeAvailability.NOT_EXECUTED
    else:
        availability = OutcomeAvailability.EXECUTED

    prov_state = services.provisioned.verifier_state if services else {}
    verification, status, evidence = verify_outcome(
        task["family"], action.value, disp, prov_state
    )

    # Build the legacy Outcome for utility computation (only if EXECUTED).
    unnecessary_tool, unnecessary_workflow = _over_routing_flags(
        action, task["family"], verification is VerificationOutcome.SUCCESS
    )
    outcome = Outcome(
        verified_success=verification is VerificationOutcome.SUCCESS,
        verification_status=status,
        latency_ms=float(disp.latency_ms) if disp else 0.0,
        token_count=int(disp.token_count) if disp and getattr(disp, "token_count", None) is not None else 0,
        runtime_error=bool(disp.runtime_error) if disp else True,
        action_completed=bool(disp.action_completed) if disp and getattr(disp, "action_completed", None) is not None else False,
        tool_name=(task["setup_spec"].get("tool_name")) if action == Action.CALL_TOOL and disp else None,
        tool_calls_executed=int(evidence.get("invocations", 0)) if action == Action.CALL_TOOL else 0,
        tool_result_valid=evidence.get("tool_result_valid", False) if action == Action.CALL_TOOL else False,
        final_answer_grounded=verification is VerificationOutcome.SUCCESS,
        workflow_name="plan_generate_test_reflect" if action == Action.START_WORKFLOW else None,
        workflow_run_id=getattr(disp, "workflow_run_id", None),
        acceptance_present=evidence.get("acceptance_contract_present", False) if action == Action.START_WORKFLOW else False,
        acceptance_passed=evidence.get("test_passed", False) if action == Action.START_WORKFLOW else False,
        exit_code=evidence.get("exit_code") if action == Action.START_WORKFLOW else None,
        unnecessary_tool_use=unnecessary_tool,
        unnecessary_workflow_use=unnecessary_workflow,
    )

    # Section 5.1: utility is ONLY computed for EXECUTED outcomes.
    utility: float | None = None
    components: dict[str, float] = {}
    if availability is OutcomeAvailability.EXECUTED:
        utility, components = utility_fn.compute(outcome)
    # NOT_EXECUTED / EXECUTION_ERROR / TIMEOUT / UNVERIFIABLE => utility is None.
    # NO -1000 placeholder is ever assigned.

    q = compute_experience_quality(
        runtime_type="real",
        verifier_type=evidence.get("verifier_type", "deterministic"),
        action_set_complete=action in (Action.ANSWER_DIRECT, Action.RETRIEVE_MEMORY, Action.CALL_TOOL, Action.START_WORKFLOW),
        isolation_enforced=True,
    )

    artifact_digest = sha256_text(canonical_json({
        "task_id": task["task_id"], "action": action.value,
        "availability": availability.value, "verification": verification.value if verification else None,
        "text": getattr(disp, "text", "") if disp else "",
    }))

    cf_outcome = CounterfactualOutcome(
        task_id=task["task_id"],
        action_id=action.value,
        availability=availability,
        verification=verification,
        utility=utility,
        reward_components=components,
        execution_metadata={
            "runtime_type": "real",
            "q_score": q.quality_score,
            "latency_ms": float(disp.latency_ms) if disp else 0.0,
            "token_count": int(disp.token_count) if disp and getattr(disp, "token_count", None) is not None else 0,
            "verification_status": status,
            "verification_evidence": evidence,
            "outcome": _outcome_to_dict(outcome),
        },
        artifact_digest=artifact_digest,
        error_type=error_type,
        error_message=error_message,
    )
    cf_outcome.validate()
    return cf_outcome


def _outcome_to_dict(outcome: Outcome) -> dict[str, Any]:
    return {
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
    }


def _build_experience(
    task: dict[str, Any],
    action: Action,
    disp: Any,
    outcome: Outcome,
    utility: float | None,
    components: dict[str, float],
    utility_config: UtilityConfig,
    *,
    q_score: float = 1.0,
) -> ExecutiveExperience:
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
            model="qual-grounded-v04",
        ),
        outcome=outcome,
        utility=utility if utility is not None else 0.0,
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


def _select_smoke_subset(tasks: list[dict[str, Any]], config: QualificationConfig) -> list[dict[str, Any]]:
    """Section 6.1: select a deterministic smoke subset -- at least one task
    from each implemented task family, with all allowed actions executed."""
    rng = random.Random(config.task_seed)
    by_split: dict[str, list[dict]] = {}
    for t in tasks:
        by_split.setdefault(t["split"], []).append(t)
    selected: list[dict] = []
    # Take a few from each split for wiring.
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
    seen = {t["task_id"] for t in selected}
    remaining = [t for t in tasks if t["task_id"] not in seen]
    needed = config.smoke_task_count - len(selected)
    if needed > 0 and remaining:
        selected.extend(remaining[:needed])
    # Ensure one of each family.
    by_family: dict[str, dict] = {}
    for t in selected:
        by_family[t["family"]] = t
    for t in tasks:
        if t["family"] not in by_family and t["task_id"] not in seen:
            selected.append(t)
            by_family[t["family"]] = t
            seen.add(t["task_id"])
            if len(selected) >= config.smoke_task_count:
                break
    return selected


async def run_counterfactuals(config: QualificationConfig) -> tuple[list[CounterfactualOutcome], list[dict[str, Any]]]:
    """Execute real counterfactuals for the benchmark.

    Returns (typed_outcomes, legacy_rows) where legacy_rows preserve the
    v0.3.2-compatible dict shape for downstream consumers, but with the
    critical fix that missing executions have utility=None (not -1000).
    """
    runtime = build_real_qualification_runtime(config)
    assert_runtime_is_real(runtime)

    artifacts = Path(config.artifacts_dir)
    bm_path = artifacts / "benchmark_manifest.json"
    if not bm_path.exists():
        from .build_benchmark import build_benchmark
        tasks, bm, sm = build_benchmark(config)
        write_json(bm_path, bm)
        write_json(artifacts / "split_manifest.json", sm)
    benchmark = read_json(bm_path)
    tasks = benchmark["tasks"]

    if config.is_smoke:
        tasks = _select_smoke_subset(tasks, config)

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

    outcomes: list[CounterfactualOutcome] = []
    action_order_log: list[dict[str, Any]] = []

    for task in tasks:
        allowed = list(task["allowed_actions"])
        # Section 9.3: randomize action execution order per task and record it.
        if config.randomize_action_order:
            order_rng = random.Random(hash((task["task_id"], config.task_seed)) & 0xFFFFFFFF)
            order_rng.shuffle(allowed)
        action_order_log.append({
            "task_id": task["task_id"],
            "action_order": list(allowed),
            "randomized": config.randomize_action_order,
        })
        for action_name in allowed:
            action = _action_for_name(action_name)
            cf_outcome = await _execute_one(runtime, task, action, utility_fn)
            outcomes.append(cf_outcome)

    # Hard assert: all rows are real runtime.
    for o in outcomes:
        if o.execution_metadata.get("runtime_type") != "real":
            raise RuntimeError(f"outcome {o.task_id}/{o.action_id} is not real runtime")

    # Write typed outcomes (the canonical v0.4.0 artifact).
    typed_path = artifacts / "counterfactual_outcomes.json"
    write_json(typed_path, [o.to_dict() for o in outcomes])

    # Write legacy-compatible rows for downstream consumers.
    legacy_rows: list[dict[str, Any]] = []
    exp_records: list[dict[str, Any]] = []
    for o in outcomes:
        legacy_rows.append(_outcome_to_legacy_row(o))
        # Build an ExecutiveExperience only for EXECUTED outcomes.
        if o.availability is OutcomeAvailability.EXECUTED:
            task = next(t for t in tasks if t["task_id"] == o.task_id)
            action = _action_for_name(o.action_id)
            outcome_obj = Outcome(
                verified_success=o.verification is VerificationOutcome.SUCCESS,
                verification_status=o.execution_metadata.get("verification_status", "skip"),
                latency_ms=o.execution_metadata.get("latency_ms", 0.0),
                token_count=o.execution_metadata.get("token_count", 0),
                runtime_error=False,
                action_completed=True,
                tool_name=task["setup_spec"].get("tool_name") if action == Action.CALL_TOOL else None,
                tool_calls_executed=o.execution_metadata.get("verification_evidence", {}).get("invocations", 0) if action == Action.CALL_TOOL else 0,
                tool_result_valid=o.execution_metadata.get("verification_evidence", {}).get("tool_result_valid", False) if action == Action.CALL_TOOL else False,
                final_answer_grounded=o.verification is VerificationOutcome.SUCCESS,
                workflow_name="plan_generate_test_reflect" if action == Action.START_WORKFLOW else None,
                acceptance_present=o.execution_metadata.get("verification_evidence", {}).get("acceptance_contract_present", False) if action == Action.START_WORKFLOW else False,
                acceptance_passed=o.execution_metadata.get("verification_evidence", {}).get("test_passed", False) if action == Action.START_WORKFLOW else False,
                exit_code=o.execution_metadata.get("verification_evidence", {}).get("exit_code") if action == Action.START_WORKFLOW else None,
                unnecessary_tool_use=o.execution_metadata.get("outcome", {}).get("unnecessary_tool_use", False),
                unnecessary_workflow_use=o.execution_metadata.get("outcome", {}).get("unnecessary_workflow_use", False),
            )
            exp = _build_experience(
                task, action, None, outcome_obj, o.utility, dict(o.reward_components),
                utility_fn.config, q_score=o.execution_metadata.get("q_score", 1.0),
            )
            exp_records.append(exp.to_dict())

    write_json(artifacts / "real_counterfactual_results.json", legacy_rows)
    write_json(artifacts / "action_order_log.json", action_order_log)
    with (artifacts / "executive_experiences.jsonl").open("w", encoding="utf-8") as f:
        for rec in exp_records:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    return outcomes, legacy_rows


def _outcome_to_legacy_row(o: CounterfactualOutcome) -> dict[str, Any]:
    """Convert a typed outcome to the legacy dict shape, preserving the
    critical fix: utility is None (not -1000) for non-EXECUTED outcomes."""
    return {
        "schema_version": "counterfactual-row/2",
        "protocol_version": "0.4.0",
        "task_id": o.task_id,
        "action": o.action_id,
        "availability": o.availability.value,
        "verification": o.verification.value if o.verification else None,
        "runtime_type": o.execution_metadata.get("runtime_type", "real"),
        "not_executed": o.availability is not OutcomeAvailability.EXECUTED,
        "verified_success": o.verification is VerificationOutcome.SUCCESS,
        "verification_status": o.execution_metadata.get("verification_status", "skip"),
        "verification_evidence": o.execution_metadata.get("verification_evidence", {}),
        "utility": o.utility,  # None for non-executed -- NEVER -1000
        "utility_components": dict(o.reward_components),
        "q_score": o.execution_metadata.get("q_score", 1.0),
        "artifact_digest": o.artifact_digest,
        "error_type": o.error_type,
        "error_message": o.error_message,
        "outcome": o.execution_metadata.get("outcome", {}),
    }


def load_typed_outcomes(config: QualificationConfig) -> list[CounterfactualOutcome]:
    """Load typed counterfactual outcomes from disk."""
    artifacts = Path(config.artifacts_dir)
    path = artifacts / "counterfactual_outcomes.json"
    if not path.exists():
        # Fall back to legacy rows.
        legacy_path = artifacts / "real_counterfactual_results.json"
        if not legacy_path.exists():
            return []
        rows = read_json(legacy_path)
        return [_legacy_row_to_outcome(r) for r in rows]
    data = read_json(path)
    return [CounterfactualOutcome.from_dict(d) for d in data]


def _legacy_row_to_outcome(r: dict[str, Any]) -> CounterfactualOutcome:
    avail = OutcomeAvailability(r.get("availability", "not_executed" if r.get("not_executed") else "executed"))
    verif_raw = r.get("verification")
    if verif_raw is None:
        verif_raw = "success" if r.get("verified_success") else "failure"
    verification = VerificationOutcome(verif_raw)
    utility = r.get("utility")
    return CounterfactualOutcome(
        task_id=r["task_id"],
        action_id=r["action"],
        availability=avail,
        verification=verification,
        utility=utility,
        reward_components=dict(r.get("utility_components", {}) or {}),
        execution_metadata={
            "runtime_type": r.get("runtime_type", "real"),
            "q_score": r.get("q_score", 1.0),
            "verification_status": r.get("verification_status", "skip"),
            "verification_evidence": r.get("verification_evidence", {}),
            "outcome": r.get("outcome", {}),
        },
        artifact_digest=r.get("artifact_digest"),
        error_type=r.get("error_type"),
        error_message=r.get("error_message"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real counterfactuals on the v0.4.0 benchmark.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "qualification"], default="smoke")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    parser.add_argument("--provider", default="qual_grounded")
    parser.add_argument("--model", default="qual-grounded-v04")
    args = parser.parse_args()
    config = QualificationConfig(
        mode=args.mode, runtime=args.runtime, provider=args.provider,
        model=args.model, artifacts_dir=args.artifacts_dir,
    )
    if config.is_simulated:
        print(config.simulated_banner, file=sys.stderr)
    if config.is_infrastructure_provider:
        print(config.infrastructure_provider_banner, file=sys.stderr)
    outcomes, rows = asyncio.run(run_counterfactuals(config))
    n_executed = sum(1 for o in outcomes if o.availability is OutcomeAvailability.EXECUTED)
    n_success = sum(1 for o in outcomes if o.verification is VerificationOutcome.SUCCESS)
    print(f"counterfactuals: {len(outcomes)} outcomes")
    print(f"  executed: {n_executed}")
    print(f"  verified successes: {n_success}")
    print(f"  not_executed: {sum(1 for o in outcomes if o.availability is OutcomeAvailability.NOT_EXECUTED)}")
    print(f"  execution_errors: {sum(1 for o in outcomes if o.availability is OutcomeAvailability.EXECUTION_ERROR)}")
    print(f"  wrote {config.artifacts_dir}/counterfactual_outcomes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
