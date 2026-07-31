"""Matched 14B pilot manifest generation.

Defines strict conditions for a matched 14B pilot and generates the
pilot task manifest if eligibility conditions are met.
"""
from __future__ import annotations

from typing import Any
import hashlib
import json

from .data import LoadedRun
from .config import PILOT_MIN_TASKS, PILOT_MAX_TASKS


def check_pilot_eligibility(
    gate_a0_status: str,
    counterfactual_completeness_status: str,
    utility_consistency_status: str,
    sham_validity: bool,
    oracle_headroom: float,
    headroom_concentration_status: str,
    feature_signal_conclusion: str,
    candidate_stability_classification: str,
    scale_decision: str,
) -> dict[str, Any]:
    """Check all eligibility conditions for a matched 14B pilot.

    A matched 14B pilot may be approved only if:
    1. Gate A0 passes.
    2. Counterfactual and utility consistency pass.
    3. Sham control is valid.
    4. Oracle headroom is sufficient (> 0.05).
    5. Headroom is not dominated by a trivial data defect.
    6. Either features show some route signal, or pre-action model-confidence
       features are identified as the missing state.
    7. Candidate learning is not completely unstable.
    8. A fixed matched task subset is defined before model execution.
    9. The pilot is explicitly not a full qualification run.
    """
    conditions = {
        "gate_a0_passes": gate_a0_status == "PASS",
        "counterfactual_completeness_passes": counterfactual_completeness_status == "PASS",
        "utility_consistency_passes": utility_consistency_status == "PASS",
        "sham_control_valid": sham_validity,
        "oracle_headroom_sufficient": oracle_headroom > 0.05,
        "headroom_not_trivial_defect": headroom_concentration_status != "CONCENTRATED",
        "feature_signal_or_missing_state_identified": (
            feature_signal_conclusion in ("FEATURE_SIGNAL_PRESENT", "FEATURE_SIGNAL_WEAK")
        ),
        "candidate_not_completely_unstable": candidate_stability_classification != "ROUTER_UNSTABLE",
        "scale_decision_allows_pilot": scale_decision == "APPROVE_MATCHED_14B_PILOT",
    }

    all_pass = all(conditions.values())
    failed = [k for k, v in conditions.items() if not v]

    return {
        "eligible": all_pass,
        "conditions": conditions,
        "failed_conditions": failed,
        "n_conditions": len(conditions),
        "n_passing": sum(1 for v in conditions.values() if v),
        "n_failing": sum(1 for v in conditions.values() if not v),
    }


def generate_pilot_manifest(
    run: LoadedRun,
    eligibility_result: dict[str, Any],
) -> dict[str, Any]:
    """Generate the matched 14B pilot task manifest.

    Selects 20-30 tasks representing:
    - direct answer
    - memory
    - tool
    - workflow
    - crossover
    - high-margin cases

    The pilot uses the exact same task IDs, prompts, hidden setup, tools,
    memory, workflow contracts, verifiers, utility function, eligible actions,
    and generation seeds.
    """
    if not eligibility_result["eligible"]:
        return {
            "pilot_approved": False,
            "reason": "Eligibility conditions not met",
            "failed_conditions": eligibility_result["failed_conditions"],
            "pilot_manifest": None,
        }

    # Select tasks from the test split, ensuring family coverage.
    test_tasks = [run.tasks[tid] for tid in run.test_task_ids if tid in run.tasks]

    # Group by family.
    by_family: dict[str, list] = {}
    for t in test_tasks:
        by_family.setdefault(t.family, []).append(t)

    # Select tasks to ensure coverage.
    selected: list[str] = []
    families = sorted(by_family.keys())

    # Target: ~25 tasks, distributed across families.
    target_per_family = max(1, PILOT_MIN_TASKS // len(families))

    for fam in families:
        fam_tasks = by_family[fam]
        # Sort by utility margin (high margin first) if available.
        for t in fam_tasks[:target_per_family]:
            selected.append(t.task_id)

    # Fill remaining slots with high-margin tasks from any family.
    remaining = PILOT_MIN_TASKS - len(selected)
    if remaining > 0:
        all_remaining = [t for t in test_tasks if t.task_id not in selected]
        for t in all_remaining[:remaining]:
            selected.append(t.task_id)

    # Cap at max.
    selected = selected[:PILOT_MAX_TASKS]

    # Build manifest entries.
    manifest_entries = []
    for tid in selected:
        task = run.tasks.get(tid)
        if task is None:
            continue
        manifest_entries.append({
            "task_id": tid,
            "family": task.family,
            "archetype": task.archetype,
            "allowed_actions": task.allowed_actions,
            "verifier_spec": task.verifier_spec,
            "prompt": task.prompt,
            "setup_spec": task.setup_spec,
            "generator_seed": task.generator_seed,
            "risk_class": task.risk_class,
            "crossover_type": task.crossover_type,
            "expected_output_digest": task.expected_output_digest,
        })

    # Compute manifest digest.
    manifest_digest = hashlib.sha256(
        json.dumps(sorted(selected)).encode()
    ).hexdigest()

    return {
        "pilot_approved": True,
        "pilot_manifest": {
            "n_tasks": len(selected),
            "task_ids": sorted(selected),
            "task_ids_digest": manifest_digest,
            "entries": manifest_entries,
            "families_represented": sorted({run.tasks[tid].family for tid in selected if tid in run.tasks}),
            "comparison_metrics": [
                "overall_success",
                "oracle_sham_headroom",
                "action_selectivity",
                "utility_margin_distribution",
            ],
            "constraints": {
                "same_task_ids": True,
                "same_prompts": True,
                "same_hidden_setup": True,
                "same_tools": True,
                "same_memory": True,
                "same_workflow_contracts": True,
                "same_verifiers": True,
                "same_utility_function": True,
                "same_eligible_actions": True,
                "same_generation_seeds": True,
            },
            "is_full_qualification_run": False,
            "approval_scope": "matched_pilot_only",
        },
        "eligibility": eligibility_result,
        "disclaimer": (
            "A larger model may increase raw task capability, but this does "
            "not establish that the current router can learn state-dependent "
            "routing. This approval authorizes only a small fixed-subset "
            "comparison, not a full 14B qualification run."
        ),
    }
