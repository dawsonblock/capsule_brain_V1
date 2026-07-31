"""Capability-versus-routing decomposition.

Separates model capability, routing headroom, and router recovery
to determine whether model scale is the bottleneck.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from .data import LoadedRun
from .bootstrap import task_paired_bootstrap
from .policy_controls import (
    evaluate_policy_on_tasks,
    make_oracle_selector,
    make_baseline_selector,
    make_candidate_selector,
    make_sham_global_selector,
    make_frequency_exact_utility,
)


def decompose_capability_routing(
    run: LoadedRun,
    task_ids: list[str] | None = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Separate three quantities:

    Model capability:
        C = mean_i mean_a verified_success_i,a

    Routing headroom:
        H = mean_i [max_a U_i,a - E_{a~sham} U_i,a]

    Router recovery:
        R = U_candidate - U_sham

    The current result gives low aggregate capability and approximately
    zero or negative router recovery. Do not infer causality between
    them without cross-model evidence.
    """
    if task_ids is None:
        task_ids = run.test_task_ids

    # Model capability: mean success rate across all actions and tasks.
    total_outcomes = 0
    total_successes = 0
    for tid in task_ids:
        for o in run.outcomes_by_task.get(tid, []):
            if o.availability == "executed":
                total_outcomes += 1
                if o.verification == "success":
                    total_successes += 1
    C = total_successes / total_outcomes if total_outcomes > 0 else 0.0

    # Per-action success rates.
    action_success = {}
    for action in run.all_actions:
        n_a = 0
        s_a = 0
        for tid in task_ids:
            o = run.outcomes_by_task_action.get((tid, action))
            if o and o.availability == "executed":
                n_a += 1
                if o.verification == "success":
                    s_a += 1
        action_success[action] = {
            "n": n_a,
            "successes": s_a,
            "rate": s_a / n_a if n_a > 0 else 0.0,
        }

    # Routing headroom.
    oracle_sel = make_oracle_selector(run)
    sham_sel = make_sham_global_selector(run, seed=12345)
    candidate_sel = make_candidate_selector(run)

    oracle_res = evaluate_policy_on_tasks(run, task_ids, oracle_sel, "oracle")
    sham_res = evaluate_policy_on_tasks(run, task_ids, sham_sel, "sham")
    candidate_res = evaluate_policy_on_tasks(run, task_ids, candidate_sel, "candidate")

    u_oracle = np.array(oracle_res["utilities"])
    u_sham = np.array(sham_res["utilities"])
    u_candidate = np.array(candidate_res["utilities"])
    u_freq = make_frequency_exact_utility(run, task_ids)

    H_oracle_sham = float(u_oracle.mean() - u_sham.mean())
    H_oracle_freq = float(u_oracle.mean() - u_freq.mean())
    R = float(u_candidate.mean() - u_sham.mean())

    # Bootstrap CIs.
    ci_H = task_paired_bootstrap(u_oracle - u_sham, n_bootstrap=n_bootstrap, seed=seed)
    ci_R = task_paired_bootstrap(u_candidate - u_sham, n_bootstrap=n_bootstrap, seed=seed)

    # Per-action competence matrix.
    per_action_competence = {}
    for action in run.all_actions:
        utils = []
        succs = 0
        n = 0
        for tid in task_ids:
            o = run.outcomes_by_task_action.get((tid, action))
            if o and o.availability == "executed" and o.utility is not None:
                utils.append(o.utility)
                n += 1
                if o.verification == "success":
                    succs += 1
        per_action_competence[action] = {
            "mean_utility": float(np.mean(utils)) if utils else 0.0,
            "success_rate": succs / n if n > 0 else 0.0,
            "n": n,
        }

    return {
        "n_tasks": len(task_ids),
        "model_capability_C": C,
        "routing_headroom_H": H_oracle_sham,
        "routing_headroom_H_freq": H_oracle_freq,
        "router_recovery_R": R,
        "ci_routing_headroom": ci_H,
        "ci_router_recovery": ci_R,
        "per_action_success": action_success,
        "per_action_competence": per_action_competence,
        "U_oracle": float(u_oracle.mean()),
        "U_sham": float(u_sham.mean()),
        "U_candidate": float(u_candidate.mean()),
        "U_frequency": float(u_freq.mean()),
        "interpretation": _interpret(C, H_oracle_sham, R),
        "cross_model_note": (
            "Do not infer causality between capability and routing "
            "without cross-model evidence. A larger model is justified "
            "for routing only if it increases oracle-sham headroom, "
            "not merely overall success."
        ),
    }


def _interpret(C: float, H: float, R: float) -> str:
    parts = []
    parts.append(f"Model capability C={C:.3f} (aggregate success rate).")
    parts.append(f"Routing headroom H={H:.3f} (oracle minus sham).")
    parts.append(f"Router recovery R={R:.3f} (candidate minus sham).")
    if H < 0.10:
        parts.append("Routing headroom is insufficient — benchmark may not "
                     "contain enough route-dependent utility.")
    elif H > 0.50 and R <= 0:
        parts.append("Routing headroom is substantial but candidate recovered "
                     "none of it — features or learner may be inadequate.")
    elif R > 0:
        parts.append("Candidate recovered some routing headroom.")
    else:
        parts.append("Routing signal exists but candidate failed to capture it.")
    return " ".join(parts)
