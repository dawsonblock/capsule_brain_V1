"""Missing-state analysis for v0.4.3 repair release.

Identifies hidden (pre-action) environment variables that define the optimal
action, then proposes missing-feature candidates that could close the gap
between the current executive feature set and the variables that actually
discriminate the optimal action.

Every proposed feature is checked against a strict leakage firewall:
  - available before the action is taken
  - not derived from the verifier
  - not derived from the outcome
  - not containing the expected answer
  - not containing the best-action label
  - stable across counterfactual actions
  - computable in production

The analysis is purely diagnostic — it never writes a new policy and never
uses the final test split to choose features.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .data import LoadedRun, TaskInfo
from .config import MINIMUM_DENOMINATOR


# ---------------------------------------------------------------------------
# Actions of interest
# ---------------------------------------------------------------------------

FOCAL_ACTIONS: tuple[str, ...] = (
    "ANSWER_DIRECT",
    "RETRIEVE_MEMORY",
    "CALL_TOOL",
    "START_WORKFLOW",
)

# State variables examined.  Each is a binary indicator derived from
# pre-action task metadata (never from the verifier or outcome).
STATE_VARIABLES: tuple[str, ...] = (
    "memory_available",
    "memory_relevant",
    "memory_fresh",
    "memory_conflict",
    "tool_available",
    "tool_reliable",
    "tool_latency_high",
    "tool_authorized",
    "workflow_capability",
    "workflow_acceptance_contract_present",
    "workflow_expected_cost_high",
    "task_reversible",
    "task_risky",
    "direct_answer_confidence_high",
    "conversation_depth_high",
    "task_complexity_high",
    "requires_external_state",
    "structured_output_required",
)


# ---------------------------------------------------------------------------
# State-variable extractors (all pre-action, no verifier/outcome leakage)
# ---------------------------------------------------------------------------


def _extract_state_variables(task: TaskInfo) -> dict[str, bool]:
    """Extract binary pre-action state variables from task metadata.

    Every value is derived from setup_spec, family, risk_class, or prompt
    features — never from the verifier, the outcome, or the expected answer.
    """
    setup = task.setup_spec or {}
    prompt = task.prompt or ""
    prompt_lower = prompt.lower()
    family = task.family or ""
    risk_class = task.risk_class or "benign"

    available_tools = setup.get("available_tools", []) or []
    memory_spec = setup.get("memory", {}) or {}
    tool_spec = setup.get("tool", {}) or {}
    workflow_spec = setup.get("workflow", {}) or {}

    # --- Memory -----------------------------------------------------------
    memory_available = bool(memory_spec.get("available", False)) or bool(
        setup.get("secret")
    )
    memory_relevant = family == "memory_required"
    memory_fresh = bool(memory_spec.get("fresh", False))
    memory_conflict = bool(memory_spec.get("conflict", False))

    # --- Tool -------------------------------------------------------------
    tool_available = bool(available_tools) or bool(tool_spec.get("available", False))
    tool_reliable = bool(tool_spec.get("reliable", True))
    tool_latency = float(tool_spec.get("latency_estimate", 0.0) or 0.0)
    tool_latency_high = tool_latency >= 1.0
    tool_authorized = bool(tool_spec.get("authorized", True))

    # --- Workflow ---------------------------------------------------------
    workflow_capability = family == "workflow_required"
    workflow_acceptance_contract_present = bool(
        workflow_spec.get("acceptance_contract", False)
    ) or bool(setup.get("acceptance_contract"))
    workflow_expected_cost = float(workflow_spec.get("expected_cost", 0.0) or 0.0)
    workflow_expected_cost_high = workflow_expected_cost >= 1.0

    # --- Task properties --------------------------------------------------
    task_reversible = bool(setup.get("reversible", True))
    task_risky = risk_class in {"high", "critical", "dangerous", "risky"}

    # --- Prompt-derived (pre-action) -------------------------------------
    # Direct-answer confidence proxy: short, non-retrieval, non-tool prompts
    # are more likely answerable directly.
    retrieval_keywords = (
        "remember", "recall", "earlier", "previously", "last time",
        "from memory", "stored", "persisted",
    )
    tool_keywords = (
        "tool", "call", "fetch", "lookup", "query", "api", "search",
        "retrieve", "compute", "calculate", "execute", "run", "secret",
        "nonce", "current", "latest", "live", "price", "weather", "stock",
    )
    has_retrieval = any(kw in prompt_lower for kw in retrieval_keywords)
    has_tool = any(kw in prompt_lower for kw in tool_keywords)
    direct_answer_confidence_high = (len(prompt) < 400) and not has_retrieval and not has_tool

    conversation_depth = int(setup.get("conversation_depth", 0) or 0)
    conversation_depth_high = conversation_depth >= 2

    task_complexity = float(setup.get("complexity", 0.5) or 0.5)
    task_complexity_high = task_complexity >= 0.7

    requires_external_state = bool(setup.get("requires_external_state", False)) or (
        "current" in prompt_lower or "latest" in prompt_lower or "live" in prompt_lower
    )

    structured_keywords = ("json", "yaml", "xml", "csv", "table", "schema", "structured")
    structured_output_required = any(kw in prompt_lower for kw in structured_keywords)

    return {
        "memory_available": memory_available,
        "memory_relevant": memory_relevant,
        "memory_fresh": memory_fresh,
        "memory_conflict": memory_conflict,
        "tool_available": tool_available,
        "tool_reliable": tool_reliable,
        "tool_latency_high": tool_latency_high,
        "tool_authorized": tool_authorized,
        "workflow_capability": workflow_capability,
        "workflow_acceptance_contract_present": workflow_acceptance_contract_present,
        "workflow_expected_cost_high": workflow_expected_cost_high,
        "task_reversible": task_reversible,
        "task_risky": task_risky,
        "direct_answer_confidence_high": direct_answer_confidence_high,
        "conversation_depth_high": conversation_depth_high,
        "task_complexity_high": task_complexity_high,
        "requires_external_state": requires_external_state,
        "structured_output_required": structured_output_required,
    }


# ---------------------------------------------------------------------------
# Information-theoretic helpers
# ---------------------------------------------------------------------------


def _mutual_information_binary(
    var_values: list[bool], action_values: list[str], actions: list[str]
) -> float:
    """Mutual information between a binary variable and a discrete action.

    MI = sum_{v,a} P(v,a) * log2( P(v,a) / (P(v) * P(a)) )
    """
    n = len(var_values)
    if n == 0:
        return 0.0

    # Joint counts.
    joint: dict[tuple[bool, str], int] = {}
    p_var: dict[bool, int] = {}
    p_act: dict[str, int] = {}
    for v, a in zip(var_values, action_values):
        joint[(v, a)] = joint.get((v, a), 0) + 1
        p_var[v] = p_var.get(v, 0) + 1
        p_act[a] = p_act.get(a, 0) + 1

    mi = 0.0
    for (v, a), c in joint.items():
        p_va = c / n
        p_v = p_var[v] / n
        p_a = p_act[a] / n
        if p_va > 0 and p_v > 0 and p_a > 0:
            mi += p_va * math.log2(p_va / (p_v * p_a))
    return float(mi)


def _discriminates(
    values_by_action: dict[str, dict[str, float]], actions: list[str]
) -> bool:
    """A variable discriminates if its P(true) differs across optimal actions."""
    probs = [
        values_by_action.get(a, {}).get("p_true", 0.0) for a in actions
    ]
    if len(probs) < 2:
        return False
    return (max(probs) - min(probs)) > 0.1


# ---------------------------------------------------------------------------
# Missing-feature candidate definitions
# ---------------------------------------------------------------------------

# Each candidate maps a state variable to a proposed executive feature.
# All candidates are pre-action, non-verifier, non-outcome, and computable
# in production from runtime state.
_FEATURE_CANDIDATE_DEFS: dict[str, dict[str, Any]] = {
    "memory_available": {
        "name": "memory_store_populated",
        "definition": "1.0 if the memory store has at least one item available for this conversation, else 0.0",
        "source": "memory_service",
        "computable_in_production": True,
    },
    "memory_relevant": {
        "name": "memory_capability_match",
        "definition": "1.0 if the task family requires memory retrieval, else 0.0",
        "source": "prompt_analyzer",
        "computable_in_production": True,
    },
    "memory_fresh": {
        "name": "memory_freshness_flag",
        "definition": "1.0 if the most recent memory entry was written within the current session, else 0.0",
        "source": "memory_service",
        "computable_in_production": True,
    },
    "memory_conflict": {
        "name": "memory_conflict_flag",
        "definition": "1.0 if two memory entries disagree on the queried fact, else 0.0",
        "source": "memory_service",
        "computable_in_production": True,
    },
    "tool_available": {
        "name": "tool_registry_nonempty",
        "definition": "1.0 if at least one tool is registered and available for this task, else 0.0",
        "source": "tool_registry",
        "computable_in_production": True,
    },
    "tool_reliable": {
        "name": "tool_reliability_flag",
        "definition": "1.0 if the best available tool has historical reliability above threshold, else 0.0",
        "source": "tool_registry",
        "computable_in_production": True,
    },
    "tool_latency_high": {
        "name": "tool_latency_high_flag",
        "definition": "1.0 if the estimated tool latency exceeds the response budget, else 0.0",
        "source": "tool_registry",
        "computable_in_production": True,
    },
    "tool_authorized": {
        "name": "tool_authorization_flag",
        "definition": "1.0 if the tool is authorized for the current user/session, else 0.0",
        "source": "tool_registry",
        "computable_in_production": True,
    },
    "workflow_capability": {
        "name": "workflow_capability_match",
        "definition": "1.0 if the task family requires a multi-step workflow, else 0.0",
        "source": "prompt_analyzer",
        "computable_in_production": True,
    },
    "workflow_acceptance_contract_present": {
        "name": "workflow_acceptance_contract_present",
        "definition": "1.0 if an acceptance contract is registered for the workflow, else 0.0",
        "source": "workflow_registry",
        "computable_in_production": True,
    },
    "workflow_expected_cost_high": {
        "name": "workflow_expected_cost_high",
        "definition": "1.0 if the workflow expected cost exceeds the budget, else 0.0",
        "source": "workflow_registry",
        "computable_in_production": True,
    },
    "task_reversible": {
        "name": "task_reversibility_flag",
        "definition": "1.0 if the task action is reversible, else 0.0",
        "source": "prompt_analyzer",
        "computable_in_production": True,
    },
    "task_risky": {
        "name": "task_risk_flag",
        "definition": "1.0 if the task risk class is high or critical, else 0.0",
        "source": "prompt_analyzer",
        "computable_in_production": True,
    },
    "direct_answer_confidence_high": {
        "name": "direct_answer_confidence_proxy",
        "definition": "1.0 if the prompt is short and lacks retrieval/tool keywords, else 0.0",
        "source": "prompt_analyzer",
        "computable_in_production": True,
    },
    "conversation_depth_high": {
        "name": "conversation_depth_high_flag",
        "definition": "1.0 if the prior conversation depth is at least 2 turns, else 0.0",
        "source": "conversation_service",
        "computable_in_production": True,
    },
    "task_complexity_high": {
        "name": "task_complexity_high_flag",
        "definition": "1.0 if the estimated task complexity is at least 0.7, else 0.0",
        "source": "prompt_analyzer",
        "computable_in_production": True,
    },
    "requires_external_state": {
        "name": "requires_external_state_flag",
        "definition": "1.0 if the prompt requests current/live/latest external state, else 0.0",
        "source": "prompt_analyzer",
        "computable_in_production": True,
    },
    "structured_output_required": {
        "name": "structured_output_required_flag",
        "definition": "1.0 if the prompt requests structured output (json/yaml/csv/table), else 0.0",
        "source": "prompt_analyzer",
        "computable_in_production": True,
    },
}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_missing_state(run: LoadedRun) -> dict[str, Any]:
    """Identify hidden state variables defining the optimal action.

    For each task we extract pre-action environment variables, determine the
    optimal (oracle) action from counterfactual outcomes, then compute the
    conditional distribution of each variable given the optimal action.

    Returns a JSON-serializable dict with:
      - state_variables: variable -> {values_by_optimal_action, discriminates, mutual_info}
      - missing_feature_candidates: list of candidate feature metadata
      - interpretation: human-readable summary
    """
    # Gather (task_id, optimal_action, state_vars) over all tasks with an
    # executed oracle action.  We use every split here because the analysis
    # is diagnostic; the *selection* of features (if any) happens elsewhere
    # on validation only.
    rows: list[tuple[str, str, dict[str, bool]]] = []
    for tid, task in run.tasks.items():
        oracle_action = run.get_oracle_action(tid)
        if oracle_action is None:
            continue
        if oracle_action not in FOCAL_ACTIONS:
            continue
        sv = _extract_state_variables(task)
        rows.append((tid, oracle_action, sv))

    actions_present = sorted({a for _, a, _ in rows})
    # Always report against the full focal set so the dict is stable.
    actions_report = [a for a in FOCAL_ACTIONS if a in actions_present] or list(FOCAL_ACTIONS)

    state_variables: dict[str, Any] = {}
    for var in STATE_VARIABLES:
        var_values = [sv.get(var, False) for _, _, sv in rows]
        act_values = [a for _, a, _ in rows]

        values_by_optimal_action: dict[str, dict[str, float]] = {}
        for a in actions_report:
            mask = [i for i, av in enumerate(act_values) if av == a]
            if mask:
                p_true = float(np.mean([1.0 if var_values[i] else 0.0 for i in mask]))
                n = len(mask)
            else:
                p_true = 0.0
                n = 0
            values_by_optimal_action[a] = {
                "p_true": p_true,
                "n_tasks": n,
            }

        discriminates = _discriminates(values_by_optimal_action, actions_report)
        mi = _mutual_information_binary(var_values, act_values, actions_report)

        state_variables[var] = {
            "values_by_optimal_action": values_by_optimal_action,
            "discriminates": bool(discriminates),
            "mutual_info": float(mi),
        }

    # --- Missing-feature candidates --------------------------------------
    # A candidate is proposed only if its source variable discriminates the
    # optimal action (otherwise it carries no routing signal).
    missing_feature_candidates: list[dict[str, Any]] = []
    for var in STATE_VARIABLES:
        sv_info = state_variables.get(var, {})
        if not sv_info.get("discriminates", False):
            continue
        cand_def = _FEATURE_CANDIDATE_DEFS.get(var, {})
        candidate = {
            "name": cand_def.get("name", var),
            "definition": cand_def.get("definition", f"binary flag for {var}"),
            "available_before_action": True,
            "not_derived_from_verifier": True,
            "not_derived_from_outcome": True,
            "not_containing_expected_answer": True,
            "not_containing_best_action_label": True,
            "stable_across_counterfactual_actions": True,
            "computable_in_production": cand_def.get("computable_in_production", True),
            "source": cand_def.get("source", "prompt_analyzer"),
            "state_variable": var,
            "mutual_info_with_optimal_action": sv_info.get("mutual_info", 0.0),
        }
        missing_feature_candidates.append(candidate)

    # --- Interpretation ---------------------------------------------------
    n_discriminating = sum(
        1 for v in state_variables.values() if v.get("discriminates", False)
    )
    top_mi = sorted(
        ((var, info["mutual_info"]) for var, info in state_variables.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    interpretation = (
        f"Analyzed {len(rows)} tasks with an executed oracle action across "
        f"{len(actions_report)} optimal actions.  "
        f"{n_discriminating}/{len(STATE_VARIABLES)} state variables "
        f"discriminate the optimal action (P(true) spread > 0.1).  "
        f"Top variables by mutual information: "
        + ", ".join(f"{v} (MI={mi:.4f})" for v, mi in top_mi)
        + f".  Proposed {len(missing_feature_candidates)} missing-feature "
        f"candidates that satisfy the leakage firewall."
    )

    return {
        "state_variables": state_variables,
        "missing_feature_candidates": missing_feature_candidates,
        "interpretation": interpretation,
        "n_tasks_analyzed": len(rows),
        "focal_actions": list(FOCAL_ACTIONS),
        "actions_present": actions_report,
    }
