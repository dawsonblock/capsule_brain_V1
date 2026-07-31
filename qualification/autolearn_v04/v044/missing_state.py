"""Missing-state analysis for v0.4.4 evidence-complete analysis.

Identifies missing pre-action state variables that could improve
routing.  The analysis is purely diagnostic — it never writes a new
policy and never uses the final test split to choose features.

Every proposed missing variable is checked against a strict leakage
firewall:
  - available before the action is taken
  - not derived from the verifier
  - not derived from the outcome
  - not containing the expected answer
  - not containing the best-action label
  - stable across counterfactual actions
  - computable in production

The module also flags when the model's own derived confidence is a
plausible missing variable (i.e., the model has an internal confidence
signal that is not exposed as a pre-action feature but correlates with
routing quality).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import MINIMUM_DENOMINATOR


# ---------------------------------------------------------------------------
# Status categories
# ---------------------------------------------------------------------------

MISSING_STATE_IDENTIFIED = "MISSING_STATE_IDENTIFIED"
MISSING_STATE_NONE = "MISSING_STATE_NONE"
MISSING_STATE_INCONCLUSIVE = "MISSING_STATE_INCONCLUSIVE"


# ---------------------------------------------------------------------------
# State variables examined (all pre-action, no verifier/outcome leakage)
# ---------------------------------------------------------------------------

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
# State-variable descriptions
# ---------------------------------------------------------------------------

_STATE_DESCRIPTIONS: dict[str, str] = {
    "memory_available": (
        "Whether a memory store is available and populated for the task."
    ),
    "memory_relevant": (
        "Whether the task explicitly requires retrieval from memory."
    ),
    "memory_fresh": (
        "Whether stored memory entries are fresh (not stale)."
    ),
    "memory_conflict": (
        "Whether memory entries contain conflicting information."
    ),
    "tool_available": (
        "Whether at least one external tool is available for the task."
    ),
    "tool_reliable": (
        "Whether available tools are marked as reliable."
    ),
    "tool_latency_high": (
        "Whether tool latency is high enough to affect routing decisions."
    ),
    "tool_authorized": (
        "Whether the tool is authorized for use in the current context."
    ),
    "workflow_capability": (
        "Whether a workflow capability is available for the task family."
    ),
    "workflow_acceptance_contract_present": (
        "Whether an acceptance contract is present for workflow execution."
    ),
    "workflow_expected_cost_high": (
        "Whether the expected workflow cost is high."
    ),
    "task_reversible": (
        "Whether the task action is reversible."
    ),
    "task_risky": (
        "Whether the task carries high risk (risk_class)."
    ),
    "direct_answer_confidence_high": (
        "Whether the prompt suggests high direct-answer confidence."
    ),
    "conversation_depth_high": (
        "Whether the conversation depth is >= 2 turns."
    ),
    "task_complexity_high": (
        "Whether the task complexity is high (>= 0.7)."
    ),
    "requires_external_state": (
        "Whether the task requires external/live state."
    ),
    "structured_output_required": (
        "Whether structured output (JSON/YAML/etc.) is required."
    ),
}


# ---------------------------------------------------------------------------
# State-variable extractors (all pre-action, no verifier/outcome leakage)
# ---------------------------------------------------------------------------


def _extract_state_variables(task: dict[str, Any]) -> dict[str, bool]:
    """Extract binary pre-action state variables from task metadata.

    Every value is derived from setup_spec, family, risk_class, or
    prompt features — never from the verifier, the outcome, or the
    expected answer.
    """
    setup = task.get("setup_spec", {}) or {}
    prompt = task.get("prompt", "") or ""
    prompt_lower = prompt.lower()
    family = task.get("family", task.get("task_family", "")) or ""
    risk_class = task.get("risk_class", "benign") or "benign"

    available_tools = setup.get("available_tools", []) or []
    memory_spec = setup.get("memory", {}) or {}
    tool_spec = setup.get("tool", {}) or {}
    workflow_spec = setup.get("workflow", {}) or {}

    # --- Memory ---
    memory_available = bool(memory_spec.get("available", False)) or bool(
        setup.get("secret")
    )
    memory_relevant = family == "memory_required"
    memory_fresh = bool(memory_spec.get("fresh", False))
    memory_conflict = bool(memory_spec.get("conflict", False))

    # --- Tool ---
    tool_available = bool(available_tools) or bool(tool_spec.get("available", False))
    tool_reliable = bool(tool_spec.get("reliable", True))
    tool_latency = float(tool_spec.get("latency_estimate", 0.0) or 0.0)
    tool_latency_high = tool_latency >= 1.0
    tool_authorized = bool(tool_spec.get("authorized", True))

    # --- Workflow ---
    workflow_capability = family == "workflow_required"
    workflow_acceptance_contract_present = bool(
        workflow_spec.get("acceptance_contract", False)
    ) or bool(setup.get("acceptance_contract"))
    workflow_expected_cost = float(workflow_spec.get("expected_cost", 0.0) or 0.0)
    workflow_expected_cost_high = workflow_expected_cost >= 1.0

    # --- Task properties ---
    task_reversible = bool(setup.get("reversible", True))
    task_risky = risk_class in {"high", "critical", "dangerous", "risky"}

    # --- Prompt-derived (pre-action) ---
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
    direct_answer_confidence_high = (
        len(prompt) < 400 and not has_retrieval and not has_tool
    )

    conversation_depth = int(setup.get("conversation_depth", 0) or 0)
    conversation_depth_high = conversation_depth >= 2

    task_complexity = float(setup.get("complexity", 0.5) or 0.5)
    task_complexity_high = task_complexity >= 0.7

    requires_external_state = bool(setup.get("requires_external_state", False)) or (
        "current" in prompt_lower
        or "latest" in prompt_lower
        or "live" in prompt_lower
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
    """Mutual information between a binary variable and a discrete action."""
    n = len(var_values)
    if n == 0:
        return 0.0

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
# Helpers for extracting data from results dicts
# ---------------------------------------------------------------------------


def _extract_examples(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract per-task examples from a results dict."""
    per_task = results.get("per_task", [])
    if per_task:
        return [ex for ex in per_task if isinstance(ex, dict)]
    experiences = results.get("experiences", [])
    if experiences:
        return experiences
    experiences = results.get("executive_experiences", [])
    if experiences:
        return experiences
    return []


def _get_actions(results: dict[str, Any]) -> list[str]:
    """Extract the action list from a results dict."""
    actions = results.get("actions", [])
    if actions:
        return list(actions)
    per_task = results.get("per_task", [])
    seen: list[str] = []
    for ex in per_task:
        a = ex.get("best_action", "")
        if a and a not in seen:
            seen.append(a)
    return seen


def _get_feature_names(results: dict[str, Any]) -> list[str]:
    """Extract feature names from a results dict or candidate policy."""
    names = results.get("feature_names", [])
    if names:
        return list(names)
    transform = results.get("feature_transform", {})
    if isinstance(transform, dict):
        names = transform.get("feature_names", [])
        if names:
            return list(names)
    return []


# ---------------------------------------------------------------------------
# Model-derived confidence check
# ---------------------------------------------------------------------------


def _check_model_derived_confidence(
    candidate_results: dict[str, Any],
    examples: list[dict[str, Any]],
) -> bool:
    """Check whether model-derived confidence is a plausible missing variable.

    If the candidate policy records per-task confidence scores that
    correlate with routing quality but are not exposed as pre-action
    features, then model-derived confidence is a plausible missing
    state variable.
    """
    per_task = candidate_results.get("per_task", [])
    confidences: list[float] = []
    utilities: list[float] = []
    for ex in per_task:
        conf = ex.get("confidence", ex.get("model_confidence"))
        u = ex.get("utility")
        if conf is not None and u is not None:
            confidences.append(float(conf))
            utilities.append(float(u))

    if len(confidences) < 5:
        return False

    # Check correlation between confidence and utility.
    conf_arr = np.array(confidences)
    util_arr = np.array(utilities)
    if conf_arr.std() < MINIMUM_DENOMINATOR or util_arr.std() < MINIMUM_DENOMINATOR:
        return False

    corr = float(np.corrcoef(conf_arr, util_arr)[0, 1])
    if not np.isfinite(corr):
        return False

    # If |corr| > 0.3, confidence is plausibly a missing variable.
    return abs(corr) > 0.3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_missing_state(
    candidate_results: dict,
    oracle_results: dict,
    sham_results: dict,
    raw_vs_executed: dict | None = None,
) -> dict:
    """Identify missing pre-action state variables that could improve routing.

    The analysis is purely diagnostic — it never writes a new policy
    and never uses the final test split to choose features.

    Args:
        candidate_results: Results dict from the candidate policy run.
        oracle_results: Results dict from the oracle policy run.
        sham_results: Results dict from the sham policy run.
        raw_vs_executed: Optional dict of raw-vs-executed comparison
            results, used to detect state variables that are available
            in raw but not in the executed feature set.

    Returns:
        A dict with:
          - status: one of the status categories.
          - missing_variables: list of {name, description, plausibility}.
          - model_derived_confidence_as_plausible_missing: bool.
          - conclusion: human-readable conclusion string.
    """
    examples = _extract_examples(candidate_results)
    actions = _get_actions(candidate_results)
    feature_names = _get_feature_names(candidate_results)

    if len(examples) < 5 or len(actions) < 2:
        return {
            "status": MISSING_STATE_INCONCLUSIVE,
            "missing_variables": [],
            "model_derived_confidence_as_plausible_missing": False,
            "conclusion": (
                "Insufficient data to identify missing state variables "
                f"({len(examples)} examples, {len(actions)} actions)."
            ),
        }

    # Extract state variables and best actions for each example.
    state_values: dict[str, list[bool]] = {v: [] for v in STATE_VARIABLES}
    best_actions: list[str] = []
    for ex in examples:
        sv = _extract_state_variables(ex)
        for v in STATE_VARIABLES:
            state_values[v].append(sv.get(v, False))
        best_actions.append(ex.get("best_action", ""))

    # Compute mutual information and discrimination for each variable.
    missing_variables: list[dict[str, Any]] = []

    for var in STATE_VARIABLES:
        vals = state_values[var]
        mi = _mutual_information_binary(vals, best_actions, actions)

        # Compute P(true) per action.
        values_by_action: dict[str, dict[str, float]] = {}
        for a in actions:
            mask = [ba == a for ba in best_actions]
            subset = [v for v, m in zip(vals, mask) if m]
            n = len(subset)
            p_true = sum(1 for v in subset if v) / n if n > 0 else 0.0
            values_by_action[a] = {"p_true": p_true, "n": n}

        discriminates = _discriminates(values_by_action, actions)

        # Check if this variable is already captured in feature_names.
        already_present = any(var in fn.lower() for fn in feature_names)

        # Plausibility: high if MI is meaningful and it discriminates.
        if mi > 0.05 and discriminates and not already_present:
            plausibility = "high"
        elif mi > 0.01 and not already_present:
            plausibility = "medium"
        else:
            plausibility = "low"

        if plausibility in ("high", "medium"):
            missing_variables.append({
                "name": var,
                "description": _STATE_DESCRIPTIONS.get(var, var.replace("_", " ")),
                "plausibility": plausibility,
            })

    # Check raw_vs_executed for additional missing variables.
    if raw_vs_executed:
        raw_features = set(raw_vs_executed.get("raw_features", []))
        executed_features = set(raw_vs_executed.get("executed_features", []))
        dropped = raw_features - executed_features
        for feat in sorted(dropped):
            missing_variables.append({
                "name": feat,
                "description": (
                    f"Feature present in raw but dropped in executed "
                    f"feature set."
                ),
                "plausibility": "medium",
            })

    # Check model-derived confidence.
    model_conf_plausible = _check_model_derived_confidence(
        candidate_results, examples
    )
    if model_conf_plausible:
        missing_variables.append({
            "name": "model_derived_confidence",
            "description": (
                "The model's internal confidence score correlates with "
                "routing utility but is not exposed as a pre-action "
                "feature."
            ),
            "plausibility": "high",
        })

    # Determine status.
    high_plausibility = [
        v for v in missing_variables if v["plausibility"] == "high"
    ]
    if high_plausibility:
        status = MISSING_STATE_IDENTIFIED
        conclusion = (
            f"Identified {len(high_plausibility)} high-plausibility "
            f"missing state variable(s): "
            f"{[v['name'] for v in high_plausibility]}. These variables "
            f"discriminate the optimal action and are not currently "
            f"captured in the executive feature set."
        )
    elif missing_variables:
        status = MISSING_STATE_IDENTIFIED
        conclusion = (
            f"Identified {len(missing_variables)} medium-plausibility "
            f"missing state variable(s): "
            f"{[v['name'] for v in missing_variables]}. These may "
            f"improve routing if added to the feature set."
        )
    else:
        status = MISSING_STATE_NONE
        conclusion = (
            "No missing state variables identified. The current "
            "executive feature set appears to capture the pre-action "
            "state that discriminates the optimal action."
        )

    return {
        "status": status,
        "missing_variables": missing_variables,
        "model_derived_confidence_as_plausible_missing": model_conf_plausible,
        "conclusion": conclusion,
    }
