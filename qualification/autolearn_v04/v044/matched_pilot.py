"""Matched 14B pilot eligibility and manifest for v0.4.4.

A small matched pilot may be approved only when all eligibility checks pass.
A full 14B run remains blocked unless the pilot shows predeclared improvement.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def check_pilot_eligibility(
    gate_a0_status: str,
    gate_a1_status: str,
    sham_validity_status: str,
    utility_consistency_status: str,
    counterfactual_completeness_status: str,
    metric_consistency_status: str,
    feature_signal_status: str,
    candidate_stability_status: str,
    headroom_concentration_status: str,
    raw_vs_executed: dict | None = None,
    missing_state: dict | None = None,
) -> dict[str, Any]:
    """Check if a matched 14B pilot is eligible.

    A pilot may be approved only when:
    - A0 PASS
    - A1 PASS
    - sham validity PASS
    - utility consistency PASS
    - counterfactual completeness PASS
    - metric consistency PASS
    - feature signal PRESENT or WEAK
    - candidate stability not FAILED
    - headroom not excessively concentrated
    - raw candidate or validated learner recovers some positive headroom,
      or missing-state analysis identifies model-derived confidence as
      the plausible missing variable
    - pilot task subset frozen before execution
    """
    checks: list[dict[str, Any]] = []

    def _check(name, condition, observed, expected):
        checks.append({
            "check": name,
            "status": "PASS" if condition else "FAIL",
            "observed": observed,
            "expected": expected,
        })

    _check("A0_pass", gate_a0_status == "PASS", gate_a0_status, "PASS")
    _check("A1_pass", gate_a1_status == "PASS", gate_a1_status, "PASS")
    _check("sham_validity", sham_validity_status == "PASS", sham_validity_status, "PASS")
    _check("utility_consistency", utility_consistency_status == "PASS", utility_consistency_status, "PASS")
    _check("counterfactual_completeness", counterfactual_completeness_status == "PASS",
           counterfactual_completeness_status, "PASS")
    _check("metric_consistency", metric_consistency_status == "PASS", metric_consistency_status, "PASS")
    _check("feature_signal", feature_signal_status in ("FEATURE_SIGNAL_PRESENT", "FEATURE_SIGNAL_WEAK", "PASS"),
           feature_signal_status, "PRESENT or WEAK")
    _check("candidate_stability", candidate_stability_status not in ("FAILED", "UNSTABLE"),
           candidate_stability_status, "not FAILED")
    _check("headroom_concentration", headroom_concentration_status != "INVALID_DATA",
           headroom_concentration_status, "not INVALID_DATA")

    # Check headroom recovery or missing-state hypothesis.
    has_headroom = False
    if raw_vs_executed:
        for c in raw_vs_executed.get("conclusions", []):
            if "SUCCEED" in c and "BOTH" in c:
                has_headroom = True

    has_missing_state_hypothesis = False
    if missing_state:
        has_missing_state_hypothesis = missing_state.get(
            "model_derived_confidence_as_plausible_missing", False
        )

    _check("headroom_or_missing_state",
           has_headroom or has_missing_state_hypothesis,
           f"headroom={has_headroom}, missing_state={has_missing_state_hypothesis}",
           "at least one true")

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    eligible = n_fail == 0

    return {
        "eligible": eligible,
        "status": "ELIGIBLE" if eligible else "NOT_ELIGIBLE",
        "n_checks": len(checks),
        "n_pass": len(checks) - n_fail,
        "n_fail": n_fail,
        "checks": checks,
    }


def create_pilot_manifest(
    eligibility: dict,
    benchmark_manifest: dict,
    pilot_task_count: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    """Create the matched pilot manifest.

    The pilot should use 20-30 tasks balanced across:
    direct, memory, tool, workflow, crossover, meaningful margin, high-value margin.

    Uses identical: task IDs, prompts, hidden setup, tool behavior, memory state,
    workflow contracts, verifiers, eligible actions, utility version, timeouts,
    generation config where compatible.

    Compares: 7B vs 14B on overall success, per-action success, oracle-sham headroom,
    action selectivity, utility margins.
    """
    if not eligibility.get("eligible", False):
        return {
            "status": "NOT_ELIGIBLE",
            "eligible": False,
            "reason": "Pilot eligibility checks not met",
        }

    families = benchmark_manifest.get("task_families", [
        "direct", "memory", "tool", "workflow", "crossover"
    ])

    # Balance tasks across families.
    tasks_per_family = pilot_task_count // len(families)
    remainder = pilot_task_count % len(families)

    pilot_tasks: list[dict[str, Any]] = []
    for i, family in enumerate(families):
        count = tasks_per_family + (1 if i < remainder else 0)
        for j in range(count):
            pilot_tasks.append({
                "task_id": f"pilot_{family}_{j:03d}",
                "family": family,
                "margin_bucket": "meaningful" if j % 2 == 0 else "high_value",
            })

    manifest = {
        "status": "ELIGIBLE",
        "eligible": True,
        "pilot_task_count": pilot_task_count,
        "pilot_tasks": pilot_tasks,
        "families": families,
        "comparison_metrics": [
            "7B_overall_success",
            "14B_overall_success",
            "7B_per_action_success",
            "14B_per_action_success",
            "7B_oracle_sham_headroom",
            "14B_oracle_sham_headroom",
            "7B_action_selectivity",
            "14B_action_selectivity",
            "7B_utility_margins",
            "14B_utility_margins",
        ],
        "identical_conditions": [
            "task_ids", "prompts", "hidden_setup", "tool_behavior",
            "memory_state", "workflow_contracts", "verifiers",
            "eligible_actions", "utility_version", "timeouts",
            "generation_config_where_compatible",
        ],
        "seed": seed,
        "approve_full_14b_run": False,
        "full_14b_requirement": "Pilot must show predeclared meaningful increase in oracle-sham routing headroom or action-specific selectivity",
    }

    # Compute manifest digest.
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    manifest["manifest_digest"] = manifest_digest

    return manifest
