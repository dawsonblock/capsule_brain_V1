"""Task/split consistency audit for v0.4.5.

Builds one canonical task registry from the benchmark and split manifests
and validates that every task ID referenced across the evidence package
maps to exactly one split.

Validates task IDs from:
  - benchmark manifest
  - split manifest
  - counterfactual outcomes
  - experience rows
  - policy evaluation results
  - safety results
  - decision traces

Rules:
  - Every task must map to exactly one split.
  - No unknown task IDs (referenced but absent from the registry).
  - No duplicate task IDs across splits.
  - No test task in experience data.
  - No safety task in route training.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_benchmark_task_ids(benchmark_manifest: dict) -> list[str]:
    """Extract the canonical task ID list from the benchmark manifest."""
    ids = benchmark_manifest.get("task_ids")
    if isinstance(ids, list):
        return [str(t) for t in ids if t]
    tasks = benchmark_manifest.get("tasks")
    if isinstance(tasks, list):
        out: list[str] = []
        for t in tasks:
            if isinstance(t, dict):
                tid = t.get("task_id") or t.get("id")
                if tid:
                    out.append(str(tid))
            elif isinstance(t, str) and t:
                out.append(t)
        return out
    return []


def _extract_split_mapping(split_manifest: dict) -> dict[str, dict[str, Any]]:
    """Return {split_name: {"count": int, "task_ids": list[str]}}."""
    splits = split_manifest.get("splits", {})
    mapping: dict[str, dict[str, Any]] = {}
    if not isinstance(splits, dict):
        return mapping
    for name, info in splits.items():
        if not isinstance(info, dict):
            continue
        ids = info.get("task_ids", [])
        if not isinstance(ids, list):
            ids = []
        mapping[name] = {
            "count": info.get("count", len(ids)),
            "task_ids": [str(t) for t in ids if t],
        }
    return mapping


def _task_to_split(split_mapping: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Build a {task_id: split_name} lookup from the split mapping."""
    out: dict[str, str] = {}
    for split_name, info in split_mapping.items():
        for tid in info["task_ids"]:
            out[tid] = split_name
    return out


def _extract_counterfactual_task_ids(counterfactual_outcomes: list[dict]) -> list[str]:
    ids: list[str] = []
    for row in counterfactual_outcomes or []:
        if not isinstance(row, dict):
            continue
        tid = row.get("task_id")
        if tid:
            ids.append(str(tid))
    return ids


def _extract_experience_task_ids(experience_rows: list[dict]) -> list[str]:
    ids: list[str] = []
    for row in experience_rows or []:
        if not isinstance(row, dict):
            continue
        tid = row.get("task_id")
        if tid:
            ids.append(str(tid))
    return ids


def _extract_policy_task_ids(policy_results: dict) -> tuple[list[str], list[str]]:
    """Return (policy_eval_task_ids, decision_trace_task_ids)."""
    policy_ids: list[str] = []
    trace_ids: list[str] = []

    if not isinstance(policy_results, dict):
        return policy_ids, trace_ids

    direct = policy_results.get("task_ids")
    if isinstance(direct, list):
        policy_ids.extend(str(t) for t in direct if t)

    results = policy_results.get("results")
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                tid = r.get("task_id")
                if tid:
                    policy_ids.append(str(tid))

    per_task = policy_results.get("per_task")
    if isinstance(per_task, list):
        for r in per_task:
            if isinstance(r, dict):
                tid = r.get("task_id")
                if tid:
                    policy_ids.append(str(tid))

    traces = policy_results.get("decision_traces")
    if isinstance(traces, list):
        for r in traces:
            if isinstance(r, dict):
                tid = r.get("task_id")
                if tid:
                    trace_ids.append(str(tid))

    return policy_ids, trace_ids


def _extract_safety_task_ids(safety_results: list[dict]) -> list[str]:
    ids: list[str] = []
    for row in safety_results or []:
        if isinstance(row, dict):
            tid = row.get("task_id")
            if tid:
                ids.append(str(tid))
    return ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_task_split_consistency(
    benchmark_manifest: dict,
    split_manifest: dict,
    counterfactual_outcomes: list[dict],
    experience_rows: list[dict],
    policy_results: dict,
    safety_results: list[dict],
) -> dict:
    """Validate task/split consistency across the evidence package.

    Builds a canonical task registry from the benchmark and split manifests
    and checks that every referenced task ID maps to exactly one split.

    Returns a dict with status ("PASS" | "FAIL" | "BLOCKED"), per-source
    task counts, and lists of unknown, duplicate, test-in-experience, and
    safety-in-training task IDs.
    """
    errors: list[str] = []

    # --- BLOCKED check: manifests missing ---
    if not isinstance(benchmark_manifest, dict) or not benchmark_manifest:
        errors.append("benchmark_manifest is missing or empty")
    if not isinstance(split_manifest, dict) or not split_manifest:
        errors.append("split_manifest is missing or empty")
    if errors:
        return {
            "status": "BLOCKED",
            "benchmark_task_count": 0,
            "split_task_count": 0,
            "counterfactual_task_count": 0,
            "experience_task_count": 0,
            "policy_task_count": 0,
            "safety_task_count": 0,
            "unknown_task_ids": [],
            "duplicate_task_ids": [],
            "test_in_experience": [],
            "safety_in_training": [],
            "errors": errors,
        }

    benchmark_task_ids = _extract_benchmark_task_ids(benchmark_manifest)
    split_mapping = _extract_split_mapping(split_manifest)

    # --- Duplicate task IDs across splits ---
    seen: dict[str, list[str]] = {}
    for split_name, info in split_mapping.items():
        for tid in info["task_ids"]:
            seen.setdefault(tid, []).append(split_name)
    duplicate_task_ids = sorted(
        tid for tid, splits in seen.items() if len(splits) > 1
    )
    for tid in duplicate_task_ids:
        errors.append(
            f"Task '{tid}' appears in multiple splits: {seen[tid]}"
        )

    # Canonical registry = benchmark task IDs union split task IDs.
    registry: set[str] = set(benchmark_task_ids)
    for info in split_mapping.values():
        registry.update(info["task_ids"])

    # --- Per-source task IDs ---
    cf_ids = _extract_counterfactual_task_ids(counterfactual_outcomes)
    exp_ids = _extract_experience_task_ids(experience_rows)
    policy_ids, trace_ids = _extract_policy_task_ids(policy_results)
    safety_ids = _extract_safety_task_ids(safety_results)

    # --- Unknown task IDs ---
    referenced: list[str] = cf_ids + exp_ids + policy_ids + trace_ids + safety_ids
    unknown_task_ids = sorted(
        tid for tid in referenced if tid not in registry
    )
    for tid in unknown_task_ids:
        errors.append(f"Unknown task ID '{tid}' is not in the task registry")

    # --- Test tasks in experience data ---
    test_task_ids = set(split_mapping.get("test", {}).get("task_ids", []))
    experience_task_set = set(exp_ids)
    test_in_experience = sorted(test_task_ids & experience_task_set)
    for tid in test_in_experience:
        errors.append(
            f"Test task '{tid}' appears in experience/training data"
        )

    # --- Safety tasks in route training ---
    safety_split_ids = set(split_mapping.get("safety", {}).get("task_ids", []))
    # Safety tasks found in experience rows.
    safety_in_training = sorted(safety_split_ids & experience_task_set)
    for tid in safety_in_training:
        errors.append(
            f"Safety task '{tid}' appears in route training data"
        )

    status = "PASS" if not errors else "FAIL"

    return {
        "status": status,
        "benchmark_task_count": len(benchmark_task_ids),
        "split_task_count": sum(len(i["task_ids"]) for i in split_mapping.values()),
        "counterfactual_task_count": len(set(cf_ids)),
        "experience_task_count": len(set(exp_ids)),
        "policy_task_count": len(set(policy_ids)),
        "safety_task_count": len(set(safety_ids)),
        "unknown_task_ids": unknown_task_ids,
        "duplicate_task_ids": duplicate_task_ids,
        "test_in_experience": test_in_experience,
        "safety_in_training": safety_in_training,
        "errors": errors,
    }
