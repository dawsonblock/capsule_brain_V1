"""Verifier registry audit for v0.4.5.

Collects every ``verifier_name`` used across the evidence package and
compares it against the authoritative verifier registry.

Failure conditions:
  - Unknown verifier (not in the registry): FAIL
  - Missing verifier version: FAIL
  - Missing verifier class: FAIL
  - Missing reliability mapping: FAIL
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Default verifier registry
# ---------------------------------------------------------------------------

DEFAULT_VERIFIER_REGISTRY: dict[str, dict[str, Any]] = {
    "exact_match": {
        "class": "ExactMatchVerifier",
        "version": "1.0",
        "reliability": {
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
        },
    },
    "tool_execution": {
        "class": "ToolExecutionVerifier",
        "version": "1.0",
        "reliability": {
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
        },
    },
    "workflow_completion": {
        "class": "WorkflowCompletionVerifier",
        "version": "1.0",
        "reliability": {
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
        },
    },
    "memory_recall": {
        "class": "MemoryRecallVerifier",
        "version": "1.0",
        "reliability": {
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_verifier_names(
    counterfactual_outcomes: list[dict],
    experience_rows: list[dict],
    safety_results: list[dict],
    policy_results: dict,
) -> dict[str, set[str]]:
    """Collect {verifier_name: set(sources)} across all evidence sources."""
    used: dict[str, set[str]] = {}

    def _add(name: str, source: str) -> None:
        if not name:
            return
        used.setdefault(str(name), set()).add(source)

    for row in counterfactual_outcomes or []:
        if isinstance(row, dict):
            _add(row.get("verifier_name"), "counterfactual_outcomes")

    for row in experience_rows or []:
        if isinstance(row, dict):
            _add(row.get("verifier_name"), "experience_rows")

    for row in safety_results or []:
        if isinstance(row, dict):
            _add(row.get("verifier_name"), "safety_results")
            # Safety results may nest verifier info under a sub-dict.
            sv = row.get("safety_verifier")
            if isinstance(sv, dict):
                _add(sv.get("verifier_name") or sv.get("name"), "safety_results")

    if isinstance(policy_results, dict):
        _add(policy_results.get("verifier_name"), "policy_results")
        for key in ("results", "per_task", "decision_traces"):
            entries = policy_results.get(key)
            if isinstance(entries, list):
                for r in entries:
                    if isinstance(r, dict):
                        _add(r.get("verifier_name"), "policy_results")

    return used


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_verifier_registry(
    counterfactual_outcomes: list[dict],
    experience_rows: list[dict],
    safety_results: list[dict],
    policy_results: dict,
    verifier_registry: dict | None = None,
) -> dict:
    """Audit verifier usage against the authoritative verifier registry.

    Collects every ``verifier_name`` referenced in the counterfactual
    outcomes, experience rows, safety results, and policy evaluations,
    then compares each against the registry.  If ``verifier_registry`` is
    None, a default registry of common verifiers is used.

    Returns a dict with status ("PASS" | "FAIL" | "BLOCKED"), the list of
    used verifiers (with name, version, class, and sources), and lists of
    unknown verifiers, missing versions, and missing classes.
    """
    errors: list[str] = []

    registry = verifier_registry
    if registry is None:
        registry = DEFAULT_VERIFIER_REGISTRY
    if not isinstance(registry, dict) or not registry:
        errors.append("verifier_registry is missing or empty")
        return {
            "status": "BLOCKED",
            "used_verifiers": [],
            "unknown_verifiers": [],
            "missing_versions": [],
            "missing_classes": [],
            "missing_reliability_mappings": [],
            "errors": errors,
        }

    used = _collect_verifier_names(
        counterfactual_outcomes, experience_rows, safety_results, policy_results,
    )

    # BLOCKED when no verifiers are referenced anywhere.
    if not used:
        errors.append("No verifier_name references found in any evidence source")
        return {
            "status": "BLOCKED",
            "used_verifiers": [],
            "unknown_verifiers": [],
            "missing_versions": [],
            "missing_classes": [],
            "missing_reliability_mappings": [],
            "errors": errors,
        }

    unknown_verifiers: list[str] = []
    missing_versions: list[str] = []
    missing_classes: list[str] = []
    missing_reliability_mappings: list[str] = []
    used_verifiers: list[dict[str, Any]] = []

    for name in sorted(used):
        sources = sorted(used[name])
        entry = registry.get(name)
        if entry is None:
            unknown_verifiers.append(name)
            errors.append(f"Unknown verifier '{name}' (sources: {sources})")
            used_verifiers.append({
                "name": name,
                "version": None,
                "class": None,
                "sources": sources,
            })
            continue

        version = entry.get("version") if isinstance(entry, dict) else None
        vclass = entry.get("class") if isinstance(entry, dict) else None
        reliability = entry.get("reliability") if isinstance(entry, dict) else None

        if not version:
            missing_versions.append(name)
            errors.append(f"Verifier '{name}' is missing a version")
        if not vclass:
            missing_classes.append(name)
            errors.append(f"Verifier '{name}' is missing a class")
        if reliability is None:
            missing_reliability_mappings.append(name)
            errors.append(f"Verifier '{name}' is missing a reliability mapping")

        used_verifiers.append({
            "name": name,
            "version": version,
            "class": vclass,
            "sources": sources,
        })

    status = "PASS" if not errors else "FAIL"

    return {
        "status": status,
        "used_verifiers": used_verifiers,
        "unknown_verifiers": unknown_verifiers,
        "missing_versions": missing_versions,
        "missing_classes": missing_classes,
        "missing_reliability_mappings": missing_reliability_mappings,
        "errors": errors,
    }
