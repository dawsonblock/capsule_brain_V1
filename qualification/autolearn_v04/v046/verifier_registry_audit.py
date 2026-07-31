"""Verifier registry audit for v0.4.6.

Audits the verifier registry by enumerating verifier identities from
outcome rows — not returning 0/0 when verifiers are actually used.

Loads the canonical verifier registry from
``qualification/autolearn_v04/common/verifier_registry.json`` and
compares every ``verifier_name`` found in the evidence against it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus

# Path to the canonical verifier registry.
_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent / "common" / "verifier_registry.json"
)


def _load_registry() -> dict[str, Any]:
    """Load the canonical verifier registry JSON."""
    try:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"verifiers": {}}


def _collect_verifier_info(
    counterfactual_outcomes: list[dict],
    experience_rows: list[dict],
    safety_results: list[dict],
    candidate_results: dict,
) -> dict[str, dict[str, Any]]:
    """Collect verifier usage info from all evidence sources.

    Returns a dict mapping verifier_name -> {version, class, sources, families}.
    """
    used: dict[str, dict[str, Any]] = {}

    def _add(name: str, version: str | None, vclass: str | None,
             source: str, family: str | None = None) -> None:
        if not name:
            return
        key = str(name)
        if key not in used:
            used[key] = {
                "name": key,
                "version": None,
                "class": None,
                "sources": set(),
                "families": set(),
            }
        entry = used[key]
        if version:
            entry["version"] = str(version)
        if vclass:
            entry["class"] = str(vclass)
        entry["sources"].add(source)
        if family:
            entry["families"].add(str(family))

    # Counterfactual outcomes.
    for row in counterfactual_outcomes or []:
        if not isinstance(row, dict):
            continue
        _add(
            row.get("verifier_name"),
            row.get("verifier_version"),
            row.get("verifier_class"),
            "counterfactual_outcomes",
            row.get("family"),
        )

    # Experience rows.
    for row in experience_rows or []:
        if not isinstance(row, dict):
            continue
        _add(
            row.get("verifier_name"),
            row.get("verifier_version"),
            row.get("verifier_class"),
            "experience_rows",
            row.get("family"),
        )
        # Also check verifier_names list (normalized task rows).
        vnames = row.get("verifier_names")
        if isinstance(vnames, list):
            for vn in vnames:
                _add(vn, None, None, "experience_rows", row.get("family"))

    # Safety results.
    for row in safety_results or []:
        if not isinstance(row, dict):
            continue
        _add(
            row.get("verifier_name"),
            row.get("verifier_version"),
            None,
            "safety_results",
        )

    # Candidate results.
    if isinstance(candidate_results, dict):
        _add(
            candidate_results.get("verifier_name"),
            candidate_results.get("verifier_version"),
            None,
            "candidate_results",
        )
        # Check per-task rows.
        task_rows = candidate_results.get("task_rows", [])
        if isinstance(task_rows, list):
            for tr in task_rows:
                if isinstance(tr, dict):
                    _add(
                        tr.get("verifier_name"),
                        tr.get("verifier_version"),
                        None,
                        "candidate_results",
                        tr.get("family"),
                    )

    return used


def audit_verifier_registry(
    counterfactual_outcomes: list[dict],
    experience_rows: list[dict],
    safety_results: list[dict],
    candidate_results: dict,
) -> dict:
    """Audit verifier usage against the canonical verifier registry.

    Enumerates every verifier identity from the outcome rows and
    compares against the registry.  Reports actual counts, not 0/0.

    Parameters
    ----------
    counterfactual_outcomes:
        List of counterfactual outcome dicts.
    experience_rows:
        List of experience row dicts (action-level or task-level).
    safety_results:
        List of safety result dicts.
    candidate_results:
        Candidate results dict.

    Returns
    -------
    dict
        used_verifier_count, registered_verifier_count, unknown_verifiers,
        version_mismatches, class_mismatches, missing_reliability_mappings,
        unsupported_family_use, status, reason
    """
    registry_data = _load_registry()
    verifiers = registry_data.get("verifiers", {})
    if not isinstance(verifiers, dict):
        verifiers = {}

    registered_verifier_count = len(verifiers)

    # Collect verifier usage from all evidence sources.
    used = _collect_verifier_info(
        counterfactual_outcomes, experience_rows, safety_results, candidate_results,
    )

    used_verifier_count = len(used)

    # BLOCKED: no verifiers found anywhere.
    if used_verifier_count == 0:
        return {
            "used_verifier_count": 0,
            "registered_verifier_count": registered_verifier_count,
            "unknown_verifiers": [],
            "version_mismatches": [],
            "class_mismatches": [],
            "missing_reliability_mappings": [],
            "unsupported_family_use": [],
            "n_registered": 0,
            "n_required": 0,
            "status": AuditStatus.BLOCKED.value,
            "reason": "no verifier_name references found in any evidence source",
        }

    unknown_verifiers: list[str] = []
    version_mismatches: list[dict[str, Any]] = []
    class_mismatches: list[dict[str, Any]] = []
    missing_reliability_mappings: list[str] = []
    unsupported_family_use: list[dict[str, Any]] = []
    errors: list[str] = []

    for name in sorted(used.keys()):
        entry = used[name]
        reg_entry = verifiers.get(name)

        if reg_entry is None:
            unknown_verifiers.append(name)
            errors.append(
                f"unknown verifier '{name}' (sources: {sorted(entry['sources'])})"
            )
            continue

        # Check version.
        reg_version = str(reg_entry.get("version", ""))
        used_version = entry.get("version") or ""
        if used_version and reg_version and used_version != reg_version:
            version_mismatches.append({
                "verifier": name,
                "used_version": used_version,
                "registered_version": reg_version,
            })
            errors.append(
                f"verifier '{name}' version mismatch: "
                f"used={used_version}, registered={reg_version}"
            )

        # Check class.
        reg_class = str(reg_entry.get("class", ""))
        used_class = entry.get("class") or ""
        if used_class and reg_class and used_class != reg_class:
            class_mismatches.append({
                "verifier": name,
                "used_class": used_class,
                "registered_class": reg_class,
            })
            errors.append(
                f"verifier '{name}' class mismatch: "
                f"used={used_class}, registered={reg_class}"
            )

        # Check reliability mapping.
        reliability = reg_entry.get("reliability")
        if reliability is None:
            missing_reliability_mappings.append(name)
            errors.append(f"verifier '{name}' missing reliability mapping in registry")

        # Check supported task families.
        supported_families = set(reg_entry.get("supported_task_families", []))
        used_families = entry.get("families", set())
        for fam in sorted(used_families):
            if supported_families and fam not in supported_families:
                unsupported_family_use.append({
                    "verifier": name,
                    "family": fam,
                    "supported_families": sorted(supported_families),
                })
                errors.append(
                    f"verifier '{name}' used for unsupported family '{fam}'"
                )

    if errors:
        return {
            "used_verifier_count": used_verifier_count,
            "registered_verifier_count": registered_verifier_count,
            "unknown_verifiers": unknown_verifiers,
            "version_mismatches": version_mismatches,
            "class_mismatches": class_mismatches,
            "missing_reliability_mappings": missing_reliability_mappings,
            "unsupported_family_use": unsupported_family_use,
            "n_registered": used_verifier_count - len(unknown_verifiers),
            "n_required": used_verifier_count,
            "status": AuditStatus.FAIL.value,
            "reason": "; ".join(errors),
        }

    return {
        "used_verifier_count": used_verifier_count,
        "registered_verifier_count": registered_verifier_count,
        "unknown_verifiers": unknown_verifiers,
        "version_mismatches": version_mismatches,
        "class_mismatches": class_mismatches,
        "missing_reliability_mappings": missing_reliability_mappings,
        "unsupported_family_use": unsupported_family_use,
        "n_registered": used_verifier_count,
        "n_required": used_verifier_count,
        "status": AuditStatus.PASS.value,
        "reason": (
            f"all {used_verifier_count} used verifier(s) are registered "
            f"and valid"
        ),
    }
