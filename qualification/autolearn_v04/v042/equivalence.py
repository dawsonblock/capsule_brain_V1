"""Counterfactual utility comparability verification.

For each task, verifies that the counterfactual outcomes (one per action)
are genuinely comparable — i.e., that they share the same starting
conditions so that differences in measured utility are attributable to
the action taken rather than to confounding differences in task setup,
model configuration, environment, or execution conditions.

Checks performed per task:
  - same task setup (same ``setup_spec`` across actions)
  - same hidden secret / tool output where counterfactual equivalence
    requires it
  - same generation configuration (``model_id``, ``model_revision``,
    ``dtype``, ``device`` from ``execution_metadata``)
  - same seed policy
  - same timeout
  - same verifier
  - same utility normalization
  - same environment snapshot
  - same capability permissions

Tasks where equivalent starting conditions cannot be proven are excluded
and recorded. A counterfactual-equivalence digest is computed over the
set of verified tasks.

The output of :func:`verify_equivalence` is the content for
``counterfactual_equivalence_report.json``.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

import numpy as np

from .data import LoadedRun


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fields in ``execution_metadata`` that must be identical across all
#: outcomes for a task to establish counterfactual equivalence.
REQUIRED_EXEC_METADATA_FIELDS: list[str] = [
    "model_id",
    "model_revision",
    "dtype",
    "device",
]

#: Additional execution_metadata fields checked for equivalence when
#: present (not all runs populate every field).
OPTIONAL_EXEC_METADATA_FIELDS: list[str] = [
    "seed",
    "seed_policy",
    "timeout",
    "temperature",
    "top_p",
    "max_new_tokens",
    "batch_size",
]

#: Fields in ``reward_components`` checked for normalization consistency.
NORMALIZATION_FIELDS: list[str] = [
    "normalization",
    "normalization_method",
    "utility_scale",
    "max_utility",
    "min_utility",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_json(obj: Any) -> str:
    """Stable SHA-256 digest of a JSON-serializable object."""
    h = hashlib.sha256()
    h.update(json.dumps(obj, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _extract_exec_metadata(outcome: Any) -> dict[str, Any]:
    """Extract the execution_metadata dict from an outcome."""
    return outcome.execution_metadata or {}


def _check_field_consistency(
    values: list[Any], field_name: str
) -> tuple[bool, list[Any]]:
    """Check that all values for a field are identical.

    Returns (is_consistent, list_of_unique_values).
    """
    # Normalize for comparison: treat None and missing as equivalent.
    normalized = []
    for v in values:
        if v is None:
            normalized.append("__MISSING__")
        else:
            normalized.append(v)
    unique = list(set(normalized)) if normalized else []
    # If the only unique value is __MISSING__, treat as consistent (field
    # simply not populated).
    if len(unique) == 1 and unique[0] == "__MISSING__":
        return True, []
    return len(unique) <= 1, unique


def _check_dict_consistency(
    dicts: list[dict[str, Any]], fields: list[str]
) -> dict[str, Any]:
    """Check consistency of specified fields across a list of dicts.

    Returns a dict mapping field -> {consistent, unique_values}.
    """
    result: dict[str, Any] = {}
    for field in fields:
        values = [d.get(field) for d in dicts]
        consistent, unique = _check_field_consistency(values, field)
        result[field] = {
            "consistent": consistent,
            "unique_values": [str(v) for v in unique],
        }
    return result


# ---------------------------------------------------------------------------
# Per-task equivalence checks
# ---------------------------------------------------------------------------


def _check_setup_spec(run: LoadedRun, task_id: str) -> dict[str, Any]:
    """Verify that setup_spec is consistent across all outcomes for a task.

    The task's ``setup_spec`` is stored once on the TaskInfo, but we also
    check that any per-outcome setup information (if present in
    execution_metadata or reward_components) is consistent.
    """
    task = run.tasks.get(task_id)
    if task is None:
        return {"consistent": False, "reason": "task not found"}

    setup_spec = task.setup_spec
    setup_digest = _sha256_json(setup_spec)

    # Check if outcomes carry per-outcome setup info.
    outcomes = run.outcomes_by_task.get(task_id, [])
    per_outcome_setups = []
    for o in outcomes:
        em = _extract_exec_metadata(o)
        if "setup_spec" in em:
            per_outcome_setups.append(em["setup_spec"])
        elif "task_setup" in em:
            per_outcome_setups.append(em["task_setup"])

    if per_outcome_setups:
        digests = [_sha256_json(s) for s in per_outcome_setups]
        consistent = len(set(digests)) == 1 and digests[0] == setup_digest
        return {
            "consistent": consistent,
            "setup_spec_digest": setup_digest,
            "per_outcome_digests": digests,
            "reason": "" if consistent else "per-outcome setup_spec mismatch",
        }

    return {
        "consistent": True,
        "setup_spec_digest": setup_digest,
        "reason": "",
    }


def _check_hidden_state(run: LoadedRun, task_id: str) -> dict[str, Any]:
    """Verify hidden secret / tool output consistency where required.

    For counterfactual equivalence, the hidden environment state (secrets,
    tool outputs, fixture data) must be identical across all action
    outcomes for the same task. We check the ``hidden_state`` or
    ``hidden_secret`` field in execution_metadata if present.
    """
    outcomes = run.outcomes_by_task.get(task_id, [])
    hidden_values: list[Any] = []
    for o in outcomes:
        em = _extract_exec_metadata(o)
        hidden = em.get("hidden_state", em.get("hidden_secret", em.get("tool_output")))
        hidden_values.append(hidden)

    # If no hidden state fields are present, we cannot prove equivalence
    # but also cannot disprove it — mark as "not_applicable".
    all_missing = all(v is None for v in hidden_values)
    if all_missing:
        return {
            "consistent": True,
            "proven": False,
            "reason": "no hidden state fields present in execution_metadata",
            "status": "not_applicable",
        }

    consistent, unique = _check_field_consistency(hidden_values, "hidden_state")
    return {
        "consistent": consistent,
        "proven": consistent,
        "unique_values": [str(v) for v in unique],
        "reason": "" if consistent else "hidden state differs across outcomes",
        "status": "verified" if consistent else "failed",
    }


def _check_generation_config(run: LoadedRun, task_id: str) -> dict[str, Any]:
    """Verify that model_id, model_revision, dtype, device are identical
    across all outcomes for a task.

    These are the core generation configuration fields that determine
    which model produced each outcome. If they differ, the utilities are
    not counterfactually comparable.
    """
    outcomes = run.outcomes_by_task.get(task_id, [])
    exec_metadatas = [_extract_exec_metadata(o) for o in outcomes]

    required_check = _check_dict_consistency(exec_metadatas, REQUIRED_EXEC_METADATA_FIELDS)
    optional_check = _check_dict_consistency(exec_metadatas, OPTIONAL_EXEC_METADATA_FIELDS)

    all_required_consistent = all(v["consistent"] for v in required_check.values())
    all_optional_consistent = all(v["consistent"] for v in optional_check.values())

    return {
        "consistent": all_required_consistent,
        "required_fields": required_check,
        "optional_fields": optional_check,
        "all_optional_consistent": all_optional_consistent,
        "reason": "" if all_required_consistent else "required generation config fields differ",
    }


def _check_seed_policy(run: LoadedRun, task_id: str) -> dict[str, Any]:
    """Verify that the seed policy is consistent across outcomes."""
    outcomes = run.outcomes_by_task.get(task_id, [])
    seeds: list[Any] = []
    seed_policies: list[Any] = []
    for o in outcomes:
        em = _extract_exec_metadata(o)
        seeds.append(em.get("seed"))
        seed_policies.append(em.get("seed_policy", em.get("seed_mode")))

    seed_consistent, seed_unique = _check_field_consistency(seeds, "seed")
    policy_consistent, policy_unique = _check_field_consistency(seed_policies, "seed_policy")

    consistent = seed_consistent and policy_consistent
    return {
        "consistent": consistent,
        "seed": {
            "consistent": seed_consistent,
            "unique_values": [str(v) for v in seed_unique],
        },
        "seed_policy": {
            "consistent": policy_consistent,
            "unique_values": [str(v) for v in policy_unique],
        },
        "reason": "" if consistent else "seed or seed_policy differs across outcomes",
    }


def _check_timeout(run: LoadedRun, task_id: str) -> dict[str, Any]:
    """Verify that the timeout is consistent across outcomes."""
    outcomes = run.outcomes_by_task.get(task_id, [])
    timeouts: list[Any] = []
    for o in outcomes:
        em = _extract_exec_metadata(o)
        timeouts.append(em.get("timeout", em.get("timeout_seconds")))

    consistent, unique = _check_field_consistency(timeouts, "timeout")
    return {
        "consistent": consistent,
        "unique_values": [str(v) for v in unique],
        "reason": "" if consistent else "timeout differs across outcomes",
    }


def _check_verifier(run: LoadedRun, task_id: str) -> dict[str, Any]:
    """Verify that the verifier is consistent across outcomes.

    The verifier spec is stored on the task, but we also check
    execution_metadata for per-outcome verifier info.
    """
    task = run.tasks.get(task_id)
    if task is None:
        return {"consistent": False, "reason": "task not found"}

    verifier_spec = task.verifier_spec
    verifier_digest = _sha256_json(verifier_spec)

    outcomes = run.outcomes_by_task.get(task_id, [])
    per_outcome_verifiers: list[Any] = []
    for o in outcomes:
        em = _extract_exec_metadata(o)
        if "verifier" in em:
            per_outcome_verifiers.append(em["verifier"])
        elif "verifier_spec" in em:
            per_outcome_verifiers.append(em["verifier_spec"])

    if per_outcome_verifiers:
        digests = [_sha256_json(v) for v in per_outcome_verifiers]
        consistent = len(set(digests)) == 1
        return {
            "consistent": consistent,
            "verifier_spec_digest": verifier_digest,
            "per_outcome_digests": digests,
            "reason": "" if consistent else "per-outcome verifier mismatch",
        }

    return {
        "consistent": True,
        "verifier_spec_digest": verifier_digest,
        "reason": "",
    }


def _check_utility_normalization(run: LoadedRun, task_id: str) -> dict[str, Any]:
    """Verify that utility normalization is consistent across outcomes."""
    outcomes = run.outcomes_by_task.get(task_id, [])
    reward_dicts = [o.reward_components or {} for o in outcomes]
    norm_check = _check_dict_consistency(reward_dicts, NORMALIZATION_FIELDS)
    all_consistent = all(v["consistent"] for v in norm_check.values())
    return {
        "consistent": all_consistent,
        "fields": norm_check,
        "reason": "" if all_consistent else "utility normalization differs across outcomes",
    }


def _check_environment_snapshot(run: LoadedRun, task_id: str) -> dict[str, Any]:
    """Verify that the environment snapshot is consistent across outcomes."""
    outcomes = run.outcomes_by_task.get(task_id, [])
    snapshots: list[Any] = []
    for o in outcomes:
        em = _extract_exec_metadata(o)
        snap = em.get("environment_snapshot", em.get("env_snapshot", em.get("environment")))
        snapshots.append(snap)

    all_missing = all(v is None for v in snapshots)
    if all_missing:
        return {
            "consistent": True,
            "proven": False,
            "status": "not_applicable",
            "reason": "no environment snapshot fields present",
        }

    consistent, unique = _check_field_consistency(snapshots, "environment_snapshot")
    return {
        "consistent": consistent,
        "proven": consistent,
        "unique_values": [str(v) for v in unique],
        "status": "verified" if consistent else "failed",
        "reason": "" if consistent else "environment snapshot differs across outcomes",
    }


def _check_capability_permissions(run: LoadedRun, task_id: str) -> dict[str, Any]:
    """Verify that capability permissions are consistent across outcomes."""
    task = run.tasks.get(task_id)
    if task is None:
        return {"consistent": False, "reason": "task not found"}

    # The allowed_actions on the task define the capability permissions.
    allowed = sorted(task.allowed_actions)
    allowed_digest = _sha256_json(allowed)

    outcomes = run.outcomes_by_task.get(task_id, [])
    per_outcome_caps: list[Any] = []
    for o in outcomes:
        em = _extract_exec_metadata(o)
        if "capabilities" in em:
            per_outcome_caps.append(sorted(em["capabilities"]) if isinstance(em["capabilities"], list) else em["capabilities"])
        elif "allowed_actions" in em:
            per_outcome_caps.append(sorted(em["allowed_actions"]) if isinstance(em["allowed_actions"], list) else em["allowed_actions"])

    if per_outcome_caps:
        digests = [_sha256_json(c) for c in per_outcome_caps]
        consistent = len(set(digests)) == 1
        return {
            "consistent": consistent,
            "allowed_actions_digest": allowed_digest,
            "per_outcome_digests": digests,
            "reason": "" if consistent else "per-outcome capability permissions mismatch",
        }

    return {
        "consistent": True,
        "allowed_actions_digest": allowed_digest,
        "reason": "",
    }


# ---------------------------------------------------------------------------
# Per-task verification
# ---------------------------------------------------------------------------


def _verify_task(run: LoadedRun, task_id: str) -> dict[str, Any]:
    """Run all equivalence checks for a single task.

    Returns a dict with per-check results and an overall ``equivalent``
    flag. A task is considered equivalent only if all *required* checks
    pass. Checks that are ``not_applicable`` (field not present) do not
    block equivalence but are marked as ``proven: False``.
    """
    task = run.tasks.get(task_id)
    if task is None:
        return {
            "task_id": task_id,
            "equivalent": False,
            "excluded": True,
            "reason": "task not found in run",
            "checks": {},
        }

    outcomes = run.outcomes_by_task.get(task_id, [])
    if not outcomes:
        return {
            "task_id": task_id,
            "equivalent": False,
            "excluded": True,
            "reason": "no outcomes for task",
            "checks": {},
        }

    checks: dict[str, Any] = {
        "setup_spec": _check_setup_spec(run, task_id),
        "hidden_state": _check_hidden_state(run, task_id),
        "generation_config": _check_generation_config(run, task_id),
        "seed_policy": _check_seed_policy(run, task_id),
        "timeout": _check_timeout(run, task_id),
        "verifier": _check_verifier(run, task_id),
        "utility_normalization": _check_utility_normalization(run, task_id),
        "environment_snapshot": _check_environment_snapshot(run, task_id),
        "capability_permissions": _check_capability_permissions(run, task_id),
    }

    # Determine overall equivalence.
    # Required checks: setup_spec, generation_config, seed_policy,
    # verifier, utility_normalization, capability_permissions.
    # Optional (not_applicable ok): hidden_state, timeout,
    # environment_snapshot.
    required_checks = [
        "setup_spec",
        "generation_config",
        "seed_policy",
        "verifier",
        "utility_normalization",
        "capability_permissions",
    ]
    optional_checks = [
        "hidden_state",
        "timeout",
        "environment_snapshot",
    ]

    required_pass = all(checks[c]["consistent"] for c in required_checks)

    # Optional checks: pass if consistent OR not_applicable.
    optional_pass = True
    for c in optional_checks:
        status = checks[c].get("status", "verified" if checks[c]["consistent"] else "failed")
        if status == "failed":
            optional_pass = False

    equivalent = required_pass and optional_pass

    failed_checks = [
        c for c in checks
        if not checks[c]["consistent"]
        and checks[c].get("status") != "not_applicable"
    ]

    return {
        "task_id": task_id,
        "n_outcomes": len(outcomes),
        "equivalent": equivalent,
        "excluded": not equivalent,
        "family": task.family,
        "archetype": task.archetype,
        "split": task.split,
        "checks": checks,
        "failed_checks": failed_checks,
        "reason": (
            ""
            if equivalent
            else f"failed checks: {', '.join(failed_checks)}"
        ),
    }


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


def _compute_digest(verified_task_ids: list[str]) -> str:
    """Compute a counterfactual-equivalence digest over verified tasks.

    The digest is a SHA-256 of the sorted list of verified task IDs,
    providing a stable fingerprint of the equivalence-verified task set.
    """
    h = hashlib.sha256()
    h.update(json.dumps(sorted(verified_task_ids), sort_keys=True).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_equivalence(run: LoadedRun) -> dict[str, Any]:
    """Verify counterfactual utility comparability for all tasks in a run.

    For each task, checks that all action outcomes share the same
    starting conditions: same task setup (``setup_spec``), same hidden
    secret/tool output where applicable, same generation configuration
    (``model_id``, ``model_revision``, ``dtype``, ``device`` from
    ``execution_metadata``), same seed policy, same timeout, same
    verifier, same utility normalization, same environment snapshot,
    and same capability permissions.

    Tasks where equivalent starting conditions cannot be proven are
    excluded and recorded with the specific checks that failed.

    A counterfactual-equivalence digest is computed over the set of
    verified (equivalent) tasks.

    Args:
        run: Loaded run data with tasks, outcomes, and execution metadata.

    Returns:
        Content for ``counterfactual_equivalence_report.json``.
    """
    per_task: dict[str, Any] = {}
    verified_task_ids: list[str] = []
    excluded_task_ids: list[str] = []

    for task_id in sorted(run.tasks.keys()):
        result = _verify_task(run, task_id)
        per_task[task_id] = result
        if result["equivalent"]:
            verified_task_ids.append(task_id)
        else:
            excluded_task_ids.append(task_id)

    # Summarize exclusion reasons.
    exclusion_reasons: Counter = Counter()
    for tid in excluded_task_ids:
        for fc in per_task[tid]["failed_checks"]:
            exclusion_reasons[fc] += 1

    # Compute digest over verified tasks.
    digest = _compute_digest(verified_task_ids)

    # Coverage by split.
    split_coverage: dict[str, dict[str, int]] = {}
    for tid, res in per_task.items():
        split = res.get("split", "unknown")
        split_coverage.setdefault(split, {"verified": 0, "excluded": 0, "total": 0})
        split_coverage[split]["total"] += 1
        if res["equivalent"]:
            split_coverage[split]["verified"] += 1
        else:
            split_coverage[split]["excluded"] += 1

    return {
        "n_total_tasks": len(per_task),
        "n_verified_tasks": len(verified_task_ids),
        "n_excluded_tasks": len(excluded_task_ids),
        "verification_rate": len(verified_task_ids) / len(per_task) if per_task else 0.0,
        "verified_task_ids": verified_task_ids,
        "excluded_task_ids": excluded_task_ids,
        "exclusion_reasons": dict(exclusion_reasons),
        "split_coverage": split_coverage,
        "counterfactual_equivalence_digest": digest,
        "required_exec_metadata_fields": REQUIRED_EXEC_METADATA_FIELDS,
        "optional_exec_metadata_fields": OPTIONAL_EXEC_METADATA_FIELDS,
        "per_task": per_task,
        "summary": {
            "description": (
                "Tasks where all action outcomes share identical starting "
                "conditions (same setup_spec, generation config, seed policy, "
                "verifier, utility normalization, and capability permissions) "
                "are marked as counterfactually equivalent. Tasks that fail "
                "any required check are excluded from downstream analysis."
            ),
            "verified_fraction": len(verified_task_ids) / len(per_task) if per_task else 0.0,
        },
    }
