"""Counterfactual completeness audit for v0.4.3.

Validates that the counterfactual outcome table is *complete* and
*comparable* across actions for every test task, so that oracle headroom
and policy comparisons are well-defined.

For each test task the audit verifies:

* **all eligible actions measured** — every action in ``allowed_actions``
  has an executed outcome.
* **same provider/model revision** — all outcomes for a task share the
  same ``model_revision`` in their ``execution_metadata``.
* **same task setup digest** — ``setup_spec`` is consistent across
  actions (it should be identical because it is the same task).
* **same verifier version** — ``verifier_spec`` is consistent across
  actions.
* **same utility version** — utility values are produced by the same
  utility version (consistent ``utility_version`` / scale).
* **same generation config** — the ``generation_config`` inside
  ``execution_metadata`` is consistent across actions.
* **same timeout** — the execution timeout is consistent across actions.
* **no action-specific hidden-answer leakage** — ``setup_spec`` must not
  contain action-specific secrets that would leak the correct action.

Each task is marked:

* ``COMPLETE`` — all checks pass.
* ``INCOMPLETE`` — some eligible actions are not measured.
* ``NONCOMPARABLE`` — setup / verifier / model mismatch across actions.
* ``UNVERIFIABLE`` — cannot determine completeness.

Only ``COMPLETE`` tasks may contribute to the primary oracle headroom.
The audit reports how headroom changes when restricted to ``COMPLETE``
tasks (oracle-sham headroom on all test tasks vs. only ``COMPLETE``
tasks).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .data import LoadedRun, TaskInfo, Outcome
from .canonical_evaluator import (
    evaluate_policy_from_counterfactuals,
    make_oracle_fn,
    make_executed_sham_fn,
    compute_headroom_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_digest(obj: Any) -> str:
    """Stable SHA-256 digest of a JSON-serializable object."""
    try:
        payload = json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(obj)
    return hashlib.sha256(payload.encode()).hexdigest()


def _executed_outcomes(run: LoadedRun, task_id: str) -> list[Outcome]:
    return [
        o for o in run.outcomes_by_task.get(task_id, [])
        if o.availability == "executed"
    ]


def _missing_actions(run: LoadedRun, task: TaskInfo) -> list[str]:
    executed_actions = {
        o.action_id for o in _executed_outcomes(run, task.task_id)
    }
    return [a for a in task.allowed_actions if a not in executed_actions]


def _model_revision(outcome: Outcome) -> str:
    em = outcome.execution_metadata or {}
    return str(em.get("model_revision", em.get("model", "unknown")))


def _generation_config(outcome: Outcome) -> Any:
    em = outcome.execution_metadata or {}
    return em.get("generation_config", em.get("generation", {}))


def _timeout(outcome: Outcome) -> Any:
    em = outcome.execution_metadata or {}
    return em.get("timeout", em.get("max_tokens", em.get("timeout_seconds", "unknown")))


def _utility_version(outcome: Outcome) -> str:
    em = outcome.execution_metadata or {}
    rc = outcome.reward_components or {}
    return str(
        em.get("utility_version", rc.get("utility_version", "normalized_v1"))
    )


def _setup_digest(task: TaskInfo) -> str:
    return _json_digest(task.setup_spec)


def _verifier_digest(task: TaskInfo) -> str:
    return _json_digest(task.verifier_spec)


def _action_specific_keys(setup_spec: dict[str, Any]) -> list[str]:
    """Return keys in setup_spec that look action-specific (leakage risk)."""
    leakage_keys: list[str] = []
    suspicious = {"answer", "correct_action", "best_action", "solution",
                  "expected_action", "oracle_action", "secret_action",
                  "target_action", "label", "gold_action"}
    for key, value in (setup_spec or {}).items():
        lk = key.lower()
        if lk in suspicious:
            leakage_keys.append(key)
            continue
        # Nested dicts that reference action ids.
        if isinstance(value, dict):
            for sub in value:
                if str(sub).lower() in suspicious:
                    leakage_keys.append(f"{key}.{sub}")
    return leakage_keys


# ---------------------------------------------------------------------------
# Per-task audit
# ---------------------------------------------------------------------------


def _audit_task(run: LoadedRun, task: TaskInfo) -> dict[str, Any]:
    """Audit a single test task for counterfactual completeness."""
    executed = _executed_outcomes(run, task.task_id)
    missing = _missing_actions(run, task)

    checks: dict[str, Any] = {
        "all_eligible_actions_measured": {
            "passed": len(missing) == 0,
            "n_allowed": len(task.allowed_actions),
            "n_executed": len(executed),
        },
    }

    # If there are no executed outcomes at all we cannot run the
    # comparability checks.
    if not executed:
        for name in (
            "same_model_revision",
            "same_task_setup_digest",
            "same_verifier_version",
            "same_utility_version",
            "same_generation_config",
            "same_timeout",
            "no_action_specific_hidden_answer_leakage",
        ):
            checks[name] = {"passed": False, "reason": "no_executed_outcomes"}
        return {
            "status": "UNVERIFIABLE",
            "checks": checks,
            "missing_actions": missing,
        }

    # --- same model revision ---
    revisions = {_model_revision(o) for o in executed}
    checks["same_model_revision"] = {
        "passed": len(revisions) <= 1,
        "values": sorted(revisions),
    }

    # --- same task setup digest ---
    # setup_spec lives on the task, so it is identical by construction;
    # we still verify that all outcomes reference the same task setup.
    setup_digests = {
        _json_digest(o.execution_metadata.get("setup_spec", task.setup_spec))
        for o in executed
    }
    checks["same_task_setup_digest"] = {
        "passed": len(setup_digests) <= 1,
        "values": sorted(setup_digests),
    }

    # --- same verifier version ---
    verifier_digests = {
        _json_digest(o.execution_metadata.get("verifier_spec", task.verifier_spec))
        for o in executed
    }
    checks["same_verifier_version"] = {
        "passed": len(verifier_digests) <= 1,
        "values": sorted(verifier_digests),
    }

    # --- same utility version ---
    utility_versions = {_utility_version(o) for o in executed}
    checks["same_utility_version"] = {
        "passed": len(utility_versions) <= 1,
        "values": sorted(utility_versions),
    }

    # --- same generation config ---
    gen_configs = {_json_digest(_generation_config(o)) for o in executed}
    checks["same_generation_config"] = {
        "passed": len(gen_configs) <= 1,
        "values": sorted(gen_configs),
    }

    # --- same timeout ---
    timeouts = {str(_timeout(o)) for o in executed}
    checks["same_timeout"] = {
        "passed": len(timeouts) <= 1,
        "values": sorted(timeouts),
    }

    # --- no action-specific hidden answer leakage ---
    leakage_keys = _action_specific_keys(task.setup_spec)
    checks["no_action_specific_hidden_answer_leakage"] = {
        "passed": len(leakage_keys) == 0,
        "leakage_keys": leakage_keys,
    }

    # --- Determine status ---
    all_measured = len(missing) == 0
    comparability_checks = (
        "same_model_revision",
        "same_task_setup_digest",
        "same_verifier_version",
        "same_utility_version",
        "same_generation_config",
        "same_timeout",
        "no_action_specific_hidden_answer_leakage",
    )
    comparable = all(
        checks[name].get("passed", False) for name in comparability_checks
    )

    if not all_measured and not comparable:
        # If both incompleteness and non-comparability are present,
        # non-comparability is the more severe diagnosis.
        status = "NONCOMPARABLE"
    elif not all_measured:
        status = "INCOMPLETE"
    elif not comparable:
        status = "NONCOMPARABLE"
    else:
        status = "COMPLETE"

    return {
        "status": status,
        "checks": checks,
        "missing_actions": missing,
    }


# ---------------------------------------------------------------------------
# Headroom comparison
# ---------------------------------------------------------------------------


def _oracle_sham_headroom(
    run: LoadedRun, task_ids: list[str]
) -> dict[str, Any]:
    """Compute oracle-sham headroom metrics on a task subset."""
    if not task_ids:
        return compute_headroom_metrics(
            candidate_mean=0.0,
            reference_mean=0.0,
            oracle_mean=0.0,
            n_tasks=0,
            reference_policy_id="sham",
            candidate_policy_id="oracle",
        )
    oracle_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_oracle_fn(run), "oracle",
    )
    sham_result = evaluate_policy_from_counterfactuals(
        run, task_ids, make_executed_sham_fn(run), "sham",
    )
    return compute_headroom_metrics(
        candidate_mean=oracle_result.mean_utility,
        reference_mean=sham_result.mean_utility,
        oracle_mean=oracle_result.mean_utility,
        n_tasks=oracle_result.n_complete,
        task_ids_digest=oracle_result.task_ids_digest,
        reference_policy_id="sham",
        candidate_policy_id="oracle",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_counterfactual_completeness(run: LoadedRun) -> dict[str, Any]:
    """Validate counterfactual completeness across all test tasks.

    Args:
        run: The loaded run with counterfactual outcomes and tasks.

    Returns:
        A JSON-serializable ``counterfactual_completeness`` dict with:

        * ``per_task`` — dict of task_id -> {status, checks, missing_actions}.
        * ``summary`` — dict of status -> count.
        * ``headroom_all_tasks`` — oracle-sham headroom on all test tasks.
        * ``headroom_complete_only`` — oracle-sham headroom on COMPLETE tasks.
        * ``interpretation`` — human-readable summary.
    """
    test_task_ids = run.test_task_ids

    per_task: dict[str, dict[str, Any]] = {}
    for tid in test_task_ids:
        task = run.tasks.get(tid)
        if task is None:
            per_task[tid] = {
                "status": "UNVERIFIABLE",
                "checks": {"task_present": {"passed": False}},
                "missing_actions": [],
            }
            continue
        per_task[tid] = _audit_task(run, task)

    # Summary counts.
    summary: dict[str, int] = {
        "COMPLETE": 0,
        "INCOMPLETE": 0,
        "NONCOMPARABLE": 0,
        "UNVERIFIABLE": 0,
    }
    for info in per_task.values():
        status = info["status"]
        summary[status] = summary.get(status, 0) + 1

    # Headroom comparisons.
    complete_task_ids = sorted(
        tid for tid, info in per_task.items() if info["status"] == "COMPLETE"
    )
    headroom_all = _oracle_sham_headroom(run, test_task_ids)
    headroom_complete = _oracle_sham_headroom(run, complete_task_ids)

    # Interpretation.
    n_complete = summary["COMPLETE"]
    n_total = len(test_task_ids)
    frac_complete = n_complete / n_total if n_total else 0.0
    h_all = headroom_all.get("recovered_headroom_fraction")
    h_comp = headroom_complete.get("recovered_headroom_fraction")
    if n_total == 0:
        interpretation = "No test tasks present; completeness cannot be assessed."
    elif n_complete == 0:
        interpretation = (
            "No test tasks are counterfactually COMPLETE. Primary oracle "
            "headroom is not estimable on comparable tasks."
        )
    else:
        interpretation = (
            f"{n_complete}/{n_total} test tasks ({frac_complete:.1%}) are "
            f"counterfactually COMPLETE. "
        )
        if h_all is not None and h_comp is not None:
            interpretation += (
                f"Oracle-sham headroom fraction is {h_all:.4f} on all test "
                f"tasks vs {h_comp:.4f} on COMPLETE-only tasks. "
            )
            if abs(h_all - h_comp) > 1e-6:
                interpretation += (
                    "Restricting to COMPLETE tasks materially changes "
                    "headroom; only COMPLETE tasks may contribute to the "
                    "primary oracle headroom."
                )
            else:
                interpretation += (
                    "Restricting to COMPLETE tasks does not materially "
                    "change headroom."
                )
        else:
            interpretation += (
                "Headroom fraction is not estimable on one or both subsets "
                "(denominator too small)."
            )

    return {
        "per_task": per_task,
        "summary": summary,
        "headroom_all_tasks": headroom_all,
        "headroom_complete_only": headroom_complete,
        "complete_task_ids": complete_task_ids,
        "interpretation": interpretation,
    }
