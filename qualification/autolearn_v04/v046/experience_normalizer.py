"""Experience normalizer for v0.4.6.

Creates a normalized task-level experience artifact from action-level rows.

Each output row represents a single experience task with aggregated
fields including feature vector, eligible actions, action utilities,
best/second-best action, utility margin, quality scores, and provenance.

For synthetic fixtures, all quality scores default to 1.0:
    q_verifier=1.0, q_execution=1.0, q_counterfactual=1.0,
    q_isolation=1.0, q_provenance=1.0, q_total=1.0, final_weight=1.0
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.common.evidence_origin import EvidenceOrigin

# Schema version for normalized experience rows.
SCHEMA_VERSION = "v2"

# Quality scores for synthetic fixtures.
SYNTHETIC_Q_VERIFIER = 1.0
SYNTHETIC_Q_EXECUTION = 1.0
SYNTHETIC_Q_COUNTERFACTUAL = 1.0
SYNTHETIC_Q_ISOLATION = 1.0
SYNTHETIC_Q_PROVENANCE = 1.0
SYNTHETIC_Q_TOTAL = 1.0
SYNTHETIC_FINAL_WEIGHT = 1.0


def _build_feature_vector(action_row: dict) -> list[float]:
    """Build a feature vector from an action-level experience row.

    Combines conversation_features, memory_features, and prompt_features
    from the row's ``state`` dict into a single flat list.
    """
    state = action_row.get("state", {})
    if not isinstance(state, dict):
        return []

    fv: list[float] = []
    for key in ("conversation_features", "memory_features", "prompt_features"):
        features = state.get(key)
        if isinstance(features, list):
            fv.extend(float(x) for x in features if isinstance(x, (int, float)))
    return fv


def _compute_feature_schema_digest(feature_vector: list[float]) -> str:
    """Compute a deterministic digest of the feature vector shape."""
    payload = json.dumps(
        {"feature_dim": len(feature_vector)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _compute_row_hash(row: dict) -> str:
    """Compute a deterministic SHA-256 hash of a row dict."""
    blob = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _is_synthetic(counterfactual_outcomes: list[dict]) -> bool:
    """Check if the evidence is synthetic based on counterfactual outcomes."""
    for row in counterfactual_outcomes:
        origin = str(row.get("provider_class", "")).upper()
        runtime = str(row.get("runtime_type", "")).lower()
        model_id = str(row.get("model_id", "")).lower()
        if (
            origin == "SYNTHETIC"
            or runtime == "synthetic"
            or "synthetic" in model_id
        ):
            return True
    return False


def _get_task_info(
    task_id: str,
    counterfactual_outcomes: list[dict],
) -> dict[str, Any]:
    """Get task-level info (split, family, archetype) from counterfactual rows."""
    for row in counterfactual_outcomes:
        if row.get("task_id") == task_id:
            return {
                "split": row.get("split", ""),
                "family": row.get("family", ""),
                "archetype": row.get("archetype", ""),
            }
    return {"split": "", "family": "", "archetype": ""}


def normalize_experience_rows(
    experience_action_rows: list[dict],
    counterfactual_outcomes: list[dict],
    benchmark_manifest: dict,
    split_manifest: dict,
    run_id: str,
) -> list[dict]:
    """Create normalized task-level experience rows from action-level rows.

    Parameters
    ----------
    experience_action_rows:
        Action-level experience rows (one per task/action pair).
    counterfactual_outcomes:
        Counterfactual outcome rows (used for origin detection and
        task metadata).
    benchmark_manifest:
        The benchmark manifest (for task metadata).
    split_manifest:
        The split manifest (for split membership).
    run_id:
        The run identifier.

    Returns
    -------
    list[dict]
        One row per experience task with all required fields.
    """
    if not experience_action_rows:
        return []

    is_synthetic = _is_synthetic(counterfactual_outcomes)

    # Group action rows by task_id.
    by_task: dict[str, list[dict]] = {}
    for row in experience_action_rows:
        tid = row.get("task_id")
        if tid is None:
            continue
        by_task.setdefault(tid, []).append(row)

    # Build benchmark task lookup.
    bm_tasks: dict[str, dict] = {}
    if isinstance(benchmark_manifest, dict):
        for t in benchmark_manifest.get("tasks", []):
            if isinstance(t, dict) and "task_id" in t:
                bm_tasks[t["task_id"]] = t

    result: list[dict] = []

    for task_id in sorted(by_task.keys()):
        action_rows = by_task[task_id]
        task_info = _get_task_info(task_id, counterfactual_outcomes)

        # Use the first action row for feature vector (all actions for a
        # task share the same initial state).
        first_row = action_rows[0]
        feature_vector = _build_feature_vector(first_row)
        feature_schema_digest = _compute_feature_schema_digest(feature_vector)

        # Eligible actions and utilities.
        eligible_actions: list[str] = []
        action_utilities: dict[str, float] = {}
        action_verified_success: dict[str, bool] = {}
        verifier_names: set[str] = set()
        counterfactual_row_hashes: list[str] = []
        source_artifact_hashes: list[str] = []

        for ar in action_rows:
            action = ar.get("action", "")
            if action:
                eligible_actions.append(action)
                utility = ar.get("utility")
                if isinstance(utility, (int, float)):
                    action_utilities[action] = float(utility)
                action_verified_success[action] = bool(
                    ar.get("verified_success", False)
                )

            # Collect verifier names.
            prov = ar.get("provenance", {})
            if isinstance(prov, dict):
                src = prov.get("source", "")
                if src:
                    verifier_names.add(src)

            # Collect experience_id as source artifact hash.
            exp_id = ar.get("experience_id")
            if exp_id:
                source_artifact_hashes.append(exp_id)

            # Compute row hash.
            counterfactual_row_hashes.append(_compute_row_hash(ar))

        # Determine best and second-best action by utility.
        sorted_actions = sorted(
            action_utilities.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        best_action = sorted_actions[0][0] if sorted_actions else ""
        best_utility = sorted_actions[0][1] if sorted_actions else 0.0
        second_best_action = sorted_actions[1][0] if len(sorted_actions) > 1 else ""
        second_best_utility = sorted_actions[1][1] if len(sorted_actions) > 1 else 0.0
        utility_margin = best_utility - second_best_utility

        # Label status: "verified" if best action has verified_success=True.
        label_status = "verified" if action_verified_success.get(best_action, False) else "unverified"

        # Quality scores.
        if is_synthetic:
            q_verifier = SYNTHETIC_Q_VERIFIER
            q_execution = SYNTHETIC_Q_EXECUTION
            q_counterfactual = SYNTHETIC_Q_COUNTERFACTUAL
            q_isolation = SYNTHETIC_Q_ISOLATION
            q_provenance = SYNTHETIC_Q_PROVENANCE
            q_total = SYNTHETIC_Q_TOTAL
            final_weight = SYNTHETIC_FINAL_WEIGHT
        else:
            # For non-synthetic (real-model) evidence, propagate quality
            # fields from the action-level rows.  Each action row may carry
            # q_verifier, q_execution, etc.  We aggregate by taking the
            # maximum across actions (best quality observed).
            q_verifier = 0.0
            q_execution = 0.0
            q_counterfactual = 0.0
            q_isolation = 0.0
            q_provenance = 0.0
            q_total = 0.0
            for ar in action_rows:
                for qfield, _ in (
                    ("q_verifier", None),
                    ("q_execution", None),
                    ("q_counterfactual", None),
                    ("q_isolation", None),
                    ("q_provenance", None),
                    ("q_total", None),
                ):
                    val = ar.get(qfield)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        # Update the corresponding local variable.
                        if qfield == "q_verifier":
                            q_verifier = max(q_verifier, float(val))
                        elif qfield == "q_execution":
                            q_execution = max(q_execution, float(val))
                        elif qfield == "q_counterfactual":
                            q_counterfactual = max(q_counterfactual, float(val))
                        elif qfield == "q_isolation":
                            q_isolation = max(q_isolation, float(val))
                        elif qfield == "q_provenance":
                            q_provenance = max(q_provenance, float(val))
                        elif qfield == "q_total":
                            q_total = max(q_total, float(val))
            # If q_total is still 0 (no quality fields on action rows),
            # fall back to a small positive value so weights are never 0.
            if q_total <= 0.0:
                q_total = 1.0
            final_weight = q_total

        result.append({
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "task_id": task_id,
            "split": task_info.get("split", first_row.get("split", "")),
            "family": task_info.get("family", first_row.get("family", "")),
            "archetype": task_info.get("archetype", first_row.get("archetype", "")),
            "feature_vector": feature_vector,
            "feature_schema_digest": feature_schema_digest,
            "eligible_actions": eligible_actions,
            "action_utilities": action_utilities,
            "action_verified_success": action_verified_success,
            "best_action": best_action,
            "second_best_action": second_best_action,
            "utility_margin": utility_margin,
            "label_status": label_status,
            "q_verifier": q_verifier,
            "q_execution": q_execution,
            "q_counterfactual": q_counterfactual,
            "q_isolation": q_isolation,
            "q_provenance": q_provenance,
            "q_total": q_total,
            "final_weight": final_weight,
            "verifier_names": sorted(verifier_names),
            "counterfactual_row_hashes": counterfactual_row_hashes,
            "source_artifact_hashes": source_artifact_hashes,
        })

    return result
