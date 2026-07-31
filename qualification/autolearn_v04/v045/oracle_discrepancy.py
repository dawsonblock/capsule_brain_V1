"""Oracle discrepancy investigation for v0.4.5.

Resolves conflicting oracle-minus-sham headroom claims.

Previous claims included oracle-minus-sham headroom = 0.632, but the current
placeholder package implies oracle mean = 5.112, sham mean = 5.069,
oracle-minus-sham = 0.043. Both cannot represent the same task set and
utility version.

This module traces every oracle value to its artifact, run ID, task set
digest, utility version, oracle construction method, action completeness
status, and source hash, then determines the most likely cause of the
discrepancy.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from qualification.autolearn_v04.common.headroom import compute_task_ids_digest


# Known historical headroom claim that this module reconciles against.
HISTORICAL_ORACLE_MINUS_SHAM_CLAIM = 0.632

# Placeholder package values that contradict the historical claim.
PLACEHOLDER_ORACLE_MEAN = 5.112
PLACEHOLDER_SHAM_MEAN = 5.069
PLACEHOLDER_ORACLE_MINUS_SHAM = PLACEHOLDER_ORACLE_MEAN - PLACEHOLDER_SHAM_MEAN  # 0.043

# Tolerance for floating-point comparison of headroom values.
_HEADROOM_TOLERANCE = 1e-6


def _is_finite(v: Any) -> bool:
    if v is None or isinstance(v, bool):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _safe_float(v: Any) -> float | None:
    if not _is_finite(v):
        return None
    return float(v)


def _looks_placeholder(value: Any) -> bool:
    """Heuristic: detect placeholder sentinel values."""
    if value is None:
        return False
    s = str(value).strip().lower()
    placeholders = {
        "placeholder",
        "todo",
        "tbd",
        "stub",
        "dummy",
        "n/a",
        "na",
        "none",
        "unknown",
        "0.0",
        "0",
    }
    return s in placeholders


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _extract_mean(results: dict) -> float | None:
    """Extract a mean utility from a results dict, preferring explicit fields."""
    if not isinstance(results, dict):
        return None
    for key in ("mean_utility", "mean", "aggregate_mean", "summary_mean"):
        v = _safe_float(results.get(key))
        if v is not None:
            return v
    # Fall back to per-task mean if present.
    per_task = results.get("per_task") or results.get("task_results") or []
    utils = [_safe_float(r.get("utility")) for r in per_task]
    utils = [u for u in utils if u is not None]
    return _mean(utils)


def _extract_task_ids(outcomes: list[dict]) -> list[str]:
    ids: list[str] = []
    for o in outcomes:
        tid = o.get("task_id") or o.get("id") or o.get("task")
        if tid is not None:
            ids.append(str(tid))
    return ids


def _extract_utility_version(results: dict, manifest: dict | None) -> str:
    for src in (results, manifest or {}):
        if isinstance(src, dict):
            for key in ("utility_version", "utility_schema_version", "normalization_version"):
                v = src.get(key)
                if v:
                    return str(v)
    return "unknown"


def _extract_task_set_digest(results: dict, outcomes: list[dict]) -> str:
    # Prefer explicit digest fields.
    for key in ("task_set_digest", "task_ids_digest", "task_digest"):
        v = results.get(key) if isinstance(results, dict) else None
        if v:
            return str(v)
    # Compute from outcomes if available.
    tids = _extract_task_ids(outcomes)
    if tids:
        return compute_task_ids_digest(tids)
    return ""


def _extract_artifact_trace(results: dict) -> dict:
    """Trace an oracle/sham result to its artifact metadata."""
    return {
        "artifact": results.get("artifact") or results.get("artifact_path") or "",
        "run_id": results.get("run_id") or results.get("run") or "",
        "source_hash": results.get("source_hash") or results.get("source_sha256") or "",
        "oracle_construction_method": results.get("oracle_construction_method")
        or results.get("construction_method")
        or "",
        "action_completeness": results.get("action_completeness")
        or results.get("action_matrix_status")
        or "",
    }


def _is_placeholder_outcomes(outcomes: list[dict]) -> bool:
    if not outcomes:
        return True
    # If every row is flagged placeholder, treat as placeholder.
    flagged = 0
    for o in outcomes:
        if o.get("placeholder") is True or _looks_placeholder(o.get("placeholder")):
            flagged += 1
        elif _looks_placeholder(o.get("source")) or _looks_placeholder(o.get("provenance")):
            flagged += 1
    return flagged == len(outcomes)


def _task_level_best_action_mean(outcomes: list[dict]) -> float | None:
    """Compute oracle mean as the per-task maximum measured utility.

    This is the canonical oracle construction: for each task, select the
    maximum measured utility among complete eligible actions, then average
    across tasks. This must NOT use the aggregate mean.
    """
    if not outcomes:
        return None
    per_task_utils: dict[str, list[float]] = {}
    for o in outcomes:
        tid = o.get("task_id") or o.get("id") or o.get("task")
        util = _safe_float(o.get("utility"))
        if tid is None or util is None:
            continue
        per_task_utils.setdefault(str(tid), []).append(util)
    if not per_task_utils:
        return None
    maxima = [max(utils) for utils in per_task_utils.values()]
    return _mean(maxima)


def investigate_oracle_discrepancy(
    oracle_results: dict,
    sham_results: dict,
    counterfactual_outcomes: list[dict],
    evidence_manifest: dict,
    benchmark_manifest: dict | None = None,
) -> dict:
    """Investigate the oracle-minus-sham headroom discrepancy.

    Traces every oracle value to its artifact, run ID, task set digest,
    utility version, oracle construction method, action completeness status,
    and source hash. Determines which of the candidate causes explains the
    discrepancy between the historical 0.632 claim and the placeholder
    0.043 value.

    Returns a dict with status, oracle_mean, sham_mean, oracle_minus_sham,
    task_set_digest, utility_version, discrepancy_found, possible_causes,
    resolution, and errors.
    """
    errors: list[str] = []
    possible_causes: list[str] = []

    # BLOCKED if no task-level outcomes are available.
    if not counterfactual_outcomes or _is_placeholder_outcomes(counterfactual_outcomes):
        return {
            "status": "BLOCKED",
            "oracle_mean": _extract_mean(oracle_results),
            "sham_mean": _extract_mean(sham_results),
            "oracle_minus_sham": None,
            "task_set_digest": _extract_task_set_digest(oracle_results, counterfactual_outcomes),
            "utility_version": _extract_utility_version(oracle_results, benchmark_manifest),
            "discrepancy_found": True,
            "possible_causes": ["cannot resolve oracle discrepancy without task-level outcomes"],
            "resolution": (
                "cannot resolve oracle discrepancy without task-level outcomes"
            ),
            "errors": errors,
        }

    oracle_trace = _extract_artifact_trace(oracle_results)
    sham_trace = _extract_artifact_trace(sham_results)

    oracle_mean_reported = _extract_mean(oracle_results)
    sham_mean_reported = _extract_mean(sham_results)

    # Canonical oracle from task-level best action.
    oracle_mean_canonical = _task_level_best_action_mean(counterfactual_outcomes)

    # Sham mean from actual task rows (if present), else reported.
    sham_task_utils: list[float] = []
    for o in counterfactual_outcomes:
        if o.get("policy") == "sham" or o.get("policy_id") == "sham":
            u = _safe_float(o.get("utility"))
            if u is not None:
                sham_task_utils.append(u)
    sham_mean_canonical = _mean(sham_task_utils) if sham_task_utils else sham_mean_reported

    # Decide which means to use for discrepancy analysis.
    oracle_mean = oracle_mean_canonical if oracle_mean_canonical is not None else oracle_mean_reported
    sham_mean = sham_mean_canonical if sham_mean_canonical is not None else sham_mean_reported

    oracle_minus_sham: float | None = None
    if oracle_mean is not None and sham_mean is not None:
        oracle_minus_sham = oracle_mean - sham_mean

    task_set_digest = _extract_task_set_digest(oracle_results, counterfactual_outcomes)
    utility_version = _extract_utility_version(oracle_results, benchmark_manifest)

    # Compare against historical claim.
    historical = HISTORICAL_ORACLE_MINUS_SHAM_CLAIM
    discrepancy_found = False
    if oracle_minus_sham is not None:
        if abs(oracle_minus_sham - historical) > _HEADROOM_TOLERANCE:
            discrepancy_found = True

    # Also compare against placeholder values.
    placeholder_match = (
        oracle_mean_reported is not None
        and sham_mean_reported is not None
        and abs(oracle_mean_reported - PLACEHOLDER_ORACLE_MEAN) < _HEADROOM_TOLERANCE
        and abs(sham_mean_reported - PLACEHOLDER_SHAM_MEAN) < _HEADROOM_TOLERANCE
    )

    # --- Cause analysis -------------------------------------------------

    # Cause 1: the 0.632 value used another sham.
    sham_ids = _extract_task_ids(
        [o for o in counterfactual_outcomes if o.get("policy") == "sham"]
    )
    if sham_mean_reported is not None and sham_mean_canonical is not None:
        if abs(sham_mean_reported - sham_mean_canonical) > _HEADROOM_TOLERANCE:
            possible_causes.append(
                "the 0.632 value used another sham "
                f"(reported sham mean={sham_mean_reported}, "
                f"task-level sham mean={sham_mean_canonical})"
            )
    if not sham_ids:
        possible_causes.append(
            "the 0.632 value used another sham (no sham task rows in counterfactual outcomes)"
        )

    # Cause 2: the 5.112 mean is placeholder data.
    if placeholder_match:
        possible_causes.append(
            f"the 5.112 mean is placeholder data "
            f"(oracle={oracle_mean_reported}, sham={sham_mean_reported} match placeholder sentinels)"
        )
    if _looks_placeholder(oracle_results.get("source")) or _looks_placeholder(
        oracle_results.get("provenance")
    ):
        possible_causes.append("the 5.112 mean is placeholder data (oracle source flagged placeholder)")
    if oracle_mean_canonical is not None and oracle_mean_reported is not None:
        if abs(oracle_mean_canonical - oracle_mean_reported) > _HEADROOM_TOLERANCE:
            possible_causes.append(
                "the 5.112 mean is placeholder data "
                f"(reported oracle mean={oracle_mean_reported} disagrees with "
                f"task-level best-action mean={oracle_mean_canonical})"
            )

    # Cause 3: normalization changed.
    uv_historical = ""
    if isinstance(evidence_manifest, dict):
        uv_historical = str(evidence_manifest.get("utility_version", ""))
    if utility_version and uv_historical and utility_version != uv_historical:
        possible_causes.append(
            f"normalization changed (historical utility_version={uv_historical}, "
            f"current utility_version={utility_version})"
        )
    if utility_version == "unknown":
        possible_causes.append(
            "normalization changed (utility version unknown in current artifacts)"
        )

    # Cause 4: task sets differ.
    historical_digest = ""
    if isinstance(evidence_manifest, dict):
        historical_digest = str(
            evidence_manifest.get("task_set_digest")
            or evidence_manifest.get("task_ids_digest")
            or ""
        )
    if historical_digest and task_set_digest and historical_digest != task_set_digest:
        possible_causes.append(
            f"task sets differ (historical digest={historical_digest[:12]}..., "
            f"current digest={task_set_digest[:12]}...)"
        )
    if not task_set_digest:
        possible_causes.append(
            "task sets differ (current task set digest unavailable)"
        )

    # Cause 5: an earlier module used task-level best action correctly while
    # the current summary does not.
    if oracle_mean_canonical is not None and oracle_mean_reported is not None:
        if abs(oracle_mean_canonical - oracle_mean_reported) > _HEADROOM_TOLERANCE:
            possible_causes.append(
                "an earlier module used task-level best action correctly while "
                f"the current summary does not (task-level best-action oracle mean="
                f"{oracle_mean_canonical}, reported oracle mean={oracle_mean_reported})"
            )

    # Cause 6: a stale mini artifact was loaded.
    oracle_artifact = oracle_trace.get("artifact", "")
    oracle_run = oracle_trace.get("run_id", "")
    if "mini" in oracle_artifact.lower() or "mini" in str(oracle_run).lower():
        possible_causes.append(
            f"a stale mini artifact was loaded (artifact={oracle_artifact}, run_id={oracle_run})"
        )
    if isinstance(evidence_manifest, dict):
        stale = evidence_manifest.get("stale_artifact") or evidence_manifest.get("stale")
        if stale:
            possible_causes.append(
                f"a stale mini artifact was loaded (evidence manifest flags stale={stale})"
            )
    if oracle_trace.get("source_hash") and isinstance(evidence_manifest, dict):
        manifest_hash = evidence_manifest.get("source_hash") or evidence_manifest.get(
            "analysis_source_tree_sha256"
        )
        if manifest_hash and oracle_trace.get("source_hash") != manifest_hash:
            possible_causes.append(
                "a stale mini artifact was loaded "
                f"(oracle source hash {oracle_trace.get('source_hash')} "
                f"!= manifest source hash {manifest_hash})"
            )

    # Deduplicate causes while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in possible_causes:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    possible_causes = deduped

    # --- Resolution -----------------------------------------------------
    if not discrepancy_found and oracle_minus_sham is not None:
        resolution = (
            f"no discrepancy: current oracle-minus-sham={oracle_minus_sham} "
            f"matches historical claim={historical}"
        )
        status = "PASS"
    elif oracle_minus_sham is None:
        resolution = (
            "cannot compute oracle-minus-sham from available data; "
            "require complete task-level outcomes"
        )
        status = "BLOCKED"
        errors.append("oracle_minus_sham could not be computed")
    else:
        # Pick the most likely primary cause.
        primary = possible_causes[0] if possible_causes else (
            "unresolved: oracle-minus-sham "
            f"{oracle_minus_sham} does not match historical claim {historical}"
        )
        resolution = (
            f"discrepancy confirmed: current oracle-minus-sham={oracle_minus_sham} "
            f"vs historical claim={historical}; primary cause: {primary}"
        )
        status = "FAIL"

    return {
        "status": status,
        "oracle_mean": oracle_mean,
        "sham_mean": sham_mean,
        "oracle_minus_sham": oracle_minus_sham,
        "task_set_digest": task_set_digest,
        "utility_version": utility_version,
        "discrepancy_found": discrepancy_found,
        "possible_causes": possible_causes,
        "resolution": resolution,
        "errors": errors,
        # Extra trace fields for auditability.
        "oracle_trace": oracle_trace,
        "sham_trace": sham_trace,
        "oracle_mean_canonical": oracle_mean_canonical,
        "oracle_mean_reported": oracle_mean_reported,
        "sham_mean_canonical": sham_mean_canonical,
        "sham_mean_reported": sham_mean_reported,
        "historical_claim": historical,
        "placeholder_match": placeholder_match,
    }
