"""Oracle discrepancy investigation for v0.4.6.

Fixes oracle-discrepancy handling so that the module does **not**
select a presumed cause.  Instead, it classifies the discrepancy
into one of a fixed set of possible causes, or returns UNRESOLVED
when insufficient evidence is available.

Possible classifications:
    DIFFERENT_TASK_SET
    DIFFERENT_SHAM_POLICY
    DIFFERENT_UTILITY_VERSION
    STALE_ARTIFACT
    SYNTHETIC_VS_REAL
    CALCULATION_DEFECT
    UNRESOLVED

Special rule:
    If evidence_origin is SYNTHETIC and historical_headroom is provided,
    return SYNTHETIC_VS_REAL.
    If insufficient evidence, return UNRESOLVED.
"""
from __future__ import annotations

import math
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.common.evidence_origin import EvidenceOrigin

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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _extract_mean(results: dict) -> float | None:
    """Extract a mean utility from a results dict."""
    if not isinstance(results, dict):
        return None
    for key in ("mean_utility", "mean", "aggregate_mean"):
        v = _safe_float(results.get(key))
        if v is not None:
            return v
    return None


def _extract_task_ids(results: dict) -> list[str]:
    """Extract task IDs from a results dict."""
    if not isinstance(results, dict):
        return []
    tids: list[str] = []
    for key in ("task_rows", "per_task", "task_results"):
        rows = results.get(key)
        if isinstance(rows, list):
            for r in rows:
                if isinstance(r, dict) and "task_id" in r:
                    tids.append(str(r["task_id"]))
    return tids


def _extract_utility_version(results: dict, manifest: dict | None) -> str:
    for src in (results, manifest or {}):
        if isinstance(src, dict):
            for key in ("utility_version", "utility_schema_version"):
                v = src.get(key)
                if v:
                    return str(v)
    return "unknown"


def _extract_task_set_digest(results: dict) -> str:
    if not isinstance(results, dict):
        return ""
    for key in ("task_set_digest", "task_ids_digest", "task_digest"):
        v = results.get(key)
        if v:
            return str(v)
    return ""


def _extract_policy_id(results: dict) -> str:
    if not isinstance(results, dict):
        return ""
    return str(results.get("policy_id", ""))


def _extract_origin(evidence_manifest: dict) -> str:
    if not isinstance(evidence_manifest, dict):
        return ""
    origin = evidence_manifest.get("evidence_origin", "")
    if not origin:
        origin = evidence_manifest.get("provider_class", "")
    return str(origin).upper()


def _looks_stale(results: dict) -> bool:
    """Heuristic: detect stale/placeholder artifacts."""
    if not isinstance(results, dict):
        return False
    for key in ("artifact", "artifact_path", "source"):
        val = str(results.get(key, "")).lower()
        if any(m in val for m in ("mini", "stale", "scratch", "tmp", "placeholder")):
            return True
    return False


def investigate_oracle_discrepancy(
    oracle_results: dict,
    sham_results: dict,
    counterfactual_outcomes: list[dict],
    evidence_manifest: dict,
    benchmark_manifest: dict,
    historical_headroom: float | None = None,
) -> dict:
    """Investigate an oracle-minus-sham headroom discrepancy.

    Does **not** select a presumed cause.  Instead, classifies the
    discrepancy into one of a fixed set of possible causes, or
    returns UNRESOLVED when insufficient evidence is available.

    Parameters
    ----------
    oracle_results:
        Oracle results dict (mean_utility, task_rows, etc.).
    sham_results:
        Sham results dict.
    counterfactual_outcomes:
        List of counterfactual outcome dicts.
    evidence_manifest:
        The evidence manifest dict (contains evidence_origin).
    benchmark_manifest:
        The benchmark manifest dict.
    historical_headroom:
        Optional historical oracle-minus-sham headroom value to
        compare against.

    Returns
    -------
    dict
        status, classification, oracle_mean, sham_mean, oracle_minus_sham,
        task_set_digest, utility_version, discrepancy_found, reason
    """
    oracle_mean = _extract_mean(oracle_results)
    sham_mean = _extract_mean(sham_results)
    oracle_minus_sham: float | None = None
    if oracle_mean is not None and sham_mean is not None:
        oracle_minus_sham = oracle_mean - sham_mean

    task_set_digest = _extract_task_set_digest(oracle_results)
    utility_version = _extract_utility_version(oracle_results, benchmark_manifest)
    origin = _extract_origin(evidence_manifest)

    # --- Special rule: SYNTHETIC + historical_headroom → SYNTHETIC_VS_REAL ---
    if origin == EvidenceOrigin.SYNTHETIC.value and historical_headroom is not None:
        return {
            "status": AuditStatus.PASS.value,
            "classification": "SYNTHETIC_VS_REAL",
            "oracle_mean": oracle_mean,
            "sham_mean": sham_mean,
            "oracle_minus_sham": oracle_minus_sham,
            "task_set_digest": task_set_digest,
            "utility_version": utility_version,
            "evidence_origin": origin,
            "discrepancy_found": True,
            "reason": (
                "evidence origin is SYNTHETIC and historical headroom "
                "was provided — discrepancy classified as SYNTHETIC_VS_REAL"
            ),
        }

    # --- Check if we have enough evidence to classify ---
    # Need at least oracle_results and sham_results with means.
    if oracle_mean is None or sham_mean is None:
        return {
            "status": AuditStatus.BLOCKED.value,
            "classification": "UNRESOLVED",
            "oracle_mean": oracle_mean,
            "sham_mean": sham_mean,
            "oracle_minus_sham": None,
            "task_set_digest": task_set_digest,
            "utility_version": utility_version,
            "evidence_origin": origin,
            "discrepancy_found": None,
            "reason": "insufficient evidence: cannot extract oracle/sham means",
        }

    # --- Check for discrepancy ---
    if historical_headroom is not None and oracle_minus_sham is not None:
        if abs(oracle_minus_sham - historical_headroom) <= _HEADROOM_TOLERANCE:
            # No discrepancy — values match.
            return {
                "status": AuditStatus.PASS.value,
                "classification": "UNRESOLVED",
                "oracle_mean": oracle_mean,
                "sham_mean": sham_mean,
                "oracle_minus_sham": oracle_minus_sham,
                "task_set_digest": task_set_digest,
                "utility_version": utility_version,
                "evidence_origin": origin,
                "discrepancy_found": False,
                "reason": (
                    "no discrepancy: oracle-minus-sham matches "
                    "historical headroom"
                ),
            }

    # If no historical headroom to compare against, and oracle_minus_sham is
    # non-negative, there is no discrepancy to investigate — but only if no
    # possible causes are identified.  We compute possible causes first and
    # then decide.
    # --- Attempt classification ---
    possible_causes: list[str] = []

    # Check DIFFERENT_TASK_SET
    oracle_tids = set(_extract_task_ids(oracle_results))
    sham_tids = set(_extract_task_ids(sham_results))
    if oracle_tids and sham_tids and oracle_tids != sham_tids:
        possible_causes.append("DIFFERENT_TASK_SET")

    # Check DIFFERENT_SHAM_POLICY
    oracle_pid = _extract_policy_id(oracle_results)
    sham_pid = _extract_policy_id(sham_results)
    if oracle_pid and sham_pid and oracle_pid == sham_pid:
        # If oracle and sham have the same policy_id, something is wrong.
        possible_causes.append("DIFFERENT_SHAM_POLICY")

    # Check DIFFERENT_UTILITY_VERSION
    oracle_uv = _extract_utility_version(oracle_results, None)
    sham_uv = _extract_utility_version(sham_results, None)
    if (
        oracle_uv != "unknown"
        and sham_uv != "unknown"
        and oracle_uv != sham_uv
    ):
        possible_causes.append("DIFFERENT_UTILITY_VERSION")

    # Check STALE_ARTIFACT
    if _looks_stale(oracle_results) or _looks_stale(sham_results):
        possible_causes.append("STALE_ARTIFACT")

    # Check SYNTHETIC_VS_REAL (without historical_headroom)
    if origin == EvidenceOrigin.SYNTHETIC.value:
        possible_causes.append("SYNTHETIC_VS_REAL")

    # Check CALCULATION_DEFECT
    # If oracle_minus_sham is negative or implausibly large.
    if oracle_minus_sham is not None:
        if oracle_minus_sham < -_HEADROOM_TOLERANCE:
            possible_causes.append("CALCULATION_DEFECT")
        elif historical_headroom is not None and abs(
            oracle_minus_sham - historical_headroom
        ) > 1.0:
            possible_causes.append("CALCULATION_DEFECT")

    # If no historical headroom to compare against, oracle_minus_sham is
    # non-negative, and no possible causes are identified, there is no
    # discrepancy to investigate.
    if (
        historical_headroom is None
        and oracle_minus_sham is not None
        and oracle_minus_sham >= -_HEADROOM_TOLERANCE
        and len(possible_causes) == 0
    ):
        return {
            "status": AuditStatus.PASS.value,
            "classification": "NO_DISCREPANCY",
            "oracle_mean": oracle_mean,
            "sham_mean": sham_mean,
            "oracle_minus_sham": oracle_minus_sham,
            "task_set_digest": task_set_digest,
            "utility_version": utility_version,
            "evidence_origin": origin,
            "discrepancy_found": False,
            "possible_causes": possible_causes,
            "reason": "no discrepancy: oracle-minus-sham is non-negative and no historical headroom to compare against",
        }

    # --- Classify ---
    if len(possible_causes) == 0:
        classification = "UNRESOLVED"
        reason = "insufficient evidence to classify the discrepancy"
        status = AuditStatus.BLOCKED.value
    elif len(possible_causes) == 1:
        classification = possible_causes[0]
        reason = f"discrepancy classified as {classification}"
        status = AuditStatus.PASS.value
    else:
        # Multiple possible causes — do not select one. Return UNRESOLVED.
        classification = "UNRESOLVED"
        reason = (
            f"multiple possible causes identified: {possible_causes} — "
            f"cannot resolve to a single cause"
        )
        status = AuditStatus.FAIL.value

    return {
        "status": status,
        "classification": classification,
        "oracle_mean": oracle_mean,
        "sham_mean": sham_mean,
        "oracle_minus_sham": oracle_minus_sham,
        "task_set_digest": task_set_digest,
        "utility_version": utility_version,
        "evidence_origin": origin,
        "discrepancy_found": True,
        "possible_causes": possible_causes,
        "reason": reason,
    }
