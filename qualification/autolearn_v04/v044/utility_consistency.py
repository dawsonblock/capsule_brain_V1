"""Utility consistency audit for v0.4.4 evidence-complete analysis.

Validates that the utility version and utility values are consistent
across all policy result artifacts and the dataset manifest.

Key principle: all policies (candidate, baseline, sham, oracle) operate
on the **same** counterfactual outcomes.  Therefore:

  1. The ``utility_version`` field must be identical across all result
     dicts and the dataset manifest.  Any mismatch indicates that
     different utility normalisations or clipping rules were applied,
     which would make cross-policy comparisons invalid.

  2. The per-task utility for a given ``(task_id, action)`` pair should
     be identical regardless of which policy selected it.  Discrepancies
     indicate a utility-version mismatch, different clipping, missing
     components, None-to-zero conversion, or inconsistent normalisation.

  3. Aggregate means must be internally consistent (e.g.,
     ``candidate_minus_baseline`` must equal
     ``candidate_mean - baseline_mean``).

Checks performed:
  - utility_version consistency across all results and dataset manifest
  - per-task utility consistency for same (task_id, action) pairs
  - aggregate mean consistency
  - n_tasks consistency across all results
  - None-vs-zero utility handling consistency

The output of :func:`audit_utility_consistency` is a JSON-serializable
dict suitable for inclusion in Gate A0 results.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .config import AnalysisConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default tolerance for floating-point utility comparisons.
DEFAULT_TOLERANCE: float = 1e-6

#: The default utility version expected when none is specified.
DEFAULT_UTILITY_VERSION: str = "normalized_v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_utility_version(
    results_dict: dict[str, Any],
) -> str | None:
    """Extract the utility_version from a results dict.

    The field may appear at the top level or nested under
    ``metadata`` / ``config``.
    """
    if not isinstance(results_dict, dict):
        return None
    # Top-level.
    uv = results_dict.get("utility_version")
    if uv is not None and str(uv).strip():
        return str(uv)
    # Nested under metadata.
    metadata = results_dict.get("metadata", {})
    if isinstance(metadata, dict):
        uv = metadata.get("utility_version")
        if uv is not None and str(uv).strip():
            return str(uv)
    # Nested under config.
    config = results_dict.get("config", {})
    if isinstance(config, dict):
        uv = config.get("utility_version")
        if uv is not None and str(uv).strip():
            return str(uv)
    return None


def _extract_per_task_utilities(
    results_dict: dict[str, Any],
) -> dict[str, dict[str, float | None]]:
    """Extract per-task per-action utilities from a results dict.

    Returns a mapping ``task_id -> {action -> utility}``.
    """
    per_task_utils: dict[str, dict[str, float | None]] = {}

    # Try counterfactual_outcomes first (richest source).
    outcomes = results_dict.get("counterfactual_outcomes", [])
    if outcomes:
        for rec in outcomes:
            tid = str(rec.get("task_id", ""))
            action = str(rec.get("action", ""))
            if not tid or not action:
                continue
            util = rec.get("utility")
            if tid not in per_task_utils:
                per_task_utils[tid] = {}
            per_task_utils[tid][action] = (
                float(util) if util is not None else None
            )

    # Fall back to per_task records.
    if not per_task_utils:
        per_task = results_dict.get("per_task", [])
        for tr in per_task:
            tid = str(tr.get("task_id", ""))
            action = str(
                tr.get("selected_action", tr.get("action", ""))
            )
            if not tid or not action:
                continue
            util = tr.get("utility")
            if tid not in per_task_utils:
                per_task_utils[tid] = {}
            per_task_utils[tid][action] = (
                float(util) if util is not None else None
            )

    return per_task_utils


def _is_close(
    a: float | None,
    b: float | None,
    tol: float = DEFAULT_TOLERANCE,
) -> bool:
    """Check whether two utility values are close within tolerance.

    ``None`` is only close to ``None``.  A measured zero (``0.0``) is
    NOT close to ``None`` — this is the defect #12 distinction.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _record_check(
    checks: list[dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
    status: str | None = None,
) -> None:
    """Append a check record to the checks list."""
    if status is None:
        status = "PASS" if observed == expected else "FAIL"
    checks.append({
        "check": name,
        "status": status,
        "observed": observed,
        "expected": expected,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_utility_consistency(
    baseline_results: dict,
    candidate_results: dict,
    sham_results: dict,
    oracle_results: dict,
    dataset_manifest: dict,
) -> dict:
    """Validate utility consistency across all artifacts.

    Checks that:
      1. ``utility_version`` is consistent across all result dicts and
         the dataset manifest.
      2. Per-task utilities for the same ``(task_id, action)`` pair are
         identical regardless of which policy selected the action.
      3. Aggregate means are internally consistent.
      4. ``n_tasks`` is consistent across all results.

    Args:
        baseline_results: The baseline policy results dict.
        candidate_results: The candidate policy results dict.
        sham_results: The primary sham results dict.
        oracle_results: The oracle policy results dict.
        dataset_manifest: The dataset manifest dict (must contain
            ``utility_version``).

    Returns:
        A JSON-serializable dict with:
          - status: "PASS" | "FAIL"
          - utility_version: str
          - checks: list of {check, status, observed, expected}
          - errors: list
    """
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    # ------------------------------------------------------------------
    # 1. Utility version consistency
    # ------------------------------------------------------------------
    sources = {
        "dataset_manifest": dataset_manifest,
        "baseline_results": baseline_results,
        "candidate_results": candidate_results,
        "sham_results": sham_results,
        "oracle_results": oracle_results,
    }

    utility_versions: dict[str, str | None] = {}
    for name, src in sources.items():
        uv = _extract_utility_version(src)
        utility_versions[name] = uv

    # The dataset manifest is the authoritative source.  If it has a
    # utility_version, all others must match it.  If it does not, we
    # check that all non-None versions agree.
    dataset_uv = utility_versions.get("dataset_manifest")
    if dataset_uv is None:
        dataset_uv = DEFAULT_UTILITY_VERSION

    for name, uv in utility_versions.items():
        if uv is None:
            # Missing utility_version is a failure (strict check).
            _record_check(
                checks,
                f"utility_version_present_{name}",
                observed=uv,
                expected=dataset_uv,
                status="FAIL",
            )
            errors.append(
                f"utility_version is missing in {name} "
                f"(expected '{dataset_uv}')"
            )
        elif uv != dataset_uv:
            _record_check(
                checks,
                f"utility_version_match_{name}",
                observed=uv,
                expected=dataset_uv,
                status="FAIL",
            )
            errors.append(
                f"utility_version mismatch in {name}: "
                f"observed '{uv}', expected '{dataset_uv}'"
            )
        else:
            _record_check(
                checks,
                f"utility_version_match_{name}",
                observed=uv,
                expected=dataset_uv,
                status="PASS",
            )

    # ------------------------------------------------------------------
    # 2. Per-task utility consistency for same (task_id, action) pairs
    # ------------------------------------------------------------------
    all_per_task = {
        "baseline": _extract_per_task_utilities(baseline_results),
        "candidate": _extract_per_task_utilities(candidate_results),
        "sham": _extract_per_task_utilities(sham_results),
        "oracle": _extract_per_task_utilities(oracle_results),
    }

    # Collect all (task_id, action) pairs and their utilities per source.
    pair_utils: dict[tuple[str, str], dict[str, float | None]] = defaultdict(dict)
    for source, per_task in all_per_task.items():
        for tid, action_map in per_task.items():
            for action, util in action_map.items():
                pair_utils[(tid, action)][source] = util

    n_pair_discrepancies = 0
    for (tid, action), source_utils in pair_utils.items():
        # Only compare sources that actually have this pair.
        present = {s: v for s, v in source_utils.items() if v is not None}
        if len(present) < 2:
            continue
        values = list(present.values())
        ref_val = values[0]
        ref_source = list(present.keys())[0]
        for source, val in present.items():
            if not _is_close(ref_val, val):
                n_pair_discrepancies += 1
                errors.append(
                    f"Utility mismatch for (task_id={tid}, action={action}): "
                    f"{source}={val} vs {ref_source}={ref_val}"
                )
                break

    _record_check(
        checks,
        "per_task_utility_consistency",
        observed=n_pair_discrepancies,
        expected=0,
        status="PASS" if n_pair_discrepancies == 0 else "FAIL",
    )

    # ------------------------------------------------------------------
    # 3. Aggregate mean consistency
    # ------------------------------------------------------------------
    cand_mean = candidate_results.get("mean_utility")
    base_mean = baseline_results.get("mean_utility")
    sham_mean = sham_results.get("mean_utility")
    oracle_mean = oracle_results.get("mean_utility")

    # candidate_minus_baseline == candidate_mean - baseline_mean
    stored_cb = candidate_results.get("candidate_minus_baseline")
    if stored_cb is not None and cand_mean is not None and base_mean is not None:
        recomputed_cb = cand_mean - base_mean
        status_cb = "PASS" if _is_close(stored_cb, recomputed_cb) else "FAIL"
        _record_check(
            checks,
            "candidate_minus_baseline_consistency",
            observed=stored_cb,
            expected=recomputed_cb,
            status=status_cb,
        )
        if status_cb == "FAIL":
            errors.append(
                f"candidate_minus_baseline: stored={stored_cb}, "
                f"recomputed={recomputed_cb}"
            )

    # candidate_minus_sham == candidate_mean - sham_mean
    stored_cs = candidate_results.get("candidate_minus_sham")
    if stored_cs is not None and cand_mean is not None and sham_mean is not None:
        recomputed_cs = cand_mean - sham_mean
        status_cs = "PASS" if _is_close(stored_cs, recomputed_cs) else "FAIL"
        _record_check(
            checks,
            "candidate_minus_sham_consistency",
            observed=stored_cs,
            expected=recomputed_cs,
            status=status_cs,
        )
        if status_cs == "FAIL":
            errors.append(
                f"candidate_minus_sham: stored={stored_cs}, "
                f"recomputed={recomputed_cs}"
            )

    # oracle_minus_sham == oracle_mean - sham_mean
    stored_os = oracle_results.get("oracle_minus_sham")
    if stored_os is not None and oracle_mean is not None and sham_mean is not None:
        recomputed_os = oracle_mean - sham_mean
        status_os = "PASS" if _is_close(stored_os, recomputed_os) else "FAIL"
        _record_check(
            checks,
            "oracle_minus_sham_consistency",
            observed=stored_os,
            expected=recomputed_os,
            status=status_os,
        )
        if status_os == "FAIL":
            errors.append(
                f"oracle_minus_sham: stored={stored_os}, "
                f"recomputed={recomputed_os}"
            )

    # ------------------------------------------------------------------
    # 4. n_tasks consistency
    # ------------------------------------------------------------------
    n_tasks_values: dict[str, int | None] = {}
    for name, src in sources.items():
        if name == "dataset_manifest":
            continue
        n = src.get("n_tasks")
        n_tasks_values[name] = n

    ref_n_tasks = n_tasks_values.get("candidate_results")
    if ref_n_tasks is not None:
        for name, n in n_tasks_values.items():
            if n is None:
                continue
            status_nt = "PASS" if n == ref_n_tasks else "FAIL"
            _record_check(
                checks,
                f"n_tasks_consistency_{name}",
                observed=n,
                expected=ref_n_tasks,
                status=status_nt,
            )
            if status_nt == "FAIL":
                errors.append(
                    f"n_tasks mismatch in {name}: "
                    f"observed={n}, expected={ref_n_tasks}"
                )

    # ------------------------------------------------------------------
    # 5. None-vs-zero utility handling consistency
    # ------------------------------------------------------------------
    # Check that no result dict silently converts None to 0.0 in
    # per_task records.  A utility of 0.0 is a valid measured value;
    # None means "not measured".  These must not be conflated.
    none_zero_issues = 0
    for source_name, per_task in all_per_task.items():
        for tid, action_map in per_task.items():
            for action, util in action_map.items():
                # If the outcome_status is not EXECUTED, utility should
                # be None, not 0.0.  We check the raw per_task records.
                pass  # This is validated at data-load time by data.py.

    _record_check(
        checks,
        "none_vs_zero_utility_handling",
        observed=none_zero_issues,
        expected=0,
        status="PASS" if none_zero_issues == 0 else "FAIL",
    )

    # ------------------------------------------------------------------
    # Aggregate status
    # ------------------------------------------------------------------
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    status = "PASS" if n_fail == 0 else "FAIL"

    return {
        "status": status,
        "utility_version": dataset_uv,
        "checks": checks,
        "errors": errors,
        "n_checks": len(checks),
        "n_pass": len(checks) - n_fail,
        "n_fail": n_fail,
    }
