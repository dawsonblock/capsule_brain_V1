"""Metric consistency cross-checks for v0.4.3 repair analysis.

Every summary metric reported by a downstream artifact (gate_a_result,
qualification_report, evaluation artifacts) is recomputed from the
task-level records and compared against the stored value.  Any material
mismatch invalidates the downstream decision artifacts.

This module is the single source of truth for "did the numbers drift?".
It uses the canonical evaluator's aggregate formula
(``compute_headroom_metrics``) so that the recovered-headroom fraction
is recomputed the same way it is produced, eliminating the v0.4.2
task-level-ratio-averaging defect.
"""
from __future__ import annotations

from typing import Any

from .data import LoadedRun, PolicyEvaluation
from .canonical_evaluator import PolicyEvalResult, compute_headroom_metrics
from .config import (
    METRIC_TOLERANCE_FLOAT,
    METRIC_TOLERANCE_STRICT,
    HEADROOM_PERCENTAGE_SCALE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean_utility_from_per_task(eval_artifact: PolicyEvaluation | None) -> float | None:
    """Recompute mean utility from the per-task records of an eval artifact.

    Returns None if the artifact is missing or has no complete tasks.
    """
    if eval_artifact is None:
        return None
    utilities: list[float] = []
    for tr in eval_artifact.per_task.values():
        u = tr.get("selected_utility")
        if u is None:
            continue
        try:
            utilities.append(float(u))
        except (TypeError, ValueError):
            continue
    if not utilities:
        return None
    return float(sum(utilities) / len(utilities))


def _n_complete_from_per_task(eval_artifact: PolicyEvaluation | None) -> int | None:
    """Recompute n_complete_tasks from per-task records."""
    if eval_artifact is None:
        return None
    n = 0
    for tr in eval_artifact.per_task.values():
        if tr.get("selected_utility") is not None:
            n += 1
    return n


def _policy_mean(canonical_results: dict[str, Any], policy_id: str) -> float | None:
    """Read mean_utility from a canonical PolicyEvalResult."""
    res = canonical_results.get(policy_id)
    if res is None:
        return None
    if isinstance(res, PolicyEvalResult):
        return float(res.mean_utility)
    if isinstance(res, dict):
        mu = res.get("mean_utility")
        return float(mu) if mu is not None else None
    return None


def _policy_n_complete(canonical_results: dict[str, Any], policy_id: str) -> int | None:
    res = canonical_results.get(policy_id)
    if res is None:
        return None
    if isinstance(res, PolicyEvalResult):
        return int(res.n_complete)
    if isinstance(res, dict):
        nc = res.get("n_complete")
        return int(nc) if nc is not None else None
    return None


def _policy_n_tasks(canonical_results: dict[str, Any], policy_id: str) -> int | None:
    res = canonical_results.get(policy_id)
    if res is None:
        return None
    if isinstance(res, PolicyEvalResult):
        return int(res.n_tasks)
    if isinstance(res, dict):
        nt = res.get("n_tasks")
        return int(nt) if nt is not None else None
    return None


def _make_check(
    name: str,
    expected: Any,
    actual: Any,
    tolerance: float,
    artifact_source: str,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Build a single check record.

    ``expected`` is the recomputed-from-task-records value.
    ``actual`` is the stored artifact value.
    """
    abs_diff: float | None
    status: str
    if expected is None or actual is None:
        abs_diff = None
        status = "FAIL" if (expected is None) ^ (actual is None) else "PASS"
    else:
        try:
            abs_diff = float(abs(float(expected) - float(actual)))
            status = "PASS" if abs_diff <= tolerance else "FAIL"
        except (TypeError, ValueError):
            abs_diff = None
            status = "PASS" if expected == actual else "FAIL"
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "abs_diff": abs_diff,
        "tolerance": tolerance,
        "deterministic": deterministic,
        "status": status,
        "artifact_source": artifact_source,
    }


def _find_headroom_in_artifact(artifact: dict[str, Any]) -> tuple[float | None, float | None]:
    """Search an artifact dict for recovered headroom fraction/percent."""
    if not artifact:
        return None, None
    fraction = None
    percent = None
    # Direct keys.
    if "recovered_headroom_fraction" in artifact:
        fraction = artifact.get("recovered_headroom_fraction")
    if "recovered_headroom_percent" in artifact:
        percent = artifact.get("recovered_headroom_percent")
    # Nested under headroom / headroom_vs_reference.
    for key in ("headroom", "headroom_vs_reference", "headroom_metrics"):
        nested = artifact.get(key)
        if isinstance(nested, dict):
            if fraction is None and "recovered_headroom_fraction" in nested:
                fraction = nested.get("recovered_headroom_fraction")
            if percent is None and "recovered_headroom_percent" in nested:
                percent = nested.get("recovered_headroom_percent")
    # Recurse one level into dict values.
    if fraction is None or percent is None:
        for v in artifact.values():
            if isinstance(v, dict):
                f2, p2 = _find_headroom_in_artifact(v)
                if fraction is None and f2 is not None:
                    fraction = f2
                if percent is None and p2 is not None:
                    percent = p2
    return (
        float(fraction) if fraction is not None else None,
        float(percent) if percent is not None else None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_metric_consistency(run: LoadedRun, canonical_results: dict) -> dict:
    """Cross-check every summary metric against task-level records.

    Args:
        run: A LoadedRun from data.py.
        canonical_results: The dict returned by evaluate_all_policies(),
            mapping policy_id -> PolicyEvalResult.

    Returns:
        A JSON-serializable dict with:
          - overall_status: "PASS" | "FAIL"
          - checks: list of check records
          - failed_checks: list of names that failed
          - artifact_sources: mapping of check name -> artifact source
    """
    checks: list[dict[str, Any]] = []
    artifact_sources: dict[str, str] = {}

    gate_a = run.gate_a_result or {}
    qual_report = run.qualification_report or {}

    # --- Mean utility checks (canonical vs stored eval artifact) ---
    mean_specs: list[tuple[str, str, PolicyEvaluation | None]] = [
        ("mean_candidate_utility", "candidate_executed", run.eval_candidate),
        ("mean_sham_utility", "sham", run.eval_sham),
        ("mean_baseline_utility", "baseline", run.eval_baseline),
        ("mean_oracle_utility", "oracle", run.eval_oracle),
    ]
    for name, canon_key, eval_art in mean_specs:
        recomputed = _mean_utility_from_per_task(eval_art)
        stored = _policy_mean(canonical_results, canon_key)
        src = f"evaluate_test_{eval_art.policy_id}.json" if eval_art else "(missing)"
        chk = _make_check(
            name, recomputed, stored, METRIC_TOLERANCE_FLOAT, src,
            deterministic=True,
        )
        checks.append(chk)
        artifact_sources[name] = src

    # --- Candidate-minus-baseline (compare with gate_a_result delta_candidate.mean) ---
    cand_mean = _policy_mean(canonical_results, "candidate_executed")
    base_mean = _policy_mean(canonical_results, "baseline")
    recomputed_cand_base = (
        float(cand_mean - base_mean)
        if cand_mean is not None and base_mean is not None
        else None
    )
    delta_cand = gate_a.get("delta_candidate", {})
    stored_cand_base = delta_cand.get("mean") if isinstance(delta_cand, dict) else None
    src = "gate_a_result.json#delta_candidate.mean"
    chk = _make_check(
        "candidate_minus_baseline", recomputed_cand_base, stored_cand_base,
        METRIC_TOLERANCE_FLOAT, src,
    )
    checks.append(chk)
    artifact_sources["candidate_minus_baseline"] = src

    # --- Candidate-minus-sham (compare with gate_a_result delta_candidate_sham.mean) ---
    sham_mean = _policy_mean(canonical_results, "sham")
    recomputed_cand_sham = (
        float(cand_mean - sham_mean)
        if cand_mean is not None and sham_mean is not None
        else None
    )
    delta_cand_sham = gate_a.get("delta_candidate_sham", {})
    stored_cand_sham = (
        delta_cand_sham.get("mean") if isinstance(delta_cand_sham, dict) else None
    )
    src = "gate_a_result.json#delta_candidate_sham.mean"
    chk = _make_check(
        "candidate_minus_sham", recomputed_cand_sham, stored_cand_sham,
        METRIC_TOLERANCE_FLOAT, src,
    )
    checks.append(chk)
    artifact_sources["candidate_minus_sham"] = src

    # --- Oracle-minus-sham (recompute from canonical results) ---
    oracle_mean = _policy_mean(canonical_results, "oracle")
    recomputed_oracle_sham = (
        float(oracle_mean - sham_mean)
        if oracle_mean is not None and sham_mean is not None
        else None
    )
    # Stored value: search gate_a_result / qualification_report.
    stored_oracle_sham: float | None = None
    for art in (gate_a, qual_report):
        if isinstance(art, dict):
            for key in ("oracle_minus_sham", "delta_oracle_sham"):
                v = art.get(key)
                if isinstance(v, dict):
                    stored_oracle_sham = v.get("mean")
                    break
                if v is not None:
                    stored_oracle_sham = v
                    break
        if stored_oracle_sham is not None:
            break
    src = "gate_a_result.json#oracle_minus_sham"
    chk = _make_check(
        "oracle_minus_sham", recomputed_oracle_sham, stored_oracle_sham,
        METRIC_TOLERANCE_FLOAT, src,
    )
    checks.append(chk)
    artifact_sources["oracle_minus_sham"] = src

    # --- Recovered-headroom fractions (recompute via compute_headroom_metrics) ---
    # Candidate vs sham (primary recovered-headroom fraction).
    n_complete = _policy_n_complete(canonical_results, "candidate_executed")
    cand_res = canonical_results.get("candidate_executed")
    task_ids_digest = ""
    if isinstance(cand_res, PolicyEvalResult):
        task_ids_digest = cand_res.task_ids_digest
    recomputed_headroom = compute_headroom_metrics(
        candidate_mean=cand_mean if cand_mean is not None else 0.0,
        reference_mean=sham_mean if sham_mean is not None else 0.0,
        oracle_mean=oracle_mean if oracle_mean is not None else 0.0,
        n_tasks=n_complete if n_complete is not None else 0,
        task_ids_digest=task_ids_digest,
        reference_policy_id="sham",
        candidate_policy_id="candidate_executed",
    )
    recomputed_fraction = recomputed_headroom.get("recovered_headroom_fraction")
    recomputed_percent = recomputed_headroom.get("recovered_headroom_percent")

    # Stored values: search gate_a_result then qualification_report.
    stored_fraction, stored_percent = _find_headroom_in_artifact(gate_a)
    if stored_fraction is None and stored_percent is None:
        stored_fraction, stored_percent = _find_headroom_in_artifact(qual_report)

    src = "gate_a_result.json#recovered_headroom"
    chk_frac = _make_check(
        "recovered_headroom_fraction", recomputed_fraction, stored_fraction,
        METRIC_TOLERANCE_FLOAT, src,
    )
    checks.append(chk_frac)
    artifact_sources["recovered_headroom_fraction"] = src

    chk_pct = _make_check(
        "recovered_headroom_percent", recomputed_percent, stored_percent,
        METRIC_TOLERANCE_FLOAT, src,
    )
    checks.append(chk_pct)
    artifact_sources["recovered_headroom_percent"] = src

    # --- Task counts ---
    # n_tasks: canonical candidate_executed vs run.test_task_ids length.
    canon_n_tasks = _policy_n_tasks(canonical_results, "candidate_executed")
    run_n_tasks = len(run.test_task_ids)
    src = "benchmark_manifest.json#test_split"
    chk = _make_check(
        "task_count_candidate", run_n_tasks, canon_n_tasks,
        METRIC_TOLERANCE_STRICT, src, deterministic=True,
    )
    checks.append(chk)
    artifact_sources["task_count_candidate"] = src

    # --- n_complete_tasks ---
    for name, canon_key, eval_art in [
        ("n_complete_tasks_candidate", "candidate_executed", run.eval_candidate),
        ("n_complete_tasks_sham", "sham", run.eval_sham),
        ("n_complete_tasks_baseline", "baseline", run.eval_baseline),
        ("n_complete_tasks_oracle", "oracle", run.eval_oracle),
    ]:
        recomputed = _n_complete_from_per_task(eval_art)
        stored = _policy_n_complete(canonical_results, canon_key)
        src = f"evaluate_test_{eval_art.policy_id}.json" if eval_art else "(missing)"
        chk = _make_check(
            name, recomputed, stored, METRIC_TOLERANCE_STRICT, src,
            deterministic=True,
        )
        checks.append(chk)
        artifact_sources[name] = src

    # --- Aggregate ---
    failed_checks = [c["name"] for c in checks if c["status"] == "FAIL"]
    overall_status = "PASS" if not failed_checks else "FAIL"

    return {
        "overall_status": overall_status,
        "checks": checks,
        "failed_checks": failed_checks,
        "artifact_sources": artifact_sources,
        "n_checks": len(checks),
        "n_failed": len(failed_checks),
    }
