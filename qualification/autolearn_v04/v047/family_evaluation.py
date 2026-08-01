"""Family-level evaluation for v0.4.7.

For each task family, compute aggregate statistics across the four
policies (baseline, candidate, sham, oracle) and flag families that
fall below the minimum support thresholds.
"""
from __future__ import annotations

import math
from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via error function approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation)."""
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _lower_confidence_bound(
    deltas: list[float], groups: list[str], confidence: float = 0.95
) -> float:
    """Cluster-robust lower confidence bound on the mean of *deltas*.

    Uses a cluster-robust standard error (collapsed by *groups*) and a
    normal approximation.  Returns 0.0 when there is no data.
    """
    if not deltas:
        return 0.0
    # Collapse deltas by group (mean within group).
    group_values: dict[str, list[float]] = {}
    for g, d in zip(groups, deltas):
        group_values.setdefault(g, []).append(d)
    cluster_means = [_mean(vs) for vs in group_values.values()]
    n = len(cluster_means)
    if n == 0:
        return 0.0
    mean = sum(cluster_means) / n
    if n < 2:
        return mean  # No SE estimate possible; be optimistic.
    se = _std(cluster_means) / math.sqrt(n)
    z = _normal_ppf(confidence)
    return mean - z * se


def _action_distribution(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    counts: dict[str, int] = {}
    for row in rows:
        action = row.get("selected_action")
        if action is None:
            action = "ABSTAIN"
        counts[str(action)] = counts.get(str(action), 0) + 1
    total = sum(counts.values())
    return {a: c / total for a, c in counts.items()}


def _rate(rows: list[dict], field: str, value: Any = True) -> float:
    if not rows:
        return 0.0
    n = sum(1 for r in rows if r.get(field) == value)
    return n / len(rows)


def _utility_mean(rows: list[dict]) -> float:
    utils = [float(r["selected_utility"]) for r in rows
             if r.get("selected_utility") is not None]
    return _mean(utils)


def _index_rows_by_task(rows: list[dict]) -> dict[str, dict]:
    return {r["task_id"]: r for r in rows}


def _paired_deltas(
    candidate_rows: list[dict],
    other_rows: list[dict],
) -> tuple[list[float], list[str]]:
    """Return (deltas, groups) for candidate - other, aligned by task_id."""
    other_by_task = _index_rows_by_task(other_rows)
    deltas: list[float] = []
    groups: list[str] = []
    for cr in candidate_rows:
        tid = cr["task_id"]
        other = other_by_task.get(tid)
        if other is None:
            continue
        cu = cr.get("selected_utility")
        ou = other.get("selected_utility")
        if cu is None or ou is None:
            continue
        deltas.append(float(cu) - float(ou))
        groups.append(str(cr.get("task_group_id", tid)))
    return deltas, groups


def evaluate_families(
    candidate_results: dict,
    baseline_results: dict,
    sham_results: dict,
    oracle_results: dict,
    min_test_tasks: int = 30,
    min_test_groups: int = 20,
) -> dict:
    """Family-level evaluation.

    For each task family, compute:
    - task_count, task_group_count
    - baseline_utility, candidate_utility, sham_utility, oracle_utility
    - candidate_baseline_delta, candidate_sham_delta
    - action_distribution
    - error_rate, timeout_rate, abstention_rate

    Families below support thresholds are marked INSUFFICIENT_SUPPORT.

    Returns dict with:
    - families: dict[family_name, family_stats]
    - critical_regressions: list of families with LCB < -delta_f
    - n_sufficient: int
    - n_insufficient: int
    """
    candidate_rows = candidate_results.get("task_rows", [])
    baseline_rows = baseline_results.get("task_rows", [])
    sham_rows = sham_results.get("task_rows", [])
    oracle_rows = oracle_results.get("task_rows", [])

    # Group candidate rows by family.
    families: dict[str, list[dict]] = {}
    for row in candidate_rows:
        fam = str(row.get("family", "unknown"))
        families.setdefault(fam, []).append(row)

    family_stats: dict[str, dict[str, Any]] = {}
    critical_regressions: list[str] = []
    n_sufficient = 0
    n_insufficient = 0

    baseline_by_task = _index_rows_by_task(baseline_rows)
    sham_by_task = _index_rows_by_task(sham_rows)
    oracle_by_task = _index_rows_by_task(oracle_rows)

    for fam, rows in families.items():
        task_ids = {r["task_id"] for r in rows}
        group_ids = {str(r.get("task_group_id", r["task_id"])) for r in rows}
        task_count = len(task_ids)
        group_count = len(group_ids)

        cand_utils = [float(r["selected_utility"]) for r in rows
                      if r.get("selected_utility") is not None]
        base_utils = [float(baseline_by_task[tid]["selected_utility"])
                      for tid in task_ids
                      if tid in baseline_by_task
                      and baseline_by_task[tid].get("selected_utility") is not None]
        sham_utils = [float(sham_by_task[tid]["selected_utility"])
                      for tid in task_ids
                      if tid in sham_by_task
                      and sham_by_task[tid].get("selected_utility") is not None]
        oracle_utils = [float(oracle_by_task[tid]["selected_utility"])
                        for tid in task_ids
                        if tid in oracle_by_task
                        and oracle_by_task[tid].get("selected_utility") is not None]

        # Paired deltas candidate vs baseline / sham within family.
        cb_deltas, cb_groups = [], []
        cs_deltas, cs_groups = [], []
        for r in rows:
            tid = r["task_id"]
            cu = r.get("selected_utility")
            bu_row = baseline_by_task.get(tid)
            su_row = sham_by_task.get(tid)
            if cu is not None and bu_row is not None and bu_row.get("selected_utility") is not None:
                cb_deltas.append(float(cu) - float(bu_row["selected_utility"]))
                cb_groups.append(str(r.get("task_group_id", tid)))
            if cu is not None and su_row is not None and su_row.get("selected_utility") is not None:
                cs_deltas.append(float(cu) - float(su_row["selected_utility"]))
                cs_groups.append(str(r.get("task_group_id", tid)))

        cb_lcb = _lower_confidence_bound(cb_deltas, cb_groups)
        cs_lcb = _lower_confidence_bound(cs_deltas, cs_groups)

        sufficient = task_count >= min_test_tasks and group_count >= min_test_groups
        status = AuditStatus.PASS.value if sufficient else "INSUFFICIENT_SUPPORT"
        if sufficient:
            n_sufficient += 1
        else:
            n_insufficient += 1

        # Critical regression: candidate-vs-baseline LCB below -delta_f.
        # delta_f defaults to 0.01 (practical threshold) when not provided.
        delta_f = 0.01
        if sufficient and cb_lcb < -delta_f:
            critical_regressions.append(
                f"{fam}: LCB(candidate-baseline)={cb_lcb:.4f} < -{delta_f}"
            )

        family_stats[fam] = {
            "family": fam,
            "status": status,
            "sufficient_support": sufficient,
            "task_count": task_count,
            "task_group_count": group_count,
            "baseline_utility": _mean(base_utils),
            "candidate_utility": _mean(cand_utils),
            "sham_utility": _mean(sham_utils),
            "oracle_utility": _mean(oracle_utils),
            "candidate_baseline_delta": _mean(cb_deltas),
            "candidate_sham_delta": _mean(cs_deltas),
            "candidate_baseline_lcb": cb_lcb,
            "candidate_sham_lcb": cs_lcb,
            "action_distribution": _action_distribution(rows),
            "error_rate": _rate(rows, "error", True),
            "timeout_rate": _rate(rows, "timeout", True),
            "abstention_rate": _rate(rows, "abstained", True),
        }

    return {
        "families": family_stats,
        "critical_regressions": critical_regressions,
        "n_sufficient": n_sufficient,
        "n_insufficient": n_insufficient,
        "min_test_tasks": min_test_tasks,
        "min_test_groups": min_test_groups,
    }
