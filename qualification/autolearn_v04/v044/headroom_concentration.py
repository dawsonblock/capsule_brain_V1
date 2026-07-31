"""Headroom concentration analysis for v0.4.4.

A correctly computed per-task oracle cannot underperform the selected sham
action on the same complete action matrix.  Therefore a *negative* per-task
oracle-minus-sham headroom is an integrity error, not a benign measurement
artifact.  When any negative per-task headroom is present the analysis does
NOT produce a normal concentration classification; it returns
``status = "INVALID_DATA"`` and reports the offending contributions.

When the per-task headroom vector is nonnegative the module computes standard
concentration diagnostics:

* ``positive_headroom_total``      — sum of all nonnegative contributions.
* ``negative_headroom_count``      — number of strictly negative contributions.
* ``negative_headroom_total``      — sum of all strictly negative contributions.
* ``top_1_share``                  — largest contribution / positive total.
* ``top_5_share``                  — top-5 contributions / positive total.
* ``top_10_share``                 — top-10 contributions / positive total.
* ``top_10_percent_share``         — top ceil(10% of n) contributions / total.
* ``gini``                         — Gini coefficient over nonnegative contribs.
* ``hhi``                          — Herfindahl-Hirschman Index over normalized
                                     nonnegative contributions.
* ``effective_headroom_task_count`` — 1 / sum(normalized_share^2), the
                                      effective number of contributing tasks.

The Gini coefficient and HHI are computed with numpy.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _per_task_utilities(results: dict, n_tasks: int) -> list[float | None]:
    """Extract per-task utilities from a policy results dict.

    Resolution mirrors the canonical evaluator: a ``per_task`` list of records
    each carrying ``task_id`` and ``utility`` is preferred; otherwise the
    aggregate ``mean_utility`` is broadcast across all tasks.
    """
    per_task = results.get("per_task", [])
    if per_task:
        utils: list[float | None] = []
        for tr in per_task[:n_tasks]:
            u = tr.get("utility")
            if u is not None:
                u = float(u)
            utils.append(u)
        # Pad if the per-task list is shorter than n_tasks.
        while len(utils) < n_tasks:
            utils.append(None)
        return utils

    mean_util = results.get("mean_utility")
    if mean_util is not None:
        mean_util = float(mean_util)
    return [mean_util] * n_tasks


def _gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient over a nonnegative array.

    Uses the standard mean-absolute-difference form::

        G = (1 / (2 * n^2 * mean)) * sum_i sum_j |x_i - x_j|

    which is equivalent to the trapezoidal Lorenz-curve form.  Returns 0.0
    when the array is empty or has zero total.
    """
    n = values.size
    if n == 0:
        return 0.0
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    # Sort for a stable, deterministic computation.
    sorted_vals = np.sort(values)
    cum = np.cumsum(sorted_vals)
    # Lorenz-curve trapezoidal form: G = 1 - 2 * area_under_lorenz.
    # area_under_lorenz = sum((cum[i] + cum[i-1]) / 2) / total, with cum[-1]=0.
    lorenz_area = float(np.sum((cum[:-1] + cum[1:]) / 2.0)) / total if n > 1 else 1.0
    # The first trapezoid includes the origin; add the (0, cum[0]) segment.
    if n > 1:
        lorenz_area = (float(cum[0]) / 2.0 + float(np.sum((cum[:-1] + cum[1:]) / 2.0))) / total
    g = 1.0 - 2.0 * lorenz_area
    # Clamp to [0, 1] against floating-point drift.
    return float(max(0.0, min(1.0, g)))


def _hhi(shares: np.ndarray) -> float:
    """Compute the Herfindahl-Hirschman Index over normalized shares.

    HHI = sum(share_i^2), where shares sum to 1.  Ranges from 1/n (perfectly
    even) to 1.0 (single-task monopoly).  Returns 0.0 for an empty array.
    """
    n = shares.size
    if n == 0:
        return 0.0
    return float(np.sum(shares ** 2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_headroom_concentration(
    candidate_results: dict,
    oracle_results: dict,
    sham_results: dict,
    n_tasks: int = 30,
) -> dict:
    """Analyze the concentration of per-task oracle-minus-sham headroom.

    Args:
        candidate_results: Candidate policy results dict (accepted for API
            symmetry; not used in the concentration math itself).
        oracle_results: Oracle policy results dict with ``per_task`` or
            ``mean_utility``.
        sham_results: Sham policy results dict with ``per_task`` or
            ``mean_utility``.
        n_tasks: Number of tasks to consider (default 30).

    Returns:
        Dict with all concentration fields plus ``status`` and
        ``negative_headroom_present``.  When negative per-task headroom is
        present ``status`` is ``"INVALID_DATA"`` and no normal concentration
        classification is produced (the concentration fields are still
        reported over the nonnegative subset for diagnostics, but the caller
        must treat the run as an integrity failure).
    """
    oracle_utils = _per_task_utilities(oracle_results, n_tasks)
    sham_utils = _per_task_utilities(sham_results, n_tasks)

    # Build the per-task oracle-minus-sham headroom vector.
    headroom: list[float] = []
    for i in range(n_tasks):
        o = oracle_utils[i]
        s = sham_utils[i]
        if o is None or s is None:
            # Treat unavailable utilities as 0.0 so the vector stays complete;
            # a genuinely missing oracle is surfaced as zero headroom rather
            # than silently dropped.
            o = 0.0 if o is None else o
            s = 0.0 if s is None else s
        headroom.append(float(o) - float(s))

    arr = np.asarray(headroom, dtype=float)

    negative_mask = arr < -1e-12
    negative_headroom_count = int(np.count_nonzero(negative_mask))
    negative_headroom_total = float(arr[negative_mask].sum())
    negative_headroom_present = negative_headroom_count > 0

    nonneg = arr[arr >= 0.0]
    positive_headroom_total = float(nonneg.sum())

    # Concentration shares over the nonnegative subset.
    top_1_share = 0.0
    top_5_share = 0.0
    top_10_share = 0.0
    top_10_percent_share = 0.0
    gini = 0.0
    hhi = 0.0
    effective_headroom_task_count = 0.0

    if positive_headroom_total > 0.0 and nonneg.size > 0:
        sorted_desc = np.sort(nonneg)[::-1]
        total = positive_headroom_total

        top_1_share = float(sorted_desc[0] / total) if sorted_desc.size >= 1 else 0.0
        top_5_share = float(sorted_desc[:5].sum() / total) if sorted_desc.size >= 1 else 0.0
        top_10_share = float(sorted_desc[:10].sum() / total) if sorted_desc.size >= 1 else 0.0

        k_10pct = max(1, int(math.ceil(0.10 * nonneg.size)))
        top_10_percent_share = float(sorted_desc[:k_10pct].sum() / total)

        gini = _gini(nonneg)

        shares = nonneg / total
        hhi = _hhi(shares)

        if hhi > 0.0:
            effective_headroom_task_count = float(1.0 / hhi)
        else:
            effective_headroom_task_count = float(nonneg.size)

    status = "INVALID_DATA" if negative_headroom_present else "OK"

    return {
        "status": status,
        "negative_headroom_present": negative_headroom_present,
        "n_tasks": n_tasks,
        "positive_headroom_total": positive_headroom_total,
        "negative_headroom_count": negative_headroom_count,
        "negative_headroom_total": negative_headroom_total,
        "top_1_share": top_1_share,
        "top_5_share": top_5_share,
        "top_10_share": top_10_share,
        "top_10_percent_share": top_10_percent_share,
        "gini": gini,
        "hhi": hhi,
        "effective_headroom_task_count": effective_headroom_task_count,
        "per_task_headroom": headroom,
    }
