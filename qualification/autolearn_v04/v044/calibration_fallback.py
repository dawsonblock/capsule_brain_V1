"""Calibration and fallback suppression audit for v0.4.4.

Quantifies the calibration and fallback effects on the candidate and decides
whether either suppresses *useful* routing — i.e. routing that would have
beaten the sham reference had it been allowed to execute.

The audit consumes the raw-vs-executed analysis (which records per-task
confidence/abstention, shift-guard, and action-mask suppression counts) plus
the candidate and sham results.  It reports:

* ``confidence_suppression_count``  — routing suppressed by confidence/abstention.
* ``shift_suppression_count``       — routing suppressed by the shift guard.
* ``mask_suppression_count``        — routing suppressed by the action mask.
* ``total_suppressed_routing``      — sum of the three suppression counts.
* ``calibration_suppresses_useful_routing`` — bool.
* ``fallback_suppresses_useful_routing``    — bool.
* ``conclusion`` — one of ``NO_SUPPRESSION``, ``CALIBRATION_SUPPRESSES``,
  ``FALLBACK_SUPPRESSES``, ``BOTH_SUPPRESS``.

Calibration suppression = confidence/abstention + action-mask suppression
(these are the calibration-stage gates).  Fallback suppression = shift-guard
suppression (the fallback-stage gate).  Either is "useful" when the candidate
mean utility is at or above the sham mean — i.e. the suppressed routing was
competitive with the reference.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suppression_count(section: dict | None, key: str = "suppressed_routing_count") -> int:
    """Read a suppression count from a raw-vs-executed sub-section."""
    if not isinstance(section, dict):
        return 0
    val = section.get(key, 0)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _mean_utility(results: dict) -> float | None:
    """Read the mean utility from a results dict."""
    mu = results.get("mean_utility")
    if mu is None:
        return None
    try:
        return float(mu)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_calibration_fallback(
    raw_vs_executed: dict,
    candidate_results: dict,
    sham_results: dict,
) -> dict:
    """Audit calibration and fallback suppression effects on the candidate.

    Args:
        raw_vs_executed: Output of ``analyze_raw_vs_executed`` carrying the
            ``calibration_suppression``, ``shift_guard_suppression``, and
            ``action_mask_suppression`` sub-sections (and optionally their
            ``conclusions`` list).
        candidate_results: Candidate policy results dict.
        sham_results: Sham policy results dict.

    Returns:
        Dict with ``status``, the three suppression counts,
        ``total_suppressed_routing``, the two ``*_suppresses_useful_routing``
        booleans, and a ``conclusion`` string.
    """
    calibration_section = raw_vs_executed.get("calibration_suppression", {})
    shift_section = raw_vs_executed.get("shift_guard_suppression", {})
    mask_section = raw_vs_executed.get("action_mask_suppression", {})

    confidence_suppression_count = _suppression_count(calibration_section)
    shift_suppression_count = _suppression_count(shift_section)
    mask_suppression_count = _suppression_count(mask_section)

    total_suppressed_routing = (
        confidence_suppression_count
        + shift_suppression_count
        + mask_suppression_count
    )

    cand_mean = _mean_utility(candidate_results)
    sham_mean = _mean_utility(sham_results)

    # "Useful" routing = routing that is at least as good as the sham
    # reference.  When the candidate mean is unavailable we cannot establish
    # usefulness, so we conservatively treat suppression as not-useful.
    useful = False
    if cand_mean is not None and sham_mean is not None:
        useful = cand_mean >= sham_mean

    calibration_suppression_total = confidence_suppression_count + mask_suppression_count
    fallback_suppression_total = shift_suppression_count

    calibration_suppresses_useful_routing = (
        calibration_suppression_total > 0 and useful
    )
    fallback_suppresses_useful_routing = (
        fallback_suppression_total > 0 and useful
    )

    if calibration_suppresses_useful_routing and fallback_suppresses_useful_routing:
        conclusion = "BOTH_SUPPRESS"
    elif calibration_suppresses_useful_routing:
        conclusion = "CALIBRATION_SUPPRESSES"
    elif fallback_suppresses_useful_routing:
        conclusion = "FALLBACK_SUPPRESSES"
    else:
        conclusion = "NO_SUPPRESSION"

    status = "FAIL" if conclusion != "NO_SUPPRESSION" else "PASS"

    return {
        "status": status,
        "confidence_suppression_count": confidence_suppression_count,
        "shift_suppression_count": shift_suppression_count,
        "mask_suppression_count": mask_suppression_count,
        "total_suppressed_routing": total_suppressed_routing,
        "calibration_suppresses_useful_routing": calibration_suppresses_useful_routing,
        "fallback_suppresses_useful_routing": fallback_suppresses_useful_routing,
        "conclusion": conclusion,
    }
