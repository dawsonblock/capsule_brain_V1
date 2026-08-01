"""Action-distribution collapse detection for v0.4.7.

Detects routing collapse where a candidate policy concentrates on a
single action, drops coverage of eligible actions, increases
abstention, or emits invalid actions.
"""
from __future__ import annotations

import math
from typing import Any

from qualification.autolearn_v04.v047.config import CollapseConfig


def _action_proportions(rows: list[dict]) -> dict[str, float]:
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


def _entropy(proportions: dict[str, float]) -> float:
    """Natural-log Shannon entropy (nats)."""
    if not proportions:
        return 0.0
    h = 0.0
    for p in proportions.values():
        if p > 0.0:
            h -= p * math.log(p)
    return h


def _js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence (natural log) between two distributions.

    Returns a value in [0, ln(2)].  Symmetric and finite.
    """
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def _kl(a: dict[str, float], b: dict[str, float]) -> float:
        kl = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            bk = b.get(k, 0.0)
            if ak > 0.0 and bk > 0.0:
                kl += ak * math.log(ak / bk)
        return kl

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _abstention_share(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    n = sum(1 for r in rows
            if r.get("abstained") is True
            or r.get("selected_action") is None
            or str(r.get("selected_action", "")).upper() == "ABSTAIN")
    return n / len(rows)


def _invalid_action_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    n = sum(1 for r in rows if r.get("invalid_action") is True)
    return n / len(rows)


def check_action_collapse(
    candidate_results: dict,
    baseline_results: dict,
    config: CollapseConfig,
) -> dict:
    """Detect routing collapse.

    Calculate per-policy action proportions and check:
    - max_single_action_share
    - min_action_coverage
    - max_abstention_increase
    - max_invalid_action_rate

    Returns dict with:
    - status: PASS | FAIL
    - entropy: float
    - max_action_share: float
    - action_coverage: int
    - js_divergence_from_baseline: float
    - abstention_share: float
    - invalid_action_rate: float
    - reasons: list[str]
    """
    cand_rows = candidate_results.get("task_rows", [])
    base_rows = baseline_results.get("task_rows", [])

    cand_props = _action_proportions(cand_rows)
    base_props = _action_proportions(base_rows)

    max_action_share = max(cand_props.values()) if cand_props else 1.0
    action_coverage = len(cand_props)
    entropy = _entropy(cand_props)
    js_div = _js_divergence(cand_props, base_props)
    abstention_share = _abstention_share(cand_rows)
    base_abstention = _abstention_share(base_rows)
    abstention_increase = abstention_share - base_abstention
    invalid_rate = _invalid_action_rate(cand_rows)

    reasons: list[str] = []
    status = "PASS"

    if max_action_share > config.max_single_action_share:
        reasons.append(
            f"max_single_action_share {max_action_share:.4f} > "
            f"{config.max_single_action_share}"
        )
        status = "FAIL"

    if action_coverage < config.min_action_coverage:
        reasons.append(
            f"action_coverage {action_coverage} < "
            f"{config.min_action_coverage}"
        )
        status = "FAIL"

    if abstention_increase > config.max_abstention_increase:
        reasons.append(
            f"abstention_increase {abstention_increase:.4f} > "
            f"{config.max_abstention_increase}"
        )
        status = "FAIL"

    if invalid_rate > config.max_invalid_action_rate:
        reasons.append(
            f"invalid_action_rate {invalid_rate:.4f} > "
            f"{config.max_invalid_action_rate}"
        )
        status = "FAIL"

    return {
        "status": status,
        "entropy": entropy,
        "max_action_share": max_action_share,
        "action_coverage": action_coverage,
        "js_divergence_from_baseline": js_div,
        "abstention_share": abstention_share,
        "baseline_abstention_share": base_abstention,
        "abstention_increase": abstention_increase,
        "invalid_action_rate": invalid_rate,
        "reasons": reasons,
    }
