"""Gate A3 — Robustness and replication for v0.4.7.

Validates that the candidate's causal effectiveness is stable across
generation and learner seeds, that no seed-level sign reversal exceeds
the catastrophic-effect floor, that family-stratified evaluation shows
no critical regression, and that there is no action-distribution
collapse or safety-event increase.
"""
from __future__ import annotations

from typing import Any

from qualification.autolearn_v04.common.audit_status import AuditStatus
from qualification.autolearn_v04.v047.config import GateA3Config
from qualification.autolearn_v04.v047.gate_schema import GateA3Result


def _status_str(s: str) -> str:
    return AuditStatus.from_str(s).value


def evaluate_gate_a3(
    replicate_results: list[dict],
    family_evaluation: dict,
    collapse_check: dict,
    safety_check: dict,
    config: GateA3Config,
) -> GateA3Result:
    """Evaluate robustness and replication.

    Each replicate_result dict has:
    - generation_seed: int
    - learner_seed: int
    - candidate_vs_baseline_delta: float
    - candidate_vs_sham_delta: float
    - candidate_vs_baseline_passes: bool
    - candidate_vs_sham_passes: bool

    Gate A3 validates:
    - At least min_generation_seeds generation seeds
    - At least min_learner_seeds learner seeds
    - No seed-level sign reversal beyond catastrophic_effect_floor
    - min_replication_pass_rate fraction of replicates pass
    - Family-stratified evaluation (from family_evaluation)
    - No critical task-family regression
    - No action-distribution collapse (from collapse_check)
    - No safety-event increase (from safety_check)

    Returns GateA3Result.
    """
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    # --- Seed diversity -------------------------------------------------
    generation_seeds = {r.get("generation_seed") for r in replicate_results
                        if r.get("generation_seed") is not None}
    learner_seeds = {r.get("learner_seed") for r in replicate_results
                     if r.get("learner_seed") is not None}
    n_gen = len(generation_seeds)
    n_learn = len(learner_seeds)
    n_replicates = len(replicate_results)

    # Check if generation is deterministic (greedy decoding).
    # If so, multiple generation seeds provide no statistical variation
    # and we should not require them.
    generation_deterministic = any(
        r.get("generation_deterministic") is True for r in replicate_results
    )

    checks["n_generation_seeds"] = n_gen
    checks["n_learner_seeds"] = n_learn
    checks["n_replicates"] = n_replicates
    checks["generation_deterministic"] = generation_deterministic

    seed_ok = True
    if generation_deterministic:
        # Deterministic generation: only require 1 generation seed.
        if n_gen < 1:
            reasons.append("no generation seeds (deterministic mode requires at least 1)")
            seed_ok = False
    elif n_gen < config.min_generation_seeds:
        reasons.append(
            f"insufficient generation seeds: {n_gen} < "
            f"{config.min_generation_seeds}"
        )
        seed_ok = False
    if n_learn < config.min_learner_seeds:
        reasons.append(
            f"insufficient learner seeds: {n_learn} < "
            f"{config.min_learner_seeds}"
        )
        seed_ok = False

    # --- Sign-reversal / catastrophic floor -----------------------------
    catastrophic_reversals: list[str] = []
    for r in replicate_results:
        delta = r.get("candidate_vs_baseline_delta", 0.0)
        if delta < config.catastrophic_effect_floor:
            catastrophic_reversals.append(
                f"gen={r.get('generation_seed')},learn={r.get('learner_seed')}: "
                f"delta={delta:.4f} < floor={config.catastrophic_effect_floor}"
            )
    checks["catastrophic_reversals"] = catastrophic_reversals
    if catastrophic_reversals:
        reasons.append(
            f"{len(catastrophic_reversals)} replicate(s) below catastrophic "
            f"effect floor {config.catastrophic_effect_floor}"
        )

    # --- Replication pass rate ------------------------------------------
    n_pass = sum(
        1 for r in replicate_results
        if r.get("candidate_vs_baseline_passes") is True
        and r.get("candidate_vs_sham_passes") is True
    )
    pass_rate = (n_pass / n_replicates) if n_replicates else 0.0
    checks["replication_pass_rate"] = pass_rate
    checks["n_replicates_passing"] = n_pass
    replication_ok = pass_rate >= config.min_replication_pass_rate
    if not replication_ok:
        reasons.append(
            f"replication pass rate {pass_rate:.4f} < "
            f"{config.min_replication_pass_rate}"
        )

    # --- Family-stratified evaluation -----------------------------------
    critical_regressions = family_evaluation.get("critical_regressions", [])
    n_sufficient = family_evaluation.get("n_sufficient", 0)
    n_insufficient = family_evaluation.get("n_insufficient", 0)
    checks["family_n_sufficient"] = n_sufficient
    checks["family_n_insufficient"] = n_insufficient
    checks["family_critical_regressions"] = critical_regressions
    family_ok = len(critical_regressions) == 0
    if critical_regressions:
        reasons.append(
            f"{len(critical_regressions)} critical family regression(s): "
            + "; ".join(critical_regressions)
        )

    # --- Collapse check -------------------------------------------------
    collapse_status = _status_str(collapse_check.get("status", AuditStatus.NOT_RUN.value))
    checks["collapse_status"] = collapse_status
    checks["collapse_reasons"] = collapse_check.get("reasons", [])
    collapse_ok = collapse_status == AuditStatus.PASS.value
    if not collapse_ok:
        reasons.append(
            f"action-distribution collapse: {collapse_check.get('reasons', [])}"
        )

    # --- Safety check ---------------------------------------------------
    safety_status = _status_str(safety_check.get("status", AuditStatus.NOT_RUN.value))
    checks["safety_status"] = safety_status
    safety_ok = safety_status == AuditStatus.PASS.value
    if not safety_ok:
        reasons.append(
            f"safety-event increase detected (status={safety_status})"
        )

    # --- Overall --------------------------------------------------------
    all_ok = (
        seed_ok
        and not catastrophic_reversals
        and replication_ok
        and family_ok
        and collapse_ok
        and safety_ok
    )
    status = AuditStatus.PASS.value if all_ok else AuditStatus.FAIL.value

    replicate_summary = {
        "n_replicates": n_replicates,
        "n_generation_seeds": n_gen,
        "n_learner_seeds": n_learn,
        "generation_deterministic": generation_deterministic,
        "replication_pass_rate": pass_rate,
        "n_replicates_passing": n_pass,
        "catastrophic_reversals": catastrophic_reversals,
        "seed_diversity_ok": seed_ok,
        "replication_ok": replication_ok,
    }

    family_summary = {
        "n_sufficient": n_sufficient,
        "n_insufficient": n_insufficient,
        "critical_regressions": critical_regressions,
        "family_ok": family_ok,
        "collapse_status": collapse_status,
        "safety_status": safety_status,
    }

    return GateA3Result(
        status=status,
        replicate_summary=replicate_summary,
        family_summary=family_summary,
        reasons=reasons,
    )
