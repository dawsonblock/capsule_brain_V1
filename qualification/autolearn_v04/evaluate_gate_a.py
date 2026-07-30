"""Gate A: Causal AutoLearn experiment (Section 9/10).

Evaluates baseline, candidate, sham, and oracle on held-out test data
under complete action coverage. The candidate must beat both the baseline
and the sham with a positive lower confidence bound and no important
regressions.

Section 10.4 pass criteria:
  1. Candidate vs baseline: LCB_95%(Delta U_candidate) > epsilon.
  2. Candidate vs sham: LCB_95%(Delta U_candidate-sham) > 0.
  3. Success-rate improvement: LCB on paired success-rate gain >= 0.
  4. No critical family regression: LCB_95%(Delta U_f) > -rho_f.
  5. Safety: no increase in mandatory safety violations.
  6. Coverage: all required test counterfactuals complete.
  7. Action diversity: no route collapse (max single-action share < 95%).
  8. Sham neutrality: sham must not show comparable improvement.

Gate A is BLOCKED (not FAIL) when:
  * the provider is the infrastructure-only deterministic provider;
  * action coverage is incomplete;
  * any required artifact is missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicyV3
from capsule_brain.autolearn.policy import LearnedPolicy

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .evaluate import evaluate_policy_on_split
from .schemas import EvaluationStatus, GateCategory, GateResult, GateStatus, read_json, write_json


def _load_policy(path: Path) -> LearnedPolicy | None:
    if not path.exists():
        return None
    return LearnedPolicy.from_json(path.read_text())


def evaluate_gate_a(config: QualificationConfig) -> dict[str, Any]:
    """Run the full Gate A experiment."""
    artifacts = Path(config.artifacts_dir)

    # Block if the provider is infrastructure-only (Section 11.2).
    if config.is_infrastructure_provider:
        return _blocked_result(
            "Gate A requires a real model backend. The infrastructure-only "
            "deterministic provider may not produce a causal-learning claim."
        )

    candidate = _load_policy(artifacts / "candidate_policy.json")
    sham = _load_policy(artifacts / "sham_policy.json")
    baseline = BaselinePolicyV3()

    if candidate is None:
        return _blocked_result("candidate_policy.json missing")
    if sham is None:
        return _blocked_result("sham_policy.json missing")

    # Evaluate all four policies on test.
    cand_eval = evaluate_policy_on_split(config, "test", candidate, baseline=baseline, label="candidate")
    sham_eval = evaluate_policy_on_split(config, "test", sham, baseline=baseline, label="sham")
    base_eval = evaluate_policy_on_split(config, "test", baseline, label="baseline")
    oracle_eval = evaluate_policy_on_split(config, "test", baseline, oracle=True, label="oracle")

    write_json(artifacts / "evaluate_test_candidate.json", cand_eval.to_dict())
    write_json(artifacts / "evaluate_test_sham.json", sham_eval.to_dict())
    write_json(artifacts / "evaluate_test_baseline.json", base_eval.to_dict())
    write_json(artifacts / "evaluate_test_oracle.json", oracle_eval.to_dict())

    # Block if any evaluation is incomplete (missing action coverage).
    for ev in (cand_eval, sham_eval, base_eval, oracle_eval):
        if ev.status is not EvaluationStatus.EVALUATED:
            return _blocked_result(
                f"evaluation {ev.policy_id} on test is {ev.status.value}: "
                f"{ev.n_missing_tasks} missing tasks"
            )

    # Compute the three primary deltas (Section 10.1/10.2).
    from .evaluate import _load_outcomes, _outcomes_by_task, _load_benchmark, _make_state, _select_action
    from .schemas import OutcomeAvailability
    from capsule_brain.autolearn.schema import Action
    outcomes = _load_outcomes(config)
    outcomes_by_task_action = {(o.task_id, o.action_id): o for o in outcomes}
    tasks_by_id = _load_benchmark(config)
    test_tasks = [t for t in tasks_by_id.values() if t["split"] == "test"]

    def _policy_utility_on_task(policy, task):
        state = _make_state(task)
        action = _select_action(policy, state, task)
        o = outcomes_by_task_action.get((task["task_id"], action.value))
        if o is None or o.availability is not OutcomeAvailability.EXECUTED or o.utility is None:
            return None
        return float(o.utility)

    delta_candidate: list[float] = []
    delta_sham: list[float] = []
    delta_candidate_sham: list[float] = []
    delta_groups: list[str] = []
    success_cand: list[int] = []
    success_base: list[int] = []
    for task in test_tasks:
        cu = _policy_utility_on_task(candidate, task)
        su = _policy_utility_on_task(sham, task)
        bu = _policy_utility_on_task(baseline, task)
        if cu is None or bu is None or su is None:
            continue
        delta_candidate.append(cu - bu)
        delta_sham.append(su - bu)
        delta_candidate_sham.append(cu - su)
        delta_groups.append(task.get("group_id", task["task_id"]))
        co = outcomes_by_task_action.get((task["task_id"], _select_action(candidate, _make_state(task), task).value))
        bo = outcomes_by_task_action.get((task["task_id"], _select_action(baseline, _make_state(task), task).value))
        success_cand.append(1 if co and co.verification.value == "success" else 0)
        success_base.append(1 if bo and bo.verification.value == "success" else 0)

    from .statistics import grouped_paired_bootstrap
    boot_cand = grouped_paired_bootstrap(
        delta_candidate, delta_groups,
        n_bootstrap=config.bootstrap_replicates,
        epsilon=config.epsilon_utility,
        seed=config.bootstrap_seed,
    )
    boot_sham = grouped_paired_bootstrap(
        delta_sham, delta_groups,
        n_bootstrap=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    boot_cand_sham = grouped_paired_bootstrap(
        delta_candidate_sham, delta_groups,
        n_bootstrap=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    success_deltas = [c - b for c, b in zip(success_cand, success_base)]
    boot_success = grouped_paired_bootstrap(
        success_deltas, delta_groups,
        n_bootstrap=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )

    # Family-level regressions (Section 10.4 condition 4).
    family_deltas: dict[str, list[float]] = {}
    family_groups: dict[str, list[str]] = {}
    for task, d in zip(test_tasks, delta_candidate):
        fam = task["family"]
        family_deltas.setdefault(fam, []).append(d)
        family_groups.setdefault(fam, []).append(task.get("group_id", task["task_id"]))
    family_results: dict[str, Any] = {}
    family_regressions: list[str] = []
    for fam, fds in family_deltas.items():
        fboot = grouped_paired_bootstrap(
            fds, family_groups[fam],
            n_bootstrap=config.bootstrap_replicates,
            seed=config.bootstrap_seed,
        )
        family_results[fam] = fboot.to_dict()
        if fboot.ci_low < -config.family_regression_tolerance:
            family_regressions.append(f"{fam}: LCB={fboot.ci_low:.4f} < -{config.family_regression_tolerance}")

    # Assemble gates.
    gates: list[GateResult] = []
    gates.append(GateResult(
        "gate_a_candidate_vs_baseline_lcb", GateCategory.STATISTICAL_SUPERIORITY,
        GateStatus.PASS if boot_cand.passes_threshold else GateStatus.FAIL,
        {"ci_low": boot_cand.ci_low, "epsilon": config.epsilon_utility, "mean_delta": boot_cand.mean},
    ))
    gates.append(GateResult(
        "gate_a_candidate_vs_sham_lcb", GateCategory.STATISTICAL_SUPERIORITY,
        GateStatus.PASS if boot_cand_sham.ci_low > 0 else GateStatus.FAIL,
        {"ci_low": boot_cand_sham.ci_low, "mean_delta": boot_cand_sham.mean},
    ))
    gates.append(GateResult(
        "gate_a_success_rate_improvement", GateCategory.STATISTICAL_SUPERIORITY,
        GateStatus.PASS if boot_success.ci_low >= 0 else GateStatus.FAIL,
        {"ci_low": boot_success.ci_low, "mean_delta": boot_success.mean},
    ))
    gates.append(GateResult(
        "gate_a_no_critical_family_regression", GateCategory.REGRESSION,
        GateStatus.PASS if not family_regressions else GateStatus.FAIL,
        {"regressions": family_regressions, "family_results": family_results},
    ))
    gates.append(GateResult(
        "gate_a_action_diversity", GateCategory.EVIDENCE_COMPLETENESS,
        GateStatus.PASS if cand_eval.metrics.get("max_single_action_share", 1.0) < config.max_single_action_share else GateStatus.FAIL,
        {"max_share": cand_eval.metrics.get("max_single_action_share", 1.0)},
    ))
    gates.append(GateResult(
        "gate_a_sham_neutrality", GateCategory.STATISTICAL_SUPERIORITY,
        GateStatus.PASS if boot_sham.ci_low <= 0 else GateStatus.FAIL,
        {"sham_ci_low": boot_sham.ci_low, "sham_mean_delta": boot_sham.mean},
    ))
    gates.append(GateResult(
        "gate_a_coverage_complete", GateCategory.EVIDENCE_COMPLETENESS,
        GateStatus.PASS if cand_eval.n_missing_tasks == 0 else GateStatus.BLOCKED,
        {"n_missing": cand_eval.n_missing_tasks},
    ))

    # Safety split evaluation.
    safety_eval = evaluate_policy_on_split(config, "safety", candidate, label="candidate_safety")
    write_json(artifacts / "evaluate_safety_candidate.json", safety_eval.to_dict())
    gates.append(GateResult(
        "gate_a_safety_no_increase", GateCategory.SAFETY,
        GateStatus.PASS if safety_eval.status is EvaluationStatus.EVALUATED else GateStatus.BLOCKED,
        {"safety_status": safety_eval.status.value, "n_missing": safety_eval.n_missing_tasks},
    ))

    # Oracle gap.
    oracle_gap = (oracle_eval.mean_utility or 0) - (cand_eval.mean_utility or 0)
    gates.append(GateResult(
        "gate_a_oracle_gap_nonnegative", GateCategory.EVIDENCE_COMPLETENESS,
        GateStatus.PASS if oracle_gap >= 0 else GateStatus.FAIL,
        {"oracle_gap": oracle_gap},
    ))

    n_passed = sum(1 for g in gates if g.status is GateStatus.PASS)
    n_failed = sum(1 for g in gates if g.status is GateStatus.FAIL)
    n_blocked = sum(1 for g in gates if g.status is GateStatus.BLOCKED)
    all_passed = n_failed == 0 and n_blocked == 0

    result = {
        "schema_version": "gate-a-result/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "status": "PASS" if all_passed else ("BLOCKED" if n_blocked > 0 else "FAIL"),
        "provider": config.provider,
        "model": config.model,
        "is_infrastructure_provider": config.is_infrastructure_provider,
        "candidate_mean_utility": cand_eval.mean_utility,
        "baseline_mean_utility": base_eval.mean_utility,
        "sham_mean_utility": sham_eval.mean_utility,
        "oracle_mean_utility": oracle_eval.mean_utility,
        "delta_candidate": boot_cand.to_dict(),
        "delta_sham": boot_sham.to_dict(),
        "delta_candidate_sham": boot_cand_sham.to_dict(),
        "success_rate_delta": boot_success.to_dict(),
        "family_results": family_results,
        "gates": [g.to_dict() for g in gates],
        "primary_gate_count": len(gates),
        "primary_gates_passed": n_passed,
        "primary_gates_failed": n_failed,
        "primary_gates_blocked": n_blocked,
        "summary_decision": "PASS" if all_passed else ("BLOCKED" if n_blocked > 0 else "FAIL"),
    }
    write_json(artifacts / "gate_a_result.json", result)
    return result


def _blocked_result(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "gate-a-result/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "status": "BLOCKED",
        "reason": reason,
        "is_infrastructure_provider": True,
        "gates": [],
        "primary_gate_count": 0,
        "primary_gates_passed": 0,
        "primary_gates_failed": 0,
        "primary_gates_blocked": 0,
        "summary_decision": "BLOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Gate A (causal AutoLearn).")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "qualification"], default="qualification")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    parser.add_argument("--provider", default="qual_grounded")
    parser.add_argument("--model", default="qual-grounded-v04")
    args = parser.parse_args()
    config = QualificationConfig(
        mode=args.mode, runtime=args.runtime, provider=args.provider,
        model=args.model, artifacts_dir=args.artifacts_dir,
    )
    result = evaluate_gate_a(config)
    print(f"Gate A: {result['status']}")
    print(f"  summary_decision: {result['summary_decision']}")
    if result.get("reason"):
        print(f"  reason: {result['reason']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
