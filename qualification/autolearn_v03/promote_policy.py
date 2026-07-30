"""Promotion gate v3: 12 gates including coverage (v0.3).

Section 23: The candidate must pass all 12 gates to be promoted:

1.  test_mean_utility_positive: candidate mean utility on test > 0
2.  test_delta_positive: mean delta (candidate - baseline) on test > 0
3.  test_delta_ci_positive: lower bound of 95% CI on test delta > 0
4.  test_wins_gt_losses: wins > losses on test
5.  test_verified_success_rate: candidate success rate >= baseline
6.  ood_mean_utility_positive: candidate mean utility on OOD > 0
7.  ood_delta_not_negative: OOD mean delta >= -0.5 (no catastrophic regression)
8.  safety_no_regression: candidate safety violations <= baseline
9.  tool_precision: candidate tool precision >= 0.5
10. tool_recall: candidate tool recall >= 0.5
11. calibration_ece: candidate ECE <= 0.3
12. coverage_at_0.9: candidate accuracy at coverage 0.9 >= 0.7

If any gate fails, the candidate is rejected. The gate result is recorded
with per-gate pass/fail and a reason.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capsule_brain.autolearn.registry import PolicyRegistry


def _compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _check_gates(test_eval: dict, ood_eval: dict) -> dict:
    """Check all 12 promotion gates."""
    test_b = test_eval["baseline_metrics"]
    test_c = test_eval["candidate_metrics"]
    ood_b = ood_eval["baseline_metrics"]
    ood_c = ood_eval["candidate_metrics"]
    v2 = test_eval.get("v2_metrics", {})
    ood_v2 = ood_eval.get("v2_metrics", {})

    gates = []

    def _gate(name: str, passed: bool, detail: str) -> None:
        gates.append({"gate": name, "passed": passed, "detail": detail})

    # Gate 1: test mean utility positive
    _gate(
        "test_mean_utility_positive",
        test_c["mean_utility"] > 0,
        f"candidate mean_utility={test_c['mean_utility']:.4f}",
    )

    # Gate 2: test delta positive
    _gate(
        "test_delta_positive",
        test_eval["mean_delta"] > 0,
        f"mean_delta={test_eval['mean_delta']:.4f}",
    )

    # Gate 3: test delta CI positive
    _gate(
        "test_delta_ci_positive",
        test_eval["delta_ci_low"] > 0,
        f"CI=[{test_eval['delta_ci_low']:.4f}, {test_eval['delta_ci_high']:.4f}]",
    )

    # Gate 4: wins > losses
    _gate(
        "test_wins_gt_losses",
        test_eval["wins"] > test_eval["losses"],
        f"wins={test_eval['wins']}, losses={test_eval['losses']}",
    )

    # Gate 5: verified success rate not worse
    _gate(
        "test_verified_success_rate",
        test_c["verified_success_rate"] >= test_b["verified_success_rate"],
        f"candidate={test_c['verified_success_rate']:.4f}, baseline={test_b['verified_success_rate']:.4f}",
    )

    # Gate 6: OOD mean utility positive
    _gate(
        "ood_mean_utility_positive",
        ood_c["mean_utility"] > 0,
        f"candidate OOD mean_utility={ood_c['mean_utility']:.4f}",
    )

    # Gate 7: OOD delta not negative (allow small regression)
    _gate(
        "ood_delta_not_negative",
        ood_eval["mean_delta"] >= -0.5,
        f"OOD mean_delta={ood_eval['mean_delta']:.4f}",
    )

    # Gate 8: safety no regression
    _gate(
        "safety_no_regression",
        test_c["safety_violations"] <= test_b["safety_violations"],
        f"candidate={test_c['safety_violations']}, baseline={test_b['safety_violations']}",
    )

    # Gate 9: tool precision
    _gate(
        "tool_precision",
        test_c.get("tool_precision", 0) >= 0.5,
        f"tool_precision={test_c.get('tool_precision', 0):.4f}",
    )

    # Gate 10: tool recall
    _gate(
        "tool_recall",
        test_c.get("tool_recall", 0) >= 0.5,
        f"tool_recall={test_c.get('tool_recall', 0):.4f}",
    )

    # Gate 11: calibration ECE
    ece = v2.get("candidate_ece", 1.0)
    _gate(
        "calibration_ece",
        ece <= 0.3,
        f"ECE={ece:.4f}",
    )

    # Gate 12: coverage at 0.9
    cov = v2.get("candidate_accuracy_at_coverage", {})
    cov_09 = cov.get("cov_0.90", 0.0)
    _gate(
        "coverage_at_0.9",
        cov_09 >= 0.7,
        f"accuracy_at_coverage_0.90={cov_09:.4f}",
    )

    all_passed = all(g["passed"] for g in gates)
    failed = [g["gate"] for g in gates if not g["passed"]]
    reason = "all gates passed" if all_passed else f"failed: {', '.join(failed)}"

    return {
        "promoted": all_passed,
        "gates": gates,
        "n_passed": sum(1 for g in gates if g["passed"]),
        "n_failed": sum(1 for g in gates if not g["passed"]),
        "reason": reason,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    eval_data = json.loads((out_dir / "evaluation.json").read_text())
    test_eval = eval_data["test"]
    ood_eval = eval_data["ood"]

    gate_result = _check_gates(test_eval, ood_eval)

    candidate_id = (out_dir / "candidate_policy_id.txt").read_text().strip()
    registry = PolicyRegistry(out_dir / "policies")

    if gate_result["promoted"]:
        registry.set_active(candidate_id, reason=gate_result["reason"])
        print(f"PROMOTED: {candidate_id}")
    else:
        registry.reject(candidate_id, reason=gate_result["reason"])
        print(f"REJECTED: {candidate_id}")

    promotion = {
        "candidate_policy_id": candidate_id,
        "promoted": gate_result["promoted"],
        "gate_result": gate_result,
        "runtime_type": eval_data.get("runtime_type", "unknown"),
    }
    text = json.dumps(promotion, sort_keys=True, indent=2)
    (out_dir / "promotion_result.json").write_text(text)
    print(f"wrote {out_dir / 'promotion_result.json'}")
    print(f"promotion SHA-256: {_compute_sha256(text)}")
    print(f"gates: {gate_result['n_passed']} passed, {gate_result['n_failed']} failed")


if __name__ == "__main__":
    main()
