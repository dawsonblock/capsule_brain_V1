"""Honest qualification report generator for v0.4.0 (Section 16).

v0.4.0 fixes:
  * Smoke mode reports PIPELINE_INTEGRITY: PASS/FAIL,
    SCIENTIFIC_QUALIFICATION: NOT_RUN, PROMOTION_DECISION: BLOCKED.
  * Honest claim language: PASS/FAIL/BLOCKED/NOT_RUN, never pseudo-PASS.
  * The report includes the disclaimer and the full gate list.
  * Infrastructure-provider runs are labeled as infrastructure-only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .schemas import read_json, write_json


DISCLAIMER = (
    "PASS means all real-execution gates and behavioral proof were "
    "satisfied on the measured subset. FAIL means the candidate did not "
    "beat the required threshold. BLOCKED means non-real runtime, "
    "infrastructure-only provider for a causal claim, or missing artifact "
    "was detected. NOT RUN means the pipeline did not complete."
)


def generate_report(config: QualificationConfig) -> dict[str, Any]:
    artifacts = Path(config.artifacts_dir)

    # Load all stage results.
    promotion = read_json(artifacts / "promotion_result.json") if (artifacts / "promotion_result.json").exists() else None
    gate_a = read_json(artifacts / "gate_a_result.json") if (artifacts / "gate_a_result.json").exists() else None
    gate_b = read_json(artifacts / "gate_b_result.json") if (artifacts / "gate_b_result.json").exists() else None
    post_promo = read_json(artifacts / "post_promotion_result.json") if (artifacts / "post_promotion_result.json").exists() else None
    provenance = read_json(artifacts / "provenance.json") if (artifacts / "provenance.json").exists() else None

    if promotion is None:
        return _not_run_result(config, "promotion_result.json missing")

    # Smoke mode: pipeline integrity only.
    if config.is_smoke:
        report = {
            "schema_version": "qualification-report/2",
            "protocol_version": PROTOCOL_VERSION,
            "package_version": PACKAGE_VERSION,
            "autolearn_version": AUTOLEARN_VERSION,
            "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
            "mode": "smoke",
            "runtime": config.runtime,
            "provider": config.provider,
            "PIPELINE_INTEGRITY": promotion.get("PIPELINE_INTEGRITY", "FAIL"),
            "SCIENTIFIC_QUALIFICATION": "NOT_RUN",
            "PROMOTION_DECISION": "BLOCKED",
            "status": "BLOCKED",
            "claim": "BLOCKED",
            "reason": "smoke mode: scientific qualification not run, promotion blocked",
            "stages": promotion.get("stages", {}),
            "disclaimer": DISCLAIMER,
        }
        write_json(artifacts / "qualification_report.json", report)
        return report

    # Qualification mode.
    status = promotion.get("status", "BLOCKED")
    claim = _claim_language(status, config)

    report = {
        "schema_version": "qualification-report/2",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "mode": config.mode,
        "runtime": config.runtime,
        "provider": config.provider,
        "is_infrastructure_provider": config.is_infrastructure_provider,
        "PIPELINE_INTEGRITY": "PASS",
        "SCIENTIFIC_QUALIFICATION": status,
        "PROMOTION_DECISION": status,
        "status": status,
        "claim": claim,
        "reason": promotion.get("reason", ""),
        "gate_a": {
            "status": gate_a.get("status") if gate_a else "NOT_RUN",
            "summary_decision": gate_a.get("summary_decision") if gate_a else "NOT_RUN",
            "candidate_mean_utility": gate_a.get("candidate_mean_utility") if gate_a else None,
            "baseline_mean_utility": gate_a.get("baseline_mean_utility") if gate_a else None,
            "sham_mean_utility": gate_a.get("sham_mean_utility") if gate_a else None,
            "oracle_mean_utility": gate_a.get("oracle_mean_utility") if gate_a else None,
        } if gate_a else {"status": "NOT_RUN"},
        "gate_b": {
            "status": gate_b.get("status") if gate_b else "NOT_RUN",
            "best_auroc": gate_b.get("best_auroc") if gate_b else None,
            "margin_over_chance": gate_b.get("margin_over_chance") if gate_b else None,
            "best_nonlatent_auroc": gate_b.get("best_nonlatent_auroc") if gate_b else None,
        } if gate_b else {"status": "NOT_RUN"},
        "post_promotion": {
            "passes": post_promo.get("passes") if post_promo else False,
            "n_behavioral_improvements": post_promo.get("n_behavioral_improvements", 0) if post_promo else 0,
            "utility_delta": post_promo.get("utility_delta") if post_promo else None,
        } if post_promo else {"passes": False},
        "promotion": {
            "primary_gates_passed": promotion.get("primary_gates_passed", 0),
            "primary_gate_count": promotion.get("primary_gate_count", 0),
            "primary_gates_failed": promotion.get("primary_gates_failed", 0),
            "primary_gates_blocked": promotion.get("primary_gates_blocked", 0),
            "summary_decision": promotion.get("summary_decision", "BLOCKED"),
        },
        "provenance": {
            "source_hash": provenance.get("source_hash", "") if provenance else "",
            "final_hash": provenance.get("final_hash", "") if provenance else "",
        } if provenance else {},
        "gates": promotion.get("gates", []),
        "disclaimer": DISCLAIMER,
    }
    write_json(artifacts / "qualification_report.json", report)
    return report


def _claim_language(status: str, config: QualificationConfig) -> str:
    if config.is_infrastructure_provider:
        return (
            "INFRASTRUCTURE-INTEGRATION ONLY: The deterministic grounded "
            "provider was used. This qualifies infrastructure only. Gate A "
            "(causal learning) is BLOCKED. No causal-learning claim is made."
        )
    if status == "PASS":
        return (
            "PASS: The candidate executive policy, trained from real "
            "counterfactual experience, improves held-out verified utility "
            "over both the frozen baseline and a sham-learning control, "
            "with no critical family regressions, and the passive CLUE "
            "latent verifier predicts verified success better than chance "
            "and stronger non-latent confidence baselines."
        )
    if status == "FAIL":
        return (
            "FAIL: The candidate did not beat the required threshold on at "
            "least one primary evidence gate. No promotion."
        )
    if status == "BLOCKED":
        return (
            "BLOCKED: A required artifact, runtime, or provider condition "
            "was not met. Scientific qualification could not complete. No "
            "promotion."
        )
    return "NOT RUN: The pipeline did not complete."


def _not_run_result(config: QualificationConfig, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "qualification-report/2",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "status": "NOT RUN",
        "claim": "NOT RUN",
        "reason": reason,
        "disclaimer": DISCLAIMER,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the v0.4.0 qualification report.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "qualification"], default="qualification")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    parser.add_argument("--provider", default="qual_grounded")
    args = parser.parse_args()
    config = QualificationConfig(
        mode=args.mode, runtime=args.runtime, provider=args.provider,
        artifacts_dir=args.artifacts_dir,
    )
    report = generate_report(config)
    print(f"report: {report['status']}")
    print(f"  PIPELINE_INTEGRITY: {report.get('PIPELINE_INTEGRITY', 'N/A')}")
    print(f"  SCIENTIFIC_QUALIFICATION: {report.get('SCIENTIFIC_QUALIFICATION', 'N/A')}")
    print(f"  PROMOTION_DECISION: {report.get('PROMOTION_DECISION', 'N/A')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
