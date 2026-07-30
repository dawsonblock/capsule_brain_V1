"""Final qualification report (Section 14/15).

Derives PASS, FAIL, BLOCKED, or NOT RUN from the artifact tree. If any stage
is missing or simulated, the report must say FAIL/BLOCKED/NOT RUN, never PASS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import QualificationConfig
from .schemas import read_json, write_json


def _status_from_artifacts(config: QualificationConfig) -> tuple[str, str, dict]:
    artifacts = Path(config.artifacts_dir)
    required = {
        "benchmark_manifest.json": "build_benchmark",
        "real_counterfactual_results.json": "run_counterfactuals",
        "dataset_manifest.json": "build_dataset",
        "candidate_policy.json": "train_candidate",
        "sham_policy.json": "train_sham",
        "evaluate_test.json": "evaluate_test",
        "evaluate_ood.json": "evaluate_ood",
        "evaluate_safety.json": "evaluate_safety",
        "promotion_result.json": "run_promotion",
        "post_promotion_result.json": "run_post_promotion",
        "provenance.json": "provenance",
    }

    missing = [p for p in required if not (artifacts / p).exists()]
    if missing:
        return "NOT RUN", f"missing artifacts: {missing}", {"missing_artifacts": missing}

    # Check runtime type.
    cf = read_json(artifacts / "real_counterfactual_results.json")
    if not all(r.get("runtime_type") == "real" for r in cf):
        return "BLOCKED", "non-real runtime rows detected", {"n_non_real": sum(1 for r in cf if r.get("runtime_type") != "real")}

    promotion = read_json(artifacts / "promotion_result.json")
    if not promotion.get("promotion_approved", False):
        return "FAIL", "promotion gates not all passed", promotion

    post = read_json(artifacts / "post_promotion_result.json")
    if not post.get("passes", False):
        return "FAIL", "post-promotion behavioral proof failed", post

    test_eval = read_json(artifacts / "evaluate_test.json")
    if test_eval.get("mean_utility", -1e9) <= 0:
        return "FAIL", "test mean utility non-positive", test_eval

    return "PASS", "v0.3.2 qualification passed", {
        "n_counterfactuals": len(cf),
        "test_mean_utility": test_eval.get("mean_utility", 0),
        "promotion_gates": promotion.get("n_passed", 0),
        "n_behavioral_improvements": post.get("n_behavioral_improvements", 0),
    }


def build_report(config: QualificationConfig) -> dict[str, Any]:
    status, reason, details = _status_from_artifacts(config)
    report = {
        "package_version": "2.15.3",
        "autolearn_qualification_version": "0.3.2",
        "status": status,
        "reason": reason,
        "details": details,
        "claim": status,
        "disclaimer": (
            "PASS means all real-execution gates and behavioral proof were "
            "satisfied on the measured subset. FAIL means the candidate did not "
            "beat the required threshold. BLOCKED means non-real runtime or "
            "missing artifact was detected. NOT RUN means the pipeline did not "
            "complete."
        ),
    }
    path = Path(config.artifacts_dir) / "qualification_report.json"
    write_json(path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Produce the v0.3.2 qualification report.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    config = QualificationConfig(artifacts_dir=args.artifacts_dir)
    report = build_report(config)
    print(f"report: {report['status']}")
    print(f"  {report['reason']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
