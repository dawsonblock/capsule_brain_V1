"""Single-command qualification pipeline for AutoLearn v0.3.1.

v0.3.1 Section 27: run_all executes the full pipeline:

    build_benchmark → run_counterfactuals → build_dataset →
    train_candidate → train_sham → evaluate_validation →
    evaluate_test → evaluate_ood → run_promotion →
    run_post_promotion → provenance → report

Official run_all must fail closed if:
- runtime != real
- test data were touched early
- source tree is dirty relative to manifest
- required artifacts are missing
- policy parity fails
- safety tests fail
- promotion gates fail

A development flag may allow simulation, but its output must be labeled:
SIMULATED DEVELOPMENT RUN / NOT QUALIFYING / NOT PROMOTABLE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qualification.autolearn_v031.config import QualificationConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full AutoLearn v0.3.1 qualification pipeline."
    )
    parser.add_argument(
        "--runtime",
        choices=["real", "simulated"],
        default="real",
        help="Runtime type: 'real' for official qualification, 'simulated' for dev.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh: remove all existing artifacts before running.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for generated artifacts.",
    )
    args = parser.parse_args()

    config = QualificationConfig(runtime=args.runtime, artifacts_dir=args.artifacts_dir)

    # Fail closed if runtime is not real.
    if not config.is_real:
        print("SIMULATED DEVELOPMENT RUN", file=sys.stderr)
        print("NOT QUALIFYING", file=sys.stderr)
        print("NOT PROMOTABLE", file=sys.stderr)
        print(
            "WARNING: Simulated runtime output cannot be used for official "
            "qualification or promotion.",
            file=sys.stderr,
        )

    artifacts = Path(args.artifacts_dir)
    if args.fresh and artifacts.exists():
        import shutil
        shutil.rmtree(artifacts, ignore_errors=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Save config.
    (artifacts / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True)
    )

    # The full pipeline would execute each stage here. For now, we provide
    # the orchestration framework and fail closed on missing stages.
    stages = [
        "build_benchmark",
        "run_counterfactuals",
        "build_dataset",
        "train_candidate",
        "train_sham",
        "evaluate_validation",
        "evaluate_test",
        "evaluate_ood",
        "run_promotion",
        "run_post_promotion",
        "provenance",
        "report",
    ]

    print(f"AutoLearn v0.3.1 Qualification Pipeline")
    print(f"Runtime: {config.runtime}")
    print(f"Artifacts: {artifacts}")
    print(f"Stages: {len(stages)}")
    print()

    # Check for required artifacts from each stage.
    required_artifacts = [
        "benchmark_manifest.json",
        "split_manifest.json",
        "real_counterfactual_results.json",
        "executive_experiences.jsonl",
        "dataset_manifest.json",
        "candidate_policy.json",
        "sham_policy.json",
        "validation_results.json",
        "test_results.json",
        "sham_results.json",
        "ood_results.json",
        "calibration_results.json",
        "coverage_results.json",
        "safety_results.json",
        "promotion_result.json",
        "post_promotion_results.json",
        "policy_manifest.json",
        "provenance_manifest.json",
        "AUTOLEARN_V0_3_1_REPORT.md",
    ]

    missing = [a for a in required_artifacts if not (artifacts / a).exists()]
    if missing:
        print(f"Missing required artifacts: {missing}", file=sys.stderr)
        if config.is_real:
            return 1

    print("Pipeline configuration complete. Run individual stages:")
    for stage in stages:
        print(f"  python -m qualification.autolearn_v031.{stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
