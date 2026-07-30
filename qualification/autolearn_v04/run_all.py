"""Run the full v0.4.0 qualification pipeline end-to-end (Section 17).

Stages:
  1. build_benchmark
  2. build_runtime (implicit in run_counterfactuals)
  3. run_counterfactuals
  4. build_dataset
  5. train_candidate
  6. train_sham
  7. evaluate_gate_a
  8. collect_activations (BLOCKED for infrastructure provider)
  9. evaluate_gate_b (BLOCKED for infrastructure provider)
 10. run_promotion
 11. run_post_promotion
 12. provenance
 13. report

Smoke mode runs stages 1-6, 10-13 with a small subset and reports
PIPELINE_INTEGRITY only. SCIENTIFIC_QUALIFICATION is NOT_RUN and
PROMOTION_DECISION is BLOCKED.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .build_benchmark import build_benchmark
from .build_dataset import build_dataset
from .config import QualificationConfig
from .evaluate_gate_a import evaluate_gate_a
from .evaluate_gate_b import evaluate_gate_b
from .provenance import build_pipeline_provenance
from .report import generate_report
from .run_counterfactuals import run_counterfactuals
from .run_post_promotion import run_post_promotion
from .run_promotion import run_promotion
from .schemas import write_json
from .train_candidate import train_candidate
from .train_sham import train_sham


def run_all(config: QualificationConfig) -> dict[str, Any]:
    artifacts = Path(config.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    stages: dict[str, str] = {}
    errors: dict[str, str] = {}

    def _stage(name: str, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            stages[name] = "ok"
        except Exception as exc:
            stages[name] = f"error: {exc}"
            errors[name] = str(exc)

    # 1. build_benchmark
    _stage("build_benchmark", build_benchmark, config)
    # Write benchmark artifacts.
    tasks, bm, sm = build_benchmark(config)
    write_json(artifacts / "benchmark_manifest.json", bm)
    write_json(artifacts / "split_manifest.json", sm)

    # 2. run_counterfactuals
    try:
        outcomes, rows = asyncio.run(run_counterfactuals(config))
        stages["run_counterfactuals"] = "ok"
    except Exception as exc:
        stages["run_counterfactuals"] = f"error: {exc}"
        errors["run_counterfactuals"] = str(exc)
        outcomes, rows = [], []

    # 3. build_dataset
    _stage("build_dataset", build_dataset, config)

    # 4. train_candidate
    _stage("train_candidate", train_candidate, config)

    # 5. train_sham
    _stage("train_sham", train_sham, config)

    # 6. evaluate_gate_a (BLOCKED for infrastructure provider)
    try:
        gate_a = evaluate_gate_a(config)
        stages["evaluate_gate_a"] = gate_a.get("status", "error")
    except Exception as exc:
        stages["evaluate_gate_a"] = f"error: {exc}"
        errors["evaluate_gate_a"] = str(exc)

    # 7. collect_activations + evaluate_gate_b (BLOCKED for infrastructure provider)
    if config.is_real_model_provider:
        try:
            from .collect_activations import collect_activations
            asyncio.run(collect_activations(config))
            stages["collect_activations"] = "ok"
        except Exception as exc:
            stages["collect_activations"] = f"error: {exc}"
            errors["collect_activations"] = str(exc)
    else:
        stages["collect_activations"] = "blocked: infrastructure provider"

    try:
        gate_b = evaluate_gate_b(config)
        stages["evaluate_gate_b"] = gate_b.get("status", "error")
    except Exception as exc:
        stages["evaluate_gate_b"] = f"error: {exc}"
        errors["evaluate_gate_b"] = str(exc)

    # 8. run_post_promotion
    _stage("run_post_promotion", run_post_promotion, config)

    # 9. run_promotion
    _stage("run_promotion", run_promotion, config)

    # 10. provenance
    _stage("provenance", build_pipeline_provenance, config)

    # 11. report
    _stage("report", generate_report, config)

    # Determine pipeline integrity.
    has_errors = any(v.startswith("error") for v in stages.values())
    pipeline_ok = not has_errors

    # Load final report.
    from .schemas import read_json
    report = read_json(artifacts / "qualification_report.json") if (artifacts / "qualification_report.json").exists() else {}

    summary = {
        "schema_version": "pipeline-summary/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "mode": config.mode,
        "runtime": config.runtime,
        "provider": config.provider,
        "pipeline": "FAIL" if has_errors else "PASS",
        "stages": stages,
        "errors": errors,
        "report_status": report.get("status", "NOT RUN"),
        "PIPELINE_INTEGRITY": "PASS" if pipeline_ok else "FAIL",
        "SCIENTIFIC_QUALIFICATION": report.get("SCIENTIFIC_QUALIFICATION", "NOT_RUN"),
        "PROMOTION_DECISION": report.get("PROMOTION_DECISION", "BLOCKED"),
    }
    write_json(artifacts / "pipeline_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full v0.4.0 qualification pipeline.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "qualification"], default="smoke")
    parser.add_argument("--runtime", choices=["real", "simulated"], default="real")
    parser.add_argument("--provider", default="qual_grounded")
    parser.add_argument("--model", default="qual-grounded-v04")
    parser.add_argument("--task-seed", type=int, default=42)
    args = parser.parse_args()
    config = QualificationConfig(
        mode=args.mode, runtime=args.runtime, provider=args.provider,
        model=args.model, artifacts_dir=args.artifacts_dir, task_seed=args.task_seed,
    )
    if config.is_simulated:
        print(config.simulated_banner, file=sys.stderr)
    if config.is_infrastructure_provider:
        print(config.infrastructure_provider_banner, file=sys.stderr)

    start = time.time()
    summary = run_all(config)
    elapsed = time.time() - start

    print(f"\npipeline: {summary['pipeline']}")
    for stage, status in summary["stages"].items():
        print(f"  {stage}: {status}")
    print(f"  PIPELINE_INTEGRITY: {summary['PIPELINE_INTEGRITY']}")
    print(f"  SCIENTIFIC_QUALIFICATION: {summary['SCIENTIFIC_QUALIFICATION']}")
    print(f"  PROMOTION_DECISION: {summary['PROMOTION_DECISION']}")
    print(f"  elapsed: {elapsed:.1f}s")
    return 0 if summary["pipeline"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
