"""Run the full v0.4.0 qualification pipeline with dependency-aware stages (Section 6/17).

Stages execute in dependency order. If a mandatory upstream stage fails,
downstream stages are BLOCKED (not raised as exceptions). Provenance and
report are always generated where possible.

Modes:
    smoke         — small wiring check, no scientific claims
    infrastructure — grounded provider, integration only, Gate A/B blocked
    scientific    — real frozen transformer, full Gate A/B evaluation
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .schemas import read_json, write_json
from .stage_dependencies import (
    PIPELINE_STAGES,
    StageStatus,
    check_required_inputs,
    check_stage_dependencies,
    should_block_downstream,
)


def run_all(config: QualificationConfig) -> dict[str, Any]:
    artifacts = Path(config.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    stage_results: dict[str, StageStatus] = {}
    stage_details: dict[str, str] = {}
    stage_errors: dict[str, str] = {}

    def _record(name: str, status: StageStatus, detail: str = "") -> None:
        stage_results[name] = status
        stage_details[name] = detail
        if status is StageStatus.FAIL:
            stage_errors[name] = detail

    def _run_stage(name: str, fn, *args, **kwargs) -> bool:
        """Run a stage with dependency checks. Returns True if executed."""
        stage = next((s for s in PIPELINE_STAGES if s.stage_name == name), None)
        if stage:
            dep_errors = check_stage_dependencies(stage, stage_results)
            if dep_errors:
                _record(name, StageStatus.BLOCKED, "; ".join(dep_errors))
                return False
            input_errors = check_required_inputs(stage, artifacts)
            if input_errors:
                _record(name, StageStatus.BLOCKED, "; ".join(input_errors))
                return False
        try:
            result = fn(*args, **kwargs)
            _record(name, StageStatus.PASS, "ok")
            return True
        except Exception as exc:
            _record(name, StageStatus.FAIL, str(exc))
            return False

    # 1. build_benchmark
    from .build_benchmark import build_benchmark as _build_bm
    if _run_stage("build_benchmark", _build_bm, config):
        tasks, bm, sm = _build_bm(config)
        write_json(artifacts / "benchmark_manifest.json", bm)
        write_json(artifacts / "split_manifest.json", sm)

    # 2. run_counterfactuals
    from .run_counterfactuals import run_counterfactuals as _run_cf
    if _run_stage("run_counterfactuals", lambda: asyncio.run(_run_cf(config))):
        pass

    # 3. diagnose_runtime_completion
    from .diagnose_runtime_completion import diagnose_runtime_completion as _diag
    _run_stage("diagnose_runtime_completion", _diag, config)

    # 4. build_dataset
    from .build_dataset import build_dataset as _build_ds
    _run_stage("build_dataset", _build_ds, config)

    # 5. train_candidate
    from .train_candidate import train_candidate as _train_cand
    _run_stage("train_candidate", _train_cand, config)

    # 6. train_sham
    from .train_sham import train_sham as _train_sham
    _run_stage("train_sham", _train_sham, config)

    # 7. evaluate_gate_a — BLOCKED for infrastructure provider
    from .evaluate_gate_a import evaluate_gate_a as _eval_a
    if config.is_infrastructure_provider:
        _record("evaluate_gate_a", StageStatus.BLOCKED,
                "Gate A requires real_model provider; infrastructure provider cannot make causal claims")
    else:
        _run_stage("evaluate_gate_a", _eval_a, config)

    # 8. collect_activations — BLOCKED for infrastructure provider
    if config.is_infrastructure_provider:
        _record("collect_activations", StageStatus.BLOCKED,
                "Activation collection requires real_model provider with hidden states")
    else:
        from .collect_activations import collect_activations as _collect
        _run_stage("collect_activations", lambda: asyncio.run(_collect(config)))

    # 9. evaluate_gate_b — BLOCKED for infrastructure provider
    from .evaluate_gate_b import evaluate_gate_b as _eval_b
    if config.is_infrastructure_provider:
        _record("evaluate_gate_b", StageStatus.BLOCKED,
                "Gate B requires real_model provider with hidden states")
    else:
        _run_stage("evaluate_gate_b", _eval_b, config)

    # 10. run_post_promotion — BLOCKED if candidate training failed
    from .run_post_promotion import run_post_promotion as _post_promo
    if stage_results.get("train_candidate") is not StageStatus.PASS:
        _record("run_post_promotion", StageStatus.BLOCKED,
                "candidate training did not complete")
    else:
        _run_stage("run_post_promotion", _post_promo, config)

    # 11. run_promotion
    from .run_promotion import run_promotion as _promo
    _run_stage("run_promotion", _promo, config)

    # 12. provenance (always run if possible)
    from .provenance import build_pipeline_provenance as _prov
    try:
        _prov(config)
        _record("provenance", StageStatus.PASS, "ok")
    except Exception as exc:
        _record("provenance", StageStatus.FAIL, str(exc))

    # 13. report (always run)
    from .report import generate_report as _report
    try:
        _report(config)
        _record("report", StageStatus.PASS, "ok")
    except Exception as exc:
        _record("report", StageStatus.FAIL, str(exc))

    # Determine pipeline integrity.
    has_failures = any(s is StageStatus.FAIL for s in stage_results.values())
    pipeline_status = "FAIL" if has_failures else "PASS"

    report = read_json(artifacts / "qualification_report.json") if (artifacts / "qualification_report.json").exists() else {}

    summary = {
        "schema_version": "pipeline-summary/2",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "mode": config.mode,
        "runtime": config.runtime,
        "provider": config.provider,
        "pipeline": pipeline_status,
        "stages": {k: v.value for k, v in stage_results.items()},
        "stage_details": stage_details,
        "errors": stage_errors,
        "report_status": report.get("status", "NOT RUN"),
        "PIPELINE_INTEGRITY": pipeline_status,
        "SCIENTIFIC_QUALIFICATION": report.get("SCIENTIFIC_QUALIFICATION", "NOT_RUN"),
        "PROMOTION_DECISION": report.get("PROMOTION_DECISION", "BLOCKED"),
    }
    write_json(artifacts / "pipeline_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full v0.4.0 qualification pipeline.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "infrastructure", "scientific"], default="smoke")
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
