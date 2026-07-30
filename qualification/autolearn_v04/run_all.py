"""Run the full v0.4.1 qualification pipeline with dependency-aware stages (Section 6/17).

v0.4.1 fixes:
    - Promotion runs BEFORE post-promotion.
    - Post-promotion requires promotion PASS.
    - Stages execute single-shot (no double build_benchmark).
    - Preflight and source_provenance stages added.
    - Provider validation stage added.

Stages execute in dependency order. If a mandatory upstream stage fails,
downstream stages are BLOCKED (not raised as exceptions). Provenance and
report are always generated where possible.

Modes:
    smoke           — small wiring check, no scientific claims
    infrastructure  — grounded provider, integration only, Gate A/B blocked
    scientific      — real frozen transformer, full Gate A/B evaluation
    scientific-mini — small real-model run for pipeline validation
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
    StageExecution,
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
    # v0.4.1: single-shot stage results cache (no reruns).
    stage_executions: dict[str, StageExecution] = {}

    def _record(name: str, status: StageStatus, detail: str = "") -> None:
        stage_results[name] = status
        stage_details[name] = detail
        if status is StageStatus.FAIL:
            stage_errors[name] = detail

    def _run_stage_single_shot(name: str, fn, *args, **kwargs) -> StageExecution:
        """Run a stage exactly once. Returns StageExecution with result.

        v0.4.1: The stage function's return value is captured in the
        StageExecution.result field. The caller must NOT call the stage
        function again to recover its return value.
        """
        stage = next((s for s in PIPELINE_STAGES if s.stage_name == name), None)
        if stage:
            dep_errors = check_stage_dependencies(stage, stage_results)
            if dep_errors:
                _record(name, StageStatus.BLOCKED, "; ".join(dep_errors))
                return StageExecution(name=name, status=StageStatus.BLOCKED,
                                      error="; ".join(dep_errors))
            input_errors = check_required_inputs(stage, artifacts)
            if input_errors:
                _record(name, StageStatus.BLOCKED, "; ".join(input_errors))
                return StageExecution(name=name, status=StageStatus.BLOCKED,
                                      error="; ".join(input_errors))
        try:
            result = fn(*args, **kwargs)
            _record(name, StageStatus.PASS, "ok")
            stage_decl = next((s for s in PIPELINE_STAGES if s.stage_name == name), None)
            produced = stage_decl.produced_outputs if stage_decl else ()
            exec_result = StageExecution(
                name=name, status=StageStatus.PASS, result=result,
                produced_artifacts=produced,
            )
            stage_executions[name] = exec_result
            return exec_result
        except Exception as exc:
            _record(name, StageStatus.FAIL, str(exc))
            exec_result = StageExecution(name=name, status=StageStatus.FAIL, error=str(exc))
            stage_executions[name] = exec_result
            return exec_result

    # 1. preflight
    from .preflight import run_preflight as _preflight
    _run_stage_single_shot("preflight", _preflight, config)

    # 2. source_provenance
    from .provenance import compute_and_write_source_provenance as _src_prov
    _run_stage_single_shot("source_provenance", _src_prov, config)

    # 3. provider_validation (scientific/infrastructure only)
    if config.mode in ("scientific", "scientific-mini", "infrastructure"):
        from .provider_validation import validate_provider as _validate
        _run_stage_single_shot("provider_validation", _validate, config)
    else:
        _record("provider_validation", StageStatus.NOT_RUN, "smoke mode skips provider validation")

    # 4. build_benchmark — single-shot, capture result
    from .build_benchmark import build_benchmark as _build_bm
    bm_exec = _run_stage_single_shot("build_benchmark", _build_bm, config)
    if bm_exec.status is StageStatus.PASS and bm_exec.result is not None:
        tasks, bm, sm = bm_exec.result
        write_json(artifacts / "benchmark_manifest.json", bm)
        write_json(artifacts / "split_manifest.json", sm)

    # 5. run_counterfactuals
    from .run_counterfactuals import run_counterfactuals as _run_cf
    _run_stage_single_shot("run_counterfactuals", lambda: asyncio.run(_run_cf(config)))

    # 6. diagnose_runtime_completion
    from .diagnose_runtime_completion import diagnose_runtime_completion as _diag
    _run_stage_single_shot("diagnose_runtime_completion", _diag, config)

    # 7. build_dataset
    from .build_dataset import build_dataset as _build_ds
    _run_stage_single_shot("build_dataset", _build_ds, config)

    # 8. train_candidate
    from .train_candidate import train_candidate as _train_cand
    _run_stage_single_shot("train_candidate", _train_cand, config)

    # 9. train_sham
    from .train_sham import train_sham as _train_sham
    _run_stage_single_shot("train_sham", _train_sham, config)

    # 10. evaluate_gate_a — BLOCKED for infrastructure provider
    from .evaluate_gate_a import evaluate_gate_a as _eval_a
    if config.is_infrastructure_provider:
        _record("evaluate_gate_a", StageStatus.BLOCKED,
                "Gate A requires real_model provider; infrastructure provider cannot make causal claims")
    else:
        _run_stage_single_shot("evaluate_gate_a", _eval_a, config)

    # 11. collect_activations — BLOCKED for infrastructure provider
    if config.is_infrastructure_provider:
        _record("collect_activations", StageStatus.BLOCKED,
                "Activation collection requires real_model provider with hidden states")
    else:
        from .collect_activations import collect_activations as _collect
        _run_stage_single_shot("collect_activations", lambda: asyncio.run(_collect(config)))

    # 12. evaluate_gate_b — BLOCKED for infrastructure provider
    from .evaluate_gate_b import evaluate_gate_b as _eval_b
    if config.is_infrastructure_provider:
        _record("evaluate_gate_b", StageStatus.BLOCKED,
                "Gate B requires real_model provider with hidden states")
    else:
        _run_stage_single_shot("evaluate_gate_b", _eval_b, config)

    # 13. run_promotion — v0.4.1: BEFORE post-promotion
    from .run_promotion import run_promotion as _promo
    _run_stage_single_shot("run_promotion", _promo, config)

    # 14. run_post_promotion — v0.4.1: AFTER promotion, requires promotion PASS
    from .run_post_promotion import run_post_promotion as _post_promo
    promo_status = stage_results.get("run_promotion")
    if promo_status is not StageStatus.PASS:
        _record("run_post_promotion", StageStatus.BLOCKED,
                f"run_promotion did not PASS (status={promo_status.value if promo_status else 'None'}); "
                "post-promotion cannot execute against unpromoted candidate")
    else:
        _run_stage_single_shot("run_post_promotion", _post_promo, config)

    # 15. provenance (always run if possible)
    from .provenance import build_pipeline_provenance as _prov
    try:
        _prov(config)
        _record("provenance", StageStatus.PASS, "ok")
    except Exception as exc:
        _record("provenance", StageStatus.FAIL, str(exc))

    # 16. report (always run)
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
        "schema_version": "pipeline-summary/3",
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
    parser = argparse.ArgumentParser(description="Run the full v0.4.1 qualification pipeline.")
    parser.add_argument("--artifacts-dir", default="artifacts_v04")
    parser.add_argument("--mode", choices=["smoke", "infrastructure", "scientific", "scientific-mini"], default="smoke")
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
