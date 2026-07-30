"""Unified orchestration engine for v0.4.1 qualification pipeline.

Replaces the separate control flows in run_all.py and
modal_scientific_executor.py with a single orchestration engine that
operates against provider/executor interfaces.

The scientific protocol, stage order, dependency handling, split
auditing, promotion logic and reporting must be identical across
LocalExecutionBackend and ModalExecutionBackend.

Modal code may manage GPU/container lifecycle, but it must not define
a different scientific pipeline.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .schemas import read_json, write_json
from .stage_dependencies import (
    PIPELINE_STAGES,
    StageExecution,
    StageStatus,
    check_required_inputs,
    check_stage_dependencies,
)


@runtime_checkable
class ExecutionBackend(Protocol):
    """Protocol for execution backends (local, Modal, etc.)."""
    backend_name: str

    async def run_counterfactuals(self, config: QualificationConfig) -> dict[str, Any]:
        """Run counterfactual executions and return results dict."""
        ...

    async def collect_activations(self, config: QualificationConfig) -> dict[str, Any]:
        """Collect hidden-state activations for Gate B."""
        ...


class LocalExecutionBackend:
    """Local execution backend using the real Capsule Brain service stack."""

    backend_name: str = "local"

    async def run_counterfactuals(self, config: QualificationConfig) -> dict[str, Any]:
        from .run_counterfactuals import run_counterfactuals
        return await run_counterfactuals(config)

    async def collect_activations(self, config: QualificationConfig) -> dict[str, Any]:
        from .collect_activations import collect_activations
        return await collect_activations(config)


class ModalExecutionBackend:
    """Modal GPU execution backend for remote model inference.

    This backend delegates model inference to a deployed Modal app
    but uses the same scientific protocol, stage order, and dependency
    handling as the local backend.
    """

    backend_name: str = "modal"

    def __init__(self, modal_client=None) -> None:
        self._client = modal_client

    async def run_counterfactuals(self, config: QualificationConfig) -> dict[str, Any]:
        from .modal_scientific_executor import run_scientific_counterfactuals
        return run_scientific_counterfactuals(config, client=self._client)

    async def collect_activations(self, config: QualificationConfig) -> dict[str, Any]:
        # Modal activation collection uses the same counterfactual path
        # with hidden-state collection enabled.
        from .modal_scientific_executor import run_scientific_counterfactuals
        return run_scientific_counterfactuals(config, client=self._client, collect_hidden_states=True)


@dataclass
class OrchestratorResult:
    """Result of running the full orchestration pipeline."""
    pipeline_status: str
    stages: dict[str, str]
    stage_details: dict[str, str]
    stage_errors: dict[str, str]
    stage_executions: dict[str, StageExecution]
    summary: dict[str, Any]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_status": self.pipeline_status,
            "stages": self.stages,
            "stage_details": self.stage_details,
            "stage_errors": self.stage_errors,
            "elapsed_seconds": self.elapsed_seconds,
            **self.summary,
        }


class PipelineOrchestrator:
    """Unified pipeline orchestrator for v0.4.1.

    Executes stages in dependency order using a pluggable execution backend.
    The scientific protocol is identical regardless of whether the backend
    is local or Modal.
    """

    def __init__(self, config: QualificationConfig, backend: ExecutionBackend | None = None) -> None:
        self.config = config
        self.backend = backend or LocalExecutionBackend()
        self.artifacts = Path(config.artifacts_dir)
        self.stage_results: dict[str, StageStatus] = {}
        self.stage_details: dict[str, str] = {}
        self.stage_errors: dict[str, str] = {}
        self.stage_executions: dict[str, StageExecution] = {}

    def _record(self, name: str, status: StageStatus, detail: str = "") -> None:
        self.stage_results[name] = status
        self.stage_details[name] = detail
        if status is StageStatus.FAIL:
            self.stage_errors[name] = detail

    def _run_stage(self, name: str, fn, *args, **kwargs) -> StageExecution:
        """Run a stage exactly once with dependency checks."""
        stage = next((s for s in PIPELINE_STAGES if s.stage_name == name), None)
        if stage:
            dep_errors = check_stage_dependencies(stage, self.stage_results)
            if dep_errors:
                self._record(name, StageStatus.BLOCKED, "; ".join(dep_errors))
                return StageExecution(name=name, status=StageStatus.BLOCKED,
                                      error="; ".join(dep_errors))
            input_errors = check_required_inputs(stage, self.artifacts)
            if input_errors:
                self._record(name, StageStatus.BLOCKED, "; ".join(input_errors))
                return StageExecution(name=name, status=StageStatus.BLOCKED,
                                      error="; ".join(input_errors))
        try:
            result = fn(*args, **kwargs)
            self._record(name, StageStatus.PASS, "ok")
            stage_decl = next((s for s in PIPELINE_STAGES if s.stage_name == name), None)
            produced = stage_decl.produced_outputs if stage_decl else ()
            exec_result = StageExecution(
                name=name, status=StageStatus.PASS, result=result,
                produced_artifacts=produced,
            )
            self.stage_executions[name] = exec_result
            return exec_result
        except Exception as exc:
            self._record(name, StageStatus.FAIL, str(exc))
            exec_result = StageExecution(name=name, status=StageStatus.FAIL, error=str(exc))
            self.stage_executions[name] = exec_result
            return exec_result

    def run(self) -> OrchestratorResult:
        """Run the full pipeline."""
        start = time.time()
        self.artifacts.mkdir(parents=True, exist_ok=True)

        # 1. preflight
        from .preflight import run_preflight
        self._run_stage("preflight", run_preflight, self.config)

        # 2. source_provenance
        from .provenance import compute_and_write_source_provenance
        self._run_stage("source_provenance", compute_and_write_source_provenance, self.config)

        # 3. provider_validation
        if self.config.mode in ("scientific", "scientific-mini", "infrastructure"):
            from .provider_validation import validate_provider
            self._run_stage("provider_validation", validate_provider, self.config)
        else:
            self._record("provider_validation", StageStatus.NOT_RUN,
                         "smoke mode skips provider validation")

        # 4. build_benchmark
        from .build_benchmark import build_benchmark
        bm_exec = self._run_stage("build_benchmark", build_benchmark, self.config)
        if bm_exec.status is StageStatus.PASS and bm_exec.result is not None:
            tasks, bm, sm = bm_exec.result
            write_json(self.artifacts / "benchmark_manifest.json", bm)
            write_json(self.artifacts / "split_manifest.json", sm)

        # 5. run_counterfactuals — uses backend
        self._run_stage("run_counterfactuals",
                        lambda: asyncio.run(self.backend.run_counterfactuals(self.config)))

        # 6. diagnose_runtime_completion
        from .diagnose_runtime_completion import diagnose_runtime_completion
        self._run_stage("diagnose_runtime_completion", diagnose_runtime_completion, self.config)

        # 7. build_dataset
        from .build_dataset import build_dataset
        self._run_stage("build_dataset", build_dataset, self.config)

        # 8. train_candidate
        from .train_candidate import train_candidate
        self._run_stage("train_candidate", train_candidate, self.config)

        # 9. train_sham
        from .train_sham import train_sham
        self._run_stage("train_sham", train_sham, self.config)

        # 10. evaluate_gate_a
        from .evaluate_gate_a import evaluate_gate_a
        if self.config.is_infrastructure_provider:
            self._record("evaluate_gate_a", StageStatus.BLOCKED,
                         "Gate A requires real_model provider")
        else:
            self._run_stage("evaluate_gate_a", evaluate_gate_a, self.config)

        # 11. collect_activations — uses backend
        if self.config.is_infrastructure_provider:
            self._record("collect_activations", StageStatus.BLOCKED,
                         "Activation collection requires real_model provider")
        else:
            self._run_stage("collect_activations",
                            lambda: asyncio.run(self.backend.collect_activations(self.config)))

        # 12. evaluate_gate_b
        from .evaluate_gate_b import evaluate_gate_b
        if self.config.is_infrastructure_provider:
            self._record("evaluate_gate_b", StageStatus.BLOCKED,
                         "Gate B requires real_model provider")
        else:
            self._run_stage("evaluate_gate_b", evaluate_gate_b, self.config)

        # 13. run_promotion (BEFORE post-promotion)
        from .run_promotion import run_promotion
        self._run_stage("run_promotion", run_promotion, self.config)

        # 14. run_post_promotion (AFTER promotion, requires PASS)
        from .run_post_promotion import run_post_promotion
        promo_status = self.stage_results.get("run_promotion")
        if promo_status is not StageStatus.PASS:
            self._record("run_post_promotion", StageStatus.BLOCKED,
                         f"run_promotion did not PASS (status={promo_status.value if promo_status else 'None'})")
        else:
            self._run_stage("run_post_promotion", run_post_promotion, self.config)

        # 15. provenance
        from .provenance import build_pipeline_provenance
        try:
            build_pipeline_provenance(self.config)
            self._record("provenance", StageStatus.PASS, "ok")
        except Exception as exc:
            self._record("provenance", StageStatus.FAIL, str(exc))

        # 16. report
        from .report import generate_report
        try:
            generate_report(self.config)
            self._record("report", StageStatus.PASS, "ok")
        except Exception as exc:
            self._record("report", StageStatus.FAIL, str(exc))

        has_failures = any(s is StageStatus.FAIL for s in self.stage_results.values())
        pipeline_status = "FAIL" if has_failures else "PASS"

        report = read_json(self.artifacts / "qualification_report.json") if (
            self.artifacts / "qualification_report.json").exists() else {}

        summary = {
            "schema_version": "pipeline-summary/3",
            "protocol_version": PROTOCOL_VERSION,
            "package_version": PACKAGE_VERSION,
            "autolearn_version": AUTOLEARN_VERSION,
            "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
            "mode": self.config.mode,
            "runtime": self.config.runtime,
            "provider": self.config.provider,
            "backend": self.backend.backend_name,
            "pipeline": pipeline_status,
            "stages": {k: v.value for k, v in self.stage_results.items()},
            "stage_details": self.stage_details,
            "errors": self.stage_errors,
            "report_status": report.get("status", "NOT RUN"),
            "PIPELINE_INTEGRITY": pipeline_status,
            "SCIENTIFIC_QUALIFICATION": report.get("SCIENTIFIC_QUALIFICATION", "NOT_RUN"),
            "PROMOTION_DECISION": report.get("PROMOTION_DECISION", "BLOCKED"),
        }
        write_json(self.artifacts / "pipeline_summary.json", summary)

        elapsed = time.time() - start
        return OrchestratorResult(
            pipeline_status=pipeline_status,
            stages={k: v.value for k, v in self.stage_results.items()},
            stage_details=self.stage_details,
            stage_errors=self.stage_errors,
            stage_executions=self.stage_executions,
            summary=summary,
            elapsed_seconds=elapsed,
        )
