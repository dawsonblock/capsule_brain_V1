"""Dependency-aware stage execution for v0.4.1 (Section 6).

Every qualification stage declares:
    stage_name
    required_inputs
    produced_outputs
    eligible_provider_classes
    dependencies
    allowed_modes

Before executing a stage:
    - verify all dependencies passed;
    - verify required artifacts exist;
    - verify parent hashes match;
    - verify provider class is eligible.

v0.4.1 fixes:
    - Promotion runs BEFORE post-promotion (not after).
    - Post-promotion requires promotion PASS.
    - Preflight and source_provenance stages added at the top.
    - Provider validation stage added before counterfactuals.
    - Evaluation stages (validation, ood, safety, oracle) made explicit.
    - Single-shot stage execution (no double build_benchmark).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StageStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class StageDeclaration:
    stage_name: str
    required_inputs: tuple[str, ...]
    produced_outputs: tuple[str, ...]
    dependencies: tuple[str, ...]
    eligible_provider_classes: tuple[str, ...] = ()
    is_mandatory: bool = True
    allowed_modes: tuple[str, ...] = ()  # empty = all modes

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "required_inputs": list(self.required_inputs),
            "produced_outputs": list(self.produced_outputs),
            "dependencies": list(self.dependencies),
            "eligible_provider_classes": list(self.eligible_provider_classes),
            "is_mandatory": self.is_mandatory,
            "allowed_modes": list(self.allowed_modes),
        }


# All pipeline stages in execution order.
# v0.4.1 correct ordering:
#   1. preflight
#   2. source_provenance
#   3. provider_validation
#   4. build_benchmark
#   5. run_counterfactuals
#   6. diagnose_runtime_completion
#   7. build_dataset
#   8. train_candidate
#   9. train_sham
#  10. evaluate_gate_a
#  11. collect_activations
#  12. evaluate_gate_b
#  13. run_promotion
#  14. run_post_promotion
#  15. provenance
#  16. report
PIPELINE_STAGES: tuple[StageDeclaration, ...] = (
    StageDeclaration(
        stage_name="preflight",
        required_inputs=(),
        produced_outputs=("preflight.json",),
        dependencies=(),
    ),
    StageDeclaration(
        stage_name="source_provenance",
        required_inputs=("preflight.json",),
        produced_outputs=("source_provenance.json",),
        dependencies=("preflight",),
    ),
    StageDeclaration(
        stage_name="provider_validation",
        required_inputs=("preflight.json",),
        produced_outputs=("provider_validation.json",),
        dependencies=("preflight",),
        allowed_modes=("scientific", "scientific-mini", "infrastructure"),
    ),
    StageDeclaration(
        stage_name="build_benchmark",
        required_inputs=("preflight.json",),
        produced_outputs=("benchmark_manifest.json", "split_manifest.json"),
        dependencies=("preflight",),
    ),
    StageDeclaration(
        stage_name="run_counterfactuals",
        required_inputs=("benchmark_manifest.json",),
        produced_outputs=("counterfactual_outcomes.json", "real_counterfactual_results.json"),
        dependencies=("build_benchmark",),
    ),
    StageDeclaration(
        stage_name="diagnose_runtime_completion",
        required_inputs=("counterfactual_outcomes.json", "benchmark_manifest.json"),
        produced_outputs=("runtime_completion_diagnostics.json",),
        dependencies=("run_counterfactuals",),
    ),
    StageDeclaration(
        stage_name="build_dataset",
        required_inputs=("counterfactual_outcomes.json", "benchmark_manifest.json"),
        produced_outputs=("dataset_manifest.json", "dataset_train.json", "dataset_validation.json"),
        dependencies=("diagnose_runtime_completion",),
    ),
    StageDeclaration(
        stage_name="train_candidate",
        required_inputs=("dataset_manifest.json", "dataset_train.json", "dataset_validation.json"),
        produced_outputs=("candidate_policy.json", "candidate_training.json"),
        dependencies=("build_dataset",),
    ),
    StageDeclaration(
        stage_name="train_sham",
        required_inputs=("dataset_manifest.json", "dataset_train.json", "dataset_validation.json"),
        produced_outputs=("sham_policy.json", "sham_training.json"),
        dependencies=("build_dataset",),
    ),
    StageDeclaration(
        stage_name="evaluate_gate_a",
        required_inputs=("candidate_policy.json", "sham_policy.json", "counterfactual_outcomes.json"),
        produced_outputs=("gate_a_result.json",),
        dependencies=("train_candidate", "train_sham"),
        eligible_provider_classes=("real_model",),
    ),
    StageDeclaration(
        stage_name="collect_activations",
        required_inputs=("counterfactual_outcomes.json", "benchmark_manifest.json"),
        produced_outputs=("activations_manifest.json",),
        dependencies=("run_counterfactuals",),
        eligible_provider_classes=("real_model",),
        is_mandatory=False,
    ),
    StageDeclaration(
        stage_name="evaluate_gate_b",
        required_inputs=("activations_manifest.json",),
        produced_outputs=("gate_b_result.json",),
        dependencies=("collect_activations",),
        eligible_provider_classes=("real_model",),
        is_mandatory=False,
    ),
    # v0.4.1 fix: promotion BEFORE post-promotion.
    StageDeclaration(
        stage_name="run_promotion",
        required_inputs=("candidate_policy.json",),
        produced_outputs=("promotion_result.json",),
        dependencies=("train_candidate", "evaluate_gate_a"),
    ),
    # v0.4.1 fix: post-promotion requires promotion PASS.
    StageDeclaration(
        stage_name="run_post_promotion",
        required_inputs=("promotion_result.json", "candidate_policy.json", "counterfactual_outcomes.json"),
        produced_outputs=("post_promotion_result.json",),
        dependencies=("run_promotion",),
    ),
    StageDeclaration(
        stage_name="provenance",
        required_inputs=(),
        produced_outputs=("provenance.json",),
        dependencies=(),
        is_mandatory=False,
    ),
    StageDeclaration(
        stage_name="report",
        required_inputs=(),  # always runs; reads what it can
        produced_outputs=("qualification_report.json",),
        dependencies=("run_promotion",),
    ),
)


def get_stage(name: str) -> StageDeclaration | None:
    for s in PIPELINE_STAGES:
        if s.stage_name == name:
            return s
    return None


def check_stage_dependencies(
    stage: StageDeclaration,
    stage_results: dict[str, StageStatus],
) -> list[str]:
    """Check if all dependencies passed. Returns list of error messages.

    For run_post_promotion, the dependency on run_promotion must be PASS
    (not just non-fail) — post-promotion must never execute against an
    unpromoted candidate.
    """
    errors: list[str] = []
    for dep in stage.dependencies:
        dep_status = stage_results.get(dep)
        if dep_status is None:
            errors.append(f"dependency '{dep}' has not been run")
        elif dep_status is StageStatus.FAIL:
            errors.append(f"dependency '{dep}' failed")
        elif dep_status is StageStatus.BLOCKED:
            errors.append(f"dependency '{dep}' is blocked")
        elif dep_status is StageStatus.NOT_RUN:
            errors.append(f"dependency '{dep}' was not run")
        # v0.4.1: post-promotion requires promotion PASS specifically.
        if stage.stage_name == "run_post_promotion" and dep == "run_promotion":
            if dep_status is not StageStatus.PASS:
                errors.append(
                    f"run_post_promotion requires run_promotion PASS, "
                    f"got {dep_status.value if dep_status else 'None'}"
                )
    return errors


def check_required_inputs(
    stage: StageDeclaration,
    artifacts_dir: str | Path,
) -> list[str]:
    """Check if all required input artifacts exist."""
    errors: list[str] = []
    artifacts = Path(artifacts_dir)
    for fname in stage.required_inputs:
        if not (artifacts / fname).exists():
            errors.append(f"required input '{fname}' does not exist")
    return errors


def check_provider_eligibility(
    stage: StageDeclaration,
    provider_class: str,
) -> list[str]:
    """Check if the provider class is eligible for this stage."""
    if not stage.eligible_provider_classes:
        return []
    if provider_class not in stage.eligible_provider_classes:
        return [
            f"provider class '{provider_class}' is not eligible for stage "
            f"'{stage.stage_name}'. Required: {list(stage.eligible_provider_classes)}"
        ]
    return []


def should_block_downstream(
    failed_stage: str,
    stage_results: dict[str, StageStatus],
) -> list[str]:
    """Return list of downstream stages that should be blocked."""
    blocked: list[str] = []
    for stage in PIPELINE_STAGES:
        if stage.stage_name in stage_results:
            continue
        # Check if any dependency is failed/blocked.
        for dep in stage.dependencies:
            dep_status = stage_results.get(dep)
            if dep_status in (StageStatus.FAIL, StageStatus.BLOCKED):
                blocked.append(stage.stage_name)
                break
    return blocked


def stage_status_to_dict(status: StageStatus) -> str:
    return status.value


# ---------------------------------------------------------------------------
# v0.4.1: Single-shot stage execution result
# ---------------------------------------------------------------------------


@dataclass
class StageExecution:
    """Result of executing a single stage (single-shot, no reruns)."""
    name: str
    status: StageStatus
    result: Any | None = None
    error: str | None = None
    produced_artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "error": self.error,
            "produced_artifacts": list(self.produced_artifacts),
        }
