"""Qualification configuration for AutoLearn v0.3.1.

Defines the configuration for the qualification pipeline, including:
- runtime type (real or simulated)
- sample sizes for each split
- utility weights
- promotion gate thresholds
- artifact paths
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    """Configuration for the v0.3.1 qualification pipeline."""

    # Runtime: "real" for official qualification, "simulated" for dev only.
    runtime: str = "real"
    # Sample sizes (Section 18).
    n_experience: int = 160
    n_validation: int = 60
    n_test: int = 100
    n_ood: int = 60
    n_safety: int = 40
    # Crossover fraction (Section 29): at least 25% of tasks should be
    # crossover cases where common routing proxies are misleading.
    crossover_fraction: float = 0.25
    # Utility weights (Section 14).
    w_success: float = 10.0
    w_latency: float = 0.50
    w_tokens: float = 0.25
    w_tool_failure: float = 1.0
    w_unnecessary_tool: float = 2.0
    w_unnecessary_workflow: float = 2.0
    w_runtime_error: float = 5.0
    w_verification_failure: float = 10.0
    w_safety_violation: float = 50.0
    # Promotion gate thresholds (Section 22).
    epsilon_utility: float = 0.1
    effective_coverage_min: float = 0.60
    tool_precision_min: float = 0.6
    tool_recall_min: float = 0.6
    workflow_precision_min: float = 0.6
    workflow_recall_min: float = 0.6
    ood_utility_floor: float = 0.0
    calibration_max: float = 0.3
    # Bootstrap settings (Section 17).
    bootstrap_replicates: int = 10000
    bootstrap_seed: int = 42
    # Artifact directory (relative to qualification root).
    artifacts_dir: str = "artifacts"
    # Random seed for task generation.
    task_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "n_experience": self.n_experience,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "n_ood": self.n_ood,
            "n_safety": self.n_safety,
            "crossover_fraction": self.crossover_fraction,
            "w_success": self.w_success,
            "w_latency": self.w_latency,
            "w_tokens": self.w_tokens,
            "w_tool_failure": self.w_tool_failure,
            "w_unnecessary_tool": self.w_unnecessary_tool,
            "w_unnecessary_workflow": self.w_unnecessary_workflow,
            "w_runtime_error": self.w_runtime_error,
            "w_verification_failure": self.w_verification_failure,
            "w_safety_violation": self.w_safety_violation,
            "epsilon_utility": self.epsilon_utility,
            "effective_coverage_min": self.effective_coverage_min,
            "tool_precision_min": self.tool_precision_min,
            "tool_recall_min": self.tool_recall_min,
            "workflow_precision_min": self.workflow_precision_min,
            "workflow_recall_min": self.workflow_recall_min,
            "ood_utility_floor": self.ood_utility_floor,
            "calibration_max": self.calibration_max,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "artifacts_dir": self.artifacts_dir,
            "task_seed": self.task_seed,
        }

    @property
    def is_real(self) -> bool:
        return self.runtime == "real"

    @property
    def is_simulated(self) -> bool:
        return self.runtime == "simulated"
