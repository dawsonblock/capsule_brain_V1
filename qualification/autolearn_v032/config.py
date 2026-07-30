"""Qualification configuration for AutoLearn v0.3.2 (Capsule Brain 2.15.3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import AUTOLEARN_QUALIFICATION_VERSION, PACKAGE_VERSION


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    """Configuration for the v0.3.2 qualification pipeline."""

    # Runtime: "real" for official qualification, "simulated" for dev only.
    runtime: str = "real"
    # When True, only execute a small smoke subset of tasks through real
    # services (proves the pipeline end-to-end). The full 420-task manifest
    # is still generated and split-integrity-validated.
    smoke: bool = True
    smoke_task_count: int = 12

    # Minimum task counts for the full benchmark manifest.
    n_experience: int = 160
    n_validation: int = 60
    n_test: int = 100
    n_ood: int = 60
    n_safety: int = 40

    # At least 25% of tasks must be crossover cases.
    crossover_fraction: float = 0.25

    # Utility weights (mirrors UtilityConfig defaults).
    w_success: float = 10.0
    w_latency: float = 0.5
    w_tokens: float = 0.25
    w_tool_failure: float = 1.0
    w_unnecessary_tool: float = 2.0
    w_unnecessary_workflow: float = 2.0
    w_runtime_error: float = 20.0
    w_safety: float = 50.0

    # Promotion gate thresholds.
    epsilon_utility: float = 0.1
    effective_coverage_min: float = 0.60
    tool_precision_min: float = 0.6
    tool_recall_min: float = 0.6
    workflow_precision_min: float = 0.6
    workflow_recall_min: float = 0.6
    ood_utility_floor: float = 0.0
    calibration_max: float = 0.3

    # Bootstrap settings.
    bootstrap_replicates: int = 10000
    bootstrap_seed: int = 42

    # Artifact directory (repository-relative).
    artifacts_dir: str = "artifacts"
    # Random seed for task generation.
    task_seed: int = 42
    # Isolation base directory for per-action environments.
    isolation_root: str = ".cf_isolation_v032"
    # Sandbox root for workflow acceptance execution.
    sandbox_root: str = ".qual_sandbox_v032"

    # Model id used for qualification (deterministic grounded provider).
    model_id: str = "qual-grounded-v032"

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "smoke": self.smoke,
            "smoke_task_count": self.smoke_task_count,
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
            "w_safety": self.w_safety,
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
            "isolation_root": self.isolation_root,
            "sandbox_root": self.sandbox_root,
            "model_id": self.model_id,
            "package_version": PACKAGE_VERSION,
            "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        }

    def to_capsule_config(self) -> dict[str, Any]:
        """Build a Capsule Brain application config for the qualification runtime.

        Uses a deterministic grounded LLM provider (registered separately) and
        host subprocess execution for workflow acceptance tests. Every service
        is real (MemoryService, ToolRegistry, WorkflowRunner, ExecutionService,
        ConversationService, AutoLearnService).
        """
        return {
            "llm_gateway": {
                "enable": True,
                "default_model": self.model_id,
                "models": {
                    self.model_id: {
                        "provider": "qual_grounded",
                        "model_name": self.model_id,
                        "capabilities": ["text", "tools"],
                    },
                },
            },
            "memory": {"db_path": ":memory:"},
            "conversation": {
                "enable": True,
                "use_tools": True,
                "db_path": ":memory:",
            },
            "autolearn": {"enable": True, "policy_root": ""},
            "execution": {
                "enable": True,
                "runner": "host",
                "allow": True,
                "unsafe_allow_host_execution": True,
                "timeout_s": 15.0,
                "max_output_chars": 20000,
                "allowed_commands": ["python"],
                "cwd_root": self.sandbox_root,
                "worker_pool": {"max_workers": 0},
            },
            "workflow": {
                "enable": True,
                "db_path": ":memory:",
            },
            "reflection": {"enable": False},
            "feedback": {"enable": False},
            "memory_consolidator": {"enable": False},
            "tracing": {"enable": False},
        }

    @property
    def is_real(self) -> bool:
        return self.runtime == "real"

    @property
    def is_simulated(self) -> bool:
        return self.runtime == "simulated"

    @property
    def simulated_banner(self) -> str:
        return (
            "SIMULATED DEVELOPMENT RUN\n"
            "NOT QUALIFYING\n"
            "NOT PROMOTABLE"
        )
