"""Qualification configuration for AutoLearn v0.4.0 (Capsule Brain 2.15.4).

Section 17: versioned configuration validated before any expensive
execution. Supports smoke and qualification modes, real model backend
selection, and all statistical thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import (
    AUTOLEARN_QUALIFICATION_VERSION,
    AUTOLEARN_VERSION,
    PACKAGE_VERSION,
    PROTOCOL_VERSION,
)


# Provider identifiers.
PROVIDER_INFRASTRUCTURE = "qual_grounded"  # deterministic, infrastructure-only
PROVIDER_LOCAL_TRANSFORMERS = "local_transformers"  # real HF transformers model

# The infrastructure provider may qualify infrastructure only -- never a
# causal learning claim (Section 11.2).
INFRASTRUCTURE_ONLY_PROVIDERS = frozenset({PROVIDER_INFRASTRUCTURE})


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    """Configuration for the v0.4.0 qualification pipeline."""

    # Mode: "smoke" for pipeline-integrity wiring check,
    # "infrastructure" for integration qualification with grounded provider,
    # "scientific" for the full real-model experiment.
    mode: str = "smoke"
    runtime: str = "real"

    # Smoke subset size (small deterministic wiring check).
    smoke_task_count: int = 12

    # Minimum task counts for the full benchmark manifest (Section 17).
    n_experience: int = 240
    n_validation: int = 100
    n_test: int = 200
    n_ood: int = 120
    n_safety: int = 80

    # At least 40% of validation/test/OOD tasks must be crossover (Section 9.5).
    crossover_fraction: float = 0.40

    # Strict group-based splitting (Section 9.4).
    split_grouping: str = "strict"

    # Utility weights (mirrors UtilityConfig defaults).
    w_success: float = 10.0
    w_latency: float = 0.5
    w_tokens: float = 0.25
    w_tool_failure: float = 1.0
    w_unnecessary_tool: float = 2.0
    w_unnecessary_workflow: float = 2.0
    w_runtime_error: float = 20.0
    w_safety: float = 50.0

    # Gate A statistical thresholds (Section 10.4).
    epsilon_utility: float = 0.01  # practical improvement threshold
    effective_coverage_min: float = 0.60
    tool_precision_min: float = 0.6
    tool_recall_min: float = 0.6
    workflow_precision_min: float = 0.6
    workflow_recall_min: float = 0.6
    ood_utility_floor: float = 0.0
    calibration_max: float = 0.3
    family_regression_tolerance: float = 0.02
    max_single_action_share: float = 0.95
    max_kl_from_baseline: float = 0.25
    kl_epsilon_smooth: float = 0.1  # smoothing for baseline distribution in KL
    ambiguity_margin: float = 0.05

    # Bootstrap settings (Section 10.3).
    bootstrap_replicates: int = 10000
    bootstrap_seed: int = 42
    confidence_level: float = 0.95
    multiple_comparison_method: str = "holm"

    # Gate B thresholds (Section 12.12).
    gate_b_minimum_auroc_margin_over_chance: float = 0.03
    gate_b_require_superiority_over_best_nonlatent: bool = True
    gate_b_distance_metric: str = "euclidean"
    gate_b_activation_dtype: str = "float32"
    gate_b_layers: str = "late_quartile"
    gate_b_n_seeds: int = 3

    # Runtime isolation (Section 9.2).
    execution_timeout_seconds: float = 120.0
    retry_count: int = 1
    isolate_filesystem: bool = True
    isolate_memory: bool = True
    randomize_action_order: bool = True

    # Candidate learner (Section 9.6/9.7).
    candidate_model_type: str = "multinomial_logistic"
    candidate_class_weighting: str = "balanced"

    # Runtime completion diagnostics (Section 9).
    min_action_completion: float = 0.50

    # Model backend (Section 11).
    provider: str = PROVIDER_INFRASTRUCTURE
    model: str = "qual-grounded-v04"
    device: str = "cpu"
    dtype: str = "float32"
    offline: bool = True

    # Artifact directory (repository-relative).
    artifacts_dir: str = "artifacts_v04"
    # Random seed for task generation.
    task_seed: int = 42
    # Isolation base directory for per-action environments.
    isolation_root: str = ".cf_isolation_v04"
    # Sandbox root for workflow acceptance execution.
    sandbox_root: str = ".qual_sandbox_v04"

    # Model id used for qualification.
    model_id: str = "qual-grounded-v04"
    tokenizer_id: str = "qual-grounded-v04"

    # CLI overrides.
    max_tasks: int | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("smoke", "infrastructure", "scientific"):
            raise ValueError(f"mode must be 'smoke', 'infrastructure', or 'scientific', got {self.mode!r}")
        if self.runtime not in ("real", "simulated"):
            raise ValueError(f"runtime must be 'real' or 'simulated', got {self.runtime!r}")

    # ----- derived properties ------------------------------------------------

    @property
    def is_smoke(self) -> bool:
        return self.mode == "smoke"

    @property
    def is_infrastructure_mode(self) -> bool:
        return self.mode == "infrastructure"

    @property
    def is_scientific(self) -> bool:
        return self.mode == "scientific"

    @property
    def is_qualification(self) -> bool:
        return self.mode in ("scientific", "qualification")

    @property
    def is_real(self) -> bool:
        return self.runtime == "real"

    @property
    def is_simulated(self) -> bool:
        return self.runtime == "simulated"

    @property
    def is_infrastructure_provider(self) -> bool:
        return self.provider in INFRASTRUCTURE_ONLY_PROVIDERS

    @property
    def is_real_model_provider(self) -> bool:
        return self.provider == PROVIDER_LOCAL_TRANSFORMERS

    @property
    def simulated_banner(self) -> str:
        return (
            "SIMULATED DEVELOPMENT RUN\n"
            "NOT QUALIFYING\n"
            "NOT PROMOTABLE"
        )

    @property
    def infrastructure_provider_banner(self) -> str:
        return (
            "INFRASTRUCTURE-INTEGRATION PROVIDER\n"
            "This provider may qualify infrastructure only.\n"
            "It MUST NOT be used for a causal-learning claim (Gate A).\n"
            "Gate A scientific qualification is BLOCKED under this provider."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "runtime": self.runtime,
            "smoke_task_count": self.smoke_task_count,
            "n_experience": self.n_experience,
            "n_validation": self.n_validation,
            "n_test": self.n_test,
            "n_ood": self.n_ood,
            "n_safety": self.n_safety,
            "crossover_fraction": self.crossover_fraction,
            "split_grouping": self.split_grouping,
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
            "family_regression_tolerance": self.family_regression_tolerance,
            "max_single_action_share": self.max_single_action_share,
            "max_kl_from_baseline": self.max_kl_from_baseline,
            "ambiguity_margin": self.ambiguity_margin,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "multiple_comparison_method": self.multiple_comparison_method,
            "gate_b_minimum_auroc_margin_over_chance": self.gate_b_minimum_auroc_margin_over_chance,
            "gate_b_require_superiority_over_best_nonlatent": self.gate_b_require_superiority_over_best_nonlatent,
            "gate_b_distance_metric": self.gate_b_distance_metric,
            "gate_b_activation_dtype": self.gate_b_activation_dtype,
            "gate_b_layers": self.gate_b_layers,
            "gate_b_n_seeds": self.gate_b_n_seeds,
            "execution_timeout_seconds": self.execution_timeout_seconds,
            "retry_count": self.retry_count,
            "isolate_filesystem": self.isolate_filesystem,
            "isolate_memory": self.isolate_memory,
            "randomize_action_order": self.randomize_action_order,
            "candidate_model_type": self.candidate_model_type,
            "candidate_class_weighting": self.candidate_class_weighting,
            "provider": self.provider,
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
            "offline": self.offline,
            "artifacts_dir": self.artifacts_dir,
            "task_seed": self.task_seed,
            "isolation_root": self.isolation_root,
            "sandbox_root": self.sandbox_root,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "max_tasks": self.max_tasks,
            "strict": self.strict,
            "package_version": PACKAGE_VERSION,
            "autolearn_version": AUTOLEARN_VERSION,
            "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
            "protocol_version": PROTOCOL_VERSION,
        }

    def to_capsule_config(self) -> dict[str, Any]:
        """Build a Capsule Brain application config for the qualification runtime."""
        return {
            "llm_gateway": {
                "enable": True,
                "default_model": self.model_id,
                "models": {
                    self.model_id: {
                        "provider": self.provider,
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

    def runtime_config_digest(self) -> str:
        """SHA-256 over the runtime-relevant config subset."""
        from .schemas import sha256_json
        return sha256_json({
            "runtime": self.runtime,
            "provider": self.provider,
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
            "execution_timeout_seconds": self.execution_timeout_seconds,
            "retry_count": self.retry_count,
            "isolate_filesystem": self.isolate_filesystem,
            "isolate_memory": self.isolate_memory,
            "randomize_action_order": self.randomize_action_order,
        })

    def verifier_config_digest(self) -> str:
        """SHA-256 over the verifier-relevant config subset."""
        from .schemas import sha256_json
        return sha256_json({
            "verifier_type": "deterministic",
            "w_success": self.w_success,
            "w_latency": self.w_latency,
            "w_tokens": self.w_tokens,
            "w_tool_failure": self.w_tool_failure,
            "w_unnecessary_tool": self.w_unnecessary_tool,
            "w_unnecessary_workflow": self.w_unnecessary_workflow,
            "w_runtime_error": self.w_runtime_error,
            "w_safety": self.w_safety,
        })
