"""Canonical typed schemas for AutoLearn v0.4.0 qualification artifacts.

Section 4 of the v0.4.0 protocol: every action-task evaluation must
represent execution state explicitly. Availability is NEVER encoded in a
numeric utility field. A missing row never receives utility.

This module defines the typed enums and frozen dataclasses that flow
through the entire pipeline, plus the JSON I/O helpers and hashing
primitives used by the provenance system.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# JSON / hashing helpers
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Stable JSON encoding (sorted keys, no extra whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return sha256_bytes(p.read_bytes())


# The SHA-256 of empty bytes. Provenance must reject this value because it
# indicates an artifact was hashed with no content (e.g. hashing the wrong
# directory, or a file that was never written).
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def write_json(path: str | Path, obj: Any) -> str:
    """Write pretty JSON and return its SHA-256."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, default=str)
    p.write_text(text, encoding="utf-8")
    return sha256_text(text)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Execution availability and verification outcome (Section 4.1)
# ---------------------------------------------------------------------------


class OutcomeAvailability(str, Enum):
    """Whether an action was actually executed on a task.

    This is SEPARATE from whether the executed action succeeded. A missing
    row must never be conflated with a failed execution.
    """

    EXECUTED = "executed"
    NOT_EXECUTED = "not_executed"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    UNVERIFIABLE = "unverifiable"


class VerificationOutcome(str, Enum):
    """The independently-verified task outcome for an executed action."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNVERIFIABLE = "unverifiable"


class IncompleteEvidenceError(RuntimeError):
    """Raised when an action is scored without an executed outcome.

    Section 5.1: the utility function must only score an action whose
    execution occurred. Calling compute_utility on a non-EXECUTED outcome
    is a programming error, not a -1000 sentinel.
    """


@dataclass(frozen=True)
class CounterfactualOutcome:
    """The canonical record for one (task, action) counterfactual execution.

    Availability and verification are represented explicitly. Utility is
    None unless the action was EXECUTED and verified. Reward components and
    execution metadata are preserved for audit.
    """

    task_id: str
    action_id: str
    availability: OutcomeAvailability
    verification: VerificationOutcome | None
    utility: float | None
    reward_components: Mapping[str, float]
    execution_metadata: Mapping[str, Any]
    artifact_digest: str | None
    error_type: str | None = None
    error_message: str | None = None

    def validate(self) -> None:
        """Enforce the availability/utility invariant at construction time."""
        if self.availability is OutcomeAvailability.EXECUTED:
            if self.verification is None:
                raise ValueError("Executed outcome requires verification.")
            if self.utility is None:
                raise ValueError("Executed outcome requires utility.")
        else:
            if self.utility is not None:
                raise ValueError(
                    "Unavailable or failed execution must not carry "
                    "behavioral utility."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "counterfactual-outcome/2",
            "protocol_version": "0.4.0",
            "task_id": self.task_id,
            "action_id": self.action_id,
            "availability": self.availability.value,
            "verification": self.verification.value if self.verification else None,
            "utility": self.utility,
            "reward_components": dict(self.reward_components),
            "execution_metadata": dict(self.execution_metadata),
            "artifact_digest": self.artifact_digest,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CounterfactualOutcome":
        avail = OutcomeAvailability(data.get("availability", "not_executed"))
        verif_raw = data.get("verification")
        verification = VerificationOutcome(verif_raw) if verif_raw else None
        return cls(
            task_id=str(data["task_id"]),
            action_id=str(data["action_id"]),
            availability=avail,
            verification=verification,
            utility=data.get("utility"),
            reward_components=dict(data.get("reward_components", {}) or {}),
            execution_metadata=dict(data.get("execution_metadata", {}) or {}),
            artifact_digest=data.get("artifact_digest"),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
        )


# ---------------------------------------------------------------------------
# Evaluation result type (Section 4.2)
# ---------------------------------------------------------------------------


class EvaluationStatus(str, Enum):
    EVALUATED = "evaluated"
    NOT_RUN = "not_run"
    BLOCKED = "blocked"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class PolicyEvaluation:
    """Result of evaluating one policy on one split.

    No aggregate mean may be emitted when evaluation is incomplete unless
    the report explicitly labels it as a partial descriptive statistic.
    """

    status: EvaluationStatus
    policy_id: str
    split_name: str
    n_manifest_tasks: int
    n_complete_tasks: int
    n_missing_tasks: int
    mean_utility: float | None
    success_rate: float | None
    metrics: Mapping[str, float]
    missing_task_ids: tuple[str, ...]
    blocking_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "policy-evaluation/1",
            "protocol_version": "0.4.0",
            "status": self.status.value,
            "policy_id": self.policy_id,
            "split_name": self.split_name,
            "n_manifest_tasks": self.n_manifest_tasks,
            "n_complete_tasks": self.n_complete_tasks,
            "n_missing_tasks": self.n_missing_tasks,
            "mean_utility": self.mean_utility,
            "success_rate": self.success_rate,
            "metrics": dict(self.metrics),
            "missing_task_ids": list(self.missing_task_ids),
            "blocking_reasons": list(self.blocking_reasons),
        }


# ---------------------------------------------------------------------------
# Gate status (Section 8.3)
# ---------------------------------------------------------------------------


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"


class GateCategory(str, Enum):
    INTEGRITY = "integrity"
    EVIDENCE_COMPLETENESS = "evidence_completeness"
    STATISTICAL_SUPERIORITY = "statistical_superiority"
    REGRESSION = "regression"
    SAFETY = "safety"
    PROVENANCE = "provenance"


@dataclass(frozen=True)
class GateResult:
    """A single promotion gate result."""

    name: str
    category: GateCategory
    status: GateStatus
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASS

    @property
    def is_primary(self) -> bool:
        """Primary gates are counted; the summary decision is not."""
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "status": self.status.value,
            "detail": dict(self.detail),
        }


# ---------------------------------------------------------------------------
# Policy provenance manifest (Section 7.1)
# ---------------------------------------------------------------------------


class ProvenanceError(RuntimeError):
    """Raised when a provenance digest is missing, empty, or mismatched."""


def validate_digest(name: str, value: str) -> None:
    """Validate that a value is a real SHA-256 digest, not empty content.

    Section 7.2: reject empty strings, the empty-content digest, and
    non-64-hex-char values.
    """
    if not value:
        raise ProvenanceError(f"{name} is empty.")
    if value == EMPTY_SHA256:
        raise ProvenanceError(f"{name} is the empty-content digest.")
    if len(value) != 64:
        raise ProvenanceError(f"{name} is not a SHA-256 digest (len={len(value)}).")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProvenanceError(f"{name} is not a hex digest.") from exc


@dataclass(frozen=True)
class PolicyProvenance:
    """Canonical provenance manifest embedded in every trained policy.

    Section 7.1: no field may be blank except optional revision and parent
    fields. Every digest is a real SHA-256 over real content.
    """

    policy_id: str
    policy_version: str
    parent_policy_id: str | None
    source_tree_digest: str
    benchmark_manifest_digest: str
    split_manifest_digest: str
    counterfactual_results_digest: str
    training_dataset_digest: str
    feature_schema_digest: str
    hyperparameter_digest: str
    learner_code_digest: str
    runtime_config_digest: str
    verifier_config_digest: str
    model_id: str
    model_revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    created_at_utc: str
    lineage_digest: str = ""

    def validate(self) -> None:
        """Validate all required digests are non-empty real SHA-256 values."""
        validate_digest("source_tree_digest", self.source_tree_digest)
        validate_digest("benchmark_manifest_digest", self.benchmark_manifest_digest)
        validate_digest("split_manifest_digest", self.split_manifest_digest)
        validate_digest("counterfactual_results_digest", self.counterfactual_results_digest)
        validate_digest("training_dataset_digest", self.training_dataset_digest)
        validate_digest("feature_schema_digest", self.feature_schema_digest)
        validate_digest("hyperparameter_digest", self.hyperparameter_digest)
        validate_digest("learner_code_digest", self.learner_code_digest)
        validate_digest("runtime_config_digest", self.runtime_config_digest)
        validate_digest("verifier_config_digest", self.verifier_config_digest)
        if not self.policy_id:
            raise ProvenanceError("policy_id is empty.")
        if not self.policy_version:
            raise ProvenanceError("policy_version is empty.")
        if not self.model_id:
            raise ProvenanceError("model_id is empty.")
        if not self.tokenizer_id:
            raise ProvenanceError("tokenizer_id is empty.")
        if not self.created_at_utc:
            raise ProvenanceError("created_at_utc is empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "policy-provenance/1",
            "protocol_version": "0.4.0",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "parent_policy_id": self.parent_policy_id,
            "source_tree_digest": self.source_tree_digest,
            "benchmark_manifest_digest": self.benchmark_manifest_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "counterfactual_results_digest": self.counterfactual_results_digest,
            "training_dataset_digest": self.training_dataset_digest,
            "feature_schema_digest": self.feature_schema_digest,
            "hyperparameter_digest": self.hyperparameter_digest,
            "learner_code_digest": self.learner_code_digest,
            "runtime_config_digest": self.runtime_config_digest,
            "verifier_config_digest": self.verifier_config_digest,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "created_at_utc": self.created_at_utc,
            "lineage_digest": self.lineage_digest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyProvenance":
        return cls(
            policy_id=str(data["policy_id"]),
            policy_version=str(data["policy_version"]),
            parent_policy_id=data.get("parent_policy_id"),
            source_tree_digest=str(data["source_tree_digest"]),
            benchmark_manifest_digest=str(data["benchmark_manifest_digest"]),
            split_manifest_digest=str(data["split_manifest_digest"]),
            counterfactual_results_digest=str(data["counterfactual_results_digest"]),
            training_dataset_digest=str(data["training_dataset_digest"]),
            feature_schema_digest=str(data["feature_schema_digest"]),
            hyperparameter_digest=str(data["hyperparameter_digest"]),
            learner_code_digest=str(data["learner_code_digest"]),
            runtime_config_digest=str(data["runtime_config_digest"]),
            verifier_config_digest=str(data["verifier_config_digest"]),
            model_id=str(data["model_id"]),
            model_revision=data.get("model_revision"),
            tokenizer_id=str(data["tokenizer_id"]),
            tokenizer_revision=data.get("tokenizer_revision"),
            created_at_utc=str(data["created_at_utc"]),
            lineage_digest=str(data.get("lineage_digest", "")),
        )


# ---------------------------------------------------------------------------
# Model execution identity (Section 11.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelExecutionIdentity:
    """Records the exact model/tokenizer/runtime used for an execution."""

    provider: str
    model_id: str
    model_revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    dtype: str
    device: str
    quantization: str | None
    generation_config_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "model-execution-identity/1",
            "provider": self.provider,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": self.dtype,
            "device": self.device,
            "quantization": self.quantization,
            "generation_config_digest": self.generation_config_digest,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


# ---------------------------------------------------------------------------
# Activation trace (Section 13.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivationTrace:
    """Stable schema for a passive hidden-state activation artifact."""

    trajectory_id: str
    task_id: str
    action_id: str
    split_name: str
    model_identity_digest: str
    token_ids_digest: str
    boundary_method: str
    start_token_index: int
    end_token_index: int
    layer_ids: tuple[int, ...]
    activation_file: str
    verifier_label: int
    verifier_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "activation-trace/1",
            "protocol_version": "0.4.0",
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "action_id": self.action_id,
            "split_name": self.split_name,
            "model_identity_digest": self.model_identity_digest,
            "token_ids_digest": self.token_ids_digest,
            "boundary_method": self.boundary_method,
            "start_token_index": self.start_token_index,
            "end_token_index": self.end_token_index,
            "layer_ids": list(self.layer_ids),
            "activation_file": self.activation_file,
            "verifier_label": self.verifier_label,
            "verifier_digest": self.verifier_digest,
        }


# ---------------------------------------------------------------------------
# Benchmark task manifest entry
# ---------------------------------------------------------------------------


FAMILIES = (
    "direct_answer",
    "memory_required",
    "tool_required",
    "workflow_required",
    "safety_adversarial",
)

SPLITS = ("experience", "validation", "test", "ood", "safety")

# Crossover archetypes (Section 9.5).
CROSSOVER_TYPES = (
    "memory_exists_direct_optimal",
    "memory_relevant_lexically_dissimilar",
    "memory_stale",
    "memory_contradictory",
    "tool_exists_direct_optimal",
    "tool_mentioned_no_tool_required",
    "tool_failure_direct_better",
    "direct_possible_expensive_workflow_succeeds",
    "workflow_required_short_prompt",
    "workflow_overkill",
    "multiple_actions_succeed_different_costs",
    "superficially_similar_different_actions",
    "unseen_tool_names",
    "unseen_workflow_capability_names",
    "changed_latency_costs",
    "changed_token_costs",
    "changed_reward_weights",
    "same_archetype_different_setup",
)


@dataclass(slots=True)
class BenchmarkTask:
    """A single benchmark task definition (manifest entry)."""

    task_id: str
    family: str
    archetype: str
    split: str
    prompt: str
    allowed_actions: tuple[str, ...]
    setup_spec: dict[str, Any]
    verifier_spec: dict[str, Any]
    expected_output_digest: str = ""
    generator_seed: int = 0
    risk_class: str = "standard"
    crossover_type: str = ""
    # Group key for strict group-based splitting (Section 9.4). All tasks
    # with the same group_id must remain in one split.
    group_id: str = ""
    # Template/generator/entity/capability families for leakage control.
    template_family: str = ""
    generator_family: str = ""
    entity_family: str = ""
    capability_family: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "archetype": self.archetype,
            "split": self.split,
            "prompt": self.prompt,
            "allowed_actions": list(self.allowed_actions),
            "setup_spec": self.setup_spec,
            "verifier_spec": self.verifier_spec,
            "expected_output_digest": self.expected_output_digest,
            "generator_seed": self.generator_seed,
            "risk_class": self.risk_class,
            "crossover_type": self.crossover_type,
            "group_id": self.group_id,
            "template_family": self.template_family,
            "generator_family": self.generator_family,
            "entity_family": self.entity_family,
            "capability_family": self.capability_family,
        }


@dataclass(frozen=True, slots=True)
class SplitEntry:
    task_id: str
    family: str
    archetype: str
    split: str
    allowed_actions: tuple[str, ...]
    verifier_type: str
    setup_digest: str
    prompt_digest: str
    crossover_type: str = ""
    group_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "archetype": self.archetype,
            "split": self.split,
            "allowed_actions": list(self.allowed_actions),
            "verifier_type": self.verifier_type,
            "setup_digest": self.setup_digest,
            "prompt_digest": self.prompt_digest,
            "crossover_type": self.crossover_type,
            "group_id": self.group_id,
        }


def split_entry_from_task(task: BenchmarkTask) -> SplitEntry:
    return SplitEntry(
        task_id=task.task_id,
        family=task.family,
        archetype=task.archetype,
        split=task.split,
        allowed_actions=task.allowed_actions,
        verifier_type=task.verifier_spec.get("type", "deterministic"),
        setup_digest=sha256_json(task.setup_spec),
        prompt_digest=sha256_text(task.prompt),
        crossover_type=task.crossover_type,
        group_id=task.group_id,
    )


# ---------------------------------------------------------------------------
# Counterfactual completeness (Section 4.3)
# ---------------------------------------------------------------------------


def is_counterfactually_complete(
    task: BenchmarkTask,
    outcomes: Mapping[tuple[str, str], CounterfactualOutcome],
) -> bool:
    """A task is eligible for counterfactual comparison only when every
    required action has an EXECUTED outcome record."""
    for action in task.allowed_actions:
        outcome = outcomes.get((task.task_id, action))
        if outcome is None:
            return False
        if outcome.availability is not OutcomeAvailability.EXECUTED:
            return False
    return True


def is_counterfactually_complete_dict(
    task: dict[str, Any],
    outcomes_by_task_action: Mapping[tuple[str, str], CounterfactualOutcome],
) -> bool:
    """Dict-based variant for use with manifest entries."""
    for action in task["allowed_actions"]:
        outcome = outcomes_by_task_action.get((task["task_id"], action))
        if outcome is None:
            return False
        if outcome.availability is not OutcomeAvailability.EXECUTED:
            return False
    return True
