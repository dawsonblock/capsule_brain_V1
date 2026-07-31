"""Common utilities for v0.4.4 analysis."""
from __future__ import annotations

from .headroom import compute_recovered_headroom, RecoveredHeadroomResult, compute_task_ids_digest
from .source_hash import hash_source_tree, SourceTreeHashResult, validate_source_hash
from .schemas import (
    GateStatus, OutcomeStatus, CounterfactualEquivalenceStatus,
    ActionMatrixStatus, HeadroomStatus, ProviderClass, AnalysisMode,
    MissingUtilityError, InvalidOutcomeError, EvidenceValidationError,
    ConfigurationError, PipelineIntegrityError,
    ProviderValidationResult, ExitCode,
)

__all__ = [
    "compute_recovered_headroom",
    "RecoveredHeadroomResult",
    "compute_task_ids_digest",
    "hash_source_tree",
    "SourceTreeHashResult",
    "validate_source_hash",
    "GateStatus",
    "OutcomeStatus",
    "CounterfactualEquivalenceStatus",
    "ActionMatrixStatus",
    "HeadroomStatus",
    "ProviderClass",
    "AnalysisMode",
    "MissingUtilityError",
    "InvalidOutcomeError",
    "EvidenceValidationError",
    "ConfigurationError",
    "PipelineIntegrityError",
    "ProviderValidationResult",
    "ExitCode",
]
