"""Common utilities for v0.4.4/v0.4.5 analysis."""
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
from .evidence_levels import (
    EvidenceLevel, EvidenceLevelAssessment,
    assess_evidence_level, detect_evidence_level, REQUIRED_EVIDENCE_LEVELS,
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
    "EvidenceLevel",
    "EvidenceLevelAssessment",
    "assess_evidence_level",
    "detect_evidence_level",
    "REQUIRED_EVIDENCE_LEVELS",
]
