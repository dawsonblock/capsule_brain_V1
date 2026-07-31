"""Common schemas, status types, and exceptions for v0.4.4 analysis."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class GateStatus(str, Enum):
    """Status of a gate or sub-gate."""
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    INVALID = "INVALID"


class OutcomeStatus(str, Enum):
    """Status of a counterfactual outcome."""
    MEASURED = "MEASURED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class CounterfactualEquivalenceStatus(str, Enum):
    """Counterfactual equivalence validation result."""
    EQUIVALENT = "EQUIVALENT"
    PARTIALLY_EQUIVALENT = "PARTIALLY_EQUIVALENT"
    NON_EQUIVALENT = "NON_EQUIVALENT"
    UNVERIFIABLE = "UNVERIFIABLE"


class ActionMatrixStatus(str, Enum):
    """Action matrix completeness status."""
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    NONCOMPARABLE = "NONCOMPARABLE"
    UNVERIFIABLE = "UNVERIFIABLE"


class HeadroomStatus(str, Enum):
    """Headroom calculation status."""
    VALID = "VALID"
    INVALID_ORACLE_RELATION = "INVALID_ORACLE_RELATION"
    NON_FINITE = "NON_FINITE"


class ProviderClass(str, Enum):
    """Provider class for qualification."""
    REAL_MODEL = "REAL_MODEL"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    SIMULATED = "SIMULATED"


class AnalysisMode(str, Enum):
    """Analysis mode for the v0.4.4 pipeline."""
    EVIDENCE_ANALYSIS = "evidence-analysis"
    INFRASTRUCTURE = "infrastructure"
    SCIENTIFIC_MINI = "scientific-mini"
    MATCHED_MODEL_PILOT = "matched-model-pilot"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MissingUtilityError(Exception):
    """Raised when an executed outcome is missing a utility value."""


class InvalidOutcomeError(Exception):
    """Raised when an outcome record is structurally invalid."""


class EvidenceValidationError(Exception):
    """Raised when evidence package validation fails."""


class ConfigurationError(Exception):
    """Raised when analysis configuration is invalid."""


class PipelineIntegrityError(Exception):
    """Raised when pipeline integrity is violated."""


# ---------------------------------------------------------------------------
# Provider validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderValidationResult:
    """Result of provider validation."""
    provider_class: str
    model_id: str
    model_revision: str
    model_digest: str
    tokenizer_id: str
    tokenizer_revision: str
    generation_config_digest: str
    provider_manifest_sha256: str
    supports_gate_a: bool
    supports_gate_b: bool
    runtime_type: str
    status: str
    reason: str


# ---------------------------------------------------------------------------
# Exit codes for CLI
# ---------------------------------------------------------------------------


class ExitCode(int, Enum):
    """CLI exit codes."""
    SUCCESS = 0          # Pipeline completed (regardless of scientific PASS/FAIL)
    INTEGRITY_FAILURE = 1
    EVIDENCE_FAILURE = 2
    CONFIG_ERROR = 3
    RUNTIME_ERROR = 4
