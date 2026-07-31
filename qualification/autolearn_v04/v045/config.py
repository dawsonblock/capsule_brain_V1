"""Configuration for v0.4.5 evidence repair analysis."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from capsule_brain.version import (
    AUTOLEARN_QUALIFICATION_VERSION,
    AUTOLEARN_VERSION,
    PACKAGE_VERSION,
)

# Evidence package directory
EVIDENCE_DIR = "qualification/evidence/seven_b_full_run"

# Schema version for evidence
SCHEMA_VERSION = "v1"

# Minimum row counts for scientific completeness
MIN_COUNTERFACTUAL_OUTCOMES = 400
MIN_EXPERIENCE_ROWS = 40
MIN_TEST_TASKS = 20

# Gate thresholds
GATE_A_EPSILON = 0.01
GATE_A_ALPHA = 0.05
GATE_A_LCB_PERCENTILE = 2.5

# Serialization parity tolerance
SERIALIZATION_PARITY_TOLERANCE = 1e-9

# Evidence weight thresholds
MIN_EVIDENCE_WEIGHT = 0.0
MIN_Q_TOTAL = 0.0

# Required evidence files
REQUIRED_EVIDENCE_FILES = (
    "EVIDENCE_MANIFEST.json",
    "benchmark_manifest.json",
    "split_manifest.json",
    "provider_manifest.json",
    "counterfactual_outcomes.jsonl",
    "executive_experiences.jsonl",
    "dataset_manifest.json",
    "candidate_policy.json",
    "sham_policy.json",
    "baseline_results.json",
    "candidate_results.json",
    "sham_results.json",
    "oracle_results.json",
    "gate_a_results.json",
    "safety_results.jsonl",
    "coverage_results.json",
    "runtime_completion_diagnostics.json",
    "source_provenance.json",
    "artifact_checksums.json",
)

# Placeholder markers to detect
PLACEHOLDER_MARKERS = (
    "placeholder",
    "real outcomes should replace this",
    "synthetic",
    "fixture only",
    "summary only",
    "not included",
)

# Valid action_status values
VALID_ACTION_STATUSES = frozenset({
    "EXECUTED", "NOT_EXECUTED", "EXECUTION_ERROR", "TIMEOUT", "UNVERIFIABLE",
})


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Configuration for a single v0.4.5 analysis run."""
    run_id: str = ""
    evidence_manifest_digest: str = ""
    mode: str = "evidence-analysis"
    random_seed: int = 42
    bootstrap_seed: int = 42
    n_bootstrap: int = 10000
    n_sham_ensemble_seeds: int = 50
    n_candidate_stability_seeds: int = 50
    gate_a_epsilon: float = GATE_A_EPSILON
    gate_a_alpha: float = GATE_A_ALPHA
    gate_a_lcb_percentile: float = GATE_A_LCB_PERCENTILE
    utility_version: str = "normalized_v1"
    primary_sham_id: str = ""
    feature_schema_digest: str = ""
    learner_config_digest: str = ""
    serialization_parity_tolerance: float = SERIALIZATION_PARITY_TOLERANCE

    def to_identity_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evidence_manifest_digest": self.evidence_manifest_digest,
            "mode": self.mode,
            "random_seed": self.random_seed,
            "bootstrap_seed": self.bootstrap_seed,
            "n_bootstrap": self.n_bootstrap,
            "n_sham_ensemble_seeds": self.n_sham_ensemble_seeds,
            "n_candidate_stability_seeds": self.n_candidate_stability_seeds,
            "gate_a_epsilon": self.gate_a_epsilon,
            "gate_a_alpha": self.gate_a_alpha,
            "gate_a_lcb_percentile": self.gate_a_lcb_percentile,
            "utility_version": self.utility_version,
            "primary_sham_id": self.primary_sham_id,
            "feature_schema_digest": self.feature_schema_digest,
            "learner_config_digest": self.learner_config_digest,
            "serialization_parity_tolerance": self.serialization_parity_tolerance,
        }

    def compute_digest(self) -> str:
        payload = json.dumps(self.to_identity_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def package_version(self) -> str:
        return PACKAGE_VERSION

    @property
    def autolearn_version(self) -> str:
        return AUTOLEARN_VERSION

    @property
    def qualification_version(self) -> str:
        return AUTOLEARN_QUALIFICATION_VERSION
