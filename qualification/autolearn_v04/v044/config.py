"""Configuration for v0.4.4 evidence-complete analysis."""
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


# ---------------------------------------------------------------------------
# Headroom
# ---------------------------------------------------------------------------
MINIMUM_DENOMINATOR = 1e-10

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
DEFAULT_N_BOOTSTRAP = 10000
DEFAULT_BOOTSTRAP_SEED = 42

# ---------------------------------------------------------------------------
# Sham ensemble
# ---------------------------------------------------------------------------
DEFAULT_SHAM_ENSEMBLE_SEEDS = 50
DEFAULT_CANDIDATE_STABILITY_SEEDS = 50

# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------
GATE_A_EPSILON = 0.01  # Practical significance threshold
GATE_A_ALPHA = 0.05    # Significance level
GATE_A_LCB_PERCENTILE = 2.5  # Lower confidence bound percentile

# Gate A0 sub-gate names (in order).
GATE_A0_SUB_GATES = [
    "A0.1_provider_eligibility",
    "A0.2_source_tree_provenance",
    "A0.3_evidence_package_checksums",
    "A0.4_version_identity",
    "A0.5_single_model_provider_identity",
    "A0.6_positive_evidence_quality_weights",
    "A0.7_counterfactual_equivalence",
    "A0.8_eligible_action_matrix_completeness",
    "A0.9_verifier_registry_completeness",
    "A0.10_utility_consistency",
    "A0.11_no_final_test_leakage",
    "A0.12_candidate_serialization_parity",
    "A0.13_sham_serialization_parity",
    "A0.14_task_split_manifest_consistency",
    "A0.15_artifact_lineage_completeness",
    "A0.16_metric_consistency",
    "A0.17_safety_evidence_integrity",
    "A0.18_no_stale_artifact_reuse",
]

# ---------------------------------------------------------------------------
# Scale decision thresholds
# ---------------------------------------------------------------------------
HEADROOM_CONCENTRATION_GINI_THRESHOLD = 0.6
HEADROOM_CONCENTRATION_HHI_THRESHOLD = 0.25
HIGH_MARGIN_RECOVERY_THRESHOLD = 0.15
CANDIDATE_STABILITY_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# Matched pilot
# ---------------------------------------------------------------------------
PILOT_TASK_COUNT_MIN = 20
PILOT_TASK_COUNT_MAX = 30

# ---------------------------------------------------------------------------
# Provider classes
# ---------------------------------------------------------------------------
PROVIDER_CLASS_REAL_MODEL = "REAL_MODEL"
PROVIDER_CLASS_INFRASTRUCTURE = "INFRASTRUCTURE"
PROVIDER_CLASS_SIMULATED = "SIMULATED"
VALID_PROVIDER_CLASSES = frozenset({
    PROVIDER_CLASS_REAL_MODEL,
    PROVIDER_CLASS_INFRASTRUCTURE,
    PROVIDER_CLASS_SIMULATED,
})


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Configuration for a single v0.4.4 analysis run.

    Only semantic values are included — no absolute filesystem paths.
    """
    run_id: str = ""
    evidence_manifest_digest: str = ""
    mode: str = "evidence-analysis"
    random_seed: int = 42
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    n_sham_ensemble_seeds: int = DEFAULT_SHAM_ENSEMBLE_SEEDS
    n_candidate_stability_seeds: int = DEFAULT_CANDIDATE_STABILITY_SEEDS
    gate_a_epsilon: float = GATE_A_EPSILON
    gate_a_alpha: float = GATE_A_ALPHA
    gate_a_lcb_percentile: float = GATE_A_LCB_PERCENTILE
    utility_version: str = "normalized_v1"
    primary_sham_id: str = ""
    feature_schema_digest: str = ""
    learner_config_digest: str = ""
    headroom_concentration_gini_threshold: float = HEADROOM_CONCENTRATION_GINI_THRESHOLD
    headroom_concentration_hhi_threshold: float = HEADROOM_CONCENTRATION_HHI_THRESHOLD
    high_margin_recovery_threshold: float = HIGH_MARGIN_RECOVERY_THRESHOLD
    candidate_stability_threshold: float = CANDIDATE_STABILITY_THRESHOLD
    pilot_task_count_min: int = PILOT_TASK_COUNT_MIN
    pilot_task_count_max: int = PILOT_TASK_COUNT_MAX

    def to_identity_dict(self) -> dict[str, Any]:
        """Return only identity-relevant fields (no paths)."""
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
            "headroom_concentration_gini_threshold": self.headroom_concentration_gini_threshold,
            "headroom_concentration_hhi_threshold": self.headroom_concentration_hhi_threshold,
            "high_margin_recovery_threshold": self.high_margin_recovery_threshold,
            "candidate_stability_threshold": self.candidate_stability_threshold,
            "pilot_task_count_min": self.pilot_task_count_min,
            "pilot_task_count_max": self.pilot_task_count_max,
        }

    def compute_digest(self) -> str:
        """Compute a stable SHA-256 digest over identity fields only.

        This digest is independent of absolute filesystem paths.
        """
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
