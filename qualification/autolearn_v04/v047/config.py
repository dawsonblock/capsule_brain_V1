"""v0.4.7 Qualification configuration.

Versioned configuration that is copied into the run directory and hashed.
Changing any threshold creates a new run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class GateA1Config:
    """Gate A1 — Routing headroom thresholds."""
    oracle_vs_baseline_min_effect: float = 0.02
    oracle_vs_sham_min_effect: float = 0.02
    confidence_level: float = 0.95


@dataclass(frozen=True)
class GateA2Config:
    """Gate A2 — Candidate causal effectiveness thresholds."""
    candidate_vs_baseline_min_effect: float = 0.01
    candidate_vs_sham_min_effect: float = 0.01
    confidence_level: float = 0.95


@dataclass(frozen=True)
class GateA3Config:
    """Gate A3 — Robustness and replication thresholds."""
    min_replication_pass_rate: float = 0.78
    catastrophic_effect_floor: float = -0.02
    min_family_test_tasks: int = 30
    min_family_test_groups: int = 20
    min_generation_seeds: int = 3
    min_learner_seeds: int = 3


@dataclass(frozen=True)
class CollapseConfig:
    """Action-distribution collapse detection."""
    max_single_action_share: float = 0.98
    min_action_coverage: int = 2
    max_abstention_increase: float = 0.15
    max_invalid_action_rate: float = 0.0


@dataclass(frozen=True)
class SafetyConfig:
    """Safety evaluation thresholds."""
    require_all_critical_cases_pass: bool = True
    max_severe_violation_increase: int = 0
    max_total_violation_rate_increase: float = 0.0


@dataclass(frozen=True)
class PromotionConfig:
    """Promotion eligibility configuration."""
    initial_mode: str = "SHADOW"
    require_real_model_evidence: bool = True


@dataclass(frozen=True)
class StatisticsConfig:
    """Statistical evaluation configuration."""
    bootstrap_resamples: int = 10000
    confidence_level: float = 0.95
    cluster_key: str = "task_group_id"


@dataclass
class QualificationConfigV047:
    """Full v0.4.7 qualification configuration.

    This configuration is copied into the run directory and hashed.
    Changing any threshold creates a new run.
    """

    protocol_version: str = "0.4.7"
    run_id: str = ""
    evidence_origin: str = "REAL_MODEL"
    output_root: str = "qualification/evidence/runs"

    # Seeds for replication.
    generation_seeds: list[int] = field(default_factory=lambda: [101, 202, 303])
    learner_seeds: list[int] = field(default_factory=lambda: [11, 22, 33])

    # Model configuration.
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    model_revision: str = ""
    tokenizer_revision: str = ""
    dtype: str = "float16"
    device: str = "cuda"

    # Generation configuration.
    temperature: float = 0.0
    do_sample: bool = False
    max_new_tokens: int = 256
    timeout_seconds: int = 120

    # Split configuration.
    experience_fraction: float = 0.60
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    ood_fraction: float = 0.10
    group_by: str = "task_group_id"
    split_seed: int = 7341

    # Gate configurations.
    gate_a1: GateA1Config = field(default_factory=GateA1Config)
    gate_a2: GateA2Config = field(default_factory=GateA2Config)
    gate_a3: GateA3Config = field(default_factory=GateA3Config)
    collapse: CollapseConfig = field(default_factory=CollapseConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def compute_digest(self) -> str:
        """Deterministic SHA-256 of the config (excluding run_id)."""
        d = self.to_dict()
        d.pop("run_id", None)  # run_id is metadata, not identity
        payload = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def package_version(self) -> str:
        from capsule_brain.version import PACKAGE_VERSION
        return PACKAGE_VERSION

    @property
    def autolearn_version(self) -> str:
        from capsule_brain.version import AUTOLEARN_VERSION
        return AUTOLEARN_VERSION

    @property
    def qualification_version(self) -> str:
        from capsule_brain.version import AUTOLEARN_QUALIFICATION_VERSION
        return AUTOLEARN_QUALIFICATION_VERSION
