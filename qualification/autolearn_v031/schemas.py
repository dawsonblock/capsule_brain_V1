"""Schemas for AutoLearn v0.3.1 qualification artifacts.

Defines the structured output schemas for:
- split_manifest.json
- counterfactual results
- verification evidence
- provenance manifest
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SplitEntry:
    """A single entry in the split manifest."""

    task_id: str
    family: str
    archetype: str
    generator: str
    seed: int
    split: str  # experience | validation | test | ood | safety
    allowed_actions: tuple[str, ...]
    verifier_type: str
    setup_digest: str
    prompt_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "archetype": self.archetype,
            "generator": self.generator,
            "seed": self.seed,
            "split": self.split,
            "allowed_actions": list(self.allowed_actions),
            "verifier_type": self.verifier_type,
            "setup_digest": self.setup_digest,
            "prompt_digest": self.prompt_digest,
        }


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """v0.3.1 Section 12: Structured verification evidence."""

    passed: bool
    status: str
    verifier_type: str
    confidence: float
    evidence_refs: tuple[str, ...]
    expected_digest: str | None
    observed_digest: str | None
    error_code: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "verifier_type": self.verifier_type,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
            "expected_digest": self.expected_digest,
            "observed_digest": self.observed_digest,
            "error_code": self.error_code,
        }


@dataclass(frozen=True, slots=True)
class ProvenanceEntry:
    """A single provenance hash entry in the lineage chain."""

    name: str
    sha256: str
    parent_hashes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "parent_hashes": list(self.parent_hashes),
        }


@dataclass(frozen=True, slots=True)
class ProvenanceManifest:
    """v0.3.1 Section 25: Complete provenance manifest."""

    source_tree_sha256: str
    qualification_source_sha256: str
    benchmark_manifest_sha256: str
    split_manifest_sha256: str
    counterfactual_results_sha256: str
    executive_experiences_sha256: str
    dataset_sha256: str
    baseline_policy_sha256: str
    sham_policy_sha256: str
    candidate_policy_sha256: str
    validation_results_sha256: str
    test_results_sha256: str
    ood_results_sha256: str
    promotion_result_sha256: str
    post_promotion_results_sha256: str
    report_sha256: str
    # Environment metadata.
    python_version: str = ""
    platform: str = ""
    package_version: str = ""
    model_id: str = ""
    model_digest: str = ""
    container_image_digest: str = ""
    runtime_config_digest: str = ""
    random_seeds: tuple[str, ...] = ()
    dependency_lock_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_tree_sha256": self.source_tree_sha256,
            "qualification_source_sha256": self.qualification_source_sha256,
            "benchmark_manifest_sha256": self.benchmark_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "counterfactual_results_sha256": self.counterfactual_results_sha256,
            "executive_experiences_sha256": self.executive_experiences_sha256,
            "dataset_sha256": self.dataset_sha256,
            "baseline_policy_sha256": self.baseline_policy_sha256,
            "sham_policy_sha256": self.sham_policy_sha256,
            "candidate_policy_sha256": self.candidate_policy_sha256,
            "validation_results_sha256": self.validation_results_sha256,
            "test_results_sha256": self.test_results_sha256,
            "ood_results_sha256": self.ood_results_sha256,
            "promotion_result_sha256": self.promotion_result_sha256,
            "post_promotion_results_sha256": self.post_promotion_results_sha256,
            "report_sha256": self.report_sha256,
            "python_version": self.python_version,
            "platform": self.platform,
            "package_version": self.package_version,
            "model_id": self.model_id,
            "model_digest": self.model_digest,
            "container_image_digest": self.container_image_digest,
            "runtime_config_digest": self.runtime_config_digest,
            "random_seeds": list(self.random_seeds),
            "dependency_lock_digest": self.dependency_lock_digest,
        }
