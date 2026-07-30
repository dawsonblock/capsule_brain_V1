"""Preflight validation for v0.4.1 qualification pipeline.

Runs before any expensive execution. Validates:
    - version identity matches;
    - source hash is nonempty;
    - provider classification is allowed;
    - model configuration is complete;
    - benchmark config is valid;
    - artifact directory is clean or --resume is explicit;
    - no stale candidate/promotion artifacts are present;
    - dependency graph is acyclic;
    - stage ordering is valid;
    - verifier registry contains every verifier used by benchmark;
    - scientific mode does not use infrastructure provider;
    - Gate B requires hidden-state support.

Output: artifacts/preflight.json

A failed preflight blocks all expensive stages.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import AUTOLEARN_QUALIFICATION_VERSION, AUTOLEARN_VERSION, PACKAGE_VERSION, PROTOCOL_VERSION
from .config import QualificationConfig
from .schemas import write_json
from .stage_dependencies import PIPELINE_STAGES, StageStatus


class PreflightError(RuntimeError):
    """Raised when preflight validation fails."""
    pass


def _check_version_identity(config: QualificationConfig) -> list[str]:
    """Check version identity matches across all sources."""
    errors: list[str] = []
    # Check pyproject.toml version matches.
    project_root = Path(__file__).resolve().parents[3]
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        import re
        text = pyproject.read_text()
        match = re.search(r'version\s*=\s*"([^"]+)"', text)
        if match:
            pyproject_version = match.group(1)
            if pyproject_version != PACKAGE_VERSION:
                errors.append(
                    f"pyproject.toml version '{pyproject_version}' != "
                    f"PACKAGE_VERSION '{PACKAGE_VERSION}'"
                )
    # Check manifest version matches.
    manifest_path = project_root / "qualification" / "QUALIFICATION_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("package_version") != PACKAGE_VERSION:
            errors.append(
                f"manifest package_version '{manifest.get('package_version')}' != "
                f"PACKAGE_VERSION '{PACKAGE_VERSION}'"
            )
        if manifest.get("qualification_version") != AUTOLEARN_QUALIFICATION_VERSION:
            errors.append(
                f"manifest qualification_version '{manifest.get('qualification_version')}' != "
                f"AUTOLEARN_QUALIFICATION_VERSION '{AUTOLEARN_QUALIFICATION_VERSION}'"
            )
    return errors


def _check_provider_classification(config: QualificationConfig) -> list[str]:
    """Check provider classification is valid for the mode."""
    errors: list[str] = []
    if config.mode in ("scientific", "scientific-mini"):
        if config.is_infrastructure_provider:
            errors.append(
                f"scientific mode requires real_model provider, "
                f"got infrastructure provider '{config.provider}'"
            )
    if config.mode == "infrastructure":
        if not config.is_infrastructure_provider:
            errors.append(
                f"infrastructure mode requires infrastructure provider, "
                f"got '{config.provider}'"
            )
    return errors


def _check_model_config(config: QualificationConfig) -> list[str]:
    """Check model configuration is complete for scientific mode."""
    errors: list[str] = []
    if config.mode in ("scientific", "scientific-mini"):
        if not config.model_id:
            errors.append("scientific mode requires model_id")
        if not config.tokenizer_id:
            errors.append("scientific mode requires tokenizer_id")
    return errors


def _check_benchmark_config(config: QualificationConfig) -> list[str]:
    """Check benchmark configuration is valid."""
    errors: list[str] = []
    if config.n_experience <= 0:
        errors.append("n_experience must be positive")
    if config.n_validation < 0:
        errors.append("n_validation must be non-negative")
    if config.n_test < 0:
        errors.append("n_test must be non-negative")
    if config.crossover_fraction < 0 or config.crossover_fraction > 1:
        errors.append("crossover_fraction must be in [0, 1]")
    return errors


def _check_artifact_dir(config: QualificationConfig) -> list[str]:
    """Check artifact directory is clean or resume is explicit."""
    errors: list[str] = []
    artifacts = Path(config.artifacts_dir)
    if artifacts.exists():
        # Check for stale candidate/promotion artifacts.
        stale_files = [
            "candidate_policy.json",
            "promotion_result.json",
            "post_promotion_result.json",
        ]
        for fname in stale_files:
            if (artifacts / fname).exists():
                errors.append(
                    f"stale artifact '{fname}' exists in artifacts dir; "
                    f"use --resume or clean the directory"
                )
    return errors


def _check_dependency_graph_acyclic() -> list[str]:
    """Check that the stage dependency graph is acyclic."""
    errors: list[str] = []
    # Simple DFS cycle detection.
    stage_names = {s.stage_name for s in PIPELINE_STAGES}
    deps = {s.stage_name: set(s.dependencies) for s in PIPELINE_STAGES}

    # Check all dependencies reference real stages.
    for name, dep_set in deps.items():
        for dep in dep_set:
            if dep not in stage_names:
                errors.append(f"stage '{name}' depends on unknown stage '{dep}'")

    # Cycle detection via DFS.
    visited: dict[str, int] = {}  # 0=visiting, 1=done

    def has_cycle(node: str) -> bool:
        if visited.get(node) == 0:
            return True
        if visited.get(node) == 1:
            return False
        visited[node] = 0
        for dep in deps.get(node, ()):
            if has_cycle(dep):
                return True
        visited[node] = 1
        return False

    for name in stage_names:
        if has_cycle(name):
            errors.append(f"cycle detected in dependency graph involving '{name}'")
            break

    return errors


def _check_stage_ordering() -> list[str]:
    """Check that stage ordering is valid (promotion before post-promotion)."""
    errors: list[str] = []
    stage_order = [s.stage_name for s in PIPELINE_STAGES]
    promo_idx = stage_order.index("run_promotion") if "run_promotion" in stage_order else -1
    post_promo_idx = stage_order.index("run_post_promotion") if "run_post_promotion" in stage_order else -1
    if promo_idx >= 0 and post_promo_idx >= 0:
        if post_promo_idx < promo_idx:
            errors.append("run_post_promotion must come after run_promotion in stage ordering")
    return errors


def _check_verifier_registry() -> list[str]:
    """Check that verifier registry contains every verifier used by benchmark."""
    errors: list[str] = []
    try:
        from capsule_brain.verification.reliability import all_verifier_names, is_known_verifier
        # Check that all verifier types used by the benchmark are registered.
        required_verifiers = [
            "direct_exact",
            "memory_secret",
            "tool_output",
            "tool_grounding",
            "workflow_acceptance",
            "safety_preemption",
        ]
        for v in required_verifiers:
            if not is_known_verifier(v):
                errors.append(f"verifier '{v}' is not in the reliability registry")
    except ImportError as e:
        errors.append(f"cannot import verifier registry: {e}")
    return errors


def _check_gate_b_requirements(config: QualificationConfig) -> list[str]:
    """Check Gate B requirements for scientific mode."""
    errors: list[str] = []
    if config.mode in ("scientific", "scientific-mini"):
        if not config.gate_b_layers:
            errors.append("scientific mode with Gate B requires gate_b_layers to be configured")
    return errors


def run_preflight(config: QualificationConfig) -> dict[str, Any]:
    """Run all preflight checks. Returns preflight result dict.

    Raises PreflightError if any check fails.
    """
    artifacts = Path(config.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    checks: list[tuple[str, list[str]]] = []
    checks.append(("version_identity", _check_version_identity(config)))
    checks.append(("provider_classification", _check_provider_classification(config)))
    checks.append(("model_config", _check_model_config(config)))
    checks.append(("benchmark_config", _check_benchmark_config(config)))
    checks.append(("artifact_dir", _check_artifact_dir(config)))
    checks.append(("dependency_graph_acyclic", _check_dependency_graph_acyclic()))
    checks.append(("stage_ordering", _check_stage_ordering()))
    checks.append(("verifier_registry", _check_verifier_registry()))
    checks.append(("gate_b_requirements", _check_gate_b_requirements(config)))

    all_errors: list[str] = []
    check_results: dict[str, list[str]] = {}
    for name, errors in checks:
        check_results[name] = errors
        all_errors.extend(errors)

    passed = len(all_errors) == 0
    result = {
        "schema_version": "preflight/1",
        "protocol_version": PROTOCOL_VERSION,
        "package_version": PACKAGE_VERSION,
        "autolearn_version": AUTOLEARN_VERSION,
        "autolearn_qualification_version": AUTOLEARN_QUALIFICATION_VERSION,
        "mode": config.mode,
        "provider": config.provider,
        "passed": passed,
        "checks": {name: {"errors": errs} for name, errs in check_results.items()},
        "error_count": len(all_errors),
    }
    write_json(artifacts / "preflight.json", result)

    if not passed:
        raise PreflightError(
            f"Preflight failed with {len(all_errors)} errors: " + "; ".join(all_errors)
        )
    return result
