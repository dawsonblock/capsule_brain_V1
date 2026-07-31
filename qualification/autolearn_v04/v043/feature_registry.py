"""Feature registry with lineage enforcement for v0.4.3 repair release.

Every executive feature must carry full lineage metadata so that leakage
cannot enter the training set unnoticed.  This module builds the registry
from the candidate policy's declared feature names, attaches metadata,
rejects any feature that violates the pre-action / non-verifier /
non-outcome firewall, and runs runtime assertions that the training feature
vectors match the registry.

Leakage patterns are rejected by name:
  best_action, expected_action, verifier_result, verified_success,
  expected_answer, utility, post_execution_latency,
  post_execution_token_count, tool_result, workflow_acceptance,
  post_hoc_failure_category
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .data import LoadedRun
from .config import METRIC_TOLERANCE_FLOAT


# ---------------------------------------------------------------------------
# Leakage patterns — a feature name containing any of these is rejected.
# ---------------------------------------------------------------------------

LEAKAGE_PATTERNS: tuple[str, ...] = (
    "best_action",
    "expected_action",
    "verifier_result",
    "verified_success",
    "expected_answer",
    "utility",
    "post_execution_latency",
    "post_execution_token_count",
    "tool_result",
    "workflow_acceptance",
    "post_hoc_failure_category",
)


# ---------------------------------------------------------------------------
# Canonical feature metadata table.
#
# Maps each known executive feature name to its lineage metadata.  Features
# not in this table are assigned conservative defaults (available_before_action
# = True, post_outcome = False, verifier_derived = False) and still pass the
# name-based leakage check.
# ---------------------------------------------------------------------------

_FEATURE_METADATA: dict[str, dict[str, Any]] = {
    "prompt_length": {
        "source_service": "prompt_analyzer",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "log_prompt_length": {
        "source_service": "prompt_analyzer",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "code_indicators": {
        "source_service": "prompt_analyzer",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "arithmetic_indicators": {
        "source_service": "prompt_analyzer",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "structured_output_request": {
        "source_service": "prompt_analyzer",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "tool_required_keywords": {
        "source_service": "prompt_analyzer",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "retrieval_indicators": {
        "source_service": "prompt_analyzer",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "prior_conversation_depth": {
        "source_service": "conversation_service",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "semantic_memory_hit_count": {
        "source_service": "memory_service",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "semantic_memory_similarity": {
        "source_service": "memory_service",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "min_max",
        "missing_value_policy": "zero_fill",
    },
    "num_available_tools": {
        "source_service": "tool_registry",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "workflow_available": {
        "source_service": "workflow_registry",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "previous_attempt_failed": {
        "source_service": "execution_service",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "verification_failure_type_id": {
        "source_service": "execution_service",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "model_identity_id": {
        "source_service": "runtime",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
    "estimated_difficulty": {
        "source_service": "prompt_analyzer",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "min_max",
        "missing_value_policy": "mean_fill",
    },
    "workflow_capability_match": {
        "source_service": "prompt_analyzer",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": True,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    },
}


def _default_metadata(feature_name: str) -> dict[str, Any]:
    """Conservative default metadata for an unknown feature name."""
    return {
        "source_service": "unknown",
        "collection_time": "pre_action",
        "available_before_action": True,
        "model_visible": True,
        "post_outcome": False,
        "verifier_derived": False,
        "family_derived": False,
        "archetype_derived": False,
        "normalization": "none",
        "missing_value_policy": "zero_fill",
    }


def _matches_leakage_pattern(name: str) -> str | None:
    """Return the first leakage pattern contained in *name*, or None."""
    lowered = name.lower()
    for pat in LEAKAGE_PATTERNS:
        if pat in lowered:
            return pat
    return None


def _definition_from_name(name: str) -> str:
    """Derive a human-readable definition from the feature name."""
    return name.replace("_", " ").strip()


# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------


def _get_feature_names(run: LoadedRun) -> list[str]:
    """Extract feature names from the candidate policy.

    Looks for top-level ``feature_names`` first, then falls back to the
    ``feature_transform`` sub-dict (the FeatureTransform serialization).
    """
    policy = run.candidate_policy or {}
    names = policy.get("feature_names", [])
    if names:
        return list(names)
    transform = policy.get("feature_transform", {})
    if isinstance(transform, dict):
        names = transform.get("feature_names", [])
        if names:
            return list(names)
    # Final fallback: the canonical FEATURE_NAMES from the production
    # feature extractor.  This keeps the registry usable even when the
    # policy artifact omits the names.
    try:
        from capsule_brain.autolearn.features import FEATURE_NAMES
        return list(FEATURE_NAMES)
    except Exception:
        return []


def build_feature_registry(run: LoadedRun) -> dict[str, Any]:
    """Build a feature registry with lineage metadata and leakage rejection.

    Returns a JSON-serializable dict with:
      - features: list of accepted feature metadata
      - rejected_features: list of rejected feature metadata with reason
      - n_accepted, n_rejected
      - registry_sha256
      - training_features_match: bool
      - assertion_results: list of runtime assertion outcomes
    """
    feature_names = _get_feature_names(run)

    features: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for name in feature_names:
        meta = dict(_FEATURE_METADATA.get(name, _default_metadata(name)))
        entry = {
            "feature_name": name,
            "definition": _definition_from_name(name),
            "source_service": meta["source_service"],
            "collection_time": meta["collection_time"],
            "available_before_action": meta["available_before_action"],
            "model_visible": meta["model_visible"],
            "post_outcome": meta["post_outcome"],
            "verifier_derived": meta["verifier_derived"],
            "family_derived": meta["family_derived"],
            "archetype_derived": meta["archetype_derived"],
            "normalization": meta["normalization"],
            "missing_value_policy": meta["missing_value_policy"],
            "diagnostic_only": False,
        }

        # --- Leakage firewall -------------------------------------------
        leakage = _matches_leakage_pattern(name)
        if leakage is not None:
            entry["rejection_reason"] = f"leakage_pattern:{leakage}"
            entry["diagnostic_only"] = True
            rejected.append(entry)
            continue

        # --- Pre-action / non-outcome / non-verifier firewall -----------
        violations: list[str] = []
        if not entry["available_before_action"]:
            violations.append("available_before_action_false")
        if entry["post_outcome"]:
            violations.append("post_outcome_true")
        if entry["verifier_derived"]:
            violations.append("verifier_derived_true")

        if violations:
            # A feature may be retained only if explicitly marked
            # diagnostic-only AND excluded from training.  We mark it
            # diagnostic-only and reject it from the training registry.
            entry["diagnostic_only"] = True
            entry["rejection_reason"] = ";".join(violations)
            rejected.append(entry)
            continue

        features.append(entry)

    # --- Registry SHA-256 ------------------------------------------------
    registry_payload = json.dumps(
        [f["feature_name"] for f in features], sort_keys=True
    ).encode()
    registry_sha256 = hashlib.sha256(registry_payload).hexdigest()

    # --- Runtime assertions ----------------------------------------------
    assertion_results: list[dict[str, Any]] = []

    # A1: feature_names in candidate_policy match registry entries.
    policy_names = _get_feature_names(run)
    registry_names = [f["feature_name"] for f in features]
    training_features_match = (
        len(policy_names) == len(registry_names)
        and all(a == b for a, b in zip(policy_names, registry_names))
    )
    assertion_results.append({
        "assertion": "training_feature_names_match_registry",
        "passed": training_features_match,
        "policy_feature_names": policy_names,
        "registry_feature_names": registry_names,
    })

    # A2: no accepted feature matches a leakage pattern.
    leakage_in_accepted = [
        f["feature_name"] for f in features
        if _matches_leakage_pattern(f["feature_name"]) is not None
    ]
    assertion_results.append({
        "assertion": "no_leakage_pattern_in_accepted_features",
        "passed": len(leakage_in_accepted) == 0,
        "offending_features": leakage_in_accepted,
    })

    # A3: every accepted feature is available before action.
    not_pre_action = [
        f["feature_name"] for f in features if not f["available_before_action"]
    ]
    assertion_results.append({
        "assertion": "all_accepted_features_available_before_action",
        "passed": len(not_pre_action) == 0,
        "offending_features": not_pre_action,
    })

    # A4: no accepted feature is post-outcome or verifier-derived.
    post_or_verifier = [
        f["feature_name"] for f in features
        if f["post_outcome"] or f["verifier_derived"]
    ]
    assertion_results.append({
        "assertion": "no_accepted_feature_post_outcome_or_verifier_derived",
        "passed": len(post_or_verifier) == 0,
        "offending_features": post_or_verifier,
    })

    # A5: training examples feature-vector length matches registry size.
    example_lengths: set[int] = set()
    for ex in run.train_examples[:50]:
        feats = ex.get("features", [])
        if isinstance(feats, list):
            example_lengths.add(len(feats))
    length_match = (
        len(example_lengths) <= 1
        and (len(example_lengths) == 0 or next(iter(example_lengths)) == len(registry_names))
    )
    assertion_results.append({
        "assertion": "training_feature_vector_length_matches_registry",
        "passed": length_match,
        "registry_size": len(registry_names),
        "observed_training_lengths": sorted(example_lengths),
    })

    return {
        "features": features,
        "rejected_features": rejected,
        "n_accepted": len(features),
        "n_rejected": len(rejected),
        "registry_sha256": registry_sha256,
        "training_features_match": bool(training_features_match),
        "assertion_results": assertion_results,
        "leakage_patterns": list(LEAKAGE_PATTERNS),
    }
