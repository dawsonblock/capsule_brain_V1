"""Feature registry for v0.4.4 evidence-complete analysis.

Registers all features used in the analysis with lineage metadata and
a SHA-256 schema digest.  Every executive feature must carry full
lineage metadata so that leakage cannot enter the training set
unnoticed.

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
# Maps each known executive feature name to its lineage metadata.
# Features not in this table are assigned conservative defaults and
# still pass the name-based leakage check.
# ---------------------------------------------------------------------------

_FEATURE_METADATA: dict[str, dict[str, Any]] = {
    "prompt_length": {
        "source": "prompt_analyzer",
        "type": "numeric",
        "description": "Length of the user prompt in characters.",
    },
    "log_prompt_length": {
        "source": "prompt_analyzer",
        "type": "numeric",
        "description": "Log-transformed prompt length.",
    },
    "code_indicators": {
        "source": "prompt_analyzer",
        "type": "numeric",
        "description": "Count of code-related indicators in the prompt.",
    },
    "arithmetic_indicators": {
        "source": "prompt_analyzer",
        "type": "numeric",
        "description": "Count of arithmetic-related indicators in the prompt.",
    },
    "structured_output_request": {
        "source": "prompt_analyzer",
        "type": "binary",
        "description": "Whether structured output is requested.",
    },
    "tool_required_keywords": {
        "source": "prompt_analyzer",
        "type": "numeric",
        "description": "Count of tool-required keywords in the prompt.",
    },
    "retrieval_indicators": {
        "source": "prompt_analyzer",
        "type": "numeric",
        "description": "Count of retrieval-related indicators in the prompt.",
    },
    "prior_conversation_depth": {
        "source": "conversation_service",
        "type": "numeric",
        "description": "Number of prior conversation turns.",
    },
    "semantic_memory_hit_count": {
        "source": "memory_service",
        "type": "numeric",
        "description": "Number of semantic memory hits for the prompt.",
    },
    "semantic_memory_similarity": {
        "source": "memory_service",
        "type": "numeric",
        "description": "Top semantic memory similarity score.",
    },
    "num_available_tools": {
        "source": "tool_registry",
        "type": "numeric",
        "description": "Number of tools available in the registry.",
    },
    "workflow_available": {
        "source": "workflow_registry",
        "type": "binary",
        "description": "Whether a workflow capability is available.",
    },
    "previous_attempt_failed": {
        "source": "execution_service",
        "type": "binary",
        "description": "Whether a previous attempt at this task failed.",
    },
    "verification_failure_type_id": {
        "source": "execution_service",
        "type": "categorical",
        "description": "Identifier for the verification failure type.",
    },
    "model_identity_id": {
        "source": "runtime",
        "type": "categorical",
        "description": "Identifier for the model identity.",
    },
    "estimated_difficulty": {
        "source": "prompt_analyzer",
        "type": "numeric",
        "description": "Estimated difficulty score for the task.",
    },
    "task_family_onehot": {
        "source": "task_metadata",
        "type": "categorical",
        "description": "One-hot encoding of the task family.",
    },
    "task_archetype_onehot": {
        "source": "task_metadata",
        "type": "categorical",
        "description": "One-hot encoding of the task archetype.",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_metadata(name: str) -> dict[str, Any]:
    """Return conservative default metadata for an unknown feature."""
    return {
        "source": "unknown",
        "type": "unknown",
        "description": name.replace("_", " ").strip(),
    }


def _matches_leakage_pattern(name: str) -> str | None:
    """Return the leakage pattern matched by ``name``, or None."""
    lowered = name.lower()
    for pat in LEAKAGE_PATTERNS:
        if pat in lowered:
            return pat
    return None


def _get_feature_names(candidate_policy: dict[str, Any]) -> list[str]:
    """Extract feature names from the candidate policy.

    Looks for top-level ``feature_names`` first, then falls back to the
    ``feature_transform`` sub-dict.
    """
    names = candidate_policy.get("feature_names", [])
    if names:
        return list(names)
    transform = candidate_policy.get("feature_transform", {})
    if isinstance(transform, dict):
        names = transform.get("feature_names", [])
        if names:
            return list(names)
    # Final fallback: the canonical FEATURE_NAMES from the production
    # feature extractor.
    try:
        from capsule_brain.autolearn.features import FEATURE_NAMES
        return list(FEATURE_NAMES)
    except Exception:
        return []


def _infer_type(name: str) -> str:
    """Infer a feature type from its name."""
    lowered = name.lower()
    if "onehot" in lowered or "one_hot" in lowered:
        return "categorical"
    if lowered.startswith("is_") or lowered.startswith("has_"):
        return "binary"
    if any(kw in lowered for kw in ("count", "num", "length", "depth", "n_")):
        return "numeric"
    return "numeric"


def _infer_source(name: str) -> str:
    """Infer a feature source from its name."""
    lowered = name.lower()
    if "memory" in lowered:
        return "memory_service"
    if "tool" in lowered:
        return "tool_registry"
    if "workflow" in lowered:
        return "workflow_registry"
    if "prompt" in lowered or "code" in lowered or "arithmetic" in lowered:
        return "prompt_analyzer"
    if "conversation" in lowered:
        return "conversation_service"
    if "family" in lowered or "archetype" in lowered:
        return "task_metadata"
    if "model" in lowered:
        return "runtime"
    if "verification" in lowered or "previous_attempt" in lowered:
        return "execution_service"
    return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_feature_registry(
    candidate_policy: dict,
    benchmark_manifest: dict,
) -> dict:
    """Register all features used in the analysis.

    Builds the registry from the candidate policy's declared feature
    names, attaches metadata, and rejects any feature that violates
    the leakage firewall.

    Args:
        candidate_policy: The candidate policy manifest dict, containing
            ``feature_names`` or a ``feature_transform`` sub-dict.
        benchmark_manifest: The benchmark manifest dict, used to
            cross-reference task families and archetypes.

    Returns:
        A dict with:
          - features: list of {name, type, source, description}.
          - feature_schema_digest: SHA-256 hex digest of the schema.
          - n_features: number of registered features.
    """
    feature_names = _get_feature_names(candidate_policy)

    # Collect known families and archetypes from the benchmark manifest.
    known_families: set[str] = set()
    known_archetypes: set[str] = set()
    tasks = benchmark_manifest.get("tasks", {})
    if isinstance(tasks, dict):
        iterable = tasks.values()
    elif isinstance(tasks, list):
        iterable = tasks
    else:
        iterable = []
    for entry in iterable:
        if isinstance(entry, dict):
            fam = entry.get("family", "")
            arch = entry.get("archetype", "")
            if fam:
                known_families.add(str(fam))
            if arch:
                known_archetypes.add(str(arch))

    features: list[dict[str, Any]] = []

    for name in feature_names:
        meta = _FEATURE_METADATA.get(name, _default_metadata(name))

        # Skip leakage-pattern features entirely.
        leakage = _matches_leakage_pattern(name)
        if leakage is not None:
            continue

        entry = {
            "name": name,
            "type": meta.get("type", _infer_type(name)),
            "source": meta.get("source", _infer_source(name)),
            "description": meta.get("description", name.replace("_", " ").strip()),
        }
        features.append(entry)

    # If no feature names were found, try to infer from benchmark manifest
    # families and archetypes.
    if not features and not feature_names:
        for fam in sorted(known_families):
            feat_name = f"family_{fam}"
            if not _matches_leakage_pattern(feat_name):
                features.append({
                    "name": feat_name,
                    "type": "binary",
                    "source": "task_metadata",
                    "description": f"One-hot indicator for family '{fam}'.",
                })
        for arch in sorted(known_archetypes):
            feat_name = f"archetype_{arch}"
            if not _matches_leakage_pattern(feat_name):
                features.append({
                    "name": feat_name,
                    "type": "binary",
                    "source": "task_metadata",
                    "description": f"One-hot indicator for archetype '{arch}'.",
                })

    # --- Feature schema SHA-256 digest ----------------------------------
    schema_payload = json.dumps(
        [
            {"name": f["name"], "type": f["type"], "source": f["source"]}
            for f in features
        ],
        sort_keys=True,
        ensure_ascii=False,
    ).encode()
    feature_schema_digest = hashlib.sha256(schema_payload).hexdigest()

    return {
        "features": features,
        "feature_schema_digest": feature_schema_digest,
        "n_features": len(features),
    }
