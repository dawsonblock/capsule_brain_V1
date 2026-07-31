"""Single-revision model identity check for v0.4.4 evidence analysis.

This module enforces that exactly one model revision, one model id, one
tokenizer revision, one generation config digest, one provider class,
and one utility version appear across the provider, dataset, and
benchmark manifests.

Missing values are failures — they are not treated as "successfully
uniform".  The previous宽松 check ``len(revisions) <= 1`` is replaced
with the strict ``len(revisions) == 1``.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_values(
    field: str,
    *manifests: dict[str, Any],
) -> list[Any]:
    """Collect non-empty values for *field* across all *manifests*."""
    values: list[Any] = []
    for m in manifests:
        if not isinstance(m, dict):
            continue
        v = m.get(field)
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        values.append(v)
    return values


def _unique(values: list[Any]) -> list[Any]:
    """Return a sorted list of unique values (preserving type for sorting)."""
    seen: list[Any] = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_model_identity(
    provider_manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    benchmark_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify single-revision model identity across all manifests.

    Returns a dict with ``status`` ("PASS" or "FAIL"), lists of unique
    values per field, per-field counts, and a list of error strings.
    """
    errors: list[str] = []

    # --- Collect values from all three manifests ---------------------------
    model_ids = _collect_values("model_id", provider_manifest, dataset_manifest, benchmark_manifest)
    model_revisions = _collect_values(
        "model_revision", provider_manifest, dataset_manifest, benchmark_manifest
    )
    tokenizer_revisions = _collect_values(
        "tokenizer_revision", provider_manifest, dataset_manifest, benchmark_manifest
    )
    generation_config_digests = _collect_values(
        "generation_config_digest",
        provider_manifest,
        dataset_manifest,
        benchmark_manifest,
    )
    provider_classes = _collect_values(
        "provider_class", provider_manifest, dataset_manifest, benchmark_manifest
    )
    utility_versions = _collect_values(
        "utility_version", provider_manifest, dataset_manifest, benchmark_manifest
    )

    # --- Unique sets -------------------------------------------------------
    unique_model_ids = _unique(model_ids)
    unique_model_revisions = _unique(model_revisions)
    unique_tokenizer_revisions = _unique(tokenizer_revisions)
    unique_generation_config_digests = _unique(generation_config_digests)
    unique_provider_classes = _unique(provider_classes)
    unique_utility_versions = _unique(utility_versions)

    # --- Counts (unique counts) -------------------------------------------
    counts: dict[str, int] = {
        "model_id": len(unique_model_ids),
        "model_revision": len(unique_model_revisions),
        "tokenizer_revision": len(unique_tokenizer_revisions),
        "generation_config_digest": len(unique_generation_config_digests),
        "provider_class": len(unique_provider_classes),
        "utility_version": len(unique_utility_versions),
    }

    # --- Strict single-revision checks (== 1, not <= 1) --------------------
    field_labels = {
        "model_id": "model_id",
        "model_revision": "model_revision",
        "tokenizer_revision": "tokenizer_revision",
        "generation_config_digest": "generation_config_digest",
        "provider_class": "provider_class",
        "utility_version": "utility_version",
    }

    unique_lists = {
        "model_id": unique_model_ids,
        "model_revision": unique_model_revisions,
        "tokenizer_revision": unique_tokenizer_revisions,
        "generation_config_digest": unique_generation_config_digests,
        "provider_class": unique_provider_classes,
        "utility_version": unique_utility_versions,
    }

    for field, count in counts.items():
        label = field_labels[field]
        if count == 0:
            errors.append(
                f"{label}: missing across all manifests "
                f"(expected exactly 1 unique value, found 0) — "
                f"missing values are failures, not successful uniformity"
            )
        elif count != 1:
            errors.append(
                f"{label}: expected exactly 1 unique value, found {count} "
                f"(values: {unique_lists[field]})"
            )

    # --- Assemble result ---------------------------------------------------
    status = "PASS" if not errors else "FAIL"

    return {
        "status": status,
        "unique_model_ids": unique_model_ids,
        "unique_model_revisions": unique_model_revisions,
        "unique_tokenizer_revisions": unique_tokenizer_revisions,
        "unique_generation_config_digests": unique_generation_config_digests,
        "unique_provider_classes": unique_provider_classes,
        "unique_utility_versions": unique_utility_versions,
        "counts": counts,
        "errors": errors,
    }
