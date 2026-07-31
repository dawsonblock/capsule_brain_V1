"""Strict real-model provider validation for v0.4.4 evidence analysis.

This module implements fail-closed provider validation.  A provider is
only accepted as a real model when *all* of the following hold:

    - provider_class == "REAL_MODEL"  (exact string match)
    - supports_gate_a_claim == True
    - runtime_type == "real"
    - model_id is nonempty
    - model_revision is nonempty
    - tokenizer_id is nonempty
    - generation_config_digest is nonempty
    - provider_manifest_sha256 is a valid 64-char hex digest

We do NOT infer a real model merely because ``is_infrastructure_provider``
is false.  Unknown provider classes fail closed.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from qualification.autolearn_v04.common.schemas import (
    ProviderClass,
    ProviderValidationResult,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VALID_PROVIDER_CLASSES: frozenset[str] = frozenset(
    c.value for c in ProviderClass
)

_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _is_valid_sha256(value: Any) -> bool:
    """Return True when *value* is a 64-character hex string."""
    return isinstance(value, str) and bool(_HEX64_RE.match(value))


def _compute_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Compute a stable SHA-256 hex digest over the manifest content."""
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _nonempty_str(value: Any) -> bool:
    """Return True when *value* is a non-empty string."""
    return isinstance(value, str) and value.strip() != ""


def validate_provider(provider_manifest: dict[str, Any]) -> ProviderValidationResult:
    """Validate a provider manifest for real-model eligibility.

    Returns a :class:`ProviderValidationResult` with ``status`` set to
    ``"PASS"`` only when every required condition is met.  Any unknown
    provider class or missing/invalid field causes a fail-closed result.
    """
    errors: list[str] = []

    # --- Field extraction ---------------------------------------------------
    provider_class = provider_manifest.get("provider_class", "")
    supports_gate_a_claim = provider_manifest.get("supports_gate_a_claim", False)
    supports_gate_b_claim = provider_manifest.get("supports_gate_b_claim", False)
    runtime_type = provider_manifest.get("runtime_type", "")
    model_id = provider_manifest.get("model_id", "")
    model_revision = provider_manifest.get("model_revision", "")
    model_digest = provider_manifest.get("model_digest", "")
    tokenizer_id = provider_manifest.get("tokenizer_id", "")
    tokenizer_revision = provider_manifest.get("tokenizer_revision", "")
    generation_config_digest = provider_manifest.get("generation_config_digest", "")

    # Compute or validate the manifest SHA-256.
    provided_sha = provider_manifest.get("provider_manifest_sha256")
    if _is_valid_sha256(provided_sha):
        manifest_sha256 = provided_sha  # type: ignore[arg-type]
    else:
        manifest_sha256 = _compute_manifest_sha256(provider_manifest)

    # --- Check 1: provider_class is a known class ---------------------------
    if provider_class not in _VALID_PROVIDER_CLASSES:
        errors.append(
            f"provider_class '{provider_class}' is not a valid class "
            f"(expected one of {sorted(_VALID_PROVIDER_CLASSES)})"
        )

    # --- Check 2: provider_class == REAL_MODEL (exact match) ----------------
    if provider_class != ProviderClass.REAL_MODEL.value:
        errors.append(
            f"provider_class must be exactly 'REAL_MODEL' "
            f"(observed '{provider_class}'); "
            f"real-model eligibility is not inferred from "
            f"is_infrastructure_provider being false"
        )

    # --- Check 3: supports_gate_a_claim == True -----------------------------
    if supports_gate_a_claim is not True:
        errors.append(
            f"supports_gate_a_claim must be True (observed "
            f"{supports_gate_a_claim!r})"
        )

    # --- Check 4: runtime_type == "real" ------------------------------------
    if runtime_type != "real":
        errors.append(
            f"runtime_type must be 'real' (observed '{runtime_type}')"
        )

    # --- Check 5: model_id nonempty -----------------------------------------
    if not _nonempty_str(model_id):
        errors.append("model_id must be a nonempty string")

    # --- Check 6: model_revision nonempty ----------------------------------
    if not _nonempty_str(model_revision):
        errors.append("model_revision must be a nonempty string")

    # --- Check 7: tokenizer_id nonempty ------------------------------------
    if not _nonempty_str(tokenizer_id):
        errors.append("tokenizer_id must be a nonempty string")

    # --- Check 8: generation_config_digest nonempty ------------------------
    if not _nonempty_str(generation_config_digest):
        errors.append("generation_config_digest must be a nonempty string")

    # --- Check 9: provider_manifest_sha256 valid (64 hex chars) -------------
    if not _is_valid_sha256(manifest_sha256):
        errors.append(
            f"provider_manifest_sha256 must be a valid 64-char hex digest "
            f"(observed '{manifest_sha256}')"
        )

    # --- Assemble result ----------------------------------------------------
    if errors:
        status = "FAIL"
        reason = "; ".join(errors)
    else:
        status = "PASS"
        reason = "provider is a valid real model"

    return ProviderValidationResult(
        provider_class=str(provider_class),
        model_id=str(model_id),
        model_revision=str(model_revision),
        model_digest=str(model_digest) if model_digest is not None else "",
        tokenizer_id=str(tokenizer_id),
        tokenizer_revision=str(tokenizer_revision) if tokenizer_revision is not None else "",
        generation_config_digest=str(generation_config_digest),
        provider_manifest_sha256=manifest_sha256,
        supports_gate_a=bool(supports_gate_a_claim),
        supports_gate_b=bool(supports_gate_b_claim),
        runtime_type=str(runtime_type),
        status=status,
        reason=reason,
    )
