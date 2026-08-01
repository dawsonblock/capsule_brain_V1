"""Promotion manifest for v0.4.7.

A ``PromotionManifest`` is the cryptographically bound record that a
candidate policy has earned promotion to shadow (or active) deployment.
It binds together every piece of evidence that justified the decision:

* the policy artifact itself (``policy_digest``)
* the source tree, benchmark, training data, and evaluation digests
* the digests of every gate result (A0–A3) and the safety audit

Because the ``promotion_id`` is the SHA-256 of all bound digests,
altering *any* input — swapping the policy, editing a gate result, or
changing the benchmark — invalidates the manifest.  ``verify_promotion_manifest``
performs this binding check independently so a reviewer does not have
to trust the manifest's own self-hash.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PromotionManifest:
    """Cryptographically bound promotion record."""

    promotion_id: str
    policy_id: str
    policy_digest: str
    run_id: str
    source_digest: str
    benchmark_digest: str
    training_data_digest: str
    evaluation_digest: str
    gate_a0_digest: str
    gate_a1_digest: str
    gate_a2_digest: str
    gate_a3_digest: str
    safety_digest: str
    protocol_version: str = "0.4.7"
    promotion_mode: str = "SHADOW"
    eligible: bool = False
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(), indent=indent, ensure_ascii=False, default=str
        )

    def compute_promotion_id(self) -> str:
        """SHA-256 of all bound digests.

        The promotion ID is recomputed from the bound evidence so that
        any alteration of the inputs produces a different ID.  The
        ``promotion_id`` field itself is excluded from the hash.
        """
        payload = json.dumps(
            {
                "policy_id": self.policy_id,
                "policy_digest": self.policy_digest,
                "run_id": self.run_id,
                "source_digest": self.source_digest,
                "benchmark_digest": self.benchmark_digest,
                "training_data_digest": self.training_data_digest,
                "evaluation_digest": self.evaluation_digest,
                "gate_a0_digest": self.gate_a0_digest,
                "gate_a1_digest": self.gate_a1_digest,
                "gate_a2_digest": self.gate_a2_digest,
                "gate_a3_digest": self.gate_a3_digest,
                "safety_digest": self.safety_digest,
                "protocol_version": self.protocol_version,
                "promotion_mode": self.promotion_mode,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_promotion_manifest(
    manifest: dict,
    policy_artifact: dict,
    gate_results: dict,
) -> dict:
    """Verify that a promotion manifest is cryptographically bound to its evidence.

    Args:
        manifest: The promotion manifest dict (as produced by
            ``PromotionManifest.to_dict``).
        policy_artifact: The policy artifact dict.  Must contain a
            ``sha256`` field (or ``policy_digest``) with the digest of
            the policy file.
        gate_results: Dict mapping gate name to its result dict.  Each
            result dict must contain a ``sha256`` field with the digest
            of the gate result file.  Expected keys: ``gate_a0``,
            ``gate_a1``, ``gate_a2``, ``gate_a3``, ``safety``.

    Returns:
        Dict with ``valid: bool`` and ``mismatches: list[str]``.

    An altered policy artifact or gate result file must invalidate the
    promotion record.
    """
    mismatches: list[str] = []

    # --- Policy digest -------------------------------------------------
    expected_policy_digest = manifest.get("policy_digest", "")
    actual_policy_digest = (
        policy_artifact.get("sha256")
        or policy_artifact.get("policy_digest")
        or ""
    )
    if not actual_policy_digest:
        mismatches.append("policy artifact is missing a sha256 digest")
    elif actual_policy_digest != expected_policy_digest:
        mismatches.append(
            f"policy_digest mismatch: manifest={expected_policy_digest!r} "
            f"artifact={actual_policy_digest!r}"
        )

    # --- Gate digests --------------------------------------------------
    gate_fields = {
        "gate_a0": "gate_a0_digest",
        "gate_a1": "gate_a1_digest",
        "gate_a2": "gate_a2_digest",
        "gate_a3": "gate_a3_digest",
        "safety": "safety_digest",
    }
    for gate_name, manifest_field in gate_fields.items():
        expected = manifest.get(manifest_field, "")
        result = gate_results.get(gate_name)
        if result is None:
            mismatches.append(f"missing gate result for {gate_name!r}")
            continue
        actual = result.get("sha256") or result.get("digest") or ""
        if not actual:
            mismatches.append(
                f"gate result {gate_name!r} is missing a sha256 digest"
            )
            continue
        if actual != expected:
            mismatches.append(
                f"{manifest_field} mismatch: manifest={expected!r} "
                f"gate_result={actual!r}"
            )

    # --- Promotion ID self-consistency ---------------------------------
    promotion_id = manifest.get("promotion_id", "")
    if promotion_id:
        rebuilt = PromotionManifest(
            promotion_id="",
            policy_id=manifest.get("policy_id", ""),
            policy_digest=manifest.get("policy_digest", ""),
            run_id=manifest.get("run_id", ""),
            source_digest=manifest.get("source_digest", ""),
            benchmark_digest=manifest.get("benchmark_digest", ""),
            training_data_digest=manifest.get("training_data_digest", ""),
            evaluation_digest=manifest.get("evaluation_digest", ""),
            gate_a0_digest=manifest.get("gate_a0_digest", ""),
            gate_a1_digest=manifest.get("gate_a1_digest", ""),
            gate_a2_digest=manifest.get("gate_a2_digest", ""),
            gate_a3_digest=manifest.get("gate_a3_digest", ""),
            safety_digest=manifest.get("safety_digest", ""),
            protocol_version=manifest.get("protocol_version", "0.4.7"),
            promotion_mode=manifest.get("promotion_mode", "SHADOW"),
        ).compute_promotion_id()
        if rebuilt != promotion_id:
            mismatches.append(
                f"promotion_id mismatch: manifest={promotion_id!r} "
                f"recomputed={rebuilt!r}"
            )

    return {
        "valid": len(mismatches) == 0,
        "mismatches": mismatches,
    }
