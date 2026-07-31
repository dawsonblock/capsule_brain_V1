"""Provider classification for honest qualification (Section 4).

Every artifact must record provider_name, provider_class, model_id,
model_revision, model_digest, tokenizer_id, tokenizer_revision, dtype,
device, generation_config, supports_hidden_states, supports_gate_a_claim,
supports_gate_b_claim.

Promotion must require provider_class == REAL_MODEL.
Infrastructure runs may qualify integration only.
Reports must never describe GroundedQualificationProvider as a real LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProviderClass(str, Enum):
    INFRASTRUCTURE = "infrastructure"
    REAL_MODEL = "real_model"
    SIMULATED = "simulated"


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    provider_name: str
    provider_class: ProviderClass
    model_id: str
    model_revision: str | None = None
    model_digest: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "float32"
    device: str = "cpu"
    generation_config: dict[str, Any] = field(default_factory=dict)
    supports_hidden_states: bool = False
    supports_gate_a_claim: bool = False
    supports_gate_b_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "provider_class": self.provider_class.value,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_digest": self.model_digest,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": self.dtype,
            "device": self.device,
            "generation_config": dict(self.generation_config),
            "supports_hidden_states": self.supports_hidden_states,
            "supports_gate_a_claim": self.supports_gate_a_claim,
            "supports_gate_b_claim": self.supports_gate_b_claim,
        }


def classify_grounded_provider() -> ProviderCapabilities:
    return ProviderCapabilities(
        provider_name="qual_grounded",
        provider_class=ProviderClass.INFRASTRUCTURE,
        model_id="qual-grounded-v04",
        supports_hidden_states=False,
        supports_gate_a_claim=False,
        supports_gate_b_claim=False,
    )


def classify_local_transformers(
    model_id: str,
    *,
    model_revision: str | None = None,
    model_digest: str | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    dtype: str = "float32",
    device: str = "cpu",
    generation_config: dict | None = None,
    supports_hidden_states: bool = True,
) -> ProviderCapabilities:
    return ProviderCapabilities(
        provider_name="local_transformers",
        provider_class=ProviderClass.REAL_MODEL,
        model_id=model_id,
        model_revision=model_revision,
        model_digest=model_digest,
        tokenizer_id=tokenizer_id or model_id,
        tokenizer_revision=tokenizer_revision,
        dtype=dtype,
        device=device,
        generation_config=generation_config or {},
        supports_hidden_states=supports_hidden_states,
        supports_gate_a_claim=True,
        supports_gate_b_claim=supports_hidden_states,
    )


def classify_modal_gpu(
    model_id: str,
    *,
    model_revision: str | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    dtype: str = "float16",
    gpu_type: str = "a10g",
    generation_config: dict | None = None,
    supports_hidden_states: bool = True,
) -> ProviderCapabilities:
    """Classify the Modal GPU provider (real frozen model on remote GPU)."""
    return ProviderCapabilities(
        provider_name="modal_gpu",
        provider_class=ProviderClass.REAL_MODEL,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id or model_id,
        tokenizer_revision=tokenizer_revision,
        dtype=dtype,
        device="cuda",
        generation_config=generation_config or {},
        supports_hidden_states=supports_hidden_states,
        supports_gate_a_claim=True,
        supports_gate_b_claim=supports_hidden_states,
    )


def classify_simulated() -> ProviderCapabilities:
    return ProviderCapabilities(
        provider_name="simulated",
        provider_class=ProviderClass.SIMULATED,
        model_id="simulated",
        supports_hidden_states=False,
        supports_gate_a_claim=False,
        supports_gate_b_claim=False,
    )


class ProviderEligibilityError(RuntimeError):
    pass


def assert_provider_eligible_for_gate(caps: ProviderCapabilities, gate: str) -> None:
    if gate == "A" and not caps.supports_gate_a_claim:
        raise ProviderEligibilityError(
            f"Provider {caps.provider_name} ({caps.provider_class.value}) "
            f"does not support Gate A claims. "
            f"Gate A requires provider_class == REAL_MODEL."
        )
    if gate == "B" and not caps.supports_gate_b_claim:
        raise ProviderEligibilityError(
            f"Provider {caps.provider_name} ({caps.provider_class.value}) "
            f"does not support Gate B claims. "
            f"Gate B requires a real model with hidden states."
        )


def is_real_model(caps: ProviderCapabilities) -> bool:
    return caps.provider_class is ProviderClass.REAL_MODEL


def is_infrastructure(caps: ProviderCapabilities) -> bool:
    return caps.provider_class is ProviderClass.INFRASTRUCTURE
