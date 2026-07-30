"""Typed verifier reliability taxonomy for experience-quality computation.

Replaces the stringly-typed verifier_type matching in compute_experience_quality
which only recognized "deterministic" and silently assigned 0.0 reliability
to every other verifier name (direct_exact, memory_secret, tool_grounding,
workflow_acceptance, safety_preemption, etc.).

Every verifier used by the qualification pipeline must be registered here.
Unknown verifier names raise UnknownVerifierTypeError unless an explicit
experimental fallback is enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class VerifierClass(Enum):
    """Broad class of verifier reliability."""
    DETERMINISTIC = "deterministic"
    RETRIEVAL_GROUNDED = "retrieval_grounded"
    SPECIALIZED_MODEL = "specialized_model"
    GENERAL_LLM_JUDGE = "general_llm_judge"


class UnknownVerifierTypeError(ValueError):
    """Raised when a verifier name is not in the registry."""
    pass


@dataclass(frozen=True)
class VerifierDescriptor:
    """Describes a verifier's reliability properties."""
    name: str
    verifier_class: VerifierClass
    reliability: float
    independently_grounded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier_name": self.name,
            "verifier_class": self.verifier_class.value,
            "reliability": self.reliability,
            "independently_grounded": self.independently_grounded,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, VerifierDescriptor] = {}


def _register(
    name: str,
    verifier_class: VerifierClass,
    reliability: float,
    independently_grounded: bool = True,
) -> None:
    _REGISTRY[name] = VerifierDescriptor(
        name=name,
        verifier_class=verifier_class,
        reliability=reliability,
        independently_grounded=independently_grounded,
    )


# Deterministic verifiers (reliability = 1.0).
_register("direct_exact", VerifierClass.DETERMINISTIC, 1.0)
_register("memory_secret", VerifierClass.DETERMINISTIC, 1.0)
_register("tool_grounding", VerifierClass.DETERMINISTIC, 1.0)
_register("tool_output", VerifierClass.DETERMINISTIC, 1.0)
_register("workflow_acceptance", VerifierClass.DETERMINISTIC, 1.0)
_register("safety_preemption", VerifierClass.DETERMINISTIC, 1.0)
_register("json_schema", VerifierClass.DETERMINISTIC, 1.0)
_register("symbolic", VerifierClass.DETERMINISTIC, 1.0)
_register("compiler", VerifierClass.DETERMINISTIC, 1.0)
_register("pytest", VerifierClass.DETERMINISTIC, 1.0)
_register("database_constraint", VerifierClass.DETERMINISTIC, 1.0)
_register("deterministic", VerifierClass.DETERMINISTIC, 1.0)

# Retrieval-grounded verifiers (reliability 0.90–0.98).
_register("retrieval_grounded", VerifierClass.RETRIEVAL_GROUNDED, 0.95)
_register("memory_grounded", VerifierClass.RETRIEVAL_GROUNDED, 0.92)

# Specialized model verifiers (configured and calibrated).
_register("specialized_model", VerifierClass.SPECIALIZED_MODEL, 0.80)

# General LLM judge (0.40–0.60).
_register("llm_judge", VerifierClass.GENERAL_LLM_JUDGE, 0.50)
_register("general_llm_judge", VerifierClass.GENERAL_LLM_JUDGE, 0.50)


def get_verifier_descriptor(name: str, *, allow_experimental_fallback: bool = False) -> VerifierDescriptor:
    """Look up a verifier by name.

    Raises UnknownVerifierTypeError if the name is not registered,
    unless allow_experimental_fallback is True (in which case it
    returns a low-reliability descriptor).
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    if allow_experimental_fallback:
        return VerifierDescriptor(
            name=name,
            verifier_class=VerifierClass.GENERAL_LLM_JUDGE,
            reliability=0.30,
            independently_grounded=False,
        )
    raise UnknownVerifierTypeError(
        f"Unknown verifier type: '{name}'. "
        f"Known types: {sorted(_REGISTRY.keys())}"
    )


def get_verifier_reliability(name: str, *, allow_experimental_fallback: bool = False) -> float:
    """Get the reliability score for a verifier name."""
    return get_verifier_descriptor(name, allow_experimental_fallback=allow_experimental_fallback).reliability


def is_known_verifier(name: str) -> bool:
    """Check if a verifier name is registered."""
    return name in _REGISTRY


def all_verifier_names() -> list[str]:
    """Return all registered verifier names."""
    return sorted(_REGISTRY.keys())


def verifier_registry_as_dict() -> dict[str, dict[str, Any]]:
    """Return the full registry as a serializable dict."""
    return {name: desc.to_dict() for name, desc in sorted(_REGISTRY.items())}
