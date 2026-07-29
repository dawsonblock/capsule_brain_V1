"""Transparent, configurable, versioned utility function.

U = w_success * verified_success
  - w_latency * normalized_latency
  - w_tokens  * normalized_tokens
  - w_failures * failure_count
  - w_intervention * operator_intervention
  - w_safety * safety_violation

Rules enforced by the default weights:
- verified correctness dominates cost (a cheap wrong answer never beats a
  correct expensive answer)
- safety violations receive a large negative utility
- unverified output (verification_status != "pass") does not receive the
  same reward as verified success — verified_success is only True when the
  verifier actually passed
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from capsule_brain.autolearn.schema import Outcome

UTILITY_CONFIG_VERSION = "utility_v1"


@dataclass(slots=True)
class UtilityConfig:
    """Versioned, configurable utility weights and normalization constants.

    Weights are intentionally exposed and not hidden. Changing them is a
    policy decision that must be recorded alongside the policy manifest.
    """

    w_success: float = 10.0
    w_latency: float = 0.05
    w_tokens: float = 0.0002
    w_failures: float = 1.0
    w_intervention: float = 3.0
    w_safety: float = 50.0
    # Normalization denominators. latency_normalize_ms maps raw ms to [0,1]
    # via latency_ms / latency_normalize_ms (clipped). token_normalize_count
    # does the same for token_count.
    latency_normalize_ms: float = 5000.0
    token_normalize_count: float = 4000.0
    config_version: str = UTILITY_CONFIG_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "w_success": self.w_success,
            "w_latency": self.w_latency,
            "w_tokens": self.w_tokens,
            "w_failures": self.w_failures,
            "w_intervention": self.w_intervention,
            "w_safety": self.w_safety,
            "latency_normalize_ms": self.latency_normalize_ms,
            "token_normalize_count": self.token_normalize_count,
            "config_version": self.config_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "UtilityConfig":
        data = data or {}
        return cls(
            w_success=float(data.get("w_success", 10.0)),
            w_latency=float(data.get("w_latency", 0.05)),
            w_tokens=float(data.get("w_tokens", 0.0002)),
            w_failures=float(data.get("w_failures", 1.0)),
            w_intervention=float(data.get("w_intervention", 3.0)),
            w_safety=float(data.get("w_safety", 50.0)),
            latency_normalize_ms=float(data.get("latency_normalize_ms", 5000.0)),
            token_normalize_count=float(data.get("token_normalize_count", 4000.0)),
            config_version=str(data.get("config_version", UTILITY_CONFIG_VERSION)),
        )


class UtilityFunction:
    """Computes scalar utility plus the full component breakdown."""

    def __init__(self, config: UtilityConfig | None = None) -> None:
        self.config = config or UtilityConfig()

    def _normalized_latency(self, latency_ms: float) -> float:
        if self.config.latency_normalize_ms <= 0:
            return 0.0
        return max(0.0, min(1.0, latency_ms / self.config.latency_normalize_ms))

    def _normalized_tokens(self, token_count: int) -> float:
        if self.config.token_normalize_count <= 0:
            return 0.0
        return max(0.0, min(1.0, token_count / self.config.token_normalize_count))

    def compute(self, outcome: Outcome) -> tuple[float, dict[str, float]]:
        """Return (utility, components) for a single outcome.

        verified_success is True only when the verifier passed. An unverified
        or failed outcome contributes zero positive reward but still incurs
        cost penalties, so a cheap wrong answer can never beat a correct
        expensive one.
        """
        success_term = self.config.w_success if outcome.verified_success else 0.0
        latency_term = self.config.w_latency * self._normalized_latency(
            outcome.latency_ms
        )
        token_term = self.config.w_tokens * self._normalized_tokens(
            outcome.token_count
        )
        failure_term = self.config.w_failures * float(outcome.tool_failures)
        intervention_term = (
            self.config.w_intervention if outcome.operator_intervention else 0.0
        )
        safety_term = self.config.w_safety if outcome.safety_violation else 0.0

        utility = (
            success_term
            - latency_term
            - token_term
            - failure_term
            - intervention_term
            - safety_term
        )
        components = {
            "success_term": success_term,
            "latency_penalty": latency_term,
            "token_penalty": token_term,
            "failure_penalty": failure_term,
            "intervention_penalty": intervention_term,
            "safety_penalty": safety_term,
            "verified_success": 1.0 if outcome.verified_success else 0.0,
            "normalized_latency": self._normalized_latency(outcome.latency_ms),
            "normalized_tokens": self._normalized_tokens(outcome.token_count),
        }
        return utility, components
