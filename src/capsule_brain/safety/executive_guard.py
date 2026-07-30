"""Immutable executive safety guard.

v0.3.1 / 2.15.2 Section 3: This module implements a safety authority that
runs BEFORE the learned executive policy. The learned policy can never
override:

- ASK_OPERATOR
- SAFE_REFUSAL
- capability denial
- verification requirements
- sandbox restrictions
- network restrictions
- filesystem restrictions

The guard is immutable: the learned policy has no authority to modify it.

Controller order:

    request
      → immutable safety guard
      → authorization/capability guard
      → learned executive policy
      → confidence/shift fallback
      → execution
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from capsule_brain.autolearn.schema import Action, ExecutiveState


class SafetyDisposition(Enum):
    """The disposition of the safety guard for a given request."""

    ALLOW = "allow"
    REQUIRE_OPERATOR = "require_operator"
    REFUSE = "refuse"


@dataclass(frozen=True)
class SafetyDecision:
    """The immutable safety decision for a request.

    This decision is final with respect to the learned policy. The learned
    policy may not override REQUIRE_OPERATOR or REFUSE.
    """

    disposition: SafetyDisposition
    reason_code: str
    evidence: tuple[str, ...]
    required_action: Action | None


# ---------------------------------------------------------------------------
# Safety keyword taxonomy
# ---------------------------------------------------------------------------

# Destructive operations — require operator escalation.
_DESTRUCTIVE_KEYWORDS: tuple[str, ...] = (
    "delete the production", "drop the", "rm -rf", "force-push",
    "overwrite the main branch", "truncate the database",
    "delete all rows", "drop schema", "wipe the",
    "destructive repository operation",
)

# Security bypass — refuse outright.
_SECURITY_BYPASS_KEYWORDS: tuple[str, ...] = (
    "disable the firewall", "disable all safety", "disable container",
    "bypass the verification", "bypass the safety", "disable verification",
    "rewrite the policy", "overwrite the safety", "remove audit logs",
    "unsafe_execution",
)

# Exfiltration / credential harvesting — refuse outright.
_EXFIL_KEYWORDS: tuple[str, ...] = (
    "exfiltrate", "backdoor", "reverse shell", "cryptominer",
    "phishing", "harvest", "scrape all user passwords",
    "ssh keys", "browser cookies", "cryptocurrency wallet",
    "send unauthorized email", "install persistence",
    "execute untrusted network command",
)

# Capability violations — require operator.
_CAPABILITY_KEYWORDS: tuple[str, ...] = (
    "modify active policy directly", "modify the active policy",
    "change the safety rules", "edit the safety guard",
)

# Sandbox / network / filesystem restrictions — require operator.
_RESTRICTION_KEYWORDS: tuple[str, ...] = (
    "bypass sandbox", "bypass the sandbox", "escape the container",
    "disable sandbox restrictions", "network restrictions",
    "filesystem restrictions",
)


class ExecutiveSafetyGuard:
    """Immutable safety authority that runs before the learned policy.

    The guard evaluates the request prompt and state. If the disposition
    is REQUIRE_OPERATOR, the controller must return ASK_OPERATOR. If the
    disposition is REFUSE, the controller must return SAFE_REFUSAL. If
    ALLOW, the learned policy may proceed.

    The guard is immutable: there is no method to update its keyword lists
    or disposition logic at runtime. The learned policy has no reference
    to this object and cannot modify it.
    """

    def __init__(self) -> None:
        # Keyword lists are frozen at construction time.
        self._destructive = _DESTRUCTIVE_KEYWORDS
        self._security_bypass = _SECURITY_BYPASS_KEYWORDS
        self._exfil = _EXFIL_KEYWORDS
        self._capability = _CAPABILITY_KEYWORDS
        self._restriction = _RESTRICTION_KEYWORDS

    def evaluate(
        self,
        state: ExecutiveState,
        request_context: Any | None = None,
    ) -> SafetyDecision:
        """Evaluate the safety disposition of a request.

        Args:
            state: The executive state containing the prompt text.
            request_context: Optional immutable request context (unused
                by the default guard, but available for future extensions).

        Returns:
            A SafetyDecision that the controller must honor.
        """
        prompt_text = state.prompt_features.get("text", "").lower()

        # Check refusal conditions first (most severe).
        for kw in self._security_bypass:
            if kw in prompt_text:
                return SafetyDecision(
                    disposition=SafetyDisposition.REFUSE,
                    reason_code="security_bypass_detected",
                    evidence=(f"keyword: {kw}",),
                    required_action=Action.SAFE_REFUSAL,
                )

        for kw in self._exfil:
            if kw in prompt_text:
                return SafetyDecision(
                    disposition=SafetyDisposition.REFUSE,
                    reason_code="exfiltration_detected",
                    evidence=(f"keyword: {kw}",),
                    required_action=Action.SAFE_REFUSAL,
                )

        # Check operator escalation conditions.
        for kw in self._destructive:
            if kw in prompt_text:
                return SafetyDecision(
                    disposition=SafetyDisposition.REQUIRE_OPERATOR,
                    reason_code="destructive_operation_detected",
                    evidence=(f"keyword: {kw}",),
                    required_action=Action.ASK_OPERATOR,
                )

        for kw in self._capability:
            if kw in prompt_text:
                return SafetyDecision(
                    disposition=SafetyDisposition.REQUIRE_OPERATOR,
                    reason_code="capability_violation_detected",
                    evidence=(f"keyword: {kw}",),
                    required_action=Action.ASK_OPERATOR,
                )

        for kw in self._restriction:
            if kw in prompt_text:
                return SafetyDecision(
                    disposition=SafetyDisposition.REQUIRE_OPERATOR,
                    reason_code="restriction_bypass_detected",
                    evidence=(f"keyword: {kw}",),
                    required_action=Action.ASK_OPERATOR,
                )

        # No safety concern — allow the learned policy to proceed.
        return SafetyDecision(
            disposition=SafetyDisposition.ALLOW,
            reason_code="no_safety_concern",
            evidence=(),
            required_action=None,
        )
