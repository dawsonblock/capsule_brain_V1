"""Safety authority for Capsule Brain.

v0.3.1 / 2.15.2: Immutable safety guard that runs BEFORE the learned
executive policy. The learned policy can never override safety decisions.
"""
from capsule_brain.safety.executive_guard import (
    ExecutiveSafetyGuard,
    SafetyDecision,
    SafetyDisposition,
)

__all__ = [
    "ExecutiveSafetyGuard",
    "SafetyDecision",
    "SafetyDisposition",
]
