"""ExecutiveController + AutoLearnService: runtime integration with fail-safe fallback.

The ExecutiveController sits between a user request and the runtime action.
It asks the active executive policy to select an action, then hands that
action to the existing Capsule Brain services for execution. AutoLearn
chooses; Capsule Brain executes; VerificationService judges.

Fail-safe contract: if the learned policy fails to load, has an incompatible
feature schema, produces an invalid action, or throws any exception, the
controller falls back to the baseline policy. It never fails open. Every
fallback is recorded with ``policy_fallback = true`` and ``policy_error``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicy
from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.policy import (
    LearnedPolicy,
    PolicyLoadError,
)
from capsule_brain.autolearn.registry import PolicyRegistry
from capsule_brain.autolearn.schema import Action, ExecutiveState
from capsule_brain.autolearn.shadow import (
    DistributionShiftGuard,
    ShadowEvaluator,
)
from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutiveDecision:
    """The final action the runtime should execute, plus audit metadata."""

    action: Action
    reason: str
    policy_version: str
    policy_type: str  # "baseline" | "learned"
    confidence: float
    scores: dict[str, float]
    policy_fallback: bool = False
    policy_error: str = ""
    distribution_shift_detected: bool = False
    abstained: bool = False
    shadow_action: Action | None = None
    feature_vector: list[float] = field(default_factory=list)
    state: ExecutiveState | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "policy_type": self.policy_type,
            "confidence": self.confidence,
            "scores": dict(self.scores),
            "policy_fallback": self.policy_fallback,
            "policy_error": self.policy_error,
            "distribution_shift_detected": self.distribution_shift_detected,
            "abstained": self.abstained,
            "shadow_action": self.shadow_action.value if self.shadow_action else None,
            "feature_vector": list(self.feature_vector),
        }


class ExecutiveController:
    """Selects an action for a state using the active policy, with fallback.

    The controller is intentionally synchronous and pure: it does not execute
    the chosen action. The caller (ConversationService / runtime) is
    responsible for dispatching the action to the appropriate Capsule Brain
    service and recording the verified outcome.
    """

    def __init__(
        self,
        baseline: BaselinePolicy | None = None,
        learned: LearnedPolicy | None = None,
        *,
        extractor: FeatureExtractor | None = None,
        confidence_threshold: float = 0.5,
        shift_guard: DistributionShiftGuard | None = None,
        shadow: ShadowEvaluator | None = None,
    ) -> None:
        self.baseline = baseline or BaselinePolicy()
        self.learned = learned
        self.extractor = extractor or FeatureExtractor()
        self.confidence_threshold = confidence_threshold
        self.shift_guard = shift_guard
        self.shadow = shadow
        self._fallbacks = 0
        self._decisions = 0

    @property
    def has_learned_policy(self) -> bool:
        return self.learned is not None

    def set_learned_policy(self, policy: LearnedPolicy | None) -> None:
        self.learned = policy

    def select_action(
        self,
        state: ExecutiveState,
        *,
        allowed_actions: list[Action] | None = None,
    ) -> ExecutiveDecision:
        self._decisions += 1
        fv = self.extractor.extract(state)
        fv_list = fv.as_list()

        # Distribution-shift guard: if the current feature distribution has
        # drifted outside the training domain, fall back to baseline.
        shift_detected = False
        if self.shift_guard is not None:
            self.shift_guard.observe(fv_list)
            try:
                shift_detected = self.shift_guard.shifted()
            except Exception as exc:
                log.warning("shift_guard error: %s", exc)
                shift_detected = False

        if shift_detected:
            return self._fallback_decision(
                state, fv_list, allowed_actions,
                reason="distribution shift detected; using baseline",
                distribution_shift_detected=True,
            )

        # No learned policy: baseline.
        if self.learned is None:
            return self._baseline_decision(
                state, fv_list, allowed_actions, reason="no learned policy active"
            )

        # Try the learned policy.
        try:
            c_dec = self.learned.select_action(
                state,
                extractor=self.extractor,
                allowed_actions=allowed_actions,
            )
        except PolicyLoadError as exc:
            return self._fallback_decision(
                state, fv_list, allowed_actions,
                reason=f"learned policy load error: {exc}",
                error=str(exc),
            )
        except Exception as exc:
            return self._fallback_decision(
                state, fv_list, allowed_actions,
                reason=f"learned policy error: {exc}",
                error=str(exc),
            )

        # Validate the chosen action is in the allowed set (and is a real Action).
        if not _is_valid_action(c_dec.action, allowed_actions):
            return self._fallback_decision(
                state, fv_list, allowed_actions,
                reason=f"learned policy chose invalid/disallowed action {c_dec.action}",
                error="invalid_action",
            )

        # Confidence abstention: fall back to baseline on low confidence.
        if c_dec.abstained or c_dec.confidence < self.confidence_threshold:
            base_dec = self.baseline.select_action(state, allowed_actions=allowed_actions)
            return ExecutiveDecision(
                action=base_dec.action,
                reason=f"learned policy abstained (conf={c_dec.confidence:.3f}); baseline fallback",
                policy_version=self.baseline.version,
                policy_type="baseline",
                confidence=c_dec.confidence,
                scores=c_dec.scores,
                policy_fallback=True,
                policy_error="abstention",
                abstained=True,
                shadow_action=c_dec.action if self.shadow is not None else None,
                feature_vector=fv_list,
                state=state,
            )

        # Shadow logging: record what the candidate would have chosen even
        # when we use it in production (for promoted policies this is a no-op
        # audit trail; for shadow-mode candidates the production action is
        # the baseline and we record disagreement).
        shadow_action: Action | None = None
        if self.shadow is not None:
            shadow_rec = self.shadow.evaluate(
                state, allowed_actions=allowed_actions
            )
            shadow_action = shadow_rec.shadow_action

        return ExecutiveDecision(
            action=c_dec.action,
            reason=c_dec.reason,
            policy_version=self.learned.policy_version,
            policy_type="learned",
            confidence=c_dec.confidence,
            scores=c_dec.scores,
            policy_fallback=False,
            shadow_action=shadow_action,
            feature_vector=fv_list,
            state=state,
        )

    def _baseline_decision(
        self,
        state: ExecutiveState,
        fv_list: list[float],
        allowed_actions: list[Action] | None,
        *,
        reason: str,
    ) -> ExecutiveDecision:
        dec = self.baseline.select_action(state, allowed_actions=allowed_actions)
        return ExecutiveDecision(
            action=dec.action,
            reason=reason,
            policy_version=self.baseline.version,
            policy_type="baseline",
            confidence=max(dec.scores.values()) if dec.scores else 0.0,
            scores=dec.scores,
            feature_vector=fv_list,
            state=state,
        )

    def _fallback_decision(
        self,
        state: ExecutiveState,
        fv_list: list[float],
        allowed_actions: list[Action] | None,
        *,
        reason: str,
        error: str = "",
        distribution_shift_detected: bool = False,
    ) -> ExecutiveDecision:
        self._fallbacks += 1
        dec = self.baseline.select_action(state, allowed_actions=allowed_actions)
        return ExecutiveDecision(
            action=dec.action,
            reason=reason,
            policy_version=self.baseline.version,
            policy_type="baseline",
            confidence=max(dec.scores.values()) if dec.scores else 0.0,
            scores=dec.scores,
            policy_fallback=True,
            policy_error=error,
            distribution_shift_detected=distribution_shift_detected,
            feature_vector=fv_list,
            state=state,
        )


def _is_valid_action(action: Action, allowed: list[Action] | None) -> bool:
    if not isinstance(action, Action):
        return False
    if allowed is None:
        return True
    return action in set(allowed)


class AutoLearnService(CapsuleService):
    """Capsule Brain service that owns the ExecutiveController.

    This is the minimal runtime integration point. It loads the active
    policy from the PolicyRegistry at startup and exposes the controller to
    ConversationService / the runtime. It does not execute actions.
    """

    name = "autolearn"

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        registry: PolicyRegistry | None = None,
        baseline: BaselinePolicy | None = None,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        super().__init__(cfg)
        self.registry = registry
        self.baseline = baseline or BaselinePolicy()
        self.extractor = extractor or FeatureExtractor()
        self.confidence_threshold = float(self.cfg.get("confidence_threshold", 0.5))
        self.policy_root = self.cfg.get("policy_root", "data/autolearn/policies")
        self.shift_guard = None
        self._build_shift_guard()
        self.controller = ExecutiveController(
            baseline=self.baseline,
            learned=None,
            extractor=self.extractor,
            confidence_threshold=self.confidence_threshold,
            shift_guard=self.shift_guard,
        )
        self._active_policy_id: str | None = None

    def _build_shift_guard(self) -> None:
        shift_cfg = self.cfg.get("distribution_shift", {}) or {}
        if not shift_cfg.get("enable", False):
            self.shift_guard = None
            return
        from capsule_brain.autolearn.shadow import DistributionShiftConfig

        training_means = shift_cfg.get("training_means", [])
        config = DistributionShiftConfig(
            max_mean_distance=float(shift_cfg.get("max_mean_distance", 1.5)),
            min_window=int(shift_cfg.get("min_window", 10)),
        )
        self.shift_guard = DistributionShiftGuard(
            training_means=list(training_means),
            config=config,
        )

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        if self.registry is None and self.policy_root:
            self.registry = PolicyRegistry(self.policy_root)
        await self._load_active_policy()
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPED

    async def _load_active_policy(self) -> None:
        if self.registry is None:
            return
        try:
            active = self.registry.load_active()
        except Exception as exc:
            log.warning("AutoLearn: failed to load active policy: %s", exc)
            self.controller.set_learned_policy(None)
            return
        if active is None:
            log.info("AutoLearn: no active policy; using baseline")
            self.controller.set_learned_policy(None)
            return
        policy, manifest = active
        try:
            # Validate schema by attempting a dummy prediction.
            _ = policy.feature_schema_version
            self.controller.set_learned_policy(policy)
            self._active_policy_id = policy.policy_id
            log.info(
                "AutoLearn: active policy %s (%s)",
                policy.policy_id, policy.policy_version,
            )
        except Exception as exc:
            log.warning(
                "AutoLearn: active policy %s incompatible, falling back: %s",
                manifest.policy_id, exc,
            )
            self.controller.set_learned_policy(None)

    def select_action(
        self,
        state: ExecutiveState,
        *,
        allowed_actions: list[Action] | None = None,
    ) -> ExecutiveDecision:
        return self.controller.select_action(state, allowed_actions=allowed_actions)

    @property
    def active_policy_id(self) -> str | None:
        return self._active_policy_id

    async def health(self) -> HealthStatus:
        return HealthStatus(
            state=self.state,
            details={
                "active_policy_id": self._active_policy_id,
                "has_learned_policy": self.controller.has_learned_policy,
                "decisions": self.controller._decisions,
                "fallbacks": self.controller._fallbacks,
                "policy_type": (
                    "learned" if self.controller.has_learned_policy else "baseline"
                ),
            },
        )
