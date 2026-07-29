"""AutoLearnService: runtime integration with fail-safe fallback (v0.2).

The ``AutoLearnService`` owns the v0.2 ``ExecutiveController`` (defined in
``controller.py``). It loads the active policy from the ``PolicyRegistry``
at startup and exposes the controller to ``ConversationService`` / the
runtime. The controller selects an action, dispatches it through real
Capsule Brain services, runs the verifier, and persists the
``ExecutiveExperience``.

For backward compatibility, the v0.1 pure ``ExecutiveController`` /
``ExecutiveDecision`` symbols are re-exported from ``controller.py``. New
code should use ``controller.ExecutiveController`` (the real dispatching
controller) and ``controller.ExecutiveDecision``.

Fail-safe contract: if the learned policy fails to load, has an incompatible
feature schema, produces an invalid action, or throws any exception, the
controller falls back to the baseline policy. It never fails open. Every
fallback is recorded with ``policy_fallback = true`` and ``policy_error``.
"""
from __future__ import annotations

import logging
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicy, BaselinePolicyV2
from capsule_brain.autolearn.controller import (
    ActionDispatcher,
    ActionResult,
    ExecutiveController,
    ExecutiveDecision,
    ExperienceSink,
    VerifierFn,
    _is_valid_action,
)
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

# Re-export for backward compatibility (v0.1 callers imported these from
# service.py).
__all__ = [
    "AutoLearnService",
    "ExecutiveController",
    "ExecutiveDecision",
]


class AutoLearnService(CapsuleService):
    """Capsule Brain service that owns the ExecutiveController.

    This is the runtime integration point. It loads the active policy from
    the PolicyRegistry at startup and exposes the controller to
    ConversationService / the runtime. The controller dispatches actions
    through real Capsule Brain services when a dispatcher is attached.
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
        # v0.2 defaults to the competent BaselinePolicyV2. Callers can
        # override with the v1 baseline for backward compatibility.
        self.baseline = baseline or BaselinePolicyV2()
        self.extractor = extractor or FeatureExtractor()
        self.confidence_threshold = float(self.cfg.get("confidence_threshold", 0.5))
        self.policy_root = self.cfg.get("policy_root", "data/autolearn/policies")
        self.shadow_mode = bool(self.cfg.get("shadow_mode", False))
        self.shift_guard = None
        self._build_shift_guard()
        self.controller = ExecutiveController(
            baseline=self.baseline,
            learned=None,
            extractor=self.extractor,
            confidence_threshold=self.confidence_threshold,
            shift_guard=self.shift_guard,
            shadow_mode=self.shadow_mode,
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

    # ------------------------------------------------------------------
    # Runtime integration points
    # ------------------------------------------------------------------

    def attach_dispatcher(self, dispatcher: ActionDispatcher) -> None:
        """Attach the real action dispatcher (provided by ConversationService)."""
        self.controller.set_dispatcher(dispatcher)

    def attach_verifier(self, verifier: VerifierFn) -> None:
        """Attach the independent verifier."""
        self.controller.set_verifier(verifier)

    def attach_experience_sink(self, sink: ExperienceSink) -> None:
        """Attach the ExecutiveExperience persistence sink."""
        self.controller.set_experience_sink(sink)

    def select_action(
        self,
        state: ExecutiveState,
        *,
        allowed_actions: list[Action] | None = None,
    ) -> ExecutiveDecision:
        return self.controller.select_action(state, allowed_actions=allowed_actions)

    async def execute(
        self,
        state: ExecutiveState,
        *,
        conversation_id: str | None = None,
        allowed_actions: list[Action] | None = None,
        task_id: str = "",
        task_family: str = "production",
    ) -> ActionResult:
        """Execute a request through the real dispatch path."""
        return await self.controller.execute(
            state,
            conversation_id=conversation_id,
            allowed_actions=allowed_actions,
            task_id=task_id,
            task_family=task_family,
        )

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
                "experiences_persisted": self.controller._experiences_persisted,
                "policy_type": (
                    "learned" if self.controller.has_learned_policy else "baseline"
                ),
                "shadow_mode": self.shadow_mode,
            },
        )
