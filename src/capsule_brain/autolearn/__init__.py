"""Capsule Brain AutoLearn v0.2.

A learned executive/router policy layer that sits on top of the frozen
Capsule Brain v2.15.0 execution substrate. AutoLearn chooses which action
the runtime should take for a given task; the ExecutiveController dispatches
that action through the *real* Capsule Brain services; the
VerificationService judges the outcome; the resulting ExecutiveExperience is
persisted. Promoting a new policy requires passing a statistical promotion
gate on held-out, independently verified tasks whose counterfactual outcomes
come from real service execution, not a simulator.

AutoLearn never modifies the safety envelope and never mutates production
behavior live. Candidate policies are trained offline from real
counterfactual action outcomes, evaluated on held-out tasks, run in shadow
mode, and only then promoted as immutable versioned artifacts.
"""
from capsule_brain.autolearn.schema import (
    Action,
    ActionMetadata,
    ExecutiveExperience,
    ExecutiveState,
    Outcome,
    Provenance,
    SCHEMA_VERSION,
)
from capsule_brain.autolearn.features import (
    FEATURE_SCHEMA_VERSION,
    FeatureExtractor,
)
from capsule_brain.autolearn.utility import (
    UtilityConfig,
    UtilityFunction,
)
from capsule_brain.autolearn.baseline import BaselinePolicy, BaselinePolicyV2
from capsule_brain.autolearn.policy import (
    LearnedPolicy,
    PolicyDecision,
    PolicyLoadError,
)
from capsule_brain.autolearn.registry import PolicyManifest, PolicyRegistry
from capsule_brain.autolearn.service import AutoLearnService, ExecutiveController
from capsule_brain.autolearn.controller import (
    ActionDispatcher,
    ActionResult,
    DispatcherResult,
    ExecutiveDecision as ControllerDecision,
    ExperienceSink,
    ExperienceStoreSink,
    JSONLExperienceSink,
    LEARNED_ACTIONS,
    MemoryExperienceSink,
    VerifierFn,
)

__all__ = [
    "Action",
    "ActionDispatcher",
    "ActionMetadata",
    "AutoLearnService",
    "BaselinePolicy",
    "BaselinePolicyV2",
    "ControllerDecision",
    "DispatcherResult",
    "ExecutiveController",
    "ExecutiveExperience",
    "ExecutiveState",
    "ExperienceSink",
    "ExperienceStoreSink",
    "FEATURE_SCHEMA_VERSION",
    "FeatureExtractor",
    "JSONLExperienceSink",
    "LEARNED_ACTIONS",
    "LearnedPolicy",
    "MemoryExperienceSink",
    "Outcome",
    "PolicyDecision",
    "PolicyLoadError",
    "PolicyManifest",
    "PolicyRegistry",
    "Provenance",
    "ActionResult",
    "SCHEMA_VERSION",
    "UtilityConfig",
    "UtilityFunction",
    "VerifierFn",
]
