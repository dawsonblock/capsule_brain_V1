"""Capsule Brain AutoLearn v0.1.

A learned executive/router policy layer that sits on top of the frozen
Capsule Brain v2.14.0 execution substrate. AutoLearn chooses which action
the runtime should take for a given task; Capsule Brain executes the action;
VerificationService judges the outcome. Promoting a new policy requires
passing a statistical promotion gate on held-out, independently verified
tasks.

AutoLearn never modifies the safety envelope and never mutates production
behavior live. Candidate policies are trained offline from counterfactual
action outcomes, evaluated on held-out tasks, run in shadow mode, and only
then promoted as immutable versioned artifacts.
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
from capsule_brain.autolearn.baseline import BaselinePolicy
from capsule_brain.autolearn.policy import (
    LearnedPolicy,
    PolicyDecision,
    PolicyLoadError,
)
from capsule_brain.autolearn.registry import PolicyManifest, PolicyRegistry
from capsule_brain.autolearn.service import AutoLearnService, ExecutiveController

__all__ = [
    "Action",
    "ActionMetadata",
    "AutoLearnService",
    "BaselinePolicy",
    "ExecutiveController",
    "ExecutiveExperience",
    "ExecutiveState",
    "FEATURE_SCHEMA_VERSION",
    "FeatureExtractor",
    "LearnedPolicy",
    "Outcome",
    "PolicyDecision",
    "PolicyLoadError",
    "PolicyManifest",
    "PolicyRegistry",
    "Provenance",
    "SCHEMA_VERSION",
    "UtilityConfig",
    "UtilityFunction",
]
