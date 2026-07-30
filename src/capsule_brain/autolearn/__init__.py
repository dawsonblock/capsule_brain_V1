"""Capsule Brain AutoLearn v0.3.

A learned executive/router policy layer that sits on top of the frozen
Capsule Brain v2.16.0 execution substrate. AutoLearn chooses which action
the runtime should take for a given task; the ExecutiveController dispatches
that action through the *real* Capsule Brain services via the public
``ConversationService.execute_executive_action()`` API; the
VerificationService judges the outcome; the resulting ExecutiveExperience is
persisted. Promoting a new policy requires passing a statistical promotion
gate on held-out, independently verified tasks whose counterfactual outcomes
come from real service execution, not a simulator.

v0.3 key changes:
- Single public action API (``execute_executive_action``) used by both
  production and counterfactual qualification.
- Canonical ``counterfactual.py`` with Protocol + Real + Simulated runtimes.
- Qualification refuses to run with the simulated runtime.
- REFLECT and ASK_OPERATOR remain outside the learned action space.
- 80-task benchmark with archetype-based splits and genuine OOD.
- 12-gate promotion gate (adds coverage gate).

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
from capsule_brain.autolearn.counterfactual import (
    CounterfactualResult,
    CounterfactualRuntime,
    CounterfactualTask,
    LEARNED_ACTIONS_V03,
    RealCounterfactualRuntime,
    SimulatedCounterfactualRuntime,
    assert_runtime_is_real,
)
from capsule_brain.autolearn.qualification_runtime import (
    QualificationProvider,
    QualificationServiceFactory,
)
from capsule_brain.autolearn.statistics import (
    BootstrapResult,
    QualificationStatistics,
    SignTestResult,
    PermutationTestResult,
    compute_qualification_statistics,
    paired_bootstrap,
    paired_permutation_test,
    sign_test,
)
from capsule_brain.autolearn.sham_learner import (
    train_sham_policy,
    shuffle_actions_within_strata,
    shuffle_labels_globally,
    shuffle_utilities,
)
from capsule_brain.autolearn.source_hash import (
    compute_source_tree_hash,
    compute_file_hash,
    compute_string_hash,
    verify_provenance,
)

__all__ = [
    "Action",
    "ActionDispatcher",
    "ActionMetadata",
    "AutoLearnService",
    "BaselinePolicy",
    "BaselinePolicyV2",
    "BootstrapResult",
    "ControllerDecision",
    "CounterfactualResult",
    "CounterfactualRuntime",
    "CounterfactualTask",
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
    "LEARNED_ACTIONS_V03",
    "LearnedPolicy",
    "MemoryExperienceSink",
    "Outcome",
    "PermutationTestResult",
    "PolicyDecision",
    "PolicyLoadError",
    "PolicyManifest",
    "PolicyRegistry",
    "Provenance",
    "QualificationProvider",
    "QualificationServiceFactory",
    "QualificationStatistics",
    "ActionResult",
    "RealCounterfactualRuntime",
    "SCHEMA_VERSION",
    "SignTestResult",
    "SimulatedCounterfactualRuntime",
    "UtilityConfig",
    "UtilityFunction",
    "VerifierFn",
    "assert_runtime_is_real",
    "compute_file_hash",
    "compute_qualification_statistics",
    "compute_source_tree_hash",
    "compute_string_hash",
    "paired_bootstrap",
    "paired_permutation_test",
    "shuffle_actions_within_strata",
    "shuffle_labels_globally",
    "shuffle_utilities",
    "sign_test",
    "train_sham_policy",
    "verify_provenance",
]
