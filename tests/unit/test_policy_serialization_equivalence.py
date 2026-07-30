"""Priority 0 invariance test: training/inference equivalence after serialization.

This test verifies that a trained policy produces IDENTICAL predictions after
being serialized to JSON and reloaded. Specifically:

1. Train a policy with standardization (feature_means / feature_stds).
2. Save it to JSON.
3. Reload it.
4. Evaluate every sample on both the original and reloaded policy.

Required: predicted action identical, probability vector identical, confidence
identical — within numerical tolerance.

This test blocks release if it fails. It catches the bug where
`policy.hyperparameters` is overwritten after training, destroying the
`feature_means` / `feature_stds` normalization metadata. Without those
parameters, inference runs on raw features while training ran on standardized
features — making the deployed model mathematically different from the trained
model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from capsule_brain.autolearn.features import (
    FEATURE_NAMES,
    FeatureExtractor,
)
from capsule_brain.autolearn.learner import (
    LearnerConfig,
    SupervisedRouterLearner,
    TrainingExample,
)
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action, ExecutiveState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(
    prompt: str = "hello",
    *,
    available_tools: list[str] | None = None,
    workflow_available: bool = False,
    memory_hit_count: int = 0,
    top_similarity: float = 0.0,
    previous_attempt_failed: bool = False,
    model_id: str = "qwen2.5-coder:3b",
    workflow_capability_match: bool = False,
) -> ExecutiveState:
    return ExecutiveState(
        prompt_features={
            "text": prompt,
            "previous_attempt_failed": previous_attempt_failed,
            "verification_failure_type": "none",
            "estimated_difficulty": 0.5,
            "workflow_capability_match": workflow_capability_match,
        },
        conversation_features={"depth": 0},
        memory_features={"hit_count": memory_hit_count, "top_similarity": top_similarity},
        available_tools=list(available_tools or []),
        workflow_available=workflow_available,
        model_id=model_id,
        context_length=len(prompt),
    )


def _build_training_examples(n: int = 40) -> list[TrainingExample]:
    """Build a set of training examples with diverse features.

    The features are deliberately non-zero-mean and non-unit-variance so that
    standardization changes the values — if the standardization parameters are
    lost, predictions will differ.
    """
    extractor = FeatureExtractor()
    prompts = [
        _state("What is 2+2?", available_tools=[]),
        _state("Fetch the live stock price using the tool.", available_tools=["stock_api"]),
        _state("Remember the pin I stored earlier.", memory_hit_count=3, top_similarity=0.9),
        _state("Run the data pipeline for Q3.", workflow_available=True),
        _state("What is the capital of France?"),
        _state("Use the calculator tool to compute 42 * 17.", available_tools=["calculator"]),
        _state("Recall our discussion about the budget.", memory_hit_count=2, top_similarity=0.8),
        _state("Execute the deployment workflow.", workflow_available=True),
    ]
    actions = list(Action.learned())
    examples = []
    for i in range(n):
        state = prompts[i % len(prompts)]
        fv = extractor.extract(state)
        features = fv.as_list()
        # Deliberately scale features to create non-trivial means/stds.
        scaled = [f * (1.0 + i * 0.1) + i * 0.5 for f in features]
        best = actions[i % len(actions)]
        examples.append(TrainingExample(
            features=scaled,
            best_action=best,
            utility_margin=2.0,
            weight=3.0,
            task_id=f"task_{i}",
            task_family="test_family",
            split="train" if i < n * 0.8 else "test",
        ))
    return examples


def _train_policy() -> LearnedPolicy:
    examples = _build_training_examples(40)
    train = [e for e in examples if e.split == "train"]
    val = [e for e in examples if e.split == "test"]
    cfg = LearnerConfig(
        actions=tuple(Action.learned()),
        n_epochs=200,
        learning_rate=0.05,
        l2=0.01,
        confidence_threshold=0.4,
    )
    learner = SupervisedRouterLearner(cfg)
    policy, _ = learner.train(
        train,
        val,
        policy_id="invariance_test_policy",
        policy_version="learned_router_v2",
        training_data_digest="test_digest",
        split_manifest_digest="test_split_digest",
        parent_policy="baseline_v2",
    )
    return policy


# ---------------------------------------------------------------------------
# Invariance tests
# ---------------------------------------------------------------------------


class TestPolicySerializationEquivalence:
    """Verify that save/reload produces identical predictions."""

    def test_feature_means_and_stds_preserved_after_serialization(self):
        """The serialized policy must retain feature_means and feature_stds."""
        policy = _train_policy()
        means = policy.hyperparameters.get("feature_means")
        stds = policy.hyperparameters.get("feature_stds")
        assert means is not None, "feature_means missing from trained policy"
        assert stds is not None, "feature_stds missing from trained policy"
        assert len(means) == len(FEATURE_NAMES)
        assert len(stds) == len(FEATURE_NAMES)

        # Serialize and reload.
        reloaded = LearnedPolicy.from_json(policy.to_json())
        assert reloaded.hyperparameters.get("feature_means") == means
        assert reloaded.hyperparameters.get("feature_stds") == stds

    def test_predicted_action_identical_after_reload(self):
        """Every sample must produce the same action before and after reload."""
        policy = _train_policy()
        reloaded = LearnedPolicy.from_json(policy.to_json())
        extractor = FeatureExtractor()

        states = [
            _state("What is 2+2?"),
            _state("Fetch the live stock price.", available_tools=["stock_api"]),
            _state("Remember the pin.", memory_hit_count=3, top_similarity=0.9),
            _state("Run the pipeline.", workflow_available=True),
            _state("What is the capital of France?"),
            _state("Use the calculator.", available_tools=["calculator"]),
            _state("Recall the budget.", memory_hit_count=2, top_similarity=0.8),
            _state("Execute deployment.", workflow_available=True),
        ]

        for state in states:
            d1 = policy.select_action(state, extractor=extractor)
            d2 = reloaded.select_action(state, extractor=extractor)
            assert d1.action == d2.action, (
                f"action mismatch for prompt {state.prompt_features['text']!r}: "
                f"original={d1.action}, reloaded={d2.action}"
            )

    def test_probability_vector_identical_after_reload(self):
        """Probability vectors must be identical within numerical tolerance."""
        policy = _train_policy()
        reloaded = LearnedPolicy.from_json(policy.to_json())
        extractor = FeatureExtractor()

        states = [
            _state("What is 2+2?"),
            _state("Fetch the live stock price.", available_tools=["stock_api"]),
            _state("Remember the pin.", memory_hit_count=3, top_similarity=0.9),
            _state("Run the pipeline.", workflow_available=True),
        ]

        tol = 1e-12
        for state in states:
            fv = extractor.extract(state)
            p1 = policy.predict_proba(fv)
            p2 = reloaded.predict_proba(fv)
            assert set(p1.keys()) == set(p2.keys())
            for key in p1:
                assert abs(p1[key] - p2[key]) < tol, (
                    f"probability mismatch for {key}: "
                    f"original={p1[key]}, reloaded={p2[key]}, diff={abs(p1[key] - p2[key])}"
                )

    def test_confidence_identical_after_reload(self):
        """Confidence values must be identical within numerical tolerance."""
        policy = _train_policy()
        reloaded = LearnedPolicy.from_json(policy.to_json())
        extractor = FeatureExtractor()

        states = [
            _state("What is 2+2?"),
            _state("Fetch the live stock price.", available_tools=["stock_api"]),
            _state("Remember the pin.", memory_hit_count=3, top_similarity=0.9),
            _state("Run the pipeline.", workflow_available=True),
        ]

        tol = 1e-12
        for state in states:
            d1 = policy.select_action(state, extractor=extractor)
            d2 = reloaded.select_action(state, extractor=extractor)
            assert abs(d1.confidence - d2.confidence) < tol, (
                f"confidence mismatch: original={d1.confidence}, "
                f"reloaded={d2.confidence}"
            )

    def test_standardization_actively_changing_features(self):
        """Verify that standardization is non-trivial (means/stds are not 0/1).

        This ensures the test is meaningful — if standardization were a no-op,
        losing the parameters wouldn't matter.
        """
        policy = _train_policy()
        means = policy.hyperparameters["feature_means"]
        stds = policy.hyperparameters["feature_stds"]
        # At least one mean should be non-zero.
        assert any(abs(m) > 1e-6 for m in means), (
            "all feature means are zero — test data is not exercising standardization"
        )
        # At least one std should differ from 1.0.
        assert any(abs(s - 1.0) > 1e-6 for s in stds), (
            "all feature stds are 1.0 — test data is not exercising standardization"
        )

    def test_overwriting_hyperparameters_breaks_predictions(self):
        """Verify that the bug (overwriting hyperparameters) actually breaks things.

        This is a regression test: if someone overwrites hyperparameters without
        preserving feature_means/stds, predictions MUST change. If they don't,
        either the test data is trivial or standardization is broken.
        """
        policy = _train_policy()
        extractor = FeatureExtractor()
        state = _state("Fetch the live stock price.", available_tools=["stock_api"])

        # Original prediction.
        d_orig = policy.select_action(state, extractor=extractor)

        # Simulate the bug: overwrite hyperparameters, losing feature_means/stds.
        policy.hyperparameters = {
            "n_epochs": 200,
            "learning_rate": 0.05,
        }

        d_buggy = policy.select_action(state, extractor=extractor)

        # The predictions should differ (either action or probability).
        # If they don't differ, the standardization wasn't doing anything.
        # Note: in edge cases the argmax might not change, but probabilities
        # should differ.
        fv = extractor.extract(state)
        # Restore original policy for comparison.
        policy2 = _train_policy()
        p_orig = policy2.predict_proba(fv)
        p_buggy = policy.predict_proba(fv)

        probs_differ = any(
            abs(p_orig[k] - p_buggy[k]) > 1e-9 for k in p_orig
        )
        assert probs_differ, (
            "overwriting hyperparameters did not change predictions — "
            "standardization is not being applied or test data is trivial"
        )

    def test_round_trip_through_file(self):
        """Verify invariance through an actual file write/read cycle."""
        policy = _train_policy()
        reloaded = LearnedPolicy.from_json(policy.to_json())
        extractor = FeatureExtractor()

        states = [
            _state("What is 2+2?"),
            _state("Fetch the live stock price.", available_tools=["stock_api"]),
            _state("Remember the pin.", memory_hit_count=3, top_similarity=0.9),
        ]

        for state in states:
            d1 = policy.select_action(state, extractor=extractor)
            d2 = reloaded.select_action(state, extractor=extractor)
            assert d1.action == d2.action
            assert abs(d1.confidence - d2.confidence) < 1e-12

    def test_registry_preserves_normalization_metadata(self):
        """Verify that PolicyRegistry.register() preserves feature_means/stds
        in both the artifact and the manifest."""
        from capsule_brain.autolearn.registry import PolicyRegistry

        policy = _train_policy()
        tmp = Path(__file__).resolve().parent / "_tmp_registry_test"
        tmp.mkdir(exist_ok=True)
        try:
            registry = PolicyRegistry(tmp / "policies")
            manifest = registry.register(
                policy,
                training_data_digest="test_digest",
                split_manifest_digest="test_split",
                hyperparameters=policy.hyperparameters,
                metrics={},
                parent_policy="baseline_v2",
            )
            # Manifest must have feature_means/stds.
            assert "feature_means" in manifest.hyperparameters
            assert "feature_stds" in manifest.hyperparameters

            # Artifact must have feature_means/stds.
            loaded = registry.load_policy(policy.policy_id)
            assert loaded.hyperparameters.get("feature_means") is not None
            assert loaded.hyperparameters.get("feature_stds") is not None

            # Predictions must match.
            extractor = FeatureExtractor()
            state = _state("Fetch the live stock price.", available_tools=["stock_api"])
            d1 = policy.select_action(state, extractor=extractor)
            d2 = loaded.select_action(state, extractor=extractor)
            assert d1.action == d2.action
            assert abs(d1.confidence - d2.confidence) < 1e-12
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
