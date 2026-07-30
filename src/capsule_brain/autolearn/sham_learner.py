"""Sham-learning control for AutoLearn v2.16.0 qualification.

Priority 4: A sham learner that uses identical optimizer, architecture,
epochs, feature extractor, and capacity as the real learner — but destroys
the experience signal. This controls for the possibility that any observed
improvement is due to optimization or regularization rather than learning
from real counterfactual experience.

Sham methods:
- ``shuffled_utilities``: Shuffle the utility margins across tasks,
  destroying the relationship between task features and action outcomes.
- ``shuffled_actions``: Shuffle the best-action labels within strata
  (task families), destroying the feature→action mapping.
- ``shuffled_labels``: Shuffle all labels globally.

Qualification requires: AutoLearn > Baseline AND AutoLearn > Sham.
"""
from __future__ import annotations

import random
from typing import Any

from capsule_brain.autolearn.learner import (
    LearnerConfig,
    SupervisedRouterLearner,
    TrainingExample,
)
from capsule_brain.autolearn.policy import LearnedPolicy


def shuffle_utilities(
    examples: list[TrainingExample],
    *,
    seed: int = 42,
) -> list[TrainingExample]:
    """Shuffle utility margins across tasks.

    This destroys the relationship between task features and action outcomes.
    The best action and features remain unchanged, but the utility margin
    (which determines the training weight) is randomly reassigned.
    """
    rng = random.Random(seed)
    margins = [e.utility_margin for e in examples]
    rng.shuffle(margins)
    shuffled = []
    for i, e in enumerate(examples):
        shuffled.append(TrainingExample(
            task_id=e.task_id,
            task_family=e.task_family,
            split=e.split,
            best_action=e.best_action,
            utility_margin=margins[i],
            weight=1.0 + min(5.0, max(0.0, margins[i])),
            features=e.features,
        ))
    return shuffled


def shuffle_actions_within_strata(
    examples: list[TrainingExample],
    *,
    seed: int = 42,
) -> list[TrainingExample]:
    """Shuffle best-action labels within task-family strata.

    This destroys the feature→action mapping while preserving the marginal
    distribution of actions within each family. The learner sees the same
    features and the same action distribution, but the labels are randomly
    reassigned within each family.
    """
    rng = random.Random(seed)
    # Group by family.
    by_family: dict[str, list[int]] = {}
    for i, e in enumerate(examples):
        by_family.setdefault(e.task_family, []).append(i)

    # Shuffle actions within each family.
    new_actions = [e.best_action for e in examples]
    for family, indices in by_family.items():
        actions = [examples[i].best_action for i in indices]
        rng.shuffle(actions)
        for j, idx in enumerate(indices):
            new_actions[idx] = actions[j]

    shuffled = []
    for i, e in enumerate(examples):
        shuffled.append(TrainingExample(
            task_id=e.task_id,
            task_family=e.task_family,
            split=e.split,
            best_action=new_actions[i],
            utility_margin=e.utility_margin,
            weight=e.weight,
            features=e.features,
        ))
    return shuffled


def shuffle_labels_globally(
    examples: list[TrainingExample],
    *,
    seed: int = 42,
) -> list[TrainingExample]:
    """Shuffle all labels globally.

    This completely destroys any feature→action relationship.
    """
    rng = random.Random(seed)
    actions = [e.best_action for e in examples]
    rng.shuffle(actions)
    shuffled = []
    for i, e in enumerate(examples):
        shuffled.append(TrainingExample(
            task_id=e.task_id,
            task_family=e.task_family,
            split=e.split,
            best_action=actions[i],
            utility_margin=e.utility_margin,
            weight=e.weight,
            features=e.features,
        ))
    return shuffled


def train_sham_policy(
    examples: list[TrainingExample],
    config: LearnerConfig,
    *,
    sham_method: str = "shuffled_actions",
    seed: int = 42,
    policy_id: str = "sham_policy",
    training_data_digest: str = "",
    split_manifest_digest: str = "",
    parent_policy: str = "baseline_v2",
) -> tuple[LearnedPolicy, Any]:
    """Train a sham policy with identical architecture but destroyed signal.

    Uses the same LearnerConfig (optimizer, epochs, L2, etc.) and the same
    SupervisedRouterLearner, but shuffles the training signal first.

    Args:
        examples: The original training examples.
        config: The same LearnerConfig used for the real policy.
        sham_method: One of "shuffled_utilities", "shuffled_actions",
            "shuffled_labels".
        seed: Random seed for reproducibility.
        policy_id: Policy ID for the sham policy.
        training_data_digest: Digest of the original training data.
        split_manifest_digest: Digest of the split manifest.
        parent_policy: Parent policy ID.

    Returns:
        Tuple of (sham_policy, training_metrics).
    """
    if sham_method == "shuffled_utilities":
        sham_examples = shuffle_utilities(examples, seed=seed)
    elif sham_method == "shuffled_actions":
        sham_examples = shuffle_actions_within_strata(examples, seed=seed)
    elif sham_method == "shuffled_labels":
        sham_examples = shuffle_labels_globally(examples, seed=seed)
    else:
        raise ValueError(f"unknown sham method: {sham_method}")

    # Split into train and validation (same as real training).
    train = [e for e in sham_examples if e.split == "train"]
    val = [e for e in sham_examples if e.split == "validation"]
    if not val:
        val = train[:max(1, len(train) // 10)]

    # Use the SAME learner with the SAME config.
    learner = SupervisedRouterLearner(config)
    policy, metrics = learner.train(
        train,
        val,
        policy_id=policy_id,
        policy_version="learned_router_v2",
        training_data_digest=training_data_digest,
        split_manifest_digest=split_manifest_digest,
        parent_policy=parent_policy,
    )

    # Mark the policy as a sham policy in hyperparameters.
    policy.hyperparameters["sham_method"] = sham_method
    policy.hyperparameters["sham_seed"] = seed

    return policy, metrics
