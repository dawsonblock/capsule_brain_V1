"""Supervised router learner: multinomial logistic regression.

Trains on (state_features, best_verified_action, utility_margin_weight)
triples produced by the counterfactual dataset builder. Uses weighted
cross-entropy:

    L(theta) = -sum_i w_i * log pi_theta(a_i* | x_i)

where w_i = f(U(best) - U(second_best)) is a bounded monotonic function of
the utility margin. Ambiguous tasks (small margin) contribute less weight;
clear routing decisions (large margin) contribute more.

Pure-Python implementation (no numpy required at runtime) so the learner
works in the minimal Capsule Brain dependency set. Uses full-batch gradient
descent with L2 regularization for determinism and reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

from capsule_brain.autolearn.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FeatureExtractor,
)
from capsule_brain.autolearn.policy import (
    LEARNED_POLICY_TYPE,
    LEARNED_POLICY_VERSION_DEFAULT,
    LearnedPolicy,
)
from capsule_brain.autolearn.schema import Action


@dataclass(slots=True)
class TrainingExample:
    features: list[float]
    best_action: Action
    utility_margin: float  # U(best) - U(second_best)
    weight: float
    task_id: str = ""
    task_family: str = "unknown"
    split: str = "train"  # train | validation | test | ood


@dataclass(slots=True)
class TrainingMetrics:
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    train_weighted_accuracy: float = 0.0
    validation_loss: float = 0.0
    validation_accuracy: float = 0.0
    validation_weighted_accuracy: float = 0.0
    n_train: int = 0
    n_validation: int = 0
    n_epochs: int = 0
    final_weights_norm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "train_weighted_accuracy": self.train_weighted_accuracy,
            "validation_loss": self.validation_loss,
            "validation_accuracy": self.validation_accuracy,
            "validation_weighted_accuracy": self.validation_weighted_accuracy,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "n_epochs": self.n_epochs,
            "final_weights_norm": self.final_weights_norm,
        }


@dataclass(slots=True)
class LearnerConfig:
    learning_rate: float = 0.1
    n_epochs: int = 400
    l2: float = 1e-3
    # Bounded monotonic weight function: w_i = 1 + min(weight_cap, max(0, dU))
    weight_cap: float = 4.0
    min_weight: float = 1.0
    confidence_threshold: float = 0.5
    temperature: float = 1.0
    actions: tuple[Action, ...] = field(default_factory=tuple)
    seed: int = 0  # reserved for future stochastic variants; full-batch GD is deterministic

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "l2": self.l2,
            "weight_cap": self.weight_cap,
            "min_weight": self.min_weight,
            "confidence_threshold": self.confidence_threshold,
            "temperature": self.temperature,
            "actions": [a.value for a in self.actions],
            "seed": self.seed,
        }


def utility_margin_weight(margin: float, cfg: LearnerConfig) -> float:
    """Bounded monotonic function of the utility margin.

    w_i = 1 + min(weight_cap, max(0, dU))
    """
    return cfg.min_weight + min(cfg.weight_cap, max(0.0, margin))


class SupervisedRouterLearner:
    """Multinomial logistic regression trained with weighted cross-entropy."""

    def __init__(
        self,
        cfg: LearnerConfig | None = None,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        self.cfg = cfg or LearnerConfig()
        self.extractor = extractor or FeatureExtractor()
        if not self.cfg.actions:
            # v0.3: the learned policy only trains on the 4 learned actions.
            # REFLECT and ASK_OPERATOR remain runtime-controlled (safety floor).
            self.cfg.actions = tuple(Action.learned())
        self.action_index = {a: i for i, a in enumerate(self.cfg.actions)}
        self.n_features = len(FEATURE_NAMES)
        self.n_actions = len(self.cfg.actions)

    def _init_params(self) -> tuple[list[list[float]], list[float]]:
        # Zero init -> uniform probabilities. Deterministic.
        W = [[0.0] * self.n_features for _ in range(self.n_actions)]
        b = [0.0] * self.n_actions
        return W, b

    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        total = sum(exps)
        if total <= 0:
            n = len(logits)
            return [1.0 / n] * n
        return [e / total for e in exps]

    def _forward(self, W, b, x: list[float]) -> list[float]:
        logits = [b[i] + sum(W[i][j] * x[j] for j in range(self.n_features))
                  for i in range(self.n_actions)]
        return self._softmax(logits)

    def _loss_and_grads(
        self,
        examples: list[TrainingExample],
        W,
        b,
    ) -> tuple[float, list[list[float]], list[float]]:
        n = len(examples)
        if n == 0:
            return 0.0, [[0.0] * self.n_features for _ in range(self.n_actions)], [0.0] * self.n_actions
        gW = [[0.0] * self.n_features for _ in range(self.n_actions)]
        gb = [0.0] * self.n_actions
        total_loss = 0.0
        for ex in examples:
            probs = self._forward(W, b, ex.features)
            ai = self.action_index[ex.best_action]
            p = max(probs[ai], 1e-12)
            total_loss += ex.weight * (-math.log(p))
            for i in range(self.n_actions):
                grad = probs[i] - (1.0 if i == ai else 0.0)
                gb[i] += ex.weight * grad
                for j in range(self.n_features):
                    gW[i][j] += ex.weight * grad * ex.features[j]
        # L2 regularization on weights (not bias).
        l2 = self.cfg.l2
        for i in range(self.n_actions):
            for j in range(self.n_features):
                gW[i][j] += l2 * W[i][j]
                total_loss += 0.5 * l2 * W[i][j] * W[i][j]
        return total_loss / n, gW, gb

    def _accuracy(
        self,
        examples: list[TrainingExample],
        W,
        b,
        weighted: bool,
    ) -> float:
        if not examples:
            return 0.0
        correct = 0.0
        total_w = 0.0
        for ex in examples:
            probs = self._forward(W, b, ex.features)
            pred = max(range(self.n_actions), key=lambda i: probs[i])
            ai = self.action_index[ex.best_action]
            w = ex.weight if weighted else 1.0
            if pred == ai:
                correct += w
            total_w += w
        return correct / total_w if total_w > 0 else 0.0

    def train(
        self,
        train: list[TrainingExample],
        validation: list[TrainingExample],
        *,
        policy_id: str,
        policy_version: str = LEARNED_POLICY_VERSION_DEFAULT,
        training_data_digest: str = "",
        split_manifest_digest: str = "",
        parent_policy: str = "",
    ) -> tuple[LearnedPolicy, TrainingMetrics]:
        # Standardize features using train-set statistics so features with
        # large scales (e.g. prompt_length) don't dominate the logits.
        # The standardization parameters are stored in the policy so the
        # same transformation is applied at inference time.
        feature_means, feature_stds = self._compute_standardization(train)
        train_norm = self._standardize_examples(train, feature_means, feature_stds)
        val_norm = self._standardize_examples(validation, feature_means, feature_stds)

        W, b = self._init_params()
        lr = self.cfg.learning_rate
        metrics = TrainingMetrics(
            n_train=len(train),
            n_validation=len(validation),
            n_epochs=self.cfg.n_epochs,
        )

        for epoch in range(self.cfg.n_epochs):
            loss, gW, gb = self._loss_and_grads(train_norm, W, b)
            for i in range(self.n_actions):
                for j in range(self.n_features):
                    W[i][j] -= lr * gW[i][j]
                b[i] -= lr * gb[i]
            if epoch == self.cfg.n_epochs - 1:
                metrics.train_loss = loss

        metrics.train_accuracy = self._accuracy(train_norm, W, b, weighted=False)
        metrics.train_weighted_accuracy = self._accuracy(train_norm, W, b, weighted=True)
        metrics.validation_loss = self._loss_and_grads(val_norm, W, b)[0]
        metrics.validation_accuracy = self._accuracy(val_norm, W, b, weighted=False)
        metrics.validation_weighted_accuracy = self._accuracy(val_norm, W, b, weighted=True)
        metrics.final_weights_norm = math.sqrt(
            sum(w * w for row in W for w in row) + sum(x * x for x in b)
        )

        policy = LearnedPolicy(
            policy_id=policy_id,
            policy_version=policy_version,
            policy_type=LEARNED_POLICY_TYPE,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_names=FEATURE_NAMES,
            actions=self.cfg.actions,
            weights=[list(row) for row in W],
            bias=list(b),
            confidence_threshold=self.cfg.confidence_threshold,
            temperature=self.cfg.temperature,
            training_data_digest=training_data_digest,
            split_manifest_digest=split_manifest_digest,
            parent_policy=parent_policy,
            hyperparameters=self.cfg.to_dict(),
            metrics=metrics.to_dict(),
            trained_at=_utc_now_iso(),
        )
        # Store standardization parameters in hyperparameters so the policy
        # can apply the same transformation at inference time.
        policy.hyperparameters["feature_means"] = list(feature_means)
        policy.hyperparameters["feature_stds"] = list(feature_stds)
        return policy, metrics

    def _compute_standardization(
        self, examples: list[TrainingExample]
    ) -> tuple[list[float], list[float]]:
        if not examples:
            return [0.0] * self.n_features, [1.0] * self.n_features
        n = len(examples)
        means = [0.0] * self.n_features
        for ex in examples:
            for j in range(self.n_features):
                means[j] += ex.features[j]
        means = [m / n for m in means]
        stds = [0.0] * self.n_features
        for ex in examples:
            for j in range(self.n_features):
                stds[j] += (ex.features[j] - means[j]) ** 2
        stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in stds]
        return means, stds

    @staticmethod
    def _standardize_examples(
        examples: list[TrainingExample],
        means: list[float],
        stds: list[float],
    ) -> list[TrainingExample]:
        out: list[TrainingExample] = []
        for ex in examples:
            normed = [
                (ex.features[j] - means[j]) / stds[j]
                for j in range(len(means))
            ]
            out.append(TrainingExample(
                features=normed,
                best_action=ex.best_action,
                utility_margin=ex.utility_margin,
                weight=ex.weight,
                task_id=ex.task_id,
                task_family=ex.task_family,
                split=ex.split,
            ))
        return out


def compute_dataset_digest(examples: list[TrainingExample]) -> str:
    """Stable SHA-256 over the (features, best_action, weight, split) triples."""
    h = hashlib.sha256()
    for ex in sorted(examples, key=lambda e: e.task_id or ""):
        h.update(json.dumps(
            {
                "features": [round(f, 8) for f in ex.features],
                "best_action": ex.best_action.value,
                "weight": round(ex.weight, 8),
                "split": ex.split,
                "task_id": ex.task_id,
                "task_family": ex.task_family,
            },
            sort_keys=True,
        ).encode("utf-8"))
    return h.hexdigest()


def _utc_now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()
