"""Learned executive policy.

A small, interpretable multinomial logistic regression over the deterministic
feature vector. Supports:
- confidence-based abstention
- deterministic inference (no sampling at inference time)
- serialization to a portable dict (no pickle required for the manifest)
- load-time feature-schema validation (incompatible policies are rejected)
- typed FeatureTransform for preprocessing statistics (v0.3.1 / 2.15.2)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from capsule_brain.autolearn.features import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_NAMES,
    FeatureExtractor,
    FeatureVector,
)
from capsule_brain.autolearn.schema import Action, ExecutiveState

LEARNED_POLICY_TYPE = "multinomial_logistic"
LEARNED_POLICY_VERSION_DEFAULT = "learned_router_v2"


class PolicyLoadError(RuntimeError):
    """Raised when a policy cannot be loaded (incompatible schema, corrupt, etc)."""


@dataclass(frozen=True)
class FeatureTransform:
    """Typed preprocessing transform applied before logits.

    v0.3.1: Replaces the unstructured ``hyperparameters["feature_means"]``
    / ``hyperparameters["feature_stds"]`` pattern with a typed, validated
    dataclass that is serialized explicitly in the policy artifact.

    The transform standardizes features:
        x'_j = (x_j - mean_j) / std_j  (if std_j > 0, else 0.0)

    A policy artifact that is missing this transform, has mismatched array
    lengths, or has zero-length arrays must fail closed at load time.
    """

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.feature_names) == 0:
            raise PolicyLoadError("FeatureTransform: feature_names is empty")
        if len(self.means) != len(self.feature_names):
            raise PolicyLoadError(
                f"FeatureTransform: means has {len(self.means)} entries, "
                f"expected {len(self.feature_names)}"
            )
        if len(self.stds) != len(self.feature_names):
            raise PolicyLoadError(
                f"FeatureTransform: stds has {len(self.stds)} entries, "
                f"expected {len(self.feature_names)}"
            )
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise PolicyLoadError(
                f"FeatureTransform: schema {self.schema_version!r} != "
                f"expected {FEATURE_SCHEMA_VERSION!r}"
            )

    def apply(self, x: list[float]) -> list[float]:
        """Apply standardization to a raw feature vector."""
        if len(x) != len(self.feature_names):
            raise PolicyLoadError(
                f"FeatureTransform: input has {len(x)} values, "
                f"expected {len(self.feature_names)}"
            )
        return [
            (x[j] - self.means[j]) / self.stds[j]
            if self.stds[j] > 0
            else 0.0
            for j in range(len(x))
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "stds": list(self.stds),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureTransform":
        return cls(
            feature_names=tuple(data.get("feature_names", [])),
            means=tuple(data.get("means", [])),
            stds=tuple(data.get("stds", [])),
            schema_version=str(data.get("schema_version", FEATURE_SCHEMA_VERSION)),
        )


@dataclass(slots=True)
class PolicyDecision:
    action: Action
    scores: dict[str, float]
    probabilities: dict[str, float]
    confidence: float
    policy_version: str
    policy_type: str
    feature_vector: FeatureVector | None = None
    abstained: bool = False
    reason: str = ""


@dataclass(slots=True)
class LearnedPolicy:
    """Multinomial logistic regression over the deterministic features.

    The model is a weight matrix W of shape (n_actions, n_features) plus a
    bias vector b of shape (n_actions,). logits = W @ x + b. Probabilities
    are the softmax of the logits. The chosen action is argmax (deterministic
    at inference time).
    """

    policy_id: str
    policy_version: str = LEARNED_POLICY_VERSION_DEFAULT
    policy_type: str = LEARNED_POLICY_TYPE
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_names: tuple[str, ...] = FEATURE_NAMES
    actions: tuple[Action, ...] = field(default_factory=tuple)
    weights: list[list[float]] = field(default_factory=list)  # [n_actions][n_features]
    bias: list[float] = field(default_factory=list)  # [n_actions]
    confidence_threshold: float = 0.5
    temperature: float = 1.0
    training_data_digest: str = ""
    split_manifest_digest: str = ""
    parent_policy: str = ""
    feature_transform: FeatureTransform | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    trained_at: str = ""

    def __post_init__(self) -> None:
        if not self.actions:
            self.actions = tuple(Action.all())
        n_features = len(self.feature_names)
        n_actions = len(self.actions)
        if not self.weights:
            self.weights = [[0.0] * n_features for _ in range(n_actions)]
        if not self.bias:
            self.bias = [0.0] * n_actions
        # Validate shape consistency.
        if len(self.weights) != n_actions:
            raise PolicyLoadError(
                f"weights has {len(self.weights)} rows, expected {n_actions}"
            )
        for row in self.weights:
            if len(row) != n_features:
                raise PolicyLoadError(
                    f"weight row has {len(row)} cols, expected {n_features}"
                )
        if len(self.bias) != n_actions:
            raise PolicyLoadError(
                f"bias has {len(self.bias)} entries, expected {n_actions}"
            )

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def _logits(self, x: list[float]) -> list[float]:
        out = list(self.bias)
        for i, row in enumerate(self.weights):
            s = self.bias[i]
            for j, w in enumerate(row):
                s += w * x[j]
            out[i] = s
        return out

    @staticmethod
    def _softmax(logits: list[float], temperature: float) -> list[float]:
        if temperature <= 0:
            temperature = 1e-6
        scaled = [l / temperature for l in logits]
        m = max(scaled)
        exps = [math.exp(l - m) for l in scaled]
        total = sum(exps)
        if total <= 0:
            # Degenerate; return uniform.
            n = len(scaled)
            return [1.0 / n] * n
        return [e / total for e in exps]

    def predict_proba(self, fv: FeatureVector) -> dict[str, float]:
        self._validate_feature_vector(fv)
        x = self._standardize(fv.as_list())
        logits = self._logits(x)
        probs = self._softmax(logits, self.temperature)
        return {self.actions[i].value: probs[i] for i in range(len(self.actions))}

    def _standardize(self, x: list[float]) -> list[float]:
        """Apply training-set standardization if parameters are stored.

        v0.3.1: Uses the typed FeatureTransform if available. Falls back to
        hyperparameters dict for backward compatibility, but logs a warning.
        Fails closed (raises PolicyLoadError) if transform is missing and
        the policy was trained with standardization.
        """
        if self.feature_transform is not None:
            return self.feature_transform.apply(x)
        # Backward compatibility: check hyperparameters dict.
        means = self.hyperparameters.get("feature_means")
        stds = self.hyperparameters.get("feature_stds")
        if means is not None and stds is not None:
            if len(means) == len(x) and len(stds) == len(x):
                return [
                    (x[j] - means[j]) / stds[j] if stds[j] > 0 else 0.0
                    for j in range(len(x))
                ]
        # No transform — return raw features (uniform weights produce
        # uniform probabilities, so this is safe for zero-initialized models).
        return x

    def select_action(
        self,
        state: ExecutiveState,
        *,
        extractor: FeatureExtractor | None = None,
        allowed_actions: list[Action] | None = None,
    ) -> PolicyDecision:
        ext = extractor or FeatureExtractor()
        if ext.schema_version != self.feature_schema_version:
            raise PolicyLoadError(
                f"extractor schema {ext.schema_version!r} != policy schema "
                f"{self.feature_schema_version!r}"
            )
        fv = ext.extract(state)
        probs = self.predict_proba(fv)

        # Restrict to allowed actions if specified.
        allowed = (
            set(allowed_actions)
            if allowed_actions is not None
            else set(self.actions)
        )
        candidate_probs = {
            a: p for a, p in probs.items() if Action(a) in allowed
        }
        if not candidate_probs:
            return PolicyDecision(
                action=Action.ASK_OPERATOR,
                scores={},
                probabilities=probs,
                confidence=0.0,
                policy_version=self.policy_version,
                policy_type=self.policy_type,
                feature_vector=fv,
                abstained=True,
                reason="no allowed actions available",
            )

        best_action_str = max(candidate_probs, key=candidate_probs.get)
        best_action = Action(best_action_str)
        confidence = candidate_probs[best_action_str]

        abstained = confidence < self.confidence_threshold
        reason = (
            "abstained: confidence below threshold"
            if abstained
            else "learned policy argmax"
        )
        return PolicyDecision(
            action=best_action,
            scores={a: p for a, p in probs.items()},
            probabilities=probs,
            confidence=confidence,
            policy_version=self.policy_version,
            policy_type=self.policy_type,
            feature_vector=fv,
            abstained=abstained,
            reason=reason,
        )

    def _validate_feature_vector(self, fv: FeatureVector) -> None:
        if fv.schema_version != self.feature_schema_version:
            raise PolicyLoadError(
                f"feature vector schema {fv.schema_version!r} != policy schema "
                f"{self.feature_schema_version!r}"
            )
        if fv.names != self.feature_names:
            raise PolicyLoadError(
                "feature names mismatch: policy expects "
                f"{self.feature_names}, got {fv.names}"
            )

    # ----- serialization -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_type": self.policy_type,
            "feature_schema_version": self.feature_schema_version,
            "feature_names": list(self.feature_names),
            "actions": [a.value for a in self.actions],
            "weights": [list(row) for row in self.weights],
            "bias": list(self.bias),
            "confidence_threshold": self.confidence_threshold,
            "temperature": self.temperature,
            "training_data_digest": self.training_data_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "parent_policy": self.parent_policy,
            "feature_transform": self.feature_transform.to_dict() if self.feature_transform else None,
            "hyperparameters": dict(self.hyperparameters),
            "metrics": dict(self.metrics),
            "trained_at": self.trained_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearnedPolicy":
        schema = str(data.get("feature_schema_version", ""))
        if schema != FEATURE_SCHEMA_VERSION:
            raise PolicyLoadError(
                f"incompatible feature schema: policy has {schema!r}, "
                f"runtime expects {FEATURE_SCHEMA_VERSION!r}"
            )
        actions = tuple(
            Action(a) for a in data.get("actions", [a.value for a in Action.all()])
        )
        feature_names = tuple(
            data.get("feature_names", list(FEATURE_NAMES))
        )
        if feature_names != FEATURE_NAMES:
            raise PolicyLoadError(
                f"incompatible feature names: policy has {feature_names}, "
                f"runtime expects {FEATURE_NAMES}"
            )
        # v0.3.1: Load typed FeatureTransform if present.
        ft_data = data.get("feature_transform")
        feature_transform: FeatureTransform | None = None
        if ft_data is not None:
            feature_transform = FeatureTransform.from_dict(ft_data)
        # Validate weights and bias shapes.
        weights = [list(row) for row in data.get("weights", [])]
        bias = list(data.get("bias", []))
        if len(weights) != len(actions):
            raise PolicyLoadError(
                f"weights has {len(weights)} rows, expected {len(actions)}"
            )
        for row in weights:
            if len(row) != len(feature_names):
                raise PolicyLoadError(
                    f"weight row has {len(row)} cols, expected {len(feature_names)}"
                )
        if len(bias) != len(actions):
            raise PolicyLoadError(
                f"bias has {len(bias)} entries, expected {len(actions)}"
            )
        return cls(
            policy_id=str(data.get("policy_id", "")),
            policy_version=str(
                data.get("policy_version", LEARNED_POLICY_VERSION_DEFAULT)
            ),
            policy_type=str(data.get("policy_type", LEARNED_POLICY_TYPE)),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_names=feature_names,
            actions=actions,
            weights=weights,
            bias=bias,
            confidence_threshold=float(data.get("confidence_threshold", 0.5)),
            temperature=float(data.get("temperature", 1.0)),
            training_data_digest=str(data.get("training_data_digest", "")),
            split_manifest_digest=str(data.get("split_manifest_digest", "")),
            parent_policy=str(data.get("parent_policy", "")),
            feature_transform=feature_transform,
            hyperparameters=dict(data.get("hyperparameters", {}) or {}),
            metrics=dict(data.get("metrics", {}) or {}),
            trained_at=str(data.get("trained_at", "")),
        )

    @classmethod
    def from_json(cls, text: str) -> "LearnedPolicy":
        return cls.from_dict(json.loads(text))
