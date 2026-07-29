"""Shadow mode + distribution-shift guard.

Shadow mode runs the candidate alongside production without executing the
candidate action (unless the task is in an explicit safe evaluation set).
Disagreement cases are high-information data and are prioritized for
evaluation.

The distribution-shift guard compares the current feature distribution to
the training distribution. If shift exceeds a threshold, the policy falls
back to baseline and logs ``distribution_shift_detected = true``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from capsule_brain.autolearn.baseline import BaselinePolicy
from capsule_brain.autolearn.features import FeatureExtractor
from capsule_brain.autolearn.policy import LearnedPolicy
from capsule_brain.autolearn.schema import Action, ExecutiveState


@dataclass(slots=True)
class ShadowRecord:
    task_id: str
    production_action: Action
    shadow_action: Action
    disagreement: bool
    shadow_confidence: float
    shadow_abstained: bool
    shadow_reason: str
    distribution_shift_detected: bool
    timestamp: str = ""


@dataclass(slots=True)
class ShadowStats:
    n: int = 0
    disagreements: int = 0
    distribution_shifts: int = 0
    abstentions: int = 0
    by_action: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "disagreements": self.disagreements,
            "distribution_shifts": self.distribution_shifts,
            "abstentions": self.abstentions,
            "by_action": dict(self.by_action),
        }


@dataclass(slots=True)
class DistributionShiftConfig:
    # Maximum allowed normalized distance between current feature mean and
    # training feature mean before the policy falls back to baseline.
    max_mean_distance: float = 1.5
    # Minimum number of recent observations required before computing a
    # distribution-shift estimate. Below this, no shift is declared.
    min_window: int = 10
    # Per-feature weights for the distance metric (uniform by default).
    feature_weights: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_mean_distance": self.max_mean_distance,
            "min_window": self.min_window,
            "feature_weights": list(self.feature_weights) if self.feature_weights else None,
        }


class DistributionShiftGuard:
    """Tracks recent feature vectors and compares them to a training reference."""

    def __init__(
        self,
        training_means: list[float] | None = None,
        config: DistributionShiftConfig | None = None,
    ) -> None:
        self.config = config or DistributionShiftConfig()
        self.training_means = list(training_means) if training_means else []
        self.recent: list[list[float]] = []

    def observe(self, features: list[float]) -> None:
        self.recent.append(list(features))
        # Keep a bounded window.
        max_window = max(self.config.min_window * 4, 200)
        if len(self.recent) > max_window:
            self.recent = self.recent[-max_window:]

    def current_distance(self) -> float:
        if len(self.recent) < self.config.min_window or not self.training_means:
            return 0.0
        n_features = len(self.training_means)
        cur_means = [0.0] * n_features
        for vec in self.recent:
            for i in range(min(n_features, len(vec))):
                cur_means[i] += vec[i]
        n = len(self.recent)
        cur_means = [m / n for m in cur_means]
        weights = (
            self.config.feature_weights
            if self.config.feature_weights
            else (1.0,) * n_features
        )
        total = 0.0
        wsum = 0.0
        for i in range(n_features):
            w = weights[i] if i < len(weights) else 1.0
            total += w * (cur_means[i] - self.training_means[i]) ** 2
            wsum += w
        return math.sqrt(total / wsum) if wsum > 0 else 0.0

    def shifted(self) -> bool:
        return self.current_distance() > self.config.max_mean_distance


class ShadowEvaluator:
    """Runs a candidate policy in shadow mode alongside a production baseline."""

    def __init__(
        self,
        baseline: BaselinePolicy,
        candidate: LearnedPolicy,
        *,
        extractor: FeatureExtractor | None = None,
        shift_guard: DistributionShiftGuard | None = None,
        confidence_threshold: float = 0.5,
        safe_evaluation_task_ids: set[str] | None = None,
    ) -> None:
        self.baseline = baseline
        self.candidate = candidate
        self.extractor = extractor or FeatureExtractor()
        self.shift_guard = shift_guard
        self.confidence_threshold = confidence_threshold
        self.safe_evaluation_task_ids = safe_evaluation_task_ids or set()
        self.records: list[ShadowRecord] = []
        self.stats = ShadowStats()

    def evaluate(
        self,
        state: ExecutiveState,
        *,
        task_id: str = "",
        allowed_actions: list[Action] | None = None,
        timestamp: str = "",
    ) -> ShadowRecord:
        # Production action: baseline policy.
        prod_dec = self.baseline.select_action(state, allowed_actions=allowed_actions)
        production_action = prod_dec.action

        # Distribution-shift check.
        shift_detected = False
        if self.shift_guard is not None:
            fv = self.extractor.extract(state)
            self.shift_guard.observe(fv.as_list())
            shift_detected = self.shift_guard.shifted()

        # Shadow action: candidate policy (with abstention).
        shadow_action = production_action
        shadow_confidence = 0.0
        shadow_abstained = True
        shadow_reason = "no candidate decision"

        if not shift_detected:
            try:
                c_dec = self.candidate.select_action(
                    state,
                    extractor=self.extractor,
                    allowed_actions=allowed_actions,
                )
                shadow_confidence = c_dec.confidence
                shadow_abstained = c_dec.abstained or c_dec.confidence < self.confidence_threshold
                if shadow_abstained:
                    shadow_action = production_action
                    shadow_reason = "candidate abstained"
                else:
                    shadow_action = c_dec.action
                    shadow_reason = "candidate argmax"
            except Exception as exc:
                shadow_abstained = True
                shadow_reason = f"candidate error: {exc}"
        else:
            shadow_reason = "distribution shift detected; using baseline"

        disagreement = production_action != shadow_action

        record = ShadowRecord(
            task_id=task_id,
            production_action=production_action,
            shadow_action=shadow_action,
            disagreement=disagreement,
            shadow_confidence=shadow_confidence,
            shadow_abstained=shadow_abstained,
            shadow_reason=shadow_reason,
            distribution_shift_detected=shift_detected,
            timestamp=timestamp,
        )
        self.records.append(record)
        self.stats.n += 1
        if disagreement:
            self.stats.disagreements += 1
        if shift_detected:
            self.stats.distribution_shifts += 1
        if shadow_abstained:
            self.stats.abstentions += 1
        self.stats.by_action[shadow_action.value] = (
            self.stats.by_action.get(shadow_action.value, 0) + 1
        )
        return record

    def disagreement_records(self) -> list[ShadowRecord]:
        return [r for r in self.records if r.disagreement]
